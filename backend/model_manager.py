import os
import json
import shutil
import time
from typing import Optional, List, Dict
from datetime import datetime
import logging

logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

MODELS_DIR = "models"
REGISTRY_FILE = os.path.join(MODELS_DIR, "registry.json")
BASE_MODEL_NAME = "Qwen/Qwen3-4B"
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "qwen3-4b-base")


class ModelManager:
    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._ensure_registry()
        self.training_status: Dict = {"status": "idle", "progress": 0, "message": ""}
        self.download_status: Dict = {"status": "idle", "progress": 0, "message": ""}
        self._chat_model = None
        self._chat_tokenizer = None
        self.loaded_chat_model_id: Optional[str] = None

    # ─── Registry ─────────────────────────────────────────────────────────────

    def _ensure_registry(self):
        if not os.path.exists(REGISTRY_FILE):
            self._save_registry({"models": {}})

    def _load_registry(self) -> dict:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)

    def _save_registry(self, data: dict):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _register_model(self, model_id: str, path: str, model_type: str,
                        parent_id: Optional[str] = None, pdfs: List[str] = None):
        registry = self._load_registry()
        registry["models"][model_id] = {
            "id": model_id,
            "path": path,
            "type": model_type,
            "parent_id": parent_id,
            "pdfs_trained_on": pdfs or [],
            "created_at": datetime.now().isoformat(),
            "deletable": model_type != "base"
        }
        self._save_registry(registry)

    def _resolve_base_path(self, model_id: str, registry: dict) -> str:
        """Walk the parent chain until we find the base model path."""
        meta = registry["models"][model_id]
        if meta["type"] == "base":
            return meta["path"]
        parent_id = meta.get("parent_id")
        if not parent_id or parent_id not in registry["models"]:
            raise ValueError(
                f"Cannot resolve base model for '{model_id}' — "
                f"parent '{parent_id}' missing from registry"
            )
        return self._resolve_base_path(parent_id, registry)

    # ─── Model listing ─────────────────────────────────────────────────────────

    def list_models(self) -> List[dict]:
        registry = self._load_registry()
        models = []
        for mid, meta in registry["models"].items():
            entry = dict(meta)
            entry["exists"] = os.path.exists(meta["path"])
            entry["size_mb"] = self._dir_size_mb(meta["path"]) if entry["exists"] else 0
            models.append(entry)
        return models

    def _dir_size_mb(self, path: str) -> float:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
        return round(total / 1024 / 1024, 1)

    # ─── Base model download ───────────────────────────────────────────────────

    def base_model_exists(self) -> bool:
        return (
            os.path.exists(BASE_MODEL_DIR) and
            any(f.endswith(".safetensors") or f.endswith(".bin")
                for f in os.listdir(BASE_MODEL_DIR))
        )

    def get_download_status(self) -> dict:
        return self.download_status

    def download_base_model(self):
        """Download Qwen3-4B from HuggingFace using snapshot_download."""
        try:
            from huggingface_hub import snapshot_download
            import threading

            self.download_status = {
                "status": "downloading", "progress": 5,
                "message": "Starting download of Qwen3-4B (~8 GB)..."
            }
            os.makedirs(BASE_MODEL_DIR, exist_ok=True)

            done  = [False]
            error = [None]

            def do_download():
                try:
                    snapshot_download(
                        repo_id=BASE_MODEL_NAME,
                        local_dir=BASE_MODEL_DIR,
                        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
                    )
                except Exception as e:
                    error[0] = e
                finally:
                    done[0] = True

            dl_thread = threading.Thread(target=do_download)
            dl_thread.start()

            TARGET_MB = 8_000
            while not done[0]:
                size = self._dir_size_mb(BASE_MODEL_DIR)
                pct  = min(95, int(size / TARGET_MB * 90) + 5)
                self.download_status = {
                    "status": "downloading",
                    "progress": pct,
                    "message": f"Downloading... {size:.0f} MB / ~{TARGET_MB} MB"
                }
                time.sleep(3)

            if error[0]:
                raise error[0]

            self._register_model("base", BASE_MODEL_DIR, "base")
            self.download_status = {
                "status": "completed", "progress": 100,
                "message": "Qwen3-4B downloaded successfully!"
            }
        except Exception as e:
            self.download_status = {
                "status": "error", "progress": 0, "message": str(e)
            }

    # ─── Training ──────────────────────────────────────────────────────────────

    def train_model(self, dataset_path: str, base_model_id: str, output_name: str):
        """
        Fine-tune using Qwen3 chat format.
        For iterative training: loads base weights, merges any existing
        adapter, then adds fresh LoRA layers for this run.
        Tuned for RTX 5070 Ti / 16 GB VRAM.
        """
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback
        from datasets import load_dataset

        try:
            registry = self._load_registry()
            if base_model_id not in registry["models"]:
                raise ValueError(f"Model '{base_model_id}' not found in registry")

            meta       = registry["models"][base_model_id]
            output_dir = os.path.join(MODELS_DIR, output_name)
            os.makedirs(output_dir, exist_ok=True)

            # ── Tune these for your GPU ──────────────────────────────────────
            MAX_SEQ_LENGTH = 4096   # 4096 ctx fills ~14 GB on 4-bit Qwen3-4B
            LORA_RANK      = 32     # 32 = better quality vs 16, ~2 GB more VRAM
            BATCH_SIZE     = 4      # per-device batch
            GRAD_ACCUM     = 4      # effective batch = 16
            MAX_STEPS      = 100
            # ────────────────────────────────────────────────────────────────

            self.training_status = {
                "status": "loading", "progress": 20,
                "message": f"Loading model from '{base_model_id}'..."
            }

            # Always start from the true base weights
            if meta["type"] == "trained":
                true_base_path = self._resolve_base_path(base_model_id, registry)
                adapter_path   = meta["path"]
            else:
                true_base_path = meta["path"]
                adapter_path   = None

            if not os.path.exists(true_base_path):
                raise FileNotFoundError(f"Base model path missing: {true_base_path}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=true_base_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,           # auto: bfloat16 on Ampere+
                load_in_4bit=True,
            )

            # Apply Qwen3 instruct chat template to tokenizer
            tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

            # If continuing from a trained model, merge its adapter first
            if adapter_path:
                self.training_status = {
                    "status": "loading", "progress": 25,
                    "message": f"Merging adapter from '{base_model_id}'..."
                }
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                model = model.merge_and_unload()

            # Add fresh LoRA adapters for this training run
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=LORA_RANK,   # alpha = rank is standard
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            self.training_status = {
                "status": "preparing", "progress": 35,
                "message": "Loading dataset..."
            }

            dataset = load_dataset("json", data_files=dataset_path, split="train")

            self.training_status = {
                "status": "training", "progress": 40,
                "message": f"Training on {len(dataset)} examples — ctx {MAX_SEQ_LENGTH}, rank {LORA_RANK}..."
            }

            class ProgressCallback(TrainerCallback):
                def __init__(self_cb, total_steps):
                    self_cb.total_steps = total_steps

                def on_log(self_cb, args, state, control, logs=None, **kwargs):
                    if state.global_step and self_cb.total_steps:
                        pct  = 40 + int(state.global_step / self_cb.total_steps * 50)
                        loss = logs.get("loss", "?") if logs else "?"
                        self.training_status = {
                            "status": "training",
                            "progress": pct,
                            "message": f"Step {state.global_step}/{self_cb.total_steps} — loss: {loss}"
                        }

            # Pre-apply chat template to entire dataset before training
            # This avoids all batching/indexing issues with formatting_func
            def apply_template(examples):
                texts = []
                for msgs in examples["messages"]:
                    text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(
                apply_template,
                batched=True,
                num_proc=2,
                desc="Applying chat template",
            )

            sft_config = SFTConfig(
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                warmup_steps=10,
                max_steps=MAX_STEPS,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
                save_strategy="no",
                max_seq_length=MAX_SEQ_LENGTH,
                dataset_num_proc=2,
                dataset_text_field="text",
                packing=True,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=sft_config,
                callbacks=[ProgressCallback(MAX_STEPS)],
            )

            trainer.train()

            self.training_status = {
                "status": "saving", "progress": 92,
                "message": "Saving trained model..."
            }

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
                json.dump({
                    "base_model_id": base_model_id,
                    "trained_at": datetime.now().isoformat(),
                    "dataset_path": dataset_path,
                }, f, indent=2)

            self._register_model(output_name, output_dir, "trained",
                                  parent_id=base_model_id)

            self.training_status = {
                "status": "completed", "progress": 100,
                "message": f"Training complete! Model saved as '{output_name}'"
            }

        except Exception as e:
            self.training_status = {
                "status": "error", "progress": 0, "message": str(e)
            }
            raise


    # ─── Delete model ──────────────────────────────────────────────────────────

    def delete_model(self, model_id: str):
        registry = self._load_registry()
        if model_id not in registry["models"]:
            raise FileNotFoundError(f"Model '{model_id}' not in registry")

        meta = registry["models"][model_id]
        if not meta.get("deletable", True):
            raise ValueError("Cannot delete the base model")

        if self.loaded_chat_model_id == model_id:
            self._chat_model     = None
            self._chat_tokenizer = None
            self.loaded_chat_model_id = None

        path = meta["path"]
        if os.path.exists(path):
            shutil.rmtree(path)

        del registry["models"][model_id]
        self._save_registry(registry)

    # ─── Chat ─────────────────────────────────────────────────────────────────

    def load_chat_model(self, model_id: str):
        """
        Load a model for inference.
        Trained model: load base weights first, then apply LoRA adapter.
        Base model: load directly.
        """
        if self.loaded_chat_model_id == model_id:
            return  # already loaded

        from unsloth import FastLanguageModel

        registry = self._load_registry()
        if model_id not in registry["models"]:
            raise FileNotFoundError(f"Model '{model_id}' not found in registry")

        meta = registry["models"][model_id]

        # Free any previously loaded model from GPU memory
        self._chat_model     = None
        self._chat_tokenizer = None

        if meta["type"] == "trained":
            true_base_path = self._resolve_base_path(model_id, registry)
            adapter_path   = meta["path"]

            if not os.path.exists(true_base_path):
                raise FileNotFoundError(f"Base model path missing: {true_base_path}")
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter path missing: {adapter_path}")

            # Step 1: load base weights
            self._chat_model, self._chat_tokenizer = FastLanguageModel.from_pretrained(
                model_name=true_base_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )

            # Step 2: apply LoRA adapter on top
            from peft import PeftModel
            self._chat_model = PeftModel.from_pretrained(
                self._chat_model,
                adapter_path
            )

        else:
            # Base model — load directly
            if not os.path.exists(meta["path"]):
                raise FileNotFoundError(f"Base model path missing: {meta['path']}")

            self._chat_model, self._chat_tokenizer = FastLanguageModel.from_pretrained(
                model_name=meta["path"],
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )

        FastLanguageModel.for_inference(self._chat_model)
        self.loaded_chat_model_id = model_id

    def chat(self, message: str, model_id: str, history: list = []) -> str:
        if self._chat_model is None or self.loaded_chat_model_id != model_id:
            self.load_chat_model(model_id)

        # Build messages with history (last 6 turns)
        messages = []
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({
                    "role":    turn["role"],
                    "content": turn["content"]
                })
        messages.append({"role": "user", "content": message})

        # Apply Qwen3 chat template
        text = self._chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False   # set True to enable Qwen3 chain-of-thought mode
        )

        inputs = self._chat_tokenizer(text, return_tensors="pt").to("cuda")

        outputs = self._chat_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self._chat_tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens, not the full prompt
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()