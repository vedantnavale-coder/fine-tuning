import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import json

class ModelTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.training_status = {"status": "idle", "progress": 0, "message": ""}
        os.makedirs(model_dir, exist_ok=True)
    
    def load_base_model(self):
        """Load Qwen 4B with Unsloth optimizations"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-4B-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    
    def train(self, dataset_path: str, output_dir: str = None):
        """Fine-tune model on dataset"""
        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "qwen_pdf_model")
        
        self.training_status = {"status": "loading", "progress": 10, "message": "Loading model..."}
        
        # Load model
        self.load_base_model()
        
        self.training_status = {"status": "preparing", "progress": 20, "message": "Loading dataset..."}
        
        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        self.training_status = {"status": "training", "progress": 30, "message": "Training started..."}
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            max_steps=100,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=50,
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=True,
            args=training_args,
        )
        
        # Train
        trainer.train()
        
        self.training_status = {"status": "saving", "progress": 90, "message": "Saving model..."}
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.training_status = {"status": "completed", "progress": 100, "message": "Training completed!"}
        
        return output_dir
    
    def get_status(self):
        """Get current training status"""
        return self.training_status
