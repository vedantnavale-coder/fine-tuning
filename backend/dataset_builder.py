import json
import os
import random
import shutil
from typing import List


INSTRUCTION_TEMPLATES = [
    "Explain the following content from the document:",
    "Summarize this passage:",
    "What does this section describe?",
    "Provide an analysis of the following text:",
    "What key information is contained in this passage?",
    "Describe what this document section covers:",
    "What is the main point of this text?",
    "Break down the following content:",
]




class DatasetBuilder:
    def __init__(self, dataset_dir: str = "dataset", min_chunk_length: int = 80, seed: int | None = None):
        self.dataset_dir = dataset_dir
        self.min_chunk_length = min_chunk_length
        if seed is not None:
            random.seed(seed)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def _clean_chunk(self, chunk: str) -> str:
        return " ".join(chunk.strip().split())

    def _is_valid_chunk(self, chunk: str) -> bool:
        return len(chunk) >= self.min_chunk_length

    def _build_message(self, chunk: str) -> dict:
        instruction = random.choice(INSTRUCTION_TEMPLATES)
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": chunk},
            ]
        }

    def create_training_dataset(
        self,
        chunks: List[str],
        output_file: str = "training_data.jsonl",
        shuffle: bool = True,
        remove_duplicates: bool = True,
    ) -> str:
        if not chunks:
            raise ValueError("No chunks provided to dataset builder.")

        output_path = os.path.join(self.dataset_dir, output_file)
        cleaned_chunks = [self._clean_chunk(c) for c in chunks if c]
        if remove_duplicates:
            cleaned_chunks = list(dict.fromkeys(cleaned_chunks))
        cleaned_chunks = [c for c in cleaned_chunks if self._is_valid_chunk(c)]
        if shuffle:
            random.shuffle(cleaned_chunks)

        written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in cleaned_chunks:
                message = self._build_message(chunk)
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
                written += 1

        if written == 0:
            raise RuntimeError("Dataset creation failed — no valid chunks written.")
        return output_path

    # ── NEW METHOD ─────────────────────────────────────────────────────────────
    def import_jsonl(self, source_path: str, output_file: str = "training_data.jsonl") -> str:
        """
        Validate and copy a pre-built JSONL file into the dataset directory.
        Every line must have a 'messages' key with at least one message.
        Returns the internal dataset path ready for training.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"JSONL file not found: {source_path}")

        output_path = os.path.join(self.dataset_dir, output_file)
        valid_lines = 0
        errors = []

        with open(source_path, "r", encoding="utf-8") as src, \
             open(output_path, "w", encoding="utf-8") as dst:
            for i, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" not in obj:
                        errors.append(f"Line {i}: missing 'messages' key")
                        continue
                    if not isinstance(obj["messages"], list) or len(obj["messages"]) == 0:
                        errors.append(f"Line {i}: 'messages' must be a non-empty list")
                        continue
                    # Strip non-training metadata fields (label, error_type, etc.)
                    clean = {"messages": obj["messages"]}
                    dst.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: JSON parse error — {e}")

        if valid_lines == 0:
            raise RuntimeError(
                f"No valid samples found in JSONL file. Errors: {errors[:5]}"
            )

        if errors:
            # Log warnings but don't fail — partial datasets are fine
            print(f"[DatasetBuilder] import_jsonl: {len(errors)} lines skipped, "
                  f"{valid_lines} valid. First errors: {errors[:3]}")

        return output_path
    def import_jsonl(self, source_path: str, output_file: str = "training_data.jsonl") -> str:
        """Validate and copy a pre-built JSONL into the dataset dir."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"JSONL file not found: {source_path}")

        output_path = os.path.join(self.dataset_dir, output_file)
        valid_lines = 0
        errors = []

        with open(source_path, "r", encoding="utf-8") as src, \
            open(output_path, "w", encoding="utf-8") as dst:
            for i, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" not in obj:
                        errors.append(f"Line {i}: missing 'messages' key")
                        continue
                    if not isinstance(obj["messages"], list) or len(obj["messages"]) == 0:
                        errors.append(f"Line {i}: 'messages' must be a non-empty list")
                        continue
                    # Strip non-training metadata (label, error_type, cache_type, etc.)
                    clean = {"messages": obj["messages"]}
                    dst.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: JSON error — {e}")

        if valid_lines == 0:
            raise RuntimeError(f"No valid samples in JSONL. Errors: {errors[:5]}")

        if errors:
            print(f"[DatasetBuilder] {len(errors)} lines skipped, {valid_lines} valid.")

        return output_path