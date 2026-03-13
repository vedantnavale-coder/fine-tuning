import json
import os
import random
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
    """
    Builds a Qwen-compatible JSONL dataset from text chunks.
    """

    def __init__(self, dataset_dir: str = "dataset", min_chunk_length: int = 80, seed: int | None = None):
        self.dataset_dir = dataset_dir
        self.min_chunk_length = min_chunk_length

        if seed is not None:
            random.seed(seed)

        os.makedirs(self.dataset_dir, exist_ok=True)

    def _clean_chunk(self, chunk: str) -> str:
        """Normalize and clean chunk text."""
        return " ".join(chunk.strip().split())

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Check if chunk is long enough to be useful."""
        return len(chunk) >= self.min_chunk_length

    def _build_message(self, chunk: str) -> dict:
        """Create Qwen chat-format message."""
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
        """
        Convert text chunks into Qwen training JSONL dataset.
        """

        if not chunks:
            raise ValueError("No chunks provided to dataset builder.")

        output_path = os.path.join(self.dataset_dir, output_file)

        # Clean chunks
        cleaned_chunks = [self._clean_chunk(c) for c in chunks if c]

        # Remove duplicates
        if remove_duplicates:
            cleaned_chunks = list(dict.fromkeys(cleaned_chunks))

        # Filter short chunks
        cleaned_chunks = [c for c in cleaned_chunks if self._is_valid_chunk(c)]

        # Shuffle for better training distribution
        if shuffle:
            random.shuffle(cleaned_chunks)

        written = 0

        with open(output_path, "w", encoding="utf-8") as f:
            write = f.write

            for chunk in cleaned_chunks:
                message = self._build_message(chunk)
                write(json.dumps(message, ensure_ascii=False) + "\n")
                written += 1

        if written == 0:
            raise RuntimeError("Dataset creation failed — no valid chunks written.")

        return output_path