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
]


class DatasetBuilder:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)

    def create_training_dataset(self, chunks: List[str],
                                 output_file: str = "training_data.jsonl") -> str:
        output_path = os.path.join(self.dataset_dir, output_file)
        written = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 80:
                    continue
                instruction = random.choice(INSTRUCTION_TEMPLATES)
                text = (
                    f"### Instruction:\n{instruction}\n"
                    f"### Response:\n{chunk}"
                )
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                written += 1
        return output_path