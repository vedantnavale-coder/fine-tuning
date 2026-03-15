import fitz
import os
import re
from typing import List


class PDFProcessor:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            text = re.sub(r'\n{3,}', '\n\n', text)
            pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
        """Chunk text with overlap to preserve context across chunks."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk.strip()) >= 100:
                chunks.append(chunk.strip())
            i += chunk_size - overlap
        return chunks

    def list_pdfs(self) -> List[str]:
        if not os.path.exists(self.upload_dir):
            return []
        # Return both PDFs and JSONL files so the UI can list and select them
        return sorted(
            f for f in os.listdir(self.upload_dir)
            if f.endswith('.pdf') or f.endswith('.jsonl')
        )