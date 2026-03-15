from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
import threading
import json
import time
from typing import Optional
from pdf_processor import PDFProcessor
from dataset_builder import DatasetBuilder
from model_manager import ModelManager

app = FastAPI(title="AI Training Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
dataset_builder = DatasetBuilder()
model_manager = ModelManager()

# Global thread tracking
training_thread = None
download_thread = None

class ChatRequest(BaseModel):
    message: str
    model_id: str = "trained"
    history: list = []

class TrainRequest(BaseModel):
    pdfs: list[str]
    base_model_id: str = "base"
    output_model_name: str = "trained_v1"

class DeleteModelRequest(BaseModel):
    model_id: str

# ─── Root ───────────────────────────────────────────────────────────────────

# ─── PDF endpoints ───────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts both PDF and JSONL files.
    PDFs go to uploads/
    JSONL files go to uploads/ too — same folder, detected by extension at train time.
    """
    filename = file.filename

    if filename.endswith('.pdf'):
        file_path = os.path.join(pdf_processor.upload_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": filename, "status": "uploaded", "type": "pdf"}

    elif filename.endswith('.jsonl'):
        file_path = os.path.join(pdf_processor.upload_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Count lines for feedback
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for l in f if l.strip())
        return {"filename": filename, "status": "uploaded", "type": "jsonl", "lines": line_count}

    else:
        raise HTTPException(400, "Only PDF or JSONL files allowed")


@app.get("/api/pdfs")
async def list_pdfs():
    """Lists both PDFs and JSONL files in the uploads folder."""
    pdfs = pdf_processor.list_pdfs()
    return {"pdfs": pdfs}


@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    path = os.path.join(pdf_processor.upload_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    raise HTTPException(404, "File not found")

# ─── Model management endpoints ──────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    return {"models": model_manager.list_models()}

@app.get("/api/models/download-status")
async def download_status():
    return model_manager.get_download_status()

@app.post("/api/models/download-base")
async def download_base_model():
    global download_thread
    if download_thread and download_thread.is_alive():
        raise HTTPException(400, "Download already in progress")

    if model_manager.base_model_exists():
        return {"status": "exists", "message": "Base model already downloaded"}

    download_thread = threading.Thread(target=model_manager.download_base_model)
    download_thread.start()
    return {"status": "started", "message": "Downloading Qwen3-4B..."}

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    try:
        model_manager.delete_model(model_id)
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")

# ─── Training endpoints ───────────────────────────────────────────────────────

@app.post("/api/train")
async def start_training(request: TrainRequest):
    """
    Same endpoint, same request body as before.

    Logic change:
    - If any file in request.pdfs ends with .jsonl → use import_jsonl() path directly.
    - If all files are PDFs → extract text and chunk as before.
    - Mixed (PDFs + JSONL) → extract PDF chunks AND merge JSONL samples together.
    """
    global training_thread
    if training_thread and training_thread.is_alive():
        raise HTTPException(400, "Training already in progress")
    if not request.pdfs:
        raise HTTPException(400, "No files selected")

    def train_task():
        try:
            # Split file list into PDFs and JSONLs
            pdf_files   = [f for f in request.pdfs if f.endswith('.pdf')]
            jsonl_files = [f for f in request.pdfs if f.endswith('.jsonl')]

            dataset_path = None

            # ── Case 1: JSONL only ────────────────────────────────────────────
            if jsonl_files and not pdf_files:
                if len(jsonl_files) == 1:
                    # Single JSONL — import directly
                    model_manager.training_status = {
                        "status": "preparing", "progress": 5,
                        "message": f"Importing JSONL dataset: {jsonl_files[0]}..."
                    }
                    source = os.path.join(pdf_processor.upload_dir, jsonl_files[0])
                    dataset_path = dataset_builder.import_jsonl(source)
                else:
                    # Multiple JSONLs — merge them first
                    model_manager.training_status = {
                        "status": "preparing", "progress": 5,
                        "message": f"Merging {len(jsonl_files)} JSONL files..."
                    }
                    merged_path = os.path.join(dataset_builder.dataset_dir, "merged.jsonl")
                    total = 0
                    with open(merged_path, "w", encoding="utf-8") as out:
                        for jf in jsonl_files:
                            source = os.path.join(pdf_processor.upload_dir, jf)
                            with open(source, "r", encoding="utf-8") as inp:
                                for line in inp:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        obj = json.loads(line)
                                        if "messages" in obj:
                                            clean = {"messages": obj["messages"]}
                                            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                                            total += 1
                                    except json.JSONDecodeError:
                                        pass
                    if total == 0:
                        raise RuntimeError("No valid samples found in any JSONL file.")
                    dataset_path = merged_path

            # ── Case 2: PDFs only (original behavior) ─────────────────────────
            elif pdf_files and not jsonl_files:
                model_manager.training_status = {
                    "status": "extracting", "progress": 5,
                    "message": f"Extracting text from {len(pdf_files)} PDF(s)..."
                }
                all_chunks = []
                for pdf_name in pdf_files:
                    pdf_path = os.path.join(pdf_processor.upload_dir, pdf_name)
                    if os.path.exists(pdf_path):
                        text   = pdf_processor.extract_text(pdf_path)
                        chunks = pdf_processor.chunk_text(text)
                        all_chunks.extend(chunks)

                model_manager.training_status = {
                    "status": "building_dataset", "progress": 15,
                    "message": f"Building dataset from {len(all_chunks)} chunks..."
                }
                dataset_path = dataset_builder.create_training_dataset(all_chunks)

            # ── Case 3: Mixed PDFs + JSONL ────────────────────────────────────
            else:
                model_manager.training_status = {
                    "status": "extracting", "progress": 5,
                    "message": f"Processing {len(pdf_files)} PDF(s) and {len(jsonl_files)} JSONL file(s)..."
                }

                # Extract PDF chunks and write as JSONL samples
                all_chunks = []
                for pdf_name in pdf_files:
                    pdf_path = os.path.join(pdf_processor.upload_dir, pdf_name)
                    if os.path.exists(pdf_path):
                        text   = pdf_processor.extract_text(pdf_path)
                        chunks = pdf_processor.chunk_text(text)
                        all_chunks.extend(chunks)

                # Write PDF-derived samples to a temp file
                merged_path = os.path.join(dataset_builder.dataset_dir, "merged.jsonl")
                written = 0
                with open(merged_path, "w", encoding="utf-8") as out:
                    # PDF chunks first
                    for chunk in all_chunks:
                        msg = dataset_builder._build_message(chunk)
                        out.write(json.dumps(msg, ensure_ascii=False) + "\n")
                        written += 1
                    # Then JSONL samples
                    for jf in jsonl_files:
                        source = os.path.join(pdf_processor.upload_dir, jf)
                        with open(source, "r", encoding="utf-8") as inp:
                            for line in inp:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if "messages" in obj:
                                        clean = {"messages": obj["messages"]}
                                        out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                                        written += 1
                                except json.JSONDecodeError:
                                    pass

                if written == 0:
                    raise RuntimeError("No valid training samples produced.")
                dataset_path = merged_path

            # ── Count samples and kick off training ───────────────────────────
            with open(dataset_path) as f:
                sample_count = sum(1 for l in f if l.strip())

            model_manager.training_status = {
                "status": "building_dataset", "progress": 15,
                "message": f"Dataset ready: {sample_count} samples. Starting training..."
            }

            model_manager.train_model(
                dataset_path=dataset_path,
                base_model_id=request.base_model_id,
                output_name=request.output_model_name,
            )

        except Exception as e:
            model_manager.training_status = {
                "status": "error", "progress": 0, "message": str(e)
            }

    training_thread = threading.Thread(target=train_task)
    training_thread.start()
    return {"status": "started"}


@app.get("/api/training/status")
async def get_training_status():
    return model_manager.training_status

# ─── Chat endpoints ───────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = model_manager.chat(
            message=request.message,
            model_id=request.model_id,
            history=request.history
        )
        return {"response": response}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/chat/load/{model_id}")
async def load_chat_model(model_id: str):
    try:
        model_manager.load_chat_model(model_id)
        return {"status": "loaded", "model_id": model_id}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/chat/loaded")
async def get_loaded_model():
    return {"loaded_model": model_manager.loaded_chat_model_id}

# Mount frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)