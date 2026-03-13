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
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    file_path = os.path.join(pdf_processor.upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/pdfs")
async def list_pdfs():
    pdfs = pdf_processor.list_pdfs()
    return {"pdfs": pdfs}

@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    path = os.path.join(pdf_processor.upload_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    raise HTTPException(404, "PDF not found")

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
    global training_thread
    if training_thread and training_thread.is_alive():
        raise HTTPException(400, "Training already in progress")
    if not request.pdfs:
        raise HTTPException(400, "No PDFs selected")
    
    def train_task():
        try:
            model_manager.training_status = {
                "status": "extracting", "progress": 5,
                "message": f"Extracting text from {len(request.pdfs)} PDF(s)..."
            }
            all_chunks = []
            for pdf_name in request.pdfs:
                pdf_path = os.path.join(pdf_processor.upload_dir, pdf_name)
                if os.path.exists(pdf_path):
                    text = pdf_processor.extract_text(pdf_path)
                    chunks = pdf_processor.chunk_text(text)
                    all_chunks.extend(chunks)
            
            model_manager.training_status = {
                "status": "building_dataset", "progress": 15,
                "message": f"Building dataset from {len(all_chunks)} chunks..."
            }
            dataset_path = dataset_builder.create_training_dataset(all_chunks)
            
            model_manager.train_model(
                dataset_path=dataset_path,
                base_model_id=request.base_model_id,
                output_name=request.output_model_name
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