# ⬡ NeuralForge — Local AI Training Platform

Fine-tune **Qwen3-4B** on your own PDF documents, locally, with a clean browser UI.

## Features

- **Auto-download Qwen3-4B** — checks if the model is already local; downloads if not
- **Iterative training** — train on the base model, then train *that* model again with new data
- **Model registry** — see all your models, their sizes, parent lineage; delete trained ones freely
- **Base model is always protected** — can never be accidentally deleted
- **Full chat interface** — load any model, set temperature, maintain conversation history
- **PDF management** — upload, list, delete documents per-session

---

## Requirements

- Python 3.10+
- CUDA GPU (RTX 3060+ recommended, 10+ GB VRAM for 4-bit)
- ~12 GB disk space for the base model

## Setup

```bash
# 1. Clone / place this folder anywhere
cd neuralforge

# 2. Install dependencies
pip install -r backend/requirements.txt

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. Start the server
bash start.sh
# or directly:
cd backend && python server.py

# 4. Open http://localhost:8000
```

---

## Workflow

### 1. Setup Tab
- Click **CHECK / DOWNLOAD** — if Qwen3-4B isn't found locally it downloads automatically (~8 GB)
- Upload your PDF documents

### 2. Train Tab
- Choose **source model**: `base` for first run, or any previously trained model for iterative training
- Give your output model a name (e.g. `legal_v1`, `finance_v2`)
- Select PDFs to train on → **START TRAINING**

### 3. Models Tab
- See all registered models with size, creation date, parent lineage
- Delete trained models when done — **base model cannot be deleted**

### 4. Chat Tab
- Select a model from the dropdown → **LOAD INTO MEMORY**
- Chat with conversation history, adjustable temperature

---

## Iterative Training Example

```
base  →  trained_v1 (trained on contracts.pdf)
            ↓
         trained_v2 (trained on more_contracts.pdf)
            ↓
         trained_v3 (trained on case_law.pdf)

Delete trained_v1 and trained_v2 when satisfied, keep base + trained_v3.
```

---

## Project Structure

```
neuralforge/
├── backend/
│   ├── server.py          # FastAPI app
│   ├── model_manager.py   # Download, train, chat, registry
│   ├── pdf_processor.py   # PDF text extraction + chunking
│   ├── dataset_builder.py # JSONL dataset creation
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                # Created at runtime
│   └── registry.json
├── uploads/               # PDF storage
├── dataset/               # Training JSONL files
└── start.sh
```