# Configuration for Local AI Training Platform

# Server Settings
HOST = "0.0.0.0"
PORT = 8000

# Directories
UPLOAD_DIR = "uploads"
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_OUTPUT_DIR = "models/qwen_pdf_model"

# Model Settings
BASE_MODEL = "unsloth/Qwen2.5-4B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA Settings
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training Settings
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_TRAINING_STEPS = 100
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10

# PDF Processing
CHUNK_SIZE = 800  # Approximate tokens per chunk
MIN_CHUNK_LENGTH = 50  # Minimum characters to include chunk

# Chat Settings
MAX_RESPONSE_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
