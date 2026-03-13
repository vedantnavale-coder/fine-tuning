#!/bin/bash

echo "========================================"
echo "AI SERVER SETUP STARTING"
echo "========================================"

# Update system
echo "Updating Ubuntu..."
sudo apt update -y
sudo apt upgrade -y

# Install system packages
echo "Installing system dependencies..."
sudo apt install -y \
git \
curl \
wget \
build-essential \
python3 \
python3-venv \
python3-pip \
python3-dev \
htop \
nvtop

# Create project directory
echo "Creating project folder..."
mkdir -p ~/ai_project
cd ~/ai_project

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo "========================================"
echo "Installing CUDA-enabled PyTorch"
echo "========================================"

pip install torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu128

echo "========================================"
echo "Installing AI libraries"
echo "========================================"

pip install \
fastapi==0.109.0 \
uvicorn[standard]==0.27.0 \
python-multipart==0.0.6 \
pymupdf==1.23.8 \
transformers>=4.40.0 \
trl>=0.8.0 \
peft>=0.10.0 \
accelerate>=0.28.0 \
bitsandbytes>=0.42.0 \
datasets>=2.16.1 \
huggingface_hub>=0.21.0 \
xformers>=0.0.24

echo "========================================"
echo "Installing Unsloth"
echo "========================================"

pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

echo "========================================"
echo "Installing Flash Attention (optional)"
echo "========================================"

pip install flash-attn --no-build-isolation || true

echo "========================================"
echo "Testing CUDA"
echo "========================================"

python3 - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "========================================"
echo "Creating project folders"
echo "========================================"

mkdir -p uploads
mkdir -p dataset
mkdir -p models
mkdir -p backend
mkdir -p frontend

echo "========================================"
echo "INSTALLATION COMPLETE"
echo "========================================"

echo "Activate environment with:"
echo "source ~/ai_project/venv/bin/activate"

echo "Run server with:"
echo "uvicorn backend.app:app --host 0.0.0.0 --port 8000"