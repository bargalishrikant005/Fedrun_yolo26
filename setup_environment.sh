#!/bin/bash
# FedASIO-YOLO26 — One-Click Environment Setup for Apple M5 (MPS)
# Usage: bash setup_environment.sh

set -e
echo "═══════════════════════════════════════════════════════════"
echo "  FedASIO-YOLO26 Environment Setup — Apple M5 MPS"
echo "═══════════════════════════════════════════════════════════"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Install Anaconda from https://www.anaconda.com"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Create environment
echo "📦 Creating conda environment 'fedasio' (Python 3.11)..."
conda create -n fedasio python=3.11 -y
conda activate fedasio

# Install PyTorch nightly with MPS support
echo "🔥 Installing PyTorch with MPS (Apple Silicon GPU) support..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify MPS
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available!'; print('✅ MPS GPU: Active')"

# Install all project dependencies
echo "📚 Installing project libraries..."
pip install \
    ultralytics \
    flwr \
    langgraph \
    langchain \
    langchain-community \
    nibabel \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    reportlab \
    albumentations \
    pydicom \
    opacus \
    pyyaml \
    tqdm \
    wandb \
    requests \
    huggingface_hub \
    transformers \
    accelerate \
    streamlit \
    pandas \
    Pillow \
    opencv-python-headless

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Environment 'fedasio' ready!"
echo ""
echo "Usage:"
echo "  conda activate fedasio"
echo "  python scripts/phase1_sample.py    # Phase 1: 5-patient sample"
echo "  python scripts/phase2_full_fl.py   # Phase 2: Full FL training"
echo "  streamlit run streamlit_app/app.py  # Launch dashboard"
echo "═══════════════════════════════════════════════════════════"
