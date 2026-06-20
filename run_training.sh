#!/bin/bash
# FedASIO-YOLO26: Run Phase 2 FL Training in Terminal
# Usage: bash run_training.sh

set -e

# Load conda setup
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "🧠 Activating 'fedasio' conda environment..."
conda activate fedasio

echo "🔗 Starting Phase 2 Federated Training..."
python scripts/phase2_full_fl.py
