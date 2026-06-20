#!/bin/bash
# FedASIO-YOLO26: Run Streamlit Clinical Dashboard in Terminal
# Usage: bash run_dashboard.sh

set -e

# Load conda setup
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "🧠 Activating 'fedasio' conda environment..."
conda activate fedasio

echo "🚀 Starting Streamlit Clinical Dashboard..."
streamlit run streamlit_app/app.py
