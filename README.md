# FedASIO-YOLO26 🧠
## Privacy-Preserving Federated Brain Tumor Instance Segmentation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch 2.14 MPS](https://img.shields.io/badge/PyTorch-2.14%20MPS-orange)](https://pytorch.org)
[![Flower FL 1.31](https://img.shields.io/badge/Flower-1.31-green)](https://flower.ai)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.5-purple)](https://langchain-ai.github.io/langgraph)
[![Ultralytics YOLO26](https://img.shields.io/badge/YOLO26--Seg-8.4.68-red)](https://ultralytics.com)

---

### Overview

**FedASIO-YOLO26** is a Q1-grade research implementation of a federated multi-agent brain tumor segmentation framework for pediatric MRI (BraTS-PEDs-v1).

| Component | Technology |
|-----------|-----------|
| **Segmentation** | YOLO26-Seg (309 layers, 3M params, 10.2 GFLOPs) |
| **Optimizer** | ASIO — Asteroid Satellite Inspired Optimization (novel) |
| **Federated Learning** | Flower 1.31 (Virtual Client Engine) |
| **Aggregation** | FedASIO (Dice-weighted), FedAvg, FedProx, FedAdam |
| **Privacy** | Opacus DP-SGD, ε=1.0, δ=1e-5, Rényi DP |
| **Multi-Agent** | LangGraph StateGraph (8 agents) |
| **Hardware** | Apple M5 · 16GB RAM · MPS GPU |
| **Dataset** | BraTS-PEDs-v1 (257 train + 91 val patients, 33 GB) |

---

### Quick Start

```bash
# 1. Setup environment (one-time)
bash setup_environment.sh

# 2. Activate environment
conda activate fedasio

# 3. Phase 1: 5-patient sample (quick sanity check ~15 min)
python scripts/phase1_sample.py

# 4. Phase 2: Full FL training (25% train / 5% val / 5% test)
python scripts/phase2_full_fl.py

# 5. Launch Streamlit clinical dashboard
streamlit run streamlit_app/app.py
```

---

### Dataset Splits

| Split | Patients | Purpose |
|-------|---------|---------|
| **Train** | 65 (25%) | Federated training — 5 clients × 13 patients |
| **Validation** | 13 (5%) | Per-round FL evaluation |
| **Test** | 13 (5%) | Final held-out evaluation |
| **Streamlit** | 166 (65%) | Live inference demonstration |

---

### Architecture

```
BraTS-PEDs-v1 NIfTI Data
        │
        ▼
┌─────────────────────────────────────────┐
│          LangGraph Multi-Agent Pipeline │
│  DataAgent → PreprocessAgent →          │
│  AugmentAgent → ASIOAgent →             │
│  SegmentAgent → EvalAgent →             │
│  XAIAgent → ReportAgent                 │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   YOLO26-Seg        │  (Segment26 head, MPS)
        │   + ASIO Optimizer  │  (6D HP, real Dice fitness)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Flower FL (flwr)   │
        │  5 Virtual Clients  │  (FedASIO aggregation)
        │  + DP-SGD Privacy   │  (ε=1.0, δ=1e-5)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Streamlit Dashboard │  + PDF Clinical Reports
        └─────────────────────┘
```

---

### ASIO Algorithm

The **Asteroid Satellite Inspired Optimization (ASIO)** is a novel topology-adaptive swarm algorithm:

- **Asteroids**: Top-k particles → fine local search via orbital attraction
- **Satellites**: Remaining → orbit nearest Asteroid via PSO-like velocity
- **Perturbation Jump**: Probabilistic escape (Kozai-Lidov mechanism)  
- **6D Search Space**: `{lr0, momentum, weight_decay, conf, iou, batch_size}`
- **Real Fitness**: YOLO validation mAP50 (not simulated)

---

### Project Structure

```
FedASIO-YOLO26/
├── agents/              # LangGraph multi-agent system
├── federated/           # Flower FL (server, client, strategy, privacy)
├── models/asio/         # ASIO optimizer
├── scripts/             # Phase 1 & 2 entry points
├── streamlit_app/       # Clinical dashboard
├── data/                # Processed YOLO slices + splits
├── reports/             # Figures, metrics, PDF reports
└── config/config.yaml   # Central configuration
```

---

### Hardware Requirements

- **CPU**: Apple M5 (tested) or any ARM64/x86_64
- **RAM**: 16 GB minimum (8 GB available for training)
- **GPU**: MPS (Apple Silicon) — auto-detected
- **Storage**: 50 GB free (dataset 33 GB + model checkpoints)

---

### Citation (Research Paper)

```bibtex
@article{fedasio_yolo26_2026,
  title   = {FedASIO-YOLO26: A Privacy-Preserving Federated Multi-Agent Framework 
             for Real-Time Brain Tumor Instance Segmentation},
  author  = {Your Name},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review}
}
```

---

**⚠️ Research Prototype — NOT for clinical use.**
