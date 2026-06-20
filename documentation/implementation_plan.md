# FedASIO-YOLO26 Implementation Plan
## Privacy-Preserving Federated Multi-Agent Brain Tumor Segmentation

> **Dataset:** BraTS-PEDs-v1 (Pediatric, 257 training + 91 validation patients, 33 GB)  
> **Hardware:** MacBook Air Apple M5 · 16 GB RAM · MPS (Metal Performance Shaders) GPU  
> **Stack:** YOLO26-Seg · ASIO Optimizer · LangGraph Multi-Agent · Flower (flwr) FL · PyTorch MPS  

---

## Dataset Overview (Confirmed)

| Split | Patients | Files per Patient | Modalities |
|-------|---------|-----------------|-----------|
| **Training** | 257 | 5 NIfTI (.nii.gz) | t1c, t1n, t2f, t2w + seg |
| **Validation** | 91 | 4 NIfTI (.nii.gz) | t1c, t1n, t2f, t2w (no seg labels) |
| **Total** | 348 | — | — |
| **Disk** | 33 GB | — | — |

**Naming convention:** `BraTS-PED-XXXXX-000-{t1c|t1n|t2f|t2w|seg}.nii.gz`

---

## Data Split Strategy

```
257 Training patients total:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (SAMPLE): 5 patients → quick sanity check + graphs
Phase 2 (FULL):   257 patients split as:
  • TRAIN:      25% = 65 patients  → Federated client training
  • TEST:        5% = 13 patients  → Post-training evaluation
  • VALIDATION:  5% = 13 patients  → During-training FL validation
  • STREAMLIT:  65% = 166 patients → Live inference demonstration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FL Clients: 5 simulated hospitals
  Client 1: 13 patients (train)
  Client 2: 13 patients (train)
  Client 3: 13 patients (train)
  Client 4: 13 patients (train)
  Client 5: 13 patients (train)
```

---

## Project Structure

```
/Users/shrikant/Downloads/FedASIO-YOLO26/
├── config/
│   └── config.yaml                   # Central configuration
│
├── data/
│   ├── raw/                          # Symlink → BraTS-PEDs-v1/Training
│   ├── processed/
│   │   ├── sample/                   # 5-patient processed slices
│   │   │   ├── images/train/
│   │   │   ├── labels/train/
│   │   │   └── data.yaml
│   │   └── full/                     # Full dataset processed
│   │       ├── train/ (65 patients)
│   │       ├── val/   (13 patients)
│   │       ├── test/  (13 patients)
│   │       └── streamlit/ (166 patients)
│   └── splits/
│       └── patient_splits.json       # Deterministic patient assignment
│
├── agents/                           # LangGraph multi-agent system
│   ├── __init__.py
│   ├── graph.py                      # LangGraph StateGraph definition
│   ├── state.py                      # AgentState TypedDict
│   ├── data_agent.py                 # NIfTI loader + BraTS-PEDs parser
│   ├── preprocessing_agent.py        # 3D→2D, normalize, slice extract
│   ├── augmentation_agent.py         # Medical augmentation (Albumentations)
│   ├── segmentation_agent.py         # YOLO26-Seg train + infer
│   ├── asio_agent.py                 # ASIO hyperparameter optimizer
│   ├── evaluation_agent.py           # Dice, IoU, mAP, HD95, F1
│   ├── xai_agent.py                  # GradCAM + confidence maps
│   ├── report_agent.py               # PDF clinical report generator
│   └── orchestrator.py               # LangGraph orchestration entry
│
├── federated/                        # Flower FL components
│   ├── __init__.py
│   ├── server.py                     # Flower server + ASIO aggregation
│   ├── client.py                     # Flower client (per hospital)
│   ├── strategy.py                   # Custom FedASIO strategy
│   ├── aggregation.py                # FedAvg / FedProx / ASIO-weighted
│   └── privacy.py                    # Differential Privacy (DP-SGD)
│
├── models/
│   ├── yolo26/
│   │   ├── yolo26n-seg.yaml          # YOLO26-Seg architecture config
│   │   └── weights/                  # Saved checkpoints
│   └── asio/
│       └── asio_optimizer.py         # ASIO algorithm (standalone)
│
├── scripts/
│   ├── phase1_sample.py              # Phase 1: 5-patient sample run
│   ├── phase2_full_fl.py             # Phase 2: Full federated training
│   ├── evaluate.py                   # Evaluation on test set
│   └── generate_graphs.py            # Publication-quality graphs
│
├── streamlit_app/
│   ├── app.py                        # Main clinical dashboard
│   ├── components/
│   │   ├── upload.py                 # DICOM/NIfTI/JPEG uploader
│   │   ├── segmentation_view.py      # Overlay + mask display
│   │   ├── metrics_panel.py          # Live metrics panel
│   │   └── xai_panel.py              # GradCAM heatmap display
│   └── pages/
│       ├── 01_Dashboard.py
│       ├── 02_Patient_Analysis.py
│       └── 03_Federated_Status.py
│
├── android_app/                      # React Native / Flutter app
│   └── README.md                     # Android app build instructions
│
├── reports/
│   ├── figures/                      # Publication graphs
│   ├── metrics/                      # CSV metric files
│   └── pdf_reports/                  # Clinical PDF reports
│
├── requirements.txt
├── setup_environment.sh              # One-click M5 environment setup
├── run_phase1.sh                     # Phase 1 launcher
└── run_phase2.sh                     # Phase 2 launcher
```

---

## Phase 1 — Environment Setup (Apple M5 MPS)

### Required Libraries

| Package | Version | Purpose | Install Method |
|---------|---------|---------|---------------|
| `torch` | 2.5+ (nightly) | PyTorch MPS backend | pip (Apple nightly) |
| `torchvision` | 0.20+ | Vision transforms | pip (Apple nightly) |
| `ultralytics` | 8.3+ | YOLO26-Seg engine | pip |
| `flwr` | 1.11+ | Flower Federated Learning | pip |
| `langgraph` | 0.2+ | Multi-agent orchestration | pip |
| `langchain` | 0.3+ | LLM chain support | pip |
| `nibabel` | 5.3+ | NIfTI file I/O | conda (already) |
| `numpy` | 2.x | Array operations | conda (already) |
| `pandas` | 2.x | DataFrames | conda (already) |
| `opencv-python-headless` | 4.x | Image processing | conda (already) |
| `scikit-learn` | 1.5+ | Metrics, splits | conda (already) |
| `scipy` | 1.13+ | Hausdorff distance | conda (already) |
| `streamlit` | 1.4+ | Clinical dashboard | conda (already) |
| `matplotlib` | 3.9+ | Publication figures | conda (already) |
| `seaborn` | 0.13+ | Statistical plots | conda (already) |
| `plotly` | 5.24+ | Interactive charts | conda (already) |
| `albumentations` | 2.x | Medical augmentation | conda (already) |
| `pydicom` | 2.4+ | DICOM file parsing | pip |
| `reportlab` | 4.x | PDF report generation | pip |
| `opacus` | 0.15+ | Differential Privacy DP-SGD | pip |
| `Pillow` | 10.x | Image I/O | pip |
| `tqdm` | 4.x | Progress bars | pip |
| `pyyaml` | 6.x | YAML config parsing | pip |
| `wandb` | 0.x | Experiment tracking | pip |

### MPS Acceleration Notes (M5 specific)
- Use `device = torch.device("mps")` for GPU acceleration
- Set `amp=False` for training (MPS stability)
- Set `workers=0` in YOLO DataLoader (macOS multiprocessing fix)
- PyTorch MPS supports: convolutions, batch norm, attention — all YOLO ops

---

## Phase 2 — Multi-Agent System Design (LangGraph)

### LangGraph State Definition

```python
# agents/state.py
class FedASIOState(TypedDict):
    patient_id: str
    modalities: dict          # {t1c, t1n, t2f, t2w} numpy arrays
    seg_mask: np.ndarray      # Ground truth segmentation
    processed_slices: list    # List of 2D slice dicts
    augmented_slices: list    # Post-augmentation
    predictions: dict         # YOLO inference results
    metrics: dict             # Dice, IoU, mAP, HD95, F1
    xai_maps: dict            # GradCAM heatmaps
    report_text: str          # Clinical narrative
    pdf_path: str             # Generated PDF
    fl_round: int             # Current FL round number
    hyperparams: dict         # ASIO-optimized HP set
    error: Optional[str]      # Error state for fault isolation
```

### LangGraph Agent Flow

```
START
  │
  ▼
DataAgent ──(load NIfTI)──────────────────────────────────────┐
  │                                                             │
  ▼                                                             │
PreprocessingAgent ──(3D→2D, normalize, extract tumor slices)──┤
  │                                                             │
  ▼                                                             │
AugmentationAgent ──(Albumentations medical transforms)─────────┤
  │                                                             │
  ▼                                                             │
ASIOAgent ──(optimize 6D hyperparameters per FL round)──────────┤
  │                                                             │
  ▼                                                             │
SegmentationAgent ──(YOLO26-Seg train/infer with ASIO HPs)──────┤
  │                                                             │
  ▼                                                             │
EvaluationAgent ──(Dice, IoU, mAP@50, mAP@50-95, HD95, F1)────┤
  │                                                             │
  ├──(if inference mode)────────────────────────────────────────┤
  ▼                                                             │
XAIAgent ──(GradCAM + confidence overlay)───────────────────────┤
  │                                                             │
  ▼                                                             │
ReportAgent ──(PDF generation + tumor volume calc)──────────────┘
  │
  ▼
END
```

---

## Phase 3 — Federated Learning Architecture (Flower)

### FL Setup

```
FEDERATED LEARNING SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Server (localhost aggregation):
  • FedASIO Strategy (custom Flower strategy)
  • ASIO: 5 particles, 6D HP space, 3 iterations/round
  • PSO: aggregation weight optimizer
  • Aggregators: FedAvg | FedProx | FedAdam | ASIO-weighted
  • Privacy: Opacus DP-SGD, ε=1.0, δ=1e-5, clip_norm=1.0

5 Simulated Clients (in-process via Flower Virtual Client Engine):
  Client 1: BraTS-PED patients 1–13   (Hospital A simulation)
  Client 2: BraTS-PED patients 14–26  (Hospital B simulation)
  Client 3: BraTS-PED patients 27–39  (Hospital C simulation)
  Client 4: BraTS-PED patients 40–52  (Hospital D simulation)
  Client 5: BraTS-PED patients 53–65  (Hospital E simulation)

Training Config (per client, per round):
  • local_epochs: 3
  • global_rounds: 50
  • model: YOLO26n-Seg
  • device: mps (M5 GPU)
  • batch_size: 8 (optimized for 16GB RAM)
  • img_size: 256×256
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### ASIO Integration in FL Rounds

```
FL Round r:
  1. Server runs ASIO (3 iterations, 5 particles)
     → ASIO fitness = mean(client val Dice scores from round r-1)
     → ASIO outputs: {lr, momentum, wd, conf, iou, batch}
  
  2. Server broadcasts: global_weights + asio_hyperparams
  
  3. Each client:
     a. Receives global model + HP config
     b. Runs local training with DP-SGD (opacus)
     c. Sends noisy gradients back (Gaussian mechanism)
  
  4. Server: PSO-weighted FedAvg on client gradients
  
  5. Evaluate global model on val set (13 patients)
     → Log: Dice, mAP, communication cost, ε used
```

---

## Phase 4 — Streamlit Clinical Dashboard

### Features
| Feature | Description |
|---------|------------|
| **MRI Upload** | DICOM (.dcm), NIfTI (.nii/.nii.gz), JPEG/PNG |
| **Model Selection** | YOLO26-ASIO, YOLO26-PSO, YOLO11-ASIO |
| **FL Round Tracker** | Live training progress panel |
| **Segmentation Overlay** | Green mask + boundary on uploaded image |
| **XAI Heatmap** | GradCAM overlay (JET colormap) |
| **Metrics Panel** | Dice, IoU, Precision, Recall, F1, HD95, Tumor Volume (cc) |
| **PDF Export** | One-click clinical report download |
| **Patient Browser** | Browse 166 streamlit-split patients |
| **Federated Dashboard** | Per-client Dice curves, communication cost, privacy budget |

---

## Phase 5 — Android App Architecture

### Technology Stack
- **Framework:** Flutter (Dart) — cross-platform, native performance
- **Backend API:** FastAPI server (Python, runs on MacBook)
- **YOLO Inference:** ONNX export of YOLO26-Seg → runs on device or via API
- **Image Input:** Camera, gallery, DICOM picker
- **Output:** Segmentation overlay + metrics display

### Android App Workflow
```
App Flow:
  User opens app
    → Select input: Camera | Gallery | DICOM file
    → Image captured/selected
    → Sent to FastAPI endpoint: POST /api/segment
    → Server: pydicom → preprocess → YOLO26-ASIO infer → response
    → App displays: Original + Segmented overlay + Dice/IoU/Volume
    → Optional: Save PDF report or share result
```

---

## Implementation Phases & Timeline

| Phase | Task | Status | Est. Time |
|-------|------|--------|-----------|
| **P0** | Environment setup (conda + pip installs) | 🔲 TODO | 30 min |
| **P1** | Data exploration + 5-patient sample pipeline | 🔲 TODO | 2 hrs |
| **P1** | Sample FL run + graphs (5 patients, 2 clients) | 🔲 TODO | 1 hr |
| **P2** | Full dataset preprocessing (25%/5%/5%/65% split) | 🔲 TODO | 3 hrs |
| **P2** | LangGraph multi-agent full build | 🔲 TODO | 4 hrs |
| **P2** | Flower FL + ASIO server/client full build | 🔲 TODO | 4 hrs |
| **P2** | ASIO optimizer with real val Dice fitness | 🔲 TODO | 2 hrs |
| **P2** | DP-SGD (Opacus) integration | 🔲 TODO | 1 hr |
| **P2** | Full evaluation + ablation graphs | 🔲 TODO | 2 hrs |
| **P3** | Streamlit clinical dashboard | 🔲 TODO | 3 hrs |
| **P4** | FastAPI server for Android backend | 🔲 TODO | 2 hrs |
| **P4** | Flutter Android app | 🔲 TODO | 6 hrs |

---

## Open Questions for Review

> [!IMPORTANT]
> **Before starting, please confirm the following:**

1. **Project folder location:** Should the project be created at `/Users/shrikant/Downloads/FedASIO-YOLO26/` or at another path (e.g., Desktop, Documents)?

2. **LLM for clinical reports:** Do you want Qwen (as in the existing project) or a different offline model for clinical report generation? Or text-only templates for Phase 1?

3. **YOLO26 architecture:** The YOLO26 model should be the custom deep-residual-attention variant. Should it use the existing `yolo26n-seg.pt` from the old project as starting weights, or train from scratch on BraTS-PEDs?

4. **Flower FL mode:** Run all 5 clients in-process (fastest for M5, uses Virtual Client Engine) or simulate separate processes (more realistic but slower)?

5. **Android priority:** Do you want the Android app built immediately after Streamlit, or first validate the full FL pipeline before moving to mobile?

6. **Conda environment:** Should we create a new conda environment named `fedasio` (recommended to keep clean) or install into the existing `base` environment?

7. **WandB tracking:** Do you want Weights & Biases experiment tracking enabled (requires free account + API key), or local-only logging?
