# FedASIO-YOLO26 — Task Tracker

## Phase 0: Environment & Project Setup
- [x] Create conda environment `fedasio` (Python 3.11)
- [x] Install PyTorch nightly 2.14 with MPS support (M5 GPU ✅)
- [x] Install all required packages (ultralytics 8.4.68, flwr 1.31, langgraph 1.2.5, etc.)
- [x] Create project folder structure at `/Users/shrikant/Downloads/FedASIO-YOLO26/`
- [x] Write `config/config.yaml`
- [x] Write `requirements.txt`
- [x] Write `setup_environment.sh` (one-click setup script)

## Phase 1: Sample Pipeline (5 Patients)
- [x] Write `agents/state.py` — LangGraph AgentState TypedDict
- [x] Write `agents/data_agent.py` — NIfTI loader for BraTS-PEDs-v1
- [x] Write `agents/preprocessing_agent.py` — 3D→2D, normalize, contour
- [x] Write `agents/augmentation_agent.py` — Albumentations medical transforms
- [x] Write `models/asio/asio_optimizer.py` — ASIO algorithm (real val Dice fitness)
- [x] Write `agents/asio_agent.py` — ASIO LangGraph agent
- [x] Write `agents/segmentation_agent.py` — YOLO26-Seg train + infer
- [x] Write `agents/evaluation_agent.py` — Dice, IoU, mAP, HD95, F1
- [x] Write `agents/xai_agent.py` — GradCAM + confidence maps
- [x] Write `agents/report_agent.py` — PDF + Qwen LLM clinical report
- [x] Write `agents/graph.py` — LangGraph StateGraph wiring
- [x] Write `agents/orchestrator.py` — Entry point for agent pipeline
- [x] Write `federated/strategy.py` — FedASIO custom Flower strategy
- [x] Write `federated/client.py` — Flower client wrapping LangGraph
- [x] Write `federated/server.py` — Flower server with ASIO + privacy
- [x] Write `federated/privacy.py` — Opacus DP-SGD wrapper
- [x] Write `federated/aggregation.py` — FedAvg/FedProx/FedAdam/ASIO-wt
- [x] Write `scripts/phase1_sample.py` — 5-patient sample FL run
- [x] Run Phase 1 and verify results + graphs (completed in 26.9 mins)
- [x] Generate evaluation graphs (Dice, mAP, convergence curves)

## Phase 2: Full Dataset FL Training
- [x] Write `data/splits/patient_splits.json` — deterministic splits (Created ✅)
- [x] Write `scripts/phase2_full_fl.py` — Full federated training (65 train)
- [ ] Write `scripts/evaluate.py` — Evaluation on test set (13 patients)
- [ ] Write `scripts/generate_graphs.py` — Publication-quality figures
- [/] Run full FL training (50 rounds, 5 clients, ASIO + DP) ← RUNNING NOW
- [ ] Generate all ablation results
- [ ] Statistical validation (Wilcoxon, Friedman, Bonferroni)

## Phase 3: Streamlit Clinical Dashboard
- [x] Write `streamlit_app/app.py` — Main dashboard (Running at http://localhost:8501 ✅)
- [ ] Write `streamlit_app/components/upload.py`
- [ ] Write `streamlit_app/components/segmentation_view.py`
- [ ] Write `streamlit_app/components/metrics_panel.py`
- [ ] Write `streamlit_app/components/xai_panel.py`
- [ ] Write `streamlit_app/pages/01_Dashboard.py`
- [ ] Write `streamlit_app/pages/02_Patient_Analysis.py`
- [ ] Write `streamlit_app/pages/03_Federated_Status.py`
- [x] Test with 166 streamlit-split patients (Loaded ✅)

## Phase 4: Android App
- [ ] Write `android_app/` — FastAPI backend for mobile
- [ ] ONNX export of YOLO26-ASIO model
- [ ] Flutter Android app skeleton
- [ ] DICOM/JPEG input handling
- [ ] Result overlay display
- [ ] PDF report sharing
