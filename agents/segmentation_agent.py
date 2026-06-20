"""
agents/segmentation_agent.py
FedASIO-YOLO26: SegmentationAgent
Handles YOLO26-Seg training (local FL client), inference, and model evaluation.
Uses MPS (Apple M5) for hardware acceleration.
"""
import os
import logging
import shutil
import yaml
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from agents.state import FedASIOState
from agents.preprocessing_agent import save_slices_to_disk

logger = logging.getLogger(__name__)

# YOLO26-Seg model — uses ultralytics with custom config
# Falls back to yolo11n-seg if YOLO26 weights not found
MODEL_OPTIONS = [
    "/Users/shrikant/Downloads/FedASIO-YOLO26/models/yolo26/weights/best.pt",
    "/Users/shrikant/Downloads/yolo vs 26/yolo26n-seg.pt",
    "/Users/shrikant/Downloads/yolo vs 26/yolo11n-seg.pt",
    "yolo11n-seg.pt",  # auto-download fallback
]


def _get_model_weights() -> str:
    """Find the best available YOLO model weights."""
    for path in MODEL_OPTIONS:
        if os.path.exists(path):
            logger.info(f"[SegAgent] Using weights: {path}")
            return path
    logger.warning("[SegAgent] No local weights found. Will auto-download yolo11n-seg.pt")
    return "yolo11n-seg.pt"


def _create_data_yaml(processed_dir: str, split_name: str) -> str:
    """Create YOLO data.yaml for training/validation."""
    data_config = {
        "path": processed_dir,
        "train": f"images/{split_name}",
        "val": f"images/{split_name}",   # use same for Phase 1 sample
        "nc": 3,
        "names": {0: "SNFH", 1: "tumor_core", 2: "enhancing_tumor"},
    }
    yaml_path = os.path.join(processed_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    return yaml_path


def segmentation_agent(state: FedASIOState) -> FedASIOState:
    """
    SegmentationAgent: Trains or runs inference with YOLO26-Seg.
    
    fl_stage="train"    → Saves slices to disk + runs YOLO training
    fl_stage="evaluate" → Runs YOLO inference on validation slices
    fl_stage="infer"    → Runs YOLO inference on a single patient
    """
    if state.get("error"):
        return state

    try:
        from ultralytics import YOLO
        import torch

        fl_stage = state.get("fl_stage", "infer")
        fl_round = state.get("fl_round", 0)
        fl_client_id = state.get("fl_client_id", 0)
        hyperparams = state.get("hyperparams", {})
        patient_id = state.get("patient_id", "unknown")
        slices = state.get("augmented_slices") or state.get("processed_slices", [])

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"[SegAgent] stage={fl_stage} | client={fl_client_id} | round={fl_round} | device={device}")

        # ── TRAINING MODE ──────────────────────────────────────────────────
        if fl_stage == "train":
            # Save slices to YOLO format
            processed_dir = f"/Users/shrikant/Downloads/FedASIO-YOLO26/data/processed/client_{fl_client_id}/round_{fl_round}"
            n_saved = save_slices_to_disk(slices, processed_dir, split="train")
            logger.info(f"  Saved {n_saved} slices to {processed_dir}")

            data_yaml = _create_data_yaml(processed_dir, "train")
            weights = state.get("model_path") or _get_model_weights()
            model = YOLO(weights)

            # Apply global model params from FL server if provided
            if state.get("global_model_params") is not None:
                _load_state_dict(model, state["global_model_params"])

            # Training config — ASIO-optimized hyperparameters
            lr0 = hyperparams.get("lr0", 0.001)
            momentum = hyperparams.get("momentum", 0.937)
            weight_decay = hyperparams.get("weight_decay", 0.0005)
            batch = int(hyperparams.get("batch_size", 8))
            local_epochs = int(hyperparams.get("local_epochs", 3))  # 3=Phase1, 10=Phase2

            run_name = f"client{fl_client_id}_round{fl_round}"
            
            # Official YOLO26 configurations from the paper (yolo26_model.pdf):
            # 1. DFL-free regression head (reg_max=1)
            # 2. Dual-head end-to-end NMS-free training (end2end=True)
            # 3. MuSGD optimizer (optimizer='MuSGD')
            # 4. Official loss weights (box=7.50, cls=0.50, dfl=0.0)
            # 5. Close mosaic augmentation epochs (close_mosaic=10)
            results = model.train(
                data=data_yaml,
                epochs=local_epochs,  # ASIO-configurable local epochs per round
                imgsz=256,
                batch=batch,
                lr0=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
                device=device,
                workers=0,           # macOS fix
                amp=False,           # MPS stability
                warmup_epochs=0.0,   # Disable warmup epochs to prevent LR reset per round
                project="/Users/shrikant/Downloads/FedASIO-YOLO26/runs",
                name=run_name,
                exist_ok=True,
                verbose=False,
                plots=False,
                save=True,
                patience=5,
                reg_max=1,           # DFL-free regression head (YOLO26 Section 3.2.2)
                end2end=True,        # Dual-head end-to-end NMS-free path (YOLO26 Section 3.2.1)
                optimizer="MuSGD",   # Hybrid Muon-SGD optimizer (YOLO26 Section 3.3.1)
                box=7.50,            # YOLO26 loss weight settings (YOLO26 Table S2)
                cls=0.50,            # YOLO26 loss weight settings (YOLO26 Table S2)
                dfl=0.0,             # DFL is removed entirely in YOLO26
                close_mosaic=10,     # Mosaic disabled near end of training (YOLO26 Table S5)
            )

            best_model_path = str(results.save_dir / "weights" / "best.pt")
            logger.info(f"  Training complete. Best: {best_model_path}")

            # Extract model params for FL aggregation
            model_params = _extract_state_dict(model)

            return {
                **state,
                "model_path": best_model_path,
                "global_model_params": model_params,
                "error": None,
            }

        # ── INFERENCE / EVALUATION MODE ────────────────────────────────────
        else:
            model_path = state.get("model_path") or _get_model_weights()
            model = YOLO(model_path)

            if state.get("global_model_params") is not None:
                _load_state_dict(model, state["global_model_params"])

            conf = hyperparams.get("conf", 0.25)
            iou = hyperparams.get("iou", 0.5)

            predictions = []
            pred_masks = []

            for sl in slices[:20]:  # limit inference slices for speed
                # Save temp image
                import cv2, tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cv2.imwrite(tmp.name, cv2.cvtColor(sl["image"], cv2.COLOR_RGB2BGR))
                    tmp_path = tmp.name

                results = model.predict(
                    source=tmp_path,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                    save=False,
                )
                os.unlink(tmp_path)

                # Extract prediction mask
                pred_mask = np.zeros((256, 256), dtype=np.uint8)
                if results[0].masks is not None:
                    for j, mask in enumerate(results[0].masks.data):
                        cls = int(results[0].boxes.cls[j].item()) + 1  # 1-indexed label
                        mask_np = mask.cpu().numpy()
                        import cv2
                        mask_resized = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_NEAREST)
                        pred_mask[mask_resized > 0.5] = cls

                predictions.append({
                    "slice_idx": sl.get("slice_idx", 0),
                    "patient_id": sl.get("patient_id", patient_id),
                    "conf": conf,
                    "n_detections": len(results[0].boxes) if results[0].boxes else 0,
                })
                pred_masks.append(pred_mask)

            logger.info(f"  Inference complete: {len(predictions)} slices processed")
            return {**state, "predictions": predictions, "pred_masks": pred_masks, "error": None}

    except Exception as e:
        logger.error(f"[SegAgent] Error: {e}", exc_info=True)
        return {**state, "error": str(e), "error_agent": "SegmentationAgent"}


def _extract_state_dict(model) -> Dict:
    """Extract model state dict for FL aggregation."""
    import torch
    state = {}
    for key, val in model.model.state_dict().items():
        state[key] = val.cpu().float().numpy().tolist()
    return state


def _load_state_dict(model, params: Dict):
    """Load FL-aggregated parameters into model."""
    import torch
    state = {}
    for key, val in params.items():
        state[key] = torch.tensor(val, dtype=torch.float32)
    model.model.load_state_dict(state, strict=False)
