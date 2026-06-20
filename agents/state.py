"""
agents/state.py
FedASIO-YOLO26: LangGraph Agent State Definition
All agents communicate through this shared state TypedDict.
"""
from typing import TypedDict, Optional, List, Dict, Any
import numpy as np


class FedASIOState(TypedDict):
    # ── Patient Identification ─────────────────────────────────────────────
    patient_id: str
    patient_dir: str

    # ── Raw MRI Data (loaded by DataAgent) ────────────────────────────────
    modalities: Dict[str, Any]        # {t1c: np.array, t1n: np.array, t2f: np.array, t2w: np.array}
    seg_mask_3d: Any                  # np.array (240,240,155) — ground truth 3D segmentation
    affine: Any                       # NIfTI affine matrix for voxel→mm conversion

    # ── Processed 2D Slices (PreprocessingAgent) ──────────────────────────
    processed_slices: List[Dict]      # [{image: np.array(256,256,3), mask: np.array(256,256), slice_idx: int}]
    tumor_slice_indices: List[int]    # Axial slice indices with tumor

    # ── Augmented Slices (AugmentationAgent) ──────────────────────────────
    augmented_slices: List[Dict]      # Same format as processed_slices, post-augmentation

    # ── ASIO Hyperparameters (ASIOAgent) ──────────────────────────────────
    hyperparams: Dict[str, float]     # {lr0, momentum, weight_decay, conf, iou, batch_size}
    asio_history: List[Dict]          # Per-iteration ASIO state for analysis

    # ── YOLO Predictions (SegmentationAgent) ──────────────────────────────
    predictions: List[Dict]           # Per-slice predictions from YOLO26-Seg
    pred_masks: List[Any]             # Binary prediction masks (np.array)
    model_path: Optional[str]         # Path to best YOLO checkpoint

    # ── Evaluation Metrics (EvaluationAgent) ──────────────────────────────
    metrics: Dict[str, float]         # {dice, iou, mAP50, mAP50_95, precision, recall, f1, hd95}
    per_class_metrics: Dict[str, Dict]  # Per-class breakdown {SNFH, TC, ET}
    tumor_volume_cc: float            # Tumor volume in cubic centimeters

    # ── XAI Outputs (XAIAgent) ────────────────────────────────────────────
    xai_maps: Dict[str, Any]          # {gradcam: np.array, confidence: np.array, overlay: np.array}
    xai_figure_paths: List[str]       # Saved XAI image paths

    # ── Clinical Report (ReportAgent) ─────────────────────────────────────
    report_text: str                  # Qwen LLM or rule-based clinical narrative
    pdf_path: Optional[str]           # Generated PDF report path
    report_metadata: Dict[str, Any]   # Timestamp, model version, FL round

    # ── Federated Learning Context ────────────────────────────────────────
    fl_round: int                     # Current global FL round
    fl_client_id: int                 # Client index (0–4)
    fl_stage: str                     # "train" | "evaluate" | "infer"
    global_model_params: Optional[Any]  # Received global model weights

    # ── Error Handling ────────────────────────────────────────────────────
    error: Optional[str]              # Error message (None = success)
    error_agent: Optional[str]        # Which agent raised the error
    retry_count: int                  # Retry attempts for current agent
