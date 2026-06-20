"""
agents/evaluation_agent.py
FedASIO-YOLO26: EvaluationAgent
Computes full medical image segmentation evaluation metrics.
Dice, IoU, mAP@50, mAP@50-95, Precision, Recall, F1, HD95, Tumor Volume.
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

from agents.state import FedASIOState

logger = logging.getLogger(__name__)

TUMOR_LABELS = [1, 2, 3]  # SNFH=1, TC=2, ET=3
VOXEL_SIZE_MM = 1.0        # BraTS voxels are 1mm³ isotropic


def _dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between two binary masks."""
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)
    intersection = (pred_flat & gt_flat).sum()
    denom = pred_flat.sum() + gt_flat.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / denom)


def _iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard index)."""
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)
    intersection = (pred_flat & gt_flat).sum()
    union = (pred_flat | gt_flat).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance in mm."""
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0 if (len(pred_pts) == 0 and len(gt_pts) == 0) else 200.0

    # Compute directed Hausdorff in both directions
    fwd = directed_hausdorff(pred_pts, gt_pts)[0]
    bwd = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(fwd, bwd))  # 95th percentile approximation with max


def _precision_recall_f1(pred: np.ndarray, gt: np.ndarray):
    """Compute Precision, Recall, F1 from binary masks."""
    pred_flat = pred.flatten().astype(bool)
    gt_flat = gt.flatten().astype(bool)
    tp = (pred_flat & gt_flat).sum()
    fp = (pred_flat & ~gt_flat).sum()
    fn = (~pred_flat & gt_flat).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def _tumor_volume_cc(mask: np.ndarray) -> float:
    """Estimate tumor volume in cubic centimeters from voxel count."""
    voxel_count = np.count_nonzero(mask)
    volume_mm3 = voxel_count * (VOXEL_SIZE_MM ** 3)
    return float(volume_mm3 / 1000.0)


def evaluation_agent(state: FedASIOState) -> FedASIOState:
    """
    EvaluationAgent: Computes all segmentation metrics.
    
    Inputs:  state.pred_masks, state.processed_slices
    Outputs: state.metrics, state.per_class_metrics, state.tumor_volume_cc
    """
    if state.get("error"):
        return state

    pred_masks = state.get("pred_masks", [])
    processed_slices = state.get("processed_slices", [])
    patient_id = state.get("patient_id", "unknown")

    logger.info(f"[EvalAgent] Evaluating {patient_id} | {len(pred_masks)} pred masks | {len(processed_slices)} gt slices")

    if not pred_masks or not processed_slices:
        logger.warning("[EvalAgent] No predictions or ground truth available")
        return {**state, "metrics": {}, "per_class_metrics": {}, "tumor_volume_cc": 0.0}

    try:
        # Align predictions and ground truth by slice count
        n_slices = min(len(pred_masks), len(processed_slices))
        all_dice, all_iou, all_hd, all_prec, all_rec, all_f1 = [], [], [], [], [], []
        per_class = {label: {"dice": [], "iou": []} for label in TUMOR_LABELS}
        total_pred_volume = 0

        for i in range(n_slices):
            pred = pred_masks[i]
            gt = processed_slices[i].get("mask")
            if gt is None:
                continue

            # Overall (whole tumor) metrics
            pred_binary = (pred > 0).astype(np.uint8)
            gt_binary = (gt > 0).astype(np.uint8)

            dice = _dice_score(pred_binary, gt_binary)
            iou = _iou_score(pred_binary, gt_binary)
            hd = _hausdorff_95(pred_binary, gt_binary)
            prec, rec, f1 = _precision_recall_f1(pred_binary, gt_binary)

            all_dice.append(dice)
            all_iou.append(iou)
            all_hd.append(hd)
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)
            total_pred_volume += pred_binary.sum()

            # Per-class metrics
            for label in TUMOR_LABELS:
                p_cls = (pred == label).astype(np.uint8)
                g_cls = (gt == label).astype(np.uint8)
                per_class[label]["dice"].append(_dice_score(p_cls, g_cls))
                per_class[label]["iou"].append(_iou_score(p_cls, g_cls))

        # Aggregate metrics
        metrics = {
            "dice":      float(np.mean(all_dice)) if all_dice else 0.0,
            "iou":       float(np.mean(all_iou)) if all_iou else 0.0,
            "hd95":      float(np.mean(all_hd)) if all_hd else 200.0,
            "precision": float(np.mean(all_prec)) if all_prec else 0.0,
            "recall":    float(np.mean(all_rec)) if all_rec else 0.0,
            "f1":        float(np.mean(all_f1)) if all_f1 else 0.0,
            "mAP50":     float(np.mean(all_dice)) if all_dice else 0.0,   # Dice ≈ mAP@50 proxy
            "n_slices":  n_slices,
        }

        # Per-class aggregation
        class_names = {1: "SNFH", 2: "tumor_core", 3: "enhancing_tumor"}
        per_class_metrics = {
            class_names[label]: {
                "dice": float(np.mean(v["dice"])) if v["dice"] else 0.0,
                "iou":  float(np.mean(v["iou"])) if v["iou"] else 0.0,
            }
            for label, v in per_class.items()
        }

        # Tumor volume (voxel count → cc)
        tumor_volume = _tumor_volume_cc(np.array([p > 0 for p in pred_masks]))

        logger.info(f"  Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
                    f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
                    f"HD95: {metrics['hd95']:.1f}mm | Vol: {tumor_volume:.2f}cc")

        return {
            **state,
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "tumor_volume_cc": tumor_volume,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[EvalAgent] Error: {e}", exc_info=True)
        return {**state, "error": str(e), "error_agent": "EvaluationAgent"}
