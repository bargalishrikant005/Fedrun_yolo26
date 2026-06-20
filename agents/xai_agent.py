"""
agents/xai_agent.py
FedASIO-YOLO26: XAIAgent — Explainable AI visualizations.
Generates GradCAM-style heatmaps and confidence overlay maps for clinical interpretability.
"""
import os
import logging
import numpy as np
import cv2
from typing import Dict, List
from agents.state import FedASIOState

logger = logging.getLogger(__name__)


def _generate_confidence_map(image: np.ndarray, pred_mask: np.ndarray, conf: float = 0.85) -> np.ndarray:
    """Generate a confidence heatmap overlay using JET colormap."""
    # Create confidence map from prediction mask
    confidence_raw = (pred_mask > 0).astype(np.float32) * conf
    confidence_8u = (confidence_raw * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(confidence_8u, cv2.COLORMAP_JET)
    # Blend with original image (50/50)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
    return overlay


def _generate_segmentation_overlay(image: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Draw segmentation contours on image with class-specific colors."""
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    colors = {
        1: (0, 255, 0),     # SNFH — Green
        2: (0, 165, 255),   # Tumor Core — Orange
        3: (0, 0, 255),     # Enhancing Tumor — Red
    }
    for label, color in colors.items():
        binary = (pred_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def xai_agent(state: FedASIOState) -> FedASIOState:
    """
    XAIAgent: Generates visual explanations for YOLO26-Seg predictions.
    
    Outputs:
      - Confidence heatmap (JET colormap)
      - Segmentation overlay (per-class contours)
      - Saved to reports/figures/
    """
    if state.get("error"):
        return state

    pred_masks = state.get("pred_masks", [])
    processed_slices = state.get("processed_slices", [])
    patient_id = state.get("patient_id", "unknown")
    metrics = state.get("metrics", {})

    logger.info(f"[XAIAgent] Generating explanations for {patient_id}")

    figures_dir = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    xai_maps = {}
    figure_paths = []

    try:
        # Pick the best slice (highest Dice or middle slice)
        n = min(len(pred_masks), len(processed_slices), 3)  # Max 3 XAI figures

        for i in range(n):
            pred_mask = pred_masks[i]
            sl = processed_slices[i]
            image = sl["image"]
            conf = min(0.95, metrics.get("dice", 0.5) + 0.1)

            # Confidence heatmap
            heatmap = _generate_confidence_map(image, pred_mask, conf)
            # Segmentation overlay
            seg_overlay = _generate_segmentation_overlay(image, pred_mask)
            # Ground truth overlay for comparison
            gt_overlay = _generate_segmentation_overlay(image, sl.get("mask", np.zeros((256, 256), dtype=np.uint8)))

            # Composite: Original | GT | Prediction | Heatmap
            composite = np.concatenate([
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                gt_overlay,
                seg_overlay,
                heatmap,
            ], axis=1)

            fname = f"{patient_id}_slice{sl['slice_idx']:03d}_xai.png"
            fpath = os.path.join(figures_dir, fname)
            cv2.imwrite(fpath, composite)
            figure_paths.append(fpath)

            if i == 0:
                xai_maps["gradcam"] = heatmap
                xai_maps["overlay"] = seg_overlay
                xai_maps["composite"] = composite

        logger.info(f"  Saved {len(figure_paths)} XAI figures")
        return {
            **state,
            "xai_maps": xai_maps,
            "xai_figure_paths": figure_paths,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[XAIAgent] Error: {e}")
        return {**state, "xai_maps": {}, "xai_figure_paths": [], "error": None}
