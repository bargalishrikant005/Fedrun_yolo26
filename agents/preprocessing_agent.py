"""
agents/preprocessing_agent.py
FedASIO-YOLO26: PreprocessingAgent
Converts 3D BraTS-PEDs NIfTI volumes to 2D YOLO-format PNG slices + polygon labels.
"""
import os
import cv2
import logging
import numpy as np
from typing import List, Dict

from agents.state import FedASIOState

logger = logging.getLogger(__name__)

IMG_SIZE = 256
# BraTS-PEDs tumor labels: 1=SNFH, 2=TC, 3=ET
TUMOR_LABELS = [1, 2, 3]
MIN_TUMOR_AREA = 50  # minimum pixels for a valid tumor region


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize, ignoring zero background voxels."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    vmin, vmax = volume[mask].min(), volume[mask].max()
    if vmax == vmin:
        return np.zeros_like(volume, dtype=np.float32)
    norm = np.zeros_like(volume, dtype=np.float32)
    norm[mask] = (volume[mask] - vmin) / (vmax - vmin + 1e-8)
    return norm


def _mask_to_yolo_polygon(mask_2d: np.ndarray, class_id: int, img_h: int, img_w: int) -> List[str]:
    """Convert a binary 2D mask to YOLO polygon format strings."""
    annotations = []
    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_TUMOR_AREA:
            continue
        if len(contour) < 3:
            continue
        # Normalize polygon points to [0,1]
        points = contour.squeeze()
        if points.ndim == 1:
            points = points[np.newaxis, :]
        norm_pts = []
        for pt in points:
            norm_pts.extend([pt[0] / img_w, pt[1] / img_h])
        if len(norm_pts) >= 6:  # minimum 3 points = 6 values
            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
            annotations.append(line)
    return annotations


def preprocessing_agent(state: FedASIOState) -> FedASIOState:
    """
    PreprocessingAgent: Converts 3D NIfTI volumes → 2D YOLO segmentation slices.
    
    Strategy:
      1. Normalize each modality
      2. Stack t1c + t2f + t2w → 3-channel RGB surrogate
      3. Extract axial slices containing tumor (non-zero seg)
      4. Resize to 256×256
      5. Generate YOLO polygon annotations from seg contours
    """
    if state.get("error"):
        return state

    modalities = state["modalities"]
    seg_mask_3d = state["seg_mask_3d"]
    patient_id = state["patient_id"]

    logger.info(f"[PreprocessingAgent] Processing {patient_id}")

    try:
        # Normalize each modality
        t1c = _normalize_volume(modalities["t1c"])
        t1n = _normalize_volume(modalities["t1n"])
        t2f = _normalize_volume(modalities["t2f"])
        t2w = _normalize_volume(modalities["t2w"])

        n_slices = t1c.shape[2]
        processed_slices = []
        tumor_slice_indices = []

        for z in range(n_slices):
            # Stack 3 channels: T1c (contrast) + T2-FLAIR + T2w
            ch0 = (t1c[:, :, z] * 255).astype(np.uint8)  # T1c — enhancing tumor
            ch1 = (t2f[:, :, z] * 255).astype(np.uint8)  # T2-FLAIR — edema
            ch2 = (t2w[:, :, z] * 255).astype(np.uint8)  # T2w — general anatomy
            rgb = np.stack([ch0, ch1, ch2], axis=-1)

            # Get seg mask for this slice
            seg_slice = seg_mask_3d[:, :, z] if seg_mask_3d is not None else None

            # Check if this slice has tumor
            has_tumor = (seg_slice is not None) and np.any(np.isin(seg_slice, TUMOR_LABELS))
            if not has_tumor:
                continue

            # Resize to target size
            rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            seg_resized = cv2.resize(seg_slice.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_NEAREST) if seg_slice is not None else None

            # Generate YOLO polygon annotations (per tumor class)
            yolo_annotations = []
            if seg_resized is not None:
                for class_id, label in enumerate(TUMOR_LABELS, start=0):
                    binary_mask = (seg_resized == label).astype(np.uint8)
                    if binary_mask.sum() > 0:
                        annots = _mask_to_yolo_polygon(binary_mask, class_id, IMG_SIZE, IMG_SIZE)
                        yolo_annotations.extend(annots)

            if not yolo_annotations:
                continue  # Skip slices where tumor contour extraction failed

            processed_slices.append({
                "image": rgb_resized,             # np.array (256, 256, 3)
                "mask": seg_resized,              # np.array (256, 256)
                "slice_idx": z,
                "yolo_annotations": yolo_annotations,
                "patient_id": patient_id,
            })
            tumor_slice_indices.append(z)

        logger.info(f"  Extracted {len(processed_slices)} tumor slices from {n_slices} total")

        if len(processed_slices) == 0:
            raise ValueError(f"No tumor slices found for patient {patient_id}")

        return {
            **state,
            "processed_slices": processed_slices,
            "tumor_slice_indices": tumor_slice_indices,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[PreprocessingAgent] Error: {e}")
        return {**state, "error": str(e), "error_agent": "PreprocessingAgent"}


def save_slices_to_disk(slices: List[Dict], output_dir: str, split: str = "train") -> int:
    """Save processed slices as PNG images + YOLO .txt label files."""
    img_dir = os.path.join(output_dir, "images", split)
    lbl_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    saved = 0
    for s in slices:
        fname = f"{s['patient_id']}_z{s['slice_idx']:03d}"
        # Save image
        img_path = os.path.join(img_dir, f"{fname}.png")
        cv2.imwrite(img_path, cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR))
        # Save YOLO label
        lbl_path = os.path.join(lbl_dir, f"{fname}.txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(s["yolo_annotations"]))
        saved += 1

    return saved
