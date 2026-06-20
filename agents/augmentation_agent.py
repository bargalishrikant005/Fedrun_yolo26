"""
agents/augmentation_agent.py
FedASIO-YOLO26: AugmentationAgent
Medical-grade image augmentation using Albumentations for BraTS-PEDs MRI data.
Augmentations are carefully chosen to be medically realistic for brain MRI.
"""
import logging
import numpy as np
from typing import List, Dict
import albumentations as A
from agents.state import FedASIOState

logger = logging.getLogger(__name__)

# Medical-grade augmentation pipeline for brain MRI
# NOTE: No color jitter — brain MRI has meaningful intensity values
MEDICAL_AUGMENTATION = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4,
        border_mode=0  # constant padding (black)
    ),
    A.ElasticTransform(
        alpha=120, sigma=6.0, p=0.3,
        border_mode=0
    ),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
], additional_targets={"mask": "mask"})


def augmentation_agent(state: FedASIOState) -> FedASIOState:
    """
    AugmentationAgent: Applies medical-grade augmentation to processed 2D slices.
    
    Inputs:  state.processed_slices
    Outputs: state.augmented_slices (original + augmented copies)
    
    Strategy: Each slice is augmented 2× during training for data diversity.
    During inference/validation, augmentation is skipped.
    """
    if state.get("error"):
        return state

    fl_stage = state.get("fl_stage", "train")
    processed_slices = state.get("processed_slices", [])
    patient_id = state.get("patient_id", "unknown")

    logger.info(f"[AugmentationAgent] {patient_id} | stage={fl_stage} | slices={len(processed_slices)}")

    if fl_stage != "train":
        # No augmentation during validation/inference
        logger.info("  Skipping augmentation (non-training stage)")
        return {**state, "augmented_slices": processed_slices}

    try:
        augmented_slices = []
        for sl in processed_slices:
            # Keep original
            augmented_slices.append(sl)

            # Generate 2 augmented versions
            for _ in range(2):
                augmented = MEDICAL_AUGMENTATION(
                    image=sl["image"],
                    mask=sl["mask"].astype(np.uint8) if sl["mask"] is not None else np.zeros((256, 256), dtype=np.uint8)
                )
                aug_image = augmented["image"]
                aug_mask = augmented["mask"]

                # Regenerate YOLO annotations from augmented mask
                from agents.preprocessing_agent import _mask_to_yolo_polygon, TUMOR_LABELS, IMG_SIZE
                yolo_annotations = []
                for class_id, label in enumerate(TUMOR_LABELS, start=0):
                    binary_mask = (aug_mask == label).astype(np.uint8)
                    if binary_mask.sum() > 0:
                        annots = _mask_to_yolo_polygon(binary_mask, class_id, IMG_SIZE, IMG_SIZE)
                        yolo_annotations.extend(annots)

                if yolo_annotations:
                    augmented_slices.append({
                        "image": aug_image,
                        "mask": aug_mask,
                        "slice_idx": sl["slice_idx"],
                        "yolo_annotations": yolo_annotations,
                        "patient_id": sl["patient_id"],
                        "augmented": True,
                    })

        logger.info(f"  Augmented: {len(processed_slices)} → {len(augmented_slices)} slices")
        return {**state, "augmented_slices": augmented_slices, "error": None}

    except Exception as e:
        logger.error(f"[AugmentationAgent] Error: {e}")
        return {**state, "augmented_slices": processed_slices, "error": None}  # fallback to original
