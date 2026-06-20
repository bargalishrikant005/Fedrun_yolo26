"""
agents/data_agent.py
FedASIO-YOLO26: DataAgent — Loads BraTS-PEDs-v1 NIfTI files.
Supports BraTS 2023/2024 naming: -t1c, -t1n, -t2f, -t2w, -seg
"""
import os
import glob
import logging
from typing import Dict, Optional
import numpy as np
import nibabel as nib

from agents.state import FedASIOState

logger = logging.getLogger(__name__)

MODALITY_SUFFIXES = {
    "t1c": ["-t1c.nii.gz", "_t1ce.nii.gz"],
    "t1n": ["-t1n.nii.gz", "_t1.nii.gz"],
    "t2f": ["-t2f.nii.gz", "_flair.nii.gz"],
    "t2w": ["-t2w.nii.gz", "_t2.nii.gz"],
    "seg": ["-seg.nii.gz", "_seg.nii.gz"],
}


def _find_file(patient_dir: str, modality: str) -> Optional[str]:
    """Find a modality file using flexible suffix matching."""
    for suffix in MODALITY_SUFFIXES[modality]:
        pattern = os.path.join(patient_dir, f"*{suffix}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def data_agent(state: FedASIOState) -> FedASIOState:
    """
    DataAgent: Loads multi-modal NIfTI MRI data for a single BraTS-PEDs patient.
    
    Inputs:  state.patient_dir
    Outputs: state.modalities, state.seg_mask_3d, state.affine
    """
    patient_dir = state["patient_dir"]
    patient_id = os.path.basename(patient_dir)
    logger.info(f"[DataAgent] Loading patient: {patient_id}")

    try:
        modalities = {}
        affine = None

        # Load each MRI modality
        for mod in ["t1c", "t1n", "t2f", "t2w"]:
            fpath = _find_file(patient_dir, mod)
            if fpath is None:
                raise FileNotFoundError(f"Modality '{mod}' not found in {patient_dir}")
            img = nib.load(fpath)
            data = img.get_fdata(dtype=np.float32)
            modalities[mod] = data
            if affine is None:
                affine = img.affine
            logger.debug(f"  Loaded {mod}: shape={data.shape}, min={data.min():.2f}, max={data.max():.2f}")

        # Load segmentation mask (may not exist for validation set)
        seg_mask_3d = None
        seg_path = _find_file(patient_dir, "seg")
        if seg_path:
            seg_img = nib.load(seg_path)
            seg_mask_3d = seg_img.get_fdata(dtype=np.float32).astype(np.uint8)
            # BraTS-PEDs label mapping: 1=SNFH, 2=TC, 3=ET
            logger.debug(f"  Loaded seg: shape={seg_mask_3d.shape}, unique={np.unique(seg_mask_3d)}")
        else:
            logger.warning(f"  No segmentation mask found (validation patient)")

        # Validate shape consistency
        shapes = [v.shape for v in modalities.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"Modality shape mismatch: {shapes}")

        return {
            **state,
            "patient_id": patient_id,
            "modalities": modalities,
            "seg_mask_3d": seg_mask_3d,
            "affine": affine,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[DataAgent] Error for {patient_id}: {e}")
        return {**state, "error": str(e), "error_agent": "DataAgent"}


def scan_dataset(raw_dir: str, limit: Optional[int] = None):
    """Scan BraTS-PEDs directory and return list of patient directories."""
    patient_dirs = sorted([
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and d.startswith("BraTS")
    ])
    if limit:
        patient_dirs = patient_dirs[:limit]
    logger.info(f"[DataAgent] Found {len(patient_dirs)} patients in {raw_dir}")
    return patient_dirs
