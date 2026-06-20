"""
federated/client.py
FedASIO-YOLO26: Flower Client — One per simulated hospital.
Wraps the LangGraph training pipeline as a Flower NumPyClient.
"""
import logging
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

from agents.graph import get_training_graph, get_inference_graph
from agents.segmentation_agent import _get_model_weights

logger = logging.getLogger(__name__)


class FedASIOClient(fl.client.NumPyClient):
    """
    Flower NumPyClient wrapping the LangGraph training pipeline.
    Each instance represents one hospital (FL client).
    """

    def __init__(
        self,
        client_id: int,
        patient_dirs: List[str],
        val_patient_dirs: List[str],
    ):
        self.client_id = client_id
        self.patient_dirs = patient_dirs          # Training patients for this client
        self.val_patient_dirs = val_patient_dirs  # Validation patients
        self.model_path = _get_model_weights()
        self.current_params = None
        logger.info(f"[Client {client_id}] Initialized with {len(patient_dirs)} train + {len(val_patient_dirs)} val patients")

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return current model parameters as numpy arrays."""
        from ultralytics import YOLO
        import torch
        model = YOLO(self.model_path)
        params = [p.cpu().detach().float().numpy() for p in model.model.parameters()]
        return params

    def _set_parameters(self, parameters: NDArrays):
        """Set model parameters from numpy arrays and save model weights."""
        from ultralytics import YOLO
        import torch
        model = YOLO(self.model_path)
        for i, (p, w) in enumerate(zip(model.model.parameters(), parameters)):
            w_tensor = torch.tensor(w, dtype=torch.float32).to(p.device)
            if p.shape != w_tensor.shape:
                raise ValueError(f"[Client {self.client_id}] Shape mismatch at parameter {i}: model shape {p.shape} vs input shape {w_tensor.shape}")
            p.data = w_tensor
        model.save(self.model_path)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Local training: run LangGraph training pipeline on client's patients.
        Returns updated parameters + metrics to server.
        """
        self._set_parameters(parameters)
        fl_round = int(config.get("fl_round", 0))
        local_epochs = int(config.get("local_epochs", 3))  # Default 3 for Phase 1, 10 for Phase 2
        hyperparams = {
            "lr0": float(config.get("hp_lr0", 0.001)),
            "momentum": float(config.get("hp_momentum", 0.937)),
            "weight_decay": float(config.get("hp_weight_decay", 0.0005)),
            "conf": float(config.get("hp_conf", 0.25)),
            "iou": float(config.get("hp_iou", 0.5)),
            "batch_size": float(config.get("hp_batch_size", 8)),
            "local_epochs": local_epochs,   # Pass epochs to SegmentationAgent
        }
        logger.info(f"[Client {self.client_id}] Fit round {fl_round} | {len(self.patient_dirs)} patients | HPs: lr={hyperparams['lr0']:.5f}")

        training_graph = get_training_graph()
        all_dice = []
        last_model_params = parameters

        for patient_dir in self.patient_dirs:
            initial_state = {
                "patient_id": "",
                "patient_dir": patient_dir,
                "modalities": {},
                "seg_mask_3d": None,
                "affine": None,
                "processed_slices": [],
                "tumor_slice_indices": [],
                "augmented_slices": [],
                "hyperparams": hyperparams,
                "asio_history": [],
                "predictions": [],
                "pred_masks": [],
                "model_path": self.model_path,
                "metrics": {},
                "per_class_metrics": {},
                "tumor_volume_cc": 0.0,
                "xai_maps": {},
                "xai_figure_paths": [],
                "report_text": "",
                "pdf_path": None,
                "report_metadata": {},
                "fl_round": fl_round,
                "fl_client_id": self.client_id,
                "fl_stage": "train",
                "global_model_params": None,
                "error": None,
                "error_agent": None,
                "retry_count": 0,
            }

            try:
                result = training_graph.invoke(initial_state)
                # Extract training Dice: read mAP50 from YOLO results_dict
                # The segmentation_agent saves the best model path; we use training mAP50 as proxy
                train_map50 = result.get("metrics", {}).get("mAP50", 0.0)
                if train_map50 == 0.0:
                    # Fallback: read directly from YOLO run results CSV if available
                    run_dir = f"/Users/shrikant/Downloads/FedASIO-YOLO26/runs/client{self.client_id}_round{fl_round}"
                    csv_path = os.path.join(run_dir, "results.csv")
                    if os.path.exists(csv_path):
                        import csv
                        with open(csv_path) as cf:
                            rows = list(csv.DictReader(cf))
                        if rows:
                            last = rows[-1]
                            # YOLO CSV col: metrics/mAP50(M) for mask
                            for col in ["metrics/mAP50(M)", "metrics/mAP50(B)", "       metrics/mAP50(M)"]:
                                val = last.get(col, "").strip()
                                if val:
                                    try:
                                        train_map50 = float(val)
                                        break
                                    except ValueError:
                                        pass
                all_dice.append(train_map50)
                if result.get("model_path"):
                    self.model_path = result["model_path"]
            except Exception as e:
                logger.error(f"[Client {self.client_id}] Patient {patient_dir} failed: {e}")

        # Return updated parameters
        updated_params = self.get_parameters({})
        mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
        logger.info(f"[Client {self.client_id}] Fit complete | Mean Dice: {mean_dice:.4f}")

        return updated_params, len(self.patient_dirs), {"dice": mean_dice, "client_id": self.client_id}

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Local evaluation on validation patients.
        Returns loss + metrics to server.
        """
        self._set_parameters(parameters)
        fl_round = int(config.get("fl_round", 0))
        logger.info(f"[Client {self.client_id}] Evaluate round {fl_round} | {len(self.val_patient_dirs)} val patients")

        inference_graph = get_inference_graph()
        all_dice = []
        all_loss = []

        for patient_dir in self.val_patient_dirs[:3]:  # Limit for speed
            initial_state = {
                "patient_id": "",
                "patient_dir": patient_dir,
                "modalities": {},
                "seg_mask_3d": None,
                "affine": None,
                "processed_slices": [],
                "tumor_slice_indices": [],
                "augmented_slices": [],
                "hyperparams": {
                    "conf": float(config.get("hp_conf", 0.25)),
                    "iou": float(config.get("hp_iou", 0.5)),
                    "batch_size": float(config.get("hp_batch_size", 8)),
                    "lr0": float(config.get("hp_lr0", 0.001)),
                    "momentum": float(config.get("hp_momentum", 0.937)),
                    "weight_decay": float(config.get("hp_weight_decay", 0.0005)),
                },
                "asio_history": [],
                "predictions": [],
                "pred_masks": [],
                "model_path": self.model_path,
                "metrics": {},
                "per_class_metrics": {},
                "tumor_volume_cc": 0.0,
                "xai_maps": {},
                "xai_figure_paths": [],
                "report_text": "",
                "pdf_path": None,
                "report_metadata": {},
                "fl_round": fl_round,
                "fl_client_id": self.client_id,
                "fl_stage": "evaluate",
                "global_model_params": None,
                "error": None,
                "error_agent": None,
                "retry_count": 0,
            }

            try:
                result = inference_graph.invoke(initial_state)
                metrics = result.get("metrics", {})
                dice = metrics.get("dice", 0.0)
                all_dice.append(dice)
                all_loss.append(1.0 - dice)  # loss = 1 - Dice
            except Exception as e:
                logger.error(f"[Client {self.client_id}] Eval patient {patient_dir} failed: {e}")

        mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
        mean_loss = float(np.mean(all_loss)) if all_loss else 1.0
        logger.info(f"[Client {self.client_id}] Eval — Loss: {mean_loss:.4f} | Dice: {mean_dice:.4f}")

        return mean_loss, len(self.val_patient_dirs), {"dice": mean_dice, "client_id": self.client_id}
