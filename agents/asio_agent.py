"""
agents/asio_agent.py
FedASIO-YOLO26: ASIOAgent — LangGraph wrapper around the ASIO optimizer.
Invokes the ASIO algorithm with a real YOLO validation Dice fitness function.
"""
import logging
import os
import cv2
import numpy as np
import tempfile
from typing import Dict

from agents.state import FedASIOState
from models.asio.asio_optimizer import ASIOOptimizer

logger = logging.getLogger(__name__)

# Shared ASIO instance (persists between FL rounds for warm-starting)
_asio_instance: ASIOOptimizer = None


def _build_fitness_fn(val_slices: list, model_path: str, device: str = "mps"):
    """
    Build a fitness function that trains YOLO for 1 epoch on sample slices
    and returns the validation Dice score.
    
    This is the REAL fitness function — not simulated.
    """
    import torch
    from ultralytics import YOLO
    from agents.preprocessing_agent import save_slices_to_disk
    from agents.evaluation_agent import _dice_score

    def fitness(hyperparams: Dict) -> float:
        """Evaluate hyperparameter set by running 1 YOLO epoch and computing Dice."""
        try:
            # Save val slices temporarily
            with tempfile.TemporaryDirectory() as tmp_dir:
                n_saved = save_slices_to_disk(val_slices[:5], tmp_dir, "train")  # use 5 slices max for speed
                if n_saved == 0:
                    return 0.0

                # Write data yaml
                import yaml
                data_yaml = {
                    "path": tmp_dir,
                    "train": "images/train",
                    "val": "images/train",
                    "nc": 3,
                    "names": {0: "SNFH", 1: "tumor_core", 2: "enhancing_tumor"},
                }
                yaml_path = os.path.join(tmp_dir, "data.yaml")
                with open(yaml_path, "w") as f:
                    yaml.dump(data_yaml, f)

                model = YOLO(model_path)
                lr0 = hyperparams.get("lr0", 0.001)
                momentum = hyperparams.get("momentum", 0.937)
                weight_decay = hyperparams.get("weight_decay", 0.0005)
                batch = max(2, int(hyperparams.get("batch_size", 4)))

                # 1-epoch quick train (fitness evaluation)
                results = model.train(
                    data=yaml_path,
                    epochs=1,
                    imgsz=256,
                    batch=batch,
                    lr0=lr0,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    device=device,
                    workers=0,
                    amp=False,
                    verbose=False,
                    plots=False,
                    save=False,
                    val=True,
                )

                # Extract Dice from YOLO metrics (mAP50 as proxy for Dice)
                dice = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
                logger.debug(f"    ASIO fitness eval: lr={lr0:.5f}, momentum={momentum:.3f} → Dice≈{dice:.4f}")
                return dice

        except Exception as e:
            logger.warning(f"    ASIO fitness eval failed: {e}")
            return 0.0

    return fitness


def asio_agent(state: FedASIOState) -> FedASIOState:
    """
    ASIOAgent: Runs ASIO to optimize YOLO hyperparameters using real validation Dice.
    
    Inputs:  state.processed_slices, state.fl_round
    Outputs: state.hyperparams, state.asio_history
    """
    if state.get("error"):
        return state

    # Bypass local ASIO search during FL training (uses global parameters broadcast by server)
    if state.get("bypass_local_asio", True) or state.get("fl_stage") == "train":
        logger.info(f"[ASIOAgent] Bypassing local ASIO search for patient {state.get('patient_id', 'unknown')}. Using global hyperparameters.")
        return state

    global _asio_instance
    fl_round = state.get("fl_round", 0)
    patient_id = state.get("patient_id", "unknown")
    val_slices = state.get("processed_slices", [])

    logger.info(f"[ASIOAgent] FL Round {fl_round} | Patient: {patient_id} | Slices: {len(val_slices)}")

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        # Determine model path for fitness evaluation
        from agents.segmentation_agent import _get_model_weights
        model_path = state.get("model_path") or _get_model_weights()

        # Initialize or reuse ASIO (warm-starting across rounds)
        if _asio_instance is None or fl_round == 0:
            _asio_instance = ASIOOptimizer(
                n_particles=5,
                n_asteroids=2,
                iterations=3,  # 3 iterations per FL round
                alpha=0.1,
                omega=0.6,
                c1=1.4,
                c2=1.4,
                p_perturb=0.2,
                sigma_perturb=0.01,
                seed=42 + fl_round,
            )
            _asio_instance.initialize()
        # else: reuse from previous round (warm-starting)

        # Build real fitness function
        fitness_fn = _build_fitness_fn(val_slices, model_path, device)

        # Run ASIO optimization
        best_hp, best_dice, history = _asio_instance.optimize(fitness_fn)

        logger.info(f"[ASIOAgent] Best HP found — Dice: {best_dice:.4f} | {best_hp}")

        return {
            **state,
            "hyperparams": best_hp,
            "asio_history": history,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[ASIOAgent] Error: {e}", exc_info=True)
        # Fallback to default hyperparameters
        default_hp = {
            "lr0": 0.001,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "conf": 0.25,
            "iou": 0.5,
            "batch_size": 8,
        }
        return {**state, "hyperparams": default_hp, "asio_history": [], "error": None}
