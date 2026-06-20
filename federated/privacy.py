"""
federated/privacy.py
FedASIO-YOLO26: Differential Privacy wrapper using Opacus DP-SGD.
Applies (epsilon, delta)-DP noise to local model gradients before FL aggregation.
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def make_private_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    target_epsilon: float = 1.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 3,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
    """
    Wrap a PyTorch model with Opacus DP-SGD for differential privacy.
    
    Args:
        model:            PyTorch model (YOLO backbone)
        optimizer:        SGD or Adam optimizer
        data_loader:      Training DataLoader
        target_epsilon:   Privacy budget ε (lower = more private)
        target_delta:     DP delta (typically 1/n, n=dataset size)
        max_grad_norm:    Gradient clipping norm (C)
        epochs:           Number of training epochs
        
    Returns:
        (private_model, private_optimizer, private_data_loader)
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        # Opacus requires BatchNorm → GroupNorm conversion
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.info(f"[PrivacyEngine] Converting incompatible layers: {len(errors)} issues")
            model = ModuleValidator.fix(model)

        privacy_engine = PrivacyEngine(accountant="rdp")  # Rényi DP accountant

        private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

        logger.info(
            f"[PrivacyEngine] DP-SGD activated — "
            f"ε={target_epsilon}, δ={target_delta}, C={max_grad_norm}, "
            f"σ={private_optimizer.noise_multiplier:.3f}"
        )

        return private_model, private_optimizer, private_loader, privacy_engine

    except ImportError:
        logger.warning("[PrivacyEngine] Opacus not available — running WITHOUT differential privacy")
        return model, optimizer, data_loader, None
    except Exception as e:
        logger.error(f"[PrivacyEngine] Error making model private: {e}")
        return model, optimizer, data_loader, None


def get_privacy_budget(privacy_engine, delta: float = 1e-5) -> Optional[float]:
    """Query the current epsilon spent so far (Rényi DP accountant)."""
    if privacy_engine is None:
        return None
    try:
        epsilon = privacy_engine.get_epsilon(delta=delta)
        return float(epsilon)
    except Exception:
        return None
