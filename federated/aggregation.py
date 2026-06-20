"""
federated/aggregation.py
FedASIO-YOLO26: Aggregation strategies for comparison.
Implements FedAvg, FedProx, FedAdam, and ASIO-Weighted for ablation studies.
"""
import logging
from typing import List, Tuple, Optional
from functools import reduce
import numpy as np

logger = logging.getLogger(__name__)


def fedavg(
    client_params: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:
    """
    Standard FedAvg aggregation (McMahan et al. 2017).
    Weights by number of local samples.
    """
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]
    aggregated = [
        sum(w * layer for w, layer in zip(weights, [cp[i] for cp in client_params]))
        for i in range(len(client_params[0]))
    ]
    logger.debug(f"[FedAvg] Aggregated {len(client_params)} clients | weights={[f'{w:.3f}' for w in weights]}")
    return aggregated


def fedprox(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_sizes: List[int],
    mu: float = 0.01,
) -> List[np.ndarray]:
    """
    FedProx aggregation (Li et al. 2020).
    Adds proximal term μ||w - w_global||² to penalize client drift.
    FedProx aggregation itself is FedAvg; mu affects local training loss.
    """
    # Aggregation is same as FedAvg; proximal regularization is in client training
    return fedavg(client_params, client_sizes)


def fedadam(
    client_grads: List[List[np.ndarray]],
    server_params: List[np.ndarray],
    m: List[np.ndarray],
    v: List[np.ndarray],
    eta: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
    step: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    FedAdam server-side optimizer (Reddi et al. 2021).
    Applies Adam update to the aggregated pseudo-gradient.
    """
    # Pseudo-gradient: average of client updates (delta = local - global)
    n = len(client_grads)
    pseudo_grad = [
        sum(g[i] for g in client_grads) / n
        for i in range(len(client_grads[0]))
    ]

    # Adam update
    new_m = [beta1 * m_i + (1 - beta1) * g_i for m_i, g_i in zip(m, pseudo_grad)]
    new_v = [beta2 * v_i + (1 - beta2) * (g_i ** 2) for v_i, g_i in zip(v, pseudo_grad)]

    # Bias correction
    m_hat = [m_i / (1 - beta1 ** step) for m_i in new_m]
    v_hat = [v_i / (1 - beta2 ** step) for v_i in new_v]

    new_params = [
        p + eta * m_h / (np.sqrt(v_h) + tau)
        for p, m_h, v_h in zip(server_params, m_hat, v_hat)
    ]

    logger.debug(f"[FedAdam] Step {step} | eta={eta}, β1={beta1}, β2={beta2}")
    return new_params, new_m, new_v


def asio_weighted_aggregation(
    client_params: List[List[np.ndarray]],
    client_dice_scores: List[float],
) -> List[np.ndarray]:
    """
    FedASIO Dice-weighted aggregation (novel contribution).
    Higher Dice clients contribute more to the global model.
    Clients with near-zero Dice are down-weighted to prevent poisoning.
    """
    # Softmax weighting with temperature for numerical stability
    scores = np.array(client_dice_scores, dtype=np.float64)
    scores = np.maximum(scores, 0.01)  # floor at 1% to avoid zero weight

    # Temperature-scaled softmax (T=0.5 sharpens the distribution)
    T = 0.5
    exp_scores = np.exp(scores / T)
    weights = exp_scores / exp_scores.sum()

    aggregated = [
        sum(w * layer for w, layer in zip(weights, [cp[i] for cp in client_params]))
        for i in range(len(client_params[0]))
    ]

    logger.info(f"[ASIO-Weighted] Dice scores: {[f'{d:.4f}' for d in client_dice_scores]} "
                f"→ Weights: {[f'{w:.3f}' for w in weights]}")
    return aggregated


def get_aggregator(strategy: str):
    """Factory function — returns the aggregation function by name."""
    strategies = {
        "fedavg": fedavg,
        "fedprox": fedprox,
        "fedadam": fedadam,
        "fedasio": asio_weighted_aggregation,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(strategies.keys())}")
    return strategies[strategy]
