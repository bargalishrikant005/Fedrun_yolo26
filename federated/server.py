"""
federated/server.py
FedASIO-YOLO26: Flower Server with ASIO hyperparameter optimization per round.
Runs in-process using Flower's Virtual Client Engine (fastest on Apple M5).
"""
import logging
import json
import os
from typing import Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from federated.strategy import FedASIOStrategy

logger = logging.getLogger(__name__)


def get_initial_parameters(model_path: str = None):
    """Extract initial model parameters from YOLO weights."""
    from ultralytics import YOLO
    from agents.segmentation_agent import _get_model_weights

    weights = model_path or _get_model_weights()
    model = YOLO(weights)
    params = [p.cpu().detach().float().numpy() for p in model.model.parameters()]
    return ndarrays_to_parameters(params)


def run_fl_server(
    n_rounds: int = 50,
    n_clients: int = 5,
    asio_hyperparams: Optional[Dict] = None,
    model_path: Optional[str] = None,
    save_dir: str = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/metrics",
):
    """
    Launch the Flower FL server with FedASIO strategy.
    Uses Virtual Client Engine (in-process) for M5 efficiency.
    """
    os.makedirs(save_dir, exist_ok=True)

    initial_params = get_initial_parameters(model_path)

    strategy = FedASIOStrategy(
        asio_hyperparams=asio_hyperparams or {
            "lr0": 0.001, "momentum": 0.937, "weight_decay": 0.0005,
            "conf": 0.25, "iou": 0.5, "batch_size": 8,
        },
        initial_parameters=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min(n_clients, 3),
        min_available_clients=n_clients,
    )

    config = ServerConfig(num_rounds=n_rounds)

    logger.info(f"[FL Server] Starting FedASIO — {n_rounds} rounds, {n_clients} clients")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
    )

    # Save training history
    history_data = {
        "losses_distributed": [(r, float(l)) for r, l in history.losses_distributed],
        "metrics_distributed": {
            k: [(r, float(v)) for r, v in vals]
            for k, vals in history.metrics_distributed.items()
        },
        "round_metrics": strategy.round_metrics,
    }
    hist_path = os.path.join(save_dir, "fl_history.json")
    with open(hist_path, "w") as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"[FL Server] History saved: {hist_path}")

    return history, strategy.round_metrics
