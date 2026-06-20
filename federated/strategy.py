"""
federated/strategy.py
FedASIO-YOLO26: Custom Flower Strategy — FedASIO
Weighted FedAvg aggregation where client weights are proportional to their Dice scores.
Integrates ASIO optimizer for server-side hyperparameter optimization each round.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, EvaluateRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, EvaluateIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class FedASIOStrategy(FedAvg):
    """
    FedASIO Strategy: Dice-weighted federated aggregation with ASIO HP optimization.
    
    Aggregation: weighted average where weight_i = client_i_dice / sum(all_dice)
    Hyperparameters: broadcast ASIO-optimized HP dict to all clients each round.
    """

    def __init__(
        self,
        asio_hyperparams: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asio_hyperparams = asio_hyperparams or {
            "lr0": 0.001,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "conf": 0.25,
            "iou": 0.5,
            "batch_size": 8,
        }
        self.round_metrics: List[Dict] = []

    def configure_fit(
        self, server_round: int, parameters: Parameters,
        client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send global model + ASIO hyperparameters to each client."""
        config = {
            "fl_round": server_round,
            **{f"hp_{k}": str(v) for k, v in self.asio_hyperparams.items()},
        }
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        logger.info(f"[Server Round {server_round}] Sending ASIO HPs: {self.asio_hyperparams}")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Dice-weighted aggregation of client model updates."""
        if not results:
            return None, {}
        if failures:
            logger.warning(f"[Server Round {server_round}] {len(failures)} clients failed")

        # Extract weights and client Dice scores
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics)
            for _, fit_res in results
        ]

        # Get Dice score for each client (fallback to equal weight)
        dice_scores = []
        for _, metrics in weights_results:
            dice = metrics.get("dice", 0.5) if metrics else 0.5
            dice_scores.append(max(float(dice), 0.01))  # avoid zero weight

        total_dice = sum(dice_scores)
        client_weights = [d / total_dice for d in dice_scores]

        logger.info(f"[Server Round {server_round}] Client Dice: {[f'{d:.4f}' for d in dice_scores]}")
        logger.info(f"[Server Round {server_round}] Client weights: {[f'{w:.3f}' for w in client_weights]}")

        # Weighted average of model parameters
        weighted_params = []
        for params, w in zip([p for p, _ in weights_results], client_weights):
            weighted_params.append([layer * w for layer in params])

        aggregated = [
            reduce(np.add, [wp[i] for wp in weighted_params])
            for i in range(len(weighted_params[0]))
        ]

        # Aggregate round metrics
        mean_dice = float(np.mean(dice_scores))
        agg_metrics = {
            "fl_round": server_round,
            "mean_dice": mean_dice,
            "max_dice": float(max(dice_scores)),
            "min_dice": float(min(dice_scores)),
            "n_clients": len(results),
        }
        self.round_metrics.append(agg_metrics)
        logger.info(f"[Server Round {server_round}] Aggregated — Mean Dice: {mean_dice:.4f}")

        return ndarrays_to_parameters(aggregated), agg_metrics

    def aggregate_evaluate(
        self, server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from all clients."""
        if not results:
            return None, {}

        all_dice = [res.metrics.get("dice", 0.0) for _, res in results if res.metrics]
        all_loss = [res.loss for _, res in results]
        mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
        mean_loss = float(np.mean(all_loss)) if all_loss else 1.0

        logger.info(f"[Server Round {server_round}] Eval — Loss: {mean_loss:.4f} | Mean Dice: {mean_dice:.4f}")
        return mean_loss, {"dice": mean_dice, "fl_round": server_round}

    def update_hyperparams(self, new_hp: Dict):
        """Called by server to update ASIO-optimized hyperparameters between rounds."""
        self.asio_hyperparams = new_hp
        logger.info(f"[FedASIO Strategy] Hyperparams updated: {new_hp}")
