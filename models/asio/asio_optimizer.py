"""
models/asio/asio_optimizer.py
FedASIO-YOLO26: ASIO — Asteroid Satellite Inspired Optimization Algorithm

Novel swarm intelligence algorithm using orbital mechanics (gravitational attraction,
satellite orbits, perturbation jumps) for 6D YOLO hyperparameter optimization.

Key Properties:
  - Asteroid particles: fine local search via orbital attraction toward gbest
  - Satellite particles: orbit nearest Asteroid via PSO-like velocity update
  - Perturbation Jump: probabilistic escape from local optima (Kozai-Lidov mechanism)
  - Dynamic allegiance: Satellites reassign host Asteroid each iteration
  - Velocity clamping: prevents 6D velocity explosion (PSO limitation)
  
Fitness Function: REAL YOLO validation Dice score (not simulated).
"""
import logging
import random
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 6D Hyperparameter Search Space
HP_BOUNDS = {
    "lr0":          (1e-5,  1e-1),    # Initial learning rate (log-scale sampled)
    "momentum":     (0.80,  0.99),    # SGD momentum
    "weight_decay": (1e-5,  1e-2),    # Weight decay (log-scale sampled)
    "conf":         (0.10,  0.90),    # YOLO confidence threshold
    "iou":          (0.30,  0.70),    # YOLO IoU threshold
    "batch_size":   (8,     32),      # Batch size (rounded to {8,16,32})
}
HP_KEYS = list(HP_BOUNDS.keys())
DIM = len(HP_KEYS)


def _sample_initial(n_particles: int) -> np.ndarray:
    """Sample initial particle positions uniformly within bounds."""
    positions = np.zeros((n_particles, DIM))
    for d, key in enumerate(HP_KEYS):
        lo, hi = HP_BOUNDS[key]
        if key in ("lr0", "weight_decay"):
            # Log-uniform sampling for learning rate and weight decay
            positions[:, d] = np.exp(
                np.random.uniform(np.log(lo), np.log(hi), n_particles)
            )
        else:
            positions[:, d] = np.random.uniform(lo, hi, n_particles)
    return positions


def _clamp(positions: np.ndarray) -> np.ndarray:
    """Clamp all particles to valid search bounds."""
    for d, key in enumerate(HP_KEYS):
        lo, hi = HP_BOUNDS[key]
        positions[:, d] = np.clip(positions[:, d], lo, hi)
    return positions


def _decode(position: np.ndarray) -> Dict:
    """Decode a particle position vector to hyperparameter dict."""
    hp = {}
    for d, key in enumerate(HP_KEYS):
        val = float(position[d])
        if key == "batch_size":
            val = int(round(val / 8.0)) * 8  # round to {8, 16, 24, 32}
            val = max(8, min(32, val))
        hp[key] = val
    return hp


class ASIOOptimizer:
    """
    Asteroid Satellite Inspired Optimization (ASIO)
    
    Topology:
      - Top-k particles by fitness → Asteroids (fine local search)
      - Remaining → Satellites (orbit nearest Asteroid via PSO-like update)
    
    Update Rules:
      Asteroid i: x_i += alpha * r * (gbest - x_i)
      Satellite i: v_i = omega*v_i + c1*r1*(host-x_i) + c2*r2*(gbest-x_i)
                   x_i += v_i
      Perturbation: x_i += N(0, sigma²) with prob p_perturb
    """

    def __init__(
        self,
        n_particles: int = 5,
        n_asteroids: int = 2,
        iterations: int = 3,
        alpha: float = 0.1,       # Orbital attraction coefficient
        omega: float = 0.6,       # Satellite inertia weight
        c1: float = 1.4,          # Cognitive coefficient (toward host Asteroid)
        c2: float = 1.4,          # Global coefficient (toward gbest)
        p_perturb: float = 0.2,   # Perturbation jump probability
        sigma_perturb: float = 0.01,  # Perturbation scale
        seed: int = 42,
    ):
        self.n_particles = n_particles
        self.n_asteroids = n_asteroids
        self.n_satellites = n_particles - n_asteroids
        self.iterations = iterations
        self.alpha = alpha
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.p_perturb = p_perturb
        self.sigma_perturb = sigma_perturb
        self.seed = seed

        # Internal state
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.personal_best: Optional[np.ndarray] = None
        self.personal_best_fitness: Optional[np.ndarray] = None
        self.global_best: Optional[np.ndarray] = None
        self.global_best_fitness: float = -np.inf
        self.history: List[Dict] = []
        self.initialized = False

    def initialize(self, warm_start: Optional[Dict] = None):
        """Initialize particle swarm. Optionally warm-start from previous best."""
        np.random.seed(self.seed)
        self.positions = _sample_initial(self.n_particles)

        if warm_start is not None:
            # Place first particle at warm-start position
            for d, key in enumerate(HP_KEYS):
                if key in warm_start:
                    self.positions[0, d] = warm_start[key]

        self.velocities = np.zeros((self.n_particles, DIM))
        # Initialize velocities with small random values
        for d, key in enumerate(HP_KEYS):
            lo, hi = HP_BOUNDS[key]
            v_max = 0.2 * (hi - lo)
            self.velocities[:, d] = np.random.uniform(-v_max, v_max, self.n_particles)

        self.personal_best = self.positions.copy()
        self.personal_best_fitness = np.full(self.n_particles, -np.inf)
        self.global_best = self.positions[0].copy()
        self.global_best_fitness = -np.inf
        self.initialized = True
        logger.debug(f"[ASIO] Initialized {self.n_particles} particles ({self.n_asteroids} Asteroids + {self.n_satellites} Satellites)")

    def optimize(self, fitness_fn: Callable[[Dict], float], warm_start: Optional[Dict] = None) -> Tuple[Dict, float, List[Dict]]:
        """
        Run ASIO optimization.
        
        Args:
            fitness_fn: Function that takes hyperparams dict → returns Dice score (float)
            warm_start: Optional previous best HP dict to seed the swarm
            
        Returns:
            (best_hyperparams, best_fitness, iteration_history)
        """
        if not self.initialized:
            self.initialize(warm_start)

        iteration_history = []

        for iteration in range(self.iterations):
            logger.info(f"[ASIO] Iteration {iteration+1}/{self.iterations}")

            # ── Step 1: Evaluate fitness for all particles ─────────────────
            fitnesses = np.zeros(self.n_particles)
            for i in range(self.n_particles):
                hp = _decode(self.positions[i])
                try:
                    fitnesses[i] = fitness_fn(hp)
                except Exception as e:
                    logger.warning(f"  Particle {i} fitness eval failed: {e}")
                    fitnesses[i] = -np.inf

            # ── Step 2: Update personal and global bests ───────────────────
            for i in range(self.n_particles):
                if fitnesses[i] > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitnesses[i]
                    self.personal_best[i] = self.positions[i].copy()
                if fitnesses[i] > self.global_best_fitness:
                    self.global_best_fitness = fitnesses[i]
                    self.global_best = self.positions[i].copy()

            # ── Step 3: Classify Asteroids and Satellites ─────────────────
            sorted_idx = np.argsort(fitnesses)[::-1]  # Descending by fitness
            asteroid_idx = sorted_idx[:self.n_asteroids]
            satellite_idx = sorted_idx[self.n_asteroids:]

            # ── Step 4: Update Asteroids (orbital attraction toward gbest) ──
            for i in asteroid_idx:
                r = np.random.uniform(0, 1, DIM)
                self.positions[i] += self.alpha * r * (self.global_best - self.positions[i])

            # ── Step 5: Update Satellites (orbit nearest Asteroid) ──────────
            for i in satellite_idx:
                # Find nearest Asteroid (dynamic allegiance)
                distances = [np.linalg.norm(self.positions[i] - self.positions[j]) for j in asteroid_idx]
                host_idx = asteroid_idx[np.argmin(distances)]
                host_pos = self.positions[host_idx]

                r1 = np.random.uniform(0, 1, DIM)
                r2 = np.random.uniform(0, 1, DIM)

                # Velocity update: PSO-like toward host Asteroid + global best
                self.velocities[i] = (
                    self.omega * self.velocities[i]
                    + self.c1 * r1 * (host_pos - self.positions[i])
                    + self.c2 * r2 * (self.global_best - self.positions[i])
                )

                # Velocity clamping (prevents 6D explosion)
                for d, key in enumerate(HP_KEYS):
                    lo, hi = HP_BOUNDS[key]
                    v_max = 0.2 * (hi - lo)
                    self.velocities[i, d] = np.clip(self.velocities[i, d], -v_max, v_max)

                self.positions[i] += self.velocities[i]

            # ── Step 6: Orbital Perturbation Jump (Kozai-Lidov escape) ──────
            for i in range(self.n_particles):
                if random.random() < self.p_perturb:
                    for d, key in enumerate(HP_KEYS):
                        lo, hi = HP_BOUNDS[key]
                        sigma = self.sigma_perturb * (hi - lo)
                        self.positions[i, d] += np.random.normal(0, sigma)

            # ── Step 7: Clamp all particles to valid search space ───────────
            self.positions = _clamp(self.positions)

            # ── Log iteration ───────────────────────────────────────────────
            best_hp = _decode(self.global_best)
            iteration_history.append({
                "iteration": iteration + 1,
                "best_fitness": float(self.global_best_fitness),
                "best_hp": best_hp,
                "mean_fitness": float(np.mean(fitnesses[fitnesses > -np.inf])) if np.any(fitnesses > -np.inf) else 0.0,
                "n_asteroids": self.n_asteroids,
                "n_satellites": self.n_satellites,
            })
            logger.info(f"  Best Dice: {self.global_best_fitness:.4f} | HP: lr={best_hp['lr0']:.5f}, momentum={best_hp['momentum']:.3f}, batch={best_hp['batch_size']}")

        best_hp = _decode(self.global_best)
        logger.info(f"[ASIO] Final best — Dice: {self.global_best_fitness:.4f} | HP: {best_hp}")
        return best_hp, float(self.global_best_fitness), iteration_history

    def get_state(self) -> Dict:
        """Return serializable optimizer state for FL round persistence."""
        return {
            "positions": self.positions.tolist() if self.positions is not None else None,
            "velocities": self.velocities.tolist() if self.velocities is not None else None,
            "global_best": self.global_best.tolist() if self.global_best is not None else None,
            "global_best_fitness": float(self.global_best_fitness),
            "history": self.history,
        }
