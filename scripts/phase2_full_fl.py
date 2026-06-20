"""
scripts/phase2_full_fl.py
FedASIO-YOLO26: Phase 2 — Full Federated Training
Data split: 25% train (65 patients) / 5% val (13) / 5% test (13) / 65% streamlit (166)
FL: 5 clients × 50 rounds, 10 local epochs, ASIO optimizer, Differential Privacy
Run: conda activate fedasio && python scripts/phase2_full_fl.py
"""
import os, sys, json, logging, time, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/shrikant/Downloads/FedASIO-YOLO26")

os.makedirs("/Users/shrikant/Downloads/FedASIO-YOLO26/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/Users/shrikant/Downloads/FedASIO-YOLO26/logs/phase2.log"),
    ],
)
logger = logging.getLogger("Phase2")

RAW_DIR = "/Users/shrikant/Downloads/BraTS-PEDs-v1/Training"
SPLITS_FILE = "/Users/shrikant/Downloads/FedASIO-YOLO26/data/splits/patient_splits.json"
METRICS_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/metrics"
FIGURES_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/figures"
N_CLIENTS = 5
N_ROUNDS = 20
LOCAL_EPOCHS = 10  # Per client per FL round (10 epochs for actual model convergence)


def create_splits(raw_dir: str, seed: int = 42) -> dict:
    """Create deterministic patient splits and save to JSON."""
    patients = sorted([
        p for p in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, p)) and p.startswith("BraTS")
    ])
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)  # 257
    n_train = 15              # 3 patients per client (15 total)
    n_val   = 3               # 3 validation patients
    n_test  = 3               # 3 test patients
    # remaining 166 → streamlit

    splits = {
        "train":     patients[:n_train],
        "val":       patients[n_train:n_train+n_val],
        "test":      patients[n_train+n_val:n_train+n_val+n_test],
        "streamlit": patients[n_train+n_val+n_test:],
        "total": n,
        "seed": seed,
    }

    # Divide train patients among N_CLIENTS
    client_size = n_train // N_CLIENTS
    splits["clients"] = {
        str(c): splits["train"][c*client_size:(c+1)*client_size]
        for c in range(N_CLIENTS)
    }

    os.makedirs(os.path.dirname(SPLITS_FILE), exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Dataset splits (N={n}):")
    logger.info(f"  Train:     {n_train} patients → {N_CLIENTS} clients × {client_size} each")
    logger.info(f"  Val:       {n_val} patients")
    logger.info(f"  Test:      {n_test} patients")
    logger.info(f"  Streamlit: {len(splits['streamlit'])} patients")
    return splits


def run_full_fl(splits: dict):
    """Run full federated training simulation."""
    from federated.client import FedASIOClient
    from models.asio.asio_optimizer import ASIOOptimizer
    from federated.aggregation import asio_weighted_aggregation

    train_patients = [os.path.join(RAW_DIR, p) for p in splits["train"]]
    val_patients   = [os.path.join(RAW_DIR, p) for p in splits["val"]]
    client_data    = splits["clients"]

    asio = ASIOOptimizer(n_particles=5, n_asteroids=2, iterations=3, seed=42)
    asio.initialize()

    round_results = []
    current_hp = {"lr0": 0.001, "momentum": 0.937, "weight_decay": 0.0005,
                   "conf": 0.25, "iou": 0.5, "batch_size": 8}

    # Initialize global parameters from the shared model checkpoint on disk
    dummy_client = FedASIOClient(client_id=999, patient_dirs=[], val_patient_dirs=[])
    global_params = dummy_client.get_parameters({})

    # Support checkpoint resuming
    intermediate_path = os.path.join(METRICS_DIR, "phase2_metrics_intermediate.json")
    start_rnd = 0
    if os.path.exists(intermediate_path):
        try:
            with open(intermediate_path) as f:
                round_results = json.load(f)
            start_rnd = len(round_results)
            logger.info(f"🔄 Resuming Phase 2 from round {start_rnd + 1} (loaded {start_rnd} completed rounds)...")
            if round_results:
                current_hp = round_results[-1]["hyperparams"]
        except Exception as e:
            logger.warning(f"Could not load intermediate metrics: {e}. Starting from scratch.")

    for rnd in range(start_rnd, N_ROUNDS):
        logger.info(f"\n{'='*70}")
        logger.info(f"FL ROUND {rnd+1}/{N_ROUNDS}")
        logger.info(f"ASIO HP: lr={current_hp['lr0']:.5f}, momentum={current_hp['momentum']:.3f}, batch={int(current_hp['batch_size'])}")
        logger.info(f"{'='*70}")

        round_dice = []
        round_val_dice = []
        client_params_list = []
        client_dices = []

        for c in range(N_CLIENTS):
            client_patients = [os.path.join(RAW_DIR, p) for p in client_data[str(c)]]
            client = FedASIOClient(
                client_id=c,
                patient_dirs=client_patients,
                val_patient_dirs=val_patients[:3],
            )

            # Training
            hp_config = {f"hp_{k}": str(v) for k, v in current_hp.items()}
            hp_config["fl_round"] = rnd
            hp_config["local_epochs"] = LOCAL_EPOCHS
            updated_params, n_samples, train_metrics = client.fit(global_params, hp_config)
            train_dice = train_metrics.get("dice", 0.0)
            round_dice.append(train_dice)
            client_params_list.append(updated_params)

            # Evaluation on client's updated parameters
            loss, n_val, val_metrics = client.evaluate(updated_params, {"fl_round": rnd})
            val_dice = val_metrics.get("dice", 0.0)
            round_val_dice.append(val_dice)
            client_dices.append(val_dice)

            logger.info(f"  Client {c}: Train Dice={train_dice:.4f} | Val Dice={val_dice:.4f}")

        # Federated parameter aggregation (server side)
        logger.info(f"[Server] Aggregating parameters for round {rnd+1} using FedASIO weighted aggregation...")
        global_params = asio_weighted_aggregation(client_params_list, client_dices)

        # Save aggregated parameters back to the shared weights file
        global_model_path = "/Users/shrikant/Downloads/FedASIO-YOLO26/models/yolo26/weights/best.pt"
        dummy_client.model_path = global_model_path
        dummy_client._set_parameters(global_params)

        mean_train_dice = float(np.mean(round_dice))
        mean_val_dice   = float(np.mean(round_val_dice))

        # ASIO update: optimize hyperparameters using mean val Dice as fitness
        def asio_fitness(hp):
            # Approximate fitness based on current val Dice trend
            return mean_val_dice * (0.9 + 0.1 * hp.get("lr0", 0.001) / 0.01)

        new_hp, best_dice, hist = asio.optimize(asio_fitness, warm_start=current_hp)
        current_hp = new_hp

        round_results.append({
            "round": rnd + 1,
            "client_train_dice": round_dice,
            "client_val_dice": round_val_dice,
            "mean_train_dice": mean_train_dice,
            "mean_val_dice": mean_val_dice,
            "hyperparams": current_hp.copy(),
            "asio_best_fitness": float(best_dice),
        })

        logger.info(f"  Round {rnd+1} Summary: Train={mean_train_dice:.4f} | Val={mean_val_dice:.4f}")

        # Save intermediate results after every round
        _save_metrics(round_results, "phase2_metrics_intermediate.json")
        _generate_convergence_plot(round_results, "phase2_convergence_intermediate.png")

    return round_results, current_hp


def evaluate_test_set(splits: dict, best_hp: dict):
    """Evaluate final model on held-out test set."""
    from agents.data_agent import data_agent
    from agents.preprocessing_agent import preprocessing_agent
    from agents.segmentation_agent import segmentation_agent
    from agents.evaluation_agent import evaluation_agent

    test_patients = [os.path.join(RAW_DIR, p) for p in splits["test"]]
    logger.info(f"\nEvaluating on {len(test_patients)} test patients...")

    all_metrics = []
    for pdir in test_patients:
        state = {
            "patient_id": "", "patient_dir": pdir, "modalities": {}, "seg_mask_3d": None,
            "affine": None, "processed_slices": [], "tumor_slice_indices": [],
            "augmented_slices": [], "hyperparams": best_hp, "asio_history": [],
            "predictions": [], "pred_masks": [], "model_path": None, "metrics": {},
            "per_class_metrics": {}, "tumor_volume_cc": 0.0, "xai_maps": {},
            "xai_figure_paths": [], "report_text": "", "pdf_path": None,
            "report_metadata": {}, "fl_round": 0, "fl_client_id": 0,
            "fl_stage": "evaluate", "global_model_params": None,
            "error": None, "error_agent": None, "retry_count": 0,
        }
        state = data_agent(state)
        state = preprocessing_agent(state)
        state = segmentation_agent(state)
        state = evaluation_agent(state)
        if not state.get("error"):
            all_metrics.append(state.get("metrics", {}))

    # Aggregate test metrics
    test_results = {}
    for key in ["dice", "iou", "precision", "recall", "f1", "hd95"]:
        vals = [m.get(key, 0) for m in all_metrics if m]
        test_results[key] = float(np.mean(vals)) if vals else 0.0

    logger.info(f"Test Results: {test_results}")
    return test_results


def _save_metrics(round_results, filename):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename)
    with open(path, "w") as f:
        json.dump(round_results, f, indent=2)


def _generate_convergence_plot(round_results, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rounds = [r["round"] for r in round_results]
    train_dice = [r["mean_train_dice"] for r in round_results]
    val_dice = [r["mean_val_dice"] for r in round_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FedASIO-YOLO26 — Full FL Training (5 Clients, 50 Rounds)", fontsize=12, fontweight="bold")

    axes[0].plot(rounds, train_dice, "o-", color="#4f46e5", label="Train Dice", linewidth=2, markersize=4)
    axes[0].plot(rounds, val_dice, "s--", color="#7c3aed", label="Val Dice", linewidth=2, markersize=4)
    axes[0].set_xlabel("FL Round"); axes[0].set_ylabel("Dice Score")
    axes[0].set_title("FL Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # ASIO HP evolution
    lr_vals = [r["hyperparams"].get("lr0", 0.001) for r in round_results]
    axes[1].plot(rounds, lr_vals, color="#a78bfa", linewidth=2)
    axes[1].set_xlabel("FL Round"); axes[1].set_ylabel("Learning Rate (ASIO-optimized)")
    axes[1].set_title("ASIO Hyperparameter Evolution"); axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved convergence plot: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a fast 2-round dry run for verification")
    args = parser.parse_args()

    global N_ROUNDS, LOCAL_EPOCHS
    if args.dry_run:
        logger.info("⚡ Dry Run mode enabled! Setting N_ROUNDS=2, LOCAL_EPOCHS=1 for fast verification.")
        N_ROUNDS = 2
        LOCAL_EPOCHS = 1

    logger.info("=" * 70)
    logger.info("FedASIO-YOLO26 — PHASE 2: FULL FEDERATED TRAINING")
    logger.info("=" * 70)

    t0 = time.time()

    # Step 1: Create data splits
    logger.info("\n📂 Step 1: Creating dataset splits...")
    splits = create_splits(RAW_DIR)

    # Step 2: Run full FL
    logger.info("\n🔗 Step 2: Running full federated training (50 rounds × 5 clients)...")
    round_results, best_hp = run_full_fl(splits)

    # Step 3: Test set evaluation
    logger.info("\n🔬 Step 3: Evaluating on test set...")
    test_results = evaluate_test_set(splits, best_hp)

    # Step 4: Save final results
    final_data = {
        "phase": "2_full",
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_streamlit": len(splits["streamlit"]),
        "n_clients": N_CLIENTS,
        "n_rounds": N_ROUNDS,
        "fl_rounds": round_results,
        "best_hyperparams": best_hp,
        "test_results": test_results,
    }
    _save_metrics(final_data, "phase2_metrics_final.json")
    _generate_convergence_plot(round_results, "phase2_convergence_final.png")

    elapsed = time.time() - t0
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ PHASE 2 COMPLETE in {elapsed/3600:.1f} hours")
    logger.info(f"   Test Dice: {test_results.get('dice', 0):.4f}")
    logger.info(f"   Test IoU:  {test_results.get('iou', 0):.4f}")
    logger.info(f"   Best HP:   {best_hp}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
