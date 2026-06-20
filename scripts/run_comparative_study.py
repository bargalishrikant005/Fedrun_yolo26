"""
scripts/run_comparative_study.py
FedASIO-YOLO26: 2-Client Federated Learning Comparative Study
Comparing:
  1. YOLO26 + ASIO (Dice-weighted aggregation & ASIO hyperparams)
  2. YOLO12 + ASIO (Dice-weighted aggregation & ASIO hyperparams)
  3. YOLO12 Baseline (FedAvg aggregation & fixed standard hyperparams)
"""
import os
import sys
import json
import logging
import time
import random
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Insert project root to path
sys.path.insert(0, "/Users/shrikant/Downloads/FedASIO-YOLO26")

os.makedirs("/Users/shrikant/Downloads/FedASIO-YOLO26/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/Users/shrikant/Downloads/FedASIO-YOLO26/logs/comparison_study.log"),
    ],
)
logger = logging.getLogger("ComparativeStudy")

RAW_DIR = "/Users/shrikant/Downloads/BraTS-PEDs-v1/Training"
METRICS_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/metrics"
FIGURES_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/figures"
SPLITS_FILE = "/Users/shrikant/Downloads/FedASIO-YOLO26/data/splits/comparison_splits.json"

N_CLIENTS = 2


def create_splits(raw_dir: str, seed: int = 42) -> dict:
    """Create deterministic patient splits for 2 clients and save to JSON."""
    patients = sorted([
        p for p in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, p)) and p.startswith("BraTS")
    ])
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = 6              # 3 patients per client for 2 clients
    n_val   = 3              # 3 validation patients
    n_test  = 3              # 3 test patients

    splits = {
        "train":     patients[:n_train],
        "val":       patients[n_train:n_train+n_val],
        "test":      patients[n_train+n_val:n_train+n_val+n_test],
        "streamlit": patients[n_train+n_val+n_test:],
        "total": n,
        "seed": seed,
    }

    # Divide train patients among N_CLIENTS (2 clients)
    client_size = n_train // N_CLIENTS
    splits["clients"] = {
        str(c): splits["train"][c*client_size:(c+1)*client_size]
        for c in range(N_CLIENTS)
    }

    os.makedirs(os.path.dirname(SPLITS_FILE), exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Dataset splits created for 2 clients (N={n}):")
    logger.info(f"  Train:     {n_train} patients → {N_CLIENTS} clients × {client_size} each")
    logger.info(f"  Val:       {n_val} patients")
    logger.info(f"  Test:      {n_test} patients")
    return splits


def get_clean_yolo12_3class_weights() -> str:
    """Ensure we have a clean 3-class YOLO12-seg weights checkpoint on disk."""
    dest_path = "/Users/shrikant/Downloads/FedASIO-YOLO26/models/yolo12_3class.pt"
    if os.path.exists(dest_path):
        logger.info(f"Using existing clean 3-class YOLO12 weights at: {dest_path}")
        return dest_path
        
    logger.info("🔧 Creating a clean 3-class YOLO12-seg model from pre-trained weights...")
    from ultralytics import YOLO
    
    # Load raw 80-class YOLO12 model
    possible_paths = [
        "/Users/shrikant/Downloads/yolo vs 26/yolo12n-seg.pt",
        "yolo12n-seg.pt",
        "/content/yolo12n-seg.pt",
        "/content/drive/MyDrive/yolo12n-seg.pt",
        "/content/drive/MyDrive/yolo vs 26/yolo12n-seg.pt",
    ]
    raw_weights = None
    for p in possible_paths:
        if os.path.exists(p):
            raw_weights = p
            break
            
    if raw_weights is None:
        raise FileNotFoundError(
            "Missing raw YOLO12 model checkpoint. Please drag & drop 'yolo12n-seg.pt' "
            "into Google Colab's file panel or upload it to Google Drive."
        )
        
    logger.info(f"Using raw weights from: {raw_weights}")
    model = YOLO(raw_weights)
    
    # Run a dummy training on 1 slice to force YOLO to reinitialize heads for nc=3
    dummy_dir = "/Users/shrikant/Downloads/FedASIO-YOLO26/data/processed/yolo12_init"
    os.makedirs(dummy_dir, exist_ok=True)
    
    from agents.segmentation_agent import _create_data_yaml
    data_yaml = _create_data_yaml(dummy_dir, "train")
    
    # Create a dummy image and label
    images_dir = os.path.join(dummy_dir, "images", "train")
    labels_dir = os.path.join(dummy_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    import cv2
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(images_dir, "dummy.png"), dummy_img)
    
    # Write a dummy YOLO label file: class x1 y1 x2 y2 ... (polygon format for segmentation)
    # Class 0, with a square polygon in the center
    with open(os.path.join(labels_dir, "dummy.txt"), "w") as lf:
        lf.write("0 0.45 0.45 0.55 0.45 0.55 0.55 0.45 0.55\n")
        
    # Run single-epoch dummy training on CPU to shape classification head to nc=3
    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=256,
        batch=1,
        device="cpu",
        workers=0,
        amp=False,
        project="/Users/shrikant/Downloads/FedASIO-YOLO26/runs",
        name="yolo12_init",
        exist_ok=True,
        verbose=False,
        plots=False,
        save=True,
    )
    
    best_init = "/Users/shrikant/Downloads/FedASIO-YOLO26/runs/yolo12_init/weights/best.pt"
    if not os.path.exists(best_init):
        # Fallback to last.pt if best.pt is not saved for 1 epoch
        best_init = "/Users/shrikant/Downloads/FedASIO-YOLO26/runs/yolo12_init/weights/last.pt"
        
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(best_init, dest_path)
    logger.info(f"✅ Clean 3-class YOLO12 weights saved to: {dest_path}")
    
    # Clean up dummy folders
    try:
        shutil.rmtree(dummy_dir)
        shutil.rmtree("/Users/shrikant/Downloads/FedASIO-YOLO26/runs/yolo12_init")
    except Exception as e:
        pass
        
    return dest_path


def run_federated_simulation(
    scenario_name: str,
    base_model_path: str,
    splits: dict,
    rounds: int,
    epochs: int,
    use_asio: bool,
) -> dict:
    """Run federated training simulation for a specific model configuration."""
    from federated.client import FedASIOClient
    from models.asio.asio_optimizer import ASIOOptimizer, HP_BOUNDS
    from federated.aggregation import asio_weighted_aggregation, fedavg

    is_yolo12 = "yolo12" in scenario_name.lower()
    if is_yolo12:
        HP_BOUNDS["lr0"] = (1e-4, 5e-3)
        HP_BOUNDS["conf"] = (0.05, 0.25)
        logger.info(f"🔧 Restricting ASIO Search bounds for YOLO12: lr0={HP_BOUNDS['lr0']}, conf={HP_BOUNDS['conf']}")
    else:
        HP_BOUNDS["lr0"] = (1e-5, 1e-1)
        HP_BOUNDS["conf"] = (0.10, 0.90)
        logger.info(f"🔧 Using standard ASIO Search bounds for YOLO26: lr0={HP_BOUNDS['lr0']}, conf={HP_BOUNDS['conf']}")

    logger.info(f"\n⚡ Starting Scenario: {scenario_name.upper()} ({rounds} rounds, {epochs} local epochs)")
    
    # Copy base weights to a scenario-specific active checkpoint file
    scenario_model_path = f"/Users/shrikant/Downloads/FedASIO-YOLO26/models/active_{scenario_name}.pt"
    shutil.copy(base_model_path, scenario_model_path)

    val_patients = [os.path.join(RAW_DIR, p) for p in splits["val"]]
    client_data = splits["clients"]

    # Initialize client variables
    dummy_client = FedASIOClient(client_id=999, patient_dirs=[], val_patient_dirs=[])
    dummy_client.model_path = scenario_model_path
    global_params = dummy_client.get_parameters({})

    # Setup ASIO Optimizer if enabled
    asio = None
    if use_asio:
        asio = ASIOOptimizer(n_particles=3, n_asteroids=1, iterations=2, seed=42)
        asio.initialize()

    current_hp = {
        "lr0": 0.001 if is_yolo12 else 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "conf": 0.15 if is_yolo12 else 0.25,
        "iou": 0.5,
        "batch_size": 8,
    }

    round_history = []

    for rnd in range(rounds):
        logger.info(f"--- {scenario_name} | Round {rnd+1}/{rounds} ---")
        
        client_params_list = []
        client_dices = []
        client_sizes = []
        round_train_dice = []
        round_val_dice = []

        for c in range(N_CLIENTS):
            client_patients = [os.path.join(RAW_DIR, p) for p in client_data[str(c)]]
            client = FedASIOClient(
                client_id=c,
                patient_dirs=client_patients,
                val_patient_dirs=val_patients,
            )
            client.model_path = scenario_model_path

            # Load global parameters to the client's local file
            client._set_parameters(global_params)

            # Training Config
            if use_asio:
                hp_config = {f"hp_{k}": str(v) for k, v in current_hp.items()}
            else:
                # Baseline uses standard fixed hyperparameters
                hp_config = {
                    "hp_lr0": "0.001" if is_yolo12 else "0.01",
                    "hp_momentum": "0.937",
                    "hp_weight_decay": "0.0005",
                    "hp_conf": "0.15" if is_yolo12 else "0.25",
                    "hp_iou": "0.5",
                    "hp_batch_size": "8",
                }
            hp_config["fl_round"] = rnd
            hp_config["local_epochs"] = epochs

            # Fit and evaluate
            updated_params, n_samples, train_metrics = client.fit(global_params, hp_config)
            train_dice = train_metrics.get("dice", 0.0)
            round_train_dice.append(train_dice)
            client_params_list.append(updated_params)
            client_sizes.append(n_samples)

            eval_config = {f"hp_{k}": str(v) for k, v in current_hp.items()}
            eval_config["fl_round"] = rnd
            loss, n_val, val_metrics = client.evaluate(updated_params, eval_config)
            val_dice = val_metrics.get("dice", 0.0)
            round_val_dice.append(val_dice)
            client_dices.append(val_dice)

            logger.info(f"  Client {c}: Train Dice={train_dice:.4f} | Val Dice={val_dice:.4f}")

        # Parameter Aggregation
        if use_asio:
            logger.info("[Server] Aggregating parameters using ASIO Dice-weighted aggregation...")
            global_params = asio_weighted_aggregation(client_params_list, client_dices)
        else:
            logger.info("[Server] Aggregating parameters using standard FedAvg...")
            global_params = fedavg(client_params_list, client_sizes)

        # Write aggregated weights back to active scenario model
        dummy_client._set_parameters(global_params)

        mean_train = float(np.mean(round_train_dice))
        mean_val = float(np.mean(round_val_dice))
        logger.info(f"Round {rnd+1} Global Summary: Train Dice={mean_train:.4f} | Val Dice={mean_val:.4f}")

        # Hyperparameter Tuning using ASIO Optimizer
        if use_asio:
            def asio_fitness(hp):
                return mean_val * (0.9 + 0.1 * hp.get("lr0", 0.01) / 0.1)
            new_hp, best_fit, _ = asio.optimize(asio_fitness, warm_start=current_hp)
            current_hp = new_hp
            logger.info(f"  ASIO Optimizer updated HP: lr0={current_hp['lr0']:.5f}, momentum={current_hp['momentum']:.3f}")

        round_history.append({
            "round": rnd + 1,
            "train_dice": mean_train,
            "val_dice": mean_val,
        })

    # Evaluate final model on the held-out test set
    test_results = evaluate_final_model(scenario_model_path, splits, current_hp)
    logger.info(f"🎯 Final Test Evaluation for {scenario_name}: {test_results}")

    # Cleanup active checkpoint file
    try:
        os.remove(scenario_model_path)
    except:
        pass

    return {
        "scenario": scenario_name,
        "rounds": round_history,
        "test_results": test_results,
    }


def evaluate_final_model(model_path: str, splits: dict, hp: dict) -> dict:
    """Evaluate final global model parameters on the test patients."""
    from agents.data_agent import data_agent
    from agents.preprocessing_agent import preprocessing_agent
    from agents.segmentation_agent import segmentation_agent
    from agents.evaluation_agent import evaluation_agent

    test_patients = [os.path.join(RAW_DIR, p) for p in splits["test"]]
    all_metrics = []

    for pdir in test_patients:
        state = {
            "patient_id": "", "patient_dir": pdir, "modalities": {}, "seg_mask_3d": None,
            "affine": None, "processed_slices": [], "tumor_slice_indices": [],
            "augmented_slices": [], "hyperparams": hp, "asio_history": [],
            "predictions": [], "pred_masks": [], "model_path": model_path, "metrics": {},
            "per_class_metrics": {}, "tumor_volume_cc": 0.0, "xai_maps": {},
            "xai_figure_paths": [], "report_text": "", "pdf_path": None,
            "report_metadata": {}, "fl_round": 99, "fl_client_id": 99,
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
    return test_results


def generate_comparison_plot(results_summary: list):
    """Plot final bar chart comparing test Dice score across the three configs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    scenarios = [r["scenario"].upper() for r in results_summary]
    dice_scores = [r["test_results"]["dice"] for r in results_summary]
    iou_scores = [r["test_results"]["iou"] for r in results_summary]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Comparative Study: YOLO26 vs YOLO12 (2 Clients)", fontsize=14, fontweight="bold")

    rects1 = ax.bar(x - width/2, dice_scores, width, label='Test Dice', color='#6366f1')
    rects2 = ax.bar(x + width/2, iou_scores, width, label='Test IoU', color='#a78bfa')

    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics across Architectures and Optimizers')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)

    # Attach score values on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "comparison_study.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved comparative bar chart: {plot_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 2-round verification")
    args = parser.parse_args()

    rounds = args.rounds
    epochs = args.epochs
    if args.dry_run:
        logger.info("⚡ Dry Run mode enabled! Overriding config to 2 rounds, 1 epoch per round.")
        rounds = 2
        epochs = 1

    t0 = time.time()

    # Step 1: Create split configuration
    logger.info("📂 Step 1: Configuring client dataset splits...")
    splits = create_splits(RAW_DIR)

    # Step 2: Initialize clean base weights
    logger.info("\n📂 Step 2: Initializing baseline weights...")
    yolo26_weights = "/Users/shrikant/Downloads/FedASIO-YOLO26/models/yolo26/weights/best.pt"
    yolo12_weights = get_clean_yolo12_3class_weights()

    # Step 3: Run the simulations
    results = []

    # Scenario 1: YOLO26 + ASIO
    res_yolo26_asio = run_federated_simulation(
        scenario_name="yolo26_asio",
        base_model_path=yolo26_weights,
        splits=splits,
        rounds=rounds,
        epochs=epochs,
        use_asio=True,
    )
    results.append(res_yolo26_asio)

    # Scenario 2: YOLO26 Baseline (FedAvg, no ASIO)
    res_yolo26_baseline = run_federated_simulation(
        scenario_name="yolo26_baseline",
        base_model_path=yolo26_weights,
        splits=splits,
        rounds=rounds,
        epochs=epochs,
        use_asio=False,
    )
    results.append(res_yolo26_baseline)

    # Scenario 3: YOLO12 + ASIO
    res_yolo12_asio = run_federated_simulation(
        scenario_name="yolo12_asio",
        base_model_path=yolo12_weights,
        splits=splits,
        rounds=rounds,
        epochs=epochs,
        use_asio=True,
    )
    results.append(res_yolo12_asio)

    # Scenario 4: YOLO12 Baseline (FedAvg, no ASIO)
    res_yolo12_baseline = run_federated_simulation(
        scenario_name="yolo12_baseline",
        base_model_path=yolo12_weights,
        splits=splits,
        rounds=rounds,
        epochs=epochs,
        use_asio=False,
    )
    results.append(res_yolo12_baseline)

    # Step 4: Write comparison summary to disk
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "comparison_study.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Saved comparison metrics: {metrics_path}")

    # Step 5: Generate evaluation chart
    generate_comparison_plot(results)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ COMPARATIVE STUDY COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")
    logger.info(f"{'Scenario':<20} | {'Test Dice':<10} | {'Test IoU':<10} | {'HD95 (mm)':<10}")
    logger.info("-" * 60)
    for r in results:
        scen = r["scenario"].upper()
        res = r["test_results"]
        logger.info(f"{scen:<20} | {res['dice']:<10.4f} | {res['iou']:<10.4f} | {res['hd95']:<10.1f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
