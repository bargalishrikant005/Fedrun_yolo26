"""
scripts/phase1_sample.py
FedASIO-YOLO26: Phase 1 — Sample Pipeline (5 patients, 2 FL clients)
Quick sanity check with full pipeline: preprocessing → ASIO → FL training → eval → XAI → report → graphs.
Run: conda activate fedasio && python scripts/phase1_sample.py
"""
import os, sys, json, logging, time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Project root
sys.path.insert(0, "/Users/shrikant/Downloads/FedASIO-YOLO26")

# Setup logging
os.makedirs("/Users/shrikant/Downloads/FedASIO-YOLO26/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/Users/shrikant/Downloads/FedASIO-YOLO26/logs/phase1.log"),
    ],
)
logger = logging.getLogger("Phase1")

RAW_DIR = "/Users/shrikant/Downloads/BraTS-PEDs-v1/Training"
PROCESSED_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/data/processed/sample"
FIGURES_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/figures"
METRICS_DIR = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports/metrics"
N_SAMPLE_PATIENTS = 5
N_FL_CLIENTS = 2
N_FL_ROUNDS = 3  # Quick sample: only 3 rounds


def prepare_sample_data():
    """Step 1: Load and preprocess 5 patients, save as YOLO format."""
    from agents.data_agent import scan_dataset, data_agent
    from agents.preprocessing_agent import preprocessing_agent, save_slices_to_disk
    from agents.augmentation_agent import augmentation_agent

    patient_dirs = scan_dataset(RAW_DIR, limit=N_SAMPLE_PATIENTS)
    logger.info(f"Processing {len(patient_dirs)} sample patients...")

    all_slices = []
    for i, pdir in enumerate(patient_dirs):
        logger.info(f"  Patient {i+1}/{len(patient_dirs)}: {os.path.basename(pdir)}")
        state = {
            "patient_id": "", "patient_dir": pdir, "modalities": {},
            "seg_mask_3d": None, "affine": None, "processed_slices": [],
            "tumor_slice_indices": [], "augmented_slices": [], "hyperparams": {},
            "asio_history": [], "predictions": [], "pred_masks": [], "model_path": None,
            "metrics": {}, "per_class_metrics": {}, "tumor_volume_cc": 0.0,
            "xai_maps": {}, "xai_figure_paths": [], "report_text": "", "pdf_path": None,
            "report_metadata": {}, "fl_round": 0, "fl_client_id": 0,
            "fl_stage": "train", "global_model_params": None,
            "error": None, "error_agent": None, "retry_count": 0,
        }
        state = data_agent(state)
        if state.get("error"):
            logger.error(f"  ❌ DataAgent failed: {state['error']}")
            continue
        state = preprocessing_agent(state)
        if state.get("error"):
            logger.error(f"  ❌ PreprocessAgent failed: {state['error']}")
            continue
        state = augmentation_agent(state)
        slices = state.get("augmented_slices", state.get("processed_slices", []))
        all_slices.extend(slices)
        logger.info(f"  ✅ {len(slices)} slices extracted")

    # Save slices
    n_saved = save_slices_to_disk(all_slices[:200], PROCESSED_DIR, "train")
    logger.info(f"Saved {n_saved} slices to {PROCESSED_DIR}")
    return all_slices[:200]


def run_sample_fl(all_slices):
    """Step 2: Run 2-client FL simulation for 3 rounds."""
    from federated.client import FedASIOClient
    from agents.data_agent import scan_dataset

    patient_dirs = scan_dataset(RAW_DIR, limit=N_SAMPLE_PATIENTS)
    split = len(patient_dirs) // N_FL_CLIENTS

    round_results = []

    for rnd in range(N_FL_ROUNDS):
        logger.info(f"\n{'='*60}")
        logger.info(f"FL ROUND {rnd+1}/{N_FL_ROUNDS}")
        logger.info(f"{'='*60}")

        round_dice = []
        for c in range(N_FL_CLIENTS):
            client_patients = patient_dirs[c*split:(c+1)*split]
            val_patients = patient_dirs[:2]  # Use first 2 as val
            client = FedASIOClient(
                client_id=c,
                patient_dirs=client_patients,
                val_patient_dirs=val_patients,
            )
            # Simulate local training
            logger.info(f"  Client {c}: {len(client_patients)} patients")
            params = client.get_parameters({})
            _, _, train_metrics = client.fit(params, {
                "fl_round": rnd,
                "hp_lr0": "0.001", "hp_momentum": "0.937",
                "hp_weight_decay": "0.0005", "hp_conf": "0.25",
                "hp_iou": "0.5", "hp_batch_size": "8",
            })
            round_dice.append(train_metrics.get("dice", 0.0))
            logger.info(f"  Client {c} Dice: {round_dice[-1]:.4f}")

        round_results.append({
            "round": rnd + 1,
            "client_dice": round_dice,
            "mean_dice": float(np.mean(round_dice)),
        })
        logger.info(f"  Round {rnd+1} Mean Dice: {np.mean(round_dice):.4f}")

    return round_results


def run_inference_and_reports(all_slices):
    """Step 3: Run inference + XAI + PDF report on sample patient."""
    from agents.data_agent import scan_dataset, data_agent
    from agents.preprocessing_agent import preprocessing_agent
    from agents.segmentation_agent import segmentation_agent
    from agents.evaluation_agent import evaluation_agent
    from agents.xai_agent import xai_agent
    from agents.report_agent import report_agent

    patient_dirs = scan_dataset(RAW_DIR, limit=1)
    pdir = patient_dirs[0]
    logger.info(f"\nRunning full inference on {os.path.basename(pdir)}")

    state = {
        "patient_id": "", "patient_dir": pdir, "modalities": {},
        "seg_mask_3d": None, "affine": None, "processed_slices": [],
        "tumor_slice_indices": [], "augmented_slices": [],
        "hyperparams": {"conf": 0.25, "iou": 0.5, "batch_size": 8, "lr0": 0.001, "momentum": 0.937, "weight_decay": 0.0005},
        "asio_history": [], "predictions": [], "pred_masks": [], "model_path": None,
        "metrics": {}, "per_class_metrics": {}, "tumor_volume_cc": 0.0,
        "xai_maps": {}, "xai_figure_paths": [], "report_text": "", "pdf_path": None,
        "report_metadata": {}, "fl_round": 0, "fl_client_id": 0,
        "fl_stage": "infer", "global_model_params": None,
        "error": None, "error_agent": None, "retry_count": 0,
    }

    state = data_agent(state)
    state = preprocessing_agent(state)
    state = segmentation_agent(state)
    state = evaluation_agent(state)
    state = xai_agent(state)
    state = report_agent(state)

    metrics = state.get("metrics", {})
    logger.info(f"Inference complete: Dice={metrics.get('dice', 0):.4f} | IoU={metrics.get('iou', 0):.4f}")
    if state.get("pdf_path"):
        logger.info(f"PDF Report saved: {state['pdf_path']}")

    return state


def generate_all_graphs(round_results, final_state):
    """Step 4: Generate publication-quality graphs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Figure 1: FL Convergence Curve ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FedASIO-YOLO26 — Phase 1 Sample Results (5 Patients, 2 Clients, 3 Rounds)",
                  fontsize=11, fontweight="bold")

    rounds = [r["round"] for r in round_results]
    mean_dice = [r["mean_dice"] for r in round_results]

    ax = axes[0]
    ax.plot(rounds, mean_dice, "o-", color="#4f46e5", linewidth=2.5, markersize=8, label="Mean Dice")
    for i, r in enumerate(round_results):
        for c, d in enumerate(r["client_dice"]):
            ax.scatter(r["round"], d, c="#94a3b8", s=50, alpha=0.7, zorder=5)
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Dice Score")
    ax.set_title("Federated Convergence")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Figure 2: Per-class Dice ─────────────────────────────────────────────
    ax = axes[1]
    metrics = final_state.get("metrics", {})
    per_class = final_state.get("per_class_metrics", {})
    class_names = list(per_class.keys()) or ["SNFH", "Tumor Core", "Enhancing Tumor"]
    class_dice = [per_class.get(c, {}).get("dice", 0.0) for c in class_names]
    colors = ["#4f46e5", "#7c3aed", "#a78bfa"]
    bars = ax.bar(class_names, class_dice, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Segmentation")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Figure 3: Metrics Summary ─────────────────────────────────────────────
    ax = axes[2]
    metric_names = ["Dice", "IoU", "Precision", "Recall", "F1"]
    metric_vals = [
        metrics.get("dice", 0), metrics.get("iou", 0),
        metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("f1", 0)
    ]
    colors2 = ["#4f46e5", "#7c3aed", "#a78bfa", "#c4b5fd", "#ddd6fe"]
    bars2 = ax.barh(metric_names, metric_vals, color=colors2, alpha=0.85)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Overall Metrics (Inference)")
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "phase1_results.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # ── Figure 2: XAI Visualization ───────────────────────────────────────────
    xai_paths = final_state.get("xai_figure_paths", [])
    if xai_paths:
        import cv2
        xai_img = cv2.imread(xai_paths[0])
        if xai_img is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(14, 4))
            ax2.imshow(cv2.cvtColor(xai_img, cv2.COLOR_BGR2RGB))
            ax2.axis("off")
            ax2.set_title("XAI Visualization: Original | Ground Truth | YOLO26 Prediction | GradCAM Confidence",
                           fontsize=10)
            fig2_path = os.path.join(FIGURES_DIR, "phase1_xai.png")
            plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {fig2_path}")

    # Save metrics JSON
    metrics_out = {
        "phase": "1_sample",
        "n_patients": N_SAMPLE_PATIENTS,
        "n_clients": N_FL_CLIENTS,
        "n_rounds": N_FL_ROUNDS,
        "fl_rounds": round_results,
        "final_metrics": metrics,
        "per_class_metrics": final_state.get("per_class_metrics", {}),
        "tumor_volume_cc": final_state.get("tumor_volume_cc", 0.0),
    }
    json_path = os.path.join(METRICS_DIR, "phase1_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Saved metrics: {json_path}")

    return fig1_path


def main():
    logger.info("="*60)
    logger.info("FedASIO-YOLO26 — PHASE 1: SAMPLE PIPELINE")
    logger.info("="*60)
    logger.info(f"Dataset: {RAW_DIR}")
    logger.info(f"Patients: {N_SAMPLE_PATIENTS} | Clients: {N_FL_CLIENTS} | Rounds: {N_FL_ROUNDS}")

    t0 = time.time()

    logger.info("\n📦 Step 1/4: Data preparation...")
    all_slices = prepare_sample_data()
    logger.info(f"  ✅ {len(all_slices)} total slices prepared")

    logger.info("\n🔗 Step 2/4: Federated training simulation...")
    round_results = run_sample_fl(all_slices)

    logger.info("\n🔬 Step 3/4: Inference + XAI + Report...")
    final_state = run_inference_and_reports(all_slices)

    logger.info("\n📊 Step 4/4: Generating graphs...")
    fig_path = generate_all_graphs(round_results, final_state)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ PHASE 1 COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"   Figures: {FIGURES_DIR}/phase1_results.png")
    logger.info(f"   Metrics: {METRICS_DIR}/phase1_metrics.json")
    if final_state.get("pdf_path"):
        logger.info(f"   PDF:     {final_state['pdf_path']}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
