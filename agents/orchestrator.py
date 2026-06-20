"""
agents/orchestrator.py
FedASIO-YOLO26: Orchestrator — top-level entry point for agent pipeline.
Supports single-patient inference, batch evaluation, and FL-integrated modes.
"""
import os
import sys
import logging
import json
from typing import Optional, List

sys.path.insert(0, "/Users/shrikant/Downloads/FedASIO-YOLO26")

from agents.graph import get_training_graph, get_inference_graph
from agents.data_agent import scan_dataset

logger = logging.getLogger(__name__)

# Empty initial state template
def _empty_state(patient_dir: str, fl_round: int = 0,
                 fl_client_id: int = 0, fl_stage: str = "infer",
                 hyperparams: dict = None, model_path: str = None) -> dict:
    return {
        "patient_id": "",
        "patient_dir": patient_dir,
        "modalities": {},
        "seg_mask_3d": None,
        "affine": None,
        "processed_slices": [],
        "tumor_slice_indices": [],
        "augmented_slices": [],
        "hyperparams": hyperparams or {
            "lr0": 0.001, "momentum": 0.937, "weight_decay": 0.0005,
            "conf": 0.25, "iou": 0.5, "batch_size": 8, "local_epochs": 3,
        },
        "asio_history": [],
        "predictions": [],
        "pred_masks": [],
        "model_path": model_path,
        "metrics": {},
        "per_class_metrics": {},
        "tumor_volume_cc": 0.0,
        "xai_maps": {},
        "xai_figure_paths": [],
        "report_text": "",
        "pdf_path": None,
        "report_metadata": {},
        "fl_round": fl_round,
        "fl_client_id": fl_client_id,
        "fl_stage": fl_stage,
        "global_model_params": None,
        "error": None,
        "error_agent": None,
        "retry_count": 0,
    }


def run_inference(patient_dir: str, model_path: Optional[str] = None,
                  hyperparams: Optional[dict] = None) -> dict:
    """
    Run full inference pipeline on a single patient directory.
    Returns final state with metrics, XAI maps, and PDF report.
    """
    graph = get_inference_graph()
    state = _empty_state(patient_dir, fl_stage="infer",
                         hyperparams=hyperparams, model_path=model_path)
    result = graph.invoke(state)
    if result.get("error"):
        logger.error(f"Pipeline error: {result['error']} in {result.get('error_agent')}")
    return result


def run_batch_inference(patient_dirs: List[str], model_path: Optional[str] = None,
                        output_dir: str = "/Users/shrikant/Downloads/FedASIO-YOLO26/reports") -> List[dict]:
    """Run inference on multiple patients and aggregate metrics."""
    import numpy as np
    results = []
    for i, pdir in enumerate(patient_dirs):
        logger.info(f"[Orchestrator] Patient {i+1}/{len(patient_dirs)}: {os.path.basename(pdir)}")
        result = run_inference(pdir, model_path=model_path)
        results.append({
            "patient_id": result.get("patient_id"),
            "metrics": result.get("metrics", {}),
            "tumor_volume_cc": result.get("tumor_volume_cc", 0.0),
            "pdf_path": result.get("pdf_path"),
            "error": result.get("error"),
        })

    # Aggregate
    valid = [r for r in results if not r["error"] and r["metrics"]]
    if valid:
        agg = {}
        for key in ["dice", "iou", "precision", "recall", "f1", "hd95"]:
            vals = [r["metrics"].get(key, 0) for r in valid]
            agg[key] = float(np.mean(vals))
        logger.info(f"[Orchestrator] Batch complete — {len(valid)}/{len(results)} success | "
                    f"Mean Dice: {agg.get('dice', 0):.4f}")

        # Save summary
        summary_path = os.path.join(output_dir, "batch_inference_summary.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"results": results, "aggregate": agg}, f, indent=2)
        logger.info(f"[Orchestrator] Summary saved: {summary_path}")

    return results


def run_training_round(patient_dir: str, fl_round: int, fl_client_id: int,
                       hyperparams: dict, model_path: Optional[str] = None) -> dict:
    """Run one FL training round for a single patient through the training graph."""
    graph = get_training_graph()
    state = _empty_state(
        patient_dir, fl_round=fl_round, fl_client_id=fl_client_id,
        fl_stage="train", hyperparams=hyperparams, model_path=model_path
    )
    return graph.invoke(state)


if __name__ == "__main__":
    """CLI usage: python agents/orchestrator.py <patient_dir>"""
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="FedASIO-YOLO26 Inference Orchestrator")
    parser.add_argument("patient_dir", help="Path to patient directory (NIfTI files)")
    parser.add_argument("--model", default=None, help="Path to YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.50, help="IoU threshold")
    args = parser.parse_args()

    hp = {"conf": args.conf, "iou": args.iou, "lr0": 0.001,
          "momentum": 0.937, "weight_decay": 0.0005, "batch_size": 8}

    result = run_inference(args.patient_dir, model_path=args.model, hyperparams=hp)
    print(f"\n{'='*50}")
    print(f"Patient:      {result.get('patient_id')}")
    print(f"Dice Score:   {result.get('metrics', {}).get('dice', 0):.4f}")
    print(f"IoU:          {result.get('metrics', {}).get('iou', 0):.4f}")
    print(f"Tumor Vol:    {result.get('tumor_volume_cc', 0):.2f} cc")
    print(f"PDF Report:   {result.get('pdf_path', 'Not generated')}")
    print(f"{'='*50}")
