"""
SONAR-GUARD — Model Evaluation
================================
Runs YOLO validation/test evaluation and produces an evaluation_report.json.

Only reports metrics that have actually been computed from real predictions.
Never fabricates mAP, precision, recall or confusion matrix values.

Usage:
    python -m src.training.evaluate --model models/yolo/best.pt --dataset data/sonar.yaml
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger, setup_logging
from src.training.dataset_paths import resolve_dataset_yaml

log = get_logger(__name__)


def evaluate(
    model_path:   Path,
    dataset_yaml: Path,
    img_size:     int  = 640,
    batch_size:   int  = 8,
    conf:         float = 0.25,
    iou:          float = 0.45,
    device:       str  = "cpu",
    split:        str  = "test",   # "val" or "test"
    output_dir:   Optional[Path] = None,
) -> dict:
    """
    Evaluate a trained YOLO model on val or test split.

    Returns:
        dict with real measured metrics from Ultralytics validation.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed.")

    model_path   = Path(model_path)
    dataset_yaml = resolve_dataset_yaml(Path(dataset_yaml))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    if output_dir is None:
        output_dir = Path("outputs") / "evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Evaluating model: %s", model_path)
    log.info("Dataset YAML    : %s", dataset_yaml)
    log.info("Split           : %s", split)
    log.info("Device          : %s", device)

    model = YOLO(str(model_path))

    t_start = time.time()
    metrics = model.val(
        data    = str(dataset_yaml),
        split   = split,
        imgsz   = img_size,
        batch   = batch_size,
        conf    = conf,
        iou     = iou,
        device  = device,
        project = str(output_dir),
        name    = f"eval_{split}",
        plots   = True,
        verbose = True,
    )
    elapsed = time.time() - t_start

    # Extract real metrics from Ultralytics results object
    report = {
        "model_path":    str(model_path),
        "dataset_yaml":  str(dataset_yaml),
        "split":         split,
        "device":        device,
        "eval_time_s":   round(elapsed, 2),
        "metrics": {},
    }

    try:
        rd = metrics.results_dict
        report["metrics"] = {
            k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in rd.items()
        }
    except Exception as e:
        log.warning("Could not extract metrics from results object: %s", e)
        report["metrics"] = {"error": str(e)}

    # Save report
    report_path = output_dir / f"eval_{split}" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("Evaluation report saved: %s", report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONAR-GUARD — Model Evaluation")
    parser.add_argument("--model",    required=True, help="Path to trained .pt file")
    parser.add_argument("--dataset",  required=True, help="Path to dataset YAML")
    parser.add_argument("--split",    default="test", choices=["val", "test"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--device",   default=None)
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    setup_logging()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    result = evaluate(
        model_path   = Path(args.model),
        dataset_yaml = Path(args.dataset),
        img_size     = args.img_size,
        batch_size   = args.batch,
        conf         = args.conf,
        iou          = args.iou,
        device       = device,
        split        = args.split,
        output_dir   = Path(args.output) if args.output else None,
    )

    print("\n--- Evaluation Results ---")
    for k, v in result.get("metrics", {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

