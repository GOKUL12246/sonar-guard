"""
SONAR-GUARD — YOLOv8 Training Script
======================================
Handles dataset YAML generation, model initialisation, training, and
checkpoint management for YOLOv8 fine-tuning on sonar imagery.

Usage (CLI):
    python -m src.training.train_yolo --dataset data/train --epochs 50

Design:
  - Auto-detects CUDA / CPU — never assumes GPU.
  - All parameters come from config.py or CLI args — nothing hardcoded.
  - Reports only actual measured metrics after real training.
  - Saves: best.pt, last.pt, training logs, metrics, confusion matrix.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import yaml

from config import cfg, Config
from src.training.dataset_paths import resolve_dataset_yaml
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset YAML generator
# ---------------------------------------------------------------------------

def generate_dataset_yaml(
    train_dir:   Path,
    val_dir:     Path,
    test_dir:    Optional[Path],
    class_names: list,
    output_path: Path,
) -> Path:
    """
    Generate a YOLOv8-compatible dataset YAML file.

    Args:
        train_dir:   Path to train/images directory.
        val_dir:     Path to val/images directory.
        test_dir:    Path to test/images directory (optional).
        class_names: Ordered list of class name strings.
        output_path: Where to save the .yaml file.

    Returns:
        Path to the written YAML file.
    """
    train_dir = Path(train_dir).resolve()
    val_dir   = Path(val_dir).resolve()

    yaml_data = {
        "path":  str(train_dir.parent.parent),   # dataset root
        "train": str(train_dir),
        "val":   str(val_dir),
        "nc":    len(class_names),
        "names": class_names,
    }
    if test_dir is not None:
        yaml_data["test"] = str(Path(test_dir).resolve())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    log.info("Dataset YAML written: %s", output_path)
    log.info("  Classes (%d): %s", len(class_names), class_names)
    return output_path


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def train(
    dataset_yaml:   Path,
    model_size:     str   = "n",
    pretrained:     str   = "yolov8n.pt",
    epochs:         int   = 50,
    img_size:       int   = 640,
    batch_size:     int   = 8,
    lr:             float = 0.01,
    device:         str   = "cpu",
    patience:       int   = 20,
    workers:        int   = 0,
    seed:           int   = 42,
    output_dir:     Optional[Path] = None,
    resume:         bool  = False,
) -> dict:
    """
    Train a YOLOv8 model on the provided dataset.

    Returns:
        dict with keys: model_path, metrics_path, training_time_s, device_used.
        All metrics come from actual Ultralytics training — never fabricated.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    dataset_yaml = Path(dataset_yaml)
    dataset_yaml = resolve_dataset_yaml(dataset_yaml)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path("outputs") / "yolo_runs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("SONAR-GUARD YOLOv8 Training")
    log.info("  Model size   : yolov8%s", model_size)
    log.info("  Pretrained   : %s", pretrained)
    log.info("  Dataset YAML : %s", dataset_yaml)
    log.info("  Epochs       : %d", epochs)
    log.info("  Image size   : %d", img_size)
    log.info("  Batch size   : %d", batch_size)
    log.info("  Device       : %s", device)
    log.info("  Seed         : %d", seed)
    log.info("=" * 60)

    # Load model
    model = YOLO(pretrained)
    log.info("Loaded pretrained weights: %s", pretrained)

    t_start = time.time()

    # Run training
    results = model.train(
        data      = str(dataset_yaml),
        epochs    = epochs,
        imgsz     = img_size,
        batch     = batch_size,
        lr0       = lr,
        device    = device,
        patience  = patience,
        workers   = workers,
        seed      = seed,
        project   = str(output_dir),
        name      = "sonarguard_yolo",
        exist_ok  = True,
        resume    = resume,
        verbose   = True,
        plots     = True,   # Saves confusion matrix, P-R curve, etc.
        save      = True,
        val       = True,
    )

    training_time = time.time() - t_start
    log.info("Training complete in %.1f s (%.1f min)", training_time, training_time / 60)

    # Locate saved model
    run_dir = output_dir / "sonarguard_yolo"
    if not (run_dir / "weights").exists():
        ultralytics_run_dir = Path("runs") / "detect" / run_dir
        if (ultralytics_run_dir / "weights").exists():
            run_dir = ultralytics_run_dir
    best_model  = run_dir / "weights" / "best.pt"
    last_model  = run_dir / "weights" / "last.pt"

    # Extract and save metrics summary
    metrics_dict = {}
    try:
        # results.results_dict contains the final epoch metrics
        metrics_dict = {
            k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in results.results_dict.items()
        }
    except Exception as e:
        log.warning("Could not extract metrics dict: %s", e)

    metrics_dict["training_time_s"] = round(training_time, 2)
    metrics_dict["device_used"]     = str(device)
    metrics_dict["epochs_trained"]  = int(epochs)
    metrics_dict["model_size"]      = model_size

    metrics_path = run_dir / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)

    summary = {
        "best_model_path": str(best_model) if best_model.exists() else None,
        "last_model_path": str(last_model) if last_model.exists() else None,
        "metrics_path":    str(metrics_path),
        "training_time_s": training_time,
        "device_used":     str(device),
    }

    log.info("Best model : %s", summary["best_model_path"])
    log.info("Last model : %s", summary["last_model_path"])
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SONAR-GUARD YOLOv8 Training Script"
    )
    parser.add_argument("--dataset",    required=True,
                        help="Path to dataset YAML or dataset root directory")
    parser.add_argument("--model-size", default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (default: n = nano)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--img-size",   type=int, default=640)
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--device",     default=None,
                        help="cuda / cpu / mps (auto-detected if omitted)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--log-level",  default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(log_level=args.log_level, log_dir=cfg.logs_dir)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.dataset)

    # If a directory is given, look for an existing YAML inside it
    if dataset_path.is_dir():
        candidates = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))
        if candidates:
            yaml_path = candidates[0]
            log.info("Using dataset YAML: %s", yaml_path)
        else:
            log.error(
                "No .yaml file found in %s. "
                "Run src.training.dataset_validator first to generate it.",
                dataset_path,
            )
            raise SystemExit(1)
    else:
        yaml_path = dataset_path

    result = train(
        dataset_yaml = yaml_path,
        model_size   = args.model_size,
        pretrained   = f"yolov8{args.model_size}.pt",
        epochs       = args.epochs,
        img_size     = args.img_size,
        batch_size   = args.batch,
        lr           = args.lr,
        device       = device,
        seed         = args.seed,
        output_dir   = Path(args.output_dir) if args.output_dir else None,
        resume       = args.resume,
    )

    print("\n--- Training Complete ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

