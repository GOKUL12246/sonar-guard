"""
SONAR-GUARD — ONNX Export Helper
===================================
Exports a trained YOLOv8 model to ONNX format for edge deployment.

Usage (from project root):
    python src/training/export_onnx.py --model models/yolo/ghost_pot_yolov8n_best.pt

Options:
    --model       Path to trained .pt file  (required)
    --imgsz       Export image size (default: 640)
    --half        Export FP16 weights (requires GPU)
    --int8        Export INT8 quantised (requires calibration data)
    --opset       ONNX opset version (default: 17)
    --output-dir  Directory to save exported model (default: models/exported)
    --device      Device: 'cpu' or 'cuda' (auto-detected if omitted)

Notes:
    - FP16 (--half) requires a CUDA-capable GPU.
    - INT8 (--int8) requires a calibration dataset and GPU.
    - CPU-only export (default) produces a full-precision ONNX model.
    - Do NOT claim FPS or latency unless actually measured on the target device.
    - Exported model is saved as: <model_stem>_opset<opset>.onnx

Requirements:
    pip install onnxruntime  # for CPU inference
    # For CUDA inference:   pip install onnxruntime-gpu
    # For TensorRT:         requires TensorRT SDK + onnxruntime-gpu
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging, get_logger

setup_logging()
log = get_logger("export_onnx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export SONAR-GUARD YOLOv8 model to ONNX for edge deployment."
    )
    p.add_argument("--model",      required=True,  help="Path to .pt weights file")
    p.add_argument("--imgsz",      type=int, default=640, help="Export image size (default: 640)")
    p.add_argument("--opset",      type=int, default=17,  help="ONNX opset version (default: 17)")
    p.add_argument("--half",       action="store_true",   help="FP16 export (GPU required)")
    p.add_argument("--int8",       action="store_true",   help="INT8 export (GPU + calibration required)")
    p.add_argument("--output-dir", default=None,          help="Output directory (default: models/exported)")
    p.add_argument("--device",     default=None,          help="Device: cpu or cuda (auto-detected)")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    # ── Validate input ─────────────────────────────────────────────────────
    if not model_path.exists():
        log.error("Model file not found: %s", model_path)
        sys.exit(1)
    if model_path.suffix != ".pt":
        log.error("Expected a .pt file, got: %s", model_path.suffix)
        sys.exit(1)

    # ── Detect device ──────────────────────────────────────────────────────
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Export device: %s", device)

    if args.half and device == "cpu":
        log.error("FP16 (--half) requires a CUDA GPU. Current device is CPU.")
        sys.exit(1)

    if args.int8 and device == "cpu":
        log.error("INT8 (--int8) quantisation requires a CUDA GPU. Current device is CPU.")
        sys.exit(1)

    # ── Output directory ───────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "models" / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    log.info("Loading model: %s", model_path)
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception as e:
        log.error("Failed to load model: %s", e)
        sys.exit(1)

    class_names = list(model.names.values()) if hasattr(model, "names") else []
    log.info("Model classes: %s", class_names)

    # ── Export ─────────────────────────────────────────────────────────────
    stem = model_path.stem
    onnx_name = f"{stem}_imgsz{args.imgsz}_opset{args.opset}"
    if args.half:
        onnx_name += "_fp16"
    if args.int8:
        onnx_name += "_int8"

    log.info("Exporting to ONNX (imgsz=%d, opset=%d, half=%s, int8=%s)…",
             args.imgsz, args.opset, args.half, args.int8)

    t_start = time.perf_counter()
    try:
        export_path = model.export(
            format   = "onnx",
            imgsz    = args.imgsz,
            opset    = args.opset,
            half     = args.half,
            int8     = args.int8,
            device   = device,
            simplify = True,
        )
        elapsed = time.perf_counter() - t_start
    except Exception as e:
        log.error("ONNX export failed: %s", e)
        sys.exit(1)

    if not export_path:
        log.error("Export returned no path — check Ultralytics output.")
        sys.exit(1)

    # Move to output_dir if needed
    src_onnx = Path(export_path)
    dst_onnx = output_dir / (onnx_name + ".onnx")
    if src_onnx.resolve() != dst_onnx.resolve():
        src_onnx.rename(dst_onnx)
        log.info("Moved ONNX model to: %s", dst_onnx)
    else:
        log.info("ONNX model saved at: %s", dst_onnx)

    # ── Verify with onnxruntime ─────────────────────────────────────────────
    log.info("Verifying ONNX model with onnxruntime…")
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape  # [batch, C, H, W]
        log.info("ONNX input: %s  shape: %s", input_name, input_shape)

        # Smoke test with random input
        dummy = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        outs = sess.run(None, {input_name: dummy})
        log.info("ONNX smoke test passed. Output shapes: %s", [o.shape for o in outs])
        onnx_verified = True
        onnx_verify_error = None

    except Exception as e:
        log.warning("ONNX verification warning: %s", e)
        onnx_verified = False
        onnx_verify_error = str(e)

    # ── Save export manifest ────────────────────────────────────────────────
    manifest = {
        "exported_at":        datetime.now(tz=timezone.utc).isoformat(),
        "source_model":       str(model_path),
        "onnx_model":         str(dst_onnx),
        "class_names":        class_names,
        "export_settings": {
            "format":  "onnx",
            "imgsz":   args.imgsz,
            "opset":   args.opset,
            "half":    args.half,
            "int8":    args.int8,
            "device":  device,
        },
        "export_time_s":      round(elapsed, 2),
        "onnx_verified":      onnx_verified,
        "onnx_verify_error":  onnx_verify_error,
        "warnings": [
            "Do not claim FPS or latency without measuring on the actual target device.",
            "FP16/INT8 optimisations require GPU — not benchmarked on CPU.",
            (
                "This model was trained on the Ghost Pot Side-Scan Sonar dataset "
                "(class: Crab-Pot). Performance on other debris types is unvalidated."
            ),
        ],
        "next_steps": [
            "Use onnxruntime for CPU edge inference.",
            "For GPU edge: use onnxruntime-gpu or TensorRT.",
            "Benchmark on the actual target edge device before claiming performance.",
            (
                "For ONNX inference in the pipeline: "
                "load with onnxruntime.InferenceSession and replace the PyTorch model."
            ),
        ],
    }

    manifest_path = output_dir / (onnx_name + "_export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Export manifest saved: %s", manifest_path)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SONAR-GUARD ONNX Export Complete")
    print("=" * 60)
    print(f"  Source model : {model_path}")
    print(f"  ONNX model   : {dst_onnx}")
    print(f"  Image size   : {args.imgsz}px")
    print(f"  Opset        : {args.opset}")
    print(f"  Precision    : {'FP16' if args.half else 'INT8' if args.int8 else 'FP32'}")
    print(f"  Export time  : {elapsed:.1f}s")
    print(f"  ONNX verified: {'✓' if onnx_verified else '✗ (see manifest)'}")
    print(f"  Classes      : {class_names}")
    print(f"  Manifest     : {manifest_path}")
    print()
    print("IMPORTANT: Do not claim FPS/latency without measuring on target device.")
    print("=" * 60)


if __name__ == "__main__":
    main()
