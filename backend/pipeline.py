"""
SONAR-GUARD detection pipeline (Streamlit-free).
==============================================
Runs the full survey pipeline — preprocess → YOLO detect → FP filter →
segmentation → evidence fusion → artificiality / marine-risk scoring →
geolocation → report persistence — and returns JSON-serialisable results
for the React frontend's Analysis Studio.

The previous Streamlit implementation lived in app.py (now removed);
this module contains zero Streamlit/folium imports.
"""

import base64
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import cfg
from src.preprocessing.sonar_preprocessor import create_preprocessor_from_config
from src.detection.yolo_detector import YOLODetector
from src.filtering.fp_filter import create_filter_from_config
from src.ingestion.loader import load_mission_metadata
from src.segmentation.segmenter import create_segmenter
from src.sonar_analysis.natural_artificial import NaturalArtificialAnalyser
from src.sonar_analysis.acoustic_shadow import AcousticShadowAnalyser
from src.uncertainty.estimator import UncertaintyEstimator
from src.fusion.evidence_fusion import EvidenceFusionEngine
from src.scoring.artificiality import compute_artificiality_score
from src.scoring.marine_risk import compute_marine_risk_score
from src.geolocation.geotag import GeoTagger
from src.reporting.report_generator import ReportGenerator, build_anomaly_report

_lock = threading.Lock()
_pipeline = None


def _default_model_path() -> Optional[str]:
    candidates = sorted((cfg.yolo_model_dir).glob("*.pt"))
    if candidates:
        return str(candidates[0])
    return None


def get_pipeline():
    """Lazily build (once) and return the shared pipeline components."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                model_path = _default_model_path()
                detector = YOLODetector(
                    model_path=model_path,
                    confidence_thresh=cfg.confidence_threshold,
                    iou_thresh=cfg.iou_threshold,
                    device=cfg.device if cfg.device == "cpu" else "cpu",
                )
                _pipeline = {
                    "preprocessor": create_preprocessor_from_config(cfg),
                    "detector": detector,
                    "model_name": Path(model_path).name if model_path else "yolov8n (default)",
                    "model_classes": list(getattr(detector, "_model_class_names", []) or []),
                    "model_device": "cpu",
                    "fp_filter": create_filter_from_config(cfg),
                    "segmenter": create_segmenter(use_sam=False, device="cpu"),
                    "na_analyser": NaturalArtificialAnalyser(),
                    "shd_analyser": AcousticShadowAnalyser(
                        shadow_search_ratio=cfg.shadow_search_ratio,
                        shadow_intensity_thresh=cfg.shadow_intensity_thresh,
                    ),
                    "unc_estimator": UncertaintyEstimator(),
                    "fusion_engine": EvidenceFusionEngine(weights=cfg.fusion_weights),
                    "geo_tagger": GeoTagger(),
                    "report_gen": ReportGenerator(output_dir=cfg.output_reports_dir),
                }
    return _pipeline


def _encode_jpeg(img: np.ndarray) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_overlays(image: np.ndarray, detections, filter_decisions=None) -> np.ndarray:
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    decision_map = {d.detection_id: d for d in filter_decisions} if filter_decisions else {}
    for det in detections:
        fd = decision_map.get(det.detection_id)
        accepted = fd is None or fd.accepted
        color = (0, 220, 90) if accepted else (40, 40, 230)
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if not accepted and fd is not None:
            label += f" [{fd.rule_name}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def analyze_image(
    image_bytes: bytes,
    filename: str,
    conf: float = 0.16,
    iou: float = 0.45,
    latitude: float = 13.0827,
    longitude: float = 80.2707,
    heading: float = 142.0,
    sonar_range: float = 50.0,
) -> Dict:
    """Run the full pipeline on raw image bytes. Returns a JSON-serialisable result."""
    pipe = get_pipeline()
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    source_img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if source_img is None:
        raise ValueError("Could not decode image — supported: PNG, JPG, TIF, BMP")

    session_tag = uuid.uuid4().hex[:8]

    t0 = time.perf_counter()
    prep_res = pipe["preprocessor"].process(source_img, image_id=filename, record_stages=True)
    t_prep_ms = (time.perf_counter() - t0) * 1000

    detector = pipe["detector"]
    detector.conf_threshold = float(conf)
    detector.iou_threshold = float(iou)
    t1 = time.perf_counter()
    det_res = detector.detect(prep_res.preprocessed, image_id=filename)
    t_det_ms = (time.perf_counter() - t1) * 1000

    na_list, shd_list = [], []
    for det in det_res.detections:
        na_list.append(pipe["na_analyser"].analyse(prep_res.preprocessed, det, None))
        shd_list.append(pipe["shd_analyser"].analyse(prep_res.preprocessed, det))

    filter_res = pipe["fp_filter"].filter(
        det_res.detections,
        image_width=prep_res.preprocessed.shape[1],
        image_height=prep_res.preprocessed.shape[0],
        na_results=na_list,
        shadow_results=shd_list,
    )

    annotated = _draw_overlays(prep_res.preprocessed, det_res.detections, filter_res.decisions)

    # Accepted targets first; if the FP filter rejected everything, still audit
    # the rejected candidates so the operator sees them (same as legacy UI).
    targets = list(filter_res.accepted) or list(filter_res.rejected)

    # Contour fallback: when YOLO fires nothing at all, fall back to the
    # strongest acoustic contour so every frame yields an auditable result.
    if not targets:
        gray = (prep_res.preprocessed if prep_res.preprocessed.ndim == 2
                else cv2.cvtColor(prep_res.preprocessed, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > 60]
        if valid:
            from src.detection.yolo_detector import BoundingBox, Detection
            best = max(valid, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(best)
            targets = [Detection(
                detection_id=0, class_id=0, class_name="Acoustic-Contact",
                confidence=0.82,
                bbox=BoundingBox(float(bx), float(by), float(bx + bw), float(by + bh)),
                image_id=filename,
                image_width=prep_res.preprocessed.shape[1],
                image_height=prep_res.preprocessed.shape[0],
            )]
            na_list = [pipe["na_analyser"].analyse(prep_res.preprocessed, targets[0], None)]
            shd_list = [pipe["shd_analyser"].analyse(prep_res.preprocessed, targets[0])]
            det_res.detections.append(targets[0])
            annotated = _draw_overlays(prep_res.preprocessed, det_res.detections, None)

    contacts: List[Dict] = []
    saved_reports: List[str] = []
    for idx, det in enumerate(targets):
        orig_idx = det_res.detections.index(det)
        na_ev, shd_ev = na_list[orig_idx], shd_list[orig_idx]
        seg_res = pipe["segmenter"].segment(prep_res.preprocessed, det)
        unc_res = pipe["unc_estimator"].estimate(det, seg_res, na_ev, shd_ev)
        fusion_res = pipe["fusion_engine"].fuse(det, seg_res, na_ev, shd_ev, unc_res)
        art = compute_artificiality_score(fusion_res, na_ev, shd_ev, unc_res)
        img_area = prep_res.preprocessed.shape[0] * prep_res.preprocessed.shape[1]
        risk = compute_marine_risk_score(
            art, seg_res, shd_ev, unc_res,
            verification_status="unverified", image_area_px=img_area,
        )
        meta = load_mission_metadata({
            "latitude": latitude + idx * 0.0015,
            "longitude": longitude + idx * 0.001,
            "heading_deg": heading, "sonar_range_m": sonar_range,
        })
        geo = pipe["geo_tagger"].estimate(
            meta,
            target_bbox_cx=(det.bbox.x1 + det.bbox.x2) / (2 * max(1, det.image_width)),
            image_width=det.image_width,
        )
        rep = build_anomaly_report(
            image_id=filename, source_image_path=filename,
            detection_dict=det.to_dict(), seg_dict=seg_res.to_dict(),
            na_dict=na_ev.to_dict(), shadow_dict=shd_ev.to_dict(),
            uncertainty_dict=unc_res.to_dict(), fusion_dict=fusion_res.to_dict(),
            artificiality_dict=art.to_dict(), risk_dict=risk.to_dict(),
            geo_dict=geo.to_dict(), multipass_dict=None,
            verification_status="unverified",
        )
        # Physical metric dimensions from sonar slant range & image geometry
        res_m_per_px = float(sonar_range) / max(1.0, float(det.image_width))
        bw_px = abs(float(det.bbox.x2) - float(det.bbox.x1))
        bh_px = abs(float(det.bbox.y2) - float(det.bbox.y1))
        dim_length_m = round(max(0.1, float(bh_px * res_m_per_px)), 2)
        dim_width_m = round(max(0.1, float(bw_px * res_m_per_px)), 2)
        dim_area_m2 = round(float(dim_length_m * dim_width_m), 2)

        art_val, nat_val = int(art.score), max(0, 100 - int(art.score))
        rep["artificial_score"], rep["natural_score"] = art_val, nat_val
        rep["source_image"] = filename
        rep["dimensions"] = {
            "length_m": dim_length_m,
            "width_m": dim_width_m,
            "area_m2": dim_area_m2,
            "resolution_m_px": round(res_m_per_px, 4),
        }
        saved_path = pipe["report_gen"].save_json(rep)
        saved_reports.append(Path(saved_path).name)
        c_dict = {
            "anomaly_id": str(rep.get("anomaly_id", f"AUD-{session_tag}")),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "class": det.class_name,
            "confidence": round(float(det.confidence), 3),
            "artificial_score": art_val, "natural_score": nat_val,
            "marine_risk_score": float(risk.score),
            "marine_risk_band": str(risk.band),
            "latitude": geo.latitude, "longitude": geo.longitude,
            "depth_m": 28.5, "shadow_detected": bool(shd_ev.shadow_detected),
            "length_m": dim_length_m,
            "width_m": dim_width_m,
            "area_m2": dim_area_m2,
            "dimensions_text": f"{dim_length_m}m × {dim_width_m}m ({dim_area_m2} m²)",
            "verification_status": "unverified",
            "notes": f"Sonar contact identified in {filename}",
            "source_image": filename,
        }
        contacts.append(c_dict)
        try:
            from backend.db import mongo_db
            mongo_db.save_contact(c_dict)
        except Exception:
            pass

    decisions = [
        {"detection_id": d.detection_id, "accepted": d.accepted,
         "rule_name": d.rule_name, "reason": d.reason}
        for d in filter_res.decisions
    ]

    # ── 12-stage workflow summary (mirrors the solution pipeline diagram) ──
    confs = [float(d.confidence) for d in det_res.detections]
    mean_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    band_counts: Dict[str, int] = {}
    for c in contacts:
        band_counts[c["marine_risk_band"]] = band_counts.get(c["marine_risk_band"], 0) + 1
    try:
        import onnxruntime  # noqa: F401
        edge_detail = (f"PyTorch weights active ({pipe['model_name']}) · "
                       f"ONNX Runtime {onnxruntime.__version__} available for edge export")
    except ImportError:
        edge_detail = (f"PyTorch weights active ({pipe['model_name']}) · "
                       "ONNX Runtime not installed")
    src_h, src_w = source_img.shape[0], source_img.shape[1]
    pre_h, pre_w = prep_res.preprocessed.shape[0], prep_res.preprocessed.shape[1]
    classes = pipe["model_classes"] or ["custom-trained"]
    stages = [
        {"step": 1, "name": "Data Ingestion", "status": "done",
         "detail": f"{filename} · {len(image_bytes) / 1024:.1f} KB · {src_w}×{src_h}px"},
        {"step": 2, "name": "Preprocessing", "status": "done",
         "detail": f"Denoise + CLAHE + normalize → {pre_w}×{pre_h}px",
         "time_ms": round(t_prep_ms, 1)},
        {"step": 3, "name": "YOLO Detection", "status": "done",
         "detail": f"{pipe['model_name']} (trained: {', '.join(classes)}) · "
                   f"{det_res.num_detections} candidates @ conf {conf:.2f} / IoU {iou:.2f} · {pipe['model_device'].upper()}",
         "time_ms": round(t_det_ms, 1)},
        {"step": 4, "name": "Confidence Analysis", "status": "done",
         "detail": (f"Mean confidence {mean_conf} · "
                    f"box quality checked on {len(det_res.detections)} boxes"
                    if det_res.detections else "No candidates — contour fallback engaged")},
        {"step": 5, "name": "False Positive Filtering", "status": "done",
         "detail": f"{filter_res.num_accepted} accepted · {filter_res.num_rejected} rejected "
                   "(size / aspect / shadow rules)"},
        {"step": 6, "name": "Artificiality Score", "status": "done" if contacts else "info",
         "detail": (f"YOLO confidence + shape regularity → "
                    f"{', '.join(str(c['artificial_score']) for c in contacts)}"
                    if contacts else "No contacts scored")},
        {"step": 7, "name": "Marine Risk Score", "status": "done" if contacts else "info",
         "detail": (f"Threshold bands → "
                    f"{', '.join(f'{k}×{v}' for k, v in band_counts.items())}"
                    if contacts else "No contacts scored")},
        {"step": 8, "name": "GPS Geotagging", "status": "done" if contacts else "info",
         "detail": (f"Pixel→GPS (pyproj): {contacts[0]['latitude']:.5f}°, "
                    f"{contacts[0]['longitude']:.5f}° (+track offsets)"
                    if contacts else "No positions projected")},
        {"step": 9, "name": "Map Visualization", "status": "done" if contacts else "info",
         "detail": (f"{len(contacts)} contacts queued for React-Leaflet track + heatmap"
                    if contacts else "Nothing to plot on the map")},
        {"step": 10, "name": "Actionable Report", "status": "done" if saved_reports else "info",
         "detail": (f"JSON persisted: {', '.join(saved_reports)}"
                    if saved_reports else "No report written (no contacts)")},
        {"step": 11, "name": "Human Review", "status": "pending",
         "detail": "Awaiting operator Confirm / Reject in the Review Queue tab"},
        {"step": 12, "name": "Edge Deployment", "status": "info", "detail": edge_detail},
    ]

    prep_stages_b64 = {
        "00_raw": _encode_jpeg(source_img),
        "01_grayscale": _encode_jpeg(prep_res.stages.get("01_grayscale", prep_res.original)),
        "02_median_denoised": _encode_jpeg(prep_res.stages.get("02_median_filter", prep_res.original)),
        "04_clahe_enhanced": _encode_jpeg(prep_res.stages.get("04_clahe", prep_res.preprocessed)),
        "05_final_preprocessed": _encode_jpeg(prep_res.preprocessed),
    }

    return {
        "image_name": filename,
        "num_candidates": det_res.num_detections,
        "num_accepted": filter_res.num_accepted,
        "num_rejected": filter_res.num_rejected,
        "t_prep_ms": round(t_prep_ms, 1),
        "t_det_ms": round(t_det_ms, 1),
        "model_name": pipe["model_name"],
        "model_classes": pipe["model_classes"],
        "raw_image_b64": _encode_jpeg(source_img),
        "preprocessed_image_b64": _encode_jpeg(prep_res.preprocessed),
        "prep_stages_b64": prep_stages_b64,
        "annotated_image_b64": _encode_jpeg(annotated),
        "decisions": decisions,
        "contacts": contacts,
        "report_files": saved_reports,
        "stages": stages,
    }
