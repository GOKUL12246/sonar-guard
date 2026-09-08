"""
SONAR-GUARD — Report Generator
================================
Generates structured JSON and CSV reports from pipeline outputs.

All values in the report come from actual computed pipeline results.
No values are fabricated. Missing values are explicitly represented as null.
"""

import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


def _safe_round(value, digits: int = 4):
    """Round float or return None/str as-is."""
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_anomaly_report(
    image_id:            str,
    source_image_path:   Optional[str]  = None,
    detection_dict:      Optional[dict] = None,
    seg_dict:            Optional[dict] = None,
    na_dict:             Optional[dict] = None,
    shadow_dict:         Optional[dict] = None,
    uncertainty_dict:    Optional[dict] = None,
    fusion_dict:         Optional[dict] = None,
    artificiality_dict:  Optional[dict] = None,
    risk_dict:           Optional[dict] = None,
    geo_dict:            Optional[dict] = None,
    multipass_dict:      Optional[dict] = None,
    verification_status: str            = "unverified",
    user_notes:          str            = "",
    timing_dict:         Optional[dict] = None,
) -> dict:
    """
    Build a single anomaly report dictionary.

    All fields come from actual pipeline outputs.
    Missing modules produce explicit null values — not fabricated data.

    Returns:
        dict — the complete anomaly report.
    """
    det = detection_dict or {}
    seg = seg_dict or {}
    na  = na_dict or {}
    shd = shadow_dict or {}
    unc = uncertainty_dict or {}
    fus = fusion_dict or {}
    art = artificiality_dict or {}
    rsk = risk_dict or {}
    geo = geo_dict or {}
    mp  = multipass_dict or {}
    tim = timing_dict or {}

    anomaly_id = str(uuid.uuid4())
    timestamp  = datetime.now(tz=timezone.utc).isoformat()

    report = {
        # ── Identity ─────────────────────────────────────────────────
        "anomaly_id":            anomaly_id,
        "timestamp_utc":         timestamp,
        "source_image":          source_image_path,
        "image_id":              image_id,

        # ── Detection ─────────────────────────────────────────────────
        "class":                 det.get("class_name"),
        "class_id":              det.get("class_id"),
        "confidence":            _safe_round(det.get("confidence")),
        "bbox":                  det.get("bbox"),
        "image_width":           det.get("image_width"),
        "image_height":          det.get("image_height"),

        # ── Segmentation ──────────────────────────────────────────────
        "segmentation_available": seg.get("mask_available", False),
        "segmentation_method":   seg.get("method"),
        "segmentation_area_px":  seg.get("mask_area_px"),
        "bbox_area_px":          seg.get("bbox_area_px"),
        "mask_to_box_ratio":     _safe_round(seg.get("mask_to_box_ratio")),

        # ── Natural/Artificial Evidence ───────────────────────────────
        "shape_evidence":        _safe_round(na.get("shape_evidence")),
        "texture_evidence":      _safe_round(na.get("texture_evidence")),
        "context_evidence":      _safe_round(na.get("context_evidence")),
        "aspect_ratio":          _safe_round(na.get("aspect_ratio")),
        "rectangularity":        _safe_round(na.get("rectangularity")),
        "compactness":           _safe_round(na.get("compactness")),

        # ── Acoustic Shadow ───────────────────────────────────────────
        "shadow_detected":       shd.get("shadow_detected"),
        "shadow_strength":       _safe_round(shd.get("shadow_strength")),
        "shadow_evidence":       _safe_round(shd.get("shadow_evidence")),
        "shadow_area_px":        shd.get("shadow_area_px"),

        # ── Uncertainty ───────────────────────────────────────────────
        "system_confidence":     _safe_round(unc.get("confidence")),
        "uncertainty":           _safe_round(unc.get("uncertainty")),
        "reliability":           _safe_round(unc.get("reliability")),
        "evidence_consistency":  _safe_round(unc.get("evidence_consistency")),
        "signals_available":     unc.get("num_signals"),
        "missing_signals":       unc.get("missing_signals"),

        # ── Evidence Fusion ───────────────────────────────────────────
        "fused_evidence_score":  _safe_round(fus.get("fused_score")),
        "fusion_components":     fus.get("components"),
        "fusion_weights":        fus.get("weights_used"),

        # ── Artificiality Score ───────────────────────────────────────
        "artificiality_score":   art.get("score"),
        "artificiality_label":   art.get("label"),

        # ── Marine Risk Score ─────────────────────────────────────────
        "marine_risk_score":     rsk.get("score"),
        "marine_risk_band":      rsk.get("band"),
        "risk_label":            "SONAR-GUARD prioritisation score — not a scientific probability",

        # ── Geolocation ───────────────────────────────────────────────
        "latitude":              geo.get("latitude"),
        "longitude":             geo.get("longitude"),
        "location_status":       geo.get("status"),
        "location_method":       geo.get("method"),
        "location_accuracy_note": geo.get("accuracy_note"),

        # ── Multi-pass ────────────────────────────────────────────────
        "multipass_available":   mp.get("available", False),
        "multipass_num_passes":  mp.get("num_passes"),
        "multipass_consistency": _safe_round(mp.get("overall_consistency")),
        "multipass_message":     mp.get("message"),

        # ── Human Verification ────────────────────────────────────────
        "verification_status":   verification_status,
        "user_notes":            user_notes,

        # ── Performance ───────────────────────────────────────────────
        "preprocessing_time_s":  _safe_round(tim.get("preprocessing_time_s")),
        "detection_time_s":      _safe_round(tim.get("detection_time_s")),
        "segmentation_time_s":   _safe_round(tim.get("segmentation_time_s")),
        "analysis_time_s":       _safe_round(tim.get("analysis_time_s")),
        "total_time_s":          _safe_round(tim.get("total_time_s")),
    }

    return report


class ReportGenerator:
    """
    Generates JSON and CSV reports from anomaly report dictionaries.
    """

    def __init__(self, output_dir: Path = Path("outputs/reports")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, report: dict, filename: Optional[str] = None) -> Path:
        """Save a single anomaly report as JSON."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_id = str(report.get("image_id") or report.get("anomaly_id") or "report")
            safe_id = re.sub(r'[\\/*?:"<>|\s]+', '_', raw_id)[:40]
            filename = f"report_{safe_id}_{ts}.json"
        else:
            filename = re.sub(r'[\\/*?:"<>|]+', '_', filename)
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        log.info("JSON report saved: %s", path)
        return path

    def save_csv(self, reports: list, filename: Optional[str] = None) -> Path:
        """Save a list of anomaly reports as CSV."""
        if not reports:
            log.warning("No reports to save as CSV.")
            return self.output_dir / "empty.csv"

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports_{ts}.csv"

        path = self.output_dir / filename

        # Flatten nested dicts for CSV
        flat_reports = [_flatten_dict(r) for r in reports]
        all_keys = list(dict.fromkeys(k for r in flat_reports for k in r))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_reports)

        log.info("CSV report saved: %s (%d records)", path, len(reports))
        return path


def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    """Flatten nested dict for CSV output."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items

