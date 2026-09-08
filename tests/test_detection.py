"""
SONAR-GUARD — Unit Tests: Detection & Reporting
"""

import json
import numpy as np
import pytest
from pathlib import Path

from src.detection.yolo_detector import Detection, BoundingBox, YOLODetector, DetectionResult
from src.reporting.report_generator import build_anomaly_report, ReportGenerator
from src.geolocation.geotag import GeoTagger, GeoLocation
from src.ingestion.loader import MissionMetadata
from src.verification.multi_pass import MultiPassVerifier, PassObservation


class TestBoundingBox:
    def test_area(self):
        bb = BoundingBox(0, 0, 100, 50)
        assert bb.area == 5000.0

    def test_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.center == (50.0, 50.0)

    def test_yolo_norm(self):
        bb = BoundingBox(100, 100, 200, 200)
        cx, cy, w, h = bb.to_yolo_norm(400, 400)
        assert abs(cx - 0.375) < 1e-6
        assert abs(cy - 0.375) < 1e-6
        assert abs(w - 0.25) < 1e-6
        assert abs(h - 0.25) < 1e-6


class TestYOLODetector:
    def test_detector_init_no_model(self):
        """Detector initialises without crash even when model is absent."""
        d = YOLODetector(model_path="nonexistent_model.pt")
        # Should fall back to yolov8n pretrained — no crash

    def test_detect_returns_result(self):
        d = YOLODetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        result = d.detect(img, image_id="test_blank")
        assert isinstance(result, DetectionResult)
        # No crash even on blank image

    def test_detect_timing_recorded(self):
        d = YOLODetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        result = d.detect(img, image_id="timing_test")
        assert result.inference_time_s >= 0


class TestGeoTagger:
    def test_no_metadata_returns_unavailable(self):
        tagger = GeoTagger()
        loc = tagger.estimate(None)
        assert not loc.available
        assert "unavailable" in loc.status.lower()

    def test_metadata_no_position_returns_unavailable(self):
        tagger = GeoTagger()
        meta = MissionMetadata(survey_name="test")  # no lat/lon
        loc = tagger.estimate(meta)
        assert not loc.available

    def test_metadata_with_position_no_geometry(self):
        tagger = GeoTagger()
        meta = MissionMetadata(latitude=12.345, longitude=78.901)
        loc = tagger.estimate(meta)
        assert loc.available
        assert abs(loc.latitude - 12.345) < 1e-5

    def test_full_geometry_gives_offset_position(self):
        tagger = GeoTagger()
        meta = MissionMetadata(
            latitude=12.000, longitude=78.000,
            heading_deg=90.0, sonar_range_m=50.0
        )
        loc = tagger.estimate(meta, target_bbox_cx=0.8, image_width=640)
        assert loc.available
        assert loc.method == "sonar_geometry_projection"
        # Position should be different from vehicle position
        assert abs(loc.latitude - 12.000) < 0.01  # reasonable offset


class TestReporting:
    def test_build_report_no_crash_missing_inputs(self):
        """Report builds even when all optional inputs are None."""
        report = build_anomaly_report(
            image_id="test", source_image_path=None,
            detection_dict=None, seg_dict=None, na_dict=None,
            shadow_dict=None, uncertainty_dict=None, fusion_dict=None,
            artificiality_dict=None, risk_dict=None, geo_dict=None,
            multipass_dict=None,
        )
        assert "anomaly_id" in report
        assert report["latitude"] is None
        assert report["longitude"] is None

    def test_report_serialisable(self):
        report = build_anomaly_report(
            image_id="test", source_image_path="img.png",
            detection_dict={"class_name": "debris", "confidence": 0.85,
                            "bbox": [10, 10, 100, 100],
                            "image_width": 640, "image_height": 480,
                            "class_id": 0},
            seg_dict=None, na_dict=None, shadow_dict=None,
            uncertainty_dict=None, fusion_dict=None,
            artificiality_dict=None, risk_dict=None,
            geo_dict={"latitude": None, "longitude": None,
                      "status": "unavailable", "method": "none", "accuracy_note": ""},
            multipass_dict=None,
        )
        # Should be JSON serialisable
        json_str = json.dumps(report, default=str)
        parsed = json.loads(json_str)
        assert parsed["image_id"] == "test"

    def test_report_generator_saves_json(self, tmp_path):
        gen = ReportGenerator(output_dir=tmp_path)
        report = build_anomaly_report(
            image_id="test", source_image_path=None,
            detection_dict=None, seg_dict=None, na_dict=None,
            shadow_dict=None, uncertainty_dict=None, fusion_dict=None,
            artificiality_dict=None, risk_dict=None, geo_dict=None,
            multipass_dict=None,
        )
        path = gen.save_json(report, filename="test_report.json")
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["image_id"] == "test"


class TestMultiPass:
    def test_single_observation_unavailable(self):
        verifier = MultiPassVerifier()
        obs = [PassObservation(pass_id="p0", confidence=0.7)]
        result = verifier.verify(obs)
        assert not result.available
        assert "single observation" in result.message.lower()

    def test_no_observations(self):
        verifier = MultiPassVerifier()
        result = verifier.verify([])
        assert not result.available

    def test_multiple_observations_available(self):
        verifier = MultiPassVerifier()
        obs = [
            PassObservation(pass_id="p0", confidence=0.72, fused_evidence=0.65),
            PassObservation(pass_id="p1", confidence=0.74, fused_evidence=0.67),
        ]
        result = verifier.verify(obs)
        assert result.available
        assert result.num_passes == 2
        assert result.mean_confidence is not None

