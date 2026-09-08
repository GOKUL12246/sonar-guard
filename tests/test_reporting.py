"""
SONAR-GUARD — Unit Tests: Reporting & Ingestion
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion.loader import load_image, load_mission_metadata, SonarImage
from src.active_learning.feedback import FeedbackStore


class TestImageLoader:
    def test_load_nonexistent_file(self):
        result = load_image(Path("nonexistent_file.png"))
        assert not result.is_valid
        assert "not found" in result.load_error.lower()

    def test_load_valid_image(self, tmp_path):
        img = np.zeros((480, 640), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)
        result = load_image(img_path)
        assert result.is_valid
        assert result.width == 640
        assert result.height == 480

    def test_load_bgr_image(self, tmp_path):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "colour.png"
        cv2.imwrite(str(img_path), img)
        result = load_image(img_path)
        assert result.is_valid
        assert result.channels == 3

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        result = load_image(bad_file)
        assert not result.is_valid
        assert "unsupported format" in result.load_error.lower()

    def test_load_mission_metadata_dict(self):
        meta = load_mission_metadata({
            "latitude": 12.345, "longitude": 78.901,
            "heading_deg": 90.0, "sonar_range_m": 50.0,
        })
        assert meta is not None
        assert meta.has_position
        assert abs(meta.latitude - 12.345) < 1e-6

    def test_load_mission_metadata_none(self):
        meta = load_mission_metadata(None)
        assert meta is None

    def test_load_mission_metadata_invalid_json(self, tmp_path):
        bad_file = tmp_path / "meta.json"
        bad_file.write_text("not valid json {{{")
        meta = load_mission_metadata(bad_file)
        assert meta is None

    def test_metadata_no_position(self):
        meta = load_mission_metadata({"survey_name": "test_survey"})
        assert meta is not None
        assert not meta.has_position
        assert "unavailable" in meta.location_status.lower()

    def test_metadata_extra_fields_preserved(self):
        meta = load_mission_metadata({
            "latitude": 10.0, "longitude": 20.0,
            "custom_field": "some_value",
        })
        assert "custom_field" in meta.extra


class TestFeedbackStore:
    def test_store_and_retrieve(self, tmp_path):
        store = FeedbackStore(feedback_file=tmp_path / "fb.jsonl")
        det_dict = {"class_name": "debris", "confidence": 0.8,
                    "bbox": [10, 10, 100, 100], "image_id": "test",
                    "image_width": 640, "image_height": 480, "class_id": 0}
        fid = store.store(
            image_id="test_img", image_path=None,
            detection_dict=det_dict,
            scores_dict={"artificiality": 75, "marine_risk": 60},
            user_decision="confirmed",
        )
        assert fid is not None
        records = store.load_all()
        assert len(records) == 1
        assert records[0]["user_decision"] == "confirmed"

    def test_invalid_decision_raises(self, tmp_path):
        store = FeedbackStore(feedback_file=tmp_path / "fb.jsonl")
        with pytest.raises(ValueError, match="Invalid user_decision"):
            store.store(
                image_id="x", image_path=None,
                detection_dict={}, scores_dict={},
                user_decision="maybe",
            )

    def test_summary_counts(self, tmp_path):
        store = FeedbackStore(feedback_file=tmp_path / "fb.jsonl",
                              retraining_threshold=5)
        det = {"class_name": "x", "confidence": 0.5, "bbox": [], "image_id": "x",
               "image_width": 640, "image_height": 480, "class_id": 0}
        store.store("a", None, det, {}, "confirmed")
        store.store("b", None, det, {}, "confirmed")
        store.store("c", None, det, {}, "rejected")
        summary = store.summary()
        assert summary["confirmed"] == 2
        assert summary["rejected"] == 1
        assert summary["total_feedback_records"] == 3
        assert not summary["ready_for_retraining"]

    def test_empty_feedback_summary(self, tmp_path):
        store = FeedbackStore(feedback_file=tmp_path / "empty_fb.jsonl")
        summary = store.summary()
        assert summary["total_feedback_records"] == 0
        assert not summary["ready_for_retraining"]

