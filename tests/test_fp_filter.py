"""
Tests for SONAR-GUARD False Positive Filter (src/filtering/fp_filter.py)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.yolo_detector import Detection, BoundingBox
from src.filtering.fp_filter import FalsePositiveFilter, FilterDecision, FilterResult


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_detection(det_id=0, conf=0.80, x1=50, y1=50, x2=150, y2=150,
                   cls_name="Crab-Pot", img_w=640, img_h=480):
    return Detection(
        detection_id=det_id,
        class_id=0,
        class_name=cls_name,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        image_id="test",
        image_width=img_w,
        image_height=img_h,
    )


def make_filter(**kwargs) -> FalsePositiveFilter:
    defaults = dict(
        min_confidence=0.25,
        min_bbox_area_px=100,
        max_aspect_ratio=10.0,
        min_shape_evidence=0.05,
        require_shadow=False,
    )
    defaults.update(kwargs)
    return FalsePositiveFilter(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────

class TestFalsePositiveFilter:

    def test_good_detection_accepted(self):
        """A well-formed detection passes all rules."""
        fp = make_filter()
        det = make_detection(conf=0.80)
        result = fp.filter([det], image_width=640, image_height=480)
        assert result.num_accepted == 1
        assert result.num_rejected == 0
        assert result.decisions[0].accepted is True

    def test_low_confidence_rejected(self):
        """Detection below confidence threshold is rejected."""
        fp = make_filter(min_confidence=0.50)
        det = make_detection(conf=0.30)
        result = fp.filter([det], 640, 480)
        assert result.num_accepted == 0
        assert result.num_rejected == 1
        assert result.decisions[0].rule_name == "low_confidence"
        assert "0.30" in result.decisions[0].reason

    def test_tiny_bbox_rejected(self):
        """Detection with bbox area below minimum is rejected."""
        fp = make_filter(min_bbox_area_px=1000)
        # 10×10 = 100px² < 1000
        det = make_detection(conf=0.90, x1=0, y1=0, x2=10, y2=10)
        result = fp.filter([det], 640, 480)
        assert result.num_rejected == 1
        assert result.decisions[0].rule_name == "tiny_bbox"

    def test_extreme_aspect_ratio_rejected(self):
        """Very thin detection (noise stripe) is rejected."""
        fp = make_filter(max_aspect_ratio=5.0)
        # 200×10 = aspect 20:1
        det = make_detection(conf=0.90, x1=0, y1=0, x2=200, y2=10)
        result = fp.filter([det], 640, 480)
        assert result.num_rejected == 1
        assert result.decisions[0].rule_name == "thin_bbox"

    def test_out_of_bounds_bbox_rejected(self):
        """Detection extending outside image is rejected."""
        fp = make_filter()
        det = make_detection(conf=0.90, x1=0, y1=0, x2=700, y2=200)  # x2 > 640
        result = fp.filter([det], 640, 480)
        assert result.num_rejected == 1
        assert result.decisions[0].rule_name == "poor_bbox_quality"

    def test_filter_result_contains_both_lists(self):
        """FilterResult separates accepted and rejected correctly."""
        fp = make_filter(min_confidence=0.50)
        det_good = make_detection(det_id=0, conf=0.90)
        det_bad  = make_detection(det_id=1, conf=0.10)
        result = fp.filter([det_good, det_bad], 640, 480)
        accepted_ids = [d.detection_id for d in result.accepted]
        rejected_ids = [d.detection_id for d in result.rejected]
        assert 0 in accepted_ids
        assert 1 in rejected_ids

    def test_decision_for_lookup(self):
        """Can look up FilterDecision by detection_id."""
        fp = make_filter()
        det = make_detection(det_id=5, conf=0.90)
        result = fp.filter([det], 640, 480)
        d = result.decision_for(5)
        assert d is not None
        assert d.detection_id == 5

    def test_empty_input(self):
        """Empty detection list returns empty result without crashing."""
        fp = make_filter()
        result = fp.filter([], 640, 480)
        assert result.num_accepted == 0
        assert result.num_rejected == 0
        assert result.decisions == []

    def test_filter_result_to_dict(self):
        """FilterResult serialises to dict without error."""
        fp = make_filter()
        det = make_detection()
        result = fp.filter([det], 640, 480)
        d = result.to_dict()
        assert "num_accepted" in d
        assert "num_rejected" in d
        assert "decisions" in d
        assert "thresholds" in d

    def test_filter_decision_to_dict(self):
        """FilterDecision serialises to dict."""
        fd = FilterDecision(detection_id=0, accepted=True)
        d = fd.to_dict()
        assert d["detection_id"] == 0
        assert d["accepted"] is True

    def test_thresholds_in_result(self):
        """FilterResult records the thresholds used."""
        fp = make_filter(min_confidence=0.42)
        result = fp.filter([], 640, 480)
        assert result.thresholds["min_confidence"] == 0.42

    def test_reason_not_empty_on_rejection(self):
        """Every rejection includes a non-empty reason string."""
        fp = make_filter(min_confidence=0.99)
        det = make_detection(conf=0.10)
        result = fp.filter([det], 640, 480)
        assert len(result.decisions[0].reason) > 10

    def test_multiple_detections_all_accepted(self):
        """Multiple valid detections all pass."""
        fp = make_filter()
        dets = [make_detection(det_id=i, conf=0.8) for i in range(5)]
        result = fp.filter(dets, 640, 480)
        assert result.num_accepted == 5
        assert result.num_rejected == 0
