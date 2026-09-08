"""
SONAR-GUARD — Unit Tests: Scoring, Fusion, Evidence
Tests evidence fusion, artificiality, and risk scoring logic.
"""

import pytest
from unittest.mock import MagicMock

from src.fusion.evidence_fusion import EvidenceFusionEngine
from src.scoring.artificiality import compute_artificiality_score, ArtificialityScore
from src.scoring.marine_risk import compute_marine_risk_score
from src.detection.yolo_detector import Detection, BoundingBox
from src.segmentation.segmenter import SegmentationResult
from src.sonar_analysis.natural_artificial import NaturalArtificialEvidence
from src.sonar_analysis.acoustic_shadow import AcousticShadowResult
from src.uncertainty.estimator import UncertaintyResult


def _make_detection(conf=0.75, cls_id=0, cls_name="marine_debris",
                    x1=50, y1=50, x2=150, y2=150):
    return Detection(
        detection_id=0, class_id=cls_id, class_name=cls_name,
        confidence=conf, bbox=BoundingBox(x1, y1, x2, y2),
        image_id="test", image_width=640, image_height=480,
    )


def _make_na(shape=0.8, texture=0.7, context=0.6):
    return NaturalArtificialEvidence(
        shape_evidence=shape, texture_evidence=texture, context_evidence=context,
        status="ok",
    )


def _make_shadow(strength=0.65, evidence=0.6, detected=True):
    return AcousticShadowResult(
        shadow_detected=detected, shadow_strength=strength,
        shadow_evidence=evidence, status="ok",
    )


def _make_unc(confidence=0.72, uncertainty=0.28, reliability=0.65, consistency=0.80):
    return UncertaintyResult(
        yolo_confidence=0.75, confidence=confidence, uncertainty=uncertainty,
        reliability=reliability, evidence_consistency=consistency,
        num_signals=5, missing_signals=[], status="ok",
    )


class TestEvidenceFusion:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            EvidenceFusionEngine(weights={"detection": 0.5, "shape": 0.3})

    def test_fused_score_in_range(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.8)
        na  = _make_na()
        shd = _make_shadow()
        unc = _make_unc()
        seg = SegmentationResult(detection_id=0, mask_to_box_ratio=0.6,
                                 mask_area_px=6000, bbox_area_px=10000,
                                 method="contours", status="OK")
        result = engine.fuse(det, seg, na, shd, unc)
        assert result.status == "ok"
        assert 0.0 <= result.fused_score <= 1.0

    def test_all_zeros_gives_low_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.0)
        result = engine.fuse(det)
        assert result.fused_score == 0.0

    def test_missing_signals_tracked(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.5)
        result = engine.fuse(det)  # no na, seg, shadow, unc
        assert "shape" in result.missing_components
        assert "shadow" in result.missing_components

    def test_contributions_sum_to_fused_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.7)
        na  = _make_na()
        shd = _make_shadow()
        unc = _make_unc()
        result = engine.fuse(det, None, na, shd, unc)
        total_contrib = sum(result.contributions.values())
        assert abs(total_contrib - result.fused_score) < 1e-4


class TestArtificialityScore:
    def test_score_in_range(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.8)
        na  = _make_na()
        shd = _make_shadow()
        unc = _make_unc()
        fusion = engine.fuse(det, None, na, shd, unc)
        art = compute_artificiality_score(fusion)
        assert 0 <= art.score <= 100

    def test_high_evidence_gives_high_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.95)
        na  = _make_na(shape=0.95, texture=0.90, context=0.88)
        shd = _make_shadow(strength=0.90, evidence=0.88)
        unc = _make_unc(confidence=0.92)
        fusion = engine.fuse(det, None, na, shd, unc)
        art = compute_artificiality_score(fusion)
        assert art.score >= 60

    def test_low_evidence_gives_low_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.1)
        fusion = engine.fuse(det)
        art = compute_artificiality_score(fusion)
        assert art.score <= 20

    def test_label_assignment(self):
        assert ArtificialityScore.label_for_score(90) == "Artificial"
        assert ArtificialityScore.label_for_score(70) == "Likely Artificial"
        assert ArtificialityScore.label_for_score(50) == "Uncertain"
        assert ArtificialityScore.label_for_score(30) == "Likely Natural"
        assert ArtificialityScore.label_for_score(10) == "Natural"

    def test_reasons_not_empty(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.7)
        na  = _make_na()
        shd = _make_shadow()
        unc = _make_unc()
        fusion = engine.fuse(det, None, na, shd, unc)
        art = compute_artificiality_score(fusion, na, shd, unc)
        assert len(art.reasons) > 0


class TestMarineRiskScore:
    def test_score_in_range(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.7)
        na  = _make_na()
        shd = _make_shadow()
        unc = _make_unc()
        fusion = engine.fuse(det, None, na, shd, unc)
        art = compute_artificiality_score(fusion)
        risk = compute_marine_risk_score(art, None, shd, unc, image_area_px=307200)
        assert 0 <= risk.score <= 100

    def test_confirmed_increases_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.7)
        fusion = engine.fuse(det)
        art = compute_artificiality_score(fusion)
        r_unverified = compute_marine_risk_score(art, verification_status="unverified")
        r_confirmed  = compute_marine_risk_score(art, verification_status="confirmed")
        assert r_confirmed.score >= r_unverified.score

    def test_rejected_decreases_score(self):
        engine = EvidenceFusionEngine()
        det = _make_detection(conf=0.7)
        fusion = engine.fuse(det)
        art = compute_artificiality_score(fusion)
        r_unverified = compute_marine_risk_score(art, verification_status="unverified")
        r_rejected   = compute_marine_risk_score(art, verification_status="rejected")
        assert r_rejected.score <= r_unverified.score

    def test_band_labels(self):
        from src.scoring.marine_risk import risk_label
        assert risk_label(10) == "Low"
        assert risk_label(35) == "Medium"
        assert risk_label(60) == "High"
        assert risk_label(85) == "Critical"

