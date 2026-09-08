"""
SONAR-GUARD — Uncertainty Estimator
=====================================
Computes system-level uncertainty and reliability for each detection.

This module does NOT simply rename YOLO confidence as uncertainty.
It combines multiple independent evidence signals to produce:
  - confidence   (aggregated evidence alignment)
  - uncertainty  (disagreement / low-evidence penalty)
  - reliability  (how much the system trusts its own output)
  - evidence_consistency (spread of individual evidence signals)

All values are computed from actual pipeline outputs.
None are fabricated or hardcoded.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.detection.yolo_detector import Detection
from src.segmentation.segmenter import SegmentationResult
from src.sonar_analysis.natural_artificial import NaturalArtificialEvidence
from src.sonar_analysis.acoustic_shadow import AcousticShadowResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyResult:
    """
    System-level uncertainty and reliability assessment.

    Attributes:
        yolo_confidence:    Raw YOLO confidence (preserved, not relabelled).
        evidence_mean:      Mean of all available evidence signals (0–1).
        evidence_std:       Standard deviation of evidence signals.
        evidence_consistency: 1 - normalised_std. Higher = more consistent signals.
        uncertainty:        System-level uncertainty (0–1; higher = less certain).
        confidence:         System-level confidence (1 - uncertainty).
        reliability:        How much to trust this detection overall (0–1).
        num_signals:        Number of evidence signals available.
        missing_signals:    List of signals that were unavailable.
        status:             Human-readable summary.
    """
    yolo_confidence:       float = 0.0
    evidence_mean:         float = 0.0
    evidence_std:          float = 0.0
    evidence_consistency:  float = 0.0
    uncertainty:           float = 1.0
    confidence:            float = 0.0
    reliability:           float = 0.0
    num_signals:           int   = 0
    missing_signals:       list  = None
    status:                str   = "not_run"

    def __post_init__(self):
        if self.missing_signals is None:
            self.missing_signals = []

    def to_dict(self) -> dict:
        return {
            "yolo_confidence":      round(self.yolo_confidence, 4),
            "evidence_mean":        round(self.evidence_mean, 4),
            "evidence_std":         round(self.evidence_std, 4),
            "evidence_consistency": round(self.evidence_consistency, 4),
            "uncertainty":          round(self.uncertainty, 4),
            "confidence":           round(self.confidence, 4),
            "reliability":          round(self.reliability, 4),
            "num_signals":          self.num_signals,
            "missing_signals":      self.missing_signals,
            "status":               self.status,
        }

    def explanation_lines(self) -> list:
        lines = []
        if self.reliability >= 0.70:
            lines.append(("✓", f"High reliability ({self.reliability:.2f}) — strong, consistent evidence"))
        elif self.reliability >= 0.45:
            lines.append(("⚠", f"Moderate reliability ({self.reliability:.2f}) — some uncertainty present"))
        else:
            lines.append(("⚠", f"Low reliability ({self.reliability:.2f}) — limited evidence, verify carefully"))
        if self.missing_signals:
            lines.append(("⚠", f"Missing signals: {', '.join(self.missing_signals)}"))
        return lines


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class UncertaintyEstimator:
    """
    Combines multiple evidence signals into a reliability assessment.

    The approach:
      1. Collect available evidence scores (YOLO conf, shape, texture,
         shadow, context, segmentation quality).
      2. Compute mean and standard deviation of available signals.
      3. High std → inconsistent signals → higher uncertainty.
      4. Few available signals → lower reliability baseline.
      5. Reliability is penalised for missing signals and inconsistency.

    This is a deterministic, transparent calculation — NOT a Bayesian model.
    If calibrated data becomes available, a Bayesian approach can replace it.
    """

    # Penalty per missing signal (reduces reliability)
    MISSING_SIGNAL_PENALTY = 0.05

    def estimate(
        self,
        detection:  Detection,
        seg_result: Optional[SegmentationResult]         = None,
        na_evidence: Optional[NaturalArtificialEvidence] = None,
        shadow_result: Optional[AcousticShadowResult]    = None,
    ) -> UncertaintyResult:
        """
        Estimate uncertainty from all available pipeline signals.

        Args:
            detection:    YOLO Detection (provides detection confidence).
            seg_result:   Segmentation result (provides mask quality signal).
            na_evidence:  Shape/texture/context evidence.
            shadow_result: Acoustic shadow evidence.

        Returns:
            UncertaintyResult — always returned.
        """
        signals = {}
        missing = []

        # ── Signal 1: YOLO detection confidence ───────────────────────
        yolo_conf = float(detection.confidence)
        signals["yolo_confidence"] = yolo_conf

        # ── Signal 2: Segmentation quality ────────────────────────────
        if seg_result and seg_result.available:
            # mask_to_box_ratio: 0 = empty mask, 1 = fully filled.
            # A reasonable segmentation is ~0.3–0.8 (not empty, not trivial).
            ratio = seg_result.mask_to_box_ratio
            seg_signal = float(1.0 - abs(ratio - 0.55) / 0.55)  # peaks at ratio=0.55
            seg_signal = max(0.0, min(1.0, seg_signal))
            signals["segmentation_quality"] = seg_signal
        else:
            missing.append("segmentation")

        # ── Signals 3–5: Shape, Texture, Context ──────────────────────
        if na_evidence and na_evidence.status == "ok":
            signals["shape_evidence"]   = na_evidence.shape_evidence
            signals["texture_evidence"] = na_evidence.texture_evidence
            signals["context_evidence"] = na_evidence.context_evidence
        else:
            missing.extend(["shape_evidence", "texture_evidence", "context_evidence"])

        # ── Signal 6: Acoustic shadow ──────────────────────────────────
        if shadow_result and shadow_result.status == "ok":
            signals["shadow_evidence"] = shadow_result.shadow_evidence
        else:
            missing.append("shadow_evidence")

        # ── Compute aggregate statistics ───────────────────────────────
        values = list(signals.values())
        n = len(values)

        if n == 0:
            return UncertaintyResult(
                yolo_confidence=yolo_conf,
                uncertainty=1.0, confidence=0.0, reliability=0.0,
                num_signals=0, missing_signals=missing,
                status="no_signals_available",
            )

        ev_mean = float(sum(values) / n)
        ev_std  = float(math.sqrt(sum((v - ev_mean) ** 2 for v in values) / max(1, n)))

        # Evidence consistency: 1 when all signals agree, 0 when maximally spread
        # Max possible std for [0,1] signals ≈ 0.5
        ev_consistency = float(max(0.0, 1.0 - (ev_std / 0.5)))

        # Uncertainty: inversely related to consistency and evidence mean
        # High evidence + high consistency → low uncertainty
        raw_uncertainty = float(1.0 - 0.5 * ev_mean - 0.5 * ev_consistency)
        uncertainty = float(max(0.0, min(1.0, raw_uncertainty)))
        confidence  = float(1.0 - uncertainty)

        # Reliability: starts from evidence_consistency × confidence,
        # then penalised for missing signals
        base_reliability = float(ev_consistency * confidence)
        missing_penalty  = len(missing) * self.MISSING_SIGNAL_PENALTY
        reliability = float(max(0.0, base_reliability - missing_penalty))

        log.debug(
            "Uncertainty det-%d | signals=%d missing=%d | "
            "conf=%.2f unc=%.2f rel=%.2f cons=%.2f",
            detection.detection_id, n, len(missing),
            confidence, uncertainty, reliability, ev_consistency,
        )

        return UncertaintyResult(
            yolo_confidence=round(yolo_conf, 4),
            evidence_mean=round(ev_mean, 4),
            evidence_std=round(ev_std, 4),
            evidence_consistency=round(ev_consistency, 4),
            uncertainty=round(uncertainty, 4),
            confidence=round(confidence, 4),
            reliability=round(reliability, 4),
            num_signals=n,
            missing_signals=missing,
            status="ok",
        )

