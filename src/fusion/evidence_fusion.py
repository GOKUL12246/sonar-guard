"""
SONAR-GUARD — Evidence Fusion Engine
======================================
Combines independent evidence signals into a single fused evidence score.

Approach:
  Deterministic weighted evidence model — transparent, explainable, validatable.
  Each weight represents the configured relative importance of each signal.
  Weights are defined in config.py and must sum to 1.0.

  This is NOT Bayesian inference. It is a weighted linear combination.
  Bayesian fusion requires calibrated likelihoods from real labelled data,
  which are documented as future work in README.md.

Output: EvidenceFusionResult (dataclass)
  - fused_score       (0–1)
  - per-component scores and contributions
  - transparency dict for the dashboard panel
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.detection.yolo_detector import Detection
from src.segmentation.segmenter import SegmentationResult
from src.sonar_analysis.natural_artificial import NaturalArtificialEvidence
from src.sonar_analysis.acoustic_shadow import AcousticShadowResult
from src.uncertainty.estimator import UncertaintyResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# Default weights — must sum to 1.0 — override via config.py
DEFAULT_WEIGHTS: Dict[str, float] = {
    "detection":   0.25,
    "shape":       0.20,
    "texture":     0.15,
    "shadow":      0.20,
    "context":     0.10,
    "uncertainty": 0.10,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvidenceFusionResult:
    """
    Result of evidence fusion for one detection.

    Attributes:
        fused_score:        Final fused evidence score (0–1).
        components:         Raw score of each evidence component.
        contributions:      Weighted contribution of each component.
        weights_used:       Weights applied to each component.
        missing_components: Components that were unavailable (received 0).
        status:             Processing status.
    """
    fused_score:         float = 0.0
    components:          Dict[str, float] = field(default_factory=dict)
    contributions:       Dict[str, float] = field(default_factory=dict)
    weights_used:        Dict[str, float] = field(default_factory=dict)
    missing_components:  list = field(default_factory=list)
    status:              str  = "not_run"

    def to_dict(self) -> dict:
        return {
            "fused_score":        round(self.fused_score, 4),
            "components":         {k: round(v, 4) for k, v in self.components.items()},
            "contributions":      {k: round(v, 4) for k, v in self.contributions.items()},
            "weights_used":       self.weights_used,
            "missing_components": self.missing_components,
            "status":             self.status,
        }

    def explanation_lines(self) -> list:
        """For the 'Evidence Fusion' panel in the dashboard."""
        lines = []
        for comp, score in self.components.items():
            weight = self.weights_used.get(comp, 0.0)
            contrib = self.contributions.get(comp, 0.0)
            missing = comp in self.missing_components
            symbol = "✗" if missing else ("✓" if score >= 0.6 else "⚠")
            suffix = " (unavailable — scored 0)" if missing else ""
            lines.append((
                symbol,
                f"{comp:20s}: score={score:.2f}  weight={weight:.2f}  "
                f"contribution={contrib:.3f}{suffix}"
            ))
        lines.append(("→", f"Fused evidence score: {self.fused_score:.4f}"))
        return lines


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class EvidenceFusionEngine:
    """
    Transparent weighted evidence fusion.

    Each signal is mapped to a [0,1] score.
    The fused score = Σ (weight_i × score_i).

    Missing signals receive a score of 0 for their weighted slot,
    which conservatively reduces the fused score.
    This is the honest choice: absence of evidence ≠ evidence of absence,
    but we cannot award points for signals we could not measure.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Per-component fusion weights summing to 1.0.
                     Defaults to DEFAULT_WEIGHTS.
        """
        w = weights or DEFAULT_WEIGHTS
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total:.4f}. "
                "Fix config.py → fusion_weights."
            )
        self.weights = w
        log.info("EvidenceFusionEngine initialised | weights: %s", w)

    def fuse(
        self,
        detection:     Detection,
        seg_result:    Optional[SegmentationResult]         = None,
        na_evidence:   Optional[NaturalArtificialEvidence]  = None,
        shadow_result: Optional[AcousticShadowResult]       = None,
        unc_result:    Optional[UncertaintyResult]           = None,
    ) -> EvidenceFusionResult:
        """
        Fuse all available evidence signals into a single score.

        Args:
            detection:     YOLO Detection (provides detection confidence).
            seg_result:    Segmentation result.
            na_evidence:   Shape/texture/context evidence.
            shadow_result: Acoustic shadow evidence.
            unc_result:    Uncertainty result (used as inverted penalty).

        Returns:
            EvidenceFusionResult with full per-component breakdown.
        """
        components: Dict[str, float] = {}
        missing: list = []

        # ── detection signal ──────────────────────────────────────────
        components["detection"] = float(detection.confidence)

        # ── shape signal ──────────────────────────────────────────────
        if na_evidence and na_evidence.status == "ok":
            components["shape"] = na_evidence.shape_evidence
        else:
            components["shape"] = 0.0
            missing.append("shape")

        # ── texture signal ─────────────────────────────────────────────
        if na_evidence and na_evidence.status == "ok":
            components["texture"] = na_evidence.texture_evidence
        else:
            components["texture"] = 0.0
            missing.append("texture")

        # ── shadow signal ──────────────────────────────────────────────
        if shadow_result and shadow_result.status == "ok":
            components["shadow"] = shadow_result.shadow_evidence
        else:
            components["shadow"] = 0.0
            missing.append("shadow")

        # ── context signal ─────────────────────────────────────────────
        if na_evidence and na_evidence.status == "ok":
            components["context"] = na_evidence.context_evidence
        else:
            components["context"] = 0.0
            missing.append("context")

        # ── uncertainty signal (inverted — low uncertainty = high score) ─
        if unc_result and unc_result.status == "ok":
            # Use confidence (= 1 - uncertainty) as the signal
            components["uncertainty"] = unc_result.confidence
        else:
            components["uncertainty"] = 0.0
            missing.append("uncertainty")

        # ── Weighted sum ───────────────────────────────────────────────
        fused = 0.0
        contributions: Dict[str, float] = {}
        for comp, score in components.items():
            w = self.weights.get(comp, 0.0)
            contrib = w * score
            contributions[comp] = round(contrib, 6)
            fused += contrib

        fused = float(max(0.0, min(1.0, fused)))

        log.debug(
            "Evidence fusion det-%d | fused=%.3f | missing=%s",
            detection.detection_id, fused, missing,
        )

        return EvidenceFusionResult(
            fused_score=round(fused, 4),
            components={k: round(v, 4) for k, v in components.items()},
            contributions=contributions,
            weights_used=dict(self.weights),
            missing_components=missing,
            status="ok",
        )

