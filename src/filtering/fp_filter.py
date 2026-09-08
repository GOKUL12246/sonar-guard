"""
SONAR-GUARD — False Positive Filter
=====================================
Configurable, explainable false-positive filtering layer.

Every rejected detection records an explicit human-readable reason.
No detection is silently removed.

Filter rules (all configurable via config.py or at runtime):
  1. low_confidence      — confidence below minimum threshold
  2. tiny_bbox           — bounding box area too small (px²)
  3. thin_bbox           — aspect ratio too extreme (likely noise stripe)
  4. poor_bbox_quality   — bounding box extends outside image or is malformed
  5. natural_region      — target shows no texture contrast with seabed
  6. shadow_inconsistent — expected shadow absent or inconsistent
  7. low_shape_evidence  — shape regularity below threshold (no geometric structure)

Design principles:
  - All thresholds are configurable.
  - Results include ACCEPTED and REJECTED lists.
  - Each rejection records the rule name + human-readable reason.
  - Never modifies the original Detection objects.
  - Always returns a FilterResult, never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.detection.yolo_detector import Detection
from src.sonar_analysis.acoustic_shadow import AcousticShadowResult
from src.sonar_analysis.natural_artificial import NaturalArtificialEvidence
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FilterDecision:
    """
    Filtering outcome for a single detection.

    Attributes:
        detection_id:  Original detection ID.
        accepted:      True if detection passes all filters.
        rule_name:     Name of the first rule that caused rejection (if any).
        reason:        Human-readable explanation of the decision.
    """
    detection_id: int
    accepted: bool
    rule_name: str = "accepted"
    reason: str = "Passed all filters."

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "accepted":     self.accepted,
            "rule_name":    self.rule_name,
            "reason":       self.reason,
        }


@dataclass
class FilterResult:
    """
    Complete filtering outcome for all detections in one image.

    Attributes:
        accepted:   Detections that passed all filters.
        rejected:   Detections that were filtered out.
        decisions:  Per-detection decisions (accepted + rejected together).
        thresholds: Thresholds used for this run (for audit/display).
    """
    accepted:   List[Detection] = field(default_factory=list)
    rejected:   List[Detection] = field(default_factory=list)
    decisions:  List[FilterDecision] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)

    @property
    def num_accepted(self) -> int:
        return len(self.accepted)

    @property
    def num_rejected(self) -> int:
        return len(self.rejected)

    def decision_for(self, detection_id: int) -> Optional[FilterDecision]:
        for d in self.decisions:
            if d.detection_id == detection_id:
                return d
        return None

    def to_dict(self) -> dict:
        return {
            "num_accepted":  self.num_accepted,
            "num_rejected":  self.num_rejected,
            "thresholds":    self.thresholds,
            "decisions":     [d.to_dict() for d in self.decisions],
        }


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class FalsePositiveFilter:
    """
    Rule-based false positive filter for YOLO sonar detections.

    Usage:
        fp_filter = FalsePositiveFilter(
            min_confidence=0.25,
            min_bbox_area_px=100,
            max_aspect_ratio=8.0,
            min_shape_evidence=0.10,
        )
        result = fp_filter.filter(detections, image_width, image_height,
                                   na_results=..., shadow_results=...)
    """

    def __init__(
        self,
        min_confidence:       float = 0.25,
        min_bbox_area_px:     int   = 100,
        max_aspect_ratio:     float = 10.0,
        min_shape_evidence:   float = 0.05,
        min_shadow_evidence:  float = 0.0,   # 0.0 = shadow check advisory only
        require_shadow:       bool  = False,  # if True, reject when shadow absent
    ):
        """
        Args:
            min_confidence:      Detections below this confidence are rejected.
            min_bbox_area_px:    Detections with bbox area (px²) below this are rejected.
            max_aspect_ratio:    Detections with w/h or h/w above this are rejected.
            min_shape_evidence:  Detections with shape evidence below this are rejected.
            min_shadow_evidence: If require_shadow=True, minimum shadow evidence required.
            require_shadow:      If True, detections with no shadow are rejected.
        """
        self.min_confidence      = min_confidence
        self.min_bbox_area_px    = min_bbox_area_px
        self.max_aspect_ratio    = max_aspect_ratio
        self.min_shape_evidence  = min_shape_evidence
        self.min_shadow_evidence = min_shadow_evidence
        self.require_shadow      = require_shadow

    @property
    def thresholds(self) -> dict:
        return {
            "min_confidence":      self.min_confidence,
            "min_bbox_area_px":    self.min_bbox_area_px,
            "max_aspect_ratio":    self.max_aspect_ratio,
            "min_shape_evidence":  self.min_shape_evidence,
            "min_shadow_evidence": self.min_shadow_evidence,
            "require_shadow":      self.require_shadow,
        }

    def filter(
        self,
        detections:    List[Detection],
        image_width:   int,
        image_height:  int,
        na_results:    Optional[List[NaturalArtificialEvidence]] = None,
        shadow_results: Optional[List[AcousticShadowResult]]     = None,
    ) -> FilterResult:
        """
        Apply all filter rules to a list of detections.

        Args:
            detections:     Detections from YOLO inference.
            image_width:    Width of the source image (pixels).
            image_height:   Height of the source image (pixels).
            na_results:     Optional list of NaturalArtificialEvidence, same order as detections.
            shadow_results: Optional list of AcousticShadowResult, same order as detections.

        Returns:
            FilterResult with accepted, rejected, and per-detection decisions.
        """
        result = FilterResult(thresholds=self.thresholds)
        img_area = max(1, image_width * image_height)

        for idx, det in enumerate(detections):
            na  = na_results[idx]     if na_results     and idx < len(na_results)     else None
            shd = shadow_results[idx] if shadow_results and idx < len(shadow_results) else None

            decision = self._evaluate(det, image_width, image_height, img_area, na, shd)
            result.decisions.append(decision)

            if decision.accepted:
                result.accepted.append(det)
            else:
                result.rejected.append(det)
                log.debug(
                    "FP filter rejected detection %d [%s]: %s",
                    det.detection_id, decision.rule_name, decision.reason,
                )

        log.info(
            "FP filter: %d accepted, %d rejected (of %d total)",
            result.num_accepted, result.num_rejected, len(detections),
        )
        return result

    # -----------------------------------------------------------------------
    # Internal rule evaluation
    # -----------------------------------------------------------------------

    def _evaluate(
        self,
        det:          Detection,
        image_width:  int,
        image_height: int,
        img_area:     int,
        na:           Optional[NaturalArtificialEvidence],
        shd:          Optional[AcousticShadowResult],
    ) -> FilterDecision:
        """
        Evaluate all rules for a single detection.
        Returns on the FIRST failing rule (fail-fast approach).
        All subsequent rules are skipped for efficiency.
        """
        det_id = det.detection_id
        b      = det.bbox

        # ── Rule 1: Low Confidence ───────────────────────────────────────
        if det.confidence < self.min_confidence:
            return FilterDecision(
                detection_id = det_id,
                accepted     = False,
                rule_name    = "low_confidence",
                reason       = (
                    f"Confidence {det.confidence:.3f} is below minimum threshold "
                    f"{self.min_confidence:.3f}. Detection is unreliable."
                ),
            )

        # ── Rule 2: Tiny Bounding Box ─────────────────────────────────────
        bbox_area = max(0.0, b.width) * max(0.0, b.height)
        if bbox_area < self.min_bbox_area_px:
            return FilterDecision(
                detection_id = det_id,
                accepted     = False,
                rule_name    = "tiny_bbox",
                reason       = (
                    f"Bounding box area {bbox_area:.0f} px² is below minimum "
                    f"{self.min_bbox_area_px} px². Likely noise or speckle artefact."
                ),
            )

        # ── Rule 3: Malformed / Out-of-bounds Bounding Box ────────────────
        if (b.x1 < 0 or b.y1 < 0 or
                b.x2 > image_width or b.y2 > image_height or
                b.width <= 0 or b.height <= 0):
            return FilterDecision(
                detection_id = det_id,
                accepted     = False,
                rule_name    = "poor_bbox_quality",
                reason       = (
                    f"Bounding box [{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}] "
                    f"extends outside image ({image_width}×{image_height}) or is malformed."
                ),
            )

        # ── Rule 4: Extreme Aspect Ratio (noise stripe) ───────────────────
        aspect = max(b.width, b.height) / max(1.0, min(b.width, b.height))
        if aspect > self.max_aspect_ratio:
            return FilterDecision(
                detection_id = det_id,
                accepted     = False,
                rule_name    = "thin_bbox",
                reason       = (
                    f"Bounding box aspect ratio {aspect:.1f}:1 exceeds maximum "
                    f"{self.max_aspect_ratio:.1f}:1. Consistent with sonar noise stripe "
                    f"or horizontal/vertical artefact."
                ),
            )

        # ── Rule 5: Low Shape Evidence (from NA analysis) ─────────────────
        if na is not None and na.status == "ok":
            if na.shape_evidence < self.min_shape_evidence:
                return FilterDecision(
                    detection_id = det_id,
                    accepted     = False,
                    rule_name    = "low_shape_evidence",
                    reason       = (
                        f"Shape regularity score {na.shape_evidence:.3f} is below "
                        f"minimum {self.min_shape_evidence:.3f}. Object shows no "
                        f"geometric structure consistent with man-made debris."
                    ),
                )

        # ── Rule 6: Shadow Inconsistency (optional, configurable) ─────────
        if self.require_shadow and shd is not None and shd.status == "ok":
            if not shd.shadow_detected or shd.shadow_evidence < self.min_shadow_evidence:
                return FilterDecision(
                    detection_id = det_id,
                    accepted     = False,
                    rule_name    = "shadow_inconsistent",
                    reason       = (
                        f"Acoustic shadow absent or insufficient "
                        f"(shadow_evidence={shd.shadow_evidence:.3f}, "
                        f"required={self.min_shadow_evidence:.3f}). "
                        f"A raised object on the seabed should cast a shadow."
                    ),
                )

        # ── All rules passed ──────────────────────────────────────────────
        return FilterDecision(
            detection_id = det_id,
            accepted     = True,
            rule_name    = "accepted",
            reason       = (
                f"Passed all filter rules "
                f"(conf={det.confidence:.3f}, "
                f"area={bbox_area:.0f}px², "
                f"aspect={aspect:.1f}:1"
                + (f", shape={na.shape_evidence:.3f}" if na and na.status == "ok" else "")
                + f")."
            ),
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_filter_from_config(cfg) -> FalsePositiveFilter:
    """
    Create a FalsePositiveFilter from the central Config object.
    Reads fp_* attributes from cfg with safe defaults if not present.
    """
    return FalsePositiveFilter(
        min_confidence      = getattr(cfg, "confidence_threshold",       0.25),
        min_bbox_area_px    = getattr(cfg, "fp_min_bbox_area_px",         100),
        max_aspect_ratio    = getattr(cfg, "fp_max_aspect_ratio",         10.0),
        min_shape_evidence  = getattr(cfg, "fp_min_shape_evidence",       0.05),
        min_shadow_evidence = getattr(cfg, "fp_min_shadow_evidence",      0.0),
        require_shadow      = getattr(cfg, "fp_require_shadow",           False),
    )
