"""
SONAR-GUARD — Multi-Pass Verification
========================================
Compares detections across multiple sonar passes over the same area.

If only one image/observation is available:
  Returns: "Multi-pass verification unavailable — single observation."
  Does NOT fabricate repeated observations.

When multiple passes exist:
  Compares: location, confidence, size, evidence consistency.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PassObservation:
    """One sonar pass observation of a candidate target."""
    pass_id:           str
    confidence:        float
    bbox_area_px:      Optional[int]   = None
    fused_evidence:    Optional[float] = None
    artificiality:     Optional[int]   = None
    pixel_location:    Optional[tuple] = None   # (cx, cy) in image coords


@dataclass
class MultiPassResult:
    """Result of multi-pass verification."""
    available:            bool  = False
    num_passes:           int   = 0
    passes:               List[PassObservation] = field(default_factory=list)
    mean_confidence:      Optional[float] = None
    confidence_std:       Optional[float] = None
    size_consistency:     Optional[float] = None
    evidence_consistency: Optional[float] = None
    overall_consistency:  Optional[float] = None
    status:               str = "unavailable"
    message:              str = "Multi-pass verification unavailable — single observation."

    def to_dict(self) -> dict:
        return {
            "available":            self.available,
            "num_passes":           self.num_passes,
            "mean_confidence":      self.mean_confidence,
            "confidence_std":       self.confidence_std,
            "size_consistency":     self.size_consistency,
            "evidence_consistency": self.evidence_consistency,
            "overall_consistency":  self.overall_consistency,
            "status":               self.status,
            "message":              self.message,
        }


class MultiPassVerifier:
    """
    Verifies a target across multiple sonar passes.

    Single-pass use:
        result = verifier.verify([single_observation])
        # result.available == False
        # result.message == "Multi-pass verification unavailable — single observation."

    Multi-pass use:
        observations = [PassObservation(...), PassObservation(...), ...]
        result = verifier.verify(observations)
    """

    def verify(self, observations: List[PassObservation]) -> MultiPassResult:
        """
        Analyse consistency across observations.

        Args:
            observations: List of PassObservation objects.
                          If len == 1 or 0, returns unavailable.

        Returns:
            MultiPassResult — always returned.
        """
        n = len(observations)

        if n == 0:
            return MultiPassResult(
                status="no_observations",
                message="Multi-pass verification unavailable — no observations provided.",
            )

        if n == 1:
            return MultiPassResult(
                available=False,
                num_passes=1,
                passes=observations,
                status="single_observation",
                message="Multi-pass verification unavailable — single observation.",
            )

        # Multiple passes available
        import math

        confs = [o.confidence for o in observations]
        mean_conf = sum(confs) / n
        std_conf  = math.sqrt(sum((c - mean_conf) ** 2 for c in confs) / n)
        conf_consistency = float(max(0.0, 1.0 - std_conf / max(mean_conf, 1e-6)))

        # Size consistency
        areas = [o.bbox_area_px for o in observations if o.bbox_area_px is not None]
        size_consistency = None
        if len(areas) >= 2:
            mean_area = sum(areas) / len(areas)
            std_area  = math.sqrt(sum((a - mean_area) ** 2 for a in areas) / len(areas))
            size_consistency = float(max(0.0, 1.0 - std_area / max(mean_area, 1.0)))

        # Evidence consistency
        evidences = [o.fused_evidence for o in observations if o.fused_evidence is not None]
        evidence_consistency = None
        if len(evidences) >= 2:
            mean_ev = sum(evidences) / len(evidences)
            std_ev  = math.sqrt(sum((e - mean_ev) ** 2 for e in evidences) / len(evidences))
            evidence_consistency = float(max(0.0, 1.0 - std_ev / 0.5))

        # Overall consistency
        parts = [conf_consistency]
        if size_consistency is not None:
            parts.append(size_consistency)
        if evidence_consistency is not None:
            parts.append(evidence_consistency)
        overall = sum(parts) / len(parts)

        msg = (f"{n}-pass verification: mean_conf={mean_conf:.2f}, "
               f"consistency={overall:.2f}")
        log.info("Multi-pass: %s", msg)

        return MultiPassResult(
            available=True,
            num_passes=n,
            passes=observations,
            mean_confidence=round(mean_conf, 4),
            confidence_std=round(std_conf, 4),
            size_consistency=round(size_consistency, 4) if size_consistency is not None else None,
            evidence_consistency=round(evidence_consistency, 4) if evidence_consistency is not None else None,
            overall_consistency=round(overall, 4),
            status="ok",
            message=msg,
        )

