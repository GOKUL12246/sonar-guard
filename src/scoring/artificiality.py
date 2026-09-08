"""
SONAR-GUARD — Artificiality Score
====================================
Converts the fused evidence score into a 0–100 Artificiality Score.

  0   = strongly natural evidence
  100 = strongly artificial (man-made) evidence

This is a SYSTEM-DEFINED prioritisation score, NOT a calibrated probability.
It is always displayed with its evidence basis — never as a bare number.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.fusion.evidence_fusion import EvidenceFusionResult
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ArtificialityScore:
    """
    Artificiality Score result.

    Attributes:
        score:          0–100 integer score.
        label:          'Natural' | 'Likely Natural' | 'Uncertain' |
                        'Likely Artificial' | 'Artificial'
        fused_evidence: The underlying fused evidence score (0–1).
        reasons:        List of (symbol, reason_string) tuples.
        status:         Processing status.
    """
    score:          int   = 0
    label:          str   = "Unknown"
    fused_evidence: float = 0.0
    reasons:        List[Tuple[str, str]] = field(default_factory=list)
    status:         str   = "not_run"

    # Label thresholds (score-based)
    _LABELS = [
        (80, "Artificial"),
        (60, "Likely Artificial"),
        (40, "Uncertain"),
        (20, "Likely Natural"),
        (0,  "Natural"),
    ]

    @classmethod
    def label_for_score(cls, score: int) -> str:
        for threshold, label in cls._LABELS:
            if score >= threshold:
                return label
        return "Natural"

    def to_dict(self) -> dict:
        return {
            "score":          self.score,
            "label":          self.label,
            "fused_evidence": round(self.fused_evidence, 4),
            "reasons":        [(s, r) for s, r in self.reasons],
            "status":         self.status,
        }


def compute_artificiality_score(
    fusion_result: EvidenceFusionResult,
    na_evidence=None,
    shadow_result=None,
    unc_result=None,
) -> ArtificialityScore:
    """
    Compute the Artificiality Score from the fused evidence result.

    Args:
        fusion_result: Output of EvidenceFusionEngine.fuse().
        na_evidence:   Optional — for additional reason generation.
        shadow_result: Optional — for additional reason generation.
        unc_result:    Optional — for uncertainty reason.

    Returns:
        ArtificialityScore with score, label, and reason list.
    """
    if fusion_result.status != "ok":
        return ArtificialityScore(status=f"fusion_error: {fusion_result.status}")

    fused = fusion_result.fused_score
    comp = fusion_result.components
    miss = fusion_result.missing_components
    det_conf = comp.get("detection", 0.0)

    # Calibrated non-linear scaling for marine debris detection
    # Ensures that detected man-made objects reflect positive artificiality likelihood
    if fused <= 0.05 and det_conf <= 0.10:
        score = int(round(fused * 100))
    else:
        # Scale active evidence smoothly into the 0-100 range
        norm_factor = max(0.0, min(1.0, (fused - 0.05) / 0.95))
        # Base neural confidence contributes strongly to artificiality prior
        det_boost = max(0.0, (det_conf - 0.20) * 20.0) if det_conf > 0.20 else 0.0
        scaled = 20.0 + (norm_factor ** 0.85) * 75.0 + min(15.0, det_boost)
        score = int(round(scaled))

    score = max(0, min(100, score))
    label = ArtificialityScore.label_for_score(score)

    # Build reason list
    reasons: List[Tuple[str, str]] = []

    # Detection confidence
    det_conf = comp.get("detection", 0.0)
    if det_conf >= 0.7:
        reasons.append(("✓", f"Strong detection confidence ({det_conf:.2f})"))
    elif det_conf >= 0.4:
        reasons.append(("⚠", f"Moderate detection confidence ({det_conf:.2f})"))
    else:
        reasons.append(("✗", f"Low detection confidence ({det_conf:.2f})"))

    # Shape
    shape = comp.get("shape", 0.0)
    if "shape" in miss:
        reasons.append(("✗", "Shape analysis unavailable"))
    elif shape >= 0.65:
        reasons.append(("✓", f"Geometric structure detected (shape={shape:.2f})"))
    elif shape >= 0.40:
        reasons.append(("⚠", f"Moderate geometric regularity (shape={shape:.2f})"))
    else:
        reasons.append(("✗", f"Irregular shape — natural indicator (shape={shape:.2f})"))

    # Texture
    tex = comp.get("texture", 0.0)
    if "texture" in miss:
        reasons.append(("✗", "Texture analysis unavailable"))
    elif tex >= 0.60:
        reasons.append(("✓", f"Texture differs from seabed (texture={tex:.2f})"))
    elif tex >= 0.35:
        reasons.append(("⚠", f"Moderate texture contrast (texture={tex:.2f})"))
    else:
        reasons.append(("✗", f"Texture similar to seabed (texture={tex:.2f})"))

    # Shadow
    shad = comp.get("shadow", 0.0)
    if "shadow" in miss:
        reasons.append(("✗", "Acoustic shadow analysis unavailable"))
    elif shad >= 0.60:
        reasons.append(("✓", f"Compatible acoustic shadow (shadow={shad:.2f})"))
    elif shad >= 0.30:
        reasons.append(("⚠", f"Weak acoustic shadow signal (shadow={shad:.2f})"))
    else:
        reasons.append(("✗", f"No acoustic shadow detected (shadow={shad:.2f})"))

    # Context contrast
    ctx = comp.get("context", 0.0)
    if "context" in miss:
        reasons.append(("✗", "Seabed context analysis unavailable"))
    elif ctx >= 0.60:
        reasons.append(("✓", f"Target contrasts with seabed (context={ctx:.2f})"))
    else:
        reasons.append(("⚠", f"Moderate seabed contrast (context={ctx:.2f})"))

    # Uncertainty
    unc_score = comp.get("uncertainty", 0.0)
    if "uncertainty" in miss:
        reasons.append(("⚠", "Uncertainty assessment unavailable"))
    elif unc_score >= 0.65:
        reasons.append(("✓", f"Consistent evidence signals (confidence={unc_score:.2f})"))
    elif unc_score >= 0.40:
        reasons.append(("⚠", f"Moderate uncertainty present (confidence={unc_score:.2f})"))
    else:
        reasons.append(("⚠", f"High uncertainty — limited evidence (confidence={unc_score:.2f})"))

    log.debug("Artificiality score: %d/100 (%s)", score, label)

    return ArtificialityScore(
        score=score,
        label=label,
        fused_evidence=fused,
        reasons=reasons,
        status="ok",
    )

