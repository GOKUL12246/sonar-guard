"""
SONAR-GUARD — Marine Risk Score
=================================
Computes a 0–100 SONAR-GUARD prioritisation score.

Bands:
  0–24   Low
  25–49  Medium
  50–74  High
  75–100 Critical

The risk score is based on measurable system inputs:
  - Artificiality score (primary driver)
  - Target size estimate (from segmentation or bbox)
  - Shadow strength (object height/relief proxy)
  - Uncertainty penalty
  - Object category if reliably identified
  - Verification status boost/penalty

LABEL REQUIRED ON ALL DISPLAYS:
  "SONAR-GUARD prioritisation score"
This is NOT a scientifically validated ecological risk probability.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.scoring.artificiality import ArtificialityScore
from src.segmentation.segmenter import SegmentationResult
from src.sonar_analysis.acoustic_shadow import AcousticShadowResult
from src.uncertainty.estimator import UncertaintyResult
from src.utils.logger import get_logger

log = get_logger(__name__)

# Risk band definitions (max_score → label, colour)
RISK_BANDS = [
    (24,  "Low",      "#2ecc71"),
    (49,  "Medium",   "#f39c12"),
    (74,  "High",     "#e74c3c"),
    (100, "Critical", "#8e44ad"),
]


def risk_label(score: int) -> str:
    for max_val, label, _ in RISK_BANDS:
        if score <= max_val:
            return label
    return "Critical"


def risk_colour(score: int) -> str:
    for max_val, _, colour in RISK_BANDS:
        if score <= max_val:
            return colour
    return "#8e44ad"


@dataclass
class MarineRiskScore:
    """
    Marine Risk Score result.

    Displayed label: "SONAR-GUARD prioritisation score"
    NOT a scientific probability.

    Attributes:
        score:              0–100 integer.
        band:               'Low' | 'Medium' | 'High' | 'Critical'
        colour:             Hex colour code for UI display.
        artificiality:      Input artificiality score (0–100).
        size_factor:        Normalised target size contribution (0–1).
        shadow_factor:      Shadow strength contribution (0–1).
        uncertainty_penalty: Uncertainty penalty applied (0–1 reduction).
        verification_modifier: Adjustment from human verification status.
        reasons:            Evidence lines for dashboard.
        status:             Processing status.
    """
    score:                  int   = 0
    band:                   str   = "Low"
    colour:                 str   = "#2ecc71"
    artificiality:          int   = 0
    size_factor:            float = 0.0
    shadow_factor:          float = 0.0
    uncertainty_penalty:    float = 0.0
    verification_modifier:  float = 0.0
    reasons:                List[Tuple[str, str]] = field(default_factory=list)
    status:                 str   = "not_run"

    def to_dict(self) -> dict:
        return {
            "score":                 self.score,
            "band":                  self.band,
            "colour":                self.colour,
            "artificiality_input":   self.artificiality,
            "size_factor":           round(self.size_factor, 4),
            "shadow_factor":         round(self.shadow_factor, 4),
            "uncertainty_penalty":   round(self.uncertainty_penalty, 4),
            "verification_modifier": round(self.verification_modifier, 4),
            "status":                self.status,
        }


def compute_marine_risk_score(
    artificiality:      ArtificialityScore,
    seg_result:         Optional[SegmentationResult]  = None,
    shadow_result:      Optional[AcousticShadowResult] = None,
    unc_result:         Optional[UncertaintyResult]    = None,
    verification_status: str = "unverified",
    image_area_px:      int  = 0,
) -> MarineRiskScore:
    """
    Compute the SONAR-GUARD Marine Risk Score.

    Args:
        artificiality:       ArtificialityScore result.
        seg_result:          Segmentation result (for size).
        shadow_result:       Shadow result (for relief proxy).
        unc_result:          Uncertainty result (for penalty).
        verification_status: 'unverified' | 'confirmed' | 'rejected' | 'rov_inspection'
        image_area_px:       Total image area in pixels (for size normalisation).

    Returns:
        MarineRiskScore.
    """
    if artificiality.status != "ok":
        return MarineRiskScore(status=f"artificiality_error: {artificiality.status}")

    art_score = artificiality.score  # 0–100
    reasons: List[Tuple[str, str]] = []

    # ── Component 1: Artificiality (primary, 60% weight) ─────────────
    art_contribution = art_score * 0.60  # 0–60

    # ── Component 2: Target size (15% weight) ─────────────────────────
    size_factor = 0.5  # neutral default (unknown size)
    if seg_result and seg_result.available and image_area_px > 0:
        # Normalise mask area against image area
        # Large objects (>5% of image) get higher size factor
        area_ratio = seg_result.mask_area_px / image_area_px
        size_factor = float(min(1.0, area_ratio / 0.05))
        reasons.append(("→", f"Target size: {seg_result.mask_area_px} px "
                             f"({area_ratio*100:.1f}% of image)"))
    elif seg_result and seg_result.bbox_area_px > 0 and image_area_px > 0:
        area_ratio = seg_result.bbox_area_px / image_area_px
        size_factor = float(min(1.0, area_ratio / 0.05))
        reasons.append(("→", f"Bbox size: {seg_result.bbox_area_px} px (no mask)"))
    else:
        reasons.append(("⚠", "Target size unknown — using neutral estimate"))

    size_contribution = size_factor * 15  # 0–15

    # ── Component 3: Shadow strength (15% weight) ─────────────────────
    shadow_factor = 0.0
    if shadow_result and shadow_result.status == "ok":
        shadow_factor = shadow_result.shadow_strength
        reasons.append(("→", f"Shadow strength: {shadow_factor:.2f}"))
    else:
        reasons.append(("⚠", "Shadow strength unavailable"))

    shadow_contribution = shadow_factor * 15  # 0–15

    # ── Uncertainty penalty (up to -10 points) ────────────────────────
    uncertainty_penalty = 0.0
    if unc_result and unc_result.status == "ok":
        uncertainty_penalty = unc_result.uncertainty * 10  # 0–10
        reasons.append(("→", f"Uncertainty penalty: -{uncertainty_penalty:.1f} "
                             f"(uncertainty={unc_result.uncertainty:.2f})"))
    else:
        reasons.append(("⚠", "Uncertainty not assessed"))

    # ── Verification modifier ─────────────────────────────────────────
    verification_modifier = 0.0
    if verification_status == "confirmed":
        verification_modifier = +10.0
        reasons.append(("✓", "Human verified: CONFIRMED (+10)"))
    elif verification_status == "rejected":
        verification_modifier = -30.0
        reasons.append(("✗", "Human verified: REJECTED (−30)"))
    elif verification_status == "rov_inspection":
        verification_modifier = +5.0
        reasons.append(("⚠", "Flagged for ROV inspection (+5)"))
    else:
        reasons.append(("⚠", "Unverified — human review recommended"))

    # ── Final score ───────────────────────────────────────────────────
    raw_score = (
        art_contribution +
        size_contribution +
        shadow_contribution -
        uncertainty_penalty +
        verification_modifier
    )
    score = int(round(max(0.0, min(100.0, raw_score))))
    band  = risk_label(score)
    colour = risk_colour(score)

    # Prepend primary reason
    reasons.insert(0, ("→", f"Artificiality score: {art_score}/100 × 0.60 = {art_contribution:.1f} pts"))

    log.debug("Risk score: %d/100 (%s)", score, band)

    return MarineRiskScore(
        score=score,
        band=band,
        colour=colour,
        artificiality=art_score,
        size_factor=round(size_factor, 4),
        shadow_factor=round(shadow_factor, 4),
        uncertainty_penalty=round(uncertainty_penalty, 2),
        verification_modifier=verification_modifier,
        reasons=reasons,
        status="ok",
    )

