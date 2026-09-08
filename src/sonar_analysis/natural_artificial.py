"""
SONAR-GUARD — Natural vs. Artificial Analysis
===============================================
Computes measurable shape, texture, and seabed-context features
from within the YOLO bounding box to produce evidence scores for
Natural ↔ Artificial classification.

All scores are computed from actual image measurements.
They represent evidence strength, NOT scientific ground truth.

Output: NaturalArtificialEvidence (dataclass)
  shape_evidence    — geometric regularity (0–1, higher = more artificial)
  texture_evidence  — texture homogeneity vs seabed (0–1)
  context_evidence  — target contrast against local seabed (0–1)
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detection.yolo_detector import Detection
from src.segmentation.segmenter import SegmentationResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NaturalArtificialEvidence:
    """
    Evidence features from shape, texture, and context analysis.

    All values are computed from actual image measurements.
    Higher values indicate stronger artificial (man-made) evidence.

    Attributes:
        shape_evidence:       Geometric regularity (rectangularity, compactness).
        texture_evidence:     Texture homogeneity relative to local seabed.
        context_evidence:     Target intensity contrast against surrounding seabed.
        aspect_ratio:         bbox width / bbox height.
        rectangularity:       mask_area / bbox_area (from segmentation).
        compactness:          4π × area / perimeter² (1.0 = perfect circle).
        contour_complexity:   Number of contour vertices (normalised).
        local_variance:       Variance of pixel intensities inside mask.
        seabed_variance:      Variance of pixel intensities in surrounding region.
        target_mean_intensity: Mean intensity inside mask/bbox.
        seabed_mean_intensity: Mean intensity in surrounding context region.
        analysis_time_s:      Wall-clock analysis time.
        status:               Human-readable processing status.
    """
    shape_evidence:         float = 0.0
    texture_evidence:       float = 0.0
    context_evidence:       float = 0.0
    aspect_ratio:           float = 0.0
    rectangularity:         float = 0.0
    compactness:            float = 0.0
    contour_complexity:     float = 0.0
    local_variance:         float = 0.0
    seabed_variance:        float = 0.0
    target_mean_intensity:  float = 0.0
    seabed_mean_intensity:  float = 0.0
    analysis_time_s:        float = 0.0
    status:                 str   = "not_run"

    def to_dict(self) -> dict:
        return {
            "shape_evidence":         round(self.shape_evidence, 4),
            "texture_evidence":       round(self.texture_evidence, 4),
            "context_evidence":       round(self.context_evidence, 4),
            "aspect_ratio":           round(self.aspect_ratio, 4),
            "rectangularity":         round(self.rectangularity, 4),
            "compactness":            round(self.compactness, 4),
            "contour_complexity":     round(self.contour_complexity, 4),
            "local_variance":         round(self.local_variance, 2),
            "seabed_variance":        round(self.seabed_variance, 2),
            "target_mean_intensity":  round(self.target_mean_intensity, 2),
            "seabed_mean_intensity":  round(self.seabed_mean_intensity, 2),
            "analysis_time_s":        round(self.analysis_time_s, 4),
            "status":                 self.status,
        }

    def explanation_lines(self) -> list:
        """Human-readable evidence lines for the 'Why was this flagged?' panel."""
        lines = []
        # Shape
        if self.shape_evidence >= 0.70:
            lines.append(("✓", f"Strong geometric structure (shape_evidence={self.shape_evidence:.2f})"))
        elif self.shape_evidence >= 0.45:
            lines.append(("⚠", f"Moderate geometric regularity (shape_evidence={self.shape_evidence:.2f})"))
        else:
            lines.append(("✗", f"Irregular/natural shape (shape_evidence={self.shape_evidence:.2f})"))
        # Texture
        if self.texture_evidence >= 0.65:
            lines.append(("✓", f"Texture differs from seabed (texture_evidence={self.texture_evidence:.2f})"))
        elif self.texture_evidence >= 0.40:
            lines.append(("⚠", f"Moderate texture contrast (texture_evidence={self.texture_evidence:.2f})"))
        else:
            lines.append(("✗", f"Texture similar to seabed (texture_evidence={self.texture_evidence:.2f})"))
        # Context
        if self.context_evidence >= 0.65:
            lines.append(("✓", f"Target contrast differs from seabed (context_evidence={self.context_evidence:.2f})"))
        elif self.context_evidence >= 0.40:
            lines.append(("⚠", f"Moderate target contrast (context_evidence={self.context_evidence:.2f})"))
        else:
            lines.append(("✗", f"Low contrast relative to seabed (context_evidence={self.context_evidence:.2f})"))
        return lines


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class NaturalArtificialAnalyser:
    """
    Computes shape, texture, and seabed-context evidence from
    the detected region and its surroundings in the sonar image.

    Parameters control the size of the seabed context region
    (how far around the bbox to sample background statistics).
    """

    def __init__(self, context_margin_ratio: float = 0.5):
        """
        Args:
            context_margin_ratio: How much extra region around the bbox to
                                  use for seabed statistics (as fraction of bbox size).
                                  0.5 = 50% extra on each side.
        """
        self.context_margin_ratio = context_margin_ratio

    def analyse(
        self,
        image:      np.ndarray,
        detection:  Detection,
        seg_result: Optional[SegmentationResult] = None,
    ) -> NaturalArtificialEvidence:
        """
        Analyse a detection's region for natural vs artificial features.

        Args:
            image:      Grayscale (HxW) preprocessed image.
            detection:  YOLO Detection object.
            seg_result: Optional segmentation result for mask-based stats.

        Returns:
            NaturalArtificialEvidence — always returned.
        """
        t_start = time.perf_counter()

        h, w = image.shape[:2]
        bbox = detection.bbox

        # Clamp bbox to image
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w - 1, int(bbox.x2))
        y2 = min(h - 1, int(bbox.y2))
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return NaturalArtificialEvidence(
                status="invalid_bbox",
                analysis_time_s=time.perf_counter() - t_start,
            )

        # Ensure grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # ── Target region ─────────────────────────────────────────────
        crop = gray[y1:y2, x1:x2].astype(np.float32)

        # ── Seabed context (annular region around bbox) ───────────────
        mx = int(max(bw, bh) * self.context_margin_ratio)
        cx1 = max(0,     x1 - mx)
        cy1 = max(0,     y1 - mx)
        cx2 = min(w - 1, x2 + mx)
        cy2 = min(h - 1, y2 + mx)
        context_full = gray[cy1:cy2, cx1:cx2].astype(np.float32)

        # Mask out the target bbox from the context region to get seabed-only
        context_mask = np.ones(context_full.shape, dtype=bool)
        oy1 = y1 - cy1;  oy2 = y2 - cy1
        ox1 = x1 - cx1;  ox2 = x2 - cx1
        context_mask[max(0,oy1):min(context_full.shape[0],oy2),
                     max(0,ox1):min(context_full.shape[1],ox2)] = False
        seabed_pixels = context_full[context_mask]

        if seabed_pixels.size == 0:
            seabed_pixels = context_full.ravel()  # fallback

        target_mean = float(np.mean(crop))
        seabed_mean = float(np.mean(seabed_pixels))
        local_var   = float(np.var(crop))
        seabed_var  = float(np.var(seabed_pixels))

        # ── SHAPE FEATURES ────────────────────────────────────────────
        aspect_ratio  = float(bw) / max(1, float(bh))

        # Rectangularity: how much of the bbox is filled
        if seg_result and seg_result.available:
            rect_score = float(seg_result.mask_to_box_ratio)
        else:
            # Estimate from Otsu threshold on the crop
            _, thresh = cv2.threshold(
                crop.astype(np.uint8), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            rect_score = float(np.sum(thresh > 0)) / max(1, thresh.size)

        # Compactness from largest contour
        contours, _ = cv2.findContours(
            ((crop > np.percentile(crop, 60)) * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        compactness  = 0.0
        contour_complexity = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_c  = cv2.contourArea(largest)
            peri_c  = cv2.arcLength(largest, True)
            if peri_c > 0:
                compactness = float(4 * np.pi * area_c / (peri_c ** 2))
            # Normalised vertex count (100 vertices = complexity 1.0)
            contour_complexity = min(1.0, len(largest) / 100.0)

        # Shape evidence score
        # Artificial objects tend to have: high rectangularity, moderate compactness,
        # regular (low complexity) contours, non-extreme aspect ratios
        rect_contrib    = rect_score                           # 0–1
        compact_contrib = min(1.0, compactness * 2)           # circles/squares score high
        reg_contrib     = 1.0 - contour_complexity             # low complexity = regular
        # Penalise very extreme aspect ratios (likely seabed structures)
        ar_penalty = 1.0 - min(1.0, abs(np.log(max(0.1, aspect_ratio))) / 2.0)
        shape_evidence = float(
            0.35 * rect_contrib +
            0.30 * compact_contrib +
            0.20 * reg_contrib +
            0.15 * ar_penalty
        )
        shape_evidence = float(np.clip(shape_evidence, 0.0, 1.0))

        # ── TEXTURE EVIDENCE ──────────────────────────────────────────
        # Artificial objects may show different texture from seabed
        # Use ratio of local variance to seabed variance as evidence
        if seabed_var > 0:
            var_ratio = local_var / seabed_var
        else:
            var_ratio = 1.0

        # A ratio far from 1.0 indicates different texture
        texture_diff  = abs(var_ratio - 1.0)
        texture_evidence = float(np.clip(texture_diff / (1.0 + texture_diff), 0.0, 1.0))

        # ── CONTEXT EVIDENCE ──────────────────────────────────────────
        # Target intensity contrast vs seabed
        intensity_diff = abs(target_mean - seabed_mean)
        # Normalise against the full 0–255 range
        context_evidence = float(np.clip(intensity_diff / 128.0, 0.0, 1.0))

        elapsed = time.perf_counter() - t_start
        log.debug(
            "NA analysis det-%d | shape=%.2f texture=%.2f context=%.2f | %.3f s",
            detection.detection_id,
            shape_evidence, texture_evidence, context_evidence, elapsed,
        )

        return NaturalArtificialEvidence(
            shape_evidence=shape_evidence,
            texture_evidence=texture_evidence,
            context_evidence=context_evidence,
            aspect_ratio=round(aspect_ratio, 3),
            rectangularity=round(rect_score, 3),
            compactness=round(compactness, 3),
            contour_complexity=round(contour_complexity, 3),
            local_variance=round(local_var, 2),
            seabed_variance=round(seabed_var, 2),
            target_mean_intensity=round(target_mean, 2),
            seabed_mean_intensity=round(seabed_mean, 2),
            analysis_time_s=round(elapsed, 4),
            status="ok",
        )

