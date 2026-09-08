"""
SONAR-GUARD — Acoustic Shadow Analysis
========================================
Detects and measures acoustic shadows in side-scan sonar imagery.

Physical basis:
  In side-scan sonar, a proud (upstanding) object on the seabed blocks
  the acoustic signal, creating a darker "shadow" region extending away
  from the sonar transducer (horizontally in a standard waterfall image).

  Shadow presence, extent, and contrast are diagnostically useful:
  - Strong, well-defined shadow → object with significant relief
  - No shadow → flat, low-relief object (e.g. net, thin debris)

Implementation:
  Image-based shadow analysis within a candidate region.
  The shadow search zone is placed on one horizontal side of the bbox
  (the away-from-centre side, approximating the sonar geometry).

  NOTE: Without precise sonar geometry (range, heading, altitude),
  the shadow direction is estimated, not exact.
  Results should be treated as evidence, not ground truth.

Output: AcousticShadowResult (dataclass)
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detection.yolo_detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AcousticShadowResult:
    """
    Acoustic shadow analysis result for one detected object.

    Attributes:
        shadow_detected:     True if a statistically significant shadow was found.
        shadow_strength:     Normalised shadow intensity (0–1; higher = darker shadow).
        shadow_area_px:      Number of shadow pixels found.
        shadow_mean_intensity: Mean intensity of shadow region (0–255, lower = darker).
        target_mean_intensity: Mean intensity of target region (0–255).
        shadow_evidence:     Combined evidence score (0–1; higher = stronger evidence).
        search_direction:    'right' or 'left' — which side was searched.
        analysis_time_s:     Wall-clock analysis time.
        status:              Processing status message.
    """
    shadow_detected:        bool  = False
    shadow_strength:        float = 0.0
    shadow_area_px:         int   = 0
    shadow_mean_intensity:  float = 0.0
    target_mean_intensity:  float = 0.0
    shadow_evidence:        float = 0.0
    search_direction:       str   = "right"
    analysis_time_s:        float = 0.0
    status:                 str   = "not_run"

    def to_dict(self) -> dict:
        return {
            "shadow_detected":       self.shadow_detected,
            "shadow_strength":       round(self.shadow_strength, 4),
            "shadow_area_px":        self.shadow_area_px,
            "shadow_mean_intensity": round(self.shadow_mean_intensity, 2),
            "target_mean_intensity": round(self.target_mean_intensity, 2),
            "shadow_evidence":       round(self.shadow_evidence, 4),
            "search_direction":      self.search_direction,
            "analysis_time_s":       round(self.analysis_time_s, 4),
            "status":                self.status,
        }

    def explanation_lines(self) -> list:
        """Human-readable evidence lines for the dashboard."""
        if self.shadow_detected:
            return [("✓", f"Acoustic shadow detected (strength={self.shadow_strength:.2f}, "
                          f"evidence={self.shadow_evidence:.2f})")]
        elif self.shadow_evidence >= 0.3:
            return [("⚠", f"Weak acoustic shadow signal (evidence={self.shadow_evidence:.2f})")]
        else:
            return [("✗", f"No acoustic shadow detected (evidence={self.shadow_evidence:.2f})")]


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class AcousticShadowAnalyser:
    """
    Image-based acoustic shadow analyser.

    Approach:
      1. Extract the target region from the YOLO bbox.
      2. Define a search zone on one side of the bbox (estimated shadow side).
      3. Compare mean intensity of search zone vs target and global seabed.
      4. A significantly darker search zone indicates shadow presence.

    Limitations:
      - Shadow direction is estimated from image position (target in right half
        → search right; left half → search left). This is a rough heuristic.
      - Without sonar heading, range, and altitude metadata, the exact shadow
        geometry cannot be computed.
      - Results should be used as probabilistic evidence, not definitive detection.
    """

    def __init__(
        self,
        shadow_search_ratio:    float = 0.5,
        shadow_intensity_thresh: float = 0.35,
        min_shadow_area_px:     int   = 30,
    ):
        """
        Args:
            shadow_search_ratio:     Fraction of bbox width to search for shadow
                                     on each side. 0.5 = 50% of bbox width.
            shadow_intensity_thresh: Relative intensity drop threshold.
                                     If shadow_zone_mean < target_mean * (1 - thresh),
                                     shadow is declared detected.
            min_shadow_area_px:      Minimum number of dark pixels to count as shadow.
        """
        self.shadow_search_ratio    = shadow_search_ratio
        self.shadow_intensity_thresh = shadow_intensity_thresh
        self.min_shadow_area_px     = min_shadow_area_px

    def analyse(
        self,
        image:     np.ndarray,
        detection: Detection,
    ) -> AcousticShadowResult:
        """
        Analyse the acoustic shadow of a detection.

        Args:
            image:     Grayscale (HxW) or BGR (HxWx3) preprocessed image.
            detection: YOLO Detection object with bounding box.

        Returns:
            AcousticShadowResult — always returned.
        """
        t_start = time.perf_counter()

        h, w = image.shape[:2]

        # Ensure grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        bbox = detection.bbox
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w - 1, int(bbox.x2))
        y2 = min(h - 1, int(bbox.y2))
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return AcousticShadowResult(
                status="invalid_bbox",
                analysis_time_s=time.perf_counter() - t_start,
            )

        # ── Target region stats ────────────────────────────────────────
        target_region = gray[y1:y2, x1:x2].astype(np.float32)
        target_mean   = float(np.mean(target_region))

        # ── Estimate shadow search direction ───────────────────────────
        # Heuristic: target in the left half → shadow to the right (and vice versa).
        # This approximates standard SSS waterfall layout where nadir is centre.
        target_cx = (x1 + x2) / 2.0
        if target_cx < w / 2:
            direction = "right"
            shadow_x1 = x2
            shadow_x2 = min(w - 1, x2 + int(bw * self.shadow_search_ratio))
        else:
            direction = "left"
            shadow_x1 = max(0, x1 - int(bw * self.shadow_search_ratio))
            shadow_x2 = x1

        shadow_y1 = y1
        shadow_y2 = y2

        # ── Extract shadow search zone ─────────────────────────────────
        if shadow_x2 <= shadow_x1:
            return AcousticShadowResult(
                search_direction=direction,
                status="shadow_zone_outside_image",
                analysis_time_s=time.perf_counter() - t_start,
            )

        shadow_zone = gray[shadow_y1:shadow_y2, shadow_x1:shadow_x2].astype(np.float32)
        if shadow_zone.size == 0:
            return AcousticShadowResult(
                search_direction=direction,
                status="empty_shadow_zone",
                analysis_time_s=time.perf_counter() - t_start,
            )

        shadow_mean = float(np.mean(shadow_zone))

        # ── Shadow detection logic ─────────────────────────────────────
        # Shadow pixels: darker than (1 - threshold) × target mean
        intensity_threshold = target_mean * (1.0 - self.shadow_intensity_thresh)
        shadow_pixel_mask   = shadow_zone < max(1.0, intensity_threshold)
        shadow_area_px      = int(np.sum(shadow_pixel_mask))

        # Normalised shadow strength: how much darker is shadow vs target
        if target_mean > 0:
            shadow_strength = float(
                np.clip(1.0 - (shadow_mean / target_mean), 0.0, 1.0)
            )
        else:
            shadow_strength = 0.0

        shadow_detected = (
            shadow_strength >= self.shadow_intensity_thresh and
            shadow_area_px  >= self.min_shadow_area_px
        )

        # ── Shadow evidence score (0–1) ────────────────────────────────
        # Combines: strength, spatial coverage, and detection flag
        coverage_ratio  = shadow_area_px / max(1, shadow_zone.size)
        shadow_evidence = float(np.clip(
            0.6 * shadow_strength +
            0.4 * coverage_ratio,
            0.0, 1.0
        ))

        elapsed = time.perf_counter() - t_start
        log.debug(
            "Shadow analysis det-%d | detected=%s strength=%.2f evidence=%.2f | %.3f s",
            detection.detection_id,
            shadow_detected, shadow_strength, shadow_evidence, elapsed,
        )

        return AcousticShadowResult(
            shadow_detected=shadow_detected,
            shadow_strength=round(shadow_strength, 4),
            shadow_area_px=shadow_area_px,
            shadow_mean_intensity=round(shadow_mean, 2),
            target_mean_intensity=round(target_mean, 2),
            shadow_evidence=round(shadow_evidence, 4),
            search_direction=direction,
            analysis_time_s=round(elapsed, 4),
            status="ok",
        )

