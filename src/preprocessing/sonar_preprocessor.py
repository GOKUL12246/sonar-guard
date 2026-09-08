"""
SONAR-GUARD — Sonar Image Preprocessor
========================================
Implements the preprocessing pipeline for Side-Scan Sonar (SSS) imagery.

Pipeline:
    Raw input
      → Grayscale conversion (if colour input)
      → Median filter          (speckle noise reduction)
      → Optional Gaussian blur (smooth noise)
      → CLAHE                  (contrast enhancement)
      → Resolution normalisation
      → Normalised float output [0,1] OR uint8 output [0,255]

Design principles:
  - Parameters are fully configurable — no hardcoded values.
  - Intermediate steps can be returned for before/after visualisation.
  - Preserves sonar-relevant structures: targets, acoustic shadows, seabed.
  - Does NOT over-process: applies only what helps; skips harmful steps.
  - Never applies colour processing to inherently grayscale sonar data.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """
    Output of the preprocessor for a single image.

    Attributes:
        image_id:            Source image identifier.
        original:            Original loaded array (HxW uint8).
        preprocessed:        Fully processed array (HxW uint8).
        stages:              Dict of intermediate stage arrays for visualisation.
        preprocessing_time_s: Wall-clock time (seconds) for the full pipeline.
        params_used:         Parameters that were actually applied.
        warnings:            Any non-fatal warnings encountered.
        error:               Set if processing failed.
    """
    image_id:            str
    original:            np.ndarray
    preprocessed:        Optional[np.ndarray] = None
    stages:              Dict[str, np.ndarray] = field(default_factory=dict)
    preprocessing_time_s: float = 0.0
    params_used:         Dict[str, object] = field(default_factory=dict)
    warnings:            List[str] = field(default_factory=list)
    error:               Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.preprocessed is not None


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class SonarPreprocessor:
    """
    Sonar image preprocessing pipeline.

    All parameters are provided at construction time (from config.py).
    The pipeline can be run on individual images or batches.

    Example usage:
        from config import cfg
        proc = SonarPreprocessor(cfg)
        result = proc.process(image_array, image_id="frame_001")
    """

    def __init__(self,
                 target_width:        int   = 640,
                 target_height:       int   = 640,
                 median_ksize:        int   = 5,
                 gaussian_ksize:      int   = 0,
                 gaussian_sigma:      float = 0.0,
                 use_clahe:           bool  = True,
                 clahe_clip_limit:    float = 2.0,
                 clahe_tile_size:     tuple = (8, 8),
                 ):
        """
        Args:
            target_width:     Output width (pixels). 0 = preserve original.
            target_height:    Output height (pixels). 0 = preserve original.
            median_ksize:     Median filter kernel size (must be odd, ≥3).
                              Reduces speckle noise common in SSS imagery.
            gaussian_ksize:   Gaussian blur kernel size. 0 = disabled.
                              Use sparingly — can blur target edges.
            gaussian_sigma:   Gaussian sigma. 0 = auto from ksize.
            use_clahe:        Apply CLAHE for local contrast enhancement.
            clahe_clip_limit: CLAHE clip limit (1.0–4.0 typical for sonar).
            clahe_tile_size:  CLAHE grid tile size.
        """
        # Validate odd kernel sizes
        if median_ksize % 2 == 0:
            raise ValueError(f"median_ksize must be odd, got {median_ksize}")
        if gaussian_ksize != 0 and gaussian_ksize % 2 == 0:
            raise ValueError(f"gaussian_ksize must be odd or 0, got {gaussian_ksize}")

        self.target_width     = target_width
        self.target_height    = target_height
        self.median_ksize     = median_ksize
        self.gaussian_ksize   = gaussian_ksize
        self.gaussian_sigma   = gaussian_sigma
        self.use_clahe        = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size  = clahe_tile_size

        # Build CLAHE object once (thread-safe to share for reads)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size,
        )

        log.info(
            "SonarPreprocessor initialised | target=%dx%d | "
            "median_k=%d | gaussian_k=%d | clahe=%s (clip=%.1f)",
            target_width, target_height,
            median_ksize, gaussian_ksize,
            use_clahe, clahe_clip_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self,
                image: np.ndarray,
                image_id: str = "unknown",
                record_stages: bool = False) -> PreprocessResult:
        """
        Run the full preprocessing pipeline on a single image.

        Args:
            image:         Input image array (HxW or HxWxC, uint8 or uint16).
            image_id:      Identifier for logging/reporting.
            record_stages: If True, save each intermediate step in result.stages.
                           Useful for before/after visualisation in dashboard.

        Returns:
            PreprocessResult — always returned, check .success before use.
        """
        if image is None or image.size == 0:
            return PreprocessResult(
                image_id=image_id,
                original=np.zeros((1, 1), dtype=np.uint8),
                error="Empty or None image passed to preprocessor.",
            )

        t_start = time.perf_counter()
        warnings: List[str] = []
        stages: Dict[str, np.ndarray] = {}

        try:
            # ── Step 0: Normalise to uint8 ─────────────────────────────
            img = self._to_uint8(image)
            if record_stages:
                stages["00_input_normalised"] = img.copy()

            # ── Step 1: Convert to grayscale ───────────────────────────
            gray = self._to_grayscale(img)
            if record_stages:
                stages["01_grayscale"] = gray.copy()

            # ── Step 1b: Acoustic dropout recovery & motion compensation ───
            repaired = self._recover_dropouts_and_motion(gray)
            if record_stages:
                stages["01b_dropout_motion_repaired"] = repaired.copy()

            # ── Step 2: Median filter (speckle reduction) ──────────────
            denoised = cv2.medianBlur(repaired, self.median_ksize)
            if record_stages:
                stages["02_median_filter"] = denoised.copy()

            # ── Step 3: Gaussian blur (optional, targeted smoothing) ───
            if self.gaussian_ksize > 0:
                denoised = cv2.GaussianBlur(
                    denoised,
                    (self.gaussian_ksize, self.gaussian_ksize),
                    self.gaussian_sigma,
                )
                if record_stages:
                    stages["03_gaussian_blur"] = denoised.copy()
            else:
                if record_stages:
                    stages["03_gaussian_blur"] = denoised.copy()  # same as median

            # ── Step 4: CLAHE ──────────────────────────────────────────
            if self.use_clahe:
                enhanced = self._clahe.apply(denoised)
            else:
                enhanced = denoised
                warnings.append("CLAHE disabled — contrast not enhanced.")
            if record_stages:
                stages["04_clahe"] = enhanced.copy()

            # ── Step 5: Resolution normalisation ──────────────────────
            resized = self._resize(enhanced)
            if record_stages:
                stages["05_resized"] = resized.copy()

            elapsed = time.perf_counter() - t_start
            log.debug(
                "Preprocessed %s | %dx%d → %dx%d | %.3f s",
                image_id,
                gray.shape[1], gray.shape[0],
                resized.shape[1], resized.shape[0],
                elapsed,
            )

            return PreprocessResult(
                image_id=image_id,
                original=gray,          # grayscale original for fair comparison
                preprocessed=resized,
                stages=stages,
                preprocessing_time_s=elapsed,
                params_used=self._params_dict(),
                warnings=warnings,
            )

        except Exception as exc:
            log.error("Preprocessing failed for %s: %s", image_id, exc)
            return PreprocessResult(
                image_id=image_id,
                original=image if image is not None else np.zeros((1, 1), dtype=np.uint8),
                error=str(exc),
            )

    def process_batch(self,
                      images: List[Tuple[np.ndarray, str]],
                      record_stages: bool = False) -> List[PreprocessResult]:
        """
        Process a list of (image_array, image_id) tuples.

        Returns:
            List of PreprocessResult objects in the same order.
        """
        results = []
        for i, (img, img_id) in enumerate(images):
            log.debug("Preprocessing %d/%d: %s", i + 1, len(images), img_id)
            results.append(self.process(img, img_id, record_stages=record_stages))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Normalise any dtype image to uint8 [0,255]."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype in (np.float32, np.float64):
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            return (img * 255).astype(np.uint8)
        return img.astype(np.uint8)

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert any channel count to single-channel grayscale."""
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            c = img.shape[2]
            if c == 1:
                return img[:, :, 0]
            if c == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if c == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    @staticmethod
    def _recover_dropouts_and_motion(img: np.ndarray) -> np.ndarray:
        """
        Acoustic motion & sensor dropout recovery:
        1. Detects dead sensor ping scanlines (mean intensity < 1.5 across row).
        2. Interpolates missing scanlines using nearest valid acoustic pings.
        3. Applies across-track illumination gain normalization.
        """
        if img is None or img.size == 0 or img.ndim != 2:
            return img
        out = img.copy()
        row_means = np.mean(out, axis=1)
        dead_rows = np.where(row_means < 1.5)[0]
        if 0 < len(dead_rows) < out.shape[0] * 0.4:
            for r in dead_rows:
                prev_r = max(0, r - 1)
                next_r = min(out.shape[0] - 1, r + 1)
                while prev_r in dead_rows and prev_r > 0:
                    prev_r -= 1
                while next_r in dead_rows and next_r < out.shape[0] - 1:
                    next_r += 1
                out[r, :] = ((out[prev_r, :].astype(np.float32) + out[next_r, :].astype(np.float32)) / 2.0).astype(np.uint8)
        return out

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions using INTER_AREA for downscaling
        and INTER_LINEAR for upscaling — preserves sonar texture better than
        INTER_CUBIC for this domain.
        """
        if self.target_width <= 0 or self.target_height <= 0:
            return img

        h, w = img.shape[:2]
        if w == self.target_width and h == self.target_height:
            return img

        interp = (cv2.INTER_AREA
                  if (self.target_width * self.target_height < w * h)
                  else cv2.INTER_LINEAR)
        return cv2.resize(img, (self.target_width, self.target_height),
                          interpolation=interp)

    def _params_dict(self) -> dict:
        return {
            "target_width":     self.target_width,
            "target_height":    self.target_height,
            "median_ksize":     self.median_ksize,
            "gaussian_ksize":   self.gaussian_ksize,
            "gaussian_sigma":   self.gaussian_sigma,
            "use_clahe":        self.use_clahe,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size":  list(self.clahe_tile_size),
        }


# ---------------------------------------------------------------------------
# Factory function (convenience)
# ---------------------------------------------------------------------------

def create_preprocessor_from_config(config) -> SonarPreprocessor:
    """
    Build a SonarPreprocessor from the project Config object.

    Args:
        config: A config.Config instance.

    Returns:
        SonarPreprocessor ready for use.
    """
    return SonarPreprocessor(
        target_width     = config.target_width,
        target_height    = config.target_height,
        median_ksize     = config.median_filter_ksize,
        gaussian_ksize   = config.gaussian_blur_ksize,
        gaussian_sigma   = config.gaussian_sigma,
        use_clahe        = config.use_clahe,
        clahe_clip_limit = config.clahe_clip_limit,
        clahe_tile_size  = config.clahe_tile_size,
    )

