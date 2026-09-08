"""
SONAR-GUARD — Pluggable Segmentation Interface
================================================
Provides pixel-level segmentation for YOLO-detected candidates.

Architecture:
  - Abstract base class defines the interface.
  - ContoursSegmenter: sonar-appropriate fallback using OpenCV contours.
    Works without SAM. Reliable on CPU.
  - SAMSegmenter:      Segment Anything Model wrapper.
    Optional — requires separate install + sufficient RAM.
    Clearly documented when unavailable.

STATUS OF SAM:
  SAM is NOT installed in the current environment.
  Memory requirement: ~2–6 GB GPU VRAM for vit_b.
  Current system: CPU only, ~3 GB free RAM.
  Recommendation: Use ContoursSegmenter for prototype.
  Enable SAM when GPU + sufficient RAM is available.
  See README.md → Segmentation section.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from src.detection.yolo_detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """
    Pixel-level segmentation result for one detected object.

    Attributes:
        detection_id:       Links back to the Detection object.
        mask:               Binary mask (HxW, uint8, 0 or 255).
                            None if segmentation unavailable.
        mask_area_px:       Number of foreground pixels.
        bbox_area_px:       Area of the YOLO bounding box in pixels.
        mask_to_box_ratio:  mask_area / bbox_area (0–1). 1.0 = box fully filled.
        contours:           List of OpenCV contours (for shape analysis).
        seg_time_s:         Wall-clock segmentation time.
        method:             "contours" | "sam" | "unavailable"
        status:             Human-readable status message.
    """
    detection_id:     int
    mask:             Optional[np.ndarray] = None
    mask_area_px:     int   = 0
    bbox_area_px:     int   = 0
    mask_to_box_ratio: float = 0.0
    contours:         list  = field(default_factory=list)
    seg_time_s:       float = 0.0
    method:           str   = "unavailable"
    status:           str   = "Not run"

    @property
    def available(self) -> bool:
        return self.mask is not None and self.method != "unavailable"

    def to_dict(self) -> dict:
        return {
            "detection_id":     self.detection_id,
            "mask_available":   self.available,
            "mask_area_px":     self.mask_area_px,
            "bbox_area_px":     self.bbox_area_px,
            "mask_to_box_ratio": round(self.mask_to_box_ratio, 4),
            "seg_time_s":       round(self.seg_time_s, 4),
            "method":           self.method,
            "status":           self.status,
        }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseSegmenter(ABC):
    """
    Abstract segmenter interface.
    All segmenters must implement segment().
    """

    @abstractmethod
    def segment(
        self,
        image:     np.ndarray,
        detection: Detection,
    ) -> SegmentationResult:
        """
        Segment the region corresponding to `detection` in `image`.

        Args:
            image:     Full grayscale (HxW) or BGR (HxWx3) image.
            detection: Detection object containing bounding box.

        Returns:
            SegmentationResult — always returned, never raises.
        """

    def segment_batch(
        self,
        image:      np.ndarray,
        detections: List[Detection],
    ) -> List[SegmentationResult]:
        """Segment all detections in an image."""
        return [self.segment(image, det) for det in detections]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable segmenter name."""


# ---------------------------------------------------------------------------
# Contour-based segmenter (sonar-appropriate fallback)
# ---------------------------------------------------------------------------

class ContoursSegmenter(BaseSegmenter):
    """
    Sonar-appropriate segmentation using OpenCV adaptive thresholding
    + contour analysis within the YOLO bounding box.

    Rationale:
      Side-scan sonar targets typically appear as bright/dark regions
      with relatively clear boundaries against the seabed. Adaptive
      thresholding inside the YOLO crop captures the foreground object
      without requiring GPU or large models.

    Limitations:
      - May over-segment or under-segment objects with ambiguous contrast.
      - Not as precise as SAM for complex shapes.
      - Results should be used as evidence features, not ground truth masks.
    """

    def __init__(
        self,
        block_size:   int   = 15,   # Adaptive threshold block size (odd)
        c_constant:   int   = 3,    # Subtracted constant for adaptive threshold
        morph_ksize:  int   = 3,    # Morphological cleanup kernel size
        min_area_px:  int   = 20,   # Minimum contour area (ignore noise)
    ):
        if block_size % 2 == 0:
            raise ValueError(f"block_size must be odd, got {block_size}")
        self.block_size  = block_size
        self.c_constant  = c_constant
        self.morph_ksize = morph_ksize
        self.min_area_px = min_area_px
        log.info("ContoursSegmenter initialised (sonar fallback — no SAM)")

    @property
    def name(self) -> str:
        return "OpenCV Adaptive Threshold + Contours (sonar fallback)"

    def segment(
        self,
        image:     np.ndarray,
        detection: Detection,
    ) -> SegmentationResult:
        """
        Segment using adaptive thresholding within the YOLO bounding box.
        """
        t_start = time.perf_counter()

        bbox = detection.bbox
        h, w = image.shape[:2]

        # Clamp bbox to image bounds
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w - 1, int(bbox.x2))
        y2 = min(h - 1, int(bbox.y2))
        bbox_area = max(1, (x2 - x1) * (y2 - y1))

        if x2 <= x1 or y2 <= y1:
            return SegmentationResult(
                detection_id=detection.detection_id,
                bbox_area_px=0,
                method="contours",
                status="Invalid bounding box — zero area.",
                seg_time_s=time.perf_counter() - t_start,
            )

        # Extract and convert crop to grayscale
        crop = image[y1:y2, x1:x2]
        if crop.ndim == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop.copy()

        # Adaptive thresholding
        if crop_gray.shape[0] < self.block_size or crop_gray.shape[1] < self.block_size:
            # Too small for adaptive threshold — use Otsu
            _, thresh = cv2.threshold(
                crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            thresh = cv2.adaptiveThreshold(
                crop_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                self.c_constant,
            )

        # Morphological cleanup — remove tiny noise blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Keep only significant contours
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area_px]

        # Build full-image mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Offset contours back to full-image coordinates
            offset_contours = [c + np.array([[[x1, y1]]]) for c in contours]
            cv2.drawContours(full_mask, offset_contours, -1, 255, thickness=cv2.FILLED)

        mask_area = int(np.sum(full_mask > 0))
        ratio = mask_area / bbox_area if bbox_area > 0 else 0.0

        elapsed = time.perf_counter() - t_start
        log.debug(
            "Segmented det-%d | mask_area=%d px | ratio=%.2f | %.3f s",
            detection.detection_id, mask_area, ratio, elapsed,
        )

        return SegmentationResult(
            detection_id=detection.detection_id,
            mask=full_mask,
            mask_area_px=mask_area,
            bbox_area_px=bbox_area,
            mask_to_box_ratio=min(1.0, ratio),
            contours=contours,
            seg_time_s=elapsed,
            method="contours",
            status="OK — OpenCV adaptive threshold segmentation",
        )


# ---------------------------------------------------------------------------
# SAM wrapper (optional — documented as unavailable)
# ---------------------------------------------------------------------------

class SAMSegmenter(BaseSegmenter):
    """
    Segment Anything Model (SAM) wrapper.

    STATUS: SAM is NOT installed in the current environment.
    Install with: pip install segment-anything
    Also requires the SAM checkpoint file (sam_vit_b_01ec64.pth, ~375 MB).

    Hardware requirements:
      - Minimum 4 GB GPU VRAM for vit_b variant.
      - CPU inference is possible but very slow (~30–60 s per image).

    When SAM becomes available, set segmenter = SAMSegmenter(checkpoint_path=...)
    The rest of the pipeline will work without any other changes.
    """

    SAM_STATUS = (
        "SAM (Segment Anything Model) is not installed. "
        "Install with: pip install segment-anything. "
        "Download checkpoint from: https://github.com/facebookresearch/segment-anything. "
        "Current hardware (CPU only, ~3 GB RAM) is marginal for SAM inference. "
        "Using ContoursSegmenter as sonar-appropriate fallback."
    )

    def __init__(self, checkpoint_path: Optional[str] = None, model_type: str = "vit_b", device: str = "cpu"):
        self._available = False
        self._model     = None
        self._predictor = None
        self.checkpoint_path = checkpoint_path
        self.model_type      = model_type
        self.device          = device

        self._try_load()

    def _try_load(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                sam.to(device=self.device)
                self._predictor = SamPredictor(sam)
                self._available = True
                log.info("SAM loaded: %s (%s)", self.model_type, self.device)
            else:
                log.warning("SAM checkpoint not found: %s", self.checkpoint_path)
        except ImportError:
            log.warning("segment-anything not installed. %s", self.SAM_STATUS)
        except Exception as exc:
            log.warning("SAM load failed: %s", exc)

    @property
    def name(self) -> str:
        status = "available" if self._available else "unavailable"
        return f"SAM ({self.model_type}) — {status}"

    def segment(self, image: np.ndarray, detection: Detection) -> SegmentationResult:
        if not self._available:
            return SegmentationResult(
                detection_id=detection.detection_id,
                method="unavailable",
                status=self.SAM_STATUS,
            )

        t_start = time.perf_counter()
        try:
            # Convert grayscale → RGB for SAM
            if image.ndim == 2:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image

            self._predictor.set_image(img_rgb)
            bbox = detection.bbox
            input_box = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            mask = (masks[0] * 255).astype(np.uint8)
            mask_area  = int(np.sum(mask > 0))
            bbox_area  = max(1, int(detection.bbox.area))
            ratio      = mask_area / bbox_area
            elapsed    = time.perf_counter() - t_start

            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=mask,
                mask_area_px=mask_area,
                bbox_area_px=bbox_area,
                mask_to_box_ratio=min(1.0, ratio),
                seg_time_s=elapsed,
                method="sam",
                status=f"SAM ({self.model_type}) — confidence {float(scores[0]):.3f}",
            )
        except Exception as exc:
            log.error("SAM inference failed: %s", exc)
            return SegmentationResult(
                detection_id=detection.detection_id,
                method="sam",
                status=f"SAM inference error: {exc}",
                seg_time_s=time.perf_counter() - t_start,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_segmenter(
    use_sam:         bool  = False,
    sam_checkpoint:  Optional[str] = None,
    sam_model_type:  str   = "vit_b",
    device:          str   = "cpu",
) -> BaseSegmenter:
    """
    Create the appropriate segmenter based on configuration.

    Args:
        use_sam:        If True, attempt to use SAM. Falls back to contours.
        sam_checkpoint: Path to SAM checkpoint file.
        sam_model_type: SAM model variant.
        device:         "cuda" or "cpu".

    Returns:
        A BaseSegmenter implementation.
    """
    if use_sam:
        sam = SAMSegmenter(
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
            device=device,
        )
        if sam._available:
            return sam
        log.warning("SAM unavailable — using ContoursSegmenter fallback.")

    return ContoursSegmenter()

