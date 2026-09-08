"""
SONAR-GUARD — YOLOv8 Detector
================================
Runs YOLO inference on a preprocessed sonar image and returns
structured DetectionResult objects.

Design:
  - Model loaded once; reused across calls (stateful detector).
  - Returns empty list (not crash) when model unavailable.
  - All confidence/IoU thresholds are runtime-configurable.
  - Measures and reports actual inference time per image.
  - Does NOT interpret YOLO confidence as a probability of real-world truth.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float   # left
    y1: float   # top
    x2: float   # right
    y2: float   # bottom

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> list:
        return [round(self.x1, 2), round(self.y1, 2),
                round(self.x2, 2), round(self.y2, 2)]

    def to_yolo_norm(self, img_w: int, img_h: int) -> tuple:
        """Return (cx, cy, w, h) normalised to [0,1]."""
        cx = (self.x1 + self.x2) / (2 * img_w)
        cy = (self.y1 + self.y2) / (2 * img_h)
        w  = self.width  / img_w
        h  = self.height / img_h
        return (cx, cy, w, h)


@dataclass
class Detection:
    """
    Single object detection result.

    Attributes:
        detection_id:     Unique ID within an image (0-indexed).
        class_id:         Integer class index from YOLO.
        class_name:       Human-readable class label.
        confidence:       YOLO detection confidence score (0–1).
                          NOTE: This is model confidence, NOT a probability
                          that the object is genuinely present or hazardous.
        bbox:             Bounding box in pixel coordinates.
        image_id:         Source image identifier.
        image_width:      Source image width.
        image_height:     Source image height.
        mask:             Binary segmentation mask (HxW uint8). None if not available.
    """
    detection_id:  int
    class_id:      int
    class_name:    str
    confidence:    float
    bbox:          BoundingBox
    image_id:      str = ""
    image_width:   int = 0
    image_height:  int = 0
    mask:          Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "class_id":     self.class_id,
            "class_name":   self.class_name,
            "confidence":   round(float(self.confidence), 4),
            "bbox":         self.bbox.to_list(),
            "image_id":     self.image_id,
            "image_width":  self.image_width,
            "image_height": self.image_height,
        }


@dataclass
class DetectionResult:
    """
    Collection of detections from one inference run on a single image.
    """
    image_id:         str
    detections:       List[Detection] = field(default_factory=list)
    inference_time_s: float = 0.0
    model_path:       str   = ""
    confidence_thresh: float = 0.25
    iou_thresh:        float = 0.45
    image_width:       int   = 0
    image_height:      int   = 0
    error:             Optional[str] = None

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "image_id":          self.image_id,
            "num_detections":    self.num_detections,
            "inference_time_s":  round(self.inference_time_s, 4),
            "model_path":        self.model_path,
            "confidence_thresh": self.confidence_thresh,
            "iou_thresh":        self.iou_thresh,
            "image_width":       self.image_width,
            "image_height":      self.image_height,
            "detections":        [d.to_dict() for d in self.detections],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    YOLOv8 inference wrapper for sonar imagery.

    The model is loaded once and reused for all subsequent calls.
    If the model file is unavailable, is_ready returns False and
    detect() returns an empty DetectionResult — no crash.

    Usage:
        detector = YOLODetector(model_path="models/yolo/best.pt")
        result = detector.detect(image_array, image_id="frame_001")
    """

    def __init__(
        self,
        model_path:        Optional[str] = None,
        confidence_thresh: float = 0.25,
        iou_thresh:        float = 0.45,
        device:            str   = "cpu",
        class_names:       Optional[List[str]] = None,
    ):
        """
        Args:
            model_path:        Path to trained .pt file. If None or missing,
                               falls back to yolov8n pretrained (COCO classes).
            confidence_thresh: Minimum confidence to include a detection.
            iou_thresh:        NMS IoU threshold.
            device:            "cuda" or "cpu".
            class_names:       Override class names (from dataset). If None,
                               uses model's built-in names.
        """
        self.confidence_thresh = confidence_thresh
        self.iou_thresh        = iou_thresh
        self.device            = device
        self.class_names       = class_names
        self.model_path        = model_path
        self._model            = None
        self._model_class_names = []
        self._load_error       = None

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the YOLO model. Failures are logged, not raised."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics not installed."
            log.error(self._load_error)
            return

        # Resolve model path
        if self.model_path and Path(self.model_path).exists():
            mp = self.model_path
            log.info("Loading trained SONAR-GUARD model: %s", mp)
        else:
            # Fallback to pretrained nano — useful for pipeline testing
            # without a trained sonar model
            mp = "yolov8n.pt"
            status = ("not provided" if not self.model_path
                      else f"not found at {self.model_path}")
            log.warning(
                "Trained sonar model %s — using pretrained YOLOv8n (COCO classes). "
                "Detection results on sonar imagery will be UNRELIABLE until a "
                "sonar-specific model is trained.",
                status,
            )

        try:
            self._model = YOLO(mp)
            # Extract class names from model
            self._model_class_names = list(self._model.names.values()) \
                if hasattr(self._model, "names") else []
            log.info("Model loaded. Classes: %s", self._model_class_names[:10])
        except Exception as exc:
            self._load_error = str(exc)
            log.error("Failed to load YOLO model (%s): %s", mp, exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._load_error is None

    @property
    def effective_class_names(self) -> List[str]:
        if self.class_names:
            return self.class_names
        return self._model_class_names

    def detect(
        self,
        image:      np.ndarray,
        image_id:   str = "unknown",
    ) -> DetectionResult:
        """
        Run YOLO inference on a single image.

        Args:
            image:    Preprocessed grayscale (HxW) or BGR (HxWx3) uint8 array.
            image_id: Identifier for logging and report.

        Returns:
            DetectionResult — always returned, never raises.
        """
        h, w = image.shape[:2]
        base_result = DetectionResult(
            image_id          = image_id,
            confidence_thresh = self.confidence_thresh,
            iou_thresh        = self.iou_thresh,
            image_width       = w,
            image_height      = h,
            model_path        = str(self.model_path or "yolov8n.pt (fallback)"),
        )

        if not self.is_ready:
            base_result.error = (
                f"Model not loaded: {self._load_error or 'unknown error'}. "
                "Please train a model first or provide a valid model path."
            )
            log.warning("Detection skipped for %s — model not ready.", image_id)
            return base_result

        # Convert grayscale → 3-channel (YOLO expects BGR or RGB)
        import cv2
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image

        try:
            t_start = time.perf_counter()
            results = self._model.predict(
                source  = img_bgr,
                conf    = self.confidence_thresh,
                iou     = self.iou_thresh,
                device  = self.device,
                verbose = False,
            )
            inf_time = time.perf_counter() - t_start

            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for det_id, box in enumerate(boxes):
                        cls_id = int(box.cls[0].item())
                        conf   = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        cls_names = self.effective_class_names
                        cls_name  = (cls_names[cls_id]
                                     if 0 <= cls_id < len(cls_names)
                                     else f"class_{cls_id}")

                        detections.append(Detection(
                            detection_id = det_id,
                            class_id     = cls_id,
                            class_name   = cls_name,
                            confidence   = conf,
                            bbox         = BoundingBox(x1, y1, x2, y2),
                            image_id     = image_id,
                            image_width  = w,
                            image_height = h,
                        ))

            log.info(
                "Detected %d objects in %s (%.3f s)",
                len(detections), image_id, inf_time,
            )

            base_result.detections       = detections
            base_result.inference_time_s = inf_time
            return base_result

        except Exception as exc:
            log.error("YOLO inference failed for %s: %s", image_id, exc)
            base_result.error = str(exc)
            return base_result

