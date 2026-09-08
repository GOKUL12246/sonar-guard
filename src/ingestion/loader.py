"""
SONAR-GUARD — Data Ingestion Module
=====================================
Loads sonar image files and associated mission metadata from disk.

Supported formats:
  Images : PNG, JPG/JPEG, TIFF, BMP
  Labels : YOLO .txt, COCO JSON, JSON/JSONL bounding boxes

Design contract:
  - Returns None / empty results rather than crashing when data is missing.
  - Caller is responsible for handling None returns gracefully.
  - Never generates fake data.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SonarImage:
    """
    Represents a single loaded sonar image with its metadata.

    Attributes:
        path:          Absolute path to the source image file.
        image_id:      Unique identifier derived from filename stem.
        data:          Loaded image as a NumPy array (HxW or HxWxC).
        width:         Image width in pixels.
        height:        Image height in pixels.
        channels:      Number of channels (1 = grayscale, 3 = colour).
        file_size_kb:  File size in kilobytes.
        load_error:    Non-None string if loading failed.
    """
    path:         Path
    image_id:     str
    data:         Optional[np.ndarray] = None
    width:        int = 0
    height:       int = 0
    channels:     int = 0
    file_size_kb: float = 0.0
    load_error:   Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.data is not None and self.load_error is None

    @property
    def shape_str(self) -> str:
        return f"{self.height}x{self.width}x{self.channels}"


@dataclass
class MissionMetadata:
    """
    Navigation / mission metadata associated with a sonar image or survey.

    All geolocation fields are Optional.
    If unavailable, the system will report "Location unavailable" rather
    than generating synthetic coordinates.

    Fields match common AUV/USV mission log formats.
    """
    image_id:        Optional[str]   = None
    latitude:        Optional[float] = None   # Decimal degrees, WGS-84
    longitude:       Optional[float] = None   # Decimal degrees, WGS-84
    altitude_m:      Optional[float] = None   # Vehicle altitude above seabed (m)
    depth_m:         Optional[float] = None   # Vehicle depth (m)
    heading_deg:     Optional[float] = None   # Vehicle heading (0–360°)
    speed_ms:        Optional[float] = None   # Vehicle speed (m/s)
    sonar_range_m:   Optional[float] = None   # Sonar range per side (m)
    sonar_frequency: Optional[float] = None   # Sonar frequency (Hz)
    timestamp_utc:   Optional[str]   = None   # ISO-8601 UTC timestamp
    survey_name:     Optional[str]   = None
    extra:           Dict[str, object] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def location_status(self) -> str:
        if self.has_position:
            return f"GPS ({self.latitude:.6f}°, {self.longitude:.6f}°)"
        return "Location unavailable — navigation metadata not provided."


# ---------------------------------------------------------------------------
# Image Loader
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image(path: Path) -> SonarImage:
    """
    Load a sonar image from disk.

    Args:
        path: Path to the image file.

    Returns:
        SonarImage — always returned; check .is_valid before use.
    """
    path = Path(path)
    image_id = path.stem

    if not path.exists():
        log.warning("Image not found: %s", path)
        return SonarImage(path=path, image_id=image_id,
                          load_error=f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.warning("Unsupported format '%s': %s", path.suffix, path)
        return SonarImage(path=path, image_id=image_id,
                          load_error=f"Unsupported format: {path.suffix}")

    file_size_kb = path.stat().st_size / 1024.0

    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imread returned None — file may be corrupted.")

        # Normalise to 8-bit if 16-bit TIFF
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        # Ensure 3-channel (grayscale → BGR for pipeline consistency)
        if img.ndim == 2:
            channels = 1
        elif img.ndim == 3:
            channels = img.shape[2]
        else:
            raise ValueError(f"Unexpected image ndim: {img.ndim}")

        h, w = img.shape[:2]
        log.debug("Loaded %s | %dx%d | %d-ch | %.1f KB",
                  path.name, w, h, channels, file_size_kb)

        return SonarImage(
            path=path,
            image_id=image_id,
            data=img,
            width=w,
            height=h,
            channels=channels,
            file_size_kb=file_size_kb,
        )

    except Exception as exc:
        log.error("Failed to load %s: %s", path, exc)
        return SonarImage(path=path, image_id=image_id,
                          load_error=str(exc),
                          file_size_kb=file_size_kb)


def load_image_directory(directory: Path,
                         recursive: bool = False) -> List[SonarImage]:
    """
    Load all supported images from a directory.

    Args:
        directory:  Directory to scan.
        recursive:  If True, scan subdirectories too.

    Returns:
        List of SonarImage objects (may include invalid ones).
    """
    directory = Path(directory)
    if not directory.exists():
        log.warning("Directory not found: %s", directory)
        return []

    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern)
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        log.warning("No supported images found in: %s", directory)
        return []

    log.info("Loading %d images from %s", len(files), directory)
    images = [load_image(p) for p in sorted(files)]

    valid   = sum(1 for img in images if img.is_valid)
    invalid = len(images) - valid
    log.info("Loaded: %d valid, %d invalid/corrupted", valid, invalid)

    return images


# ---------------------------------------------------------------------------
# Metadata Loader
# ---------------------------------------------------------------------------

def load_mission_metadata(metadata_source) -> Optional[MissionMetadata]:
    """
    Load mission/navigation metadata from various sources.

    Supported input types:
      - dict:  Pre-parsed metadata dictionary
      - str/Path: Path to JSON file containing metadata

    Returns:
        MissionMetadata if successfully parsed, None otherwise.
        Never raises — returns None on any failure.
    """
    if metadata_source is None:
        return None

    if isinstance(metadata_source, dict):
        return _parse_metadata_dict(metadata_source)

    path = Path(metadata_source)
    if not path.exists():
        log.warning("Metadata file not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_metadata_dict(data)
    except json.JSONDecodeError as e:
        log.error("Invalid JSON in metadata file %s: %s", path, e)
        return None
    except Exception as e:
        log.error("Failed to load metadata from %s: %s", path, e)
        return None


def _parse_metadata_dict(data: dict) -> MissionMetadata:
    """
    Parse a metadata dictionary into a MissionMetadata object.
    Unknown/extra keys are stored in .extra — nothing is discarded.
    """
    known_keys = {
        "image_id", "latitude", "longitude", "altitude_m", "depth_m",
        "heading_deg", "speed_ms", "sonar_range_m", "sonar_frequency",
        "timestamp_utc", "survey_name",
    }

    def _safe_float(key: str) -> Optional[float]:
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            log.warning("Could not parse metadata field '%s' = %r as float", key, val)
            return None

    meta = MissionMetadata(
        image_id        = data.get("image_id"),
        latitude        = _safe_float("latitude"),
        longitude       = _safe_float("longitude"),
        altitude_m      = _safe_float("altitude_m"),
        depth_m         = _safe_float("depth_m"),
        heading_deg     = _safe_float("heading_deg"),
        speed_ms        = _safe_float("speed_ms"),
        sonar_range_m   = _safe_float("sonar_range_m"),
        sonar_frequency = _safe_float("sonar_frequency"),
        timestamp_utc   = data.get("timestamp_utc"),
        survey_name     = data.get("survey_name"),
        extra           = {k: v for k, v in data.items() if k not in known_keys},
    )

    if meta.has_position:
        log.debug("Metadata loaded: %s", meta.location_status)
    else:
        log.debug("Metadata loaded — no GPS position available")

    return meta


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_batch(image_paths: List[Path]) -> List[SonarImage]:
    """
    Load a batch of images by explicit path list.

    Args:
        image_paths: List of file paths to load.

    Returns:
        List of SonarImage objects.
    """
    return [load_image(p) for p in image_paths]

