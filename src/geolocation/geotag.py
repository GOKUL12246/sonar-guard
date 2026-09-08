"""
SONAR-GUARD — Geolocation Module
===================================
Estimates target geographic coordinates from mission metadata and sonar geometry.

CRITICAL DESIGN PRINCIPLE:
  This module NEVER generates fake GPS coordinates.
  If navigation metadata is unavailable, it returns:
    latitude  = null
    longitude = null
    status    = "Location unavailable — navigation metadata not provided."

When metadata IS available:
  Uses sonar geometry to estimate target position from:
  - Vehicle latitude/longitude
  - Vehicle heading
  - Sonar range (metres per side)
  - Pixel position of target within image (normalised across-track offset)

The estimated position is an approximation — accuracy depends on
navigation metadata quality. This is documented clearly.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.ingestion.loader import MissionMetadata
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GeoLocation:
    """
    Geographic location estimate for a detected target.

    Attributes:
        latitude:   Decimal degrees WGS-84 (None if unavailable).
        longitude:  Decimal degrees WGS-84 (None if unavailable).
        status:     Human-readable location status.
        method:     How the location was derived.
        accuracy_note: Honest description of estimation accuracy.
    """
    latitude:      Optional[float] = None
    longitude:     Optional[float] = None
    status:        str = "Location unavailable — navigation metadata not provided."
    method:        str = "none"
    accuracy_note: str = ""

    @property
    def available(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> dict:
        return {
            "latitude":      self.latitude,
            "longitude":     self.longitude,
            "status":        self.status,
            "method":        self.method,
            "accuracy_note": self.accuracy_note,
        }


class GeoTagger:
    """
    Estimates target geolocation from mission navigation metadata.

    If metadata is None or position is unavailable:
      Returns GeoLocation with latitude=None, longitude=None.

    If metadata has position and sonar geometry parameters:
      Projects the target's across-track pixel offset into
      a geographic displacement using the sonar range and vehicle heading.

    NOTE: This is an approximation. Real-world accuracy requires:
    - Accurate vehicle position (GPS quality)
    - Calibrated sonar range
    - Accurate heading
    - Known sonar orientation (port/starboard)
    """

    def estimate(
        self,
        metadata:          Optional[MissionMetadata],
        target_bbox_cx:    Optional[float] = None,   # normalised [0,1] across-track
        image_width:       int = 0,
    ) -> GeoLocation:
        """
        Estimate target geolocation.

        Args:
            metadata:         Mission/navigation metadata.
            target_bbox_cx:   Target centre-x normalised to [0,1] in image.
                              0 = left edge, 1 = right edge.
                              Used to estimate across-track offset.
            image_width:      Image width in pixels (for context).

        Returns:
            GeoLocation — always returned, never raises.
        """
        # No metadata at all
        if metadata is None:
            log.debug("No mission metadata — location unavailable.")
            return GeoLocation()

        # Metadata exists but no position fix
        if not metadata.has_position:
            log.debug("Metadata present but no GPS position — location unavailable.")
            return GeoLocation(
                status="Location unavailable — GPS position not in mission metadata.",
                method="metadata_no_position",
            )

        lat0 = metadata.latitude
        lon0 = metadata.longitude

        # Vehicle position only — no sonar geometry for projection
        if metadata.sonar_range_m is None or metadata.heading_deg is None:
            log.info(
                "Vehicle position available but sonar geometry missing — "
                "using vehicle position as approximate target location."
            )
            return GeoLocation(
                latitude=lat0,
                longitude=lon0,
                status=f"Approximate: vehicle position ({lat0:.6f}°, {lon0:.6f}°) "
                       f"— sonar range/heading not provided for offset estimation.",
                method="vehicle_position_only",
                accuracy_note=(
                    "WARNING: This is the vehicle position, not a computed target position. "
                    "Target may be up to one sonar range (unknown) away. "
                    "Provide sonar_range_m and heading_deg for better accuracy."
                ),
            )

        # Full geometry available — estimate across-track offset
        sonar_range_m = metadata.sonar_range_m
        heading_deg   = metadata.heading_deg

        # Across-track offset: 0.5 = nadir (directly below), 0 or 1 = full range
        if target_bbox_cx is not None:
            # Standard SSS: left half = port side, right half = starboard
            # offset_fraction: 0 = nadir, 1 = full range
            offset_fraction = abs(target_bbox_cx - 0.5) * 2.0
            across_track_m  = offset_fraction * sonar_range_m
            # Determine which side (port = left, starboard = right)
            if target_bbox_cx < 0.5:
                side_heading = (heading_deg - 90) % 360   # port
            else:
                side_heading = (heading_deg + 90) % 360   # starboard
        else:
            # No target position — use nadir
            across_track_m = 0.0
            side_heading   = heading_deg
            log.debug("Target x position unknown — using nadir as location.")

        # Convert metres to degrees using approximate Earth radius
        R_EARTH_M = 6_371_000.0
        d_lat = (across_track_m * math.cos(math.radians(side_heading))) / R_EARTH_M
        d_lon = (across_track_m * math.sin(math.radians(side_heading))) / (
            R_EARTH_M * math.cos(math.radians(lat0))
        )

        est_lat = lat0 + math.degrees(d_lat)
        est_lon = lon0 + math.degrees(d_lon)

        log.info(
            "Geolocation estimated: (%.6f, %.6f) | "
            "offset=%.1f m from vehicle | heading=%.1f°",
            est_lat, est_lon, across_track_m, heading_deg,
        )

        return GeoLocation(
            latitude=round(est_lat, 7),
            longitude=round(est_lon, 7),
            status=f"Estimated: ({est_lat:.6f}°, {est_lon:.6f}°)",
            method="sonar_geometry_projection",
            accuracy_note=(
                f"Estimated from vehicle position + sonar geometry. "
                f"Across-track offset: {across_track_m:.1f} m. "
                f"Accuracy depends on navigation data quality. "
                f"Not suitable for precision navigation without independent verification."
            ),
        )

