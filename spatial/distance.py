"""
Relative distance estimation module for VisionMate.

Calculates relative proximity categories (far, medium, close, very_close)
and relative size ratios based on 2D bounding-box areas within camera frame coordinates.

Note on Limitations:
Monocular cameras do not directly provide 3D physical distance in meters/feet.
Apparent 2D bounding-box size depends both on camera distance and the actual physical
size of the detected object (e.g., a nearby handheld phone vs. a distant person).
This module provides a relative proximity score and category, not physical depth.
"""

from typing import Dict, Any, Tuple


class DistanceEstimator:
    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        far_threshold: float = 0.02,
        medium_threshold: float = 0.08,
        close_threshold: float = 0.20
    ):
        """
        Initialize the DistanceEstimator.

        Args:
            frame_width (int): Width of the camera frame in pixels (default 1280).
            frame_height (int): Height of the camera frame in pixels (default 720).
            far_threshold (float): Relative area ratio boundary for 'far' category (default 0.02).
            medium_threshold (float): Relative area ratio boundary for 'medium' category (default 0.08).
            close_threshold (float): Relative area ratio boundary for 'close' category (default 0.20).
        """
        self.far_threshold = far_threshold
        self.medium_threshold = medium_threshold
        self.close_threshold = close_threshold

        self.set_frame_size(frame_width, frame_height)

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Update the frame dimensions dynamically.

        Args:
            width (int): Frame width in pixels (> 0).
            height (int): Frame height in pixels (> 0).
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame dimensions: width={width}, height={height}. Dimensions must be positive.")
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.frame_area = self.frame_width * self.frame_height

    def _validate_and_normalize_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Validate and clamp bounding box coordinates to valid frame dimensions.

        Args:
            bbox: Tuple of (x1, y1, x2, y2).

        Returns:
            Tuple[int, int, int, int]: Normalized bounding box coordinates.
        """
        if bbox is None or not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected a 4-element tuple (x1, y1, x2, y2).")

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Bbox coordinates must be numeric integers: {bbox}") from e

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bounding box boundaries: x1 ({x1}) >= x2 ({x2}) or y1 ({y1}) >= y2 ({y2}).")

        x1_clamped = max(0, min(self.frame_width, x1))
        y1_clamped = max(0, min(self.frame_height, y1))
        x2_clamped = max(0, min(self.frame_width, x2))
        y2_clamped = max(0, min(self.frame_height, y2))

        return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)

    def get_bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Calculate the pixel area (width x height) of a bounding box.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            int: Bounding box area in square pixels.
        """
        x1, y1, x2, y2 = self._validate_and_normalize_bbox(bbox)
        width = x2 - x1
        height = y2 - y1
        return width * height

    def get_relative_size(self, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Calculate the ratio of the bounding box area relative to the total frame area.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            dict: {
                "bbox_area": int,
                "frame_area": int,
                "relative_size": float (0.0 to 1.0)
            }
        """
        bbox_area = self.get_bbox_area(bbox)
        relative_size = round(bbox_area / self.frame_area, 4)
        return {
            "bbox_area": bbox_area,
            "frame_area": self.frame_area,
            "relative_size": relative_size
        }

    def estimate_distance(self, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Estimate relative distance category and proximity score based on relative bounding box area.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            dict: {
                "category": "far" | "medium" | "close" | "very_close",
                "relative_size": float,
                "proximity_score": float (0.0 to 1.0)
            }
        """
        size_info = self.get_relative_size(bbox)
        relative_size = size_info["relative_size"]

        # Category classification based on configurable relative size thresholds
        if relative_size < self.far_threshold:
            category = "far"
        elif relative_size < self.medium_threshold:
            category = "medium"
        elif relative_size < self.close_threshold:
            category = "close"
        else:
            category = "very_close"

        # Proximity score normalized from 0.0 (negligible size) to 1.0 (covers 50%+ of frame)
        proximity_score = round(min(1.0, relative_size / 0.50), 4)

        return {
            "category": category,
            "relative_size": relative_size,
            "proximity_score": proximity_score
        }

    def analyze_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a tracked object dictionary with distance/proximity metadata.

        Args:
            track (dict): Tracked object dictionary containing a "bbox" key.

        Returns:
            dict: Enriched dictionary containing original track data plus a "distance" dict.
        """
        if not isinstance(track, dict) or "bbox" not in track:
            raise ValueError("Input track must be a dictionary containing a 'bbox' key.")

        distance_info = self.estimate_distance(track["bbox"])

        enriched_track = dict(track)
        enriched_track["distance"] = distance_info
        return enriched_track
