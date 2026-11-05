"""
Position analysis module for VisionMate.

Calculates relative 2D spatial positions (horizontal/vertical region and center coordinates)
for detected objects within camera frame coordinates.
"""

from typing import Dict, Any, Tuple


class ObjectPositionAnalyzer:
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """
        Initialize the ObjectPositionAnalyzer.

        Args:
            frame_width (int): Width of the camera frame in pixels (default 1280).
            frame_height (int): Height of the camera frame in pixels (default 720).
        """
        self.set_frame_size(frame_width, frame_height)

        # Horizontal region boundaries (as fractions of frame_width)
        # Left: 0.0 - 0.33, Center: 0.33 - 0.66, Right: 0.66 - 1.0
        self.horiz_left_thresh = 0.33
        self.horiz_right_thresh = 0.66

        # Vertical region boundaries (as fractions of frame_height)
        # Top: 0.0 - 0.33, Middle: 0.33 - 0.66, Bottom: 0.66 - 1.0
        self.vert_top_thresh = 0.33
        self.vert_bottom_thresh = 0.66

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Update the frame dimensions dynamically.

        Args:
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame dimensions: width={width}, height={height}. Dimensions must be positive.")
        self.frame_width = int(width)
        self.frame_height = int(height)

    def _validate_and_normalize_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Validate and clamp bounding box coordinates to valid frame dimensions.

        Args:
            bbox: Tuple of (x1, y1, x2, y2).

        Returns:
            Tuple[int, int, int, int]: Normalized and clamped bounding box coordinates.
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

    def get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate the center point (center_x, center_y) of a bounding box.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            Tuple[int, int]: (center_x, center_y) in pixels.
        """
        x1, y1, x2, y2 = self._validate_and_normalize_bbox(bbox)
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        return (center_x, center_y)

    def get_position(self, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Determine horizontal, vertical, and center point position in the 2D frame.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            dict: {
                "horizontal": "left" | "center" | "right",
                "vertical": "top" | "middle" | "bottom",
                "center": (center_x, center_y)
            }
        """
        center_x, center_y = self.get_center(bbox)

        norm_x = center_x / self.frame_width
        norm_y = center_y / self.frame_height

        # Horizontal region
        if norm_x < self.horiz_left_thresh:
            horizontal = "left"
        elif norm_x > self.horiz_right_thresh:
            horizontal = "right"
        else:
            horizontal = "center"

        # Vertical region
        if norm_y < self.vert_top_thresh:
            vertical = "top"
        elif norm_y > self.vert_bottom_thresh:
            vertical = "bottom"
        else:
            vertical = "middle"

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "center": (center_x, center_y)
        }

    def describe_position(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Generate a natural-language description suitable for text-to-speech responses.

        Args:
            bbox: (x1, y1, x2, y2) tuple.

        Returns:
            str: Descriptive string such as "left", "slightly left", "center", "upper right", "lower center", etc.
        """
        center_x, center_y = self.get_center(bbox)
        norm_x = center_x / self.frame_width
        norm_y = center_y / self.frame_height

        # Fine-grained horizontal description
        if norm_x < 0.15:
            h_desc = "far left"
        elif norm_x < 0.35:
            h_desc = "left"
        elif norm_x < 0.45:
            h_desc = "slightly left"
        elif norm_x <= 0.55:
            h_desc = "center"
        elif norm_x <= 0.65:
            h_desc = "slightly right"
        elif norm_x <= 0.85:
            h_desc = "right"
        else:
            h_desc = "far right"

        # Fine-grained vertical modifier
        if norm_y < 0.33:
            v_prefix = "upper"
        elif norm_y > 0.66:
            v_prefix = "lower"
        else:
            v_prefix = ""

        if v_prefix:
            if h_desc == "center":
                return f"{v_prefix} center"
            else:
                return f"{v_prefix} {h_desc}"
        else:
            return h_desc

    def analyze_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a tracked object dictionary with spatial position data.

        Args:
            track (dict): Tracked object dictionary containing a "bbox" key.

        Returns:
            dict: Enriched dictionary containing original track data plus a "position" dict.
        """
        if not isinstance(track, dict) or "bbox" not in track:
            raise ValueError("Input track must be a dictionary containing a 'bbox' key.")

        position_info = self.get_position(track["bbox"])

        enriched_track = dict(track)
        enriched_track["position"] = position_info
        return enriched_track
