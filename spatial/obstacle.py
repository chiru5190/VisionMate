"""
Obstacle analysis module for VisionMate.

Evaluates processed 2D object detections against safety parameters to identify
immediate obstacles, generating structured proximity warnings ordered by risk severity.
"""

from typing import Dict, Any, List, Optional, Set, Iterable

# Default COCO/YOLO classes categorized as potential physical navigation obstacles
DEFAULT_OBSTACLE_CLASSES: Set[str] = {
    "person",
    "chair",
    "couch",
    "table",
    "dining table",
    "bed",
    "backpack",
    "suitcase",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "bench",
    "potted plant",
}

# Distance category rank mapping (higher rank = closer proximity)
DISTANCE_RANKS: Dict[str, int] = {
    "far": 0,
    "medium": 1,
    "close": 2,
    "very_close": 3,
}


class ObstacleAnalyzer:
    """
    Deterministic obstacle analyzer that filters live detections and produces structured proximity warnings.
    """

    def __init__(
        self,
        min_confidence: float = 0.40,
        warning_distance: str = "close",
        critical_distance: str = "very_close",
        obstacle_classes: Optional[Iterable[str]] = None
    ) -> None:
        """
        Initialize the ObstacleAnalyzer with safety thresholds and target obstacle classes.

        Args:
            min_confidence (float): Minimum detection confidence required to consider an obstacle (default 0.40).
            warning_distance (str): Distance category threshold for "warning" severity (default "close").
            critical_distance (str): Distance category threshold for "critical" severity (default "very_close").
            obstacle_classes (iterable, optional): List/set of class names treated as obstacles.
                                                  Defaults to DEFAULT_OBSTACLE_CLASSES if None.
        """
        self.min_confidence = float(min_confidence)
        self.warning_distance = warning_distance.lower()
        self.critical_distance = critical_distance.lower()

        if obstacle_classes is not None:
            self.obstacle_classes: Set[str] = {
                str(c).strip().lower() for c in obstacle_classes if str(c).strip()
            }
        else:
            self.obstacle_classes = set(DEFAULT_OBSTACLE_CLASSES)

    def is_obstacle_class(self, class_name: str) -> bool:
        """
        Check if a given class name is included in the active obstacle configuration.

        Args:
            class_name (str): Object class label.

        Returns:
            bool: True if class is a recognized obstacle, False otherwise.
        """
        if not class_name:
            return False
        c_clean = class_name.strip().lower()
        return c_clean in self.obstacle_classes

    def analyze(self, detections: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Analyze a list of frame detections and return structured obstacle warning dictionaries.

        Args:
            detections (list, optional): List of detection dictionaries containing spatial and distance metadata.

        Returns:
            list: List of structured warning dictionaries ordered by proximity score descending.
                  Returns [] if no obstacles exceed warning thresholds.
        """
        if not isinstance(detections, list) or not detections:
            return []

        warnings: List[Dict[str, Any]] = []

        for det in detections:
            if not isinstance(det, dict):
                continue

            # 1. Check confidence threshold
            confidence = float(det.get("confidence", 0.0))
            if confidence < self.min_confidence:
                continue

            # 2. Check obstacle class category
            raw_class_name = str(det.get("class_name", "")).strip().lower()
            if not self.is_obstacle_class(raw_class_name):
                continue

            # 3. Extract distance category and proximity score
            dist_info = det.get("distance")
            dist_category = "far"
            proximity_score = 0.0

            if isinstance(dist_info, dict):
                dist_category = str(dist_info.get("category", "far")).strip().lower()
                proximity_score = float(dist_info.get("proximity_score", 0.0))
            elif isinstance(dist_info, str):
                dist_category = dist_info.strip().lower()

            # Fallback proximity score from distance category rank if proximity_score is 0
            dist_rank = DISTANCE_RANKS.get(dist_category, 0)
            if proximity_score <= 0.0:
                proximity_score = float(dist_rank)

            # 4. Filter by distance warning thresholds
            warning_rank = DISTANCE_RANKS.get(self.warning_distance, 2)
            critical_rank = DISTANCE_RANKS.get(self.critical_distance, 3)

            if dist_rank < warning_rank:
                # Below warning distance (e.g., "far" or "medium")
                continue

            # Determine severity and message key
            if dist_rank >= critical_rank or dist_category == self.critical_distance:
                severity = "critical"
                message_key = "obstacle_very_close"
            else:
                severity = "warning"
                message_key = "obstacle_close"

            # 5. Extract horizontal spatial position
            pos_info = det.get("position")
            horiz_position = "center"
            if isinstance(pos_info, dict):
                horiz_position = str(pos_info.get("horizontal", "center")).strip().lower()
            elif isinstance(pos_info, str):
                horiz_position = pos_info.strip().lower()

            warning_entry = {
                "object": raw_class_name,
                "position": horiz_position,
                "distance": dist_category,
                "severity": severity,
                "message_key": message_key,
                "proximity_score": proximity_score,
                "confidence": confidence,
            }

            warnings.append(warning_entry)

        # 6. Order warnings by proximity score descending (closest objects first)
        warnings.sort(
            key=lambda w: (w["proximity_score"], w["confidence"]),
            reverse=True
        )

        return warnings
