"""
Object Query Memory module for VisionMate.

Maintains short-term detection history across recent frames to serve object spatial queries
(e.g., "Where is my phone?") even when an object is briefly missed in the exact current frame.
Groups observations of the same physical object while keeping distinct objects separate.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Set


def _compute_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate center pixel coordinates (cx, cy) from bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _compute_norm_center_distance(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    frame_w: int = 640,
    frame_h: int = 480
) -> float:
    """Calculate normalized center-point Euclidean distance between two bboxes."""
    cx1, cy1 = _compute_center(bbox1)
    cx2, cy2 = _compute_center(bbox2)
    dx = (cx1 - cx2) / max(1.0, float(frame_w))
    dy = (cy1 - cy2) / max(1.0, float(frame_h))
    return math.sqrt(dx * dx + dy * dy)


def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)

    area1 = float(max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1))
    area2 = float(max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2))

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class ObjectQueryMemory:
    """
    Lightweight, short-term detection memory for spatial object queries.
    """

    def __init__(
        self,
        max_history_frames: int = 30,
        iou_threshold: float = 0.5,
        spatial_distance_threshold: float = 0.15,
        frame_width: int = 640,
        frame_height: int = 480
    ):
        """
        Initialize the ObjectQueryMemory.

        Args:
            max_history_frames (int): Number of recent frame snapshots to retain (default 30).
            iou_threshold (float): Bounding box IoU threshold to group same-object observations (default 0.5).
            spatial_distance_threshold (float): Normalized center distance threshold to group same-object observations (default 0.15).
            frame_width (int): Camera frame width in pixels (default 640).
            frame_height (int): Camera frame height in pixels (default 480).
        """
        if max_history_frames < 1:
            raise ValueError("max_history_frames must be at least 1.")

        self.max_history_frames = max_history_frames
        self.iou_threshold = iou_threshold
        self.spatial_distance_threshold = spatial_distance_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Snapshot history list of Dicts: {"frame_idx": int, "detections": List[Dict]}
        self._history: List[Dict[str, Any]] = []
        self._frame_count: int = 0

    def clear(self) -> None:
        """
        Reset all stored detection history and frame counter.
        """
        self._history.clear()
        self._frame_count = 0

    def update(self, detections: Optional[List[Dict[str, Any]]]) -> None:
        """
        Record a frame snapshot of perception detections into recent history.

        Args:
            detections (list, optional): List of detection dictionaries for current frame.
        """
        self._frame_count += 1
        valid_input = detections if isinstance(detections, list) else []

        cleaned_detections = []
        for det in valid_input:
            if not isinstance(det, dict):
                continue
            class_name = str(det.get("class_name", "")).strip().lower()
            bbox = det.get("bbox")
            if not class_name or not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                continue

            cleaned_detections.append({
                "class_id": det.get("class_id", 0),
                "class_name": class_name,
                "confidence": float(det.get("confidence", 0.0)),
                "track_id": det.get("track_id"),
                "bbox": tuple(int(c) for c in bbox),
                "position": det.get("position"),
                "distance": det.get("distance"),
                "is_stale": det.get("is_stale", False),
                "recovered": det.get("recovered", False)
            })

        self._history.append({
            "frame_idx": self._frame_count,
            "detections": cleaned_detections
        })

        # Maintain sliding history window
        if len(self._history) > self.max_history_frames:
            self._history.pop(0)

    def find_object(self, object_name: str) -> List[Dict[str, Any]]:
        """
        Search recent detection history for an object by class name or common alias.

        Args:
            object_name (str): Requested object query string (e.g., "phone", "cell phone", "bottle").

        Returns:
            list: List of structured physical object result dictionaries, ranked by recency,
                  stability (observation count), and confidence. Returns [] if not found.
        """
        if not object_name or not isinstance(object_name, str):
            return []

        query = object_name.strip().lower()
        if not query:
            return []

        # Alias mappings for common user query terms
        alias_map = {
            "phone": ["cell phone", "phone", "mobile"],
            "cup": ["cup", "mug"],
            "bottle": ["bottle", "water bottle"],
            "keys": ["keys", "keychain"],
            "laptop": ["laptop", "computer"]
        }
        target_names = set(alias_map.get(query, [query]))

        # Collect matching observations from history
        matching_obs = []
        for snapshot in self._history:
            frame_idx = snapshot["frame_idx"]
            for det in snapshot["detections"]:
                cname = det["class_name"]
                if cname in target_names or any(t in cname for t in target_names):
                    matching_obs.append({
                        "frame_idx": frame_idx,
                        "data": det
                    })

        if not matching_obs:
            return []

        # Group observations into physical object clusters
        clusters: List[List[Dict[str, Any]]] = []

        for obs in matching_obs:
            obs_det = obs["data"]
            obs_track_id = obs_det.get("track_id")
            obs_bbox = obs_det["bbox"]

            merged = False
            for cluster in clusters:
                rep = cluster[-1]["data"]
                rep_track_id = rep.get("track_id")
                rep_bbox = rep["bbox"]

                # Criteria 1: Identical non-None track ID
                if obs_track_id is not None and rep_track_id is not None and obs_track_id == rep_track_id:
                    merged = True

                # Criteria 2: Substantial bounding box IoU
                elif _compute_iou(obs_bbox, rep_bbox) >= self.iou_threshold:
                    merged = True

                # Criteria 3: Spatial center proximity
                elif _compute_norm_center_distance(obs_bbox, rep_bbox, self.frame_width, self.frame_height) <= self.spatial_distance_threshold:
                    merged = True

                if merged:
                    cluster.append(obs)
                    break

            if not merged:
                clusters.append([obs])

        # Synthesize structured result per physical object cluster
        results = []
        for cluster in clusters:
            latest_item = max(cluster, key=lambda x: (x["frame_idx"], x["data"]["confidence"]))
            latest_det = latest_item["data"]
            latest_frame = latest_item["frame_idx"]

            unique_frames = len(set(item["frame_idx"] for item in cluster))
            max_conf = max(item["data"]["confidence"] for item in cluster)
            recency_ago = self._frame_count - latest_frame
            is_in_current_frame = (latest_frame == self._frame_count)

            results.append({
                "class_id": latest_det.get("class_id", 0),
                "class_name": latest_det["class_name"],
                "track_id": latest_det.get("track_id"),
                "confidence": round(latest_det.get("confidence", 0.0), 4),
                "max_confidence": round(max_conf, 4),
                "bbox": latest_det["bbox"],
                "position": latest_det.get("position"),
                "distance": latest_det.get("distance"),
                "recency_frames_ago": recency_ago,
                "is_in_current_frame": is_in_current_frame,
                "observation_count": unique_frames,
                "latest_seen_frame": latest_frame,
            })

        # Rank results: prefer current frame, newest frame, higher stability (obs count), higher confidence
        results.sort(
            key=lambda r: (
                r["is_in_current_frame"],
                r["latest_seen_frame"],
                r["observation_count"],
                r["max_confidence"]
            ),
            reverse=True
        )

        return results
