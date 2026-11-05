"""
Track recovery manager module for VisionMate.

Associates newly appearing ByteTrack IDs with recently lost logical tracks using
motion-aware spatial prediction, bounding box area trend similarity, trajectory direction,
temporal decay, and ambiguity protection to reduce ID fragmentation without neural
re-identification or permanent memory.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Set


class TrackRecoveryManager:
    """
    Lightweight, rule-based track recovery manager operating between ByteTrack and TemporalDetectionStabilizer.
    """

    def __init__(
        self,
        max_recovery_frames: int = 15,
        max_center_distance: float = 0.20,
        min_size_ratio: float = 0.5,
        max_size_ratio: float = 2.0,
        min_recovery_score: float = 0.70,
        frame_width: int = 640,
        frame_height: int = 480,
        max_history_positions: int = 5,
        ambiguity_threshold: float = 0.015,
        min_motion_threshold: float = 0.002
    ):
        """
        Initialize the TrackRecoveryManager.

        Args:
            max_recovery_frames (int): Max consecutive frames to retain a lost track for potential recovery (default 15).
            max_center_distance (float): Max normalized center-point Euclidean distance (0.0 to 1.0) for matching (default 0.20).
            min_size_ratio (float): Minimum area ratio boundary (new_area / ref_area) for matching (default 0.5).
            max_size_ratio (float): Maximum area ratio boundary (new_area / ref_area) for matching (default 2.0).
            min_recovery_score (float): Minimum combined recovery score required to accept candidate (default 0.70).
            frame_width (int): Camera frame width in pixels (default 640).
            frame_height (int): Camera frame height in pixels (default 480).
            max_history_positions (int): Max recent position entries stored for velocity and area trend (default 5).
            ambiguity_threshold (float): Minimum score difference between top 2 candidates to prevent ambiguous assignment (default 0.015).
            min_motion_threshold (float): Minimum normalized movement speed per frame to activate velocity prediction (default 0.002).
        """
        if max_recovery_frames < 1:
            raise ValueError("max_recovery_frames must be at least 1.")
        if max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive.")
        if min_recovery_score <= 0 or min_recovery_score > 1.0:
            raise ValueError("min_recovery_score must be between 0 and 1.")
        if max_history_positions < 1:
            raise ValueError("max_history_positions must be at least 1.")

        self.max_recovery_frames = max_recovery_frames
        self.max_center_distance = max_center_distance
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.min_recovery_score = min_recovery_score
        self.max_history_positions = max_history_positions
        self.ambiguity_threshold = ambiguity_threshold
        self.min_motion_threshold = min_motion_threshold

        self.set_frame_size(frame_width, frame_height)

        # Mapping of raw ByteTrack ID -> assigned logical_track_id
        self._raw_to_logical: Dict[int, int] = {}

        # History of recently lost logical tracks available for recovery
        # Key: logical_track_id (int) -> Value: Dict of track metadata & frames_since_seen
        self._history: Dict[int, Dict[str, Any]] = {}

        # Tracks active in the immediately preceding frame
        self._previous_active: Dict[int, Dict[str, Any]] = {}

        # Position history per logical track for velocity/trend estimation
        # Key: logical_track_id -> List[Dict]
        self._track_pos_history: Dict[int, List[Dict[str, Any]]] = {}

        # Global frame counter
        self._frame_count: int = 0

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Update frame dimensions dynamically for normalized distance calculations.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive integers.")
        self.frame_width = int(width)
        self.frame_height = int(height)

    def clear(self) -> None:
        """
        Reset all internal recovery history, position histories, and ID mappings.
        """
        self._raw_to_logical.clear()
        self._history.clear()
        self._previous_active.clear()
        self._track_pos_history.clear()
        self._frame_count = 0

    def _get_bbox_metrics(self, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Calculate center and area metrics from bounding box coordinates.
        """
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        norm_cx = center_x / self.frame_width
        norm_cy = center_y / self.frame_height
        area = float(width * height)
        frame_area = max(1.0, float(self.frame_width * self.frame_height))
        relative_size = area / frame_area

        return {
            "center_x": center_x,
            "center_y": center_y,
            "norm_cx": norm_cx,
            "norm_cy": norm_cy,
            "bbox_width": float(width),
            "bbox_height": float(height),
            "area": area,
            "relative_size": relative_size,
        }

    def _update_pos_history(self, logical_id: int, metrics: Dict[str, float]) -> None:
        """
        Append current frame position metrics to logical track history.
        """
        if logical_id not in self._track_pos_history:
            self._track_pos_history[logical_id] = []

        pos_entry = {
            "norm_cx": metrics["norm_cx"],
            "norm_cy": metrics["norm_cy"],
            "area": metrics["area"],
            "frame": self._frame_count
        }

        self._track_pos_history[logical_id].append(pos_entry)
        if len(self._track_pos_history[logical_id]) > self.max_history_positions:
            self._track_pos_history[logical_id].pop(0)

    def _estimate_motion(self, logical_id: int) -> Tuple[float, float, float, bool, float]:
        """
        Estimate velocity vector (vx, vy), velocity magnitude (v_mag), has_motion flag,
        and mean area from recent position history.

        Returns:
            Tuple[vx, vy, v_mag, has_motion, mean_area]
        """
        history = self._track_pos_history.get(logical_id, [])
        if not history:
            return 0.0, 0.0, 0.0, False, 1.0

        mean_area = sum(h["area"] for h in history) / len(history)

        if len(history) < 2:
            return 0.0, 0.0, 0.0, False, mean_area

        pos_start = history[0]
        pos_end = history[-1]
        dt = pos_end["frame"] - pos_start["frame"]

        if dt <= 0:
            return 0.0, 0.0, 0.0, False, mean_area

        vx = (pos_end["norm_cx"] - pos_start["norm_cx"]) / dt
        vy = (pos_end["norm_cy"] - pos_start["norm_cy"]) / dt
        v_mag = math.sqrt(vx * vx + vy * vy)
        has_motion = v_mag >= self.min_motion_threshold

        return vx, vy, v_mag, has_motion, mean_area

    def _compute_recovery_score(
        self, new_metrics: Dict[str, float], lost_data: Dict[str, Any]
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Compute candidate recovery score using motion-aware spatial prediction,
        size trend consistency, trajectory direction, and temporal decay.

        Returns:
            Tuple[center_dist, size_ratio, spatial_score, size_score, motion_score, temporal_score, combined_score]
        """
        last_cx = lost_data["last_norm_cx"]
        last_cy = lost_data["last_norm_cy"]
        k_frames = max(1, lost_data["frames_since_seen"])

        # 1. Motion Prediction
        vx = lost_data.get("vx", 0.0)
        vy = lost_data.get("vy", 0.0)
        v_mag = lost_data.get("v_mag", 0.0)
        has_motion = lost_data.get("has_motion", False)

        pred_cx = last_cx + vx * k_frames
        pred_cy = last_cy + vy * k_frames

        # Clamp prediction to frame boundary
        pred_cx = max(0.0, min(1.0, pred_cx))
        pred_cy = max(0.0, min(1.0, pred_cy))

        # Center distance from last observed position
        dx_last = new_metrics["norm_cx"] - last_cx
        dy_last = new_metrics["norm_cy"] - last_cy
        dist_last = math.sqrt(dx_last * dx_last + dy_last * dy_last)

        # Center distance from predicted position
        dx_pred = new_metrics["norm_cx"] - pred_cx
        dy_pred = new_metrics["norm_cy"] - pred_cy
        dist_pred = math.sqrt(dx_pred * dx_pred + dy_pred * dy_pred)

        if has_motion:
            dist_used = min(dist_last, dist_pred)
        else:
            dist_used = dist_last

        center_dist = dist_last

        # 2. Adaptive Spatial Tolerance
        if has_motion:
            adaptive_max_cd = min(
                self.max_center_distance * 1.25,
                self.max_center_distance + (0.5 * v_mag * k_frames)
            )
        else:
            adaptive_max_cd = self.max_center_distance

        max_cd = max(1e-6, adaptive_max_cd)
        spatial_score = max(0.0, 1.0 - (dist_used / max_cd))

        # 3. Size Trend Consistency
        ref_area = lost_data.get("mean_area", lost_data["last_area"])
        size_ratio = new_metrics["area"] / max(1.0, ref_area)

        size_diff = abs(1.0 - size_ratio)
        size_score = max(0.0, 1.0 - size_diff)

        # 4. Trajectory Direction Consistency
        disp_x = new_metrics["norm_cx"] - last_cx
        disp_y = new_metrics["norm_cy"] - last_cy
        disp_mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)

        if has_motion and disp_mag > 1e-4 and v_mag > 1e-4:
            cos_sim = (disp_x * vx + disp_y * vy) / (disp_mag * v_mag)
            motion_score = max(0.0, (cos_sim + 1.0) / 2.0)
        else:
            motion_score = 1.0

        # 5. Temporal Decay
        temporal_score = max(0.0, 1.0 - (0.025 * (k_frames - 1)))

        # 6. Combined Score
        if has_motion:
            combined_score = (
                0.45 * spatial_score +
                0.25 * size_score +
                0.15 * motion_score +
                0.15 * temporal_score
            )
        else:
            combined_score = (
                0.55 * spatial_score +
                0.35 * size_score +
                0.10 * temporal_score
            )

        return (
            center_dist,
            size_ratio,
            spatial_score,
            size_score,
            motion_score,
            temporal_score,
            combined_score,
        )

    def update(self, tracks: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process incoming raw ByteTrack detections, applying deterministic track recovery
        for fragmented IDs before passing to TemporalDetectionStabilizer.

        Args:
            tracks (list): Raw detections list from ObjectTracker.

        Returns:
            list: Detections list with logical_track_id, track_id, and recovery metadata attached.
        """
        self._frame_count += 1
        valid_input_tracks = tracks if isinstance(tracks, list) else []

        # 1. Filter valid incoming raw tracks
        current_raw_tracks: List[Dict[str, Any]] = []
        current_raw_ids: Set[int] = set()

        for t in valid_input_tracks:
            if not isinstance(t, dict) or "track_id" not in t or t["track_id"] is None:
                continue
            try:
                raw_id = int(t["track_id"])
            except (ValueError, TypeError):
                continue

            bbox = t.get("bbox")
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                continue

            t_copy = dict(t)
            t_copy["_raw_id"] = raw_id
            current_raw_tracks.append(t_copy)
            current_raw_ids.add(raw_id)

        # 2. Identify active tracks from previous frame that disappeared in this frame
        for prev_logical_id, prev_data in list(self._previous_active.items()):
            prev_raw_id = prev_data.get("original_track_id", prev_logical_id)
            if prev_raw_id not in current_raw_ids:
                if prev_logical_id not in self._history:
                    # Move to recovery history with estimated motion signals
                    metrics = self._get_bbox_metrics(prev_data["bbox"])
                    vx, vy, v_mag, has_motion, mean_area = self._estimate_motion(prev_logical_id)

                    self._history[prev_logical_id] = {
                        "logical_track_id": prev_logical_id,
                        "previous_track_id": prev_raw_id,
                        "class_id": prev_data.get("class_id"),
                        "class_name": str(prev_data.get("class_name", "")).strip().lower(),
                        "bbox": prev_data["bbox"],
                        "last_norm_cx": metrics["norm_cx"],
                        "last_norm_cy": metrics["norm_cy"],
                        "last_area": metrics["area"],
                        "mean_area": mean_area,
                        "vx": vx,
                        "vy": vy,
                        "v_mag": v_mag,
                        "has_motion": has_motion,
                        "last_confidence": prev_data.get("confidence", 0.0),
                        "frames_since_seen": 0,
                        "metrics": metrics
                    }

        # 3. Increment frames_since_seen for history items and expire old tracks
        for logical_id in list(self._history.keys()):
            self._history[logical_id]["frames_since_seen"] += 1
            if self._history[logical_id]["frames_since_seen"] > self.max_recovery_frames:
                del self._history[logical_id]
                # Purge stale raw_to_logical and position history mappings for expired logical_id
                raw_to_del = [r for r, l in self._raw_to_logical.items() if l == logical_id]
                for r in raw_to_del:
                    del self._raw_to_logical[r]
                self._track_pos_history.pop(logical_id, None)

        # 4. Classify incoming raw tracks into continuous vs unassigned
        continuous_raw_tracks: List[Dict[str, Any]] = []
        unassigned_raw_tracks: List[Dict[str, Any]] = []
        claimed_continuous_logical: Set[int] = set()

        for t in current_raw_tracks:
            raw_id = t["_raw_id"]
            if raw_id in self._raw_to_logical:
                candidate_logical = self._raw_to_logical[raw_id]
                if candidate_logical not in claimed_continuous_logical:
                    continuous_raw_tracks.append(t)
                    claimed_continuous_logical.add(candidate_logical)
                else:
                    unassigned_raw_tracks.append(t)
            else:
                unassigned_raw_tracks.append(t)

        # Remove continuous tracks from recovery history if they were briefly recorded
        for logical_id in claimed_continuous_logical:
            self._history.pop(logical_id, None)

        # 5. Build candidate recovery scoring matrix for unassigned tracks
        candidate_edges: List[Dict[str, Any]] = []
        raw_candidates_map: Dict[int, List[Dict[str, Any]]] = {}
        lost_candidates_map: Dict[int, List[Dict[str, Any]]] = {}

        for t in unassigned_raw_tracks:
            raw_id = t["_raw_id"]
            class_name = str(t.get("class_name", "")).strip().lower()
            metrics = self._get_bbox_metrics(t["bbox"])
            t["_metrics"] = metrics

            for lost_logical_id, lost_data in self._history.items():
                # Rule 1: Mandatory Class Match
                if lost_data["class_name"] != class_name:
                    continue

                (
                    center_dist,
                    size_ratio,
                    spatial_score,
                    size_score,
                    motion_score,
                    temporal_score,
                    combined_score,
                ) = self._compute_recovery_score(metrics, lost_data)

                # Rule 2: Adaptive Spatial Tolerance Check
                has_motion = lost_data.get("has_motion", False)
                v_mag = lost_data.get("v_mag", 0.0)
                k_frames = max(1, lost_data["frames_since_seen"])
                if has_motion:
                    adaptive_max_cd = min(
                        self.max_center_distance * 1.25,
                        self.max_center_distance + (0.5 * v_mag * k_frames)
                    )
                else:
                    adaptive_max_cd = self.max_center_distance

                if center_dist > adaptive_max_cd:
                    continue

                # Rule 3: Size ratio threshold
                if size_ratio < self.min_size_ratio or size_ratio > self.max_size_ratio:
                    continue

                # Rule 4: Score threshold check
                if combined_score < self.min_recovery_score:
                    continue

                cand_info = {
                    "score": combined_score,
                    "raw_id": raw_id,
                    "lost_logical_id": lost_logical_id,
                    "spatial_score": spatial_score,
                    "size_score": size_score,
                    "motion_score": motion_score,
                    "temporal_score": temporal_score,
                    "center_dist": center_dist,
                    "size_ratio": size_ratio,
                    "has_motion": has_motion
                }

                candidate_edges.append(cand_info)
                raw_candidates_map.setdefault(raw_id, []).append(cand_info)
                lost_candidates_map.setdefault(lost_logical_id, []).append(cand_info)

        # 6. Ambiguity Protection
        ambiguous_raw_ids: Set[int] = set()
        for raw_id, cands in raw_candidates_map.items():
            if len(cands) >= 2:
                cands.sort(key=lambda x: x["score"], reverse=True)
                if cands[0]["score"] - cands[1]["score"] < self.ambiguity_threshold:
                    ambiguous_raw_ids.add(raw_id)

        ambiguous_lost_ids: Set[int] = set()
        for lost_id, cands in lost_candidates_map.items():
            if len(cands) >= 2:
                cands.sort(key=lambda x: x["score"], reverse=True)
                if cands[0]["score"] - cands[1]["score"] < self.ambiguity_threshold:
                    ambiguous_lost_ids.add(lost_id)

        # 7. One-to-One Greedy Assignment
        candidate_edges.sort(key=lambda x: x["score"], reverse=True)

        assigned_raw_info: Dict[int, Dict[str, Any]] = {}
        claimed_lost_ids: Set[int] = set()

        for edge in candidate_edges:
            raw_id = edge["raw_id"]
            lost_id = edge["lost_logical_id"]

            if raw_id in assigned_raw_info:
                continue
            if lost_id in claimed_lost_ids:
                continue
            if raw_id in ambiguous_raw_ids or lost_id in ambiguous_lost_ids:
                continue

            assigned_raw_info[raw_id] = {
                "logical_id": lost_id,
                "recovered": True,
                "recovered_from_id": self._history[lost_id]["previous_track_id"],
                "score": edge["score"],
                "spatial_score": edge["spatial_score"],
                "size_score": edge["size_score"],
                "motion_score": edge["motion_score"],
                "temporal_score": edge["temporal_score"],
                "recovery_reason": "recovered_motion" if edge["has_motion"] else "recovered_spatial"
            }
            claimed_lost_ids.add(lost_id)

        # Purge assigned lost tracks from history
        for lost_id in claimed_lost_ids:
            self._history.pop(lost_id, None)

        # Handle remaining unassigned tracks
        for t in unassigned_raw_tracks:
            raw_id = t["_raw_id"]
            if raw_id not in assigned_raw_info:
                reason = "ambiguity_rejected" if raw_id in ambiguous_raw_ids else "new_track"
                assigned_raw_info[raw_id] = {
                    "logical_id": raw_id,
                    "recovered": False,
                    "recovered_from_id": None,
                    "score": 0.0,
                    "spatial_score": 0.0,
                    "size_score": 0.0,
                    "motion_score": 0.0,
                    "temporal_score": 0.0,
                    "recovery_reason": reason
                }

        # Handle continuous tracks
        for t in continuous_raw_tracks:
            raw_id = t["_raw_id"]
            logical_id = self._raw_to_logical[raw_id]
            assigned_raw_info[raw_id] = {
                "logical_id": logical_id,
                "recovered": False,
                "recovered_from_id": None,
                "score": 0.0,
                "spatial_score": 0.0,
                "size_score": 0.0,
                "motion_score": 0.0,
                "temporal_score": 0.0,
                "recovery_reason": "continuous"
            }

        # 8. Assemble output tracks
        output_tracks: List[Dict[str, Any]] = []
        current_active_logical: Dict[int, Dict[str, Any]] = {}

        for t in current_raw_tracks:
            raw_id = t["_raw_id"]
            metrics = t.get("_metrics") or self._get_bbox_metrics(t["bbox"])
            info = assigned_raw_info[raw_id]
            logical_id = info["logical_id"]

            self._raw_to_logical[raw_id] = logical_id
            self._update_pos_history(logical_id, metrics)

            out_t = dict(t)
            out_t.pop("_raw_id", None)
            out_t.pop("_metrics", None)

            out_t["logical_track_id"] = logical_id
            out_t["track_id"] = logical_id
            out_t["original_track_id"] = raw_id
            out_t["recovered"] = info["recovered"]
            out_t["recovered_from_track_id"] = info["recovered_from_id"]
            out_t["recovery_score"] = round(info["score"], 4)
            out_t["spatial_score"] = round(info["spatial_score"], 4)
            out_t["size_score"] = round(info["size_score"], 4)
            out_t["motion_score"] = round(info["motion_score"], 4)
            out_t["temporal_score"] = round(info["temporal_score"], 4)
            out_t["recovery_reason"] = info["recovery_reason"]

            output_tracks.append(out_t)
            current_active_logical[logical_id] = out_t

        self._previous_active = current_active_logical

        # Deterministic ordering by logical_track_id
        output_tracks.sort(key=lambda x: int(x["logical_track_id"]))
        return output_tracks
