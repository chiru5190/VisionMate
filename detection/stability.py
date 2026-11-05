"""
Temporal detection stability & class-consistency module for VisionMate.

Retains recently observed ByteTrack objects across short temporary detection misses
and suppresses transient single-frame classification noise to improve perception continuity.
"""

from typing import Dict, Any, List, Optional


class TemporalDetectionStabilizer:
    """
    Deterministic temporal tracker wrapper that maintains short-term memory of recent
    ByteTrack identities and enforces class consistency across transient classification errors.
    """

    def __init__(
        self,
        max_missing_frames: int = 8,
        class_change_threshold: int = 3
    ):
        """
        Initialize the TemporalDetectionStabilizer.

        Args:
            max_missing_frames (int): Maximum consecutive missing frames before a track expires (default 8).
            class_change_threshold (int): Consecutive matching observations required to change an established class (default 3).
        """
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be a non-negative integer.")
        if class_change_threshold < 1:
            raise ValueError("class_change_threshold must be a positive integer.")

        self.max_missing_frames = max_missing_frames
        self.class_change_threshold = class_change_threshold

        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._missing_counters: Dict[int, int] = {}

        # Class consistency state tracking per track_id
        self._stable_class_info: Dict[int, Dict[str, Any]] = {}
        self._candidate_class_info: Dict[int, Optional[Dict[str, Any]]] = {}
        self._class_change_counts: Dict[int, int] = {}

    def clear(self) -> None:
        """
        Clear all stored tracks, counters, and class consistency state.
        """
        self._active_tracks.clear()
        self._missing_counters.clear()
        self._stable_class_info.clear()
        self._candidate_class_info.clear()
        self._class_change_counts.clear()

    def update(self, tracks: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Update track state with new frame detections, enforcing temporal retention and class consistency.

        Args:
            tracks (list): Current frame track dictionaries from ObjectTracker.

        Returns:
            list: Stabilized list of detection dictionaries, enriched with 'is_stale',
                  'missing_frames', 'stable_class_name', 'candidate_class_name', and 'class_change_count'.
        """
        valid_input_tracks = tracks if isinstance(tracks, list) else []
        seen_ids = set()

        # 1. Process current incoming tracks
        for t in valid_input_tracks:
            if not isinstance(t, dict):
                continue

            track_id = t.get("track_id")
            if track_id is None:
                continue

            try:
                track_id_int = int(track_id)
            except (ValueError, TypeError):
                continue

            seen_ids.add(track_id_int)
            self._missing_counters[track_id_int] = 0

            observed_class_id = t.get("class_id")
            observed_class_name = str(t.get("class_name", "")).strip()

            # Rule 1: Initial Class Setup for new track_id
            if track_id_int not in self._stable_class_info:
                self._stable_class_info[track_id_int] = {
                    "class_id": observed_class_id,
                    "class_name": observed_class_name
                }
                self._candidate_class_info[track_id_int] = None
                self._class_change_counts[track_id_int] = 0
            else:
                stable_name = self._stable_class_info[track_id_int]["class_name"]

                # Rule 2: Same Class Observed
                if observed_class_name == stable_name:
                    self._candidate_class_info[track_id_int] = None
                    self._class_change_counts[track_id_int] = 0
                else:
                    # Rule 3 & 4: Different Class Observed (Transient vs Persistent Change)
                    candidate = self._candidate_class_info[track_id_int]
                    if candidate and candidate.get("class_name") == observed_class_name:
                        self._class_change_counts[track_id_int] += 1
                    else:
                        self._candidate_class_info[track_id_int] = {
                            "class_id": observed_class_id,
                            "class_name": observed_class_name
                        }
                        self._class_change_counts[track_id_int] = 1

                    # Rule 4: Threshold reached -> Accept new stable class
                    if self._class_change_counts[track_id_int] >= self.class_change_threshold:
                        self._stable_class_info[track_id_int] = {
                            "class_id": observed_class_id,
                            "class_name": observed_class_name
                        }
                        self._candidate_class_info[track_id_int] = None
                        self._class_change_counts[track_id_int] = 0

            # Construct output track with enforced stable class identity
            stable_info = self._stable_class_info[track_id_int]
            candidate_info = self._candidate_class_info[track_id_int]

            track_copy = dict(t)
            track_copy["class_id"] = stable_info["class_id"]
            track_copy["class_name"] = stable_info["class_name"]
            track_copy["is_stale"] = False
            track_copy["missing_frames"] = 0
            track_copy["stable_class_name"] = stable_info["class_name"]
            track_copy["candidate_class_name"] = candidate_info["class_name"] if candidate_info else None
            track_copy["class_change_count"] = self._class_change_counts[track_id_int]

            self._active_tracks[track_id_int] = track_copy

        # 2. Update missing counters for tracks not seen in current frame
        missing_ids = set(self._active_tracks.keys()) - seen_ids
        for m_id in list(missing_ids):
            self._missing_counters[m_id] = self._missing_counters.get(m_id, 0) + 1
            missing_count = self._missing_counters[m_id]

            if missing_count <= self.max_missing_frames:
                stable_info = self._stable_class_info[m_id]
                candidate_info = self._candidate_class_info.get(m_id)

                # Maintain last known state and position while marked stale
                stale_track = dict(self._active_tracks[m_id])
                stale_track["class_id"] = stable_info["class_id"]
                stale_track["class_name"] = stable_info["class_name"]
                stale_track["is_stale"] = True
                stale_track["missing_frames"] = missing_count
                stale_track["stable_class_name"] = stable_info["class_name"]
                stale_track["candidate_class_name"] = candidate_info["class_name"] if candidate_info else None
                stale_track["class_change_count"] = self._class_change_counts.get(m_id, 0)

                self._active_tracks[m_id] = stale_track
            else:
                # Expire track and purge all associated class consistency history
                del self._active_tracks[m_id]
                del self._missing_counters[m_id]
                self._stable_class_info.pop(m_id, None)
                self._candidate_class_info.pop(m_id, None)
                self._class_change_counts.pop(m_id, None)

        # 3. Deterministic ordering: fresh tracks first (by track_id), then stale tracks (by track_id)
        fresh_tracks = [
            t for tid, t in self._active_tracks.items()
            if not t.get("is_stale", False)
        ]
        stale_tracks = [
            t for tid, t in self._active_tracks.items()
            if t.get("is_stale", False)
        ]

        fresh_tracks.sort(key=lambda x: int(x["track_id"]))
        stale_tracks.sort(key=lambda x: int(x["track_id"]))

        return fresh_tracks + stale_tracks
