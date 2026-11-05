"""
VisionMate - Controlled YOLO Detection Confidence Experiment Script

Evaluates physical webcam object tracking performance across different YOLO confidence thresholds
(0.35 baseline vs 0.30 vs 0.25) without modifying production files.

Measures:
- Raw YOLO/ByteTrack detection reliability & detection gaps
- Stabilized & recovered phone presence metrics
- Track ID fragmentation (raw ID changes vs logical ID changes)
- Confidence distributions & live FPS
- False positive / extra object observations by class

Usage:
  # Mode 2 — Controlled CLI Runs (Recommended)
  python scratch/evaluate_confidence.py --confidence 0.35
  python scratch/evaluate_confidence.py --confidence 0.30
  python scratch/evaluate_confidence.py --confidence 0.25

  # Mode 1 — Manual Interactive Run
  python scratch/evaluate_confidence.py
"""

import sys
import os
import time
import json
import argparse
import datetime
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from detection.tracker import ObjectTracker
from detection.recovery import TrackRecoveryManager
from detection.stability import TemporalDetectionStabilizer


def find_working_camera():
    """
    Test camera indices 0, 1, 2, 3 in sequence and return the first working VideoCapture.
    """
    for index in [0, 1, 2, 3]:
        print(f"Testing camera index: {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"--> Using camera index: {index}")
                return cap, index
            cap.release()
    return None, -1


class ConfidenceEvaluator:
    def __init__(self, confidence: float = 0.35, duration: int = 0, results_file: str = "scratch/confidence_results.json"):
        self.target_confidence = float(confidence)
        self.target_duration = int(duration)
        self.results_file = results_file

        self.cap = None
        self.frame_width = 640
        self.frame_height = 480

        self.tracker = None
        self.recovery_manager = None
        self.stabilizer = None

        # Evaluator Metrics State
        self.reset_metrics()

    def reset_metrics(self):
        """
        Reset all session performance metrics.
        """
        self.total_frames = 0
        self.raw_phone_frames = 0
        self.stabilized_phone_frames = 0
        self.recovered_phone_frames = 0

        self.raw_phone_gaps = 0
        self.current_raw_gap = 0
        self.longest_raw_gap = 0
        self.in_raw_gap = False

        self.last_raw_phone_id = None
        self.raw_id_changes = 0

        self.last_logical_phone_id = None
        self.logical_id_changes = 0

        self.fps_list = []
        self.phone_confidences = []

        self.other_object_detections_count = 0
        self.other_object_counts_by_class = {}

        self.session_start_time = time.time()

    def initialize_pipeline(self, target_conf: float, frame_w: int, frame_h: int):
        """
        Cleanly initialize or re-initialize object tracker, recovery manager, and stabilizer.
        """
        self.target_confidence = float(target_conf)
        self.frame_width = frame_w
        self.frame_height = frame_h

        print(f"\n[Evaluator] Initializing perception pipeline at confidence={self.target_confidence:.2f}...")
        
        # Initialize YOLO26n ByteTrack with specified confidence
        self.tracker = ObjectTracker(
            model_path_or_detector="yolo26n.pt",
            confidence=self.target_confidence,
            tracker_type="models/visionmate_bytetrack.yaml" if os.path.exists("models/visionmate_bytetrack.yaml") else "bytetrack.yaml"
        )

        self.recovery_manager = TrackRecoveryManager(
            max_recovery_frames=15,
            max_center_distance=0.20,
            min_size_ratio=0.5,
            max_size_ratio=2.0,
            min_recovery_score=0.70,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )

        self.stabilizer = TemporalDetectionStabilizer(
            max_missing_frames=8,
            class_change_threshold=3
        )

        self.reset_metrics()
        print(f"[Evaluator] Perception pipeline initialized cleanly for confidence={self.target_confidence:.2f}.")

    def record_frame_metrics(self, raw_tracks, recovered_tracks, stable_tracks, frame_fps):
        """
        Extract metrics distinguishing raw YOLO/ByteTrack detections, recovery, and stability.
        """
        self.total_frames += 1
        if frame_fps > 0:
            self.fps_list.append(frame_fps)

        # 1. RAW YOLO/ByteTrack Phone Detections
        raw_phone_tracks = [
            t for t in raw_tracks
            if str(t.get("class_name", "")).strip().lower() == "cell phone"
        ]

        if raw_phone_tracks:
            self.raw_phone_frames += 1

            # Pick highest confidence phone detection in this frame
            primary_raw_phone = max(raw_phone_tracks, key=lambda x: x.get("confidence", 0.0))
            raw_conf = primary_raw_phone.get("confidence", 0.0)
            raw_id = primary_raw_phone.get("track_id")

            self.phone_confidences.append(raw_conf)

            # Check gap state
            if self.in_raw_gap:
                self.in_raw_gap = False
            self.current_raw_gap = 0

            # Check raw ID changes
            if self.last_raw_phone_id is not None and raw_id != self.last_raw_phone_id:
                self.raw_id_changes += 1
            self.last_raw_phone_id = raw_id
        else:
            # Phone missed in raw YOLO output
            if self.last_raw_phone_id is not None:
                if not self.in_raw_gap:
                    self.in_raw_gap = True
                    self.raw_phone_gaps += 1
                self.current_raw_gap += 1
                if self.current_raw_gap > self.longest_raw_gap:
                    self.longest_raw_gap = self.current_raw_gap

        # 2. RECOVERED Phone Detections
        recovered_phone_tracks = [
            t for t in recovered_tracks
            if str(t.get("class_name", "")).strip().lower() == "cell phone" and t.get("recovered", False)
        ]
        if recovered_phone_tracks:
            self.recovered_phone_frames += 1

        # 3. STABILIZED Phone Detections
        stable_phone_tracks = [
            t for t in stable_tracks
            if str(t.get("class_name", "")).strip().lower() == "cell phone"
        ]

        if stable_phone_tracks:
            self.stabilized_phone_frames += 1
            primary_stable = stable_phone_tracks[0]
            logical_id = primary_stable.get("track_id")

            if self.last_logical_phone_id is not None and logical_id != self.last_logical_phone_id:
                self.logical_id_changes += 1
            self.last_logical_phone_id = logical_id

        # 4. EXTRA / NON-PHONE OBJECT DETECTIONS
        raw_other_tracks = [
            t for t in raw_tracks
            if str(t.get("class_name", "")).strip().lower() != "cell phone"
        ]
        self.other_object_detections_count += len(raw_other_tracks)

        for ot in raw_other_tracks:
            cname = str(ot.get("class_name", "unknown")).strip().lower()
            self.other_object_counts_by_class[cname] = self.other_object_counts_by_class.get(cname, 0) + 1

    def compute_summary_results(self) -> dict:
        """
        Compile final statistical report structure.
        """
        tf = max(1, self.total_frames)

        raw_rate = (self.raw_phone_frames / tf) * 100.0
        stab_rate = (self.stabilized_phone_frames / tf) * 100.0

        avg_fps = round(float(np.mean(self.fps_list)), 2) if self.fps_list else "N/A"
        min_fps = round(float(np.min(self.fps_list)), 2) if self.fps_list else "N/A"
        max_fps = round(float(np.max(self.fps_list)), 2) if self.fps_list else "N/A"

        avg_conf = round(float(np.mean(self.phone_confidences)), 4) if self.phone_confidences else "N/A"
        min_conf = round(float(np.min(self.phone_confidences)), 4) if self.phone_confidences else "N/A"

        duration_sec = round(time.time() - self.session_start_time, 1)

        summary = {
            "confidence_threshold": self.target_confidence,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_seconds": duration_sec,
            "total_frames": self.total_frames,
            "raw_phone_detected_frames": self.raw_phone_frames,
            "raw_phone_detection_rate_pct": round(raw_rate, 2),
            "stabilized_phone_detected_frames": self.stabilized_phone_frames,
            "stabilized_phone_detection_rate_pct": round(stab_rate, 2),
            "recovered_phone_frames": self.recovered_phone_frames,
            "raw_phone_detection_gaps": self.raw_phone_gaps,
            "longest_raw_gap_frames": self.longest_raw_gap,
            "raw_track_id_changes": self.raw_id_changes,
            "logical_track_id_changes": self.logical_id_changes,
            "average_fps": avg_fps,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "average_phone_confidence": avg_conf,
            "minimum_phone_confidence": min_conf,
            "total_extra_object_detections": self.other_object_detections_count,
            "observed_extra_detections_by_class": dict(sorted(self.other_object_counts_by_class.items(), key=lambda x: x[1], reverse=True))
        }

        return summary

    def save_results(self):
        """
        Append or update evaluation results in scratch/confidence_results.json.
        """
        summary = self.compute_summary_results()
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        existing_data = []
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, "r") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except Exception:
                existing_data = []

        # Remove previous entry for same confidence threshold if present, then append fresh result
        existing_data = [item for item in existing_data if item.get("confidence_threshold") != self.target_confidence]
        existing_data.append(summary)

        with open(self.results_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        print(f"\n==========================================================")
        print(f" Evaluation Results Saved to '{self.results_file}' ")
        print(f"==========================================================")
        print(json.dumps(summary, indent=2))
        print("==========================================================")

    def draw_eval_hud(self, frame: np.ndarray, raw_tracks, stable_tracks, fps: float):
        """
        Render real-time assessment HUD overlay on the frame window.
        """
        h, w = frame.shape[:2]

        # 1. Top Panel: System Status & Confidence Setting
        cv2.rectangle(frame, (10, 10), (w - 10, 50), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 50), (0, 255, 255), 1)

        title = f"CONFIDENCE EVALUATION | Threshold: {self.target_confidence:.2f}"
        fps_text = f"FPS: {fps:.1f} | Frames: {self.total_frames}"

        cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, fps_text, (w - 230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Phone Status Indicator (Top-Right Box)
        raw_phones = [t for t in raw_tracks if str(t.get("class_name", "")).strip().lower() == "cell phone"]
        stable_phones = [t for t in stable_tracks if str(t.get("class_name", "")).strip().lower() == "cell phone"]

        box_y1, box_y2 = 60, 155
        box_x1, box_x2 = w - 280, w - 10

        if raw_phones:
            primary_raw = max(raw_phones, key=lambda x: x.get("confidence", 0.0))
            raw_conf = primary_raw.get("confidence", 0.0)
            raw_id = primary_raw.get("track_id", "N/A")

            logical_id = stable_phones[0].get("track_id", "N/A") if stable_phones else "N/A"
            is_recovered = stable_phones[0].get("recovered", False) if stable_phones else False

            bg_color = (0, 100, 0)
            border_color = (0, 255, 0)
            status_str = "PHONE: DETECTED"
            conf_str = f"CONF: {raw_conf:.2f}"
            ids_str = f"RAW ID: {raw_id} | LOGICAL: {logical_id}"
            tag_str = "[RECOVERED]" if is_recovered else "[FRESH YOLO]"
        elif stable_phones:
            stale_phone = stable_phones[0]
            logical_id = stale_phone.get("track_id", "N/A")
            bg_color = (0, 70, 140)
            border_color = (0, 165, 255)
            status_str = "PHONE: STALE/MISS"
            conf_str = "RAW CONF: N/A"
            ids_str = f"STALE LOGICAL ID: {logical_id}"
            tag_str = f"[STALE {stale_phone.get('missing_frames', 1)}F]"
        else:
            bg_color = (30, 30, 80)
            border_color = (0, 0, 255)
            status_str = "PHONE: NOT DETECTED"
            conf_str = "RAW CONF: N/A"
            ids_str = "NO ACTIVE TRACK"
            tag_str = "[MISSED]"

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_color, 2)

        cv2.putText(frame, status_str, (box_x1 + 10, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, tag_str, (box_x1 + 170, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1)
        cv2.putText(frame, conf_str, (box_x1 + 10, box_y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        cv2.putText(frame, ids_str, (box_x1 + 10, box_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 3. Live Evaluation Metrics Panel (Left Side Overlay)
        tf = max(1, self.total_frames)
        raw_pct = (self.raw_phone_frames / tf) * 100.0
        stab_pct = (self.stabilized_phone_frames / tf) * 100.0
        avg_conf = float(np.mean(self.phone_confidences)) if self.phone_confidences else 0.0

        p_x1, p_y1 = 10, 60
        p_x2, p_y2 = 320, 230

        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (15, 15, 15), -1)
        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 255, 0), 1)

        cv2.putText(frame, "EVALUATION METRICS", (p_x1 + 10, p_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

        m_lines = [
            f"Raw YOLO Det Rate: {raw_pct:.1f}% ({self.raw_phone_frames}/{tf})",
            f"Stabilized Det Rate: {stab_pct:.1f}% ({self.stabilized_phone_frames}/{tf})",
            f"Raw Gaps: {self.raw_phone_gaps}  | Max Gap: {self.longest_raw_gap} frames",
            f"Raw ID Changes: {self.raw_id_changes}  | Logical Changes: {self.logical_id_changes}",
            f"Avg Phone Conf: {avg_conf:.3f}",
            f"Extra Detections: {self.other_object_detections_count}"
        ]

        for idx, line in enumerate(m_lines):
            cv2.putText(frame, line, (p_x1 + 10, p_y1 + 42 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

        # 4. Render Detections Bounding Boxes
        for t in stable_tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            cname = t["class_name"]
            conf = t["confidence"]
            is_stale = t.get("is_stale", False)
            is_recovered = t.get("recovered", False)

            if cname == "cell phone":
                box_c = (0, 165, 255) if is_stale else ((255, 255, 0) if is_recovered else (0, 255, 0))
                tag = " [STALE]" if is_stale else (" [RECOVERED]" if is_recovered else "")
            else:
                box_c = (255, 0, 255)  # Magenta for non-phone objects
                tag = " [EXTRA]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_c, 2)
            lbl = f"ID {track_id} | {cname}{tag} | {conf:.2f}"
            cv2.putText(frame, lbl, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_c, 2)

        # 5. Bottom Control Banner
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (255, 255, 255), 1)
        banner_txt = "Controls: [1] Conf 0.35 | [2] Conf 0.30 | [3] Conf 0.25 | [S] Save Results | [Q/ESC] Quit"
        cv2.putText(frame, banner_txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def run(self):
        """
        Execute live camera evaluation session.
        """
        print("==========================================================")
        print(f" VisionMate Confidence Evaluation Session (Conf={self.target_confidence:.2f}) ")
        print("==========================================================")

        self.cap, cam_idx = find_working_camera()
        if self.cap is None:
            print("\nERROR: Could not open any camera.")
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("ERROR: Could not read frame from camera.")
            self.cap.release()
            return

        frame_h, frame_w = frame.shape[:2]
        print(f"Webcam active at resolution: {frame_w}x{frame_h}")

        self.initialize_pipeline(self.target_confidence, frame_w, frame_h)

        cv2.namedWindow("VisionMate - Confidence Evaluation", cv2.WINDOW_AUTOSIZE)

        fps = 0.0
        prev_time = time.time()
        start_session_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("ERROR: Failed to read camera frame.")
                    break

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
                prev_time = curr_time

                # Check duration limit if specified
                if self.target_duration > 0 and (curr_time - start_session_time) >= self.target_duration:
                    print(f"\nTarget duration of {self.target_duration}s reached. Saving results...")
                    self.save_results()
                    break

                # 1. Raw YOLO / ByteTrack Perception
                raw_tracks = self.tracker.track(frame)

                # 2. Track Recovery Layer
                recovered_tracks = self.recovery_manager.update(raw_tracks)

                # 3. Temporal Stabilizer Layer
                stable_tracks = self.stabilizer.update(recovered_tracks)

                # 4. Record Frame Metrics
                self.record_frame_metrics(raw_tracks, recovered_tracks, stable_tracks, fps)

                # 5. Render HUD Overlay
                self.draw_eval_hud(frame, raw_tracks, stable_tracks, fps)

                # 6. Display Frame
                cv2.imshow("VisionMate - Confidence Evaluation", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]: # 27 = ESC
                    print("\nUser requested exit.")
                    self.save_results()
                    break
                elif key == ord('s') or key == ord('S'):
                    print("\nManual save triggered by user.")
                    self.save_results()
                elif key == ord('1'):
                    if self.target_confidence != 0.35:
                        self.save_results()
                        self.initialize_pipeline(0.35, frame_w, frame_h)
                        start_session_time = time.time()
                elif key == ord('2'):
                    if self.target_confidence != 0.30:
                        self.save_results()
                        self.initialize_pipeline(0.30, frame_w, frame_h)
                        start_session_time = time.time()
                elif key == ord('3'):
                    if self.target_confidence != 0.25:
                        self.save_results()
                        self.initialize_pipeline(0.25, frame_w, frame_h)
                        start_session_time = time.time()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received.")
            self.save_results()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Evaluation session finished.")


def main():
    parser = argparse.ArgumentParser(description="VisionMate YOLO Detection Confidence Threshold Evaluation")
    parser.add_argument("--confidence", "-c", type=float, default=0.35, help="YOLO confidence threshold (0.35, 0.30, or 0.25)")
    parser.add_argument("--duration", "-d", type=int, default=0, help="Optional duration in seconds per test run (e.g. 45)")
    parser.add_argument("--output", "-o", type=str, default="scratch/confidence_results.json", help="Path to save evaluation results JSON")

    args = parser.parse_args()

    evaluator = ConfidenceEvaluator(
        confidence=args.confidence,
        duration=args.duration,
        results_file=args.output
    )
    evaluator.run()


if __name__ == "__main__":
    main()
