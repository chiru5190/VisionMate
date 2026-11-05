"""
VisionMate - Performance & Quality Profiler Script

Measures execution time per frame for each component in the VisionMate webcam pipeline:
1. YOLO11n detection/tracking
2. TrackRecoveryManager
3. TemporalDetectionStabilizer
4. PositionAnalyzer
5. DistanceEstimator
6. Assistant processing
7. Camera read & GUI rendering
8. Total frame processing time

Also collects detection quality and stability metrics:
- Raw phone detected frames & rate %
- Stabilized phone detected frames & rate %
- Recovered phone frames
- Raw phone detection gaps & longest gap
- Raw track ID changes vs logical track ID changes
- Average & minimum phone confidence
- Total extra object detections & breakdown by class
"""

import sys
import os
import time
import json
import argparse
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from detection.tracker import ObjectTracker
from detection.recovery import TrackRecoveryManager
from detection.stability import TemporalDetectionStabilizer
from spatial.position import ObjectPositionAnalyzer
from spatial.distance import DistanceEstimator
from assistant.controller import AssistantController


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


def run_profiler(duration: int = 60, confidence_threshold: float = 0.25, imgsz: int = 640, output_file: str = None, model_path: str = "yolo26n.pt"):
    eff_imgsz = imgsz if imgsz is not None else 640
    if output_file is None:
        output_file = f"scratch/profiling_quality_{eff_imgsz}.json"

    print("==========================================================")
    print("        VisionMate Performance & Quality Profiler         ")
    print("==========================================================")
    print("Target Duration  : " + str(duration) + " seconds")
    print("YOLO Confidence  : " + str(confidence_threshold))
    print("YOLO Image Size  : " + str(eff_imgsz))
    print("Output Results   : " + str(output_file))
    print("----------------------------------------------------------")

    cap, cam_idx = find_working_camera()
    if cap is None:
        print("\nERROR: Could not open any webcam (tried indices 0, 1, 2, 3).")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        ret, sample_frame = cap.read()
        if ret and sample_frame is not None:
            frame_height, frame_width = sample_frame.shape[:2]
        else:
            frame_width, frame_height = 640, 480

    print(f"Webcam active at resolution: {frame_width}x{frame_height}")

    print("Initializing pipeline components...")
    tracker = ObjectTracker(
        model_path_or_detector=model_path,
        confidence=confidence_threshold,
        tracker_type="models/visionmate_bytetrack.yaml",
        imgsz=eff_imgsz
    )
    recovery_manager = TrackRecoveryManager(
        max_recovery_frames=15,
        max_center_distance=0.20,
        min_size_ratio=0.5,
        max_size_ratio=2.0,
        min_recovery_score=0.70,
        frame_width=frame_width,
        frame_height=frame_height
    )
    stabilizer = TemporalDetectionStabilizer(max_missing_frames=8, class_change_threshold=3)
    position_analyzer = ObjectPositionAnalyzer(frame_width=frame_width, frame_height=frame_height)
    distance_estimator = DistanceEstimator(frame_width=frame_width, frame_height=frame_height)
    assistant_controller = AssistantController()

    print("Components initialized successfully. Warming up pipeline (10 frames)...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            raw = tracker.track(frame)
            rec = recovery_manager.update(raw)
            stab = stabilizer.update(rec)

    # Timing metrics
    timings = {
        "camera_read": [],
        "yolo_detection_tracking": [],
        "track_recovery_manager": [],
        "temporal_detection_stabilizer": [],
        "position_analyzer": [],
        "distance_estimator": [],
        "assistant_processing": [],
        "render_and_display": [],
        "total_frame": []
    }

    # Quality Telemetry State
    total_frames = 0
    raw_phone_frames = 0
    stabilized_phone_frames = 0
    recovered_phone_frames = 0

    raw_phone_gaps = 0
    current_raw_gap = 0
    longest_raw_gap = 0
    in_raw_gap = False

    last_raw_phone_id = None
    raw_id_changes = 0

    last_logical_phone_id = None
    logical_id_changes = 0

    phone_confidences = []
    other_object_detections_count = 0
    other_object_counts_by_class = {}

    start_wall_time = time.time()

    print(f"\n>>> Profiling session started. Running for approximately {duration} seconds...\n")

    try:
        while True:
            elapsed_wall = time.time() - start_wall_time
            if elapsed_wall >= duration:
                print(f"\nReached target duration of {duration}s ({elapsed_wall:.2f}s elapsed). Stopping capture.")
                break

            t_frame_start = time.perf_counter()

            # 0. Camera Read
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t_camera = (time.perf_counter() - t0) * 1000.0

            if not ret or frame is None:
                print("Warning: Failed to read frame from webcam.")
                continue

            total_frames += 1

            # 1. YOLO11n Detection / Tracking
            t0 = time.perf_counter()
            raw_tracks = tracker.track(frame)
            t_yolo = (time.perf_counter() - t0) * 1000.0

            # 2. TrackRecoveryManager
            t0 = time.perf_counter()
            recovered_tracks = recovery_manager.update(raw_tracks)
            t_recovery = (time.perf_counter() - t0) * 1000.0

            # 3. TemporalDetectionStabilizer
            t0 = time.perf_counter()
            stable_tracks = stabilizer.update(recovered_tracks)
            t_stability = (time.perf_counter() - t0) * 1000.0

            # 4. PositionAnalyzer
            t0 = time.perf_counter()
            current_detections = []
            for track in stable_tracks:
                if "position" not in track:
                    track = position_analyzer.analyze_track(track)
                current_detections.append(track)
            t_position = (time.perf_counter() - t0) * 1000.0

            # 5. DistanceEstimator
            t0 = time.perf_counter()
            for track in current_detections:
                if "distance" not in track:
                    track = distance_estimator.analyze_track(track)
            t_distance = (time.perf_counter() - t0) * 1000.0

            # 6. Assistant Processing
            t0 = time.perf_counter()
            dummy_resp = None
            t_assistant = (time.perf_counter() - t0) * 1000.0

            # --- QUALITY METRICS ACCUMULATION ---
            # Raw phone detections
            raw_phone_tracks = [t for t in raw_tracks if str(t.get("class_name", "")).strip().lower() == "cell phone"]
            if raw_phone_tracks:
                raw_phone_frames += 1
                primary_raw = max(raw_phone_tracks, key=lambda x: x.get("confidence", 0.0))
                phone_confidences.append(primary_raw.get("confidence", 0.0))
                raw_id = primary_raw.get("track_id")

                if in_raw_gap:
                    in_raw_gap = False
                current_raw_gap = 0

                if last_raw_phone_id is not None and raw_id != last_raw_phone_id:
                    raw_id_changes += 1
                last_raw_phone_id = raw_id
            else:
                if last_raw_phone_id is not None:
                    if not in_raw_gap:
                        in_raw_gap = True
                        raw_phone_gaps += 1
                    current_raw_gap += 1
                    if current_raw_gap > longest_raw_gap:
                        longest_raw_gap = current_raw_gap

            # Recovered phone detections
            rec_phone_tracks = [t for t in recovered_tracks if str(t.get("class_name", "")).strip().lower() == "cell phone" and t.get("recovered", False)]
            if rec_phone_tracks:
                recovered_phone_frames += 1

            # Stabilized phone detections
            stable_phone_tracks = [t for t in stable_tracks if str(t.get("class_name", "")).strip().lower() == "cell phone"]
            if stable_phone_tracks:
                stabilized_phone_frames += 1
                logical_id = stable_phone_tracks[0].get("track_id")
                if last_logical_phone_id is not None and logical_id != last_logical_phone_id:
                    logical_id_changes += 1
                last_logical_phone_id = logical_id

            # Extra object detections
            raw_other_tracks = [t for t in raw_tracks if str(t.get("class_name", "")).strip().lower() != "cell phone"]
            other_object_detections_count += len(raw_other_tracks)
            for ot in raw_other_tracks:
                cname = str(ot.get("class_name", "unknown")).strip().lower()
                other_object_counts_by_class[cname] = other_object_counts_by_class.get(cname, 0) + 1

            # 7. Render & Display
            t0 = time.perf_counter()
            for t in current_detections:
                x1, y1, x2, y2 = t["bbox"]
                track_id = t["track_id"]
                class_name = t["class_name"]
                det_conf = t["confidence"]
                is_stale = t.get("is_stale", False)
                is_recovered = t.get("recovered", False)

                box_color = (0, 165, 255) if is_stale else ((255, 255, 0) if is_recovered else (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID {track_id} | {class_name}", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            cv2.rectangle(frame, (10, 10), (260, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"PROFILING... Frame: {total_frames}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("VisionMate Performance & Quality Profiler", frame)
            cv2.waitKey(1)
            t_render = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_frame_start) * 1000.0

            timings["camera_read"].append(t_camera)
            timings["yolo_detection_tracking"].append(t_yolo)
            timings["track_recovery_manager"].append(t_recovery)
            timings["temporal_detection_stabilizer"].append(t_stability)
            timings["position_analyzer"].append(t_position)
            timings["distance_estimator"].append(t_distance)
            timings["assistant_processing"].append(t_assistant)
            timings["render_and_display"].append(t_render)
            timings["total_frame"].append(t_total)

            if total_frames % 150 == 0:
                cur_fps = total_frames / elapsed_wall
                print(f"[{elapsed_wall:.1f}s / {duration}s] Processed {total_frames} frames | Live FPS: {cur_fps:.2f}")

    except KeyboardInterrupt:
        print("\nProfiling interrupted by user.")
    finally:
        total_session_time = time.time() - start_wall_time
        cap.release()
        cv2.destroyAllWindows()

    if total_frames == 0:
        print("ERROR: No frames were processed.")
        return

    avg_fps = total_frames / total_session_time
    avg_total_ms = float(np.mean(timings["total_frame"]))

    raw_phone_pct = round((raw_phone_frames / total_frames) * 100.0, 2)
    stabilized_phone_pct = round((stabilized_phone_frames / total_frames) * 100.0, 2)
    avg_phone_conf = round(float(np.mean(phone_confidences)), 4) if phone_confidences else 0.0
    min_phone_conf = round(float(np.min(phone_confidences)), 4) if phone_confidences else 0.0

    sorted_other_classes = dict(sorted(other_object_counts_by_class.items(), key=lambda item: item[1], reverse=True))

    results_summary = {
        "model": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_threshold": confidence_threshold,
        "imgsz": eff_imgsz,
        "session_duration_seconds": round(total_session_time, 2),
        "total_frames": total_frames,
        "average_fps": round(avg_fps, 2),
        "raw_phone_detected_frames": raw_phone_frames,
        "raw_phone_detection_rate_pct": raw_phone_pct,
        "stabilized_phone_detected_frames": stabilized_phone_frames,
        "stabilized_phone_detection_rate_pct": stabilized_phone_pct,
        "recovered_phone_frames": recovered_phone_frames,
        "raw_phone_detection_gaps": raw_phone_gaps,
        "longest_raw_gap_frames": longest_raw_gap,
        "raw_track_id_changes": raw_id_changes,
        "logical_track_id_changes": logical_id_changes,
        "average_phone_confidence": avg_phone_conf,
        "minimum_phone_confidence": min_phone_conf,
        "total_extra_object_detections": other_object_detections_count,
        "extra_detections_by_class": sorted_other_classes,
        "performance_components": {}
    }

    component_keys = [
        ("1. YOLO detection/tracking", "yolo_detection_tracking"),
        ("2. TrackRecoveryManager", "track_recovery_manager"),
        ("3. TemporalDetectionStabilizer", "temporal_detection_stabilizer"),
        ("4. PositionAnalyzer", "position_analyzer"),
        ("5. DistanceEstimator", "distance_estimator"),
        ("6. Assistant processing", "assistant_processing"),
        ("7. Camera Read (Webcam I/O)", "camera_read"),
        ("8. Render & GUI Display", "render_and_display"),
        ("9. Total frame processing time", "total_frame")
    ]

    print("\n" + "=" * 75)
    print("           VISIONMATE PERFORMANCE & QUALITY PROFILING REPORT           ")
    print("=" * 75)
    print(f"Session Duration : {total_session_time:.2f} seconds")
    print(f"Total Frames     : {total_frames}")
    print(f"Average FPS      : {avg_fps:.2f} FPS")
    print(f"YOLO Confidence  : {confidence_threshold:.2f}")
    print(f"Image Size (imgsz): {eff_imgsz}")
    print("-" * 75)
    print("PERFORMANCE TIMINGS:")
    print(f"{'Component':<35} | {'Avg (ms)':<10} | {'Max (ms)':<10} | {'% Total':<8}")
    print("-" * 75)

    for label, key in component_keys:
        arr = timings[key]
        avg_ms = float(np.mean(arr))
        max_ms = float(np.max(arr))
        pct = (avg_ms / avg_total_ms * 100.0) if avg_total_ms > 0 else 0.0

        print(f"{label:<35} | {avg_ms:<10.3f} | {max_ms:<10.3f} | {pct:<8.2f}%")

        results_summary["performance_components"][label] = {
            "avg_ms": round(avg_ms, 4),
            "max_ms": round(max_ms, 4),
            "percentage_of_total_frame": round(pct, 2)
        }

    print("-" * 75)
    print("DETECTION QUALITY & STABILITY METRICS:")
    print(f"  Raw Phone Detected Frames      : {raw_phone_frames} / {total_frames} ({raw_phone_pct}%)")
    print(f"  Stabilized Phone Frames        : {stabilized_phone_frames} / {total_frames} ({stabilized_phone_pct}%)")
    print(f"  Recovered Phone Frames         : {recovered_phone_frames}")
    print(f"  Raw Phone Detection Gaps       : {raw_phone_gaps} (Longest Gap: {longest_raw_gap} frames)")
    print(f"  Raw Track ID Changes           : {raw_id_changes}")
    print(f"  Logical Track ID Changes       : {logical_id_changes}")
    print(f"  Phone Confidence (Avg / Min)   : {avg_phone_conf:.4f} / {min_phone_conf:.4f}")
    print(f"  Total Extra Object Detections  : {other_object_detections_count}")
    top_extras = list(sorted_other_classes.items())[:5]
    print(f"  Top Extra Detection Classes    : {top_extras}")
    print("=" * 75 + "\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"Profiling quality report saved to: {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionMate Performance & Quality Profiler")
    parser.add_argument("--duration", type=int, default=60, help="Profiling duration in seconds (default 60)")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default 640)")
    parser.add_argument("--output", type=str, default=None, help="Path to output json report")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO model weights file (default yolo26n.pt)")
    args = parser.parse_args()

    out_file = args.output if args.output else f"scratch/profiling_quality_{args.imgsz}.json"

    run_profiler(duration=args.duration, confidence_threshold=args.confidence, imgsz=args.imgsz, output_file=out_file, model_path=args.model)
