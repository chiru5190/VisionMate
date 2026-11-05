"""
VisionMate - Comprehensive YOLO11n vs YOLO26n Head-to-Head Benchmark
Measures:
1. Average FPS
2. YOLO inference latency (Avg, Min, Max, P95 in ms)
3. Total pipeline latency (Avg, Min, Max, P95 in ms)
4. GPU memory usage (allocated, peak, reserved in MB)
5. Object detection frequency (frames with detections, detection rate %, class breakdown)
6. Missed detection gaps (total gap events, avg gap, max gap)
7. Tracking ID stability (track count, ID fragmentation, recovery count)
8. False detections / transient detections filtered by stabilizer
"""

import sys
import os
import time
import json
import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from detection.tracker import ObjectTracker
from detection.recovery import TrackRecoveryManager
from detection.stability import TemporalDetectionStabilizer
from spatial.position import ObjectPositionAnalyzer
from spatial.distance import DistanceEstimator
from spatial.obstacle import ObstacleAnalyzer
from assistant.controller import AssistantController
from assistant.query_memory import ObjectQueryMemory


def run_benchmark_for_model(model_name: str, duration_sec: int = 60, conf_threshold: float = 0.35, imgsz: int = 640):
    print(f"\n==========================================================")
    print(f"  BENCHMARKING: {model_name} (Duration: {duration_sec}s, Imgsz: {imgsz}, Conf: {conf_threshold})")
    print(f"==========================================================")

    # Reset CUDA stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        initial_gpu_mem_mb = 0.0

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam at index 0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        frame_w, frame_h = 640, 480

    print(f"Webcam active: {frame_w}x{frame_h}")

    # Initialize full VisionMate pipeline
    tracker = ObjectTracker(
        model_path_or_detector=model_name,
        confidence=conf_threshold,
        tracker_type="models/visionmate_bytetrack.yaml",
        imgsz=imgsz
    )
    recovery_manager = TrackRecoveryManager(
        max_recovery_frames=15,
        max_center_distance=0.20,
        min_size_ratio=0.5,
        max_size_ratio=2.0,
        min_recovery_score=0.70,
        frame_width=frame_w,
        frame_height=frame_h
    )
    stabilizer = TemporalDetectionStabilizer(max_missing_frames=8, class_change_threshold=3)
    position_analyzer = ObjectPositionAnalyzer(frame_width=frame_w, frame_height=frame_h)
    distance_estimator = DistanceEstimator(frame_width=frame_w, frame_height=frame_h)
    obstacle_analyzer = ObstacleAnalyzer(min_confidence=0.40, warning_distance="close", critical_distance="very_close")
    query_memory = ObjectQueryMemory(max_history_frames=30)
    controller = AssistantController(query_memory=query_memory)

    # Warmup (10 frames)
    print("Warming up GPU pipeline (10 frames)...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            raw = tracker.track(frame)
            rec = recovery_manager.update(raw)
            stab = stabilizer.update(rec)

    # Metrics collectors
    latencies_yolo = []
    latencies_pipeline = []
    total_frames = 0
    frames_with_detections = 0
    detections_by_class = {}
    
    unique_raw_ids = set()
    unique_logical_ids = set()
    recovered_count = 0
    obstacle_warning_count = 0

    # Missed detection gaps telemetry across tracked entities
    active_track_gaps = {}  # track_id -> current_gap
    completed_gaps = []

    # Test sample assistant commands at intervals
    test_queries = ["Where is my bottle?", "Find my phone.", "Where is my laptop?", "Where is the person?"]
    query_responses = {}

    start_time = time.time()
    next_query_time = start_time + 10.0
    query_idx = 0

    print(f"Beginning {duration_sec}-second live capture and perception profiling...")

    while True:
        loop_start = time.perf_counter()
        elapsed = time.time() - start_time
        if elapsed >= duration_sec:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        total_frames += 1

        # 1. YOLO inference + ByteTrack tracking latency
        t_yolo_start = time.perf_counter()
        raw_tracks = tracker.track(frame)
        t_yolo_end = time.perf_counter()
        yolo_latency_ms = (t_yolo_end - t_yolo_start) * 1000.0
        latencies_yolo.append(yolo_latency_ms)

        # 2. Track Recovery
        recovered_tracks = recovery_manager.update(raw_tracks)

        # 3. Temporal Stabilizer
        stable_tracks = stabilizer.update(recovered_tracks)

        # 4. Spatial Analysis
        current_detections = []
        for track in stable_tracks:
            if "position" not in track:
                track = position_analyzer.analyze_track(track)
            if "distance" not in track:
                track = distance_estimator.analyze_track(track)
            current_detections.append(track)

        # 5. Query memory & Obstacle Analysis
        query_memory.update(current_detections)
        warnings = obstacle_analyzer.analyze(current_detections)
        if warnings:
            obstacle_warning_count += len(warnings)

        # 6. Periodic assistant query check
        if time.time() >= next_query_time and query_idx < len(test_queries):
            cmd = test_queries[query_idx]
            resp = controller.process_command(cmd, current_detections)
            query_responses[cmd] = resp
            query_idx += 1
            next_query_time = time.time() + 12.0

        loop_end = time.perf_counter()
        pipeline_latency_ms = (loop_end - loop_start) * 1000.0
        latencies_pipeline.append(pipeline_latency_ms)

        # Track telemetry
        if len(raw_tracks) > 0:
            frames_with_detections += 1

        for t in raw_tracks:
            unique_raw_ids.add(t["track_id"])
            cls = t["class_name"]
            detections_by_class[cls] = detections_by_class.get(cls, 0) + 1

        for t in recovered_tracks:
            unique_logical_ids.add(t.get("logical_track_id", t["track_id"]))
            if t.get("recovered", False):
                recovered_count += 1

        # Check gaps: tracks present in stabilizer vs raw detections
        raw_ids_now = {t["track_id"] for t in raw_tracks}
        for st in stable_tracks:
            tid = st["track_id"]
            if tid not in raw_ids_now:
                # Detection was missed this frame, carried by stabilizer
                active_track_gaps[tid] = active_track_gaps.get(tid, 0) + 1
            else:
                if tid in active_track_gaps and active_track_gaps[tid] > 0:
                    completed_gaps.append(active_track_gaps[tid])
                    active_track_gaps[tid] = 0

    cap.release()
    total_duration = time.time() - start_time

    # Record any ongoing gaps
    for tid, gap in active_track_gaps.items():
        if gap > 0:
            completed_gaps.append(gap)

    # Compute GPU memory
    if torch.cuda.is_available():
        gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        gpu_allocated_mb = 0.0
        gpu_peak_mb = 0.0
        gpu_reserved_mb = 0.0

    avg_fps = total_frames / total_duration if total_duration > 0 else 0.0
    detection_rate_pct = (frames_with_detections / total_frames * 100.0) if total_frames > 0 else 0.0

    yolo_avg = float(np.mean(latencies_yolo)) if latencies_yolo else 0.0
    yolo_p95 = float(np.percentile(latencies_yolo, 95)) if latencies_yolo else 0.0
    yolo_min = float(np.min(latencies_yolo)) if latencies_yolo else 0.0
    yolo_max = float(np.max(latencies_yolo)) if latencies_yolo else 0.0

    pipe_avg = float(np.mean(latencies_pipeline)) if latencies_pipeline else 0.0
    pipe_p95 = float(np.percentile(latencies_pipeline, 95)) if latencies_pipeline else 0.0
    pipe_min = float(np.min(latencies_pipeline)) if latencies_pipeline else 0.0
    pipe_max = float(np.max(latencies_pipeline)) if latencies_pipeline else 0.0

    total_gaps = len(completed_gaps)
    avg_gap = float(np.mean(completed_gaps)) if completed_gaps else 0.0
    max_gap = int(np.max(completed_gaps)) if completed_gaps else 0

    results = {
        "model": model_name,
        "duration_seconds": round(total_duration, 2),
        "total_frames": total_frames,
        "average_fps": round(avg_fps, 2),
        "yolo_latency_ms": {
            "avg": round(yolo_avg, 2),
            "min": round(yolo_min, 2),
            "max": round(yolo_max, 2),
            "p95": round(yolo_p95, 2)
        },
        "pipeline_latency_ms": {
            "avg": round(pipe_avg, 2),
            "min": round(pipe_min, 2),
            "max": round(pipe_max, 2),
            "p95": round(pipe_p95, 2)
        },
        "gpu_memory_mb": {
            "allocated": round(gpu_allocated_mb, 2),
            "peak": round(gpu_peak_mb, 2),
            "reserved": round(gpu_reserved_mb, 2)
        },
        "detection_frequency": {
            "frames_with_detections": frames_with_detections,
            "detection_rate_pct": round(detection_rate_pct, 2),
            "detections_by_class": dict(sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True))
        },
        "missed_detection_gaps": {
            "total_gaps": total_gaps,
            "avg_gap_frames": round(avg_gap, 2),
            "longest_gap_frames": max_gap
        },
        "tracking_stability": {
            "unique_raw_ids": len(unique_raw_ids),
            "unique_logical_ids": len(unique_logical_ids),
            "recovered_tracks_count": recovered_count,
            "id_fragmentation_ratio": round(len(unique_raw_ids) / max(len(unique_logical_ids), 1), 3)
        },
        "assistant_queries": query_responses,
        "obstacle_warnings_triggered": obstacle_warning_count
    }

    print(f"\n--- {model_name} RESULTS SUMMARY ---")
    print(f"Total Frames Processed  : {total_frames} ({avg_fps:.2f} FPS)")
    print(f"YOLO Inference Latency  : Avg={yolo_avg:.2f} ms | P95={yolo_p95:.2f} ms | Max={yolo_max:.2f} ms")
    print(f"Total Pipeline Latency  : Avg={pipe_avg:.2f} ms | P95={pipe_p95:.2f} ms | Max={pipe_max:.2f} ms")
    print(f"GPU Memory (Peak / Alloc): {gpu_peak_mb:.1f} MB / {gpu_allocated_mb:.1f} MB (Reserved: {gpu_reserved_mb:.1f} MB)")
    print(f"Detection Frequency     : {detection_rate_pct:.1f}% of frames ({frames_with_detections}/{total_frames})")
    print(f"Detected Classes Count  : {detections_by_class}")
    print(f"Missed Detection Gaps   : Total={total_gaps} gaps | Avg={avg_gap:.1f} frames | Longest={max_gap} frames")
    print(f"Tracking ID Stability   : Raw IDs={len(unique_raw_ids)} | Logical IDs={len(unique_logical_ids)} | Recoveries={recovered_count}")
    print(f"Obstacle Warnings       : {obstacle_warning_count}")
    print(f"Sample Query Responses  : {query_responses}")

    return results


def main():
    duration = 60
    conf = 0.35
    imgsz = 640

    print("====================================================================")
    print("      VISIONMATE COMPARATIVE BENCHMARK: YOLO11n vs YOLO26n          ")
    print("====================================================================")

    res_11n = run_benchmark_for_model("yolo11n.pt", duration_sec=duration, conf_threshold=conf, imgsz=imgsz)
    
    # Brief cooldown between benchmarks
    time.sleep(3.0)

    res_26n = run_benchmark_for_model("yolo26n.pt", duration_sec=duration, conf_threshold=conf, imgsz=imgsz)

    comparison = {
        "benchmark_parameters": {
            "duration_seconds": duration,
            "resolution": "640x480",
            "imgsz": imgsz,
            "confidence_threshold": conf,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__
        },
        "yolo11n": res_11n,
        "yolo26n": res_26n
    }

    out_file = "scratch/yolo11_vs_yolo26_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n\n" + "=" * 80)
    print(f"{'METRIC COMPARISON':<35} | {'YOLO11n':<18} | {'YOLO26n':<18} | {'DELTA':<10}")
    print("=" * 80)
    
    fps11, fps26 = res_11n["average_fps"], res_26n["average_fps"]
    fps_delta = fps26 - fps11
    print(f"{'Average FPS':<35} | {fps11:<18.2f} | {fps26:<18.2f} | {fps_delta:+6.2f} FPS")

    lat11, lat26 = res_11n["yolo_latency_ms"]["avg"], res_26n["yolo_latency_ms"]["avg"]
    lat_delta = lat26 - lat11
    print(f"{'YOLO Inference Latency (avg ms)':<35} | {lat11:<18.2f} | {lat26:<18.2f} | {lat_delta:+6.2f} ms")

    p95_11, p95_26 = res_11n["yolo_latency_ms"]["p95"], res_26n["yolo_latency_ms"]["p95"]
    p95_delta = p95_26 - p95_11
    print(f"{'YOLO Inference Latency (p95 ms)':<35} | {p95_11:<18.2f} | {p95_26:<18.2f} | {p95_delta:+6.2f} ms")

    pipe11, pipe26 = res_11n["pipeline_latency_ms"]["avg"], res_26n["pipeline_latency_ms"]["avg"]
    pipe_delta = pipe26 - pipe11
    print(f"{'Total Pipeline Latency (avg ms)':<35} | {pipe11:<18.2f} | {pipe26:<18.2f} | {pipe_delta:+6.2f} ms")

    gpu11, gpu26 = res_11n["gpu_memory_mb"]["peak"], res_26n["gpu_memory_mb"]["peak"]
    gpu_delta = gpu26 - gpu11
    print(f"{'GPU Peak Memory (MB)':<35} | {gpu11:<18.1f} | {gpu26:<18.1f} | {gpu_delta:+6.1f} MB")

    det11, det26 = res_11n["detection_frequency"]["detection_rate_pct"], res_26n["detection_frequency"]["detection_rate_pct"]
    det_delta = det26 - det11
    print(f"{'Detection Frequency (%)':<35} | {det11:<18.1f} | {det26:<18.1f} | {det_delta:+6.1f} %")

    gap11, gap26 = res_11n["missed_detection_gaps"]["total_gaps"], res_26n["missed_detection_gaps"]["total_gaps"]
    gap_delta = gap26 - gap11
    print(f"{'Missed Detection Gaps (count)':<35} | {gap11:<18} | {gap26:<18} | {gap_delta:+6d}")

    max_gap11, max_gap26 = res_11n["missed_detection_gaps"]["longest_gap_frames"], res_26n["missed_detection_gaps"]["longest_gap_frames"]
    max_gap_delta = max_gap26 - max_gap11
    print(f"{'Longest Missed Gap (frames)':<35} | {max_gap11:<18} | {max_gap26:<18} | {max_gap_delta:+6d}")

    frag11, frag26 = res_11n["tracking_stability"]["id_fragmentation_ratio"], res_26n["tracking_stability"]["id_fragmentation_ratio"]
    frag_delta = frag26 - frag11
    print(f"{'Tracking ID Fragmentation Ratio':<35} | {frag11:<18.3f} | {frag26:<18.3f} | {frag_delta:+6.3f}")

    rec11, rec26 = res_11n["tracking_stability"]["recovered_tracks_count"], res_26n["tracking_stability"]["recovered_tracks_count"]
    rec_delta = rec26 - rec11
    print(f"{'Recovered Tracks Count':<35} | {rec11:<18} | {rec26:<18} | {rec_delta:+6d}")

    print("=" * 80)
    print(f"Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()

