"""
VisionMate Live Webcam Interactive Demo Script.

Connects OpenCV webcam feed, YOLO26 + ByteTrack multi-object tracking,
2D spatial position analysis, and relative distance estimation with
real-time desktop visual overlays.
"""

import sys
import os
import time
import cv2
import numpy as np

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from detection.tracker import ObjectTracker
from spatial.position import ObjectPositionAnalyzer
from spatial.distance import DistanceEstimator


def find_working_camera():
    """
    Test camera indices 0, 1, 2, 3 in sequence and return the first working VideoCapture object.
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


def main():
    print("==========================================================")
    print("           VisionMate Live Perception Demo                ")
    print("==========================================================")

    # 1. Camera Initialization
    cap, cam_index = find_working_camera()
    if cap is None:
        print("\nERROR: Could not open any webcam (tried indices 0, 1, 2, 3).")
        print("Please check that a physical camera is connected and not in use by another app.")
        sys.exit(1)

    try:
        # Read actual resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Fallback if CAP_PROP returned 0
        if frame_width <= 0 or frame_height <= 0:
            ret, sample_frame = cap.read()
            if ret and sample_frame is not None:
                frame_height, frame_width = sample_frame.shape[:2]
            else:
                frame_width, frame_height = 640, 480

        print(f"Webcam active at resolution: {frame_width}x{frame_height}")

        # 2. Perception Modules Setup
        print("Loading YOLO26n ByteTrack Model...")
        tracker = ObjectTracker(model_path_or_detector="yolo26n.pt", confidence=0.45)
        
        position_analyzer = ObjectPositionAnalyzer(frame_width=frame_width, frame_height=frame_height)
        distance_estimator = DistanceEstimator(frame_width=frame_width, frame_height=frame_height)
        print("Perception pipeline initialized successfully.")

        print("\n----------------------------------------------------------")
        print("  Controls: Press 'Q' or 'ESC' in the video window to quit")
        print("----------------------------------------------------------\n")

        fps = 0.0
        prev_time = time.time()

        # 3. Real-Time Perception Loop
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from webcam.")
                break

            # Calculate live FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # A. Object Detection & ByteTrack Persistent Tracking
            tracks = tracker.track(frame)

            # B. Enrich tracks with 2D Position & Relative Distance
            enriched_tracks = []
            for track in tracks:
                pos_track = position_analyzer.analyze_track(track)
                full_track = distance_estimator.analyze_track(pos_track)
                enriched_tracks.append(full_track)

            # C. Render Visual Overlays on Frame
            for t in enriched_tracks:
                x1, y1, x2, y2 = t["bbox"]
                track_id = t["track_id"]
                class_name = t["class_name"]
                confidence = t["confidence"]

                horiz = t["position"]["horizontal"].upper()
                vert = t["position"]["vertical"].upper()
                dist_cat = t["distance"]["category"].upper().replace("_", " ")

                # 1. Bounding box (Bright Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. Formatted Multi-line Labels
                label_id = f"ID {track_id} | {class_name} | {confidence:.2f}"
                label_pos = f"Position: {horiz} / {vert}"
                label_dist = f"Distance: {dist_cat}"

                # Render label background boxes for maximum legibility
                base_y = max(y1 - 40, 45)

                # Draw top ID/Class label
                cv2.putText(frame, label_id, (x1, base_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                # Draw Position label
                cv2.putText(frame, label_pos, (x1, base_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
                # Draw Distance label
                cv2.putText(frame, label_dist, (x1, base_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            # 4. Header Overlay (FPS & Object Count)
            cv2.rectangle(frame, (10, 10), (220, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (220, 75), (0, 255, 0), 1)
            cv2.putText(frame, "VisionMate Demo", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Objects: {len(enriched_tracks)}", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 5. Display Window
            cv2.imshow("VisionMate - Live Demo", frame)

            # Check keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # ESC = 27
                print("\nExit key pressed by user. Shutting down...")
                break

    except Exception as e:
        print(f"\nUnexpected error during demo execution: {e}")
    finally:
        # Guarantee camera release and window destruction
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and OpenCV windows closed cleanly.")


if __name__ == "__main__":
    main()
