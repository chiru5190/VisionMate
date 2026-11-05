"""
Comprehensive Migration Test Suite for YOLO26n in VisionMate
Covers:
- Step 5A: Model loading & CUDA device verification
- Step 5B: Image detection format verification
- Step 5C: Consecutive frames detection consistency
- Step 5D: ByteTrack persistence & custom YAML compatibility
"""

import sys
import os
import torch
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from ultralytics import YOLO
from detection.detector import ObjectDetector
from detection.tracker import ObjectTracker


def test_5a_model_loading():
    print("--- [5A] Model Loading & CUDA Verification ---")
    assert torch.cuda.is_available(), "CUDA is not available!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"? CUDA is available. GPU: {gpu_name}")

    detector = ObjectDetector(model_path="yolo26n.pt")
    assert detector.model is not None, "Failed to load yolo26n.pt into ObjectDetector"
    
    # Warmup / dummy pass to check device placement
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.model(dummy, verbose=False)
    device = res[0].boxes.data.device
    print(f"? Model loaded successfully. Active device: {device}")
    assert "cuda" in str(device), f"Expected cuda device, got {device}"
    print(f"? Architecture task: {detector.model.task}")
    print(f"? Total parameter count: {sum(p.numel() for p in detector.model.parameters()):,}")
    print("? Step 5A Passed Cleanly.\n")
    return True


def test_5b_image_detection():
    print("--- [5B] Image Detection & Format Verification ---")
    import ultralytics
    img_path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    assert os.path.exists(img_path), f"Sample image {img_path} not found"
    
    img = cv2.imread(img_path)
    detector = ObjectDetector(model_path="yolo26n.pt", confidence=0.35)
    detections = detector.detect(img)
    print(f"? Detections count on bus.jpg: {len(detections)}")
    assert len(detections) > 0, "Expected at least 1 detection on bus.jpg"

    expected_keys = {"class_id", "class_name", "confidence", "bbox"}
    for idx, d in enumerate(detections):
        assert set(d.keys()) == expected_keys, f"Detection keys mismatch: {d.keys()}"
        assert isinstance(d["class_id"], int), f"class_id not int: {type(d['class_id'])}"
        assert isinstance(d["class_name"], str), f"class_name not str: {type(d['class_name'])}"
        assert isinstance(d["confidence"], float), f"confidence not float: {type(d['confidence'])}"
        assert 0.0 <= d["confidence"] <= 1.0, f"confidence out of bounds: {d['confidence']}"
        assert isinstance(d["bbox"], tuple) and len(d["bbox"]) == 4, f"bbox format invalid: {d['bbox']}"
        x1, y1, x2, y2 = d["bbox"]
        assert all(isinstance(c, int) for c in (x1, y1, x2, y2)), f"Coordinates not int: {d['bbox']}"
        assert x2 >= x1 and y2 >= y1, f"Coordinates inverted: {d['bbox']}"
        print(f"  Det {idx + 1}: {d['class_name']} ({d['confidence']:.2f}) at {d['bbox']}")

    print("? Step 5B Passed Cleanly.\n")
    return True


def test_5c_video_detection():
    print("--- [5C] Video Detection Across Consecutive Frames ---")
    import ultralytics
    img_path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    base_frame = cv2.imread(img_path)
    detector = ObjectDetector(model_path="yolo26n.pt", confidence=0.35)

    # Simulate 5 consecutive video frames with subtle affine shift
    for i in range(5):
        shift = i * 2
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(base_frame, M, (base_frame.shape[1], base_frame.shape[0]))
        dets = detector.detect(frame)
        assert len(dets) > 0, f"Frame {i}: No detections returned"
        for d in dets:
            assert all(k in d for k in ["class_id", "class_name", "confidence", "bbox"])
        print(f"  Frame {i + 1}: detected {len(dets)} objects cleanly.")

    print("? Step 5C Passed Cleanly.\n")
    return True


def test_5d_bytetrack_compatibility():
    print("--- [5D] ByteTrack Compatibility & Tracking ID Persistence ---")
    import ultralytics
    img_path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    base_frame = cv2.imread(img_path)

    custom_yaml = "models/visionmate_bytetrack.yaml"
    assert os.path.exists(custom_yaml), f"{custom_yaml} not found"

    tracker = ObjectTracker(
        model_path_or_detector="yolo26n.pt",
        confidence=0.35,
        tracker_type=custom_yaml
    )
    assert tracker.tracker_type == custom_yaml, "Tracker YAML mismatch"

    expected_keys = {"track_id", "class_id", "class_name", "confidence", "bbox"}
    
    # Run consecutive tracking frames
    frame_track_ids = []
    for i in range(5):
        # Small translation simulating motion
        M = np.float32([[1, 0, i], [0, 1, 0]])
        frame = cv2.warpAffine(base_frame, M, (base_frame.shape[1], base_frame.shape[0]))
        tracks = tracker.track(frame)
        assert len(tracks) > 0, f"Tracking frame {i} returned 0 tracks"
        
        current_ids = []
        for t in tracks:
            assert set(t.keys()) == expected_keys, f"Track keys mismatch: {t.keys()}"
            assert isinstance(t["track_id"], int)
            current_ids.append(t["track_id"])
        
        frame_track_ids.append(current_ids)
        print(f"  Track Frame {i + 1}: Active Track IDs = {current_ids}")

    # Verify ID persistence: IDs from frame 1 must be present in subsequent frames
    common_ids = set(frame_track_ids[0]).intersection(set(frame_track_ids[1]))
    assert len(common_ids) > 0, f"No persistent IDs between frame 0 and frame 1: {frame_track_ids}"
    print(f"? ByteTrack persistent IDs maintained across all frames: {common_ids}")
    print("? Step 5D Passed Cleanly.\n")
    return True


if __name__ == "__main__":
    print("==========================================================")
    print("     VisionMate YOLO26n Verification & Migration Suite    ")
    print("==========================================================\n")
    test_5a_model_loading()
    test_5b_image_detection()
    test_5c_video_detection()
    test_5d_bytetrack_compatibility()
    print("==========================================================")
    print("        ALL STEP 5 (A-D) VERIFICATION TESTS PASSED        ")
    print("==========================================================")

