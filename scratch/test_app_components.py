import sys
import os
import py_compile

sys.path.insert(0, os.path.abspath("."))

def validate_app_components():
    print("--- VisionMate App Component Validation Suite ---")

    # 1. Compile app.py syntax check
    app_path = "app.py"
    py_compile.compile(app_path, doraise=True)
    print("? app.py syntax is valid.")

    # 2. Test Component Imports
    from detection.tracker import ObjectTracker
    from detection.recovery import TrackRecoveryManager
    from detection.stability import TemporalDetectionStabilizer
    from spatial.position import ObjectPositionAnalyzer
    from spatial.distance import DistanceEstimator
    from spatial.obstacle import ObstacleAnalyzer
    from assistant.controller import AssistantController
    from assistant.query_memory import ObjectQueryMemory

    print("? Component imports succeeded.")

    # 3. Test Tracker & Recovery Initialization
    tracker = ObjectTracker(model_path_or_detector="yolo26n.pt", confidence=0.35)
    assert tracker.model is not None, "Tracker model failed to load."
    recovery_manager = TrackRecoveryManager(max_recovery_frames=15, frame_width=640, frame_height=480)
    stabilizer = TemporalDetectionStabilizer(max_missing_frames=8, class_change_threshold=3)
    print("? ObjectTracker, TrackRecoveryManager, & TemporalDetectionStabilizer initialized.")

    # 4. Test Spatial Analyzers Initialization
    pos_analyzer = ObjectPositionAnalyzer(frame_width=640, frame_height=480)
    dist_estimator = DistanceEstimator(frame_width=640, frame_height=480)
    obstacle_analyzer = ObstacleAnalyzer(min_confidence=0.40, warning_distance="close", critical_distance="very_close")
    print("? ObjectPositionAnalyzer, DistanceEstimator, & ObstacleAnalyzer initialized.")

    # 5. Test Pipeline Flow Integration
    raw_tracks = [{"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 200, 200)}]
    recovered = recovery_manager.update(raw_tracks)
    assert len(recovered) == 1 and recovered[0]["logical_track_id"] == 10
    stable = stabilizer.update(recovered)
    assert len(stable) == 1 and stable[0]["track_id"] == 10
    
    # Enrich with position & distance
    track_pos = pos_analyzer.analyze_track(stable[0])
    track_dist = dist_estimator.analyze_track(track_pos)
    
    # Test Obstacle Analysis
    warnings = obstacle_analyzer.analyze([track_dist])
    assert isinstance(warnings, list)
    print("? Pipeline flow (ByteTrack -> Recovery -> Stabilizer -> Spatial -> Obstacle) executed cleanly.")

    # 6. Test AssistantController Initialization & Execution
    controller = AssistantController()
    res = controller.process_command("Find my bottle", [])
    assert res == "I can't see your bottle right now.", f"Unexpected response: {res}"
    print("? AssistantController initialized and processed command cleanly.")

    print("\nSUCCESS: Static/component validation passed for app.py and obstacle analyzer integration!")


if __name__ == "__main__":
    validate_app_components()
