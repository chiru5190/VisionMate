with open("app.py", "r") as f:
    content = f.read()

# 1. Add import
if "from spatial.obstacle import ObstacleAnalyzer" not in content:
    content = content.replace(
        "from spatial.distance import DistanceEstimator",
        "from spatial.distance import DistanceEstimator\nfrom spatial.obstacle import ObstacleAnalyzer"
    )

# 2. Add overlay function
overlay_func = """def render_obstacle_overlay(frame, warning: dict):
    if not warning or not isinstance(warning, dict):
        return

    frame_h, frame_w = frame.shape[:2]
    obj_str = str(warning.get("object", "")).upper()
    pos_str = str(warning.get("position", "")).upper()
    dist_str = str(warning.get("distance", "")).upper().replace("_", " ")
    severity = str(warning.get("severity", "warning")).lower()

    text = f"{obj_str} - {pos_str} - {dist_str}"

    if severity == "critical":
        bg_color = (0, 0, 160)
        border_color = (0, 0, 255)
        text_color = (0, 255, 255)
        header_text = "[CRITICAL OBSTACLE]"
    else:
        bg_color = (0, 75, 150)
        border_color = (0, 165, 255)
        text_color = (255, 255, 255)
        header_text = "[OBSTACLE WARNING]"

    box_w = max(320, int(len(text) * 11) + 20)
    x2 = frame_w - 10
    x1 = max(230, x2 - box_w)
    y1 = 10
    y2 = 65

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    cv2.putText(frame, header_text, (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_color, 2)
    cv2.putText(frame, text, (x1 + 10, y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1)


"""

if "def render_obstacle_overlay" not in content:
    content = content.replace("def render_response_overlay", overlay_func + "def render_response_overlay")

# 3. Init analyzer in main()
if "obstacle_analyzer = ObstacleAnalyzer" not in content:
    content = content.replace(
        'distance_estimator = DistanceEstimator(frame_width=frame_width, frame_height=frame_height)',
        'distance_estimator = DistanceEstimator(frame_width=frame_width, frame_height=frame_height)\n        obstacle_analyzer = ObstacleAnalyzer(\n            min_confidence=0.40,\n            warning_distance="close",\n            critical_distance="very_close"\n        )'
    )

# 4. Call obstacle_analyzer.analyze(current_detections) and render overlay
if "obstacle_warnings = obstacle_analyzer.analyze" not in content:
    content = content.replace(
        "query_memory.update(current_detections)",
        "query_memory.update(current_detections)\n\n            # Analyze proximity obstacles\n            obstacle_warnings = obstacle_analyzer.analyze(current_detections)"
    )

if "render_obstacle_overlay(frame" not in content:
    content = content.replace(
        "if active_response:\n                render_response_overlay(frame, active_response)",
        "if obstacle_warnings:\n                render_obstacle_overlay(frame, obstacle_warnings[0])\n\n            if active_response:\n                render_response_overlay(frame, active_response)"
    )

with open("app.py", "w") as f:
    f.write(content)

print("app.py updated successfully!")
