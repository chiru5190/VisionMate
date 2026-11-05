from ultralytics import YOLO
import cv2
import os

DEFAULT_TRACKER_CONFIG = "models/visionmate_bytetrack.yaml" if os.path.exists("models/visionmate_bytetrack.yaml") else "bytetrack.yaml"


class ObjectTracker:
    def __init__(self, model_path_or_detector="yolo26n.pt", confidence=0.35, tracker_type=DEFAULT_TRACKER_CONFIG, imgsz=None):
        """
        Initialize the multi-object tracker using ByteTrack.

        Args:
            model_path_or_detector: Path to YOLO model (str), YOLO instance, or ObjectDetector instance.
            confidence (float): Minimum confidence threshold (default 0.35).
            tracker_type (str): Tracker configuration file (default 'models/visionmate_bytetrack.yaml').
            imgsz (int, optional): Inference image resolution size (default None).
        """
        if hasattr(model_path_or_detector, "model") and isinstance(model_path_or_detector.model, YOLO):
            self.model = model_path_or_detector.model
        elif isinstance(model_path_or_detector, YOLO):
            self.model = model_path_or_detector
        elif isinstance(model_path_or_detector, str):
            self.model = YOLO(model_path_or_detector)
        else:
            raise ValueError("model_path_or_detector must be a file path str, YOLO instance, or ObjectDetector instance.")

        self.confidence = confidence
        self.tracker_type = tracker_type
        self.imgsz = imgsz

    def track(self, frame, imgsz=None):
        """
        Track objects in a single video frame using ByteTrack.

        Args:
            frame: OpenCV BGR image.
            imgsz (int, optional): Override inference image resolution size.

        Returns:
            list: Tracked objects dictionaries containing:
                - track_id (int)
                - class_id (int)
                - class_name (str)
                - confidence (float)
                - bbox (tuple: x1, y1, x2, y2)
        """
        target_imgsz = imgsz if imgsz is not None else self.imgsz

        kwargs = {
            "conf": self.confidence,
            "persist": True,
            "tracker": self.tracker_type,
            "verbose": False
        }
        if target_imgsz is not None:
            kwargs["imgsz"] = target_imgsz

        results = self.model.track(
            frame,
            **kwargs
        )

        tracks = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                track_id = int(box.id[0])
                class_name = self.model.names[class_id]

                tracks.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return tracks

    def draw_tracks(self, frame, tracks):
        """
        Draw bounding boxes, labels, and track IDs on the frame.

        Args:
            frame: OpenCV frame.
            tracks: Output from track().

        Returns:
            frame: Annotated frame.
        """
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]
            class_name = track["class_name"]
            confidence = track["confidence"]

            label = f"ID {track_id} | {class_name} | {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Text background rectangle for readability
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            text_y = max(y1 - 10, text_height + 10)
            cv2.rectangle(
                frame,
                (x1, text_y - text_height - 4),
                (x1 + text_width, text_y + baseline - 4),
                (255, 0, 0),
                -1
            )

            # Text label
            cv2.putText(
                frame,
                label,
                (x1, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame
