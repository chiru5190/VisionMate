import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import cv2


class ObjectDetector:
    def __init__(self, model_path="yolo26n.pt", confidence=0.5):
        """
        Initialize the YOLO object detector.

        Args:
            model_path (str): Path/name of the YOLO model.
            confidence (float): Minimum confidence threshold.
        """
        resolved_path = model_path
        if isinstance(model_path, str) and not os.path.isabs(model_path) and not os.path.exists(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(project_root, model_path)
            if os.path.exists(candidate):
                resolved_path = candidate

        self.model = YOLO(resolved_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Detect objects in a single frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            list: Detected objects.
        """

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: OpenCV frame.
            detections: Output from detect().

        Returns:
            frame: Annotated frame.
        """

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            class_name = detection["class_name"]
            confidence = detection["confidence"]

            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame
