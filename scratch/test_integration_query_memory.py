"""
Integration test suite for ObjectQueryMemory + AssistantController + ObjectResponseGenerator.

Tests end-to-end functionality for:
1. Current object query
2. Temporarily missing object
3. Duplicate same-object detections
4. Two separated same-class objects
5. Unknown object
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath("."))

from assistant.query_memory import ObjectQueryMemory
from assistant.controller import AssistantController
from assistant.responses import ObjectResponseGenerator


class TestQueryMemoryIntegration(unittest.TestCase):

    def setUp(self):
        self.memory = ObjectQueryMemory(max_history_frames=30)
        self.controller = AssistantController(query_memory=self.memory)

    def test_1_current_object_query(self):
        """Test querying an object currently visible in the active frame."""
        detections = [
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.88,
                "track_id": 1,
                "bbox": (100, 100, 200, 300),
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            }
        ]
        
        # Process frame
        self.memory.update(detections)

        # User asks for phone
        response = self.controller.process_command("Where is my phone?")
        
        self.assertIn("I found your cell phone", response)
        self.assertIn("left", response)
        self.assertNotIn("recently saw", response)

    def test_2_temporarily_missing_object(self):
        """Test querying an object that was seen recently but is missing from the current frame."""
        past_detections = [
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.85,
                "track_id": 1,
                "bbox": (100, 100, 200, 300),
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            }
        ]
        
        # Update memory for 5 frames with phone visible
        for _ in range(5):
            self.memory.update(past_detections)

        # Phone disappears for next 3 frames
        empty_frame = []
        for _ in range(3):
            self.memory.update(empty_frame)

        # User asks for phone
        response = self.controller.process_command("Where is my cell phone?")

        # Response must clearly indicate recent observation, not current visibility
        self.assertIn("I recently saw your cell phone", response)
        self.assertIn("left", response)

    def test_3_duplicate_same_object_detections(self):
        """Test that multiple duplicate bounding boxes for the same physical object are merged."""
        duplicate_detections = [
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.80,
                "track_id": 1,
                "bbox": (100, 100, 200, 300),
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            },
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.75,
                "track_id": 1,  # Same track ID / overlapping bbox
                "bbox": (105, 102, 198, 298),
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            }
        ]
        
        self.memory.update(duplicate_detections)
        response = self.controller.process_command("Find my phone")

        # Should report as 1 phone, NOT 2 phones
        self.assertIn("I found your cell phone", response)
        self.assertNotIn("I found 2", response)

    def test_4_two_separated_same_class_objects(self):
        """Test that two clearly separated objects of the same class remain separate."""
        separated_detections = [
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.85,
                "track_id": 1,
                "bbox": (50, 100, 150, 300),  # Far left
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            },
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.80,
                "track_id": 2,
                "bbox": (450, 100, 550, 300),  # Far right
                "position": {"horizontal": "right", "vertical": "middle"},
                "distance": {"category": "close"}
            }
        ]

        self.memory.update(separated_detections)
        response = self.controller.process_command("Where are my phones?")

        # Should report 2 distinct phones
        self.assertIn("I found 2 cell phones", response)
        self.assertIn("left", response)
        self.assertIn("right", response)

    def test_5_unknown_object(self):
        """Test querying an object that has not been observed."""
        detections = [
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.85,
                "track_id": 1,
                "bbox": (100, 100, 200, 300),
                "position": {"horizontal": "left", "vertical": "middle"},
                "distance": {"category": "close"}
            }
        ]
        self.memory.update(detections)

        # Query for bottle (not in memory)
        response = self.controller.process_command("Where is my bottle?")
        self.assertEqual(response, "I can't see your bottle right now.")


if __name__ == '__main__':
    unittest.main()
