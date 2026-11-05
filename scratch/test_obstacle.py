"""
Unit test suite for ObstacleAnalyzer in VisionMate spatial package.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath("."))

from spatial.obstacle import ObstacleAnalyzer, DEFAULT_OBSTACLE_CLASSES


class TestObstacleAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ObstacleAnalyzer(
            min_confidence=0.40,
            warning_distance="close",
            critical_distance="very_close"
        )

    def test_1_confidence_filtering(self):
        """Detections with confidence below min_confidence should be ignored."""
        detections = [
            {
                "class_name": "chair",
                "confidence": 0.35,  # Below 0.40
                "position": {"horizontal": "left"},
                "distance": {"category": "very_close", "proximity_score": 0.9}
            },
            {
                "class_name": "chair",
                "confidence": 0.45,  # Above 0.40
                "position": {"horizontal": "right"},
                "distance": {"category": "very_close", "proximity_score": 0.9}
            }
        ]

        warnings = self.analyzer.analyze(detections)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["position"], "right")

    def test_2_distance_filtering(self):
        """Far and medium objects should produce no warning; close produces warning; very_close produces critical."""
        detections = [
            {
                "class_name": "person",
                "confidence": 0.80,
                "position": {"horizontal": "center"},
                "distance": {"category": "far", "proximity_score": 0.1}
            },
            {
                "class_name": "couch",
                "confidence": 0.85,
                "position": {"horizontal": "left"},
                "distance": {"category": "medium", "proximity_score": 0.4}
            },
            {
                "class_name": "table",
                "confidence": 0.75,
                "position": {"horizontal": "right"},
                "distance": {"category": "close", "proximity_score": 0.7}
            },
            {
                "class_name": "chair",
                "confidence": 0.90,
                "position": {"horizontal": "center"},
                "distance": {"category": "very_close", "proximity_score": 0.95}
            }
        ]

        warnings = self.analyzer.analyze(detections)
        self.assertEqual(len(warnings), 2)

        # Ordering should be proximity score descending: chair (0.95) then table (0.7)
        self.assertEqual(warnings[0]["object"], "chair")
        self.assertEqual(warnings[0]["severity"], "critical")
        self.assertEqual(warnings[0]["message_key"], "obstacle_very_close")
        self.assertEqual(warnings[0]["distance"], "very_close")

        self.assertEqual(warnings[1]["object"], "table")
        self.assertEqual(warnings[1]["severity"], "warning")
        self.assertEqual(warnings[1]["message_key"], "obstacle_close")
        self.assertEqual(warnings[1]["distance"], "close")

    def test_3_non_obstacle_filtering(self):
        """Objects not in obstacle classes (e.g. cell phone, cup) should be ignored."""
        detections = [
            {
                "class_name": "cell phone",
                "confidence": 0.95,
                "position": {"horizontal": "center"},
                "distance": {"category": "very_close", "proximity_score": 0.98}
            },
            {
                "class_name": "cup",
                "confidence": 0.88,
                "position": {"horizontal": "left"},
                "distance": {"category": "close", "proximity_score": 0.75}
            }
        ]

        warnings = self.analyzer.analyze(detections)
        self.assertEqual(warnings, [])

    def test_4_configurable_obstacle_classes(self):
        """Custom obstacle classes configuration should be respected."""
        custom_analyzer = ObstacleAnalyzer(
            obstacle_classes=["cell phone", "laptop"]
        )

        detections = [
            {
                "class_name": "cell phone",
                "confidence": 0.80,
                "position": {"horizontal": "left"},
                "distance": {"category": "close", "proximity_score": 0.7}
            },
            {
                "class_name": "chair",  # Default obstacle, but not in custom list
                "confidence": 0.80,
                "position": {"horizontal": "right"},
                "distance": {"category": "very_close", "proximity_score": 0.9}
            }
        ]

        warnings = custom_analyzer.analyze(detections)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["object"], "cell phone")

    def test_5_empty_and_invalid_inputs(self):
        """Empty or invalid detection inputs should return an empty list []."""
        self.assertEqual(self.analyzer.analyze([]), [])
        self.assertEqual(self.analyzer.analyze(None), [])
        self.assertEqual(self.analyzer.analyze([None, "invalid"]), [])

    def test_6_proximity_ordering(self):
        """Multiple warnings should be returned ordered by proximity_score descending."""
        detections = [
            {
                "class_name": "person",
                "confidence": 0.80,
                "position": {"horizontal": "center"},
                "distance": {"category": "close", "proximity_score": 0.65}
            },
            {
                "class_name": "car",
                "confidence": 0.85,
                "position": {"horizontal": "left"},
                "distance": {"category": "very_close", "proximity_score": 0.99}
            },
            {
                "class_name": "bicycle",
                "confidence": 0.70,
                "position": {"horizontal": "right"},
                "distance": {"category": "close", "proximity_score": 0.75}
            }
        ]

        warnings = self.analyzer.analyze(detections)
        self.assertEqual(len(warnings), 3)
        # Expected order: car (0.99), bicycle (0.75), person (0.65)
        self.assertEqual(warnings[0]["object"], "car")
        self.assertEqual(warnings[1]["object"], "bicycle")
        self.assertEqual(warnings[2]["object"], "person")


if __name__ == "__main__":
    unittest.main()
