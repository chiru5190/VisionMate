import sys
import os

sys.path.insert(0, os.path.abspath("."))

from assistant.query_memory import ObjectQueryMemory


def run_query_memory_tests():
    print("--- VisionMate ObjectQueryMemory Test Suite ---")
    qm = ObjectQueryMemory(max_history_frames=30, frame_width=640, frame_height=480)

    # TEST 1: Object found in current frame
    t_phone = {
        "track_id": 10,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.85,
        "bbox": (100, 100, 200, 200),
        "position": {"horizontal": "center", "vertical": "middle"},
        "distance": {"category": "near"}
    }
    qm.update([t_phone])
    res1 = qm.find_object("cell phone")
    print(f"Test 1 (Found in current frame): {res1}")
    assert len(res1) == 1
    assert res1[0]["class_name"] == "cell phone"
    assert res1[0]["is_in_current_frame"] is True
    assert res1[0]["recency_frames_ago"] == 0
    assert res1[0]["observation_count"] == 1
    assert res1[0]["bbox"] == (100, 100, 200, 200)

    # TEST 2: Object temporarily missing from current frame but present in recent history
    qm.clear()
    for _ in range(5):
        qm.update([t_phone])  # Frame 1..5
    for _ in range(3):
        qm.update([])         # Frame 6..8 (phone missing)

    res2 = qm.find_object("phone")
    print(f"Test 2 (Temporarily missing object found in history): {res2}")
    assert len(res2) == 1
    assert res2[0]["class_name"] == "cell phone"
    assert res2[0]["is_in_current_frame"] is False
    assert res2[0]["recency_frames_ago"] == 3
    assert res2[0]["observation_count"] == 5

    # TEST 3: Duplicate same-object detections are merged
    qm.clear()
    for i in range(10):
        # Slightly jittering bounding box of same physical phone
        jitter_phone = dict(t_phone)
        jitter_phone["bbox"] = (100 + i % 2, 100 + i % 2, 200 + i % 2, 200 + i % 2)
        qm.update([jitter_phone])

    res3 = qm.find_object("cell phone")
    print(f"Test 3 (Duplicate same-object merged): {res3}")
    assert len(res3) == 1
    assert res3[0]["observation_count"] == 10
    assert res3[0]["is_in_current_frame"] is True

    # TEST 4: Two physically separated same-class objects remain separate
    qm.clear()
    phone_left = {
        "track_id": 10,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.85,
        "bbox": (50, 50, 150, 150),
        "position": {"horizontal": "left", "vertical": "top"}
    }
    phone_right = {
        "track_id": 20,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.90,
        "bbox": (450, 300, 550, 400),
        "position": {"horizontal": "right", "vertical": "bottom"}
    }
    qm.update([phone_left, phone_right])
    res4 = qm.find_object("cell phone")
    print(f"Test 4 (Two separated same-class objects remain separate): {res4}")
    assert len(res4) == 2
    track_ids = {r["track_id"] for r in res4}
    assert track_ids == {10, 20}

    # TEST 5: Unknown object returns no result
    res5 = qm.find_object("spaceship")
    print(f"Test 5 (Unknown object returns empty): {res5}")
    assert res5 == []

    # TEST 6: Newest / high-confidence observation is preferred
    qm.clear()
    phone_old = {
        "track_id": 10,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.60,
        "bbox": (100, 100, 200, 200)
    }
    phone_new_high_conf = {
        "track_id": 30,
        "class_id": 67,
        "class_name": "cell phone",
        "confidence": 0.95,
        "bbox": (300, 300, 400, 400)
    }
    qm.update([phone_old])       # Frame 1
    qm.update([])                # Frame 2
    qm.update([phone_new_high_conf]) # Frame 3
    res6 = qm.find_object("cell phone")
    print(f"Test 6 (Newest / high-confidence preferred): {res6}")
    assert len(res6) == 2
    assert res6[0]["track_id"] == 30  # Newest & higher confidence phone ranked first
    assert res6[0]["confidence"] == 0.95
    assert res6[1]["track_id"] == 10

    # TEST 7: Clear method resets state
    qm.clear()
    assert qm.find_object("cell phone") == []

    print("\nSUCCESS: All ObjectQueryMemory unit tests passed cleanly!")


if __name__ == "__main__":
    run_query_memory_tests()
