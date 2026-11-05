import sys
import os

sys.path.insert(0, os.path.abspath("."))

from detection.recovery import TrackRecoveryManager


def run_tests():
    print("--- VisionMate Track Recovery Manager Test Suite ---")
    rm = TrackRecoveryManager(max_recovery_frames=3, frame_width=640, frame_height=480)

    # TEST 1 — New track
    t10_phone = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 200, 200)}
    res1 = rm.update([t10_phone])
    print(f"Test 1 (New track): {res1}")
    assert len(res1) == 1
    assert res1[0]["logical_track_id"] == 10
    assert res1[0]["track_id"] == 10
    assert res1[0]["recovered"] is False

    # TEST 2 — Continuous same ID
    res2 = rm.update([t10_phone])
    print(f"Test 2 (Continuous same ID): {res2}")
    assert len(res2) == 1
    assert res2[0]["logical_track_id"] == 10

    # TEST 3 — Brief ID replacement
    rm.clear()
    rm.update([t10_phone])  # Frame 1: active ID 10
    rm.update([])           # Frame 2: lost ID 10
    t20_phone = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (105, 105, 205, 205)}
    res3 = rm.update([t20_phone]) # Frame 3: raw ID 20 appears nearby
    print(f"Test 3 (Brief ID replacement): {res3}")
    assert len(res3) == 1
    assert res3[0]["logical_track_id"] == 10
    assert res3[0]["track_id"] == 10
    assert res3[0]["original_track_id"] == 20
    assert res3[0]["recovered"] is True

    # TEST 4 — Different class
    rm.clear()
    rm.update([t10_phone])
    rm.update([])
    t20_bottle = {"track_id": 20, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (105, 105, 205, 205)}
    res4 = rm.update([t20_bottle])
    print(f"Test 4 (Different class reject): {res4}")
    assert len(res4) == 1
    assert res4[0]["logical_track_id"] == 20
    assert res4[0]["recovered"] is False

    # TEST 5 — Too far away
    rm.clear()
    rm.update([t10_phone])
    rm.update([])
    t20_far_phone = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (500, 350, 600, 450)}
    res5 = rm.update([t20_far_phone])
    print(f"Test 5 (Too far away reject): {res5}")
    assert len(res5) == 1
    assert res5[0]["logical_track_id"] == 20
    assert res5[0]["recovered"] is False

    # TEST 6 — Huge size difference
    rm.clear()
    t10_small = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 140, 140)} # area 1600
    rm.update([t10_small])
    rm.update([])
    t20_huge = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 400, 400)}  # area 90000
    res6 = rm.update([t20_huge])
    print(f"Test 6 (Huge size difference reject): {res6}")
    assert len(res6) == 1
    assert res6[0]["logical_track_id"] == 20
    assert res6[0]["recovered"] is False

    # TEST 7 — Multiple objects
    rm.clear()
    t10_bot = {"track_id": 10, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (50, 50, 100, 100)}
    t20_ph = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (300, 300, 400, 400)}
    rm.update([t10_bot, t20_ph])
    rm.update([])  # Both lost

    t30_bot = {"track_id": 30, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (55, 55, 105, 105)}
    t40_ph = {"track_id": 40, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (305, 305, 405, 405)}
    res7 = rm.update([t30_bot, t40_ph])
    print(f"Test 7 (Multiple objects independent recovery): {res7}")
    assert len(res7) == 2
    bot_res = next(t for t in res7 if t["class_name"] == "bottle")
    ph_res = next(t for t in res7 if t["class_name"] == "cell phone")
    assert bot_res["logical_track_id"] == 10 and bot_res["recovered"] is True
    assert ph_res["logical_track_id"] == 20 and ph_res["recovered"] is True

    # TEST 8 — One-to-one matching
    rm.clear()
    rm.update([t10_bot])
    rm.update([])
    t30_bot = {"track_id": 30, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (55, 55, 105, 105)}
    t31_bot = {"track_id": 31, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (58, 58, 108, 108)}
    res8 = rm.update([t30_bot, t31_bot])
    print(f"Test 8 (One-to-one matching): {res8}")
    assert len(res8) == 2
    # Only one can inherit logical ID 10; the other starts fresh
    logical_ids = [t["logical_track_id"] for t in res8]
    assert 10 in logical_ids
    assert len(set(logical_ids)) == 2

    # TEST 9 — Recovery expiration
    rm.clear()
    rm_exp = TrackRecoveryManager(max_recovery_frames=2, frame_width=640, frame_height=480)
    rm_exp.update([t10_phone])
    rm_exp.update([])  # frames_since_seen = 1
    rm_exp.update([])  # frames_since_seen = 2
    rm_exp.update([])  # frames_since_seen = 3 (> 2 -> expired)
    res9 = rm_exp.update([t20_phone])
    print(f"Test 9 (Expiration after max_recovery_frames=2): {res9}")
    assert len(res9) == 1
    assert res9[0]["logical_track_id"] == 20
    assert res9[0]["recovered"] is False

    # TEST 10 — Recovery after 1-frame gap
    rm.clear()
    rm.update([t10_phone])
    rm.update([])
    res10 = rm.update([t20_phone])
    print(f"Test 10 (1-frame gap recovery): {res10}")
    assert len(res10) == 1
    assert res10[0]["logical_track_id"] == 10
    assert res10[0]["recovered"] is True

    # TEST 11 — Position change within threshold
    rm.clear()
    rm.update([t10_phone]) # bbox (100, 100, 200, 200) -> center (150, 150)
    rm.update([])
    t20_moved = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (140, 120, 240, 220)} # center (190, 170) -> dx=40/640=0.06, dy=20/480=0.04
    res11 = rm.update([t20_moved])
    print(f"Test 11 (Position change recovery): {res11}")
    assert len(res11) == 1
    assert res11[0]["logical_track_id"] == 10
    assert res11[0]["recovered"] is True

    # TEST 12 — Invalid input
    rm.clear()
    assert rm.update(None) == []
    assert rm.update([]) == []
    invalid_t = {"class_name": "bottle", "confidence": 0.80, "bbox": (10, 10, 20, 20)}
    assert rm.update([invalid_t]) == []
    print("Test 12 (Invalid input safety): Passed cleanly.")

    print("\nSUCCESS: All 12 recovery manager unit tests passed cleanly!")


if __name__ == "__main__":
    run_tests()
