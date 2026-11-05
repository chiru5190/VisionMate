import sys
import os

sys.path.insert(0, os.path.abspath("."))

from detection.stability import TemporalDetectionStabilizer


def run_tests():
    print("--- VisionMate Temporal Stabilizer & Class Consistency Test Suite ---")
    stabilizer = TemporalDetectionStabilizer(max_missing_frames=3, class_change_threshold=3)

    # TEST 1 — New detection
    track_15 = {"track_id": 15, "class_id": 67, "class_name": "cell phone", "confidence": 0.65, "bbox": (100, 100, 200, 200)}
    res1 = stabilizer.update([track_15])
    print(f"Test 1 (New detection): {res1}")
    assert len(res1) == 1
    assert res1[0]["track_id"] == 15
    assert res1[0]["is_stale"] is False
    assert res1[0]["missing_frames"] == 0

    # TEST 2 — Continuous detection
    res2 = stabilizer.update([track_15])
    print(f"Test 2 (Continuous detection): {res2}")
    assert len(res2) == 1
    assert res2[0]["is_stale"] is False
    assert res2[0]["missing_frames"] == 0

    # TEST 3 — One missed frame
    res3 = stabilizer.update([])
    print(f"Test 3 (One missed frame): {res3}")
    assert len(res3) == 1
    assert res3[0]["track_id"] == 15
    assert res3[0]["is_stale"] is True
    assert res3[0]["missing_frames"] == 1

    # TEST 4 — Temporary disappearance and reappearance
    stabilizer.clear()
    stabilizer.update([track_15])  # Frame 1: fresh
    stabilizer.update([])          # Frame 2: stale (1)
    stabilizer.update([])          # Frame 3: stale (2)
    res4 = stabilizer.update([track_15]) # Frame 4: reappears
    print(f"Test 4 (Reappearance): {res4}")
    assert len(res4) == 1
    assert res4[0]["track_id"] == 15
    assert res4[0]["is_stale"] is False
    assert res4[0]["missing_frames"] == 0

    # TEST 5 — Expiration
    stabilizer.clear()
    stabilizer.update([track_15])  # Frame 1: detected
    stabilizer.update([])          # Frame 2: missing (1)
    stabilizer.update([])          # Frame 3: missing (2)
    stabilizer.update([])          # Frame 4: missing (3)
    res5 = stabilizer.update([])   # Frame 5: missing (4 > 3 threshold)
    print(f"Test 5 (Expiration after max_missing_frames=3): {res5}")
    assert len(res5) == 0

    # TEST 6 — Multiple objects
    stabilizer.clear()
    t10 = {"track_id": 10, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (50, 50, 100, 100)}
    t20 = {"track_id": 20, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (200, 200, 300, 300)}
    t30 = {"track_id": 30, "class_id": 67, "class_name": "cell phone", "confidence": 0.90, "bbox": (400, 400, 500, 500)}

    stabilizer.update([t10, t20, t30])
    res6 = stabilizer.update([t10, t30])  # t20 is missed
    print(f"Test 6 (Multiple objects with t20 missed): {res6}")
    assert len(res6) == 3
    t10_res = next(t for t in res6 if t["track_id"] == 10)
    t20_res = next(t for t in res6 if t["track_id"] == 20)
    t30_res = next(t for t in res6 if t["track_id"] == 30)
    assert t10_res["is_stale"] is False
    assert t20_res["is_stale"] is True and t20_res["missing_frames"] == 1
    assert t30_res["is_stale"] is False

    # TEST 7 — Independent tracking
    print(f"Test 7 (Independent tracking): t10={t10_res['is_stale']}, t20={t20_res['is_stale']}")
    assert t10_res["track_id"] != t20_res["track_id"]
    assert t10_res["is_stale"] != t20_res["is_stale"]

    # TEST 8 — None input
    res8 = stabilizer.update(None)
    print(f"Test 8 (None input): {res8}")
    assert isinstance(res8, list)

    # TEST 9 — Empty input
    res9 = stabilizer.update([])
    print(f"Test 9 (Empty input): {res9}")
    assert isinstance(res9, list)

    # TEST 10 — Missing track_id
    stabilizer.clear()
    invalid_track = {"class_name": "bottle", "confidence": 0.90, "bbox": (0, 0, 10, 10)}
    res10 = stabilizer.update([invalid_track])
    print(f"Test 10 (Missing track_id): {res10}")
    assert len(res10) == 0

    # TEST 11 — Stable class initialization
    stabilizer.clear()
    t54_phone = {"track_id": 54, "class_id": 67, "class_name": "cell phone", "confidence": 0.65, "bbox": (100, 100, 200, 200)}
    res11 = stabilizer.update([t54_phone])
    print(f"\nTest 11 (Stable class init): {res11[0]}")
    assert res11[0]["class_name"] == "cell phone"
    assert res11[0]["stable_class_name"] == "cell phone"
    assert res11[0]["class_change_count"] == 0

    # TEST 12 — Same class remains stable
    stabilizer.update([t54_phone])
    res12 = stabilizer.update([t54_phone])
    print(f"Test 12 (Same class continuous): {res12[0]}")
    assert res12[0]["class_name"] == "cell phone"
    assert res12[0]["candidate_class_name"] is None

    # TEST 13 — One-frame misclassification
    t54_brush = {"track_id": 54, "class_id": 79, "class_name": "toothbrush", "confidence": 0.50, "bbox": (100, 100, 200, 200)}
    res13 = stabilizer.update([t54_brush])
    print(f"Test 13 (1-frame misclass): {res13[0]}")
    assert res13[0]["class_name"] == "cell phone"
    assert res13[0]["candidate_class_name"] == "toothbrush"
    assert res13[0]["class_change_count"] == 1

    # TEST 14 — Two-frame misclassification
    res14 = stabilizer.update([t54_brush])
    print(f"Test 14 (2-frame misclass): {res14[0]}")
    assert res14[0]["class_name"] == "cell phone"
    assert res14[0]["candidate_class_name"] == "toothbrush"
    assert res14[0]["class_change_count"] == 2

    # TEST 15 — Three-frame persistent class change
    res15 = stabilizer.update([t54_brush])
    print(f"Test 15 (3-frame persistent change): {res15[0]}")
    assert res15[0]["class_name"] == "toothbrush"
    assert res15[0]["stable_class_name"] == "toothbrush"
    assert res15[0]["candidate_class_name"] is None
    assert res15[0]["class_change_count"] == 0

    # TEST 16 — Recovery after transient error
    stabilizer.clear()
    stabilizer.update([t54_phone])  # Frame 1: cell phone
    stabilizer.update([t54_brush])  # Frame 2: toothbrush (count 1)
    res16 = stabilizer.update([t54_phone])  # Frame 3: cell phone returns
    print(f"Test 16 (Recovery after 1-frame error): {res16[0]}")
    assert res16[0]["class_name"] == "cell phone"
    assert res16[0]["candidate_class_name"] is None
    assert res16[0]["class_change_count"] == 0

    # TEST 17 — Different IDs remain independent
    stabilizer.clear()
    t10_bot = {"track_id": 10, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (10, 10, 50, 50)}
    t20_phone = {"track_id": 20, "class_id": 67, "class_name": "cell phone", "confidence": 0.70, "bbox": (100, 100, 150, 150)}
    t20_brush = {"track_id": 20, "class_id": 79, "class_name": "toothbrush", "confidence": 0.50, "bbox": (100, 100, 150, 150)}

    stabilizer.update([t10_bot, t20_phone])
    res17 = stabilizer.update([t10_bot, t20_brush])
    print(f"Test 17 (Independent ID consistency): {res17}")
    t10_res17 = next(t for t in res17 if t["track_id"] == 10)
    t20_res17 = next(t for t in res17 if t["track_id"] == 20)
    assert t10_res17["class_name"] == "bottle" and t10_res17["class_change_count"] == 0
    assert t20_res17["class_name"] == "cell phone" and t20_res17["class_change_count"] == 1

    # TEST 18 — Missing track does not change class
    stabilizer.clear()
    stabilizer.update([t54_phone])
    stabilizer.update([])
    res18 = stabilizer.update([])
    print(f"Test 18 (Missing track retains class): {res18[0]}")
    assert res18[0]["class_name"] == "cell phone"
    assert res18[0]["is_stale"] is True

    # TEST 19 — Expired track clears class history
    stabilizer.clear()
    st2 = TemporalDetectionStabilizer(max_missing_frames=2, class_change_threshold=3)
    st2.update([t54_phone])
    st2.update([])
    st2.update([])
    st2.update([])  # Expired
    t80_brush = {"track_id": 80, "class_id": 79, "class_name": "toothbrush", "confidence": 0.90, "bbox": (200, 200, 250, 250)}
    res19 = st2.update([t80_brush])
    print(f"Test 19 (Expired track clears history, new ID 80): {res19[0]}")
    assert res19[0]["track_id"] == 80
    assert res19[0]["class_name"] == "toothbrush"
    assert res19[0]["stable_class_name"] == "toothbrush"

    # TEST 20 — Same ID reappears with different class
    stabilizer.clear()
    stabilizer.update([t54_phone])
    stabilizer.update([])  # Stale
    stabilizer.update([t54_brush])  # Frame 3: count 1
    stabilizer.update([t54_brush])  # Frame 4: count 2
    res20 = stabilizer.update([t54_brush])  # Frame 5: count 3 -> changes class
    print(f"Test 20 (Reappearance with different class threshold 3): {res20[0]}")
    assert res20[0]["class_name"] == "toothbrush"
    assert res20[0]["stable_class_name"] == "toothbrush"

    print("\nSUCCESS: All 20 stabilizer & class-consistency unit tests passed cleanly!")


if __name__ == "__main__":
    run_tests()
