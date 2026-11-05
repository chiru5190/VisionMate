import sys
import os

sys.path.insert(0, os.path.abspath("."))

from detection.recovery import TrackRecoveryManager


def run_enhanced_tests():
    print("--- VisionMate Enhanced Track Recovery Manager Test Suite ---")

    # 1. Stationary object disappearing and returning
    rm = TrackRecoveryManager(max_recovery_frames=5, frame_width=640, frame_height=480)
    t10_stat = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 200, 200)}
    rm.update([t10_stat])
    rm.update([t10_stat])
    rm.update([t10_stat])
    rm.update([])  # lost frame 1
    rm.update([])  # lost frame 2

    t50_stat = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (100, 100, 200, 200)}
    res1 = rm.update([t50_stat])
    print(f"Test 1 (Stationary object recovery): {res1}")
    assert len(res1) == 1
    assert res1[0]["logical_track_id"] == 10
    assert res1[0]["recovered"] is True
    assert res1[0]["recovery_reason"] == "recovered_spatial"
    assert res1[0]["spatial_score"] == 1.0

    # 2. Moving object disappearing and returning with motion prediction
    rm.clear()
    t10_m1 = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 200, 200)}  # center 150
    t10_m2 = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (120, 100, 220, 200)}  # center 170 (+20px/frame)
    t10_m3 = {"track_id": 10, "class_id": 67, "class_name": "cell phone", "confidence": 0.85, "bbox": (140, 100, 240, 200)}  # center 190 (+20px/frame)
    rm.update([t10_m1])
    rm.update([t10_m2])
    rm.update([t10_m3])
    rm.update([])  # lost frame 1 (predicted center 210)
    rm.update([])  # lost frame 2 (predicted center 230)

    # Frame 6: expected predicted center 250 -> bbox (200, 100, 300, 200)
    t50_move = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (200, 100, 300, 200)}
    res2 = rm.update([t50_move])
    print(f"Test 2 (Moving object prediction recovery): {res2}")
    assert len(res2) == 1
    assert res2[0]["logical_track_id"] == 10
    assert res2[0]["recovered"] is True
    assert res2[0]["recovery_reason"] == "recovered_motion"

    # 3. Same-class objects close together
    rm.clear()
    t10 = {"track_id": 10, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (100, 100, 150, 150)}
    t20 = {"track_id": 20, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (300, 100, 350, 150)}
    rm.update([t10, t20])
    rm.update([])  # both lost

    t40 = {"track_id": 40, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (102, 100, 152, 150)}
    t41 = {"track_id": 41, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (302, 100, 352, 150)}
    res3 = rm.update([t40, t41])
    print(f"Test 3 (Same-class objects close together): {res3}")
    assert len(res3) == 2
    r10 = next(t for t in res3 if t["original_track_id"] == 40)
    r20 = next(t for t in res3 if t["original_track_id"] == 41)
    assert r10["logical_track_id"] == 10 and r10["recovered"] is True
    assert r20["logical_track_id"] == 20 and r20["recovered"] is True

    # 4. Two possible recovery candidates (higher score wins)
    rm.clear()
    rm.update([t10_stat])
    rm.update([])
    t50_cand1 = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (101, 101, 201, 201)}
    t51_cand2 = {"track_id": 51, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (150, 100, 250, 200)}
    res4 = rm.update([t50_cand1, t51_cand2])
    print(f"Test 4 (Two candidate assignment): {res4}")
    assert len(res4) == 2
    c1 = next(t for t in res4 if t["original_track_id"] == 50)
    c2 = next(t for t in res4 if t["original_track_id"] == 51)
    assert c1["logical_track_id"] == 10 and c1["recovered"] is True
    assert c2["logical_track_id"] == 51 and c2["recovered"] is False

    # 5. Object changing direction
    rm.clear()
    rm.update([t10_m1])
    rm.update([t10_m2])
    rm.update([t10_m3])
    rm.update([])
    t50_reverse = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (130, 100, 230, 200)}
    res5 = rm.update([t50_reverse])
    print(f"Test 5 (Object changing direction): {res5}")
    assert len(res5) == 1
    assert res5[0]["logical_track_id"] == 10
    assert res5[0]["recovered"] is True

    # 6. Object changing apparent size (camera distance change)
    rm.clear()
    rm.update([t10_stat])
    rm.update([])
    t50_bigger = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (85, 85, 215, 215)}  # center (150,150), area 16900 vs 10000
    res6 = rm.update([t50_bigger])
    print(f"Test 6 (Apparent size change): {res6}")
    assert len(res6) == 1
    assert res6[0]["logical_track_id"] == 10
    assert res6[0]["recovered"] is True

    # 7. Recovery at exact maximum allowed disappearance boundary
    rm.clear()
    rm_boundary = TrackRecoveryManager(max_recovery_frames=3, frame_width=640, frame_height=480)
    rm_boundary.update([t10_stat])
    rm_boundary.update([]) # missing 1
    rm_boundary.update([]) # missing 2
    t50_bound = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (101, 101, 201, 201)} # reappears on missing 3 (seen count 3 <= 3)
    res7 = rm_boundary.update([t50_bound])
    print(f"Test 7 (Exact max_recovery_frames boundary recovery): {res7}")
    assert len(res7) == 1
    assert res7[0]["logical_track_id"] == 10
    assert res7[0]["recovered"] is True

    # 8. Ambiguous recovery (equal top scores rejected)
    rm.clear()
    rm_ambig = TrackRecoveryManager(max_recovery_frames=5, ambiguity_threshold=0.05)
    t10_a = {"track_id": 10, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (100, 100, 150, 150)}
    t11_a = {"track_id": 11, "class_id": 39, "class_name": "bottle", "confidence": 0.85, "bbox": (101, 100, 151, 150)}
    rm_ambig.update([t10_a, t11_a])
    rm_ambig.update([])  # both lost

    t50_ambig = {"track_id": 50, "class_id": 39, "class_name": "bottle", "confidence": 0.80, "bbox": (100, 100, 150, 150)}
    res8 = rm_ambig.update([t50_ambig])
    print(f"Test 8 (Ambiguous recovery protection): {res8}")
    assert len(res8) == 1
    assert res8[0]["logical_track_id"] == 50
    assert res8[0]["recovered"] is False
    assert res8[0]["recovery_reason"] == "ambiguity_rejected"

    # 9. One-to-one assignment enforcement
    rm.clear()
    rm.update([t10_stat])
    rm.update([])
    t50_o1 = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (100, 100, 200, 200)}
    t51_o2 = {"track_id": 51, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (105, 105, 205, 205)}
    res9 = rm.update([t50_o1, t51_o2])
    print(f"Test 9 (One-to-one assignment): {res9}")
    assert len(res9) == 2
    logical_ids = [t["logical_track_id"] for t in res9]
    assert 10 in logical_ids
    assert len(set(logical_ids)) == 2

    # 10. Velocity/prediction fallback with insufficient history
    rm.clear()
    rm.update([t10_stat])
    rm.update([])
    t50_fallback = {"track_id": 50, "class_id": 67, "class_name": "cell phone", "confidence": 0.80, "bbox": (102, 102, 202, 202)}
    res10 = rm.update([t50_fallback])
    print(f"Test 10 (Insufficient history fallback): {res10}")
    assert len(res10) == 1
    assert res10[0]["logical_track_id"] == 10
    assert res10[0]["recovered"] is True

    print("\nSUCCESS: ALL 10 ENHANCED TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_enhanced_tests()
