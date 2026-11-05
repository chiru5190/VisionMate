"""
VisionMate Research - Model Comparison & Research Metrics Reporting (Step 6)
Generates:
1. YOLO11n vs YOLO26n Baseline Comparison
2. YOLO11n vs YOLO26n Fine-Tuned Comparison
3. Before-training vs After-fine-tuning Comparison for both models
4. Relative metric improvement percentages
5. Per-class AP analysis
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath("."))

def pct_delta(val_new, val_old):
    if val_old == 0:
        return 0.0
    return round(((val_new - val_old) / val_old) * 100.0, 2)


def generate_comparison_report(eval_summary_json: str = "runs/research/all_evaluations_summary.json"):
    if not os.path.exists(eval_summary_json):
        print(f"ERROR: Summary file {eval_summary_json} does not exist. Run evaluate_models.py first.")
        return None

    with open(eval_summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    b11 = data.get("yolo11n_baseline")
    b26 = data.get("yolo26n_baseline")
    f11 = data.get("yolo11n_finetuned")
    f26 = data.get("yolo26n_finetuned")

    report = {
        "title": "VisionMate Research: YOLO11n vs YOLO26n COCO Evaluation & Fine-Tuning Analysis",
        "comparisons": {}
    }

    # 1. Baseline Pretrained Comparison: YOLO11n vs YOLO26n
    if b11 and b26:
        baseline_comp = {
            "mAP50": {"yolo11n": b11["map50"], "yolo26n": b26["map50"], "diff": round(b26["map50"] - b11["map50"], 4), "pct_change": pct_delta(b26["map50"], b11["map50"])},
            "mAP50_95": {"yolo11n": b11["map50_95"], "yolo26n": b26["map50_95"], "diff": round(b26["map50_95"] - b11["map50_95"], 4), "pct_change": pct_delta(b26["map50_95"], b11["map50_95"])},
            "precision": {"yolo11n": b11["precision"], "yolo26n": b26["precision"], "diff": round(b26["precision"] - b11["precision"], 4), "pct_change": pct_delta(b26["precision"], b11["precision"])},
            "recall": {"yolo11n": b11["recall"], "yolo26n": b26["recall"], "diff": round(b26["recall"] - b11["recall"], 4), "pct_change": pct_delta(b26["recall"], b11["recall"])},
            "f1_score": {"yolo11n": b11["f1_score"], "yolo26n": b26["f1_score"], "diff": round(b26["f1_score"] - b11["f1_score"], 4), "pct_change": pct_delta(b26["f1_score"], b11["f1_score"])},
            "inference_ms": {"yolo11n": b11["latency_ms"]["inference"], "yolo26n": b26["latency_ms"]["inference"], "diff": round(b26["latency_ms"]["inference"] - b11["latency_ms"]["inference"], 3), "pct_change": pct_delta(b26["latency_ms"]["inference"], b11["latency_ms"]["inference"])},
            "fps": {"yolo11n": b11["fps"], "yolo26n": b26["fps"], "diff": round(b26["fps"] - b11["fps"], 2), "pct_change": pct_delta(b26["fps"], b11["fps"])},
            "parameters": {"yolo11n": b11["parameters"], "yolo26n": b26["parameters"], "diff": b26["parameters"] - b11["parameters"]}
        }
        report["comparisons"]["baseline_yolo11n_vs_yolo26n"] = baseline_comp

    # 2. Before vs After Fine-Tuning: YOLO11n
    if b11 and f11:
        yolo11_ft_comp = {
            "mAP50": {"before": b11["map50"], "after": f11["map50"], "pct_change": pct_delta(f11["map50"], b11["map50"])},
            "mAP50_95": {"before": b11["map50_95"], "after": f11["map50_95"], "pct_change": pct_delta(f11["map50_95"], b11["map50_95"])},
            "precision": {"before": b11["precision"], "after": f11["precision"], "pct_change": pct_delta(f11["precision"], b11["precision"])},
            "recall": {"before": b11["recall"], "after": f11["recall"], "pct_change": pct_delta(f11["recall"], b11["recall"])}
        }
        report["comparisons"]["yolo11n_before_vs_after_finetune"] = yolo11_ft_comp

    # 3. Before vs After Fine-Tuning: YOLO26n
    if b26 and f26:
        yolo26_ft_comp = {
            "mAP50": {"before": b26["map50"], "after": f26["map50"], "pct_change": pct_delta(f26["map50"], b26["map50"])},
            "mAP50_95": {"before": b26["map50_95"], "after": f26["map50_95"], "pct_change": pct_delta(f26["map50_95"], b26["map50_95"])},
            "precision": {"before": b26["precision"], "after": f26["precision"], "pct_change": pct_delta(f26["precision"], b26["precision"])},
            "recall": {"before": b26["recall"], "after": f26["recall"], "pct_change": pct_delta(f26["recall"], b26["recall"])}
        }
        report["comparisons"]["yolo26n_before_vs_after_finetune"] = yolo26_ft_comp

    # 4. Fine-Tuned YOLO11n vs Fine-Tuned YOLO26n
    if f11 and f26:
        ft_comp = {
            "mAP50": {"yolo11n_ft": f11["map50"], "yolo26n_ft": f26["map50"], "diff": round(f26["map50"] - f11["map50"], 4), "pct_change": pct_delta(f26["map50"], f11["map50"])},
            "mAP50_95": {"yolo11n_ft": f11["map50_95"], "yolo26n_ft": f26["map50_95"], "diff": round(f26["map50_95"] - f11["map50_95"], 4), "pct_change": pct_delta(f26["map50_95"], f11["map50_95"])},
            "precision": {"yolo11n_ft": f11["precision"], "yolo26n_ft": f26["precision"], "diff": round(f26["precision"] - f11["precision"], 4), "pct_change": pct_delta(f26["precision"], f11["precision"])},
            "recall": {"yolo11n_ft": f11["recall"], "yolo26n_ft": f26["recall"], "diff": round(f26["recall"] - f11["recall"], 4), "pct_change": pct_delta(f26["recall"], f11["recall"])}
        }
        report["comparisons"]["finetuned_yolo11n_vs_yolo26n"] = ft_comp

    # 5. Per-Class Analysis (Baseline)
    if b11 and b26:
        classes = sorted(set(b11["per_class_map50_95"].keys()).union(set(b26["per_class_map50_95"].keys())))
        per_class_diff = {}
        for c in classes:
            ap11 = b11["per_class_map50_95"].get(c, 0.0)
            ap26 = b26["per_class_map50_95"].get(c, 0.0)
            per_class_diff[c] = {
                "yolo11n": ap11,
                "yolo26n": ap26,
                "delta": round(ap26 - ap11, 4)
            }
        report["per_class_baseline_comparison"] = per_class_diff

    out_file = "runs/research/model_comparison_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("================================================================================")
    print("      RESEARCH COMPARISON: YOLO11n vs YOLO26n BASELINE VALIDATION               ")
    print("================================================================================")
    if b11 and b26:
        print(f"{'METRIC':<20} | {'YOLO11n Baseline':<18} | {'YOLO26n Baseline':<18} | {'DELTA':<10} | {'% CHANGE':<10}")
        print("-" * 80)
        for k in ["mAP50", "mAP50_95", "precision", "recall", "f1_score", "inference_ms", "fps"]:
            sub = baseline_comp[k]
            v11 = sub["yolo11n"]
            v26 = sub["yolo26n"]
            d = sub["diff"]
            pct = sub["pct_change"]
            print(f"{k:<20} | {v11:<18.4f} | {v26:<18.4f} | {d:+9.4f} | {pct:+8.2f}%")
        print("=" * 80)

    print(f"\nDetailed report saved to: {out_file}\n")
    return report


if __name__ == "__main__":
    generate_comparison_report()

