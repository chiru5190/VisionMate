"""
VisionMate Research - Pilot Training Verification Suite
Runs a controlled 1-epoch pilot for both YOLO11n and YOLO26n under identical settings:
- imgsz: 640
- batch: 2
- workers: 2
- device: 0
- amp: True
- cache: False

Detects any CUDA Out-Of-Memory errors and logs peak memory consumption.
"""

import sys
import os
import json
import time
import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("."))

def run_single_pilot(model_checkpoint: str, exp_name: str, config_path: str = "research/configs/base_training_config.yaml"):
    print(f"\n==========================================================")
    print(f"  LAUNCHING PILOT: {model_checkpoint} -> {exp_name}")
    print(f"==========================================================")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force 1-epoch pilot constraints
    cfg["epochs"] = 1
    cfg["batch"] = 2
    cfg["imgsz"] = 640
    cfg["workers"] = 2
    cfg["device"] = 0
    cfg["amp"] = True
    cfg["cache"] = False
    cfg["plots"] = True
    cfg["val"] = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_alloc = torch.cuda.memory_allocated() / (1024**2)
    else:
        initial_alloc = 0.0

    start_time = time.time()
    error_occurred = None
    results = None
    peak_vram_mb = 0.0

    try:
        model = YOLO(model_checkpoint)
        project_abs = os.path.abspath("runs/research")
        results = model.train(
            project=project_abs,
            name=exp_name,
            exist_ok=True,
            **cfg
        )
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    except torch.cuda.OutOfMemoryError as oom:
        error_occurred = f"CUDA OutOfMemoryError: {oom}"
        print(f"ERROR: {error_occurred}")
    except Exception as e:
        error_occurred = f"Training error: {e}"
        print(f"ERROR: {error_occurred}")

    duration = time.time() - start_time
    if results and hasattr(results, "save_dir"):
        save_dir = str(results.save_dir)
    else:
        # Fallback check both direct and nested detect paths
        candidate = os.path.join(os.path.abspath("runs/research"), exp_name)
        nested = os.path.join(os.path.abspath("runs"), "detect", "runs", "research", exp_name)
        save_dir = candidate if os.path.exists(candidate) else nested

    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")

    status = "SUCCESS" if (error_occurred is None and os.path.exists(last_pt)) else "FAILED"

    summary = {
        "model_checkpoint": model_checkpoint,
        "exp_name": exp_name,
        "status": status,
        "duration_seconds": round(duration, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "initial_vram_mb": round(initial_alloc, 2),
        "save_dir": save_dir,
        "best_checkpoint_exists": os.path.exists(best_pt),
        "last_checkpoint_exists": os.path.exists(last_pt),
        "error": error_occurred
    }

    print(f"\nPilot Result for {model_checkpoint}:")
    print(f"  Status        : {status}")
    print(f"  Duration      : {duration:.2f} s")
    print(f"  Peak VRAM     : {peak_vram_mb:.2f} MB (out of 4096 MB)")
    print(f"  Last Pt Exists: {os.path.exists(last_pt)}")
    if error_occurred:
        print(f"  Error details : {error_occurred}")

    return summary


def run_full_pilot():
    print("##########################################################")
    print("     VisionMate Research: Step 3 Hardware & Pilot Suite   ")
    print("##########################################################")

    pilot_11n = run_single_pilot("yolo11n.pt", "pilot_yolo11n")
    pilot_26n = run_single_pilot("yolo26n.pt", "pilot_yolo26n")

    overall_status = "ALL_PASSED" if (pilot_11n["status"] == "SUCCESS" and pilot_26n["status"] == "SUCCESS") else "FAILED"

    combined = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "yolo11n_pilot": pilot_11n,
        "yolo26n_pilot": pilot_26n
    }

    out_json = "runs/research/pilot_summary.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\n==========================================================")
    print(f"Overall Pilot Status: {overall_status}")
    print(f"Pilot summary saved to: {out_json}")
    print("==========================================================")
    return combined


if __name__ == "__main__":
    run_full_pilot()

