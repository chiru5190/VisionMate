"""
VisionMate Research - YOLO26n Batch Size Pilot Testing (Batch 8 & Batch 16)
Measures:
- Complete 1-epoch training pipeline execution
- Peak allocated GPU memory (MB)
- Peak reserved GPU memory (MB)
- Stability, loss convergence, and VRAM headroom on NVIDIA GeForce RTX 2050 (4 GB)
"""

import os
import sys
import time
import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("."))

def run_batch_pilot(batch_size: int, exp_name: str, config_path: str = "research/configs/base_training_config.yaml"):
    print(f"\n==========================================================")
    print(f"  RUNNING YOLO26n PILOT: Batch Size = {batch_size} ({exp_name})")
    print(f"==========================================================")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["epochs"] = 1
    cfg["batch"] = batch_size
    cfg["imgsz"] = 640
    cfg["device"] = 0
    cfg["amp"] = True
    cfg["cache"] = False
    cfg["val"] = True
    cfg["plots"] = False
    cfg["verbose"] = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot test!")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
    initial_reserved_mb = torch.cuda.memory_reserved() / (1024**2)

    base_model = "yolo26n.pt"
    assert os.path.exists(base_model), f"{base_model} not found!"
    model = YOLO(base_model)

    start_time = time.time()
    project_abs = os.path.abspath("runs/research/pilots")

    results = model.train(
        project=project_abs,
        name=exp_name,
        exist_ok=True,
        **cfg
    )
    duration = time.time() - start_time

    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    headroom_mb = total_gpu_mb - peak_reserved_mb

    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join(project_abs, exp_name)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")

    summary = {
        "batch_size": batch_size,
        "exp_name": exp_name,
        "status": "SUCCESS" if os.path.exists(last_pt) else "FAILED",
        "duration_seconds": round(duration, 2),
        "initial_allocated_mb": round(initial_alloc_mb, 2),
        "peak_allocated_mb": round(peak_alloc_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "vram_headroom_mb": round(headroom_mb, 2),
        "total_gpu_mb": round(total_gpu_mb, 2),
        "vram_utilization_pct": round((peak_reserved_mb / total_gpu_mb) * 100.0, 2),
        "save_dir": save_dir,
        "best_checkpoint_exists": os.path.exists(best_pt),
        "last_checkpoint_exists": os.path.exists(last_pt)
    }

    print("\n----------------------------------------------------------")
    print(f"Pilot Summary for Batch Size {batch_size}:")
    print(f"  Status           : {summary['status']}")
    print(f"  Duration         : {summary['duration_seconds']}s")
    print(f"  Peak Allocated   : {summary['peak_allocated_mb']} MB")
    print(f"  Peak Reserved    : {summary['peak_reserved_mb']} MB")
    print(f"  VRAM Headroom    : {summary['vram_headroom_mb']} MB ({100.0 - summary['vram_utilization_pct']:.1f}% free)")
    print(f"  Artifacts saved  : {save_dir}")
    print("----------------------------------------------------------\n")
    return summary


def main():
    print("==========================================================")
    print("   VisionMate Research: Batch Size 8 & 16 Pilot Suite     ")
    print("==========================================================")

    # 1. Run Pilot for Batch Size 8
    res_b8 = run_batch_pilot(batch_size=8, exp_name="pilot_batch8")
    assert res_b8["status"] == "SUCCESS", "Batch size 8 pilot failed!"

    # 2. Run Pilot for Batch Size 16
    res_b16 = run_batch_pilot(batch_size=16, exp_name="pilot_batch16")
    assert res_b16["status"] == "SUCCESS", "Batch size 16 pilot failed!"

    print("==========================================================")
    print("              PILOT COMPARISON & SELECTION                ")
    print("==========================================================")
    print(f"Batch 8  -> Peak Allocated: {res_b8['peak_allocated_mb']} MB | Peak Reserved: {res_b8['peak_reserved_mb']} MB | Headroom: {res_b8['vram_headroom_mb']} MB")
    print(f"Batch 16 -> Peak Allocated: {res_b16['peak_allocated_mb']} MB | Peak Reserved: {res_b16['peak_reserved_mb']} MB | Headroom: {res_b16['vram_headroom_mb']} MB")
    
    selected_batch = 16 if res_b16["vram_headroom_mb"] >= 1000.0 else 8
    print(f"\nRecommended Selected Batch Size: {selected_batch}")
    print("==========================================================")

if __name__ == "__main__":
    main()

