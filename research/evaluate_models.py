"""
VisionMate Research - Comprehensive Model Evaluation Script (Step 4)
Evaluates any YOLO checkpoint (baseline pretrained or fine-tuned) under identical conditions:
- Precision, Recall, mAP50, mAP50-95, F1-Score
- Per-class AP & metrics
- Confusion matrix & Precision-Recall curves
- Inference latency & FPS
- Parameter count & FLOPs estimate
- Peak GPU memory consumption
"""

import sys
import os
import json
import time
import torch
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("."))

def benchmark_standardized_latency(
    model_obj: YOLO,
    device: str = "cuda:0",
    imgsz: int = 640,
    warmup_iters: int = 30,
    bench_iters: int = 100
) -> dict:
    """
    Standardized Latency and FPS Measurement Methodology:
    -----------------------------------------------------
    1. Warm-Up: Executes 30 full forward passes to ensure CUDA kernel compilation,
       GPU clock boosting, and PyTorch CUDA cache initialization before measurement.
    2. Batch Size: Standardized at batch_size=1 (represents real-time edge/robotic frame streaming).
    3. Hardware Synchronization: Explicit torch.cuda.synchronize() brackets timing events
       to prevent asynchronous execution from skewing wall-clock measurements.
    4. Boundary Differentiation:
       - Pure Inference Latency: GPU-accelerated forward pass measured via torch.cuda.Event.
       - End-to-End Latency: Raw BGR NumPy frame -> Letterbox resize -> Model Forward -> NMS -> Output Boxes.
    """
    is_cuda = torch.cuda.is_available() and "cuda" in str(device).lower()
    dummy_tensor = torch.zeros((1, 3, imgsz, imgsz), device=device if is_cuda else "cpu")

    # 1. Warm-Up passes
    for _ in range(warmup_iters):
        _ = model_obj(dummy_tensor, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()

    # 2. Pure Inference Latency (Tensor -> Forward Pass)
    pure_latencies_ms = []
    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(bench_iters):
            starter.record()
            _ = model_obj(dummy_tensor, verbose=False)
            ender.record()
            torch.cuda.synchronize()
            pure_latencies_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            _ = model_obj(dummy_tensor, verbose=False)
            pure_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_pure_ms = float(np.mean(pure_latencies_ms))
    p95_pure_ms = float(np.percentile(pure_latencies_ms, 95))
    pure_fps = 1000.0 / avg_pure_ms if avg_pure_ms > 0 else 0.0

    # 3. End-to-End Latency (NumPy BGR Frame -> Letterbox -> Predict -> NMS)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    e2e_latencies_ms = []
    for _ in range(bench_iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model_obj(dummy_frame, device=device if is_cuda else "cpu", verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        e2e_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_e2e_ms = float(np.mean(e2e_latencies_ms))
    p95_e2e_ms = float(np.percentile(e2e_latencies_ms, 95))
    e2e_fps = 1000.0 / avg_e2e_ms if avg_e2e_ms > 0 else 0.0

    return {
        "pure_inference_avg_ms": round(avg_pure_ms, 2),
        "pure_inference_p95_ms": round(p95_pure_ms, 2),
        "pure_inference_fps": round(pure_fps, 1),
        "e2e_latency_avg_ms": round(avg_e2e_ms, 2),
        "e2e_latency_p95_ms": round(p95_e2e_ms, 2),
        "e2e_fps": round(e2e_fps, 1),
        "methodology": {
            "warmup_iterations": warmup_iters,
            "benchmark_iterations": bench_iters,
            "batch_size": 1,
            "synchronization": "torch.cuda.synchronize() and torch.cuda.Event" if is_cuda else "perf_counter",
            "resolution": f"{imgsz}x{imgsz}"
        }
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    data_yaml: str = "datasets/coco_experiment.yaml",
    imgsz: int = 640,
    batch: int = 2,
    device: int = 0,
    project: str = "runs/research/eval",
    name: str = None
):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at: {checkpoint_path}"
    if name is None:
        base = os.path.splitext(os.path.basename(checkpoint_path))[0]
        parent = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path))) or "direct"
        name = f"eval_{parent}_{base}"

    print(f"\n==========================================================")
    print(f"  EVALUATING MODEL: {checkpoint_path}")
    print(f"  Destination     : {project}/{name}")
    print(f"==========================================================")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(checkpoint_path)
    total_params = sum(p.numel() for p in model.parameters())

    start_eval = time.time()
    project_abs = os.path.abspath(project)
    val_res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_abs,
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True
    )
    eval_duration = time.time() - start_eval

    # Metrics extraction
    r_dict = val_res.results_dict
    precision = float(r_dict.get("metrics/precision(B)", 0.0))
    recall = float(r_dict.get("metrics/recall(B)", 0.0))
    map50 = float(r_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(r_dict.get("metrics/mAP50-95(B)", 0.0))
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Latencies
    speed = val_res.speed  # preprocess, inference, loss, postprocess in ms
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    loss_ms = float(speed.get("loss", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

    # Memory
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

    # Per-class metrics
    per_class_ap = {}
    class_names = val_res.names
    if hasattr(val_res.box, "maps") and val_res.box.maps is not None:
        for cls_idx, ap in enumerate(val_res.box.maps):
            if cls_idx in class_names:
                cls_name = class_names[cls_idx]
                per_class_ap[cls_name] = round(float(ap), 4)

    # Standardized Interactive Benchmark (batch_size=1, CUDA synchronization, warm-up)
    std_bench = benchmark_standardized_latency(
        model_obj=model,
        device=f"cuda:{device}" if isinstance(device, int) else str(device),
        imgsz=imgsz,
        warmup_iters=30,
        bench_iters=50
    )

    eval_summary = {
        "checkpoint": checkpoint_path,
        "dataset": data_yaml,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": total_params,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "dataloader_batch_speed_ms": {
            "preprocess": round(preprocess_ms, 3),
            "inference": round(inference_ms, 3),
            "postprocess": round(postprocess_ms, 3),
            "total_latency": round(total_latency_ms, 3)
        },
        "latency_ms": {
            "preprocess": round(preprocess_ms, 3),
            "inference": round(inference_ms, 3),
            "postprocess": round(postprocess_ms, 3),
            "total_latency": round(total_latency_ms, 3)
        },
        "dataloader_fps": round(fps, 2),
        "standardized_benchmark": std_bench,
        "pure_inference_ms": std_bench["pure_inference_avg_ms"],
        "pure_inference_fps": std_bench["pure_inference_fps"],
        "e2e_interactive_fps": std_bench["e2e_fps"],
        "peak_vram_mb": round(peak_vram_mb, 2),
        "per_class_map50_95": per_class_ap,
        "artifacts_dir": os.path.join(project, name)
    }

    out_json = os.path.join(project, name, "evaluation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n--- EVALUATION METRICS: {checkpoint_path} ---")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-Score           : {f1_score:.4f}")
    print(f"mAP50              : {map50:.4f}")
    print(f"mAP50-95           : {map50_95:.4f}")
    print(f"Dataloader Latency : {total_latency_ms:.2f} ms ({fps:.1f} FPS, batch={batch})")
    print(f"Pure Inference     : {std_bench['pure_inference_avg_ms']:.2f} ms ({std_bench['pure_inference_fps']:.1f} FPS, batch=1, CUDA Event)")
    print(f"Interactive E2E    : {std_bench['e2e_latency_avg_ms']:.2f} ms ({std_bench['e2e_fps']:.1f} FPS, batch=1, synchronized)")
    print(f"Parameters         : {total_params:,}")
    print(f"Peak VRAM          : {peak_vram_mb:.1f} MB")
    print(f"Summary JSON       : {out_json}\n")

    return eval_summary


def evaluate_all(
    baseline_11: str = "yolo11n.pt",
    baseline_26: str = "yolo26n.pt",
    finetuned_11: str = "runs/research/yolo11n_coco/weights/best.pt",
    finetuned_26: str = "runs/research/yolo26n_coco/weights/best.pt",
    data_yaml: str = "datasets/coco_experiment.yaml"
):
    print("##########################################################")
    print("      VisionMate Research: Step 4 Full Evaluation Suite   ")
    print("##########################################################")

    all_results = {}

    # 1. Baseline Pretrained YOLO11n
    if os.path.exists(baseline_11):
        all_results["yolo11n_baseline"] = evaluate_checkpoint(baseline_11, data_yaml=data_yaml, name="yolo11n_baseline")

    # 2. Baseline Pretrained YOLO26n
    if os.path.exists(baseline_26):
        all_results["yolo26n_baseline"] = evaluate_checkpoint(baseline_26, data_yaml=data_yaml, name="yolo26n_baseline")

    # 3. Fine-tuned YOLO11n (or pilot checkpoint)
    target_ft11 = finetuned_11 if os.path.exists(finetuned_11) else "runs/research/pilot_yolo11n/weights/best.pt"
    if os.path.exists(target_ft11):
        all_results["yolo11n_finetuned"] = evaluate_checkpoint(target_ft11, data_yaml=data_yaml, name="yolo11n_finetuned")
    else:
        print(f"Notice: Fine-tuned YOLO11n not found at {finetuned_11} or pilot path.")

    # 4. Fine-tuned YOLO26n (or pilot checkpoint)
    target_ft26 = finetuned_26 if os.path.exists(finetuned_26) else "runs/research/pilot_yolo26n/weights/best.pt"
    if os.path.exists(target_ft26):
        all_results["yolo26n_finetuned"] = evaluate_checkpoint(target_ft26, data_yaml=data_yaml, name="yolo26n_finetuned")
    else:
        print(f"Notice: Fine-tuned YOLO26n not found at {finetuned_26} or pilot path.")

    summary_file = "runs/research/all_evaluations_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll evaluations saved to: {summary_file}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on COCO")
    parser.add_argument("--model", type=str, default=None, help="Single model checkpoint to evaluate")
    parser.add_argument("--data", type=str, default="datasets/coco_experiment.yaml")
    args = parser.parse_args()

    if args.model:
        evaluate_checkpoint(args.model, data_yaml=args.data)
    else:
        evaluate_all(data_yaml=args.data)

