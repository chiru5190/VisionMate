"""
VisionMate Research - YOLO26n Continued Fine-Tuning on COCO
Initializes from official pretrained checkpoint: yolo26n.pt
Saves artifacts to: runs/research/yolo26n_coco/
"""

import sys
import os
import yaml
import argparse
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("."))

def train_yolo26n(
    config_path: str = "research/configs/base_training_config.yaml",
    epochs: int = None,
    batch: int = None,
    project: str = "runs/research",
    name: str = "yolo26n_coco",
    exist_ok: bool = True
):
    print("==========================================================")
    print("      VisionMate Research: YOLO26n COCO Fine-Tuning       ")
    print("==========================================================")

    # 1. Load base configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch is not None:
        cfg["batch"] = batch

    # Hardware check
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected, falling back to CPU.")
        cfg["device"] = "cpu"
        cfg["amp"] = False
    else:
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial Allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # 2. Initialize from pretrained checkpoint
    base_model = "yolo26n.pt"
    assert os.path.exists(base_model), f"Checkpoint {base_model} not found."
    print(f"Loading official pretrained weights: {base_model}")
    model = YOLO(base_model)

    print(f"Starting fine-tuning with parameters:")
    print(f"  Dataset    : {cfg.get('data')}")
    print(f"  Epochs     : {cfg.get('epochs')}")
    print(f"  Batch size : {cfg.get('batch')}")
    print(f"  Image size : {cfg.get('imgsz')}")
    print(f"  Device     : {cfg.get('device')}")
    print(f"  AMP        : {cfg.get('amp')}")
    print(f"  Optimizer  : {cfg.get('optimizer')}")
    print(f"  Output dir : {project}/{name}")

    # 3. Train
    project_abs = os.path.abspath(project)
    results = model.train(
        project=project_abs,
        name=name,
        exist_ok=exist_ok,
        **cfg
    )

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print("\n----------------------------------------------------------")
    print("YOLO26n Fine-Tuning Run Completed Successfully!")
    print(f"Peak VRAM during training: {peak_vram_mb:.2f} MB")
    print(f"Artifacts saved in: {project}/{name}")
    print("----------------------------------------------------------\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26n COCO Fine-Tuning")
    parser.add_argument("--config", type=str, default="research/configs/base_training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--project", type=str, default="runs/research")
    parser.add_argument("--name", type=str, default="yolo26n_coco")
    args = parser.parse_args()

    train_yolo26n(
        config_path=args.config,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name
    )

