"""
VisionMate Research - Hardware & AMP Verification Script
Verifies:
- NVIDIA GPU device properties (RTX 2050, 4GB VRAM)
- CUDA Memory availability
- PyTorch AMP (Automatic Mixed Precision) functionality
"""

import sys
import torch
import torch.nn as nn

def check_hardware():
    print("==========================================================")
    print("          VisionMate Hardware & AMP Verification           ")
    print("==========================================================")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! A GPU is required for this experiment.")

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024 ** 3)
    free_mem_gb, total_mem_gb_alloc = torch.cuda.mem_get_info(0)
    free_mem_gb = free_mem_gb / (1024 ** 3)

    print(f"CUDA Devices Available : {device_count}")
    print(f"Device 0 Name          : {device_name}")
    print(f"Compute Capability     : {capability[0]}.{capability[1]}")
    print(f"Total VRAM             : {total_mem_gb:.2f} GB")
    print(f"Free VRAM Currently    : {free_mem_gb:.2f} GB")
    print(f"PyTorch Version        : {torch.__version__}")
    print(f"CUDA Driver / Toolchain: {torch.version.cuda}")

    # Verify AMP support with dummy model & tensor
    print("\nVerifying Automatic Mixed Precision (AMP) on cuda:0...")
    torch.cuda.empty_cache()
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.SiLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 80)
    ).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda')

    dummy_input = torch.randn(2, 3, 640, 640, device='cuda')
    target = torch.randint(0, 80, (2,), device='cuda')
    criterion = nn.CrossEntropyLoss()

    with torch.amp.autocast('cuda'):
        output = model(dummy_input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"AMP Forward & Backward succeeded!")
    print(f"Peak VRAM used during AMP test: {peak_mem_mb:.2f} MB")
    print("==========================================================")
    print("         HARDWARE & AMP VERIFICATION COMPLETED: PASS      ")
    print("==========================================================")

if __name__ == "__main__":
    check_hardware()

