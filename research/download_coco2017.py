"""
VisionMate Research - Official COCO 2017 Dataset Downloader & Verification
Downloads and organizes the full official dataset under D:\\VisionMate\\datasets\\coco:
- Images: train2017 (118,287 images) and val2017 (5,000 images)
- Annotations: instances_train2017.json, instances_val2017.json
- YOLO Labels: train2017 (118,287 label files) and val2017 (5,000 label files)
- Avoids duplicate downloads if files/archives already exist
- Verifies image counts, annotation files, and all 80 COCO categories
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import json

DATASET_ROOT = r"D:\VisionMate\datasets\coco"
ZIPS_DIR = os.path.join(DATASET_ROOT, "zips")

DOWNLOAD_TARGETS = [
    {
        "name": "COCO 2017 Validation Images (5k)",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "zip_name": "val2017.zip",
        "extract_to": os.path.join(DATASET_ROOT, "images"),
        "expected_check": os.path.join(DATASET_ROOT, "images", "val2017"),
        "expected_count": 5000
    },
    {
        "name": "COCO 2017 Converted YOLO Labels",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip",
        "zip_name": "coco2017labels-segments.zip",
        "extract_to": DATASET_ROOT,
        "expected_check": os.path.join(DATASET_ROOT, "labels", "val2017"),
        "expected_count": 5000
    },
    {
        "name": "COCO 2017 Official Annotations JSON",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "zip_name": "annotations_trainval2017.zip",
        "extract_to": DATASET_ROOT,
        "expected_check": os.path.join(DATASET_ROOT, "annotations", "instances_val2017.json"),
        "expected_count": None
    },
    {
        "name": "COCO 2017 Training Images (118k)",
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "zip_name": "train2017.zip",
        "extract_to": os.path.join(DATASET_ROOT, "images"),
        "expected_check": os.path.join(DATASET_ROOT, "images", "train2017"),
        "expected_count": 118287
    }
]

def check_disk_space():
    total, used, free = shutil.disk_usage(r"D:\VisionMate")
    free_gb = free / (1024 ** 3)
    print(f"Checking disk space on D: drive: {free_gb:.2f} GB free")
    if free_gb < 40.0:
        raise RuntimeError(f"Insufficient disk space on D:! Required: ~40 GB, Available: {free_gb:.2f} GB")
    return free_gb

def download_with_progress(url: str, output_path: str):
    if os.path.exists(output_path):
        remote_size = None
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
        except Exception:
            pass

        local_size = os.path.getsize(output_path)
        if remote_size and local_size == remote_size:
            print(f"  [Already Downloaded] {os.path.basename(output_path)} ({local_size / (1024**2):.1f} MB), skipping.")
            return

    print(f"  Downloading from: {url}")
    print(f"  Saving to       : {output_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, (downloaded / total_size) * 100.0)
            mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progress: {mb:.1f} / {tot_mb:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print("\n    Download complete.")

def extract_archive(zip_path: str, extract_to: str):
    print(f"  Extracting {os.path.basename(zip_path)} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("    Extraction complete.")

def verify_dataset():
    print("\n==========================================================")
    print("         VERIFYING OFFICIAL COCO 2017 DATASET            ")
    print("==========================================================")

    train_img_dir = os.path.join(DATASET_ROOT, "images", "train2017")
    val_img_dir = os.path.join(DATASET_ROOT, "images", "val2017")
    train_lbl_dir = os.path.join(DATASET_ROOT, "labels", "train2017")
    val_lbl_dir = os.path.join(DATASET_ROOT, "labels", "val2017")
    ann_json = os.path.join(DATASET_ROOT, "annotations", "instances_val2017.json")

    train_imgs = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    val_imgs = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
    train_lbls = len(os.listdir(train_lbl_dir)) if os.path.exists(train_lbl_dir) else 0
    val_lbls = len(os.listdir(val_lbl_dir)) if os.path.exists(val_lbl_dir) else 0

    print(f"Train Images Count : {train_imgs:,} (Expected: 118,287)")
    print(f"Val Images Count   : {val_imgs:,} (Expected: 5,000)")
    print(f"Train Labels Count : {train_lbls:,} (Expected: ~118,287)")
    print(f"Val Labels Count   : {val_lbls:,} (Expected: 5,000)")

    # Category verification from JSON
    if os.path.exists(ann_json):
        with open(ann_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = data.get("categories", [])
        print(f"Categories Detected: {len(categories)} / 80 official COCO classes")
        sample_cats = [c["name"] for c in categories[:10]]
        print(f"Sample Categories  : {sample_cats}")
    else:
        print(f"Notice: Annotations JSON not yet present at {ann_json}")

    print("==========================================================\n")

def run(download_images: bool = True):
    check_disk_space()
    os.makedirs(ZIPS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "images"), exist_ok=True)

    for item in DOWNLOAD_TARGETS:
        if not download_images and "Images" in item["name"]:
            print(f"Skipping {item['name']} (download_images=False)")
            continue

        zip_path = os.path.join(ZIPS_DIR, item["zip_name"])
        print(f"\n[Processing] {item['name']}")
        download_with_progress(item["url"], zip_path)
        extract_archive(zip_path, item["extract_to"])

    verify_dataset()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Official COCO2017 Dataset Downloader")
    parser.add_argument("--skip-images", action="store_true", help="Download only annotations and labels")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing downloaded dataset without downloading")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset()
    else:
        run(download_images=not args.skip_images)

