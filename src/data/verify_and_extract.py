"""Script to verify and extract train2017.zip upon download completion."""

import os
import sys
import zipfile
import subprocess
from pathlib import Path


def main() -> None:
    zip_path = Path("data/coco/zips/train2017.zip")
    extract_target = Path("data/coco")
    train_dir = Path("data/coco/train2017")

    print(f"Checking zip file: {zip_path}")
    if not zip_path.exists():
        print(f"Error: {zip_path} does not exist!")
        sys.exit(1)

    size_gb = zip_path.stat().st_size / (1024**3)
    print(f"File size: {size_gb:.2f} GB")

    print("Verifying zip header integrity...")
    if not zipfile.is_zipfile(zip_path):
        print("Error: train2017.zip is not a valid zip archive (download may still be incomplete).")
        sys.exit(1)

    print("Valid zip archive detected! Extracting with tar.exe for high throughput...")
    cmd = ["tar.exe", "-xf", str(zip_path), "-C", str(extract_target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"tar extraction returned error: {result.stderr}")
        print("Falling back to Python zipfile extraction...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_target)
    
    print("Extraction completed!")
    if train_dir.exists():
        img_count = sum(1 for _ in train_dir.glob("*.jpg"))
        print(f"Total images found in {train_dir}: {img_count} (Expected: ~118,287)")
    else:
        print(f"Warning: {train_dir} not found after extraction!")


if __name__ == "__main__":
    main()
