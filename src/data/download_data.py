"""COCO 2017 dataset download and subset preparation module.

Supports two download modes:
- Mode A ('zip'): Full zip archives download and extraction (~19.3 GB total download).
- Mode B ('direct'): Annotations zip + direct image download via COCO image URLs.

Splits train2017 into:
- 14,000 train subset images (train_subset14k.json)
- 1,000 val-heldout subset images (val_heldout1k.json)
Zero data leakage is guaranteed via fixed random seed and mutual exclusivity assertion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}

ESTIMATED_SIZES = {
    "annotations": 241 * 1024 * 1024,      # ~241 MB
    "val2017": 1024 * 1024 * 1024,         # ~1.0 GB
    "train2017": 18 * 1024 * 1024 * 1024,  # ~18.0 GB
}


class DownloadProgressBar:
    """Console progress bar for urllib downloads."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        if percent != self.last_percent and percent % 5 == 0:
            self.last_percent = percent
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r{self.description}: {percent:3d}% ({downloaded_mb:,.1f} MB / {total_mb:,.1f} MB)"
            )
            sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")
            sys.stdout.flush()


def download_file(url: str, dest_path: Path, description: str) -> None:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return

    temp_path = dest_path.with_suffix(".tmp")
    print(f"Starting download: {url} -> {dest_path}")
    progress = DownloadProgressBar(description)
    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=progress)
        temp_path.rename(dest_path)
        print(f"Download completed: {dest_path}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def extract_zip(zip_path: Path, extract_to: Path, check_dir: Path | None = None) -> None:
    """Safely extract a zip file to destination directory if not already extracted."""
    if check_dir and check_dir.exists() and any(check_dir.iterdir()):
        print(f"Directory already exists and is non-empty: {check_dir}. Skipping extraction.")
        return

    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def split_coco_annotations(
    coco_data: Dict[str, Any],
    train_count: int = 14000,
    val_heldout_count: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split COCO annotation dict deterministically into train and val-heldout subsets.

    Args:
        coco_data: Loaded COCO instances annotation dictionary.
        train_count: Number of images for training subset (default: 14,000).
        val_heldout_count: Number of images for held-out validation subset (default: 1,000).
        seed: Deterministic random seed for shuffling images.

    Returns:
        Tuple of (train_subset_dict, val_heldout_dict).
    """
    images = list(coco_data["images"])
    total_requested = train_count + val_heldout_count
    if len(images) < total_requested:
        raise ValueError(
            f"Dataset contains {len(images)} images, less than requested {total_requested}."
        )

    # Sort images first by ID to guarantee order across platforms before shuffling
    images.sort(key=lambda img: img["id"])
    rng = random.Random(seed)
    rng.shuffle(images)

    train_images = images[:train_count]
    val_heldout_images = images[train_count : train_count + val_heldout_count]

    train_image_ids: Set[int] = {img["id"] for img in train_images}
    val_heldout_image_ids: Set[int] = {img["id"] for img in val_heldout_images}

    # CRITICAL ZERO-LEAKAGE ASSERTION
    overlap = train_image_ids & val_heldout_image_ids
    assert len(overlap) == 0, (
        f"CRITICAL ERROR: Overlap detected between train and val-heldout subsets: {len(overlap)} images!"
    )
    assert len(train_images) == train_count, (
        f"Expected {train_count} train images, got {len(train_images)}"
    )
    assert len(val_heldout_images) == val_heldout_count, (
        f"Expected {val_heldout_count} held-out images, got {len(val_heldout_images)}"
    )

    # Filter annotations by image_id
    train_annotations: List[Dict[str, Any]] = []
    val_heldout_annotations: List[Dict[str, Any]] = []

    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in train_image_ids:
            train_annotations.append(ann)
        elif img_id in val_heldout_image_ids:
            val_heldout_annotations.append(ann)

    info = coco_data.get("info", {})
    licenses = coco_data.get("licenses", [])
    categories = coco_data.get("categories", [])

    train_dict = {
        "info": info,
        "licenses": licenses,
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }

    val_heldout_dict = {
        "info": info,
        "licenses": licenses,
        "images": val_heldout_images,
        "annotations": val_heldout_annotations,
        "categories": categories,
    }

    return train_dict, val_heldout_dict


def split_train_val_files(
    instances_train_path: Path,
    train_subset_path: Path,
    val_heldout_path: Path,
    train_count: int = 14000,
    val_heldout_count: int = 1000,
    seed: int = 42,
) -> None:
    """Read train annotations JSON, perform deterministic split and save subset files."""
    print(f"Reading full train annotations from {instances_train_path}...")
    with open(instances_train_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    print(
        f"Splitting into {train_count} train and {val_heldout_count} val-heldout (seed={seed})..."
    )
    train_dict, val_heldout_dict = split_coco_annotations(
        coco_data,
        train_count=train_count,
        val_heldout_count=val_heldout_count,
        seed=seed,
    )

    train_subset_path.parent.mkdir(parents=True, exist_ok=True)
    val_heldout_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing train subset JSON to {train_subset_path}...")
    with open(train_subset_path, "w", encoding="utf-8") as f:
        json.dump(train_dict, f)

    print(f"Writing val-heldout subset JSON to {val_heldout_path}...")
    with open(val_heldout_path, "w", encoding="utf-8") as f:
        json.dump(val_heldout_dict, f)

    print(
        f"Successfully generated: {train_subset_path} ({len(train_dict['images'])} images, {len(train_dict['annotations'])} boxes)"
    )
    print(
        f"Successfully generated: {val_heldout_path} ({len(val_heldout_dict['images'])} images, {len(val_heldout_dict['annotations'])} boxes)"
    )


def execute_mode_zip(data_dir: Path, skip_images: bool = False) -> None:
    """Download full zip archives for annotations, val2017, and train2017."""
    zips_dir = data_dir / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotations
    ann_zip = zips_dir / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"], ann_zip, "COCO Annotations")
    extract_zip(ann_zip, data_dir, check_dir=data_dir / "annotations")

    # 2. val2017
    if not skip_images:
        val_zip = zips_dir / "val2017.zip"
        download_file(COCO_URLS["val2017"], val_zip, "COCO val2017 (5k images)")
        extract_zip(val_zip, data_dir, check_dir=data_dir / "val2017")

        # 3. train2017
        train_zip = zips_dir / "train2017.zip"
        download_file(COCO_URLS["train2017"], train_zip, "COCO train2017 (118k images)")
        extract_zip(train_zip, data_dir, check_dir=data_dir / "train2017")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare COCO 2017 subset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/coco",
        help="Root directory for COCO dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zip", "annotations-only"],
        default="zip",
        help="Download mode: 'zip' for full download, 'annotations-only' for downloading annotations first",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=14000,
        help="Number of images for train subset (default: 14000)",
    )
    parser.add_argument(
        "--val-heldout-count",
        type=int,
        default=1000,
        help="Number of images for val-heldout subset (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting train2017 (default: 42)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge download size and skip interactive prompt",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    print("=" * 60)
    print(" COCO 2017 Dataset Preparation Pipeline")
    print("=" * 60)
    print(f"Target directory: {data_dir}")
    print(f"Download mode:    {args.mode}")
    print(f"Train subset:     {args.train_count:,} images")
    print(f"Val-heldout:      {args.val_heldout_count:,} images")
    print(f"Split seed:       {args.seed}")

    total_est = sum(ESTIMATED_SIZES.values()) / (1024 * 1024 * 1024)
    print(f"Estimated download size (Mode 'zip'): ~{total_est:.1f} GB")
    print(f"Estimated disk space required:         ~25-30 GB")

    if not args.yes:
        print("\n[WARNING] This action will download large files.")
        confirm = input("Do you want to proceed with download? (y/N): ").strip().lower()
        if confirm != "y":
            print("Download cancelled by user.")
            sys.exit(0)

    data_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "zip":
        execute_mode_zip(data_dir, skip_images=False)
    elif args.mode == "annotations-only":
        execute_mode_zip(data_dir, skip_images=True)

    # Perform split
    instances_train_path = data_dir / "annotations" / "instances_train2017.json"
    train_subset_path = data_dir / "train_subset14k.json"
    val_heldout_path = data_dir / "val_heldout1k.json"

    if instances_train_path.exists():
        split_train_val_files(
            instances_train_path=instances_train_path,
            train_subset_path=train_subset_path,
            val_heldout_path=val_heldout_path,
            train_count=args.train_count,
            val_heldout_count=args.val_heldout_count,
            seed=args.seed,
        )
    else:
        print(f"[NOTE] Annotations file not found yet at {instances_train_path}")


if __name__ == "__main__":
    main()
