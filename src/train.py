"""Main training script for YOLO-style object detector (Milestone M4 & M6).

Usage:
    python src/train.py --config configs/base_config.yaml --epochs 20
    python src/train.py --config configs/base_config.yaml --max-images 8 --epochs 1  # Smoke run

TODO: Full training loop implementation in Milestone M4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-style detector on COCO subset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images for smoke run / dry-run testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(" YOLO Detector Training CLI (Skeleton)")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Epochs:     {args.epochs}")
    print(f"Resume:     {args.resume}")
    print(f"Max images: {args.max_images}")
    print("\n[NOTE] Training engine will be implemented in Milestone M4.")


if __name__ == "__main__":
    main()
