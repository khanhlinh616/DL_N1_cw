"""Single image demo inference script (Milestone M9).

Usage:
    python src/demo.py --image path/to/image.jpg --checkpoint outputs/checkpoints/best_map.pth

TODO: Implement inference pipeline in Milestone M9.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detector demo on a single image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input RGB image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_map.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence score threshold for filtering detections",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/demo",
        help="Directory to save visualized detection output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(" YOLO Detector Single-Image Demo CLI (Skeleton)")
    print("=" * 60)
    print(f"Image:       {args.image}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Conf thresh: {args.conf_thresh}")
    print(f"Output dir:  {args.output_dir}")
    print("\n[NOTE] Demo inference pipeline will be finalized in Milestone M9.")


if __name__ == "__main__":
    main()
