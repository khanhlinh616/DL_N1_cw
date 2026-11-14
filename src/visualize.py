"""Prediction visualization and qualitative error analysis script (Milestone M8).

Features:
- Overlay Ground Truth boxes (green) vs Predicted boxes (blue/red).
- Export >= 15 sample images for final report verification.
- Group analysis: top detections, small object misses, heavy occlusion cases, false positives.

TODO: Finalize visualization in Milestone M8.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector predictions and error cases")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/evaluation_results/predictions_val2017.json",
        help="Path to COCO predictions JSON",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/coco/annotations/instances_val2017.json",
        help="Path to ground truth instances JSON",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/coco/val2017",
        help="Directory containing original test images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/eval_samples",
        help="Directory to save visual comparison images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of samples to visualize (minimum 15 required for M8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(" YOLO Detector Visualization CLI (Skeleton)")
    print("=" * 60)
    print(f"Predictions: {args.predictions}")
    print(f"Annotations: {args.annotations}")
    print(f"Image dir:   {args.image_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Num samples: {args.num_samples}")
    print("\n[NOTE] Visualizer will be finalized in Milestone M8.")


if __name__ == "__main__":
    main()
