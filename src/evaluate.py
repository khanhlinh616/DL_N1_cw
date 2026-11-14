"""Evaluation script for YOLO-style detector on COCO validation sets.

Invariant Rules (RULES.md §17.3):
- Held-out 1k dataset is used during development, tuning, and ablations.
- val2017 (5k images) can ONLY be evaluated ONCE in Milestone M8.
- Must reject running on val2017 if '--final' flag is omitted.
- Writing 'outputs/final_eval.lock' and logging to 'docs/audit-log.md'.
- Re-running when lock file exists is strictly forbidden.

TODO: Full COCOeval integration in Milestone M5 & M8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-style detector on COCO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_map.pth",
        help="Path to model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["heldout", "val2017"],
        default="heldout",
        help="Dataset split to evaluate on: 'heldout' (1k) or 'val2017' (5k)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Mandatory safety flag to execute official evaluation on val2017 (Milestone M8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation_results",
        help="Directory to save predictions JSON",
    )
    return parser.parse_args()


def check_eval_safety(dataset: str, is_final: bool, output_dir: Path) -> None:
    """Enforce RULES.md §17.3 safety constraints on test set evaluation."""
    if dataset == "val2017":
        if not is_final:
            print("\n[BLOCKED] Running on 'val2017' requires the '--final' flag.")
            print("RULES.md §17.3: val2017 is strictly reserved for one-time final evaluation at M8.")
            sys.exit(1)

        lock_file = output_dir.parent / "final_eval.lock"
        if lock_file.exists():
            print(f"\n[BLOCKED] Lock file exists: {lock_file}")
            print("RULES.md §17.3: Official evaluation on val2017 has already been performed.")
            print("Repeated runs on the test set are forbidden to prevent data snooping.")
            sys.exit(1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    check_eval_safety(args.dataset, args.final, output_dir)

    print("=" * 60)
    print(" YOLO Detector Evaluation CLI (Skeleton)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Final flag: {args.final}")
    print("\n[NOTE] Evaluation pipeline will be implemented in Milestone M5.")


if __name__ == "__main__":
    main()
