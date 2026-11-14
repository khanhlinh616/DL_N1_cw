"""Official and Intermediate Evaluation CLI and Pipeline for YOLO-style Detector.

Strict Compliance Rules (RULES.md §17.3):
- Held-out 1k dataset (or subsets thereof) is strictly used during development and tuning.
- val2017 (5,000 images) is reserved STRICTLY for one-time final evaluation at Milestone M8.
- Evaluation on val2017 MUST require the '--final' flag; otherwise the script blocks execution.
- Executing with '--final' creates 'outputs/final_eval.lock' and logs an audit record to 'docs/audit-log.md'.
- Re-running when 'outputs/final_eval.lock' exists is strictly prohibited.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from src.core.decoder import YoloDecoder, to_coco_results
from src.data.dataset import COCODataset, detection_collate_fn
from src.models.detector import build_detector


def get_git_commit() -> str:
    """Retrieve current short git commit hash."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def check_eval_safety(dataset: str, is_final: bool, output_dir: Path) -> None:
    """Enforce RULES.md §17.3 safety constraints on test set evaluation."""
    if dataset == "val2017":
        if not is_final:
            print("\n[BLOCKED] Execution halted: Running on 'val2017' requires the '--final' flag.")
            print("RULES.md §17.3: val2017 is strictly reserved for one-time final evaluation at Milestone M8.")
            print("To evaluate during development/tuning, use '--dataset heldout'.")
            sys.exit(1)

        lock_file = output_dir.parent / "final_eval.lock"
        if lock_file.exists():
            print(f"\n[BLOCKED] Execution halted: Final evaluation lock file already exists at '{lock_file}'.")
            print("RULES.md §17.3: Official evaluation on val2017 has already been completed.")
            print("Repeated runs on the test set are forbidden to maintain zero data snooping.")
            sys.exit(1)


def record_final_eval(
    output_dir: Path,
    checkpoint_path: str,
    metrics: Dict[str, float],
    audit_file: Path = Path("docs/audit-log.md"),
) -> None:
    """Create final_eval.lock and log audit entry after successful final evaluation."""
    lock_file = output_dir.parent / "final_eval.lock"
    commit_hash = get_git_commit()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lock_payload = {
        "timestamp": timestamp,
        "commit": commit_hash,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_payload, f, indent=2)

    # Append audit log table entry
    if audit_file.exists():
        row = (
            f"| {timestamp} | {commit_hash} | `{checkpoint_path}` | `val2017 (5k)` | "
            f"Baseline | {metrics.get('mAP_50_95', 0.0):.4f} | COMPLETED | Official Milestone M8 Evaluation |\n"
        )
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(row)
    print(f"\n[AUDIT] Locked evaluation state saved to '{lock_file}' and logged in '{audit_file}'.")


def evaluate_coco(
    model: nn.Module,
    dataloader: DataLoader,
    coco_gt: Optional[COCO] = None,
    device: Optional[torch.device] = None,
    output_json_path: Optional[Path | str] = None,
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.60,
    max_detections: int = 100,
) -> Dict[str, float]:
    """Run full evaluation on a dataset using YoloDecoder and pycocotools COCOeval.

    Args:
        model: Trained or initialized YOLODetector model.
        dataloader: PyTorch DataLoader yielding (images, targets).
        coco_gt: Ground truth pycocotools.coco.COCO instance.
        device: Device to run inference on.
        output_json_path: Optional path to save prediction results JSON.
        conf_threshold: Confidence threshold for candidate detection filtering.
        nms_iou_threshold: IoU threshold for Batched NMS.
        max_detections: Maximum detections per image retained after NMS.

    Returns:
        Dictionary mapping standard metric names to float scores.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(dev)

    decoder = YoloDecoder(
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )

    all_coco_results: List[Dict[str, Any]] = []
    evaluated_img_ids: List[int] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(dev)
            raw_predictions = model(images)

            # Metadata extraction for coordinate restoration
            metadatas = [
                {
                    "orig_size": t["orig_size"],
                    "scale_ratio": t["scale_ratio"],
                    "pad_offsets": t["pad_offsets"],
                    "image_id": t["image_id"],
                }
                for t in targets
            ]

            batch_detections = decoder(
                raw_predictions,
                metadatas=metadatas,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )

            batch_img_ids = [int(t["image_id"]) for t in targets]
            evaluated_img_ids.extend(batch_img_ids)

            batch_coco_results = to_coco_results(batch_detections, image_ids=batch_img_ids)
            all_coco_results.extend(batch_coco_results)

    # Save predictions to file if requested
    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(all_coco_results, f, indent=2)
        print(f"[EVAL] Saved {len(all_coco_results)} predictions to '{out_p}'.")

    # If no ground truth COCO instance, return basic prediction counts
    if coco_gt is None:
        return {"num_predictions": float(len(all_coco_results))}

    # Pycocotools edge-case: empty results list crashes coco.loadRes([])
    if len(all_coco_results) == 0:
        print("[WARNING] Zero detections above confidence threshold were produced by the model.")
        return {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_s": 0.0,
            "mAP_m": 0.0,
            "mAP_l": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_s": 0.0,
            "AR_m": 0.0,
            "AR_l": 0.0,
        }

    # Execute official pycocotools COCOeval
    coco_dt = coco_gt.loadRes(all_coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Restrict evaluation to exactly the evaluated image IDs (crucial for subset/smoke runs)
    coco_eval.params.imgIds = list(set(evaluated_img_ids))

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_s": float(stats[3]),
        "mAP_m": float(stats[4]),
        "mAP_l": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_s": float(stats[9]),
        "AR_m": float(stats[10]),
        "AR_l": float(stats[11]),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-style detector on COCO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model weights checkpoint (.pth). If omitted, initializes model without weights.",
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
        help="Directory to save predictions JSON and evaluation results",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=None,
        help="Optional override path to annotations JSON file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional override path to images directory",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.05,
        help="Confidence threshold for prediction filtering (default: 0.05)",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.60,
        help="IoU threshold for Batched NMS (default: 0.60)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Maximum detections per image retained after NMS (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on maximum number of images evaluated (for pilot/smoke tests)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on ('cpu' or 'cuda')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Enforce safety checks
    check_eval_safety(args.dataset, args.final, output_dir)

    # 2. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(" YOLO Detector Evaluation CLI")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Dataset split:   {args.dataset}")
    print(f"Final mode:      {args.final}")
    print(f"Conf threshold:  {args.conf_thresh}")
    print(f"NMS threshold:   {args.nms_thresh}")
    print(f"Max detections:  {args.max_detections}")

    # 3. Resolve paths
    if args.dataset == "val2017":
        ann_file = Path(args.ann_file or "data/coco/annotations/instances_val2017.json")
        images_dir = Path(args.images_dir or "data/coco/val2017")
    else:
        ann_file = Path(args.ann_file or "data/coco/val_heldout1k.json")
        images_dir = Path(args.images_dir or "data/coco/train2017")

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print(f"Annotations:     {ann_file}")
    print(f"Images dir:      {images_dir}")

    # 4. Build Model
    model = build_detector(num_classes=80, pretrained_backbone=False)
    if args.checkpoint is not None and Path(args.checkpoint).exists():
        print(f"Loading weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Note: Running with initialized model (no checkpoint provided or found).")

    # 5. Build Dataset and DataLoader
    val_dataset = COCODataset(root=images_dir, annotation_file=ann_file, img_size=512, is_train=False)

    if args.max_images is not None and args.max_images < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(args.max_images)))  # type: ignore[assignment]
        print(f"Subsetting evaluation to first {args.max_images} images.")

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    coco_gt = COCO(str(ann_file))
    output_json = output_dir / f"predictions_{args.dataset}.json"

    # 6. Run Evaluation
    print("\nExecuting evaluation pipeline...")
    metrics = evaluate_coco(
        model=model,
        dataloader=dataloader,
        coco_gt=coco_gt,
        device=device,
        output_json_path=output_json,
        conf_threshold=args.conf_thresh,
        nms_iou_threshold=args.nms_thresh,
        max_detections=args.max_detections,
    )

    # 7. Print Results Table
    print("\n" + "=" * 65)
    print(f" Evaluation Metrics Summary ({args.dataset})")
    print("=" * 65)
    print(f" mAP@[.50:.95] : {metrics.get('mAP_50_95', 0.0):.4f}")
    print(f" mAP@.50       : {metrics.get('mAP_50', 0.0):.4f}")
    print(f" mAP@.75       : {metrics.get('mAP_75', 0.0):.4f}")
    print(f" mAP (small)   : {metrics.get('mAP_s', 0.0):.4f}")
    print(f" mAP (medium)  : {metrics.get('mAP_m', 0.0):.4f}")
    print(f" mAP (large)   : {metrics.get('mAP_l', 0.0):.4f}")
    print(f" AR@100        : {metrics.get('AR_100', 0.0):.4f}")
    print("=" * 65)

    # 8. If final run on val2017, record audit
    if args.dataset == "val2017" and args.final:
        record_final_eval(output_dir, args.checkpoint or "none", metrics)


if __name__ == "__main__":
    main()
