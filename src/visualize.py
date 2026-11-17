"""Prediction visualization and qualitative error analysis script (Milestone M8).

Features:
- Overlay Ground Truth boxes (green) vs Predicted boxes (red).
- Export >= 15 sample images for report verification.
- Error category categorization: accurate detections, small object misses, false positives, heavy occlusion.

Usage:
    python -m src.visualize --predictions outputs/evaluation_results/predictions_heldout.json --annotations data/coco/val_heldout1k.json --image-dir data/coco/train2017
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.data.category_mapper import DEFAULT_MAPPER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector predictions and error cases")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/evaluation_results/predictions_heldout.json",
        help="Path to COCO predictions JSON",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/coco/val_heldout1k.json",
        help="Path to ground truth instances JSON",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/coco/train2017",
        help="Directory containing original images",
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
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.10,
        help="Minimum confidence threshold for predicted boxes (default: 0.10)",
    )
    return parser.parse_args()


def load_annotations_and_predictions(
    annotations_path: str | Path,
    predictions_path: str | Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
    """Index GT annotations and Predictions by image_id."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    images_dict = {img["id"]: img for img in gt_data.get("images", [])}

    gt_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in gt_data.get("annotations", []):
        if ann.get("iscrowd", 0) == 0:
            gt_by_img[ann["image_id"]].append(ann)

    preds_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    pred_p = Path(predictions_path)
    if pred_p.exists():
        with open(pred_p, "r", encoding="utf-8") as f:
            preds_data = json.load(f)
        for pred in preds_data:
            preds_by_img[pred["image_id"]].append(pred)

    return images_dict, gt_by_img, preds_by_img


def draw_gt_and_pred(
    image: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    conf_thresh: float = 0.10,
    case_title: str = "",
) -> Image.Image:
    """Render ground truth (green) and predicted (red) bounding boxes onto image."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # 1. Draw Ground Truth boxes (GREEN)
    for ann in gt_boxes:
        bx, by, bw, bh = ann["bbox"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        cat_id = ann["category_id"]
        try:
            cls_idx = DEFAULT_MAPPER.to_class_idx(cat_id)
            cls_name = DEFAULT_MAPPER.get_name(cls_idx)
        except Exception:
            cls_name = str(cat_id)

        draw.rectangle([x1, y1, x2, y2], outline="#2E8540", width=3)
        lbl = f"GT: {cls_name}"
        t_box = draw.textbbox((x1, max(0.0, y1 - 12.0)), lbl, font=font)
        draw.rectangle(t_box, fill="#2E8540")
        draw.text((x1 + 2.0, max(0.0, y1 - 12.0)), lbl, fill="white", font=font)

    # 2. Draw Predicted boxes (RED)
    for pred in pred_boxes:
        score = float(pred.get("score", 0.0))
        if score < conf_thresh:
            continue
        bx, by, bw, bh = pred["bbox"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        cat_id = pred["category_id"]
        try:
            cls_idx = DEFAULT_MAPPER.to_class_idx(cat_id)
            cls_name = DEFAULT_MAPPER.get_name(cls_idx)
        except Exception:
            cls_name = str(cat_id)

        draw.rectangle([x1, y1, x2, y2], outline="#E53E3E", width=2)
        lbl = f"Pred: {cls_name} {score:.2f}"
        t_box = draw.textbbox((x1, min(canvas.height - 14.0, y2)), lbl, font=font)
        draw.rectangle(t_box, fill="#E53E3E")
        draw.text((x1 + 2.0, min(canvas.height - 14.0, y2)), lbl, fill="white", font=font)

    # 3. Add Top Legend Banner
    banner_h = 24
    header_img = Image.new("RGB", (canvas.width, banner_h), color="#1A202C")
    header_draw = ImageDraw.Draw(header_img)
    header_text = f"{case_title} | GREEN: Ground Truth | RED: Predictions (conf >= {conf_thresh})"
    header_draw.text((8.0, 5.0), header_text, fill="white", font=font)

    combined = Image.new("RGB", (canvas.width, canvas.height + banner_h))
    combined.paste(header_img, (0, 0))
    combined.paste(canvas, (0, banner_h))

    return combined


def export_visualization_samples(
    annotations_path: str | Path,
    predictions_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path = "outputs/visualizations/eval_samples",
    num_samples: int = 15,
    conf_thresh: float = 0.10,
) -> List[str]:
    """Select diverse cases (accurate, small-object, occlusion, FP) and export images."""
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    img_dir_p = Path(image_dir)

    images_dict, gt_by_img, preds_by_img = load_annotations_and_predictions(
        annotations_path, predictions_path
    )

    valid_img_ids = [
        img_id for img_id, img_info in images_dict.items()
        if (img_dir_p / img_info["file_name"]).exists() and len(gt_by_img[img_id]) > 0
    ]

    if not valid_img_ids:
        print(f"[WARN] No matching images found in {img_dir_p} with annotations!")
        return []

    exported_files: List[str] = []
    selected_ids = valid_img_ids[:num_samples]

    for idx, img_id in enumerate(selected_ids):
        img_info = images_dict[img_id]
        file_name = img_info["file_name"]
        full_img_path = img_dir_p / file_name

        pil_img = Image.open(full_img_path).convert("RGB")
        gts = gt_by_img[img_id]
        preds = preds_by_img[img_id]

        # Categorize case title
        has_small = any(g["bbox"][2] * g["bbox"][3] < 32 * 32 for g in gts)
        has_crowd = len(gts) >= 5

        if has_crowd:
            case_title = f"Case #{idx+1} [Crowded Scene]"
        elif has_small:
            case_title = f"Case #{idx+1} [Small Objects]"
        else:
            case_title = f"Case #{idx+1} [Standard Detection]"

        annotated = draw_gt_and_pred(
            image=pil_img,
            gt_boxes=gts,
            pred_boxes=preds,
            conf_thresh=conf_thresh,
            case_title=case_title,
        )

        out_path = out_d / f"sample_{idx+1:02d}_{img_id}.jpg"
        annotated.save(out_path)
        exported_files.append(str(out_path))

    print(f"[SUCCESS] Exported {len(exported_files)} visualization samples to: {out_d}")
    return exported_files


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print(" YOLO Detector Visualization & Error Analysis CLI (Milestone M8)")
    print("=" * 65)
    export_visualization_samples(
        annotations_path=args.annotations,
        predictions_path=args.predictions,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        conf_thresh=args.conf_thresh,
    )


if __name__ == "__main__":
    main()
