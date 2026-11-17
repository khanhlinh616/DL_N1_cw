"""Unit tests for visualization script (Milestone M8)."""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from src.visualize import (
    load_annotations_and_predictions,
    draw_gt_and_pred,
    export_visualization_samples,
)


def test_load_annotations_and_predictions(tmp_path: Path) -> None:
    ann_path = tmp_path / "ann.json"
    pred_path = tmp_path / "pred.json"

    ann_data = {
        "images": [{"id": 1, "file_name": "001.jpg"}],
        "annotations": [{"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "iscrowd": 0}],
    }
    pred_data = [
        {"image_id": 1, "category_id": 1, "bbox": [12, 12, 48, 48], "score": 0.9},
    ]

    with open(ann_path, "w") as f:
        json.dump(ann_data, f)
    with open(pred_path, "w") as f:
        json.dump(pred_data, f)

    images_dict, gt_by_img, preds_by_img = load_annotations_and_predictions(ann_path, pred_path)

    assert 1 in images_dict
    assert len(gt_by_img[1]) == 1
    assert len(preds_by_img[1]) == 1


def test_draw_gt_and_pred() -> None:
    img = Image.new("RGB", (300, 200), color=(200, 200, 200))
    gts = [{"category_id": 1, "bbox": [10, 10, 50, 50]}]
    preds = [{"category_id": 1, "bbox": [15, 15, 45, 45], "score": 0.85}]

    annotated = draw_gt_and_pred(img, gts, preds, conf_thresh=0.1, case_title="Test Case")
    assert isinstance(annotated, Image.Image)
    assert annotated.width == 300
    assert annotated.height == 224  # 200 + 24 banner


def test_export_visualization_samples_smoke(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    dummy_img = Image.new("RGB", (100, 100), color=(150, 150, 150))
    dummy_img.save(img_dir / "001.jpg")

    ann_path = tmp_path / "ann.json"
    pred_path = tmp_path / "pred.json"

    with open(ann_path, "w") as f:
        json.dump({
            "images": [{"id": 1, "file_name": "001.jpg"}],
            "annotations": [{"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}],
        }, f)
    with open(pred_path, "w") as f:
        json.dump([], f)

    out_dir = tmp_path / "viz_out"
    exported = export_visualization_samples(
        annotations_path=ann_path,
        predictions_path=pred_path,
        image_dir=img_dir,
        output_dir=out_dir,
        num_samples=1,
    )

    assert len(exported) == 1
    assert Path(exported[0]).exists()
