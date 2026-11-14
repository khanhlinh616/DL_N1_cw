"""Unit tests for Evaluation Pipeline and Safety Constraints (Milestone M5).

Verifies:
- Safety checks block unauthorized access to val2017 without --final.
- Lock file prevents duplicate evaluation runs on val2017.
- evaluate_coco runs end-to-end on synthetic data and returns valid COCOeval metrics dictionary.
- Output predictions JSON is properly formatted and saved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from src.data.dataset import COCODataset, detection_collate_fn
from src.evaluate import check_eval_safety, evaluate_coco
from src.models.detector import build_detector
from tests.test_dataset import create_synthetic_coco_env


def test_safety_check_blocks_val2017_without_final(tmp_path: Path) -> None:
    """Verify check_eval_safety blocks execution on val2017 if is_final=False."""
    with pytest.raises(SystemExit) as exc_info:
        check_eval_safety("val2017", is_final=False, output_dir=tmp_path / "out")
    assert exc_info.value.code == 1


def test_safety_check_blocks_val2017_if_locked(tmp_path: Path) -> None:
    """Verify check_eval_safety blocks execution if final_eval.lock already exists."""
    out_dir = tmp_path / "outputs" / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = tmp_path / "outputs" / "final_eval.lock"
    lock_file.write_text("locked", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        check_eval_safety("val2017", is_final=True, output_dir=out_dir)
    assert exc_info.value.code == 1


def test_evaluate_coco_synthetic_pipeline(tmp_path: Path) -> None:
    """Test evaluate_coco runs end-to-end on a synthetic dataset."""
    img_dir, ann_file = create_synthetic_coco_env(tmp_path)
    dataset = COCODataset(root=img_dir, annotation_file=ann_file, img_size=512, is_train=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)

    coco_gt = COCO(str(ann_file))
    model = build_detector(num_classes=80, pretrained_backbone=False)
    pred_json = tmp_path / "predictions_test.json"

    metrics = evaluate_coco(
        model=model,
        dataloader=loader,
        coco_gt=coco_gt,
        device=torch.device("cpu"),
        output_json_path=pred_json,
        conf_threshold=0.01,
        nms_iou_threshold=0.6,
    )

    assert pred_json.exists()
    assert "mAP_50_95" in metrics
    assert "mAP_50" in metrics
    assert "mAP_75" in metrics
    assert "AR_100" in metrics
    assert isinstance(metrics["mAP_50_95"], float)
