"""Unit tests for inference coordinate decoder, Batched NMS, and COCO results format (Milestone M5).

Requirements from DEFINE_AND_DESIGN.md §2.4.3 & §2.7.4:
- Center offsets decode into [-0.5, 1.5] * stride.
- Width and height decode into [0, 16 * stride].
- Scores equal sigmoid(obj) * sigmoid(cls).
- Batched NMS correctly suppresses overlapping boxes within the same class while isolating different classes.
- Coordinates properly inverted back to original image space via letterbox parameters.
- to_coco_results outputs valid COCOeval JSON structures with mapped category IDs in [1, 90].
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest
import torch

from src.core.decoder import (
    YoloDecoder,
    decode_box_coordinates,
    to_coco_results,
)
from src.data.category_mapper import DEFAULT_MAPPER


def test_decode_box_coordinates_formula_and_ranges() -> None:
    """Verify coordinate decoding formulas and mathematical bounds."""
    # Stride 8 (P3)
    stride = 8
    H, W = 4, 4

    # 1. Zero logits: tx=0 -> sigmoid(0)=0.5 -> cx = (grid_x + 2*0.5 - 0.5)*8 = (grid_x + 0.5)*8
    #               tw=0 -> sigmoid(0)=0.5 -> w = (2*0.5)^2 * 4 * 8 = 32
    box_preds = torch.zeros((1, 4, H, W), dtype=torch.float32)
    boxes = decode_box_coordinates(box_preds, stride=stride)  # [1, 4, 4, 4]

    assert boxes.shape == (1, H, W, 4)
    # Check grid (0, 0): cx = 0.5 * 8 = 4.0, cy = 4.0, w = 32.0, h = 32.0
    # x1 = cx - w/2 = 4 - 16 = -12.0, x2 = 4 + 16 = 20.0
    x1, y1, x2, y2 = boxes[0, 0, 0].tolist()
    assert math.isclose(x2 - x1, 32.0, abs_tol=1e-5)
    assert math.isclose(y2 - y1, 32.0, abs_tol=1e-5)
    assert math.isclose((x1 + x2) / 2.0, 4.0, abs_tol=1e-5)
    assert math.isclose((y1 + y2) / 2.0, 4.0, abs_tol=1e-5)

    # 2. Extreme logits: check offset bounds [-0.5, 1.5] * stride
    # tx = +100 -> sigmoid ~ 1.0 -> cx ~ (grid_x + 1.5) * stride
    # tx = -100 -> sigmoid ~ 0.0 -> cx ~ (grid_x - 0.5) * stride
    box_preds_high = torch.full((1, 4, H, W), 100.0)
    boxes_high = decode_box_coordinates(box_preds_high, stride=stride)
    cx_high = (boxes_high[..., 0] + boxes_high[..., 2]) / 2.0
    w_high = boxes_high[..., 2] - boxes_high[..., 0]

    # For grid_x = 0: cx = 1.5 * 8 = 12.0, w = (2*1)^2 * 4 * 8 = 16 * 8 = 128.0
    assert math.isclose(cx_high[0, 0, 0].item(), 1.5 * stride, rel_tol=1e-4)
    assert math.isclose(w_high[0, 0, 0].item(), 16.0 * stride, rel_tol=1e-4)

    box_preds_low = torch.full((1, 4, H, W), -100.0)
    boxes_low = decode_box_coordinates(box_preds_low, stride=stride)
    cx_low = (boxes_low[..., 0] + boxes_low[..., 2]) / 2.0
    w_low = boxes_low[..., 2] - boxes_low[..., 0]

    # For grid_x = 0: cx = -0.5 * 8 = -4.0, w ~ 0.0
    assert math.isclose(cx_low[0, 0, 0].item(), -0.5 * stride, rel_tol=1e-4)
    assert math.isclose(w_low[0, 0, 0].item(), 0.0, abs_tol=1e-4)


def test_yolo_decoder_shapes_and_composite_scores() -> None:
    """Verify YoloDecoder correctly aggregates 5,376 anchors across P3, P4, P5."""
    decoder = YoloDecoder(strides=[8, 16, 32], conf_threshold=0.05, nms_iou_threshold=0.60)

    B = 2
    raw_predictions = {
        "P3": {
            "box": torch.zeros((B, 4, 64, 64)),
            "obj": torch.zeros((B, 1, 64, 64)),  # sigmoid(0) = 0.5
            "cls": torch.zeros((B, 80, 64, 64)),  # sigmoid(0) = 0.5
        },
        "P4": {
            "box": torch.zeros((B, 4, 32, 32)),
            "obj": torch.zeros((B, 1, 32, 32)),
            "cls": torch.zeros((B, 80, 32, 32)),
        },
        "P5": {
            "box": torch.zeros((B, 4, 16, 16)),
            "obj": torch.zeros((B, 1, 16, 16)),
            "cls": torch.zeros((B, 80, 16, 16)),
        },
    }

    all_boxes, all_scores = decoder.decode_predictions(raw_predictions)

    # Total anchors = 64*64 + 32*32 + 16*16 = 4096 + 1024 + 256 = 5376
    assert all_boxes.shape == (B, 5376, 4)
    assert all_scores.shape == (B, 5376, 80)

    # When obj=0 and cls=0, composite score = sigmoid(0)*sigmoid(0) = 0.5 * 0.5 = 0.25
    assert torch.allclose(all_scores, torch.tensor(0.25), atol=1e-5)


def test_batched_nms_class_isolation() -> None:
    """Verify Batched NMS isolates suppression across different classes while suppressing same class."""
    decoder = YoloDecoder(conf_threshold=0.1, nms_iou_threshold=0.5, max_detections=100)

    # 3 candidate boxes:
    # Box A: [10, 10, 50, 50], score 0.9 for class 0 (person)
    # Box B: [12, 12, 48, 48] (IoU ~ 0.8 with Box A), score 0.8 for class 0 (person) -> MUST BE SUPPRESSED
    # Box C: [10, 10, 50, 50] (identical box), score 0.85 for class 2 (car) -> MUST BE KEPT (different class)
    boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],  # Box A
            [12.0, 12.0, 48.0, 48.0],  # Box B
            [10.0, 10.0, 50.0, 50.0],  # Box C
        ]
    )

    scores = torch.zeros((3, 80))
    scores[0, 0] = 0.90  # Box A, class 0
    scores[1, 0] = 0.80  # Box B, class 0
    scores[2, 2] = 0.85  # Box C, class 2

    det = decoder.postprocess_single_image(boxes, scores)

    # Expect exactly 2 boxes remaining: Box A (class 0) and Box C (class 2)
    assert len(det["boxes"]) == 2
    assert len(det["scores"]) == 2
    assert len(det["labels"]) == 2

    # Check labels: one class 0, one class 2
    labels = det["labels"].tolist()
    assert 0 in labels
    assert 2 in labels
    # Box B should have been suppressed by Box A
    scores_list = det["scores"].tolist()
    assert math.isclose(max(scores_list), 0.90, abs_tol=1e-4)


def test_coordinate_restoration_to_original_image() -> None:
    """Verify inversion from 512x512 letterbox canvas back to original image dimensions."""
    decoder = YoloDecoder(conf_threshold=0.1)

    # Original image: (orig_h=600, orig_w=1200) -> aspect 2:1
    # Letterbox to 512x512:
    # scale r = 512 / 1200 = 0.426667
    # new_w = 512, new_h = 600 * 0.426667 = 256
    # pad_w = 0, pad_h = (512 - 256) / 2 = 128
    orig_h, orig_w = 600, 1200
    r = 512.0 / 1200.0
    pad_w = 0.0
    pad_h = (512.0 - 256.0) / 2.0  # 128.0

    metadata = {
        "orig_size": (orig_h, orig_w),
        "scale_ratio": r,
        "pad_offsets": (pad_w, pad_h),
        "image_id": 42,
    }

    # Suppose model predicted a box at center of valid region:
    # On letterbox canvas: x in [0, 512], y in [128, 384]
    box_canvas = torch.tensor([[50.0, 128.0 + 40.0, 200.0, 128.0 + 100.0]])
    score_canvas = torch.zeros((1, 80))
    score_canvas[0, 5] = 0.95

    det = decoder.postprocess_single_image(box_canvas, score_canvas, metadata=metadata)

    pred_box = det["boxes"][0]
    # Expected original coordinates:
    # x1_orig = (50 - 0) / r
    # y1_orig = ((128 + 40) - 128) / r = 40 / r
    # x2_orig = 200 / r
    # y2_orig = 100 / r
    assert math.isclose(pred_box[0].item(), 50.0 / r, rel_tol=1e-4)
    assert math.isclose(pred_box[1].item(), 40.0 / r, rel_tol=1e-4)
    assert math.isclose(pred_box[2].item(), 200.0 / r, rel_tol=1e-4)
    assert math.isclose(pred_box[3].item(), 100.0 / r, rel_tol=1e-4)

    # Ensure coordinates are within [0, orig_w] and [0, orig_h]
    assert 0.0 <= pred_box[0] <= orig_w
    assert 0.0 <= pred_box[1] <= orig_h
    assert 0.0 <= pred_box[2] <= orig_w
    assert 0.0 <= pred_box[3] <= orig_h


def test_empty_predictions_handling() -> None:
    """Verify decoder gracefully handles empty predictions without error."""
    decoder = YoloDecoder(conf_threshold=0.5)

    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    scores = torch.full((1, 80), 0.1)  # All below 0.5

    det = decoder.postprocess_single_image(boxes, scores)
    assert det["boxes"].shape == (0, 4)
    assert det["scores"].shape == (0,)
    assert det["labels"].shape == (0,)

    coco_res = to_coco_results([det], image_ids=[999])
    assert coco_res == []


def test_to_coco_results_format() -> None:
    """Verify to_coco_results produces the exact JSON dictionary format required by COCOeval."""
    detections = [
        {
            "boxes": torch.tensor([[100.256, 150.789, 250.333, 350.888]]),  # x1, y1, x2, y2
            "scores": torch.tensor([0.923456]),
            "labels": torch.tensor([0]),  # class 0 -> category_id 1 (person)
        }
    ]
    image_ids = [12345]

    coco_results = to_coco_results(detections, image_ids, mapper=DEFAULT_MAPPER)

    assert len(coco_results) == 1
    res = coco_results[0]

    assert res["image_id"] == 12345
    assert res["category_id"] == 1  # Person in COCO
    # COCO bbox format: [x, y, w, h]
    # x = 100.26, y = 150.79, w = 250.333 - 100.256 = 150.08, h = 350.888 - 150.789 = 200.10
    x, y, w, h = res["bbox"]
    assert math.isclose(x, 100.26, abs_tol=0.01)
    assert math.isclose(y, 150.79, abs_tol=0.01)
    assert math.isclose(w, 150.08, abs_tol=0.02)
    assert math.isclose(h, 200.10, abs_tol=0.02)
    assert math.isclose(res["score"], 0.9235, abs_tol=1e-4)
