"""Unit tests for transforms module (Milestone M1).

Verifies:
- Output image shape is strictly [3, 512, 512].
- Letterbox preserves aspect ratio with correct padding offsets.
- Perfect roundtrip box recovery (original -> letterbox -> inverse letterbox) with error < 1 pixel.
- Horizontal flip preserves box bounds correctly.
- Edge cases: empty box sets [0, 4], various aspect ratios (wide, tall, square).
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.transforms import (
    DetectionTransforms,
    inverse_letterbox_boxes,
    letterbox_boxes,
    letterbox_image,
    random_horizontal_flip,
)


@pytest.mark.parametrize(
    "orig_w, orig_h",
    [
        (640, 480),   # Standard landscape
        (1920, 1080), # Wide 16:9
        (480, 640),   # Portrait
        (500, 1500),  # Very tall
        (1500, 500),  # Very wide
        (800, 800),   # Square
        (512, 512),   # Already target size
    ],
)
def test_letterbox_roundtrip_box_accuracy(orig_w: int, orig_h: int) -> None:
    """Test that original box -> letterbox -> inverse letterbox error is strictly < 1.0 px."""
    img = Image.new("RGB", (orig_w, orig_h), color=(100, 150, 200))
    orig_boxes = torch.tensor(
        [
            [10.0, 15.0, 100.0, 120.0],
            [orig_w * 0.2, orig_h * 0.3, orig_w * 0.6, orig_h * 0.7],
            [orig_w - 50.0, orig_h - 60.0, float(orig_w), float(orig_h)],
        ],
        dtype=torch.float32,
    )

    lb_img, r, pad_offsets = letterbox_image(img, target_size=512)
    assert lb_img.size == (512, 512)

    lb_boxes = letterbox_boxes(orig_boxes, scale_ratio=r, pad_offsets=pad_offsets)

    # All letterboxed boxes must be within [0, 512]
    assert (lb_boxes >= 0.0).all()
    assert (lb_boxes <= 512.0).all()

    # Invert back to original coordinates
    recovered_boxes = inverse_letterbox_boxes(
        lb_boxes, scale_ratio=r, pad_offsets=pad_offsets, orig_size=(orig_h, orig_w)
    )

    # Check maximum coordinate error
    error = torch.max(torch.abs(orig_boxes - recovered_boxes)).item()
    assert error < 1.0, f"Roundtrip coordinate error {error:.4f} px exceeded 1.0 px threshold!"


def test_empty_boxes_handling() -> None:
    """Test that transform handles empty box tensor [0, 4] gracefully."""
    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    lb_boxes = letterbox_boxes(empty_boxes, scale_ratio=0.5, pad_offsets=(50.0, 50.0))
    assert lb_boxes.shape == (0, 4)

    restored = inverse_letterbox_boxes(
        lb_boxes, scale_ratio=0.5, pad_offsets=(50.0, 50.0), orig_size=(480, 640)
    )
    assert restored.shape == (0, 4)


def test_horizontal_flip_coordinates() -> None:
    """Test that horizontal flip mirrors coordinates accurately."""
    img = Image.new("RGB", (512, 512), color=(50, 50, 50))
    boxes = torch.tensor([[100.0, 50.0, 200.0, 150.0]], dtype=torch.float32)

    # Force flip with p=1.0
    _, flipped_boxes, did_flip = random_horizontal_flip(img, boxes, img_size=512, p=1.0)
    assert did_flip is True

    # Expected: x1' = 512 - 200 = 312, x2' = 512 - 100 = 412
    assert torch.isclose(flipped_boxes[0, 0], torch.tensor(312.0))
    assert torch.isclose(flipped_boxes[0, 1], torch.tensor(50.0))
    assert torch.isclose(flipped_boxes[0, 2], torch.tensor(412.0))
    assert torch.isclose(flipped_boxes[0, 3], torch.tensor(150.0))


def test_detection_transforms_output_shapes_and_metadata() -> None:
    """Test DetectionTransforms callable pipeline."""
    img = Image.new("RGB", (800, 600))
    boxes = torch.tensor([[50.0, 40.0, 300.0, 400.0]], dtype=torch.float32)
    labels = torch.tensor([5], dtype=torch.int64)

    pipeline = DetectionTransforms(img_size=512, is_train=False)
    img_tensor, out_boxes, out_labels, metadata = pipeline(img, boxes, labels)

    assert img_tensor.shape == (3, 512, 512)
    assert img_tensor.dtype == torch.float32
    assert out_boxes.shape == (1, 4)
    assert out_labels.shape == (1,)

    # Verify metadata fields
    assert "orig_size" in metadata
    assert metadata["orig_size"] == (600, 800)
    assert "scale_ratio" in metadata
    assert "pad_offsets" in metadata
    assert isinstance(metadata["scale_ratio"], float)
    assert len(metadata["pad_offsets"]) == 2
