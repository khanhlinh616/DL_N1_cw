"""Unit tests for box operations and CIoU cross-validation (Milestone M3).

Verifies:
- Coordinate conversions cxcywh <-> xyxy.
- IoU pairwise matrix computation.
- Strict cross-validation of custom CIoU loss against torchvision.ops.complete_box_iou_loss.
"""

from __future__ import annotations

import torch
from torchvision.ops import complete_box_iou_loss

from src.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    ciou_loss,
)


def test_box_conversions_roundtrip() -> None:
    """Test cxcywh <-> xyxy roundtrip consistency."""
    boxes_xyxy = torch.tensor(
        [
            [10.0, 20.0, 100.0, 150.0],
            [50.0, 60.0, 70.0, 90.0],
            [0.0, 0.0, 512.0, 512.0],
        ],
        dtype=torch.float32,
    )

    boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
    recovered_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

    max_diff = torch.max(torch.abs(boxes_xyxy - recovered_xyxy)).item()
    assert max_diff < 1e-6


def test_box_iou_pairwise() -> None:
    """Test pairwise IoU computation."""
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    b2 = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # Identical -> IoU = 1.0
            [10.0, 10.0, 20.0, 20.0], # Non-overlapping -> IoU = 0.0
            [0.0, 0.0, 5.0, 10.0],   # Half overlap -> IoU = 0.5
        ],
        dtype=torch.float32,
    )

    ious = box_iou(b1, b2)
    assert ious.shape == (1, 3)
    assert torch.isclose(ious[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(ious[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(ious[0, 2], torch.tensor(0.5), atol=1e-5)


def test_ciou_loss_matches_torchvision() -> None:
    """CRITICAL: Cross-validate custom CIoU loss against torchvision complete_box_iou_loss."""
    b1 = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [25.0, 30.0, 120.0, 200.0],
            [100.0, 100.0, 300.0, 400.0],
            [40.0, 60.0, 80.0, 120.0],
        ],
        dtype=torch.float32,
    )
    b2 = torch.tensor(
        [
            [15.0, 15.0, 55.0, 55.0],
            [30.0, 25.0, 115.0, 210.0],
            [90.0, 110.0, 310.0, 390.0],
            [40.0, 60.0, 80.0, 120.0],  # Exactly identical -> loss should be 0.0
        ],
        dtype=torch.float32,
    )

    custom_loss = ciou_loss(b1, b2)
    torchvision_loss = complete_box_iou_loss(b1, b2, reduction="none")

    # Identical box should have 0.0 loss
    assert torch.isclose(custom_loss[3], torch.tensor(0.0), atol=1e-5)

    # Maximum difference across pairs must be negligible (< 1e-6)
    max_diff = torch.max(torch.abs(custom_loss - torchvision_loss)).item()
    assert max_diff < 1e-6, f"CIoU divergence from torchvision: {max_diff:.8e}"
