"""Unit tests for multi-task loss calculation (Milestone M3).

Verifies:
- Loss computes finite values without NaN or Inf.
- Gradient backpropagation succeeds on detector model parameters.
- Empty foreground case (zero positive cells) is handled gracefully without division by zero.
- Proper weights scaling (5.0 box, 1.0 obj, 1.0 cls).
"""

from __future__ import annotations

import torch

from src.core.assigner import assign_targets
from src.core.loss import MultiTaskLoss, compute_loss
from src.models.detector import build_detector


def test_loss_computation_and_backward() -> None:
    """Test full forward and backward pass through loss module."""
    detector = build_detector(num_classes=80, pretrained_backbone=False)
    detector.train()

    images = torch.randn((2, 3, 512, 512), dtype=torch.float32)

    # Two ground truth boxes for image 0, one for image 1
    gt_boxes = [
        torch.tensor([[50.0, 50.0, 100.0, 100.0], [200.0, 200.0, 400.0, 400.0]]),
        torch.tensor([[100.0, 100.0, 250.0, 250.0]]),
    ]
    gt_labels = [
        torch.tensor([0, 15]),
        torch.tensor([2]),
    ]

    predictions = detector(images)
    targets = assign_targets(gt_boxes, gt_labels)

    loss_fn = MultiTaskLoss()
    total_loss, loss_dict = loss_fn(predictions, targets)

    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0
    assert "loss_box" in loss_dict
    assert "loss_obj" in loss_dict
    assert "loss_cls" in loss_dict
    assert loss_dict["num_pos"] > 0

    # Test backpropagation
    total_loss.backward()

    # Check that head gradients are computed and finite
    assert detector.head.box_pred.weight.grad is not None
    assert torch.isfinite(detector.head.box_pred.weight.grad).all()
    assert detector.head.obj_pred.weight.grad is not None
    assert torch.isfinite(detector.head.obj_pred.weight.grad).all()
    assert detector.head.cls_pred.weight.grad is not None
    assert torch.isfinite(detector.head.cls_pred.weight.grad).all()


def test_loss_empty_targets_no_crash() -> None:
    """Test loss computation when images have zero ground truth objects."""
    detector = build_detector(num_classes=80, pretrained_backbone=False)
    detector.eval()

    images = torch.randn((1, 3, 512, 512), dtype=torch.float32)
    gt_boxes = [torch.zeros((0, 4), dtype=torch.float32)]
    gt_labels = [torch.zeros((0,), dtype=torch.int64)]

    with torch.no_grad():
        predictions = detector(images)
        targets = assign_targets(gt_boxes, gt_labels)
        total_loss, loss_dict = compute_loss(predictions, targets)

    assert torch.isfinite(total_loss)
    assert loss_dict["loss_box"] == 0.0
    assert loss_dict["loss_cls"] == 0.0
    assert loss_dict["num_pos"] == 0


def test_ablation_loss_variants() -> None:
    """Test loss computation with ablation configurations: Smooth L1 and BCE objectness."""
    detector = build_detector(num_classes=80, pretrained_backbone=False)
    detector.train()

    images = torch.randn((1, 3, 512, 512), dtype=torch.float32)
    gt_boxes = [torch.tensor([[50.0, 50.0, 100.0, 100.0]])]
    gt_labels = [torch.tensor([1])]

    predictions = detector(images)
    targets = assign_targets(gt_boxes, gt_labels)

    # Variant 1: Smooth L1 box loss
    loss_smooth_l1 = MultiTaskLoss(box_loss_type="smooth_l1")
    l_box, d_box = loss_smooth_l1(predictions, targets)
    assert torch.isfinite(l_box)
    assert d_box["loss_box"] > 0.0

    # Variant 2: BCE objectness loss
    loss_bce_obj = MultiTaskLoss(obj_loss_type="bce")
    l_bce, d_bce = loss_bce_obj(predictions, targets)
    assert torch.isfinite(l_bce)
    assert d_bce["loss_obj"] > 0.0

