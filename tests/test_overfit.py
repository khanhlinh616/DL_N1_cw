"""Overfitting test on 10 images to verify pipeline convergence (Milestone M3).

Requirements from RULES.md §17.6 and DEFINE_AND_DESIGN.md:
- Overfit on 10 images: loss decreases smoothly towards near 0.
- Recall reaches ≈ 1.0 (all ground truth boxes are detected with IoU >= 0.5 and score > 0.5).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.core.assigner import assign_targets
from src.core.loss import MultiTaskLoss
from src.models.detector import build_detector
from src.utils.box_ops import box_iou


@pytest.mark.slow
def test_overfit_10_images() -> None:
    """Overfit detector on 10 images and verify smooth loss decay and recall ≈ 1.0."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    num_images = 10
    detector = build_detector(num_classes=80, pretrained_backbone=False)
    detector.train()
    detector.freeze_backbone()  # Train Head and FPN neck for efficient convergence

    # Create 10 synthetic images with distinct boxes across P3, P4, P5
    images = torch.randn((num_images, 3, 512, 512), device=device)

    gt_boxes = []
    gt_labels = []

    for i in range(num_images):
        if i % 3 == 0:
            # Small box -> P3 (scale ~ 30)
            box = torch.tensor([[100.0 + i * 10, 100.0 + i * 10, 130.0 + i * 10, 130.0 + i * 10]], device=device)
            label = torch.tensor([i % 80], device=device)
        elif i % 3 == 1:
            # Medium box -> P4 (scale ~ 90)
            box = torch.tensor([[150.0 + i * 5, 150.0 + i * 5, 240.0 + i * 5, 240.0 + i * 5]], device=device)
            label = torch.tensor([(i * 3) % 80], device=device)
        else:
            # Large box -> P5 (scale ~ 200)
            box = torch.tensor([[80.0, 80.0, 280.0, 280.0]], device=device)
            label = torch.tensor([(i * 7) % 80], device=device)

        gt_boxes.append(box)
        gt_labels.append(label)

    targets = assign_targets(gt_boxes, gt_labels)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, detector.parameters()),
        lr=0.006,
        weight_decay=1e-4,
    )
    criterion = MultiTaskLoss()

    initial_loss = None
    final_loss = None

    # Train for 40 iterations
    for step in range(40):
        optimizer.zero_grad()
        predictions = detector(images)
        loss_total, loss_dict = criterion(predictions, targets)
        loss_total.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss_total.item()
        final_loss = loss_total.item()

    # 1. Verify loss drop: must drop by at least 95%
    print(f"Overfit 10 images - Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.05, (
        f"Loss did not drop sufficiently: initial {initial_loss:.4f}, final {final_loss:.4f}"
    )

    # 2. Verify recall on ground truth boxes
    detector.eval()
    with torch.no_grad():
        eval_preds = detector(images)

    recalled_count = 0
    total_gt = num_images

    strides = {"P3": 8, "P4": 16, "P5": 32}

    for b in range(num_images):
        gt_box = gt_boxes[b][0]  # [4]
        gt_cls = gt_labels[b][0].item()

        found_match = False
        for lvl_name, stride in strides.items():
            pos_mask = targets[lvl_name]["pos_mask"][b]  # [H, W]
            if not pos_mask.any():
                continue

            # Decode predictions at this level
            box_pred = eval_preds[lvl_name]["box"][b : b + 1]  # [1, 4, H, W]
            obj_pred = eval_preds[lvl_name]["obj"][b, 0]  # [H, W]
            cls_pred = eval_preds[lvl_name]["cls"][b]  # [80, H, W]

            decoded = criterion._decode_box_predictions(box_pred, stride=stride)[0]  # [4, H, W]
            decoded_perm = decoded.permute(1, 2, 0)  # [H, W, 4]

            # Check positive cells
            pos_indices = pos_mask.nonzero(as_tuple=False)
            for j, i in pos_indices:
                pred_b = decoded_perm[j, i : i + 1]  # [1, 4]
                iou = box_iou(pred_b, gt_box.unsqueeze(0))[0, 0].item()
                obj_score = torch.sigmoid(obj_pred[j, i]).item()
                pred_class = torch.argmax(cls_pred[:, j, i]).item()

                if iou >= 0.50 and obj_score >= 0.50 and pred_class == gt_cls:
                    found_match = True
                    break
            if found_match:
                break

        if found_match:
            recalled_count += 1

    recall = recalled_count / float(total_gt)
    print(f"Overfit Recall: {recall:.2f} ({recalled_count}/{total_gt} boxes detected)")
    assert recall >= 0.90, f"Expected recall >= 0.90, got {recall:.2f}"
