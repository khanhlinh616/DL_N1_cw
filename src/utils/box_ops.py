"""Bounding box operations and geometry utilities.

Includes:
- Conversions between xyxy and cxcywh formats.
- Standard Intersection-over-Union (IoU).
- Complete IoU (CIoU) and CIoU Loss with aspect ratio and center distance penalty.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] boxes to [x1, y1, x2, y2] format.

    Args:
        boxes: Tensor of shape [..., 4] in [cx, cy, w, h].

    Returns:
        Tensor of shape [..., 4] in [x1, y1, x2, y2].
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] boxes to [cx, cy, w, h] format.

    Args:
        boxes: Tensor of shape [..., 4] in [x1, y1, x2, y2].

    Returns:
        Tensor of shape [..., 4] in [cx, cy, w, h].
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute pairwise Intersection-over-Union (IoU) between two sets of boxes.

    Args:
        boxes1: Tensor [N, 4] in [x1, y1, x2, y2].
        boxes2: Tensor [M, 4] in [x1, y1, x2, y2].
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor [N, M] with IoU values in [0, 1].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + eps)


def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Complete IoU (CIoU) Loss between matching pairs of boxes.

    Matches torchvision.ops.complete_box_iou_loss formulation:
        L_CIoU = 1 - IoU + (rho^2 / c^2) + alpha * v

    Args:
        pred_boxes: Predicted boxes [N, 4] in [x1, y1, x2, y2].
        target_boxes: Ground truth boxes [N, 4] in [x1, y1, x2, y2].
        eps: Numerical stability constant.

    Returns:
        Tensor [N] containing element-wise CIoU loss.
    """
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return torch.empty((0,), dtype=pred_boxes.dtype, device=pred_boxes.device)

    x1, y1, x2, y2 = pred_boxes.unbind(-1)
    x1_g, y1_g, x2_g, y2_g = target_boxes.unbind(-1)

    w1 = (x2 - x1).clamp(min=eps)
    h1 = (y2 - y1).clamp(min=eps)
    w2 = (x2_g - x1_g).clamp(min=eps)
    h2 = (y2_g - y1_g).clamp(min=eps)

    # 1. Intersection over Union
    inter_x1 = torch.max(x1, x1_g)
    inter_y1 = torch.max(y1, y1_g)
    inter_x2 = torch.min(x2, x2_g)
    inter_y2 = torch.min(y2, y2_g)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / (union_area + eps)

    # 2. Distance between centers (rho^2)
    cx1, cy1 = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cx2, cy2 = (x1_g + x2_g) * 0.5, (y1_g + y2_g) * 0.5
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # 3. Smallest enclosing box diagonal (c^2)
    enc_x1 = torch.min(x1, x1_g)
    enc_y1 = torch.min(y1, y1_g)
    enc_x2 = torch.max(x2, x2_g)
    enc_y2 = torch.max(y2, y2_g)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # 4. Aspect ratio consistency (v and alpha)
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    loss = 1.0 - ciou
    return loss
