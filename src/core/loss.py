"""Multi-Task Loss Function for YOLO-style object detector.

Combines:
1. CIoU Loss on bounding boxes (evaluated strictly on positive cells).
2. Focal Loss on objectness (evaluated across ALL grid cells at all levels).
3. BCE Loss on multi-class classification (evaluated strictly on positive cells).

Formula:
    L_total = lambda_box * L_CIoU + lambda_obj * L_Focal_obj + lambda_cls * L_BCE_cls
Default weights:
    lambda_box = 5.0, lambda_obj = 1.0, lambda_cls = 1.0
Focal Loss parameters:
    gamma = 2.0, alpha = 0.25
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.box_ops import ciou_loss


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Numerically stable sigmoid focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw unnormalized prediction logits tensor of arbitrary shape.
        targets: Binary targets of same shape with values in {0.0, 1.0}.
        alpha: Weighting factor for foreground class (default: 0.25).
        gamma: Focusing parameter for modulating hard examples (default: 2.0).
        reduction: 'none' | 'mean' | 'sum'.

    Returns:
        Loss tensor matching input shape or reduced scalar.
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = torch.pow((1.0 - p_t).clamp(min=0.0), gamma)
    loss = alpha_t * modulating_factor * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss module combining CIoU, Focal Objectness, and BCE Classification."""

    def __init__(
        self,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_classes: int = 80,
        strides: List[int] = [8, 16, 32],
    ) -> None:
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_classes = num_classes
        self.strides = strides
        self.level_names = ["P3", "P4", "P5"]

    def _decode_box_predictions(
        self,
        box_preds: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """Decode raw box logits into [x1, y1, x2, y2] coordinates on 512x512 space.

        Formula (Section 2.4.3 & 2.7):
            cx = (grid_x + 2*sigmoid(tx) - 0.5) * s
            cy = (grid_y + 2*sigmoid(ty) - 0.5) * s
            w = (2*sigmoid(tw))^2 * 4 * s
            h = (2*sigmoid(th))^2 * 4 * s

        Args:
            box_preds: Tensor [B, 4, H, W].
            stride: FPN level stride s.

        Returns:
            Decoded boxes [B, 4, H, W] in [x1, y1, x2, y2].
        """
        B, _, H, W = box_preds.shape
        device = box_preds.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        tx = box_preds[:, 0]
        ty = box_preds[:, 1]
        tw = box_preds[:, 2]
        th = box_preds[:, 3]

        cx = (grid_x + 2.0 * torch.sigmoid(tx) - 0.5) * stride
        cy = (grid_y + 2.0 * torch.sigmoid(ty) - 0.5) * stride

        w = torch.pow(2.0 * torch.sigmoid(tw), 2) * 4.0 * stride
        h = torch.pow(2.0 * torch.sigmoid(th), 2) * 4.0 * stride

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss across all FPN levels.

        Args:
            predictions: Output from YOLODetector:
                {lvl: {'box': [B,4,H,W], 'obj': [B,1,H,W], 'cls': [B,80,H,W]}}
            targets: Output from assign_targets:
                {lvl: {'pos_mask': [B,H,W], 'target_boxes': [B,4,H,W],
                       'target_obj': [B,1,H,W], 'target_cls': [B,H,W]}}

        Returns:
            Tuple of:
                - loss_total: Scalar loss tensor with attached gradient computation graph.
                - loss_dict: Dict with detached float scalar values for logging.
        """
        device = next(iter(predictions.values()))["obj"].device

        total_num_pos = 0
        total_ciou_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        for lvl_idx, lvl_name in enumerate(self.level_names):
            if lvl_name not in predictions or lvl_name not in targets:
                continue

            stride = self.strides[lvl_idx]
            pred_dict = predictions[lvl_name]
            target_dict = targets[lvl_name]

            pos_mask = target_dict["pos_mask"]  # [B, H, W]
            target_boxes = target_dict["target_boxes"]  # [B, 4, H, W]
            target_obj = target_dict["target_obj"]  # [B, 1, H, W]
            target_cls = target_dict["target_cls"]  # [B, H, W]

            pred_box = pred_dict["box"]  # [B, 4, H, W]
            pred_obj = pred_dict["obj"]  # [B, 1, H, W]
            pred_cls = pred_dict["cls"]  # [B, 80, H, W]

            # 1. Objectness Loss (Focal Loss on ALL cells)
            obj_loss_lvl = sigmoid_focal_loss(
                pred_obj,
                target_obj,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )
            total_obj_loss = total_obj_loss + obj_loss_lvl

            num_pos_lvl = pos_mask.sum().item()
            total_num_pos += num_pos_lvl

            if num_pos_lvl > 0:
                # 2. Box Regression Loss (CIoU strictly on positive cells)
                decoded_boxes = self._decode_box_predictions(pred_box, stride=stride)
                # Permute to [B, H, W, 4] for boolean masking
                decoded_boxes_perm = decoded_boxes.permute(0, 2, 3, 1)
                target_boxes_perm = target_boxes.permute(0, 2, 3, 1)

                pos_pred_boxes = decoded_boxes_perm[pos_mask]  # [N_pos_lvl, 4]
                pos_target_boxes = target_boxes_perm[pos_mask]  # [N_pos_lvl, 4]

                lvl_ciou = ciou_loss(pos_pred_boxes, pos_target_boxes)
                total_ciou_loss = total_ciou_loss + lvl_ciou.sum()

                # 3. Class Classification Loss (BCE strictly on positive cells)
                pred_cls_perm = pred_cls.permute(0, 2, 3, 1)  # [B, H, W, 80]
                pos_pred_cls = pred_cls_perm[pos_mask]  # [N_pos_lvl, 80]
                pos_target_cls = target_cls[pos_mask]  # [N_pos_lvl]

                target_one_hot = F.one_hot(pos_target_cls, num_classes=self.num_classes).float()
                lvl_cls_loss = F.binary_cross_entropy_with_logits(
                    pos_pred_cls, target_one_hot, reduction="sum"
                )
                total_cls_loss = total_cls_loss + lvl_cls_loss

        # Normalization constant: standard RetinaNet / YOLOX normalizes by total positive anchors
        norm_factor = max(1.0, float(total_num_pos))

        loss_box = total_ciou_loss / norm_factor
        loss_obj = total_obj_loss / norm_factor
        loss_cls = total_cls_loss / norm_factor

        loss_total = (
            self.lambda_box * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_cls * loss_cls
        )

        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_box": loss_box.item(),
            "loss_obj": loss_obj.item(),
            "loss_cls": loss_cls.item(),
            "num_pos": total_num_pos,
        }

        return loss_total, loss_dict


def compute_loss(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, Dict[str, torch.Tensor]],
    lambda_box: float = 5.0,
    lambda_obj: float = 1.0,
    lambda_cls: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Functional interface matching Section 2.5.1 specification."""
    loss_module = MultiTaskLoss(
        lambda_box=lambda_box,
        lambda_obj=lambda_obj,
        lambda_cls=lambda_cls,
    )
    return loss_module(predictions, targets)
