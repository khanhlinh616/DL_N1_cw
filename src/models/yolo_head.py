"""Decoupled YOLO Detection Head (YOLOX/YOLOv8 style).

Features:
- Decoupled branches for Bounding Box regression, Objectness, and Class classification.
- Shared weights across multi-scale FPN levels (P3, P4, P5).
- Focal loss bias initialization for objectness branch:
  bias = -log((1 - pi) / pi) ≈ -4.595 (with pi = 0.01).
- Outputs raw logits for numerical stability in loss and decoder.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + Per-Level BatchNorm2d + SiLU activation.

    Shares convolution weights across multi-scale FPN levels, with level-specific
    BatchNorm layers to avoid cross-scale running statistics drift.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        levels: Tuple[str, ...] = ("P3", "P4", "P5"),
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bns = nn.ModuleDict({lvl: nn.BatchNorm2d(out_channels) for lvl in levels})
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, level: str = "P3") -> torch.Tensor:
        bn = self.bns[level] if level in self.bns else next(iter(self.bns.values()))
        return self.act(bn(self.conv(x)))


class DecoupledYOLOHead(nn.Module):
    """Shared Decoupled Detection Head with 3 distinct prediction branches."""

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_classes: int = 80,
        focal_bias_pi: float = 0.01,
    ) -> None:
        """Initialize Decoupled YOLO Head.

        Args:
            in_channels: Number of input channels from FPN (default: 256).
            feat_channels: Number of channels in intermediate conv layers (default: 256).
            num_classes: Number of target object categories (default: 80).
            focal_bias_pi: Prior probability for foreground objectness (default: 0.01).
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.focal_bias_pi = focal_bias_pi

        # 1. Box Regression Branch (2 x Conv 3x3 -> 1x1 Conv -> 4 outputs: tx, ty, tw, th)
        self.box_conv1 = ConvBlock(in_channels, feat_channels, kernel_size=3)
        self.box_conv2 = ConvBlock(feat_channels, feat_channels, kernel_size=3)
        self.box_pred = nn.Conv2d(feat_channels, 4, kernel_size=1, stride=1, padding=0)

        # 2. Objectness Branch (2 x Conv 3x3 -> 1x1 Conv -> 1 logit: tobj)
        self.obj_conv1 = ConvBlock(in_channels, feat_channels, kernel_size=3)
        self.obj_conv2 = ConvBlock(feat_channels, feat_channels, kernel_size=3)
        self.obj_pred = nn.Conv2d(feat_channels, 1, kernel_size=1, stride=1, padding=0)

        # 3. Class Classification Branch (2 x Conv 3x3 -> 1x1 Conv -> 80 logits: tcls)
        self.cls_conv1 = ConvBlock(in_channels, feat_channels, kernel_size=3)
        self.cls_conv2 = ConvBlock(feat_channels, feat_channels, kernel_size=3)
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal and set Focal Loss bias on objectness."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Standard detector initialization: final projection convs use small gaussian weights
        for pred_layer in [self.box_pred, self.obj_pred, self.cls_pred]:
            nn.init.normal_(pred_layer.weight, std=0.01)
            if pred_layer.bias is not None:
                nn.init.constant_(pred_layer.bias, 0.0)

        # CRITICAL: Focal bias initialization on final objectness layer
        # b = -log((1 - pi) / pi) ≈ -4.595 for pi = 0.01
        initial_bias = -math.log((1.0 - self.focal_bias_pi) / self.focal_bias_pi)
        nn.init.constant_(self.obj_pred.bias, initial_bias)

    def forward_single_level(
        self, x: torch.Tensor, level: str = "P3"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on a single FPN level feature map.

        Args:
            x: Feature tensor of shape [B, 256, H, W].
            level: FPN level identifier ('P3', 'P4', or 'P5').

        Returns:
            Tuple of:
                - box_preds: [B, 4, H, W] (tx, ty, tw, th)
                - obj_preds: [B, 1, H, W] (raw logits)
                - cls_preds: [B, num_classes, H, W] (raw logits)
        """
        # Box branch
        feat_box = self.box_conv2(self.box_conv1(x, level), level)
        box_preds = self.box_pred(feat_box)

        # Objectness branch
        feat_obj = self.obj_conv2(self.obj_conv1(x, level), level)
        obj_preds = self.obj_pred(feat_obj)

        # Classification branch
        feat_cls = self.cls_conv2(self.cls_conv1(x, level), level)
        cls_preds = self.cls_pred(feat_cls)

        return box_preds, obj_preds, cls_preds

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass across all FPN levels with shared weights.

        Args:
            features: Dict containing feature maps for levels, e.g.:
                {'P3': [B, 256, 64, 64], 'P4': [B, 256, 32, 32], 'P5': [B, 256, 16, 16]}

        Returns:
            Dict containing predictions per level:
            {
                'P3': {'box': [B, 4, 64, 64], 'obj': [B, 1, 64, 64], 'cls': [B, 80, 64, 64]},
                'P4': {'box': [B, 4, 32, 32], 'obj': [B, 1, 32, 32], 'cls': [B, 80, 32, 32]},
                'P5': {'box': [B, 4, 16, 16], 'obj': [B, 1, 16, 16], 'cls': [B, 80, 16, 16]},
            }
        """
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for lvl_name, feat in features.items():
            box, obj, cls = self.forward_single_level(feat, level=lvl_name)
            outputs[lvl_name] = {
                "box": box,
                "obj": obj,
                "cls": cls,
            }
        return outputs
