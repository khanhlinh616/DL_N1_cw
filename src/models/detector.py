"""Complete One-Stage YOLO-style Object Detector.

Combines:
- ResNet-50 backbone with FrozenBatchNorm2d
- Feature Pyramid Network (FPN) neck
- Shared Decoupled YOLO Head across P3, P4, P5
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.backbone_fpn import ResNet50FPN
from src.models.yolo_head import DecoupledYOLOHead


class YOLODetector(nn.Module):
    """Full One-Stage Object Detector with ResNet-50-FPN and Decoupled Head."""

    def __init__(
        self,
        backbone_fpn: ResNet50FPN,
        head: DecoupledYOLOHead,
    ) -> None:
        super().__init__()
        self.backbone_fpn = backbone_fpn
        self.head = head

    def freeze_backbone(self) -> None:
        """Freeze all ResNet-50 backbone layers (used in initial training epochs)."""
        self.backbone_fpn.freeze_backbone()

    def unfreeze_backbone(self, trainable_layers: int = 3) -> None:
        """Unfreeze top backbone stages (layers 2-4) for end-to-end fine-tuning."""
        self.backbone_fpn.unfreeze_backbone(trainable_layers=trainable_layers)

    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass of the full detector.

        Args:
            images: Batch of images of shape [B, 3, 512, 512].

        Returns:
            Dict containing raw prediction logits for each FPN level:
            {
                'P3': {'box': [B, 4, 64, 64], 'obj': [B, 1, 64, 64], 'cls': [B, 80, 64, 64]},
                'P4': {'box': [B, 4, 32, 32], 'obj': [B, 1, 32, 32], 'cls': [B, 80, 32, 32]},
                'P5': {'box': [B, 4, 16, 16], 'obj': [B, 1, 16, 16], 'cls': [B, 80, 16, 16]},
            }
        """
        features = self.backbone_fpn(images)
        predictions = self.head(features)
        return predictions


def build_detector(
    num_classes: int = 80,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    in_channels: int = 256,
    feat_channels: int = 256,
    focal_bias_pi: float = 0.01,
) -> YOLODetector:
    """Factory function to build the complete detector according to specification.

    Args:
        num_classes: Number of object categories (default: 80).
        pretrained_backbone: Whether to load ImageNet-1K pretrained weights into ResNet-50.
        trainable_backbone_layers: Number of stages to unfreeze from top (0-3).
        in_channels: FPN out channels (default: 256).
        feat_channels: Channels inside Head conv layers (default: 256).
        focal_bias_pi: Foreground objectness prior probability (default: 0.01).

    Returns:
        Instantiated YOLODetector.
    """
    backbone_fpn = ResNet50FPN(
        pretrained=pretrained_backbone,
        out_channels=in_channels,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    head = DecoupledYOLOHead(
        in_channels=in_channels,
        feat_channels=feat_channels,
        num_classes=num_classes,
        focal_bias_pi=focal_bias_pi,
    )
    return YOLODetector(backbone_fpn=backbone_fpn, head=head)
