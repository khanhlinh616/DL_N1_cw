"""ResNet-50 backbone with FrozenBatchNorm and Feature Pyramid Network (FPN).

Extracts multi-scale feature maps:
- P3: stride 8,  256 channels, shape [B, 256, 64, 64] for 512x512 input
- P4: stride 16, 256 channels, shape [B, 256, 32, 32] for 512x512 input
- P5: stride 32, 256 channels, shape [B, 256, 16, 16] for 512x512 input
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet50FPN(nn.Module):
    """ResNet-50 Backbone with FrozenBatchNorm2d and FPN."""

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
        trainable_backbone_layers: int = 3,
    ) -> None:
        """Initialize ResNet-50 backbone with FPN.

        Args:
            pretrained: Whether to load ImageNet-1K pretrained weights.
            out_channels: Number of channels in FPN feature maps (default: 256).
            trainable_backbone_layers: Number of trainable stages from top (0 to 3).
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        # Use FrozenBatchNorm2d to preserve ImageNet statistics when batch size is small
        backbone = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)

        # Stage 0: stem
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Residual layers
        self.layer1 = backbone.layer1  # stride 4, 256 ch (not used in FPN)
        self.layer2 = backbone.layer2  # stride 8, 512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch
        self.layer4 = backbone.layer4  # stride 32, 2048 ch

        # FPN neck
        in_channels_list = [512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )

        self.out_channels = out_channels
        self.strides = [8, 16, 32]
        self.fpn_keys = ["P3", "P4", "P5"]

        # Set initial freezing according to trainable_backbone_layers
        self.set_trainable_layers(trainable_backbone_layers)

    def set_trainable_layers(self, trainable_layers: int) -> None:
        """Configure which backbone stages are trainable.

        Args:
            trainable_layers: Number of stages to unfreeze from the top (0 to 3).
                - 0: All backbone layers frozen (e.g. first 3 epochs).
                - 1: Only layer4 is trainable.
                - 2: layer3 and layer4 are trainable.
                - 3: layer2, layer3, and layer4 are trainable (standard fine-tuning).
        """
        assert 0 <= trainable_layers <= 3, f"trainable_layers must be in [0, 3], got {trainable_layers}"

        # 1. Always freeze stem and layer1
        layers_to_freeze = [self.conv1, self.bn1, self.layer1]
        for module in layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        # 2. Configure layers 2, 3, 4
        all_layers = [self.layer2, self.layer3, self.layer4]
        # Number of layers to freeze from bottom: 3 - trainable_layers
        num_frozen = 3 - trainable_layers

        for i, layer in enumerate(all_layers):
            should_freeze = i < num_frozen
            for param in layer.parameters():
                param.requires_grad = not should_freeze

        # 3. Always keep FrozenBatchNorm2d frozen without grads
        for module in self.modules():
            if isinstance(module, FrozenBatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False

        # 4. FPN neck is always trainable
        for param in self.fpn.parameters():
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Freeze all layers of the ResNet-50 backbone (used in first 3 epochs)."""
        self.set_trainable_layers(trainable_layers=0)

    def unfreeze_backbone(self, trainable_layers: int = 3) -> None:
        """Unfreeze upper stages of the backbone for end-to-end fine-tuning."""
        self.set_trainable_layers(trainable_layers=trainable_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass extracting multi-scale feature maps.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            Dict containing:
                - 'P3': Feature tensor [B, 256, H/8, W/8]
                - 'P4': Feature tensor [B, 256, H/16, W/16]
                - 'P5': Feature tensor [B, 256, H/32, W/32]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Backbone stages
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN neck
        features_dict = {"0": c3, "1": c4, "2": c5}
        fpn_outputs = self.fpn(features_dict)

        return {
            "P3": fpn_outputs["0"],
            "P4": fpn_outputs["1"],
            "P5": fpn_outputs["2"],
        }
