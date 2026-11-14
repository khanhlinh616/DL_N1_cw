"""Unit tests for neural network architecture (Milestone M2).

Verifies:
- Feature map shapes at P3 (stride 8), P4 (stride 16), P5 (stride 32).
- Output shapes for Box [B, 4, H, W], Obj [B, 1, H, W], Cls [B, 80, H, W] branches.
- Focal loss bias initialization: obj_pred.bias ≈ -4.595 (pi = 0.01).
- Shared weights property of Decoupled Head.
- End-to-end detector forward pass.
- Backbone freezing (epoch 1-3) and unfreezing (epoch 4+) parameter gradients.
"""

from __future__ import annotations

import math
import pytest
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from src.models.backbone_fpn import ResNet50FPN
from src.models.detector import YOLODetector, build_detector
from src.models.yolo_head import DecoupledYOLOHead


def test_backbone_fpn_shapes() -> None:
    """Test feature map output shapes from ResNet-50 + FPN."""
    model = ResNet50FPN(pretrained=False, out_channels=256)
    model.eval()

    dummy_input = torch.zeros((2, 3, 512, 512), dtype=torch.float32)
    with torch.no_grad():
        features = model(dummy_input)

    assert "P3" in features
    assert "P4" in features
    assert "P5" in features

    # Stride 8: 512 / 8 = 64
    assert features["P3"].shape == (2, 256, 64, 64)
    # Stride 16: 512 / 16 = 32
    assert features["P4"].shape == (2, 256, 32, 32)
    # Stride 32: 512 / 32 = 16
    assert features["P5"].shape == (2, 256, 16, 16)


def test_decoupled_head_shapes_and_focal_bias() -> None:
    """Test Decoupled Head outputs and objectness bias initialization."""
    head = DecoupledYOLOHead(
        in_channels=256,
        feat_channels=256,
        num_classes=80,
        focal_bias_pi=0.01,
    )
    head.eval()

    # 1. Verify Focal bias initialization
    expected_bias = -math.log((1.0 - 0.01) / 0.01)  # ≈ -4.59512
    actual_bias = head.obj_pred.bias.item()
    assert math.isclose(actual_bias, expected_bias, rel_tol=1e-3), (
        f"Expected obj bias {expected_bias:.4f}, got {actual_bias:.4f}"
    )

    # 2. Verify forward output shapes across levels
    dummy_features = {
        "P3": torch.zeros((2, 256, 64, 64), dtype=torch.float32),
        "P4": torch.zeros((2, 256, 32, 32), dtype=torch.float32),
        "P5": torch.zeros((2, 256, 16, 16), dtype=torch.float32),
    }

    with torch.no_grad():
        preds = head(dummy_features)

    for lvl, (h, w) in [("P3", (64, 64)), ("P4", (32, 32)), ("P5", (16, 16))]:
        assert lvl in preds
        assert preds[lvl]["box"].shape == (2, 4, h, w)
        assert preds[lvl]["obj"].shape == (2, 1, h, w)
        assert preds[lvl]["cls"].shape == (2, 80, h, w)


def test_full_detector_forward_pass() -> None:
    """Test full YOLODetector forward pass end-to-end."""
    detector = build_detector(num_classes=80, pretrained_backbone=False)
    detector.eval()

    dummy_images = torch.randn((2, 3, 512, 512), dtype=torch.float32)
    with torch.no_grad():
        predictions = detector(dummy_images)

    assert set(predictions.keys()) == {"P3", "P4", "P5"}
    assert predictions["P3"]["box"].shape == (2, 4, 64, 64)
    assert predictions["P3"]["obj"].shape == (2, 1, 64, 64)
    assert predictions["P3"]["cls"].shape == (2, 80, 64, 64)


def test_backbone_freezing_and_unfreezing() -> None:
    """Test freezing ResNet-50 backbone for initial epochs and unfreezing for fine-tuning."""
    detector = build_detector(num_classes=80, pretrained_backbone=False)

    # 1. Freeze entire backbone (epochs 1-3)
    detector.freeze_backbone()

    # Backbone parameters should NOT require grad
    for name, param in detector.backbone_fpn.named_parameters():
        if "fpn" not in name:
            assert param.requires_grad is False, f"Backbone param {name} should be frozen!"
        else:
            assert param.requires_grad is True, f"FPN neck param {name} must remain trainable!"

    # Head parameters must remain trainable
    for name, param in detector.head.named_parameters():
        assert param.requires_grad is True, f"Head param {name} must remain trainable!"

    # 2. Unfreeze Layers 2-4 (epochs 4+)
    detector.unfreeze_backbone(trainable_layers=3)

    # Layer 1 must remain frozen
    for param in detector.backbone_fpn.layer1.parameters():
        assert param.requires_grad is False

    # Layer 2, 3, 4 conv weights must be trainable
    assert detector.backbone_fpn.layer2[0].conv1.weight.requires_grad is True
    assert detector.backbone_fpn.layer3[0].conv1.weight.requires_grad is True
    assert detector.backbone_fpn.layer4[0].conv1.weight.requires_grad is True

    # FrozenBatchNorm2d must ALWAYS stay requires_grad = False
    for module in detector.modules():
        if isinstance(module, FrozenBatchNorm2d):
            for param in module.parameters():
                assert param.requires_grad is False
