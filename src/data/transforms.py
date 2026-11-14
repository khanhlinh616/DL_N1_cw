"""Data transformations and augmentations for YOLO-style detector.

Includes:
- Letterbox resize with padding to 512x512 while preserving aspect ratio.
- Exact forward and inverse coordinate transforms for bounding boxes.
- Random horizontal flip with coordinate mirroring.
- Color jitter (photometric distortion).
- Tensor conversion and ImageNet normalization.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms as T


def letterbox_image(
    image: Image.Image,
    target_size: int = 512,
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[float, float]]:
    """Resize image to target_size x target_size with padding, keeping aspect ratio.

    Args:
        image: Input PIL Image.
        target_size: Target square dimension (default: 512).
        fill_color: RGB fill value for letterbox padding borders (default: 114).

    Returns:
        Tuple of:
            - padded_image: PIL Image of size (target_size, target_size).
            - scale_ratio: float r = min(target_size/H, target_size/W).
            - pad_offsets: Tuple (pad_w, pad_h) in pixels on target space.
    """
    orig_w, orig_h = image.size
    r = min(target_size / orig_h, target_size / orig_w)
    new_w = min(target_size, max(1, int(round(orig_w * r))))
    new_h = min(target_size, max(1, int(round(orig_h * r))))

    # Resample image
    resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Compute padding
    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    # Create new canvas
    padded_image = Image.new("RGB", (target_size, target_size), fill_color)
    padded_image.paste(resized_image, (int(round(pad_w)), int(round(pad_h))))

    # Return actual float offsets applied for coordinate consistency
    actual_pad_w = float(int(round(pad_w)))
    actual_pad_h = float(int(round(pad_h)))

    return padded_image, r, (actual_pad_w, actual_pad_h)


def letterbox_boxes(
    boxes: torch.Tensor,
    scale_ratio: float,
    pad_offsets: Tuple[float, float],
) -> torch.Tensor:
    """Transform bounding boxes from original coordinates to letterboxed 512x512 space.

    Args:
        boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format.
        scale_ratio: Scaling factor r.
        pad_offsets: Tuple of (pad_w, pad_h).

    Returns:
        Transformed boxes tensor [N, 4] clipped to [0, 512].
    """
    if boxes.numel() == 0:
        return boxes.clone()

    pad_w, pad_h = pad_offsets
    transformed = boxes.clone()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale_ratio + pad_w
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale_ratio + pad_h
    return transformed


def inverse_letterbox_boxes(
    boxes: torch.Tensor,
    scale_ratio: float,
    pad_offsets: Tuple[float, float],
    orig_size: Tuple[int, int],
) -> torch.Tensor:
    """Restore bounding boxes from 512x512 letterboxed space back to original image coordinates.

    Args:
        boxes: Tensor of shape [K, 4] in [x1, y1, x2, y2] format on 512x512.
        scale_ratio: Scaling factor r used during letterboxing.
        pad_offsets: Tuple of (pad_w, pad_h).
        orig_size: Original image dimensions as (H_orig, W_orig).

    Returns:
        Tensor of shape [K, 4] in [x1, y1, x2, y2] format on original image coordinate space,
        clipped to [0, W_orig] and [0, H_orig].
    """
    if boxes.numel() == 0:
        return boxes.clone()

    orig_h, orig_w = orig_size
    pad_w, pad_h = pad_offsets

    restored = boxes.clone()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - pad_w) / scale_ratio
    restored[:, [1, 3]] = (restored[:, [1, 3]] - pad_h) / scale_ratio

    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(min=0.0, max=float(orig_w))
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(min=0.0, max=float(orig_h))
    return restored


def random_horizontal_flip(
    image: Image.Image,
    boxes: torch.Tensor,
    img_size: int = 512,
    p: float = 0.5,
) -> Tuple[Image.Image, torch.Tensor, bool]:
    """Apply random horizontal flip to image and boxes."""
    if random.random() < p:
        flipped_img = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if boxes.numel() > 0:
            flipped_boxes = boxes.clone()
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            flipped_boxes[:, 0] = img_size - x2
            flipped_boxes[:, 2] = img_size - x1
            return flipped_img, flipped_boxes, True
        return flipped_img, boxes, True
    return image, boxes, False


class DetectionTransforms:
    """End-to-end preprocessing pipeline for object detection."""

    def __init__(
        self,
        img_size: int = 512,
        is_train: bool = True,
        flip_prob: float = 0.5,
        color_jitter: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.img_size = img_size
        self.is_train = is_train
        self.flip_prob = flip_prob if is_train else 0.0
        self.color_jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            if (is_train and color_jitter)
            else None
        )
        self.normalize = T.Normalize(mean=list(mean), std=list(std))

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply transforms to image, boxes, and metadata.

        Args:
            image: PIL Image in RGB format.
            boxes: FloatTensor [N, 4] in [x1, y1, x2, y2] format on original image.
            labels: LongTensor [N] with class_idx.

        Returns:
            Tuple of:
                - img_tensor: FloatTensor [3, 512, 512] normalized.
                - boxes_out: FloatTensor [N, 4] on 512x512 space.
                - labels_out: LongTensor [N].
                - metadata: Dict with 'orig_size', 'scale_ratio', 'pad_offsets'.
        """
        orig_w, orig_h = image.size
        orig_size = (orig_h, orig_w)

        # 1. Letterbox resize
        lb_img, r, pad_offsets = letterbox_image(image, target_size=self.img_size)
        lb_boxes = letterbox_boxes(boxes, scale_ratio=r, pad_offsets=pad_offsets)

        # 2. Augmentations (train only)
        if self.is_train:
            # Color jitter
            if self.color_jitter is not None:
                lb_img = self.color_jitter(lb_img)
            # Random horizontal flip
            lb_img, lb_boxes, _ = random_horizontal_flip(
                lb_img, lb_boxes, img_size=self.img_size, p=self.flip_prob
            )

        # 3. Convert to tensor [3, H, W] in [0, 1] and normalize
        img_tensor = TF.to_tensor(lb_img)
        img_tensor = self.normalize(img_tensor)

        metadata = {
            "orig_size": orig_size,
            "scale_ratio": r,
            "pad_offsets": pad_offsets,
        }

        return img_tensor, lb_boxes, labels, metadata
