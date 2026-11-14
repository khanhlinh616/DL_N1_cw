"""COCO Dataset module for object detection.

Returns:
- image: FloatTensor [3, 512, 512]
- target: Dict {
    'boxes': FloatTensor [N, 4] (format xyxy on 512x512),
    'labels': LongTensor [N] (class_idx in [0, 79]),
    'image_id': int,
    'orig_size': Tuple[int, int] (H_orig, W_orig),
    'scale_ratio': float,
    'pad_offsets': Tuple[float, float] (pad_w, pad_h),
  }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from src.data.category_mapper import CategoryMapper, DEFAULT_MAPPER
from src.data.transforms import DetectionTransforms


class COCODataset(Dataset):
    """PyTorch Dataset loading COCO annotations with full metadata preservation."""

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        img_size: int = 512,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        filter_empty: bool = True,
        category_mapper: Optional[CategoryMapper] = None,
    ) -> None:
        """Initialize COCO dataset.

        Args:
            root: Path to directory containing image files (e.g. data/coco/train2017).
            annotation_file: Path to COCO instances JSON file.
            img_size: Target image size for letterbox transform (default: 512).
            transforms: Optional custom callable transforms. If None, default DetectionTransforms is used.
            is_train: Whether dataset is used for training (affects augmentations).
            filter_empty: If True, exclude images that have no annotations (standard for training).
            category_mapper: CategoryMapper instance. Defaults to standard 80-class mapper.
        """
        self.root = Path(root).resolve()
        self.annotation_file = Path(annotation_file).resolve()
        self.img_size = img_size
        self.is_train = is_train
        self.mapper = category_mapper or DEFAULT_MAPPER

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        self.coco = COCO(str(self.annotation_file))
        all_img_ids = sorted(self.coco.getImgIds())

        if filter_empty and is_train:
            valid_ids = []
            for img_id in all_img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
                if len(ann_ids) > 0:
                    valid_ids.append(img_id)
            self.image_ids = valid_ids
        else:
            self.image_ids = all_img_ids

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = DetectionTransforms(img_size=self.img_size, is_train=self.is_train)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load single image and target dict with metadata.

        Returns:
            image: FloatTensor [3, 512, 512]
            target: Dict containing:
                - boxes: FloatTensor [N, 4] in xyxy format on 512 space
                - labels: LongTensor [N] with class indices in [0, 79]
                - image_id: int
                - orig_size: Tuple[int, int] (H_orig, W_orig)
                - scale_ratio: float
                - pad_offsets: Tuple[float, float] (pad_w, pad_h)
        """
        image_id = self.image_ids[index]
        img_meta = self.coco.loadImgs([image_id])[0]
        file_name = img_meta["file_name"]
        img_path = self.root / file_name

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []

        for ann in anns:
            if self.is_train and ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0.0 or h <= 0.0:
                continue

            # Convert [x, y, w, h] -> [x1, y1, x2, y2]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(orig_w), float(x + w))
            y2 = min(float(orig_h), float(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                class_idx = self.mapper.to_class_idx(ann["category_id"])
            except KeyError:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(class_idx)

        if boxes_list:
            raw_boxes = torch.tensor(boxes_list, dtype=torch.float32)
            raw_labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            raw_boxes = torch.zeros((0, 4), dtype=torch.float32)
            raw_labels = torch.zeros((0,), dtype=torch.int64)

        # Apply transforms (letterbox, flip, color jitter, normalize)
        img_tensor, out_boxes, out_labels, metadata = self.transforms(
            image, raw_boxes, raw_labels
        )

        target: Dict[str, Any] = {
            "boxes": out_boxes,
            "labels": out_labels,
            "image_id": image_id,
            "orig_size": metadata["orig_size"],
            "scale_ratio": metadata["scale_ratio"],
            "pad_offsets": metadata["pad_offsets"],
        }

        return img_tensor, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collate function for DataLoader yielding batch tensor [B, 3, 512, 512].

    Args:
        batch: List of tuples (image_tensor, target_dict).

    Returns:
        images: FloatTensor [B, 3, 512, 512]
        targets: List of target dicts of length B.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
