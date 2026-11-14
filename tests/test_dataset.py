"""Unit tests for COCODataset and DataLoader integration (Milestone M1).

Verifies:
- COCODataset item structure matches Section 1.4 specification.
- Full required metadata is present in target dict (image_id, orig_size, scale_ratio, pad_offsets).
- detection_collate_fn creates batch with shape [B, 3, 512, 512].
- Real data test with COCO val2017 confirms end-to-end functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import COCODataset, detection_collate_fn


def create_synthetic_coco_env(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a temporary synthetic dataset on disk with 3 dummy images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    sizes = [(640, 480), (800, 600), (500, 700)]
    for i, (w, h) in enumerate(sizes):
        img_id = 100 + i
        file_name = f"dummy_{img_id}.jpg"
        img = Image.new("RGB", (w, h), color=((i * 40) % 255, (i * 70) % 255, 120))
        img.save(img_dir / file_name)

        images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        # Add 2 boxes per image: [x, y, w, h]
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # person -> class_idx 0
                "bbox": [50.0, 60.0, 100.0, 120.0],
                "area": 12000.0,
                "iscrowd": 0,
            }
        )
        ann_id += 1
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 3,  # car -> class_idx 2
                "bbox": [200.0, 150.0, 150.0, 80.0],
                "area": 12000.0,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
    ]

    coco_dict = {
        "info": {"description": "Synthetic Test COCO"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    ann_file = tmp_path / "annotations.json"
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)

    return img_dir, ann_file


def test_coco_dataset_synthetic(tmp_path: Path) -> None:
    """Test COCODataset with synthetic images and annotations."""
    img_dir, ann_file = create_synthetic_coco_env(tmp_path)
    dataset = COCODataset(root=img_dir, annotation_file=ann_file, img_size=512, is_train=False)

    assert len(dataset) == 3

    img_tensor, target = dataset[0]

    # 1. Image tensor shape
    assert img_tensor.shape == (3, 512, 512)
    assert img_tensor.dtype == torch.float32

    # 2. Target boxes & labels
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].shape == (2,)
    assert target["labels"][0].item() == 0  # person -> 0
    assert target["labels"][1].item() == 2  # car -> 2

    # 3. Metadata fields
    assert "image_id" in target
    assert target["image_id"] == 100
    assert "orig_size" in target
    assert target["orig_size"] == (480, 640)
    assert "scale_ratio" in target
    assert isinstance(target["scale_ratio"], float)
    assert "pad_offsets" in target
    assert len(target["pad_offsets"]) == 2


def test_dataloader_batch_shape(tmp_path: Path) -> None:
    """Test DataLoader and detection_collate_fn batch shape [B, 3, 512, 512]."""
    img_dir, ann_file = create_synthetic_coco_env(tmp_path)
    dataset = COCODataset(root=img_dir, annotation_file=ann_file, img_size=512, is_train=False)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )

    batch_images, batch_targets = next(iter(loader))

    assert batch_images.shape == (2, 3, 512, 512)
    assert len(batch_targets) == 2
    for target in batch_targets:
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        assert "orig_size" in target
        assert "scale_ratio" in target
        assert "pad_offsets" in target


def test_coco_dataset_real_val2017() -> None:
    """Smoke test on real COCO val2017 images if available on disk."""
    val_root = Path("data/coco/val2017")
    val_ann = Path("data/coco/annotations/instances_val2017.json")

    if not (val_root.exists() and val_ann.exists()):
        pytest.skip("COCO val2017 not found on disk, skipping real data test.")

    dataset = COCODataset(root=val_root, annotation_file=val_ann, img_size=512, is_train=False)
    assert len(dataset) == 5000

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=detection_collate_fn)
    batch_images, batch_targets = next(iter(loader))

    assert batch_images.shape == (4, 3, 512, 512)
    assert len(batch_targets) == 4
    for target in batch_targets:
        assert target["boxes"].shape[1] == 4 if target["boxes"].numel() > 0 else True
        assert (target["labels"] >= 0).all()
        assert (target["labels"] < 80).all()
        assert target["scale_ratio"] > 0.0
