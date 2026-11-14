"""Unit tests for dataset split logic (Milestone M0).

Verifies:
- Deterministic behavior with fixed random seed.
- Zero data leakage (zero overlap between train and val-heldout).
- Exact counts for images and annotations.
- Correct COCO format structure preservation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.download_data import split_coco_annotations, split_train_val_files


def create_dummy_coco_data(num_images: int = 100, num_categories: int = 80) -> Dict[str, Any]:
    """Helper to generate synthetic COCO annotation structure for testing."""
    categories = [
        {"id": i + 1, "name": f"cat_{i+1}", "supercategory": "thing"}
        for i in range(num_categories)
    ]
    images = [
        {
            "id": 1000 + i,
            "file_name": f"{1000 + i:012d}.jpg",
            "width": 640,
            "height": 480,
        }
        for i in range(num_images)
    ]
    annotations = []
    ann_id = 1
    for img in images:
        # Give each image 2 bounding boxes
        for b in range(2):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img["id"],
                    "category_id": (ann_id % num_categories) + 1,
                    "bbox": [50.0, 50.0, 100.0, 100.0],
                    "area": 10000.0,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {"description": "Test Synthetic COCO Dataset", "year": 2017},
        "licenses": [{"id": 1, "name": "Test License"}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def test_split_coco_annotations_counts_and_zero_overlap() -> None:
    """Test that split produces exact counts and strictly zero overlap."""
    data = create_dummy_coco_data(num_images=100)
    train_count = 80
    val_count = 15
    seed = 42

    train_dict, val_dict = split_coco_annotations(
        data, train_count=train_count, val_heldout_count=val_count, seed=seed
    )

    # 1. Exact image counts
    assert len(train_dict["images"]) == train_count
    assert len(val_dict["images"]) == val_count

    # 2. Zero overlap assertion
    train_ids = {img["id"] for img in train_dict["images"]}
    val_ids = {img["id"] for img in val_dict["images"]}
    assert len(train_ids & val_ids) == 0, "Train and Val subsets must not share any image IDs!"

    # 3. Annotation integrity
    for ann in train_dict["annotations"]:
        assert ann["image_id"] in train_ids
        assert ann["image_id"] not in val_ids

    for ann in val_dict["annotations"]:
        assert ann["image_id"] in val_ids
        assert ann["image_id"] not in train_ids

    # 4. Each image had 2 boxes
    assert len(train_dict["annotations"]) == train_count * 2
    assert len(val_dict["annotations"]) == val_count * 2

    # 5. Categories preserved
    assert len(train_dict["categories"]) == 80
    assert len(val_dict["categories"]) == 80


def test_split_determinism() -> None:
    """Test that same seed yields identical splits across multiple runs."""
    data = create_dummy_coco_data(num_images=50)

    train_1, val_1 = split_coco_annotations(data, train_count=35, val_heldout_count=10, seed=42)
    train_2, val_2 = split_coco_annotations(data, train_count=35, val_heldout_count=10, seed=42)

    ids_train_1 = [img["id"] for img in train_1["images"]]
    ids_train_2 = [img["id"] for img in train_2["images"]]
    assert ids_train_1 == ids_train_2

    ids_val_1 = [img["id"] for img in val_1["images"]]
    ids_val_2 = [img["id"] for img in val_2["images"]]
    assert ids_val_1 == ids_val_2


def test_split_insufficient_images_error() -> None:
    """Test that requesting more images than available raises ValueError."""
    data = create_dummy_coco_data(num_images=20)
    with pytest.raises(ValueError, match="less than requested"):
        split_coco_annotations(data, train_count=18, val_heldout_count=5, seed=42)


def test_split_train_val_files(tmp_path: Path) -> None:
    """Test file reading, splitting, and writing to disk."""
    data = create_dummy_coco_data(num_images=50)
    dummy_input = tmp_path / "instances_dummy.json"
    train_out = tmp_path / "train_subset.json"
    val_out = tmp_path / "val_heldout.json"

    with open(dummy_input, "w", encoding="utf-8") as f:
        json.dump(data, f)

    split_train_val_files(
        instances_train_path=dummy_input,
        train_subset_path=train_out,
        val_heldout_path=val_out,
        train_count=30,
        val_heldout_count=10,
        seed=42,
    )

    assert train_out.exists()
    assert val_out.exists()

    with open(train_out, "r", encoding="utf-8") as f:
        loaded_train = json.load(f)
    assert len(loaded_train["images"]) == 30

    with open(val_out, "r", encoding="utf-8") as f:
        loaded_val = json.load(f)
    assert len(loaded_val["images"]) == 10
