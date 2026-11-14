"""Unit tests for category mapper (Milestone M1).

Verifies:
- 100% two-way bidirectional mapping for all 80 classes.
- Exactly 10 missing category IDs are rejected.
- Out-of-bounds indices are rejected.
- Class names list is properly ordered and populated.
"""

from __future__ import annotations

import pytest

from src.data.category_mapper import (
    DEFAULT_MAPPER,
    MISSING_CATEGORY_IDS,
    CategoryMapper,
    category_id_to_class_idx,
    class_idx_to_category_id,
)


def test_bidirectional_mapping_completeness() -> None:
    """Test that all 80 classes map cleanly 2-way with 100% fidelity."""
    mapper = CategoryMapper()
    assert mapper.num_classes == 80

    for cat_id in mapper._cat_id_to_class_idx.keys():
        class_idx = mapper.to_class_idx(cat_id)
        assert 0 <= class_idx < 80
        recovered_cat_id = mapper.to_category_id(class_idx)
        assert recovered_cat_id == cat_id

    for class_idx in range(80):
        cat_id = mapper.to_category_id(class_idx)
        recovered_idx = mapper.to_class_idx(cat_id)
        assert recovered_idx == class_idx


def test_convenience_functions() -> None:
    """Test module-level convenience functions."""
    assert category_id_to_class_idx(1) == 0  # person -> 0
    assert class_idx_to_category_id(0) == 1
    assert category_id_to_class_idx(90) == 79  # toothbrush -> 79
    assert class_idx_to_category_id(79) == 90


def test_missing_ids_rejected() -> None:
    """Test that all 10 missing COCO category IDs raise KeyError."""
    assert len(MISSING_CATEGORY_IDS) == 10
    for missing_id in MISSING_CATEGORY_IDS:
        with pytest.raises(KeyError, match=f"Invalid category_id {missing_id}"):
            category_id_to_class_idx(missing_id)


def test_out_of_bounds_rejected() -> None:
    """Test out-of-bounds category_id and class_idx raise KeyError."""
    with pytest.raises(KeyError):
        category_id_to_class_idx(0)
    with pytest.raises(KeyError):
        category_id_to_class_idx(91)

    with pytest.raises(KeyError):
        class_idx_to_category_id(-1)
    with pytest.raises(KeyError):
        class_idx_to_category_id(80)


def test_class_names() -> None:
    """Test class names retrieval."""
    names = DEFAULT_MAPPER.class_names
    assert len(names) == 80
    assert names[0] == "person"
    assert names[1] == "bicycle"
    assert names[79] == "toothbrush"
