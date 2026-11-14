"""Unit tests for anchor-free label assigner (Milestone M3).

Verifies:
- Proper level selection based on scale thresholds (64 and 168).
- Selection of exactly 3 positive cells (center + 2 nearest neighbors).
- Safe boundary handling at grid edges (no out-of-bounds or negative indexing).
- Target objectness 1.0 on positive cells and 0.0 on negative cells.
"""

from __future__ import annotations

import torch

from src.core.assigner import assign_targets


def test_scale_threshold_level_selection() -> None:
    """Test scale thresholds: <64 -> P3, 64-168 -> P4, >=168 -> P5."""
    # Small box: scale = sqrt(30 * 30) = 30 < 64 -> P3
    small_box = torch.tensor([[100.0, 100.0, 130.0, 130.0]])
    # Medium box: scale = sqrt(100 * 100) = 100 in [64, 168) -> P4
    med_box = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    # Large box: scale = sqrt(200 * 200) = 200 >= 168 -> P5
    large_box = torch.tensor([[100.0, 100.0, 300.0, 300.0]])

    labels = torch.tensor([5])

    # 1. Test Small Box -> P3
    targets_p3 = assign_targets([small_box], [labels])
    assert targets_p3["P3"]["pos_mask"].sum().item() > 0
    assert targets_p3["P4"]["pos_mask"].sum().item() == 0
    assert targets_p3["P5"]["pos_mask"].sum().item() == 0

    # 2. Test Medium Box -> P4
    targets_p4 = assign_targets([med_box], [labels])
    assert targets_p4["P3"]["pos_mask"].sum().item() == 0
    assert targets_p4["P4"]["pos_mask"].sum().item() > 0
    assert targets_p4["P5"]["pos_mask"].sum().item() == 0

    # 3. Test Large Box -> P5
    targets_p5 = assign_targets([large_box], [labels])
    assert targets_p5["P3"]["pos_mask"].sum().item() == 0
    assert targets_p5["P4"]["pos_mask"].sum().item() == 0
    assert targets_p5["P5"]["pos_mask"].sum().item() > 0


def test_positive_cells_count_exactly_three() -> None:
    """Test that a non-boundary object assigns exactly 3 positive grid cells."""
    # Place a box in the interior of the grid (P3 stride 8, cell ~ (20, 20))
    box = torch.tensor([[162.0, 162.0, 192.0, 192.0]])  # center at (177, 177)
    labels = torch.tensor([10])

    targets = assign_targets([box], [labels])

    pos_mask_p3 = targets["P3"]["pos_mask"][0]
    num_positive = pos_mask_p3.sum().item()

    assert num_positive == 3, f"Expected exactly 3 positive cells, got {num_positive}"

    # Target objectness should be 1.0 on exactly these 3 cells
    assert (targets["P3"]["target_obj"][0, 0][pos_mask_p3] == 1.0).all()
    # All other cells should be 0.0
    assert (targets["P3"]["target_obj"][0, 0][~pos_mask_p3] == 0.0).all()


def test_boundary_handling() -> None:
    """Test boundary cases where center is at the very edge of the image."""
    # Top-left corner box (center at (2, 2) -> cell (0, 0))
    corner_box = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    # Bottom-right corner box (center at (510, 510) -> cell (63, 63))
    br_box = torch.tensor([[508.0, 508.0, 512.0, 512.0]])
    labels = torch.tensor([1, 2])

    targets = assign_targets([torch.cat([corner_box, br_box], dim=0)], [labels])

    # Should not crash, and positive cells must be valid
    pos_mask = targets["P3"]["pos_mask"][0]
    assert pos_mask.sum().item() >= 2
    # Check top-left cell (0, 0) is marked
    assert pos_mask[0, 0].item() is True
    # Check bottom-right cell (63, 63) is marked
    assert pos_mask[63, 63].item() is True
