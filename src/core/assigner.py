"""Anchor-free center-grid label assigner for YOLO-style detector.

Assignment rules:
1. Scale-based FPN level selection according to S = sqrt(w * h):
   - S < 64:      P3 (stride 8, grid 64x64)
   - 64 <= S < 168: P4 (stride 16, grid 32x32)
   - S >= 168:    P5 (stride 32, grid 16x16)
2. Positive cell selection (Center cell + 2 nearest neighbors, max 3 cells):
   - Primary: (i, j) = (floor(gx), floor(gy))
   - Horizontal: i - 1 if dx < 0.5 and i > 0 else i + 1 if dx >= 0.5 and i < W - 1
   - Vertical:   j - 1 if dy < 0.5 and j > 0 else j + 1 if dy >= 0.5 and j < H - 1
3. Target objectness: 1.0 at positive cells, 0.0 at all other cells.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


LEVEL_CONFIGS = [
    {"name": "P3", "stride": 8, "grid_shape": (64, 64), "min_scale": 0.0, "max_scale": 64.0},
    {"name": "P4", "stride": 16, "grid_shape": (32, 32), "min_scale": 64.0, "max_scale": 168.0},
    {"name": "P5", "stride": 32, "grid_shape": (16, 16), "min_scale": 168.0, "max_scale": float("inf")},
]


def assign_targets(
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    fpn_strides: List[int] = [8, 16, 32],
    grid_shapes: List[Tuple[int, int]] = [(64, 64), (32, 32), (16, 16)],
    scale_thresholds: Tuple[float, float] = (64.0, 168.0),
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Assign ground truth boxes and classes to FPN grid cells.

    Args:
        gt_boxes: List of B tensors, each [N_i, 4] in xyxy format on 512x512 space.
        gt_labels: List of B tensors, each [N_i] in [0, 79].
        fpn_strides: Strides for P3, P4, P5 (default: [8, 16, 32]).
        grid_shapes: Grid heights and widths [(H_3, W_3), (H_4, W_4), (H_5, W_5)].
        scale_thresholds: Scale thresholds (64.0, 168.0) dividing P3, P4, P5.

    Returns:
        Dict mapping level name ('P3', 'P4', 'P5') to target dict:
        {
            'P3': {
                'pos_mask': [B, H_3, W_3] (BoolTensor),
                'target_boxes': [B, 4, H_3, W_3] (FloatTensor in xyxy on 512),
                'target_obj': [B, 1, H_3, W_3] (FloatTensor, 1.0 at pos, 0.0 at neg),
                'target_cls': [B, H_3, W_3] (LongTensor, class_idx at pos),
            }, ...
        }
    """
    batch_size = len(gt_boxes)
    device = gt_boxes[0].device if batch_size > 0 and gt_boxes[0].numel() > 0 else torch.device("cpu")

    level_names = ["P3", "P4", "P5"]
    targets: Dict[str, Dict[str, torch.Tensor]] = {}

    for lvl_idx, lvl_name in enumerate(level_names):
        h, w = grid_shapes[lvl_idx]
        targets[lvl_name] = {
            "pos_mask": torch.zeros((batch_size, h, w), dtype=torch.bool, device=device),
            "target_boxes": torch.zeros((batch_size, 4, h, w), dtype=torch.float32, device=device),
            "target_obj": torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=device),
            "target_cls": torch.zeros((batch_size, h, w), dtype=torch.int64, device=device),
        }

    thresh_low, thresh_high = scale_thresholds

    for b in range(batch_size):
        boxes = gt_boxes[b]
        labels = gt_labels[b]
        num_gt = boxes.shape[0]
        if num_gt == 0:
            continue

        # Sort GT boxes by area descending so smaller boxes overwrite larger boxes in case of cell collision
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        scales = torch.sqrt(areas.clamp(min=0.0))

        sort_indices = torch.argsort(areas, descending=True)

        for idx in sort_indices:
            box = boxes[idx]
            label = labels[idx].item()
            scale = scales[idx].item()

            # 1. Select FPN level according to scale
            if scale < thresh_low:
                lvl_idx = 0  # P3
            elif scale < thresh_high:
                lvl_idx = 1  # P4
            else:
                lvl_idx = 2  # P5

            lvl_name = level_names[lvl_idx]
            stride = fpn_strides[lvl_idx]
            grid_h, grid_w = grid_shapes[lvl_idx]

            # 2. Compute center coordinates and cell index
            cx = (box[0] + box[2]).item() * 0.5
            cy = (box[1] + box[3]).item() * 0.5

            gx = cx / stride
            gy = cy / stride

            i = int(math.floor(gx))
            j = int(math.floor(gy))

            # Clamp main cell within valid grid bounds
            i = min(max(0, i), grid_w - 1)
            j = min(max(0, j), grid_h - 1)

            dx = gx - i
            dy = gy - j

            # 3. Select up to 3 cells: primary + horizontal neighbor + vertical neighbor
            selected_cells: List[Tuple[int, int]] = [(i, j)]

            # Horizontal neighbor
            if dx < 0.5 and i > 0:
                selected_cells.append((i - 1, j))
            elif dx >= 0.5 and i < grid_w - 1:
                selected_cells.append((i + 1, j))

            # Vertical neighbor
            if dy < 0.5 and j > 0:
                selected_cells.append((i, j - 1))
            elif dy >= 0.5 and j < grid_h - 1:
                selected_cells.append((i, j + 1))

            # 4. Assign targets for each selected cell
            for ci, cj in selected_cells:
                targets[lvl_name]["pos_mask"][b, cj, ci] = True
                targets[lvl_name]["target_obj"][b, 0, cj, ci] = 1.0
                targets[lvl_name]["target_boxes"][b, :, cj, ci] = box
                targets[lvl_name]["target_cls"][b, cj, ci] = label

    return targets
