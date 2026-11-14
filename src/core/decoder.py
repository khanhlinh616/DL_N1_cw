"""Offset decoding, score calculation, and Batched NMS for YOLO-style detector.

Decoding formulas (from DEFINE_AND_DESIGN.md §2.4.3):
- Center: c_hat = (grid + 2*sigmoid(t) - 0.5) * stride  (range [-0.5, 1.5] * stride)
- Size:   w_hat, h_hat = (2*sigmoid(t))^2 * 4 * stride   (range [0, 16*stride])
- Score:  score = sigmoid(t_obj) * sigmoid(t_cls)

Batched NMS: per-class NMS with IoU threshold, top-K=100.

TODO: Implement in Milestone M5.
"""

from __future__ import annotations
