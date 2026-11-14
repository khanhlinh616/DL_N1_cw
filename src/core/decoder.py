"""Prediction Decoder, Coordinate Inversion, and Batched NMS for YOLO Detector.

Formulas (DEFINE_AND_DESIGN.md §2.4.3 & §2.7.4):
- Center: cx = (grid_x + 2*sigmoid(tx) - 0.5) * stride   [-0.5, 1.5] * stride
          cy = (grid_y + 2*sigmoid(ty) - 0.5) * stride   [-0.5, 1.5] * stride
- Size:   w = (2*sigmoid(tw))^2 * 4 * stride             [0, 16 * stride]
          h = (2*sigmoid(th))^2 * 4 * stride             [0, 16 * stride]
- Score:  score = sigmoid(t_obj) * sigmoid(t_cls)
- Original image restoration:
          x1_orig = (x1 - pad_w) / r
          y1_orig = (y1 - pad_h) / r
          x2_orig = (x2 - pad_w) / r
          y2_orig = (y2 - pad_h) / r
- Batched NMS: per-class non-maximum suppression (IoU threshold = 0.60, max 100 boxes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from src.data.category_mapper import CategoryMapper, DEFAULT_MAPPER


def decode_box_coordinates(
    box_preds: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Decode raw box prediction logits into [x1, y1, x2, y2] coordinates on 512x512 space.

    Args:
        box_preds: Tensor of shape [B, 4, H, W] representing (tx, ty, tw, th).
        stride: FPN level stride (e.g. 8 for P3, 16 for P4, 32 for P5).

    Returns:
        Tensor of shape [B, H, W, 4] with (x1, y1, x2, y2) coordinates.
    """
    B, _, H, W = box_preds.shape
    device = box_preds.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    tx = box_preds[:, 0]
    ty = box_preds[:, 1]
    tw = box_preds[:, 2]
    th = box_preds[:, 3]

    cx = (grid_x + 2.0 * torch.sigmoid(tx) - 0.5) * stride
    cy = (grid_y + 2.0 * torch.sigmoid(ty) - 0.5) * stride

    w = torch.pow(2.0 * torch.sigmoid(tw), 2) * 4.0 * stride
    h = torch.pow(2.0 * torch.sigmoid(th), 2) * 4.0 * stride

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H, W, 4]
    return boxes


class YoloDecoder(nn.Module):
    """Decodes multi-scale feature logits into bounding boxes and executes Batched NMS."""

    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        conf_threshold: float = 0.05,
        nms_iou_threshold: float = 0.60,
        max_detections: int = 100,
        num_classes: int = 80,
    ) -> None:
        """Initialize YOLO prediction decoder.

        Args:
            strides: FPN feature map strides [8, 16, 32].
            conf_threshold: Minimum confidence score threshold (default: 0.05).
            nms_iou_threshold: IoU threshold for Batched NMS (default: 0.60).
            max_detections: Maximum detections per image after NMS (default: 100).
            num_classes: Number of object categories (default: 80).
        """
        super().__init__()
        self.strides = strides
        self.level_names = ["P3", "P4", "P5"]
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes

    def decode_predictions(
        self,
        raw_predictions: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode raw logits across all FPN levels into concatenated boxes and scores.

        Args:
            raw_predictions: Dict from YOLODetector:
                {
                    'P3': {'box': [B, 4, 64, 64], 'obj': [B, 1, 64, 64], 'cls': [B, 80, 64, 64]},
                    'P4': {'box': [B, 4, 32, 32], 'obj': [B, 1, 32, 32], 'cls': [B, 80, 32, 32]},
                    'P5': {'box': [B, 4, 16, 16], 'obj': [B, 1, 16, 16], 'cls': [B, 80, 16, 16]},
                }

        Returns:
            Tuple of:
                - all_boxes: [B, 5376, 4] in (x1, y1, x2, y2) on 512x512 space.
                - all_scores: [B, 5376, 80] composite scores = sigmoid(obj) * sigmoid(cls).
        """
        batch_boxes: List[torch.Tensor] = []
        batch_scores: List[torch.Tensor] = []

        for lvl_idx, lvl_name in enumerate(self.level_names):
            if lvl_name not in raw_predictions:
                continue

            stride = self.strides[lvl_idx]
            preds = raw_predictions[lvl_name]
            box_pred = preds["box"]  # [B, 4, H, W]
            obj_pred = preds["obj"]  # [B, 1, H, W]
            cls_pred = preds["cls"]  # [B, 80, H, W]

            B, _, H, W = box_pred.shape

            # 1. Decode box coordinates to [B, H*W, 4]
            decoded_boxes = decode_box_coordinates(box_pred, stride=stride)
            flat_boxes = decoded_boxes.reshape(B, H * W, 4)
            batch_boxes.append(flat_boxes)

            # 2. Compute composite scores = sigmoid(t_obj) * sigmoid(t_cls)
            obj_prob = torch.sigmoid(obj_pred)  # [B, 1, H, W]
            cls_prob = torch.sigmoid(cls_pred)  # [B, 80, H, W]
            composite_score = obj_prob * cls_prob  # [B, 80, H, W]

            # Permute to [B, H*W, 80]
            flat_scores = composite_score.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            batch_scores.append(flat_scores)

        all_boxes = torch.cat(batch_boxes, dim=1)  # [B, 5376, 4]
        all_scores = torch.cat(batch_scores, dim=1)  # [B, 5376, 80]
        return all_boxes, all_scores

    def postprocess_single_image(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        conf_threshold: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Filter predictions by confidence, execute Batched NMS, and restore original image scale.

        Args:
            boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2) on 512x512 space.
            scores: Tensor of shape [N, 80] with composite class scores.
            metadata: Image metadata dict containing 'scale_ratio', 'pad_offsets', 'orig_size', 'image_id'.
            conf_threshold: Score filtering threshold (defaults to self.conf_threshold).
            nms_iou_threshold: NMS IoU threshold (defaults to self.nms_iou_threshold).
            max_detections: Max output detections (defaults to self.max_detections).

        Returns:
            Dict containing:
                - 'boxes': [K, 4] in (x1, y1, x2, y2) on original image space.
                - 'scores': [K] float scores.
                - 'labels': [K] int64 class indices in [0, 79].
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_thresh = nms_iou_threshold if nms_iou_threshold is not None else self.nms_iou_threshold
        max_det = max_detections if max_detections is not None else self.max_detections

        device = boxes.device

        # 1. Candidate filtering: find (anchor_idx, class_idx) pairs above threshold
        mask = scores > conf_thresh
        if not mask.any():
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
            empty_labels = torch.zeros((0,), dtype=torch.int64, device=device)
            return {"boxes": empty_boxes, "scores": empty_scores, "labels": empty_labels}

        anchor_indices, class_indices = mask.nonzero(as_tuple=True)
        cand_boxes = boxes[anchor_indices]  # [M, 4]
        cand_scores = scores[anchor_indices, class_indices]  # [M]
        cand_labels = class_indices  # [M]

        # 2. Batched NMS (isolated per class category)
        keep = batched_nms(cand_boxes, cand_scores, cand_labels, iou_threshold=nms_thresh)
        keep = keep[:max_det]

        final_boxes = cand_boxes[keep]
        final_scores = cand_scores[keep]
        final_labels = cand_labels[keep]

        # 3. Restore coordinates to original image coordinates using metadata
        if metadata is not None:
            scale_ratio = float(metadata.get("scale_ratio", 1.0))
            pad_w, pad_h = metadata.get("pad_offsets", (0.0, 0.0))
            orig_h, orig_w = metadata.get("orig_size", (512, 512))

            x1 = (final_boxes[:, 0] - pad_w) / scale_ratio
            y1 = (final_boxes[:, 1] - pad_h) / scale_ratio
            x2 = (final_boxes[:, 2] - pad_w) / scale_ratio
            y2 = (final_boxes[:, 3] - pad_h) / scale_ratio

            # Clamp to valid image boundaries
            x1 = torch.clamp(x1, min=0.0, max=float(orig_w))
            y1 = torch.clamp(y1, min=0.0, max=float(orig_h))
            x2 = torch.clamp(x2, min=0.0, max=float(orig_w))
            y2 = torch.clamp(y2, min=0.0, max=float(orig_h))

            final_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels,
        }

    def forward(
        self,
        raw_predictions: Dict[str, Dict[str, torch.Tensor]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        conf_threshold: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Complete decoding and Batched NMS across a batch of images.

        Args:
            raw_predictions: Model raw output dict for P3, P4, P5.
            metadatas: Optional list of metadata dicts corresponding to each image in batch.

        Returns:
            List of B dicts, each with 'boxes', 'scores', 'labels'.
        """
        all_boxes, all_scores = self.decode_predictions(raw_predictions)
        batch_size = all_boxes.size(0)

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            meta_b = metadatas[b] if metadatas is not None and b < len(metadatas) else None
            det = self.postprocess_single_image(
                boxes=all_boxes[b],
                scores=all_scores[b],
                metadata=meta_b,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            if meta_b is not None and "image_id" in meta_b:
                det["image_id"] = torch.tensor(meta_b["image_id"], dtype=torch.int64)
            results.append(det)

        return results


def to_coco_results(
    detections: List[Dict[str, Any]],
    image_ids: List[int],
    mapper: Optional[CategoryMapper] = None,
) -> List[Dict[str, Any]]:
    """Convert decoded detection dicts to official COCOeval JSON format.

    Format per detection item:
    {
        "image_id": int,
        "category_id": int (COCO ID in [1, 90]),
        "bbox": [x, y, width, height] (float rounded to 2 decimals),
        "score": float (rounded to 4 decimals),
    }

    Args:
        detections: List of detection dicts (one per image) with 'boxes', 'scores', 'labels'.
        image_ids: List of corresponding COCO image_ids.
        mapper: CategoryMapper instance (default: DEFAULT_MAPPER).

    Returns:
        List of COCO prediction dicts.
    """
    cat_mapper = mapper or DEFAULT_MAPPER
    coco_results: List[Dict[str, Any]] = []

    for det, img_id in zip(detections, image_ids):
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if boxes.numel() == 0:
            continue

        boxes_cpu = boxes.detach().cpu()
        scores_cpu = scores.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(len(scores_cpu)):
            x1, y1, x2, y2 = boxes_cpu[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            score = float(scores_cpu[i].item())
            class_idx = int(labels_cpu[i].item())

            try:
                category_id = cat_mapper.to_category_id(class_idx)
            except KeyError:
                continue

            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": category_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 4),
                }
            )

    return coco_results
