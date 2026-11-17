"""Single image demo inference script (Milestone M9).

Usage:
    python -m src.demo --image data/coco/val2017/000000000139.jpg --checkpoint outputs/checkpoints/best_map.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.core.decoder import YoloDecoder
from src.data.category_mapper import DEFAULT_MAPPER
from src.data.transforms import letterbox_image
from src.models.detector import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detector demo on a single image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input RGB image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_map.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence score threshold for filtering detections (default: 0.25)",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.60,
        help="NMS IoU threshold (default: 0.60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/demo",
        help="Directory to save visualized detection output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cpu' or 'cuda')",
    )
    return parser.parse_args()


def preprocess_image(
    image_path: str | Path,
    img_size: int = 512,
) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
    """Load and preprocess image for detector inference."""
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found: {img_p}")

    orig_image = Image.open(img_p).convert("RGB")
    orig_w, orig_h = orig_image.size

    padded_image, scale_ratio, pad_offsets = letterbox_image(orig_image, target_size=img_size)

    tensor = TF.to_tensor(padded_image)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    metadata = {
        "orig_size": (orig_h, orig_w),
        "scale_ratio": scale_ratio,
        "pad_offsets": pad_offsets,
        "image_id": 0,
    }
    return tensor.unsqueeze(0), orig_image, metadata


def draw_detections(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> Image.Image:
    """Draw bounding boxes and class labels onto PIL image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Use default font
    font = ImageFont.load_default()

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    for i in range(len(boxes_np)):
        x1, y1, x2, y2 = boxes_np[i]
        score = float(scores_np[i])
        cls_idx = int(labels_np[i])
        class_name = DEFAULT_MAPPER.get_name(cls_idx)

        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline="#E53E3E", width=3)

        label_text = f"{class_name} {score:.2f}"

        # Draw text background
        text_bbox = draw.textbbox((x1, max(0.0, y1 - 14.0)), label_text, font=font)
        draw.rectangle(text_bbox, fill="#E53E3E")
        draw.text((x1 + 2.0, max(0.0, y1 - 14.0)), label_text, fill="white", font=font)

    return annotated


def run_demo(
    image_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.60,
    output_dir: str | Path = "outputs/visualizations/demo",
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute single-image inference and visualization."""
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor, orig_image, metadata = preprocess_image(image_path, img_size=512)
    tensor = tensor.to(device)

    # Build model
    model = build_detector(num_classes=80, pretrained_backbone=False)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded weights from: {checkpoint_path}")
    else:
        print("[WARN] Checkpoint not found or not specified. Running with model weights as initialized.")

    model = model.to(device)
    model.eval()

    decoder = YoloDecoder(
        num_classes=80,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_threshold,
    )

    with torch.no_grad():
        raw_preds = model(tensor)
        detections_list = decoder(
            raw_preds,
            metadatas=[metadata],
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_threshold,
            max_detections=100,
        )

    detections = detections_list[0]
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    print(f"\n[INFO] Detections count above score {conf_threshold}: {len(boxes)}")
    for i in range(min(15, len(boxes))):
        cls_name = DEFAULT_MAPPER.get_name(int(labels[i]))
        print(f"  #{i+1}: {cls_name:<15} score={scores[i]:.3f} box={[round(x, 1) for x in boxes[i].tolist()]}")

    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    annotated_img = draw_detections(orig_image, boxes, scores, labels)

    input_stem = Path(image_path).stem
    out_file = out_d / f"{input_stem}_detected.jpg"
    annotated_img.save(out_file)
    print(f"[SUCCESS] Visualized detection saved to: {out_file}\n")

    return {
        "output_image": str(out_file),
        "num_detections": len(boxes),
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist(),
    }


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print(" YOLO Detector Single-Image Demo CLI")
    print("=" * 65)
    run_demo(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        conf_threshold=args.conf_thresh,
        nms_threshold=args.nms_thresh,
        output_dir=args.output_dir,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
