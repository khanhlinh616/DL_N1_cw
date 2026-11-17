"""Unit tests for demo inference script (Milestone M9)."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from PIL import Image

from src.demo import preprocess_image, draw_detections, run_demo


def test_preprocess_image(tmp_path: Path) -> None:
    # Create dummy RGB image
    dummy_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img_path = tmp_path / "test.jpg"
    dummy_img.save(img_path)

    tensor, orig_img, metadata = preprocess_image(img_path, img_size=512)

    assert tensor.shape == (1, 3, 512, 512)
    assert orig_img.size == (640, 480)
    assert metadata["orig_size"] == (480, 640)
    assert "scale_ratio" in metadata
    assert "pad_offsets" in metadata


def test_draw_detections() -> None:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0]])
    scores = torch.tensor([0.85, 0.45])
    labels = torch.tensor([0, 1])

    annotated = draw_detections(img, boxes, scores, labels)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (200, 200)


def test_run_demo_smoke(tmp_path: Path) -> None:
    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img_path = tmp_path / "smoke.jpg"
    dummy_img.save(img_path)

    res = run_demo(
        image_path=img_path,
        checkpoint_path=None,
        conf_threshold=0.5,
        output_dir=tmp_path / "demo_out",
        device_override="cpu",
    )

    assert "output_image" in res
    assert Path(res["output_image"]).exists()
    assert "boxes" in res
