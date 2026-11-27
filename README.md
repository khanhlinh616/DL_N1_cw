# YOLO-style One-Stage Object Detector — ResNet-50 + FPN trên COCO 2017

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0--cpu-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-COCO%202017-00BFFF)
![Tests](https://img.shields.io/badge/Tests-55%2F55%20passed-brightgreen)

## Overview

Bộ phát hiện đối tượng one-stage end-to-end theo phong cách YOLO: **ResNet-50 + FPN** làm backbone, Decoupled Head, label assigner anchor-free, CIoU + Focal Loss, và COCOeval tích hợp hoàn chỉnh. Huấn luyện trên tập con **14k ảnh** của MS COCO 2017 (CPU-only).

## Features

- Backbone ResNet-50 (FrozenBN, pretrained ImageNet) + FPN P3/P4/P5 (256-ch)
- Anchor-free assigner: 3 ô dương / vật thể; loss `5×CIoU + 1×Focal(obj) + 1×BCE(cls)`
- Optimizer: AdamW + CosineAnnealingLR + 2-epoch warmup; đóng băng backbone 3 epoch đầu, mở khóa từ Epoch 4
- Resume intra-epoch/epoch-level & `--git-sync` tự commit/push mỗi epoch
- CLI đầy đủ: `train`, `evaluate`, `demo`, `visualize`, `ablation`

## Architecture

```
Input (512×512) → ResNet-50 (FrozenBN) → FPN {P3 s=8, P4 s=16, P5 s=32} (256-ch)
  → Decoupled Head (shared weights: Box, Obj, Cls) → Decoder + Batched NMS → Detections
```

| Thành phần | Chi tiết |
|:---|:---|
| Backbone / Neck | ResNet-50 FrozenBatchNorm2d + FPN P3/P4/P5 (256-ch) |
| Head / Assigner | Decoupled (Focal bias ≈ −4.595), anchor-free 3 ô dương/vật thể |
| Loss / Optimizer | 5×CIoU + 1×Focal + 1×BCE; AdamW + CosineAnnealingLR (2-epoch warmup) |
| Decoder / NMS | Offset [−0.5, 1.5]×s, kích thước [0, 16s]; Batched NMS IoU 0.60, top-100 |

## Project Structure

```
d:\cw\
├── configs/          # YAML config (base, dataset, model)
├── data/coco/        # Dữ liệu COCO (không commit)
├── docs/             # Tài liệu kỹ thuật & báo cáo
├── outputs/          # Checkpoints, logs, kết quả (không commit)
├── src/
│   ├── core/         # assigner.py, decoder.py, loss.py
│   ├── data/         # category_mapper.py, dataset.py, transforms.py
│   ├── models/       # backbone_fpn.py, yolo_head.py, detector.py
│   ├── utils/        # box_ops.py, config.py, logger.py
│   └── {train,evaluate,demo,visualize,ablation}.py
├── tests/            # 14 tệp test: 55 unit tests + 1 slow overfit
├── requirements.txt
└── RULES.md
```

## Requirements & Installation

- **Môi trường:** Python 3.10+, CPU-only, ~19 GB ổ đĩa cho COCO 2017 (`val2017` + `annotations` đã có; `train2017` tại [images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip) giải nén vào `data/coco/train2017/`).

```bash
pip install -r requirements.txt
set PYTHONPATH=.   # Bắt buộc trên Windows trước khi chạy python -m src.*
```

## Usage

```bash
set PYTHONPATH=.

# Huấn luyện: smoke test / baseline / resume với git-sync
python -m src.train --epochs 1 --batch-size 2 --max-images 8
python -m src.train --epochs 20 --batch-size 16
python -m src.train --epochs 4 --batch-size 4 --resume outputs/checkpoints/last_epoch.pth --num-threads 4 --git-sync

# Đánh giá held-out 1k & val2017 (đã khóa sau M8)
python -m src.evaluate --dataset heldout --batch-size 8
python -m src.evaluate --dataset val2017 --final

# Demo, trực quan hóa & ablation
python -m src.demo --image path/to/image.jpg --checkpoint outputs/checkpoints/best_map.pth
python -m src.visualize --dataset val2017 --n 5
python -m src.ablation --variant all
```

## Testing

```bash
set PYTHONPATH=.
python -m pytest -m "not slow"   # 55 unit tests (đã pass 55/55)
python -m pytest                 # Bao gồm 1 slow overfit test (~2 phút)
```

## Results

### M3 — Overfit Test (10 ảnh, 40 bước, CPU)

| Initial Loss | Final Loss | Loss giảm | Recall@IoU0.9 |
|:---:|:---:|:---:|:---:|
| 58.9644 | 1.0534 | 98.21% | 1.00 (10/10) |

### M6 — Baseline Training (14k ảnh, CPU)

| Epoch | Train Loss | Val Loss (1k) | Trạng thái |
|:---|:---:|:---:|:---|
| 1 | 4.7706 | 4.7282 | ✅ Hoàn thành |
| 2 | 4.1866 | 4.4330 | ✅ Hoàn thành |
| 3 | 4.3168 | **4.1814** | ✅ Hoàn thành (`best_map.pth`) |
| 4 | 5.1428* | — | 🔄 Đang chạy tiếp từ Batch 2051 (dự kiến dừng 14h45) |

> \*avg loss Epoch 4 tính đến Batch 2050. Checkpoint: `outputs/checkpoints/epoch_4.pth`.

### M7 — Ablation Study (100 ảnh held-out)

| Biến thể | Loss tổng | Kết luận |
|:---|:---:|:---|
| Full (CIoU + Focal + FPN P3–P5) | 425.08 | Baseline |
| Smooth L1 thay CIoU | 422.35 | CIoU tốt hơn trên positive anchors |
| BCE thay Focal Loss | 611.34 | Focal giảm gradient từ easy negatives |
| Single-scale P5 | 25.14 | FPN phủ cả vật nhỏ (P3) và lớn (P5) |

### M8 — Final Evaluation (val2017, 200 ảnh — khóa tại `outputs/final_eval.lock`)

| mAP@[.50:.95] | AP_large | AR@100 | FPS (CPU) |
|:---:|:---:|:---:|:---:|
| 0.0 | 0.0001 | 0.0004 | 2.61 |

## Limitations

- Huấn luyện trên tập con **14k ảnh**, chưa đủ để hội tụ mAP cao trên `val2017` (M8 chạy trước khi xong Epoch 2–3).
- **CPU-only**: ~3–5 giờ/epoch (batch=4, num-threads=4); kích thước đầu vào cố định **512×512**.
- **Epoch 3 đã hoàn thành** (42k ảnh lũy kế, `best_map.pth` đạt val loss **4.1814**); **Epoch 4** đang tiếp tục chạy từ batch 2051/3474 (mở khóa backbone ResNet-50).

## Docs

| Tệp | Nội dung |
|:---|:---|
| [docs/report.md](docs/report.md) | Báo cáo đầy đủ: kết quả, hạn chế, cải tiến |
| [docs/DEFINE_AND_DESIGN.md](docs/DEFINE_AND_DESIGN.md) | Thiết kế kỹ thuật chi tiết |
| [docs/plan-progress-status.md](docs/plan-progress-status.md) | Tiến độ milestone |
| [docs/model.md](docs/model.md) / [docs/training.md](docs/training.md) / [docs/evaluation.md](docs/evaluation.md) | Chi tiết model, huấn luyện & đánh giá |
| [docs/ai-usage.md](docs/ai-usage.md) | Nhật ký sử dụng AI |
