# YOLO-style One-Stage Object Detector — ResNet-50 + FPN trên COCO 2017

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0--cpu-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-COCO%202017-00BFFF)
![Tests](https://img.shields.io/badge/Tests-55%2F55%20passed-brightgreen)

## Overview

Bộ phát hiện đối tượng one-stage end-to-end theo phong cách YOLO: **ResNet-50 + FPN** làm backbone, Decoupled Head, label assigner anchor-free, CIoU + Focal Loss, và COCOeval tích hợp hoàn chỉnh. Huấn luyện trên tập con **14k ảnh** của MS COCO 2017, chạy CPU-only.

---

## Features

- Backbone ResNet-50 (FrozenBN, pretrained ImageNet) + FPN P3/P4/P5 (256-ch)
- Anchor-free assigner: 3 ô dương / vật thể
- Loss: 5×CIoU + 1×Focal(obj) + 1×BCE(cls)
- Optimizer: AdamW + CosineAnnealingLR + 2-epoch warmup
- Resume từ checkpoint (intra-epoch, epoch-level)
- `--git-sync`: auto commit & push sau mỗi epoch
- CLI: train, evaluate, demo, visualize, ablation

---

## Architecture

```
Input (512×512)
  └─ ResNet-50 (FrozenBN, pretrained ImageNet)
       └─ FPN → {P3 stride 8, P4 stride 16, P5 stride 32}  (256-ch)
            └─ Decoupled YOLO Head (shared weights, 3 branches)
                 ├─ Box branch   → (Δx, Δy, Δw, Δh)
                 ├─ Obj branch   → objectness score
                 └─ Cls branch   → 80-class logits
                      └─ Decoder + Batched NMS → Detections
```

| Thành phần | Chi tiết |
|:---|:---|
| Backbone | ResNet-50 FrozenBatchNorm2d |
| Neck | FPN P3/P4/P5, 256 channels |
| Head | Decoupled, shared weights, Focal bias ≈ −4.595 |
| Assigner | Anchor-free, 3 ô dương / vật thể |
| Loss | 5×CIoU + 1×Focal(obj) + 1×BCE(cls) |
| Optimizer | AdamW + CosineAnnealingLR + 2-epoch warmup |
| NMS | Batched NMS, IoU thr 0.60, top-100 |

---

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
│   ├── train.py      # CLI huấn luyện (lưu epoch_N.pth)
│   ├── evaluate.py   # CLI đánh giá (COCOeval)
│   ├── demo.py       # CLI demo trực quan
│   ├── visualize.py  # CLI trực quan hóa GT vs Pred
│   └── ablation.py   # CLI thực nghiệm bóc tách (M7)
├── tests/            # 14 tệp test, 55 unit tests + 1 slow
├── requirements.txt
└── RULES.md
```

---

## Requirements

- Python 3.10+
- CPU-only (không cần GPU)
- ~19 GB dung lượng ổ đĩa cho dữ liệu COCO

```bash
pip install -r requirements.txt
```

**Dữ liệu:**
- `val2017` (778 MB) + `annotations` (241 MB): đã sẵn có
- `train2017` (~18.5 GB): tải thủ công từ [images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip) → giải nén vào `data/coco/train2017/`

---

## Installation

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Thiết lập PYTHONPATH (bắt buộc trên Windows)
set PYTHONPATH=.
```

---

## Usage

### Huấn luyện

```bash
set PYTHONPATH=.

# Smoke test (không cần đủ dữ liệu)
python -m src.train --epochs 1 --batch-size 2 --max-images 8

# Baseline đầy đủ (cần train2017 ≥14k ảnh)
python -m src.train --epochs 20 --batch-size 16

# Tiếp tục từ checkpoint
python -m src.train --epochs 20 --resume outputs/checkpoints/last_epoch.pth

# Với auto git-sync sau mỗi epoch
python -m src.train --epochs 3 --batch-size 4 --resume outputs/checkpoints/epoch_3.pth --num-threads 4 --git-sync
```

### Đánh giá

```bash
# Đánh giá trên held-out 1k
python -m src.evaluate --dataset heldout --batch-size 8

# Đánh giá chính thức val2017 (chỉ chạy 1 lần tại M8, đã khoá)
python -m src.evaluate --dataset val2017 --final
```

### Demo & Visualization

```bash
# Demo trên 1 ảnh
python -m src.demo --image path/to/image.jpg --checkpoint outputs/checkpoints/best_map.pth

# Trực quan hóa GT vs Pred
python -m src.visualize --dataset val2017 --n 5

# Ablation study
python -m src.ablation --variant all
```

---

## Testing

```bash
set PYTHONPATH=.

# 55 unit tests
python -m pytest -m "not slow"

# Bao gồm 1 slow overfit test (chạy ~2 phút)
python -m pytest
```

Kết quả: **55/55 unit tests passed**, 1 slow overfit test passed.

---

## Results

### M3 — Overfit Test (10 ảnh, 40 bước, CPU)

| Chỉ số | Giá trị |
|:---|:---|
| Loss ban đầu | 58.9644 |
| Loss cuối | 1.0534 |
| Loss giảm | 98.21% |
| Recall@IoU0.9 | 1.00 (10/10) |

### M6 — Baseline Training (14k ảnh, CPU)

| Epoch | Train Loss | Val Loss (1k) |
|:---|:---:|:---:|
| 1 | 4.7706 | 4.7282 |
| 2 (best) | 4.1866 | 4.4330 |
| 3 | _đang chạy, dừng tại Batch 1500/3474_ | — |

### M7 — Ablation Study (100 ảnh held-out)

| Biến thể | Loss tổng | Kết luận |
|:---|:---:|:---|
| Full (CIoU + Focal + FPN P3–P5) | 425.08 | Baseline |
| Smooth L1 thay CIoU | 422.35 | CIoU tốt hơn trên positive anchors |
| BCE thay Focal Loss | 611.34 | Focal suppresses easy negatives hiệu quả hơn |
| Single-scale P5 | 25.14 | FPN phủ cả vật nhỏ (P3) và lớn (P5) |

### M8 — Final Evaluation (val2017, 200 ảnh, checkpoint best_map.pth)

| Chỉ số | Giá trị |
|:---|:---:|
| mAP@[.50:.95] | 0.0 |
| AP_large | 0.0001 |
| AR@100 | 0.0004 |
| FPS (CPU) | 2.61 |

> Kết quả M8 đã khoá (`outputs/final_eval.lock`) — không tái chạy.

---

## Limitations

- Chỉ huấn luyện trên **14k ảnh** (tập con), không đủ để hội tụ mAP cao trên val2017.
- **CPU-only**: 1 epoch ≈ 30–50 phút (batch=4, num-threads=4); không thực tế cho full training.
- Kích thước ảnh cố định **512×512**; không hỗ trợ multi-scale inference.
- Epoch 3 đang dở: checkpoint `epoch_3.pth` lưu tại Batch 1499 (43.2% epoch).

---

## Docs

| Tệp | Nội dung |
|:---|:---|
| [docs/report.md](docs/report.md) | Báo cáo đầy đủ: kết quả, hạn chế, cải tiến |
| [docs/DEFINE_AND_DESIGN.md](docs/DEFINE_AND_DESIGN.md) | Thiết kế kỹ thuật chi tiết |
| [docs/plan-progress-status.md](docs/plan-progress-status.md) | Tiến độ milestone |
| [docs/model.md](docs/model.md) | Kiến trúc model chi tiết |
| [docs/training.md](docs/training.md) | Hướng dẫn huấn luyện |
| [docs/evaluation.md](docs/evaluation.md) | Hướng dẫn đánh giá |
| [docs/ai-usage.md](docs/ai-usage.md) | Nhật ký sử dụng AI |
