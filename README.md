# YOLO-style One-Stage Object Detector — ResNet-50 + FPN trên COCO 2017

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-COCO%202017-00BFFF)
![Tests](https://img.shields.io/badge/Tests-55%2F55%20passed-brightgreen)

## Overview

Bộ phát hiện đối tượng one-stage end-to-end theo phong cách YOLO: dùng **ResNet-50 (FrozenBN) + FPN** làm backbone/neck, kết hợp **Decoupled YOLO Head**, bộ gán nhãn **Anchor-free**, hàm mất mát **CIoU + Focal + BCE**, và bộ giải mã **Batched NMS** tích hợp **COCOeval**. Dự án huấn luyện trên tập con **14.000 ảnh** của MS COCO 2017 (môi trường CPU-only) và tách riêng **1.000 ảnh val-heldout** để chống rò rỉ dữ liệu.

## Features

- **Backbone & Neck:** ResNet-50 pretrained ImageNet (`FrozenBatchNorm2d`) + FPN `{P3, P4, P5}` cố định 256 kênh (`29.71M` tổng tham số).
- **Decoupled Head:** Tách 3 nhánh Box, Objectness (khởi tạo Focal bias $b \approx -4.595$), Class (80 lớp COCO); chia sẻ trọng số Conv nhưng tách BatchNorm theo tầng.
- **Anchor-Free Assigner & Loss:** Gán tối đa 3 ô dương gần tâm/vật thể; hàm loss $5.0\,\mathcal{L}_{\text{CIoU}} + 1.0\,\mathcal{L}_{\text{Focal\_obj}} + 1.0\,\mathcal{L}_{\text{BCE\_cls}}$.
- **Training Engine:** `AdamW` + `CosineAnnealingLR` (warmup 2 epochs), Gradient Clipping `10.0`, đóng băng backbone 3 epoch đầu và mở khóa `layer2–4` từ Epoch 4.
- **Checkpoint & Sync:** Lưu checkpoint mỗi 50 batches (`epoch_{N}.pth`), hỗ trợ resume chính xác từng batch và tự đồng bộ GitHub qua `--git-sync`.
- **CLI đầy đủ:** `train`, `evaluate`, `demo`, `visualize`, `ablation`.

## Architecture

```text
Input [B, 3, 512, 512] (Letterbox)
  → ResNet-50 (FrozenBN) → FPN {P3 s=8, P4 s=16, P5 s=32} (256-ch)
  → Decoupled Head (Box [4], Obj [1], Cls [80])
  → Decoder (offset [-0.5, 1.5]×s, dim [0, 16s]) → Batched NMS (IoU 0.60, top-100)
```

| Thành phần | Chi tiết kỹ thuật |
|:---|:---|
| Backbone / Neck | ResNet-50 `FrozenBatchNorm2d` + FPN P3/P4/P5 (5.376 ô lưới) |
| Head / Assigner | Decoupled Head (b ≈ −4.595), Anchor-free 3 ô dương/vật thể |
| Loss / Optimizer | 5×CIoU + 1×Focal + 1×BCE; AdamW (`lr=1e-3`) + CosineAnnealingLR |
| Decoder / NMS | Score = σ(obj)×σ(cls) ≥ 0.05; Batched NMS theo lớp, khôi phục ảnh gốc |

## Project Structure

```text
DL_N1_cw/
├── configs/          # YAML configs (base, dataset, model)
├── data/coco/        # Dữ liệu COCO 2017 (không commit — xem .gitignore)
├── docs/             # Tài liệu kỹ thuật, báo cáo, nhật ký & tiến độ
├── outputs/          # Checkpoints, logs, eval & viz (không commit)
├── src/
│   ├── core/         # assigner.py, decoder.py, loss.py
│   ├── data/         # category_mapper.py, dataset.py, download_data.py, transforms.py
│   ├── models/       # backbone_fpn.py, yolo_head.py, detector.py
│   ├── utils/        # box_ops.py, config.py, logger.py
│   └── {train,evaluate,demo,visualize,ablation}.py
├── tests/            # 14 tệp test_*.py + verify_split.py (55 unit + 1 slow)
├── requirements.txt
└── RULES.md
```

## Requirements & Installation

- **Môi trường:** Python 3.10+, PyTorch 2.0+ (CPU/CUDA), tối thiểu ~19 GB ổ đĩa cho COCO 2017 (`val2017` + `annotations` + `train2017`).

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="."   # PowerShell (Windows) | CMD: set PYTHONPATH=.
```

## Usage

```bash
set PYTHONPATH=.

# Huấn luyện: smoke test / baseline 5 epochs / resume Epoch 4 với giới hạn 4 luồng CPU
python -m src.train --epochs 1 --batch-size 2 --max-images 8
python -m src.train --epochs 5 --batch-size 4 --num-threads 4 --git-sync
python -u -m src.train --epochs 5 --batch-size 4 --resume outputs/checkpoints/epoch_4.pth --num-threads 4 --git-sync

# Đánh giá nội bộ (held-out 1k) & đánh giá chính thức (val2017 — đã khóa sau M8)
python -m src.evaluate --dataset heldout --batch-size 8
python -m src.evaluate --dataset val2017 --final

# Demo 1 ảnh, trực quan hóa 15 mẫu & chạy ablation
python -m src.demo --image data/coco/val2017/000000000139.jpg --checkpoint outputs/checkpoints/best_map.pth
python -m src.visualize --predictions outputs/evaluation_results/predictions_heldout.json --num-samples 15
python -m src.ablation --checkpoint outputs/checkpoints/best_map.pth --max-images 100
```

## Testing

```bash
set PYTHONPATH=.
python -m pytest -m "not slow"   # 55 unit tests nhanh (55/55 passed)
python -m pytest                 # Bao gồm 1 slow overfit test 10 ảnh (~2 phút)
```

## Milestones Status (M0–M9)

| Mốc | Nội dung chính | Trạng thái |
|:---|:---|:---|
| **M0–M1** | Split 14k/1k (`seed=42`), Category mapping 80 lớp, Letterbox $512^2$ ($\Delta < 1$ px) | ✅ Hoàn thành |
| **M2–M3** | ResNet-50 (FrozenBN) + FPN + Decoupled Head; Assigner 3 ô + MultiTaskLoss (Overfit 98.21%) | ✅ Hoàn thành |
| **M4–M5** | Training Engine (`AdamW`, Cosine LR, resume, `--git-sync`); Decoder + Batched NMS + Pilot 100 ảnh | ✅ Hoàn thành |
| **M6** | Baseline 14k (kế hoạch 5 epochs CPU): Ep1–3 xong (`best_map.pth` Val 4.1814), Ep4 đạt 60.4% | ⏸ Tạm dừng Ep4 |
| **M7** | 3 bài Ablation Studies trên `val-heldout 1k` (CIoU, Focal Loss, FPN) | ✅ Hoàn thành |
| **M8–M9** | Final Eval `val2017` (khóa `final_eval.lock`), Visualizer 15 mẫu, CLI & 55/55 unit tests | ✅ Hoàn thành |

## Checkpoint Status & Results

### Trạng thái Checkpoint hiện tại (`outputs/checkpoints/`)
- `best_map.pth` / `epoch_3.pth` / `last_epoch.pth`: Lưu tại **Epoch 3** (Val Loss tốt nhất: **4.1814**).
- `epoch_4.pth`: Tạm dừng **Epoch 4** tại Batch **2100/3474** (60.4%, lúc 14:45 ngày 27/11/2026), còn 1.374 batches cuối Epoch 4 + Epoch 5 theo kế hoạch **5 epochs**.

### M3 — Overfit Test (10 ảnh, 40 bước, CPU)
| Initial Loss | Final Loss | Mức giảm Loss | Recall @ IoU 0.9 |
|:---:|:---:|:---:|:---:|
| 58.9644 | 1.0534 | **98.21%** | **1.00** (10/10 boxes) |

### M6 — Baseline Training (Mục tiêu 5 Epochs, 14k Train / 1k Val Held-out, CPU)
| Epoch | Backbone | Train Loss | Val Loss (1k) | Trạng thái |
|:---:|:---|:---:|:---:|:---|
| 1 | Frozen | 4.7706 | 4.7282 | ✅ Hoàn thành (`epoch_1.pth`) |
| 2 | Frozen | 4.1866 | 4.4330 | ✅ Hoàn thành (`epoch_2.pth`) |
| 3 | Frozen | 4.3168 | **4.1814** | ✅ Hoàn thành (`epoch_3.pth`, `best_map.pth`) |
| 4 | Unfrozen (L2–4) | 5.1313* | — | ⏸ Tạm dừng Batch 2100/3474 (60.4%, `epoch_4.pth`) |
| 5 | Unfrozen (L2–4) | — | — | ⏳ Kế hoạch chạy tiếp sau khi xong Epoch 4 |

> \*Avg loss của Epoch 4 tính đến Batch 2100 (giảm từ 6.1682 sau khi mở khóa backbone).

### M7 — Ablation Study (100 ảnh held-out) & M8 — Final Eval (`val2017`, đã khóa)
- **M7 Ablation:** Full Pipeline (`425.08`) vs Smooth L1 (`422.35`, kém khớp hình học 2D) vs BCE (`611.34`, tăng 43.8% loss nền) vs Single-scale P5 (`25.14`, bỏ sót 84.3% vật nhỏ/vừa).
- **M8 Final Eval (`outputs/final_eval.lock`):** 200 ảnh / 20.000 predictions — `mAP@[.50:.95] = 0.0`, `AP_large = 0.0001`, `AR@100 = 0.0004`, tốc độ **2.61 FPS (CPU)**.

## Limitations

- Huấn luyện trên tập con **14k ảnh** (~12% COCO) nên chưa đủ để hội tụ `mAP` cao trên `val2017` (M8 đánh giá trước khi hoàn tất Epoch 3–5).
- Chạy **CPU-only** (~3–5 giờ/epoch với `batch_size=4`), kế hoạch thực tế giới hạn **5 epochs** để phù hợp tài nguyên phần cứng; độ phân giải đầu vào cố định $512 \times 512$.

## Docs

| Tệp | Nội dung |
|:---|:---|
| [docs/report.md](docs/report.md) | Báo cáo dự án: kiến trúc, thực nghiệm, hạn chế & hướng phát triển |
| [docs/DEFINE_AND_DESIGN.md](docs/DEFINE_AND_DESIGN.md) | Tài liệu định nghĩa và thiết kế kỹ thuật gốc |
| [docs/plan-progress-status.md](docs/plan-progress-status.md) | Theo dõi tiến độ Milestones M0–M9 và chi tiết Epoch 1–4 |
| [docs/model.md](docs/model.md) / [docs/data.md](docs/data.md) | Chi tiết kiến trúc ResNet-50-FPN-Head và Pipeline dữ liệu COCO |
| [docs/training.md](docs/training.md) / [docs/evaluation.md](docs/evaluation.md) | Quy trình huấn luyện, giải mã dự đoán và chuẩn đánh giá COCOeval |
| [docs/demo.md](docs/demo.md) / [docs/audit-log.md](docs/audit-log.md) | Hướng dẫn Demo/Visualize và Nhật ký kiểm toán `val2017` |
| [docs/ai-usage.md](docs/ai-usage.md) | Nhật ký minh bạch sử dụng AI theo từng milestone |
