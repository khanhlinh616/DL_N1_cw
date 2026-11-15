# YOLO-style One-Stage Object Detector — ResNet-50 + FPN trên COCO 2017

Dự án cài đặt bộ phát hiện đối tượng one-stage end-to-end theo phong cách YOLO:
ResNet-50 + FPN làm backbone, Decoupled Head, label assigner anchor-free,
CIoU + Focal Loss, và COCOeval tích hợp hoàn chỉnh. Huấn luyện trên tập con 14k
ảnh của MS COCO 2017.

---

## Bài toán

Phát hiện đối tượng (Object Detection) trên 80 lớp COCO 2017.  
Đầu vào: ảnh RGB bất kỳ. Đầu ra: danh sách `[x1, y1, x2, y2, score, label]`.

**Lý do chọn One-Stage + Anchor-free:**
- One-stage loại bỏ bước proposal → inference nhanh hơn hai giai đoạn.
- Anchor-free đơn giản hoá pipeline, tránh hyperparameter anchor thủ công.
- CIoU hội tụ nhanh hơn IoU/GIoU; Focal Loss chống mất cân bằng nền/vật.

---

## Kiến trúc

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
| Decoder | offset [−0.5, 1.5]×s, kích thước [0, 16s] |
| NMS | Batched NMS, IoU thr 0.60, top-100 |

---

## Cấu trúc thư mục

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
│   ├── train.py      # CLI huấn luyện
│   ├── evaluate.py   # CLI đánh giá (COCOeval)
│   └── demo.py       # CLI demo trực quan (M9)
├── tests/            # 12 tệp test, 48 unit tests + 1 slow
├── requirements.txt
└── RULES.md
```

---

## Cài đặt

```bash
# Yêu cầu Python 3.10+
pip install -r requirements.txt
```

**Dữ liệu:**
```bash
# val2017 (778 MB) + annotations (241 MB): đã sẵn có
# train2017 (~18.5 GB): tải thủ công từ images.cocodataset.org/zips/train2017.zip
# Giải nén vào data/coco/train2017/
```

---

## Chạy

### Huấn luyện
```bash
# Smoke test (không cần đủ dữ liệu)
set PYTHONPATH=.
python -m src.train --epochs 1 --batch-size 2 --max-images 8

# Baseline đầy đủ (cần train2017 ≥14k ảnh)
python -m src.train --epochs 20 --batch-size 16

# Tiếp tục từ checkpoint
python -m src.train --epochs 20 --resume outputs/checkpoints/last_epoch.pth
```

### Đánh giá
```bash
# Đánh giá trên held-out 1k
python -m src.evaluate --dataset heldout --batch-size 8

# Đánh giá chính thức val2017 (chỉ chạy 1 lần tại M8)
python -m src.evaluate --dataset val2017 --final
```

### Demo trực quan
```bash
# Demo trên 1 ảnh (M9)
python -m src.demo --image data/coco/val2017/000000000139.jpg
```

---

## Kiểm thử

```bash
set PYTHONPATH=.
python -m pytest -v -m "not slow"   # 48 unit tests (~52 giây)
python -m pytest -v                 # Tất cả bao gồm overfit test (~90 giây)
```

---

## Kết quả

Xem chi tiết: [docs/report.md](docs/report.md)

| Giai đoạn | Chỉ số | Giá trị |
|:---|:---|:---|
| M3 Overfit 10 ảnh | Loss drop | 98.21% (58.96 → 1.05) |
| M3 Overfit 10 ảnh | Recall@IoU0.9 | 1.00 (10/10) |
| M6 Baseline (20 epochs) | mAP@[.5:.95] | Chưa chạy |
| M8 Final eval (val2017) | mAP@[.5:.95] | Chưa chạy |

> Số liệu M6-M8 sẽ được cập nhật sau khi hoàn tất tải train2017.zip (~18.5 GB).

---

## Giới hạn kỹ thuật

- Subset 14k ảnh thay vì 118k → mAP thấp hơn mô hình đầy đủ.
- Phân giải 512×512 cố định, vật nhỏ có thể bị bỏ sót.
- CPU-only trong môi trường hiện tại → huấn luyện chậm.

---

## Tài liệu

| Tệp | Nội dung |
|:---|:---|
| [docs/report.md](docs/report.md) | Báo cáo đầy đủ: kết quả, hạn chế, cải tiến |
| [docs/DEFINE_AND_DESIGN.md](docs/DEFINE_AND_DESIGN.md) | Thiết kế kỹ thuật chi tiết |
| [docs/plan-progress-status.md](docs/plan-progress-status.md) | Tiến độ milestone |
| [docs/ai-usage.md](docs/ai-usage.md) | Nhật ký sử dụng AI |
| [docs/model.md](docs/model.md) | Kiến trúc model chi tiết |
| [docs/training.md](docs/training.md) | Hướng dẫn huấn luyện |
| [docs/evaluation.md](docs/evaluation.md) | Hướng dẫn đánh giá |
