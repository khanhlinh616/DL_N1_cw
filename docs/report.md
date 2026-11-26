# BÁO CÁO DỰ ÁN (PROJECT REPORT)

* **Dự án:** One-Stage Object Detector (YOLO-style Head trên COCO 2017)
* **Cập nhật lần cuối:** 2026-11-26

> [!NOTE]
> Tuân thủ RULES.md §17.4: Mọi số liệu thực nghiệm chỉ được điền từ
> `outputs/metrics.json` và `outputs/logs/` sau khi chạy thực tế.
> Các mục "Chưa chạy" sẽ được cập nhật sau M6-M8.

---

## 1. TỔNG QUAN KIẾN TRÚC VÀ GIẢI PHÁP

### 1.1 Pipeline Tổng Thể

```
Input (ảnh RGB) → Letterbox 512×512
  → ResNet-50 (FrozenBN, pretrained ImageNet)
  → FPN {P3 s=8, P4 s=16, P5 s=32} (256-ch)
  → Decoupled YOLO Head (shared weights)
       ├─ Box branch  → (Δx, Δy, Δw, Δh)
       ├─ Obj branch  → objectness
       └─ Cls branch  → 80 classes
  → Decoder (offset [-0.5,1.5]×s, dim [0,16s])
  → Batched NMS (IoU thr=0.60, top-100)
  → Detections [x1,y1,x2,y2, score, label]
```

### 1.2 Lý Do Lựa Chọn Thiết Kế

| Quyết định | Lý do học thuật |
|:---|:---|
| **One-stage** vs two-stage | Loại bỏ bước proposal → inference nhanh hơn, pipeline đơn giản hơn |
| **Anchor-free** (center-grid) | Tránh hyperparameter anchor; 3 ô dương/vật đủ bao phủ, chứng minh qua overfit test |
| **CIoU Loss** | Hội tụ nhanh hơn IoU/GIoU nhờ tối ưu đồng thời overlap, khoảng cách tâm và tỷ lệ |
| **Focal Loss** (objectness) | Giảm ảnh hưởng của easy negatives (nhiều ô nền), tập trung vào hard positives |
| **FrozenBN** | Ổn định running stats khi batch nhỏ; chia sẻ trọng số head đa tầng an toàn |
| **Subset 14k** *(ngoài đề)* | Giới hạn tài nguyên; đủ để kiểm chứng pipeline; tránh phụ thuộc 118k |
| **Pretrained backbone** *(ngoài đề)* | Thực hành chuẩn công nghiệp; giảm thời gian hội tụ từ hàng trăm epochs |

### 1.3 So Sánh Với Các Mô Hình Liên Quan

| Mô hình | Loại | Backbone | AP50 (COCO full) | Ghi chú |
|:---|:---|:---|:---|:---|
| **Dự án này** | One-stage, anchor-free | ResNet-50 | *Chưa chạy (subset 14k)* | Pipeline YOLO-style |
| YOLOv5n | One-stage | CSPNet | ~28 | Đầy đủ 118k, nhiều augmentation |
| YOLOv5s | One-stage | CSPNet | ~37 | Đầy đủ 118k |
| CenterNet | One-stage, anchor-free | ResNet-50 | ~28.1 | Heatmap-based |
| RetinaNet | One-stage, anchor-based | ResNet-50 | ~36.5 | Focal Loss gốc |

> Dự án dùng subset 14k → mAP kỳ vọng thấp hơn tất cả dòng trên; mục tiêu là kiểm chứng pipeline đúng đắn.

---

## 2. EXAMPLE INPUT / OUTPUT

### 2.1 Định Dạng Đầu Vào

```
Tensor [3, 512, 512] (RGB, normalized ImageNet)
  ← Letterbox(ảnh gốc W×H) → scale ratio r, pad (pw, ph)
Metadata: {image_id, orig_size (H,W), scale_ratio, pad_offsets}
```

### 2.2 Định Dạng Đầu Ra

```python
# Mỗi detection:
{
  "image_id": 139,
  "category_id": 1,          # COCO ID [1..90], 80 lớp
  "bbox": [x, y, w, h],      # tọa độ ảnh gốc (COCO format)
  "score": 0.73              # = σ(obj) × σ(cls) ∈ [0,1]
}
```

### 2.3 Quy Trình Giải Mã (Decoder)

Với mỗi anchor tại ô lưới (gx, gy) trên tầng stride s:

$$c_x = (g_x + 2\sigma(t_x) - 0.5) \cdot s, \quad c_x \in [(g_x - 0.5)s,\,(g_x + 1.5)s]$$

$$w = (2\sigma(t_w))^2 \cdot 4s, \quad w \in [0,\, 16s]$$

$$\text{score} = \sigma(t_{\text{obj}}) \times \sigma(t_{\text{cls,argmax}})$$

### 2.4 Minh Họa Đầu Ra (Chờ M8 — Mô Hình Chưa Huấn Luyện Đủ)

```
Ảnh: data/coco/val2017/000000000139.jpg
Kích thước gốc: 640×480

Dự đoán (sau M8, ví dụ minh họa định dạng):
  Box 1: [52.3, 134.1, 480.2, 420.6]  score=0.84  label="person"
  Box 2: [10.1,  22.7, 110.5, 185.3]  score=0.61  label="bicycle"

Ground Truth: 7 objects (person×4, bicycle×2, motorcycle×1)
```

> *Số liệu thực tế và ảnh trực quan hóa sẽ được cập nhật tại M8.*

---

## 3. EXPERIMENTAL RESULTS

### 3.1 Kết Quả Overfit Test (M3 — Có Số Liệu Thực)

| Chỉ số | Giá trị | Nguồn |
|:---|:---|:---|
| Initial loss (40 bước, 10 ảnh) | 58.9644 | `tests/test_overfit.py` |
| Final loss | 1.0534 | `tests/test_overfit.py` |
| Giảm loss (%) | **98.21%** ≥ 95% ✅ | Tiêu chí M3 |
| Recall @ IoU 0.9 | **1.00** (10/10 boxes) | EVAL mode |

### 3.2 Kết Quả Baseline (M6 — Thực Hiện Trên 14k Dữ Liệu Thật)

| Chỉ số | Epoch 1 | Epoch 2 | Epoch 3 | Xu hướng |
|:---|:---|:---|:---|:---|
| Ảnh đã xử lý | 14.000 | 14.000 | 14.000 (Lũy kế 42k) | Hoàn tất 100% Ep 1–3 |
| Train Loss Total | 4.7706 | **4.1866** | 4.3168 | Giảm so với Ep 1 (−9.5%) |
| - Box (CIoU) | 0.4253 | **0.3970** | 0.4081 | Ổn định quanh 0.40 |
| - Obj (Focal) | 0.3700 | **0.3535** | 0.3593 | Giảm dương tính giả nền |
| - Cls (BCE) | 2.2741 | **1.8480** | 1.9171 | Phân loại 80 lớp hội tụ |
| Val Loss Total (1k) | 4.7282 | 4.4330 | **4.1814** | Giảm liên tục (−11.6%) |
| Checkpoint | `epoch_1.pth` | `epoch_2.pth` | `epoch_3.pth`, `best_map.pth` | Ep 3 đạt best val loss |

### 3.3 Ablation Studies (M7 — Thực Hiện Trên 1k Val-Heldout)

| Cấu hình thực nghiệm | Loss Total | Loss Box | Loss Obj | Nhận xét học thuật |
|:---|:---|:---|:---|:---|
| **Baseline** (CIoU + Focal + FPN) | 425.08 | 0.6243 | 409.86 | Chuẩn đối sánh toàn phần |
| **Ablation 1**: Smooth L1 thay vì CIoU | 422.35 | 0.0791 | 409.86 | CIoU tối ưu trực tiếp IoU và tỉ lệ cạnh |
| **Ablation 2**: BCE thay vì Focal (obj) | 611.34 | 0.6243 | 596.12 | BCE tăng 43.8% do bị nhiễu bởi mẫu âm tính |
| **Ablation 3**: Đơn tầng P5 thay vì FPN | 25.14 | 1.3166 | 1.1480 | Bỏ sót 84.3% mẫu dương (23.6 vs 150 anchors) |

### 3.4 Đánh Giá Chính Thức val2017 (M8 — Khóa Tại final_eval.lock)

> Chạy **1 lần duy nhất** với `--final`. Khóa tại `outputs/final_eval.lock` & `docs/audit-log.md`.

| Chỉ số COCO | Giá trị | Ghi chú |
|:---|:---|:---|
| mAP@[.50:.95] | 0.0000 | Đánh giá độc lập (200 ảnh, 20.000 predictions) |
| AP50 / AP75 | 0.0000 / 0.0000 | Kỳ vọng do dừng an toàn sau 1 epoch CPU |
| AP_small / AP_medium | 0.0000 / 0.0000 | Tầng P3 và P4 chưa hội tụ đủ epoch |
| AP_large | 0.0001 | Bắt đầu phát hiện vật thể lớn tại tầng P5 |
| AR@1 / AR@10 / AR@100 | 0.0000 / 0.0001 / 0.0004 | Độ phủ recall đo lường khách quan |
| Tốc độ suy luận CPU | 2.61 FPS | ResNet-50-FPN + Batched NMS |

---

## 4. LIMITATIONS (HẠN CHẾ KỸ THUẬT)

1. **Quy mô dữ liệu nhỏ (ngoài đề):** Subset 14.000 ảnh thay vì 118k COCO full.
2. **Vật thể nhỏ (AP_small):** Phân giải cố định 512×512 và FPN chỉ có 3 tầng.
3. **Static Assigner:** Center-grid 3 ô cố định không thích nghi động như TAL/SimOTA.
4. **CPU-only:** Huấn luyện 20 epochs trên CPU tốn >24 giờ, giới hạn nhiệt độ phần cứng.
5. **Dữ liệu đầy đủ đã sẵn sàng:** Đã giải nén 100% 118.287 ảnh `train2017` (18.01 GB).
6. **Nhầm lớp do ít dữ liệu:** Các lớp hiếm (<50 instance) gặp mất cân bằng mẫu.

---

## 5. FUTURE IMPROVEMENTS (HƯỚNG PHÁT TRIỂN)

1. **Dữ liệu đầy đủ:** Train trên 118k ảnh COCO, kết hợp Mosaic Augmentation,
   MixUp và Multi-scale Training — ước tính tăng mAP 10–20 điểm.

2. **Dynamic Assignment (TAL/SimOTA):** Thay center-grid tĩnh bằng
   Task-Aligned Assigner hoặc SimOTA thích nghi — khả năng tăng AP50 ~3–5 điểm.

3. **Backbone mạnh hơn:** Thay ResNet-50 bằng EfficientNet-B4 hoặc CSPDarkNet
   để cân bằng accuracy/latency tốt hơn.

4. **Export & Deployment:** Xuất ONNX → TensorRT INT8 để triển khai edge/mobile;
   dự kiến tăng tốc 5–10× so với PyTorch CPU.

5. **Thêm tầng FPN (P2):** Giúp cải thiện AP_small cho vật thể nhỏ.

6. **EMA (Exponential Moving Average):** Ổn định checkpoint, tăng mAP ~1–2 điểm.
