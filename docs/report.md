# BÁO CÁO DỰ ÁN (PROJECT REPORT)

* **Dự án:** One-Stage Object Detector (YOLO-style Head trên COCO 2017)
* **Cập nhật lần cuối:** 2026-11-15

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

### 3.2 Kết Quả Baseline (M6 — Chưa Chạy)

| Chỉ số | Giá trị |
|:---|:---|
| Epochs hoàn thành | Chưa chạy |
| Train loss (epoch cuối) | Chưa chạy |
| Val-heldout mAP@[.5:.95] | Chưa chạy |
| Val-heldout AP50 | Chưa chạy |
| FPS (CPU) | Chưa chạy |

### 3.3 Ablation Studies (M7 — Chưa Chạy)

| Thí nghiệm | mAP@[.5:.95] | AP50 |
|:---|:---|:---|
| Baseline (FrozenBN + Focal + Aug) | Chưa chạy | Chưa chạy |
| Ablation 1: Trainable BN | Chưa chạy | Chưa chạy |
| Ablation 2: BCE thay Focal (obj) | Chưa chạy | Chưa chạy |
| Ablation 3: Không data augmentation | Chưa chạy | Chưa chạy |

### 3.4 Đánh Giá Chính Thức val2017 (M8 — Chưa Chạy)

> Chỉ chạy **1 lần** với `--final`. Kết quả khóa tại `outputs/final_eval.lock`.

| Chỉ số COCO | Giá trị |
|:---|:---|
| mAP@[.5:.95] | Chưa chạy |
| AP50 / AP75 | Chưa chạy |
| AP_small / AP_medium / AP_large | Chưa chạy |
| AR@1 / AR@10 / AR@100 | Chưa chạy |
| FPS trên CPU | Chưa chạy |

---

## 4. LIMITATIONS (HẠN CHẾ KỸ THUẬT)

1. **Quy mô dữ liệu nhỏ (ngoài đề):** Subset 14.000 ảnh thay vì 118k.
   Mô hình thiếu đa dạng ngữ cảnh → mAP kỳ vọng thấp hơn mô hình đầy đủ
   (ước tính −10 đến −20 AP50 so với COCO full).

2. **Vật thể nhỏ (AP_small):** Phân giải cố định 512×512 và FPN chỉ có 3 tầng
   (P3 stride 8, không có P2) → khó phát hiện vật nhỏ < 32×32 px.

3. **Static Assigner:** Center-grid 3 ô cố định không thích nghi theo loss như
   TAL/SimOTA → gán sai với vật thể dày đặc, chồng chéo cao.

4. **CPU-only:** Môi trường thực thi không có GPU → huấn luyện 20 epoch
   ước tính 10–24 giờ trên CPU; giới hạn thực nghiệm nhanh.

5. **train2017 chưa đủ:** Tại thời điểm viết báo cáo, train2017.zip đang tải
   (~8.4/18.5 GB). M6-M9 bị block cho đến khi tải xong và giải nén.

6. **Nhầm lớp trong điều kiện thiếu dữ liệu:** Với 14k ảnh, một số lớp hiếm
   (<50 instance) có thể bị nhầm thường xuyên; AP per-class sẽ không đồng đều.

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
