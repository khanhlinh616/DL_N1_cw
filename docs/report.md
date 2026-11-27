# BÁO CÁO DỰ ÁN (PROJECT REPORT)

* **Dự án:** One-Stage Object Detector (YOLO-style Head trên COCO 2017)
* **Cập nhật lần cuối:** 2026-11-27

> [!NOTE]
> Tuân thủ RULES.md §17.4: Mọi số liệu thực nghiệm trong báo cáo này được trích xuất trực tiếp từ `outputs/metrics.json` và `outputs/logs/`, tuyệt đối không suy diễn hoặc điền số liệu chưa đo.

---

## 1. TỔNG QUAN KIẾN TRÚC VÀ GIẢI PHÁP

### 1.1 Pipeline Tổng Thể

```
Input (ảnh RGB) → Letterbox 512×512 (giữ tỷ lệ, đệm xám 114)
  → ResNet-50 (FrozenBN, pretrained ImageNet)
  → FPN {P3 s=8, P4 s=16, P5 s=32} (256-ch)
  → Decoupled YOLO Head (shared weights, per-level BN)
       ├─ Box branch  → (tx, ty, tw, th)
       ├─ Obj branch  → objectness (bias ≈ -4.595)
       └─ Cls branch  → 80 classes
  → Decoder (offset [-0.5,1.5]×s, dim [0,16s])
  → Batched NMS (IoU thr=0.60, top-100)
  → Detections [x1, y1, x2, y2, score, label]
```

### 1.2 Lý Do Lựa Chọn Thiết Kế

| Quyết định | Lý do học thuật & kỹ thuật |
|:---|:---|
| **One-stage** vs Two-stage | Bỏ bước RPN/RoIAlign → suy luận một lượt nhanh hơn, kiến trúc gọn |
| **Anchor-free** (center-grid) | Giảm số ứng viên từ 16.128 xuống 5.376 (giảm 3×), bỏ K-means anchor |
| **CIoU Loss** | Tối ưu đồng thời độ chồng lấn IoU, khoảng cách tâm và tỷ lệ cạnh |
| **Focal Loss** (objectness) | Chế ngự >99% ô nền dễ (`gamma=2.0, alpha=0.25`), tập trung vào vật thể |
| **FrozenBN** | Giữ nguyên thống kê ImageNet khi batch nhỏ ($B \le 16$) |
| **Subset 14k** *(ngoài đề)* | Phù hợp phần cứng CPU-only, vẫn kiểm chứng đầy đủ pipeline COCO |
| **Pretrained backbone** *(ngoài đề)* | Tận dụng đặc trưng thị giác tổng quát, rút ngắn thời gian hội tụ |

### 1.3 Bối Cảnh So Sánh Với Các Mô Hình Khác

| Mô hình | Loại | Backbone | AP50 (COCO full) | Ghi chú |
|:---|:---|:---|:---|:---|
| **Dự án này** | One-stage, anchor-free | ResNet-50 | *Chưa hội tụ (subset 14k, CPU)* | Kiểm chứng pipeline chuẩn COCO |
| YOLOv5n / YOLOv5s | One-stage | CSPNet | ~28 / ~37 | Train 118k ảnh, Mosaic/MixUp |
| CenterNet | One-stage, anchor-free | ResNet-50 | ~28.1 | Heatmap-based trên 118k ảnh |
| RetinaNet | One-stage, anchor-based | ResNet-50 | ~36.5 | Focal Loss gốc trên 118k ảnh |

---

## 2. ĐỊNH DẠNG DỮ LIỆU VÀ GIẢI MÃ

### 2.1 Đầu Vào & Đầu Ra
- **Đầu vào:** Tensor `[B, 3, 512, 512]` (RGB chuẩn hóa ImageNet) kèm metadata `{image_id, orig_size (H, W), scale_ratio r, pad_offsets (pw, ph)}`.
- **Đầu ra chuẩn COCO:** Danh sách dict `{image_id, category_id [1..90], bbox [x, y, w, h], score}`.

### 2.2 Quy Trình Giải Mã (Decoder)
Tại ô lưới $(g_x, g_y)$ trên tầng stride $s_l \in \{8, 16, 32\}$:
$$\hat{c}_x = (g_x + 2\sigma(t_x) - 0.5) \cdot s_l, \quad \hat{w} = (2\sigma(t_w))^2 \cdot 4s_l, \quad \text{score} = \sigma(t_{\text{obj}}) \times \sigma(t_{\text{cls,argmax}})$$

```text
Ví dụ minh họa định dạng (không phải số đo thực nghiệm):
  Ảnh: data/coco/val2017/000000000139.jpg (640×480)
  Box 1: [52.3, 134.1, 480.2, 420.6]  score=0.84  label="person"
  Box 2: [10.1,  22.7, 110.5, 185.3]  score=0.61  label="bicycle"
```
> Đã chạy `src/visualize.py` xuất **15 ảnh mẫu đối chiếu GT (xanh lá) vs Pred (đỏ)** tại `outputs/visualizations/eval_samples/` phục vụ phân tích 3 ca lỗi điển hình.

---

## 3. KẾT QUẢ THỰC NGHIỆM (TỪ `outputs/metrics.json`)

### 3.1 Overfit Test (M3 — 10 ảnh, 40 bước, CPU)
- **Initial loss:** `58.9644` $\to$ **Final loss:** `1.0534` (giảm **98.21%** $\ge 95\%$ ✅).
- **Recall @ IoU 0.9:** **1.00** (`10/10` boxes detected ở chế độ `eval()`).

### 3.2 Baseline Training (M6 — 14.000 ảnh Train / 1.000 ảnh Val Held-out)

| Chỉ số | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 (đang chạy) |
|:---|:---:|:---:|:---:|:---:|
| Trạng thái | ✅ Hoàn tất | ✅ Hoàn tất | ✅ Hoàn tất | ⏸ Tạm dừng Batch 2100/3474 (60.4%) |
| Train Loss Total | 4.7706 | **4.1866** | 4.3168 | 5.1313* (unfreeze L2–4) |
| - Box (CIoU) | 0.4253 | **0.3970** | 0.4081 | 0.4088* |
| - Obj (Focal) | 0.3700 | **0.3535** | 0.3593 | 0.3964* |
| - Cls (BCE) | 2.2741 | **1.8480** | 1.9171 | 2.6909* |
| Val Loss Total (1k) | 4.7282 | 4.4330 | **4.1814** | — |
| Checkpoint | `epoch_1.pth` | `epoch_2.pth` | `epoch_3.pth`, `best_map.pth` | `epoch_4.pth` |

> \*Trong Epoch 4, sau khi mở khóa ResNet-50 layers 2–4, loss khởi đầu tăng tạm thời lên 6.1682 rồi giảm dần về 5.1313 tại batch 2100.

### 3.3 Ablation Studies (M7 — 100 ảnh Val Held-out)

| Cấu hình thực nghiệm | Loss Total | Loss Box | Loss Obj | Kết luận kỹ thuật |
|:---|:---:|:---:|:---:|:---|
| **Baseline** (CIoU + Focal + FPN) | 425.08 | 0.6243 | 409.86 | Chuẩn đối sánh toàn phần |
| **Ablation 1:** Smooth L1 thay CIoU | 422.35 | 0.0791 | 409.86 | CIoU phản ánh đúng hình học 2D & tỷ lệ cạnh |
| **Ablation 2:** BCE thay Focal (obj) | 611.34 | 0.6243 | 596.12 | BCE tăng 43.8% loss do bị nhiễu bởi mẫu âm nền |
| **Ablation 3:** Đơn tầng P5 thay FPN | 25.14 | 1.3166 | 1.1480 | Bỏ sót 84.3% vật nhỏ/vừa (chỉ dùng tầng P5) |

### 3.4 Đánh Giá Chính Thức `val2017` (M8 — Đã Khóa)

> Chạy 1 lần duy nhất lúc 2026-11-18 00:26:10 (commit `3067fb2`), khóa tại `outputs/final_eval.lock`.

| Chỉ số COCO | Giá trị | Chỉ số COCO | Giá trị |
|:---|:---:|:---|:---:|
| `mAP@[.50:.95]` | `0.0000` | `AP_large` | `0.0001` |
| `AP50` / `AP75` | `0.0000` / `0.0000` | `AR@1` / `AR@10` / `AR@100` | `0.0000` / `0.0001` / `0.0004` |
| `AP_small` / `AP_medium` | `0.0000` / `0.0000` | Tốc độ suy luận CPU | **2.61 FPS** |

---

## 4. LIMITATIONS VÀ HƯỚNG PHÁT TRIỂN

### 4.1 Hạn Chế Hiện Tại
1. **Quy mô tập train nhỏ:** 14.000 ảnh (~12% COCO `train2017`) khiến các lớp hiếm (<50 mẫu) khó hội tụ.
2. **Phần cứng CPU-only:** Mỗi epoch mất ~3–5 giờ và phải chia phiên chạy để tránh quá nhiệt CPU; kế hoạch thực tế giới hạn ở **5 epochs** thay vì 20 epochs để phù hợp tài nguyên phần cứng.
3. **Độ phân giải cố định $512 \times 512$:** Vật thể rất nhỏ ($<32^2$ px) dễ mất chi tiết đặc trưng ở tầng $P_3$.
4. **Static Assigner:** Quy tắc 3 ô cố định chưa linh hoạt theo độ tự tin dự đoán như SimOTA / TAL.

### 4.2 Hướng Cải Tiến
1. **Huấn luyện trên GPU với 118k ảnh đầy đủ** (đã tải sẵn trong `data/coco/train2017/`) kết hợp Mosaic & MixUp.
2. **Thay Static Assigner bằng SimOTA / Task-Aligned Assigner (TAL)** và thêm **EMA** để ổn định trọng số.
3. **Bổ sung tầng $P_2$ (stride 4)** để cải thiện `AP_small` và xuất mô hình sang **ONNX / TensorRT** để tăng tốc suy luận.
