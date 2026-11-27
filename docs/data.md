# TÀI LIỆU PIPELINE DỮ LIỆU (DATA PIPELINE)

* **Module liên quan:** [`src/data/download_data.py`](file:///d:/cw/src/data/download_data.py), [`src/data/category_mapper.py`](file:///d:/cw/src/data/category_mapper.py), [`src/data/transforms.py`](file:///d:/cw/src/data/transforms.py), [`src/data/dataset.py`](file:///d:/cw/src/data/dataset.py)
* **Cấu hình:** [`configs/dataset_coco_subset.yaml`](file:///d:/cw/configs/dataset_coco_subset.yaml)
* **Kiểm thử:** `tests/test_data_split.py`, `test_category_mapper.py`, `test_transforms.py`, `test_dataset.py`

---

## 1. PHÂN CHIA TẬP DỮ LIỆU (ZERO DATA LEAKAGE)

Dữ liệu MS COCO 2017 được chia cố định với `seed=42`, bảo đảm không chồng lấn (`len(train_ids & val_ids) == 0`):

| Tập dữ liệu | Tệp Annotation | Nguồn ảnh | Số ảnh | Số Annotations | Mục đích sử dụng |
|:---|:---|:---|---:|---:|:---|
| **Train Subset** | `train_subset14k.json` | `train2017/` | 14.000 | 102.736 | Huấn luyện mô hình (M6) |
| **Val Held-out** | `val_heldout1k.json` | `train2017/` | 1.000 | 6.903 | Theo dõi val loss, lưu `best_map.pth`, Ablation (M7) |
| **Test Set** | `instances_val2017.json` | `val2017/` | 5.000 | — | Đánh giá chính thức **1 lần duy nhất** ở M8 |

**Top 5 lớp phổ biến nhất trong tập Train 14k (và Val 1k):**
- `person`: 31.766 (val: 2.130) · `car`: 5.139 (val: 390) · `chair`: 4.573 (val: 315) · `bottle`: 2.878 (val: 169) · `book`: 2.734 (val: 165).

---

## 2. CATEGORY MAPPING (80 LỚP COCO)

COCO 2017 có 80 lớp đối tượng nhưng chỉ số `category_id` trải từ 1 đến 90 do khuyết 10 ID lịch sử:
`[12, 26, 29, 30, 45, 66, 68, 69, 71, 83]`.

Nếu đưa trực tiếp `category_id` vào đầu ra 80 lớp sẽ bị lệch chỉ số hoặc lỗi `IndexError`. Module [`category_mapper.py`](file:///d:/cw/src/data/category_mapper.py) ánh xạ 2 chiều chính xác:
- `coco_to_model(cat_id)`: Chuyển `category_id` $\in [1, 90]$ $\to$ `class_idx` $\in [0, 79]$ khi nạp dữ liệu huấn luyện.
- `model_to_coco(cls_idx)`: Chuyển `class_idx` $\in [0, 79]$ $\to$ `category_id` $\in [1, 90]$ khi xuất JSON cho `COCOeval`.

---

## 3. TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU (`transforms.py`)

### 3.1 Letterbox Resizing ($512 \times 512$)
Thay vì kéo dãn (resize trực tiếp làm méo vật thể), Letterbox giữ nguyên tỷ lệ khung hình và đệm viền xám $(114, 114, 114)$:
1. Tính tỷ lệ co giãn: $r = \min(512 / H_{\text{orig}}, 512 / W_{\text{orig}})$.
2. Kích thước mới: $W_{\text{new}} = \text{round}(W_{\text{orig}} \cdot r)$, $H_{\text{new}} = \text{round}(H_{\text{orig}} \cdot r)$.
3. Khoảng đệm hai bên: $\text{pad}_w = (512 - W_{\text{new}}) / 2$, $\text{pad}_h = (512 - H_{\text{new}}) / 2$.
4. Biến đổi hộp bao: $x' = x \cdot r + \text{pad}_w$, $y' = y \cdot r + \text{pad}_h$.

> **Ví dụ ảnh $640 \times 480$:** $r = \min(512/480, 512/640) = 0.8$ $\implies$ ảnh mới $512 \times 384$, $\text{pad}_w = 0$, $\text{pad}_h = (512 - 384)/2 = 64\text{ px}$. Sai số khôi phục hộp bao khép kín $\Delta < 1.0\text{ px}$.

### 3.2 Augmentation (Chỉ áp dụng khi `is_train=True`)
| Phép biến đổi | Tham số | Xử lý hộp bao đi kèm |
|:---|:---|:---|
| **Random Horizontal Flip** | Xác suất $p = 0.5$ | Lật ngang tọa độ: $x_1' = 512 - x_2$, $x_2' = 512 - x_1$ |
| **Color Jitter** | Brightness / Contrast / Saturation $\in [0.9, 1.1]$ | Giữ nguyên tọa độ hộp bao |

---

## 4. PYTORCH DATASET VÀ BATCH COLLATING (`dataset.py`)

- **`COCODataset`:**
  - Đọc ảnh RGB, loại bỏ annotation `iscrowd == 1` hoặc hộp có chiều rộng/cao $\le 0$.
  - Chuyển đổi hộp từ `[x, y, w, h]` (COCO) sang `[x1, y1, x2, y2]` trên lưới $512 \times 512$.
  - Chuẩn hóa pixel về $[0, 1]$ rồi chuẩn hóa theo ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - Trả về kèm metadata: `image_id`, `orig_size (H, W)`, `scale_ratio` ($r$), `pad_offsets` $(\text{pad}_w, \text{pad}_h)$.
- **`detection_collate_fn`:** Ghép ảnh thành tensor `[B, 3, 512, 512]`, giữ `targets` dưới dạng list các dict theo từng ảnh (vì số vật thể mỗi ảnh khác nhau).
