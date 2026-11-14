# TÀI LIỆU PIPELINE DỮ LIỆU (DATA PIPELINE)

* **Module liên quan:** `src/data/` (`download_data.py`, `category_mapper.py`, `transforms.py`, `dataset.py`)
* **Kiểm thử:** `tests/test_data_split.py`, `test_category_mapper.py`, `test_transforms.py`, `test_dataset.py`

---

## 1. PHÂN CHIA TẬP DỮ LIỆU (ZERO DATA LEAKAGE)

Dữ liệu gốc từ MS COCO 2017 được phân chia theo nguyên tắc phòng chống rò rỉ dữ liệu tuyệt đối:
- **Tập Train Subset (14.000 ảnh):** Trích xuất ngẫu nhiên từ `instances_train2017.json` với seed cố định `seed=42`, lưu tại `data/coco/train_subset14k.json`.
- **Tập Validation Held-out (1.000 ảnh):** Trích xuất độc lập từ `instances_train2017.json` (không giao thoa với tập 14k train: `assert len(train_ids & val_ids) == 0`), lưu tại `data/coco/val_heldout1k.json`. Dùng để theo dõi loss, lưu `best_map.pth` và đo ablation studies.
- **Tập Test Set (5.000 ảnh `val2017`):** Toàn bộ 5.000 ảnh từ `instances_val2017.json`. Tập này **chỉ được đánh giá đúng 1 lần duy nhất ở M8**, không tham gia chọn siêu tham số hay checkpoint.

---

## 2. CATEGORY MAPPING (80 LỚP COCO)

COCO có 80 danh mục đối tượng nhưng dải `category_id` trải dài từ 1 đến 90 (có 10 ID bị khuyết: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83).
- **Lớp liên tục cho nơ-ron:** $\text{class\_idx} \in [0, 79]$.
- **Lớp COCO chuẩn:** $\text{category\_id} \in [1, 90]$.
- Module [`category_mapper.py`](file:///c:/Project/cw_nhom1/cw_Nhom1/src/data/category_mapper.py) cung cấp hàm chuyển đổi 2 chiều:
  - `coco_to_model(cat_id)`: Ánh xạ từ COCO sang index nội bộ huấn luyện.
  - `model_to_coco(cls_idx)`: Ánh xạ từ index nội bộ sang COCO Category ID chuẩn xuất file đánh giá.

---

## 3. TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU (TRANSFORMS)

### 3.1 Letterbox Resizing
- Thay đổi kích thước ảnh về $512 \times 512$ mà vẫn giữ nguyên tỷ lệ khung hình ban đầu.
- Phần dư được bù màu xám chuẩn $(114, 114, 114)$.
- Lưu trữ đầy đủ bộ metadata: tỷ lệ thu phóng $r$, padding ngang/dọc $(\text{pad}_w, \text{pad}_h)$, kích thước gốc $(H_{orig}, W_{orig})$ và `image_id`.
- Sai số khôi phục hộp bao khép kín: $\Delta < 1.0\text{ px}$.

### 3.2 Augmentation (Chỉ áp dụng khi Huấn luyện)
- **Random Horizontal Flip:** Xác suất $p = 0.5$, tự động lật tọa độ hộp bao: $x_1' = W - x_2, x_2' = W - x_1$.
- **Color Jitter:** Biến thiên nhẹ độ sáng, độ tương phản và độ bão hòa màu ($[0.9, 1.1]$).

---

## 4. PYTORCH DATASET VÀ BATCH COLLATING

- `COCODataset`: Đọc ảnh, chuyển đổi format tọa độ từ $[x, y, w, h]$ sang $[x_1, y_1, x_2, y_2]$, chuẩn hóa pixel về dải $[0, 1]$ với mean/std ImageNet.
- `detection_collate_fn`: Ghép các ảnh thành batch tensor $[B, 3, 512, 512]$, đồng thời đóng gói danh sách bounding boxes, labels và metadata theo từng ảnh mà không bị cắt xén.
