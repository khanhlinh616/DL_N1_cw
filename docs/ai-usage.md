# NHẬT KÝ SỬ DỤNG AI (AI USAGE LOG)

* **Quy tắc tuân thủ:** Ghi nhận minh bạch mục đích prompt, công cụ sử dụng, đóng góp của sinh viên, thành phần AI sinh ra và phương pháp kiểm chứng (theo RULES.md §3 & §17.11).

---

## MILESTONE M0: SETUP & DATA SPLIT

### 1. Công cụ sử dụng
- Antigravity AI Agent (Deep Learning Engineer role).

### 2. Mục đích prompt
- Thiết lập cấu trúc repo ban đầu: `.gitignore`, `requirements.txt`, các file cấu hình `configs/`.
- Cài đặt script tải và tách dữ liệu `src/data/download_data.py` tuân thủ nguyên tắc chống rò rỉ (zero data leakage) với `assert` không giao nhau và seed cố định (`seed=42`).
- Viết bộ unit test `tests/test_data_split.py` kiểm chứng thuật toán chia tập dữ liệu.

### 3. Đóng góp của sinh viên
- Xác lập thiết kế kiến trúc chuẩn tại `docs/DEFINE_AND_DESIGN.md`.
- Duyệt và phê duyệt phương án tải dữ liệu zip đầy đủ.
- Cung cấp link remote repository GitHub và yêu cầu push đồng bộ.

### 4. Thành phần do AI tạo ra
- File `.gitignore` và `requirements.txt`.
- Module `src/data/download_data.py` (hàm `split_coco_annotations` và `split_train_val_files`).
- File unit test `tests/test_data_split.py` và script xác thực `tests/verify_split.py`.
- Các file cấu hình YAML trong `configs/`.

### 5. Phương pháp kiểm chứng (Validation)
- Chạy `pytest tests/test_data_split.py -v`: 4/4 bài test passed trong 0.15s.
- Kiểm tra trực tiếp trên 14k train và 1k held-out: `Overlap train & heldout: 0`.

---

## MILESTONE M1: DATASET & METADATA PIPELINE

### 1. Công cụ sử dụng
- Antigravity AI Agent (Deep Learning Engineer role).

### 2. Mục đích prompt
- Xây dựng bảng ánh xạ danh mục hai chiều 80 lớp chuẩn COCO 2017 (`category_id` 1–90 $\leftrightarrow$ `class_idx` 0–79).
- Xây dựng pipeline tiền xử lý và tăng cường dữ liệu `transforms.py`: Letterbox $512 \times 512$, đóng gói metadata ($r, \text{pad}_w, \text{pad}_h$), thuật toán khôi phục hộp bao về ảnh gốc, Random Horizontal Flip, Color Jitter.
- Cài đặt `COCODataset` và `detection_collate_fn` trả về batch tensor $[B, 3, 512, 512]$ kèm đầy đủ metadata.

### 3. Đóng góp của sinh viên
- Yêu cầu kiểm tra chặt chẽ sai số tọa độ khi letterbox và khôi phục về ảnh gốc ($< 1.0\text{ px}$).
- Yêu cầu commit và push liên tục lên remote GitHub.

### 4. Thành phần do AI tạo ra
- `src/data/category_mapper.py` (Class `CategoryMapper` và hàm ánh xạ tiện ích).
- `src/data/transforms.py` (`letterbox_image`, `letterbox_boxes`, `inverse_letterbox_boxes`, `random_horizontal_flip`, `DetectionTransforms`).
- `src/data/dataset.py` (`COCODataset`, `detection_collate_fn`).
- Bộ unit test `tests/test_category_mapper.py`, `tests/test_transforms.py`, `tests/test_dataset.py`.

### 5. Phương pháp kiểm chứng (Validation)
- `pytest tests/test_category_mapper.py`: 5/5 passed. Ánh xạ 2 chiều 80 lớp đạt 100%; 10 ID khuyết đều bị từ chối chính xác.
- `pytest tests/test_transforms.py`: 10/10 passed. Sai số khôi phục tọa độ hộp bao qua letterbox $< 1.0\text{ px}$ trên 7 tỷ lệ khung hình khác nhau.
- `pytest tests/test_dataset.py`: 3/3 passed. Shape batch $[B, 3, 512, 512]$ được kiểm chứng trên cả dữ liệu giả lập và 5.000 ảnh thật `val2017`.
- Tổng cộng toàn bộ test suite: 22/22 unit tests passed.
