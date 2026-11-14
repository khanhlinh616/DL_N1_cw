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
- Kiểm tra kết quả test và nghiệm thu tiêu chí hoàn thành của M0.

### 4. Thành phần do AI tạo ra
- File `.gitignore` và `requirements.txt`.
- Module `src/data/download_data.py` (hàm `split_coco_annotations` và `split_train_val_files`).
- File unit test `tests/test_data_split.py`.
- Các file cấu hình YAML trong `configs/`.

### 5. Phương pháp kiểm chứng (Validation)
- Chạy `pytest tests/test_data_split.py -v`: 4/4 bài test passed trong 0.15s.
- Kiểm tra thuộc tính `len(set(train_ids) & set(val_ids)) == 0` đảm bảo không có rò rỉ dữ liệu.
