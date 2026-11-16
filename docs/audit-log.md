# NHẬT KÝ KIỂM TOÁN VÀ ĐÁNH GIÁ (AUDIT LOG)

* **Mục đích:** Ghi nhận minh bạch mọi lần chạy đánh giá chính thức, đặc biệt là bài kiểm tra độc lập duy nhất trên tập `val2017` (5.000 ảnh) theo quy chuẩn chống rò rỉ dữ liệu (RULES.md §17.3).
* **Quy tắc bất biến:** Không xóa, không sửa tay hoặc thay đổi lịch sử đánh giá.

---

## 1. QUY CHẾ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (`val2017`)

1. Quá trình phát triển mô hình, điều chỉnh siêu tham số và ablation studies **chỉ được phép** sử dụng tập `val-heldout 1k`.
2. Tập `val2017` (5k ảnh) chỉ được mở khóa để chạy đánh giá chính thức **đúng 1 lần duy nhất** khi toàn bộ quá trình huấn luyện và tối ưu đã hoàn tất tại Milestone M8.
3. Khi thực hiện lệnh đánh giá `python src/evaluate.py --dataset val2017 --final`:
   - Hệ thống tự động tạo file khóa `outputs/final_eval.lock`.
   - Một bản ghi chi tiết sẽ được ghi thêm vào Bảng Nhật ký bên dưới.
   - Nếu file `final_eval.lock` đã tồn tại, chương trình sẽ lập tức báo lỗi và từ chối chạy lại.

---

## 2. BẢNG NHẬT KÝ THỰC THI (AUDIT RECORDS)

| Thời gian (Timestamp) | Commit Hash | Checkpoint Path | Tập dữ liệu | Config | mAP@[.5:.95] | Trạng thái | Người thực thi / Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Chưa ghi nhận* | - | - | - | - | - | PENDING | Chờ hoàn thành Milestone M8 |

---

## 3. LỊCH SỬ THAY ĐỔI CẤU HÌNH QUAN TRỌNG

- **2026-11-14 (M0):** Phân chia cố định 14.000 train và 1.000 held-out từ `train2017` bằng seed 42, kiểm tra giao thoa = 0.
- **2026-11-14 (M1):** Cố định hệ tọa độ Letterbox $512 \times 512$, độ lệch khép kín $< 1.0\text{ px}$.
- **2026-11-14 (M2):** Cố định kiến trúc ResNet-50-FPN + Decoupled Head, khởi tạo Focal bias $b \approx -4.595$.
- **2026-11-15 (M3):** Assigner 3 ô dương + MultiTaskLoss (CIoU + Focal + BCE); Overfit test 10 ảnh giảm loss 98.21%, Recall 1.0.
- **2026-11-15 (M4):** Xây dựng Training Engine (AdamW, Cosine LR, AMP, Clip 10.0), chạy smoke run 1 epoch, lưu checkpoint.
- **2026-11-15 (M5):** Triển khai Decoder tọa độ, Batched NMS, tích hợp COCOeval, pilot test 100 ảnh held-out (8.289 predictions).
- **2026-11-15 (Data):** Tải và giải nén hoàn tất 100% 118.287 ảnh `train2017` bằng `tar.exe`, xác thực 0 ảnh thiếu.
- **2026-11-16 (M6):** Khởi chạy Baseline trên 14k ảnh train; đạt ~13.3k ảnh (75 phút) và dừng an toàn theo yêu cầu người dùng do CPU quá nhiệt.
- **2026-11-16 (Merge & Sync):** Hợp nhất (merge) lịch sử gốc (commit 952fd34 từ DL Team 1) và gắn lại chính xác toàn bộ thẻ tags M0–M6 theo từng commit nguyên bản (`8a60937`, `ee9d166`, `86d3c17`, `952fd34`, `70f6d9c`, `18cc511`, `93d9134`). Đồng bộ duy nhất 1 nhánh `main` lên GitHub `khanhlinh616/DL_N1_cw`.
