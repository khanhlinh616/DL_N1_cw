# NHẬT KÝ KIỂM TOÁN VÀ ĐÁNH GIÁ (AUDIT LOG)

* **Mục đích:** Ghi nhận minh bạch mọi lần chạy đánh giá chính thức (đặc biệt là `val2017`) và lịch sử các mốc cấu hình / huấn luyện thực tế theo RULES.md §17.3.
* **Quy tắc bất biến:** Không xóa hoặc chỉnh sửa sai lệch lịch sử đánh giá đã khóa.

---

## 1. QUY CHẾ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (`val2017`)

1. Phát triển mô hình, chọn checkpoint và chạy ablation studies **chỉ được phép** dùng tập `val-heldout 1k`.
2. Tập `val2017` (5.000 ảnh) chỉ được đánh giá **đúng 1 lần duy nhất** ở Milestone M8.
3. Khi chạy `python -m src.evaluate --dataset val2017 --final`:
   - Hệ thống tự động tạo file khóa `outputs/final_eval.lock`.
   - Tự động ghi 1 bản ghi vào Bảng Nhật ký bên dưới.
   - Mọi lần chạy lại sau khi đã có file khóa đều bị từ chối với mã lỗi `1`.

---

## 2. BẢNG NHẬT KÝ ĐÁNH GIÁ CHÍNH THỨC (AUDIT RECORDS)

| Thời gian | Commit | Checkpoint | Tập dữ liệu | Config | mAP@[.5:.95] | Trạng thái | Ghi chú |
|:---|:---|:---|:---|:---|:---:|:---|:---|
| 2026-11-18 00:26:10 | `3067fb2` | `outputs/checkpoints/best_map.pth` | `val2017` (200/5k) | Baseline | `0.0000` | COMPLETED | Official Milestone M8 Evaluation (đã khóa) |

---

## 3. THỐNG KÊ CHECKPOINT ĐÃ LƯU (`outputs/checkpoints/`)

| Tệp Checkpoint | Thời điểm lưu | Dung lượng | Nội dung / Trạng thái |
|:---|:---|---:|:---|
| `epoch_1.pth` | 2026-11-21 23:19 | ~290 MB | Hoàn thành 100% Epoch 1 (Train 4.7706 / Val 4.7282) |
| `epoch_2.pth` | 2026-11-24 22:43 | ~162 MB | Hoàn thành 100% Epoch 2 (Train 4.1866 / Val 4.4330) |
| `epoch_3.pth` | 2026-11-26 22:16 | ~162 MB | Hoàn thành 100% Epoch 3 (Train 4.3168 / **Val 4.1814**) |
| `best_map.pth` | 2026-11-26 22:16 | ~162 MB | Bản sao trọng số tốt nhất tại Epoch 3 (Val 4.1814) |
| `last_epoch.pth` | 2026-11-26 22:16 | ~162 MB | Checkpoint epoch hoàn chỉnh gần nhất (Epoch 3) |
| `epoch_4.pth` | 2026-11-27 14:46 | ~339 MB | Tạm dừng Epoch 4 tại Batch 2100/3474 (60.4%, avg loss 5.1313) |

---

## 4. LỊCH SỬ CÁC MỐC PHÁT TRIỂN VÀ PHIÊN HUẤN LUYỆN

- **2026-11-14 (M0–M2):** Chia cố định 14k train / 1k held-out (`seed=42`, overlap = 0); cài đặt Letterbox 512 ($\Delta < 1\text{ px}$); hoàn thiện ResNet-50 FrozenBN + FPN + Decoupled Head ($b \approx -4.595$).
- **2026-11-15 (M3–M5):** Cài đặt Assigner 3 ô + MultiTaskLoss (Overfit 10 ảnh giảm 98.21%, Recall 1.0); Training Engine + smoke run; Decoder + Batched NMS + pilot 100 ảnh held-out (8.289 predictions); tải đủ 118.287 ảnh `train2017`.
- **2026-11-16 (M6 & Sync):** Bắt đầu train baseline 14k trên CPU; hợp nhất nhánh lịch sử DL Team 1 (`952fd34`), gắn tag M0–M6 và đồng bộ nhánh `main`.
- **2026-11-18 (M7–M8):** Chạy 3 bài Ablation trên `val-heldout 1k` (CIoU, Focal, FPN); chạy đánh giá chính thức duy nhất trên `val2017` và khóa `outputs/final_eval.lock`.
- **2026-11-21 (Epoch 1):** Hoàn tất 100% Epoch 1 (Train 4.7706, Val 4.7282), lưu `epoch_1.pth`.
- **2026-11-24 (Epoch 2):** Hoàn tất 100% Epoch 2 (Train 4.1866, Val 4.4330), lưu `epoch_2.pth`.
- **2026-11-26 (Epoch 3):** Hoàn tất 100% Epoch 3 (Train 4.3168, Val **4.1814** — best val loss), cập nhật `best_map.pth`.
- **2026-11-27 (Epoch 4 — phiên 1 & 2):** Mở khóa ResNet-50 layers 2–4; chạy tới Batch 2100/3474 (60.4%), avg loss giảm từ 6.1682 xuống 5.1313, dừng theo lịch lúc 14h45 để nghỉ CPU, lưu `epoch_4.pth`.
