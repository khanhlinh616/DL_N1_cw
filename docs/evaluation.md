# TÀI LIỆU ĐÁNH GIÁ VÀ GIẢI MÃ DỰ ĐOÁN (EVALUATION & INFERENCE)

* **Module liên quan:** `src/core/decoder.py`, `src/evaluate.py`, `src/utils/box_ops.py`
* **Kiểm thử:** `tests/test_decoder.py`

---

## 1. BỘ GIẢI MÃ TỌA ĐỘ VÀ NMS (INFERENCE DECODER)

Nhận đầu vào là tensor logits thô từ Detection Head tại 3 tầng FPN ($P_3, P_4, P_5$) và metadata ảnh:
1. **Công thức giải mã tâm và kích thước hộp (theo phong cách YOLOv5/v8):**
   - Tọa độ tâm:
     $$\hat{c}_x = \left(\text{grid}_x + 2\sigma(t_x) - 0.5\right) \times s_l \quad (\text{miền offset: } [-0.5, 1.5] \times s_l)$$
     $$\hat{c}_y = \left(\text{grid}_y + 2\sigma(t_y) - 0.5\right) \times s_l$$
   - Chiều rộng và chiều cao:
     $$\hat{w} = \left(2\sigma(t_w)\right)^2 \times 4 \times s_l \quad (\text{miền kích thước: } [0, 16 \times s_l])$$
     $$\hat{h} = \left(2\sigma(t_h)\right)^2 \times 4 \times s_l$$
2. **Tính điểm tin cậy tổng hợp:**
   $$\text{score} = \sigma(t_{\text{obj}}) \times \sigma(t_{\text{cls}}) \in [0, 1]$$
   Lọc trước các ứng viên có $\text{score} < 0.05$ để giảm tải tính toán cho NMS.
3. **Batched Non-Maximum Suppression (Batched NMS):**
   - Thực hiện NMS độc lập theo từng lớp đối tượng bằng cách dịch chuyển tọa độ theo lớp: $b' = b + 4096 \times c$.
   - Ngưỡng $\text{IoU}_{\text{threshold}} = 0.5$, lấy tối đa $K = 100$ bounding boxes có điểm tin cậy cao nhất mỗi ảnh.
4. **Khôi phục tọa độ về không gian ảnh gốc:**
   - Trừ khoảng đệm: $x = x_{\text{letterbox}} - \text{pad}_w, \quad y = y_{\text{letterbox}} - \text{pad}_h$.
   - Chia tỷ lệ: $x_{\text{orig}} = x / r, \quad y_{\text{orig}} = y / r$.
   - Cắt tọa độ trong phạm vi $[0, W_{\text{orig}}]$ và $[0, H_{\text{orig}}]$.

---

## 2. QUY CHUẨN ĐÁNH GIÁ COCO (COCO EVALUATION)

### 2.1 Thẩm định trên tập Held-out (1.000 ảnh)
- Được thực hiện định kỳ trong quá trình huấn luyện sau mỗi epoch.
- Xuất file dự đoán `outputs/evaluation_results/predictions_val_heldout.json`.
- Chạy `COCOeval` để lấy `mAP@[.5:.95]`, dùng để quyết định lưu trọng số tốt nhất `best_map.pth`.

### 2.2 Đánh giá chính thức trên `val2017` (5.000 ảnh) — Quy tắc bất biến (§17.3)
- **Tập `val2017` chỉ được đánh giá đúng 1 lần duy nhất ở Milestone M8.**
- Script [`evaluate.py`](file:///d:/cw/src/evaluate.py) bắt buộc phải có cờ `--final`. Nếu chạy trên `val2017` mà thiếu cờ này, chương trình sẽ tự động từ chối thực thi.
- Khi cờ `--final` được kích hoạt:
  1. Ghi file khóa an toàn `outputs/final_eval.lock`.
  2. Ghi một dòng nhật ký vào [`docs/audit-log.md`](file:///d:/cw/docs/audit-log.md) lưu trữ: thời gian chạy, commit hash, config, điểm số mAP.
  3. Nếu file lock đã tồn tại, chương trình sẽ ngăn chặn mọi lần chạy lại tiếp theo để loại trừ nguy cơ tinh chỉnh tham số theo tập test (Data Leakage).

### Lệnh thực thi:
```bash
set PYTHONPATH=.

# Đánh giá trên tập validation held-out 1k
python -m src.evaluate --checkpoint outputs/checkpoints/best_map.pth --dataset heldout

# Đánh giá chính thức 1 lần trên val2017 (M8 — đã thực hiện và khóa an toàn)
python -m src.evaluate --checkpoint outputs/checkpoints/best_map.pth --dataset val2017 --final
```
