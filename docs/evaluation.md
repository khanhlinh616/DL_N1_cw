# TÀI LIỆU ĐÁNH GIÁ VÀ GIẢI MÃ DỰ ĐOÁN (EVALUATION & INFERENCE)

* **Module liên quan:** [`src/core/decoder.py`](file:///d:/cw/src/core/decoder.py), [`src/evaluate.py`](file:///d:/cw/src/evaluate.py), [`src/utils/box_ops.py`](file:///d:/cw/src/utils/box_ops.py)
* **Kiểm thử:** `tests/test_decoder.py` (6/6 passed), `tests/test_evaluate.py` (3/3 passed)

---

## 1. BỘ GIẢI MÃ TỌA ĐỘ VÀ NMS (`src/core/decoder.py`)

Nhận logits thô từ 3 tầng FPN ($P_3, P_4, P_5$ — tổng 5.376 ô lưới) và metadata Letterbox:
1. **Giải mã tâm và kích thước hộp (tầng stride $s_l$):**
   $$\hat{c}_x = \left(\text{grid}_x + 2\sigma(t_x) - 0.5\right) \cdot s_l, \quad \hat{c}_y = \left(\text{grid}_y + 2\sigma(t_y) - 0.5\right) \cdot s_l$$
   $$\hat{w} = \left(2\sigma(t_w)\right)^2 \cdot 4 s_l, \quad \hat{h} = \left(2\sigma(t_h)\right)^2 \cdot 4 s_l$$
   *(Miền offset tâm $[-0.5, 1.5] \cdot s_l$; miền kích thước $[0, 16 s_l]$: $P_3 \le 128$, $P_4 \le 256$, $P_5 \le 512$ px.)*
2. **Điểm tin cậy tổng hợp và lọc ngưỡng:**
   $$\text{score} = \sigma(t_{\text{obj}}) \times \sigma(t_{\text{cls}}) \in [0, 1]$$
   Loại bỏ ứng viên có $\text{score} < 0.05$ trước khi chạy NMS.
3. **Batched NMS theo từng lớp:**
   - Dịch chuyển tọa độ theo lớp: $b' = b + 4096 \times c$ để các lớp khác nhau không triệt tiêu lẫn nhau.
   - Ngưỡng $\text{IoU}_{\text{threshold}} = 0.60$, giữ tối đa $K = 100$ hộp có điểm cao nhất mỗi ảnh.
4. **Khôi phục tọa độ ảnh gốc & ánh xạ lớp:**
   - $x_{\text{orig}} = (x_{\text{letterbox}} - \text{pad}_w) / r$, $y_{\text{orig}} = (y_{\text{letterbox}} - \text{pad}_h) / r$, cắt vào $[0, W_{\text{orig}}]$ và $[0, H_{\text{orig}}]$.
   - Chuyển `class_idx` $[0, 79] \to$ `category_id` $[1, 90]$ qua `model_to_coco()`.

---

## 2. GIẢI THÍCH CÁC CHỈ SỐ COCO (`COCOeval`)

| Chỉ số | Ý nghĩa |
|:---|:---|
| **`mAP@[.50:.95]`** | Trung bình AP trên 80 lớp và 10 ngưỡng IoU từ $0.50$ đến $0.95$ (bước $0.05$) — chỉ số chuẩn chính |
| **`AP50` / `AP75`** | AP tại ngưỡng IoU $\ge 0.50$ (dễ) và $\ge 0.75$ (đòi hỏi khớp khung chặt) |
| **`AP_small` / `AP_medium` / `AP_large`** | AP phân theo diện tích vật: $< 32^2$ px, $32^2\text{–}96^2$ px, $> 96^2$ px |
| **`AR@1` / `AR@10` / `AR@100`** | Average Recall khi giới hạn tối đa 1, 10 hoặc 100 dự đoán mỗi ảnh |

---

## 3. KIỂM THỬ PILOT 100 ẢNH (M5) & ĐÁNH GIÁ CHÍNH THỨC M8 (`val2017`)

- **Pilot Smoke Test M5 (`outputs/metrics.json` → `pilot_eval_m5`):** Chạy `evaluate.py --dataset heldout --max-images 100` trước huấn luyện M6 → xuất **8.289 predictions** hợp lệ chuẩn COCO, `mAP@[.50:.95] = 0.0` (đúng kỳ vọng khi chưa train). Bản chạy kiểm chứng độc lập lưu tại `outputs/evaluation_results/m5_smoke_100/`.
- **Đánh giá chính thức M8 (`val2017`):** Thực hiện 1 lần duy nhất (2026-11-18 00:26:10, commit `3067fb2`, 200 ảnh / 20.000 predictions):

| Chỉ số | Giá trị | Chỉ số | Giá trị |
|:---|:---:|:---|:---:|
| `mAP@[.50:.95]` | `0.0000` | `AP_large` | `0.0001` |
| `AP50` / `AP75` | `0.0000` / `0.0000` | `AR@1` / `AR@10` / `AR@100` | `0.0000` / `0.0001` / `0.0004` |
| `AP_small` / `AP_medium` | `0.0000` / `0.0000` | Tốc độ suy luận CPU | **2.61 FPS** |

> **Phân tích nguyên nhân mAP thấp ở thời điểm M8:** Bản đánh giá M8 chạy khi checkpoint mới chỉ hoàn tất phần đầu giai đoạn warmup trên tập con 14k (chưa hội tụ sâu qua các epoch sau như Epoch 3), nên chỉ mới bắt đầu xuất hiện tín hiệu ở vật thể lớn (`AP_large = 0.0001`).

---

## 4. QUY TRÌNH CHỐNG RÒ RỈ DỮ LIỆU & LỆNH THỰC THI

- Tập `val-heldout 1k` dùng cho đánh giá nội bộ trong quá trình train/ablation. Xem ma trận & biểu đồ chi tiết qua từng epoch tại [`docs/epoch-matrix.md`](file:///d:/cw/docs/epoch-matrix.md).
- Tập `val2017` bắt buộc cờ `--final`. Sau khi chạy M8, hệ thống đã tạo file khóa `outputs/final_eval.lock` và ghi nhận vào [`docs/audit-log.md`](file:///d:/cw/docs/audit-log.md) — **tuyệt đối không chạy lại** để giữ tính khách quan.

```bash
set PYTHONPATH=.

# Đánh giá nội bộ trên tập validation held-out 1k
python -m src.evaluate --checkpoint outputs/checkpoints/best_map.pth --dataset heldout

# Đánh giá chính thức M8 (đã khóa an toàn tại outputs/final_eval.lock)
python -m src.evaluate --checkpoint outputs/checkpoints/best_map.pth --dataset val2017 --final
```
