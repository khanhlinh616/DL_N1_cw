# TÀI LIỆU QUY TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)

* **Module liên quan:** `src/core/assigner.py`, `src/core/loss.py`, `src/train.py`, `src/utils/logger.py`
* **Kiểm thử:** `tests/test_assigner.py`, `tests/test_loss.py`, `tests/test_overfit.py`

---

## 1. GÁN NHÃN Ô LƯỚI (LABEL ASSIGNER)

Bộ gán nhãn anchor-free mở rộng theo phong cách YOLOv5/v8, chọn tầng và các ô lân cận:
1. **Phân tầng FPN theo kích thước hộp bao ($S = \sqrt{w \times h}$):**
   - $S < 64\text{ px} \implies P_3$ (stride 8, lưới $64 \times 64$)
   - $64 \le S < 168\text{ px} \implies P_4$ (stride 16, lưới $32 \times 32$)
   - $S \ge 168\text{ px} \implies P_5$ (stride 32, lưới $16 \times 16$)
2. **Quy tắc 3 ô lân cận gần tâm nhất:**
   - Ô chính (Primary): tọa độ ô sàn $(i, j) = (\lfloor g_x \rfloor, \lfloor g_y \rfloor)$.
   - Ô lân cận ngang: nếu phần dư $dx < 0.5$ và $i > 0$ thì lấy $i - 1$, ngược lại nếu $dx \ge 0.5$ và $i < W - 1$ thì lấy $i + 1$.
   - Ô lân cận dọc: nếu phần dư $dy < 0.5$ và $j > 0$ thì lấy $j - 1$, ngược lại nếu $dy \ge 0.5$ và $j < H - 1$ thì lấy $j + 1$.
   - Mỗi Ground Truth được gán tối đa 3 ô dương (`pos_mask = True`), tăng số lượng mẫu học hồi quy cho mạng.

---

## 2. HÀM MẤT MÁT ĐA NHIỆM (MULTI-TASK LOSS)

Tổng hàm mất mát gồm 3 thành phần:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{obj}} \mathcal{L}_{\text{Focal\_obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE\_cls}}$$
với trọng số mặc định: $\lambda_{\text{box}} = 5.0, \lambda_{\text{obj}} = 1.0, \lambda_{\text{cls}} = 1.0$.

1. **Complete IoU (CIoU) Loss:**
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
   Tính toán nghiêm ngặt chỉ trên các ô dương (`pos_mask == True`).
2. **Sigmoid Focal Loss cho Objectness:**
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\alpha = 0.25, \gamma = 2.0)$$
   Được đánh giá trên **toàn bộ mọi ô lưới** ở cả 3 tầng $P_3, P_4, P_5$ để giải quyết triệt để mất cân bằng tiền cảnh / hậu cảnh.
3. **BCE Loss cho Phân loại đa lớp:**
   Được tính bằng Binary Cross Entropy with Logits cho 80 lớp chỉ trên các ô dương.

---

## 3. ENGINE HUẤN LUYỆN VÀ TỐI ƯU HÓA (`src/train.py`)

- **Optimizer:** AdamW với learning rate cơ sở $\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$.
- **Learning Rate Scheduler:** Cosine Annealing giảm dần về $\text{lr}_{\text{min}} = 10^{-5}$ trong suốt chu kỳ 20 epochs.
- **Mixed Precision:** Kích hoạt `torch.cuda.amp.autocast()` và `GradScaler` để tiết kiệm $40\%$ VRAM và tăng tốc $2\times$ trên GPU có Tensor Cores.
- **Gradient Clipping:** Cắt ngưỡng gradient `clip_grad_norm_(parameters, max_norm=10.0)` chống tràn số và chống bùng nổ gradient.
- **Quản lý Checkpoint:**
  - Kiểm tra thẩm định định kỳ trên tập `val-heldout 1k` sau mỗi epoch.
  - Lưu checkpoint tốt nhất tại `outputs/checkpoints/best_map.pth`.
  - Lưu checkpoint gần nhất `outputs/checkpoints/last_epoch.pth` hỗ trợ `--resume`.

### Lệnh thực thi:
```bash
# Huấn luyện đầy đủ 20 epochs
python src/train.py --config configs/base_config.yaml --epochs 20

# Kiểm thử nhanh pipeline (Smoke run)
python src/train.py --config configs/base_config.yaml --max-images 8 --epochs 1
```
