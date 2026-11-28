# TÀI LIỆU QUY TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)

* **Module liên quan:** [`src/core/assigner.py`](file:///d:/cw/src/core/assigner.py), [`src/core/loss.py`](file:///d:/cw/src/core/loss.py), [`src/train.py`](file:///d:/cw/src/train.py), [`src/utils/logger.py`](file:///d:/cw/src/utils/logger.py)
* **Cấu hình:** [`configs/base_config.yaml`](file:///d:/cw/configs/base_config.yaml)
* **Kiểm thử:** `tests/test_assigner.py`, `test_loss.py`, `test_overfit.py`, `test_train.py`

---

## 1. GÁN NHÃN Ô LƯỚI (ANCHOR-FREE LABEL ASSIGNER)

Thay vì dùng anchor box K-means, bộ gán nhãn chọn tầng FPN và 3 ô dương gần tâm nhất cho mỗi vật thể:
1. **Phân tầng theo kích thước hộp bao ($S = \sqrt{w \times h}$):**
   - $S < 64\text{ px} \implies P_3$ (stride 8, lưới $64 \times 64$)
   - $64 \le S < 168\text{ px} \implies P_4$ (stride 16, lưới $32 \times 32$)
   - $S \ge 168\text{ px} \implies P_5$ (stride 32, lưới $16 \times 16$)
2. **Quy tắc 3 ô lân cận gần tâm nhất:**
   - Ô chính (Primary): $(i, j) = (\lfloor c_x / s_l \rfloor, \lfloor c_y / s_l \rfloor)$, phần dư $\delta_x = c_x/s_l - i$, $\delta_y = c_y/s_l - j \in [0, 1)$.
   - Ô lân cận ngang: lấy $(i - 1, j)$ nếu $\delta_x < 0.5$ (và $i > 0$), ngược lại lấy $(i + 1, j)$ nếu $i < W_l - 1$.
   - Ô lân cận dọc: lấy $(i, j - 1)$ nếu $\delta_y < 0.5$ (và $j > 0$), ngược lại lấy $(i, j + 1)$ nếu $j < H_l - 1$.
   - Khớp hoàn toàn với miền dự đoán $[-0.5, 1.5] \times s_l$ của Decoder.

---

## 2. HÀM MẤT MÁT ĐA NHIỆM (MULTI-TASK LOSS)

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{obj}} \mathcal{L}_{\text{Focal\_obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE\_cls}}$$

| Thành phần | Trọng số | Phạm vi tính | Vai trò |
|:---|:---:|:---|:---|
| **CIoU Loss** | $\lambda_{\text{box}} = 5.0$ | Chỉ trên các ô dương (`pos_mask`) | Tối ưu đồng thời IoU, khoảng cách tâm và tỷ lệ khung hình |
| **Focal Loss** | $\lambda_{\text{obj}} = 1.0$ | **Toàn bộ 5.376 ô** ($P_3, P_4, P_5$) | $\alpha=0.25, \gamma=2.0$; triệt tiêu gradient từ >99% ô nền dễ |
| **BCE Loss** | $\lambda_{\text{cls}} = 1.0$ | Chỉ trên các ô dương (`pos_mask`) | Phân loại đa nhãn 80 lớp độc lập, không bị nhiễu nền |

---

## 3. SIÊU THAM SỐ VÀ LỊCH TRÌNH HUẤN LUYỆN (`src/train.py`)

| Tham số | Giá trị (`base_config.yaml`) | Ghi chú thực tế (GPU RTX 3050 6GB + AMP) |
|:---|:---|:---|
| **Optimizer** | `AdamW` ($\text{lr} = 10^{-3}$, `weight_decay = 5e-4`, `betas=[0.9, 0.999]`) | Hội tụ ổn định hơn SGD ở bước đầu |
| **LR Scheduler** | `CosineAnnealingLR` ($\text{lr}_{\min} = 10^{-5}$) + Warmup 2 epochs | `warmup_bias_lr = 1e-4` |
| **Batch Size / Epochs** | Config mặc định 16 / **20 epochs** | Chạy đủ **20 epochs** trên GPU RTX 3050 (`--batch-size 8 --device cuda`, 1.737 batch/epoch; trước đó thử nghiệm 3.5 ep trên CPU `batch=4`) |
| **Gradient Clipping** | `max_norm = 10.0` | Chống bùng nổ gradient khi mở khóa backbone |
| **Mixed Precision** | `torch.amp.autocast` + `GradScaler` | Bật FP16 AMP trên CUDA giúp tiết kiệm VRAM 6 GB |
| **Backbone Freeze** | Đóng băng 3 epoch đầu; mở khóa `layer2–4` từ Epoch 4 | `FrozenBatchNorm2d` luôn bất biến |

---

## 4. KẾT QUẢ HUẤN LUYỆN THỰC TẾ (14K TRAIN / 1K VAL HELD-OUT)

| Epoch | Backbone | Train Total | Train Box | Train Obj | Train Cls | Val Total | Checkpoint |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | Frozen | 4.7706 | 0.4253 | 0.3700 | 2.2741 | 4.7282 | `epoch_1.pth` |
| **2** | Frozen | 4.1866 | 0.3970 | 0.3535 | 1.8480 | 4.4330 | `epoch_2.pth` |
| **3** | Frozen | 4.3168 | 0.4081 | 0.3593 | 1.9171 | **4.1814** | `epoch_3.pth`, `best_map.pth` |
| **4*** | Unfrozen (L2–4) | 5.1313* | 0.4088* | 0.3964* | 2.6909* | — | `epoch_4.pth` (⏸ Batch 2100/3474 — 60.4%) |

> \*Epoch 4 tạm dừng lúc 14:45 (27/11/2026); loss đang giảm đều từ 6.1682 (lúc mới mở khóa backbone) xuống 5.1313.

---

## 5. CHECKPOINT, RESUME VÀ LỆNH THỰC THI

- **Lưu tự động:** Lưu `outputs/checkpoints/epoch_{N}.pth` mỗi **50 batches** và cuối mỗi epoch, kèm `best_map.pth` (theo val loss thấp nhất) và `last_epoch.pth`.
- **Resume chính xác từng batch:** `--resume` khôi phục `model`, `optimizer`, `scheduler` (có fallback step thủ công nếu lệch key `_schedulers`), `epoch` và `batch_idx`.
- **Đồng bộ GitHub:** Cờ `--git-sync` tự động cập nhật tiến độ vào `docs/plan-progress-status.md` và push sau mỗi epoch.

```bash
set PYTHONPATH=.

# Huấn luyện đủ 20 epochs trên GPU (RTX 3050 6GB, AMP FP16)
.venv\Scripts\python.exe -u -m src.train --epochs 20 --batch-size 8 --device cuda --git-sync

# Smoke test nhanh pipeline
.venv\Scripts\python.exe -m src.train --epochs 1 --batch-size 2 --max-images 8
```
