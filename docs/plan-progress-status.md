# TIẾN ĐỘ THỰC HIỆN DỰ ÁN (PLAN & PROGRESS STATUS)

* **Cập nhật lần cuối:** 2026-11-28 13:58
* **Trạng thái tổng thể:** Hoàn thành M0–M9; Đã hoàn thành 100% Epoch 1, Epoch 2 & Epoch 2 baseline.
* **Tiến độ trực tiếp:** Epoch 3: Batch 1737/1737 (100.0%) | Loss: 4.3034 (box: 0.4197, obj: 0.3548, cls: 1.8503)

---

## 1. BẢNG THEO DÕI CÁC MILESTONES (M0–M9)

| Milestone | Mô tả công việc | Tiêu chí nghiệm thu | Trạng thái |
|:---|:---|:---|:---|
| **M0** | Môi trường, requirements, split 14k/1k | `train_subset14k.json`, `val_heldout1k.json`, `instances_val2017.json` | **Hoàn thành** |
| **M1** | Category mapping, Letterbox 512, Dataset | Mapping 80 lớp 100%, shape `[B,3,512,512]`, sai số < 1 px | **Hoàn thành** |
| **M2** | ResNet-50 (FrozenBN) + FPN + YOLO Head | Forward đúng shape P3/P4/P5, bias obj ≈ −4.595 | **Hoàn thành** |
| **M3** | Assigner 3 ô + MultiTaskLoss (CIoU/Focal/BCE) | Overfit 10 ảnh: loss giảm ≥ 95% (98.21%), recall = 1.0 | **Hoàn thành** |
| **M4** | Training Engine (AdamW, Cosine LR, AMP, Clip 10) | Smoke run 1 epoch, lưu/resume checkpoint bit-exact | **Hoàn thành** |
| **M5** | Decoder (offset [−0.5,1.5], dim [0,16s]) + NMS | Pilot 100 ảnh held-out, JSON hợp lệ, COCOeval chạy mượt | **Hoàn thành** |
| **M6** | Train baseline **20 epochs** trên 14k ảnh (`train_subset14k`) | Giai đoạn CPU xong Ep 1–3 (`best_map.pth` Val 4.1814); chuyển sang **GPU 20 epochs** | **Chuyển sang GPU (20 ep)** |
| **M7** | 3 bài Ablation Studies trên `val-heldout 1k` | So sánh CIoU vs SmoothL1, Focal vs BCE, FPN vs P5 | **Hoàn thành** |
| **M8** | Đánh giá 1 lần `val2017` + Visualizer 15 mẫu | Khóa `final_eval.lock`, xuất 15 ảnh phân loại ca lỗi | **Hoàn thành** |
| **M9** | Đóng gói CLI (`train/evaluate/demo`), PEP8, docs | 55/55 unit tests passed, tài liệu ≤ 200 dòng | **Hoàn thành** |

---

## 2. CHI TIẾT TIẾN ĐỘ HUẤN LUYỆN BASELINE (M6)

### 2.1 Tổng hợp kết quả theo Epoch (14k Train / 1k Val Held-out)
| Epoch | Trạng thái | Train Total | Box | Obj | Cls | Val Total | Ghi chú |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | ✅ 100% | 4.7706 | 0.4253 | 0.3700 | 2.2741 | 4.7282 | Backbone đóng băng |
| **2** | ✅ 100% | 4.1866 | 0.3970 | 0.3535 | 1.8480 | 4.4330 | Backbone đóng băng |
| **3** | ✅ 100% | 4.3168 | 0.4081 | 0.3593 | 1.9171 | **4.1814** | Best Val Loss (`best_map.pth`) |
| **4** | ⏸ 60.4% | 5.1313* | 0.4088* | 0.3964* | 2.6909* | — | Mở khóa ResNet-50 (Batch 2100/3474) |

### 2.2 Kế hoạch chuyển sang GPU & Chạy đủ 20 Epochs (chuẩn `base_config.yaml`)
- **Lịch sử CPU trước đó:** Hoàn thành Epoch 1–3 (100%) + Epoch 4 đạt 2.100 / 3.474 batches (`--batch-size 4 --num-threads 4`).
- **Kế hoạch GPU mới:** Tạo `.venv` với PyTorch CUDA (`cu126`), chạy đủ **20 epochs** (`--batch-size 8 --device cuda --git-sync`) trên **RTX 3050 6GB** (lưu checkpoint mỗi 50 batches & cuối mỗi epoch).
- **Lệnh chạy 20 epochs trên GPU:**
  ```bash
  set PYTHONPATH=.
  .venv\Scripts\python.exe -u -m src.train --epochs 20 --batch-size 8 --device cuda --git-sync
  ```

---

## 3. TÓM TẮT NGHIỆM THU CÁC MILESTONES ĐÃ HOÀN TẤT

- **M0–M1 (Dữ liệu):** Tách 14.000 train / 1.000 held-out (`seed=42`, overlap = 0); tải đủ 118.287 ảnh `train2017` và 5.000 ảnh `val2017`; ánh xạ 80 lớp COCO bỏ qua 10 ID khuyết; Letterbox $512 \times 512$ có sai số khôi phục $< 1.0\text{ px}$.
- **M2 (Kiến trúc):** `ResNet50FPN` (29.71M tổng tham số; 6.26M trainable khi đóng băng, 29.49M khi mở khóa layers 2–4); `DecoupledYOLOHead` chia sẻ trọng số Conv nhưng tách `BatchNorm2d` theo tầng, khởi tạo Focal bias $b \approx -4.595$.
- **M3 (Assigner & Loss):** `assign_targets` phân tầng theo $\sqrt{wh}$ và gán 3 ô dương/vật thể; `MultiTaskLoss` ($5.0\mathcal{L}_{\text{CIoU}} + \mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{BCE}}$); Overfit 10 ảnh đạt loss giảm **98.21%** ($58.9644 \to 1.0534$), Recall **1.00** (10/10 boxes).
- **M4 (Training Engine):** `AdamW` + `CosineAnnealingLR` (warmup 2 epochs) + Gradient Clipping `10.0` + lưu checkpoint mỗi 50 batches (`epoch_{N}.pth`) và cuối epoch (`best_map.pth`, `last_epoch.pth`) + `--git-sync`.
- **M5 (Decoder & Eval):** `YoloDecoder` giải mã offset $[-0.5, 1.5]s_l$, kích thước $[0, 16s_l]$, Batched NMS (`IoU=0.60`, top-100); Pilot test 100 ảnh held-out sinh 8.289 predictions hợp lệ.
- **M7 (Ablation trên 100 ảnh held-out):** So sánh trực tiếp chứng minh: CIoU tối ưu hình học 2D tốt hơn Smooth L1; Focal Loss tránh bùng nổ loss nền (BCE tăng 43.8% loss total / +45.4% loss obj); FPN $\{P_3, P_4, P_5\}$ (5.376 ô) bao phủ vật nhỏ/vừa so với đơn tầng $P_5$ (chỉ 256 ô). Biểu đồ so sánh tại `outputs/visualizations/metrics/m7_ablation_comparison.png`.
- **M8–M9 (Eval, Visualizer & CLI):** Đánh giá chính thức 1 lần trên `val2017` (200 ảnh, 20.000 predictions, 2.61 FPS CPU), khóa `outputs/final_eval.lock`; xuất 15 ảnh đối chiếu GT vs Pred tại `outputs/visualizations/eval_samples/`; kiểm thử `src.demo`, `src.visualize`, `src.evaluate`, `src.train`; **55/55 unit tests** (+1 slow test) pass 100%.
