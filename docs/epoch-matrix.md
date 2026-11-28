# Ma trận Đánh giá & Trực quan hóa qua từng Epoch

> Nguồn dữ liệu thực nghiệm: `outputs/logs/train_history.json` và `outputs/evaluation_results/epoch_*_metrics.json` (`val-heldout 1k`, 1.000 ảnh).

## 1. Biểu đồ Trực quan hóa

![Loss Curves](../outputs/visualizations/metrics/loss_curves.png)

![COCO Trends](../outputs/visualizations/metrics/coco_ap_ar_trends.png)

![COCO Matrix Heatmap](../outputs/visualizations/metrics/coco_matrix_heatmap.png)

## 2. Bảng Loss qua các Epoch

| Epoch | LR | Train Total | Train Box | Train Obj | Train Cls | Val Total | Val Box | Val Obj | Val Cls |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `1e-04` | 4.7706 | 0.4253 | 0.3700 | 2.2741 | **4.7282** | 0.4158 | 0.3660 | 2.2830 |
| 2 | `1e-05` | 4.1866 | 0.3970 | 0.3535 | 1.8480 | **4.4330** | 0.4049 | 0.3605 | 2.0480 |
| 3 | `1e-03` | 4.3168 | 0.4081 | 0.3593 | 1.9171 | **4.1814** | 0.3997 | 0.3549 | 1.8277 |

## 3. Ma trận 12 chỉ số COCO (`val-heldout 1k`)

| Metric | Epoch 2 | Epoch 3 | Delta (Ep3-Ep2) |
|:---| :---: | :---: | :---: |
| **mAP@[.50:.95]** | 0.0773 | 0.0997 | +0.0224 |
| **mAP@.50** | 0.1838 | 0.2333 | +0.0495 |
| **mAP@.75** | 0.0572 | 0.0685 | +0.0113 |
| **AP (small)** | 0.0317 | 0.0506 | +0.0189 |
| **AP (medium)** | 0.1132 | 0.1345 | +0.0213 |
| **AP (large)** | 0.1024 | 0.1536 | +0.0512 |
| **AR@1** | 0.0960 | 0.1120 | +0.0160 |
| **AR@10** | 0.1690 | 0.2120 | +0.0430 |
| **AR@100** | 0.1883 | 0.2354 | +0.0471 |
| **AR (small)** | 0.0620 | 0.0870 | +0.0250 |
| **AR (medium)** | 0.2090 | 0.2610 | +0.0520 |
| **AR (large)** | 0.2620 | 0.3530 | +0.0910 |

## 4. Ảnh dự đoán minh họa & Lưu ý tái tạo

- **Ảnh dự đoán Epoch 3** (khung xanh: Ground Truth, khung đỏ: Prediction): `outputs/visualizations/eval_samples_epoch3/`
- **Cập nhật lại bảng & biểu đồ sau khi train thêm epoch:**

```powershell
$env:PYTHONPATH="."
python scripts/eval_epochs.py --epochs 1 2 3 4 5
```
