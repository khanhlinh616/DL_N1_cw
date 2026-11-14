# TIẾN ĐỘ THỰC HIỆN DỰ ÁN (PLAN & PROGRESS STATUS)

* **Cập nhật lần cuối:** 2026-11-14
* **Trạng thái tổng thể:** Hoàn thành M0, M1, M2, M3; Sẵn sàng cho M4.

---

## 1. BẢNG THEO DÕI CÁC MILESTONES

| Milestone | Mô tả công việc | Tiêu chí hoàn thành | Trạng thái |
| :--- | :--- | :--- | :--- |
| **M0** | Môi trường, requirements, split dữ liệu 14k/1k | Sinh `train_subset14k.json`, `val_heldout1k.json`, `instances_val2017.json` | **Hoàn thành** |
| **M1** | Category mapping, transforms, Dataset metadata | Unit test mapping 80 lớp 100%, shape [B,3,512,512], metadata đầy đủ, roundtrip box < 1px | **Hoàn thành** |
| **M2** | Backbone ResNet50 (FrozenBN) + FPN + YOLO Head | Forward pass GPU/CPU thành công, đúng shape tại P3, P4, P5, bias obj ≈ -4.595 | **Hoàn thành** |
| **M3** | Assigner (3 ô) + MultiTaskLoss (CIoU, Focal, BCE) | Unit test loss, Overfit 10 ảnh: loss mượt về 0, recall ≈ 1.0 | **Hoàn thành** |
| **M4** | Training engine (AdamW, Cosine LR, AMP, clip 10) | Chạy 1 epoch thử nghiệm, lưu checkpoint theo val-heldout | Chưa bắt đầu |
| **M5** | Decoder (offset [-0.5, 1.5], dải 16s), Batched NMS | Smoke test 100 ảnh held-out, xuất JSON hợp lệ, chạy COCOeval | Chưa bắt đầu |
| **M6** | Train baseline 20 epochs trên 14k ảnh train | Hoàn thành 20 epochs, lưu `best_map.pth` theo val-heldout | Chưa bắt đầu |
| **M7** | 3 bài Ablation studies (mỗi bài 5-10 epochs) | Bảng so sánh 3 ablation chỉ đo trên val-heldout 1k | Chưa bắt đầu |
| **M8** | Final evaluation trên val2017 (5k) + Visualizer | Chạy COCOeval đúng 1 lần trên val2017, xuất ≥15 ảnh (tốt & 3 ca lỗi) | Chưa bắt đầu |
| **M9** | Đóng gói, dọn dẹp, kiểm tra CLI và tài liệu | CLI train/eval/demo chạy độc lập mượt mà, PEP8 chuẩn | Chưa bắt đầu |

---

## 2. KẾT QUẢ NGHIỆM THU CHI TIẾT

### Milestone M0: Setup & Data Split
- [x] Tạo `.gitignore` chặn triệt để `/data/`, `/outputs/`, checkpoint, secrets.
- [x] Tạo `requirements.txt` và các file `configs/`.
- [x] Tách 14k train / 1k held-out từ COCO 2017 với seed 42 cố định, assert overlap = 0.
- [x] Đã tải và giải nén toàn bộ 5.000 ảnh `val2017`. File `train2017.zip` đang tải nền.

### Milestone M1: Dataset & Metadata Pipeline
- [x] `src/data/category_mapper.py`: Ánh xạ 2 chiều 80 lớp chuẩn COCO, loại trừ 10 ID khuyết.
- [x] `src/data/transforms.py`: Letterbox 512x512 giữ aspect ratio, sai số khôi phục < 1.0 px trên 7 tỷ lệ khung hình.
- [x] `src/data/dataset.py`: `COCODataset` và `detection_collate_fn` trả về batch `[B, 3, 512, 512]` kèm đầy đủ metadata.

### Milestone M2: Model Architecture
- [x] `src/models/backbone_fpn.py`: ResNet-50 kết hợp `FrozenBatchNorm2d` và FPN P3 (stride 8), P4 (stride 16), P5 (stride 32), cố định 256 kênh.
- [x] Hỗ trợ cơ chế đóng băng toàn bộ backbone ở 3 epoch đầu (`freeze_backbone()`) và mở khóa Layers 2–4 từ epoch 4 (`unfreeze_backbone(trainable_layers=3)`). `FrozenBatchNorm2d` luôn giữ `requires_grad = False`.
- [x] `src/models/yolo_head.py`: Decoupled Head phong cách YOLOX/YOLOv8 với 3 nhánh riêng biệt (Box, Obj, Cls), chia sẻ trọng số đa tầng (shared weights).
- [x] Khởi tạo Focal bias ban đầu cho nhánh Objectness: $b = -\log((1-\pi)/\pi) \approx -4.595$ ($\pi = 0.01$).
- [x] `src/models/detector.py`: `YOLODetector` và factory `build_detector()`.
- [x] Toàn bộ 26/26 unit tests passed.

### Milestone M3: Label Assigner & MultiTask Loss
- [x] `src/utils/box_ops.py`: Triển khai các hàm biến đổi format box (`xyxy2cxcywh`, `cxcywh2xyxy`), `box_iou` tính IoU ma trận cặp đôi, và `ciou_loss` tính Complete IoU Loss chuẩn xác (đối chiếu đối đẳng với `torchvision.ops.complete_box_iou_loss`).
- [x] `src/core/assigner.py`: `assign_targets` phân tầng theo quy mô box ($\sqrt{wh} < 64 \to P3$, $64 \le \sqrt{wh} < 168 \to P4$, $\ge 168 \to P5$). Thuật toán gán 3 ô dương (ô tâm + 2 ô lân cận gần tâm nhất theo phần dư $\delta_x, \delta_y$), gán `target_obj = 1.0` cho 3 ô dương và `0.0` cho các ô còn lại, xử lý chính xác biên lưới.
- [x] `src/core/loss.py`: `MultiTaskLoss` với $\mathcal{L}_{\text{total}} = 5.0 \cdot \mathcal{L}_{\text{CIoU}} + 1.0 \cdot \mathcal{L}_{\text{Focal\_obj}} + 1.0 \cdot \mathcal{L}_{\text{BCE\_cls}}$, giải mã box theo offset $[-0.5, 1.5]$ và kích thước $[0, 16s]$, chuẩn hóa theo tổng số ô dương `norm_factor = max(1, num_pos)`.
- [x] `src/models/yolo_head.py`: Tối ưu hóa khởi tạo trọng số các lớp chiếu (`normal_(std=0.01)`) và phân tách BatchNorm theo tầng (`P3`, `P4`, `P5`) trong khối tích chập dùng chung để bảo toàn running statistics độc lập giữa các độ phân giải không gian.
- [x] `tests/test_overfit.py`: Thực nghiệm overfit 10 ảnh kiểm chứng pipeline:
  - Initial loss: `58.9644` $\to$ Final loss: `1.0534` (giảm **98.21%** $\ge 95\%$).
  - Recall kiểm thử ở chế độ `EVAL` end-to-end đạt **1.00** (**10/10** boxes detected $\ge 0.90$).
- [x] Toàn bộ **36/36 tests** của dự án PASSED 100%.
