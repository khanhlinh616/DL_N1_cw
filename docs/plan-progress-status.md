# TIẾN ĐỘ THỰC HIỆN DỰ ÁN (PLAN & PROGRESS STATUS)

* **Cập nhật lần cuối:** 2026-11-14
* **Trạng thái tổng thể:** Hoàn thành M0 & M1; Sẵn sàng cho M2.

---

## 1. BẢNG THEO DÕI CÁC MILESTONES

| Milestone | Mô tả công việc | Tiêu chí hoàn thành | Trạng thái |
| :--- | :--- | :--- | :--- |
| **M0** | Môi trường, requirements, split dữ liệu 14k/1k | Sinh `train_subset14k.json`, `val_heldout1k.json`, `instances_val2017.json` | **Hoàn thành** |
| **M1** | Category mapping, transforms, Dataset metadata | Unit test mapping 80 lớp 100%, shape [B,3,512,512], metadata đầy đủ, roundtrip box < 1px | **Hoàn thành** |
| **M2** | Backbone ResNet50 (FrozenBN) + FPN + YOLO Head | Forward pass GPU/CPU thành công, đúng shape tại P3, P4, P5 | Chưa bắt đầu |
| **M3** | Assigner (3 ô) + MultiTaskLoss (CIoU, Focal, BCE) | Unit test loss, Overfit 10 ảnh: loss mượt về 0, recall ≈ 1.0 | Chưa bắt đầu |
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
- [x] Tạo `requirements.txt` và các file `configs/` (`dataset_coco_subset.yaml`, `base_config.yaml`, `model_resnet50_fpn.yaml`).
- [x] Module `src/data/download_data.py`: Tải annotations COCO, chia 14k train / 1k held-out với seed 42 cố định.
- [x] Đã assert `overlap == 0` giữa train và val-heldout trên toàn bộ 14k và 1k ảnh.
- [x] Đã tải và giải nén toàn bộ 5.000 ảnh `val2017`. File `train2017.zip` đang tải nền.
- [x] Khởi tạo git, push lên remote `https://github.com/khanhlinh616/DL_N1_cw.git`.

### Milestone M1: Dataset & Metadata Pipeline
- [x] `src/data/category_mapper.py`: Ánh xạ 2 chiều 80 lớp `category_id (1-90) <-> class_idx (0-79)`, loại trừ 10 ID khuyết.
- [x] `src/data/transforms.py`: Letterbox 512x512 giữ aspect ratio, lưu `scale_ratio` và `pad_offsets`.
- [x] Khôi phục tọa độ hộp bao (inverse letterbox) kiểm chứng sai số tuyệt đối `< 1.0 pixel` trên 7 tỷ lệ khung hình khác nhau.
- [x] Random Horizontal Flip xử lý tọa độ chuẩn xác (`x1' = 512 - x2, x2' = 512 - x1`).
- [x] `src/data/dataset.py`: Trả về tensor ảnh `[3, 512, 512]` và dict metadata: `image_id`, `orig_size`, `scale_ratio`, `pad_offsets`.
- [x] DataLoader + `detection_collate_fn` trả về batch `[B, 3, 512, 512]`. Kiểm chứng khép kín trên cả dữ liệu giả lập và 5.000 ảnh thật `val2017`.
- [x] Toàn bộ 22/22 unit tests passed.
