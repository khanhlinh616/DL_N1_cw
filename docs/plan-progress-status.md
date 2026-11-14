# TIẾN ĐỘ THỰC HIỆN DỰ ÁN (PLAN & PROGRESS STATUS)

* **Cập nhật lần cuối:** 2026-11-14
* **Trạng thái tổng thể:** Đang thực hiện M0 (Setup & Data Split)

---

## 1. BẢNG THEO DÕI CÁC MILESTONES

| Milestone | Mô tả công việc | Tiêu chí hoàn thành | Trạng thái |
| :--- | :--- | :--- | :--- |
| **M0** | Môi trường, requirements, split dữ liệu 14k/1k | Sinh `train_subset14k.json`, `val_heldout1k.json`, `instances_val2017.json` | **Đang thực hiện** |
| **M1** | Category mapping, transforms, Dataset metadata | Unit test mapping 80 lớp 100%, shape [B,3,512,512], metadata đầy đủ | Chưa bắt đầu |
| **M2** | Backbone ResNet50 (FrozenBN) + FPN + YOLO Head | Forward pass GPU/CPU thành công, đúng shape tại P3, P4, P5 | Chưa bắt đầu |
| **M3** | Assigner (3 ô) + MultiTaskLoss (CIoU, Focal, BCE) | Unit test loss, Overfit 10 ảnh: loss mượt về 0, recall ≈ 1.0 | Chưa bắt đầu |
| **M4** | Training engine (AdamW, Cosine LR, AMP, clip 10) | Chạy 1 epoch thử nghiệm, lưu checkpoint theo val-heldout | Chưa bắt đầu |
| **M5** | Decoder (offset [-0.5, 1.5], dải 16s), Batched NMS | Smoke test 100 ảnh held-out, xuất JSON hợp lệ, chạy COCOeval | Chưa bắt đầu |
| **M6** | Train baseline 20 epochs trên 14k ảnh train | Hoàn thành 20 epochs, lưu `best_map.pth` theo val-heldout | Chưa bắt đầu |
| **M7** | 3 bài Ablation studies (mỗi bài 5-10 epochs) | Bảng so sánh 3 ablation chỉ đo trên val-heldout 1k | Chưa bắt đầu |
| **M8** | Final evaluation trên val2017 (5k) + Visualizer | Chạy COCOeval đúng 1 lần trên val2017, xuất ≥15 ảnh (tốt & 3 ca lỗi) | Chưa bắt đầu |
| **M9** | Đóng gói, dọn dẹp, kiểm tra CLI và tài liệu | CLI train/eval/demo chạy độc lập mượt mà, PEP8 chuẩn | Chưa bắt đầu |

---

## 2. CHI TIẾT TIẾN ĐỘ HIỆN TẠI: MILESTONE M0
- [x] Tạo file `.gitignore` loại trừ `data/`, `outputs/`, `.pth`, secrets, cache.
- [x] Tạo `requirements.txt` chuẩn hóa các thư viện cần thiết.
- [x] Tạo các file cấu hình YAML trong `configs/` (`dataset_coco_subset.yaml`, `base_config.yaml`, `model_resnet50_fpn.yaml`).
- [x] Viết module `src/data/download_data.py` hỗ trợ 2 chế độ, in dung lượng, seed=42, assert zero overlap.
- [x] Viết unit tests `tests/test_data_split.py` kiểm chứng zero leakage, seed tất định, 4/4 tests PASSED.
- [ ] Tải và giải nén annotation, tách file JSON thực tế `train_subset14k.json` và `val_heldout1k.json`.
- [ ] Tải các gói ảnh zip theo yêu cầu người dùng.
