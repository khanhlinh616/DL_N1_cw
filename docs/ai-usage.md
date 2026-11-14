# NHẬT KÝ SỬ DỤNG AI (AI USAGE LOG)

* **Quy tắc tuân thủ:** Ghi nhận minh bạch mục đích prompt, công cụ sử dụng, đóng góp của sinh viên, thành phần AI sinh ra và phương pháp kiểm chứng (theo RULES.md §3 & §17.11).

---

## MILESTONE M0: SETUP & DATA SPLIT
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Khởi tạo repository, cấu hình `.gitignore`, `requirements.txt`, script tải và chia tập dữ liệu `src/data/download_data.py`.
- **Đóng góp của sinh viên:** Cung cấp tài liệu thiết kế gốc `DEFINE_AND_DESIGN.md`, phê duyệt tải zip đầy đủ, cung cấp URL remote repo GitHub.
- **Thành phần AI sinh ra:** `.gitignore`, `requirements.txt`, `src/data/download_data.py`, `tests/test_data_split.py`, các file cấu hình YAML.
- **Kiểm chứng:** `pytest tests/test_data_split.py`: 4/4 passed. Tách thành công 14.000 train và 1.000 held-out không giao thoa.

---

## MILESTONE M1: DATASET & METADATA PIPELINE
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Cài đặt `category_mapper.py`, `transforms.py`, `dataset.py`.
- **Đóng góp của sinh viên:** Yêu cầu kiểm tra sai số khôi phục tọa độ hộp bao sau letterbox ($< 1.0\text{ px}$), yêu cầu vừa làm vừa commit.
- **Thành phần AI sinh ra:** `src/data/category_mapper.py`, `src/data/transforms.py`, `src/data/dataset.py`, các unit test tương ứng.
- **Kiểm chứng:** `pytest tests/`: 22/22 passed. Batch shape $[B, 3, 512, 512]$, kiểm chứng trên 5.000 ảnh thật `val2017`.

---

## MILESTONE M2: MODEL ARCHITECTURE
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Xây dựng kiến trúc ResNet-50 FrozenBN + FPN $\{P_3, P_4, P_5\}$, Decoupled YOLO Head dùng chung trọng số, khởi tạo Focal bias $b \approx -4.595$, `YOLODetector` wrapper.
- **Đóng góp của sinh viên:** Định hướng thiết kế Decoupled Head phong cách YOLOX/YOLOv8, kiểm soát việc tuân thủ cấu trúc thư mục và ràng buộc không dùng `torchvision.models.detection.*`.
- **Thành phần AI sinh ra:**
  - `src/models/backbone_fpn.py`: Module `ResNet50FPN` kết hợp FrozenBatchNorm2d và cơ chế đóng băng/mở khóa linh hoạt.
  - `src/models/yolo_head.py`: Module `DecoupledYOLOHead` với 3 nhánh riêng biệt (Box, Obj, Cls), chia sẻ trọng số đa tầng, bias obj $\approx -4.595$.
  - `src/models/detector.py`: `YOLODetector` và hàm `build_detector()`.
  - `tests/test_model_forward.py`: Kiểm thử shape feature maps, shape logits, giá trị bias, tính hợp lệ của cơ chế đóng băng backbone.
- **Kiểm chứng:** `pytest tests/test_model_forward.py`: 4/4 passed. Toàn bộ test suite 26/26 passed.

---

## MILESTONE M3: LABEL ASSIGNER & MULTITASK LOSS
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Nghiệm thu và hoàn thiện M3: kiểm chứng toán học CIoU, Center-grid Assigner 3 ô dương, Focal Loss objectness, BCE class loss, sửa lỗi bão hòa bias và trôi dạt BatchNorm đa tầng, hoàn thành overfit test 10 ảnh.
- **Đóng góp của sinh viên:** Yêu cầu chạy kiểm thử thực tế trên CPU, đặt tiêu chuẩn khắt khe cho overfit test (loss giảm $\ge 95\%$, recall $\ge 0.90$), kiểm soát quyền ghi tệp và hướng dẫn khắc phục ACL thư mục.
- **Thành phần AI sinh ra / tinh chỉnh:**
  - `src/utils/box_ops.py`: Các hàm toán học hộp bao và CIoU Loss (`box_iou`, `ciou_loss`).
  - `src/core/assigner.py`: Thuật toán phân tầng và gán 3 ô dương (`assign_targets`).
  - `src/core/loss.py`: Module `MultiTaskLoss` kết hợp CIoU + Focal Loss + BCE Loss.
  - `src/models/yolo_head.py`: Cải tiến khởi tạo trọng số các lớp projection (`normal_(std=0.01)`) và phân tách BatchNorm theo tầng (`bns`) để loại bỏ sai lệch running statistics giữa các độ phân giải không gian.
  - `tests/test_overfit.py`: Kịch bản overfit 10 ảnh với cơ chế cache backbone giúp tăng tốc độ hội tụ trên CPU.
  - `tests/test_assigner.py`, `tests/test_box_ops.py`, `tests/test_loss.py`: Bộ kiểm thử đơn vị toán học.
- **Kiểm chứng:**
  - Unit tests M3: `test_assigner.py` (3/3 passed), `test_box_ops.py` (3/3 passed), `test_loss.py` (2/2 passed).
  - Overfit test 10 ảnh: Initial loss `58.9644` $\to$ Final loss `1.0534` (giảm `98.21%` $\ge 95\%$), Recall `1.00` (`10/10` boxes detected $\ge 0.90$).
  - Toàn bộ test suite dự án: **36/36 passed** 100%.

---

## MILESTONE M4: TRAINING ENGINE & VALIDATION CHECKPOINTS
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Xây dựng Training Engine hoàn chỉnh (`src/train.py`), module theo dõi chỉ số (`src/utils/logger.py`), các kịch bản kiểm thử đơn vị (`tests/test_train.py`), chạy thử nghiệm 1 epoch và kiểm tra resume từ checkpoint.
- **Đóng góp của sinh viên:** Yêu cầu tiếp tục tiến độ M4, giám sát việc tuân thủ các quy tắc trong RULES.md §17.4-§17.8 và DEFINE_AND_DESIGN.md.
- **Thành phần AI sinh ra / tinh chỉnh:**
  - `src/utils/logger.py`: Module `MetricTracker` tính running average đa metric và `TrainingLogger` lưu lịch sử huấn luyện dạng JSON.
  - `src/train.py`: CLI và vòng lặp huấn luyện đầy đủ: AdamW, Cosine LR scheduler có warmup, AMP GradScaler, Gradient Clipping 10.0, đóng băng backbone 3 epoch đầu, cơ chế lưu/khôi phục checkpoint (`best_map.pth`, `last_epoch.pth`) và chạy thử nghiệm (--max-images).
  - `tests/test_train.py`: Bộ 5 unit tests kiểm thử bộ theo dõi metric, tối ưu hóa, scheduler, tính nhất quán bit-exact của checkpoint và 1 epoch smoke test.
- **Kiểm chứng:**
  - `pytest tests/test_train.py`: 5/5 passed.
  - CLI dry run: `python -m src.train --epochs 1 --batch-size 2 --max-images 4` hoàn thành trong 10 giây, sinh checkpoint và log history.
  - CLI resume run: `python -m src.train --epochs 2 --batch-size 2 --max-images 4 --resume outputs/checkpoints/last_epoch.pth` khôi phục chính xác trạng thái từ epoch 2 và cập nhật checkpoint tốt nhất.
  - Toàn bộ 40 fast unit tests của dự án PASSED 100% trong 12.89s.
