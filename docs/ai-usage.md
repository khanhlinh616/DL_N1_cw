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

---

## MILESTONE M5: PREDICTION DECODER & COCOEVAL INTEGRATION
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Triển khai module giải mã tọa độ hộp bao `src/core/decoder.py` (công thức offset $[-0.5, 1.5]$ và kích thước $[0, 16s]$, Batched NMS, khôi phục kích thước ảnh gốc), module và CLI đánh giá `src/evaluate.py` tích hợp `pycocotools.cocoeval`, bộ unit test `tests/test_decoder.py` và `tests/test_evaluate.py`, thực hiện pilot smoke test trên 100 ảnh held-out thật.
- **Đóng góp của sinh viên:** Yêu cầu nghiệm thu M5 theo đúng tiêu chuẩn thiết kế DEFINE_AND_DESIGN.md §2.4.3 & §2.7.4, giám sát chặt chẽ quy tắc chống rò rỉ dữ liệu (RULES.md §17.3 không chạm `val2017` trước M8), yêu cầu giải thích logic giải mã để nắm vững bản chất học thuật trước khi tích hợp.
- **Thành phần AI sinh ra / tinh chỉnh:**
  - `src/core/decoder.py`: Các hàm `decode_box_coordinates`, class `YoloDecoder` (kết hợp multi-scale logits, tính composite confidence $\text{score} = \sigma(obj) \times \sigma(cls)$, lọc ngưỡng, Batched NMS theo lớp, khôi phục tọa độ theo scale ratio và padding) và hàm `to_coco_results`.
  - `src/evaluate.py`: Hàm `evaluate_coco` và CLI hoàn chỉnh hỗ trợ `--dataset heldout`/`val2017`, `--final`, `--max-images`, `--output-dir`, cơ chế kiểm soát khóa an toàn `outputs/final_eval.lock` và ghi nhật ký kiểm toán `docs/audit-log.md`.
  - `tests/test_decoder.py`: Bộ 6 unit tests bao phủ toàn diện từ công thức giải mã, tính chất toán học của cận dải, gom logits 3 tầng (5.376 anchors), kiểm chứng cô lập lớp trong Batched NMS, khôi phục tọa độ ảnh gốc đến định dạng xuất COCO JSON.
  - `tests/test_evaluate.py`: Bộ 3 unit tests kiểm thử cơ chế chặn an toàn khi thiếu cờ `--final`, chặn khi có file khóa, và pipeline đánh giá trọn vẹn trên dữ liệu kiểm thử.
- **Kiểm chứng:**
  - `pytest tests/test_decoder.py`: 6/6 passed.
  - `pytest tests/test_evaluate.py`: 3/3 passed.
  - Kiểm thử an toàn CLI: `python -m src.evaluate --dataset val2017` bị chặn lập tức với mã lỗi 1 và thông điệp cảnh báo vi phạm quy tắc.
  - Pilot Smoke Test 100 ảnh held-out thật: `python -m src.evaluate --dataset heldout --max-images 100` hoàn tất thành công, tạo file kết quả chuẩn `outputs/evaluation_results/predictions_heldout.json` (8.289 predictions), kích hoạt `COCOeval` tổng hợp và xuất bảng 12 chỉ số chuẩn mực COCO.
  - Toàn bộ bộ test của dự án: **48/48 unit tests** (+ 1 slow overfit test) PASSED 100%.

---

## MILESTONE M6: BASELINE MODEL TRAINING (14K SUBSET)
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Khởi động vòng lặp huấn luyện Baseline trên 14.000 ảnh COCO train subset (`train_subset14k.json`, kế hoạch 5 epochs trên CPU) và đánh giá định kỳ trên 1.000 ảnh val held-out (`val_heldout1k.json`).
- **Đóng góp của sinh viên:** Xác nhận chuyển giao milestone M6 sau khi kiểm chứng hoàn tất 100% dữ liệu 118k ảnh `train2017` và nghiệm thu M5.
- **Thành phần AI sinh ra / điều phối:** Điều phối tiến trình huấn luyện nền, ghi nhận metric losses và lưu trữ checkpoints `best_map.pth`.
- **Kiểm chứng:** Hoàn thành 100% Epoch 1 (Train 4.7706, Val 4.7282), Epoch 2 (Train 4.1866, Val 4.4330) và Epoch 3 (Train 4.3168, Val 4.1814 — lưu `best_map.pth`). Đang chạy tiếp Epoch 4 mở khóa backbone ResNet-50 (đạt 60.4%, batch 2100/3474, loss 5.1313). Checkpoint lưu trữ an toàn từng epoch và mỗi 50 batch (`outputs/checkpoints/epoch_{N}.pth`), hỗ trợ resume chính xác.
- **Hợp nhất lịch sử:** Hợp nhất nhánh lịch sử ban đầu (commit `952fd34` từ DL Team 1) với toàn bộ tiến trình M0–M6, gắn lại tags M0–M6 theo đúng commit hash ban đầu và đồng bộ duy nhất nhánh `main`.

---

## MILESTONE M7: ABLATION STUDIES (VAL-HELDOUT 1K)
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Cài đặt module bóc tách `src/ablation.py` hỗ trợ các biến thể hàm loss (Smooth L1 vs CIoU, Standard BCE vs Focal Loss, Đơn tầng P5 vs FPN {P3,P4,P5}) và thực thi đo lường khách quan trên tập `val-heldout 1k`.
- **Đóng góp của sinh viên:** Yêu cầu tuân thủ nguyên tắc không rò rỉ dữ liệu (RULES.md §17.3), chỉ chạy ablation trên held-out.
- **Thành phần AI sinh ra / tinh chỉnh:** `src/core/loss.py` (hỗ trợ `box_loss_type`, `obj_loss_type`), `src/ablation.py`, `tests/test_loss.py`.
- **Kiểm chứng:** Đo lường thực tế trên 100 ảnh held-out: CIoU tối ưu hình học tốt hơn Smooth L1 (`425.08` vs `422.35`), BCE tăng 43.8% loss total (`611.34`, loss obj `596.12`) do nhiễu mẫu âm tính, đơn tầng P5 (`25.14`, chỉ xét 256/5.376 ô lưới) bỏ sót vật nhỏ/vừa ở P3/P4. Cập nhật `outputs/metrics.json` và `docs/report.md`.

---

## MILESTONES M8 & M9: VISUALIZER, FINAL EVAL & DEMO CLI
- **Công cụ:** Antigravity AI Agent.
- **Mục đích prompt:** Nâng cấp hệ thống lưu checkpoint theo từng epoch (`epoch_N.pth`), in tiến độ batch chi tiết có flush, cài đặt hoàn chỉnh CLI suy luận ảnh đơn lẻ `src/demo.py`, CLI trực quan hóa GT vs Pred `src/visualize.py` trích xuất 15 ca mẫu phân loại, và thực thi đánh giá chính thức có khóa an toàn trên `val2017`.
- **Đóng góp của sinh viên:** Yêu cầu lưu checkpoint theo từng epoch, kiểm soát tiến độ, chỉ đạo đồng bộ duy nhất nhánh `main` lên repository nhóm.
- **Thành phần AI sinh ra / tinh chỉnh:** `src/train.py` (lưu `epoch_{N}.pth`, log batch), `src/demo.py`, `src/visualize.py`, `src/evaluate.py`, `tests/test_demo.py`, `tests/test_visualize.py`.
- **Kiểm chứng:** Test suite đạt 55/55 passed 100%, chạy thử thành công `src.demo` xuất ảnh phát hiện hộp bao, chạy thử `src.visualize` xuất đủ 15 ca mẫu đối chiếu, đánh giá chính thức `val2017` xuất 20.000 dự đoán và khóa an toàn tại `outputs/final_eval.lock`.
