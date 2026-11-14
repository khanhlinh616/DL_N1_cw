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
