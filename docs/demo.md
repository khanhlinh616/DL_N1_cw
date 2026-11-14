# TÀI LIỆU DEMO VÀ TRỰC QUAN HÓA (DEMO & VISUALIZATION)

* **Module liên quan:** `src/demo.py`, `src/visualize.py`
* **Đầu ra:** Thư mục `outputs/visualizations/`

---

## 1. SUY LUẬN TRÊN ẢNH ĐƠN LẺ (`src/demo.py`)

Công cụ dòng lệnh cho phép người dùng chạy thử mô hình trên một ảnh bất kỳ từ nguồn bên ngoài hoặc từ tập kiểm thử:
- **Tiền xử lý:** Tự động Letterbox ảnh về kích thước $512 \times 512$, chuẩn hóa RGB theo ImageNet.
- **Dự đoán:** Chạy forward qua mạng, giải mã bounding boxes và lọc NMS.
- **Trực quan hóa:** Vẽ hộp bao dự đoán kèm tên lớp đối tượng và điểm tin cậy `score`.
- **Lưu file:** Xuất ảnh kết quả có vẽ box ra thư mục chỉ định.

### Lệnh thực thi:
```bash
# Chạy demo trên 1 ảnh với ngưỡng tin cậy 0.30
python src/demo.py \
    --image data/coco/val2017/000000000139.jpg \
    --checkpoint outputs/checkpoints/best_map.pth \
    --conf-thresh 0.30 \
    --output-dir outputs/visualizations/demo
```

---

## 2. TRỰC QUAN HÓA VÀ PHÂN TÍCH CA LỖI (`src/visualize.py`)

Phục vụ việc nghiệm thu Milestone M8 theo yêu cầu của đề tài:
- **So sánh trực quan Ground Truth vs. Dự đoán:**
  - Hộp Ground Truth được viền màu **xanh lá cây (Green)** kèm nhãn chuẩn.
  - Hộp Dự đoán được viền màu **xanh dương / đỏ** kèm nhãn dự đoán và điểm tin cậy $\text{score}$.
- **Tiêu chí nghiệm thu M8:**
  - Xuất tối thiểu **15 ảnh mẫu trực quan hóa** vào thư mục `outputs/visualizations/`.
  - Phân loại rõ ràng:
    1. Các trường hợp phát hiện xuất sắc (đối tượng rõ ràng, kích thước đa dạng, khớp chuẩn IoU).
    2. Ít nhất **3 ca lỗi điển hình**:
       - *Bỏ sót vật thể cực nhỏ (Small Object Miss)*.
       - *Che khuất nặng hoặc dính chùm (Severe Occlusion / Cluttered Scene)*.
       - *Nhận diện nhầm lớp hoặc phát hiện sai hậu cảnh (False Positive)*.

### Lệnh thực thi:
```bash
# Xuất 15 ảnh mẫu phân tích kết quả sau bài đánh giá M8
python src/visualize.py \
    --predictions outputs/evaluation_results/predictions_val2017.json \
    --annotations data/coco/annotations/instances_val2017.json \
    --image-dir data/coco/val2017 \
    --output-dir outputs/visualizations/eval_samples \
    --num-samples 15
```
