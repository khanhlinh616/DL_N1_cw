# TÀI LIỆU DEMO VÀ TRỰC QUAN HÓA (DEMO & VISUALIZATION)

* **Module liên quan:** [`src/demo.py`](file:///d:/cw/src/demo.py), [`src/visualize.py`](file:///d:/cw/src/visualize.py)
* **Kiểm thử:** `tests/test_demo.py`, `tests/test_visualize.py`
* **Đầu ra:** Thư mục `outputs/visualizations/`

---

## 1. SUY LUẬN TRÊN ẢNH ĐƠN LẺ (`src/demo.py`)

Công cụ CLI chạy mô hình trên bất kỳ ảnh RGB nào theo đúng 5 bước:
1. **Preprocess:** Nạp ảnh bằng PIL, Letterbox về $512 \times 512$ (giữ tỷ lệ, đệm màu xám `114`), chuẩn hóa ImageNet.
2. **Forward:** Chạy `YOLODetector` ở chế độ `eval()` tạo logits trên 3 tầng $P_3, P_4, P_5$.
3. **Decode & NMS:** Giải mã hộp bao qua `YoloDecoder`, lọc `score >= conf_thresh`, áp dụng Batched NMS (`nms_thresh = 0.60`), khôi phục tọa độ về kích thước ảnh gốc.
4. **Draw:** Vẽ khung chữ nhật và nhãn dạng `"{class_name}: {score:.2f}"` lên ảnh gốc.
5. **Save:** Lưu ảnh kết quả vào thư mục `--output-dir`.

### Tham số CLI của `src/demo.py`
| Cờ (Flag) | Mặc định | Ý nghĩa |
|:---|:---|:---|
| `--image` | *(Bắt buộc)* | Đường dẫn ảnh RGB đầu vào |
| `--checkpoint` | `outputs/checkpoints/best_map.pth` | Đường dẫn file trọng số `.pth` |
| `--conf-thresh` | `0.25` | Ngưỡng điểm tin cậy tối thiểu để giữ box |
| `--nms-thresh` | `0.60` | Ngưỡng IoU cho Batched NMS |
| `--output-dir` | `outputs/visualizations/demo` | Thư mục lưu ảnh sau khi vẽ box |
| `--device` | `None` (tự chọn) | Thiết bị tính toán (`cpu` hoặc `cuda`) |

```bash
set PYTHONPATH=.
python -m src.demo --image data/coco/val2017/000000000139.jpg --checkpoint outputs/checkpoints/best_map.pth --conf-thresh 0.30
```

**Ví dụ định dạng kết quả đầu ra:**
```text
Ảnh: data/coco/val2017/000000000139.jpg (640x480)
  Box 1: [52.3, 134.1, 480.2, 420.6]  score=0.84  label="person"
  Box 2: [10.1,  22.7, 110.5, 185.3]  score=0.61  label="bicycle"
```

---

## 2. TRỰC QUAN HÓA VÀ PHÂN TÍCH CA LỖI (`src/visualize.py`)

Phục vụ nghiệm thu Milestone M8 bằng cách đối chiếu trực tiếp Ground Truth và Dự đoán:
- **Quy ước màu sắc:**
  - **Xanh lá (Green):** Hộp Ground Truth kèm tên lớp chuẩn.
  - **Đỏ (Red):** Hộp Dự đoán kèm tên lớp và điểm tin cậy `score`.
- **Tiêu chí nghiệm thu M8 (đã hoàn thành — 15 ảnh mẫu tại `outputs/visualizations/eval_samples/`):**
  1. **Ca phát hiện tốt:** Khớp lớp chính xác, IoU cao với Ground Truth.
  2. **3 ca lỗi điển hình:**
     - *Bỏ sót vật thể nhỏ (Small Object Miss):* Vật $< 32^2$ px bị mất thông tin ở độ phân giải $512 \times 512$.
     - *Che khuất nặng / cảnh đông đúc (Severe Occlusion):* Nhiều vật chồng lấn bị NMS gộp hoặc mất tâm lưới.
     - *Nhận diện nhầm / sai hậu cảnh (False Positive):* Dự đoán nhầm lớp tương đồng hoặc gán nhãn cho vùng nền.

### Tham số CLI chính của `src/visualize.py`
| Cờ (Flag) | Mặc định | Ý nghĩa |
|:---|:---|:---|
| `--predictions` | `outputs/evaluation_results/predictions_heldout.json` | Tệp dự đoán định dạng COCO JSON |
| `--annotations` | `data/coco/val_heldout1k.json` | Tệp Ground Truth COCO JSON |
| `--image-dir` | `data/coco/train2017` | Thư mục chứa ảnh gốc |
| `--output-dir` | `outputs/visualizations/eval_samples` | Thư mục lưu ảnh đối chiếu |
| `--num-samples` | `15` | Số lượng ảnh mẫu cần xuất |
| `--conf-thresh` | `0.10` | Ngưỡng confidence tối thiểu khi vẽ dự đoán |

```bash
set PYTHONPATH=.
python -m src.visualize --predictions outputs/evaluation_results/predictions_heldout.json --annotations data/coco/val_heldout1k.json --num-samples 15
```
