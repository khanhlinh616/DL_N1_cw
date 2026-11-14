# BÁO CÁO DỰ ÁN (PROJECT REPORT)

* **Dự án:** One-Stage Object Detector (YOLO-style Head trên COCO 2017)
* **Nhóm thực hiện:** Nhóm 1

---

## 1. TỔNG QUAN KIẾN TRÚC VÀ GIẢI PHÁP

Dự án cài đặt bộ phát hiện đối tượng one-stage end-to-end theo phong cách YOLO trên tập dữ liệu MS COCO 2017:
- **Backbone & Neck:** ResNet-50 kết hợp `FrozenBatchNorm2d` và Feature Pyramid Network (FPN) trích xuất đặc trưng tại 3 tỷ lệ: $P_3$ (stride 8), $P_4$ (stride 16), $P_5$ (stride 32), cố định 256 channels.
- **Detection Head:** Decoupled Head phong cách YOLOX/YOLOv8 với 3 nhánh độc lập (Box, Objectness, Classification), chia sẻ trọng số đa tầng (shared weights). Khởi tạo Focal bias cho Objectness $b \approx -4.595$.
- **Label Assigner:** Anchor-free center-grid assignment chọn tầng theo kích thước vật thể ($S < 64 \to P_3$, $64 \le S < 168 \to P_4$, $S \ge 168 \to P_5$) và chọn tối đa 3 ô dương lân cận tâm.
- **Loss Function:** $\mathcal{L} = 5.0 \times \mathcal{L}_{\text{CIoU}} + 1.0 \times \mathcal{L}_{\text{Focal\_obj}} + 1.0 \times \mathcal{L}_{\text{BCE\_cls}}$.
- **Inference Decoder:** Giải mã offset tọa độ với miền giá trị $[-0.5, 1.5] \times s$ và dải kích thước $[0, 16 \times s]$, kết hợp Batched NMS loại trừ trùng lặp.

---

## 2. EXAMPLE INPUT / OUTPUT

### 2.1 Đầu vào (Input)
- Ảnh màu RGB kích thước bất kỳ (ví dụ $640 \times 480 \times 3$).
- Chuẩn hóa qua phép biến đổi `Letterbox` về tensor $[3, 512, 512]$, padding màu xám 114, giữ nguyên tỷ lệ khung hình.
- Metadata kèm theo: `image_id`, `orig_size` $(H_{orig}, W_{orig})$, `scale_ratio` $r$, `pad_offsets` $(\text{pad}_w, \text{pad}_h)$.

### 2.2 Đầu ra (Output)
- Danh sách bounding boxes trên hệ tọa độ ảnh gốc: $[x_1, y_1, x_2, y_2]$.
- Nhãn phân loại: COCO Category ID ($1 \to 90$) tương ứng 80 lớp đối tượng.
- Điểm tin cậy: $\text{score} = \sigma(\text{obj}) \times \sigma(\text{cls}) \in [0, 1]$, lọc ngưỡng $\text{score} \ge 0.05$.
- File ảnh trực quan hóa: Hộp xanh lá (Ground Truth) và hộp đỏ/xanh dương (Dự đoán kèm nhãn + score).

---

## 3. EXPERIMENTAL RESULTS

> *Ghi chú tuân thủ RULES.md §17.4: Mọi số liệu dưới đây chỉ được điền từ file metrics thực tế trong `outputs/` sau khi hoàn thành huấn luyện và đánh giá.*

### 3.1 Kết quả Baseline (14k Train / 1k Val-heldout / 20 Epochs)
- **Epoch hoàn thành:** *Chưa chạy (sẽ cập nhật ở M6)*
- **Validation Loss (Held-out 1k):** *Chưa chạy*
- **mAP@[.5:.95] (Held-out 1k):** *Chưa chạy*
- **AP50 / AP75 (Held-out 1k):** *Chưa chạy*

### 3.2 Kết quả Ablation Studies (Chỉ đo trên Val-heldout 1k)
| Thí nghiệm | Thiết lập | mAP@[.5:.95] | AP50 | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Chuẩn (FrozenBN, Focal, CIoU, Data Aug) | *Chưa chạy* | *Chưa chạy* | Đo ở M6 |
| **Ablation 1** | Tác động của FrozenBatchNorm vs Trainable BN | *Chưa chạy* | *Chưa chạy* | Đo ở M7 |
| **Ablation 2** | Tác động của Focal Loss vs BCE Loss (obj) | *Chưa chạy* | *Chưa chạy* | Đo ở M7 |
| **Ablation 3** | Tác động của Data Augmentation (Flip/Color) | *Chưa chạy* | *Chưa chạy* | Đo ở M7 |

### 3.3 Đánh giá chính thức trên tập kiểm thử độc lập (`val2017` 5.000 ảnh)
> *Chỉ chạy đúng 1 lần duy nhất tại M8 với cờ `--final` và khóa lock an toàn.*
- **mAP@[.5:.95]:** *Chưa chạy*
- **AP50 / AP75:** *Chưa chạy*
- **AP_small / AP_medium / AP_large:** *Chưa chạy*
- **Tốc độ suy luận (FPS trên CPU/GPU):** *Chưa chạy*

---

## 4. LIMITATIONS (HẠN CHẾ KỸ THUẬT)

1. **Quy mô tập huấn luyện:** Huấn luyện trên subset 14.000 ảnh thay vì toàn bộ 118k ảnh của COCO do giới hạn tài nguyên và thời gian huấn luyện.
2. **Độ phân giải đầu vào:** Cố định $512 \times 512$ giúp tiết kiệm bộ nhớ nhưng có thể giảm khả năng phát hiện các vật thể cực nhỏ so với phân giải gốc hoặc $640 \times 640$.
3. **Phần cứng thực thi:** Huấn luyện không có cụm GPU phân tán nhiều node, phụ thuộc vào mixed-precision AMP trên 1 card đồ họa.

---

## 5. FUTURE IMPROVEMENTS (HƯỚNG PHÁT TRIỂN)

1. Mở rộng huấn luyện full dataset 118k ảnh COCO trong 50–100 epochs kèm Multi-scale training.
2. Nâng cấp cơ chế gán nhãn động nâng cao (SimOTA / Task-Aligned Assigner) thay vì anchor-free tĩnh.
3. Xuất mô hình sang định dạng ONNX / TensorRT và tích hợp lượng tử hóa INT8 phục vụ triển khai biên (Edge AI).
