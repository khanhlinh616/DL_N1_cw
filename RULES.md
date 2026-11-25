# RULES

## Quy tắc cốt lõi & Definition of Done
- **Definition of Done:** Đủ yêu cầu chức năng, code chạy, test passed, không secrets (`.env`/API key), đã code review, cập nhật docs, commit & push GitHub. Kiểm tra, đánh giá kỹ thay vì chỉ thấy chạy được.
- **Báo cáo (`report.md`):** Example In/Out, Results, Limitations, Future Improvements.
- **AI Usage (`ai-usage.md`):** Tools used, Main purposes, Student contributions, AI-generated components, Prompt purpose, Student modification, Validation. Không để AI push code chưa chạy thử/chưa hiểu; không sửa sai kết quả đánh giá.
- **Quy trình Build MVP (từng module):** Xác định mục tiêu, In/Out, Interface, Constraints trước khi sửa; sửa từng module một.
  `Describe → Generate → Review → Run (Smoke test) → Dry-run → Test → Refactor → Commit`
- **Usage & Docs:** `README.md` (Overview, Problem, Features, Architecture, Folder structure, Install/Run/Test), `DEFINE_AND_DESIGN.md`, `plan-progress-status.md`.

---

# MASTER RULES — QUY TẮC CHUNG CHO MỌI DỰ ÁN

## 1. Nguyên tắc làm việc & Tài liệu nguồn
- Đọc kỹ yêu cầu, đề bài, rubric, tài liệu và mã nguồn trước khi làm; xác định rõ mục tiêu, phạm vi, In/Out, ràng buộc.
- Ưu tiên nguồn chính thức; không tự ý mở rộng phạm vi, không bịa số liệu/kết quả/trạng thái, không báo cáo đã chạy/kiểm tra nếu chưa làm.
- Phân biệt rõ thông tin từ nguồn, đã kiểm chứng, giả định và đề xuất kỹ thuật. Giữ nguyên thuật ngữ, công thức, ký hiệu từ tài liệu.

## 2. Chỉnh sửa dự án & Chất lượng mã nguồn
- Kiểm tra cấu trúc, cấu hình, test hiện tại trước khi sửa; ưu tiên thay đổi tối thiểu, có mục tiêu, giữ nguyên phần đang hoạt động tốt.
- Không tự ý đổi tên file/API/định dạng hoặc xóa dữ liệu, checkpoint, kết quả, cấu hình giá trị. Kiểm tra phụ thuộc trước khi sửa module dùng chung.
- Code rõ ràng, tách trách nhiệm hợp lý, không trùng lặp logic, không thêm abstraction/thư viện thừa, xử lý ngoại lệ và biên phù hợp.
- Không hardcode secrets, mật khẩu, API key; tôn trọng phiên bản thư viện và giới hạn phần cứng.

## 3. Dữ liệu, Giải pháp kỹ thuật & Kiểm thử
- Xác định nguồn gốc, cấu trúc, kiểu và giới hạn dữ liệu; không tự tạo/sửa số liệu thực nghiệm; tách rạch ròi train/val/test, chống rò rỉ dữ liệu.
- Chọn giải pháp dựa trên bài toán, tài nguyên và khả năng kiểm chứng; không tự thay đổi phương pháp bắt buộc của đề bài.
- Chạy test hiện có trước và chạy lại test liên quan sau khi sửa; kiểm tra trường hợp biên và tương thích liên module.
- Không bỏ qua lỗi/warning/test thất bại; báo cáo trung thực test đã chạy, kết quả thực tế và phần chưa kiểm chứng.

## 4. Tài liệu, Văn phong & Tổ chức dự án
- Viết trực tiếp, rõ ràng, tập trung cơ chế, lý do và kết quả thực tế; loại bỏ câu sáo rỗng, lặp ý, tự khen không có bằng chứng.
- Phân biệt rõ mã nguồn, dữ liệu, cấu hình, kiểm thử, tài liệu và kết quả. Không đưa file tạm, cache, dữ liệu lớn, checkpoint vào git.
- Bảo đảm đường dẫn, cấu hình và hướng dẫn nhất quán với thực tế.

## 5. Tài nguyên, Bảo mật & Độ tin cậy
- Kiểm tra giới hạn phần cứng (RAM, CPU, GPU, disk, thời gian); ưu tiên giải pháp chạy được trên môi trường thực tế; không mặc định có GPU/Internet.
- Không lộ thông tin nhạy cảm; không chạy thao tác phá hủy dữ liệu khi chưa đánh giá rủi ro; không vô hiệu hóa kiểm tra hay che giấu lỗi để pass test.

## 6. Quy trình thực hiện & Báo cáo kết quả
- **Quy trình:** Phân tích yêu cầu → Đọc tài liệu & kiểm tra workspace → Xác định ràng buộc → Lựa chọn phương án tối thiểu → Sửa trực tiếp → Kiểm tra tương thích & chạy test → Rà soát & báo cáo thực tế.
- **Báo cáo:** Ngắn gọn, đủ ý: đã làm gì, file đã đổi, test đã chạy & kết quả, giới hạn/việc chưa xong, điểm cần xác nhận.
- **Ưu tiên khi xung đột:** An toàn & toàn vẹn dữ liệu > Yêu cầu bắt buộc > Tính chính xác > Tương thích hệ thống > Khả năng kiểm thử/bảo trì > Đơn giản > Phong cách.

---

## 16. Danh mục Docs
- **Quy trình:** `description - gen - smoke run - run - check des`
- **Giới hạn độ dài:** **Mỗi file `.md` tối đa 200 dòng**, lược bỏ lý thuyết thừa để trình bày code và ý chính (giữ nguyên cấu trúc `/docs` gồm `DEFINE_AND_DESIGN.md`, `report.md`, `data.md`, `model.md`, `training.md`, `evaluation.md`, `demo.md`, `ai-usage.md`, `plan-progress-status.md`, `audit-log.md`).

---

## 17. QUY TẮC RIÊNG CHO DỰ ÁN (YOLO-style detector trên COCO 2017)

### 17.1 Nguồn sự thật và thứ tự ưu tiên
- Đề bài > `RULES.md` > `docs/DEFINE_AND_DESIGN.md` > code. Đầu phiên đọc `RULES.md` và `docs/DEFINE_AND_DESIGN.md`.
- Muốn đổi thiết kế: sửa tài liệu trước, người dùng duyệt rồi mới sửa code. Cấu trúc thư mục theo mục 2.3 của thiết kế.

### 17.2 Ràng buộc bất biến
- Detector one-stage YOLO-style: ResNet-50 (FrozenBN) → FPN `{P3, P4, P5}` → Decoupled Head, ảnh `512×512`.
- Tự viết: Head, Assigner, Loss (CIoU, Focal, BCE), Decoder. Chỉ dùng `torchvision` cho `resnet50`, FPN, `nms`, `box_iou`.
- Công thức giải mã, quy tắc gán 3 ô dương và công thức loss phải đúng mục 2.4.3 và 2.7 của thiết kế.

### 17.3 Dữ liệu và val2017
- Train 14k và held-out 1k lấy từ `train2017`, seed cố định, không giao nhau. Chọn checkpoint, chỉnh siêu tham số và ablation chỉ dùng `held-out 1k`.
- `val2017` chỉ chạy đúng 1 lần ở M8. `evaluate.py` phải chặn nếu thiếu `--final`, đồng thời ghi `outputs/final_eval.lock` và log vào `docs/audit-log.md`. Không xóa lock để chạy lại.

### 17.4 Kết quả thực nghiệm & Tái lập
- Mọi số liệu (`mAP`, `AP50`, `loss`, `FPS`) lấy từ `outputs/` (`metrics.json`, log), lưu kèm config, seed, commit hash, thời gian chạy. Không sửa tay kết quả; chưa chạy ghi "chưa chạy".
- Đặt seed cho `python`, `numpy`, `torch`; lưu config và phiên bản thư viện vào checkpoint; `requirements.txt` ghi phiên bản cụ thể.

### 17.5 Kiểm thử và xác minh
- Dùng `pytest` chạy được trên CPU, test nặng gắn marker `slow`.
- Test bắt buộc: shape từng tầng, ánh xạ category 2 chiều, letterbox khép kín (sai số < 1 px), Assigner (đúng 3 ô, xử lý biên), CIoU đối chiếu `torchvision.ops.complete_box_iou_loss`.
- Trước M6 phải qua overfit test 10 ảnh (loss về gần 0, recall ≈ 1.0). Kiểm tra không NaN/Inf.
- **Smoke run** = 1–2 iteration hoặc ≤100 ảnh; **dry-run** = hết pipeline với `--max-images 8`, không ghi checkpoint chính.

### 17.6 Tài nguyên, Git & Biên Milestone
- Hỏi xác nhận trước khi tải >1 GB hoặc chạy lệnh >30 phút. Lưu checkpoint mỗi epoch, hỗ trợ `--resume`; không xóa checkpoint/kết quả cũ; chạy được trên CPU.
- Git: 1 commit/nhánh cho mỗi milestone (`M3: assigner + loss (tests pass)`). `.gitignore` chặn `data/`, `outputs/`, `*.pth`, `.env`. Quét secrets trước khi commit; không commit khi test chưa pass.
- Dừng tại biên milestone; khi tài liệu mâu thuẫn hoặc thiếu thông tin: dừng, nêu 2 phương án, chờ chọn.
- **Mẫu báo cáo cuối milestone:** (1) Đã làm gì, (2) File đã đổi, (3) Test/lệnh & kết quả thật, (4) Tiêu chí hoàn thành, (5) Chưa làm/lỗi, (6) Điều cần xác nhận.
- **AI usage:** Cập nhật `docs/ai-usage.md` sau mỗi milestone; giải thích ngắn logic Assigner, Loss, Decoder trước khi merge.
