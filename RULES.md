# RULES

## 1. Definition of Done
### A. Task chỉ hoàn thành khi có đầy đủ các kỹ thuật:
- Đủ yêu cầu (đủ các yêu cầu chức năng).
- Code chạy.
- Test passed.
- No secrets.
- Code review.
- GitHub.
- Docs updated.
- Committed.
- Committed kh dc hiển thị trên git.

→ Viết code xong phải kiểm tra, đánh giá thay vì chỉ thấy chạy được là hoàn thành.

### Implement – Plan
- Code trước, nhưng nếu mục tiêu lớn thì cần có các bước, ghi chú.
- Progress status: md.

---

## 2. Các mục trong báo cáo
- Example In-out.
- Results.
- Limitations.
- Future Improvements.

---

## 3. AI Usage – md
- Tools used.
- Main purposes.
- Student contributions.
- Important AI-generated components.
- Prompt purpose.
- Student modification.
- Validation.
- Hội thoại / [không rõ] / [không rõ].

---

## 4. Một số lưu ý khi dùng AI
- Không để AI đẩy code mà không chạy thử hoặc không hiểu.
- Tạo test làm pipeline “passed”.
- Không đưa API key/password lên GitHub.
- Không commit .env, token hoặc secrets.
- Tạo rule check lại.
- Không thay đổi báo cáo / SD? [không rõ].
- Không sửa kết quả đánh giá theo kiểu [không rõ].

---

## 5. Prompt tốt
- Context nhỏ + mục tiêu + tiêu chí chung.

---

## 6. Plan
- Mục tiêu + In/Out + Feature.
- Constraints → Architecture + Modules + Milestones + Risks → Chia viết code.

---

## 7. Implement
- Module + code + In/Out + Interface + Constraints.

---

## 8. Usage và Document
- **README.md**:
  - Project overview.
  - Problem statement.
  - Main feature.
  - System architecture.
  - Folder structure.
  - Installation / Run / Test.
  - Model [phần tiếp theo không rõ trong ảnh].
  - Knowledge distillation.
- `Project description.md`
- `Project design.md`

---

## 9. Build MVP – Implement module by module
- MVP trước.
- Nhận input → thực hiện chức năng chính → kiểm tra xem có đủ tính năng hay cần thêm tính năng phụ.
- Một module / lần.
- Xác định mục tiêu, input, output, interface rõ ràng trước khi sửa code.
- Một thay đổi / lần.
- Đừng sửa nhiều module cùng lúc để tránh quan trọng hơn việc [không rõ] và prompt.
- **Quy trình:**
  `Describe → Generate → Review → Run (Smoke test) → Dry-run → Test → Refactor → Commit`

---

# MASTER RULES — QUY TẮC CHUNG CHO MỌI DỰ ÁN

## 1. Nguyên tắc làm việc
- Đọc kỹ toàn bộ yêu cầu, tài liệu, đề bài, rubric và mã nguồn hiện có trước khi thực hiện.
- Xác định rõ mục tiêu, phạm vi, đầu vào, đầu ra, ràng buộc kỹ thuật và tiêu chí hoàn thành.
- Ưu tiên nội dung được cung cấp bởi người dùng, đề bài, giáo trình và tài liệu chính thức của dự án.
- Không tự ý thêm yêu cầu, thay đổi mục tiêu hoặc mở rộng phạm vi nếu không có căn cứ.
- Không bịa thông tin, kết quả, số liệu, nguồn trích dẫn, tính năng hoặc trạng thái hoàn thành.
- Phân biệt rõ thông tin từ tài liệu, thông tin đã kiểm chứng, giả định và đề xuất kỹ thuật.
- Khi thiếu thông tin quan trọng, kiểm tra những nguồn có sẵn trước; chỉ hỏi lại khi không thể đưa ra quyết định an toàn và chính xác.
- Không tuyên bố đã đọc, kiểm tra, chạy, sửa hoặc xác minh điều gì nếu thực tế chưa thực hiện.

## 2. Đọc và tuân thủ tài liệu nguồn
- Đọc đúng nội dung nguồn được cung cấp; không chỉ dựa vào tên file, mục lục, trích đoạn hoặc phần tóm tắt khi nhiệm vụ yêu cầu đọc đầy đủ.
- Giữ nguyên thuật ngữ chuyên môn, công thức, ký hiệu, định nghĩa, cấu trúc lập luận và mức độ chi tiết quan trọng của tài liệu.
- Không âm thầm thay thế nội dung nguồn bằng kiến thức bên ngoài, kể cả khi kiến thức bên ngoài có vẻ hợp lý hơn.
- Không bỏ ý, đổi nghĩa, tự hòa giải những điểm mâu thuẫn hoặc bổ sung phần tài liệu không đề cập.
- Nếu cần dùng kiến thức ngoài nguồn, phải phân biệt rõ với nội dung được trích xuất từ nguồn.
- Khi đề bài có rubric hoặc yêu cầu cụ thể, kiểm tra từng tiêu chí tương ứng trước khi kết luận đã hoàn thành.
- Với yêu cầu chính xác theo tài liệu, cần đối chiếu trực tiếp nội dung liên quan; không khẳng định chính xác hoàn toàn nếu chưa đủ căn cứ.

## 3. Quy tắc chỉnh sửa dự án có sẵn
- Trước khi sửa, kiểm tra cấu trúc thư mục, file cấu hình, mã nguồn, tài liệu, test và trạng thái hiện tại của dự án.
- Xác định vấn đề thực sự cần sửa trước khi thay đổi.
- Ưu tiên chỉnh sửa tối thiểu, có mục tiêu và có thể kiểm chứng.
- Giữ nguyên những phần đang hoạt động đúng, không liên quan đến yêu cầu.
- Không xây dựng lại toàn bộ dự án chỉ để xử lý một vấn đề cục bộ.
- Không tự ý đổi tên file, thư mục, hàm, class, biến, API, định dạng đầu vào hoặc đầu ra nếu không cần thiết.
- Không xóa dữ liệu, checkpoint, kết quả thực nghiệm, cấu hình hoặc tài liệu có giá trị.
- Trước khi sửa các thành phần có nguy cơ gây mất dữ liệu hoặc phá vỡ tính tương thích, phải xác định ảnh hưởng và có phương án bảo toàn.
- Kiểm tra các thành phần phụ thuộc trước khi sửa một thành phần dùng chung.
- Nếu dự án chưa có cấu trúc phù hợp, đề xuất hoặc xây dựng cấu trúc dựa trên yêu cầu thực tế; không áp đặt một mẫu kiến trúc cố định cho mọi dự án.

## 4. Quy tắc chất lượng mã nguồn
- Viết code rõ ràng, nhất quán, có thể đọc, bảo trì và kiểm thử.
- Tách các chức năng theo trách nhiệm hợp lý; tránh trùng lặp logic và phụ thuộc không cần thiết.
- Sử dụng tên biến, hàm, class và module thể hiện đúng mục đích.
- Xử lý các trường hợp đầu vào không hợp lệ, lỗi có thể xảy ra và tình huống biên phù hợp với phạm vi dự án.
- Không thêm abstraction, framework, thư viện hoặc tầng kiến trúc chỉ để làm dự án trông phức tạp hơn.
- Không lặp lại code, thêm file hoặc tạo cấu hình không mang lại giá trị rõ ràng.
- Không hardcode thông tin nhạy cảm, mật khẩu, API key hoặc thông tin môi trường.
- Tôn trọng phiên bản ngôn ngữ, thư viện, nền tảng và giới hạn tài nguyên hiện có.
- Giữ tương thích với giao diện và hợp đồng của những thành phần liên quan.
- Không sửa code đang đúng chỉ để tạo ra khác biệt về hình thức.

## 5. Quy tắc dữ liệu và kết quả
- Xác định nguồn gốc, cấu trúc, kiểu dữ liệu, ý nghĩa và giới hạn của dữ liệu trước khi sử dụng.
- Không tự tạo dữ liệu thực nghiệm hoặc sửa số liệu để kết quả trông tốt hơn.
- Không xóa, lọc hoặc biến đổi dữ liệu mà không có lý do phù hợp với mục tiêu.
- Kiểm tra dữ liệu thiếu, trùng lặp, sai kiểu, giá trị bất thường và các vấn đề liên quan khi chúng có ý nghĩa đối với bài toán.
- Giữ lại dữ liệu gốc khi cần thiết để có thể truy xuất và kiểm tra.
- Phân biệt dữ liệu huấn luyện, validation, test hoặc các tập dữ liệu có vai trò khác nhau khi dự án yêu cầu.
- Tránh rò rỉ dữ liệu và các phép biến đổi sử dụng thông tin từ tập đánh giá.
- Không suy diễn quan hệ nhân quả từ tương quan hoặc khẳng định kết luận vượt quá bằng chứng.
- Mọi bảng, biểu đồ và kết luận định lượng phải dựa trên dữ liệu thực tế hoặc nguồn được ghi rõ.

## 6. Quy tắc lựa chọn giải pháp kỹ thuật
- Chọn giải pháp dựa trên bài toán, tài liệu nguồn, yêu cầu, độ phức tạp, tài nguyên và khả năng kiểm chứng.
- Phân biệt yêu cầu bắt buộc với lựa chọn kỹ thuật do người triển khai đề xuất.
- Không mặc định một thuật toán, framework, mô hình hoặc kiến trúc là tốt nhất cho mọi trường hợp.
- Giải thích ngắn gọn lý do chọn giải pháp khi quyết định đó ảnh hưởng đáng kể đến thiết kế.
- Chỉ thêm tối ưu hóa khi có nhu cầu thực tế hoặc bằng chứng về vấn đề cần giải quyết.
- Cân nhắc độ chính xác, khả năng bảo trì, độ phức tạp, hiệu năng và giới hạn môi trường.
- Không thêm tính năng ngoài phạm vi chỉ vì chúng có thể hữu ích.
- Không tự thay đổi phương pháp đã được đề bài yêu cầu bắt buộc.

## 7. Kiểm thử và xác minh
- Xác định cách kiểm tra phù hợp với từng loại dự án.
- Ưu tiên chạy test hiện có trước khi sửa nếu điều kiện cho phép, để biết trạng thái ban đầu.
- Sau khi sửa, chạy các test và kiểm tra có liên quan trực tiếp đến thay đổi.
- Kiểm tra cả đầu vào hợp lệ, trường hợp biên và lỗi có thể xảy ra khi cần thiết.
- Với dự án có nhiều thành phần, kiểm tra sự tương thích giữa các thành phần bị ảnh hưởng.
- Không chỉ dựa vào việc code không báo lỗi cú pháp để kết luận chương trình hoạt động đúng.
- Không bỏ qua lỗi, warning hoặc test thất bại có ý nghĩa.
- Nếu không thể chạy test do thiếu thư viện, quyền truy cập, tài nguyên hoặc môi trường, phải nêu rõ giới hạn.
- Báo cáo chính xác test nào đã chạy, kết quả thực tế và phần nào chưa được xác minh.
- Không tuyên bố mọi thứ hoạt động hoàn hảo khi chỉ kiểm tra một phần.

## 8. Quy tắc tài liệu và văn phong
- Viết tài liệu phục vụ người đọc hiểu dự án, cách hoạt động, quyết định kỹ thuật và kết quả thực tế.
- Dùng câu trực tiếp, rõ ràng, tự nhiên và vừa đủ.
- Ưu tiên giải thích cơ chế, lý do hoặc kết quả cụ thể thay vì các tuyên bố chung chung.
- Loại bỏ câu sáo rỗng, câu lặp ý, phần tự khen dự án và những đoạn không bổ sung thông tin.
- Không lạm dụng các cụm như “đảm bảo tính tối ưu”, “tuân thủ đầy đủ”, “đáp ứng mọi yêu cầu”, “như đã đề cập ở trên” hoặc “dựa trên quy trình đã đối chiếu” khi không cần thiết.
- Không viết câu chỉ để tuyên bố rằng dự án tuân thủ một quy trình, tiêu chuẩn hoặc yêu cầu mà không giải thích thông tin thực chất.
- Không tự nhận xét mô hình, hệ thống hoặc kết quả là tốt, hiệu quả, chính xác hay tối ưu nếu chưa có bằng chứng tương ứng.
- Giữ thuật ngữ nhất quán và giải thích thuật ngữ khi đối tượng đọc cần.
- Không cố tình tạo lỗi ngữ pháp hoặc làm giảm tính chuyên nghiệp để thay đổi phong cách.
- Giữ lại những câu tổng kết, chuyển ý hoặc giải thích nếu chúng thực sự cần thiết cho mạch nội dung.
- Không thêm phần giới thiệu, kết luận, mục lục hoặc tài liệu tham khảo nếu chúng không cần thiết hoặc không được yêu cầu.

## 9. Quy tắc tổ chức dự án
- Tổ chức file theo chức năng và mức độ phức tạp thực tế.
- Phân biệt mã nguồn, dữ liệu, cấu hình, kiểm thử, tài liệu, kết quả và tài nguyên được sinh ra khi phù hợp.
- Tránh cấu trúc quá sâu, quá nhiều file nhỏ hoặc nhiều tầng thư mục không cần thiết.
- Không tạo file trùng lặp hoặc giữ nhiều phiên bản không rõ vai trò.
- Không đưa file tạm, cache, môi trường ảo, dữ liệu lớn hoặc checkpoint không cần thiết vào gói bàn giao.
- Bảo đảm các đường dẫn, cấu hình và hướng dẫn sử dụng nhất quán với cấu trúc thực tế.
- Không giả định một cấu trúc thư mục duy nhất phù hợp với mọi loại dự án.

## 10. Tài nguyên và môi trường thực thi
- Kiểm tra giới hạn phần cứng, bộ nhớ, dung lượng lưu trữ, thời gian chạy và môi trường triển khai.
- Ưu tiên giải pháp có thể thực thi trong môi trường thực tế của người dùng.
- Không mặc định có GPU, Internet, quyền quản trị hoặc khả năng cài đặt mọi thư viện.
- Hạn chế tải lại dữ liệu, tính toán hoặc cài đặt không cần thiết.
- Không xóa môi trường, dữ liệu hoặc tài nguyên đã tạo mà chưa đánh giá ảnh hưởng.
- Nếu cần thay đổi dependency, kiểm tra tính tương thích và chỉ thêm những gói thực sự cần thiết.
- Không tuyên bố đã chạy trên môi trường cụ thể nếu chưa thực sự kiểm tra.

## 11. Bảo mật và độ tin cậy
- Không để lộ thông tin xác thực, dữ liệu riêng tư hoặc thông tin nhạy cảm trong mã nguồn và tài liệu.
- Không thực hiện thao tác phá hủy dữ liệu hoặc thay đổi môi trường có phạm vi rộng mà không đánh giá rủi ro.
- Không chạy lệnh, script hoặc dependency không đáng tin cậy mà không xem xét tác động.
- Kiểm tra các điểm có thể gây lỗi bảo mật phù hợp với loại dự án.
- Không vô hiệu hóa kiểm tra, xử lý lỗi hoặc cơ chế bảo vệ chỉ để khiến test vượt qua.
- Không che giấu lỗi hoặc sửa kết quả đầu ra để tạo ấn tượng rằng hệ thống hoạt động đúng.

## 12. Quy trình thực hiện công việc
Thực hiện theo trình tự phù hợp với nhiệm vụ:
1. Tiếp nhận và phân tích yêu cầu.
2. Đọc tài liệu và kiểm tra workspace.
3. Xác định trạng thái hiện tại và các ràng buộc.
4. Xác định phần cần làm hoặc cần sửa.
5. Lựa chọn phương án phù hợp và tối thiểu.
6. Thực hiện trực tiếp trên các file liên quan.
7. Kiểm tra thay đổi và tính tương thích.
8. Chạy test hoặc xác minh trong phạm vi có thể.
9. Rà soát nội dung, cấu trúc và các vấn đề còn tồn tại.
10. Báo cáo kết quả thực tế.

*Đây là trình tự làm việc chung, không phải danh sách bắt buộc phải tạo thành các file hoặc mục tài liệu riêng trong dự án.*

## 13. Quy tắc báo cáo kết quả
Sau khi hoàn thành, báo cáo ngắn gọn nhưng đủ thông tin:
- Những gì đã thực hiện.
- Những file hoặc thành phần đã thay đổi.
- Các kiểm tra và test đã thực sự chạy.
- Kết quả xác minh.
- Những giới hạn, lỗi hoặc việc chưa hoàn thành.
- Những quyết định cần người dùng xác nhận, nếu có.
- Không liệt kê những việc chưa thực hiện như thể đã hoàn thành.
- Không tự khẳng định dự án hoàn hảo, tối ưu tuyệt đối hoặc đáp ứng mọi tiêu chí khi chưa có bằng chứng.
- Không tạo báo cáo dài dòng chỉ để mô tả lại từng thao tác nhỏ.

## 14. Quy tắc ưu tiên khi có xung đột
Khi các yêu cầu mâu thuẫn nhau, ưu tiên theo thứ tự:
1. An toàn, bảo mật và tính toàn vẹn dữ liệu.
2. Yêu cầu bắt buộc của người dùng và đề bài.
3. Tính chính xác của kiến thức, dữ liệu và kết quả.
4. Khả năng hoạt động đúng và tương thích với hệ thống hiện có.
5. Khả năng kiểm thử, bảo trì và tái sử dụng.
6. Tính đơn giản, rõ ràng và hiệu quả.
7. Sở thích về trình bày hoặc phong cách không ảnh hưởng đến các yêu cầu trên.

*Không âm thầm bỏ qua xung đột quan trọng. Nếu không thể giải quyết mà không thay đổi mục tiêu hoặc vi phạm yêu cầu bắt buộc, hãy nêu rõ vấn đề.*

## 15. Chỉ dẫn cuối cùng
- Trước khi thực hiện bất kỳ thay đổi nào, hãy hiểu trạng thái hiện tại và mục tiêu cần đạt.
- Ưu tiên tính chính xác, khả năng kiểm chứng, thay đổi tối thiểu và kết quả thực tế. Không bịa, không tự ý mở rộng phạm vi, không phá vỡ phần đang hoạt động tốt và không tạo thêm nội dung chỉ để làm dự án trông phức tạp hơn.
- Hãy thực hiện công việc trực tiếp khi đủ thông tin. Chỉ dừng lại để hỏi người dùng khi có vấn đề quan trọng không thể giải quyết an toàn hoặc chính xác bằng thông tin hiện có.

## 16. Danh mục Docs
- **Quy trình:** `description - gen - smoke run - run - check des`
- **Cấu trúc thư mục `/docs`:**
- **Mỗi file md tối đa 200 dòng, tách ra nhiều file hoặc lược bỏ lý thuyết đề trình bày code và ý chính, không được làm hơn :**
- **Giữ lại file md t làm trước đó:** 
```text
/docs
├── DEFINE_AND_DESIGN.md      (hoặc define.md + design.md)
├── report.md                 (báo cáo chính: Example In/Out, Results, Limitations, Future Improvements)
├── data.md
├── model.md
├── training.md
├── evaluation.md
├── demo.md
├── ai-usage.md
├── plan-progress-status.md
└── audit-log.md
```
## 17. QUY TẮC RIÊNG CHO DỰ ÁN (YOLO-style detector trên COCO 2017)

### 17.1 Nguồn sự thật và thứ tự ưu tiên
- Đề bài > RULES.md > docs/DEFINE_AND_DESIGN.md > code.
- Đầu mỗi phiên làm việc: đọc RULES.md và docs/DEFINE_AND_DESIGN.md trước khi làm.
- Muốn đổi thiết kế: sửa tài liệu trước, người dùng duyệt, rồi mới sửa code.
- Dự án này dùng cấu trúc thư mục ở mục 2.3 của thiết kế (áp dụng thay §9 cho dự án này).

### 17.2 Ràng buộc bất biến (chi tiết ở DEFINE_AND_DESIGN)
- Detector one-stage YOLO-style;.
- ResNet-50 (FrozenBN) → FPN P3/P4/P5 → Decoupled Head, ảnh 512×512.
- Tự viết: Head, Assigner, Loss (CIoU, Focal, BCE), Decoder. Chỉ dùng torchvision cho resnet50, FPN, nms, box_iou.
- Công thức giải mã, quy tắc gán ô dương và công thức loss phải đúng mục 2.4.3 và 2.7.

### 17.3 Dữ liệu và val2017
- Train 14k và held-out 1k đều lấy từ train2017, seed cố định, không giao nhau.
- Chọn checkpoint, chỉnh siêu tham số và ablation chỉ dùng held-out 1k.
- val2017 chỉ chạy đúng một lần ở M8. `evaluate.py` phải từ chối chạy val2017 nếu thiếu cờ `--final`,
  đồng thời ghi `outputs/final_eval.lock` và một dòng vào audit log. Không xóa lock để chạy lại.

### 17.4 Kết quả thực nghiệm
- Mọi số liệu (mAP, AP50, loss, FPS) lấy từ file thực tế trong `outputs/` (`metrics.json`, log).
- Mỗi lần chạy lưu kèm: config, seed, commit hash, thời gian chạy.
- Không sửa tay file kết quả hoặc log. Chưa chạy thì ghi "chưa chạy" trong docs.
- Không chỉnh ngưỡng hoặc tham số theo kết quả trên val2017 (đó là rò rỉ dữ liệu).

### 17.5 Tái lập
- Đặt seed cho python, numpy, torch. Lưu config và phiên bản thư viện vào checkpoint.
- `requirements.txt` ghi phiên bản cụ thể.

### 17.6 Kiểm thử và xác minh
- Dùng pytest. Test chạy được trên CPU, thời gian ngắn; test nặng gắn marker `slow`.
- Test bắt buộc: shape từng tầng, ánh xạ category hai chiều, letterbox khép kín (sai số < 1 px),
  Assigner (đúng 3 ô, xử lý biên), CIoU đối chiếu `torchvision.ops.complete_box_iou_loss`.
- Trước M6 phải qua overfit test 10 ảnh (loss về gần 0, recall ≈ 1.0).
- Kiểm tra loss ban đầu hợp lý, không NaN/Inf; dừng và báo nếu gặp NaN.
- Định nghĩa: **smoke run** = chạy 1–2 iteration hoặc ≤100 ảnh; **dry-run** = chạy hết pipeline với
  `--max-images 8`, không ghi checkpoint chính.

### 17.7 Tài nguyên
- Hỏi xác nhận trước khi tải dữ liệu >1 GB hoặc chạy lệnh >30 phút.
- Lưu checkpoint mỗi epoch, hỗ trợ `--resume`. Không xóa checkpoint và kết quả cũ.
- Không giả định có GPU: code phải chạy được (chậm) trên CPU để test.

### 17.8 Git
- Một commit (hoặc nhánh) cho mỗi milestone, message dạng `M3: assigner + loss (tests pass)`.
- `.gitignore` gồm `data/`, `outputs/`, `*.pth`, `.env`. Chạy quét secrets trước khi commit.
- Không commit khi test chưa pass.

### 17.9 Dừng tại biên milestone
- Làm đúng một milestone mỗi lượt, không tự sang milestone tiếp theo.
- Khi đề bài, RULES và thiết kế mâu thuẫn nhau hoặc thiếu thông tin: dừng, nêu 2 phương án, chờ chọn
  (quy tắc này áp dụng thay cho §15 trong dự án này).

### 17.10 Mẫu báo cáo cuối milestone
1. Đã làm gì. 2. File đã đổi. 3. Test/lệnh đã chạy và kết quả thật.
4. Tiêu chí hoàn thành của milestone: tick từng ý kèm bằng chứng.
5. Chưa làm hoặc còn lỗi. 6. Điều cần người dùng xác nhận.

### 17.11 AI usage
- Sau mỗi milestone cập nhật `docs/ai-usage.md`: công cụ, mục đích prompt, phần AI sinh ra,
  phần sinh viên sửa, cách đã kiểm chứng.
- Với Assigner, Loss, Decoder: agent giải thích ngắn logic chính để sinh viên hiểu trước khi merge.