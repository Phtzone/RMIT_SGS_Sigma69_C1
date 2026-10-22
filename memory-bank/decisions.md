# Mục tiêu, quyền hạn và quyết định

## Mục tiêu

Tối ưu **Macro F1** cho bài phân loại prompt ba nhãn trên Kaggle T4 ×2; tạo notebook, báo cáo và CSV hoàn chỉnh. **Yêu cầu mới ngày 22/10/2026: bỏ quy tắc thời gian, chạy hết ma trận đã chọn.** Không cam kết điểm hoặc thứ hạng trước thực nghiệm.

Người dùng ban đầu yêu cầu nghiên cứu không gồm model, sau đó chấp thuận kế hoạch có nghiên cứu/model. Quyết định mới nhất cho phép dùng các checkpoint pretrained trong kế hoạch. Không thêm dữ liệu ngoài, dịch/sinh dữ liệu hoặc pseudo-label test ở vòng đầu.

## Quy tắc bắt buộc

- Giữ nguyên từng byte ba CSV gốc; không sửa nhãn, xóa dòng hoặc ghi đè dữ liệu.
- Được chuẩn bị mã, chạy/cập nhật notebook, huấn luyện, tạo/kiểm tra CSV, xem kết quả và tải artifacts.
- **Không tự gửi submission bằng CLI/API/trình duyệt/script; không chọn final submission; không thêm auto-submit.** Chỉ thực hiện khi người dùng yêu cầu rõ ràng việc nộp bài. Nguồn: [AGENTS.md](../AGENTS.md).
- Dùng `text` làm đầu vào; `language` để chia tập/báo cáo, `id` để đối chiếu. Không dùng luật gán nhãn cứng từ template hay gần trùng train–test.
- Chỉ chọn cấu hình/checkpoint/ensemble/hệ số trên dev. Khóa phương án trước audit; không mở thêm vòng tìm kiếm theo điểm audit.

## Phương pháp đã triển khai

- Hai baseline: `FacebookAI/xlm-roberta-base` và `microsoft/mdeberta-v3-base`, mỗi tiến trình một GPU.
- Thứ tự nhãn và cột xác suất cố định: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`.
- Full fine-tune; CE; AdamW LR 2e-5, weight decay 0,01, warmup 10%, linear decay, clip 1,0; tối đa 3 epoch, seed 42.
- Batch 8 × accumulation 4 = effective batch 32; FP16 trên CUDA; dynamic padding; giữ nửa đầu/nửa cuối trong 256 token rồi thêm special tokens.
- Đánh giá mỗi nửa epoch. OOM giảm batch 8 → 4 → 2 → 1, giữ effective batch và dùng gradient checkpointing khi cần.
- Hai biến thể tối đa từ pretrained gốc của họ baseline tốt hơn: 512 token hoặc CE trọng số `sqrt(n_max/n_class)` tính từ train split. Chênh dev dưới 0,001 ưu tiên baseline nhanh hơn.
- Giữ ứng viên tốt nhất mỗi họ; thử trọng số XLM-R 0 / 0,25 / 0,5 / 0,75 / 1. Sau đó thử nhân xác suất jailbreak/injection với 0,8 / 1 / 1,25, benign giữ 1. Mỗi bước chỉ giữ nếu tăng dev Macro F1 ít nhất 0,002.
- Audit một lần sau khi khóa; sau đó refit toàn bộ 14.000 mẫu với số epoch đã chọn. Điểm audit thuộc checkpoint trước refit.
- Nếu refit ensemble không hoàn tất toàn bộ thành phần, dùng phương án checkpoint đã đánh giá. Lưu dự phòng từ checkpoint hợp lệ; không tạo nhãn giả khi chưa có model hợp lệ.

## Chế độ thời gian hiện tại

Đã bỏ giới hạn bốn giờ, các mốc 105/150/165/220/240 phút, điều kiện bỏ thử nghiệm theo dự báo và tự ngắt chương trình theo deadline cuộc thi. Notebook dùng `--no-time-limit`; worker nhận `deadline_unix: null`. Một GPU thì chạy tuần tự. Giữ ma trận hai baseline + hai biến thể, tối đa ba epoch mỗi thử nghiệm; không mở tìm kiếm vô hạn. Bộ thu kết quả cũng mặc định chờ đến khi job kết thúc. Quota/giới hạn phiên nền tảng và deadline nộp bài của cuộc thi vẫn tồn tại; không tự nộp bài.

Prompt Guard cần quyền truy cập nên không mặc định chạy. Protect AI detector bị loại vì nhị phân/tiếng Anh/không phát hiện jailbreak; mmBERT và EuroBERT để vòng mở rộng. Nguồn và lý do đầy đủ trong [PLAN.md](../PLAN.md) và [báo cáo dữ liệu](../reports/DATA_REVIEW.md).
