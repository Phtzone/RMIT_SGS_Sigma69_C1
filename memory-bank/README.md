# Memory bank — RMIT_SGS_Sigma69_C1

Ngày lưu ngữ cảnh: **22/10/2026**, múi giờ **Asia/Bangkok (UTC+7)**.

Đây là bản bàn giao ngữ cảnh để tiếp tục công việc ở phiên mới. Trạng thái Kaggle trong tài liệu là **lần quan sát gần nhất**, không phải trạng thái trực tiếp.

## Thứ tự đọc

1. [AGENTS.md](../AGENTS.md): quy tắc của dự án, đặc biệt quyền gửi submission.
2. [active-context.md](active-context.md): trạng thái hiện tại và việc tiếp theo.
3. [decisions.md](decisions.md): mục tiêu, phạm vi được phép và phương pháp đã chọn.
4. [data-context.md](data-context.md): dữ liệu, hash và cách chia tập.
5. [implementation.md](implementation.md): mã nguồn, kiểm thử, môi trường và lệnh tiếp tục.

## Cách duy trì

- Cập nhật `active-context.md` sau khi kiểm tra Kaggle hoặc có kết quả thực nghiệm mới; ghi thời điểm và nguồn kiểm chứng.
- Phân biệt việc đã triển khai, đã kiểm thử CPU và đã chạy thực tế trên GPU. Không suy ra điểm số từ kế hoạch hay model card.
- Không lưu token, API key, OAuth, nội dung credentials hoặc dữ liệu CSV gốc trong thư mục này.
- Một yêu cầu chạy notebook hoặc tạo CSV **không phải** yêu cầu nộp bài. Chỉ gửi submission/chọn final submission khi người dùng yêu cầu rõ ràng.
- Khi nội dung ghi nhớ cũ mâu thuẫn với trạng thái thực tế, kiểm tra lại file/artifacts/API và cập nhật; không coi bản ghi nhớ là bằng chứng thời gian thực.

Tài liệu chi tiết: [kế hoạch](../PLAN.md), [runbook](../RUNBOOK.md), [nghiên cứu](../RESEARCH_C1.md), [báo cáo dữ liệu](../reports/DATA_REVIEW.md), [biên bản triển khai](../IMPLEMENTATION.md).
