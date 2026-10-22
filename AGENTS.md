# Project instructions

## Submission policy

- Không tự ý gửi submission lên Kaggle hoặc bất kỳ nền tảng chấm thi nào. Chỉ thực hiện khi người dùng yêu cầu rõ ràng việc gửi/nộp bài.
- Quyền huấn luyện, chạy/cập nhật notebook, tạo hoặc kiểm tra `submission.csv`, theo dõi kết quả và tải artifacts không đồng nghĩa với quyền gửi submission.
- Quy tắc này áp dụng cho mọi cách gửi bài: CLI/API, giao diện trình duyệt, script và automation. Không thêm chức năng tự động gửi submission vào pipeline.
- Khi chưa có yêu cầu nộp bài, chỉ chuẩn bị file và báo cáo kết quả để người dùng xem xét. Không tự chọn final submission trên Kaggle.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
