# Triển khai và cách tiếp tục

## Kết quả chạy thật version 5

Kaggle COMPLETE sau khoảng 62 phút; refit mDeBERTa 256 CE thường đủ 3 epoch trên 14.000 mẫu. CSV local 6.000 dòng được kiểm tra ID/hash/xác suất; dev 0,949005 và audit trước refit 0,935382 được tính lại khớp. Bằng chứng `artifacts/kaggle-v5-output/result-verification.json`; [báo cáo](../reports/KAGGLE_V5_RESULTS.md). Chưa xác nhận tải đủ checkpoint local; đọc monitor-status. Không gửi submission.

## Bản đồ mã nguồn

| File | Trách nhiệm |
|---|---|
| `c1/data.py` | Kiểm tra dữ liệu/hash, dò gần trùng, nhóm và manifest split |
| `c1/training.py` | Tokenize head–tail, train/refit, AMP/OOM, checkpoint, inference |
| `c1/metrics.py` | Xác suất/ID, Macro F1 và lát cắt, ensemble/hệ số, kiểm tra CSV |
| `c1/pipeline.py` | Tiến trình GPU, baseline/biến thể, khóa phương án, audit/refit/fallback; mặc định không cắt theo thời gian |
| `c1/__main__.py` | CLI `prepare`, `train`, `evaluate`, `predict`, `validate-submission`, `run` |
| `tools/build_notebook.py` | Nhúng package hiện tại vào notebook độc lập, không chứa CSV/credentials |
| `tools/check_tokenizers.py` | Kiểm tra tokenizer/config chính thức và ghi revision |
| `tools/monitor_kaggle.py` | Theo dõi phiên, tải artifacts/checkpoint cuối, kiểm tra CSV; chỉ đọc Kaggle, không gửi submission |
| `RMIT_SGS_Sigma69_C1.ipynb` | Notebook Kaggle có cấu hình, cài dependencies, chạy và kiểm tra đầu ra |
| `kaggle/kernel-metadata.json` | Định danh notebook private và cấu hình triển khai |

Notebook giải nén vào `/kaggle/working/c1-project`; kết quả tại `/kaggle/working/c1-output`. Log chi tiết worker ở `c1-output/runs/`; log API Kaggle có thể chưa thể hiện tiến độ worker.

## Bằng chứng kiểm tra đã có

- Full suite gần nhất sau bỏ giới hạn thời gian: **53 passed**; gồm train/save/reload model nhỏ thật trên CPU, chuẩn bị dữ liệu, khóa audit, refit, inference tiến trình riêng, xuất CSV và collector. Reviewer kiểm tra độc lập 13 test liên quan đều qua.
- Notebook được kiểm tra payload đúng bằng mã nguồn hiện tại và biên dịch mọi cell Python không có magic. Kiểm tra notebook riêng gần nhất: 1 passed.
- Đã sửa và có regression cho overflow AMP (skip optimizer/LR đúng), lưu checkpoint gián đoạn, baseline đầu thất bại và inference lỗi tái sử dụng nhầm file NPZ cũ.
- Checkpoint được xuất bản theo generation bất biến; chỉ cập nhật con trỏ kết quả sau khi lưu thành công. Inference dùng tên generation mới và yêu cầu process thành công.
- Ba hash gốc khớp sau triển khai; `compileall` và `git diff --check` đã qua ở phiên trước. Đây không phải bằng chứng train T4 thành công.
- Tokenizer/config XLM-R và mDeBERTa thật đã được tải/kiểm tra; revision lần kiểm tra: XLM-R `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, mDeBERTa `a0484667b22365f84929a935b5e50a51f71f159d`. Runtime lưu revision thực dùng riêng.

## Môi trường

- Workspace: `E:\RMIT\RMIT_SGS_Sigma69_C1`, shell PowerShell; dùng `.\.venv\Scripts\python.exe` và `.\.venv\Scripts\kaggle.exe`.
- Python local 3.14.6, torch 2.14.1+cpu, transformers 4.57.6; Kaggle CLI 2.2.4. Python alias WindowsApps không dùng được trong phiên trước.
- Máy local có RTX 4050 Laptop 6 GB nhưng môi trường test là CPU. GPU thực nghiệm theo quyết định của người dùng là Kaggle T4 ×2.
- Kaggle đã xác thực OAuth; không đọc/in/lưu nội dung credentials. Lệnh mạng trong sandbox từng bị WinError 10013 và cần cơ chế escalation của công cụ.
- Git đã từ chối tạo branch do không cho sửa metadata; thay đổi đang ở checkout hiện tại, chưa commit/push. Giữ nguyên `PLAN.md` do người dùng cung cấp và demo gốc.
- Không có `.codegraph/` ở lần kiểm tra tạo memory bank. Nếu xuất hiện sau này, tuân thủ [AGENTS.md](../AGENTS.md) trước khi tìm/đọc mã.
- `.gitignore` bỏ qua `.venv/`, `artifacts/`, cache và credentials. Không nhầm artifacts bị ignore với file chưa từng được tạo.

## Lệnh đọc trạng thái / lấy kết quả

```powershell
& '.\.venv\Scripts\kaggle.exe' kernels status phtzone/rmit-sgs-sigma69-c1
& '.\.venv\Scripts\kaggle.exe' kernels logs phtzone/rmit-sgs-sigma69-c1
& '.\.venv\Scripts\kaggle.exe' kernels files phtzone/rmit-sgs-sigma69-c1
& '.\.venv\Scripts\kaggle.exe' kernels output phtzone/rmit-sgs-sigma69-c1 --path artifacts/kaggle-output
& '.\.venv\Scripts\kaggle.exe' quota
```

Không tự đẩy version mới trước khi kiểm tra phiên đang chạy. Hiện tại version 5 dùng `KaggleApi.kernels_push('kaggle', acc='NvidiaTeslaT4')`, không truyền timeout. Metadata đã pin image GPU từng chạy thành công trên UI; version 5 qua API đã xác nhận GPU và Internet hoạt động. Các lần CPU/HTTP 409 cũ không phải trạng thái hiện tại.

## Lệnh kiểm tra local khi thay đổi mã

```powershell
& '.\.venv\Scripts\python.exe' tools/build_notebook.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m c1 validate-submission --data-dir rmit-hackathon-2026 --submission artifacts/kaggle-output/submission.csv
```

Đường dẫn CSV tải xuống phải kiểm tra thực tế trước khi chạy validate. Chỉ build lại notebook khi mã nguồn thay đổi; không cần chạy train hoặc suite nặng cho chỉnh sửa tài liệu đơn thuần.

Artifacts cần đọc sau run: `EXPERIMENT_REPORT.md`, `dev_metrics.json`, `audit_metrics.json`, `selection.json`, `selection.sha256`, `final_models.json`, `environment.txt`, `events.json`, `failure.json` nếu có, `submission.csv` và bản dự phòng. Báo cáo phải nêu rõ dùng checkpoint train split hay refit toàn bộ.

**Không có lệnh nộp bài trong memory bank; chỉ nộp khi người dùng yêu cầu rõ ràng.**
