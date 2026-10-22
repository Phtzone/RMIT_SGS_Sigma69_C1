# Chạy C1 trên Kaggle T4 ×2

## Trạng thái

Đã triển khai pipeline, kiểm thử và kết nối Kaggle API với `phtzone`. **Version 5** chạy qua API lúc **12:07 UTC+7 ngày 22/10/2026** với hai Tesla T4, CUDA 12.8 và Internet; log xác nhận `time limits: disabled`. Full suite: **53 passed**. **Chưa có checkpoint/điểm thực nghiệm được xác nhận tại lần cập nhật này.** Trạng thái bàn giao tại [memory-bank/active-context.md](memory-bank/active-context.md).

CSV trong `rmit-hackathon-2026/` được giữ nguyên. Các artifacts được tạo riêng. Theo yêu cầu mới, mặc định chạy hết các thử nghiệm, audit, refit và inference, không có giới hạn thời gian do pipeline đặt ra.

## Cách chạy trên Kaggle

1. Import notebook **[RMIT_SGS_Sigma69_C1.ipynb](RMIT_SGS_Sigma69_C1.ipynb)** vào Kaggle; giữ notebook private trong khi thi.
2. Thêm dữ liệu cuộc thi `rmit-hackathon-2026` qua Add Input. Chọn **GPU T4 ×2**, bật Internet. Đối chiếu Rules, quota và lượt nộp trên tài khoản.
3. Kiểm tra cell cấu hình. Deadline nộp bài là **18:00 ngày 22/10/2026 UTC+7**; đây là thông tin để người dùng quyết định nộp bài, không phải điều kiện dừng chương trình. Quota/giới hạn phiên của Kaggle vẫn áp dụng.
4. Run All. Notebook tự giải nén mã nguồn hiện tại vào `/kaggle/working/c1-project`; không cần upload repo hoặc CSV riêng.
5. Xem log trong `c1-output/runs/`. Khi hoàn tất, cell cuối kiểm tra `c1-output/submission.csv` và hiển thị link tải.
6. Save Version kèm outputs để giữ checkpoint và báo cáo. Chỉ gửi CSV lên cuộc thi hoặc chọn final submission khi người dùng yêu cầu rõ ràng; xem [quy tắc submission](AGENTS.md).

Notebook không tự nộp bài, không sử dụng API key và không chấp nhận Rules thay người dùng. Mỗi lần thực nghiệm dùng OUTPUT_DIR mới để bảo toàn audit của lần trước.

## Chạy từ mã nguồn

### Thu kết quả của phiên Kaggle đang chạy

`tools/monitor_kaggle.py` chỉ đọc trạng thái và tải artifacts, không gửi bài. Bộ thu kiểm tra version hiện hành trước khi đọc/tải; nếu người dùng tạo version mới, bộ thu dừng để tránh nhầm kết quả. Chỉ chạy một bộ thu cho mỗi thư mục đầu ra:

```powershell
.\.venv\Scripts\python.exe tools/monitor_kaggle.py --version 5 --output-dir artifacts/kaggle-v5-output
```

Đọc `monitor-status.json` để xem trạng thái cập nhật mỗi 30 giây. Khi hoàn tất, xem `validation.json`, `COLLECTION_REPORT.md` và báo cáo thí nghiệm tải về; nếu có lỗi, xem `validation-error.json` và log. Máy local cần bật để bộ thu hoạt động, còn job Kaggle chạy độc lập. Checkpoint tải về là các thành phần cuối được chỉ định trong `final_models.json`; checkpoint lịch sử vẫn ở outputs Kaggle.

### Thực thi pipeline

Yêu cầu Python từ 3.11 trở lên. Cài môi trường riêng và dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m c1 prepare --data-dir rmit-hackathon-2026 --output-dir artifacts/prepared
```

Chạy toàn bộ trên môi trường có CUDA, từ thư mục gốc dự án:

```bash
python -m c1 run --data-dir /kaggle/input/competitions/rmit-hackathon-2026 --output-dir /kaggle/working/c1-output --no-time-limit
```

Các lệnh riêng:

```bash
python -m c1 train --config artifacts/runs/xlmr_baseline/config.json
python -m c1 evaluate --data-dir rmit-hackathon-2026 --output-dir artifacts
python -m c1 evaluate --data-dir rmit-hackathon-2026 --output-dir artifacts --audit --gpu-count 2
python -m c1 predict --config artifacts/predict_config.json
python -m c1 validate-submission --data-dir rmit-hackathon-2026 --submission artifacts/submission.csv
```

`train` cần config JSON đã lưu từ một lần chạy hoặc cấu hình theo mẫu bên dưới. `evaluate` mặc định chỉ chọn theo dev; `--audit` yêu cầu `selection.json` đã khóa và không được chạy audit lần thứ hai.

```json
{
  "run_id": "xlmr_baseline",
  "family": "xlmr",
  "model_id": "FacebookAI/xlm-roberta-base",
  "model_revision": "main",
  "data_dir": "rmit-hackathon-2026",
  "prepared_dir": "artifacts/prepared",
  "run_dir": "artifacts/runs/xlmr_baseline",
  "max_length": 256,
  "epochs": 3.0,
  "lr": 0.00002,
  "batch_size": 8,
  "grad_accum": 4,
  "weighted": false,
  "seed": 42,
  "full_train": false,
  "device": "cuda"
}
```

Predict config: `checkpoint`, `data_dir`, `prepared_dir`, `split` (`test`, `audit` hoặc `dev`), `max_length`, `batch_size`, `device`, `output` (file `.npz`). Lệnh predict tạo xác suất và ID, không tự điều chỉnh nhãn.

## Quy trình và điều kiện dừng

- Prepare kiểm tra schema/hash, tạo nhóm train-only và 20 fold; 0–2 audit, 3–5 dev, 6–19 train.
- Hai baseline chạy độc lập, một model mỗi T4; chỉ input `text`. `language` và `id` phục vụ kiểm tra/báo cáo.
- Head–tail truncation giữ đều đầu/cuối trong ngân sách token, dynamic padding, FP16 và effective batch 32.
- Mỗi nửa epoch đánh giá dev; checkpoint và xác suất là một phiên bản bất biến, `result.json` được xuất bản sau khi lưu thành công. Giữ best, best trước và latest để giới hạn dung lượng.
- Sau 100 optimizer steps, `benchmark.json` ghi tốc độ để báo cáo; log tiến độ hiển thị mỗi phút khi chờ worker.
- Chạy cả hai biến thể 512 token và weighted cross-entropy, không loại thử nghiệm theo thời gian dự báo.
- Chọn ensemble với trọng số 0,25/0,5/0,75 và điều chỉnh xác suất lớp trong 0,8/1/1,25; chỉ giữ bước tăng dev Macro F1 ít nhất 0,002. Hai đầu 0/1 tương ứng model đơn.
- Khóa phương án bằng SHA-256; audit chỉ để báo cáo. Điểm audit không phải điểm của model refit toàn bộ dữ liệu.
- Full refit dùng số epoch đã chọn. Chỉ dùng ensemble refit khi mọi thành phần hoàn tất; nếu không, giữ toàn bộ phương án checkpoint đã đánh giá.
- Nếu bước sau thất bại mà có submission dự phòng đã kiểm tra, xuất nó thành `submission.csv`, lưu nguyên nhân trong `failure.json`. Nếu chưa có checkpoint hợp lệ, dừng với lỗi thay vì tạo nhãn giả.

Chế độ mặc định không đặt deadline cho worker hoặc inference và không bỏ qua refit theo thời gian. Các tùy chọn `--budget-minutes` / `--competition-deadline` chỉ còn để tương thích khi người dùng chủ động yêu cầu giới hạn mới; notebook hiện không dùng chúng. Có inference dự phòng trước refit để giữ CSV hợp lệ nếu bước sau lỗi.

## Artifacts

| Artifact | Nội dung |
|---|---|
| `prepared/manifest.csv` | ID, nhóm, fold và split đã chốt |
| `prepared/data_report.json`, `DATA_REPORT.md` | Số liệu, phương pháp và hạn chế của phép dò gần trùng |
| `prepared/data_hashes.json` | SHA-256 ba CSV gốc |
| `runs/*/config.json`, `result.json`, `train.log` | Cấu hình và kết quả từng lần chạy |
| `runs/*/token_statistics.json` | Độ dài tokenizer trên train/dev; không dùng audit để chọn cấu hình |
| `runs/*/checkpoint-*/` | Weights, tokenizer và xác suất dev khớp checkpoint |
| `selection.json`, `selection.sha256` | Chính sách được khóa trước audit |
| `dev_metrics.json`, `audit_metrics.json`, `dev_errors.csv` | Macro F1, từng lớp/ngôn ngữ, confusion matrix và tối đa 60 lỗi dev |
| `final_models.json` | Checkpoint cuối và trạng thái full-data refit |
| `environment.txt` | Phiên bản thư viện trong phiên chạy Kaggle |
| `EXPERIMENT_REPORT.md`, `events.json`, `failure.json` | Báo cáo, các bước bỏ qua và lỗi nếu có |
| `submission.csv`, `submission_fallback.csv` | Bài nộp chính và dự phòng |

Khi sửa mã nguồn, tạo lại notebook trước khi import:

```bash
python tools/build_notebook.py
python -m pytest -q
```

Phép dò gần trùng là gần đúng, có thể bỏ sót bản dịch/paraphrase. Chia theo nhóm giảm rò rỉ đã quan sát, không bảo đảm mọi nhóm ngữ nghĩa đã được tách. Không có điểm validation hoặc leaderboard nào được suy ra từ paper/model card.
