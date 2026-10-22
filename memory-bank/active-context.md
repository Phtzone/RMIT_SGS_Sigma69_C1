# Ngữ cảnh đang làm việc

## Trạng thái hiện tại

- Kaggle version 5 `phtzone/rmit-sgs-sigma69-c1` (kernel 140144687) đã **COMPLETE**, khoảng 62 phút, từ 12:07:57 đến khoảng 13:10 UTC+7 ngày 22/10/2026. T4 ×2, Internet hoạt động. Không giới hạn thời gian pipeline.
- [Báo cáo kết quả](../reports/KAGGLE_V5_RESULTS.md). Yêu cầu mới: tư vấn bước tối ưu để thắng giải. [Lộ trình đề xuất](../reports/C1_NEXT_STEPS.md) chưa triển khai hoặc khởi chạy.
- Model thắng mDeBERTa-v3-base 256 token đầu–cuối, CE thường, 3 epoch, LR 2e-5, batch 8 × accumulation 4, seed 42, FP16.
- Dev thô 0,946733; sau hệ số BENIGN/JAILBREAK/INJECTION **1 / 0,8 / 1,25**, dev **0,949005**. Audit **0,935382** đo model train trên split trước refit, không phải model cuối.
- Refit toàn bộ 14.000 mẫu đủ 3 epoch trong 18,58 phút. Checkpoint `/kaggle/working/c1-output/runs/mdeberta_full/final`; revision `a0484667b22365f84929a935b5e50a51f71f159d`.
- **Không gửi hoặc chọn final submission** nếu chưa được yêu cầu rõ ràng. Xem [AGENTS.md](../AGENTS.md). Chưa xác nhận điểm leaderboard của CSV v5.

## Kết quả cuối và kiểm chứng

| Run | Dev Macro F1 thô | Wall phút |
|---|---:|---:|
| XLM-R 256 | 0,926540 | 9,49 |
| mDeBERTa 256 | 0,946733 | 15,65 |
| mDeBERTa 512 | 0,938939 | 23,86 |
| mDeBERTa weighted 256 | 0,942487 | 15,12 |

- Ensemble tăng 0,000794 (<0,002), không giữ. Các số checkpoint trung gian đã lỗi thời.
- CSV `artifacts/kaggle-v5-output/c1-output/submission.csv`: 6.000 dòng hợp lệ, đúng ID/thứ tự; BENIGN 3.859 / JAILBREAK 1.780 / INJECTION 361.
- SHA-256 CSV `36146d8ac0f8d5ae219c4701378c6f3ecc603849f755a9fddc9260db7c0332b4`.
- [result-verification.json](../artifacts/kaggle-v5-output/result-verification.json): tính lại dev/audit và confusion matrix khớp; CSV khớp toàn bộ xác suất refit; 12 NPZ hợp lệ; seal selection hợp lệ; ba hash gốc không đổi; không giao nhóm; 262 cặp train–train không vượt split.
- Audit injection F1 0,89130, recall 82% (bỏ sót 27/150). Injection dài >1.024 ký tự đúng 5/14; 11 injection gần trùng đều nhầm benign. Audit đã mở; không dùng nó tune rồi gọi là đánh giá độc lập mới.

## Split và môi trường

- Manifest Kaggle thực tế `artifacts/kaggle-v5-output/c1-output/prepared/manifest.csv`: **9.808 / 2.096 / 2.096**, đủ 12 tổ hợp nhãn–ngôn ngữ mỗi phần.
- Manifest local cũ khác 6.477 ID về split, group ID giống. Kaggle scikit-learn 1.6.1 / NumPy 2.1.3; local 1.9.1 / 2.5.4. Chưa xác minh riêng nguyên nhân; vòng sau dùng manifest lưu sẵn.
- Kaggle Python 3.13.15, torch 2.11.0+cu128, transformers 4.57.6. Metadata pin GPU image `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`.
- Full suite trước push v5 53 passed; reviewer 13 passed. Lần báo cáo không đổi code train.
- PowerShell pipe Unicode vào Python đã làm mất dấu tài liệu mới; đã sửa. Dùng apply_patch/file UTF-8 hoặc JSON ASCII escapes khi đưa văn bản qua stdin. CSV và JSON số liệu không bị ảnh hưởng.

## Thu artifacts

- Watcher `tools/monitor_kaggle.py`; PID trong `artifacts/kaggle-v5-output/monitor.pid` (restart 27240). Log mới `monitor-restart.stdout.log` / `monitor-restart.stderr.log`; trạng thái `monitor-status.json`.
- `remote_status=COMPLETE, stage=waiting` có thể đang tải checkpoint; kiểm tra process và `.part`. Trọng số có trên Kaggle, chưa xác nhận bản local đầy đủ. CSV/báo cáo local đã có.
- Watcher cũ 16640 dừng sau 10 ConnectionError lúc 12:27; restart lúc 13:00. Log không có chữ restart là lịch sử cũ. Collector chỉ đọc, không nộp bài.
- v2/v3 lỗi CPU/mạng trước train; v4 UI có GPU. Ảnh Draft CPU cũ không phản ánh v5.
- API version_label là nhãn, không phải số phiên; dùng metadata.current_version_number trước/sau để bảo vệ việc tải.

## Bước tiếp theo

1. Giữ nguyên baseline/CSV gốc. Không chạy lại chỉ để xem kết quả.
2. Lộ trình tuning là đề xuất; chưa tăng epoch/pruning/cache/DDP trong code, chưa khởi chạy job mới.
3. Deadline API kiểm tra lại: **18:00 UTC+7 ngày 22/10/2026**. Không biến thành tự ngắt pipeline hoặc tự nộp.
4. Tách mục tiêu top C1 và giải tổng thể. README ghi nội dung Challenge do người dùng cung cấp: bài hợp lệ đầu tiên đạt đầy đủ điểm C1; cần đối chiếu rubric mới nhất trước phân bổ toàn bộ công sức vào leaderboard.
5. Chỉ nộp hoặc chọn final submission khi người dùng yêu cầu rõ ràng.
