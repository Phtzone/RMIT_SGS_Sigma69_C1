# Phân công bốn thành viên và chiến lược branch

## 1. Quy ước Git

- `main`: chỉ chứa phiên bản đã tích hợp và có khả năng tạo submission.
- `develop`: branch tích hợp của đội, tạo từ `main`.
- Mỗi thành viên tạo feature branch từ `develop` và mở pull request về `develop`.
- Không push trực tiếp vào `main` hoặc force-push branch dùng chung.
- Mỗi PR phải ghi cách chạy, đầu ra và kiểm tra đã thực hiện.

Tên thành viên chưa được cung cấp nên bảng dùng `Thành viên 1–4`; thay bằng tên thật nhưng giữ nguyên branch và phạm vi.

Trên Kaggle, một trong bốn thành viên giữ vai trò Team Leader, tạo `RMIT_SGS_Sigma69_C1` và mời đúng ba người còn lại. Thành viên 4 phụ trách submit sau khi Team Leader xác nhận tab Team đã hiển thị đủ bốn người.

## 2. Phân công

| Thành viên | Vai trò | Branch | Công việc chính | Đầu ra bắt buộc |
|---|---|---|---|---|
| Thành viên 1 | Data & preprocessing owner | `feat/data-pipeline` | EDA; kiểm tra schema; split; tokenizer; xử lý prompt dài; data collator | Module data dùng chung, EDA report, split/config cố định |
| Thành viên 2 | Model training owner | `feat/distilbert-qlora` | Wolf full fine-tuning mặc định; giữ preset DistilBERT QLoRA; train loop; lưu model/adapter; smoke test GPU | Model factory, training entry point, checkpoint và log |
| Thành viên 3 | Evaluation & experiment owner | `feat/evaluation-tuning` | Macro F1; per-class/per-language metrics; weighted/focal loss; confusion matrix; thí nghiệm | Metric/loss module, bảng thí nghiệm, cấu hình tốt nhất có căn cứ |
| Thành viên 4 | Inference, Kaggle submission & integration owner | `feat/inference-submission` | Test inference; submission validator; tạo Kaggle Notebook; lưu version; nộp bài; tích hợp PR | `submission.csv` hợp lệ, Kaggle Notebook/version URL, submission được chấm điểm |

## 3. Chi tiết nhiệm vụ

### Thành viên 1 — `feat/data-pipeline`

- Kiểm tra số dòng, cột, null, duplicate ID và miền nhãn.
- Báo cáo phân bố `language × label` và độ dài text.
- Tạo split tái lập với seed `42`.
- Cung cấp interface chung:

```python
load_data(data_dir) -> train_df, test_df
make_splits(train_df, seed) -> train_df, valid_df
build_dataset(df, tokenizer, with_labels) -> Dataset
```

- So sánh truncation đầu chuỗi và head-tail, nhưng không tự chọn phương án cuối nếu chưa có metric từ Thành viên 3.

### Thành viên 2 — `feat/distilbert-qlora`

- Giữ tên branch đã phân công; V5 trong `final_97.ipynb` mặc định dùng Wolf Defender theo cấu hình train V3 đã có kết quả. Backbone đã được BTC duyệt và người dùng chấp nhận. Cấu hình cụ thể theo [architecture.md](architecture.md).
- Wolf: fine-tune toàn bộ model/head ba lớp, mixed precision FP16 trên T4. DistilBERT là preset thay thế dùng NF4, double quantization và QLoRA; chỉ train adapter/head.
- Xác nhận model chạy được ít nhất một batch trên Kaggle T4.
- Lưu full model với Wolf hoặc adapter/head với DistilBERT, kèm config và trainable parameter report.
- Ghi rõ preset đã chọn cho cả nhóm; không trộn hai pretrained backbone trong một giải pháp.

### Thành viên 3 — `feat/evaluation-tuning`

- Dùng `f1_score(..., average="macro")` làm metric chọn model.
- Báo cáo F1 cho cả ba lớp, confusion matrix và F1 theo ngôn ngữ.
- So sánh tối thiểu: loss thường, weighted cross entropy và focal loss nếu còn thời gian.
- Ghi từng thí nghiệm vào bảng: commit, config, seed, validation Macro F1, thời gian và ghi chú.
- Không dùng public leaderboard để thay thế validation nội bộ.

### Thành viên 4 — `feat/inference-submission`

- Nạp model/adapter đúng cấu hình đã chọn và predict đủ test set.
- Giữ nguyên ID và tạo đúng hai cột `id,label`.
- Viết validator dừng ngay nếu sai 6.000 dòng, ID hoặc label.
- Tích hợp notebook `RMIT_SGS_Sigma69_C1` chạy từ data tới submission.
- Kiểm tra notebook ghi rõ backbone và phương pháp train ở đầu; ghi QLoRA config nếu dùng preset DistilBERT.
- Tạo notebook thật trên Kaggle Code, gắn competition data và model source tương ứng preset (mặc định Wolf).
- Lưu version bằng chế độ chạy lại toàn bộ và kiểm tra `/kaggle/working/submission.csv` trong output.
- Chỉ nộp bài sau khi một thành viên khác kiểm tra version, file output và tên team.
- Ghi notebook URL, version, submission description, thời điểm nộp, trạng thái và điểm vào submission log.
- Quản lý tối đa 6 lượt nộp: chỉ submit artifact đã qua validator và được đội duyệt.

## 4. Hợp đồng giữa các branch

Các branch phải dùng chung:

```text
MODEL_PRESET = wolf
MODEL_ID = patronus-studio/wolf-defender-prompt-injection
LABELS = [BENIGN, JAILBREAK, PROMPT_INJECTION]
SEED = 42
PRIMARY_METRIC = macro_f1
SUBMISSION_COLUMNS = [id, label]
```

Nếu chọn preset `distilbert`, model ID là `distilbert/distilbert-base-multilingual-cased`; các branch dùng cùng preset/config cho một lần chạy. Tên branch lịch sử được giữ nguyên.

Thành viên 1 chốt schema dataset trước. Thành viên 2 và 3 không tự tạo data format khác. Thành viên 4 chỉ tích hợp API công khai từ các module, không copy code sang notebook trước khi các PR được review.

## 5. Thứ tự merge

1. `feat/data-pipeline`: merge sớm để khóa schema và split.
2. `feat/distilbert-qlora`: merge sau khi smoke test một batch thành công.
3. `feat/evaluation-tuning`: merge metric trước; merge lựa chọn loss/config sau khi có baseline.
4. `feat/inference-submission`: merge validator sớm, notebook tích hợp sau cùng.
5. Chạy end-to-end trên `develop`, tạo release candidate, rồi mở PR `develop -> main`.

Thành viên 3 có thể viết metric song song với data pipeline. Thành viên 4 có thể viết submission validator từ schema chính thức trước khi model hoàn tất.

## 6. Quy tắc review và hoàn thành

Một task chỉ hoàn thành khi:

- Có code hoặc tài liệu đúng phạm vi branch.
- Có lệnh chạy hoặc cell hướng dẫn rõ ràng.
- Không chứa secret, data cuộc thi, checkpoint lớn hay đường dẫn máy cá nhân.
- PR đã được ít nhất một thành viên khác review.
- Các kiểm tra liên quan chạy qua.
- Không phá vỡ pipeline end-to-end trên `develop`.

## 7. Kế hoạch submission

| Lượt | Mục tiêu |
|---:|---|
| 1 | Xác nhận pipeline và format bằng baseline hợp lệ |
| 2 | Baseline backbone đã chọn trên toàn bộ train theo cấu hình được duyệt |
| 3 | Loss xử lý mất cân bằng tốt nhất |
| 4 | Max length/head-tail tốt nhất |
| 5 | Hyperparameter tốt nhất theo validation |
| 6 | Final candidate hoặc phương án dự phòng ổn định |

Không bắt buộc dùng đủ sáu lượt. Giữ ít nhất một lượt cho lỗi bất ngờ hoặc final candidate.
