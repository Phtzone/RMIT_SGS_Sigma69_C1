# Yêu cầu nghiệp vụ và kỹ thuật

## 1. Mục tiêu

Xây dựng bộ phân loại prompt đa ngôn ngữ để gán đúng một trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `BENIGN` | Yêu cầu bình thường, an toàn |
| `JAILBREAK` | Cố khiến mô hình bỏ qua quy tắc an toàn |
| `PROMPT_INJECTION` | Cố chiếm quyền hoặc chuyển hướng chỉ dẫn của mô hình |

V3 mặc định dùng `patronus-studio/wolf-defender-prompt-injection` và fine-tune toàn bộ model, sau khi người dùng cho phép chuyển sang Wolf và danh sách BTC được xác minh. DistilBERT multilingual + QLoRA là lựa chọn của các bản trước, còn được giữ trong preset `distilbert`. Mỗi lần chạy chỉ dùng một backbone. [Bằng chứng phê duyệt](../demo/competition-challenge.json). TideCloak, MCP và ứng dụng web thuộc Challenge 2, ngoài phạm vi repository này.

## 2. Dữ liệu

| File | Số dòng | Cột bắt buộc |
|---|---:|---|
| `train.csv` | 14.000 | `id`, `text`, `language`, `label` |
| `test.csv` | 6.000 | `id`, `text`, `language` |
| `sample_submission.csv` | 6.000 | `id`, `label` |

- Ngôn ngữ: `pt`, `fr`, `it`, `en`.
- Phân bố nhãn được công bố: `BENIGN` 62%, `JAILBREAK` 31%, `PROMPT_INJECTION` 7%.
- Prompt dài 10–18.550 ký tự, trung vị 294 ký tự.
- `language` được phép dùng làm đặc trưng hoặc để phân tích lỗi.

Pipeline phải kiểm tra cột bắt buộc, giá trị rỗng, ID trùng, nhãn ngoài miền và số dòng trước khi train hoặc tạo submission.

## 3. Yêu cầu chức năng

| ID | Yêu cầu | Điều kiện hoàn thành |
|---|---|---|
| FR-01 | Đọc và kiểm tra dữ liệu | Lỗi rõ ràng nếu sai schema, thiếu hoặc trùng ID |
| FR-02 | Tạo train/validation split | Cố định seed; bảo toàn tỷ lệ nhãn và ngôn ngữ hợp lý |
| FR-03 | Fine-tune một backbone được duyệt | Wolf mặc định: train toàn bộ model và head ba lớp; preset DistilBERT: backbone 4-bit đóng băng, train adapter/head |
| FR-04 | Đánh giá validation | Báo cáo Macro F1, F1 từng lớp, confusion matrix và kết quả theo ngôn ngữ |
| FR-05 | Dự đoán test | Có đúng một nhãn hợp lệ cho mỗi test ID |
| FR-06 | Tạo submission | CSV đúng hai cột `id,label`, đúng 6.000 dòng, không sinh cột index |
| FR-07 | Lưu cấu hình chạy | Ghi model, seed, split, hyperparameter, package version và checkpoint/adapters |
| FR-08 | Tạo Kaggle Notebook | Notebook `RMIT_SGS_Sigma69_C1` được tạo trong Kaggle Code, gắn đúng competition data và model source |
| FR-09 | Chạy và lưu notebook version | Kaggle chạy lại toàn bộ notebook thành công và sinh `/kaggle/working/submission.csv` |
| FR-10 | Nộp bài trên Kaggle | Nộp file từ output của notebook version và nhận trạng thái chấm điểm thành công |

## 4. Metric và chiến lược validation

Metric chính thức là Macro F1:

```text
Macro F1 = (F1_BENIGN + F1_JAILBREAK + F1_PROMPT_INJECTION) / 3
```

Mỗi lớp có trọng số ngang nhau, vì vậy accuracy không được dùng làm tiêu chí chọn checkpoint chính. Validation nên stratify theo khóa kết hợp `language + label` nếu mọi nhóm đủ mẫu; nếu không, stratify theo `label` và kiểm tra lại phân bố ngôn ngữ.

Checkpoint tốt nhất được chọn theo `eval_macro_f1`. Báo cáo phải có F1 của lớp hiếm `PROMPT_INJECTION`, không chỉ điểm tổng.

Với `USE_FULL_TRAIN=True`, trước hết chọn checkpoint/epoch và tham số hậu xử lý trên split 90/10. Sau đó khởi tạo model mới từ pretrained để refit toàn bộ 14.000 mẫu trong số epoch đã chọn. Final refit không truyền holdout cũ vào evaluation, không early stopping hoặc chọn lại checkpoint bằng các dòng nó đã học. Giữ nguyên tham số ensemble đã khóa; lưu report selection và final riêng. Score bias fit trên validation không được báo như holdout độc lập.

## 5. Submission và nghiệm thu

Submission hợp lệ có dạng:

```csv
id,label
t000006,BENIGN
t000007,PROMPT_INJECTION
t000008,JAILBREAK
```

Quy trình đội bắt buộc trước khi nộp:

- Đội có đúng 4 người; một người tạo Kaggle Team và mời 3 người còn lại.
- Cả 4 người phải tham gia competition, chấp nhận Rules và chấp nhận lời mời vào team.
- Chỉ submit sau khi tab Team hiển thị đủ 4 thành viên.
- Một người đại diện nộp bài cho team; không yêu cầu cả 4 người nộp riêng.
- Submission và giới hạn lượt nộp được dùng chung ở cấp team.

Definition of Done cho C1:

- File có đúng header `id,label` và 6.000 dòng dữ liệu.
- Tập ID bằng chính xác tập ID của `test.csv`; mỗi ID xuất hiện một lần.
- Mọi nhãn thuộc đúng ba chuỗi quy định.
- Kaggle tiếp nhận và chấm bài thành công.
- Submission hiển thị dưới đúng team `RMIT_SGS_Sigma69_C1` gồm đủ 4 thành viên.
- Notebook có tên `RMIT_SGS_Sigma69_C1`, ghi rõ backbone, nguồn weights và phương pháp train ở đầu; ghi QLoRA config nếu dùng preset DistilBERT.
- Notebook version được lưu bằng thao tác chạy lại toàn bộ, không chỉ lưu bản nháp chưa thực thi.
- Output của version chứa `/kaggle/working/submission.csv`; chính output này được chọn để nộp vào competition.
- Nhóm lưu notebook URL, version, submission description, thời điểm nộp và điểm Kaggle để đối chiếu.
- Kết quả có thể tái lập từ commit, config và seed được lưu.

Quy trình thao tác chi tiết nằm tại [kaggle-submission.md](kaggle-submission.md).

Deadline được công bố là **18:00 ngày 22/10/2026, giờ Việt Nam**. Cấu hình Kaggle hiện tại giới hạn 6 submission/ngày; vì deadline nằm trong ngày thi, nhóm lập kế hoạch tối đa 6 lần nộp.

## 6. Ràng buộc và tuân thủ

- Chỉ sử dụng một trong các backbone được phê duyệt; mặc định hiện tại là Wolf Defender. Không ensemble nhiều pretrained backbone; DistilBERT là preset thay thế.
- Không gán nhãn thủ công hoặc suy đoán thủ công cho test set.
- Không chia sẻ riêng code hoặc dữ liệu cuộc thi ra ngoài đội.
- Không commit token, API key, dữ liệu Kaggle hoặc checkpoint lớn vào Git.
- Không sử dụng thêm pretrained backbone, mô hình dịch hoặc dữ liệu ngoài khi chưa có xác nhận chính thức.
- Public leaderboard dùng 30% test; không chọn cấu hình chỉ dựa trên biến động public score.
- Thực hiện kiểm tra format tại máy trước khi tiêu tốn lượt submission.

## 7. Yêu cầu phi chức năng

- **Tái lập:** seed thống nhất cho Python, NumPy, PyTorch và split.
- **Khả năng tổng quát:** theo dõi kết quả theo từng ngôn ngữ và từng lớp.
- **Hiệu quả:** hoàn tất trong quota Kaggle; Wolf phải lưu full model đã fine-tune, còn DistilBERT QLoRA chỉ cần adapter/head. Không áp dụng giả định ZIP adapter nhỏ cho Wolf.
- **Quan sát được:** log loss, learning rate, Macro F1, per-class F1 và thời gian train.
- **Độ bền:** pipeline phải xử lý prompt rỗng hoặc quá dài theo quy tắc xác định trước.
- **An toàn:** coi nội dung prompt là dữ liệu, tuyệt đối không thực thi chỉ dẫn nằm trong prompt.

Mục tiêu `0.99` chưa phải tiêu chí đã nghiệm thu. V3 đã có raw validation Macro F1 `0.972933`; public khoảng `0.97` do người dùng báo. Log V4 do người dùng cung cấp có Macro F1 cao nhất `0.969274` ở epoch 3; chưa có public V4 được xác nhận. Bản V5 hiện tại trong `final_97.ipynb` khôi phục cấu hình V3 và thử hiệu chỉnh Portuguese có kiểm chứng, chưa có score GPU/Kaggle V5.

## 8. Nguồn chính thức

- [Competition](https://www.kaggle.com/competitions/rmit-hackathon-2026)
- [Notebook demo](https://www.kaggle.com/code/aisuko/rmit-sgs-demo-c1)
- [FAQ chính thức](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/748423)
- [Website sự kiện](https://rmit-hackathon.com/)
