# RMIT_SGS_Sigma69_C1

Tài liệu yêu cầu và quy định tuân thủ cho **Challenge 1 — Multilingual Prompt-Attack Classification**, thuộc RMIT GenAI & Cybersecurity Hackathon 2026.

Tài liệu học trước: [Nội dung ôn tập và bài thi tham khảo C1](RESEARCH_C1.md) — tập trung vào bảo mật prompt, dữ liệu, đánh giá và submission; không nghiên cứu model.

## 1. Nguồn thông tin và phạm vi

- [Trang cuộc thi Kaggle](https://www.kaggle.com/competitions/rmit-hackathon-2026): nội dung Overview, Description, Challenge và Evaluation do người dùng cung cấp.
- [Ảnh hướng dẫn của ban tổ chức](Pic/): deadline, quy tắc đặt tên và hướng dẫn nộp bài.
- [Notebook demo chính thức](https://www.kaggle.com/code/aisuko/rmit-sgs-demo-c1), bản tải tại [demo/rmit-sgs-demo-c1.ipynb](demo/rmit-sgs-demo-c1.ipynb).
- Cập nhật do người dùng xác nhận: được phép sử dụng model ngoài danh sách ban đầu.
- Metadata Kaggle đã kiểm tra trong phiên làm việc: metric, deadline, giới hạn đội, lượt nộp và tỷ lệ public leaderboard.
- [Website hackathon](https://rmit-hackathon.com/) và Discussion của cuộc thi là nơi ban tổ chức công bố cập nhật, FAQ và quy định mới.

Tài liệu này tổng hợp các nguồn trên, **không thay thế toàn bộ tab Rules hoặc các cập nhật chính thức**. Nội dung đầy đủ của tab Rules chưa được đọc trong phiên làm việc. Khi có thông tin khác nhau, cần kiểm tra thông báo mới nhất hoặc hỏi ban tổ chức trước khi thực hiện phần bị ảnh hưởng.

Phạm vi của repository này là **C1**. Yêu cầu TideCloak và MCP server của Tide thuộc C2, không phải yêu cầu C1 trong các tài liệu đã cung cấp.

## 2. Mục tiêu và nhãn phân loại

Fine-tune model để phân loại mỗi prompt vào đúng một trong ba lớp:

| Nhãn chính xác | Ý nghĩa |
|---|---|
| `BENIGN` | Yêu cầu bình thường, an toàn |
| `JAILBREAK` | Cố thuyết phục model bỏ qua các quy tắc an toàn |
| `PROMPT_INJECTION` | Cố chiếm quyền điều khiển hoặc chuyển hướng chỉ dẫn của model |

Prompt thuộc bốn ngôn ngữ: **Portuguese, French, Italian và English**. Phần lớn dữ liệu không phải tiếng Anh. Đây là bài toán phân loại đa lớp: mỗi dòng nhận một nhãn, không phải nhiều nhãn đồng thời.

## 3. Dữ liệu

| Tập dữ liệu | Số dòng | Nội dung được mô tả |
|---|---:|---|
| `train.csv` | 14.000 | Các cột `id`, `text`, `language`, `label` |
| `test.csv` | 6.000 | Prompt cần dự đoán; nhãn bị ẩn |

Phân bố lớp trong tập train theo đề bài:

| Lớp | Tỷ lệ |
|---|---:|
| `BENIGN` | 62% |
| `JAILBREAK` | 31% |
| `PROMPT_INJECTION` | 7% |

Cần kiểm tra schema và dữ liệu thực tế sau khi tải. Các tỷ lệ trên là số liệu mô tả của đề, không phải số đếm đã tính lại từ CSV trong repository.

## 4. Yêu cầu triển khai

Được phép sử dụng model ngoài danh sách ban đầu theo cập nhật người dùng xác nhận.

1. Fine-tune model trên `train.csv`.
2. Dự đoán một nhãn hợp lệ cho mọi dòng trong `test.csv`.

## 5. Cách chấm điểm

- Metric chính thức: **Macro F1**, càng cao càng tốt; tối đa là `1.0`.
- Tính F1 riêng cho ba lớp, sau đó lấy trung bình với trọng số bằng nhau:

  `Macro F1 = (F1_BENIGN + F1_JAILBREAK + F1_PROMPT_INJECTION) / 3`

- `PROMPT_INJECTION` chỉ chiếm 7% train nhưng đóng góp ngang với mỗi lớp còn lại vào Macro F1.
- Theo mô tả của ban tổ chức, dự đoán toàn bộ là `BENIGN` chỉ đạt khoảng `0.26`.
- Metadata Kaggle cho biết public leaderboard sử dụng **30% tập test**. Điểm public không đại diện đầy đủ cho toàn bộ tập test.

Theo nội dung Challenge, đội đạt đầy đủ điểm Challenge 1 sau khi nộp thành công kết quả hợp lệ đầu tiên; sau đó tiếp tục cải thiện điểm leaderboard. Đây là phát biểu về điểm C1 trong đề, không phải cam kết về thứ hạng, giải thưởng hoặc tổng điểm hackathon.

## 6. Định dạng submission bắt buộc

Nộp CSV có đúng hai cột theo thứ tự:

```csv
id,label
t000006,BENIGN
t000007,PROMPT_INJECTION
t000008,JAILBREAK
```

Yêu cầu:

- Đúng **6.000 dòng dữ liệu và một dòng header**.
- Mỗi `id` trong `test.csv` xuất hiện đúng một lần; không thiếu, thừa hoặc trùng ID.
- `label` phải đúng một trong ba chuỗi: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`.
- Không nộp xác suất hoặc mã số lớp thay cho chuỗi nhãn.
- Không xuất thêm cột index của pandas.

Nên giữ nguyên thứ tự dòng của `test.csv` khi tạo submission để dễ đối chiếu và kiểm tra.

## 7. Tên đội, notebook và điều kiện nộp bài

Quy tắc đặt tên:

```text
<university>_<campus>_<team>_C<challenge>
```

- Với C1: `<challenge>` là `1`.
- `<campus>` thuộc `SGS`, `HNI`, `MEL`, `MYS`.
- `<university>` và `<campus>` chỉ dùng chữ cái.
- `<team>` được dùng chữ cái, chữ số, `_` và `-`.
- Tên áp dụng cho đội/repository hiện tại: **`RMIT_SGS_Sigma69_C1`**. Dùng tên này cho đội và notebook/kernel theo hướng dẫn được cung cấp.
- Đăng nhập hoặc tạo tài khoản Kaggle, thành lập hoặc tham gia đội trên Kaggle và kiểm tra điều kiện tham gia trước khi nộp.
- Metadata Kaggle ghi tối đa **4 thành viên/đội**.

## 8. Deadline và giới hạn lượt nộp

**Deadline: 18:00 ngày 22/10/2026, giờ Việt Nam (UTC+7).** Metadata Kaggle tương ứng là `2026-10-22T11:00:00Z`.

Có điểm chưa thống nhất về lượt nộp:

| Nguồn | Thông tin |
|---|---|
| Nội dung Challenge do người dùng cung cấp | Chỉ có **6 lượt nộp** |
| Metadata Kaggle đã kiểm tra | Tối đa **6 lượt nộp/ngày** |

**Quy ước làm việc tạm thời của đội:** lập kế hoạch trong phạm vi **6 lượt tổng cộng** cho đến khi ban tổ chức làm rõ. Đây là lựa chọn thận trọng của đội, không phải quy định mới của cuộc thi. Không dùng lượt nộp chỉ để kiểm tra lỗi định dạng có thể phát hiện tại máy.

## 9. Quy định khác từ slide

- Không sửa, cập nhật hoặc thay thế file trong repository GitHub đã nộp sau deadline; phiên bản đã nộp được dùng để đánh giá.
- Bảng nộp bài trong slide yêu cầu C1 nộp qua Kaggle; yêu cầu GitHub chứa source code, README và video được nêu cho C2. Không suy diễn rằng C1 bắt buộc nộp GitHub từ quy định chung trên.
- Theo dõi quota công cụ AI, Kaggle và GPU, đặc biệt khi fine-tune; chủ động phân bổ tài nguyên vì quota và khả năng cấp GPU có thể hạn chế.

## 10. Gợi ý kỹ thuật từ đề — không phải yêu cầu bắt buộc

- Encoder: dùng `AutoModelForSequenceClassification` với `num_labels=3`.
- LLM: có thể fine-tune bằng LoRA/QLoRA hoặc dùng classification head có đầu ra được giới hạn.
- Xử lý mất cân bằng bằng class weights, focal loss hoặc resampling.
- Khai thác cột `language` và đánh giá theo từng ngôn ngữ.
- Ưu tiên khả năng tổng quát hóa; tránh tối ưu quá mức theo public leaderboard.

## 11. Lưu ý về notebook demo hiện có

Bản demo đã tải có **2 cell**: chuẩn bị dữ liệu/tokenizer và khởi tạo model `FacebookAI/xlm-roberta-base`.

- Chỉ lấy **200 dòng đầu của train**, không phải toàn bộ 14.000 dòng.
- Chia train/validation theo tỷ lệ 90%/10%, stratify theo nhãn, seed `42`.
- Giới hạn token là `256`; khai báo `EPOCHS=1`, `LR=2e-5`, batch size mỗi GPU là `16`.
- Chưa có bước train, đánh giá, dự đoán hoặc xuất submission.

Đây là mã khởi đầu, chưa phải giải pháp hoàn chỉnh để nộp. Các hyperparameter của demo không phải giới hạn bắt buộc của cuộc thi. Notebook chưa được chạy trong phiên làm việc này.

## 12. Checklist trước khi nộp

- [ ] Kiểm tra cập nhật chính thức, tab Rules và các điểm chưa rõ trước khi áp dụng phương án liên quan.
- [ ] Đội và notebook có tên `RMIT_SGS_Sigma69_C1`; đội không quá 4 thành viên.
- [ ] Đã fine-tune trên `train.csv`; không chỉ khởi tạo model hoặc chạy inference từ model gốc.
- [ ] Đã đánh giá Macro F1 trên validation; kiểm tra F1 từng lớp và từng ngôn ngữ như bước kiểm soát chất lượng của đội.
- [ ] CSV có đúng header `id,label` và 6.000 dòng dữ liệu.
- [ ] Tập ID khớp hoàn toàn `test.csv`, không trùng hoặc thiếu.
- [ ] Tất cả nhãn đúng chính tả, đúng chữ hoa; không có giá trị rỗng hoặc cột index thừa.
- [ ] Kiểm tra lượt nộp còn lại và nộp trước deadline.
- [ ] Xác nhận Kaggle tiếp nhận, chấm điểm thành công; lưu lại notebook/version và file đã nộp để đối chiếu.
