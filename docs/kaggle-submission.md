# Quy trình tạo Kaggle Notebook và nộp bài

Tài liệu này là checklist bắt buộc cho Challenge 1. C1 được triển khai và nộp qua Kaggle; GitHub là nơi phối hợp source code, không thay thế Kaggle submission.

## 1. Trách nhiệm

- Thành viên 4 là người thao tác notebook và submission chính.
- Ít nhất một thành viên khác phải review notebook version và file output trước khi bấm Submit.
- Mọi thành viên phải tham gia đúng Kaggle Team trước deadline.

## 2. Chuẩn bị Kaggle Team

Đội có đúng **4 thành viên tổng cộng**. Một trong bốn người là Team Leader và tạo team; người này chỉ mời **3 thành viên còn lại**, không mời thêm 4 người vì như vậy đội sẽ có 5 người và vượt giới hạn.

1. Cả 4 thành viên đăng nhập Kaggle, tham gia competition và chấp nhận Competition Rules bằng tài khoản riêng.
2. Team Leader mở tab **Team** và tạo team.
3. Đặt tên team chính xác:

```text
RMIT_SGS_Sigma69_C1
```

4. Team Leader gửi lời mời cho 3 thành viên còn lại bằng đúng tài khoản Kaggle của họ.
5. Ba thành viên mở lời mời và bấm chấp nhận; gửi lời mời chưa có nghĩa là đã vào team.
6. Team Leader kiểm tra trang Team hiển thị đúng 4 thành viên trước khi submit.
7. Sau khi đội hình đã đủ, một người đại diện thực hiện submission. Các thành viên khác không nộp lại cùng một file vì submission và quota được tính chung cho team.
8. Kiểm tra submission được ghi nhận dưới team `RMIT_SGS_Sigma69_C1`, không phải tài khoản hoặc team khác.

```text
Tổng số người: 4
Team Leader: 1 người tạo team
Thành viên được mời: 3 người
Người submit: 1 trong 4 người, ưu tiên Thành viên 4 theo kế hoạch
```

## 3. Tạo notebook

1. Mở tab **Code** của competition.
2. Chọn **New Notebook** hoặc **Copy & Edit** notebook demo chính thức.
3. Đặt tên notebook:

```text
RMIT_SGS_Sigma69_C1
```

4. Gắn competition data source `rmit-hackathon-2026`.
5. V3 mặc định gắn [Wolf Defender của BTC](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default), version 1, và dùng `MODEL_PRESET="wolf"`. Nếu chọn preset `distilbert`, gắn model source chính thức `distilbert/distilbert-base-multilingual-cased` tương ứng.
6. Bật một GPU T4 theo thiết kế notebook; không chạy distributed training với preset 4-bit.
7. Không phụ thuộc vào tải model trực tiếp từ Internet; sử dụng model source do ban tổ chức cung cấp.

Đường dẫn trong notebook phải xuất phát từ Kaggle input, ví dụ:

```python
DATA_DIR = "/kaggle/input/competitions/rmit-hackathon-2026"
OUTPUT_FILE = "/kaggle/working/submission.csv"
```

Đường dẫn model phải được kiểm tra từ panel Input sau khi add model. Với Wolf, đặt `WOLF_LOCAL_PATH` vào thư mục model đã gắn chứa `config.json` để không phụ thuộc tải mạng khi chạy version. Nguồn/version dùng trong notebook là `aisuko/wolf-defender-prompt-injection/transformers/default/1`; [bằng chứng BTC phê duyệt](../demo/competition-challenge.json) đã được lưu.

## 4. Nội dung notebook bắt buộc

Notebook cuối phải chạy theo thứ tự:

1. Tiêu đề, team, model được chọn và phương pháp train; QLoRA config chỉ áp dụng cho preset DistilBERT.
2. In phiên bản Python, PyTorch, Transformers, PEFT và bitsandbytes.
3. Cố định seed.
4. Đọc toàn bộ `train.csv` và `test.csv`.
5. Kiểm tra schema, số dòng, ID và nhãn.
6. Tạo train/validation split.
7. Nạp Wolf với head ba lớp để full fine-tuning, hoặc DistilBERT 4-bit NF4 và LoRA theo preset đã chọn.
8. Fine-tune và chọn checkpoint theo validation Macro F1.
9. Báo cáo Macro F1, per-class F1 và kết quả theo ngôn ngữ.
10. Predict đủ 6.000 dòng test.
11. Tạo và validate `submission.csv`.
12. Ghi file tại `/kaggle/working/submission.csv`.

Cell cuối phải kiểm tra ít nhất:

```python
assert list(submission.columns) == ["id", "label"]
assert len(submission) == 6000
assert submission["id"].is_unique
assert set(submission["id"]) == set(test["id"])
assert set(submission["label"]) <= {
    "BENIGN", "JAILBREAK", "PROMPT_INJECTION"
}
submission.to_csv("/kaggle/working/submission.csv", index=False)
```

## 5. Lưu version chạy thành công

1. Chọn **Save Version**.
2. Chọn chế độ chạy lại toàn bộ notebook, thường hiển thị là **Save & Run All** hoặc **Commit**.
3. Ghi version note gồm commit Git, cấu hình chính và validation Macro F1.
4. Chờ notebook chạy hoàn tất; bản nháp hoặc version lỗi không đủ điều kiện nộp.
5. Mở tab Output và xác nhận có `submission.csv`.
6. Tải file output xuống để kiểm tra độc lập nếu còn thời gian.

Không sửa notebook đang chạy. Nếu cần sửa, tạo version mới và ghi rõ thay đổi.

## 6. Nộp vào competition

1. Từ output của notebook version, chọn **Submit to Competition**. Nếu giao diện không hiện nút này, mở trang **Submit Predictions** của competition và chọn output của notebook.
2. Chọn đúng `submission.csv` từ version vừa review.
3. Submission description phải ngắn gọn nhưng truy vết được, ví dụ:

```text
v5 wolf full-ft maxlen256 head weighted-ce seed42 selection pt-gated
```

4. Kiểm tra lại team `RMIT_SGS_Sigma69_C1`.
5. Bấm Submit trước deadline.
6. Chờ trạng thái hoàn tất và score xuất hiện; submission lỗi format không được xem là hoàn thành.

## 7. Bằng chứng cần lưu

Sau mỗi lượt nộp, cập nhật bảng sau trong pull request hoặc team log:

| Lượt | Thời gian ICT | Notebook version/URL | Git commit | Cấu hình | Validation Macro F1 | Public score | Trạng thái |
|---:|---|---|---|---|---:|---:|---|
| 1 |  |  |  |  |  |  |  |

Ngoài bảng trên, lưu lại:

- file `submission.csv` đã nộp;
- notebook version URL;
- screenshot hoặc link trạng thái Kaggle đã chấm;
- full model/config với Wolf hoặc adapter/config với DistilBERT, đúng artifact tạo submission;
- lỗi Kaggle nếu lượt nộp thất bại.

Không commit Kaggle credentials, cookies, API token hoặc dữ liệu cuộc thi vào repository.

## 8. Điều kiện hoàn thành

Công việc chỉ hoàn thành khi:

- Kaggle Notebook đúng tên tồn tại và team truy cập được;
- một version đã chạy end-to-end thành công;
- output có `submission.csv` hợp lệ;
- file đã được submit dưới đúng team;
- Kaggle hiển thị trạng thái chấm thành công và score;
- notebook URL, version và submission record đã được lưu.
