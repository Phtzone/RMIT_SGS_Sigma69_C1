# C1 — dữ liệu thực tế và giới hạn kiểm chứng

Đọc trực tiếp ba CSV trong repository; dữ liệu gốc giữ nguyên. Các kết quả dưới đây là phân tích dữ liệu và kiểm tra triển khai, không phải điểm huấn luyện hoặc leaderboard.

## Schema và số lượng

| File | Số dòng dữ liệu | Schema |
|---|---:|---|
| train.csv | 14.000 | id,text,language,label |
| test.csv | 6.000 | id,text,language |
| sample_submission.csv | 6.000 | id,label |

Train có 8.680 BENIGN, 4.340 JAILBREAK và 980 PROMPT_INJECTION. Ngôn ngữ train: pt 6.300, fr 3.500, it 2.800, en 1.400. Test lần lượt là 2.700, 1.500, 1.200 và 600. Không phát hiện ô rỗng hay ID trùng trong các file đã đọc. Sample submission khớp thứ tự ID test và toàn bộ là BENIGN.

| SHA-256 | Giá trị |
|---|---|
| train.csv | `b6809aea692dbbb4978f45fa603bb001e10ebfbffe9bfbff964e44e2856be795` |
| test.csv | `137925562e9b8403a6f725431489b73b8f267d24423784d1ea0f0a7dc4a455c6` |
| sample_submission.csv | `b25f6188461fb1ad555aa5ce5f7fe7eba356827e8bd541c63b9f59bf6f594f73` |

## Split đã tạo bằng pipeline

| Split | Số dòng | Vai trò |
|---|---:|---|
| train | 9.799 | Huấn luyện và tính class weights |
| dev | 2.100 | Chọn checkpoint, cấu hình và chính sách quyết định |
| audit | 2.101 | Kiểm tra một lần sau khi khóa chính sách |

20-fold StratifiedGroupKFold, seed 42, phân tầng label × language. Fold 0–2 thuộc audit, 3–5 thuộc dev, 6–19 thuộc train. Mọi split có đủ ba lớp và đủ 12 tổ hợp nhãn/ngôn ngữ. Mỗi nhóm gần trùng nằm trọn trong một fold.

Phép dò hiện thực dùng 24 affine MinHash trên BLAKE2s-32 của cụm ba từ, tám band, ba hash/band; sau đó kiểm tra Jaccard chính xác trên tập cụm từ ở ngưỡng 0,8. Kết quả: **262 cặp train–train**, 105 dòng trong 34 nhóm không đơn lẻ; **208 cặp train–test**, liên quan 51 dòng test. Không có bucket nào vượt ngưỡng 200 trong lần chạy này.

Các con số khác nhẹ với 264/205 trong nghiên cứu trước vì phép sinh hash và định nghĩa từ được cố định lại trong Python; MinHash chỉ tìm ứng viên gần đúng. Không nên cố ép số liệu khớp một ước lượng trước đó. Chỉ cạnh train–train tạo nhóm; thêm mẫu test không thể thay đổi nhóm train, kể cả khi bucket diagnostic quá lớn.

## Điều ảnh hưởng đến kết quả

- Trung vị độ dài theo lớp: benign 115, jailbreak 565, injection 99 ký tự. Mẫu dài nhất 18.550 ký tự thuộc benign. Thống kê Python đếm Unicode code point; có thể khác bộ đếm UTF-16 trước đây ở văn bản có emoji. Token length được tính riêng theo từng tokenizer khi huấn luyện.
- Có văn bản tiếng Việt ghi nhãn ngôn ngữ `pt` và văn bản trộn ngôn ngữ. Không dùng `language` để định tuyến model hoặc đặt luật dự đoán.
- Các template và từ khóa không thay thế được ý định. Ví dụ mẫu yêu cầu lộ chỉ dẫn có thể mang nhãn injection, trong khi một số yêu cầu mang ý định nguy hiểm được gán jailbreak mà không có lệnh ghi đè rõ ràng.
- So khớp từ loại bỏ dấu câu và emoji trong chuỗi có từ; hai chuỗi có thể cùng đặc trưng nhưng khác nội dung. Văn bản không có từ dùng toàn bộ chuỗi đã chuẩn hóa khoảng trắng, tránh gom mọi emoji thành một nhóm.
- Phép dò có thể bỏ sót bản dịch/paraphrase. Validation theo nhóm giảm một nguồn rò rỉ đã quan sát, không chứng minh không còn mọi nguồn rò rỉ.

Không sửa nhãn, dịch/sinh thêm dữ liệu hoặc truyền nhãn từ train sang test. Mẫu có vẻ bất thường được ghi nhận như giới hạn, không bị loại khỏi train.

## Kiểm chứng và cách tái tạo

```bash
python -m c1 prepare --data-dir rmit-hackathon-2026 --output-dir artifacts/prepared
python -m pytest -q
```

Prepare tạo manifest, diagnostic pairs, hash và báo cáo JSON/Markdown. Kiểm thử tích hợp offline dùng model BERT rất nhỏ tự khởi tạo để thực sự chạy train → lưu/nạp → lựa chọn → audit → full refit → xuất CSV. Model này chỉ kiểm tra tính đúng của pipeline; không được dùng làm bài thi.

Chưa khởi chạy XLM-R/mDeBERTa trên Kaggle. Không có credentials hoặc phiên trình duyệt Kaggle trong môi trường làm việc hiện tại. Quota, Rules, deadline và số lượt nộp phải đối chiếu trên tài khoản trước khi Run All.

Tokenizer/config chính thức của hai model đã được tải và kiểm tra độc lập trên ví dụ đa ngôn ngữ, không tải trọng số. Head–tail encoding tạo đúng 256 token. Revision quan sát: XLM-R `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, mDeBERTa `a0484667b22365f84929a935b5e50a51f71f159d`. Khi chạy thật, worker lưu revision đã resolve để cố định các lượt thử tiếp theo.

## Nguồn phương pháp

- [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base): baseline đa ngôn ngữ đã có trong demo.
- [mDeBERTa-v3 base](https://huggingface.co/microsoft/mdeberta-v3-base): ứng viên so sánh; không suy từ kết quả XNLI thành Macro F1 C1.
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): chia nhóm đồng thời cố giữ phân bố nhãn.
- [Điều chỉnh quyết định](https://scikit-learn.org/stable/modules/classification_threshold.html): dùng dữ liệu riêng với huấn luyện và tránh tối ưu trên tập kiểm tra cuối.
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): FP16, gradient scaling và xử lý overflow.
- [Class-balanced loss](https://arxiv.org/abs/1901.05555): bối cảnh nghiên cứu mất cân bằng; biến thể triển khai ở đây dùng sqrt inverse-frequency, không phải công thức effective-number của paper.

Chi tiết cấu hình và giới hạn thời gian: [RUNBOOK](../RUNBOOK.md). Kế hoạch được người dùng chấp thuận: [PLAN](../PLAN.md).
