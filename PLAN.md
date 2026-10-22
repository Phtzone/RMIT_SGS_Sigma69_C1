# Kế hoạch tối ưu C1 trên Kaggle T4 ×2

## 1. Kết quả nghiên cứu dữ liệu và hướng lựa chọn

**Mục tiêu:** tìm phương án có Macro F1 tốt nhất qua đánh giá độc lập, chạy hết ma trận thực nghiệm đã chọn rồi tạo CSV hoàn chỉnh. Theo yêu cầu mới ngày 22/10/2026, bỏ giới hạn bốn giờ và các mốc cắt theo phút. Chưa thể cam kết điểm hoặc thứ hạng trước thực nghiệm; chỉ nộp khi người dùng yêu cầu rõ ràng.

Đã đọc toàn bộ 20.000 prompt để tính thống kê, dò gần trùng và xem các mẫu đại diện. CSV gốc được giữ nguyên; SHA-256 đã được đối chiếu.

| Phát hiện thực tế | Ý nghĩa đối với phương pháp |
|---|---|
| Train có 8.680 benign, 4.340 jailbreak, 980 injection | Phải theo dõi F1 từng lớp; accuracy không đủ |
| Mỗi ngôn ngữ đều có đúng tỷ lệ nhãn 62%/31%/7% | Không có lợi thế từ phân bố nhãn theo `language`; dùng trường này để chia và báo cáo |
| Có nội dung tiếng Việt ghi `pt`, cùng văn bản trộn ngôn ngữ | Không dùng `language` để ép định tuyến sang model riêng |
| Không trùng hoàn toàn, kể cả sau chuẩn hóa chữ thường/khoảng trắng | Chưa đủ để kết luận không rò rỉ |
| Dò gần trùng phát hiện 264 cặp train–train, liên quan 109 dòng trong 36 nhóm; 232 cặp thuộc injection | Gom nhóm trước khi chia dữ liệu, đặc biệt để tránh đánh giá quá cao lớp hiếm |
| Có 205 cặp train–test gần nhau, liên quan 51 dòng test | Ghi nhận để hiểu phân bố; không tự sao chép nhãn sang test |
| Trung vị độ dài: benign 115, jailbreak 565, injection 99 ký tự | Độ dài có thể là dấu hiệu đặc thù của bộ dữ liệu; cần kiểm tra kết quả theo nhóm độ dài |
| Có 913 dòng train chứa các biến thể đã tìm của `TARGETLANGUAGE`, đều benign | Có dấu hiệu template; không chuyển thành luật gán nhãn cứng |
| Mẫu dài nhất có 18.550 ký tự; chỉ dẫn quan trọng có thể nằm cuối | Đo độ dài bằng tokenizer thực tế và so sánh cách xử lý đoạn dài |

**Giới hạn của phép dò:** dùng MinHash tìm ứng viên rồi kiểm tra Jaccard trên cụm ba từ ở ngưỡng 0,8. Đây là các mẫu gần nhau trên đặc trưng văn bản, không phải 264 bản sao ngữ nghĩa được xác nhận. Phương pháp có thể bỏ sót bản dịch và diễn đạt lại; việc bỏ dấu câu/emoji khi tạo đặc trưng cũng có thể làm hai mẫu trông giống nhau hơn.

Lựa chọn model cho vòng đầu:

| Phương án | Quyết định |
|---|---|
| **XLM-RoBERTa base** | Baseline chính vì đã có trong demo và hỗ trợ đa ngôn ngữ. [Model card](https://huggingface.co/FacebookAI/xlm-roberta-base) |
| **mDeBERTa-v3 base** | Đối thủ trực tiếp; Microsoft công bố kết quả chuyển giao đa ngôn ngữ tốt, nhưng phải kiểm chứng trên C1. [Model card](https://huggingface.co/microsoft/mdeberta-v3-base) |
| **Prompt Guard 86M** | Phù hợp chủ đề và dùng nền mDeBERTa, nhưng checkpoint cần quyền truy cập; không đưa vào lịch chạy mặc định. [Model card](https://huggingface.co/meta-llama/Prompt-Guard-86M) |
| **Protect AI injection detector** | Không chọn: tài liệu ghi chỉ hỗ trợ tiếng Anh, phân loại nhị phân và không phát hiện jailbreak. [Giới hạn chính thức](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) |
| **mmBERT / EuroBERT** | Để vòng mở rộng sau ma trận đầu; khả năng đa ngôn ngữ/ngữ cảnh dài đáng quan tâm nhưng chưa xác minh hiệu quả và môi trường chạy cho bài này. [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base), [EuroBERT](https://huggingface.co/EuroBERT/EuroBERT-210m) |

## 2. Chuẩn bị dữ liệu và thiết kế đánh giá

- Đọc nguyên văn `text`; giữ dấu, emoji, xuống dòng, HTML và ký tự đặc biệt. Không sửa nhãn, xóa dòng hay ghi đè CSV.
- Tạo đặc trưng chuẩn hóa riêng để dò nhóm; chỉ dùng quan hệ **train–train** để xác định nhóm chia dữ liệu. Các thành phần liên thông ở ngưỡng Jaccard ≥ 0,8 thuộc cùng nhóm.
- Chốt ba phần khoảng **70% huấn luyện, 15% phát triển, 15% kiểm tra cuối**. Dùng `StratifiedGroupKFold` với 20 fold, seed 42, phân tầng theo `label × language`; fold 0–2 là kiểm tra cuối, 3–5 là phát triển, phần còn lại là huấn luyện. Đây là cách tạo split, không phải huấn luyện 20 model. [Tài liệu chia theo nhóm](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- Kiểm tra không giao nhau về ID/nhóm; đủ ba nhãn trong mỗi phần; lưu số mẫu thực tế của 12 tổ hợp nhãn–ngôn ngữ.
- Chỉ dùng tập phát triển để chọn checkpoint, cấu hình, ensemble và điều chỉnh quyết định. Tập kiểm tra cuối chỉ mở sau khi khóa phương án.
- Đầu vào model là `text`. `id` dùng để đối chiếu; `language` dùng để chia và báo cáo, không đưa vào đặc trưng vòng đầu.

Mỗi lần đánh giá phải có Macro F1, precision/recall/F1 từng lớp, confusion matrix, kết quả từng ngôn ngữ và số mẫu. Báo cáo riêng văn bản dài, mẫu có template và nhóm gần trùng để phát hiện việc học thuộc dấu hiệu.

## 3. Ma trận thực nghiệm và cách chọn kết quả

### Hai baseline chạy đồng thời

Một tiến trình dùng GPU 0 cho XLM-R, một tiến trình dùng GPU 1 cho mDeBERTa. Mỗi model nằm trọn trên một GPU; không giả định bộ nhớ hai T4 tự cộng lại.

Cấu hình khởi đầu chung:

- Classification head ba nhãn, thứ tự cố định: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`.
- Fine-tune toàn bộ model; cross-entropy thông thường.
- AdamW, learning rate `2e-5`, weight decay `0.01`, warmup 10%, linear decay, gradient clipping `1.0`.
- Tối đa ba epoch, seed 42; đánh giá và lưu checkpoint mỗi nửa epoch; chọn theo Macro F1 tập phát triển.
- Microbatch 8, gradient accumulation 4: effective batch 32 trên mỗi tiến trình.
- FP16 với gradient scaling, dynamic padding. [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
- Độ dài tối đa 256 token. Nếu vượt giới hạn, chia đều ngân sách token nội dung cho đầu và cuối văn bản, rồi thêm special token bằng tokenizer.
- Nếu hết bộ nhớ: giảm batch `8 → 4 → 2 → 1`, tăng accumulation tương ứng; bật gradient checkpointing khi cần.

Đo tốc độ sau 100 bước để báo cáo tiến độ, không dùng tốc độ để bỏ qua thử nghiệm. Tạo CSV dự phòng từ checkpoint hợp lệ khi GPU được giải phóng.

### Tối đa hai thử nghiệm cải tiến

Chọn model baseline có Macro F1 phát triển cao hơn; nếu chênh dưới `0,001`, ưu tiên model chạy nhanh hơn. Từ cùng checkpoint gốc, cùng split và seed, chạy hai biến thể độc lập:

1. **Biến thể đoạn dài:** tăng giới hạn lên 512 token, vẫn giữ đầu–cuối; các lựa chọn khác giữ như baseline.
2. **Biến thể mất cân bằng:** giữ 256 token, đổi sang weighted cross-entropy với trọng số tỷ lệ `sqrt(n_max / n_class)`, tính trên phần huấn luyện.

Dùng trọng số vừa phải để tránh mặc định tăng quá mạnh lớp injection. Class-balanced loss và focal loss có cơ sở nghiên cứu, nhưng không mặc định tốt hơn cross-entropy trên dữ liệu này; focal loss có thể làm các mẫu khó hoặc nhãn bất thường chi phối hơn. [Nghiên cứu class-balanced loss](https://arxiv.org/abs/1901.05555).

Đọc tối đa 60 lỗi tiêu biểu trên tập phát triển, ưu tiên nhầm injection/benign và jailbreak/injection. Ghi nhận lỗi để giải thích kết quả, không sửa nhãn thủ công.

### Ensemble và điều chỉnh quyết định

- Giữ ứng viên tốt nhất của mỗi họ model.
- Thử trung bình xác suất với trọng số XLM-R trong `{0; 0,25; 0,5; 0,75; 1}`.
- Sau khi chọn trọng số ensemble, thử nhân xác suất jailbreak và injection với hệ số trong `{0,8; 1; 1,25}`; benign giữ hệ số 1. Chọn nhãn bằng argmax để luôn có đúng một lớp.
- Chỉ giữ ensemble hoặc điều chỉnh nếu tăng Macro F1 phát triển ít nhất `0,002` so với phương án đơn giản hơn tương ứng. Đây là ngưỡng quyết định thực nghiệm, không phải bảo đảm có ý nghĩa thống kê.
- Không chỉnh ngưỡng riêng cho từng ngôn ngữ trong vòng đầu vì số mẫu injection ở một số nhóm nhỏ.
- Khóa toàn bộ lựa chọn trước khi đánh giá tập kiểm tra cuối. Việc điều chỉnh quyết định cần dữ liệu tách khỏi dữ liệu huấn luyện. [Hướng dẫn tránh overfitting khi chọn ngưỡng](https://scikit-learn.org/stable/modules/classification_threshold.html).

Kết quả kiểm tra cuối dùng để báo cáo khả năng tổng quát hóa, không để mở thêm vòng dò cấu hình. Sau đó huấn luyện lại phương án đã chọn trên toàn bộ 14.000 dòng, với số epoch tương ứng checkpoint tốt nhất.

## 4. Trình tự thực hiện

1. Kiểm tra GPU, dữ liệu và tokenizer; tạo nhóm và split cố định.
2. Chạy hai baseline, sau đó hai biến thể của họ model được chọn; mỗi thử nghiệm tối đa ba epoch như cấu hình đã định.
3. Chọn ensemble/hệ số trên dev, khóa phương án, đánh giá audit một lần.
4. Refit các thành phần được chọn trên toàn bộ train, dự đoán test và kiểm tra CSV.
5. Lưu notebook, checkpoint và báo cáo để người dùng xem xét.

Không dùng giới hạn bốn giờ, mốc kết thúc giai đoạn hoặc deadline cuộc thi để ngắt chương trình. Một GPU thì chạy tuần tự. Quota và giới hạn phiên do Kaggle quản lý vẫn áp dụng; nếu có lỗi thì giữ checkpoint/CSV dự phòng hợp lệ. Deadline nộp bài vẫn là 18:00 ngày 22/10/2026 UTC+7 theo thông tin đã kiểm tra, nhưng không tự nộp bài.

## 5. Đầu ra và tiêu chí hoàn thành

Bổ sung một notebook Kaggle hoàn chỉnh và bộ chạy có các bước `prepare`, `train`, `evaluate`, `predict`, `validate-submission`. Cấu hình và kết quả mỗi lần thử được lưu riêng để có thể chạy lại.

Đầu ra gồm:

- Báo cáo dữ liệu, phương pháp dò gần trùng, SHA-256 và danh sách ID của các split.
- Cấu hình, phiên bản thư viện, revision model, seed, checkpoint và xác suất dự đoán.
- Bảng so sánh thí nghiệm; điểm phát triển và điểm kiểm tra cuối ghi riêng. Điểm kiểm tra cuối thuộc model trước khi huấn luyện lại toàn bộ train.
- `submission.csv` cùng notebook có tên đội `RMIT_SGS_Sigma69_C1`.
- Báo cáo Markdown cập nhật phát hiện thực tế và các nguồn nghiên cứu.

Kiểm tra bắt buộc:

- Hash ba CSV gốc không đổi; không có giao nhau về ID hoặc nhóm giữa các split.
- Bộ mã hóa nhãn và thứ tự cột xác suất giống nhau ở mọi model.
- Xác suất hữu hạn, mỗi dòng có đúng một dự đoán; kết quả ensemble khớp theo ID.
- CSV cuối có đúng header `id,label`, đủ 6.000 dòng, đúng thứ tự test, không thiếu/thừa/trùng ID và không có cột index.
- Nạp lại checkpoint và dự đoán một batch thành công trước khi chạy toàn bộ test.

**Mặc định hiện tại:** Kaggle T4 ×2, chạy hết ma trận không cắt theo thời gian, được nghiên cứu model, giữ nguyên dữ liệu gốc. Vòng đầu dùng dữ liệu cuộc thi cùng checkpoint pretrained; không bổ sung dữ liệu ngoài, dịch/sinh dữ liệu hay pseudo-label test. Trạng thái chạy thực tế ở `memory-bank/active-context.md`.
