# Đối chiếu notebook Melbourne và hướng nâng cấp

Ngày kiểm tra: 22/10/2026, giờ Việt Nam. File được đọc: [rmit-mel-segfault-c1.ipynb](../rmit-mel-segfault-c1.ipynb). Chưa chạy lại notebook này trong phiên kiểm tra.

## Wolf Defender đã có trong danh sách BTC phê duyệt

Đã đọc trực tiếp nội dung Challenge từ API công khai của Kaggle: `POST https://www.kaggle.com/api/i/competitions.PageService/ListPages`, body `{"competitionId":169360}`. Trang trả về có `id=877551`, `name="Challenge"`, `isPublished=true`.

Danh sách hiện có **10 lựa chọn**, trong đó ghi rõ `patronus-studio/wolf-defender-prompt-injection`, loại encoder đã được huấn luyện cho an toàn, với [model source do BTC cung cấp](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default). Challenge yêu cầu chọn **một** backbone được duyệt và fine-tune trên train. Do đó có căn cứ dùng Wolf thay cho DistilBERT; không trộn thêm một backbone khác vào cùng giải pháp. [Nguồn: Challenge của cuộc thi](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview/challenge).

DistilBERT + QLoRA trong các tài liệu trước là lựa chọn ban đầu của dự án, không phải model duy nhất được cuộc thi cho phép. Người dùng đã cho phép chuyển sang Wolf nếu BTC duyệt; điều kiện đó đã được xác minh bằng nội dung chính thức trên, không suy từ thứ hạng notebook tham khảo.

Bản trích cấu trúc gồm URL, thời điểm, page ID, SHA-256 nội dung trang và danh sách model được lưu tại [demo/competition-challenge.json](../demo/competition-challenge.json).

## Notebook tham khảo đang làm gì?

| Thành phần | Quan sát trực tiếp trong file |
|---|---|
| Model source | `/kaggle/input/models/aisuko/wolf-defender-prompt-injection/transformers/default/1` |
| Kiến trúc được loader báo | `ModernBertForSequenceClassification` |
| Head | Nạp checkpoint hai nhãn, khởi tạo lại classifier thành ba nhãn với `ignore_mismatched_sizes=True` |
| Huấn luyện | Fine-tune toàn bộ model; không có quantization hoặc LoRA |
| Tokenization | 256 token đầu, dynamic padding |
| Validation | 90/10, seed 42, stratify `language + label` |
| Loss | Weighted cross-entropy, class weight là căn bậc hai inverse-frequency từ train split |
| Batch / LR | Batch 16, accumulation 2, learning rate `3e-5`, warmup 100 steps |
| Chọn checkpoint | Validation Macro F1, evaluation và save theo epoch |
| Hậu xử lý | Grid search bias cho JAILBREAK/PROMPT_INJECTION trên logits validation |
| Test | Dự đoán 6.000 dòng, cộng bias đã chọn, xuất `id,label` |

Model Wolf bắt đầu từ một checkpoint khác và được fine-tune toàn bộ. Vì vậy không thể quy toàn bộ mức cải thiện so với DistilBERT QLoRA cho bias hoặc số epoch.

## Kết quả có bằng chứng trong output

| Phép đo | Giá trị |
|---|---:|
| Epoch 1 raw validation Macro F1 | 0.946800 |
| Epoch 2 raw validation Macro F1 | **0.960082** |
| Macro F1 sau chọn bias trên chính validation | **0.9647113691** |
| Bias đã lưu | `[0.0, -1.0, -0.75]` |
| Raw theo ngôn ngữ: English | 0.927766 |
| Raw theo ngôn ngữ: French | 0.987540 |
| Raw theo ngôn ngữ: Italian | 0.972816 |
| Raw theo ngôn ngữ: Portuguese | 0.947991 |

Điểm public khoảng `0.97` là thông tin người dùng cung cấp. File không có phản hồi submission/leaderboard để kiểm chứng độc lập score đó hoặc gắn chắc chắn nó với đúng source đang thấy.

## Những giới hạn cần sửa khi áp dụng

1. **Source và output không cùng trạng thái.** Code khai báo bốn epoch, nhưng progress/output ghi hai epoch và `global_step=394`. Code lưu `submission_wolf4.csv`, `val_wolf4.npy`, `test_wolf4.npy`, `out_wolf4`; danh sách output lại ghi `submission.csv`, `val_wolf.npy`, `test_wolf.npy`, `out`. Không thể cam kết chạy source hiện tại tái tạo đúng score đã lưu.
2. **Bias score có thiên lệch tối ưu.** Notebook chọn bias và tính `0.964711` trên cùng 1.400 mẫu. Không gọi đây là kiểm định độc lập. V2 của nhóm đã có cross-fit cho bias/ensemble, nên tiếp tục dùng biện pháp đó thay vì chép nguyên grid search rồi báo điểm fit.
3. **F1 tăng không có nghĩa mọi dấu hiệu đều tốt lên.** Validation loss tăng từ `0.127164` ở epoch 1 lên `0.169654` ở epoch 2 dù Macro F1 tăng. Cần giữ tiêu chí chọn checkpoint theo metric cuộc thi và theo dõi cả loss.
4. **Chưa kiểm tra trùng/template giữa hai phía split.** Giữ kiểm tra trùng và ID split từ pipeline nhóm; nếu có nhóm trùng đã xác định thì tránh để một nhóm ở cả train và validation.
5. **Chọn data/model bằng kết quả `os.walk` đầu tiên dễ nhầm input.** Cần cấu hình đường dẫn rõ ràng, kiểm tra đúng model type/source và schema trước train.
6. **Chưa có submission validator đầy đủ.** Giữ kiểm tra chính xác ID, số dòng, miền label và thứ tự `id,label`; lưu label mapping một lần để tránh nhầm classifier hai lớp gốc với ba lớp cuộc thi.
7. **Chưa refit toàn bộ train.** Reference chỉ train phần 90%. Nếu thêm refit, phải tách selection và final training, không chấm final model trên validation cũ đã đưa vào fit.

## Hướng áp dụng cho Sigma69

Ưu tiên một notebook Wolf Defender theo quyền dùng backbone vừa xác minh. Giữ bản DistilBERT v2 đã chạy để đối chiếu và giữ nguyên notebook Melbourne gốc làm nguồn tham khảo. Các tham số LoRA r64/alpha128 của kế hoạch DistilBERT không áp dụng máy móc cho Wolf full fine-tuning.

Cấu hình Wolf mặc định mới dùng tối đa 4 epoch, LR `3e-5`, patience 2, `MAX_LENGTH=256`, lấy đầu chuỗi, class-weight power `0.5`, focal gamma `0`, batch 8/accumulation 4. Bật gradient checkpointing, dùng SDPA, giữ trọng số FP32 với mixed precision FP16 trên T4 và tắt reference compile. Đây là lựa chọn triển khai cần chạy kiểm chứng; không coi điểm hai epoch trong notebook Melbourne là kết quả của cấu hình mới.

Nhánh TF-IDF/SVM của nhóm và cross-fit hậu xử lý được giữ. Grid bias mở rộng để bao phủ vùng notebook Melbourne đã khảo sát, nhưng tham số cuối được chọn theo pipeline nhóm; không chép cố định `[0,-1,-0.75]` lên model chưa kiểm tra. Không bật bias riêng theo ngôn ngữ mặc định.

Pipeline cần giữ các kiểm tra dữ liệu/submission, lưu config/logits/split IDs và báo cáo theo lớp/ngôn ngữ. Tách rõ model selection trên holdout khỏi final refit toàn train; khóa epoch và tham số hậu xử lý trước refit. Khi refit, không evaluation/early stopping trên holdout cũ. Giữ cả submission selection và final để truy vết lựa chọn.

Raw validation, cross-fit hậu xử lý và score fit bias phải được ghi riêng. `0.99` là mục tiêu, chưa có GPU run hoặc score Kaggle của phiên bản nâng cấp để xác nhận. Không dự báo tổng mức tăng bằng cách cộng các lợi ích giả định của refit, focal loss, epoch và bias.
