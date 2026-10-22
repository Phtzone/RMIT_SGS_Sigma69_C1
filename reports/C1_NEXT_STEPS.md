# Lộ trình tối ưu C1 sau version 5

Ngày 22/10/2026. **Đề xuất để người dùng xem xét, chưa chạy tuning mới.** Giữ nguyên CSV, không tự gửi submission. Không thêm giới hạn thời gian tự ngắt.

## Ưu tiên kết quả cuộc thi

Baseline hiện có: dev 0,949005 sau điều chỉnh, audit 0,935382 trước refit; chưa có điểm leaderboard được xác nhận cho CSV v5. Không thể suy ra khả năng thắng giải chỉ từ dev.
Deadline API vừa đối chiếu là **18:00 UTC+7 ngày 22/10/2026**. Đây là hạn nộp nền tảng, không phải ngân sách tự ngắt của pipeline.
README lưu nội dung Challenge do người dùng cung cấp: bài hợp lệ đầu tiên đạt đầy đủ điểm C1. Website chính thức xác nhận một warm-up và hai challenge, nhưng chưa xác minh được rubric giải tổng thể hoặc toàn bộ Rules Kaggle mới nhất. Vì vậy cần đối chiếu rubric C1/C2 trước khi dành toàn bộ công sức để tăng vài phần nghìn F1. [Website ban tổ chức](https://rmit-hackathon.com/), [tài liệu đã lưu](../README.md).
Nếu người dùng quyết định nộp, dùng CSV v5 đã kiểm chứng để có mốc leaderboard; ghi lại phiên/model/hash/điểm. Đây là bước đề xuất cho người dùng, không phải quyền agent tự nộp. Điểm public chỉ dùng đối chiếu độ lệch, không dò liên tục hệ số theo leaderboard.

## 1. Đóng băng mốc so sánh

- Giữ bản CSV, notebook, selection, xác suất, báo cáo v5 và manifest Kaggle.
- Tái sử dụng split 9.808/2.096/2.096; pin thư viện. Không tạo lại split theo seed trên môi trường khác: đã có 6.477 ID khác phần chia so với chuẩn bị local.
- Mọi thử nghiệm đánh giá phải khởi tạo từ pretrained gốc, không từ model refit 14.000 mẫu đã nhìn thấy dev/audit.
- Audit 0,935382 chỉ là bằng chứng cho phương án v5. Sau khi đã xem lỗi, không dùng cùng audit để tune rồi gọi nó là kiểm tra độc lập mới. [Scikit-learn về thiên lệch khi chọn model](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## 2. Phân tích lỗi dev có mục tiêu

Đọc tối đa 60 lỗi dev, đối chiếu phần token được giữ/bị cắt và ba nhóm: injection→benign, jailbreak→benign, benign→attack. Tách văn bản dài, trộn ngôn ngữ và nhóm gần trùng. Chỉ ghi nhận, không sửa CSV/nhãn.
Audit cho thấy injection recall 82%, injection dài đúng 5/14, nhưng quyết định cấu hình tiếp theo phải được kiểm chứng trên dev/CV. Không gán luật cứng theo từ khóa, ngôn ngữ hoặc sao nhãn gần trùng sang test.

## 3. Ma trận nhỏ, ưu tiên mDeBERTa 256

| Thử nghiệm đề xuất | Thay đổi so với baseline | Giả thuyết cần kiểm tra |
|---|---|---|
| A | LR 1e-5, 3 epoch, CE thường | Bước cập nhật nhỏ hơn có giúp tổng quát hóa |
| B | LR 2e-5, tối đa 5 epoch, chọn checkpoint trên dev | Best hiện ở epoch cuối 3; thử xem còn cải thiện, không mặc định train lâu hơn tốt hơn |
| C | 256 token chia đầu–giữa–cuối, giữ các cấu hình khác | Có thông tin quan trọng bị mất ở giữa; chỉ làm nếu lỗi dev hỗ trợ giả thuyết |
| D | Thêm seed 17 và 2026 cho cấu hình tốt nhất | Phân biệt cải thiện ổn định với may mắn của seed |

Chạy A/B song song trên hai T4. Tối đa 5 epoch là số bước thí nghiệm, không phải giới hạn giờ. Nếu dùng early stopping, chỉ xét sau ít nhất 1,5 epoch, patience 2 lần đánh giá nửa epoch; cần lưu state optimizer/scheduler/scaler/RNG nếu resume cùng trial. Đây là thay đổi chưa được triển khai; code hiện còn kiểm tra epochs <= 3.
Giữ batch hiệu dụng 32. Bản 512 token và weighted CE đã thua; không ưu tiên chạy lại cùng cấu hình. Nghiên cứu về fine-tuning cung cấp hướng thử learning rate/độ dài, không bảo đảm tăng điểm C1. [Sun và cộng sự](https://arxiv.org/abs/1905.05583).

## 4. Kiểm tra độ ổn định trước refit

- Đối chiếu nhiều seed trên cùng dev; theo dõi Macro F1 và precision/recall/F1 injection, không chỉ recall.
- Với tối đa hai cấu hình cuối, có thể làm 3-fold StratifiedGroupKFold trên pool train+dev, loại audit khỏi pool, giữ nhóm gần trùng nguyên vẹn. Lưu fold IDs trước chạy.
- Thu out-of-fold probabilities để xem lỗi và chọn trọng số/hệ số. Điểm OOF đã dùng chọn cấu hình là điểm chọn model, không tuyên bố là test độc lập không thiên lệch.
- So sánh mức tăng giữa seed/fold; mức +0,002 chỉ là ngưỡng thực hành trước đó, không chứng minh ý nghĩa thống kê.

## 5. Ensemble và quyết định cuối

Ưu tiên trung bình xác suất 2–3 mDeBERTa khác seed nếu chứng minh cải thiện; chỉ thêm XLM-R hoặc model khác khi sửa được lỗi bổ sung. Luôn khớp ID/thứ tự nhãn. Giữ tìm kiếm hệ số quyết định nhỏ, trên dev/OOF; không chỉnh riêng từng ngôn ngữ khi injection ít mẫu.
Refit toàn bộ train chỉ sau khi chốt phương án; không refit mỗi trial. Nếu ensemble được chọn, có thể train hai thành phần song song trên hai GPU. Nếu chỉ một thành phần, DDP là tùy chọn cần benchmark chứ không bảo đảm nhanh gấp đôi. [PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## 6. Tăng tốc hỗ trợ tuning

Cache tokenization theo hash dữ liệu/tokenizer revision/cách cắt; gom batch theo độ dài; benchmark microbatch 16×accumulation 2; giảm đồng bộ CUDA dùng riêng đo thời gian; chỉ lưu checkpoint khi điểm tăng và checkpoint phục hồi định kỳ. Giữ kiểm tra NaN/overflow và định danh checkpoint an toàn. [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Chỉ bật tối ưu sau benchmark thực tế và kiểm tra tính đúng. FP16 và dynamic padding đã có; không tính chúng là cải thiện mới.

## Điều kiện giữ một ứng viên

Cải thiện Macro F1 dev/CV có lặp lại qua seed/fold, không chỉ nâng một lát cắt ít mẫu; thời gian inference/độ phức tạp hợp lý; xác suất hữu hạn và CSV 6.000 ID đúng thứ tự; raw hashes nguyên vẹn. Có báo cáo model trước refit tách khỏi model cuối.
Không cam kết điểm hoặc thứ hạng. Không tự gửi bài, chọn final submission, sửa nhãn hoặc thêm dữ liệu ngoài.
