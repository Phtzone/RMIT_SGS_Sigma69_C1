# Tài liệu Challenge 1

Thư mục này là nguồn thống nhất để bốn thành viên triển khai Challenge 1.

| Tài liệu | Nội dung |
|---|---|
| [requirements.md](requirements.md) | Yêu cầu nghiệp vụ, dữ liệu, metric, submission và điều kiện nghiệm thu |
| [architecture.md](architecture.md) | Wolf Defender v5 dựa trên cấu hình v3, hậu xử lý và quy trình selection/refit |
| [team-plan.md](team-plan.md) | Phân công bốn thành viên, tên branch, đầu ra và thứ tự merge |
| [kaggle-submission.md](kaggle-submission.md) | Quy trình bắt buộc để tạo notebook, lưu version và nộp bài trên Kaggle |
| [colab-data.md](colab-data.md) | Tải bộ CSV và kết nối dữ liệu với notebook trên Google Colab |
| [experiments.md](experiments.md) | Kết quả baseline/v2/v3 đã lưu, log v4 do người dùng cung cấp và thử nghiệm v5 |
| [upgrade-099-plan.md](upgrade-099-plan.md) | Mục tiêu 0.99, cấu hình Wolf mới, kế hoạch DistilBERT trước đó và giới hạn bằng chứng |
| [melbourne-reference-review.md](melbourne-reference-review.md) | Đối chiếu notebook tham khảo, source/output khác nhau và bằng chứng BTC duyệt Wolf |

Notebook hiện tại là **[final_97.ipynb](../final_97.ipynb)**, bản v5 dùng `MODEL_PRESET="wolf"`, 256 token `head` và cách nạp `v3_legacy`; `distilbert` là lựa chọn thay thế, không chạy chung hai backbone. Upload file này và [ZIP dữ liệu](../rmit-hackathon-2026.zip) lên Colab theo [hướng dẫn](colab-data.md). V5 mặc định tắt refit (`USE_FULL_TRAIN=False`) và thử hiệu chỉnh BENIGN/JAILBREAK tiếng Bồ Đào Nha có điều kiện bằng nested cross-fit. Nguồn phê duyệt model được lưu tại [competition-challenge.json](../demo/competition-challenge.json).

[V3 đã chạy](../notebooks/archive/final_97_executed.ipynb) đạt raw validation `0.972933` ở epoch 4; người dùng báo public khoảng `0.97`. Log v4 do người dùng cung cấp đạt tốt nhất `0.969274` ở epoch 3; chưa có notebook v4 đã chạy hoặc public score v4 trong repository. V5 chưa có kết quả GPU và chưa có bằng chứng đạt `0.99`.

Khi tài liệu nội bộ khác với thông báo mới của ban tổ chức, ưu tiên thông báo chính thức mới nhất và cập nhật lại tài liệu này trong cùng pull request.
