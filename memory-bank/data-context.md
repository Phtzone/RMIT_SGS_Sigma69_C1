# Dữ liệu và thiết kế đánh giá

## Kết quả split Kaggle version 5 — ưu tiên khi tái lập

Version 5 dùng **9.808 train / 2.096 dev / 2.096 audit**, lưu tại `artifacts/kaggle-v5-output/c1-output/prepared/manifest.csv`. Hash CSV gốc trùng local; group ID giống nhưng 6.477 ID khác phần chia. Môi trường scikit-learn/NumPy khác nhau, chưa xác minh riêng nguyên nhân. Dùng manifest Kaggle để so sánh tiếp. Mỗi split đủ 12 tổ hợp nhãn–ngôn ngữ, không giao nhóm. Xem [báo cáo](../reports/KAGGLE_V5_RESULTS.md).

Nguồn: `rmit-hackathon-2026/`. Phát hiện được lưu trong [reports/DATA_REVIEW.md](../reports/DATA_REVIEW.md); artifacts chuẩn bị tại `artifacts/prepared/`.

## Dữ liệu gốc

| File | Số dòng dữ liệu | Cột |
|---|---:|---|
| `train.csv` | 14.000 | `id,text,language,label` |
| `test.csv` | 6.000 | `id,text,language` |
| `sample_submission.csv` | 6.000 | `id,label` |

- Train: benign 8.680, jailbreak 4.340, injection 980 (62% / 31% / 7%).
- Ngôn ngữ train: pt 6.300, fr 3.500, it 2.800, en 1.400; test lần lượt 2.700 / 1.500 / 1.200 / 600. Tỷ lệ nhãn giống nhau trong từng ngôn ngữ.
- Không phát hiện giá trị thiếu/rỗng, ID trùng hoặc văn bản trùng hoàn toàn trong kiểm tra trước. Có ngôn ngữ ghi sai và văn bản trộn ngôn ngữ.
- Trung vị ký tự benign 115, jailbreak 565, injection 99; dài nhất khoảng 18.550 ký tự. Cách đếm Unicode Python và UTF-16 có thể khác ở emoji.
- Biến thể marker TARGETLANGUAGE đã dò xuất hiện ở 913 dòng train đều benign và 397 dòng test. Đây là dấu hiệu cần báo cáo, không phải luật dự đoán.

## SHA-256 đã đối chiếu

```text
train.csv             b6809aea692dbbb4978f45fa603bb001e10ebfbffe9bfbff964e44e2856be795
test.csv              137925562e9b8403a6f725431489b73b8f267d24423784d1ea0f0a7dc4a455c6
sample_submission.csv b25f6188461fb1ad555aa5ce5f7fe7eba356827e8bd541c63b9f59bf6f594f73
```

## Gần trùng và split thực tế

- Triển khai dùng 24 giá trị MinHash trên BLAKE2s 32-bit của cụm ba từ, 8 band × 3, rồi kiểm tra Jaccard ≥ 0,8; giới hạn bucket 200. Không có bucket nào bị cắt ở dữ liệu thực.
- Kết quả triển khai: **262 cặp train–train**, liên quan **105 dòng / 34 nhóm**; **208 cặp train–test**, liên quan **51 dòng test**.
- Số 264/205 trong kế hoạch là kết quả dò thăm dò trước đó; khác nhau do hash/tokenization. Không ghi đè kết quả thực bằng con số kế hoạch.
- Chỉ quan hệ train–train quyết định nhóm thành phần liên thông. Train–test chỉ để chẩn đoán, không nối nhóm hoặc sao chép nhãn.
- `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` phân tầng `label × language`. Fold 0–2 audit, 3–5 dev, 6–19 train.
- Số dòng: **train 9.799, dev 2.100, audit 2.101**. Đã kiểm tra không giao nhóm/ID giữa split và có đủ 12 tổ hợp nhãn–ngôn ngữ trong mỗi phần.
- Phép dò gần đúng có thể bỏ sót bản dịch/paraphrase; nhóm gần trùng không đồng nghĩa bản sao ngữ nghĩa đã xác nhận.

Manifest, cặp gần trùng, hash và báo cáo: `manifest.csv`, `near_pairs.csv`, `data_hashes.json`, `data_report.json`, `DATA_REPORT.md` trong `artifacts/prepared/`.

## Điều kiện đầu ra

Xác suất hữu hạn, đúng thứ tự ba nhãn và khớp ID trước ensemble. CSV cuối phải có header chính xác `id,label`, 6.000 dòng, đúng thứ tự test, không thiếu/thừa/trùng ID, không cột index. Nạp lại checkpoint và predict một batch thành công trước inference toàn bộ.
