# Kết quả C1 — Kaggle version 5

Ngày 22/10/2026. [Notebook Kaggle](https://www.kaggle.com/code/phtzone/rmit-sgs-sigma69-c1). **COMPLETE sau khoảng 62 phút trên T4 ×2. Chưa gửi submission.**

Model thắng: **mDeBERTa-v3-base, 256 token đầu–cuối, cross-entropy thường**. Dev sau điều chỉnh quyết định **0,949005**; audit **0,935382**. Audit thuộc model train trên split trước refit, không phải model cuối đã học toàn bộ 14.000 mẫu. Chưa xác nhận điểm leaderboard của CSV này.

## Thực nghiệm

Mọi thử nghiệm: 3 epoch, seed 42, LR 2e-5, microbatch 8 × accumulation 4, FP16. Thời gian là wall time từng worker; hai thử nghiệm trong mỗi giai đoạn chạy song song.

| Run | Token | Weighted CE | Dev Macro F1 thô | Epoch tốt nhất | Phút/worker |
|---|---:|---|---:|---:|---:|
| XLM-R baseline | 256 | Không | 0,926540 | 3 | 9,49 |
| mDeBERTa baseline | 256 | Không | 0,946733 | 3 | 15,65 |
| mDeBERTa long | 512 | Không | 0,938939 | 3 | 23,86 |
| mDeBERTa weighted | 256 | Có | 0,942487 | 3 | 15,12 |

Điểm này thay thế các checkpoint trung gian đã báo khi chạy. Weighted CE và 512 token không thắng baseline trong lần thử này.

## Phương án cuối

- Ensemble tốt nhất 25% XLM-R + 75% mDeBERTa đạt 0,947527, tăng 0,000794 (< 0,002), nên không giữ.
- Hệ số BENIGN / JAILBREAK / PROMPT_INJECTION = **1 / 0,8 / 1,25**, rồi argmax. Dev tăng 0,002272 lên 0,949005. Đây là điều chỉnh quyết định, không chứng minh xác suất được hiệu chuẩn.
- Seal selection SHA-256 hợp lệ; lựa chọn khóa trước audit, không dùng audit chọn cấu hình.
- Refit toàn bộ 14.000 mẫu đủ 3 epoch, hoàn tất trong **18,58 phút**; dùng cùng cấu hình và hệ số. Không báo điểm holdout của model refit đã học mọi dòng train.
- Revision: `a0484667b22365f84929a935b5e50a51f71f159d`.

## Audit: 2.096 mẫu

| Lớp | Precision | Recall | F1 | Số mẫu |
|---|---:|---:|---:|---:|
| BENIGN | 0,9578 | 0,9743 | 0,9660 | 1.282 |
| JAILBREAK | 0,9474 | 0,9503 | 0,9489 | 664 |
| PROMPT_INJECTION | 0,9762 | 0,8200 | 0,8913 | 150 |

Confusion matrix: hàng là nhãn thật, cột là dự đoán.

| Thật / Dự đoán | BENIGN | JAILBREAK | PROMPT_INJECTION |
|---|---:|---:|---:|
| BENIGN | 1.249 | 30 | 3 |
| JAILBREAK | 33 | 631 | 0 |
| PROMPT_INJECTION | 22 | 5 | 123 |

| Ngôn ngữ khai báo | Macro F1 audit | Số mẫu |
|---|---:|---:|
| en | 0,9024 | 217 |
| fr | 0,9626 | 512 |
| it | 0,9534 | 441 |
| pt | 0,9162 | 926 |

Injection bỏ sót **27/150**, recall 82%. Nhóm >1.024 ký tự có Macro F1 0,7897; injection dài đúng 5/14. Tiếng Anh chỉ có 14 injection; số mẫu nhỏ cần được nêu khi diễn giải.
11 injection thuộc nhóm gần trùng audit đều nhầm benign; các mẫu này không độc lập. Nhóm template toàn benign đúng 122/122; Macro F1 cố định ba lớp = 0,3333 do hai lớp vắng mặt, không phải accuracy 33%.

## Dữ liệu và tái lập

- Split Kaggle thực tế: **9.808 train / 2.096 dev / 2.096 audit**; đủ 12 tổ hợp nhãn × ngôn ngữ mỗi phần.
- Mỗi ID train có mặt đúng một lần; không giao nhóm; 262 cặp train–train không vượt ranh giới split.
- Ba CSV gốc có hash trùng local/Kaggle, dữ liệu không bị sửa.
- Manifest Kaggle khác bản chuẩn bị local ở **6.477 ID về phần chia**, group ID giống nhau. Kaggle scikit-learn 1.6.1 / NumPy 2.1.3; local 1.9.1 / 2.5.4. Chưa tái lập riêng nguyên nhân. Báo cáo này dùng manifest Kaggle.
- Vòng sau tái sử dụng manifest Kaggle và pin môi trường. Audit đã được xem; không dùng nó chọn cấu hình mới rồi gọi cùng tập là kiểm tra độc lập mới.

| CSV | SHA-256 |
|---|---|
| train.csv | `b6809aea692dbbb4978f45fa603bb001e10ebfbffe9bfbff964e44e2856be795` |
| test.csv | `137925562e9b8403a6f725431489b73b8f267d24423784d1ea0f0a7dc4a455c6` |
| sample_submission.csv | `b25f6188461fb1ad555aa5ce5f7fe7eba356827e8bd541c63b9f59bf6f594f73` |

## Submission và kiểm chứng

- [submission.csv](../artifacts/kaggle-v5-output/c1-output/submission.csv): **6.000 dòng**, header `id,label`, ID đầy đủ/duy nhất/đúng thứ tự, nhãn hợp lệ, không có cột index.
- Dự đoán: **3.859 BENIGN / 1.780 JAILBREAK / 361 PROMPT_INJECTION**, không phải nhãn thật test.
- CSV khớp toàn bộ xác suất refit + chính sách khóa. Đã kiểm tra 12 NPZ; tính lại Macro F1/confusion matrix dev và audit khớp báo cáo.
- Pipeline nạp lại checkpoint và thử một batch trước inference toàn test; inference cuối thành công.
- SHA-256 submission: `36146d8ac0f8d5ae219c4701378c6f3ecc603849f755a9fddc9260db7c0332b4`.
- [Bằng chứng kiểm chứng](../artifacts/kaggle-v5-output/result-verification.json).
- Trọng số cuối đã có trên Kaggle tại `c1-output/runs/mdeberta_full/final`; chưa xác nhận tải đủ bản sao lớn về local. CSV và báo cáo đã có local. Xem [monitor-status.json](../artifacts/kaggle-v5-output/monitor-status.json).

## Artifact nguồn

- [experiments.json](../artifacts/kaggle-v5-output/c1-output/experiments.json)
- [selection.json](../artifacts/kaggle-v5-output/c1-output/selection.json)
- [dev_metrics.json](../artifacts/kaggle-v5-output/c1-output/dev_metrics.json)
- [audit_metrics.json](../artifacts/kaggle-v5-output/c1-output/audit_metrics.json)
- [final_models.json](../artifacts/kaggle-v5-output/c1-output/final_models.json)
- [manifest.csv](../artifacts/kaggle-v5-output/c1-output/prepared/manifest.csv)

**Không gửi hoặc chọn final submission trên Kaggle.**
