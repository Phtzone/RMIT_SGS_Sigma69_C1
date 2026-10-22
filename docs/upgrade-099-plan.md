# Kế hoạch nâng cấp Notebook hướng tới Macro F1 0.99
**Dự án:** RMIT_SGS_Sigma69_C1 — Multilingual Prompt-Attack Classification  
**Notebook hiện tại:** [final_97.ipynb](../final_97.ipynb), bản v5 dựa trên cấu hình Wolf v3 đã chạy.  
**Ngày:** 22/10/2026 | **Deadline:** 18:00 cùng ngày

## Kết quả mới nhất và hướng thử nghiệm v5

[Bản v3 đã chạy](../notebooks/archive/final_97_executed.ipynb) được bảo toàn nguyên trạng. Raw validation Macro F1 tại epoch 4 là **0.972933** trên 1.400 mẫu; người dùng báo public khoảng **0.97**. Log v4 người dùng cung cấp có Macro F1 theo epoch là `0.938744`, `0.960004`, **`0.969274`**, `0.967447`: tốt nhất ở epoch 3, thấp hơn v3 khoảng `0.003659`. Chưa có notebook v4 đã chạy hoặc public score v4 ở repo; [bản archive v4](../notebooks/archive/final_97_v4_source.ipynb) chỉ là source. V5 chưa có score GPU; `0.99` vẫn là mục tiêu cần kiểm chứng.

V3 còn 35 lỗi validation, trong đó 27 lỗi nhầm BENIGN/JAILBREAK tiếng Bồ Đào Nha. Có 45/195 mẫu JAILBREAK tiếng Bồ Đào Nha bị cắt ở context 256; chưa có ID lỗi để chứng minh truncation là nguyên nhân. V4 thay đồng thời context 512 `head_tail`, RoPE/tokenizer và batch, nên không thể kết luận riêng context dài hơn gây giảm điểm. V5 quay về 256 token đầu, batch/cách nạp v3 và chỉ thêm hiệu chỉnh nhỏ cho cặp BENIGN/JAILBREAK tiếng Bồ Đào Nha sau hậu xử lý toàn cục. Không đổi class-weight power hoặc LR cùng lúc; không thêm backbone, seed ensemble hoặc dữ liệu ngoài.

Upload **`final_97.ipynb` ở thư mục gốc** và `rmit-hackathon-2026.zip` lên Colab, chạy từ đầu trong runtime mới. Mặc định `USE_FULL_TRAIN=False` để dùng CSV selection trước; refit chỉ chạy khi chủ động bật `True`. Lần v3 ở 256 token mất khoảng 27:37 selection + 26:48 refit trên T4; đây là số đo lịch sử, không cam kết thời gian lần mới. Khôi phục cấu hình không bảo đảm tái lập đúng trọng số hoặc điểm v3.

## Cập nhật hướng triển khai: Wolf Defender đã được BTC duyệt

Người dùng cho phép chuyển sang Wolf nếu thuộc danh sách BTC. Đã xác minh trang Challenge qua API công khai Kaggle `competitions.PageService/ListPages`, competition ID `169360`, page ID `877551`: danh sách hiện có 10 model và bao gồm `patronus-studio/wolf-defender-prompt-injection`, với [Kaggle model của BTC](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default). Điều kiện chuyển model đã được đáp ứng. [Nguồn chính thức](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview/challenge).

Hướng hiện tại ưu tiên Wolf Defender theo notebook Melbourne và giữ pipeline validation/submission của nhóm. Các mục LoRA/focal/hyperparameter DistilBERT dưới đây là **phương án DistilBERT đã đề xuất trước khi người dùng cho phép đổi backbone**, không phải cấu hình bắt buộc của Wolf. Phần tách selection/refit, tránh rò rỉ và đánh giá hậu xử lý tiếp tục áp dụng cho cả hai hướng. Chi tiết nguồn tham khảo tại [melbourne-reference-review.md](melbourne-reference-review.md).

### Cấu hình Wolf mặc định v5

| Thành phần | Cấu hình |
|---|---|
| Backbone | `patronus-studio/wolf-defender-prompt-injection`, full fine-tuning |
| Cách nạp | `WOLF_LOADING_MODE="v3_legacy"`, tái hiện config/tokenizer v3; `checkpoint_config` là thử nghiệm riêng dùng ánh xạ RoPE v4 |
| Selection | Tối đa 4 epoch, LR `3e-5`, early-stopping patience 2, tự chọn Macro F1 cao nhất; seed/split seed 42 |
| Input | 256 token `head`, dynamic padding, giữ special tokens |
| Loss | Class-weight power `0.5`, focal gamma `0` |
| Batch | Train batch 8, accumulation 4, effective batch 32; eval batch 16 |
| Bộ nhớ / precision | Gradient checkpointing, SDPA, trọng số FP32 và mixed precision FP16 trên T4; tắt reference compile |
| Hậu xử lý toàn cục | Grid v3: lexical weight `[0, 0.25, 0.5, 0.75, 0.9]` và bias toàn cục, guard cross-fit gain ≥ `0.002` |
| Hiệu chỉnh tiếng Bồ Đào Nha | Chỉ cặp BENIGN/JAILBREAK gần ranh giới; delta `[-0.3, -0.15, 0, 0.15, 0.3]`; có nested cross-fit và tự tắt khi không đạt điều kiện |
| Final | Mặc định `USE_FULL_TRAIN=False`; khi bật `True`, refit mới trên 14k với epoch và tham số hậu xử lý đã khóa |

Model vẫn là `aisuko/wolf-defender-prompt-injection/transformers/default/1`; bằng chứng danh sách BTC được lưu ở [competition-challenge.json](../demo/competition-challenge.json). Cách nạp `v3_legacy` giữ cách `AutoConfig` của Transformers 4.57.6 diễn giải checkpoint như lần v3, đồng thời thực hiện bản vá tokenizer v3 trên các file nhỏ trong thư mục riêng, không copy toàn bộ weights.

Hiệu chỉnh mới chỉ được phép tác động dòng `pt` có dự đoán trước hiệu chỉnh là BENIGN/JAILBREAK và độ chênh tuyệt đối log-score hai lớp không quá `0.75`. Nested cross-fit phải cho gain tổng ít nhất `0.002`, gain tiếng Bồ Đào Nha dương, ít nhất 2/3 outer fold tăng và fit trên toàn validation không giảm điểm. Nếu không đạt, giữ delta bằng 0; các ngôn ngữ khác và dự đoán PROMPT_INJECTION không đổi bởi bước này. `pt_pair_tuning.json` lưu quyết định và số liệu. `submission_selection_global.csv` giữ kết quả trước hiệu chỉnh; `submission_selection.csv` giữ kết quả được chọn. Cùng một lần train cung cấp cả hai để đối chiếu, không cần train lại chỉ để bật/tắt bước hiệu chỉnh.

Không ép chọn epoch 2 hoặc 3: v3 tốt nhất ở epoch 4, v4 ở epoch 3; `load_best_model_at_end=True` chọn theo kết quả lần mới. Chỉ nộp file sau khi xem báo cáo validation và kiểm tra CSV. Nested cross-fit vẫn không phải đánh giá hoàn toàn độc lập vì validation từng được dùng chọn checkpoint và quyết định phương pháp thử; không suy ra score private hoặc bảo đảm tăng lên `0.99`.

---

## Kết quả v2 trước khi chuyển sang Wolf — giữ để đối chiếu

| Metric | Giá trị |
|---|---|
| Validation Macro F1 (raw, epoch 5) | **0.9197** |
| Validation Macro F1 (post-processing cross-fit) | **0.952069** |
| Validation Macro F1 (fit post-processing trên toàn validation) | **0.958402** |
| Public leaderboard v1 (người dùng báo lại) | **0.87** |
| Public leaderboard v2 | Chưa có bằng chứng trong repository |

`0.99` là mục tiêu thử nghiệm, chưa phải kết quả hoặc mức điểm được bảo đảm. Điểm `0.958402` dùng chính validation để chọn bias/ensemble rồi chấm; cross-fit giảm phần thiên lệch đó nhưng vẫn dùng tập đã chọn checkpoint. Không lấy validation v2 trừ public v1 để suy ra khoảng cách do phân bố test.

### Điểm yếu theo ngôn ngữ của v2 (validation)

| Language | Macro F1 | F1_BENIGN | F1_JAILBREAK | F1_PROMPT_INJECTION |
|---|---|---|---|---|
| `pt` | 0.883 | 0.895 | **0.802** ← yếu | 0.953 |
| `en` | 0.889 | 0.983 | 0.977 | **0.706** ← yếu |
| `fr` | 0.949 | 0.981 | 0.991 | 0.875 |
| `it` | 0.976 | 0.988 | 0.989 | 0.950 |

**Kết luận chẩn đoán:**
- `JAILBREAK` tiếng Bồ chiếm 195 mẫu validation → kéo điểm chung xuống mạnh nhất.
- `PROMPT_INJECTION` tiếng Anh chỉ ~10 mẫu validation → F1 biến động mạnh khi thay đổi vài dự đoán; chưa thể kết luận thiếu dữ liệu là nguyên nhân duy nhất.
- Epoch 5 là checkpoint tốt nhất trong lần chạy đã lưu → có cơ sở thử thêm epoch. Loss train giảm không tự chứng minh chưa overfit.

---

## Kế hoạch DistilBERT trước đó và nguyên tắc refit dùng chung

### 1. Refit toàn bộ 14.000 mẫu sau selection

**Vấn đề:** Notebook đang giữ 10% (1.400 mẫu) làm validation. Có thể dùng toàn bộ 14.000 mẫu khi tạo model final, sau khi hoàn tất chọn epoch/cấu hình bằng holdout.

**Thêm vào cell cấu hình (Cell 2):**

```python
USE_FULL_TRAIN = True   # Selection 90/10 trước; sau đó refit mới trên 14k
```

**Giữ split selection và bổ sung một giai đoạn refit riêng:**

```python
fit_df, valid_df = train_test_split(
    train_df, test_size=VAL_SIZE, random_state=SPLIT_SEED,
    stratify=train_df["language"] + "_" + train_df["label"]
)
# 1. Selection: fit trên fit_df, chọn best epoch bằng valid_df.
#    Fit vectorizer/SVM chỉ trên fit_df; cross-fit hậu xử lý trên validation.
# 2. Khóa best epoch và toàn bộ tham số/điều kiện bật ensemble.
# 3. Nếu USE_FULL_TRAIN: tạo model + adapter/head MỚI từ pretrained,
#    train toàn bộ train_df trong số epoch đã chọn, tính lại class weights.
#    Fit lại vectorizer/SVM trên train_df; dự đoán test với ensemble đã khóa.
#    Không truyền valid_df làm eval_dataset trong giai đoạn này.
```

> [!WARNING]
> Final refit phải đặt `eval_strategy="no"`, `load_best_model_at_end=False`, bỏ `metric_for_best_model`, `eval_dataset` và early-stopping callback. Không tạo holdout nằm trong tập đã fit. Validation report chỉ thuộc model selection; model final không được chấm trên holdout cũ mà nó đã học. Giữ submission selection để đối chiếu. Chuyển bias từ model selection sang refit là giả định chưa được kiểm chứng độc lập.

---

### 2. Tăng số Epoch và giảm Learning Rate

**Giả thuyết:** V2 đạt checkpoint tốt nhất ở epoch 5; thử tăng giới hạn epoch và giảm learning rate, tiếp tục theo dõi validation để phát hiện overfit.

**Thay đổi trong cell cấu hình (Cell 2):**

```python
# Trước
EPOCHS = 5
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 2

# Sau
EPOCHS = 8
LEARNING_RATE = 5e-5     # giảm LR để hội tụ tinh hơn giai đoạn cuối
EARLY_STOPPING_PATIENCE = 3
```

Đây là learning rate của preset DistilBERT đề xuất từ v3, không phải cấu hình Wolf v5 và không tự resume v2. Early stopping chỉ áp dụng cho selection; refit dùng số epoch đã chốt.

---

### 3. Focal Loss thay vì Weighted Cross-Entropy

**Giả thuyết:** Focal Loss nhấn mạnh mẫu khó; cần kiểm chứng liệu có cải thiện PROMPT_INJECTION tiếng Anh hay chỉ tập trung thêm vào nhiễu/nhãn khó.

**Thêm vào cell cấu hình (Cell 2):**

```python
FOCAL_GAMMA = 2.0   # 0 = CE thường, 2 = focal chuẩn
```

**Thay thế hàm `weighted_loss` trong Cell 10:**

```python
def weighted_focal_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits.float()
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)  # Xác suất lớp thật, không detach gradient.
    weight = class_weights.to(device=logits.device, dtype=logits.dtype)[labels]
    focal_loss = weight * ((1.0 - pt) ** FOCAL_GAMMA) * ce_loss
    count = labels.numel() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(count, device=logits.device, dtype=logits.dtype).clamp_min(1)
    return focal_loss.sum() / denominator
```

`p_t` phải được suy ra từ CE chưa nhân trọng số lớp; `exp(-weighted_ce)` không phải xác suất lớp thật. Khi gamma bằng 0, công thức trở về weighted CE chuẩn hóa theo số mẫu.

**Trong cell Trainer, đổi:**

```python
# Trước
compute_loss_func=weighted_loss,

# Sau
compute_loss_func=weighted_focal_loss,
```

---

### 4. Tăng LoRA Rank

**Thay đổi trong cell cấu hình (Cell 2):**

```python
# Trước
LORA_R = 32
LORA_ALPHA = 64

# Sau
LORA_R = 64      # tăng gấp đôi capacity adapter
LORA_ALPHA = 128  # giữ tỉ lệ alpha/r = 2
```

In số tham số trainable và bộ nhớ thực tế sau khi tạo model. Chưa đo VRAM của preset DistilBERT này nên chưa thể khẳng định chắc chắn cấu hình vừa T4.

> [!TIP]
> Nếu OOM: đặt `TRAIN_BATCH_SIZE = 4`, `GRAD_ACCUM_STEPS = 8` để giữ effective batch 32.

---

### 5. Tăng CLASS_WEIGHT_POWER

**Thay đổi trong cell cấu hình (Cell 2):**

```python
# Trước
CLASS_WEIGHT_POWER = 0.5   # sqrt → INJECTION weight = 2.35

# Sau
CLASS_WEIGHT_POWER = 0.65  # compromise giữa sqrt và inverse-freq
```

---

### 6. Bias riêng theo ngôn ngữ — đề xuất DistilBERT trước đó

Post-processing DistilBERT có global bias và ensemble QLoRA/SVM. Bias grid của notebook Melbourne đã có chức năng tương tự trong v2; không coi đó là cải tiến hoàn toàn mới. Đề xuất lịch sử này áp dụng bias theo ngôn ngữ rộng hơn, khác bước v5 chỉ tác động cặp BENIGN/JAILBREAK tiếng Bồ Đào Nha được giới hạn ở đầu tài liệu.

> [!WARNING]
> Per-language bias dễ overfit trên validation nhỏ (~10 mẫu `en` INJECTION). **Bỏ qua nếu ít thời gian.** Chỉ bật nếu cross-fit gain ≥ 0.005.

---

## Thứ tự thử nghiệm DistilBERT trước đó — không áp dụng mặc định cho Wolf

| # | Thay đổi | Cần kiểm chứng | Mức tăng F1 |
|---|---|---|---|
| **1** | Selection rồi refit toàn bộ 14k | Không rò rỉ holdout, epoch/ensemble được khóa | Chưa đo |
| **2** | EPOCHS=8, LR=5e-5, PATIENCE=3 | Checkpoint và per-class F1 tốt nhất | Chưa đo |
| **3** | Focal Loss (gamma=2) | Công thức, gradient và precision/recall | Chưa đo |
| **4** | LoRA r64, alpha=128 | Trainable parameters, VRAM | Chưa đo |
| **5** | CLASS_WEIGHT_POWER=0.65 | Recall lớp hiếm và false positive | Chưa đo |

Không có căn cứ cộng các mức tăng của từng thay đổi hoặc dự báo từ public v1 `0.87` lên `0.99`. Khi thay đồng thời nhiều yếu tố, cần ablation nếu muốn xác định nguyên nhân cải thiện. Các đề xuất DistilBERT này chưa có kết quả GPU; kết quả Wolf v3 được ghi ở đầu tài liệu.

---

## Checklist trước khi submit

- [ ] Giữ `USE_FULL_TRAIN=False` để xem selection trước; chỉ bật `True` nếu chủ động thử refit
- [ ] Giữ bản v2 đã chạy trong `notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_v2_executed.ipynb`
- [ ] Giữ bản v3 đã chạy trong `notebooks/archive/final_97_executed.ipynb`
- [ ] Giữ source v4 trong `notebooks/archive/final_97_v4_source.ipynb`; không gọi đây là notebook chứa output train
- [ ] Hoàn tất selection trước refit; lưu best epoch, split IDs và ensemble params
- [ ] Final refit không evaluation trên các dòng đã học; lưu riêng submission selection
- [ ] Đổi `RUN_NAME` để tránh load nhầm checkpoint cũ
- [ ] Wolf v5: context 256 `head`, batch 8/accumulation 4, eval batch 16, `WOLF_LOADING_MODE="v3_legacy"`; tối đa 4 epoch, LR `3e-5`, patience 2, power `0.5`, gamma `0`
- [ ] So sánh `pt_pair_tuning.json` và hai file selection trước/sau hiệu chỉnh; giữ fallback khi không qua điều kiện cross-fit
- [ ] Nếu chủ động chọn lại DistilBERT: 8 epoch, LR `5e-5`, patience 3, r64/alpha128, power `0.65`, gamma `2`
- [ ] Chạy toàn bộ notebook từ đầu không lỗi
- [ ] `submission.csv` có đúng **6.000 dòng** + header `id,label`
- [ ] Tất cả nhãn đúng chính tả: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`
- [ ] Không có index pandas thừa
- [ ] Kiểm tra lượt submit còn lại trước khi bấm nộp

---

## Lưu ý

> [!CAUTION]
> DistilBERT multilingual là lựa chọn ban đầu của dự án, không phải model duy nhất được phép. Wolf Defender đã được xác minh trong danh sách BTC; dùng một backbone được duyệt cho mỗi giải pháp. Không ensemble thêm DistilBERT với Wolf. Xem [đối chiếu notebook Melbourne](melbourne-reference-review.md).

> [!NOTE]
> Validation v2 và public v1 thuộc hai phiên bản khác nhau. Chưa có cơ sở dùng khoảng cách này để kết luận distribution shift hoặc bảo đảm refit toàn bộ train sẽ thu hẹp khoảng cách.

> [!TIP]
> Bật `MOUNT_DRIVE = True` và đặt `OUTPUT_ROOT` trên Drive để giữ checkpoint nếu Colab ngắt. Selection rồi refit tốn hai lượt train; chưa có phép đo thời lượng của bản mới.
