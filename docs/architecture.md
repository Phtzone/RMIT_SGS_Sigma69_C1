# Kiến trúc v5: Wolf Defender và preset DistilBERT QLoRA

Notebook đang chạy là [final_97.ipynb](../final_97.ipynb), mặc định Wolf Defender full fine-tuning. DistilBERT QLoRA được giữ làm preset thay thế và để đối chiếu với v1/v2. Chỉ chọn một pretrained backbone trong mỗi lần chạy. Wolf đã có trong danh sách BTC: [bằng chứng nguồn](../demo/competition-challenge.json), [đối chiếu notebook Melbourne](melbourne-reference-review.md).

## 1. Luồng xử lý

```text
train/test CSV
    -> schema validation và EDA
    -> stratified train/validation split
    -> tokenizer tương ứng backbone đã chọn
    -> Wolf full fine-tuning HOẶC DistilBERT 4-bit NF4 + LoRA
    -> classification head 3 lớp
    -> weighted loss và Macro F1
    -> selection: best checkpoint + TF-IDF/SVM + cross-fit hậu xử lý toàn cục
    -> nested cross-fit hiệu chỉnh nhỏ BENIGN/JAILBREAK tiếng Bồ Đào Nha
    -> khóa epoch và ensemble params
    -> tùy chọn refit model mới trên toàn bộ train, không eval holdout cũ
    -> test inference với tham số hậu xử lý đã khóa
    -> submission validator
    -> submission.csv
```

## 2. Backbone và label mapping

```text
MODEL_PRESET = wolf  # hoặc distilbert
wolf = patronus-studio/wolf-defender-prompt-injection
distilbert = distilbert/distilbert-base-multilingual-cased
BENIGN = 0
JAILBREAK = 1
PROMPT_INJECTION = 2
```

Label mapping phải được định nghĩa một lần trong config hoặc module dùng chung. Không lặp mapping riêng trong train và inference.

## 3. Cấu hình model

### Wolf mặc định v5

Nguồn weights chính xác: `aisuko/wolf-defender-prompt-injection/transformers/default/1`. Trên Colab, để trống `WOLF_LOCAL_PATH` để cell model tải bằng KaggleHub; có thể thay bằng thư mục model đã tải/gắn trên Kaggle. Kiểm tra `model_type="modernbert"`, nạp tokenizer cùng nguồn và khởi tạo head ba lớp thay head hai lớp gốc.

Wolf fine-tune toàn bộ model, không quantization hoặc LoRA. V5 trở về cấu hình v3: tối đa 4 epoch, LR `3e-5`, early-stopping patience 2, class-weight power `0.5`, focal gamma `0`, train batch 8, gradient accumulation 4 và eval batch 16. Effective train batch là 32; seed và split seed đều bằng 42. Dùng gradient checkpointing và SDPA; giữ weights FP32, mixed precision FP16 trên T4, tắt ModernBERT reference compile.

Artifact Wolf có metadata từ Transformers 5, trong khi notebook khóa Transformers `4.57.6`. Mặc định `WOLF_LOADING_MODE="v3_legacy"` tái hiện cách v3 gọi `AutoConfig` và nạp tokenizer: giữ cách phiên bản 4.57 diễn giải config, bỏ hai trường tokenizer không tương thích như bản đã chạy. Chỉ chuẩn bị các file tokenizer nhỏ trong thư mục riêng, không sao chép weights. Chế độ `checkpoint_config` ánh xạ `rope_parameters` global/local theo metadata checkpoint như v4; phải coi đây là một thí nghiệm riêng, không trộn với baseline. Cách nạp và giá trị config thực tế được ghi vào provenance. Khôi phục cấu hình không bảo đảm tạo cùng trọng số hoặc cùng score với lần v3 cũ.

### DistilBERT QLoRA: baseline lịch sử và preset thay thế

DistilBERT là encoder tương đối nhỏ nên QLoRA không chắc nhanh hoặc chính xác hơn LoRA FP16. Phần dưới mô tả baseline lịch sử; cấu hình preset hiện tại được ghi riêng để tránh nạp nhầm.

### Lượng tử hóa

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

Kaggle T4 dùng `float16`; không bật `bfloat16`. Gọi `prepare_model_for_kbit_training()` trước khi gắn adapter.

### LoRA

Baseline thống nhất:

| Tham số | Giá trị khởi đầu |
|---|---|
| `task_type` | `SEQ_CLS` |
| `target_modules` | `q_lin`, `v_lin` |
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0,05 |
| `bias` | `none` |
| `modules_to_save` | `pre_classifier`, `classifier` |

Sau khi tạo model, phải in và lưu số trainable parameters để xác nhận backbone thực sự được đóng băng và adapter/classification head đang được train.

Preset `distilbert` hiện tại giữ cấu hình từ v3: rank 64/alpha 128, sáu target `q_lin`, `k_lin`, `v_lin`, `out_lin`, `lin1`, `lin2`, tối đa 8 epoch, LR `5e-5`, patience 3, class-weight power `0.65`, focal gamma `2`. Đây là cấu hình thử nghiệm chưa có kết quả GPU mới, không phải tham số của Wolf.

## 4. Tokenization

- Wolf v5 mặc định dùng `max_length=256`, `head`, giống cấu hình v3 đã chạy. DistilBERT giữ `512` và `head_tail`. Cả hai dùng dynamic padding và giữ special tokens của chính tokenizer.
- DistilBERT không nhận chuỗi dài hơn giới hạn vị trí của model.
- V3 giữ 256 token đầu; output cho thấy 45/195 mẫu JAILBREAK tiếng Bồ Đào Nha trong validation bị cắt. V4 dùng 512 `head_tail` đạt Macro F1 tốt nhất `0.969274`, thấp hơn v3 `0.972933` theo log đã có. V4 đồng thời đổi cách nạp RoPE và batch nên không thể kết luận riêng tăng context gây giảm điểm. V5 giữ báo cáo lỗi theo độ dài/ngôn ngữ để chẩn đoán tiếp.
- Không thay tokenizer bằng tokenizer của backbone khác.

## 5. Train và validation

Cấu hình khởi đầu của baseline DistilBERT lịch sử (preset hiện tại ở mục 3):

| Tham số | Giá trị đề xuất |
|---|---|
| Epoch | 3, kèm early stopping |
| Learning rate | thử `1e-4`, `2e-4` |
| Effective batch size | 32 |
| Optimizer | AdamW/paged AdamW 8-bit nếu môi trường hỗ trợ |
| LR scheduler | linear hoặc cosine, warmup 5–10% |
| Seed | 42 |
| Best model metric | `macro_f1` |

Do mất cân bằng lớp, baseline dùng weighted cross entropy với trọng số tính từ train split. Nhánh evaluation chịu trách nhiệm so sánh weighted loss với focal loss hoặc resampling. Chỉ thay một nhóm tham số mỗi thí nghiệm.

V5 mặc định `USE_FULL_TRAIN=False`. Selection chỉ fit trên 90% train, tự chọn checkpoint theo Macro F1 cao nhất, fit TF-IDF/SVM trên cùng phần fit và cross-fit hậu xử lý trong validation. Không ép epoch 2 hoặc 3: v3 tốt nhất ở epoch 4, v4 tốt nhất ở epoch 3. Nếu chủ động đặt `USE_FULL_TRAIN=True`, khởi tạo model mới từ pretrained, tính lại class weights, fit model/vectorizer/SVM trên toàn bộ train theo epoch đã chốt; áp dụng nguyên tham số hậu xử lý, gồm hiệu chỉnh tiếng Bồ Đào Nha nếu được chấp nhận. Final refit không có `eval_dataset`, dùng `eval_strategy="no"`, không early stopping hoặc `load_best_model_at_end`. Cả hai chế độ đều xuất CSV hợp lệ.

Hậu xử lý toàn cục giữ grid v3: lexical weight `[0, 0.25, 0.5, 0.75, 0.9]`, bias toàn cục và guard cross-fit tăng ít nhất `0.002` so với neural thô. Không đạt thì giữ dự đoán thô. Bước mới chỉ xét dòng `language="pt"` mà dự đoán trước hiệu chỉnh là BENIGN hoặc JAILBREAK và chênh lệch tuyệt đối log-score hai lớp không quá `0.75`; chỉ thử delta `[-0.3, -0.15, 0, 0.15, 0.3]`. Các ngôn ngữ khác và dự đoán PROMPT_INJECTION không đổi bởi bước này.

Nested cross-fit so sánh nhánh toàn cục và nhánh có hiệu chỉnh trong từng outer fold, fit tham số bằng phần còn lại. Chỉ chấp nhận khi Macro F1 tổng tăng ít nhất `0.002`, Macro F1 tiếng Bồ Đào Nha tăng, ít nhất 2/3 outer fold tăng điểm và phép fit trên toàn validation không cho thấy giảm điểm. Không đạt điều kiện thì delta bằng 0. Lưu quyết định và số liệu trong `pt_pair_tuning.json`; `submission_selection_global.csv` giữ kết quả trước hiệu chỉnh, `submission_selection.csv` là kết quả được chọn. Tăng điểm fit trên toàn validation không được coi là bằng chứng độc lập.

Report validation chỉ thuộc selection. Cross-fit hậu xử lý vẫn chưa loại hết thiên lệch do validation từng dùng để chọn checkpoint/cấu hình. Không báo điểm trên holdout đã đưa vào full train như đánh giá độc lập. Test chỉ được dự đoán vì không có nhãn thật.

## 6. Tài nguyên Kaggle

- Notebook hiện yêu cầu một GPU; preset 4-bit không dùng distributed training.
- Không hard-code đường dẫn ngoài `/kaggle/input` và `/kaggle/working` trong notebook cuối.
- Wolf lưu full weights tại `selection_model`/`final_model`; DistilBERT lưu adapter/head tại `selection_adapter`/`final_adapter`. Cả hai lưu tokenizer, config, label mapping, metric và tham số ensemble. ZIP Wolf sẽ lớn hơn ZIP adapter.
- Khóa phiên bản `transformers`, `peft`, `accelerate` và `bitsandbytes` đã chạy thành công.

Nếu preset DistilBERT gặp lỗi bitsandbytes 4-bit, ghi lại lỗi và phiên bản package; không âm thầm đổi phương pháp train. Wolf là preset full fine-tuning đã được người dùng chấp nhận, không phải fallback tự động khi QLoRA lỗi.

## 7. Kiểm tra bắt buộc

- Model đã chọn load thành công trên GPU và chạy được một training step.
- Wolf train toàn bộ weights/head; DistilBERT chỉ train adapter/head với backbone đóng băng. Lưu trainable parameter report để kiểm chứng.
- Một batch chứa đủ `input_ids`, `attention_mask`, `labels` đúng dtype.
- `compute_metrics` trả về `macro_f1` và per-class F1.
- Inference không shuffle ID.
- Submission validator phát hiện thiếu ID, trùng ID, label sai và cột thừa.

V3 đã hoàn tất GPU training và đạt raw validation Macro F1 `0.972933` trong output đã lưu. Người dùng cung cấp log v4 tốt nhất `0.969274`, chưa cung cấp notebook đã chạy hoặc public score v4. Các mục trên cần kiểm chứng lại với v5; chưa có score GPU v5. Mục tiêu `0.99` chưa được xác nhận.
