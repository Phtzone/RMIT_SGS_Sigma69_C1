# Kiến trúc DistilBERT multilingual + QLoRA

## 1. Luồng xử lý

```text
train/test CSV
    -> schema validation và EDA
    -> stratified train/validation split
    -> DistilBERT tokenizer
    -> DistilBERT 4-bit NF4 + LoRA adapters
    -> classification head 3 lớp
    -> weighted loss và Macro F1
    -> test inference
    -> submission validator
    -> submission.csv
```

## 2. Backbone và label mapping

```text
model_id = distilbert/distilbert-base-multilingual-cased
BENIGN = 0
JAILBREAK = 1
PROMPT_INJECTION = 2
```

Label mapping phải được định nghĩa một lần trong config hoặc module dùng chung. Không lặp mapping riêng trong train và inference.

## 3. Cấu hình QLoRA baseline

DistilBERT là encoder tương đối nhỏ nên QLoRA không chắc nhanh hoặc chính xác hơn LoRA FP16. Tuy nhiên QLoRA là kiến trúc đã được nhóm chọn và phải được triển khai đúng để tạo baseline.

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

## 4. Tokenization

- Baseline dùng `max_length=512`, truncation và dynamic padding.
- DistilBERT không nhận chuỗi dài hơn giới hạn vị trí của model.
- Vì prompt có thể rất dài, nhóm cần so sánh ít nhất:
  - lấy 512 token đầu;
  - chiến lược head-tail, giữ cả phần đầu và cuối prompt.
- Không thay tokenizer bằng tokenizer của backbone khác.

## 5. Train và validation

Cấu hình khởi đầu:

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

## 6. Tài nguyên Kaggle

- Ưu tiên một GPU T4 để tránh xung đột giữa model 4-bit, `device_map="auto"` và distributed training.
- Không hard-code đường dẫn ngoài `/kaggle/input` và `/kaggle/working` trong notebook cuối.
- Lưu LoRA adapter, tokenizer, config, label mapping và metric; không cần lưu lại toàn bộ backbone.
- Khóa phiên bản `transformers`, `peft`, `accelerate` và `bitsandbytes` đã chạy thành công.

Nếu bitsandbytes 4-bit không tương thích với Kaggle image, ghi lại lỗi và phiên bản package trước khi thay đổi hướng. Fallback LoRA FP16 chỉ được dùng sau khi nhóm thống nhất vì không còn là QLoRA.

## 7. Kiểm tra bắt buộc

- QLoRA model load thành công trên GPU và chạy được một training step.
- Trainable parameter ratio hợp lý; classification head được train.
- Một batch chứa đủ `input_ids`, `attention_mask`, `labels` đúng dtype.
- `compute_metrics` trả về `macro_f1` và per-class F1.
- Inference không shuffle ID.
- Submission validator phát hiện thiếu ID, trùng ID, label sai và cột thừa.
