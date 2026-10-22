"""Build the standalone V3 notebook from the preserved, executed V2 notebook.

Does not install packages, download weights or run training. Rebuilding clears
outputs in both active notebook copies; the original run is archived once.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "RMIT_SGS_Sigma69_C1_Colab.ipynb"
UPLOADED = ROOT / "RMIT_SGS_Sigma69_C1_Colab (1).ipynb"
BACKUP = ROOT / "notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_v2_executed.ipynb"
if not BACKUP.exists():
    original = UPLOADED.read_bytes()
    parsed = json.loads(original)
    assert any(c.get("outputs") for c in parsed["cells"]), "Expected executed V2 source"
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_bytes(original)
notebook = json.loads(BACKUP.read_text(encoding="utf-8"))
cells = notebook["cells"]


def source(index):
    value = cells[index]["source"]
    return value if isinstance(value, str) else "".join(value)


def put(index, text):
    cells[index]["source"] = text.strip() + "\n"


def replace(index, old, new):
    text = source(index)
    assert old in text, (index, old)
    put(index, text.replace(old, new))


put(0, '''# RMIT_SGS_Sigma69_C1 — V3, học từ notebook Melbourne

**Backbone mặc định: `patronus-studio/wolf-defender-prompt-injection` (ModernBERT),
fine-tune toàn bộ**, từ Kaggle `aisuko/wolf-defender-prompt-injection/transformers/default/1`.

Chạy trên **Colab GPU trong runtime mới**. Upload ZIP dữ liệu ở mục 3 hoặc dùng Drive.
Notebook độc lập; không cần upload các script trong repository.

**Kết quả đã có:** DistilBERT V2 raw validation Macro F1 `0.919711`,
cross-fit postprocess `0.952069`, fitted validation `0.958402`.
Notebook Melbourne dùng Wolf Defender/ModernBERT: raw validation `0.960082`,
fitted bias `0.964711`; public `0.97` do người dùng cung cấp.
Code Melbourne ghi 4 epoch nhưng output lưu 2 epoch, nên đây chưa phải bản tái lập chính xác.

**V3:** chọn một backbone bằng `MODEL_PRESET`; mở rộng global bias theo Melbourne,
chỉ dùng hiệu chỉnh nếu qua cross-fit; giữ checkpoint theo raw Macro F1;
sau đó tùy chọn train lại từ pretrained trên đủ 14.000 mẫu với số epoch đã chọn.
`USE_FULL_TRAIN=True` tốn thêm một lượt train. Full refit không có validation độc lập;
mọi báo cáo validation thuộc mô hình ở bước chọn cấu hình.

`distilbert`: QLoRA r64, 8 epoch tối đa, focal loss, head-tail 512 token.
`wolf`: full fine-tune ModernBERT, weighted CE, 4 epoch tối đa, 256 token như Melbourne.
Wolf đã được đối chiếu trong danh sách được phép trên trang Challenge của BTC, 22/10/2026.
Hai backbone không được train/blend cùng nhau. TF-IDF + SVM chỉ học từ train.csv.

Mục tiêu **0.99 chưa được kiểm chứng**; chưa chạy GPU bản V3.
Bản V2 đã chạy được lưu nguyên vẹn tại `notebooks/archive/` trong repo.
Cuối notebook tạo `/content/submission.csv` và ZIP artifacts. Tải CSV bằng panel Files;
ZIP chứa model/adapter và báo cáo, không phải prediction file để nộp Kaggle.
''')

replace(2, '"matplotlib>=3.8,<4", "safetensors>=0.4.3",',
        '"matplotlib>=3.8,<4", "safetensors>=0.4.3", "kagglehub>=0.3.13,<1",')

put(4, '''# Chon MOT backbone. Chay lai tu dau khi doi preset, khong tai dung logits cu.
MODEL_PRESET = "wolf"                       # "wolf" hoac "distilbert"; chi MOT backbone
WOLF_APPROVAL_CONFIRMED = True              # Challenge page 877551, 22/10/2026
WOLF_APPROVAL_SOURCE = "https://www.kaggle.com/competitions/rmit-hackathon-2026/overview/challenge"
WOLF_KAGGLE_HANDLE = "aisuko/wolf-defender-prompt-injection/transformers/default/1"
WOLF_LOCAL_PATH = ""                        # Tuy chon: folder model da tai/gan tren Kaggle

DATA_MODE = "upload"                       # "upload" hoac "path"
DATA_DIR = "/content/rmit-hackathon-2026"
MOUNT_DRIVE = False
OUTPUT_ROOT = "/content/rmit_c1_runs"        # Doi sang Drive de giu checkpoints
SEED = 42
SPLIT_SEED = 42
VAL_SIZE = 0.10
USE_FULL_TRAIN = True                      # 90/10 selection -> fresh refit tren 14k
FULL_TRAIN_EPOCHS = None                   # None = epoch cua checkpoint tot nhat
TRAIN_BATCH_SIZE = 8                       # OOM: giam 4, tang GRAD_ACCUM_STEPS len 8
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
USE_CLASS_WEIGHTS = True
GRADIENT_CHECKPOINTING = True

if MODEL_PRESET == "distilbert":
    MODEL_SOURCE = "distilbert/distilbert-base-multilingual-cased"
    MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
    EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE = 8, 5e-5, 3
    MAX_LENGTH, LONG_TEXT_MODE = 512, "head_tail"
    CLASS_WEIGHT_POWER, FOCAL_GAMMA = 0.65, 2.0
elif MODEL_PRESET == "wolf":
    if not WOLF_APPROVAL_CONFIRMED or not WOLF_APPROVAL_SOURCE.strip():
        raise ValueError("Can doi chieu whitelist BTC cho Wolf va ghi WOLF_APPROVAL_SOURCE.")
    MODEL_SOURCE = WOLF_LOCAL_PATH          # Tu tai dung model version 1 neu de trong
    MODEL_REVISION = None
    EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE = 4, 3e-5, 2
    MAX_LENGTH, LONG_TEXT_MODE = 256, "head"
    CLASS_WEIGHT_POWER, FOCAL_GAMMA = 0.5, 0.0  # Weighted CE de bat dau gan Melbourne
else:
    raise ValueError("MODEL_PRESET phai la distilbert hoac wolf.")
RUN_NAME = f"{MODEL_PRESET}_v3"

# Cac tham so LoRA chi dung khi MODEL_PRESET == "distilbert".
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.05
LORA_TARGETS = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
HEAD_MODULES = ["pre_classifier", "classifier"]
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
LANGUAGES = ["pt", "fr", "it", "en"]
ENABLE_LEXICAL_BRANCH = True
LEXICAL_MAX_FEATURES, LEXICAL_C, LEXICAL_GLOBAL_WEIGHT = 300000, 2.0, 0.5
MIN_POSTPROCESS_GAIN = 0.002
# Giu cac moc bias nho V2 va them range rong Melbourne; KHONG copy bias da fit cua ho.
POSTPROCESS_FIT_KWARGS = {
    "jailbreak_bias_values": [-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1.],
    "injection_bias_values": [-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1., 1.5, 2.],
    "bias_l2": 0.002,
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
assert DATA_MODE in {"upload", "path"}
assert LONG_TEXT_MODE in {"head", "head_tail"}
assert 0 < VAL_SIZE < 1
assert 8 <= MAX_LENGTH <= 512 and MAX_LENGTH % 8 == 0
assert TRAIN_BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0
assert EPOCHS >= 1 and 0 <= CLASS_WEIGHT_POWER <= 1 and FOCAL_GAMMA >= 0
assert 0 <= LEXICAL_GLOBAL_WEIGHT <= 1
print("Preset:", MODEL_PRESET, "| full refit:", USE_FULL_TRAIN)
print("Effective batch:", TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
''')

replace(5, '"safetensors", "joblib"]', '"safetensors", "joblib", "kagglehub"]')
replace(5, 'AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,',
        'AutoConfig, AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification,')
replace(5, 'ADAPTER_DIR = ARTIFACT_DIR / "adapter"',
        'ADAPTER_DIR = ARTIFACT_DIR / ("selection_model" if MODEL_PRESET == "wolf" else "selection_adapter")')
replace(5, 'print("Output:", RUN_DIR)', 'print("Output:", RUN_DIR)\nprint("Training preset:", MODEL_PRESET)')

put(9, '''## 4. Holdout để chọn checkpoint và class weights

Luôn chia 90/10 theo language × label, kể cả khi `USE_FULL_TRAIN=True`.
Chỉ bước refit cuối mới dùng toàn bộ train; không dùng validation đã học để chọn model.
Class weights được tính riêng trên tập train của từng bước.
''')
text = source(10)
start = text.index('class_counts =')
put(10, text[:start] + '''def compute_class_weights(label_ids, num_classes, power=0.65, enabled=True):
    ids = np.asarray(label_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0 or not np.isin(ids, np.arange(num_classes)).all():
        raise ValueError("Label IDs khong hop le.")
    if not 0 <= power <= 1:
        raise ValueError("Class weight power phai trong [0, 1].")
    counts = np.bincount(ids, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError("Moi class phai co mau train.")
    values = (len(ids) / counts.astype(np.float64)) ** (power if enabled else 0.0)
    values /= np.average(values, weights=counts)
    return values.astype(np.float32)

class_counts = np.bincount(fit_df["label_id"].to_numpy(), minlength=len(LABELS))
power = CLASS_WEIGHT_POWER if USE_CLASS_WEIGHTS else 0.0
weight_values = compute_class_weights(fit_df["label_id"], len(LABELS), power, USE_CLASS_WEIGHTS)
class_weights = torch.tensor(weight_values, dtype=torch.float32)
print("Selection training:", len(fit_df), "| Validation:", len(valid_df))
display(pd.DataFrame({"label": LABELS, "train_count": class_counts, "weight": weight_values}))
''')
replace(12, '# Vocabulary/IDF va classifier chi hoc tu fit_df, khong fit tren validation/test.',
        '# Chi fit tren frame co nhan cua tung stage; khong fit tren test.')
replace(12, 'ARTIFACT_DIR / "lexical.joblib"', 'ARTIFACT_DIR / "selection_lexical.joblib"')
replace(12, 'submission se dung DistilBERT.', 'submission se dung backbone da chon.')
put(11, '''### Nhánh bổ trợ TF-IDF ký tự + SVM

Học prompt không cắt ngắn, gồm classifier chung và classifier theo ngôn ngữ.
Ở selection, chỉ fit trên 12.600 mẫu; validation/test chỉ transform/predict.
Ở full refit, fit lại trên 14.000 mẫu có nhãn. Không fit trên test.csv.
Global/local cố định 50/50; cross-fit chọn mức phối hợp với backbone.
''')
put(13, '''## 5. Tokenizer và prompt dài

Wolf mặc định lấy đầu prompt trong 256 token theo Melbourne; DistilBERT giữ đầu–cuối trong 512 token.
Tokenizer tự thêm token đặc biệt trước khi cắt; giữ nguyên các token này ở cả hai chế độ.
Có thể thử head-tail/512 cho Wolf trong lần chạy riêng và so sánh holdout.
Lưu tỷ lệ bị cắt theo language/label; không thêm prefix ngôn ngữ vào đầu vào backbone.
''')

text = source(14)
tokenizer_tail = text[text.index('def encode_texts'):]
put(14, '''# Wolf dung dung Kaggle model/version trong notebook Melbourne, khong thay bang repo gan ten.
if MODEL_PRESET == "wolf":
    if not MODEL_SOURCE:
        import kagglehub
        try:
            MODEL_SOURCE = kagglehub.model_download(WOLF_KAGGLE_HANDLE)
        except Exception as exc:
            raise RuntimeError(
                "Khong tai duoc Wolf tu KaggleHub. Kiem tra Internet/Kaggle login, "
                "hoac dat WOLF_LOCAL_PATH toi folder model co config.json."
            ) from exc
    model_root = Path(MODEL_SOURCE)
    candidates = ([model_root / "config.json"] if (model_root / "config.json").is_file()
                  else list(model_root.rglob("config.json")))
    if len(candidates) != 1:
        raise ValueError("WOLF_LOCAL_PATH phai tro dung mot model folder co config.json.")
    MODEL_SOURCE = str(candidates[0].parent)

base_config = AutoConfig.from_pretrained(
    MODEL_SOURCE, revision=MODEL_REVISION, trust_remote_code=False,
)
expected_type = "modernbert" if MODEL_PRESET == "wolf" else "distilbert"
if base_config.model_type != expected_type:
    raise ValueError(f"Preset {MODEL_PRESET} can model_type={expected_type}.")
if MAX_LENGTH > base_config.max_position_embeddings:
    raise ValueError("MAX_LENGTH vuot gioi han position embeddings.")

def prepare_wolf_v457_compatibility(config):
    """Map the V5 Wolf RoPE schema onto pinned Transformers 4.57.6 fields."""
    rope = getattr(config, "rope_parameters", None)
    try:
        full_theta = float(rope["full_attention"]["rope_theta"])
        local_theta = float(rope["sliding_attention"]["rope_theta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Wolf config thieu rope_parameters cua checkpoint da kiem tra.") from exc
    config.global_rope_theta = full_theta
    config.local_rope_theta = local_theta
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        interval = int(config.global_attn_every_n_layers)
        expected = [
            "full_attention" if index % interval == 0 else "sliding_attention"
            for index in range(int(config.num_hidden_layers))
        ]
        if list(layer_types) != expected:
            raise ValueError("Wolf layer_types khong tuong thich voi ModernBERT 4.57.6.")
    return config

if MODEL_PRESET == "wolf":
    base_config = prepare_wolf_v457_compatibility(base_config)

RESOLVED_MODEL_REVISION = getattr(base_config, "_commit_hash", None)
LOAD_KWARGS = {"trust_remote_code": False}
if MODEL_PRESET == "distilbert":
    LOAD_KWARGS["revision"] = RESOLVED_MODEL_REVISION or MODEL_REVISION
MODEL_PROVENANCE = {
    "preset": MODEL_PRESET, "source": MODEL_SOURCE,
    "revision": LOAD_KWARGS.get("revision"),
    "kaggle_handle": WOLF_KAGGLE_HANDLE if MODEL_PRESET == "wolf" else None,
    "approval_source": WOLF_APPROVAL_SOURCE if MODEL_PRESET == "wolf" else "docs/requirements.md",
}
if MODEL_PRESET == "wolf":
    # Wolf v1 was exported by Transformers 5 and declares TokenizersBackend,
    # a class unavailable in the pinned 4.57.6. Load tokenizer.json directly.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_SOURCE, **LOAD_KWARGS)
    if not tokenizer.is_fast:
        raise ValueError("Wolf tokenizer phai la fast tokenizer.")
    for name in ("pad", "bos", "eos", "cls", "sep"):
        tokenizer_id = getattr(tokenizer, f"{name}_token_id")
        config_id = getattr(base_config, f"{name}_token_id")
        if tokenizer_id != config_id:
            raise ValueError(f"Wolf tokenizer/config lech {name}_token_id: {tokenizer_id} != {config_id}")
    probe = tokenizer("prompt test", add_special_tokens=True)["input_ids"]
    if base_config.cls_token_id not in probe or base_config.sep_token_id not in probe:
        raise ValueError("Wolf tokenizer khong chen dung CLS/SEP.")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=True, **LOAD_KWARGS)

''' + tokenizer_tail)
text = source(14)
encode_start = text.index('def encode_texts')
encode_end = text.index('class PromptDataset')
put(14, text[:encode_start] + '''def encode_texts(texts):
    # Let the fast tokenizer's real postprocessor insert CLS/SEP. The generic
    # PreTrainedTokenizerFast.build_inputs_with_special_tokens does not do so.
    encoded = tokenizer(
        texts, add_special_tokens=True, truncation=False,
        return_special_tokens_mask=True, return_attention_mask=False,
        return_token_type_ids=False,
    )
    final_ids, lengths = [], []
    for token_ids, special_mask in zip(encoded["input_ids"], encoded["special_tokens_mask"]):
        positions = [i for i, special in enumerate(special_mask) if not special]
        lengths.append(len(positions))
        budget = MAX_LENGTH - sum(special_mask)
        if budget < 1:
            raise ValueError("MAX_LENGTH khong du cho special tokens va noi dung.")
        if len(positions) > budget:
            if LONG_TEXT_MODE == "head":
                chosen = positions[:budget]
            else:
                head_count = (budget + 1) // 2
                tail_count = budget - head_count
                chosen = positions[:head_count] + (positions[-tail_count:] if tail_count else [])
            kept = set(chosen)
            token_ids = [token_id for i, token_id in enumerate(token_ids) if special_mask[i] or i in kept]
        assert len(token_ids) <= MAX_LENGTH
        final_ids.append(token_ids)
    return {"input_ids": final_ids, "attention_mask": [[1] * len(ids) for ids in final_ids]}, lengths

''' + text[encode_end:])
put(15, '''## 6. Khởi tạo backbone đã chọn

Wolf dùng full fine-tune, head 3 lớp được khởi tạo lại, SDPA và AMP; tắt compile nội bộ
ModernBERT để chạy đơn giản trên T4. DistilBERT dùng QLoRA NF4 và train cả hai lớp head.
Factory luôn khởi tạo mới từ pretrained cho cả selection và full refit.
''')
put(16, (ROOT / "scripts/_v3_model_cell.py").read_text(encoding="utf-8"))
put(17, '''## 7. Macro F1 và loss

Weighted focal loss tính `p_t` từ cross-entropy **chưa nhân class weight**, giữ gradient
qua hệ số focal. `gamma=0` trở về weighted CE. Mẫu số là tổng số mẫu trong bước
gradient accumulation, kể cả microbatch cuối nhỏ hơn. Wolf mặc định CE như Melbourne;
DistilBERT mặc định focal gamma 2 theo kế hoạch. Không suy ra gain trước khi chạy.
''')
text = source(18)
put(18, text[:text.index('def weighted_loss')] + '''def make_weighted_focal_loss(weights, gamma):
    if gamma < 0:
        raise ValueError("Focal gamma phai >= 0.")
    # Bind weights rieng cho tung Trainer; full refit khong doi closure cua selection.
    fixed_weights = torch.as_tensor(weights, dtype=torch.float32).clone()

    def weighted_focal_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.float()
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha = fixed_weights.to(device=logits.device, dtype=logits.dtype)[labels]
        losses = alpha * (1.0 - pt).pow(gamma) * ce
        count = labels.numel() if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(count, device=logits.device, dtype=logits.dtype).clamp_min(1)
        return losses.sum() / denominator

    return weighted_focal_loss


def selected_checkpoint_epoch(state):
    # early stopping co the ket thuc muon hon checkpoint tot nhat.
    import math
    checkpoint = getattr(state, "best_model_checkpoint", None)
    if not checkpoint:
        raise ValueError("Thieu best_model_checkpoint de chon epoch refit.")
    try:
        step = int(str(checkpoint).replace("\\\\", "/").rstrip("/").rsplit("checkpoint-", 1)[1])
    except (ValueError, IndexError) as exc:
        raise ValueError("Khong doc duoc step cua best checkpoint.") from exc
    matches = [row for row in state.log_history
               if row.get("step") == step and "eval_macro_f1" in row and "epoch" in row]
    if not matches:
        raise ValueError("Khong co eval log khop best checkpoint.")
    epoch = float(matches[-1]["epoch"])
    if not math.isfinite(epoch) or epoch <= 0:
        raise ValueError("Epoch cua checkpoint khong hop le.")
    return int(math.ceil(epoch - 1e-9))


weighted_focal_loss = make_weighted_focal_loss(class_weights, FOCAL_GAMMA)
''')
replace(19, 'compute_loss_func=weighted_loss,', 'compute_loss_func=weighted_focal_loss,')
replace(19, 'push_to_hub=False,', 'push_to_hub=False,\n    torch_compile=False,')
put(20, '''## 8. Huấn luyện và chọn checkpoint

Wolf train tối đa 4 epoch; DistilBERT tối đa 8 epoch. Early stopping theo raw Macro F1,
sau đó lấy checkpoint có Macro F1 cao nhất. Training loss giảm không đảm bảo điểm test tăng.
Hết VRAM: giảm batch 8 → 4 và tăng accumulation 4 → 8, giữ effective batch 32.
Lưu OUTPUT_ROOT trên Drive để giữ checkpoint khi Colab ngắt phiên.
''')
replace(21, '# Luu adapter + ca hai classification head truoc buoc evaluation/inference.',
        '# Luu selection model (Wolf) hoac adapter + heads (DistilBERT) truoc refit.')
replace(21, 'print("Adapter saved:", ADAPTER_DIR)', 'print("Selection weights saved:", ADAPTER_DIR)')

# Preserve the V2 postprocess functions but widen two bias axes, guarded by cross-fit.
text = source(27)
text = text.replace('    bias_l2: float = 0.002,', '''    jailbreak_bias_values: Optional[Sequence[float]] = None,
    injection_bias_values: Optional[Sequence[float]] = None,
    bias_l2: float = 0.002,''')
text = text.replace('    if 1.0 not in neural_temperatures', '''    jailbreak_grid = tuple(bias_values if jailbreak_bias_values is None else jailbreak_bias_values)
    injection_grid = tuple(bias_values if injection_bias_values is None else injection_bias_values)
    if any(0.0 not in grid or not np.isfinite(grid).all() for grid in (jailbreak_grid, injection_grid)):
        raise ValueError("Moi bias grid can 0 va cac gia tri huu han.")
    if 1.0 not in neural_temperatures''')
text = text.replace('product(bias_values, repeat=2)', 'product(jailbreak_grid, injection_grid)')
text = text.replace('        n_splits=3,\n    )', '        n_splits=3,\n        fit_kwargs=POSTPROCESS_FIT_KWARGS,\n    )')
text = text.replace('    "bias_values": [-0.30, -0.15, 0.0, 0.15, 0.30], "bias_l2": 0.002,',
                    '    **POSTPROCESS_FIT_KWARGS,')
# Confusion-count implementation avoids thousands of sklearn calls in the expanded grid.
tree = ast.parse(text)
macro = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "macro_f1")
lines = text.splitlines(keepends=True)
lines[macro.lineno - 1:macro.end_lineno] = ['''def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int = 3) -> float:
    """Exact unweighted macro F1, including classes with zero support."""
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Expected aligned one-dimensional labels.")
    if not np.isin(truth, np.arange(class_count)).all() or not np.isin(predicted, np.arange(class_count)).all():
        raise ValueError("Class index outside allowed range.")
    matrix = np.bincount(class_count * truth + predicted, minlength=class_count ** 2).reshape(class_count, class_count)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    values = np.divide(2.0 * matrix.diagonal(), denominator,
                       out=np.zeros(class_count, dtype=np.float64), where=denominator > 0)
    return float(values.mean())
''']
put(27, "".join(lines))
put(26, '''## 10. Ensemble và global bias theo hướng Melbourne

Tìm bias JAILBREAK trong [-1, 1], PROMPT_INJECTION trong [-1, 2], giữ thêm các mức nhỏ V2.
Không chép bias `[-1, -0.75]` đã fit của Melbourne sang mô hình này.
Cross-fit 3 fold trên validation; chỉ bật nếu gain ít nhất `MIN_POSTPROCESS_GAIN`.
Không thêm bias theo ngôn ngữ vì nhóm English/PROMPT_INJECTION chỉ khoảng 10 mẫu validation.
Macro F1 fitted trên toàn bộ validation là số dùng chọn cấu hình, không phải điểm test.
''')
replace(29, '# Giu CSV QLoRA rieng de so sanh; CSV chinh theo postprocess guard.',
        '# Luu selection CSV truoc full refit de co ket qua neu runtime bi ngat.')
replace(29, '"submission_distilbert_raw.csv"', '"submission_selection_raw.csv"')
replace(29, 'shutil.copy2(SUBMISSION_PATH, ARTIFACT_DIR / "submission.csv")',
        'shutil.copy2(SUBMISSION_PATH, ARTIFACT_DIR / "submission_selection.csv")\nshutil.copy2(SUBMISSION_PATH, "/content/submission_selection.csv")')
replace(29, 'print("Final fitted validation report (selection set, NOT an independent score):")',
        'print("Selection fitted validation report (NOT an independent score):")')
put(28, '''### Xuất CSV và báo cáo của bước selection

`submission_selection.csv` giữ dự đoán đã hiệu chỉnh của model 90/10;
`submission_selection_raw.csv` giữ dự đoán raw. CSV được lưu trước khi train full-data.
`submission.csv` sẽ được cập nhật bằng model refit khi bước tiếp theo hoàn tất.
''')
put(30, '''## 12. Lưu cấu hình và đóng gói

ZIP chứa selection và final weights: full model cho Wolf, adapter + heads cho DistilBERT.
Vì Wolf lưu toàn bộ weights, ZIP lớn hơn bản QLoRA. Bao gồm tokenizer, lexical joblib của từng
stage, postprocess.json, logits/ID, split, cấu hình và CSV. Đọc active_weights/active_lexical
trong run_config.json để tái lập submission; không ghép nhầm các stage.
DistilBERT cần tải lại đúng pretrained revision khi nạp adapter. Checkpoint chứa optimizer
ở RUN_DIR không nằm trong ZIP. Validation chỉ thuộc selection; refit dùng đủ 14.000 mẫu.
''')
replace(32, '## 12. Tải kết quả', '## 13. Tải kết quả')
replace(31, '"notebook_version": "v2",', '''"notebook_version": "v3",
    "model_preset": MODEL_PRESET,
    "model_provenance": MODEL_PROVENANCE,
    "selection": SELECTION_REPORT,
    "full_refit": FINAL_FIT_REPORT,
    "active_weights": FINAL_FIT_REPORT["active_adapter"],
    "active_lexical": FINAL_FIT_REPORT["active_lexical"],
    "weights_format": "full_model" if MODEL_PRESET == "wolf" else "peft_adapter",
    "modernbert_reference_compile": False if MODEL_PRESET == "wolf" else None,''')
replace(31, '"training_rows": len(fit_df),', '"training_rows": FINAL_FIT_REPORT["training_rows"],')
replace(31, '"class_weights": weight_values.tolist(),',
        '"class_weights": FINAL_FIT_REPORT.get("class_weights", weight_values.tolist()),')
replace(31, '"loss": "power_frequency_weighted_cross_entropy_mean_over_samples" if USE_CLASS_WEIGHTS else "cross_entropy",',
        '"loss": "weighted_focal_mean_over_samples" if FOCAL_GAMMA else "weighted_cross_entropy_mean_over_samples",\n    "focal_gamma": FOCAL_GAMMA,')
replace(31, '"compute_dtype": str(COMPUTE_DTYPE), "skip_modules": HEAD_MODULES,\n    },',
        '"compute_dtype": str(COMPUTE_DTYPE), "skip_modules": HEAD_MODULES,\n    } if MODEL_PRESET == "distilbert" else None,')
replace(31, '"target_modules": LORA_TARGETS, "modules_to_save": HEAD_MODULES,\n    },',
        '"target_modules": LORA_TARGETS, "modules_to_save": HEAD_MODULES,\n    } if MODEL_PRESET == "distilbert" else None,')
replace(31, '"best_checkpoint": trainer.state.best_model_checkpoint,',
        '"best_checkpoint": SELECTION_REPORT["best_checkpoint"],')
replace(31, '"best_validation_macro_f1": trainer.state.best_metric,',
        '"best_validation_macro_f1": SELECTION_REPORT["best_validation_macro_f1"],')
replace(31, 'json.dumps(valid_result.metrics, indent=2)',
        'json.dumps({"stage": "selection_only", "metrics": valid_result.metrics}, indent=2)')

put(35, '''## Nguồn và cách đọc kết quả

- File tham khảo trong repo: `rmit-mel-segfault-c1.ipynb`; xem `docs/melbourne-reference-review.md`.
- [Cuộc thi và quy định](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview).
- [Wolf Defender trên Kaggle](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/transformers/default/1).
- [Trainer API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
- [Focal loss](https://arxiv.org/abs/1708.02002).
- [KaggleHub model download](https://github.com/Kaggle/kagglehub#download-models).

Trong ZIP, `selection_report.json` và `validation_*` thuộc bước chọn mô hình;
`final_fit_report.json` ghi model nào tạo submission, số mẫu/epoch và đường dẫn weights.
Wolf lưu full model, DistilBERT lưu PEFT adapter: đọc `weights_format` trong `run_config.json`.
Khi load lại Wolf, đặt `reference_compile=False`, SDPA, đúng tokenizer và mapping nhãn.
Ensemble inference cần cả model/adapter, lexical joblib tương ứng và `postprocess.json`.
Không cộng các mức gain dự kiến để suy ra điểm 0.99; phải kiểm chứng từng lần chạy.
''')

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": '''## 11. Refit cuối trên toàn bộ dữ liệu có nhãn

Giữ nguyên checkpoint selection và `submission_selection.csv`. Khởi tạo lại từ pretrained,
train đủ 14.000 mẫu trong số epoch tốt nhất đã chọn. Không early-stop/evaluate trên dữ liệu
đã học, không tune lại bias. `USE_FULL_TRAIN=False` bỏ qua bước này.
Các temperature/bias đã chọn có thể chuyển sang model mới không hoàn hảo; giữ cả hai CSV để đối chiếu.
'''},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": (ROOT / "scripts/_v3_refit_cell.py").read_text(encoding="utf-8")},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": '''# Xuat du doan cua active model; final_valid_* van la selection predictions.
submission = make_submission(final_test_pred, SUBMISSION_PATH)
shutil.copy2(SUBMISSION_PATH, ARTIFACT_DIR / "submission.csv")
make_submission(test_logits.argmax(axis=1), ARTIFACT_DIR / "submission_raw.csv")
np.save(ARTIFACT_DIR / "test_final_probabilities.npy", final_test_proba)
print("Submission:", SUBMISSION_PATH, "| rows:", len(submission))
print("Weights:", FINAL_FIT_REPORT["active_adapter"])
print("SHA256:", hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest())
display(submission["label"].value_counts().reindex(LABELS, fill_value=0))
'''},
]
cells[30:30] = new_cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
notebook["nbformat_minor"] = 5
for index, cell in enumerate(cells):
    cell["metadata"] = {}
    cell["id"] = f"v3-{index:03d}"
    if not isinstance(cell["source"], str):
        cell["source"] = "".join(cell["source"])
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        compile(cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"]),
                f"cell_{index}", "exec")
payload = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
for destination in (MAIN, UPLOADED):
    destination.write_text(payload, encoding="utf-8")
print("Built both V3 notebooks; preserved V2 SHA256:", hashlib.sha256(BACKUP.read_bytes()).hexdigest())
