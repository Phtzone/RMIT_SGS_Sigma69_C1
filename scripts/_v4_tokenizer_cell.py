"""Standalone notebook cell: load the approved source without editing its files."""


def normalize_wolf_config(raw_config):
    """Translate supported v5 ModernBERT settings to Transformers 4.57.6."""
    from copy import deepcopy
    import math

    config = deepcopy(raw_config)
    if config.get("model_type") != "modernbert":
        raise ValueError("Wolf source must contain a ModernBERT config.")
    changes = {}
    rope = config.get("rope_parameters")
    if rope is not None:
        if not isinstance(rope, dict) or set(rope) != {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported Wolf rope_parameters: expected full/sliding attention settings.")
        for kind, legacy_name in (("full_attention", "global_rope_theta"),
                                  ("sliding_attention", "local_rope_theta")):
            settings = rope[kind]
            if not isinstance(settings, dict) or settings.get("rope_type", "default") != "default":
                raise ValueError(f"Unsupported {kind} RoPE type; do not silently approximate this checkpoint.")
            if set(settings) - {"rope_type", "rope_theta"}:
                raise ValueError(f"Unsupported extra {kind} RoPE settings: {settings}.")
            theta = settings.get("rope_theta")
            if isinstance(theta, bool) or not isinstance(theta, (int, float)):
                raise ValueError(f"Missing/nonnumeric {kind} rope_theta.")
            if not math.isfinite(theta) or theta <= 0:
                raise ValueError(f"Invalid {kind} rope_theta.")
            if legacy_name in config and config[legacy_name] is not None:
                if float(config[legacy_name]) != float(theta):
                    raise ValueError(f"Conflicting {legacy_name} and rope_parameters in source config.")
            config[legacy_name] = float(theta)
            changes[legacy_name] = float(theta)
        # The older implementation consumes the translated scalar fields.
        config.pop("rope_parameters")

    for name in ("global_rope_theta", "local_rope_theta"):
        if name in config and config[name] is not None:
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {name} in source config.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name} in source config.")
    if config.get("rope_scaling") is not None:
        raise ValueError("Scaled RoPE needs a separate compatibility implementation.")

    layer_types = config.get("layer_types")
    if layer_types is not None:
        depth = config.get("num_hidden_layers")
        interval = config.get("global_attn_every_n_layers", 3)
        if type(depth) is not int or depth <= 0 or type(interval) is not int or interval <= 0:
            raise ValueError("Invalid ModernBERT layer count/attention interval.")
        expected = ["full_attention" if index % interval == 0 else "sliding_attention"
                    for index in range(depth)]
        if layer_types != expected:
            raise ValueError("Source layer_types cannot be reproduced by this pinned ModernBERT runtime.")
        config.pop("layer_types")
        changes["layer_schedule_verified"] = True

    if "norm_eps" not in config and "layer_norm_eps" in config:
        config["norm_eps"] = config["layer_norm_eps"]
        changes["norm_eps"] = config["norm_eps"]
    # Keep checkpoint pooling (Wolf uses mean); do not replace it with CLS.
    if config.get("classifier_pooling", "cls") not in {"mean", "cls"}:
        raise ValueError("Unsupported classifier pooling in Wolf source.")
    return config, {
        "source_transformers_version": raw_config.get("transformers_version"),
        "source_rope_parameters": deepcopy(rope),
        "changes": changes,
        "classifier_pooling": config.get("classifier_pooling", "cls"),
    }


def load_wolf_tokenizer(source_dir, compatibility_dir):
    """Load the real fast-tokenizer graph, preserving its IDs and postprocessor."""
    import hashlib
    import json
    from pathlib import Path
    from tokenizers import AddedToken, Tokenizer
    from transformers import PreTrainedTokenizerFast

    source_dir, compatibility_dir = Path(source_dir), Path(compatibility_dir)
    tokenizer_file = source_dir / "tokenizer.json"
    config_file = source_dir / "tokenizer_config.json"
    if not tokenizer_file.is_file() or not config_file.is_file():
        raise ValueError("Wolf folder must include tokenizer.json and tokenizer_config.json.")
    settings = json.loads(config_file.read_text(encoding="utf-8"))
    # Construct explicitly instead of dispatching the v5 TokenizersBackend name
    # through the v4 AutoTokenizer registry. No model weights are copied.
    backend = Tokenizer.from_file(str(tokenizer_file))
    original_vocab = backend.get_vocab(with_added_tokens=True)

    def token_value(value):
        if isinstance(value, str):
            content, normalized = value, value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            content = value["content"]
            fields = {key: item for key, item in value.items()
                      if key in {"content", "single_word", "lstrip", "rstrip", "normalized", "special"}}
            normalized = AddedToken(**fields)
        else:
            raise ValueError(f"Unsupported special-token entry: {value!r}")
        if content not in original_vocab:
            raise ValueError(f"Special token {content!r} is absent from tokenizer.json; refusing to add vocabulary.")
        return normalized

    arguments = {
        key: settings[key] for key in (
            "model_max_length", "padding_side", "truncation_side",
            "clean_up_tokenization_spaces", "spaces_between_special_tokens",
        ) if key in settings
    }
    for key in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
        if settings.get(key) is not None:
            arguments[key] = token_value(settings[key])
    extra = settings.get("extra_special_tokens", [])
    if isinstance(extra, dict):
        extra = list(extra.values())
    if not isinstance(extra, list):
        raise ValueError("extra_special_tokens must be a list or mapping.")
    additional = settings.get("additional_special_tokens", [])
    if not isinstance(additional, list):
        raise ValueError("additional_special_tokens must be a list.")
    seen, normalized_extra = set(), []
    for value in additional + extra:
        normalized = token_value(value)
        content = str(normalized)
        if content not in seen:
            seen.add(content)
            normalized_extra.append(normalized)
    if normalized_extra:
        arguments["additional_special_tokens"] = normalized_extra
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **arguments)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    if tokenizer.get_vocab() != original_vocab:
        raise ValueError("Tokenizer compatibility changed token IDs or vocabulary.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Wolf tokenizer needs a padding token for dynamic batching.")
    # Only tokenizer/config assets are written under this run's own directory.
    compatibility_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(compatibility_dir))
    report = {
        "source_tokenizer_class": settings.get("tokenizer_class"),
        "runtime_tokenizer_class": type(tokenizer).__name__,
        "normalized_extra_special_tokens": [str(value) for value in normalized_extra],
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "vocabulary_unchanged": True,
        "postprocessor_preserved": True,
        "compatibility_directory": str(compatibility_dir),
    }
    return tokenizer, report


MODEL_COMPATIBILITY = {"preset": MODEL_PRESET}
if MODEL_PRESET == "wolf":
    # Reset only Wolf's temporary runtime source; keep DistilBERT's pinned ID.
    MODEL_SOURCE = WOLF_LOCAL_PATH
    if not MODEL_SOURCE:
        import kagglehub
        MODEL_SOURCE = kagglehub.model_download(WOLF_KAGGLE_HANDLE)
    model_root = Path(MODEL_SOURCE)
    candidates = ([model_root / "config.json"] if (model_root / "config.json").is_file()
                  else list(model_root.rglob("config.json")))
    if len(candidates) != 1:
        raise ValueError("WOLF_LOCAL_PATH must identify exactly one model folder with config.json.")
    MODEL_SOURCE = str(candidates[0].parent)
    source_config = json.loads(candidates[0].read_text(encoding="utf-8"))
    normalized_config, config_compatibility = normalize_wolf_config(source_config)
    base_config = AutoConfig.for_model(
        normalized_config.pop("model_type"), **normalized_config,
    )
    tokenizer, tokenizer_compatibility = load_wolf_tokenizer(
        MODEL_SOURCE, RUN_DIR / "wolf_tokenizer_compat",
    )
    MODEL_COMPATIBILITY.update({
        "config": config_compatibility,
        "tokenizer": tokenizer_compatibility,
        "source_config_sha256": hashlib.sha256(candidates[0].read_bytes()).hexdigest(),
        "resolved_global_rope_theta": base_config.global_rope_theta,
        "resolved_local_rope_theta": base_config.local_rope_theta,
    })
else:
    base_config = AutoConfig.from_pretrained(
        MODEL_SOURCE, revision=MODEL_REVISION, trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE, revision=getattr(base_config, "_commit_hash", None) or MODEL_REVISION,
        use_fast=True, trust_remote_code=False,
    )

expected_type = "modernbert" if MODEL_PRESET == "wolf" else "distilbert"
if base_config.model_type != expected_type:
    raise ValueError(f"Preset {MODEL_PRESET} requires model_type={expected_type}.")
if MAX_LENGTH > base_config.max_position_embeddings:
    raise ValueError("MAX_LENGTH exceeds checkpoint positional capacity.")
if max(tokenizer.get_vocab().values()) >= base_config.vocab_size:
    raise ValueError("Tokenizer IDs exceed checkpoint vocabulary.")
if tokenizer.pad_token_id != base_config.pad_token_id:
    raise ValueError("Tokenizer/checkpoint padding IDs differ.")
RESOLVED_MODEL_REVISION = getattr(base_config, "_commit_hash", None)
LOAD_KWARGS = {"trust_remote_code": False}
if MODEL_PRESET == "distilbert":
    LOAD_KWARGS["revision"] = RESOLVED_MODEL_REVISION or MODEL_REVISION
MODEL_PROVENANCE = {
    "preset": MODEL_PRESET, "source": MODEL_SOURCE,
    "revision": LOAD_KWARGS.get("revision"),
    "kaggle_handle": WOLF_KAGGLE_HANDLE if MODEL_PRESET == "wolf" else None,
    "approval_source": WOLF_APPROVAL_SOURCE if MODEL_PRESET == "wolf" else "docs/requirements.md",
    "compatibility": MODEL_COMPATIBILITY,
}
(ARTIFACT_DIR / "model_compatibility.json").write_text(
    json.dumps(MODEL_COMPATIBILITY, indent=2, ensure_ascii=False), encoding="utf-8",
)
(ARTIFACT_DIR / "model_provenance.json").write_text(
    json.dumps(MODEL_PROVENANCE, indent=2, ensure_ascii=False), encoding="utf-8",
)


def encode_texts(texts):
    # Preserve the actual tokenizer postprocessor, including inserted tokens.
    encoded = tokenizer(
        texts, add_special_tokens=True, truncation=False,
        return_special_tokens_mask=True, return_attention_mask=False,
        return_token_type_ids=False,
    )
    final_ids, lengths = [], []
    for token_ids, special_mask in zip(encoded["input_ids"], encoded["special_tokens_mask"]):
        positions = [index for index, special in enumerate(special_mask) if not special]
        lengths.append(len(positions))
        budget = MAX_LENGTH - sum(special_mask)
        if budget < 1:
            raise ValueError("MAX_LENGTH cannot fit special tokens and content.")
        if len(positions) > budget:
            if LONG_TEXT_MODE == "head":
                chosen = positions[:budget]
            else:
                head_count = (budget + 1) // 2
                tail_count = budget - head_count
                chosen = positions[:head_count] + (positions[-tail_count:] if tail_count else [])
            kept = set(chosen)
            token_ids = [token_id for index, token_id in enumerate(token_ids)
                         if special_mask[index] or index in kept]
        assert len(token_ids) <= MAX_LENGTH
        final_ids.append(token_ids)
    return {"input_ids": final_ids, "attention_mask": [[1] * len(ids) for ids in final_ids]}, lengths


class PromptDataset(Dataset):
    def __init__(self, frame, labelled):
        self.features = {"input_ids": [], "attention_mask": []}
        self.labels = frame["label_id"].tolist() if labelled else None
        self.raw_lengths = []
        for start in range(0, len(frame), 256):
            encoded, lengths = encode_texts(frame["text"].iloc[start:start + 256].tolist())
            self.raw_lengths.extend(lengths)
            for key in self.features:
                self.features[key].extend(encoded[key])

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.features.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[index])
        return item


train_dataset = PromptDataset(fit_df, labelled=True)
valid_dataset = PromptDataset(valid_df, labelled=True)
test_dataset = PromptDataset(test_df, labelled=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
budget = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
length_reports = []
for split, frame, dataset in [("train", fit_df, train_dataset),
                              ("validation", valid_df, valid_dataset),
                              ("test", test_df, test_dataset)]:
    report = frame[["id", "language"]].copy()
    report["label"] = frame["label"].to_numpy() if "label" in frame else "UNKNOWN"
    report["split"] = split
    report["raw_tokens"] = dataset.raw_lengths
    report["truncated"] = report["raw_tokens"] > budget
    length_reports.append(report)
token_lengths = pd.concat(length_reports, ignore_index=True)
token_lengths.to_csv(ARTIFACT_DIR / "token_lengths.csv", index=False)
display(token_lengths.groupby(["split", "language", "label"])["truncated"].agg(["size", "sum", "mean"]))
print("Datasets:", len(train_dataset), len(valid_dataset), len(test_dataset))
print("Model revision:", RESOLVED_MODEL_REVISION or MODEL_REVISION)
print("Model compatibility:", json.dumps(MODEL_COMPATIBILITY, indent=2, ensure_ascii=False))
