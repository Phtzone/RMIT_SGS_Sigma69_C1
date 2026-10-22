def build_model():
    """Reload the approved source and reset the seed for selection or final refit."""
    from copy import deepcopy

    set_seed(SEED)
    if MODEL_PRESET == "wolf":
        if base_config.model_type != "modernbert":
            raise ValueError("Preset wolf yeu cau checkpoint Wolf-Defender ModernBERT.")
        config = deepcopy(base_config)
        config.num_labels = len(LABELS)
        config.id2label = dict(ID2LABEL)
        config.label2id = dict(LABEL2ID)
        config.problem_type = "single_label_classification"
        # ModernBERT co torch.compile noi bo, doc lap voi Trainer.torch_compile.
        # Dat lai False moi lan load: ModernBertConfig.to_dict() bo truong nay.
        config.reference_compile = False
        loaded = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SOURCE,
            config=config,
            # Checkpoint Wolf co 2 nhan; khoi tao lai classifier cho 3 nhan C1.
            ignore_mismatched_sizes=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
            use_safetensors=True,
            **LOAD_KWARGS,
        )
        assert loaded.config.model_type == "modernbert"
        assert loaded.config.reference_compile is False
        assert loaded.classifier.out_features == len(LABELS)
        assert not getattr(loaded, "is_loaded_in_4bit", False)
        # Full fine-tuning: FP32 parameters; Trainer bat AMP FP16/BF16.
        assert all(parameter.requires_grad for parameter in loaded.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())
        print("Wolf-Defender: full fine-tuning, 3-label classifier, SDPA, reference_compile=False.")
    elif MODEL_PRESET == "distilbert":
        if base_config.model_type != "distilbert":
            raise ValueError("Preset distilbert yeu cau DistilBERT multilingual.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            # Transformers 4.57.6 su dung skip_modules cho ca 4-bit quantizer.
            llm_int8_skip_modules=list(HEAD_MODULES),
        )
        loaded = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SOURCE,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            problem_type="single_label_classification",
            quantization_config=quant_config,
            device_map={"": 0},
            dtype=COMPUTE_DTYPE,
            attn_implementation="sdpa",
            use_safetensors=True,
            **LOAD_KWARGS,
        )
        assert getattr(loaded, "is_loaded_in_4bit", False), "Backbone chua load 4-bit."
        assert any(isinstance(module, bnb.nn.Linear4bit) for module in loaded.modules())
        for name in HEAD_MODULES:
            assert not isinstance(getattr(loaded, name), bnb.nn.Linear4bit), f"{name} khong duoc quantize."
        loaded = prepare_model_for_kbit_training(
            loaded,
            use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        for suffix in LORA_TARGETS:
            hits = [name for name, module in loaded.named_modules() if name.endswith("." + suffix)]
            assert len(hits) == base_config.n_layers, (suffix, hits)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            modules_to_save=list(HEAD_MODULES),
            bias="none",
        )
        loaded = get_peft_model(loaded, lora_config)
        trainable_names = [name for name, parameter in loaded.named_parameters() if parameter.requires_grad]
        assert trainable_names
        assert any("lora_" in name for name in trainable_names)
        for head in HEAD_MODULES:
            assert any(f".{head}.modules_to_save." in name for name in trainable_names)
        assert all(
            "lora_" in name or any(f".{head}.modules_to_save." in name for head in HEAD_MODULES)
            for name in trainable_names
        ), "Co tham so backbone dang train ngoai adapter/head."
        loaded.print_trainable_parameters()
    else:
        raise ValueError(f"MODEL_PRESET khong hop le: {MODEL_PRESET}")

    assert loaded.config.num_labels == len(LABELS)
    assert loaded.config.id2label == ID2LABEL
    trainable_names = [name for name, parameter in loaded.named_parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in loaded.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in loaded.parameters())
    print(f"Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)")
    (ARTIFACT_DIR / "trainable_parameters.txt").write_text(
        "\n".join(trainable_names) + "\n", encoding="utf-8",
    )
    return loaded


model = build_model()
