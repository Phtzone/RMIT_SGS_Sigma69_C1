"""Notebook cell source: refit a fresh model on all labelled training rows.

Paste after the selection submission/artifacts cell. This source intentionally
depends on the notebook namespace and must not be run as a standalone script.
Validation predictions below always belong to the held-out selection stage.
"""

# Capture the selection state before replacing trainer/model. These Python and
# NumPy objects contain no live GPU model or optimizer references.
SELECTED_EPOCH = int(selected_checkpoint_epoch(trainer.state))
SELECTION_TRAINING_LOG = json.loads(json.dumps(trainer.state.log_history))
SELECTION_TRAINING_ARGUMENTS = json.loads(training_args.to_json_string())
SELECTION_TRAIN_METRICS = dict(train_result.metrics)
FROZEN_POSTPROCESS_CONFIG = json.loads(json.dumps(POSTPROCESS_CONFIG))
SELECTION_REPORT = {
    "training_rows": int(len(fit_df)),
    "validation_rows": int(len(valid_df)),
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_validation_macro_f1": float(trainer.state.best_metric),
    "selected_epoch": SELECTED_EPOCH,
    "adapter": str(ADAPTER_DIR.relative_to(ARTIFACT_DIR)),
    "lexical": "selection_lexical.joblib" if ENABLE_LEXICAL_BRANCH else None,
    "postprocess_selection": json.loads(json.dumps(POSTPROCESS_SELECTION_REPORT)),
}
(ARTIFACT_DIR / "selection_report.json").write_text(
    json.dumps(SELECTION_REPORT, indent=2, ensure_ascii=False), encoding="utf-8",
)
(ARTIFACT_DIR / "selection_training_log.json").write_text(
    json.dumps(SELECTION_TRAINING_LOG, indent=2), encoding="utf-8",
)
(ARTIFACT_DIR / "selection_training_arguments.json").write_text(
    json.dumps(SELECTION_TRAINING_ARGUMENTS, indent=2), encoding="utf-8",
)
(ARTIFACT_DIR / "selection_train_metrics.json").write_text(
    json.dumps(SELECTION_TRAIN_METRICS, indent=2), encoding="utf-8",
)

# Preserve the 90/10 model's predictions even when full-train predictions later
# replace the notebook's active test variables and submission.csv.
selection_test_logits = np.array(test_logits, dtype=np.float32, copy=True)
selection_test_probabilities = np.array(final_test_proba, copy=True)
np.save(ARTIFACT_DIR / "selection_test_logits.npy", selection_test_logits)
np.save(ARTIFACT_DIR / "selection_test_probabilities.npy", selection_test_probabilities)
if test_lexical_scores is not None:
    np.save(ARTIFACT_DIR / "selection_test_lexical_scores.npy", test_lexical_scores)

FINAL_FIT_REPORT = {
    "enabled": bool(USE_FULL_TRAIN),
    "completed": not USE_FULL_TRAIN,
    "training_rows": int(len(fit_df)),
    "epochs": SELECTED_EPOCH,
    "epoch_source": "holdout_best_checkpoint",
    "active_adapter": SELECTION_REPORT["adapter"],
    "active_lexical": SELECTION_REPORT["lexical"],
    "postprocess_frozen": True,
    "validation_evaluated_after_refit": False,
    "full_train_validation_macro_f1": None,
    "note": (
        "Validation and cross-fit scores refer only to the selection model. "
        "A refit on all labelled rows has no untouched validation score; "
        "its frozen temperatures/biases may transfer imperfectly."
    ),
}

if USE_FULL_TRAIN:
    refit_epochs = float(SELECTED_EPOCH if FULL_TRAIN_EPOCHS is None else FULL_TRAIN_EPOCHS)
    if not np.isfinite(refit_epochs) or refit_epochs <= 0:
        raise ValueError("FULL_TRAIN_EPOCHS phai la so duong hoac None.")
    assert len(train_df) == 14000
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    full_train_df = train_df.reset_index(drop=True).copy()
    full_train_dataset = PromptDataset(full_train_df, labelled=True)
    full_weight_values = compute_class_weights(
        full_train_df["label_id"].to_numpy(), len(LABELS),
        power=CLASS_WEIGHT_POWER, enabled=USE_CLASS_WEIGHTS,
    )
    full_class_weights = torch.tensor(full_weight_values, dtype=torch.float32)
    full_weighted_focal_loss = make_weighted_focal_loss(full_class_weights, FOCAL_GAMMA)
    full_train_df[["id", "language", "label"]].to_csv(
        ARTIFACT_DIR / "full_train_rows.csv", index=False,
    )
    FINAL_FIT_REPORT.update({
        "training_rows": int(len(full_train_df)),
        "epochs": refit_epochs,
        "epoch_source": "holdout_best_checkpoint" if FULL_TRAIN_EPOCHS is None else "fixed_config_override",
        "class_weights": np.asarray(full_weight_values).tolist(),
    })
    print(
        "Full refit:", len(full_train_df), "rows | epochs =", refit_epochs,
        "| fresh backbone/head (adapter neu bat) | khong evaluate/tune tren validation.",
    )

    # Release Accelerator's references as well as Trainer's model, optimizer and
    # callbacks before allocating another backbone on the same GPU.
    trainer.accelerator.free_memory()
    del trainer, model, train_result
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(SEED)
    model = build_model()
    training_args = TrainingArguments(
        output_dir=str(RUN_DIR / "full_train"),
        run_name=RUN_NAME + "_full_train",
        num_train_epochs=refit_epochs,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        fp16=not USE_BF16,
        bf16=USE_BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=25,
        label_names=["labels"],
        remove_unused_columns=True,
        eval_accumulation_steps=16,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        push_to_hub=False,
        torch_compile=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_loss_func=full_weighted_focal_loss,
    )
    trainer.model_accepts_loss_kwargs = False
    # Explicitly start from the factory's fresh model, never resume the 90/10 fit.
    train_result = trainer.train(resume_from_checkpoint=False)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    ADAPTER_DIR = ARTIFACT_DIR / ("final_model" if MODEL_PRESET == "wolf" else "final_adapter")
    trainer.model.save_pretrained(str(ADAPTER_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    trainer.model.config.to_json_file(str(ARTIFACT_DIR / "base_config.json"))

    test_result = trainer.predict(test_dataset, metric_key_prefix="test")
    test_logits = test_result.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_logits = np.asarray(test_logits, dtype=np.float32)
    if test_logits.shape != (len(test_df), len(LABELS)) or not np.isfinite(test_logits).all():
        raise ValueError("Full-train test logits sai shape hoac co NaN/Inf.")

    # Refit the vocabulary, IDF and classifiers on labelled train.csv only.
    # No validation or test scores/labels enter model or postprocessor selection.
    test_lexical_scores = None
    if ENABLE_LEXICAL_BRANCH:
        lexical_bundle = fit_lexical_branch(full_train_df)
        test_lexical_scores = lexical_predict(lexical_bundle, test_df)
        joblib.dump(lexical_bundle, ARTIFACT_DIR / "final_lexical.joblib", compress=3)
        np.save(ARTIFACT_DIR / "test_lexical_scores.npy", test_lexical_scores)
        FINAL_FIT_REPORT["active_lexical"] = "final_lexical.joblib"

    # Apply exactly the holdout-selected configuration; no postprocess fit call.
    if FROZEN_POSTPROCESS_CONFIG["enabled"]:
        final_test_scores = postprocess_log_scores(
            test_logits, test_lexical_scores, FROZEN_POSTPROCESS_CONFIG["parameters"],
        )
    else:
        final_test_scores = log_softmax_np(test_logits, 1.0)
    final_test_proba = softmax_from_log_scores(final_test_scores)
    final_test_pred = final_test_proba.argmax(axis=1)
    assert POSTPROCESS_CONFIG == FROZEN_POSTPROCESS_CONFIG
    np.testing.assert_allclose(final_test_proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(final_test_proba).all()
    np.save(ARTIFACT_DIR / "test_logits.npy", test_logits)
    np.save(ARTIFACT_DIR / "test_final_probabilities.npy", final_test_proba)
    FINAL_FIT_REPORT.update({
        "completed": True,
        "active_adapter": str(ADAPTER_DIR.relative_to(ARTIFACT_DIR)),
        "train_metrics": dict(train_result.metrics),
        "optimizer_steps": int(trainer.state.global_step),
    })
else:
    print("USE_FULL_TRAIN=False: giu model va submission tu holdout selection.")

(ARTIFACT_DIR / "final_fit_report.json").write_text(
    json.dumps(FINAL_FIT_REPORT, indent=2, ensure_ascii=False), encoding="utf-8",
)
print("Active model/adapter:", FINAL_FIT_REPORT["active_adapter"])
print("Active lexical model:", FINAL_FIT_REPORT["active_lexical"])
print("Validation reports van thuoc selection stage; chay cell xuat CSV tiep theo.")
