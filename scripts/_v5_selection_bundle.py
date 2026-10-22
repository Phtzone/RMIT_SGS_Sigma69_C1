"""Append to selection CSV export: save a small bundle for future CPU tuning."""

# These predictions are captured BEFORE any full-data refit replaces test_logits.
np.save(ARTIFACT_DIR / "selection_test_logits.npy", np.asarray(test_logits, dtype=np.float32))
np.save(ARTIFACT_DIR / "selection_test_probabilities.npy", np.asarray(final_test_proba))
if valid_lexical_scores is not None:
    np.save(ARTIFACT_DIR / "validation_lexical_scores.npy", valid_lexical_scores)
if test_lexical_scores is not None:
    np.save(ARTIFACT_DIR / "selection_test_lexical_scores.npy", test_lexical_scores)

pt_report = POSTPROCESS_SELECTION_REPORT.get("portuguese_pair", {
    "enabled": False, "reason": "disabled_by_configuration",
})
(ARTIFACT_DIR / "pt_pair_tuning.json").write_text(
    json.dumps(pt_report, indent=2, ensure_ascii=False), encoding="utf-8",
)
(ARTIFACT_DIR / "global_postprocess.json").write_text(
    json.dumps(GLOBAL_POSTPROCESS_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8",
)
raw_macro_current = float(macro_f1(valid_true, valid_logits.argmax(axis=1)))
selection_comparison = {
    "stage": "selection",
    "v3_raw_validation_reference": 0.9729325186,
    "v4_raw_validation_user_reported": 0.969274,
    "current_raw_validation_macro_f1": raw_macro_current,
    "raw_delta_vs_v3": raw_macro_current - 0.9729325186,
    "global_crossfit_macro_f1": GLOBAL_POSTPROCESS_SELECTION_REPORT["cross_fitted_postprocess_macro_f1"],
    "global_gate_passed": bool(GLOBAL_POSTPROCESS_CONFIG["enabled"]),
    "portuguese_pair_enabled": bool(pt_report["enabled"]),
    "portuguese_pair_nested_gain": pt_report.get("nested_gain"),
    "final_fitted_validation_macro_f1": float(macro_f1(valid_true, final_valid_pred)),
    "note": "Fitted scores are diagnostics. Checkpoint selection used this validation; no Kaggle gain is guaranteed.",
}
(ARTIFACT_DIR / "selection_comparison.json").write_text(
    json.dumps(selection_comparison, indent=2, ensure_ascii=False), encoding="utf-8",
)
if "portuguese_pair_baseline_oof_pred" in globals():
    pt_oof = valid_df[["id", "language", "label"]].copy()
    pt_oof["baseline_prediction"] = [ID2LABEL[int(i)] for i in portuguese_pair_baseline_oof_pred]
    pt_oof["candidate_prediction"] = [ID2LABEL[int(i)] for i in portuguese_pair_candidate_oof_pred]
    pt_oof.to_csv(ARTIFACT_DIR / "validation_portuguese_pair_oof.csv", index=False)

# Use an explicit allowlist: never include checkpoints, optimizer state, or a
# final-refit array whose rows came from a different model than validation.
selection_bundle_names = [
    "validation_logits.npy", "validation_ids.csv", "test_ids.csv", "split.csv",
    "selection_test_logits.npy", "selection_test_probabilities.npy",
    "validation_lexical_scores.npy", "selection_test_lexical_scores.npy",
    "submission_selection.csv", "submission_selection_raw.csv", "submission_selection_global.csv",
    "postprocess.json", "global_postprocess.json", "postprocess_selection.json",
    "pt_pair_tuning.json", "selection_comparison.json", "validation_predictions.csv",
    "validation_errors_raw.csv", "validation_portuguese_pair_oof.csv",
    "model_provenance.json", "model_compatibility.json",
]
required_selection_files = {
    "validation_logits.npy", "validation_ids.csv", "test_ids.csv", "split.csv",
    "selection_test_logits.npy", "submission_selection.csv", "submission_selection_raw.csv",
    "submission_selection_global.csv", "postprocess.json", "global_postprocess.json",
    "selection_comparison.json",
}
missing_selection_files = sorted(name for name in required_selection_files if not (ARTIFACT_DIR / name).is_file())
if missing_selection_files:
    raise FileNotFoundError(f"Selection bundle missing: {missing_selection_files}")
available_selection_files = [name for name in selection_bundle_names if (ARTIFACT_DIR / name).is_file()]
selection_manifest = {
    "schema_version": 1, "stage": "selection_only", "contains_model_weights": False,
    "notebook_version": NOTEBOOK_VERSION, "run_id": RUN_ID,
    "model_preset": MODEL_PRESET, "model_provenance": MODEL_PROVENANCE,
    "wolf_loading_mode": WOLF_LOADING_MODE if MODEL_PRESET == "wolf" else None,
    "labels": list(LABELS), "label2id": LABEL2ID,
    "data_sha256": DATA_HASHES, "split_seed": SPLIT_SEED,
    "fit_rows": len(fit_df), "validation_rows": len(valid_df), "test_rows": len(test_df),
    "best_validation_macro_f1": float(trainer.state.best_metric),
    "selected_epoch": int(selected_checkpoint_epoch(trainer.state)),
    "lexical_enabled": bool(valid_lexical_scores is not None),
    "test_logits_file": "selection_test_logits.npy",
    "validation_logits_file": "validation_logits.npy",
    "file_sha256": {name: hashlib.sha256((ARTIFACT_DIR / name).read_bytes()).hexdigest()
                    for name in available_selection_files},
}
(ARTIFACT_DIR / "selection_manifest.json").write_text(
    json.dumps(selection_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
)
SELECTION_PREDICTIONS_PATH = Path("/content/selection_predictions.zip")
with zipfile.ZipFile(SELECTION_PREDICTIONS_PATH, "w", compression=zipfile.ZIP_DEFLATED) as packed:
    for name in available_selection_files + ["selection_manifest.json"]:
        packed.write(ARTIFACT_DIR / name, arcname=name)
with zipfile.ZipFile(SELECTION_PREDICTIONS_PATH) as packed:
    for name, digest in selection_manifest["file_sha256"].items():
        assert hashlib.sha256(packed.read(name)).hexdigest() == digest
    assert packed.read("submission_selection.csv") == (ARTIFACT_DIR / "submission_selection.csv").read_bytes()
print("Raw Macro F1:", round(raw_macro_current, 6), "| V3 reference: 0.972933")
print("Portuguese pair enabled:", pt_report["enabled"], "| nested gain:", pt_report.get("nested_gain"))
print("Download small predictions bundle:", SELECTION_PREDICTIONS_PATH,
      f"({SELECTION_PREDICTIONS_PATH.stat().st_size / 2**20:.2f} MiB)")
