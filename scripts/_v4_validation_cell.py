"""Notebook cell: held-out predictions and diagnostics; no test labels are read."""


def build_prediction_audit(frame, true_ids, scores, raw_lengths, content_budget,
                           label_names, stage="raw", scores_are_probabilities=False):
    """Keep scores tied to row IDs; normalized scores are NOT calibrated confidence.

    Stage-prefixed columns remain interpretable if a downstream report replaces
    ``predicted_label`` with a postprocessed prediction. All rows are returned;
    callers select errors only when writing an error-only artifact.
    """
    required = {"id", "language", "label", "text"}
    if not required.issubset(frame.columns) or not frame["id"].is_unique:
        raise ValueError("Validation audit can cot id/language/label/text va ID duy nhat.")
    if not stage.isidentifier():
        raise ValueError("Audit stage phai la ten hop le, vi du raw hoac final.")
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(true_ids)
    lengths = np.asarray(raw_lengths)
    class_count = len(label_names)
    if len(frame) == 0 or class_count < 2 or len(set(label_names)) != class_count:
        raise ValueError("Validation phai co mau va it nhat hai nhan rieng biet.")
    if values.shape != (len(frame), class_count) or not np.isfinite(values).all():
        raise ValueError("Validation scores sai shape hoac co NaN/Inf.")
    if truth.shape != (len(frame),) or not np.isin(truth, np.arange(class_count)).all():
        raise ValueError("Validation true_ids sai shape hoac ngoai mien nhan.")
    truth = truth.astype(np.int64)
    expected_labels = np.asarray(label_names, dtype=object)[truth]
    if not np.array_equal(expected_labels, frame["label"].to_numpy()):
        raise ValueError("Validation nhan/ID order khong khop true_ids.")
    if (lengths.shape != (len(frame),) or not np.isfinite(lengths).all()
            or (lengths < 0).any() or not np.equal(lengths, np.floor(lengths)).all()):
        raise ValueError("raw_lengths phai la so token nguyen khong am, dung thu tu ID.")
    if not isinstance(content_budget, (int, np.integer)) or content_budget < 1:
        raise ValueError("Content token budget phai la so nguyen duong.")
    if scores_are_probabilities:
        if (values < 0).any() or (values > 1).any():
            raise ValueError("Normalized scores phai nam trong [0, 1].")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
            raise ValueError("Moi hang normalized scores phai co tong bang 1.")
        normalized = values
    else:
        shifted = values - values.max(axis=1, keepdims=True)
        normalized = np.exp(shifted)
        normalized /= normalized.sum(axis=1, keepdims=True)
    predicted = values.argmax(axis=1)
    ordered_scores = np.sort(normalized, axis=1)
    rows = frame[["id", "language", "label", "text"]].reset_index(drop=True).copy()
    rows["predicted_label"] = np.asarray(label_names, dtype=object)[predicted]
    rows[f"{stage}_predicted_label"] = rows["predicted_label"]
    rows["raw_tokens"] = lengths.astype(np.int64)
    rows["truncated"] = rows["raw_tokens"] > content_budget
    rows["length_bucket"] = pd.cut(
        rows["raw_tokens"], bins=[-1, 128, 256, 512, 1024, np.inf],
        labels=["0-128", "129-256", "257-512", "513-1024", "1025+"],
    )
    rows[f"{stage}_is_error"] = predicted != truth
    rows[f"{stage}_top_score"] = normalized.max(axis=1)
    rows[f"{stage}_true_class_score"] = normalized[np.arange(len(frame)), truth]
    rows[f"{stage}_score_margin"] = ordered_scores[:, -1] - ordered_scores[:, -2]
    return rows


def summarize_prediction_audit(rows, stage="raw"):
    """Count errors with support, avoiding conclusions from F1 on tiny groups."""
    def aggregate(keys):
        summary = rows.groupby(keys, observed=True, dropna=False).agg(
            support=("id", "size"), errors=(f"{stage}_is_error", "sum"),
        ).reset_index()
        summary["support"] = summary["support"].astype(int)
        summary["errors"] = summary["errors"].astype(int)
        summary["correct"] = summary["support"] - summary["errors"]
        summary["error_rate"] = summary["errors"] / summary["support"]
        return summary

    return (aggregate(["language", "label", "truncated"]),
            aggregate(["length_bucket"]))


valid_result = trainer.predict(valid_dataset, metric_key_prefix="validation")
valid_logits = valid_result.predictions
if isinstance(valid_logits, tuple):
    valid_logits = valid_logits[0]
valid_logits = np.asarray(valid_logits, dtype=np.float32)
valid_true = valid_df["label_id"].to_numpy()
raw_errors = build_prediction_audit(
    valid_df, valid_true, valid_logits, valid_dataset.raw_lengths, budget, LABELS,
)
valid_pred = np.argmax(valid_logits, axis=-1)
report = classification_report(
    valid_true, valid_pred, labels=list(range(len(LABELS))),
    target_names=LABELS, output_dict=True, zero_division=0,
)
report_df = pd.DataFrame(report).T
display(report_df)
report_df.to_csv(ARTIFACT_DIR / "validation_class_report.csv")

language_rows = []
for language in LANGUAGES:
    mask = valid_df["language"].eq(language).to_numpy()
    if not mask.any():
        continue
    f1_values = f1_score(
        valid_true[mask], valid_pred[mask], labels=list(range(len(LABELS))),
        average=None, zero_division=0,
    )
    row = {"language": language, "rows": int(mask.sum()), "macro_f1": float(f1_values.mean())}
    row.update({f"f1_{label}": float(score) for label, score in zip(LABELS, f1_values)})
    row.update({f"support_{label}": int((valid_true[mask] == index).sum())
                for index, label in enumerate(LABELS)})
    language_rows.append(row)
language_report = pd.DataFrame(language_rows)
display(language_report)
language_report.to_csv(ARTIFACT_DIR / "validation_by_language.csv", index=False)

validation_predictions = raw_errors.drop(columns=["text"]).copy()
validation_predictions.to_csv(ARTIFACT_DIR / "validation_predictions.csv", index=False)
raw_errors.to_csv(ARTIFACT_DIR / "validation_prediction_audit_raw.csv", index=False)
raw_errors.loc[raw_errors["raw_is_error"]].to_csv(
    ARTIFACT_DIR / "validation_errors_raw.csv", index=False,
)
validation_group_errors, validation_length_errors = summarize_prediction_audit(raw_errors)
validation_group_errors.to_csv(ARTIFACT_DIR / "validation_errors_by_group_raw.csv", index=False)
validation_length_errors.to_csv(ARTIFACT_DIR / "validation_errors_by_length_raw.csv", index=False)
print("Selection-only errors:", int(raw_errors["raw_is_error"].sum()), "/", len(raw_errors))
print("Softmax scores va top-1/top-2 margin KHONG phai do tin cay da calibration.")
display(validation_group_errors.sort_values(["errors", "support"], ascending=False).head(12))
display(validation_length_errors)
error_preview = raw_errors.loc[raw_errors["raw_is_error"]].sort_values(
    "raw_score_margin", ascending=False,
).head(20).copy()
error_preview["text"] = error_preview["text"].str.slice(0, 180)
display(error_preview[["id", "language", "label", "predicted_label", "raw_tokens",
                       "truncated", "raw_score_margin", "text"]])

matrix = confusion_matrix(valid_true, valid_pred, labels=list(range(len(LABELS))))
pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
    ARTIFACT_DIR / "validation_confusion_matrix.csv",
)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(
    ax=ax, values_format="d", cmap="Blues", xticks_rotation=30, colorbar=False,
)
ax.set_title("Selection validation confusion matrix")
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "validation_confusion_matrix.png", dpi=150)
plt.show()
plt.close(fig)
np.save(ARTIFACT_DIR / "validation_logits.npy", valid_logits)
valid_df[["id", "language", "label"]].to_csv(ARTIFACT_DIR / "validation_ids.csv", index=False)
