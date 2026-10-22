"""Exercise the actual final_97.ipynb validation cell on CPU.

Requires numpy, pandas and scikit-learn. GPU prediction and plotting are stubbed;
the notebook's diagnostics, reports and CSV/NumPy artifact writes execute.
Run: python scripts/check_validation.py
"""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_97.ipynb"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]


def validation_cell_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
               for cell in notebook["cells"] if cell["cell_type"] == "code"]
    matches = [source for source in sources if "def build_prediction_audit(" in source]
    assert len(matches) == 1, "Expected exactly one diagnostics cell in final_97.ipynb."
    compile(matches[0], "notebook_validation_cell", "exec")
    return matches[0]


def fixture():
    truth = np.array([0, 1, 2, 0, 1, 2, 0])
    predicted = np.array([0, 0, 2, 1, 1, 0, 0])
    lengths = [128, 129, 256, 512, 1025, 510, 511]
    frame = pd.DataFrame({
        "id": [f"v{i}" for i in range(7)],
        "language": ["pt"] * 4 + ["en"] * 3,
        "label": [LABELS[index] for index in truth],
        "text": ["full text " * 100] * 7,
        "label_id": truth,
    }, index=[11, 22, 33, 44, 55, 66, 77])
    return frame, truth, predicted, lengths


def check_helpers(source):
    functions = ast.Module(
        body=[node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"np": np, "pd": pd}
    exec(compile(functions, "notebook_validation_helpers", "exec"), namespace)
    build = namespace["build_prediction_audit"]
    summarize = namespace["summarize_prediction_audit"]
    frame, truth, predicted, lengths = fixture()
    logits = np.eye(3)[predicted] * 1000
    rows = build(frame, truth, logits, lengths, 510, LABELS)
    assert rows.id.tolist() == frame.id.tolist(), "Diagnostics changed row/ID alignment."
    assert rows["raw_is_error"].sum() == 3
    assert rows.truncated.tolist() == [False, False, False, True, True, False, True]
    assert rows.length_bucket.astype(str).tolist() == [
        "0-128", "129-256", "129-256", "257-512", "1025+", "257-512", "257-512",
    ]
    assert rows.raw_score_margin.eq(1).all()
    assert np.isfinite(rows.select_dtypes(include="number")).all().all()
    assert rows.raw_true_class_score.tolist() == [1, 0, 1, 0, 1, 0, 1]
    assert rows.text.tolist() == frame.text.tolist(), "Saved prompts were shortened."
    groups, buckets = summarize(rows)
    assert groups.support.sum() == buckets.support.sum() == 7
    assert groups.errors.sum() == buckets.errors.sum() == 3
    assert (groups.correct + groups.errors == groups.support).all()

    normalized = np.eye(3)[predicted] * 0.8 + 0.2 / 3
    final = build(frame, truth, normalized, lengths, 510, LABELS,
                  stage="final", scores_are_probabilities=True)
    np.testing.assert_allclose(final.final_score_margin, 0.8)
    assert final.final_is_error.sum() == 3
    same = build(frame, truth, np.log(normalized), lengths, 510, LABELS, stage="final")
    np.testing.assert_allclose(final.final_score_margin, same.final_score_margin)
    assert "raw_is_error" not in final and "raw_is_error" in rows
    tied = build(frame, truth, np.zeros((7, 3)), lengths, 510, LABELS)
    assert tied.raw_score_margin.eq(0).all()
    np.testing.assert_allclose(tied.raw_top_score, 1 / 3)

    invalid_cases = [
        {"scores": logits[:-1]},
        {"scores": np.full((7, 3), np.nan)},
        {"true_ids": truth[::-1]},
        {"raw_lengths": [-1] * 7},
        {"raw_lengths": [0.5] * 7},
        {"content_budget": 0},
        {"scores": normalized * 0.5, "scores_are_probabilities": True},
        {"frame": frame.assign(id="duplicate")},
    ]
    for changes in invalid_cases:
        arguments = dict(frame=frame, true_ids=truth, scores=logits, raw_lengths=lengths,
                         content_budget=510, label_names=LABELS)
        arguments.update(changes)
        try:
            build(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid diagnostics input accepted: {list(changes)}")
    print("Validation helpers: IDs, scores/margins, token boundaries, groups and invalid input OK.")


class FigureStub:
    def tight_layout(self):
        pass

    def savefig(self, destination, dpi):
        assert destination.name == "validation_confusion_matrix.png" and dpi == 150


class AxisStub:
    def set_title(self, title):
        assert "validation" in title.lower()


class ConfusionDisplayStub:
    def __init__(self, matrix, display_labels):
        assert matrix.sum() == 7 and display_labels == LABELS

    def plot(self, **kwargs):
        pass


def check_full_cell(source, perfect=False):
    frame, truth, predicted, lengths = fixture()
    if perfect:
        predicted = truth.copy()
    logits = np.eye(3)[predicted] * 1000
    expected_errors = int((predicted != truth).sum())
    valid_dataset = SimpleNamespace(raw_lengths=lengths)

    class TrainerStub:
        calls = 0

        def predict(self, dataset, metric_key_prefix):
            assert dataset is valid_dataset and metric_key_prefix == "validation"
            self.calls += 1
            return SimpleNamespace(predictions=(logits,), metrics={"validation_macro_f1": 0.5})

    trainer = TrainerStub()
    plot_stub = SimpleNamespace(
        subplots=lambda **kwargs: (FigureStub(), AxisStub()),
        show=lambda: None,
        close=lambda figure: None,
    )
    with TemporaryDirectory() as directory:
        artifacts = Path(directory)
        namespace = {
            "np": np, "pd": pd, "trainer": trainer, "valid_dataset": valid_dataset,
            "valid_df": frame, "budget": 510, "LABELS": LABELS, "LANGUAGES": ["pt", "en"],
            "ID2LABEL": dict(enumerate(LABELS)), "ARTIFACT_DIR": artifacts,
            "classification_report": classification_report, "confusion_matrix": confusion_matrix,
            "f1_score": f1_score, "display": lambda table: None, "plt": plot_stub,
            "ConfusionMatrixDisplay": ConfusionDisplayStub,
        }
        exec(compile(source, "notebook_validation_cell", "exec"), namespace)
        assert trainer.calls == 1, "Validation diagnostics must not invoke extra model fits/predictions."
        assert len(namespace["raw_errors"]) == len(namespace["validation_predictions"]) == 7
        assert len(pd.read_csv(artifacts / "validation_errors_raw.csv")) == expected_errors
        exported = pd.read_csv(artifacts / "validation_prediction_audit_raw.csv")
        assert exported.text.tolist() == frame.text.tolist()
        assert exported.id.tolist() == frame.id.tolist()
        assert namespace["language_report"]["support_PROMPT_INJECTION"].sum() == 2
        assert np.load(artifacts / "validation_logits.npy").shape == (7, 3)
        assert namespace["matrix"].sum() == 7
        for filename in ("validation_class_report.csv", "validation_by_language.csv",
                         "validation_predictions.csv", "validation_ids.csv",
                         "validation_confusion_matrix.csv", "validation_errors_by_group_raw.csv",
                         "validation_errors_by_length_raw.csv"):
            assert (artifacts / filename).is_file(), f"Missing validation artifact: {filename}"
    print(f"Entire validation cell/artifacts: OK ({expected_errors} errors; GPU/plot surfaces stubbed).")


def main():
    source = validation_cell_source()
    check_helpers(source)
    check_full_cell(source)
    check_full_cell(source, perfect=True)
    print("Validation CPU checks passed. No GPU training was executed.")


if __name__ == "__main__":
    main()
