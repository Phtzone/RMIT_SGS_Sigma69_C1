"""CPU checks for the Colab notebook; never invokes its GPU training cells.

Requires numpy, pandas, scikit-learn, nbformat (already used by the notebook).
Run: python scripts/check_notebook.py
"""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_97.ipynb"
notebook = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(notebook)
code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
for index, source in enumerate(code):
    compile(source, f"cell_{index}", "exec")
assert all(cell.execution_count is None and not cell.outputs
           for cell in notebook.cells if cell.cell_type == "code")
print(f"Notebook schema/syntax: {len(code)} code cells OK; outputs cleared.")

backup = ROOT / "notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_baseline_087.ipynb"
old = json.loads(backup.read_text(encoding="utf-8"))
assert any(cell.get("outputs") for cell in old["cells"])
assert "0.8863657407407408" in backup.read_text(encoding="utf-8")
print("Executed baseline preserved; SHA256:", hashlib.sha256(backup.read_bytes()).hexdigest())
wolf_backup = ROOT / "notebooks/archive/final_97_executed.ipynb"
assert hashlib.sha256(wolf_backup.read_bytes()).hexdigest() == (
    "cea7a8332988234fcf2a232c3b25d9256964519cd2c7a2f90d11131a6d02e9a5"
)
print("Executed Wolf V3 preserved byte-for-byte.")

def source_with(needle):
    return next(source for source in code if needle in source)

def definitions(source, namespace, imports=False, names=None):
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if names is None or node.name in names:
                selected.append(node)
        elif imports and isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "notebook_functions", "exec"), namespace)

labels = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
env = {"np": np, "pd": pd, "LABELS": labels, "ID2LABEL": dict(enumerate(labels)),
       "LANGUAGES": ["pt", "fr", "it", "en"], "SEED": 42,
       "LEXICAL_C": 2.0, "LEXICAL_MAX_FEATURES": 300000, "LEXICAL_GLOBAL_WEIGHT": 0.5}

# Verify that head-tail actually retains both ends within the special-token budget.
class TokenizerStub:
    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"]
        return {"input_ids": [[-1] + list(range(int(text))) + [-2] for text in texts],
                "special_tokens_mask": [[1] + [0] * int(text) + [1] for text in texts]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        raise AssertionError("Generic fast tokenizer must use its postprocessor instead")

env.update(tokenizer=TokenizerStub(), MAX_LENGTH=512, LONG_TEXT_MODE="head_tail")
definitions(source_with("def encode_texts"), env, names={"encode_texts"})
encoded, lengths = env["encode_texts"](["10", "700"])
assert lengths == [10, 700]
assert encoded["input_ids"][0] == [-1] + list(range(10)) + [-2]
assert encoded["input_ids"][1] == [-1] + list(range(255)) + list(range(445, 700)) + [-2]
env["LONG_TEXT_MODE"] = "head"
encoded, _ = env["encode_texts"](["700"])
assert encoded["input_ids"][0] == [-1] + list(range(510)) + [-2]
print("Token budget/head-tail: OK.")

# Run the exact CPU lexical branch against the provided split (no GPU training).
train = pd.read_csv(ROOT / "rmit-hackathon-2026/train.csv", dtype=str, keep_default_na=False)
train["label_id"] = train.label.map({label: i for i, label in enumerate(labels)})
fit, valid = train_test_split(train, test_size=0.1, random_state=42,
                             stratify=train.language + "::" + train.label)
fit, valid = fit.reset_index(drop=True), valid.reset_index(drop=True)
definitions(source_with("def fit_lexical_branch"), env, imports=True)
bundle = env["fit_lexical_branch"](fit)
scores = env["lexical_predict"](bundle, valid)
assert scores.shape == (1400, 3) and np.isfinite(scores).all()
assert np.array_equal(bundle["global_model"].classes_, np.arange(3))
vocab_before = dict(bundle["vectorizer"].vocabulary_)
env["lexical_predict"](bundle, valid.assign(text="zzzzqqqqvvvv unseen validation only"))
assert vocab_before == bundle["vectorizer"].vocabulary_
with TemporaryDirectory() as directory:
    saved = Path(directory) / "lexical.joblib"
    env["joblib"].dump(bundle, saved)
    restored = env["joblib"].load(saved)
    np.testing.assert_allclose(env["lexical_predict"](restored, valid), scores)
print("Lexical branch/serialization: OK. Exploratory validation Macro F1:",
      round(f1_score(valid.label_id, scores.argmax(1), average="macro"), 6))

# Synthetic score fixtures verify the postprocess gate, normalization and data boundary.
definitions(source_with("def run_postprocessing"), env, imports=True)
env["EXPECTED_POSTPROCESS_LABELS"] = labels
config_tree = ast.parse(source_with("POSTPROCESS_FIT_KWARGS ="))
grid = next(ast.literal_eval(node.value) for node in config_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "POSTPROCESS_FIT_KWARGS" for t in node.targets))
rng = np.random.default_rng(42)
for count in (1, 7, 100):
    for _ in range(20):
        expected, predicted = rng.integers(0, 3, size=(2, count))
        np.testing.assert_allclose(env["macro_f1"](expected, predicted),
                                   f1_score(expected, predicted, labels=np.arange(3),
                                            average="macro", zero_division=0))
y = np.tile(np.arange(3), 80)
languages = np.repeat(["pt", "fr", "it", "en"], 60)
frame = pd.DataFrame({"id": [f"v{i}" for i in range(len(y))], "language": languages, "label_id": y})
strong = np.eye(3)[y] * 4 + rng.normal(0, 0.1, (len(y), 3))
weak = strong.copy()
wrong = np.arange(0, len(y), 4)
weak[wrong] = np.roll(weak[wrong], 1, axis=1)
test = strong[:24].copy()
run = env["run_postprocessing"]
result = run(weak, test, frame, labels, strong, test, fit_kwargs=grid)
assert result["POSTPROCESS_CONFIG"]["enabled"]
assert result["POSTPROCESS_SELECTION_REPORT"]["cross_fitted_gain"] > 0.1
np.testing.assert_allclose(result["final_test_proba"].sum(axis=1), 1.0)
assert np.isfinite(result["final_test_proba"]).all()
json.dumps(result["POSTPROCESS_CONFIG"])
json.dumps(result["POSTPROCESS_SELECTION_REPORT"])
changed_test = run(weak, -test, frame, labels, strong, -test, fit_kwargs=grid)
assert result["POSTPROCESS_CONFIG"] == changed_test["POSTPROCESS_CONFIG"], "Test scores affected fitting!"
identity = run(strong, test, frame, labels, fit_kwargs=grid)
assert not identity["POSTPROCESS_CONFIG"]["enabled"]
np.testing.assert_array_equal(identity["final_test_pred"], test.argmax(1))
try:
    run(weak, test, frame, labels, strong, None)
except ValueError:
    pass
else:
    raise AssertionError("Mismatched validation/test lexical branch was not rejected")
print("Postprocessing: gate pass/fail, disabled lexical, probability sums, no test-dependent fitting OK.")

# CSV ordering, class names, uniqueness and on-disk validation.
definitions(source_with("def validate_submission"), env)
ids = [f"t{i:06d}" for i in range(6000)]
env["test_df"] = pd.DataFrame({"id": ids[::-1]})
env["sample_df"] = pd.DataFrame({"id": ids})
with TemporaryDirectory() as directory:
    result = env["make_submission"](np.arange(6000) % 3, Path(directory) / "submission.csv")
    assert result.id.tolist() == ids
    assert result.iloc[-1].label == "BENIGN"  # first test row has reversed ID
    for bad in (result.assign(label="INVALID"), result.iloc[:-1], result.iloc[::-1]):
        try:
            env["validate_submission"](bad, env["test_df"], env["sample_df"])
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid submission accepted")
print("Submission checks: order, valid labels, row count, read-back OK.")
print("All CPU checks passed. Neither Wolf nor DistilBERT GPU training was executed.")
