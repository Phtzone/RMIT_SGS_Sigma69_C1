"""CPU checks of the notebook's global policy plus nested Portuguese pair gate.

Run: python scripts/check_v5_postprocess.py
No training, model download, external data, or GPU is used.
"""

import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "final_97.ipynb").read_text(encoding="utf-8"))
sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
global_source = next(source for source in sources if "def run_postprocessing(" in source)
helper_source = (ROOT / "scripts/_v5_postprocess_cell.py").read_text(encoding="utf-8")
assert global_source.rstrip().endswith(helper_source.rstrip()), "Regenerate notebook from tested PT helper."


def load_definitions(source, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "postprocess_functions", "exec"), namespace)


env = {"EXPECTED_POSTPROCESS_LABELS": ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]}
load_definitions(global_source, env)
apply_pair = env["apply_portuguese_pair_bias"]
apply_selected = env["apply_selected_postprocess"]
run_global = env["run_postprocessing"]
run_pair = env["run_portuguese_pair_tuning"]
labels = env["EXPECTED_POSTPROCESS_LABELS"]
identity_grid = {"neural_temperatures": (1.0,), "lexical_temperatures": (1.0,),
                 "lexical_weights": (0.0,), "bias_values": (0.0,)}

# Every protected row must retain its exact scores, including a negative
# JAILBREAK correction that would otherwise make PROMPT_INJECTION the winner.
rows = np.array([[0.10, 0.0, -3.0], [0.0, 0.10, -3.0], [0.0, 0.10, 3.0],
                 [0.10, 0.0, -3.0], [2.0, 0.0, -3.0], [0.0, 0.2, 0.1]])
row_languages = np.array(["pt", "pt", "pt", "fr", "pt", "pt"])
positive = apply_pair(rows, row_languages, 0.15)
assert positive[0].argmax() == 1
np.testing.assert_array_equal(positive[[2, 3, 4]], rows[[2, 3, 4]])
negative = apply_pair(rows, row_languages, -0.30)
assert negative[1].argmax() == 0
np.testing.assert_array_equal(negative[[2, 3, 4, 5]], rows[[2, 3, 4, 5]])
np.testing.assert_array_equal(apply_pair(rows, row_languages, 0), rows)
print("Pair application: margin limit, language protection, PI protection, identity OK.")

# A known stable Portuguese-only boundary error is correctable across all folds.
languages = np.repeat(["pt", "fr", "it", "en"], 108)
y = np.tile(np.repeat(np.arange(3), 36), 4)
frame = pd.DataFrame({"id": [f"v{i}" for i in range(len(y))],
                      "language": languages, "label_id": y})
strong = np.eye(3)[y] * 3.0
weak = strong.copy()
pt_jailbreak = (languages == "pt") & (y == 1)
weak[pt_jailbreak] = [0.10, 0.0, -3.0]
test = rows.copy()
test_langs = row_languages.copy()


def global_result(logits, test_values=test, lexical=None, test_lexical=None, grid=identity_grid):
    return run_global(logits, test_values, frame, labels, lexical, test_lexical,
                      fit_kwargs=grid)


def pair_result(logits, base, test_values=test, lexical=None, test_lexical=None, **kwargs):
    return run_pair(logits, test_values, frame, test_langs, base, lexical, test_lexical,
                    global_fit_kwargs=kwargs.pop("global_fit_kwargs", identity_grid),
                    verbose=False, **kwargs)


base = global_result(weak)
original_base = deepcopy(base)
assert not base["POSTPROCESS_CONFIG"]["enabled"]
result = pair_result(weak, base)
report = result["POSTPROCESS_SELECTION_REPORT"]["portuguese_pair"]
assert report["enabled"] and report["positive_folds"] == 3
assert report["nested_gain"] > 0.002
assert result["POSTPROCESS_CONFIG"]["portuguese_pair"]["bias"] == 0.15
assert not result["POSTPROCESS_CONFIG"]["enabled"], "Rejected global gate was re-enabled!"
assert result["POSTPROCESS_CONFIG"]["parameters"] == base["POSTPROCESS_CONFIG"]["parameters"]
np.testing.assert_array_equal(result["final_valid_pred"], y)
np.testing.assert_allclose(result["final_test_proba"].sum(axis=1), 1)
np.testing.assert_allclose(apply_selected(test, None, result["POSTPROCESS_CONFIG"], test_langs),
                           result["final_test_scores"])
assert base["POSTPROCESS_CONFIG"] == original_base["POSTPROCESS_CONFIG"]
assert base["POSTPROCESS_SELECTION_REPORT"] == original_base["POSTPROCESS_SELECTION_REPORT"]
np.testing.assert_array_equal(base["final_test_pred"], original_base["final_test_pred"])
json.dumps(result["POSTPROCESS_CONFIG"])
json.dumps(result["POSTPROCESS_SELECTION_REPORT"])
print("Nested gate: stable held-out gain enables one small bias; original global gate retained.")

# Verify recorded fold boundaries: every outer evaluation row is absent from
# every inner training set and from the rows used to fit the scalar.
all_rows = set(range(len(y)))
seen = set()
for fold in report["folds"]:
    fit_indices = np.asarray(fold["fit_indices"])
    fit_rows, held_rows = set(fit_indices.tolist()), set(fold["holdout_indices"])
    assert not fit_rows & held_rows and fit_rows | held_rows == all_rows
    assert not seen & held_rows
    seen |= held_rows
    inner_seen = set()
    for inner in fold["inner_splits_local_to_fit_indices"]:
        inner_train = set(fit_indices[inner["fit_indices"]].tolist())
        inner_valid = set(fit_indices[inner["holdout_indices"]].tolist())
        assert not inner_train & inner_valid
        assert inner_train | inner_valid == fit_rows
        assert not (inner_train | inner_valid) & held_rows
        assert not inner_seen & inner_valid
        inner_seen |= inner_valid
    assert inner_seen == fit_rows
assert seen == all_rows
print("Nested fold boundaries: all rows held out once, no outer-row leakage into inner fits.")

# Verify actual fitting calls, not just the report: a harmless equal-class
# offset fingerprints each row without changing any probabilities or labels.
fingerprinted = weak + np.arange(len(y))[:, None] * 1e-6
fingerprint_lookup = {tuple(row): i for i, row in enumerate(fingerprinted)}
fingerprint_base = global_result(fingerprinted)
original_fit = env["fit_postprocessor"]
observed_fit_rows = []


def recording_fit(logits, truth, lexical=None, **kwargs):
    indices = [fingerprint_lookup[tuple(row)] for row in logits]
    observed_fit_rows.append(indices)
    np.testing.assert_array_equal(truth, y[indices])
    return original_fit(logits, truth, lexical, **kwargs)


env["fit_postprocessor"] = recording_fit
try:
    observed = pair_result(fingerprinted, fingerprint_base)
finally:
    env["fit_postprocessor"] = original_fit
expected_fit_rows = []
for fold in observed["POSTPROCESS_SELECTION_REPORT"]["portuguese_pair"]["folds"]:
    outer_fit = np.asarray(fold["fit_indices"])
    for inner in fold["inner_splits_local_to_fit_indices"]:
        expected_fit_rows.append(outer_fit[inner["fit_indices"]].tolist())
assert observed_fit_rows == expected_fit_rows
print("Actual fit-call audit: every global fit uses only its documented inner training rows.")

# Test values are used only AFTER all fitting and gates; changing them cannot
# alter any configuration, selected bias, or validation report.
changed_test = -7.0 * test[:, ::-1]
changed_base = global_result(weak, test_values=changed_test)
changed = pair_result(weak, changed_base, test_values=changed_test)
assert changed["POSTPROCESS_CONFIG"] == result["POSTPROCESS_CONFIG"]
assert changed["POSTPROCESS_SELECTION_REPORT"] == result["POSTPROCESS_SELECTION_REPORT"]
print("No fitting on test: changed test scores leave all parameters and selection reports identical.")

# A perfect baseline must be exactly retained, not merely have similar F1.
perfect_base = global_result(strong)
perfect = pair_result(strong, perfect_base)
assert not perfect["POSTPROCESS_CONFIG"]["portuguese_pair"]["enabled"]
assert perfect["POSTPROCESS_CONFIG"]["portuguese_pair"]["bias"] == 0
for name in ("final_valid_pred", "final_test_pred", "final_valid_scores", "final_test_scores",
             "final_valid_proba", "final_test_proba"):
    np.testing.assert_array_equal(perfect[name], perfect_base[name])
print("No-gain gate: retains baseline scores and predictions exactly.")

# Even a consistent gain is rejected if it misses the configured minimum.
rejected = pair_result(weak, base, min_pair_gain=1.0)
assert not rejected["POSTPROCESS_CONFIG"]["portuguese_pair"]["enabled"]
np.testing.assert_array_equal(rejected["final_test_pred"], base["final_test_pred"])

# The nested candidate alone is insufficient: if the deployed global policy
# already fixes those errors, a final in-sample sanity check must veto the pair.
already_fixed = deepcopy(base)
already_fixed["POSTPROCESS_CONFIG"]["parameters"]["class_bias"] = [0.0, 0.20, 0.0]
already_fixed["POSTPROCESS_CONFIG"]["enabled"] = True
for split, values in (("valid", weak), ("test", test)):
    scores = env["postprocess_log_scores"](values, None, already_fixed["POSTPROCESS_CONFIG"]["parameters"])
    already_fixed[f"final_{split}_scores"] = scores
    already_fixed[f"final_{split}_proba"] = env["softmax_from_log_scores"](scores)
    already_fixed[f"final_{split}_pred"] = scores.argmax(axis=1)
sanity_veto = pair_result(weak, already_fixed)
sanity_report = sanity_veto["POSTPROCESS_SELECTION_REPORT"]["portuguese_pair"]
assert sanity_report["outer_gate_passed"]
assert sanity_report["all_validation_diagnostic_gain"] == 0
assert not sanity_report["enabled"]
np.testing.assert_array_equal(sanity_veto["final_test_pred"], already_fixed["final_test_pred"])

# Preserve an accepted global blend: the optional stage never replaces its
# parameters or turns it off after evaluating a language correction.
blend_grid = {"neural_temperatures": (1.0,), "lexical_temperatures": (1.0,),
              "lexical_weights": (0.0, 0.75), "bias_values": (0.0,)}
bad = np.roll(strong, 1, axis=1)
blend_base = global_result(bad, lexical=strong, test_lexical=test, grid=blend_grid)
assert blend_base["POSTPROCESS_CONFIG"]["enabled"]
blend = pair_result(bad, blend_base, lexical=strong, test_lexical=test,
                    global_fit_kwargs=blend_grid)
assert blend["POSTPROCESS_CONFIG"]["enabled"]
assert blend["POSTPROCESS_CONFIG"]["parameters"] == blend_base["POSTPROCESS_CONFIG"]["parameters"]
assert not blend["POSTPROCESS_CONFIG"]["portuguese_pair"]["enabled"]

# Insufficient support falls back without weakening the existing policy.
insufficient = pair_result(weak, base, min_pt_class_support=1000)
assert not insufficient["POSTPROCESS_CONFIG"]["portuguese_pair"]["enabled"]
assert "insufficient" in insufficient["POSTPROCESS_SELECTION_REPORT"]["portuguese_pair"]["reason"]
np.testing.assert_array_equal(insufficient["final_test_pred"], base["final_test_pred"])

for invalid in ({"bias_values": (-0.5, 0.0)}, {"bias_values": (0.15,)}, {"n_splits": 2},
                {"max_margin": 0.0}, {"min_pair_gain": 0.0}):
    try:
        pair_result(weak, base, **invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid configuration accepted: {invalid}")
try:
    apply_selected(test, None, result["POSTPROCESS_CONFIG"])
except ValueError:
    pass
else:
    raise AssertionError("Enabled pair policy silently accepted missing test languages")
print("Threshold/support failures, accepted global policy preservation, malformed inputs OK.")
print("All V5 post-processing CPU checks passed. No GPU/model training was executed.")
