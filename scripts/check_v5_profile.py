"""Check restored V3 training choices and execute V5's complete postprocess cell.

The scores below are synthetic contract fixtures, not a model/F1 experiment.
No GPU, model weights, network or training is used.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def one(sources, marker):
    found = [source for source in sources if marker in source]
    assert len(found) == 1, marker
    return found[0]


old = cells(ROOT / "notebooks/archive/final_97_executed.ipynb")
current = cells(ROOT / "final_97.ipynb")
old_env, current_env = ({"os": SimpleNamespace(environ={})} for _ in range(2))
exec(one(old, 'MODEL_PRESET = "wolf"'), old_env)
exec(one(current, 'MODEL_PRESET = "wolf"'), current_env)
same_settings = (
    "MODEL_PRESET", "WOLF_KAGGLE_HANDLE", "SEED", "SPLIT_SEED", "VAL_SIZE",
    "TRAIN_BATCH_SIZE", "EVAL_BATCH_SIZE", "GRAD_ACCUM_STEPS", "EPOCHS",
    "MAX_LENGTH", "LONG_TEXT_MODE", "LEARNING_RATE", "EARLY_STOPPING_PATIENCE",
    "WEIGHT_DECAY", "WARMUP_RATIO", "USE_CLASS_WEIGHTS", "CLASS_WEIGHT_POWER",
    "FOCAL_GAMMA", "GRADIENT_CHECKPOINTING", "LABELS", "LABEL2ID", "LANGUAGES",
    "ENABLE_LEXICAL_BRANCH", "LEXICAL_C", "LEXICAL_MAX_FEATURES", "LEXICAL_GLOBAL_WEIGHT",
    "MIN_POSTPROCESS_GAIN", "POSTPROCESS_FIT_KWARGS",
)
for key in same_settings:
    assert current_env[key] == old_env[key], (key, current_env[key], old_env[key])
assert current_env["WOLF_LOADING_MODE"] == "v3_legacy"
assert current_env["USE_FULL_TRAIN"] is False
assert current_env["ENABLE_PT_PAIR_TUNING"] is True
assert current_env["PT_PAIR_MIN_GAIN"] == 0.002
assert 0.0 in current_env["PT_PAIR_BIAS_VALUES"]
for marker in ("strata = train_df", "def build_model():", "def make_weighted_focal_loss(",
               "training_args = TrainingArguments("):
    # Refit's TrainingArguments line is indented, so this prefix selects the
    # initial Trainer cell explicitly instead of matching by a substring.
    if marker.startswith("training_args"):
        old_source = next(source for source in old if source.startswith(marker))
        new_source = next(source for source in current if source.startswith(marker))
    else:
        old_source, new_source = one(old, marker), one(current, marker)
    assert old_source.replace("\r\n", "\n") == new_source.replace("\r\n", "\n"), marker
print("V3 baseline restored: training/split/loss/global policy settings and Trainer source match.")

# Execute the complete code cell, including global-to-PT integration and global
# publication of its outputs. Unit tests separately audit fitting boundaries.
y = np.tile(np.repeat(np.arange(3), 36), 4)
languages = np.repeat(["pt", "fr", "it", "en"], 108)
valid = pd.DataFrame({"id": [f"v{i}" for i in range(len(y))],
                      "language": languages, "label_id": y})
logits = np.eye(3)[y] * 3.0
logits[(languages == "pt") & (y == 1)] = [0.10, 0.0, -3.0]
test_logits = np.tile([[0.10, 0.0, -3.0], [0.0, 0.1, 3.0], [3.0, 0.0, 0.0]], (4, 1))
test = pd.DataFrame({"id": [f"t{i}" for i in range(len(test_logits))],
                     "language": np.repeat(["pt", "fr", "it", "en"], 3)})
source = one(current, "def run_postprocessing(")
for enabled in (False, True):
    env = dict(current_env)
    env.update(
        valid_logits=logits.copy(), test_logits=test_logits.copy(), valid_true=y,
        valid_df=valid.copy(), test_df=test.copy(),
        valid_lexical_scores=None, test_lexical_scores=None,
        ENABLE_PT_PAIR_TUNING=enabled,
        POSTPROCESS_FIT_KWARGS={
            "neural_temperatures": (1.0,), "lexical_temperatures": (1.0,),
            "lexical_weights": (0.0,), "bias_values": (0.0,),
        },
    )
    exec(compile(source, "active_v5_postprocess_cell", "exec"), env)
    policy = env["POSTPROCESS_CONFIG"]
    reconstructed = env["apply_selected_postprocess"](
        test_logits, None, policy, test.language.to_numpy(),
    )
    np.testing.assert_array_equal(reconstructed.argmax(axis=1), env["final_test_pred"])
    np.testing.assert_array_equal(env["global_test_pred"], test_logits.argmax(axis=1))
    assert policy.get("portuguese_pair", {}).get("enabled", False) is enabled
    assert policy["labels"] == env["LABELS"]
    assert "search_grid" in policy and "score_interpretation" in policy
    json.dumps(policy)
    json.dumps(env["POSTPROCESS_SELECTION_REPORT"])
    print("Complete postprocess cell and frozen inference policy: OK; PT enabled =", enabled)
print("Profile/integration CPU checks passed. Synthetic scores are not model performance.")
