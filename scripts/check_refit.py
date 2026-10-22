"""CPU contract checks for the notebook's two-stage final refit.

Uses small fake model/Trainer objects and the real 14k/6k row boundaries. Never
loads model weights, imports PyTorch, or executes GPU training. Run from any cwd:
    python scripts/check_refit.py
"""

import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_97.ipynb"
SOURCE_MARKER = "# Capture the selection state before replacing trainer/model."
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
matching_cells = [
    "".join(cell.get("source", []))
    for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
    and SOURCE_MARKER in "".join(cell.get("source", []))
]
assert len(matching_cells) == 1, "Expected exactly one full-refit cell in the active notebook."
source = matching_cells[0]
source_name = str(NOTEBOOK)
compiled_cell = compile(source, source_name, "exec")


def check_case(model_preset, use_full_train, lexical_enabled, postprocess_enabled, override=None,
               pair_enabled=False):
    events = []
    selection_weights_name = "selection_model" if model_preset == "wolf" else "selection_adapter"
    final_weights_name = "final_model" if model_preset == "wolf" else "final_adapter"
    labels = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
    targets = np.tile([0, 0, 0, 0, 0, 0, 1, 1, 1, 2], 1400)
    train = pd.DataFrame({
        "id": [f"r{i:05d}" for i in range(14000)],
        "language": np.tile(["en", "fr", "pt", "it"], 3500),
        "label_id": targets,
        "label": np.asarray(labels)[targets],
        "text": "training example",
    })
    fit = train.iloc[:12600].copy()
    valid = train.iloc[12600:].copy()
    test = pd.DataFrame({
        "id": [f"t{i:05d}" for i in range(6000)],
        "language": np.tile(["en", "fr", "pt", "it"], 1500),
        "text": "test example",
    })
    old_logits = np.tile([2.0, 0.0, -1.0], (6000, 1))
    new_logits = np.tile([0.0, 3.0, -1.0], (6000, 1))
    lexical_scores = np.tile([-1.0, 0.0, 2.0], (6000, 1))
    original_valid_logits = np.ones((1400, 3))
    original_valid_predictions = np.zeros(1400, dtype=np.int64)
    selected_config = {
        "enabled": bool(postprocess_enabled),
        "parameters": {
            "neural_temperature": 2.0 if postprocess_enabled else 1.0,
            "lexical_temperature": 0.5 if lexical_enabled else None,
            "lexical_weight": 0.5 if lexical_enabled and postprocess_enabled else 0.0,
            "class_bias": [0.0, -0.3, 0.3] if postprocess_enabled else [0.0, 0.0, 0.0],
        },
        "portuguese_pair": {
            "enabled": bool(pair_enabled),
            "bias": 0.15 if pair_enabled else 0.0,
            "max_margin": 0.75,
            "minimum_outer_gain": 0.002,
        },
    }

    class FakeModel:
        def __init__(self, stage):
            self.stage = stage
            self.config = SimpleNamespace(to_json_file=lambda path: Path(path).write_text("{}"))

        def save_pretrained(self, path, safe_serialization=True):
            assert self.stage == "final" and safe_serialization
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "mock-model.json").write_text('{"fresh": true}')

    class FakeDataset:
        def __init__(self, frame, labelled):
            self.frame = frame
            self.labelled = labelled
            if labelled:
                assert "label_id" in frame
            else:
                assert "label_id" not in frame and "label" not in frame

    class FakeAccelerator:
        def __init__(self, model):
            self.tracked_model = model

        def free_memory(self):
            events.append("free_memory")
            self.tracked_model = None

    class FakeArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            assert kwargs["eval_strategy"] == "no"
            assert kwargs["save_strategy"] == "no"
            assert kwargs["load_best_model_at_end"] is False
            assert kwargs["torch_compile"] is False
            assert "metric_for_best_model" not in kwargs

        def to_json_string(self):
            return json.dumps(self.__dict__)

    class FakeTrainer:
        def __init__(self, **kwargs):
            assert "eval_dataset" not in kwargs
            assert "compute_metrics" not in kwargs
            assert "callbacks" not in kwargs
            self.model = kwargs["model"]
            self.model_wrapped = self.model
            self.args = kwargs["args"]
            self.dataset = kwargs["train_dataset"]
            self.loss = kwargs["compute_loss_func"]
            assert len(self.dataset.frame) == 14000 and self.dataset.labelled
            assert set(self.dataset.frame.id) == set(train.id)
            assert self.model.stage == "final"
            self.state = SimpleNamespace(
                global_step=1314, best_model_checkpoint=None, best_metric=None,
                log_history=[{"loss": 0.1, "epoch": self.args.num_train_epochs}],
            )

        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is False
            assert self.model_accepts_loss_kwargs is False
            events.append("train_fresh")
            np.testing.assert_allclose(self.loss.weights, env["full_weight_values"])
            assert self.loss.gamma == env["FOCAL_GAMMA"]
            return SimpleNamespace(metrics={"train_loss": 0.1, "train_runtime": 1.0})

        def save_metrics(self, split, metrics):
            assert split == "train" and "eval_macro_f1" not in metrics

        def save_state(self):
            events.append("save_final_state")

        def predict(self, dataset, metric_key_prefix):
            assert dataset is env["test_dataset"] and not dataset.labelled
            assert len(dataset.frame) == 6000 and metric_key_prefix == "test"
            events.append("predict_test")
            return SimpleNamespace(predictions=(new_logits.copy(),))

    def weights(label_ids, class_count, *, power, enabled):
        assert len(label_ids) == 14000 and class_count == 3
        values = (len(label_ids) / np.bincount(label_ids).astype(float)) ** (power if enabled else 0)
        return (values / np.average(values, weights=np.bincount(label_ids))).astype(np.float32)

    def make_loss(values, gamma):
        return SimpleNamespace(weights=np.array(values, copy=True), gamma=gamma)

    def predict_lexical(bundle, frame):
        assert bundle["train_ids"] == set(train.id)
        assert frame is test and "label" not in frame
        return lexical_scores.copy()

    def fit_lexical(frame):
        assert len(frame) == 14000
        assert set(frame.id).isdisjoint(test.id)
        events.append("refit_lexical")
        return {"train_ids": set(frame.id)}

    def dump_lexical(bundle, destination, compress):
        assert set(bundle["train_ids"]) == set(train.id)
        assert Path(destination).name == "final_lexical.joblib" and compress == 3
        Path(destination).write_text("mock")

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Refit tried to retune on old validation scores.")

    def log_softmax(values, temperature=1.0):
        values = np.asarray(values) / temperature
        values = values - values.max(axis=1, keepdims=True)
        return values - np.log(np.exp(values).sum(axis=1, keepdims=True))

    def softmax(values):
        return np.exp(log_softmax(values))

    def apply_frozen(logits, lexical, config, languages):
        # The entire policy is mandatory: PT may be enabled even when the
        # original global blend gate is False. Check actual test-row order.
        assert config == selected_config
        assert config is env["FROZEN_POSTPROCESS_CONFIG"]
        np.testing.assert_array_equal(np.asarray(languages), test["language"].to_numpy())
        np.testing.assert_array_equal(logits, new_logits)
        events.append("apply_frozen")
        if config["portuguese_pair"]["enabled"]:
            events.append("apply_frozen_pair")
            assert config["portuguese_pair"]["bias"] == 0.15
        parameters = config["parameters"]
        combined = log_softmax(logits, parameters["neural_temperature"])
        fraction = parameters["lexical_weight"]
        if fraction:
            combined = (1 - fraction) * combined + fraction * log_softmax(
                lexical, parameters["lexical_temperature"],
            )
        # Pair-score arithmetic is covered in check_v5_postprocess.py; this
        # refit spy verifies call routing, frozen config and row alignment.
        return combined + np.array(parameters["class_bias"])

    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir()
        (artifact_dir / selection_weights_name).mkdir()
        selection_submission = artifact_dir / "submission_selection.csv"
        selection_submission.write_text("id,label\nt0,BENIGN\n")
        selection_csv_before = selection_submission.read_bytes()
        env = {
            "np": np, "pd": pd, "json": json, "gc": gc,
            "torch": SimpleNamespace(
                tensor=lambda x, dtype: np.asarray(x, dtype=dtype), float32=np.float32,
                cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache")),
            ),
            "RUN_DIR": run_dir, "ARTIFACT_DIR": artifact_dir,
            "ADAPTER_DIR": artifact_dir / selection_weights_name,
            "MODEL_PRESET": model_preset,
            "USE_FULL_TRAIN": use_full_train, "FULL_TRAIN_EPOCHS": override,
            "ENABLE_LEXICAL_BRANCH": lexical_enabled,
            "LABELS": labels, "fit_df": fit, "valid_df": valid,
            "train_df": train, "test_df": test,
            "test_dataset": FakeDataset(test, labelled=False),
            "PromptDataset": FakeDataset,
            "train_result": SimpleNamespace(metrics={"train_loss": 0.2}),
            "training_args": SimpleNamespace(to_json_string=lambda: '{"eval_strategy":"epoch"}'),
            "test_logits": old_logits.copy(),
            "test_lexical_scores": lexical_scores.copy() if lexical_enabled else None,
            "final_test_proba": softmax(old_logits),
            "final_test_pred": old_logits.argmax(axis=1),
            "final_test_scores": log_softmax(old_logits),
            "valid_logits": original_valid_logits,
            "final_valid_pred": original_valid_predictions,
            "POSTPROCESS_CONFIG": json.loads(json.dumps(selected_config)),
            "POSTPROCESS_SELECTION_REPORT": {"cross_fitted_postprocess_macro_f1": 0.952069},
            "compute_class_weights": weights,
            "make_weighted_focal_loss": make_loss,
            "selected_checkpoint_epoch": lambda state: 3,
            "SEED": 42, "RUN_NAME": "test_v5", "CLASS_WEIGHT_POWER": 0.65,
            "USE_CLASS_WEIGHTS": True, "FOCAL_GAMMA": 2.0,
            "LEARNING_RATE": 5e-5, "TRAIN_BATCH_SIZE": 8, "EVAL_BATCH_SIZE": 16,
            "GRAD_ACCUM_STEPS": 4, "WEIGHT_DECAY": 0.01, "WARMUP_RATIO": 0.1,
            "USE_BF16": False, "GRADIENT_CHECKPOINTING": True,
            "TrainingArguments": FakeArguments, "Trainer": FakeTrainer,
            "tokenizer": SimpleNamespace(save_pretrained=lambda path: None),
            "data_collator": object(),
            "set_seed": lambda seed: events.append(("seed", seed)),
            "fit_lexical_branch": fit_lexical, "lexical_predict": predict_lexical,
            "joblib": SimpleNamespace(dump=dump_lexical),
            "apply_selected_postprocess": apply_frozen,
            "postprocess_log_scores": lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Refit bypassed the full frozen global+pair policy")
            ),
            "softmax_from_log_scores": softmax, "log_softmax_np": log_softmax,
            "fit_postprocessor": forbidden_fit, "run_postprocessing": forbidden_fit,
            "cross_fitted_postprocess": forbidden_fit,
            "run_portuguese_pair_tuning": forbidden_fit,
            "_fit_portuguese_pair_bias": forbidden_fit,
            "_crossfit_global_log_scores": forbidden_fit,
        }
        env["model"] = FakeModel("selection")
        env["trainer"] = SimpleNamespace(
            model=env["model"], model_wrapped=env["model"],
            accelerator=FakeAccelerator(env["model"]),
            state=SimpleNamespace(
                best_model_checkpoint="checkpoint-1182", best_metric=0.9197106836910539,
                epoch=5, log_history=[{"epoch": 3, "step": 1182, "eval_macro_f1": 0.9197106836910539}],
            ),
        )
        old_model_ref = weakref.ref(env["model"])

        def build_model():
            # Includes old Trainer.model_wrapped and Accelerator tracking refs.
            assert old_model_ref() is None, "Old GPU model still referenced during rebuild."
            events.append("build_fresh")
            return FakeModel("final")

        env["build_model"] = build_model
        exec(compiled_cell, env)
        report = json.loads((artifact_dir / "final_fit_report.json").read_text())
        assert report["completed"]
        assert report["full_train_validation_macro_f1"] is None
        assert report["validation_evaluated_after_refit"] is False
        assert report["postprocess_frozen"] is True
        assert env["valid_logits"] is original_valid_logits
        assert env["final_valid_pred"] is original_valid_predictions
        assert env["POSTPROCESS_CONFIG"] == selected_config
        assert env["SELECTION_REPORT"]["best_checkpoint"] == "checkpoint-1182"
        assert env["SELECTION_REPORT"]["best_validation_macro_f1"] == 0.9197106836910539
        assert env["SELECTION_REPORT"]["selected_epoch"] == 3
        assert env["SELECTION_TRAINING_ARGUMENTS"]["eval_strategy"] == "epoch"
        assert selection_submission.read_bytes() == selection_csv_before
        np.testing.assert_array_equal(np.load(artifact_dir / "selection_test_logits.npy"), old_logits)
        np.testing.assert_allclose(env["final_test_proba"].sum(axis=1), 1.0)
        if use_full_train:
            assert report["training_rows"] == 14000
            assert report["epochs"] == (3 if override is None else override)
            assert report["active_adapter"] == final_weights_name
            assert (artifact_dir / final_weights_name / "mock-model.json").is_file()
            assert len(pd.read_csv(artifact_dir / "full_train_rows.csv")) == 14000
            assert events.index("free_memory") < events.index("build_fresh") < events.index("train_fresh")
            np.testing.assert_array_equal(env["test_logits"], new_logits)
            assert ("refit_lexical" in events) == lexical_enabled
            assert events.count("apply_frozen") == 1, "Every full refit must apply the complete frozen policy."
            assert events.count("apply_frozen_pair") == int(pair_enabled)
            assert report["active_lexical"] == ("final_lexical.joblib" if lexical_enabled else None)
        else:
            assert events == []
            assert report["training_rows"] == 12600
            assert report["active_adapter"] == selection_weights_name
            np.testing.assert_array_equal(env["test_logits"], old_logits)
        print("Refit contract OK:", model_preset, "full_train=", use_full_train,
              "lexical=", lexical_enabled, "global=", postprocess_enabled,
              "pair=", pair_enabled, "epoch_override=", override)


for preset in ["wolf", "distilbert"]:
    check_case(preset, True, True, True)
    check_case(preset, True, False, False, override=2)
    check_case(preset, True, False, True)
    check_case(preset, False, True, True)
    check_case(preset, False, False, False)
    check_case(preset, True, False, False, pair_enabled=True)
    check_case(preset, True, True, True, pair_enabled=True)
    check_case(preset, False, False, False, pair_enabled=True)
print("All final-refit CPU contract checks passed; no GPU training executed.")
