"""CPU execution checks of the actual notebook's small selection ZIP export.

Uses generated 12600/1400/6000 row fixtures in temporary directories. It never
downloads weights, runs training, modifies data, or creates a real /content path.
Run: python scripts/check_selection_bundle.py
"""

import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
helper = (ROOT / "scripts/_v5_selection_bundle.py").read_text(encoding="utf-8").strip()
notebook = json.loads((ROOT / "final_97.ipynb").read_text(encoding="utf-8"))
matching = ["".join(cell.get("source", [])) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def make_submission(" in "".join(cell["source"])]
assert len(matching) == 1, "Expected exactly one notebook submission-export cell."
assert helper in matching[0], "Active notebook does not embed the current selection-bundle helper exactly."
source = matching[0][matching[0].index(helper):]


class RedirectColabPath(ast.NodeTransformer):
    """Redirect the actual Colab ZIP destination into this test's temp folder."""

    def __init__(self, destination):
        self.destination = str(destination)
        self.replacements = 0

    def visit_Constant(self, node):
        if node.value == "/content/selection_predictions.zip":
            self.replacements += 1
            return ast.copy_location(ast.Constant(value=self.destination), node)
        return node


def macro_f1(truth, prediction):
    return float(f1_score(truth, prediction, labels=np.arange(3), average="macro", zero_division=0))


def check_case(lexical_enabled, missing_required=False):
    labels = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = dict(enumerate(labels))
    rng = np.random.default_rng(42)
    truth = np.tile([0, 0, 0, 0, 0, 0, 1, 1, 1, 2], 140)
    valid = pd.DataFrame({
        "id": [f"v{i:05d}" for i in range(1400)],
        "language": np.tile(["pt", "en", "fr", "it"], 350),
        "label_id": truth,
        "label": np.asarray(labels)[truth],
    })
    fit = pd.DataFrame({"id": [f"f{i:05d}" for i in range(12600)]})
    test = pd.DataFrame({"id": [f"t{i:05d}" for i in range(6000)],
                         "language": np.tile(["it", "fr", "pt", "en"], 1500)})
    valid_logits = rng.normal(size=(1400, 3)) + np.eye(3)[truth] * 4
    selection_logits = rng.normal(size=(6000, 3))
    exp_scores = np.exp(selection_logits - selection_logits.max(axis=1, keepdims=True))
    selection_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    valid_lexical = rng.normal(size=(1400, 3)) if lexical_enabled else None
    test_lexical = rng.normal(size=(6000, 3)) if lexical_enabled else None
    base_config = {
        "enabled": False,
        "parameters": {"neural_temperature": 1.0, "lexical_temperature": None,
                       "lexical_weight": 0.0, "class_bias": [0.0, 0.0, 0.0]},
    }
    with TemporaryDirectory(prefix="sigma69-selection-bundle-") as directory:
        run_dir = Path(directory)
        artifacts = run_dir / "artifacts"
        artifacts.mkdir()
        destination = run_dir / "selection_predictions.zip"
        np.save(artifacts / "validation_logits.npy", valid_logits.astype(np.float32))
        if not missing_required:
            valid.to_csv(artifacts / "validation_ids.csv", index=False)
        test.to_csv(artifacts / "test_ids.csv", index=False)
        pd.concat([fit.assign(split="fit"), valid[["id"]].assign(split="validation")],
                  ignore_index=True).to_csv(artifacts / "split.csv", index=False)
        submission = pd.DataFrame({"id": test.id,
                                   "label": np.asarray(labels)[selection_logits.argmax(axis=1)]})
        csv_names = ["submission_selection.csv", "submission_selection_raw.csv",
                     "submission_selection_global.csv"]
        for name in csv_names:
            submission.to_csv(artifacts / name, index=False)
        csv_before = {name: (artifacts / name).read_bytes() for name in csv_names}
        (artifacts / "postprocess.json").write_text(json.dumps(base_config), encoding="utf-8")
        (artifacts / "postprocess_selection.json").write_text("{}", encoding="utf-8")
        valid.assign(predicted_label=np.asarray(labels)[valid_logits.argmax(axis=1)]).to_csv(
            artifacts / "validation_predictions.csv", index=False,
        )
        # These stale files must NEVER enter a selection-only prediction bundle.
        stale_model = artifacts / "final_model"
        stale_model.mkdir()
        (stale_model / "model.safetensors").write_bytes(b"stale model weights")
        (artifacts / "final_model.bin").write_bytes(b"stale flat model weights")
        (artifacts / "optimizer.pt").write_bytes(b"stale optimizer state")
        np.save(artifacts / "test_logits.npy", np.full((6000, 3), 99.0))
        np.save(artifacts / "final_test_logits.npy", np.full((6000, 3), -99.0))

        state = SimpleNamespace(best_metric=macro_f1(truth, valid_logits.argmax(axis=1)),
                                best_model_checkpoint="checkpoint-1576")

        def selected_epoch(seen_state):
            assert seen_state is state
            return 4

        env = {
            "np": np, "pd": pd, "Path": Path, "json": json,
            "hashlib": hashlib, "zipfile": zipfile, "ARTIFACT_DIR": artifacts,
            "valid_df": valid, "fit_df": fit, "test_df": test,
            "valid_true": truth, "valid_logits": valid_logits, "test_logits": selection_logits,
            "final_test_proba": selection_probabilities,
            "final_valid_pred": valid_logits.argmax(axis=1),
            "valid_lexical_scores": valid_lexical, "test_lexical_scores": test_lexical,
            "POSTPROCESS_SELECTION_REPORT": {"portuguese_pair": {"enabled": False, "nested_gain": 0.0}},
            "GLOBAL_POSTPROCESS_CONFIG": base_config,
            "GLOBAL_POSTPROCESS_SELECTION_REPORT": {"cross_fitted_postprocess_macro_f1": state.best_metric},
            "macro_f1": macro_f1, "ID2LABEL": id2label, "LABELS": labels, "LABEL2ID": label2id,
            "NOTEBOOK_VERSION": "v5-test", "RUN_ID": "selection_fixture",
            "MODEL_PRESET": "wolf", "WOLF_LOADING_MODE": "v3_legacy",
            "MODEL_PROVENANCE": {"preset": "wolf", "kaggle_handle": "approved/model/transformers/default/1"},
            "DATA_HASHES": {"train.csv": "a" * 64, "test.csv": "b" * 64,
                            "sample_submission.csv": "c" * 64},
            "SPLIT_SEED": 42, "trainer": SimpleNamespace(state=state),
            "selected_checkpoint_epoch": selected_epoch,
        }
        if lexical_enabled:
            env["portuguese_pair_baseline_oof_pred"] = valid_logits.argmax(axis=1)
            env["portuguese_pair_candidate_oof_pred"] = valid_logits.argmax(axis=1)
        redirect = RedirectColabPath(destination)
        tree = redirect.visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        assert redirect.replacements == 1, "Expected one Colab ZIP output destination."
        compiled = compile(tree, "notebook_selection_export", "exec")
        if missing_required:
            try:
                exec(compiled, env)
            except FileNotFoundError as exc:
                assert "validation_ids.csv" in str(exc)
                assert not destination.exists(), "An incomplete selection ZIP was emitted."
            else:
                raise AssertionError("Missing validation IDs did not block ZIP export.")
            print("Selection bundle rejects a missing required validation file before writing ZIP.")
            return

        print("Executing export on SYNTHETIC scores; displayed F1 is not a trained-model result.")
        exec(compiled, env)
        assert env["SELECTION_PREDICTIONS_PATH"] == destination
        assert destination.is_file()
        with zipfile.ZipFile(destination) as packed:
            members = packed.namelist()
            assert len(members) == len(set(members)), "Duplicate ZIP entries."
            manifest = json.loads(packed.read("selection_manifest.json"))
            assert manifest["stage"] == "selection_only"
            assert manifest["contains_model_weights"] is False
            assert manifest["notebook_version"] == "v5-test"
            assert manifest["fit_rows"] == 12600 and manifest["validation_rows"] == 1400
            assert manifest["test_rows"] == 6000 and manifest["selected_epoch"] == 4
            assert manifest["test_logits_file"] == "selection_test_logits.npy"
            assert manifest["validation_logits_file"] == "validation_logits.npy"
            assert manifest["lexical_enabled"] is lexical_enabled
            assert manifest["labels"] == labels and manifest["label2id"] == label2id
            assert manifest["data_sha256"] == env["DATA_HASHES"]
            assert set(members) == set(manifest["file_sha256"]) | {"selection_manifest.json"}
            for name, digest in manifest["file_sha256"].items():
                assert hashlib.sha256(packed.read(name)).hexdigest() == digest
                assert packed.read(name) == (artifacts / name).read_bytes()
            for name in ("test_logits.npy", "final_test_logits.npy", "final_model.bin",
                         "final_model/model.safetensors", "optimizer.pt"):
                assert name not in members
            assert all("/" not in name and "\\" not in name for name in members)
            stored_logits = np.load(BytesIO(packed.read("selection_test_logits.npy")), allow_pickle=False)
            assert stored_logits.shape == (6000, 3) and stored_logits.dtype == np.float32
            np.testing.assert_array_equal(stored_logits, selection_logits.astype(np.float32))
            np.testing.assert_array_equal(
                np.load(BytesIO(packed.read("validation_logits.npy")), allow_pickle=False),
                valid_logits.astype(np.float32),
            )
            np.testing.assert_array_equal(
                np.load(BytesIO(packed.read("selection_test_probabilities.npy")), allow_pickle=False),
                selection_probabilities,
            )
            archived_test = pd.read_csv(BytesIO(packed.read("test_ids.csv")))
            archived_valid = pd.read_csv(BytesIO(packed.read("validation_ids.csv")))
            assert archived_test.id.tolist() == test.id.tolist()
            assert archived_valid.id.tolist() == valid.id.tolist()
            for name in csv_names:
                assert packed.read(name) == csv_before[name]
                assert (artifacts / name).read_bytes() == csv_before[name]
            lexical_names = ("validation_lexical_scores.npy", "selection_test_lexical_scores.npy")
            for name in lexical_names:
                assert (name in members) == lexical_enabled
            if lexical_enabled:
                np.testing.assert_array_equal(
                    np.load(BytesIO(packed.read(lexical_names[0])), allow_pickle=False), valid_lexical,
                )
                np.testing.assert_array_equal(
                    np.load(BytesIO(packed.read(lexical_names[1])), allow_pickle=False), test_lexical,
                )
                assert "validation_portuguese_pair_oof.csv" in members
            else:
                assert "validation_portuguese_pair_oof.csv" not in members
        print(f"Selection bundle OK: lexical={lexical_enabled}; rows, hashes, CSVs, model exclusion verified.")


check_case(lexical_enabled=True)
check_case(lexical_enabled=False)
check_case(lexical_enabled=False, missing_required=True)
print("All selection-bundle CPU checks passed; no GPU/model execution.")
