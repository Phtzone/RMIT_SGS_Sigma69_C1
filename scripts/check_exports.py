"""Run the actual notebook's CSV and ZIP export cells using temporary CPU fixtures.

No model is loaded or trained. Colab paths are redirected into TemporaryDirectory.
Requires numpy and pandas. Run: python scripts/check_exports.py
"""

import ast
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def notebook_sources():
    notebook = json.loads((ROOT / "final_97.ipynb").read_text(encoding="utf-8"))
    return [cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"]


def unique_source(sources, marker):
    matches = [source for source in sources if marker in source]
    assert len(matches) == 1, f"Expected exactly one code cell containing {marker!r}."
    return matches[0]


class RedirectColabPaths(ast.NodeTransformer):
    """Change only literal /content paths, preserving the real export code."""

    def __init__(self, content_directory):
        self.content_directory = content_directory

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/content/"):
            relative = Path(node.value.removeprefix("/content/"))
            assert not relative.is_absolute() and ".." not in relative.parts
            replacement = ast.Constant(value=str(self.content_directory / relative))
            return ast.copy_location(replacement, node)
        return node


def execute_cell(source, namespace, content_directory):
    tree = RedirectColabPaths(content_directory).visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    exec(compile(tree, "notebook_export_cell", "exec"), namespace)


def main():
    sources = notebook_sources()
    config_source = unique_source(sources, 'MODEL_PRESET = "wolf"')
    submission_source = unique_source(sources, "def make_submission(")
    final_export_source = unique_source(sources, "# Xuat du doan cua active model")
    archive_source = unique_source(sources, "run_config = {")

    with TemporaryDirectory() as directory:
        content = Path(directory) / "content"
        run = content / "run"
        artifacts = run / "artifacts"
        artifacts.mkdir(parents=True)
        namespace = {
            "os": SimpleNamespace(environ={}), "np": np, "pd": pd, "Path": Path,
            "json": json, "shutil": shutil, "zipfile": zipfile, "hashlib": hashlib,
            "datetime": datetime, "timezone": timezone, "display": lambda value: None,
        }
        execute_cell(config_source, namespace, content)
        # Reuse real schema/order validation and submission creation functions.
        functions = ast.Module(body=[
            node for node in ast.parse(submission_source).body
            if isinstance(node, ast.FunctionDef) and node.name in {"make_submission", "validate_submission"}
        ], type_ignores=[])
        exec(compile(functions, "notebook_submission_functions", "exec"), namespace)
        ids = [f"t{index:06d}" for index in range(6000)]
        predictions = np.arange(6000) % 3
        namespace.update({
            "RUN_ID": "cpu_export_regression", "RUN_DIR": run, "ARTIFACT_DIR": artifacts,
            "SUBMISSION_PATH": content / "submission.csv",
            "test_df": pd.DataFrame({"id": ids[::-1]}),
            "sample_df": pd.DataFrame({"id": ids}),
            "valid_df": pd.DataFrame(index=np.arange(1400)),
            "final_test_pred": predictions,
            "final_test_proba": np.eye(3)[predictions],
            "test_logits": np.eye(3)[(predictions + 1) % 3] * 3,
            "MODEL_SOURCE": "cpu-fixture/model", "RESOLVED_MODEL_REVISION": None,
            "MODEL_PROVENANCE": {"preset": "wolf", "source": "cpu-fixture/model"},
            "FINAL_FIT_REPORT": {
                "enabled": True, "completed": True, "training_rows": 14000,
                "active_adapter": "final_model", "active_lexical": "final_lexical.joblib",
                "validation_evaluated_after_refit": False, "full_train_validation_macro_f1": None,
            },
            "SELECTION_REPORT": {
                "best_checkpoint": "cpu-fixture/checkpoint-1576", "best_validation_macro_f1": 0.97,
                "training_rows": 12600, "validation_rows": 1400,
            },
            "POSTPROCESS_CONFIG": {"enabled": False},
            "weight_values": np.ones(3), "power": namespace["CLASS_WEIGHT_POWER"],
            "GPU_CAPABILITY": (7, 5), "COMPUTE_DTYPE": "float16",
            "PACKAGE_VERSIONS": {"numpy": np.__version__, "pandas": pd.__version__},
            "DATA_HASHES": {"train.csv": "cpu-fixture", "test.csv": "cpu-fixture"},
            "valid_result": SimpleNamespace(metrics={"validation_macro_f1": 0.97}),
            "training_args": SimpleNamespace(to_json_string=lambda: json.dumps({"eval_strategy": "no"})),
            "trainer": SimpleNamespace(state=SimpleNamespace(log_history=[{"step": 1, "loss": 0.1}])),
            "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda index: "CPU test stub")),
        })
        make_submission = namespace["make_submission"]
        selection_path = artifacts / "submission_selection.csv"
        make_submission(np.zeros(6000, dtype=int), selection_path)
        selection_bytes = selection_path.read_bytes()
        public_selection_path = content / "submission_selection.csv"
        shutil.copy2(selection_path, public_selection_path)

        execute_cell(final_export_source, namespace, content)
        current_bytes = namespace["SUBMISSION_PATH"].read_bytes()
        assert current_bytes != selection_bytes
        assert (artifacts / "submission.csv").read_bytes() == current_bytes
        assert selection_path.read_bytes() == public_selection_path.read_bytes() == selection_bytes
        exported = pd.read_csv(namespace["SUBMISSION_PATH"], dtype=str)
        assert exported.id.tolist() == ids
        assert exported.label.iloc[0] == namespace["ID2LABEL"][int(predictions[-1])]
        assert len(pd.read_csv(artifacts / "submission_raw.csv")) == 6000
        np.testing.assert_array_equal(np.load(artifacts / "test_final_probabilities.npy"), namespace["final_test_proba"])

        # Reproduce the stale-artifact failure after exporting the new public CSV.
        # Archive creation must synchronize from the current public CSV first.
        (artifacts / "submission.csv").write_bytes(selection_bytes)
        assert (artifacts / "submission.csv").read_bytes() != current_bytes
        execute_cell(archive_source, namespace, content)
        digest = hashlib.sha256(current_bytes).hexdigest()
        assert namespace["run_config"]["submission_sha256"] == digest
        assert (artifacts / "submission.csv").read_bytes() == current_bytes
        assert selection_path.read_bytes() == public_selection_path.read_bytes() == selection_bytes
        with zipfile.ZipFile(namespace["ARCHIVE_PATH"]) as packed:
            assert packed.read("submission.csv") == current_bytes
            assert packed.read("submission_selection.csv") == selection_bytes
            config = json.loads(packed.read("run_config.json"))
            assert config["submission_sha256"] == digest
            assert config["active_weights"] == "final_model"
            assert config["notebook_version"] == namespace["NOTEBOOK_VERSION"]
            assert config["max_length"] == namespace["MAX_LENGTH"]
            metrics = json.loads(packed.read("validation_metrics.json"))
            assert metrics["stage"] == "selection_only"
        assert namespace["PUBLIC_ARCHIVE_PATH"].resolve().is_relative_to(content.resolve())
        assert namespace["PUBLIC_ARCHIVE_PATH"].read_bytes() == namespace["ARCHIVE_PATH"].read_bytes()
    print("CSV/ZIP export checks passed: 6000 IDs ordered, stale artifact synchronized, hashes matched, selection preserved.")
    print("Only temporary files were created. No GPU training or model download was executed.")


if __name__ == "__main__":
    main()
