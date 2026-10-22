"""Check the active notebook's training helpers on CPU without GPU training.

Run: python scripts/check_v3_training.py
Requires numpy and nbformat. The NumPy shim executes the actual notebook loss
callback; it checks values/scaling, not PyTorch autograd or GPU compatibility.
"""

import ast
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "final_97.ipynb",
]


class Tensor(np.ndarray):
    """Minimal tensor operations needed by the notebook's focal loss callback."""

    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    @property
    def device(self):
        return "cpu"

    def float(self):
        # Keep double precision for the finite-difference mathematical check.
        return Tensor(self, dtype=np.float64)

    def to(self, device=None, dtype=None):
        return Tensor(self, dtype=dtype)

    def clone(self):
        return Tensor(self.copy())

    def numel(self):
        return self.size

    def clamp_min(self, minimum):
        return Tensor(np.maximum(self, minimum))

    def pow(self, exponent):
        return Tensor(np.power(self, exponent))


class TorchShim:
    Tensor = Tensor
    float32 = np.float64  # Use doubles for the finite-difference oracle.
    long = np.int64

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        return Tensor(value, dtype=dtype)

    tensor = as_tensor

    @staticmethod
    def exp(value):
        return Tensor(np.exp(value))


class FunctionalShim:
    @staticmethod
    def cross_entropy(logits, labels, weight=None, reduction="mean"):
        values = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        log_prob = values - np.logaddexp.reduce(values, axis=1, keepdims=True)
        losses = -log_prob[np.arange(len(targets)), targets]
        if weight is not None:
            losses *= np.asarray(weight)[targets]
        if reduction == "none":
            return Tensor(losses)
        if reduction == "sum":
            return Tensor(losses.sum())
        if reduction == "mean":
            denominator = len(targets) if weight is None else np.asarray(weight)[targets].sum()
            return Tensor(losses.sum() / denominator)
        raise AssertionError(f"Unexpected reduction: {reduction}")


def read_notebooks():
    documents = []
    for path in NOTEBOOKS:
        document = nbformat.read(path, as_version=4)
        nbformat.validate(document)
        assert document.metadata.kernelspec.name == "python3", path
        assert document.metadata.language_info.name == "python", path
        for index, cell in enumerate(document.cells):
            if cell.cell_type == "code":
                assert cell.execution_count is None and not cell.outputs, (path, index)
                compile(cell.source, f"{path.name}:cell_{index}", "exec")
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    print("Active notebook: schema, syntax, metadata and cleared outputs OK.")
    return nbformat.from_dict(documents[0])


def load_helpers(notebook):
    wanted = {
        "compute_class_weights", "make_weighted_focal_loss", "selected_checkpoint_epoch",
        "build_model", "encode_texts",
    }
    selected = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                selected.append(node)
    found = [node.name for node in selected]
    assert set(found) == wanted and len(found) == len(wanted), (
        f"Expected each training helper exactly once; found {found}"
    )
    focal_factory = next(node for node in selected if node.name == "make_weighted_focal_loss")
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"detach", "numpy", "item"}
        for node in ast.walk(focal_factory)
    ), "Focal loss must keep its probability modulation in the autograd graph"
    namespace = {
        "np": np, "torch": TorchShim, "F": FunctionalShim,
        "math": math, "Path": Path, "re": re,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "v3_notebook_helpers", "exec"), namespace)
    return namespace


def check_weights(function):
    counts = np.array([8, 4, 1])
    targets = np.repeat(np.arange(3), counts)
    weights = function(targets, 3, power=0.65, enabled=True)
    expected = (len(targets) / counts.astype(float)) ** 0.65
    expected /= np.average(expected, weights=counts)
    np.testing.assert_allclose(weights, expected, rtol=1e-6)
    np.testing.assert_allclose(np.average(weights, weights=counts), 1.0, rtol=1e-6)
    assert weights[2] > weights[1] > weights[0] > 0
    np.testing.assert_array_equal(function(targets, 3, power=0.65, enabled=False), np.ones(3))
    np.testing.assert_array_equal(function(targets, 3, power=0, enabled=True), np.ones(3))
    try:
        function(np.array([0, 0, 1]), 3)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("Missing classes should be rejected instead of making infinite weights")
    print("Class weights: inverse-frequency power, sample normalization, disabled mode and missing-class guard OK.")


def loss_value(callback, logits, labels, denominator=None):
    return float(callback(
        SimpleNamespace(logits=Tensor(logits)),
        Tensor(labels, dtype=np.int64),
        num_items_in_batch=denominator,
    ))


def numerical_gradient(function, matrix):
    gradient = np.zeros_like(matrix)
    epsilon = 1e-6
    for index in np.ndindex(matrix.shape):
        plus, minus = matrix.copy(), matrix.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return gradient


def check_focal_loss(factory):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(5, 4))
    parameters = rng.normal(scale=0.6, size=(4, 3))
    labels = np.array([0, 1, 2, 1, 0])
    weights = np.array([0.8, 1.2, 2.4])
    logits = features @ parameters
    log_prob = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    probability = np.exp(log_prob)
    pt = probability[np.arange(len(labels)), labels]
    ce = -log_prob[np.arange(len(labels)), labels]
    alpha = weights[labels]

    for gamma in (0.0, 2.0):
        callback = factory(weights, gamma)
        expected = np.sum(alpha * (1 - pt) ** gamma * ce) / len(labels)
        observed = loss_value(callback, logits, labels)
        np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-10)
        # The same denominator must be passed to every microbatch, including
        # a smaller final microbatch. There must be no extra division by GA.
        splits = (np.arange(3), np.arange(3, 5))
        accumulated = sum(loss_value(callback, logits[ids], labels[ids], 5) for ids in splits)
        np.testing.assert_allclose(accumulated, observed, rtol=1e-10)
        np.testing.assert_allclose(loss_value(callback, logits, labels, Tensor(5)), observed)

        numeric = numerical_gradient(
            lambda matrix: loss_value(callback, features @ matrix, labels), parameters,
        )
        factor = alpha * (1 - pt) ** gamma
        if gamma:
            factor += alpha * gamma * ce * pt * (1 - pt) ** (gamma - 1)
        analytic = features.T @ (
            factor[:, None] * (probability - np.eye(3)[labels])
        ) / len(labels)
        np.testing.assert_allclose(numeric, analytic, rtol=1e-5, atol=1e-8)
        accumulated_numeric = numerical_gradient(
            lambda matrix: sum(
                loss_value(callback, features[ids] @ matrix, labels[ids], 5) for ids in splits
            ), parameters,
        )
        np.testing.assert_allclose(accumulated_numeric, numeric, rtol=1e-5, atol=1e-8)
        print(f"Focal gamma={gamma:g}: correct value; finite-difference and uneven-batch scaling OK.")

    # Same class weight, one easy and one difficult example: gamma=2 reduces
    # the easy example's contribution more strongly.
    probabilities = np.array([[0.95, 0.03, 0.02], [0.20, 0.45, 0.35]])
    cross_entropy = factory(np.ones(3), 0.0)
    focal = factory(np.ones(3), 2.0)
    multipliers = []
    for row in probabilities:
        row_logits = np.log(row)[None, :]
        multipliers.append(loss_value(focal, row_logits, [0]) / loss_value(cross_entropy, row_logits, [0]))
    np.testing.assert_allclose(multipliers, [0.0025, 0.64], rtol=1e-8)
    assert multipliers[0] < multipliers[1]
    mutable_weights = weights.copy()
    frozen_loss = factory(mutable_weights, 2.0)
    before = loss_value(frozen_loss, logits, labels)
    mutable_weights *= 10
    np.testing.assert_allclose(loss_value(frozen_loss, logits, labels), before)
    print("Focal hard-example emphasis, gamma=0 equivalence and isolated weight closures OK.")


def check_selected_epoch(function):
    state = SimpleNamespace(
        best_model_checkpoint="/content/run/checkpoints/checkpoint-120",
        best_metric=0.96,
        epoch=8.0,
        log_history=[
            {"step": 60, "epoch": 1.2, "eval_macro_f1": 0.92},
            {"step": 120, "epoch": 2.4, "loss": 0.15},
            {"step": 120, "epoch": 2.4, "eval_macro_f1": 0.96},
            {"step": 240, "epoch": 4.8, "eval_macro_f1": 0.94},
            {"step": 400, "epoch": 8.0, "train_loss": 0.08},
        ],
    )
    assert function(state) == 3, "Full-data refit must use the selected checkpoint epoch, rounded up"
    state.best_model_checkpoint = "/content/run/checkpoints/checkpoint-240"
    assert function(state) == 5
    for missing in (None, "/content/run/checkpoints/checkpoint-999"):
        state.best_model_checkpoint = missing
        try:
            function(state)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing selected epoch must not silently use the last epoch")
    print("Refit duration: best checkpoint epoch, ceiling and missing-evidence failure OK.")


def check_model_factory(function):
    """Verify factory dispatch/configuration with model-load spies, never download weights."""
    namespace = function.__globals__
    labels = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
    id2label = dict(enumerate(labels))
    load_calls, seed_calls, quant_calls, lora_calls = [], [], [], []

    class Linear4bit:
        pass

    class ModelStub:
        def __init__(self, config, quantized):
            self.config = config
            self.classifier = SimpleNamespace(out_features=3)
            self.pre_classifier = object()
            self.is_loaded_in_4bit = quantized
            self.peft = False

        def named_parameters(self):
            names = (["base.lora_A", "base.pre_classifier.modules_to_save.default.weight",
                      "base.classifier.modules_to_save.default.weight"]
                     if self.peft else ["backbone.weight", "classifier.weight"])
            return [(name, SimpleNamespace(
                requires_grad=True, dtype=TorchShim.float32, numel=lambda: 16,
            )) for name in names]

        def parameters(self):
            return [parameter for _, parameter in self.named_parameters()]

        def modules(self):
            return [Linear4bit(), self.classifier, self.pre_classifier]

        def named_modules(self):
            return [(f"layer.{index}.{suffix}", Linear4bit())
                    for index in range(2) for suffix in namespace["LORA_TARGETS"]]

        def print_trainable_parameters(self):
            pass

    def load(source, **kwargs):
        load_calls.append((source, kwargs))
        config = kwargs.get("config") or SimpleNamespace(
            model_type="distilbert", num_labels=kwargs["num_labels"], id2label=kwargs["id2label"],
        )
        return ModelStub(config, "quantization_config" in kwargs)

    def quantize(**kwargs):
        quant_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def lora(**kwargs):
        lora_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def wrap_peft(model, config):
        model.peft = True
        return model

    namespace.update(
        set_seed=seed_calls.append, SEED=42, LABELS=labels,
        ID2LABEL=id2label, LABEL2ID={label: index for index, label in id2label.items()},
        LOAD_KWARGS={"revision": "fixed-revision", "trust_remote_code": False},
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load),
        BitsAndBytesConfig=quantize, bnb=SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit)),
        HEAD_MODULES=["pre_classifier", "classifier"],
        LORA_TARGETS=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
        COMPUTE_DTYPE="float16", GRADIENT_CHECKPOINTING=True,
        LORA_R=64, LORA_ALPHA=128, LORA_DROPOUT=0.05,
        LoraConfig=lora, TaskType=SimpleNamespace(SEQ_CLS="SEQ_CLS"),
        prepare_model_for_kbit_training=lambda model, **kwargs: model,
        get_peft_model=wrap_peft,
    )
    with TemporaryDirectory() as directory:
        namespace["ARTIFACT_DIR"] = Path(directory)
        original = SimpleNamespace(model_type="modernbert", num_labels=2, id2label={0: "safe", 1: "attack"})
        namespace.update(MODEL_PRESET="wolf", MODEL_SOURCE="approved-wolf", base_config=original)
        wolf = function()
        assert original.num_labels == 2 and original.id2label == {0: "safe", 1: "attack"}
        assert wolf.config.num_labels == 3 and wolf.config.id2label == id2label
        assert wolf.config.reference_compile is False
        kwargs = load_calls[-1][1]
        assert kwargs["ignore_mismatched_sizes"] is True
        assert kwargs["dtype"] == TorchShim.float32 and kwargs["attn_implementation"] == "sdpa"
        assert kwargs["trust_remote_code"] is False and kwargs["revision"] == "fixed-revision"
        assert "quantization_config" not in kwargs and "device_map" not in kwargs
        assert not quant_calls and not lora_calls, "Wolf full tuning must not apply the DistilBERT QLoRA setup"
        wolf_refit = function()
        assert wolf_refit is not wolf and wolf_refit.config is not wolf.config
        assert len(load_calls) == 2, "Full-data refit must reload the original pretrained checkpoint"
        assert wolf_refit.config.reference_compile is False and original.num_labels == 2

        namespace.update(MODEL_PRESET="distilbert", MODEL_SOURCE="approved-distilbert",
                         base_config=SimpleNamespace(model_type="distilbert", n_layers=2))
        distilbert = function()
        assert distilbert.peft and distilbert.is_loaded_in_4bit
        assert quant_calls[-1]["load_in_4bit"] and quant_calls[-1]["bnb_4bit_quant_type"] == "nf4"
        assert lora_calls[-1]["r"] == 64 and lora_calls[-1]["lora_alpha"] == 128
        assert set(lora_calls[-1]["target_modules"]) == set(namespace["LORA_TARGETS"])
        assert seed_calls == [42, 42, 42], "Each fresh model/refit must reset its training seed"
        assert (Path(directory) / "trainable_parameters.txt").is_file()
    print("Model factories: Wolf FP32/full tuning/head reset/no compile and DistilBERT QLoRA dispatch OK (load spies).")


def check_tokenizer(function):
    """Generic fast tokenizers add tokens in their postprocessor, not their builder."""
    class GenericFastTokenizerStub:
        def __call__(self, texts, **kwargs):
            assert kwargs.get("add_special_tokens") is True
            assert kwargs.get("truncation") is False
            assert kwargs.get("return_special_tokens_mask") is True
            assert kwargs.get("return_attention_mask") is False
            assert kwargs.get("return_token_type_ids") is False
            rows = [[-101] + list(range(int(text))) + [-102] for text in texts]
            masks = [[1] + [0] * int(text) + [1] for text in texts]
            return {"input_ids": rows, "special_tokens_mask": masks}

        def build_inputs_with_special_tokens(self, ids):
            raise AssertionError("Generic fast tokenizer builder does not add CLS/SEP")

    namespace = function.__globals__
    namespace.update(tokenizer=GenericFastTokenizerStub(), MAX_LENGTH=512, LONG_TEXT_MODE="head_tail")
    encoded, lengths = function(["0", "10", "510", "700"])
    expected = [
        [-101, -102],
        [-101] + list(range(10)) + [-102],
        [-101] + list(range(510)) + [-102],
        [-101] + list(range(255)) + list(range(445, 700)) + [-102],
    ]
    assert lengths == [0, 10, 510, 700]
    assert encoded["input_ids"] == expected
    assert encoded["attention_mask"] == [[1] * len(row) for row in expected]
    assert all(len(row) <= 512 for row in encoded["input_ids"])

    namespace["LONG_TEXT_MODE"] = "head"
    encoded, lengths = function(["10", "700"])
    assert encoded["input_ids"] == [
        [-101] + list(range(10)) + [-102],
        [-101] + list(range(510)) + [-102],
    ]
    assert lengths == [10, 700]
    namespace["MAX_LENGTH"] = 1
    try:
        function(["10"])
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("Tokenizer accepted a budget smaller than its mandatory special tokens")
    print("Generic fast tokenizer: real CLS/SEP retained in head/head-tail modes, raw lengths and budget guard OK.")


def main():
    notebook = read_notebooks()
    helpers = load_helpers(notebook)
    check_weights(helpers["compute_class_weights"])
    check_focal_loss(helpers["make_weighted_focal_loss"])
    check_selected_epoch(helpers["selected_checkpoint_epoch"])
    check_model_factory(helpers["build_model"])
    check_tokenizer(helpers["encode_texts"])
    print("All training-helper CPU checks passed. PyTorch autograd, GPU training and Kaggle score were not evaluated.")


if __name__ == "__main__":
    main()
