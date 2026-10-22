"""Optional Portuguese BENIGN/JAILBREAK correction after the global policy.

This cell relies on the global post-processing helpers defined in the preceding
cell. It does not run training, read labels from test, or execute on import.
"""

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _identity_postprocess_parameters() -> dict[str, Any]:
    return {
        "neural_temperature": 1.0,
        "lexical_temperature": None,
        "lexical_weight": 0.0,
        "class_bias": [0.0, 0.0, 0.0],
    }


def apply_portuguese_pair_bias(
    log_scores: np.ndarray,
    languages: Sequence[str],
    bias: float = 0.0,
    max_margin: float = 0.75,
) -> np.ndarray:
    """Change only low-margin Portuguese BENIGN/JAILBREAK decisions.

    Positive bias favors JAILBREAK; negative bias favors BENIGN. Skip a proposed
    change if it would make PROMPT_INJECTION win. An existing PROMPT_INJECTION
    winner and every non-Portuguese row remain unchanged in score space.
    """
    scores = _as_score_matrix("pair_log_scores", log_scores)
    languages = np.asarray(languages, dtype=str)
    if languages.shape != (len(scores),):
        raise ValueError("languages must contain exactly one value per score row.")
    if not np.isfinite(bias) or not np.isfinite(max_margin) or max_margin <= 0:
        raise ValueError("Pair bias must be finite and max_margin must be positive.")
    adjusted = scores.copy()
    if float(bias) == 0.0:
        return adjusted
    original_pred = scores.argmax(axis=1)
    eligible = (
        (languages == "pt")
        & (original_pred != 2)
        & (np.abs(scores[:, 1] - scores[:, 0]) <= float(max_margin))
    )
    adjusted[eligible, 1] += float(bias)
    new_pi = eligible & (adjusted.argmax(axis=1) == 2)
    adjusted[new_pi] = scores[new_pi]
    return adjusted


def apply_selected_postprocess(
    neural_logits: np.ndarray,
    lexical_scores: Optional[np.ndarray],
    policy: Mapping[str, Any],
    languages: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Apply frozen global and optional pair policies, including after refit."""
    scores = postprocess_log_scores(neural_logits, lexical_scores, policy["parameters"])
    pair = policy.get("portuguese_pair", {})
    if pair.get("enabled", False):
        if languages is None:
            raise ValueError("Frozen Portuguese pair policy requires row languages.")
        scores = apply_portuguese_pair_bias(
            scores, languages, float(pair["bias"]), float(pair["max_margin"])
        )
    return scores


def _fit_portuguese_pair_bias(
    scores: np.ndarray,
    y_true: np.ndarray,
    languages: np.ndarray,
    *,
    bias_values: Sequence[float],
    max_margin: float,
    bias_l2: float,
    min_pt_class_support: int,
) -> dict[str, Any]:
    """Fit one shrunk scalar on supplied development rows only."""
    baseline_score = macro_f1(y_true, scores.argmax(axis=1))
    support = [int(((languages == "pt") & (y_true == label)).sum()) for label in (0, 1)]
    best = {
        "bias": 0.0,
        "fit_macro_f1": baseline_score,
        "penalized_objective": baseline_score,
        "pt_benign_support": support[0],
        "pt_jailbreak_support": support[1],
    }
    if min(support) < min_pt_class_support:
        best["reason"] = "insufficient_portuguese_pair_support"
        return best
    for bias in bias_values:
        predicted = apply_portuguese_pair_bias(scores, languages, bias, max_margin).argmax(axis=1)
        score = macro_f1(y_true, predicted)
        objective = score - bias_l2 * float(bias) ** 2
        better = objective > best["penalized_objective"] + 1e-12
        tied = abs(objective - best["penalized_objective"]) <= 1e-12
        if better or (tied and abs(bias) < abs(best["bias"])):
            best.update(bias=float(bias), fit_macro_f1=float(score),
                        penalized_objective=float(objective))
    return best


def _crossfit_global_log_scores(
    logits: np.ndarray,
    y_true: np.ndarray,
    strata: np.ndarray,
    lexical: Optional[np.ndarray],
    *,
    n_splits: int,
    seed: int,
    fit_kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return scores for rows excluded from their global-policy fitting fold."""
    scores = np.full(logits.shape, np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fit_index, held_index in splitter.split(np.zeros(len(y_true)), strata):
        params = fit_postprocessor(
            logits[fit_index], y_true[fit_index],
            None if lexical is None else lexical[fit_index], **dict(fit_kwargs),
        )
        scores[held_index] = postprocess_log_scores(
            logits[held_index], None if lexical is None else lexical[held_index], params
        )
        splits.append({"fit_indices": fit_index.tolist(), "holdout_indices": held_index.tolist()})
    if not np.isfinite(scores).all():
        raise RuntimeError("Global cross-fitting left a validation row unpredicted.")
    return scores, splits


def run_portuguese_pair_tuning(
    valid_logits: np.ndarray,
    test_logits: np.ndarray,
    valid_df: Any,
    test_languages: Sequence[str],
    base_result: Mapping[str, Any],
    valid_lexical_scores: Optional[np.ndarray] = None,
    test_lexical_scores: Optional[np.ndarray] = None,
    *,
    split_seed: int = 42,
    min_postprocess_gain: float = 0.002,
    min_pair_gain: float = 0.002,
    n_splits: int = 3,
    global_fit_kwargs: Optional[Mapping[str, Any]] = None,
    bias_values: Sequence[float] = (-0.30, -0.15, 0.0, 0.15, 0.30),
    max_margin: float = 0.75,
    bias_l2: float = 0.002,
    min_pt_class_support: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a pair correction against an independently fitted fold baseline.

    Each outer training portion independently chooses its global blend/bias via
    inner cross-fitting. The pair bias is fit using inner out-of-fold scores and
    is scored on the untouched outer holdout. Both compared predictions come
    from the SAME fold-specific global pipeline. The existing deployed global
    gate is retained. Cross-fitting does not undo earlier checkpoint selection
    on this validation set, so these scores are not independent model estimates.
    """
    logits = _as_score_matrix("valid_logits", valid_logits, len(valid_df))
    test = _as_score_matrix("test_logits", test_logits)
    required = {"id", "language", "label_id"}
    if not required.issubset(valid_df.columns) or valid_df["id"].duplicated().any():
        raise ValueError("valid_df needs unique id plus language and label_id columns.")
    y_true = valid_df["label_id"].to_numpy(dtype=np.int64)
    languages = valid_df["language"].astype(str).to_numpy()
    test_languages = np.asarray(test_languages, dtype=str)
    if test_languages.shape != (len(test),):
        raise ValueError("test_languages must align with test_logits.")
    if not np.isin(y_true, [0, 1, 2]).all():
        raise ValueError("Validation labels must be class indices 0, 1, 2.")
    if (valid_lexical_scores is None) != (test_lexical_scores is None):
        raise ValueError("Provide both validation and test lexical scores, or neither.")
    lexical = (None if valid_lexical_scores is None else
               _as_score_matrix("valid_lexical_scores", valid_lexical_scores, len(logits)))
    test_lexical = (None if test_lexical_scores is None else
                    _as_score_matrix("test_lexical_scores", test_lexical_scores, len(test)))
    bias_values = tuple(float(value) for value in bias_values)
    if not bias_values or 0.0 not in bias_values or not np.isfinite(bias_values).all():
        raise ValueError("Pair bias grid must be finite and contain zero.")
    if any(abs(value) > 0.30 for value in bias_values):
        raise ValueError("Pair correction is limited to biases between -0.30 and +0.30.")
    if n_splits < 3 or min_pair_gain <= 0 or min_postprocess_gain < 0 or bias_l2 < 0:
        raise ValueError("Need >=3 folds, positive pair gain, nonnegative global gain and penalty.")
    if not np.isfinite(max_margin) or max_margin <= 0 or min_pt_class_support < 1:
        raise ValueError("Need positive max_margin and class-support threshold.")
    fit_kwargs = dict(global_fit_kwargs or {})
    pair_fit_kwargs = dict(bias_values=bias_values, max_margin=max_margin,
                           bias_l2=bias_l2, min_pt_class_support=min_pt_class_support)
    result = dict(base_result)
    policy = deepcopy(base_result["POSTPROCESS_CONFIG"])
    selection_report = deepcopy(base_result["POSTPROCESS_SELECTION_REPORT"])
    baseline_valid_scores = apply_selected_postprocess(logits, lexical, policy, languages)
    baseline_test_scores = apply_selected_postprocess(test, test_lexical, policy, test_languages)
    if policy.get("portuguese_pair", {}).get("enabled", False):
        raise ValueError("Pair fitting must start from the original global-only policy.")
    np.testing.assert_array_equal(baseline_valid_scores.argmax(axis=1), base_result["final_valid_pred"])
    pair_policy = {
        "enabled": False, "bias": 0.0, "max_margin": float(max_margin),
        "minimum_outer_gain": float(min_pair_gain), "bias_grid": list(bias_values),
        "bias_l2": float(bias_l2), "minimum_pt_class_support": int(min_pt_class_support),
    }
    policy["portuguese_pair"] = pair_policy
    report = {
        "enabled": False,
        "global_policy_retained": True,
        "comparison": "pair correction versus the same independently selected global pipeline",
        "outer_folds": int(n_splits),
        "minimum_required_gain": float(min_pair_gain),
        "warnings": [
            "Outer held-out labels never fit that fold's global gate or pair bias.",
            "The checkpoint was selected on this validation set; this is not an independent model estimate.",
            "All-validation scores are fitting diagnostics, not evidence of expected Kaggle performance.",
        ],
    }
    strata = np.asarray([f"{lang}::{label}" for lang, label in zip(languages, y_true)])
    _, counts = np.unique(strata, return_counts=True)
    # Every inner fold must still be stratifiable after removing one outer fold.
    enough_strata = len(counts) >= 3 and counts.min() - np.ceil(counts.min() / n_splits) >= n_splits
    enough_pt = all(int(((languages == "pt") & (y_true == label)).sum()) >= min_pt_class_support
                    for label in (0, 1))
    if not enough_strata or not enough_pt:
        report["reason"] = "insufficient_rows_for_nested_crossfit_or_portuguese_support"
        selection_report["portuguese_pair"] = report
        result.update(POSTPROCESS_CONFIG=policy, POSTPROCESS_SELECTION_REPORT=selection_report)
        if verbose:
            print("Portuguese pair adjustment: retained global baseline; insufficient validation support.")
        return result

    baseline_oof = np.full(len(logits), -1, dtype=np.int64)
    candidate_oof = np.full(len(logits), -1, dtype=np.int64)
    outer_reports = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    for fold, (fit_index, held_index) in enumerate(splitter.split(np.zeros(len(logits)), strata), 1):
        fit_lexical = None if lexical is None else lexical[fit_index]
        inner_scores, inner_splits = _crossfit_global_log_scores(
            logits[fit_index], y_true[fit_index], strata[fit_index], fit_lexical,
            n_splits=n_splits, seed=split_seed + fold, fit_kwargs=fit_kwargs,
        )
        raw_fit_score = macro_f1(y_true[fit_index], logits[fit_index].argmax(axis=1))
        inner_gain = macro_f1(y_true[fit_index], inner_scores.argmax(axis=1)) - raw_fit_score
        global_enabled = bool(inner_gain >= min_postprocess_gain)
        if global_enabled:
            fold_params = fit_postprocessor(logits[fit_index], y_true[fit_index], fit_lexical,
                                            **fit_kwargs)
            pair_fit_scores = inner_scores
        else:
            fold_params = _identity_postprocess_parameters()
            pair_fit_scores = log_softmax_np(logits[fit_index])
        pair_fit = _fit_portuguese_pair_bias(
            pair_fit_scores, y_true[fit_index], languages[fit_index], **pair_fit_kwargs
        )
        held_scores = postprocess_log_scores(
            logits[held_index], None if lexical is None else lexical[held_index], fold_params
        )
        base_pred = held_scores.argmax(axis=1)
        candidate_pred = apply_portuguese_pair_bias(
            held_scores, languages[held_index], pair_fit["bias"], max_margin
        ).argmax(axis=1)
        baseline_oof[held_index] = base_pred
        candidate_oof[held_index] = candidate_pred
        base_score = macro_f1(y_true[held_index], base_pred)
        candidate_score = macro_f1(y_true[held_index], candidate_pred)
        outer_reports.append({
            "fold": fold, "fit_indices": fit_index.tolist(), "holdout_indices": held_index.tolist(),
            "inner_splits_local_to_fit_indices": inner_splits,
            "global_gate_passed": global_enabled, "inner_global_gain": float(inner_gain),
            "pair_fit": pair_fit, "baseline_macro_f1": base_score,
            "candidate_macro_f1": candidate_score, "gain": float(candidate_score - base_score),
        })
        if verbose:
            print(f"Portuguese pair fold {fold}/{n_splits}: bias={pair_fit['bias']:+.2f}, "
                  f"held-out gain={candidate_score - base_score:+.6f}")
    if (baseline_oof < 0).any() or (candidate_oof < 0).any():
        raise RuntimeError("Nested cross-fitting did not predict every validation row.")
    preserved = (languages != "pt") | (baseline_oof == 2)
    if not np.array_equal(candidate_oof[preserved], baseline_oof[preserved]):
        raise AssertionError("Pair adjustment changed a protected language or PI winner.")
    if not np.array_equal(candidate_oof == 2, baseline_oof == 2):
        raise AssertionError("Pair adjustment changed PROMPT_INJECTION membership.")
    baseline_macro = macro_f1(y_true, baseline_oof)
    candidate_macro = macro_f1(y_true, candidate_oof)
    outer_gain = candidate_macro - baseline_macro
    pt_mask = languages == "pt"
    pt_gain = macro_f1(y_true[pt_mask], candidate_oof[pt_mask]) - macro_f1(y_true[pt_mask], baseline_oof[pt_mask])
    positive_folds = sum(fold["gain"] > 1e-12 for fold in outer_reports)
    required_positive = int(np.ceil(2 * n_splits / 3))
    outer_gate = bool(outer_gain >= min_pair_gain and pt_gain > 0 and positive_folds >= required_positive)

    # Fit the deployed scalar using OOF global scores. If the original global
    # gate failed, fit to raw scores so a rejected blend cannot be reintroduced.
    if base_result["POSTPROCESS_CONFIG"]["enabled"]:
        final_fit_scores, _ = _crossfit_global_log_scores(
            logits, y_true, strata, lexical, n_splits=n_splits,
            seed=split_seed, fit_kwargs=fit_kwargs,
        )
    else:
        final_fit_scores = log_softmax_np(logits)
    final_pair = _fit_portuguese_pair_bias(final_fit_scores, y_true, languages, **pair_fit_kwargs)
    candidate_valid_scores = apply_portuguese_pair_bias(
        baseline_valid_scores, languages, final_pair["bias"], max_margin
    )
    diagnostic_base = macro_f1(y_true, baseline_valid_scores.argmax(axis=1))
    diagnostic_candidate = macro_f1(y_true, candidate_valid_scores.argmax(axis=1))
    diagnostic_gain = diagnostic_candidate - diagnostic_base
    enabled = bool(outer_gate and final_pair["bias"] != 0 and diagnostic_gain > 0)
    pair_policy.update(enabled=enabled, bias=float(final_pair["bias"]) if enabled else 0.0)
    report.update({
        "enabled": enabled, "outer_gate_passed": outer_gate,
        "nested_baseline_macro_f1": baseline_macro, "nested_candidate_macro_f1": candidate_macro,
        "nested_gain": float(outer_gain), "portuguese_nested_gain": float(pt_gain),
        "positive_folds": int(positive_folds), "required_positive_folds": required_positive,
        "folds": outer_reports, "final_fit": final_pair,
        "all_validation_baseline_macro_f1": diagnostic_base,
        "all_validation_candidate_macro_f1": diagnostic_candidate,
        "all_validation_diagnostic_gain": float(diagnostic_gain),
        "reason": "passed_nested_and_sanity_gates" if enabled else "global_baseline_retained",
    })
    selection_report["portuguese_pair"] = report
    selection_report["language_specific_bias"] = enabled
    # Preserve enabled/mode: these describe the ORIGINAL GLOBAL gate. Consumers
    # must call apply_selected_postprocess even when that gate is False.
    final_valid_scores = apply_selected_postprocess(logits, lexical, policy, languages)
    final_test_scores = apply_selected_postprocess(test, test_lexical, policy, test_languages)
    result.update(
        final_valid_scores=final_valid_scores, final_test_scores=final_test_scores,
        final_valid_proba=softmax_from_log_scores(final_valid_scores),
        final_test_proba=softmax_from_log_scores(final_test_scores),
        final_valid_pred=final_valid_scores.argmax(axis=1),
        final_test_pred=final_test_scores.argmax(axis=1),
        portuguese_pair_baseline_oof_pred=baseline_oof,
        portuguese_pair_candidate_oof_pred=candidate_oof,
        POSTPROCESS_CONFIG=policy, POSTPROCESS_SELECTION_REPORT=selection_report,
    )
    if not enabled:
        # Keep original arrays exactly, including their probability rounding.
        for name in ("final_valid_pred", "final_test_pred", "final_valid_scores",
                     "final_test_scores", "final_valid_proba", "final_test_proba"):
            result[name] = base_result[name]
    if verbose:
        print("Portuguese pair adjustment:", "enabled" if enabled else "global baseline retained",
              f"| nested gain={outer_gain:+.6f} | positive folds={positive_folds}/{n_splits}")
    return result


# Append after the original global block. Preserve both prediction sets so the
# exported pair candidate can be compared with the original selected policy.
if {"_postprocess_result", "valid_logits", "test_logits", "test_df"}.issubset(globals()):
    GLOBAL_POSTPROCESS_CONFIG = deepcopy(POSTPROCESS_CONFIG)
    GLOBAL_POSTPROCESS_SELECTION_REPORT = deepcopy(POSTPROCESS_SELECTION_REPORT)
    global_valid_pred = final_valid_pred.copy()
    global_test_pred = final_test_pred.copy()
    global_test_proba = final_test_proba.copy()
    if bool(globals().get("ENABLE_PT_PAIR_TUNING", False)):
        _postprocess_result = run_portuguese_pair_tuning(
            valid_logits=valid_logits,
            test_logits=test_logits,
            valid_df=valid_df,
            test_languages=test_df["language"].astype(str).to_numpy(),
            base_result=_postprocess_result,
            valid_lexical_scores=globals().get("valid_lexical_scores"),
            test_lexical_scores=globals().get("test_lexical_scores"),
            split_seed=int(SPLIT_SEED),
            min_postprocess_gain=float(MIN_POSTPROCESS_GAIN),
            min_pair_gain=float(globals().get("PT_PAIR_MIN_GAIN", 0.002)),
            global_fit_kwargs=POSTPROCESS_FIT_KWARGS,
            bias_values=globals().get("PT_PAIR_BIAS_VALUES", (-0.30, -0.15, 0.0, 0.15, 0.30)),
            max_margin=float(globals().get("PT_PAIR_MAX_MARGIN", 0.75)),
        )
        globals().update(_postprocess_result)
    print("Global parameters retained:", POSTPROCESS_CONFIG["parameters"])
    print("Portuguese pair policy:", POSTPROCESS_CONFIG.get("portuguese_pair", {"enabled": False}))
