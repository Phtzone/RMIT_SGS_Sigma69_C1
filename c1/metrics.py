"""Metrics, ID-safe probability artifacts and submission validation."""

from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from c1 import LABELS, LABEL_TO_ID


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def check_probabilities(probabilities, count=None):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(LABELS) or (count is not None and len(p) != count):
        raise ValueError('Expected probabilities with shape (rows, 3)')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and between zero and one')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError('Probability rows must sum to one')
    return p


def save_predictions(path, ids, probabilities):
    ids = np.asarray(ids, dtype=str)
    if ids.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError('Prediction IDs must be unique')
    p = check_probabilities(probabilities, len(ids))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, ids=ids, probabilities=p, label_order=np.asarray(LABELS))
    os.replace(path.with_suffix('.tmp'), path)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        if tuple(data['label_order'].tolist()) != LABELS:
            raise ValueError('Prediction label order does not match LABELS')
        ids = data['ids'].astype(str)
        p = check_probabilities(data['probabilities'], len(ids))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction IDs')
    return ids, p


def align_predictions(ids, probabilities, target_ids):
    ids, target_ids = list(map(str, ids)), list(map(str, target_ids))
    p = check_probabilities(probabilities, len(ids))
    if len(set(ids)) != len(ids) or len(set(target_ids)) != len(target_ids) or set(ids) != set(target_ids):
        raise ValueError('Prediction IDs must match target IDs exactly')
    index = {value: position for position, value in enumerate(ids)}
    return p[[index[value] for value in target_ids]]


def score_predictions(y_true, y_pred, rows=None):
    y, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError('Nonempty aligned one-dimensional labels required')
    if not np.isin(y, range(3)).all() or not np.isin(pred, range(3)).all():
        raise ValueError('Unexpected numeric label')
    report = classification_report(y, pred, labels=list(range(3)), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    result = {
        'macro_f1': float(f1_score(y, pred, labels=list(range(3)), average='macro', zero_division=0)),
        'support': len(y), 'per_class': {name: report[name] for name in LABELS},
        'confusion_matrix': confusion_matrix(y, pred, labels=list(range(3))).tolist(),
        'label_order': list(LABELS),
    }
    if rows is not None:
        rows = rows.reset_index(drop=True)
        if len(rows) != len(y):
            raise ValueError('Metric rows not aligned')
        result['by_language'] = {}
        for language in sorted(rows.language.unique()):
            mask = (rows.language == language).to_numpy()
            result['by_language'][language] = score_predictions(y[mask], pred[mask])
        lengths = rows.text.str.len().to_numpy()
        masks = {'chars_le_256': lengths <= 256, 'chars_257_1024': (lengths > 256) & (lengths <= 1024),
                 'chars_gt_1024': lengths > 1024,
                 'template_marker': rows.text.str.contains(
                     r'TARGETLANGUAGE|LANGUE CIBLE|LINGUA DI DESTINAZIONE', case=False, regex=True).to_numpy()}
        if 'group_size' in rows:
            masks['near_duplicate_group'] = rows.group_size.to_numpy() > 1
        elif 'group' in rows:
            masks['near_duplicate_group_in_slice'] = rows.group.duplicated(keep=False).to_numpy()
        result['slices'] = {key: score_predictions(y[mask], pred[mask]) for key, mask in masks.items() if mask.any()}
    return result


def choose_policy(probabilities, y_true, min_gain=.002):
    """Coarse dev-only search. Audit labels must never be passed here."""
    if not probabilities or set(probabilities) - {'xlmr', 'mdeberta'}:
        raise ValueError('Expected one or both configured model families')
    p = {name: check_probabilities(value, len(y_true)) for name, value in probabilities.items()}
    score = lambda value: float(f1_score(y_true, np.argmax(value, axis=1), labels=[0, 1, 2],
                                        average='macro', zero_division=0))
    singles = {name: score(value) for name, value in p.items()}
    best = max(sorted(singles), key=singles.get)
    weights, combined, best_score = {best: 1.0}, p[best], singles[best]
    search = []
    if len(p) == 2:
        candidates = []
        for w in (.25, .5, .75):
            value = w * p['xlmr'] + (1 - w) * p['mdeberta']
            candidates.append((score(value), w, value))
        candidate_score, w, value = max(candidates, key=lambda item: item[0])
        search = [{'xlmr_weight': float(w), 'macro_f1': float(s)} for s, w, _ in candidates]
        if candidate_score >= best_score + min_gain:
            weights, combined, best_score = {'xlmr': w, 'mdeberta': 1 - w}, value, candidate_score
    factors = [1., 1., 1.]
    calibrated = []
    for jailbreak, injection in itertools.product((.8, 1., 1.25), repeat=2):
        candidate_factors = [1., jailbreak, injection]
        calibrated.append((score(combined * candidate_factors), candidate_factors))
    candidate_score, candidate_factors = max(calibrated, key=lambda item: item[0])
    if candidate_score >= best_score + min_gain:
        factors, best_score = candidate_factors, candidate_score
    return {'weights': weights, 'class_factors': factors, 'dev_macro_f1': best_score,
            'single_model_scores': singles, 'ensemble_search': search, 'min_gain': min_gain,
            'selection_split': 'dev', 'label_order': list(LABELS)}


def predict_labels(probabilities, policy):
    if tuple(policy['label_order']) != LABELS:
        raise ValueError('Policy label order mismatch')
    weights = policy['weights']
    if not weights or not np.isclose(sum(weights.values()), 1) or min(weights.values()) < 0:
        raise ValueError('Invalid ensemble weights')
    arrays = [check_probabilities(probabilities[name]) for name in weights]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError('Ensemble row counts mismatch')
    combined = sum(weight * array for weight, array in zip(weights.values(), arrays))
    factors = np.asarray(policy['class_factors'], dtype=float)
    if factors.shape != (3,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Invalid class factors')
    return np.argmax(combined * factors, axis=1)


def validate_submission(path, test_rows):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'label']:
            raise ValueError('Submission must have exactly header id,label')
        records = list(reader)
    ids = [row['id'] for row in records]
    target_ids = test_rows.id.astype(str).tolist()
    if len(ids) != len(target_ids) or len(set(ids)) != len(ids) or ids != target_ids:
        raise ValueError('Submission IDs must match test IDs once, in original order')
    if any(set(row) != {'id', 'label'} or row['label'] not in LABELS for row in records):
        raise ValueError('Submission contains an invalid label or extra CSV field')
    return {'valid': True, 'rows': len(ids), 'label_counts': {
        name: sum(row['label'] == name for row in records) for name in LABELS}}


def write_submission(path, test_rows, prediction_ids, labels):
    ids, labels = list(map(str, prediction_ids)), list(labels)
    if len(ids) != len(labels) or len(set(ids)) != len(ids) or set(ids) != set(test_rows.id.astype(str)):
        raise ValueError('Submission prediction IDs do not match test')
    values = {}
    for identifier, label in zip(ids, labels):
        if isinstance(label, (int, np.integer)) and label in range(3):
            label = LABELS[int(label)]
        if label not in LABELS:
            raise ValueError('Invalid submission label')
        values[identifier] = label
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label'])
        writer.writerows((identifier, values[identifier]) for identifier in test_rows.id.astype(str))
    validate_submission(temp, test_rows)
    os.replace(temp, path)
    return validate_submission(path, test_rows)


def error_examples(rows, predictions, limit=60):
    frame = rows.copy()
    frame['prediction'] = [LABELS[int(value)] for value in predictions]
    frame = frame[frame.label != frame.prediction]
    if frame.empty:
        return frame
    # Round-robin language and confusion pairs, rather than only the first errors.
    groups = [group for _, group in frame.groupby(['language', 'label', 'prediction'], sort=True)]
    selected = []
    for position in range(len(frame)):
        for group in groups:
            if position < len(group):
                selected.append(group.iloc[position])
                if len(selected) == limit:
                    return pd.DataFrame(selected)
    return pd.DataFrame(selected)
