import json

import numpy as np
import pandas as pd
import pytest

from c1 import LABELS
from c1.metrics import (
    align_predictions, choose_policy, load_predictions, predict_labels,
    save_predictions, score_predictions, validate_submission, write_submission,
)


def test_macro_f1_includes_missing_predicted_classes():
    report = score_predictions([0, 0, 1, 2], [0, 0, 0, 0])
    assert report['macro_f1'] == pytest.approx(2 / 9)
    assert report['per_class']['PROMPT_INJECTION']['recall'] == 0


def test_probability_files_align_by_id_and_reject_missing_rows(tmp_path):
    path = tmp_path / 'predictions.npz'
    save_predictions(path, ['b', 'a'], [[0, 1, 0], [1, 0, 0]])
    ids, probs = load_predictions(path)
    aligned = align_predictions(ids, probs, ['a', 'b'])
    np.testing.assert_array_equal(aligned, [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        align_predictions(ids, probs, ['a', 'c'])


@pytest.mark.parametrize('probabilities', [
    [[float('nan'), 0, 1]], [[0.1, 0.2, 0.3]], [[1, 0]], [[1.1, -0.1, 0]],
])
def test_invalid_probabilities_are_rejected(tmp_path, probabilities):
    with pytest.raises(ValueError):
        save_predictions(tmp_path / 'bad.npz', ['a'], probabilities)


def test_wrong_label_order_is_rejected(tmp_path):
    path = tmp_path / 'wrong.npz'
    np.savez(path, ids=['a'], probabilities=[[1., 0., 0.]], label_order=LABELS[::-1])
    with pytest.raises(ValueError, match='label'):
        load_predictions(path)


def test_submission_reorders_by_test_ids_and_roundtrips(tmp_path):
    path = tmp_path / 'submission.csv'
    test = pd.DataFrame({'id': ['a', 'b', 'c']})
    write_submission(path, test, ['c', 'a', 'b'], [2, 0, 1])
    report = validate_submission(path, test)
    assert report['rows'] == 3
    assert path.read_text().splitlines() == [
        'id,label', 'a,BENIGN', 'b,JAILBREAK', 'c,PROMPT_INJECTION',
    ]


@pytest.mark.parametrize('text', [
    'id,label\na,BENIGN\na,JAILBREAK\n',
    'id,label\na,BENIGN\nc,JAILBREAK\n',
    'id,label\nb,BENIGN\na,JAILBREAK\n',
    'id,label\na,benign\nb,JAILBREAK\n',
    'index,id,label\n0,a,BENIGN\n1,b,JAILBREAK\n',
])
def test_submission_rejects_duplicate_missing_order_label_or_index(tmp_path, text):
    path = tmp_path / 'submission.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        validate_submission(path, pd.DataFrame({'id': ['a', 'b']}))


def test_policy_keeps_perfect_single_model_and_no_calibration():
    y = np.array([0, 1, 2] * 10)
    good = np.eye(3)[y] * .9 + .1 / 3
    bad = np.eye(3)[(y + 1) % 3] * .9 + .1 / 3
    policy = choose_policy({'xlmr': good, 'mdeberta': bad}, y)
    assert policy['weights'] == {'xlmr': 1.0}
    assert policy['class_factors'] == [1., 1., 1.]
    assert policy['dev_macro_f1'] == 1
    assert predict_labels({'xlmr': good}, policy).tolist() == y.tolist()
    json.dumps(policy)


def test_policy_recovers_complementary_errors_with_ensemble():
    y = np.array([0, 1, 2] * 10)
    first = np.eye(3)[y] * .8 + .2 / 3
    second = first.copy()
    first[0] = [.44, .46, .10]
    second[1] = [.46, .44, .10]
    policy = choose_policy({'xlmr': first, 'mdeberta': second}, y)
    assert len(policy['weights']) == 2
    assert policy['dev_macro_f1'] == 1


def test_score_slices_include_support_and_do_not_require_all_classes():
    rows = pd.DataFrame({'language': ['en', 'pt', 'pt'], 'text': ['a', 'b', 'c'],
                         'label': list(LABELS), 'group': ['a', 'b', 'b']})
    report = score_predictions([0, 1, 2], [0, 1, 2], rows)
    assert report['by_language']['en']['support'] == 1
    assert report['by_language']['en']['macro_f1'] == pytest.approx(1 / 3)
