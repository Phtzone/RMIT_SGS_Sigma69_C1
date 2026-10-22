import json
import time

import numpy as np
import pandas as pd
import pytest

from c1 import LABELS
from c1.metrics import save_predictions
from c1.pipeline import Budget, choose_candidate, select_models, resolve_runtime_paths, next_baseline_fits, predict_selection


def test_unlimited_budget_keeps_all_experiments_and_has_no_worker_deadline(tmp_path):
    from c1.pipeline import base_config
    budget = Budget(start=0., minutes=None)
    assert budget.can_finish(10**12, now=10**12, phase=150)
    assert next_baseline_fits([], budget, now=10**12)
    assert budget.worker_deadline(105, reserve=180, max_seconds=300) is None
    cfg = base_config('xlmr', 'test', tmp_path, tmp_path, budget.worker_deadline(220, reserve=120))
    assert cfg['deadline_unix'] is None
    json.dumps(cfg, allow_nan=False, default=str)


def test_unlimited_wait_allows_real_worker_to_finish(tmp_path):
    import subprocess
    import sys
    from c1.pipeline import wait_workers
    target = tmp_path / 'done.txt'
    class Job:
        def __init__(self):
            self.process = subprocess.Popen([sys.executable, '-c',
                'import sys,time; from pathlib import Path; time.sleep(.15); Path(sys.argv[1]).write_text("done")', str(target)])
        def stop(self):
            if self.process.poll() is None:
                self.process.terminate()
        def result(self):
            return {'done': target.read_text()} if target.exists() else None
    assert wait_workers([Job()], None) == [{'done': 'done'}]


def test_budget_reserves_export_and_obeys_competition_deadline():
    budget = Budget(start=1000., minutes=240., deadline=1000. + 7200.)
    assert budget.end == 8200.
    assert budget.phase_end(150) == 5500.
    assert budget.phase_end(220) == 7600.
    assert not budget.can_finish(300, now=7400., phase=220)
    assert budget.can_finish(100, now=7400., phase=220)


def test_budget_rejects_expired_or_impractical_run():
    with pytest.raises(ValueError):
        Budget(start=1000, minutes=240, deadline=999)
    with pytest.raises(ValueError):
        Budget(start=1000, minutes=0)


def test_baseline_tie_uses_measured_speed():
    a = {'dev_macro_f1': .9, 'train_seconds': 120}
    b = {'dev_macro_f1': .9005, 'train_seconds': 240}
    assert choose_candidate([a, b]) is a
    assert choose_candidate([a, {**b, 'dev_macro_f1': .902}])['train_seconds'] == 240


def test_failed_first_baseline_can_try_second_if_time_remains():
    budget = Budget(start=0., minutes=240.)
    assert next_baseline_fits([], budget, now=500.)
    assert not next_baseline_fits([], budget, now=6200.)


def test_model_selection_ignores_audit_labels_and_seals_policy(tmp_path):
    data = tmp_path / 'data'
    prepared = tmp_path / 'prepared'
    data.mkdir()
    prepared.mkdir()
    ids = [f'd{i}' for i in range(6)]
    rows = pd.DataFrame({'id': ids + ['audit'], 'text': ['x'] * 7,
                         'language': ['en'] * 7, 'label': list(LABELS) * 2 + ['BENIGN']})
    rows.to_csv(data / 'train.csv', index=False)
    pd.DataFrame({'id': rows.id, 'group': rows.id, 'fold': [3] * 6 + [0],
                  'split': ['dev'] * 6 + ['audit']}).to_csv(prepared / 'manifest.csv', index=False)
    p = np.eye(3)[[0, 1, 2, 0, 1, 2]] * .9 + .1 / 3
    pred = tmp_path / 'dev.npz'
    save_predictions(pred, ids[::-1], p[::-1])
    result = {'family': 'xlmr', 'run_id': 'a', 'dev_predictions': str(pred),
              'dev_macro_f1': 1., 'train_seconds': 1., 'checkpoint': str(tmp_path / 'model'),
              'best_epoch': 1., 'max_length': 256, 'weighted': False,
              'config': {'model_id': 'unused', 'model_revision': 'main'}}
    selection = select_models([result], data, prepared, tmp_path / 'out')
    assert selection['policy']['dev_macro_f1'] == 1.
    assert (tmp_path / 'out' / 'selection.sha256').exists()
    rows.loc[6, 'label'] = 'JAILBREAK'
    rows.to_csv(data / 'train.csv', index=False)
    second = select_models([result], data, prepared, tmp_path / 'other')
    assert selection['policy'] == second['policy']


def test_runtime_paths_must_not_write_inside_raw_data(tmp_path):
    raw = tmp_path / 'raw'
    with pytest.raises(ValueError):
        resolve_runtime_paths(raw, raw / 'artifacts')


def test_failed_prediction_cannot_reuse_old_valid_artifact(tmp_path):
    raw, out, broken = tmp_path / 'raw', tmp_path / 'output', tmp_path / 'broken'
    raw.mkdir(); broken.mkdir()
    (broken / 'config.json').write_text('invalid json')
    pd.DataFrame({'id': ['x'], 'text': ['hello'], 'language': ['en']}).to_csv(raw / 'test.csv', index=False)
    stale = out / 'predictions' / 'test' / 'xlmr' / 'probabilities.npz'
    save_predictions(stale, ['x'], [[1, 0, 0]])
    selection = {'models': {'xlmr': {'checkpoint': str(broken), 'max_length': 12}}}
    with pytest.raises(RuntimeError, match='prediction failed'):
        predict_selection(selection, raw, out, 'test', 1, time.time() + 45, device='cpu')
