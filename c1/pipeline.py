"""Subprocess orchestration and dev-only selection, with optional time limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

from c1 import LABELS, LABEL_TO_ID
from c1.metrics import (
    align_predictions, choose_policy, error_examples, load_predictions,
    predict_labels, score_predictions, write_json, write_submission,
)

MODELS = {'xlmr': 'FacebookAI/xlm-roberta-base', 'mdeberta': 'microsoft/mdeberta-v3-base'}


@dataclass
class Budget:
    start: float
    minutes: float | None = 240.
    deadline: float | None = None

    def __post_init__(self):
        if (self.minutes is not None and self.minutes <= 0) or (self.deadline is not None and self.deadline <= self.start):
            raise ValueError('Budget/deadline leaves no execution time')
        self.end = min(self.start + self.minutes * 60 if self.minutes is not None else float('inf'),
                       self.deadline or float('inf'))

    def phase_end(self, standard_minute):
        return self.start + (self.end - self.start) * standard_minute / 240.

    def can_finish(self, estimate_seconds, now=None, phase=150):
        return (time.time() if now is None else now) + 1.2 * estimate_seconds < self.phase_end(phase)

    def worker_deadline(self, phase=240, reserve=0, max_seconds=None):
        if not np.isfinite(self.end):
            return None
        end = self.phase_end(phase) - reserve
        return min(end, time.time() + max_seconds) if max_seconds is not None else end


def resolve_runtime_paths(data_dir, output_dir):
    raw, out = Path(data_dir).resolve(), Path(output_dir).resolve()
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError('Output must be separate from the original data directory')
    return raw, out


def choose_candidate(results):
    valid = [r for r in results if r.get('dev_macro_f1') is not None and np.isfinite(r['dev_macro_f1'])]
    if not valid:
        raise ValueError('No evaluated checkpoint available')
    high = max(r['dev_macro_f1'] for r in valid)
    tied = [r for r in valid if high - r['dev_macro_f1'] < .001]
    return min(tied, key=lambda r: r.get('wall_seconds', r.get('train_seconds', float('inf'))))


def next_baseline_fits(results, budget, now=None):
    # If the first model failed before evaluation, still allow the other family.
    estimate = (results[0].get('wall_seconds', results[0]['train_seconds']) if results else 300.)
    return budget.can_finish(estimate, now=now, phase=105)


def split_rows(data_dir, prepared_dir, split):
    frame = pd.read_csv(Path(data_dir) / 'train.csv', dtype=str, keep_default_na=False)
    manifest = pd.read_csv(Path(prepared_dir) / 'manifest.csv', dtype={'id': str, 'group': str})
    if frame.id.duplicated().any() or manifest.id.duplicated().any() or set(frame.id) != set(manifest.id):
        raise ValueError('Prepared split does not match original train IDs')
    manifest['group_size'] = manifest.groupby('group')['id'].transform('size')
    # Original CSV order is the stable order for worker outputs and all metrics.
    joined = frame.merge(manifest, on='id', how='left', validate='one_to_one', sort=False)
    return joined[joined.split == split].reset_index(drop=True)


def select_models(results, data_dir, prepared_dir, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'audit_metrics.json').exists():
        raise ValueError('Cannot reselect after audit; start a fresh independent run')
    rows = split_rows(data_dir, prepared_dir, 'dev')
    if rows.empty:
        raise ValueError('No development rows')
    chosen = {}
    for family in MODELS:
        candidates = [r for r in results if r.get('family') == family and r.get('dev_predictions')]
        if candidates:
            # Within a family, respect the best measured score; speed breaks exact ties.
            chosen[family] = max(candidates, key=lambda r: (r['dev_macro_f1'], -r.get('train_seconds', 0)))
    arrays = {}
    for family, result in chosen.items():
        ids, p = load_predictions(result['dev_predictions'])
        arrays[family] = align_predictions(ids, p, rows.id)
    labels = rows.label.map(LABEL_TO_ID).to_numpy()
    policy = choose_policy(arrays, labels)
    chosen = {family: chosen[family] for family in policy['weights']}
    selection = {'policy': policy, 'models': chosen, 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                 'audit_used_for_selection': False, 'training_scope': 'split train only'}
    write_json(out / 'selection.json', selection)
    digest = hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest()
    (out / 'selection.sha256').write_text(digest, encoding='ascii')
    pred = predict_labels(arrays, policy)
    write_json(out / 'dev_metrics.json', score_predictions(labels, pred, rows))
    error_examples(rows, pred).to_csv(out / 'dev_errors.csv', index=False)
    write_json(out / 'experiments.json', results)
    return selection


def verify_selection(output_dir):
    out = Path(output_dir)
    expected = (out / 'selection.sha256').read_text(encoding='ascii').strip()
    if hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest() != expected:
        raise ValueError('Locked model selection has changed')
    return json.loads((out / 'selection.json').read_text(encoding='utf-8'))


class Worker:
    def __init__(self, config, gpu=0, predict=False):
        self.config = config
        self.run_dir = Path(config.get('run_dir', Path(config['output']).parent if 'output' in config else '.'))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.run_dir / ('predict_config.json' if predict else 'config.json')
        write_json(self.config_path, config)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
        self.log = (self.run_dir / ('predict.log' if predict else 'train.log')).open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'c1.training', '--predict-config' if predict else '--config', str(self.config_path)],
            stdout=self.log, stderr=subprocess.STDOUT, env=env,
        )
        print(f'Started {config.get("run_id", "prediction")} on GPU {gpu}; log: {self.log.name}', flush=True)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()

    def result(self):
        path = self.run_dir / 'result.json'
        if not path.exists():
            return None
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            checkpoint = Path(result['checkpoint'])
            if not (checkpoint / 'config.json').exists():
                return None
            return result
        except (ValueError, KeyError, OSError):
            return None


def wait_workers(workers, deadline, callback=None):
    next_report = 0.
    try:
        while any(worker.process.poll() is None for worker in workers) and (deadline is None or time.time() < deadline):
            if time.time() >= next_report:
                for worker in workers:
                    report = {'run': getattr(worker, 'config', {}).get('run_id', 'prediction'),
                              'process_status': worker.process.poll()}
                    folder = getattr(worker, 'run_dir', None)
                    if folder is not None:
                        for filename in ('benchmark.json', 'result.json'):
                            path = folder / filename
                            if path.exists():
                                try:
                                    values = json.loads(path.read_text(encoding='utf-8'))
                                    report.update({k: values[k] for k in ('optimizer_steps', 'seconds_per_step',
                                        'dev_macro_f1', 'best_epoch', 'completed') if k in values})
                                except (ValueError, OSError):
                                    pass
                    print('Progress: ' + json.dumps(report), flush=True)
                next_report = time.time() + 60
            if callback:
                callback(workers)
            time.sleep(2. if deadline is None else min(2., max(0., deadline - time.time())))
    finally:
        for worker in workers:
            worker.stop()
    return [r for worker in workers if (r := worker.result()) is not None]


def base_config(family, run_id, raw, out, deadline, **overrides):
    config = {'run_id': run_id, 'family': family, 'model_id': MODELS[family], 'model_revision': 'main',
              'data_dir': str(raw), 'prepared_dir': str(out / 'prepared'),
              'run_dir': str(out / 'runs' / run_id), 'max_length': 256, 'epochs': 3., 'lr': 2e-5,
              'batch_size': 8, 'grad_accum': 4, 'weighted': False, 'seed': 42,
              'deadline_unix': deadline, 'full_train': False, 'device': 'cuda'}
    config.update(overrides)
    return config


def predict_selection(selection, raw, out, split, gpu_count, deadline, models=None, device='cuda'):
    models = models or selection['models']
    items, predictions = list(models.items()), {}
    for offset in range(0, len(items), gpu_count):
        workers = []
        current = items[offset:offset + gpu_count]
        for gpu, (family, result) in enumerate(current):
            pred_dir = out / 'predictions' / split / family / uuid.uuid4().hex
            config = {'checkpoint': result['checkpoint'], 'data_dir': str(raw),
                      'prepared_dir': str(out / 'prepared'), 'split': split,
                      'max_length': result['max_length'], 'batch_size': 16, 'device': device,
                      'output': str(pred_dir / 'probabilities.npz'), 'run_dir': str(pred_dir)}
            workers.append(Worker(config, gpu=gpu, predict=True))
        wait_workers(workers, deadline)
        for (family, _), worker in zip(current, workers):
            path = Path(worker.config['output'])
            if worker.process.returncode != 0 or not path.exists():
                raise RuntimeError(f'{split} prediction failed; inspect {worker.run_dir / "predict.log"}')
            predictions[family] = load_predictions(path)
    return predictions


def export_selection(selection, raw, out, gpu_count, deadline, models=None, name='submission.csv', device='cuda'):
    from c1.data import load_competition
    _, test, _ = load_competition(raw)
    outputs = predict_selection(selection, raw, out, 'test', gpu_count, deadline, models=models, device=device)
    aligned = {family: align_predictions(ids, p, test.id) for family, (ids, p) in outputs.items()}
    pred = predict_labels(aligned, selection['policy'])
    return write_submission(out / name, test, test.id, pred)


def audit_selection(raw, out, gpu_count, deadline, device='cuda'):
    selection = verify_selection(out)
    if (out / 'audit_metrics.json').exists():
        raise ValueError('Audit has already been evaluated; do not tune on it')
    rows = split_rows(raw, out / 'prepared', 'audit')
    outputs = predict_selection(selection, raw, out, 'audit', gpu_count, deadline, device=device)
    aligned = {family: align_predictions(ids, p, rows.id) for family, (ids, p) in outputs.items()}
    report = score_predictions(rows.label.map(LABEL_TO_ID), predict_labels(aligned, selection['policy']), rows)
    report['scope'] = 'Frozen split-trained models, before full-data refit. Not a score for the refitted models.'
    write_json(out / 'audit_metrics.json', report)
    return report


def gpu_inventory():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Enable Kaggle GPU T4 x2; prepare/tests can run on CPU.')
    return [{'index': i, 'name': torch.cuda.get_device_name(i),
             'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
            for i in range(min(2, torch.cuda.device_count()))]


def run_pipeline(data_dir, output_dir, budget_minutes=None, competition_deadline=None):
    from c1.data import assert_hashes, compute_hashes, prepare
    raw, out = resolve_runtime_paths(data_dir, output_dir)
    if (out / 'run_started.json').exists():
        raise ValueError('Use a fresh output directory for each run; existing audit must remain sealed')
    deadline = datetime.fromisoformat(competition_deadline).timestamp() if competition_deadline else None
    if competition_deadline and datetime.fromisoformat(competition_deadline).tzinfo is None:
        raise ValueError('Competition deadline must include timezone, for example +07:00')
    budget = Budget(time.time(), budget_minutes, deadline)
    devices = gpu_inventory()
    out.mkdir(parents=True, exist_ok=True)
    hashes = compute_hashes(raw)
    write_json(out / 'run_started.json', {'started': budget.start, 'end': budget.worker_deadline(),
                                         'time_limit_enabled': np.isfinite(budget.end).item(),
                                         'gpu': devices, 'raw_hashes': hashes})
    print('Pipeline started; time limits: ' + ('enabled' if np.isfinite(budget.end) else 'disabled'), flush=True)
    (out / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    results, events = [], []
    try:
        report = prepare(raw, out / 'prepared')
        write_json(out / 'preparation_summary.json', report)
        if time.time() >= budget.phase_end(105) - 300:
            raise RuntimeError('Preparation exhausted baseline budget')
        fallback_done = False

        def fallback_callback(active):
            nonlocal fallback_done
            if fallback_done:
                return
            # A finished worker has a stable checkpoint and releases its GPU for inference.
            for worker in active:
                if worker.process.poll() is None:
                    continue
                result = worker.result()
                if result and not result.get('full_train'):
                    family = result['family']
                    gpu = active.index(worker)
                    # Inference is assigned to the released physical GPU, not GPU 0 unconditionally.
                    pred_dir = out / 'fallback' / family
                    cfg = {'checkpoint': result['checkpoint'], 'data_dir': str(raw),
                           'prepared_dir': str(out / 'prepared'), 'split': 'test',
                           'max_length': result['max_length'], 'batch_size': 16, 'device': 'cuda',
                           'output': str(pred_dir / 'probabilities.npz'), 'run_dir': str(pred_dir)}
                    predictor = Worker(cfg, gpu=gpu, predict=True)
                    wait_workers([predictor], budget.worker_deadline(105, max_seconds=300))
                    if predictor.process.returncode == 0 and Path(cfg['output']).exists():
                        from c1.data import load_competition
                        _, test, _ = load_competition(raw)
                        ids, p = load_predictions(cfg['output'])
                        p = align_predictions(ids, p, test.id)
                        write_submission(out / 'submission_fallback.csv', test, test.id, np.argmax(p, axis=1))
                        fallback_done = True
                    return

        # Two independent jobs on two devices; single-device mode starts XLM-R first.
        families = list(MODELS)
        for offset in range(0, len(families), len(devices)):
            if offset and not next_baseline_fits(results, budget):
                events.append('Skipped second baseline to protect later phases')
                break
            active = [Worker(base_config(family, family + '_baseline', raw, out,
                                          budget.worker_deadline(105, reserve=180)), gpu=gpu)
                      for gpu, family in enumerate(families[offset:offset + len(devices)])]
            results.extend(wait_workers(active, budget.worker_deadline(105), fallback_callback))
            fallback_callback(active)
        champion = choose_candidate(results)
        events.append(f'Baseline champion: {champion["run_id"]}')
        refinements = [(champion['family'] + '_long', {'max_length': 512}, 2.),
                       (champion['family'] + '_weighted', {'weighted': True}, 1.)]
        active = []
        for run_id, overrides, multiplier in refinements:
            estimate = champion.get('wall_seconds', champion['train_seconds']) * multiplier
            if not budget.can_finish(estimate, phase=150):
                events.append(f'Skipped {run_id}: projected runtime exceeds search budget')
                continue
            if len(active) == len(devices):
                results.extend(wait_workers(active, budget.worker_deadline(150)))
                active = []
                if not budget.can_finish(estimate, phase=150):
                    continue
            config = base_config(champion['family'], run_id, raw, out,
                                 budget.worker_deadline(150, reserve=120),
                                 model_revision=champion.get('model_revision', 'main'), **overrides)
            active.append(Worker(config, gpu=len(active)))
        if active:
            results.extend(wait_workers(active, budget.worker_deadline(150)))
        selection = select_models(results, raw, out / 'prepared', out)
        audit_selection(raw, out, len(devices), budget.worker_deadline(165))
        # Ensure a frozen, evaluated submission exists before risking the remaining time on refits.
        export_selection(selection, raw, out, len(devices), budget.worker_deadline(175), name='submission_fallback.csv')
        refits, refit_jobs = {}, []
        for family, selected in selection['models'].items():
            estimate = selected.get('wall_seconds', selected['train_seconds']) / .7
            if not budget.can_finish(estimate, phase=220):
                events.append(f'Skipped {family} full refit: insufficient budget')
                continue
            config = base_config(family, family + '_full', raw, out, budget.worker_deadline(220, reserve=120),
                                 epochs=selected['best_epoch'], max_length=selected['max_length'],
                                 weighted=selected['weighted'], full_train=True,
                                 model_revision=selected.get('model_revision', 'main'))
            refit_jobs.append(Worker(config, gpu=len(refit_jobs) % len(devices)))
            if len(refit_jobs) == len(devices):
                for result in wait_workers(refit_jobs, budget.worker_deadline(220)):
                    if result.get('full_train') and result.get('completed', False):
                        refits[result['family']] = result
                refit_jobs = []
        if refit_jobs:
            for result in wait_workers(refit_jobs, budget.worker_deadline(220)):
                if result.get('full_train') and result.get('completed', False):
                    refits[result['family']] = result
        # Atomic policy: use the entire refitted ensemble only when all its components completed.
        final_models = refits if set(refits) == set(selection['models']) else selection['models']
        final_report = export_selection(selection, raw, out, len(devices), budget.worker_deadline(),
                                        models=final_models)
        write_json(out / 'final_models.json', {'models': final_models, 'policy': selection['policy'],
                   'full_data_refit': final_models is refits, 'submission': final_report})
        write_run_report(out, results, events)
        return {'output_dir': str(out), 'submission': final_report,
                'full_data_refit': final_models is refits}
    except Exception as error:
        # Preserve a real evaluated fallback if later training/inference exhausts time.
        # The raw-data integrity guard in finally still takes precedence over this fallback.
        from c1.data import load_competition
        from c1.metrics import validate_submission
        fallback = out / 'submission_fallback.csv'
        write_json(out / 'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        if not fallback.exists():
            raise
        _, test, _ = load_competition(raw)
        final_report = validate_submission(fallback, test)
        shutil.copyfile(fallback, out / 'submission.csv')
        events.append(f'Used evaluated fallback after {type(error).__name__}: {error}')
        if (out / 'selection.json').exists():
            write_run_report(out, results, events)
        return {'output_dir': str(out), 'submission': final_report,
                'full_data_refit': False, 'status': 'fallback', 'reason': str(error)}
    finally:
        assert_hashes(raw, hashes)
        write_json(out / 'events.json', events)


def write_run_report(out, results, events):
    selection = verify_selection(out)
    audit_path = out / 'audit_metrics.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
    lines = ['# C1 experiment report', '', 'Raw CSVs were preserved. Selection used dev labels only.', '',
             '| Run | Family | Max tokens | Weighted | Dev Macro F1 | Best epoch | Seconds |',
             '|---|---|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["run_id"]} | {r["family"]} | {r["max_length"]} | {r["weighted"]} | '
                     f'{r["dev_macro_f1"]:.6f} | {r["best_epoch"]:.3f} | {r["train_seconds"]:.1f} |')
    lines += ['', f'Chosen dev Macro F1: {selection["policy"]["dev_macro_f1"]:.6f}',
              f'Audit Macro F1 before full-data refit: {audit["macro_f1"]:.6f}' if audit else 'Audit unavailable.',
              '', 'Audit scores are not scores for full-data-refitted models.', '', '## Execution decisions', '']
    lines += ['- ' + event for event in events]
    (out / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
