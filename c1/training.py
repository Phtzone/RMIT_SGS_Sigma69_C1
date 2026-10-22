"""Single-device transformer training and checkpoint inference.

The worker never reads audit examples during model selection. All probability
artifacts use the shared, explicit three-class column order.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np

from . import LABELS, LABEL_TO_ID


def head_tail_encode(tokenizer, text, max_length):
    budget = int(max_length) - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 0:
        raise ValueError('max_length is smaller than the special-token budget')
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > budget:
        head = (budget + 1) // 2
        tail = budget // 2
        ids = ids[:head] + (ids[-tail:] if tail else [])
    return tokenizer.prepare_for_model(ids, add_special_tokens=True,
        truncation=False, return_attention_mask=True)


def class_weights(labels):
    counts = np.bincount(labels, minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError('Training data must contain all three labels')
    return np.sqrt(counts.max() / counts)


def accumulation_windows(length, accumulation):
    if accumulation < 1:
        raise ValueError('grad_accum must be positive')
    for start in range(0, length, accumulation):
        yield start, min(length, start + accumulation)


def epoch_windows(epoch_batches, remaining, accumulation):
    """Flush accumulation at the half-epoch and final boundaries."""
    half = min(math.ceil(epoch_batches / 2), remaining)
    yield from accumulation_windows(half, accumulation)
    for start, end in accumulation_windows(remaining - half, accumulation):
        yield half + start, half + end


def _optimizer_step(model, optimizer, scaler, scheduler, amp, overflow_state,
                    max_overflows=8, gradient_multiplier=1.0):
    """Let GradScaler skip overflowed updates before advancing the scheduler."""
    import torch
    scaler.unscale_(optimizer)
    if gradient_multiplier != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_multiplier)
    # AMP already recorded nonfinite gradients in unscale_. It must reach
    # step/update to skip the optimizer and lower the scale on such a batch.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = amp and scaler.get_scale() < old_scale
    if skipped:
        overflow_state['consecutive_overflows'] += 1
        if overflow_state['consecutive_overflows'] >= max_overflows:
            raise FloatingPointError(f'FP16 gradient overflow persisted for {max_overflows} consecutive updates; previous valid checkpoint retained')
        return False
    overflow_state['consecutive_overflows'] = 0
    scheduler.step()
    return True


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _rows(data_dir, name):
    with (Path(data_dir) / (name + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _split_rows(config, split):
    if split == 'test':
        return _rows(config['data_dir'], 'test')
    with (Path(config['prepared_dir']) / 'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    wanted = {row['id'] for row in manifest if row['split'] == split}
    rows = [row for row in _rows(config['data_dir'], 'train') if row['id'] in wanted]
    if len(rows) != len(wanted):
        raise ValueError('Manifest IDs missing or repeated in train.csv')
    return rows


def _check_mapping(model):
    mapping = {int(k): v for k, v in model.config.id2label.items()}
    if mapping != dict(enumerate(LABELS)):
        raise ValueError('Checkpoint label order does not match LABELS')


def _probability_batch(model, tokenizer, texts, max_length, device, features=None):
    import torch
    features = features if features is not None else [head_tail_encode(tokenizer, x, max_length) for x in texts]
    batch = tokenizer.pad(features, padding=True, return_tensors='pt')
    amp = str(device).startswith('cuda')
    with torch.autocast(device_type='cuda' if amp else 'cpu', dtype=torch.float16, enabled=amp):
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
    return torch.softmax(logits.float(), -1).cpu().numpy()


def _probabilities(model, tokenizer, texts, max_length, batch_size, device, encoded=None):
    import torch
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    arrays = []
    with torch.inference_mode():
        start = 0
        while start < len(texts):
            failed = False
            try:
                probabilities = _probability_batch(model, tokenizer, texts[start:start + batch_size],
                    max_length, device, None if encoded is None else encoded[start:start + batch_size])
            except torch.cuda.OutOfMemoryError as error:
                if batch_size == 1:
                    raise
                error.__traceback__ = None
                failed = True
            if failed:
                # The exception and its failed forward frame are gone before cleanup.
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            arrays.append(probabilities)
            start += batch_size
    probabilities = np.concatenate(arrays) if arrays else np.empty((0, 3), dtype=np.float32)
    if probabilities.shape != (len(texts), 3) or not np.isfinite(probabilities).all():
        raise FloatingPointError('Model produced invalid probabilities; no scores saved')
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise FloatingPointError('Model probabilities are not normalized')
    return probabilities


def predict_checkpoint(checkpoint, texts: list[str], max_length, batch_size=16, device='cuda'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    _check_mapping(model)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Deliberately smoke-test the reloaded checkpoint before the full workload.
    if texts:
        _probabilities(model, tokenizer, texts[:1], max_length, 1, device)
    return _probabilities(model, tokenizer, texts, max_length, batch_size, device)


def _save_predictions(path, rows, probabilities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array([r['id'] for r in rows], dtype=str),
        probabilities=probabilities, label_order=np.array(LABELS, dtype=str))


def run_experiment(config: dict) -> dict:
    """Train from scratch, retrying CUDA OOM with smaller microbatches."""
    wall_start = time.perf_counter()
    import torch
    runtime = dict(wall_start=wall_start, train_seconds=0.0, optimizer_steps=0, overflow_steps=0)
    config = dict(config)
    defaults = dict(max_length=256, epochs=3.0, lr=2e-5, batch_size=8,
        grad_accum=4, weighted=False, seed=42, full_train=False, device='cuda', model_revision='main')
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config['family'] not in ('xlmr', 'mdeberta'):
        raise ValueError('family must be xlmr or mdeberta')
    if not 0 < float(config['epochs']) <= 3:
        raise ValueError('epochs must be in (0, 3]')
    if int(config['batch_size']) < 1 or int(config['grad_accum']) < 1:
        raise ValueError('batch_size and grad_accum must be positive')
    run_dir = Path(config['run_dir']).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config['run_dir'] = str(run_dir)
    effective = int(config['batch_size']) * int(config['grad_accum'])
    batch_size = int(config['batch_size'])
    while True:
        try:
            result = _train(config, batch_size, math.ceil(effective / batch_size), runtime)
            result['wall_seconds'] = time.perf_counter() - wall_start
            _json(run_dir / 'result.json', result)
            return result
        except torch.cuda.OutOfMemoryError as error:
            if batch_size == 1:
                # A previously evaluated checkpoint remains usable by the scheduler.
                result_path = run_dir / 'result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                    result.update(stopped_reason='cuda_oom', completed=False,
                        wall_seconds=time.perf_counter() - wall_start,
                        train_seconds=runtime['train_seconds'])
                    _json(result_path, result)
                    return result
                raise
            error.__traceback__ = None
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)


def _train(config, batch_size, accumulation, runtime):
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    device = config['device']
    model = AutoModelForSequenceClassification.from_pretrained(config['model_id'],
        revision=config['model_revision'], num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True).to(device)
    revision = getattr(model.config, '_commit_hash', None) or config['model_revision']
    config['model_revision'] = revision
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=revision)
    _check_mapping(model)
    if batch_size < config['batch_size'] and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    train = _rows(config['data_dir'], 'train') if config['full_train'] else _split_rows(config, 'train')
    dev = [] if config['full_train'] else _split_rows(config, 'dev')
    if not train or (not config['full_train'] and not dev):
        raise ValueError('Training and development partitions must be nonempty')
    labels = [LABEL_TO_ID[row['label']] for row in train]
    weights = class_weights(labels)  # Validate all labels even for unweighted runs.
    encoded = [head_tail_encode(tokenizer, row['text'], config['max_length']) for row in train]
    dev_encoded = [head_tail_encode(tokenizer, row['text'], config['max_length']) for row in dev]
    statistics = {'scope': 'selection train and dev splits only; raw content tokens before truncation and special tokens'}
    for split in ('train', 'dev'):
        # Full refits still report only the selection partitions, never audit.
        rows = _split_rows(config, split) if config['full_train'] else (train if split == 'train' else dev)
        lengths = np.array([len(tokenizer.encode(row['text'], add_special_tokens=False)) for row in rows])
        statistics[split] = dict(count=len(rows), quantiles={str(q): float(np.quantile(lengths, q))
            for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)} if len(rows) else {},
            over256=int((lengths > 256).sum()), over512=int((lengths > 512).sum()))
    _json(Path(config['run_dir']) / 'token_statistics.json', statistics)
    batches_per_epoch = math.ceil(len(train) / batch_size)
    steps_per_epoch = len(list(epoch_windows(batches_per_epoch, batches_per_epoch, accumulation)))
    total_batches = math.ceil(batches_per_epoch * config['epochs'])
    total_steps = sum(len(list(epoch_windows(batches_per_epoch, min(batches_per_epoch, total_batches - i), accumulation)))
        for i in range(0, total_batches, batches_per_epoch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    amp = str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device) if config['weighted'] else None
    run_dir = Path(config['run_dir'])
    result = dict(run_id=config['run_id'], family=config['family'], checkpoint=str(run_dir / ('final' if config['full_train'] else 'best')),
        max_length=config['max_length'], weighted=config['weighted'], best_epoch=None,
        dev_macro_f1=None, train_seconds=runtime['train_seconds'], wall_seconds=time.perf_counter() - runtime['wall_start'],
        completed=False, seconds_per_step=None, dev_predictions=None,
        model_revision=revision, full_train=config['full_train'], config=config,
        microbatch_size=batch_size, grad_accum=accumulation)
    previous = run_dir / 'result.json'
    if previous.exists() and not config['full_train']:
        prior = json.loads(previous.read_text(encoding='utf-8'))
        if prior.get('dev_macro_f1') is not None:
            result = prior
            result.update(config=config, microbatch_size=batch_size, grad_accum=accumulation)
    train_seconds = runtime['train_seconds']
    optimizer_steps = runtime['optimizer_steps']
    done = 0
    evaluated_at = -1
    stopped = False
    overflow_state = {'consecutive_overflows': 0}

    def benchmark():
        seconds = train_seconds / max(optimizer_steps, 1)
        _json(run_dir / 'benchmark.json', dict(train_seconds=train_seconds,
            wall_seconds=time.perf_counter() - runtime['wall_start'],
            seconds_per_step=seconds, optimizer_steps=optimizer_steps, steps_per_epoch=steps_per_epoch,
            overflow_steps=runtime['overflow_steps'], scheduler_steps=schedule.last_epoch,
            estimated_total_seconds=seconds * total_steps,
            duration_scope='training forward/backward/optimizer only; excludes tokenization, evaluation and checkpoint I/O'))
        return seconds

    def evaluate(epoch):
        nonlocal evaluated_at
        probabilities = _probabilities(model, tokenizer, [r['text'] for r in dev], config['max_length'], batch_size, device, dev_encoded)
        score = float(f1_score([LABEL_TO_ID[r['label']] for r in dev], probabilities.argmax(1),
            labels=[0, 1, 2], average='macro', zero_division=0))
        # A result pointer only refers to an immutable, fully persisted generation.
        # Interrupted writes therefore cannot alter the previously published best.
        checkpoint = run_dir / ('checkpoint-' + uuid.uuid4().hex)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        published = dict(result)
        if result['dev_macro_f1'] is None or score > result['dev_macro_f1']:
            prediction_path = checkpoint / 'dev_predictions.npz'
            _save_predictions(prediction_path, dev, probabilities)
            if result['dev_macro_f1'] is not None:
                published['previous_checkpoint'] = result['checkpoint']
            published.update(checkpoint=str(checkpoint), dev_macro_f1=score,
                best_epoch=float(epoch), dev_predictions=str(prediction_path))
        published.update(latest_checkpoint=str(checkpoint), train_seconds=train_seconds, seconds_per_step=benchmark(),
            wall_seconds=time.perf_counter() - runtime['wall_start'], completed=done >= total_batches,
            optimizer_steps=optimizer_steps, scheduler_steps=schedule.last_epoch, overflow_steps=runtime['overflow_steps'])
        _json(run_dir / 'result.json', published)
        result.clear()
        result.update(published)
        # Keep current best, previous best and latest; prune only after publication.
        keep = {Path(published[key]).resolve() for key in ('checkpoint', 'previous_checkpoint', 'latest_checkpoint') if published.get(key)}
        for candidate in run_dir.glob('checkpoint-*'):
            resolved = candidate.resolve()
            if candidate.is_dir() and not candidate.is_symlink() and resolved.parent == run_dir.resolve() and resolved not in keep:
                try:
                    shutil.rmtree(resolved)
                except OSError:
                    pass  # A valid published checkpoint is more valuable than cleanup.
        evaluated_at = done

    for epoch_index in range(math.ceil(config['epochs'])):
        order = np.random.permutation(len(train))
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        remaining = min(len(batches), total_batches - done)
        if remaining <= 0:
            break
        half_boundary = math.ceil(len(batches) / 2)
        for start, end in epoch_windows(batches_per_epoch, remaining, accumulation):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            window = batches[start:end]
            denominator = sum(float(weights[np.array(labels)[idx]].sum()) if config['weighted'] else len(idx) for idx in window)
            elapsed_start = time.perf_counter()
            processed_denominator = 0.0
            for idx in window:
                batch = tokenizer.pad([encoded[int(i)] for i in idx], padding=True, return_tensors='pt')
                targets = torch.tensor([labels[int(i)] for i in idx], device=device)
                with torch.autocast(device_type='cuda' if amp else 'cpu', dtype=torch.float16, enabled=amp):
                    logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), targets, weight=weight_tensor, reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss (including FP16 overflow); previous valid checkpoint retained')
                scaler.scale(loss / denominator).backward()
                processed_denominator += float(weights[np.array(labels)[idx]].sum()) if config['weighted'] else len(idx)
                done += 1
                if config.get('deadline_unix') is not None and time.time() >= config['deadline_unix']:
                    stopped = True
                    break
            # Flush an incomplete deadline window with the same mean-loss normalization.
            advanced = _optimizer_step(model, optimizer, scaler, schedule, amp, overflow_state,
                gradient_multiplier=denominator / processed_denominator)
            if amp:
                torch.cuda.synchronize()
            train_seconds += time.perf_counter() - elapsed_start
            optimizer_steps += int(advanced)
            runtime['overflow_steps'] += int(not advanced)
            runtime.update(train_seconds=train_seconds, optimizer_steps=optimizer_steps)
            if advanced and optimizer_steps == 100:
                benchmark()
            local_done = done - epoch_index * batches_per_epoch
            if not config['full_train'] and (local_done >= half_boundary and evaluated_at < epoch_index * batches_per_epoch + half_boundary or local_done == remaining or stopped):
                evaluate(min(config['epochs'], epoch_index + local_done / batches_per_epoch))
            if stopped:
                break
        if stopped:
            break
    if config['full_train']:
        model.save_pretrained(result['checkpoint'])
        tokenizer.save_pretrained(result['checkpoint'])
        result['best_epoch'] = min(config['epochs'], done / batches_per_epoch)
    elif evaluated_at != done:
        evaluate(min(config['epochs'], done / batches_per_epoch))
    result.update(train_seconds=train_seconds, seconds_per_step=benchmark(),
        wall_seconds=time.perf_counter() - runtime['wall_start'], completed=done >= total_batches,
        optimizer_steps=optimizer_steps, scheduler_steps=schedule.last_epoch, overflow_steps=runtime['overflow_steps'],
        stopped_reason='deadline' if stopped else 'completed')
    _json(run_dir / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config')
    group.add_argument('--predict-config')
    args = parser.parse_args()
    config = json.loads(Path(args.config or args.predict_config).read_text(encoding='utf-8'))
    if args.config:
        run_experiment(config)
    else:
        if config['split'] not in ('test', 'audit', 'dev'):
            raise ValueError('Prediction split must be test, audit, or dev')
        rows = _split_rows(config, config['split'])
        probabilities = predict_checkpoint(config['checkpoint'], [r['text'] for r in rows],
            config['max_length'], config.get('batch_size', 16), config.get('device', 'cuda'))
        _save_predictions(config['output'], rows, probabilities)


if __name__ == '__main__':
    main()
