"""Offline CPU integration: real model training through sealed audit and export."""

import importlib.util
import json
import time

import numpy as np
import pandas as pd
import pytest

from c1 import LABELS


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='Install ML requirements')
def test_offline_prepare_train_select_audit_refit_export(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from c1.data import prepare, assert_hashes
    from c1.metrics import validate_submission
    from c1.pipeline import select_models, audit_selection, export_selection
    from c1.training import run_experiment

    torch.set_num_threads(1)
    model_dir, raw, out = tmp_path / 'tiny-model', tmp_path / 'raw', tmp_path / 'output'
    model_dir.mkdir(); raw.mkdir(); out.mkdir()
    (model_dir / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n', encoding='utf-8')
    BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt')).save_pretrained(model_dir)
    BertForSequenceClassification(BertConfig(vocab_size=7, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, num_labels=3)).save_pretrained(model_dir)
    rows = []
    for language in ('en', 'pt', 'fr', 'it'):
        for label in LABELS:
            for _ in range(20):
                i = len(rows)
                rows.append({'id': f'train{i}', 'text': f'hello unique{i} item{i}', 'language': language, 'label': label})
    pd.DataFrame(rows).to_csv(raw / 'train.csv', index=False)
    test = pd.DataFrame({'id': ['test0', 'test1', 'test2'], 'text': ['hello world'] * 3,
                         'language': ['en', 'pt', 'it']})
    test.to_csv(raw / 'test.csv', index=False)
    pd.DataFrame({'id': test.id, 'label': ['BENIGN'] * 3}).to_csv(raw / 'sample_submission.csv', index=False)
    report = prepare(raw, out / 'prepared')
    config = {'run_id': 'tiny', 'family': 'xlmr', 'model_id': str(model_dir),
              'data_dir': str(raw), 'prepared_dir': str(out / 'prepared'), 'run_dir': str(out / 'run'),
              'device': 'cpu', 'epochs': .5, 'batch_size': 16, 'grad_accum': 2, 'max_length': 12}
    result = run_experiment(config)
    selection = select_models([result], raw, out / 'prepared', out)
    audit = audit_selection(raw, out, 1, time.time() + 90, device='cpu')
    assert audit['support'] > 0 and 0 <= audit['macro_f1'] <= 1
    with pytest.raises(ValueError, match='already'):
        audit_selection(raw, out, 1, time.time() + 90, device='cpu')
    with pytest.raises(ValueError, match='audit'):
        select_models([result], raw, out / 'prepared', out)
    full = run_experiment({**config, 'run_id': 'full', 'run_dir': str(out / 'full'),
                           'full_train': True, 'epochs': result['best_epoch']})
    assert full['completed'] and full['dev_macro_f1'] is None
    exported = export_selection(selection, raw, out, 1, time.time() + 90,
                                models={'xlmr': full}, device='cpu')
    assert exported['valid'] and exported['rows'] == 3
    assert validate_submission(out / 'submission.csv', test)['rows'] == 3
    assert_hashes(raw, report['hashes'])
    assert json.loads((out / 'audit_metrics.json').read_text())['scope'].startswith('Frozen')
