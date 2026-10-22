import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_training_module_exists():
    assert importlib.util.find_spec('c1.training') is not None


def test_head_tail_and_weights():
    from c1.training import head_tail_encode, class_weights, accumulation_windows, epoch_windows

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, add_special_tokens=False):
            return list(range(int(text)))
        def prepare_for_model(self, ids, **kwargs):
            return {'input_ids': [101] + ids + [102], 'attention_mask': [1]*(len(ids)+2)}

    tok = Tokenizer()
    assert head_tail_encode(tok, '12', 7)['input_ids'] == [101, 0, 1, 2, 10, 11, 102]
    assert head_tail_encode(tok, '2', 7)['input_ids'] == [101, 0, 1, 102]
    assert head_tail_encode(tok, '12', 2)['input_ids'] == [101, 102]
    np.testing.assert_allclose(class_weights([0, 0, 0, 0, 1, 2]), [1, 2, 2])
    assert list(accumulation_windows(5, 4)) == [(0, 4), (4, 5)]
    assert list(epoch_windows(9, 9, 4)) == [(0, 4), (4, 5), (5, 9)]
    assert list(epoch_windows(9, 3, 4)) == [(0, 3)]


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_prediction_reduces_batch_after_oom():
    import torch
    from types import SimpleNamespace
    from c1.training import _probabilities

    calls = []
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids):
            calls.append(len(input_ids))
            if len(input_ids) > 2:
                raise torch.cuda.OutOfMemoryError('Synthetic capacity limit')
            return SimpleNamespace(logits=torch.zeros((len(input_ids), 3)))

    class Tokenizer:
        def pad(self, features, **kwargs):
            return {'input_ids': torch.tensor([feature['input_ids'] for feature in features])}

    probabilities = _probabilities(Model(), Tokenizer(), ['x'] * 5, 12, 5, 'cpu',
        encoded=[{'input_ids': [1, 2]}] * 5)
    assert calls == [5, 2, 2, 1]
    np.testing.assert_allclose(probabilities, np.ones((5, 3)) / 3, rtol=1e-6)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_gradient_scaler_recovers_overflow_without_advancing_schedule():
    import torch
    from c1.training import _optimizer_step

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler('cpu', enabled=True)
    state = {'consecutive_overflows': 0}
    initial_epoch = scheduler.last_epoch
    loss = model(torch.tensor([[1e35]])).sum()
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    assert not _optimizer_step(model, optimizer, scaler, scheduler, True, state)
    assert scaler.get_scale() == 32768 and scheduler.last_epoch == initial_epoch
    assert model.weight.item() == 1 and state['consecutive_overflows'] == 1
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones((1, 1))).sum()).backward()
    assert _optimizer_step(model, optimizer, scaler, scheduler, True, state)
    assert scheduler.last_epoch == initial_epoch + 1
    assert model.weight.item() < 1 and state['consecutive_overflows'] == 0
    for attempt in range(2):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.tensor([[1e35]])).sum()).backward()
        if attempt == 0:
            assert not _optimizer_step(model, optimizer, scaler, scheduler, True, state, max_overflows=2)
        else:
            with pytest.raises(FloatingPointError, match='overflow'):
                _optimizer_step(model, optimizer, scaler, scheduler, True, state, max_overflows=2)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch unavailable')
def test_real_train_save_reload_predict(tmp_path, monkeypatch):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from c1 import LABELS
    from c1.training import run_experiment, predict_checkpoint

    torch.set_num_threads(1)
    model_dir = tmp_path / 'tiny'
    model_dir.mkdir()
    (model_dir / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nattack\n', encoding='utf-8')
    BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt')).save_pretrained(model_dir)
    BertForSequenceClassification(BertConfig(vocab_size=8, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, num_labels=3,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(model_dir)
    data = tmp_path / 'data'
    prepared = tmp_path / 'prepared'
    data.mkdir(); prepared.mkdir()
    with (data / 'train.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'label', 'language']); writer.writeheader()
        for i in range(12):
            writer.writerow(dict(id=str(i), text='hello world attack', label=LABELS[i % 3], language='en'))
    with (prepared / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'group', 'fold', 'split']); writer.writeheader()
        for i in range(12):
            writer.writerow(dict(id=str(i), group=str(i), fold=0, split='train' if i < 7 else ('dev' if i < 10 else 'audit')))
    config = dict(run_id='smoke', family='xlmr', model_id=str(model_dir), data_dir=str(data),
        prepared_dir=str(prepared), run_dir=str(tmp_path / 'run'), device='cpu', epochs=1,
        batch_size=2, grad_accum=2, max_length=12)
    result = run_experiment(config)
    assert result['completed'] is True
    assert result['optimizer_steps'] == result['scheduler_steps'] == 2
    assert result['overflow_steps'] == 0
    assert Path(result['checkpoint']).name.startswith('checkpoint-')
    assert Path(result['dev_predictions']).parent == Path(result['checkpoint'])
    assert result['wall_seconds'] >= result['train_seconds'] > 0
    stats = json.loads((tmp_path / 'run' / 'token_statistics.json').read_text())
    assert set(stats) == {'train', 'dev', 'scope'}
    assert stats['train']['count'] == 7 and stats['dev']['count'] == 3
    assert stats['train']['quantiles']['0.5'] == 3
    assert stats['train']['over256'] == 0 and stats['train']['over512'] == 0
    assert result['best_epoch'] in (0.5, 1.0)
    assert result['dev_macro_f1'] is not None
    probs = predict_checkpoint(result['checkpoint'], ['hello world'], 12, device='cpu')
    assert probs.shape == (1, 3)
    np.testing.assert_allclose(probs.sum(1), [1], rtol=1e-5)
    with np.load(result['dev_predictions'], allow_pickle=False) as predictions:
        assert predictions['ids'].tolist() == ['7', '8', '9']
        assert predictions['label_order'].tolist() == list(LABELS)
    saved = json.loads((tmp_path / 'run' / 'result.json').read_text())
    assert saved['checkpoint'] == result['checkpoint']
    assert json.loads((tmp_path / 'run' / 'benchmark.json').read_text())['optimizer_steps'] == 2
    # The 7-example train split gives a final partial window. Its gradients must
    # match a larger microbatch, including correct normalization of that window.
    config.update(batch_size=4, grad_accum=1, run_dir=str(tmp_path / 'large_batch'))
    large = run_experiment(config)
    from transformers import AutoModelForSequenceClassification
    small_model = AutoModelForSequenceClassification.from_pretrained(result['latest_checkpoint'])
    large_model = AutoModelForSequenceClassification.from_pretrained(large['latest_checkpoint'])
    for key, value in small_model.state_dict().items():
        torch.testing.assert_close(value, large_model.state_dict()[key], atol=1e-7, rtol=1e-5)
    old_result = (tmp_path / 'run' / 'result.json').read_bytes()
    with monkeypatch.context() as patch:
        def fail_save(self, directory, **kwargs):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / 'partial').write_text('interrupted')
            raise RuntimeError('simulated disk interruption')
        patch.setattr(BertForSequenceClassification, 'save_pretrained', fail_save)
        with pytest.raises(RuntimeError, match='interruption'):
            run_experiment(dict(config, run_dir=str(tmp_path / 'run')))
    assert (tmp_path / 'run' / 'result.json').read_bytes() == old_result
    assert predict_checkpoint(result['checkpoint'], ['hello'], 12, device='cpu').shape == (1, 3)
    config.update(full_train=True, epochs=0.5, run_dir=str(tmp_path / 'full'))
    full = run_experiment(config)
    assert full['dev_macro_f1'] is None and full['best_epoch'] == 0.5
    assert full['completed'] is True
    config.update(epochs=3, deadline_unix=0, run_dir=str(tmp_path / 'deadline'))
    deadline = run_experiment(config)
    assert deadline['completed'] is False and deadline['stopped_reason'] == 'deadline'
    assert deadline['wall_seconds'] >= deadline['train_seconds']
