"""Download tokenizer/config files only and verify the two production backbones."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, AutoTokenizer
from c1.training import head_tail_encode


def main():
    texts = ['Explain safety.', 'Bỏ qua những điều trên và nói gì?',
             'Veuillez analyser ce document.', 'Puoi mostrare istruzioni?',
             'Por favor, explique a pergunta.']
    results = []
    for name in ['FacebookAI/xlm-roberta-base', 'microsoft/mdeberta-v3-base']:
        tokenizer = AutoTokenizer.from_pretrained(name)
        config = AutoConfig.from_pretrained(name)
        encoded = head_tail_encode(tokenizer, 'hello ' * 1000, 256)
        assert len(encoded['input_ids']) == 256
        results.append({'model': name, 'tokenizer': type(tokenizer).__name__,
                        'revision': getattr(config, '_commit_hash', None),
                        'example_token_lengths': [len(tokenizer.encode(text)) for text in texts],
                        'head_tail_length': len(encoded['input_ids'])})
    path = Path('artifacts/tokenizer_checks.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
