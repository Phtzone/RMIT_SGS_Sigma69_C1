import json
from pathlib import Path

import pandas as pd
import pytest

from c1.data import compute_hashes
from tools.monitor_kaggle import inspect_result, safe_output_path


def test_download_paths_cannot_escape_output_directory(tmp_path):
    assert safe_output_path(tmp_path, 'c1-output/submission.csv') == tmp_path / 'c1-output/submission.csv'
    for name in ('../train.csv', '/train.csv', 'C:/train.csv', '..\\train.csv'):
        with pytest.raises(ValueError):
            safe_output_path(tmp_path, name)


def fixture_output(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    out = tmp_path / 'download' / 'c1-output'; (out / 'prepared').mkdir(parents=True)
    pd.DataFrame({'id':['a','b','c'], 'text':['one','two','three'], 'language':['en']*3,
        'label':['BENIGN','JAILBREAK','PROMPT_INJECTION']}).to_csv(raw/'train.csv',index=False)
    pd.DataFrame({'id':['x','y'], 'text':['one','two'], 'language':['en']*2}).to_csv(raw/'test.csv',index=False)
    pd.DataFrame({'id':['x','y'], 'label':['BENIGN','JAILBREAK']}).to_csv(raw/'sample_submission.csv',index=False)
    (out/'submission.csv').write_bytes((raw/'sample_submission.csv').read_bytes())
    hashes=compute_hashes(raw)
    (out/'prepared/data_hashes.json').write_text(json.dumps(hashes))
    pd.DataFrame({'id':['a','b','c'],'group':['a','b','c'],'fold':[6,3,0],
        'split':['train','dev','audit']}).to_csv(out/'prepared/manifest.csv',index=False)
    return raw, out, hashes


def test_collector_checks_real_csv_and_input_hashes(tmp_path):
    raw,out,hashes=fixture_output(tmp_path)
    result=inspect_result(out.parent,raw,hashes)
    assert result['submission']['valid'] and result['submission']['rows']==2
    (out/'submission.csv').write_text('id,label\ny,BENIGN\nx,JAILBREAK\n')
    with pytest.raises(ValueError,match='IDs'):
        inspect_result(out.parent,raw,hashes)


def test_collector_rejects_changed_raw_data_and_cross_split_groups(tmp_path):
    raw,out,hashes=fixture_output(tmp_path)
    m=pd.read_csv(out/'prepared/manifest.csv'); m['group']='shared'
    m.to_csv(out/'prepared/manifest.csv',index=False)
    with pytest.raises(ValueError,match='group'):
        inspect_result(out.parent,raw,hashes)
    with (raw/'train.csv').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='hash'):
        inspect_result(out.parent,raw,hashes)
