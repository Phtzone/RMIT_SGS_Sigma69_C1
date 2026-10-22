import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def test_generated_notebook_contains_executable_current_sources_and_no_data(tmp_path):
    path = tmp_path / 'RMIT_SGS_Sigma69_C1.ipynb'
    subprocess.run([sys.executable, 'tools/build_notebook.py', '--output', str(path)], check=True)
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    assert notebook['metadata']['kaggle']['isGpuEnabled'] is True
    code_cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in code_cells:
        compile(source, '<notebook>', 'exec')
    # Fail before downloading dependencies when Kaggle silently provisions CPU.
    assert 'torch.cuda.is_available()' in code_cells[0]
    assert 'nvidia-smi' in code_cells[0]
    assert 'huggingface.co' in code_cells[0]
    assert 'check=True' in code_cells[1]
    run_cell = next(source for source in code_cells if "'-m', 'c1', 'run'" in source)
    assert '--budget-minutes' not in run_cell
    assert '--competition-deadline' not in run_cell
    assert '--no-time-limit' in run_cell
    payload_cell = next(source for source in code_cells if 'SOURCE_ARCHIVE =' in source)
    payload = payload_cell.split('SOURCE_ARCHIVE = ', 1)[1].splitlines()[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(json.loads(payload)))) as archive:
        names = archive.namelist()
        assert not any(name.endswith('.csv') for name in names)
        assert not any('kaggle.json' in name or name.endswith('.env') for name in names)
        for source_path in Path('c1').glob('*.py'):
            assert archive.read(source_path.as_posix()) == source_path.read_bytes()
            compile(archive.read(source_path.as_posix()), source_path.as_posix(), 'exec')
