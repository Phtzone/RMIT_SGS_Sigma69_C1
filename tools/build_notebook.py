"""Build a self-contained Kaggle notebook from the current package sources."""

import argparse
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile


def build(output):
    root = Path(__file__).resolve().parents[1]
    buffer = io.BytesIO()
    files = sorted((root / 'c1').glob('*.py')) + [root / 'requirements.txt']
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode('ascii')
    cells = []

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})

    def code(source):
        source = textwrap.dedent(source).strip() + '\n'
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {},
                      'outputs': [], 'source': source.splitlines(keepends=True)})

    markdown('''# RMIT_SGS_Sigma69_C1

Pipeline XLM-R + mDeBERTa, grouped train/dev/audit, lựa chọn theo Macro F1,
ensemble, full-data refit và submission. Notebook tự chứa mã nguồn; không chứa CSV hoặc credentials.

Trước khi chạy: tham gia cuộc thi, thêm **rmit-hackathon-2026** vào Input,
chọn **GPU T4 ×2**, bật Internet và đối chiếu Rules, quota, số lượt nộp, deadline.
Notebook không tự gửi bài Kaggle. Dữ liệu trong Input chỉ được đọc.

Theo yêu cầu mới: chạy hết các thử nghiệm và full-data refit, không cắt theo số phút.
Giới hạn phiên/quota do Kaggle quản lý vẫn có hiệu lực. Deadline cuộc thi là thông tin
để người dùng quyết định nộp bài, không phải điều kiện dừng chương trình.
Điểm kiểm tra cuối là điểm trước full-data refit.
''')
    code('''
        import json, shutil, socket, subprocess, sys
        from datetime import datetime, timezone
        import torch
        print('Runtime preflight:', datetime.now(timezone.utc).isoformat(), flush=True)
        print('Python:', sys.version, 'Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
        if shutil.which('nvidia-smi'):
            subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)
        print('GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('Kaggle runtime has no CUDA GPU. Check Settings: Accelerator T4 x2 and account verification; metadata alone is insufficient.')
        for host in ('pypi.org', 'huggingface.co'):
            with socket.create_connection((host, 443), timeout=15):
                print('HTTPS connection available:', host, flush=True)
    ''')
    code('''
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'transformers>=4.46,<5', 'accelerate>=1,<2',
                        'scikit-learn>=1.5,<2', 'pandas>=2.2,<4', 'numpy>=1.26,<3',
                        'protobuf>=4,<7', 'sentencepiece', 'safetensors'], check=True)
    ''')
    code(textwrap.dedent('''
        import base64, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        PROJECT = Path('/kaggle/working/c1-project')
        PROJECT.mkdir(parents=True, exist_ok=True)
    ''') + '\nSOURCE_ARCHIVE = ' + json.dumps(payload) + '''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
    for member in archive.infolist():
        destination = (PROJECT / member.filename).resolve()
        if PROJECT.resolve() not in destination.parents:
            raise ValueError('Invalid source archive path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
os.chdir(PROJECT)
print('Source extracted to', PROJECT)
''')
    code('''
        # Chỉ thay đường dẫn khi Kaggle gắn Input ở vị trí khác.
        expected = [Path('/kaggle/input/competitions/rmit-hackathon-2026'),
                    Path('/kaggle/input/rmit-hackathon-2026')]
        candidates = [p for p in expected if (p / 'train.csv').exists()]
        if not candidates:
            candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('train.csv')
                                 if (p.parent / 'test.csv').exists()
                                 and (p.parent / 'sample_submission.csv').exists()})
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one competition input; found {candidates}. Set DATA_DIR explicitly.')
        DATA_DIR = candidates[0]
        OUTPUT_DIR = Path('/kaggle/working/c1-output')
        import torch
        print('Input:', DATA_DIR)
        print('GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        if not torch.cuda.is_available():
            raise RuntimeError('Enable GPU T4 x2 in Kaggle notebook Settings.')
        if torch.cuda.device_count() < 2:
            print('One GPU detected: pipeline will use the single-device fallback schedule.')
    ''')
    code('''
        command = [sys.executable, '-u', '-m', 'c1', 'run',
                   '--data-dir', str(DATA_DIR), '--output-dir', str(OUTPUT_DIR),
                   '--no-time-limit']
        result = subprocess.run(command, cwd=PROJECT)
        if result.returncode:
            print('Inspect c1-output/runs/*/train.log and c1-output/failure.json for the error.')
            raise RuntimeError(f'Pipeline exited with code {result.returncode}')
    ''')
    code('''
        subprocess.run([sys.executable, '-m', 'c1', 'validate-submission',
                        '--data-dir', str(DATA_DIR),
                        '--submission', str(OUTPUT_DIR / 'submission.csv')], check=True)
        import pandas as pd
        display(pd.read_csv(OUTPUT_DIR / 'submission.csv').head())
        report = OUTPUT_DIR / 'EXPERIMENT_REPORT.md'
        if report.exists():
            from IPython.display import Markdown, display
            display(Markdown(report.read_text()))
        from IPython.display import FileLink
        display(FileLink(str(OUTPUT_DIR / 'submission.csv')))
        print('Save Version with outputs before ending the Kaggle session.')
    ''')
    markdown('''## Artifacts

- `c1-output/submission.csv`: bài nộp đã kiểm tra.
- `c1-output/submission_fallback.csv`: bài dự phòng từ checkpoint đã đánh giá.
- `c1-output/prepared/`: manifest, hash CSV, thống kê dữ liệu và gần trùng.
- `c1-output/runs/`: cấu hình, train log, checkpoint, token statistics và xác suất dev.
- `selection.json` + `selection.sha256`: phương án khóa trước khi đánh giá audit.
- `dev_metrics.json`, `audit_metrics.json`, `EXPERIMENT_REPORT.md`: kết quả và quyết định.
- `failure.json`: nếu pipeline phải dùng bài dự phòng vì một bước sau thất bại.

Nếu chạy lại, chọn OUTPUT_DIR mới để giữ nguyên audit của lần trước. Khi hết quota hoặc
không có thời gian, dùng bài dự phòng hợp lệ. Không mở audit để tiếp tục dò tham số.
''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': True,
                   'language': 'python', 'sourceType': 'notebook',
                   'competitionDataSources': ['rmit-hackathon-2026']},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    for index, cell in enumerate(cells):
        cell['id'] = f'c1-{index:02d}'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='RMIT_SGS_Sigma69_C1.ipynb')
    build(parser.parse_args().output)
