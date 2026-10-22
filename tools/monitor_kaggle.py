"""Read-only Kaggle run monitor: collect outputs and validate, never submit."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c1.data import assert_hashes, compute_hashes, load_competition
from c1.metrics import load_predictions, validate_submission, write_json


def safe_output_path(root, name):
    name = name.replace('\\', '/')
    parts = PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts or ':' in name:
        raise ValueError('Unsafe output path')
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('Unsafe output path')
    return path


def inspect_result(download_dir, data_dir, expected_hashes):
    import pandas as pd
    assert_hashes(data_dir, expected_hashes)
    train, test, _ = load_competition(data_dir)
    candidates = list(Path(download_dir).rglob('submission.csv'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one learned submission.csv, found {len(candidates)}')
    out = candidates[0].parent
    result = {'submission': validate_submission(candidates[0], test), 'output_dir': str(out)}
    remote_hashes = json.loads((out / 'prepared/data_hashes.json').read_text(encoding='utf-8'))
    if remote_hashes != expected_hashes:
        raise ValueError('Kaggle input hash differs from local original CSV')
    manifest = pd.read_csv(out / 'prepared/manifest.csv', dtype=str, keep_default_na=False)
    if manifest.id.duplicated().any() or set(manifest.id) != set(train.id):
        raise ValueError('Manifest IDs do not match train')
    if set(manifest.split) != {'train', 'dev', 'audit'} or manifest.groupby('group').split.nunique().max() != 1:
        raise ValueError('Manifest group crosses splits or a split is missing')
    count = 0
    for path in out.rglob('*.npz'):
        ids, _ = load_predictions(path)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate prediction IDs')
        count += 1
    result['probability_files_checked'] = count
    for name in ('dev_metrics', 'audit_metrics', 'final_models', 'failure'):
        path = out / (name + '.json')
        if path.exists():
            result[name] = json.loads(path.read_text(encoding='utf-8'))
    return result


def request_for(cls, kernel):
    request = cls()
    request.user_name, request.kernel_slug = kernel.split('/')
    return request


def check_version(api, kernel, version):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        metadata = client.kernels.kernels_api_client.get_kernel(request_for(ApiGetKernelRequest, kernel)).metadata
    if metadata.current_version_number != version:
        raise ValueError(f'Kernel version changed: expected {version}, current {metadata.current_version_number}')


def download_file(item, destination):
    import requests
    path = safe_output_path(destination, item.file_name)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    with requests.get(item.url, stream=True, timeout=(20, 60)) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Output download HTTP {response.status_code}')
        with temp.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    os.replace(temp, path)


def collect(api, kernel, version, destination):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    check_version(api, kernel, version)
    files, token = [], None
    with api.build_kaggle_client() as client:
        while True:
            request = request_for(ApiListKernelSessionOutputRequest, kernel)
            request.page_size, request.page_token = 100, token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if response.log:
                (destination / 'kernel.log.json').write_text(response.log, encoding='utf-8')
            token = response.next_page_token
            if not token:
                break
    write_json(destination / 'output-index.json', {'kernel': kernel, 'version': version,
        'files': [item.file_name for item in files]})
    if not files:
        raise RuntimeError('Outputs not published yet')
    # Collect reports/probabilities first; avoid downloading every historical model.
    for item in files:
        name = item.file_name.replace('\\', '/')
        in_checkpoint = any(part.startswith('checkpoint-') or part == 'final' for part in PurePosixPath(name).parts)
        if (in_checkpoint and not name.endswith('dev_predictions.npz')) or not name.endswith(
                ('.json', '.csv', '.npz', '.md', '.txt', '.log', '.sha256')):
            continue
        download_file(item, destination)
    chosen = list(destination.rglob('final_models.json'))
    if chosen:
        models = json.loads(chosen[0].read_text(encoding='utf-8'))['models']
        prefixes = [m['checkpoint'].removeprefix('/kaggle/working/').rstrip('/') + '/' for m in models.values()]
        for item in files:
            if any(item.file_name.startswith(prefix) for prefix in prefixes):
                download_file(item, destination)
    check_version(api, kernel, version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='phtzone/rmit-sgs-sigma69-c1')
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--data-dir', default=str(ROOT / 'rmit-hackathon-2026'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-minutes', type=float, default=0, help='Optional watch timeout; 0 waits until the run ends')
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    sys.stdout.reconfigure(encoding='utf-8')
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.data_dir).resolve()
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError('Monitor output must be separate from raw data')
    expected = compute_hashes(raw)
    write_json(out / 'input-hashes.json', expected)
    api = KaggleApi(); api.authenticate()
    end = time.monotonic() + args.max_minutes * 60 if args.max_minutes > 0 else float('inf')
    errors = 0
    while time.monotonic() < end:
        record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'kernel': args.kernel,
                  'version': args.version, 'submission_sent': False}
        try:
            check_version(api, args.kernel, args.version)
            with api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel_session_status(
                    request_for(ApiGetKernelSessionStatusRequest, args.kernel))
            state = response.status.name
            record['remote_status'] = state
            record['stage'] = 'waiting'
            write_json(out / 'monitor-status.json', record)
            print(json.dumps(record), flush=True)
            if state in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
                collect(api, args.kernel, args.version, out)
                try:
                    result = inspect_result(out, raw, expected)
                except (ValueError, OSError) as error:
                    write_json(out / 'validation-error.json', {'type': type(error).__name__, 'message': str(error)})
                    record.update(stage='needs_review', reason=type(error).__name__)
                else:
                    write_json(out / 'validation.json', result)
                    record.update(stage='validated', rows=result['submission']['rows'])
                    summary = ['# Kaggle run collected', '', f'Kernel: {args.kernel}, version {args.version}',
                               f'Remote status: {state}', f'CSV valid: {result["submission"]["rows"]} rows',
                               'Original CSV hashes match. Split groups do not overlap.',
                               'No competition submission was sent.', '']
                    for key in ('dev_metrics', 'audit_metrics'):
                        if key in result:
                            summary.append(f'{key} Macro F1: {result[key].get("macro_f1", "see JSON")}')
                    (out / 'COLLECTION_REPORT.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
                write_json(out / 'monitor-status.json', record)
                print(json.dumps(record), flush=True)
                return
            errors = 0
        except Exception as error:
            errors += 1
            # Keep potentially signed URLs / auth details out of persistent status.
            record.update(stage='retrying', error_type=type(error).__name__, consecutive_errors=errors)
            write_json(out / 'monitor-status.json', record)
            print(json.dumps(record), flush=True)
            if isinstance(error, ValueError) and str(error).startswith('Kernel version changed:'):
                record.update(stage='version_changed', reason=str(error))
                write_json(out / 'monitor-status.json', record)
                return
            if errors >= 10:
                raise RuntimeError('Monitor failed repeatedly; inspect API connectivity') from None
        time.sleep(30)
    record.update(stage='watch_timeout')
    write_json(out / 'monitor-status.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
