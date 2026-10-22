"""CLI: python -m c1 --help."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='RMIT C1 reproducible classification pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Validate CSVs, group similar train rows and seal splits')
    prep.add_argument('--data-dir', required=True)
    prep.add_argument('--output-dir', required=True)
    prep.add_argument('--seed', type=int, default=42)
    for name in ('train', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
    evaluate = commands.add_parser('evaluate', help='Dev selection or one sealed audit')
    evaluate.add_argument('--data-dir', required=True)
    evaluate.add_argument('--output-dir', required=True)
    evaluate.add_argument('--audit', action='store_true')
    evaluate.add_argument('--gpu-count', type=int, default=1)
    validate = commands.add_parser('validate-submission')
    validate.add_argument('--data-dir', required=True)
    validate.add_argument('--submission', required=True)
    run = commands.add_parser('run', help='GPU experiments, audit, full refit and CSV; no time cap by default')
    run.add_argument('--data-dir', required=True)
    run.add_argument('--output-dir', required=True)
    time_mode = run.add_mutually_exclusive_group()
    time_mode.add_argument('--budget-minutes', type=float, default=None, help='Optional explicit time cap')
    time_mode.add_argument('--no-time-limit', action='store_true', help='Run all stages without software time caps (default)')
    run.add_argument('--competition-deadline', help='Optional explicit ISO8601 stop time with timezone')
    args = parser.parse_args()
    if args.command == 'run' and args.no_time_limit and args.competition_deadline:
        parser.error('--no-time-limit cannot be combined with --competition-deadline')
    if args.command == 'prepare':
        from c1.data import prepare
        result = prepare(args.data_dir, args.output_dir, seed=args.seed)
    elif args.command == 'train':
        from c1.training import run_experiment
        result = run_experiment(json.loads(Path(args.config).read_text(encoding='utf-8')))
    elif args.command == 'predict':
        import subprocess
        import sys
        subprocess.run([sys.executable, '-m', 'c1.training', '--predict-config', args.config], check=True)
        result = {'prediction_config': args.config, 'completed': True}
    elif args.command == 'validate-submission':
        from c1.data import load_competition
        from c1.metrics import validate_submission
        _, test, _ = load_competition(args.data_dir)
        result = validate_submission(args.submission, test)
    elif args.command == 'evaluate':
        from c1.pipeline import audit_selection, select_models
        out = Path(args.output_dir)
        if args.audit:
            result = audit_selection(Path(args.data_dir).resolve(), out.resolve(), args.gpu_count, None)
        else:
            results = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((out / 'runs').glob('*/result.json'))]
            results = [r for r in results if not r.get('full_train')]
            result = select_models(results, args.data_dir, out / 'prepared', out)
    else:
        from c1.pipeline import run_pipeline
        result = run_pipeline(args.data_dir, args.output_dir, args.budget_minutes, args.competition_deadline)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
