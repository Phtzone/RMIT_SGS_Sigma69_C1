# C1 implementation record

Plan: [PLAN.md](PLAN.md). Raw competition CSVs must remain byte-for-byte unchanged.

## Work items

- [x] Data preparation, immutable hashes, approximate similarity groups, fixed splits and data report.
- [x] Training/prediction worker, half-epoch selection, time budget and OOM recovery.
- [x] Metrics, model/ensemble selection, locked audit and submission validation.
- [x] Full experiment orchestration, portable Kaggle notebook and documentation; software time caps removed by user request on 2026-10-22.
- [x] Unit/integration verification and whole-pipeline review.

## Environment decisions

- Git denied creating a feature branch; implementation proceeds in the current checkout without changing Git metadata.
- Kaggle CLI is authenticated as `phtzone`; no connected browser is available. API versions 2 and 3 received CPU-only runtimes. After user UI launch, live logs at 04:55:02 UTC on 2026-10-22 confirmed two Tesla T4 GPUs, torch 2.11.0+cu128 and working Internet. The new session is RUNNING; checkpoint/metric evidence remains pending. See `memory-bank/active-context.md`.
- Use a workspace-local virtual environment for CPU verification. Two-T4 training is verified only after execution in Kaggle.
- Do not modify the user-provided PLAN.md or the original demo.

## Verification evidence

- `.venv/Scripts/python.exe -m pytest -q`: **53 passed** after removal of software time caps. Includes real offline tiny-model train/save/reload, sealed audit, full refit, subprocess inference, CSV export, collection guards and a real worker with no deadline.
- Independent reviewer reproduced and verified fixes for normal AMP overflow, interrupted checkpoint publication, failed first baseline and stale inference artifact reuse. Final scoped regression + offline integration: **2 passed**.
- Real data preparation: train 9,799; dev 2,100; audit 2,101; no group crosses a fold. Recomputed near pairs: 262 train-train and 208 train-test under the implemented approximate method.
- All three input SHA-256 fingerprints match the pre-implementation values.
- Downloaded official tokenizer/config files only for XLM-R and mDeBERTa; multilingual examples and 256-token head-tail encoding succeeded. Resolved revisions are in `artifacts/tokenizer_checks.json`.
- Generated notebook embeds current package sources, contains no competition CSVs/credentials, and every non-magic code cell compiles. Payload/source equality is tested.
- Local test environment: Python 3.14.6, torch 2.14.1+cpu, transformers 4.57.6. Kaggle version 5 preflight confirms two T4 GPUs, torch 2.11.0+cu128 and Internet; both baseline workers have started, with experiment metrics still pending.

## Remaining execution step

Monitor the existing private Save Version run at `phtzone/rmit-sgs-sigma69-c1`; GPU/network preflight and dependency installation have passed. GPU preflight runs before dependency installation, and pip failures stop execution. Local notebook validation and an actual CPU preflight run passed; CSV hashes remain unchanged. Download and inspect checkpoints/metrics and validate the learned 6,000-row CSV when available. Submission/scoring requires a separate explicit user request; the pipeline never submits.
