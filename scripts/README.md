# Commands

Run commands from the repository root. Root-level scripts locate that root with
`Path(__file__).resolve().parents[1]`; scripts one directory deeper use `parents[2]`.
Shared runtime code belongs in `models/`, `data/`, `envs/`, and `configs/`.

## Main pipeline

| Command | Purpose |
|---|---|
| `run_experiment.py` | Reuse/build a validated shared cache, then resume/train one seed. |
| `runpod_update_matched_study.py` | Execute the update-matched RunPod study; deploy one dataset worker per Pod. |
| `run_study.py` | Earlier episode-holdout fixed-endpoint workflow; `--dry-run` prints commands only. |
| `prepare_protocol.py` | Prepare episode splits and training-only statistics without a VLM. |
| `precompute.py` | Cache frozen features (`--dataset --exp`). |
| `train.py` | Train the adapter/action head on cached features. |
| `evaluate.py`, `eval_libero.py` | Closed-loop policy evaluation. |
| `offline_eval.py` | Historical open-loop evaluation. |
| `validate_harness.py`, `validate_harness_aloha.py` | Simulator harness checks. |
| `train_lt_onthefly.py`, `convert_language_table.py` | Language Table training/data variants. |
| `analysis.py`, `compare.py`, `export_bank.py` | Legacy PushT analysis and probe-bank export. |

The [camera-ready workflow](../docs/CAMERA_READY.md) identifies the current opt-in
protocol and deferred experiment commands. Historical entrypoints do not
constitute revised experiment results.

The current update-matched RunPod workers are launched with
`scripts/runpod_update_matched_study.py`, one dataset per Pod. For their locked
plan, current run roots, live status commands, recovery procedure, and required
cross-Pod final analysis, use the [team continuity guide](../docs/team_continuity_guide.md).

The `configs/camera_ready_study.json` and `scripts/run_study.py` instructions
below belong to the earlier episode-holdout workflow. Do not use them to launch
the active update-matched RunPod matrix; its locked configuration and operating
instructions are in `configs/runpod_update_matched_study.json` and the
[team continuity guide](../docs/team_continuity_guide.md).

## One full experiment

```powershell
$py = 'C:\Users\SK\miniconda3\envs\vla\python.exe'
& $py scripts/run_experiment.py --dataset aloha --exp exp05 --seed 0 --split-manifest asset/protocols/aloha/97fa699cb307366377cec2eb0f29795d967710f9ffe0fea9d1e1c50108b8d537.json --run-root asset/runs/hf_warmup_patience
```

Repeat this command to resume. A completed compatible run is skipped. Change
`--seed` for an independent policy using the same validated feature cache.

```text
asset/protocols/<dataset>/<protocol_id>.json         episode splits/statistics
asset/cache/<dataset>/<backbone>/<identity>/         shared frozen features
asset/runs/hf_warmup_patience/<dataset>/<experiment>/
  seed_000/checkpoints/                             best, last, final
  seed_000/history.jsonl                            per-epoch training record
  seed_000/run_manifest.json                        configuration/result
  seed_000/pipeline.json                            latest orchestration status
  seed_000/attempts/<UTC timestamp>/                logs, checks, source snapshot
```

Cache identities include the protocol, encoder settings/code, and source
weights/processor hashes, but no training seed. Encoding writes a `.partial.h5`
file; only a complete cache passing provenance, shape, finite-value, raw ALOHA
boundary and batch-one parity checks is published. Interrupted partial files are
preserved under timestamped names. A bad published cache causes an explicit
error. OS file locks prevent duplicate writers or duplicate runs of one seed.
Training uses Hugging Face linear warmup/cosine decay toward a 1% LR floor.
ALOHA reaches this floor by epoch 100 (`lr_decay_epochs=100`) and holds it;
the independent maximum remains 300 epochs. The endpoint is shared by both
backbones. Other benchmarks default to decay across their full epoch budget.
Validation patience (30 epochs by default) is enabled as soon as the configured
warmup optimizer updates finish. For ALOHA, warmup is 300 updates and the first
eligible validation is epoch 5. ALOHA's diagnostic `min_epochs=100` blocks
stopping until epoch 100, without deferring the patience count. Training then
stops on exhausted patience or its 300-epoch maximum. State is resumable across
warmup, decay and the constant-floor phase. The previous OneCycle pilot stays under
`asset/runs/episode_holdout/`; its changed training identity requires a separate
run root for the new recipe. Shared caches are reused.
`--run-root` optionally selects a campaign directory. No rollout is launched.

## Supporting scripts

- `data/`: downloaders, conversion, cache migration, dataset inspection.
- `validation/`: milestone gates.
- `validation/observe_training.py`: bounded real-data CUDA/DataLoader/optimizer
  and interrupted-resume check; writes only to a new readiness directory.
- `eval/`: instruction, OOD, and token-ablation evaluations.
- `analysis/`: head diagnostics, attention/PCA probes, paired statistics, figures.
- `report/`, `plots/`, `media/`: report, figure, and video generators.
- `archive/`: superseded experiments retained for reference.
- Root `.ps1`/`.sh` files: existing launch/status/migration commands; paths retained
  because external scheduled tasks may reference them.

The frozen-backbone implementation is now `models/backbones.py`. The latent-study
`backbones.py` only re-exports it for older study commands. `analysis/__init__.py`
keeps package imports safe alongside the historical `scripts/analysis.py` CLI.
Root compatibility commands forward to this pipeline; they contain no optimizer,
cache-generation, or rollout implementation of their own.
