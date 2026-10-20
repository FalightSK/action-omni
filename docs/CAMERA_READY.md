# Camera-ready workflow

The implementation is opt-in. Three ALOHA GR00T training seeds and 50-start
screens are complete under the original OneCycle episode-holdout pilot, stored
in `asset/runs/episode_holdout/aloha/exp05_groot_transfer/`. The updated
Hugging Face warmup/cosine schedule and patience after warmup below are a separate recipe;
its seed-0 usability diagnostic completed at 100 epochs and 60/100 confirmation successes.
The prospective matched study described below is a further, separately locked protocol.
Commands use a new output root to keep
the pilot distinct. Timestamped attempt directories contain training logs,
source snapshots and cache checks. Shared feature caches are separate from
training seeds; compatible restarts reuse the cache and resume `last.pt`.
See [plan.md](../plan.md) for current execution status and
[commands](../scripts/README.md#one-full-experiment) for the directory layout.

The author authorized implementing the replication plan. Historical bounded
readiness observations are infrastructure checks, not policy-success evidence.

Use the existing `vla` Python for preparation, caching, training, ALOHA, and
analysis; use `vla_libero` for LIBERO simulation. These are entrypoints under
`scripts/`; the legacy root `train.py` is not the revised protocol entrypoint.

## Prospective matched study

The tracked protocol is `configs/camera_ready_study.json`. It supersedes the
older loss-selection examples below for new matched runs. Train new heads at
seeds 3, 4 and 5 for each arm. Use a fixed endpoint for both backbones within
each benchmark: ALOHA 100 epochs, LIBERO 120. Minimum equals maximum; final
checkpoint selection is independent of test success. Save its hash before
evaluation. Keep the earlier seeds and the 60% usability diagnostic separate.

```powershell
$py = 'C:\Users\SK\miniconda3\envs\vla\python.exe'
$liberoPy = 'C:\Users\SK\miniconda3\envs\vla_libero\python.exe'
& $py scripts/run_study.py --dry-run --libero-python $liberoPy
# Run the six ALOHA policies first; caches are shared across training seeds.
& $py scripts/run_study.py --datasets aloha
# Subsequently run the six LIBERO policies with its simulator environment.
& $py scripts/run_study.py --datasets libero --libero-python $liberoPy
# Verify completed records and produce the combined paired report.
& $py scripts/run_study.py --libero-python $liberoPy
```

The runner alternates backbone order across seeds, trains/evaluates sequentially,
and refuses changed locked code/configuration or mismatched evaluation records.
Repeating the same command resumes training and verifies existing tests rather
than overwriting them. The lock and status are under
`asset/runs/camera_ready_fixed_endpoint_v1/locked_study.json` and
`study_status.json`; logs, source snapshots, checkpoints and curves remain
traceable to each policy. Plot titles are derived from the actual run metadata.

ALOHA uses 200 shared starts (IDs 10000–10199) per policy; LIBERO uses all 50
standard starts on each of 10 tasks with canonical instructions. The full
matrix is 12 trained policies / 4,200 rollouts, not 4,200 training replications.
The six planned per-seed paired tests retain Holm family size 6, including when
only the ALOHA subset has completed. Mean and sample SD are across training
seeds; paired episode tests are conditional on each trained pair. Three seeds
still give limited precision for training-population conclusions.

Both simulators enable deterministic inference with an explicit cuBLAS workspace,
disabled TF32 and deterministic PyTorch kernels. Unsupported kernels fail
explicitly. ALOHA `--trajectory FILE.npz` records state/action/reward without
requesting attention weights; `--trace` retains the separate attention diagnostic.
Video outputs are grouped by result filename to avoid overwriting other cohorts.
Small replay cohorts are mechanism/readiness checks, not success-rate estimates.
The prospective ALOHA evaluation sets MuJoCo offscreen anti-aliasing samples to
zero. This removes observed same-seed pixel drift, but changes visual inputs
from the submitted-paper renderer. Record the setting alongside every result.

Before a long run, check the actual policy with
`scripts/validation/replay_aloha.py --checkpoint PATH --output-dir NEW_DIRECTORY`.
This repeats the same two starts without/without/with video and writes
`replay_check.json`, failing if state/action/reward arrays differ. Enabling
deterministic kernels alone is not proof of successful replay; inspect this gate.
The gate passed on the GR00T diagnostic checkpoint with `--render-samples 0`;
the saved evidence is `asset/runs/readiness/fixed_endpoint_renderer0/replay_check.json`.

This prospective study changes training/checkpoint selection and ALOHA camera
rendering from the submitted paper. An exact historical reproduction still requires the missing original
logs/checkpoints and settings; describe that distinction in the manuscript.

## Available now: inspect and test without retraining

```powershell
$py = 'C:\Users\SK\miniconda3\envs\vla\python.exe'
& $py -m unittest discover -s tests -v
& $py scripts/train.py --dataset aloha --exp exp05 --split-manifest asset/protocols/aloha/97fa699cb307366377cec2eb0f29795d967710f9ffe0fea9d1e1c50108b8d537.json --seed 0 --dry-run
```

### Bounded GPU training observation

```powershell
& $py scripts/validation/observe_training.py --samples 16 --workers 6
```

This requires CUDA and existing local backbone/data files. It encodes 16 real
training frames per benchmark, repeats that small cache to exercise the configured
head/batch sizes, and runs the actual trainer with a three-epoch maximum
and two warmup steps. It checks both optimizer groups, then intentionally interrupts
after a committed epoch-1 checkpoint and compares resumed weights, optimizer
moments, and scheduler state with uninterrupted training. ALOHA and LIBERO use
their existing prepared normalization; this partial cache is not a full
episode-holdout training artifact.

Results and checkpoints live in a new `asset/runs/readiness/<timestamp>/`
directory. `observation.json` records success/failure, per-step gradients and
parameter changes, timing, peak GPU allocation, and resume differences. Validation
losses from repeated tiny subsets are infrastructure observations only. This
does not measure policy performance, full-cache throughput, or sustained
long-session thermal stability.

The DiT output projection is initialized to zero, so the adapter can have zero
gradient on the very first optimizer step. The observation requires later
nonzero gradients and parameter changes in both optimizer groups.

The manifests have already been prepared locally. For a fresh checkout/asset
transfer, the following reads raw metadata/data and writes the manifest; it does
not load a VLM or train a head. Existing manifest files are not overwritten.

```powershell
& $py scripts/prepare_protocol.py --dataset aloha --output asset/protocols/aloha/proposed.json
& $py scripts/prepare_protocol.py --dataset libero --output asset/protocols/libero/proposed.json
```

Changing source data changes its protocol identity. Use an explicitly reviewed
new protocol/output namespace rather than mixing earlier caches/checkpoints.

## Encode and train one authorized experiment

Example for **one** ALOHA condition and training seed. Repeat conditionally in
the priority order in `plan.md`; no all-jobs launcher is supplied.

```powershell
$protocol = 'asset/protocols/aloha/97fa699cb307366377cec2eb0f29795d967710f9ffe0fea9d1e1c50108b8d537.json'
& $py scripts/run_experiment.py --dataset aloha --exp exp05 --split-manifest $protocol --seed 0 --run-root asset/runs/hf_warmup_patience
```

Repeat the same command to resume; no manual precompute relaunch is necessary.
The cache is reusable for later authorized seeds 1 and 2. The trainer places each policy in its own
`seed_000`, `seed_001`, or `seed_002` directory. Use `--output-dir` for a deliberate
alternative when invoking the trainer directly; the runner accepts `--run-root`.
Protocol caches require full HDF5 data and refuse truncation,
legacy/mismatched metadata, and overwriting.
The revised encoder removes masked processor padding and uses a fixed right-zero
padding layout for cached and live features. This is an opt-in protocol change;
old caches cannot be assumed compatible.
It also disables BF16 reduced-precision accumulation during encoding, restoring
the prior CUDA flag afterwards. The precision choice is part of cache identity.
Protocol precompute checks stored features against the live batch-one path
before marking the cache complete; a failed check leaves it ineligible.

Resume requires the same protocol,
seed, configuration, code identity, cache, and saved RNG/optimizer state.

### Learning-rate schedule and stopping (19 October update)

The canonical trainer now uses AdamW with fixed default betas, linear warmup,
and cosine decay to a configurable positive floor. It uses the installed
Transformers [cosine schedule with a minimum learning rate](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_with_min_lr_schedule_with_warmup).
The 1% floor is a starting setting to evaluate, not a demonstrated optimum.

For ALOHA, `num_epochs=300` is the hard training maximum, while
`lr_decay_epochs=100` is the earlier cosine endpoint (7,000 updates, including
300 warmup updates). Both GR00T and Qwen3-VL inherit this same setting.
Decoder/adapter peak learning rates are 3e-4 / 1.5e-4.
`min_lr_ratio=0.01` gives final rates 3e-6 / 1.5e-6. Both groups follow the same
multiplier, preserving their 2:1 ratio. The scheduler is stepped after the
optimizer only until the configured endpoint; later updates hold the floor
so the cosine cannot rebound. With the current ALOHA dataset/batch, decoder
LR is approximately 1.62e-4 at epoch 50 (54% of peak), 5.03e-5 at 75 (17%),
and 3e-6 at 100 (1%). This 100-epoch decay is a diagnostic preset, not a
demonstrated optimum. The authorized usability run now sets `min_epochs=100`
to ensure it reaches this endpoint; this is independent of patience counting.
`lr_decay_epochs=None` follows the hard epoch maximum on other benchmarks;
a shortened `--epochs` run also caps the decay horizon to that shorter budget.

Patience is disabled only while `global_step < warmup_steps`. It becomes
eligible at the first end-of-epoch validation at or after warmup completion.
ALOHA has 70 optimizer updates per epoch, so its first eligible validation is
epoch 5 (350 updates, after crossing the 300-update warmup boundary). Thirty
consecutive eligible validations without a new minimum loss stop training;
improvement resets the count. Cosine decay continues while patience is active,
followed by a constant floor if training reaches the earlier decay endpoint.
ALOHA early stopping is blocked until `min_epochs=100`, while the counter
continues to track eligible validations. At or after that minimum the run ends
when patience expires, subject to the hard epoch-300 limit. A validation
improvement resets the count normally. There is no extra training tail.
LIBERO uses the same update-based rule
with a 120-epoch maximum. Validation cadence remains once per epoch.
`best.pt` tracks the minimum loss across all epochs, including warmup;
`last.pt` preserves optimizer/scheduler/RNG/patience state. Logs/history record
warmup versus cosine decay, learning rate, and patience.

This is a new training recipe, separate from the completed OneCycle pilot.
The new `--run-root` above protects the pilot and reuses the compatible feature
cache. Changed training code/signatures and stopping-rule metadata reject
resuming an old OneCycle or post-schedule-patience run as this recipe; its saved
source snapshot is required to reproduce its original
training/inference provenance. The author's training/evaluation request
authorized one ALOHA seed-0 usability diagnostic, with its candidate checkpoint
selection and two confirmation cohorts recorded before launch in
`asset/runs/hf_warmup_patience/aloha/exp05_groot_transfer/seed_000/diagnostic_plan.json`.

That diagnostic is complete: 100 epochs / 7,000 updates. Minimum-loss epoch 22
scored 1/50 on the selection cohort versus epoch 100 at 34/50. The selected
epoch-100 policy then scored 25/50 and 35/50 on untouched cohorts, combined
60% (Wilson 95% interval 50.2–69.1%). For diagnostic inference, load the
checkpoint recorded in that run's `selected_policy.json` (`checkpoints/final.pt`).
The run's `reliability_report.md` records the results and limitations. This
does not silently replace the earlier pilot's loss-selection protocol; lock
a shared selection rule before launching further matched paper comparisons.

Config keys: ALOHA `exp05` (GR00T), `exp06` (Qwen3-VL), optional `exp07` (Cosmos);
LIBERO `exp05`/`exp06` (two views), `exp03`/`exp04` (one-view controls).

## Later: paired evaluation

The primary ALOHA evaluation is uninstrumented. Save trace capture for a
separately named diagnostic run because trace mode changes the attention call
to request weights. For the earlier OneCycle seed 0, the matched screen produced 5/50
successes untraced and 6/50 traced; that comparison does not establish a
systematic instrumentation effect. Three uninstrumented training seeds scored
5/50, 18/50, and 7/50 on the same scene IDs.

```powershell
$env:MUJOCO_GL = 'glfw'
$checkpoint = 'asset/runs/hf_warmup_patience/aloha/exp05_groot_transfer/seed_000/checkpoints/best.pt'
$run = 'asset/runs/hf_warmup_patience/aloha/exp05_groot_transfer/seed_000'
& $py scripts/evaluate.py --dataset aloha --exp exp05 --checkpoint $checkpoint --episodes 200 --no-video --output "$run/evaluation_primary_200.json"

# Optional, separate trace cohort. --trace records the attention-instrumented run.
& $py scripts/evaluate.py --dataset aloha --exp exp05 --checkpoint $checkpoint --episodes 200 --no-video --trace "$run/evaluation_trace_200.trace.npz" --output "$run/evaluation_trace_200.json"
```

LIBERO, using the separate simulator environment:

```powershell
$liberoPy = 'C:\Users\SK\miniconda3\envs\vla_libero\python.exe'
$env:MUJOCO_GL = 'wgl'
$run = 'asset/runs/hf_warmup_patience/libero/exp05_groot_2view/seed_000'
& $liberoPy scripts/eval_libero.py --exp exp05 --output-dir $run --episodes 20 --conditions canonical --videos 0
```

Use identical start IDs and matching training-seed identifiers for the other
endpoint. Saved metadata restores the actual training normalization and checks
the current live encoder against the cache's model/processor identity. Scene
seeds are distinct from training seeds. Do not pass horizon overrides to a
protocol checkpoint. Do not overwrite completed evaluations.

## Later: diagnostics and manuscript exports

```powershell
$run = 'asset/runs/hf_warmup_patience/aloha/exp05_groot_transfer/seed_000'
$cache = (Get-Content "$run/pipeline.json" -Raw | ConvertFrom-Json).cache
& $py scripts/analysis/camera_ready_diagnostics.py --checkpoint $checkpoint --cache $cache --results "$run/evaluation_trace_200.json" --output "$run/diagnostics_trace_200.json"
```

For LIBERO offline diagnostics omit `--results`. The command reads only the
saved head/cache, not the VLM, and performs no optimization. Optional `--device
cuda` changes inference hardware. The default is CPU.

Build a JSON reporting manifest with paths relative to that manifest. Add one
entry per completed matched training-seed pair. Include both benchmark labels
in the same primary-family manifest (six planned tests). Camera comparisons
belong to a separate manifest and use `"vary": "camera"`.

```json
{
  "family_size": 6,
  "labels": {"ALOHA backbone": {"left": "GR00T", "right": "Qwen3-VL"}},
  "pairs": [
    {"label": "ALOHA backbone", "vary": "backbone",
     "left": "../aloha/exp05_groot_transfer/seed_000/sim_results.json",
     "right": "../aloha/exp06_qwen3vl_transfer/seed_000/sim_results.json"}
  ]
}
```

This is a schema/example, not a declaration that those result files exist.

```powershell
& $py scripts/analysis/paired_results.py --manifest path/to/comparisons.json --output path/to/paired_analysis.json --report-dir path/to/manuscript_exports
```

The output contains raw/adjusted p-values, paired intervals and counts, individual
training-seed results, and sample SD. Optional exports are a LaTeX table and
performance PDF. Review the counts, scope, and conclusions before integrating
them into `manuscript/main.tex`; that draft never automatically inserts stale
generated results. The historical analysis/plot scripts are not the revised path.

See `docs/camera_ready_validation.md` for the current validation boundary and
remaining real-model/simulator gates.
