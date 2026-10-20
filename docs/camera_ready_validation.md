# Camera-ready implementation and validation record

Date: 18 October 2026 (JST).

## Initial implementation boundary

At the initial implementation stage, the author requested code and an editable LaTeX draft, with retraining later
after confirmation. No real-data head training, optimizer run, full VLM cache
generation, performance rollout batch, or optional Cosmos training was started.
Only protocol preparation, dry runs, synthetic tests, source checks, and LaTeX
compilation were performed. Tests use tiny temporary synthetic caches and an
untrained small head for forward inference; their outputs are not paper results.

The author subsequently authorized a bounded real-data training observation to
check GPU access, loading, and gradient updates. That separate readiness check
does not authorize full policy retraining or performance rollouts.

## Implemented

| Area | Change | Validation |
|---|---|---|
| Data split | Persisted, task-stratified whole-episode holdout, split seed 42. | Synthetic disjointness checks and real metadata preparation. |
| Normalization | ALOHA statistics and gripper calibration derived from training episodes only. | Changing held-out synthetic values leaves training statistics unchanged. |
| Cache | Full-dataset protocol checks, episode mapping, configuration and encoder identity, source-asset hashes, explicit completion flag; remove processor pads and right-zero-pad consistently in the opt-in path. | Temporary HDF5 roundtrip; reject partial writes/configuration mismatches; left/right/multi-layer padding regression test. |
| Training | Explicit seed/output/protocol arguments; seeded shuffling/workers; validation RNG isolation; checkpoint metadata/RNG; strict protocol resume signature; actual final epoch. | Synthetic checks plus the bounded real-data CUDA optimizer/resume observation below. Resume using a complete protocol cache remains pending. |
| Evaluation | Restore saved configuration/normalization; verify live encoder identity for protocol checkpoints; record scene and policy seeds; reject incompatible head tensors; keep primary rollouts uninstrumented and make traces opt-in. | Three independent GR00T ALOHA heads evaluated on the same 50 scene starts: 10%, 36%, and 14% success; mean 20%, sample SD 14 percentage points. |
| Confidence intervals | Correct Wilson implementation replaces mislabeled binomial plug-in interval. | Reference value and all-success/all-failure checks. |
| Paired results | Exact McNemar, paired/task-stratified bootstrap, Holm adjustment, per-training-seed mean/sample SD, strict pairing and controlled-factor checks. | Known discordant-table calculations; duplicate/unmatched seed/configuration rejection. |
| Offline diagnostics | Checkpoint-specific episode holdout, four seeded draws, training-action scales, per-episode metrics and resampling summaries. | Deterministic inference on a synthetic cache using a small actual head; no optimization. |
| Failure analysis | Common training-calibrated threshold, sensitivity check, post-step reward/state alignment, operational categories, commanded-versus-observed opening, episode-averaged attention. | Synthetic behavior categories and same-cohort consistency checks. |
| Reports | Optional table and performance-figure export from recorded outcomes; no copied historical rates. | Temporary synthetic table/PDF export test. |
| Cosmos | Registered `aloha/exp07` intermediate-checkpoint configuration with exp05/06 settings. | Configuration import/resolution; training remains optional and deferred. |
| Manuscript | Editable `manuscript/main.tex`, historical/pending evidence labels, corrected methods and claims. | Desktop compiler success with six-page guard. Visual preview review remains pending. |

## Real dataset preparation

The generated manifests live in the ignored asset directory; their source files
are hashed and the manifests contain their exact episode assignments. Recreate
them with the preparation command if the asset directory is not transferred.

| Dataset | Total episodes | Train / validation episodes | Train / validation frames |
|---|---:|---:|---:|
| ALOHA transfer-cube | 50 | 45 / 5 | 18,000 / 2,000 |
| LIBERO-Goal | 500 | 450 / 50, five held-out episodes per task | 57,490 / 6,238 |

ALOHA source identity includes the video as well as the three parquet shards.
LIBERO source identity includes all ten demonstration HDF5 files. No frames
were VLM-encoded during this preparation.

## Runnable checks

Run from the repository root in the existing `vla` environment:

```powershell
& C:\Users\SK\miniconda3\envs\vla\python.exe -m unittest discover -s tests -v
```

The implemented suite now contains 23 tests. It covers meaningful failure cases and
forward inference, not a real-data optimization run. Python compilation and CLI
help checks are also used; they do not establish simulator correctness.

Safe setting inspection, without training or creating a run directory:

```powershell
& C:\Users\SK\miniconda3\envs\vla\python.exe scripts/train.py --dataset aloha --exp exp05 --split-manifest asset/protocols/aloha/97fa699cb307366377cec2eb0f29795d967710f9ffe0fea9d1e1c50108b8d537.json --seed 0 --dry-run
& C:\Users\SK\miniconda3\envs\vla\python.exe scripts/train.py --dataset libero --exp exp05 --split-manifest asset/protocols/libero/4d1d6267e4270ac86caf491ea9b5fb630eb0e3a8a728822f96e94be9b3e3b0ad.json --seed 0 --dry-run
```

## Remaining gates before a full experiment

- Author review of this implementation, the actual split, and the manuscript.
- Cache/live encoder parity using actual model weights and observations.
- Simulator action-plumbing, gripper semantics, and harness checks in their
  appropriate environments.
- Measured cache, epoch, and rollout runtime/memory projection.
- Complete-protocol checkpoint roundtrip/resume using an eligible full cache;
  shortened-schedule real GPU resume has passed separately below.
- Visual manuscript review and official conference-template/author information.

Repeated-seed determinism is not a promise of bitwise equality across GPUs,
PyTorch versions, or nondeterministic kernels. Runtime identities and settings
must accompany later results.

The legacy `head_diagnostics.py`, `failure_taxonomy.py`, and existing paper plots
remain historical entrypoints; their hard-coded old run locations and frame
split must not be used for revised results. Use the new checkpoint-specific
diagnostic and paired-result commands instead. Legacy results are rejected by
the revised report path when protocol or provenance metadata are absent.

Automatic per-episode retry and partial-evaluation recovery are not implemented
in this pass. Evaluation exceptions abort rather than silently count as policy
failures or replace starts; only completed evaluations are eligible for reporting.

## Project structure check (18 October 2026)

Research and review records now live in `docs/`; the active manuscript outline
is `manuscript/outline.md`, while the open `manuscript/main.tex` and root
`plan.md` paths remain unchanged. Runtime loading uses `models/backbones.py`
through a regular import, and existing latent-study commands retain an import
shim. Root train/cache/evaluation commands forward to the canonical `scripts/`
pipeline, retaining supported PushT argument mapping and public imports.

The three new layout regression tests check argument forwarding, import safety,
legacy public imports, shared loader/resource paths, and live-model loader
delegation with mocked weights. Together with the 16 existing synthetic tests,
all 19 pass. No actual backbone weights are loaded by these checks. The encoder
fingerprint now hashes the canonical loader path; old protocol fingerprints
are not treated as compatible. Assets and historical reports were not moved
or rewritten. Remaining real-model/simulator gates above still apply.

Additional structure checks passed: 186 Python source files parse; local links
in the active guides and moved documents resolve; 11 legacy/canonical CLI help
commands run from a temporary directory outside the repository; ALOHA and
LIBERO exp05 seed-0 dry runs accept their existing prepared protocols. These
checks do not run the underlying training, encoding, or simulator commands.

## Bounded real-data CUDA observation (18 October 2026)

**Passed** on the NVIDIA GeForce RTX 5080 Laptop GPU, PyTorch 2.10.0 / CUDA 13.0.
The initial matrix/backward probe also passed; BF16 support is available.
The author authorized this sample separately from full paper retraining.

Run command, from the repository root in `vla`:

```powershell
python scripts/validation/observe_training.py --samples 16 --workers 6
```

Artifacts: `asset/runs/readiness/20261018_213803/observation.json`, including
per-step group gradient norms and parameter changes, phase timings, GPU peaks,
and resume comparison. Each benchmark directory contains its tiny HDF5 cache
and `baseline`/`resumed` checkpoints. This directory is ignored by Git.

| Benchmark/config | Batch | Workers per loader | Observed optimizer steps | Peak allocated / reserved GPU GiB | Resumed model max difference |
|---|---:|---:|---:|---:|---:|
| ALOHA / exp05 | 256 | 6 | 18 | 2.95 / 3.29 | 0.0 |
| LIBERO two-camera / exp05 | 128 | 6 | 18 | 2.40 / 2.82 | 0.0 |

Both runs used the actual registered heads in float32, real GR00T-derived
features stored/read through the production HDF5 path, and the production
trainer's AdamW, clipping, validation, checkpointing, and resume logic. Six
Windows worker processes loaded each training stream successfully. Each
benchmark encoded only 16 real training frames, using normalization from the
prepared protocol. Cache rows were then repeated to exercise the configured
batch size without encoding the full dataset.

Each uninterrupted run completed three smoke epochs / nine updates. A second
run was intentionally interrupted after atomically saving epoch 1 / step 3,
then resumed for six more updates. Final model tensors matched exactly for
both benchmarks; optimizer moments and scheduler state also matched under the
script's comparison checks. Gradients were finite, and both adapter and decoder
had nonzero gradients and weight changes. The adapter's first-step zero
gradient is expected from the zero-initialized DiT output projection; 16 of
18 observed steps per benchmark had effective adapter updates, and all 18 had
effective decoder updates. Training/validation losses remained finite.

The existing 19 regression tests also passed after this observation script was
added. No full policy run or simulator rollout was launched.

**Limits:** only exp05/GR00T was encoded here, covering the two primary head
geometries rather than all backbone families. The schedule was shortened to
three epochs and two warmup steps. Repeating tiny subsets means smoke train
and validation streams can contain the same source frames; their losses are
not validation-quality or task-performance estimates. The cache is explicitly
partial/smoke-only and has no eligible camera-ready protocol identity. This
checks short-run GPU/loading/update/resume correctness, not sustained thermal
stability, full-cache disk throughput, full encoding batch sizes, simulator
correctness, or the complete strict-protocol end-to-end path. Those remaining
gates above still apply before committing the long experiment matrix.

## Full-run precision correction (18 October 2026)

The author subsequently authorized one full ALOHA exp05 / seed-0 training run.
The first complete 20,000-frame cache took about seven minutes to encode, but
its batch-32 features failed comparison with the earlier batch-4 observation:
minimum cosine similarity was about 0.99407 and maximum relative RMS difference
was about 0.10878. Training was stopped before a completed epoch was recorded.
The attempt is not eligible paper evidence. Its compact batch-parity records are
retained in `docs/camera_ready_evidence/`; generated run outputs are superseded.

A controlled probe held input IDs/pixels and weights fixed while toggling
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`. Disabling
it made the tested batch-32 versus batch-4 and batch-1 outputs exactly equal,
including all captured hidden layers in the first-four comparison. Probe
results are in [`batch_probe.json`](camera_ready_evidence/batch_probe.json).
This isolates the numerical
setting in the tested build; it does not establish a robot-policy mechanism.
[PyTorch's numerical-accuracy documentation](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)
describes this BF16 setting and its default behavior.

The revised protocol now records
`vlm_bf16_reduced_precision_reduction=False` in its cache configuration.
The shared `VLAModel.encode_vlm` applies it for cached and live encoding and
restores the prior process setting even when encoding raises. Legacy defaults
are retained. A regression test checks application/restoration and the HDF5
test rejects a precision-setting mismatch; all 20 tests pass.

The corrected full pipeline is rebuilding from scratch in the standard exp05
directory, with a new source snapshot and encoder/cache identity. Precompute
now requires exact stored-BF16 equality with batch-one encoding at selected
first/last-batch rows before marking a protocol cache complete. A separate
full-cache validation checks metadata, token masks, finite states/actions, and
150 real action/state rows spanning the boundaries of all 50 demonstrations
before training can start. The 300-epoch maximum, training batch, seed, split,
and stopping rule have not been tuned in response to an outcome.

### Reusable experiment execution (18 October, 22:49 JST)

The subsequent corrected encoding process stopped at approximately 54% without
a recorded exception; both launcher and encoder were absent when checked. Its
partial HDF5 could not be opened. Its run directory was later cleared from disk;
only its compact BF16 diagnostic records were retained separately.

The author requested an execution restructure and a fresh start of one
experiment. `scripts/run_experiment.py` replaces the ignored per-attempt launcher.
Prepared manifests have immutable copies under `asset/protocols/<dataset>/`.
Shared caches are keyed by protocol, encoder configuration/code and actual source
weight/processor hashes under `asset/cache/<dataset>/<backbone>/`. Seed-specific
outputs use `asset/runs/episode_holdout/<dataset>/<experiment>/seed_NNN/`.
The cache key excludes the training seed; valid caches are reused on relaunch.
The canonical trainer now uses the same undated episode-holdout output layout.

Encoding writes a partial file and publishes it only after full provenance,
geometry, all-value finiteness, first/last batch-one parity, and ALOHA raw-data
boundary checks pass. Encoding batch size is stored for interpreting parity;
changing a training or encoding batch setting does not itself force feature
regeneration. Interrupted files and attempt logs/snapshots are preserved.
OS locks reject duplicate cache writers and duplicate runs of one seed, and
release automatically on process exit. Invalid published caches fail explicitly.
Complete matching runs are skipped; unfinished runs reuse the cache and resume
the canonical trainer's epoch checkpoint.

All **23 tests pass**, including actual synthetic HDF5 validation, no-encoding
cache reuse, preservation, failed-publication rejection, lock exclusion/release,
and rejection of missing parity, changed source hashes and nonfinite BF16 values.
CLI help and the production ALOHA training dry-run also passed; `git diff --check`
found no whitespace errors. No new dependencies were introduced.

At 22:49 JST one full ALOHA GR00T exp05 seed-0 pipeline was launched via the
native Windows process provider, outside the tool shell's process tree and with
its window hidden. The initial launcher PID was 41856. It uses the unchanged
20,000-frame, 45/5-episode split and registered 300-epoch maximum, batch 256,
warmup 300 and patience 50. Per-attempt logs and `pipeline.json` report actual
progress. Launch alone does not establish training completion or policy success.

### First full ALOHA replication cohort (19 October)

Three independently initialized GR00T heads completed the shared 45/5 episode
split, and each was evaluated on the same 50 simulator scene IDs without
attention instrumentation. Policy-sampling seeds were derived from the training
seed, so scene starts are matched but flow-sampling draws are not.

| Training seed | Best epoch | Best validation flow loss | Successes / 50 | Rate | Wilson 95% rollout CI |
|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 0.16551 | 5 | 10% | [4.3%, 21.4%] |
| 1 | 17 | 0.16341 | 18 | 36% | [24.1%, 49.9%] |
| 2 | 24 | 0.16080 | 7 | 14% | [7.0%, 26.2%] |

The per-policy mean is 20.0% with a sample SD of 14.0 percentage points across
three heads. These Wilson intervals describe rollout uncertainty conditional on
each trained policy; the SD describes variation across only three policies.
The current episode-holdout ALOHA GR00T results are far below the historical
61.25% result and vary sharply despite similar validation losses. The historical
result is therefore not reproduced by the revised training protocol.

For seed 0, attention-instrumented evaluation scored 6/50 versus 5/50
uninstrumented on the same starts. A 10-start epoch-50 checkpoint screen and a
separate 4-action replanning diagnostic each scored 0/10; neither justified
changing the primary 16/8 horizon or selecting a checkpoint from a small screen.
Primary ALOHA evaluation defaults to no trace; pass `--trace PATH` only for a
separately named attention diagnostic.

An independent holdout calculation reproduced seed 0's checkpoint validation
loss. Four seeded flow samples gave frame-weighted open-loop correlation 0.963
and normalized MAE 0.164; left-gripper open-action prevalence was 19.6% in the
targets and 20.5% in predictions. Cache/action alignment, saved normalization,
GR00T weight grafting (494 tensors, no missing/unexpected tensors), and the
simulator's 14-joint ordering were checked. These checks found no data-loader,
loss, or action-scaling defect that explains the weak closed-loop results.

### Checkpoint-selection diagnostic (19 October)

Seed 1's epoch-25 checkpoint had worse held-out flow loss (0.1759) than the
selected epoch-17 checkpoint (0.1634). It was evaluated post hoc because an
initial 10-start screen favored epoch 25; that screen was not used as a final
test. On the same first 50 starts, epoch 17 scored 18/50 and epoch 25 scored
24/50. To check whether this was cohort-specific, both checkpoints were then
evaluated on a separate, identically paired 50-start cohort (seed offset 2000):
epoch 17 scored 29/50 and epoch 25 scored 22/50. Within each cohort, the two
checkpoints used identical initial-state IDs, environment seeds, and policy
sampling seeds. Exact two-sided McNemar tests were not significant (first
cohort: 8 epoch-17-only versus 14 epoch-25-only successes, $p=0.2863$;
confirmation cohort: 17 versus 10, $p=0.2478$).

The direction reversed across cohorts, so these screens do not establish that
either checkpoint is better or validate selecting checkpoints by rollout
success. Keep epoch 17 as the prespecified minimum-validation-loss checkpoint
for seed 1 and do not replace the primary result with the more favorable
post-hoc screen. The results instead show that 50-start estimates are sensitive
to the sampled cohort and that this small pilot cannot support a stable
closed-loop checkpoint ranking. A future checkpoint-selection study should
define a separate rollout-validation cohort before training and preserve a
distinct final evaluation cohort; it must not tune checkpoints on the reported
test starts.

### Schedule completion before patience (19 October; superseded)

This intermediate setup was superseded by the author's warmup-based
clarification below before any full policy training used it.

At the author's request, the next recipe replaces OneCycle with the installed
Transformers linear-warmup/cosine-to-floor scheduler and fixed-beta AdamW.
The existing scheduled epoch budget, peak group rates, 300 warmup updates,
batch size and episode split are retained. `min_lr_ratio=0.01` is configurable;
ALOHA's decoder/adapter floors are 3e-6 / 1.5e-6. This floor is an initial
setting, not a task-specific optimum demonstrated by experiments.

Patience remains zero through all 300 scheduled ALOHA epochs (120 on LIBERO).
Thereafter the rate stays at its floor and 30 consecutive epochs without a
new minimum validation loss trigger stopping. New minima reset the count;
`best.pt` continues to track all epochs. Scheduler calls stop at the cosine
endpoint so the LR cannot rebound in the patience phase. Checkpoints and
history record the schedule identity, phase, LR and patience as appropriate;
incompatible scheduler resumes are rejected. Existing OneCycle policies and
their reported results are not changed. New output commands use
`asset/runs/schedule_then_patience/` and reuse compatible encoder caches.

The CPU trainer regression executes the complete schedule plus a patience
tail, injects a late validation improvement to check counter reset, verifies
the positive LR floor and 2:1 decoder/adapter ratio, and compares exact model,
optimizer, scheduler and patience state after interruptions before schedule
completion, during the tail, and after the late improvement. Reopening a
completed run performs no new updates; a changed LR floor is rejected.
All **24 tests and 6 subtests pass** in the `vla` environment. The production
ALOHA protocol dry-run and `git diff --check` pass. The real-data observation
script still stops at its three-epoch smoke checkpoint, preserving its bounded
scope; it has not been rerun for this schedule change. No new full policy
training or performance evaluation has started, so improved success remains
unproven.

### Current rule: Hugging Face schedule, patience after warmup (19 October)

The author clarified that patience should start when warmup optimizer updates
finish, not at the end of the full cosine schedule. The installed Hugging Face
`get_cosine_with_min_lr_schedule_with_warmup` remains the scheduler, with
300 warmup updates, existing peak rates, fixed-beta AdamW and a configurable
1% floor. `num_epochs` is the hard maximum: 300 for ALOHA, 120 for LIBERO.
The subsequent faster-decay update below gives ALOHA an earlier decay endpoint.
There is no training beyond the hard maximum.

The patience gate is `global_step >= warmup_steps`, evaluated at the normal
end-of-epoch validation. Before that point the non-improvement count stays
zero; afterward each validation without a new minimum increments it and a new
minimum resets it. ALOHA has 70 updates per epoch, so warmup finishes during
epoch 5 and that epoch's validation is the first eligible one. Patience is 30
validation epochs. Cosine decay keeps running while patience is enabled;
early stopping can occur before the full schedule finishes. Best-checkpoint
selection continues across all epochs, including warmup.

Checkpoints/manifests record the stopping rule, warmup boundary and first
eligible validation epoch. Resume rejects the superseded stopping rule as well
as an incompatible LR schedule; optimizer, scheduler, RNG and patience states
are restored for compatible runs. Future commands use
`asset/runs/hf_warmup_patience/` with the existing shared feature cache.
All **24 tests and 8 subtests pass**, including patience during decay, zero and
exact/mid-epoch warmup boundaries, reset on improvement, hard-cap termination,
LR-floor/group-ratio checks and exact interrupted resume. The ALOHA dry-run and
whitespace check pass. The bounded GPU observation retains its original
three-epoch cap. No new full training or closed-loop evaluation was launched.

### Faster ALOHA decay (19 October)

At the author's request, `lr_decay_epochs` now separates the cosine endpoint
from the hard training maximum. ALOHA GR00T and Qwen3-VL share a 100-epoch
endpoint, including the unchanged 300 warmup updates. The existing Hugging
Face scheduler reaches the 1% floor there, and scheduler stepping is then
disabled so later optimizer updates hold that rate without a cosine rebound.
The hard maximum remains 300 epochs and patience remains enabled immediately
after warmup; it can still stop the run before epoch 100. No minimum training
epoch floor was added. Other benchmarks default to full-budget decay, and
short smoke budgets cap the configured decay endpoint to the shorter maximum.

The decoder's actual configured schedule gives approximately 54% of peak at
epoch 50, 17% at epoch 75 and 1% at 100; it remains at 1% at epochs 150 and
300. The adapter retains half the decoder rate. Decay and maximum-step
metadata are recorded separately and incompatible resumes are rejected.
The 100-epoch endpoint is an initial diagnostic choice, not an experimentally
validated optimum. The existing feature cache identity is unaffected.

All **24 tests and 8 subtests pass**, including exact model/scheduler resume
after the floor is reached, continued floor holding, and changed-decay-horizon
rejection. Both ALOHA backbone configs resolve to the same 100-epoch endpoint.
No full training or policy-success evaluation was started.

### Authorized 100-epoch usability diagnostic (19 October; complete)

The author requested actual training and evaluation. Added a minimum-epoch
stopping gate for the diagnostic (`min_epochs=100`), while keeping patience
counting after warmup and the 300-epoch hard maximum. The CPU regression
verifies that resume continues before the minimum even if patience is already
exhausted; all 24 tests and 8 subtests pass. The real-data smoke observation
explicitly uses `min_epochs=0` to retain its three-epoch scope.

One new GR00T seed-0 run reused the validated cache and completed 100 epochs /
7,000 optimizer updates. The loss-selected checkpoint was epoch 22 (validation
flow loss 0.16536); epoch 100 had worse validation loss, 0.27574. A diagnostic
plan was saved before training: compare these candidates on 50 validation
starts with offset 3000, select the larger success count (ties favor the
offline-best checkpoint), then evaluate the selected policy on two untouched
50-start cohorts with offsets 4000 and 5000. All runs use uninstrumented 16/8
action horizons and the same 400-step limit.

| Candidate | Validation flow loss | Successes on matched 50-start selection cohort |
|---|---:|---:|
| Offline best, epoch 22 | 0.16536 | 1/50 (2%) |
| Later policy, epoch 100 | 0.27574 | 34/50 (68%) |

The paired table has 1 shared success, 16 shared failures, 0 early-only and
33 late-only successes; exact McNemar p is 2.3283e-10, conditional on these
checkpoints and starts. The late-minus-early difference is 66 points with a
paired bootstrap interval [52, 78]. This is a validation comparison used for
selection; it is not an independent final test or evidence across training
seeds. The epoch-100 checkpoint was selected, with its hash and predeclared
confirmation cohorts saved in `selected_policy.json` before confirmation.
Both untouched confirmation batches are complete:

| Cohort | Success | Wilson 95% interval |
|---|---:|---:|
| Offset 4000 | 25/50 (50%) | 36.6–63.4% |
| Offset 5000 | 35/50 (70%) | 56.2–80.9% |
| Combined | **60/100 (60%)** | **50.2–69.1%** |

Ninety attempts reached the lift stage; 30 of the 40 failed after lifting.
Maximum reward counts were 0: 3, 1: 7, 2: 30, 4: 60. These stage counts do
not identify a causal gripper or attention mechanism. The policy learned useful
task behavior but still fails 40% of fresh starts, so it is not dependable.
The two cohort rates also illustrate rollout variation for this one policy.

All 200 diagnostic rollouts cover 150 unique starts. Verified matched selection
sampling, disjoint confirmation starts, checkpoint hashes, counts and inference
settings. The final estimate excludes selection rollouts. Use the selected
epoch-100 `checkpoints/final.pt`, rather than the minimum-loss `best.pt`, for
this diagnostic. Its optional checkpoint `val_loss` metadata is absent; the
load banner's `nan` is missing metadata, while history records finite 0.27574.
Saved full evidence in the run's `diagnostic_results.json` and
`reliability_report.md`, with the curve in `training_loss.png`.

This identifies offline-loss checkpoint selection as a practical weakness;
longer training and the changed schedule were not isolated experimentally.
Before the paper's next matched comparisons, lock one common selection rule
for both backbones and keep rollout-validation starts separate from final tests.
No training-population reliability or robot-pretraining benefit is inferred
from this single-run diagnostic. The historical paper protocol is not a matched
comparison with this revised run.

## Prospective ALOHA replay gate

A same-seed replay of the diagnostic policy under the previous four-sample
offscreen renderer produced small pixel differences despite identical simulator
state. The differences propagated to later actions and could change success.
For the new matched study, the ALOHA evaluator explicitly sets MuJoCo offscreen
anti-aliasing samples to zero. A simulator-only probe repeated camera frames
exactly; the full-policy gate then repeated two fixed starts twice without video
and once with video. Every recorded state, action, and reward array matched
exactly (`asset/runs/readiness/fixed_endpoint_renderer0/replay_check.json`).
The gate validates these settings on this machine and checkpoint only. Two
starts do not estimate policy success, and the prospective visual-input setting
must be disclosed because it differs from the historical evaluation.
