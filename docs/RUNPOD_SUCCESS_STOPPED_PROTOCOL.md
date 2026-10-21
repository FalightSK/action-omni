# RunPod success-stopped study protocol

This document describes the experiment workflow and its implementation. Measured
results, checkpoints, run histories, and infrastructure status are stored separately.

## Study plans and initialization

- `configs/runpod_success_stopped_study.json` defines the online-encoder workflow.
- `configs/runpod_cached_success_study.json` defines the cached-feature workflow.
- `configs/runpod_fresh_cached_success_study.json` starts fresh adapter and decoder
  weights while reusing the pinned cached-feature producer. The training namespace
  is `runpod_success_stopped_fresh_cached_v1_<dataset>`; the cache producer namespace
  remains `runpod_success_stopped_v6_<dataset>`.

`cache_source_for_plan` checks the cache producer plan digest and requires the fresh
plan to preserve the scientific configuration. Cache reuse does not resume model,
optimizer, scheduler, or random-number-generator state from an earlier study.
Fresh startup uses `prepare_cached_study.py --reuse-only`; it does not encode
replacement features when a required cache is unavailable or incompatible.

Both GR00T N1.7 (`exp05`) and Qwen3-VL (`exp06`) remain frozen. Training updates only
the token adapter and action decoder. Training seeds are 3, 4, and 5. Backbones,
tokenizers, datasets, and split manifests are pinned in the plan.

## Training and selection

| Setting | ALOHA Transfer Cube | LIBERO Goal |
|---|---|---|
| Reference steps per epoch | 140 | 448 |
| Effective batch size | 128 | 128 |
| Development cohort | 50 episodes, reset seeds 20000-20049 | 10 starts per task, IDs 0-9 |
| Final cohort | 200 episodes, reset seeds 10000-10199 | 40 starts per task, IDs 10-49 |
| Tasks and cameras | Transfer Cube, top camera | 10 Goal tasks, agentview and wrist |
| Final episodes per policy | 200 | 400 |

The learning-rate schedule uses 5376 warmup updates, cosine decay to the 53760-update
reference point, then a constant floor at 0.01 of peak LR. The reference point is
not a maximum training budget. Development evaluation starts at epoch 100 and then
runs every 10 epochs. Patience is 30 elapsed training epochs without a strict
improvement, with no maximum epoch cap. Select the highest development success rate
and retain the earliest checkpoint on ties.

Development evaluation is asynchronous. Each policy may have at most one unresolved
development task. At the next evaluation boundary, resolve that task before
submitting another. A patience boundary requires the boundary checkpoint's result
before stopping. Apply each result to the evaluated checkpoint's epoch.

Training and validation demonstrations use episode-disjoint manifests shared across
backbones and training seeds. Use the saved normalization and protocol configuration
when loading a checkpoint; do not replace them with loader defaults.

## Checkpoint durability and resume

`checkpoint_io.py` publishes SHA-addressed immutable recovery revisions. The trainer
verifies an immutable recovery before committing controller state. Local `last.pt`
remains available, while periodic durable synchronization skips the mutable
`checkpoints/last.pt` alias. Resume selects a verified recovery revision and checks
saved configuration, source identities, encoder assets, protocol, and checkpoint SHA.

Recovery includes model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG state,
training and validation generators, optimizer step, and selection-controller state.
A conflicting checkpoint at the same epoch/step/revision must be rejected.

`check_cached_checkpoint.py` performs an isolated real one-update checkpoint round
trip for a selected dataset. Its diagnostic namespace and seed are separate from
the scientific training runs. A diagnostic does not count as a trained policy or a
rollout success measurement.

## Evaluation and provenance

`success_queue.py` publishes development and final tasks with checkpoint and
protocol identity. `runpod_eval_worker.py` validates and evaluates those tasks.
Final evaluation uses the selected checkpoint only after development selection;
final outcomes must not tune training or choose checkpoints.

Pairing claims require matching actual simulator-state and initial-observation
fingerprints for the corresponding training seed, task, and start ID. Preserve raw
outcomes and retain comparisons as descriptive when required fingerprints differ.
Reset-cohort separation does not establish that demonstration scenes are absent,
and Goal evaluation does not establish unseen-task generalization. Training seeds,
not individual rollout episodes, are the independent training replications.

`prepare_simulation_cohorts.py` builds the cohort manifest. Simulator checks are
available through `check_simulation_renderer.py` and `smoke_runpod_simulators.py`.
`plot_study_losses.py` and `analysis/runpod_paper_results.py` consume recorded data;
missing rollouts must remain missing rather than being inferred from losses.

## Operational entrypoints

`runpod_bootstrap.py` materializes a verified source archive and stages pinned assets
before starting the study runner. RunPod lifecycle actions require explicit
resource ownership and authorization. The optional local `runpod_queue_manager.py`
has a read-only `--dry-run` mode; trainer stopping is separately opt-in and gated by
durable completion evidence. It is not started by importing the module or merging
this code.

Keep generated results, model weights, feature caches, dataset files, credentials,
and machine-specific Pod configuration outside version control. The fresh/cache
study plans and implementation are reproducible code; generated run records are
separate experiment artifacts.
