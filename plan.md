# Camera-ready revision plan

Updated: 19 October 2026 (Asia/Tokyo).

Status: the corrected pipeline completed three independent ALOHA transfer-cube / GR00T N1.7 training runs and uninstrumented 50-start evaluations. Success was 5/50, 18/50, and 7/50 (mean 20.0%, sample SD 14.0 percentage points), far below the historical 61.25% result and variable across training seeds despite similar validation losses. Audits found no data-loader, loss, normalization, or action-scaling defect that explains the gap. A paired checkpoint screen reversed direction across two 50-start cohorts and does not justify changing the validation-loss checkpoint selection rule. `scripts/evaluate.py` now runs primary ALOHA rollouts without trace instrumentation unless `--trace PATH` is supplied. The episode-holdout pilot is diagnostic and does not complete the matched Qwen3-VL or LIBERO comparisons. See `docs/camera_ready_validation.md` for the evidence and `docs/CAMERA_READY.md` for the remaining run procedure.

Schedule update: the author's latest clarification supersedes the earlier schedule-completion rule. Use Hugging Face linear warmup/cosine decay to a 1% floor and enable patience at the first validation after `warmup_steps` optimizer updates finish. ALOHA uses 300 warmup updates (about 4.3 epochs), so patience is eligible from epoch 5; 30 non-improving validation epochs stop the run, subject to a hard 300-epoch maximum. At the author's request for faster decay, `lr_decay_epochs=100` now reaches the floor at epoch 100 and holds it afterward; both ALOHA backbones use this setting. This is a diagnostic preset and does not enforce a minimum training duration. Best-checkpoint selection spans all epochs. Use `asset/runs/hf_warmup_patience/` for subsequent runs and reuse the validated feature cache; the completed OneCycle pilot remains separate. This setup update does not launch retraining.

This is the current source of truth for the camera-ready revision. It supersedes the pre-review experimental and writing instructions in `manuscript/outline.md`, including its former "no new experiments" constraint and predetermined conclusions. The active manuscript outline is maintained at the top of that file. Existing results are historical evidence, not targets the new experiments must reproduce.

## 1. Objective and locked decisions

Produce a scientifically defensible, reproducible paper within the six-page conference limit. Strengthen the existing matched backbone comparisons before expanding the study, and make the final claims follow the new evidence.

Preserve the contributions reviewers valued:

- Controlled comparisons with matched settings within each benchmark.
- Closed-loop policy evaluation rather than reliance on demonstration loss alone.
- An interpretable analysis of observed ALOHA handover failures.
- A direct comparison between selected offline diagnostics and observed rollout performance.

### Decisions agreed with the author

| Item | Decision |
|---|---|
| Manuscript length | Six pages maximum. Budget references within those six pages unless official instructions explicitly allow otherwise. |
| Editable source | On 18 October the author explicitly requested a new LaTeX manuscript. `manuscript/main.tex` is now an editable revision draft; the original source is still useful for reconciliation when supplied. |
| Primary experiments | GR00T N1.7 versus Qwen3-VL-2B on ALOHA transfer-cube and two-camera LIBERO-Goal. |
| Training replication | Target three independent training seeds per primary configuration. |
| Validation | Replace random frame splitting with episode-disjoint validation. |
| Compute priority | Finish backbone replication first; camera controls are secondary; Cosmos is a time-gated extension. |
| Optional model extension | Cosmos-Reason2-2B on ALOHA, with the same protocol and three training seeds. |
| Deferred experiments | Push-T, Qwen3.5 capacity controls, additional model lineages, additional tasks, and a complete checkpoint-ladder rerun. |
| Interpretation | Report observed effects and failure patterns; do not infer a universal pretraining benefit, equivalence from a null, or a causal visual-attention mechanism. |
| Expected outcome | No required sign, magnitude, significance level, or preservation of the submitted headline. |
| Supplement | Do not depend on a supplement being permitted. Keep detailed artifacts in the repository for reproducibility, subject to venue rules. |

### Deadline and working cutoffs

- Conference deadline: **20 October 2026, 23:59 Anywhere on Earth (UTC-12)**.
- Equivalent deadline: **21 October 2026, 11:59 UTC / 20:59 JST**.
- Experiment cutoff: **20 October 2026, 20:59 JST**, leaving 24 hours for analysis, manuscript integration, and final checks.
- Target complete reviewable PDF: **21 October 2026, 12:00 JST**.
- Target submission-ready package: **21 October 2026, 18:00 JST**. Actual submission remains the author's action unless separately requested.

These are scheduling cutoffs, not a claim that all target experiments will fit. Measure throughput before committing the remaining compute budget.

## 2. Evidence reviewed and current state

### Inputs

- Submitted manuscript: `C:/Users/SK/Downloads/paper_76.pdf`, six pages.
- Four reviews supplied by the author in this conversation.
- Repository summaries: `README.md`, `docs/RESULTS.md`, `docs/OBJECTIVE.md`, `docs/REFERENCES.md`, and the existing `manuscript/outline.md`.
- Training, caching, dataset, evaluation, model-loading, and analysis implementations.
- Saved analysis summaries under `asset/analysis/`, including policy agreement, failure taxonomy, head diagnostics, cue probes, and drift analysis.

The manuscript and repository documents are evidence to inspect. Their previous instructions and conclusions do not override the author's revised scope.

### Available and missing artifacts

| Artifact | Audit finding | Consequence |
|---|---|---|
| GR00T, Qwen3-VL, and Cosmos weights | Present locally. | Backbone downloads are not an assumed prerequisite. |
| ALOHA and LIBERO demonstration data | Present locally. | A fresh, explicit split can be constructed. |
| Analysis JSONs, figures, and sampled latent files | Present. | Useful for audit and historical comparisons; not substitutes for missing primary records. |
| `asset/runs/` | Empty at inspection. | No local primary trained policies, full training caches, or per-episode result files were recovered there. |
| Full policy checkpoints and both original ALOHA rollout batches | Not found in the project search. | Plan to rebuild the core results unless compatible artifacts are subsequently supplied. |
| Editable manuscript source | Not found in the project search. | Await the author's source; protocol and experiment work need not wait. |
| GPU | RTX 5080 Laptop GPU, approximately 16 GB VRAM. | Profile actual memory and throughput; do not assume concurrent GPU jobs will be faster. |
| Python environments | Existing `vla` and `vla_libero` environments are present. | Reuse them; preserve the benchmark-specific simulator separation. |

Sampled representation-analysis HDF5 files are not full policy-training caches. Do not silently train on those smaller subsets and call the result a replication of the full-data experiment.

### Preserve existing work

At the audit, `scripts/analysis/plots_paper.py` already had local modifications. `manuscript/outline.md`, several review documents, and the cue/drift analysis scripts were untracked. Inspect and preserve these changes; do not reset, clean, or overwrite unrelated work. The old outline is retained as a historical record within `manuscript/outline.md`.

## 3. Reviewer-to-change map

| Feedback | Planned response | Why it improves the paper | Completion evidence |
|---|---|---|---|
| R1: paired ALOHA statistical analysis | Use matched scene starts, paired outcome tables, exact McNemar tests, and paired difference intervals for each trained-policy pair. | Uses the paired design instead of treating matched episodes as unrelated samples. | Regenerated paired tables and statistics with identifiable episode keys. |
| R1: explain the two independent runs | Separate independent head training, evaluation batches, and repeated policy sampling in both methods and captions. | Prevents evaluation resampling from being mistaken for training replication. | Explicit counts of trained policies and rollout batches. |
| R1: keep conclusions specific | Restrict statements to the tested frozen endpoints, heads, tasks, and simulation settings. | Aligns the conclusions with the actual external validity. | Abstract, introduction, results, and conclusion use the same scope. |
| R1: gripper pattern is not a proven cause | Describe the operationally measured failure category and distinguish it from a mechanism hypothesis. | Preserves useful failure analysis without causal overreach. | No claim that attention or a missing visual cue causes the gap. |
| R2: more independent training seeds | Target three independently initialized and trained heads per primary configuration. | Measures training variability, which additional rollout starts cannot measure. | Per-seed checkpoints, logs, configuration manifests, and results. |
| R2: mean and standard deviation across policies | Show each seed and report unweighted mean plus sample SD across trained policies. | Makes stability visible rather than concealing it in pooled episode counts. | Primary table and figure with labeled units of uncertainty. |
| R2: LIBERO training-run count unclear | State the independent training count separately for every LIBERO configuration. | Removes ambiguity about replication and camera controls. | Counts in the methods and results table agree with the saved runs. |
| R2: validation data and separation | Hold out whole demonstration episodes, stratified by LIBERO task; compute learned normalization statistics from training data only. | Removes overlapping trajectories/action chunks across training and validation. | Persisted split manifest and zero-overlap checks. |
| R2: validation loss versus test success | Define demonstration validation and simulator evaluation as different datasets, distributions, and quantities. | Makes model selection and final testing interpretable. | Clear definitions and checkpoint-selection rule. |
| R2: answer RQ2 directly | State whether the selected offline measures reflected the observed closed-loop comparisons under the revised protocol. | Gives a direct answer without claiming universal metric failure. | A short answer paragraph tied to the regenerated offline/rollout results. |
| R2: evaluate intermediate lineage checkpoint | Add Cosmos on ALOHA only after core replication and required camera work are secured. | Helps distinguish the intermediate adaptation from the subsequent robot-training stage. | Three Cosmos policies and matched evaluations, or an explicit statement that this extension was not completed. |
| R2: additional families/datasets or narrower claims | Narrow claims in the camera-ready paper; defer a second matched lineage and more tasks. | Avoids presenting a small study as general evidence across VLA systems. | Limitations explain what remains untested. |
| R2: consider Qwen3.5-0.8B | Defer the capacity-control experiment. | An unmatched smaller model would add another scientific question and consume limited time. | No implication that this suggested control was performed. |
| R3: broader backbones and task complexity | Improve replication now, narrow scope, and describe broader testing as future work. Optional Cosmos adds one checkpoint, not another lineage. | Strengthens reliability while being honest that broad generalization remains unresolved. | Discussion explicitly acknowledges the remaining external-validity gap. |
| R4: title implies too general an answer | Use a neutral empirical-study title. | Sets expectations consistent with one lineage and two simulated benchmarks. | Revised title and matching research questions. |
| R4: economic wording without cost accounting | Remove payoff/pay-off, what-it-buys, break-even, and unsupported pretraining-cost arguments. | Eliminates an implied economic analysis the experiment does not contain. | Text and figure-label search completed. |

Replication strengthens internal reliability; it does not itself satisfy R3's request for additional families and tasks. State this explicitly rather than claiming every request has been experimentally resolved.

## 4. Corrections required before new conclusions

### 4.1 Training runs versus rollout batches

The available repository documentation describes two ALOHA evaluation batches with disjoint scene seeds. It does not establish two independently trained heads per backbone. Treat independent-training replication of the submitted results as unverified unless the original logs/checkpoints demonstrate it.

Use these terms consistently:

- **Training seed:** a separate head initialization and stochastic training run.
- **Evaluation batch:** a group of rollout starts for an already trained policy.
- **Environment seed / initial-state ID:** identifies the sampled scene/start.
- **Policy sampling seed:** controls stochastic action generation during evaluation.

Do not compute a training SD from the two historical evaluation-batch success rates.

### 4.2 Methods must match the executed experiment

The current configuration/model audit gives approximately:

| Benchmark | Trainable head parameters | Decoder width | Batch size | Maximum epochs | Action/state handling |
|---|---:|---:|---:|---:|---|
| Two-camera LIBERO | 19,165,319 | 320 | 128 | 120 | Raw dataset actions/states in the current loader. |
| ALOHA | 14,014,990 | 256 | 256 | 300 | Normalized states/actions, with action denormalization at rollout. |

These describe the inspected current code, not proof of the missing historical runs' settings. Resolve the submitted manuscript's claim of one identical 19.2M head and one training recipe across all benchmarks. Preserve matched settings within each benchmark; disclose cross-benchmark differences rather than changing head capacity merely to preserve the old prose.

Measure and report actual image geometry, token counts, optimizer steps, completed epochs, and checkpoint epochs from each new run. Do not carry the LIBERO-specific optimizer-step count or square input description across to ALOHA without checking it.

### 4.3 Historical statistical and numerical issues

- The submitted ALOHA Wilson intervals, [56.4, 65.9] and [46.9, 56.6], overlap slightly. Delete the claim that they do not overlap. Interval overlap is not the paired hypothesis test.
- Saved ALOHA batch-1 discordant counts, 47 versus 25, give exact two-sided McNemar p approximately 0.0128. The pooled paired p-value cannot be recovered from the second batch's marginal success totals alone.
- Saved LIBERO discordant counts, 18 versus 13, give exact McNemar p approximately 0.473 rather than the unpaired p = 0.40. This is a historical conditional comparison, not inference over training seeds.
- A nonsignificant contrast does not prove equivalence or an absolute effect smaller than approximately seven points. Replace that assertion with the actual effect estimate and appropriate difference interval.
- Detecting a large camera effect does not demonstrate adequate power to detect a small backbone effect.
- The saved LIBERO ladder includes a positive GR00T point estimate at epoch 50. Distinguish "no corrected statistical evidence of an advantage" from "never leads." The new manuscript will not depend on unreproduced old ladders.
- P(lift | contact) is higher for GR00T in the submitted table. Remove the assertion that both earlier transitions point against it.
- The failure taxonomy uses a 200-episode instrumented cohort while the cited approximately 11.9-point handover gap uses the pooled cohort. The taxonomy's approximately 11.7-point excess cannot be said to explain the entire pooled gap by direct subtraction across cohorts.
- The existing general evaluator prints a binomial plug-in interval under a Wilson label. Use a verified Wilson implementation for per-policy success intervals and a separate paired interval for differences.

Historical p-values and percentages belong in an audit record if useful. They are not defaults for the revised tables or figures.

### 4.4 Mechanistic interpretation

The saved episode-grouped cue probe reports near-perfect prediction of opening from proprioception, as well as high prediction from both backbones' image features. This is expert-demonstration evidence and does not identify which cues the deployed policy uses. It nevertheless makes an exclusive visual-cue account unjustified.

- Remove "opening is cued only visually," "no proprioceptive correlate," and equivalent statements.
- Do not replace the visual explanation with a claimed proprioceptive cause.
- Similar routing diagnostics do not establish identical learned policies or identical solutions.
- Averaging error over dimensions/chunks can dilute a critical behavior difference; it does not mathematically guarantee that the metric cannot express it.
- Weight differences establish that released tensors differ; by themselves they do not isolate which adaptation stage or dataset caused a performance difference.
- Attention allocations, spatial probes, and failure categories are diagnostic observations, not causal interventions or automatically validated success predictors.

## 5. Execution phases and dependencies

| Phase | Deliverable | Depends on | Status |
|---|---|---|---|
| A | Finalized plan and active manuscript outline | Discussion and repository audit | Completed by this documentation update. |
| B | Original editable source and experiment provenance inventory | Author-supplied source; current audit | Inventory complete; original source not supplied. Author authorized a new LaTeX draft. |
| C | Corrected split, seed, metadata, normalization, and evaluation plumbing | Plan | Implemented and exercised by all three completed ALOHA pilot runs; automatic episode retries/partial recovery remain deferred. |
| D | Passing correctness checks and measured runtime projection | Phase C | Real CUDA training, cache validation, optimizer/resume checks, and simulator evaluations completed. All 23 tests plus 3 subtests pass; full-workload projections and long-duration stability remain unmeasured. |
| E | Primary backbone replications and matched evaluations | Phase D | Partial: three of twelve target policies (ALOHA GR00T seeds 0–2) and 50-start screens are complete. Matched Qwen3-VL ALOHA and all primary LIBERO runs remain pending. |
| F | Camera controls, then optional Cosmos extension | Primary results secured; runtime gate | Not started; retain as lower priority until the matched backbone matrix is complete. |
| G | Recomputed analysis and reviewer response matrix | Completed eligible runs | ALOHA pilot statistics, offline checks, failure diagnostics, and checkpoint screen are recorded. Cross-backbone paired analysis and full reviewer-response matrix remain pending. |
| H | Revised six-page manuscript and verified PDF | Draft plus final analysis | The active draft includes the pilot and diagnostic caveats and compiles successfully. Final results integration, visual review, author details, and source reconciliation remain pending. |

Do not create a new experiment-management framework. Extend the existing entrypoints and save simple machine-readable manifests/results. The author later authorized full training and evaluation after the initial preparation stage; the three-seed ALOHA GR00T pilot is now complete. Keep any further runs within the agreed core matrix, preserve the fixed protocol and cohort provenance, and do not add optional experiments before securing the matched backbone results.

## 6. Phase B: source and provenance

- [ ] Reconcile the new LaTeX draft with the original editable manuscript when the author supplies it. The new source already compiles in the desktop editor.
- [ ] Preserve the submitted PDF and identify the exact source version from which it was produced if possible.
- [ ] Check any subsequently supplied checkpoints/results against dataset, read layer, camera configuration, normalization, model state, split, and code version.
- [ ] Record the source of every recovered artifact. Do not merge unverified summaries with fresh run records.
- [ ] Preserve the old analysis summaries and submitted figures as historical artifacts.
- [ ] Record the existing environment/package versions, backbone file identity, and relevant code revision before new runs.
- [x] Create and update the editable LaTeX draft explicitly requested by the author, preserving historical-result labels and adding the completed ALOHA pilot. Independent source reconciliation remains pending.

Recovery rule: recovered old policies can help audit the submission, but a policy trained with the old frame split cannot be counted as an episode-holdout replication. New primary results must share the revised protocol. Data manifests have been prepared locally without loading or training a VLM: ALOHA has 18,000 training / 2,000 validation frames; LIBERO has 57,490 / 6,238. Exact identities are recorded in the manifests and `report/protocol_summary.json`.

## 7. Phase C: reproducible data and training protocol

### 7.1 Split definition

- Build stable raw sample identities before caching: dataset, task, demonstration episode, and frame index.
- Sort task and episode identities, then construct the split with `numpy.random.default_rng(42)`.
- For each LIBERO task, assign `max(1, floor(0.10 * episode_count))` whole episodes to validation; all remaining episodes train. Require at least two episodes per task and fail clearly otherwise.
- For ALOHA's 50 demonstrations, assign five whole episodes to validation and 45 to training.
- Persist exact episode membership and row mapping. Do not generate a new split for each training seed, backbone, or camera condition.
- Every frame and action chunk from an episode belongs to that episode's split. Retain existing within-episode padding; never create chunks across demonstrations.
- Store split identity in caches, checkpoints, diagnostics, and result manifests.

This controls training stochasticity on a fixed data split; it does not estimate variation across alternative demonstration splits.

### 7.2 Normalization and cache provenance

- For ALOHA, derive the existing mean/std normalization from training episodes only. Preserve the existing normalization convention and epsilon unless a correctness defect requires a documented change.
- Use those same statistics for training, validation, offline diagnostics, and simulator inference. Do not recalculate them per backbone or policy seed.
- Retain LIBERO's existing raw action/state representation. State this explicitly in the paper.
- Cache frozen-backbone embeddings once per dataset/backbone/camera condition and reuse them across training seeds.
- Record source model identity, read layer, final-norm handling, processor settings, token geometry, camera order, raw-row mapping, and normalization identity.
- A pure image/text embedding cache may be reused only after verifying that its rows and encoder settings match. Cached states/actions must also match the new training-only normalization. Do not infer compatibility from shape alone.
- Make metadata mismatches fail clearly rather than silently falling back to another cache or subset.

### 7.3 Training seeds and budget

- Use training seeds **0, 1, 2** for the primary comparisons.
- Seed Python, NumPy, PyTorch CPU/CUDA, initialization, data-loader workers/shuffling, dropout, and flow-matching noise. A split generator alone is not a training seed.
- Use the same seed identifiers across matched backbones and camera conditions. Distinct training seeds must initialize/train new heads; do not copy or fine-tune another seed's trained head.
- Preserve current per-benchmark head architecture, maximum epoch count, batch size, learning-rate recipe, chunk length, executed horizon, inference denoising steps, and configured early-stopping rule.
- Apply the same maximum schedule and stopping rule within a benchmark. Record actual completed epochs/optimizer steps; do not claim identical executed budgets if early stopping differs.
- Choose `best.pt` by minimum validation loss only, never rollout success. Resolve an exact validation-loss tie by retaining the earlier checkpoint.
- Use fixed validation random draws, with validation seed 4242, consistently across epochs and matched policies; restore training RNG state after validation.
- Save actual checkpoint epoch, best-checkpoint criterion, full configuration, training seed, split/cache identity, optimizer/scheduler state, and all RNG states needed for reproducible resume.
- Fix `final.pt` metadata to describe the actual completed epoch if training stops early.
- Never change a backbone's settings to improve its result after inspecting its test success.

### 7.4 Storage and minimal interfaces

Use a separate namespace such as:

```text
asset/protocols/<dataset>/<protocol_id>.json
asset/cache/<dataset>/<backbone>/<identity>/vlm_embeddings.h5
asset/runs/episode_holdout/<dataset>/<condition>/
  seed_000/                         # independent head, logs, results, traces
  seed_001/
  seed_002/
```

Keep large binaries in the existing ignored asset hierarchy. Keep concise protocol descriptions and final reporting documentation in the repository. Do not overwrite legacy paths by default.

Extend existing scripts only where necessary:

- `scripts/precompute.py`: consume the persisted split/normalization provenance and save episode/frame mappings with the cache.
- `scripts/train.py`: add explicit training-seed, split-manifest, and output-directory inputs; reuse existing cache controls.
- `scripts/evaluate.py`: retain checkpoint/output controls and add explicit policy-sampling reproducibility and complete evaluation metadata.
- `scripts/eval_libero.py`: handle each seed's checkpoint/output location correctly and reject material state-dict/configuration mismatches.
- Existing diagnostics: consume the persisted split instead of reconstructing the legacy frame-random split.
- Revised table/figure export is implemented in `scripts/analysis/paired_results.py --report-dir`, deriving values from eligible records. The existing `scripts/analysis/plots_paper.py` remains a historical entrypoint; its hard-coded old results must not be used for the revision.

Preserve existing callers where practical; use explicit new arguments for the camera-ready protocol rather than silently changing legacy defaults. Reuse installed numerical libraries and existing model loaders. Add no dependency unless the current environments cannot perform the required operation.

## 8. Phase D: correctness and feasibility gate

Complete small, meaningful checks before committing to long training. Smoke checks are not reported success-rate experiments.

- [x] Training/validation episode sets are disjoint and exhaustive in the prepared manifests; matched arms consume the same persisted assignments.
- [ ] Chunks remain within their originating episode and split.
- [x] ALOHA normalization is computed from training episodes only; synthetic holdout perturbation leaves those statistics unchanged. Live simulator roundtrip validation remains pending.
- [ ] Cached and live features agree for the same raw observations within the expected precision; compare real tokens, masks, padding, camera order, layer read, and final norm.
- [ ] Verify actual image sizes/token grids rather than copying the manuscript descriptions.
- [ ] Model/config/head shapes agree; unexplained missing or unexpected checkpoint tensors cause failure.
- [ ] Short same-seed training checks reproduce initialization and sample order; distinct seeds produce distinct initialization/training streams.
- [ ] Resume restores RNG state, optimizer/scheduler state, and epoch accounting; incompatible resumes are rejected.
- [x] Bounded real-data GPU observation: ALOHA exp05 at batch 256 and LIBERO exp05 at batch 128, six workers, finite gradients/effective updates, and interrupted resume matching the uninterrupted model exactly. This does not close the complete-protocol or long-duration gates above.
- [ ] Simulator observations, action scaling, executed horizon, and gripper polarity agree with the training representation.
- [ ] LIBERO demonstration replay and ALOHA action-plumbing checks have no unexplained regressions. A replay rate is not a policy ceiling, and the ALOHA plumbing check is not full success-detector validation.
- [x] Statistics checks reproduce known 2x2-table calculations and a verified Wilson formula; tests cover zero-discordance and all-success/all-failure cases.
- [x] Analysis rejects unmatched/duplicate episode keys and inconsistent cohorts instead of truncating arrays to equal length.

Measure cache production, representative training epoch time, and rollout throughput in both environments. Include model-load, validation, checkpointing, analysis, and I/O costs. Project the cost of the core set before adding optional work; use measured estimates rather than promising a fixed number of GPU-hours.

## 9. Phases E/F: experiment matrix and priority

### 9.1 Primary comparisons: 12 policies

| Benchmark | Condition | Existing config | Training seeds | Evaluation per policy |
|---|---|---|---|---|
| ALOHA transfer-cube | GR00T, one top camera | `aloha/exp05` | 0, 1, 2 | 200 shared scene starts |
| ALOHA transfer-cube | Qwen3-VL, one top camera | `aloha/exp06` | 0, 1, 2 | Same 200 starts |
| LIBERO-Goal | GR00T, two cameras | `libero/exp05` | 0, 1, 2 | 10 tasks x 20 initial states |
| LIBERO-Goal | Qwen3-VL, two cameras | `libero/exp06` | 0, 1, 2 | Same task/initial-state IDs |

Run one complete GR00T/Qwen pair at a time, alternating the backbone trained first across seeds. Finish seed 0 for both benchmarks before seed 1, then seed 2. This protects balanced coverage if the runtime budget is exhausted. A run includes its required evaluation; do not spend all remaining time training unevaluated policies.

The target primary set contains **2,400 evaluation episodes**. These are 12 trained policies, not 2,400 independent training replications. Reused evaluation starts across policies are repeated measures.

### 9.2 Secondary camera controls

After primary backbone replication is secured, train `libero/exp03` and `libero/exp04` with the same new protocol:

- First complete the matched one-camera pair for seed 0, giving one camera contrast per backbone against its two-camera seed-0 policy.
- Then complete seeds 1 and 2 if the runtime gate permits. Do not start an additional camera seed pair that is unlikely to finish evaluation by the cutoff.
- Three seeds for both one-camera configurations add six policies and 1,200 episodes, for 3,600 total primary-plus-camera episodes.
- If only seed 0 is completed, camera effects are secondary observations from one policy pair per backbone. Do not imply replicated camera effects or treat episode intervals as training uncertainty.
- If no new camera controls finish, omit the historical quantitative camera result from the new primary results. Explain the reduced scope; do not mix old frame-split camera policies with new episode-split policies to calculate a camera effect.

No backbone conclusion should depend on a single-camera distribution-mismatch argument. Quantitative camera contrasts concern the tested observation configurations, not a general proof that observability is the binding constraint.

### 9.3 Optional Cosmos extension

Consider Cosmos only after the core primary results and feasible camera controls are complete. Start it only if the projected full three-seed training/evaluation/analysis workload, with **50% runtime contingency**, fits before the experiment cutoff.

- Add an ALOHA configuration for Cosmos-Reason2-2B using the existing loader, layer-16 read with appropriate final-norm handling, and the same ALOHA head, processor budget, split, normalization, schedule, and evaluation starts.
- Train seeds 0, 1, 2; evaluate 200 starts per policy. This adds three policies and 600 episodes.
- Compare Cosmos to both Qwen3-VL and GR00T within the same training-seed identifiers.
- Report it as an intermediate checkpoint from the same lineage, not a new family or independent pretraining lineage.
- It helps examine where the endpoint contrast arises; it does not fully identify a particular pretraining dataset as the cause.
- Do not claim the intermediate checkpoint was trained unless its runs exist. If interrupted by the cutoff, archive partial artifacts and omit the incomplete extension from confirmatory claims.

### 9.4 Deferred alternatives and rationale

| Deferred addition | Reason for deferral | When it becomes worthwhile |
|---|---|---|
| Push-T | Adds task diversity but leaves the existing training-variance and one-lineage concerns unresolved. Current Push-T code uses another backbone/read configuration and is not a ready-made matched comparison. | After primary replication, with time for a matched backbone pair, episode holdout, three training seeds, and rollout evaluation. |
| Qwen3.5-0.8B | An unmatched capacity control addresses a separate question and introduces architecture/capacity differences. | A dedicated capacity study or a longer follow-up. |
| Second matched lineage | Potentially the strongest external-validity extension, but needs a verified ancestor/descendant pair and substantial new training/validation work. | A follow-up study with adequate compute and task coverage. |
| Full checkpoint ladders | Expensive and not required to resolve the main reviewer concerns. | A planned training-efficiency study, with replicated curves and consistent evaluation budgets. |
| Causal activation patching | Would investigate mechanism but requires additional intervention design and validation. | A focused mechanistic follow-up. |

Do not present an unrelated extra pretrained model as a matched robot-pretraining intervention.

## 10. Evaluation protocol

### Shared starts and sampling

- LIBERO: use the same 20 initial-state IDs per task, IDs 0 through 19 from the benchmark's available states, in every primary/camera policy evaluation. Record task ID, initial-state ID, instruction condition, and environment seed.
- ALOHA: persist 200 starts using the existing formula `(episode_index + 0) * 137 + 42`, with episode indices 0 through 199. Use the same list for all policies. Record the actual resulting environment seed, not just the offset or row number.
- Derive and record a deterministic policy-sampling seed from the benchmark, training-seed identifier, task ID, and initial-state ID using a stable documented derivation, such as NumPy `SeedSequence` with fixed integer components. Do not use Python's process-randomized `hash()`.
- Matched backbone comparisons use the same sampling-seed derivation. Training seed, environment seed, and action-sampling seed remain separate recorded fields.
- Primary LIBERO evaluation uses canonical instructions only. The old swapped-instruction zero-success claim must be removed from the new main evidence unless rerun under the new policies; a small smoke check cannot validate an all-task 0/200 claim.
- Keep maximum horizon, chunk execution, action representation, denoising steps, and success definitions fixed within a benchmark.
- Turn video off for the full evaluation; retain a small reproducibly selected set of qualitative examples after quantitative evaluation, without choosing examples solely for favorable outcomes.

### Trace the same ALOHA cohort

Record rewards, actual states, commanded actions, gripper state, and available per-replan attention in the primary ALOHA evaluation. The final success summary and failure analysis must be derived from this same rollout cohort. Do not add a separately resampled instrumented set and mix its denominators with the headline set.

An infrastructure exception is not automatically a policy failure. Log it and retry the same start with the same seeds at most twice. Persistent errors remain explicitly missing; do not substitute easier starts or discard ordinary failures. Incomplete required evaluations must be labeled incomplete, with their actual denominators.

### Result records

Each run must identify the dataset, condition, training seed, checkpoint identity and actual epoch, code/protocol identity, split/normalization/cache identity, environment versions, and execution settings. Every episode must have an unambiguous pairing key, actual scene/start seed, policy sampling seed, outcome, termination reason, and trace linkage when applicable.

Do not silently concatenate batches containing duplicate starts, infer pairing from list order, or drop extra rows using `min(len(a), len(b))`.

## 11. Phase G: statistical analysis

### 11.1 Primary reporting units

For every trained policy, report successes/attempts, success rate, selected epoch, actual training budget, and a verified Wilson 95% interval conditional on that policy and evaluation set.

For each configuration with three trained policies, report:

- All three seed-specific success rates.
- Unweighted mean success rate across policies.
- Sample SD across policies (`ddof=1`), in percentage points.
- Actual independent training count, separately from episode count.

For each paired backbone comparison, report each training seed's success-rate difference and the mean plus sample SD of those differences. Do not substitute SD across evaluation batches or an episode-pooled interval for training variability. With one policy, training SD is undefined, not zero; with two policies, label the estimate as based on two.

Three seeds give a limited estimate of training variability. Do not use thousands of rollout episodes to claim a highly precise population-level training effect.

### 11.2 Paired inference conditional on each trained pair

- Construct the 2x2 outcome table: both succeed, both fail, GR00T only, Qwen3-VL only.
- Use the exact two-sided McNemar test on discordant outcomes.
- Estimate the difference as mean paired success difference, in percentage points.
- Produce a paired percentile-bootstrap 95% interval using 20,000 resamples and analysis RNG seed 20261018. Resample whole matched episode pairs; for LIBERO, resample starts within each fixed task and retain equal task weights.
- Label these intervals as empirical evaluation-start uncertainty conditional on the fixed trained policies and tested tasks. They do not account for new training seeds, unseen tasks, or alternative data splits.
- Keep task-level LIBERO rates descriptive at 20 starts per task. Do not interpret isolated per-task differences as resolved discoveries.

Primary family: two benchmark contrasts times three training-seed pairs, six planned tests. Report nominal and Holm-adjusted p-values, with a planned family size of six. If a comparison is unavailable, mark it unavailable; use p = 1 internally only for conservative adjustment bookkeeping and never present that placeholder as an observed test.

Camera tests form a separate secondary family, up to two backbone-specific camera contrasts times three seeds. State which contrasts were completed and apply Holm adjustment within the reported secondary family. Cosmos-versus-stock and Cosmos-versus-GR00T contrasts form an explicitly exploratory extension family of six tests if all three seeds complete. Do not retrospectively describe these additions as preregistered.

### 11.3 Stage decomposition

- Derive contact, lift, handover, and terminal-success indicators from the same primary rollout trace, validating reward semantics.
- Report raw counts and denominators for unconditional reach rates and conditional transition rates.
- Treat conditional stage-rate differences as descriptive comparisons of each policy's reached-state cohort. These cohorts need not contain the same episodes.
- If a paired stage test is reported, test an unconditional per-start stage indicator and label it exploratory. Do not apply ordinary McNemar to differently filtered post-lift subsets.
- Do not infer that nonsignificant early-stage differences establish equality.
- Only merge handover and terminal success if the new traces demonstrate that distinction is absent in this evaluation set; otherwise report both.
- Report stage variation across training seeds and do not treat every stage of an episode as an independent trial.

## 12. Failure analysis and offline diagnostics

### 12.1 Failure taxonomy

- Verify picker/receiver identity and gripper polarity from the simulator definition and demonstrations; do not infer different arm assignments independently for different models without explaining why.
- Calibrate one common raw-unit gripper threshold from the training demonstrations, before inspecting compared policies' outcomes. Use the pooled training-demonstration receiver-gripper 5th/95th percentiles and the existing 0.6 interpolation convention.
- Use the same threshold for all backbones and seeds. Check sensitivity at interpolation fractions 0.5, 0.6, and 0.7; report if the qualitative pattern changes.
- Use operational labels: "receiver never crosses the opening threshold after lift," "opens but does not recross the closing threshold," and "opens and subsequently closes without completing transfer."
- Do not call the third category "closed on nothing" unless object/contact measurements actually establish that interpretation.
- Distinguish commanded opening from observed gripper opening where the trace permits; a closed gripper alone does not prove the policy never commanded opening.
- Report failure-mode counts per all episodes and per post-lift episode, with explicit denominators and training-seed variability.
- Report rates descriptively by default. Do not preserve historical mode-level p-values when the cohorts or classifier change.
- Check counts sum to the full cohort and connect every plot to those same counts.

The claim supported is a difference in observed failure patterns, if it survives replication and threshold sensitivity. It is not proof that a missing visual cue, attention allocation, or pretraining dataset caused that difference.

### 12.2 Attention and supporting probes

- Average the first five available post-lift replans within each episode before summarizing across episodes.
- Record the number of available replans and episodes used; avoid giving long failed episodes more statistical weight.
- Keep successful/failed strata descriptive and state that outcome selection changes the compared populations.
- Do not treat replans as independent training or episode replicates.
- Retain attention/probe material only if it clarifies an observation within the six-page budget. It is optional, not a causal explanation required to complete the paper.
- Preserve the cue-probe and drift results in the evidence record, including results that weaken the preferred mechanism. Do not replace one unsupported causal story with another.

### 12.3 Offline diagnostics

- Recompute validation flow-matching loss and existing open-loop action error/correlation on the new held-out episodes.
- Use the same episode/frame identities, action scaling, and seeded stochastic draws across compared backbones. Use all held-out episodes; any frame subsampling must be deterministic and documented.
- For diagnostic Monte Carlo estimates, use four fixed stochastic draws per selected sample and record the draw seeds. Keep these diagnostic estimates distinct from the single fixed-draw validation measure used for checkpoint selection.
- Summarize variability by demonstration episode, not by pretending adjacent frames are independent. ALOHA has only five held-out episodes, which limits precision.
- Define normalized action error explicitly and keep its scale identical within a comparison. Do not compare loss magnitudes across benchmarks as if they were on a common scale.
- Distinguish validation loss used for selection, recomputed diagnostic loss, and simulator test success.
- Treat attention and positional-encoding sensitivity as representation/usage diagnostics, not standard success predictors whose failure has been established by two models.
- Do not claim that a small percentage difference is statistically zero without its uncertainty.
- Do not calculate a general predictive correlation from only two backbone endpoints or treat correlated seeds/checkpoints as independent models.

Answer RQ2 directly from the regenerated results: whether the selected offline measures reflected the direction and size of the observed closed-loop contrasts, with any failures and exceptions stated. If the new diagnostics distinguish the policies, revise the offline-mismatch conclusion accordingly.

## 13. Phase H: six-page manuscript revision

### 13.1 Title and research questions

Working title: **An Empirical Study of Robot Pretraining in Frozen VLA Backbones**.

Research questions:

1. Under matched settings within each benchmark, how do the tested stock and robot-adapted frozen backbone endpoints compare in closed-loop performance across independently trained heads?
2. Do the selected offline validation measures reflect the observed closed-loop comparisons in these settings?

If Cosmos completes, add a subsidiary question about the intermediate checkpoint's position in the observed ALOHA comparison. Do not rewrite the paper as a broad study of multiple pretraining families.

### 13.2 Section allocation

| Content | Target pages | Required content |
|---|---:|---|
| Introduction and related work | 1.00 | Specific questions, controlled study, relation to existing evaluation findings, bounded contributions. |
| Experimental design and statistics | 1.25 | Per-benchmark architecture/recipe, episode split, independent seeds, selection rule, evaluation starts, pairing, reporting units. |
| Results | 2.25 | Replicated backbone comparisons; eligible camera controls; observed failures; offline comparison; optional Cosmos only if complete. |
| Discussion, limitations, conclusion | 0.75 | Direct answers, uncertainty, cross-testbed confounds, scope, remaining work. |
| References | 0.75 | Verified, relevant primary references. |
| Total | 6.00 | Layout may redistribute space while staying within the hard limit. |

Do not reduce font sizes or margins to conceal excessive content. Remove redundant material first. Verify official venue requirements once the author supplies the source/venue information; do not invent supplement permissions or camera-ready rules.

### 13.3 Tables and figures

- Primary results table: benchmark, backbone/cameras, independent training count, per-seed success rates, mean +/- sample SD, and evaluation count per policy.
- Primary performance figure: individual training-seed points and aggregate summaries; show paired differences where space permits. Label training SD separately from episode intervals.
- Failure figure/table: regenerated counts and rates from the same ALOHA cohort, showing training-seed variation and actual denominators.
- Offline comparison figure/table: revised held-out measures alongside closed-loop outcomes, with defined units and limits on interpretation.
- Optional Cosmos rows should fit into the existing result structure rather than requiring another long section.
- Do not retain old checkpoint ladders merely to fill space. Remove unsupported training-efficiency conclusions with them.
- Use the submitted convention consistently: GR00T blue and Qwen3-VL orange. Update script colors/captions together if needed; distinguish conditions using labels/markers as well as color.
- Generate all quantitative labels from final analysis records. No manual p-values or copied historical rates.

### 13.4 Required wording changes

| Previous framing | Revised rule |
|---|---|
| "Robot pretraining is not a general prior" | Describe the observed consistency or inconsistency of benefit in the tested settings; do not assert a universal conclusion. |
| "No advantage at any stage" | State the evaluated policy/seed results and uncertainty; remove reliance on unreproduced historical ladders. |
| "The head reaches the same solution" | Describe similar measured diagnostics without asserting identical policies. |
| "Observability is the binding constraint" | Report the measured camera contrast, if rerun, without identifying an exclusive cause. |
| "The backbone effect is below about seven points" | Report the actual paired effect interval; do not claim equivalence without a justified equivalence analysis. |
| "A single mode explains the whole gap" | Report same-cohort failure differences and acknowledge the descriptive decomposition and other modes. |
| "Opening is cued only visually" | Remove; the available evidence does not establish exclusive visual dependence. |
| "Evaluation found the cause" | Evaluation identified an associated failure pattern. |
| "The arithmetic guarantees offline blindness" | Aggregation may dilute behavior-critical differences; whether it does so here is empirical. |
| "Offline metrics do not predict success" | Selected offline diagnostics were or were not sufficient for these observed comparisons, according to the revised evidence. |
| "Replicated" without a unit | Specify independent training replication or a second evaluation batch. |
| "Payoff," "what it buys," cost is "repaid" | Use neutral descriptions of measured performance differences. |

The endpoint contrast includes multiple adaptation stages. Without the optional intermediate comparison, do not attribute the entire difference specifically to the final robot-data stage. Even with Cosmos, scope the interpretation to these released checkpoints and this protocol.

### 13.5 Limitations that must remain visible

- One backbone lineage; Cosmos does not change that count.
- Few independent head-training seeds and one fixed demonstration split.
- Frozen-backbone policies only; full fine-tuning may yield another ordering.
- Per-benchmark head and training-recipe differences, plus other simultaneous cross-testbed differences, prevent causal attribution of the benchmark reversal to a single task property.
- Simulation only, on LIBERO-Goal and ALOHA transfer-cube.
- No causal intervention establishes the source of a gripper/attention difference.
- Null results are not equivalence, and fixed tested tasks do not establish unseen-task generalization.
- Very small validation-episode counts constrain offline uncertainty estimates.

### 13.6 References and final assembly

- Verify the metadata and supporting claims for every retained reference against primary sources; reference verification is not yet complete.
- Check model/version names and cited released configurations, distinguishing repository defaults from the actual checkpoint used.
- Update abstract rollout/policy counts from the completed run manifest. Do not count shared starts as independently trained policies or double-count the same traced episode as an extra evaluation.
- Reconcile every number appearing in multiple places through generated analysis outputs.
- Replace anonymous placeholders with author information supplied by the author and follow the venue's camera-ready requirements.
- Produce an internal reviewer-response matrix listing each concern, the actual change, manuscript location, evidence, and any explicitly deferred request. Do not claim an experiment completed because it was planned.

## 14. Deadline management and failure rules

### Scheduling

1. Start with protocol corrections and a bounded correctness/throughput check. Use the initial implementation window to establish a realistic projection.
2. Reserve time for 12 core policies plus their required evaluations before allocating optional experiments.
3. Work in complete matched pairs and complete training-seed rounds across the two primary benchmarks.
4. After core replication, run feasible camera controls. Consider Cosmos only if its full three-seed set plus analysis and a 50% runtime contingency fits before 20 October, 20:59 JST.
5. Stop starting optional experiments as soon as that gate fails. Do not trade the final verification day for an additional favorable-looking result.
6. Freeze the reporting set at the experiment cutoff; analyze, integrate, and verify the manuscript during the final 24 hours.

### Fallbacks

| Situation | Required response |
|---|---|
| Three core seeds do not fit | Keep complete matched seed pairs, report actual counts, and explicitly retain the training-variance limitation. Prioritize core backbone pairs over optional work. |
| A seed is partially complete | Preserve its artifacts and explain status. Use completed matched pairs for paired comparisons; never select which seeds to include based on outcomes. |
| Camera controls do not fit | Reduce them to the completed matched seed-0 pair or omit them. Remove unsupported camera magnitudes from the new main results. |
| Cosmos does not fit | Omit it; state that the intermediate comparison remains future work and maintain endpoint-level attribution. |
| Original source arrives late | Use the explicitly authorized new LaTeX draft and reconcile with the original source later. Keep author details and unsupported results pending. |
| A correctness defect changes the protocol | Document it, version the protocol, and rerun all affected matched comparisons. Do not pool incompatible versions. |
| Core results change or reverse | Update the thesis, abstract, figures, and conclusion. Do not adjust settings to recover the old headline. |
| New validation metrics reflect rollout differences | Report that result and narrow or remove the offline-mismatch claim. |
| A run fails technically | Resume only with compatible saved state; log retries and missing evaluations. Do not silently count crashes as ordinary policy failure or replace difficult starts. |
| A venue rule conflicts with assumed layout | Follow the verified rule, preserve the six-page hard limit, and reduce optional content first. |

## 15. Completion checklist

### Experimental integrity

- [ ] No demonstration episode or action chunk crosses training and validation.
- [ ] All corresponding conditions use the same split and training-only normalization.
- [ ] Independent training counts are explicit and backed by separately trained checkpoints/logs.
- [ ] Every headline policy result has a checkpoint, configuration, seed, split/cache provenance, and complete episode records.
- [ ] Pairing keys, actual start IDs, sampling seeds, missing episodes, and retries are recorded.
- [ ] No historical frame-split policy is pooled with the episode-holdout primary set.
- [ ] All settings that differ across benchmarks are documented rather than described as identical.

### Analysis integrity

- [ ] Mean +/- sample SD uses trained policies as the unit.
- [ ] Conditional episode intervals and training variability are labeled separately.
- [ ] Paired tests and intervals regenerate from the underlying outcome records.
- [ ] Multiple-testing families, nominal p-values, and adjusted p-values are explicit.
- [ ] Failure categories, cohort sizes, denominators, and headline success counts agree.
- [ ] Threshold sensitivity and command-versus-observation distinctions are checked.
- [ ] No frame/replan count is presented as independent training replication.
- [ ] Offline diagnostics use the new holdout and report their actual uncertainty and limitations.
- [ ] Figures contain no hard-coded historical values.

### Manuscript and delivery

- [ ] Working title and all claims fit the actual completed study.
- [ ] RQ1 and RQ2 have direct, evidence-based answers.
- [ ] Cost-benefit language, unproven causal explanations, and false equivalence bounds are removed.
- [ ] All four reviews have a documented response; deferred requests are identified honestly.
- [ ] References and model/configuration citations are verified.
- [ ] Abstract, tables, figures, captions, and conclusion agree numerically.
- [ ] The final PDF is visually checked at normal reading size and remains within six pages.
- [ ] Author details and required submission metadata are correct.
- [ ] The final source, analysis summaries, protocol description, and reviewer-response matrix are preserved.

## 16. Progress log

| Date | Update |
|---|---|
| 2026-10-18 JST | Inspected the submitted six-page PDF, four reviews, current code/configuration, available data/weights, and analysis summaries. Identified missing primary run artifacts and methods/statistics/interpretation discrepancies. |
| 2026-10-18 JST | Author chose episode holdout, backbone replication first, optional expansion only within the deadline, six-page maximum, and supplying the original manuscript source. |
| 2026-10-18 JST | Compared Push-T with another pretraining comparison. Prioritized core replication and optional Cosmos; deferred Push-T and another matched lineage to follow-up work. |
| 2026-10-18 JST | Created this detailed current plan and updated the active manuscript outline. At that point new experiments and manuscript edits remained pending. |
| 2026-10-18 JST | Author authorized code implementation and a new LaTeX draft, explicitly deferring retraining until confirmation. Implemented the opt-in protocol, metadata/seed/checkpoint changes, paired reporter, checkpoint-specific diagnostics, and optional Cosmos config. Prepared real-data episode splits without encoding or training. |
| 2026-10-18 JST | Wrote `manuscript/main.tex` with historical/pending evidence labels. Desktop compiler succeeded with a six-page guard. Synthetic validation covers 16 tests, including inference without optimization and canonical padding across batch layouts. Live parity/harness, optimizer/resume and throughput checks remain pending; implementation/reviewer documentation records these limits. |
| 2026-10-18 JST | At the author's request, realigned the existing open manuscript with all six pages of `paper_76.pdf`: restored the original seven-section progression, six numerical result tables, and the same 13 cited works in original order. Verified primary reference records, corrected RoboVLMs' first-author metadata and identified N1.7 by its released model card. Retained reviewer-driven statistical/causal corrections and historical/pending labels; figures are not yet reconstructed. Citation/label checks and the desktop compiler with six-page guard passed. No training or rollouts were launched. |
| 2026-10-18 JST | Restructured with the requested ponytail skill: research/review/workflow documents moved into `docs/`, outline moved to `manuscript/outline.md`, runtime backbone loading moved into `models/backbones.py` with an import compatibility shim, and root CLI wrappers now delegate to the canonical pipeline. Updated links, structure guides, direct dependency declarations, and provenance hashing. The active `plan.md`, open `manuscript/main.tex`, configuration import paths, assets, and existing command locations are preserved. All 19 synthetic/import/dispatch checks pass; real-model/simulator gates and retraining remain deferred. |
| 2026-10-18 JST | Author separately authorized a short training/readiness check. Added and ran `scripts/validation/observe_training.py` on actual CUDA with 16 real GR00T feature frames per benchmark, production-size ALOHA/LIBERO heads and batches (256/128), six HDF5 workers, and shortened schedules. All 36 observed optimizer steps across both benchmarks had finite gradients; both optimizer groups updated. Interrupted/resumed final models matched uninterrupted training exactly; optimizer/scheduler comparisons passed. Peak allocated GPU memory was 2.95/2.40 GiB. Evidence is `asset/runs/readiness/20261018_213803/observation.json`; validation/workflow docs record limitations. All 19 existing tests still pass. Full caches, full policy retraining, simulator rollouts, and long-duration stability remain outside this check. |
| 2026-10-18 JST | Author requested one completed experiment training run. The initial ALOHA GR00T exp05 seed-0 pipeline used the old `camera_ready_2026_10_18` layout and was later stopped before a completed epoch because the batch-parity audit failed. It was superseded by the reusable-cache runner; other policies/rollouts were not launched. |
| 2026-10-18 JST | Full-cache parity checks caught BF16 batch-dependent features before a completed training epoch. Stopped and preserved the attempt as `exp05_groot_transfer_attempt_20261018_2204` (ineligible for results). A controlled GPU probe found exact batch-1/4/32 agreement when BF16 reduced-precision accumulation was disabled. Applied that setting to the shared cached/live encoder, recorded it in cache configuration, added exception-safe restoration and a regression test (20 tests pass), and added mandatory real batch-one parity checks before cache completion. Restarted the same condition/seed from scratch with a new cache identity and source snapshot; training settings/outcome selection remain unchanged. |

| 2026-10-18 JST | At the author's request, replaced the ignored one-off launcher with tracked `scripts/run_experiment.py`: immutable protocol copies, content-addressed caches shared across seeds, seed-specific outputs, timestamped logs/source snapshots, atomic cache publication after validation, OS duplicate-run locks, and cache/checkpoint reuse on restart. Preserved the interrupted 54% attempt and marked it ineligible. All 23 tests, CLI help, production dry-run and whitespace checks passed. Launched the same full ALOHA GR00T exp05 seed-0 condition at 22:49 JST via the native Windows process provider (PID 41856), with hidden UI and logs outside the tool shell's process tree. Live state is under `asset/runs/episode_holdout/aloha/exp05_groot_transfer/seed_000/`; full training and closed-loop success are not yet completed results. |
| 2026-10-19 JST | Completed the full ALOHA GR00T seed-0 run and matched 50-scene evaluation. The best checkpoint (epoch 17) scored 5/50 untraced and 6/50 with attention tracing; an epoch-50 10-scene screen scored 0/10. A 4-action replanning diagnostic also scored 0/10. Recomputed holdout loss and per-dimension action metrics, verified action ordering/scaling, cache alignment and GR00T tensor grafting; no implementation defect explains the weak closed-loop score. Changed primary ALOHA evaluation to be uninstrumented by default (`--trace` is opt-in), fixed the no-trace path and CP874 warning, and recorded the evidence in `docs/camera_ready_validation.md`. The single policy is not yet a usable replication of the historical 61.25%; next work is targeted policy/training investigation, then the planned independent seeds. |
| 2026-10-19 JST | Completed the planned three ALOHA GR00T training seeds and uninstrumented 50-scene evaluations. Success was 5/50, 18/50, and 7/50 (mean 20.0%, sample SD 14.0 percentage points); all used the same scene IDs, while policy-sampling seeds differed by training seed. The historical 61.25% is not reproduced under episode holdout, and nearly identical offline losses do not explain the large closed-loop spread. Updated run records and validation docs; revise manuscript status/results accordingly. No code-level data/loss/scaling defect was found, so do not tune to recover the historical rate without a new justified protocol. |
| 2026-10-19 JST | Compared seed-1 epoch 17 (minimum validation loss) with post-hoc epoch 25 on two separately sampled, paired 50-start cohorts. The observed direction reversed (18 vs 24 successes, then 29 vs 22); exact McNemar p-values were 0.2863 and 0.2478. No checkpoint replacement is justified. Recorded the cohort-sensitive result and a prospective separate rollout-validation/final-test recommendation in `docs/camera_ready_validation.md`; checkpoint selection remains validation-loss based for the completed pilot. |
| 2026-10-19 JST | Author requested schedule completion before patience and a standard training schedule. Replaced OneCycle with the installed Transformers warmup/cosine-to-floor scheduler and fixed-beta AdamW; retained existing peak rates, warmup updates and scheduled epoch budgets. Added a configurable 1% floor, post-schedule patience, phase/LR/count logging, and explicit incompatible-schedule resume rejection. CPU regression checks full schedule completion, late improvement resetting patience, LR-floor/adapter-ratio preservation, exact interrupted resume before/after decay and after a late improvement, and no restart after completed patience. Full retraining has not started. |
| 2026-10-19 JST | Author clarified that patience must begin after warmup optimizer updates, not after the full cosine schedule. Updated the Hugging Face trainer to gate patience on `global_step >= warmup_steps`, restore the hard epoch maximum, remove the extra tail, and record/reject mismatched stopping rules on resume. ALOHA's first eligible validation is epoch 5. Updated CPU regression coverage for mid-epoch/exact-boundary/zero warmup, counter reset, early stopping during decay, hard cap/LR floor, exact resume, and incompatible stopping-rule rejection. This supersedes the immediately preceding unrun setup; no full policy training has started. |
| 2026-10-19 JST | Author requested faster learning-rate decay. Added an independent `lr_decay_epochs` setting, with ALOHA's shared GR00T/Qwen preset at 100 epochs and the hard maximum at 300. Hugging Face cosine reaches the 1% floor at 100 and remains fixed thereafter; patience still starts after warmup. Other benchmarks keep their full-budget decay unless configured otherwise. Tests cover floor holding, resume after reaching it, and changed-horizon rejection; all 24 tests and 8 subtests pass. This does not set a minimum epoch floor, start retraining, or establish improved closed-loop performance. |

Update the progress log, completed checkboxes, actual run counts, runtime projections, and deviations as execution proceeds. Keep planned, running, completed, failed, and deferred work distinct.
