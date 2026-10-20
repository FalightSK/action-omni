# Camera-ready matched-study plan and results

Updated 20 October 2026. All six ALOHA policies in `camera_ready_fixed_endpoint_v1` are complete and pass the locked verifier; they remain separate from the new update-matched follow-up. The RunPod follow-up is now deployed on two B300 MIG Pods, one for ALOHA and one for two-camera LIBERO. At the latest live check, both workers were still precomputing GR00T (`exp05`) features for seed 3; no new policy training or evaluation had started. See the [team continuity guide](docs/team_continuity_guide.md) for live paths, monitoring, recovery, and finalization instructions.

## Current status and scope

- The six ALOHA policies are complete and passed the locked result verifier: GR00T N1.7 and Qwen3-VL, each trained with seeds 3, 4, and 5.
- Each policy was evaluated on 200 starts. Within each training seed, both backbones used the same initial states, environment seeds, and policy-sampling seeds. The fixed checkpoint was epoch 100.
- The active ALOHA worker was at about 19/625 precompute batches and the LIBERO worker at about 10/2,656 at the latest log check. Their `study_status.json` files show `running` / `precompute`, with no completed policies. These are live-run observations, not final experiment results.
- The RunPod protocol and plan are locked. Both workers are detached with `nohup`; closing the laptop does not stop them. Do not launch duplicates. Run artifacts sync to `/workspace-global/action-omni/study_v1/runs/`; active embedding caches remain on each Pod's local disk and must be regenerated if lost.
- Preserve the previously requested battery gate for new local heavy tasks: after charge falls below 35%, wait until it reaches 70% before starting another one. This gate does not interrupt or pause the already-running cloud workers.
- Detailed verification and calculation notes are in [docs/aloha_results_analysis.md](docs/aloha_results_analysis.md). The submitted baseline is [paper_76.pdf](C:/Users/SK/Downloads/paper_76.pdf).
- The ALOHA-only paired test output was regenerated from the six verified evaluation files: `asset/runs/camera_ready_fixed_endpoint_v1/paired_aloha_results.json` (input manifest: `paired_aloha.json`).

## Full result

Rates are successes divided by 200 rollouts. Per-policy intervals are 95% Wilson intervals. The paired difference is Qwen3-VL minus GR00T; its interval is a 20,000-resample paired-bootstrap 95% interval. The McNemar test uses discordant outcomes from the matched starts.

| Training seed | GR00T N1.7 | Qwen3-VL | Qwen − GR00T, paired 95% CI | Qwen-only / GR00T-only | Exact McNemar p | Holm p across 3 ALOHA seed pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 85/200 (42.5%; 35.9–49.4%) | 104/200 (52.0%; 45.1–58.8%) | +9.5 pp (+0.5 to +18.5) | 52 / 33 | 0.05025 | 0.05025 |
| 4 | 107/200 (53.5%; 46.6–60.3%) | 61/200 (30.5%; 24.5–37.2%) | −23.0 pp (−31.5 to −14.5) | 20 / 66 | 6.67 × 10⁻⁷ | 2.00 × 10⁻⁶ |
| 5 | 104/200 (52.0%; 45.1–58.8%) | 74/200 (37.0%; 30.6–43.9%) | −15.0 pp (−23.5 to −6.5) | 27 / 57 | 0.001404 | 0.002809 |

Across independently trained policies:

| Backbone | Seed 3 | Seed 4 | Seed 5 | Mean success ± sample SD across training seeds |
|---|---:|---:|---:|---:|
| GR00T N1.7 | 42.5% | 53.5% | 52.0% | 49.3% ± 6.0 pp |
| Qwen3-VL | 52.0% | 30.5% | 37.0% | 39.8% ± 11.0 pp |

The seed-matched Qwen-minus-GR00T differences are +9.5, −23.0, and −15.0 percentage points. Their mean is −9.5 pp, with a sample SD of 16.9 pp. The average favors GR00T, but seed 3 reverses direction. With only three independent training seeds per backbone, these across-seed summaries are descriptive and do not establish a stable population-level advantage.

The paired rollout tests answer a different question: they compare these particular trained policies on matched starts. The 200 rollouts per policy are not 200 independent training runs. Seed 3 is borderline: its bootstrap interval excludes zero slightly, while its exact McNemar p-value is just above 0.05; after Holm correction it is not below 0.05. Seeds 4 and 5 favor GR00T on their matched rollout cohorts.

## Comparison with the submitted paper

| Measure | Submitted paper | New ALOHA study |
|---|---:|---:|
| GR00T success | 245/400 = 61.25% | 49.3% mean across 3 training seeds |
| Qwen3-VL success | 207/400 = 51.75% | 39.8% mean across 3 training seeds |
| GR00T-minus-Qwen gap | +9.5 pp | +9.5 pp difference between means |
| Independent training seeds reported/evaluated | 2 per backbone in the paper | 3 per backbone |
| Evaluation design | Two 200-start cohorts per backbone, as reported | 200 matched starts per backbone within each training-seed pair |

The new study reproduces the paper's average 9.5-point gap numerically, but both new mean success rates are about 11.9 points lower. The per-seed effect is also less uniform: one of three pairs favors Qwen3-VL, and the training-seed variation is substantial. Therefore:

- **What improved:** the evidence is more carefully replicated and reported. There are three independently trained policies per backbone, matched rollout starts, paired tests, explicit per-policy outcomes, and mean ± sample SD across training seeds. The new study also makes the distinction between rollout uncertainty and training-run variation explicit.
- **What did not improve:** the observed absolute ALOHA success rates are lower than the submitted-paper rates. The average backbone gap is unchanged, and the direction is not consistent across the three seed pairs.
- **What cannot be concluded:** this is not an exact reproduction of the submitted training recipe, so the lower absolute rates cannot be attributed to one cause. The results do not show that the new policies perform better than the submitted policies or that the backbone effect generalizes beyond these tested models and ALOHA setup.

The protocols differ materially. The submitted paper describes 53,760 optimizer updates per configuration, batch size 128, no state/action normalization, and a 19.2M-parameter head. The new ALOHA runs used 100 fixed epochs, batch size 256, training-episode-only normalization with a 45/5 episode split, a revised head configuration, and the epoch-100 checkpoint. They used the same 200 start IDs (10000–10199) within each seed pair and disabled rendered evaluation samples. The submitted paper reports two separate 200-start cohorts per backbone. These differences make the result a prospective matched comparison within the new protocol, not a direct reproduction of the historical run.

## Offline metrics and observed failures

For every training seed, GR00T has lower best and final validation loss than the same-seed Qwen3-VL policy. However, the closed-loop winner differs by seed: Qwen3-VL wins seed 3, while GR00T wins seeds 4 and 5. On these six policies, offline validation loss did not consistently identify which policy would achieve more closed-loop successes.

| Seed | GR00T best validation loss (epoch) | GR00T epoch-100 loss | Qwen3-VL best validation loss (epoch) | Qwen3-VL epoch-100 loss |
|---:|---:|---:|---:|---:|
| 3 | 0.1622 (19) | 0.2707 | 0.1649 (20) | 0.2845 |
| 4 | 0.1599 (17) | 0.2748 | 0.1686 (16) | 0.2813 |
| 5 | 0.1625 (18) | 0.2657 | 0.1664 (21) | 0.2766 |

All six policies reached their minimum validation loss around epochs 16–21, and all had higher validation loss at epoch 100. Final training losses were about 0.039–0.041 versus final validation losses around 0.266–0.284. This pattern is consistent with late-training overfit, but does not prove why success was lower than in the submitted paper. The fixed epoch-100 checkpoint was part of the locked comparison; do not replace it post hoc with a validation-selected checkpoint and present that as the same experiment.

All unsuccessful rollouts in the new records ended as 400-step truncations: 304 of 600 GR00T rollouts and 361 of 600 Qwen3-VL rollouts. The records contain no gripper traces or videos. Report truncation as an observed failure pattern; these results cannot establish that gripper opening caused the failures.

## Camera-ready action plan

1. **Freeze the verified ALOHA evidence.** Preserve the six run records, checkpoint provenance, shared-start manifests, and verifier outputs. Keep the submitted-paper numbers separate from the new runs.
2. **Report both levels of variation.** Show each policy's success count and interval, the per-seed matched difference and paired test, and the across-training-seed mean ± sample SD. State that paired rollout tests are conditional on the trained policies and that three seeds remain a small training sample.
3. **State the comparison directly.** Say that the mean ALOHA gap is +9.5 pp for GR00T in both the submitted paper and the new study, while the new absolute rates are about 11.9 pp lower and one new seed pair reverses direction.
4. **Disclose the protocol differences.** Identify the different update budget, batch size, normalization, head configuration, checkpoint endpoint, and evaluation cohort/rendering. Do not call the new results an exact reproduction or claim that a single change caused the lower rates.
5. **Answer the offline-metric question plainly.** In this ALOHA sample, lower validation loss did not reliably predict the closed-loop winner. Keep the train/validation divergence as an observation and late overfitting as a possible interpretation, not a proven cause.
6. **Keep failure claims observational.** Report the 400-step truncation counts. Do not attribute failure to gripper opening without trace evidence.
7. **Keep claims bounded to the experiment.** Describe the tested frozen-backbone configurations and ALOHA transfer-cube task. Avoid general claims about all robot pretraining, model families, or tasks. Use neutral wording instead of economic terms such as “pay off.”
8. **If stronger performance evidence is desired later, design a separate ALOHA follow-up before running it.** The highest-value diagnosis is a predeclared checkpoint-selection comparison using fresh selection starts and untouched confirmation starts, because validation loss rises well before epoch 100. To explain the historical absolute-rate gap, a later reproduction would need to match the submitted recipe as closely as the recoverable provenance allows and change one factor at a time. Neither follow-up should reuse the current test starts or be described as part of these six completed policies.

## Reviewer feedback addressed by this plan

| Reviewer concern | ALOHA-specific response |
|---|---|
| R1: paired ALOHA analysis and clarity about two independent runs | Report matched starts and discordances per seed, paired uncertainty/tests, and separate training-seed mean ± SD. Distinguish the paper's reported runs from the three newly trained policies. |
| R2: training-run count, training variance, validation split, and whether offline loss answers the question | State three independent training seeds per backbone, report their per-policy outcomes and mean ± SD, document the episode-disjoint 45/5 split and training-only normalization, and answer that loss did not consistently predict the closed-loop winner. |
| R3: broader backbone and task generality | Treat broader generality as untested. This ALOHA study supports only narrow claims about the two evaluated backbone configurations. |
| R4: title and cost-benefit wording | Frame the result as an empirical ALOHA study of the tested frozen backbones. Remove “pay off,” “what it buys,” and similar wording because pretraining cost was not measured. |

## RunPod update-matched follow-up

This is a new follow-up protocol, separate from the six completed ALOHA policies above. It does not relabel the existing 7,000-update ALOHA policies as reproductions of the paper's 53,760-update training budget.

### Scope and matrix

- Compare the two frozen backbones, GR00T N1.7 and Qwen3-VL-2B, on ALOHA transfer-cube and LIBERO-Goal with **two cameras only**. There is no one-camera LIBERO arm. The one-camera setup has been identified as invalid for this study; it must not be pooled into the new evidence or used to support a camera-count conclusion.
- Train three independent policy seeds per backbone/task cell: 2 backbones × 2 tasks × 3 seeds = **12 core policies**. The six existing ALOHA policies do not replace these runs because they stopped at 7,000 updates and used a different training configuration.
- Keep Cosmos-Reason2-2B on ALOHA as a separate reviewer-requested extension, not part of the 12-policy core. If included, train three seeds and report it as an intermediate checkpoint in the same model lineage; it does not establish generalization across unrelated pretraining lineages.
- Removing one-camera LIBERO means this follow-up cannot reproduce or retain a one-camera-versus-two-camera claim. The manuscript must state that the invalid one-camera condition was excluded and narrow camera-related conclusions accordingly.

### Training protocol and checkpointing

- Count **optimizer updates** as calls to `optimizer.step()`, not epochs, loader batches, or accumulated microbatches. Set the core training horizon to **53,760 updates per policy**. Use effective batch size 128 to match the paper's stated batch size.
- Continue using the requested Hugging Face scheduler implementation with the full 53,760-update horizon and a predeclared warmup fraction. If the current 10% warmup is retained, that is 5,376 updates. Because the submitted paper used OneCycle, this is update-matched but not scheduler-identical; report that distinction rather than calling it an exact recipe reproduction. Do not let validation patience stop these fixed-horizon runs early.
- The launched protocol locks the training/validation episode split, model revisions, data fingerprints, seeds, batch size, learning-rate schedule, and probe steps in `configs/runpod_update_matched_study.json` and each `locked_study.json`. Reuse a cache only when its recorded provenance matches the model, input, camera, and data configuration.
- The locked update-matched mapping is ALOHA updates 26,880 and 53,760, and LIBERO updates 13,440, 26,880, 40,320, and 53,760. These correspond to ALOHA epochs 192/384 and LIBERO epochs 30/60/90/120 under the active loader geometry. The manuscript's historical epoch numbers are not the same schedule; keep that distinction explicit and do not change the locked probes while workers are running.
- Save full resumable state (policy head, optimizer, scheduler, RNG, sampler state, and `global_step`) every 2,000 updates, retaining the latest two recovery states per run. Save a lightweight policy-weight snapshot at every task-specific probe and at the final update. Never select a checkpoint using closed-loop test results.
- Write structured loss records keyed by optimizer update. Record averaged training loss, episode-disjoint validation loss, learning rate, and—if available—gradient norm at least every 1,000 updates and at every probe. Save raw records and plot per-seed curves plus mean and spread across the three training seeds.

### Evaluation and offline-metric analysis

- Evaluate all core policies on matched, predeclared starts. At interim probes use 50 ALOHA starts and 100 LIBERO starts (10 tasks × 10); at the final update use 200 ALOHA starts and all 50 standard initial states for each of the 10 LIBERO-Goal tasks (500). Reuse the same probe and final cohorts across both backbones and all training seeds. Record per-episode outcomes so paired analyses can be recomputed.
- The 12-policy core requires 6,300 closed-loop episodes: 4,200 final episodes plus 2,100 interim-probe episodes. The optional three-seed Cosmos ALOHA extension adds 750 episodes under the same probe/final counts.
- Align each probe's training loss and validation loss with its closed-loop success. Plot the loss trajectories and corresponding closed-loop success at the same optimizer updates. Test whether loss consistently ranks backbones and independent policies within each task, treating checkpoints as repeated observations within a policy. An inconsistent association supports a task-specific finding that these losses were uninformative for these tested decisions; it does not show that offline metrics are universally uninformative.
- Treat checkpoint observations as repeated measurements within a trained policy, not independent training runs. Report policy-level means and sample SD across three seeds, paired rollout comparisons on matched starts, and uncertainty clustered by training seed. State that this is task- and backbone-specific evidence; it cannot establish that offline loss is universally uninformative.
- Report the gripper-opening pattern only if it is present in the new recorded traces, and describe it as an observed failure pattern rather than a proven cause.

### Storage, cost, and launch gates

- The two workers write to the mounted RunPod Global Volume at `/workspace-global/action-omni/study_v1/`; they sync run-root files about once per minute. The very large active feature caches remain on Pod-local storage. Keep the Pods and volume until all required caches, checkpoints, logs, and final reports are preserved. No laptop-to-Pod file transfer is part of this workflow.
- At the latest observation, cache throughput was far below the scripts' initial estimate: roughly 44 seconds per ALOHA batch and 80–90 seconds per LIBERO batch. Re-estimate timing and credit from current logs and the RunPod dashboard before making later deployment decisions; early-batch extrapolations can change.
- Each Pod runs a single dataset worker. When both report `worker_complete`, a teammate still needs to combine the suffixed ALOHA/LIBERO run folders and run the paired and loss analyses; the separate workers do not emit the cross-dataset summary automatically. Follow the [team continuity guide](docs/team_continuity_guide.md).
- The final report must separate the completed ALOHA study, the update-matched follow-up, and any future Cosmos extension. Cosmos is not part of the current locked run. Never pool results from these protocols as if they shared a training recipe.

### Reviewer mapping

| Change | Reviewer feedback addressed | Limit |
|---|---|---|
| Three independent seeds for each of four core cells; matched rollout starts and policy-level summaries | R1 paired ALOHA analysis; R2 independent training variance and mean ± SD | Three seeds remain a small sample. |
| Two-camera LIBERO only and removal of the invalid one-camera comparison | R1/R4 scope and camera-claim precision | The camera-count effect is no longer tested. |
| Training/validation loss curves aligned with closed-loop probe outcomes | R2 direct answer on offline metrics; preserves the paper's offline-versus-closed-loop contribution | Supports only task- and backbone-specific conclusions. |
| Optional Cosmos ALOHA extension | R2 intermediate-lineage checkpoint; partial response to R3 model breadth | Does not test a separate pretraining lineage. |
