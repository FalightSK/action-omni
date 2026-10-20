# Camera-ready matched-study plan and results

Updated 20 October 2026. All six ALOHA policies are complete and pass the locked verifier. The matched study is paused at the user's request: LIBERO is deferred until a RunPod environment is provided. The GR00T seed-3 LIBERO embedding precompute completed and its cache passed provenance, geometry, finite-value, and parity checks; the worker was stopped in preflight before training or evaluation. The validated cache and earlier interrupted cache files are preserved. Resume only after the RunPod setup is provided.

## Current status and scope

- The six ALOHA policies are complete and passed the locked result verifier: GR00T N1.7 and Qwen3-VL, each trained with seeds 3, 4, and 5.
- Each policy was evaluated on 200 starts. Within each training seed, both backbones used the same initial states, environment seeds, and policy-sampling seeds. The fixed checkpoint was epoch 100.
- LIBERO is currently deferred at the user's request pending a RunPod environment. One GR00T seed-3 embedding cache is complete and validated, but no LIBERO policy training or evaluation has started. The coordinator and monitor are paused; all ALOHA results and cache artifacts are preserved.
- Preserve the battery gate for every new heavy training/evaluation stage: if charge is below 35%, wait to start until it reaches 70%; do not interrupt a running stage solely for this gate.
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
