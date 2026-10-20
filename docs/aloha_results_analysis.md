# ALOHA matched-study result analysis

Updated 20 October 2026. This report covers the six completed ALOHA policies only: GR00T N1.7 and Qwen3-VL, trained with seeds 3, 4, and 5. Each record passes the locked verifier. The separate LIBERO portion is deferred pending a RunPod environment, so this analysis does not make combined cross-benchmark claims.

## Verification and comparison design

All six evaluation records pass the locked verifier in scripts/run_study.py. Each policy completed 100 training epochs and was evaluated from the fixed epoch-100 checkpoint. Within each training seed, the two backbones were tested on the same 200 initial states, environment seeds, and policy-sampling seeds. The paired reporter rejects mismatched starts or provenance, and the three runs share the same protocol ID. The 200-episode comparisons therefore measure rollout uncertainty conditional on each pair of trained policies; they do not replace independent training runs. The paired analysis was regenerated from these records with `scripts/analysis/paired_results.py`; its input manifest and machine-readable output are `asset/runs/camera_ready_fixed_endpoint_v1/paired_aloha.json` and `asset/runs/camera_ready_fixed_endpoint_v1/paired_aloha_results.json`.

The table reports Qwen3-VL minus GR00T. Per-policy intervals are 95% Wilson intervals; difference intervals are 20,000-resample paired-bootstrap intervals. McNemar tests use the discordant paired outcomes. The Holm column adjusts only the three completed ALOHA seed-pair tests. The original locked plan specified six total comparisons, so that ALOHA-only Holm correction is not the completed global adjustment. Bonferroni values for a six-test family are included as a conservative correction for each observed contrast.

| Training seed | GR00T successes (rate; Wilson 95% CI) | Qwen3-VL successes (rate; Wilson 95% CI) | Qwen − GR00T (paired bootstrap 95% CI) | Qwen-only / GR00T-only | Exact McNemar p | Holm p, ALOHA 3 | Bonferroni p, family 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 85/200 (42.5%; 35.9–49.4%) | 104/200 (52.0%; 45.1–58.8%) | +9.5 pp (+0.5 to +18.5) | 52 / 33 | 0.05025 | 0.05025 | 0.30151 |
| 4 | 107/200 (53.5%; 46.6–60.3%) | 61/200 (30.5%; 24.5–37.2%) | −23.0 pp (−31.5 to −14.5) | 20 / 66 | 6.67e−7 | 2.00e−6 | 4.00e−6 |
| 5 | 104/200 (52.0%; 45.1–58.8%) | 74/200 (37.0%; 30.6–43.9%) | −15.0 pp (−23.5 to −6.5) | 27 / 57 | 0.001404 | 0.002809 | 0.008427 |

The two-sided exact McNemar tests are conditional on the particular trained policies. Seed 3 is close to the 0.05 threshold; its bootstrap interval and exact test fall on opposite sides of that threshold because they use different inferential procedures. It should not be described as a decisive result. Seeds 4 and 5 favor GR00T on their paired rollout cohorts.

## Variation across independently trained policies

| Backbone | Seed 3 | Seed 4 | Seed 5 | Mean success rate ± sample SD across training seeds |
|---|---:|---:|---:|---:|
| GR00T N1.7 | 42.5% | 53.5% | 52.0% | 49.3% ± 6.0 pp |
| Qwen3-VL | 52.0% | 30.5% | 37.0% | 39.8% ± 11.0 pp |

The seed-paired Qwen-minus-GR00T differences are +9.5, −23.0, and −15.0 points, averaging −9.5 ± 16.9 points (sample SD). Thus the observed average favors GR00T by 9.5 points, but the direction reverses in one of three seed pairs and the between-training-seed spread is large. With only three independent policies per backbone, this is a descriptive estimate, not strong evidence that the difference will hold across new training runs. Do not pool the 600 rollouts per backbone as if they were independent training replications.

## Relation to the submitted ALOHA result

The submitted paper reported two independent runs per backbone, each on a distinct 200-start rollout cohort, and pooled 245/400 GR00T successes (61.25%) and 207/400 Qwen3-VL successes (51.75%), a 9.5-point difference. The original checkpoints, training logs, and per-episode outcomes were not recovered in the repository audit, so the original training-run provenance and all paired tests cannot be independently rechecked here. The new matched study has the same difference in its three-seed means: 49.3% versus 39.8%. Its absolute rates are each about 11.9 points lower than the submitted pooled rates, while the average gap is unchanged. The historical and new runs are kept separate.

This is a prospective matched comparison, not an exact reproduction of the submitted recipe. The submitted paper describes 53,760 optimizer updates per configuration at batch size 128, no state or action normalization, and a 19.2M-parameter head. The new locked ALOHA runs use a 100-epoch fixed-final checkpoint, batch size 256, training-episode-only normalization, and the revised 45/5 episode holdout. They evaluate 200 shared scene IDs (10000–10199) per seed pair with `render_samples=0`; the submitted paper reports two independent policy runs per backbone on disjoint rollout seed ranges. Training seeds, training budget, normalization, head configuration, and evaluation rendering therefore differ. The available historical records do not specify enough to verify or compare checkpoint selection directly. The current numbers cannot identify which change explains the lower absolute rates.

The most defensible interpretation is that the average backbone gap reproduced numerically, but the policy-level evidence is less uniform than the submitted summary implied: one of three new seed pairs favors Qwen3-VL, and the GR00T–Qwen3-VL contrast varies by 16.9 points SD across paired training seeds. The same +9.5-point mean should not be treated as proof that the original effect is stable across new training runs.

## Offline loss and closed-loop success

All three GR00T policies have lower best and final offline validation loss than the Qwen policy with the same seed. Yet Qwen succeeds more often in the seed-3 closed-loop comparison, while GR00T succeeds more often in seeds 4 and 5. In this ALOHA sample, validation loss does not reliably predict the closed-loop winner for each policy pair.

| Seed | GR00T best validation loss (epoch) | GR00T epoch-100 validation loss | Qwen best validation loss (epoch) | Qwen epoch-100 validation loss |
|---:|---:|---:|---:|---:|
| 3 | 0.1622 (19) | 0.2707 | 0.1649 (20) | 0.2845 |
| 4 | 0.1599 (17) | 0.2748 | 0.1686 (16) | 0.2813 |
| 5 | 0.1625 (18) | 0.2657 | 0.1664 (21) | 0.2766 |

For all six policies the validation minimum occurs around epochs 16–21, then is higher at the fixed epoch-100 endpoint. Final training losses are about 0.039–0.041, while final validation losses are about 0.266–0.284. This is a clear train/validation gap and is consistent with late-training overfit, but it does not establish the cause of the closed-loop differences. The locked comparison intentionally uses epoch 100 for both arms; changing to the best-validation checkpoint after seeing these results would change the selection protocol.

The later fixed endpoint is a plausible contributor to the lower absolute success rates relative to the submitted run, because the validation loss is already worse by epoch 100. It is not a demonstrated cause: the historical training budget, normalization, head configuration, seeds, and rollout cohort also changed, and the original checkpoint-selection rule cannot be verified from the recovered artifacts. To isolate checkpoint timing, compare best-validation and fixed-final checkpoints under a newly predeclared rollout cohort for both backbones; do not reuse the current test starts to select a checkpoint.

## Recorded failure pattern and limits

Every unsuccessful rollout in these six evaluations ended with the recorded termination reason “truncated” at 400 steps: 304 GR00T failures and 361 Qwen3-VL failures. This supports a narrow statement about time-limit terminations in these runs, not a causal explanation of why the policy failed.

The evaluation records contain no commanded or observed gripper trace, and video/render samples were disabled. These results cannot confirm a gripper-opening mechanism or establish that gripper behavior caused the failures. If the paper retains a gripper-opening observation from other evidence, describe it as an observed pattern and avoid causal wording.

## What this supports in the paper

- Report every policy’s success rate and the mean ± sample SD across the three independent training seeds.
- Report paired rollout tests as conditional on the trained policies, with matched-start counts, discordances, intervals, and the chosen multiplicity correction.
- State the result narrowly: GR00T had the higher three-seed ALOHA mean in this run set, but the seed-3 pair reversed direction and n=3 does not establish a general advantage.
- Answer the offline-metric question directly: validation loss did not identify the closed-loop winner consistently across training-seed pairs.
- Keep failure language observational. The current structured records show 400-step truncation, not a causal gripper diagnosis.
- Keep title and conclusions limited to the evaluated frozen-backbone configurations and ALOHA task. This analysis does not establish generalization to other models or tasks.

The locked evaluation uses render_samples=0. Since camera rendering differs from the submitted-paper setting, describe this protocol change and do not present the new runs as an exact reproduction of the historical numerical result.
