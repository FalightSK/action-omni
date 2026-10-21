# Reviewer-linked camera-ready changes

- **R1 (paired ALOHA and training runs):** The separate fixed-100-epoch ALOHA
  study reports three independent training-seed pairs, per-policy rates,
  discordant counts, exact paired tests, and three-comparison correction.
  Six local test records and the full regenerated paired statistics are
  available under `asset/runs/camera_ready_fixed_endpoint_v1/`. The historical
  submitted campaign is kept separate.
- **R2 (validation and offline measures):** Methods state whole-episode
  train/validation splits, train-only ALOHA normalization, checkpoint
  selection, sample counts, and the distinction between validation chunks
  and closed-loop starts. A focused ALOHA figure compares epoch 50 and 250
  in the four complete trajectories, showing training loss, validation flow
  loss, and development success together. The LIBERO
  seed-3 result explicitly shows flow loss and MAE disagreeing with success,
  while MSE agrees. Table I explicitly shows the missing Qwen LIBERO final
  seed and the unequal development-selected epochs.
- **R3 (breadth):** The manuscript limits inference to two frozen endpoints
  from one lineage, ALOHA and LIBERO-Goal simulation, and the reported
  training recipes. It does not claim added backbone families or physical
  robot experiments.
- **R4 (title and wording):** The title now states the empirical frozen
  backbone scope. Cost-benefit language is removed because training cost was
  not measured.

The current results bundle at
`tmp/experiment_result/action_omni_experiment_results_2026-10-21` passed
all 60 manifest byte-count and SHA-256 checks. Its LIBERO per-task counts
reproduce all three final totals. Its 37 measured ALOHA rows have distinct
seed/backbone/epoch keys, 50 episodes each, and matching saved loss records.
Development ALOHA rates are not pooled with fixed-endpoint final rates.
