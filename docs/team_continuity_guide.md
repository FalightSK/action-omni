# Action-Omni team walkthrough and continuity guide

> **Current operations notice (20 October 2026):** Production RunPod actions are
> MCP-only and use `scripts/runpod_bootstrap.py` with the v3
> `scripts/runpod_success_study.py` runner. SSH/nohup commands and the older
> update-matched study details below are historical notes, not current launch
> instructions.

**Status snapshot:** 20 October 2026. The live Pod state below was checked during this session. Refresh it from the RunPod logs before making operational decisions.

This guide is for a teammate picking up the experiment and manuscript work. The canonical protocol is [`configs/runpod_update_matched_study.json`](../configs/runpod_update_matched_study.json); the research decisions and camera-ready plan are in [`plan.md`](../plan.md). This guide explains how the running cloud jobs fit together, where their records live, and what remains after they finish.

## Read this first

- The original submitted paper and the six completed `camera_ready_fixed_endpoint_v1` ALOHA policies are **historical evidence**. The new RunPod study is a separate, update-matched follow-up; never pool the results as if they used one recipe.
- Two B300 MIG Pods are running detached workers: one for ALOHA and one for LIBERO. Closing or sleeping the laptop does not stop them. Stopping or deleting a Pod does.
- At the latest log check, **neither Pod had started policy training**. Both were still encoding frozen-backbone features for GR00T (`exp05`, seed 3). The latest observed progress was approximately ALOHA `28/625` batches and LIBERO `14/2656` batches. No policy had completed.
- Do not launch another copy while either worker is alive. The unsuffixed run folder `runpod_update_matched_v1` is a stale first-launch folder; the active studies use the suffixed `_aloha` and `_libero` folders.
- The active Pod source directories are not Git checkouts. The lock file records code hashes, not the source contents. The local repository also has uncommitted changes; preserve them and the active Pods until a reproducible code snapshot is secured.

## Research context for the next author

The paper asks whether robot pretraining of a frozen visual-language backbone improves a learned VLA policy in closed-loop manipulation, and whether offline loss can predict rollout performance. Its evidence is one backbone lineage (Qwen3-VL → Cosmos → GR00T) and two simulated tasks. Reviewers liked the controlled comparison, closed-loop evaluation, and offline-versus-rollout question. They asked for clearer training-seed accounting, a defensible episode-disjoint validation split, paired ALOHA statistics, narrower claims, and observational language for failure patterns. The current follow-up addresses seed and loss reporting with three independent seeds and training/validation trajectories; it does not add an independent model lineage, and it must not be described as answering generalization across all backbones or tasks. See [`docs/reviewer_response_matrix.md`](reviewer_response_matrix.md) and [`docs/RESULTS.md`](RESULTS.md).

The editable camera-ready source is [`manuscript/main.tex`](../manuscript/main.tex). The original submitted PDF was supplied from `C:\Users\SK\Downloads\paper_76.pdf`, outside the repository; teammates should obtain an accessible copy from the owner rather than assuming it is in Git.

## What the experiment is doing

The study compares frozen GR00T N1.7 (`exp05`) and stock Qwen3-VL-2B (`exp06`) on ALOHA transfer-cube and two-camera LIBERO-Goal. Each dataset/backbone cell has three independently trained policy seeds: 3, 4, and 5, for six policies per Pod and twelve policies in total. LIBERO uses both `agentview` and `robot0_eye_in_hand`; the invalid one-camera arm is excluded. Cosmos is not part of this locked run.

Each policy gets 53,760 optimizer updates at effective batch size 128. The schedule has 5,376 warmup updates followed by cosine decay to a 1% learning-rate floor. This matches the paper's stated update budget, but the scheduler and several other protocol details differ; report it as an update-matched follow-up, not an exact reproduction. Training runs to the fixed endpoint rather than stopping early on validation patience.

The worker handles policies in this order: GR00T seeds 3, 4, and 5, then Qwen3-VL seeds 3, 4, and 5. It encodes one shared feature cache per backbone and dataset, trains each policy, then evaluates its saved checkpoints. ALOHA has a 50-episode intermediate probe at update 26,880 and a 200-episode final evaluation. LIBERO has 10 episodes per task at updates 13,440, 26,880, and 40,320, then 50 per task at the final update across all ten tasks. Four `--workers` means data-loader workers; it does not mean four training Pods.

The complete stage flow is:

```text
locked split and source fingerprints
    → frozen-backbone feature precompute and cache validation
    → fixed-update policy training with recovery checkpoints and loss history
    → probe/final closed-loop evaluation
    → paired statistics and training/validation-loss analysis
```

Precompute is the current bottleneck. The observed batch timings imply that the first ALOHA cache may take several hours and the first LIBERO cache may take multiple days. Those are rough extrapolations from early batches, not completion promises. The code's initial time estimates were much shorter than the observed rates, so use the live progress log to revise expectations.

## Pods and storage

Use the RunPod dashboard's **Connect** panel to confirm current connection details. These were the observed names and SSH endpoints on 20 October; endpoints can change.

| Pod | Dataset | Pod ID | Observed SSH endpoint |
|---|---|---|---|
| `action-omni-aloha` | ALOHA | `u4530gsf6t3v8d` | `root@195.26.233.158`, port `12289` |
| `action-omni-libero` | LIBERO | `qbcyn0aiuuh6vg` | `root@195.26.233.158`, port `15756` |

On each Pod, the checkout is `/workspace/action-omni`. ALOHA uses `.venv`; LIBERO uses `.venv310` and `MUJOCO_GL=egl`. Do not move files between the laptop and Pods unless the project owner explicitly requests it. Models and datasets are staged from their pinned Hugging Face revisions, and the workers sync run artifacts to the mounted Global Volume.

| Purpose | ALOHA | LIBERO |
|---|---|---|
| Active local run root | `/workspace/action-omni/asset/runs/runpod_update_matched_v1_aloha` | `/workspace/action-omni/asset/runs/runpod_update_matched_v1_libero` |
| Durable run root | `/workspace-global/action-omni/study_v1/runs/runpod_update_matched_v1_aloha` | `/workspace-global/action-omni/study_v1/runs/runpod_update_matched_v1_libero` |
| Detached worker log | `/workspace-global/action-omni/workers/aloha/worker.log` | `/workspace-global/action-omni/workers/libero/worker.log` |
| Active status file | `<local run root>/study_status.json` | `<local run root>/study_status.json` |

The worker syncs files beneath each run root to the Global Volume approximately once per minute, including current logs, histories, checkpoints, evaluation records, and manifests. The large active embedding cache is under `/workspace/action-omni/asset/cache/` and is **not** copied to the Global Volume. Do not delete or stop a Pod while its cache is still needed; if local cache data is lost, precompute must be repeated. Keep the Global Volume attached and do not delete it when retiring a Pod.

The Global Volume also holds `study_v1/assets_manifest.json`, pinned model/data assets, and a copy of the plan JSON. The `runpod_update_matched_v1` folder without a dataset suffix is from the first launch attempt; its `running` status is stale and does not mean a worker is still active. The per-dataset suffixed folders are authoritative.

## Check status and follow logs

From PowerShell, connect using the current details shown in RunPod:

```powershell
ssh -p 12289 root@195.26.233.158  # ALOHA; verify the endpoint in RunPod first
ssh -p 15756 root@195.26.233.158  # LIBERO; verify the endpoint in RunPod first
```

After connecting to a Pod, read the status and follow the log path named by it. Example for ALOHA:

```bash
run=/workspace/action-omni/asset/runs/runpod_update_matched_v1_aloha
cat "$run/study_status.json"
log="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$run/study_status.json")"
tail -f "$log"
```

For LIBERO, replace `_aloha` with `_libero`. The durable copy can be inspected at `/workspace-global/action-omni/study_v1/runs/…` if the local Pod filesystem is unavailable. The worker's `updated_utc` field currently changes at stage boundaries, not for every precompute batch; a timestamp that stays at the start of precompute is not proof that the worker is stuck. Read the current stage log for batch progress.

GPU telemetry is limited on these MIG Pods. The logs report `Device: cuda`, the precompute process opens the assigned `/dev/nvidia*` device, and the model is moved to CUDA. `nvidia-smi` may still show utilization `N/A` or memory “Insufficient Permissions.” CPU use is expected during image decode, transforms, data loading, and HDF5 writes; the RunPod CPU meter alone does not show that the VLM was placed on CPU. The accessible telemetry does not give a reliable GPU-utilization percentage.

Two log details need context:

- ALOHA may warn that `meta/stats.json` is missing when the default config is first created. The precompute then loads the locked split manifest and applies its training-only normalization before encoding; this warning alone does not mean the active protocol uses unit normalization.
- The shared `worker.log` can contain the traceback from the first LIBERO attempt, which hit a CUDA 13 NVRTC library-path problem. The launched retry sets `LD_LIBRARY_PATH` explicitly. Use the fresh per-stage log referenced in `_libero/study_status.json` to determine current health, not the old traceback by itself.

## Restart and recovery rules

The workers were launched with `nohup`, so an SSH disconnect or sleeping laptop is safe. Do not rerun a launch command just because the terminal closed. First inspect `study_status.json`, the current stage log, and active processes:

```bash
pgrep -af 'runpod_update_matched_study.py|precompute.py|scripts/train.py|evaluate.py|eval_libero.py'
```

If the process is still present, leave it alone and keep monitoring. If it is absent, inspect the last stage log and durable run tree before restarting. The runner reuses a valid complete cache, skips a policy whose manifest is already complete, and the trainer can resume from its recovery checkpoint. Restart with the same plan and same run root only after confirming no worker remains. Do not delete `.partial.h5`, `last.pt`, `previous.pt`, manifests, or logs to “clean up” a failed attempt. A changed plan or workflow code must use a new run root because `locked_study.json` protects the existing run identity.

The active launch commands are reproduced below for recovery. **Do not run them unless the status/log has failed and `pgrep` confirms no worker remains.**

```bash
# ALOHA Pod only
cd /workspace/action-omni
mkdir -p /workspace-global/action-omni/workers/aloha
nohup env LD_LIBRARY_PATH=/workspace/action-omni/.venv/lib/python3.12/site-packages/nvidia/cu13/lib \
  .venv/bin/python -u scripts/runpod_update_matched_study.py --datasets aloha --workers 4 \
  --run-root /workspace/action-omni/asset/runs/runpod_update_matched_v1_aloha \
  --durable-root /workspace-global/action-omni/study_v1 \
  >> /workspace-global/action-omni/workers/aloha/worker.log 2>&1 < /dev/null &

# LIBERO Pod only
cd /workspace/action-omni
mkdir -p /workspace-global/action-omni/workers/libero
nohup env MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/action-omni/.venv310/lib/python3.10/site-packages/nvidia/cu13/lib \
  .venv310/bin/python -u scripts/runpod_update_matched_study.py --datasets libero --workers 4 \
  --run-root /workspace/action-omni/asset/runs/runpod_update_matched_v1_libero \
  --durable-root /workspace-global/action-omni/study_v1 \
  >> /workspace-global/action-omni/workers/libero/worker.log 2>&1 < /dev/null &
```

Do not start a second worker to “speed it up”; it would compete for the same Pod GPU and could overwrite the same status and run files.

## Finish the study after both workers complete

Because the Pods each run one dataset, their expected terminal state is `worker_complete`, not the multi-dataset `complete` state. The separate workers do **not** automatically produce the combined six-pair statistics or cross-dataset loss plots. A teammate must do the final merge after both workers report `worker_complete` and all six policies per dataset have complete manifests and evaluation files.

Run these steps on either Pod while both durable dataset folders are mounted. This copies between cloud-mounted run folders, not between the laptop and a Pod:

```bash
final=/workspace-global/action-omni/study_v1/runs/runpod_update_matched_v1_combined
mkdir -p "$final/aloha" "$final/libero"
cp -a /workspace-global/action-omni/study_v1/runs/runpod_update_matched_v1_aloha/aloha/. "$final/aloha/"
cp -a /workspace-global/action-omni/study_v1/runs/runpod_update_matched_v1_libero/libero/. "$final/libero/"
```

Create `$final/paired_final_manifest.json`:

```bash
cat > "$final/paired_final_manifest.json" <<'JSON'
{
  "family_size": 6,
  "pairs": [
    {"label":"ALOHA","left":"aloha/exp05_groot_transfer/seed_003/evaluation_final.json","right":"aloha/exp06_qwen3vl_transfer/seed_003/evaluation_final.json","vary":"backbone"},
    {"label":"ALOHA","left":"aloha/exp05_groot_transfer/seed_004/evaluation_final.json","right":"aloha/exp06_qwen3vl_transfer/seed_004/evaluation_final.json","vary":"backbone"},
    {"label":"ALOHA","left":"aloha/exp05_groot_transfer/seed_005/evaluation_final.json","right":"aloha/exp06_qwen3vl_transfer/seed_005/evaluation_final.json","vary":"backbone"},
    {"label":"LIBERO","left":"libero/exp05_groot_2view/seed_003/evaluation_final.json","right":"libero/exp06_qwen3vl_2view/seed_003/evaluation_final.json","vary":"backbone"},
    {"label":"LIBERO","left":"libero/exp05_groot_2view/seed_004/evaluation_final.json","right":"libero/exp06_qwen3vl_2view/seed_004/evaluation_final.json","vary":"backbone"},
    {"label":"LIBERO","left":"libero/exp05_groot_2view/seed_005/evaluation_final.json","right":"libero/exp06_qwen3vl_2view/seed_005/evaluation_final.json","vary":"backbone"}
  ],
  "labels": {
    "ALOHA": {"left":"GR00T N1.7","right":"Qwen3-VL-2B"},
    "LIBERO": {"left":"GR00T N1.7, two-view","right":"Qwen3-VL-2B, two-view"}
  }
}
JSON
```

Then run:

```bash
cd /workspace/action-omni
python scripts/analysis/paired_results.py \
  --manifest "$final/paired_final_manifest.json" \
  --output "$final/paired_final.json" \
  --report-dir "$final/reports"
python scripts/plot_study_losses.py \
  --plan configs/runpod_update_matched_study.json \
  --run-root "$final"
```

The loss script verifies all twelve fixed-endpoint manifests and every probe/final evaluation before writing `loss_by_update_aloha.png`, `loss_by_update_libero.png`, `loss_success_alignment.json`, and `loss_summary.json`. Preserve these outputs alongside `paired_final.json`, `reports/`, and the per-policy artifacts. Review them against [`docs/aloha_results_analysis.md`](aloha_results_analysis.md), [`docs/reviewer_response_matrix.md`](reviewer_response_matrix.md), and the manuscript before changing claims.

## Credentials and source preservation

Never put RunPod API keys or Hugging Face tokens in this guide, source files, shell history, or chat. A Hugging Face token was pasted into the earlier conversation; rotate it in Hugging Face settings and reauthenticate securely if the Pods need a new credential. Do not copy a replacement token into the repository.

The local repository currently has uncommitted research-code changes, and the Pod trees are not Git repositories. `locked_study.json` contains hashes of important workflow files but cannot recreate their contents. Before deleting either Pod, preserve the exact source version and confirm it matches those hashes; do not edit the locked scripts while this experiment is running. Keep the old camera-ready ALOHA result files separate from the new update-matched follow-up.

## Manuscript and handoff checklist

When the computations are complete:

1. Verify all 12 run manifests reach exactly 53,760 optimizer updates, with the expected checkpoint files and valid cache/protocol provenance.
2. Check every planned probe and final rollout exists, with matched starts across backbones and per-episode records preserved.
3. Run the combined paired and loss analyses above. Report rollout uncertainty separately from the three independent training seeds.
4. Compare the new ALOHA values to the submitted paper without calling them an exact reproduction. Keep conclusions limited to these backbones, tasks, cameras, and training protocol.
5. Update [`manuscript/main.tex`](../manuscript/main.tex), tables, figures, reviewer responses, and [`plan.md`](../plan.md) only from verified artifacts; compile the existing manuscript before submission.
6. Preserve the RunPod Global Volume and source snapshot before stopping or deleting Pods. Keep credentials out of the handoff and repository.

The previously requested laptop-battery gate applies to new local heavy work: after charge falls below 35%, wait until it reaches 70% before starting another heavy local task. The detached cloud workers do not depend on laptop battery.
