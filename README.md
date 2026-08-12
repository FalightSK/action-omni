# VLA anatomy — which components decide closed-loop success

A controlled study of **vision-language-action** architecture. One identical
19.2M-parameter head (token adapter + DiT flow-matching decoder) is trained
against **frozen** published backbones, layer-matched, on the same data and
schedule, and compared by **closed-loop rollout** rather than offline loss.

**Question.** Existing work studies how to choose *data* and how to choose a
*backbone*. This study asks what happens *between* the components: how backbone
pretraining, camera configuration, action space and task structure interact to
decide whether a policy actually succeeds in the loop.

**Method.** Move one factor at a time and roll out. Every arm shares the head,
the read depth, the training recipe and the evaluation protocol, so a difference
in success rate is attributable to the factor that moved.

See [OBJECTIVE.md](OBJECTIVE.md) for the framing and [RESULTS.md](RESULTS.md)
for the full design, hypotheses, statistics and limitations.

---

## The arms

Two backbones, one lineage — stock **Qwen3-VL-2B** → Cosmos-Reason2-2B →
**GR00T N1.7**. Both hops are verified real finetunes (584/625 and 476/493
tensors differ), so the pair spans the whole robot-pretraining treatment. Both
are read at **layer 16**, which is GR00T's own `select_layer` and intermediate
for Qwen3-VL; the language stack's final RMSNorm is applied to the intermediate
read so the pair differs by weights and not by normalisation.

| Testbed | Arm | Backbone | Views | Val loss | Success rate |
|---|---|---|---|---|---|
| LIBERO-Goal | exp03 | GR00T N1.7 | 1 | 0.0316 | 62.5% (n=200) |
| LIBERO-Goal | exp04 | Qwen3-VL-2B | 1 | 0.0332 | 68.0% (n=200) |
| LIBERO-Goal | exp05 | GR00T N1.7 | **2** | 0.0352 | **91.5%** (n=200) |
| LIBERO-Goal | exp06 | Qwen3-VL-2B | **2** | 0.0352 | **89.0%** (n=200) |
| ALOHA transfer-cube | exp05 | GR00T N1.7 | 1 | 0.0171 | **61.25%** (n=400) |
| ALOHA transfer-cube | exp06 | Qwen3-VL-2B | 1 | 0.0179 | 51.75% (n=400) |

---

## Findings

### 1. Cameras dominate backbones

On LIBERO-Goal, adding the wrist camera is worth **+29.0 points** (GR00T) and
**+21.0 points** (Qwen3-VL), p < 10⁻⁷. Swapping a robot-pretrained backbone for
its stock root, at the benchmark's own two-camera spec, is worth **2.5 points**
(91.5% vs 89.0%, p = 0.40).

An order of magnitude separates the observation configuration from the backbone.

### 2. The backbone null is bounded — it does not survive a bimanual task

ALOHA transfer-cube, n = 400 per arm (two independent 200-episode runs on
disjoint seed ranges):

| Arm | Run 1 | Run 2 | Pooled | Wilson 95% CI |
|---|---|---|---|---|
| GR00T N1.7 | 60.0% | 62.5% | **61.25%** (245/400) | [56.4, 65.9] |
| Qwen3-VL-2B | 49.0% | 54.5% | 51.75% (207/400) | [46.9, 56.6] |

**+9.5 points, z = 2.71, p = 0.0067** — clears the Bonferroni bar for the six
comparisons in this study (0.05/6 = 0.0083). The intervals do not overlap.

The gap is **one transition**, not general competence. ALOHA scores
touch / lift / handover / success, and `max_reward == 3` never occurs in 800
episodes, so the ladder is effectively touch → lift → handover:

| Stage | GR00T | Qwen3-VL | Δ | p |
|---|---|---|---|---|
| P(touch) | 89.2% | 92.5% | −3.3 | 0.11 |
| P(lift \| touch) | 95.8% | 93.8% | +2.0 | 0.22 |
| **P(handover \| lift)** | **71.6%** | **59.7%** | **+12.0** | **0.0009** |

Both early stages run slightly *against* the pretrained arm. The advantage is
localised to the one stage bimanual pretraining plausibly covers, and the
localisation is tighter than the top-line result.

### 3. Pretraining buys ~2× training efficiency, and the gap does not close

Closed-loop checkpoint ladder, paired seeds, 50 episodes per point:

| Epoch | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| GR00T | 8.0% | 28.0% | 46.0% | 54.0% | 72.0% | 66.0% |
| Qwen3-VL | **0.0%** | **0.0%** | 22.0% | 28.0% | 48.0% | 46.0% |

The stock backbone scores **0/50 through epoch 50** while the pretrained arm is
already at 28% (McNemar p = 0.0001), and needs **~2× the epochs** to reach any
given rate (2.4× / 2.8× / 2.2× / 1.9× at the 20/30/40/46% targets). The gap does
not close by epoch 300, so pretraining buys both speed and a residual advantage
— not speed alone.

### 4. Offline metrics do not rank these policies

On LIBERO, validation loss **anti-predicts** closed-loop success. On ALOHA it is
worse than a bad predictor — it is **blind**:

| Measure (stock − pretrained, % of pretrained) | Δ |
|---|---|
| velocity loss (overall) | +0.1% |
| velocity loss (mid phase, contains handover) | +1.9% |
| open-loop action error (nMAE, 14 dims) | −0.4% |
| PE sensitivity | +1.1% |
| attention mass on image | −5.2% |
| **closed-loop success rate** | **−15.5%** |

Four accuracy measures are flat against a gap resolved at p = 0.0067 with a
mechanism at p = 0.0009. Selecting a backbone or a checkpoint by flow-matching
loss would have carried no information about which policy is better.

### 5. Action space determines how much the policy uses vision

Zeroing the 2D positional encoding shifts actions by **0.097–0.111** on LIBERO
(end-effector control) but only **0.038** on ALOHA (joint control with fully
observable 14-DOF proprioception) — 2.5–3× less. Where the state already
determines the arm configuration, the head sources far less from image position.

### 6. Text ablation has a floor, and it must be measured

ALOHA has a single fixed instruction, so zeroing its text tokens removes no task
information — yet loss still rises **1.14×–1.24×**. That residual is the cost of
an off-distribution perturbation alone, i.e. the metric's floor. LIBERO arms,
whose instructions vary across ten goals, read **5.4×–7.4×** against that floor.
Reporting an ablation ratio without its floor overstates the effect.

---

## Architecture

```
PIL image(s) + task text
       │
Frozen VLM backbone  ❄  read at layer 16, final RMSNorm applied
       │  hidden states (B, T, D)
       │
VLMTokenAdapter  🔥  ~5.9M params
  PerTokenLoRA        rank 16,  h' = h + 0.1·B(A(h))
  SpatialAwareMLP     2D positional encoding (+ per-view embedding when T>1 view)
  AttentionReadout    T tokens → context vector
       │  context → adaLN (global)
       │  tokens  → DiT cross-attention (local, per denoising step)
       │
DiTFlowDecoder  🔥  10.1M params
  16 action tokens, self-attn + cross-attn to the VLM tokens
  rectified flow:  x_t = (1-t)·x₀ + t·actions,  3-step Euler at inference
       │
action chunk → execute inference_horizon steps → replan
```

Token geometry varies by backbone and view count: Qwen3-VL family **88** tokens
(64 image, 1 view) or **152** (128 image, 2 views); ALOHA **96** (54 image on a
6×9 grid).

**Two conditioning pathways.** The readout compresses all tokens to one vector
that modulates every LayerNorm (global); cross-attention exposes all tokens as
K,V at every denoising step (local). Removing cross-attention costs +261%
validation loss — the local pathway is load-bearing.

---

## Hard constraints

1. **State = environment observations only** — never action predictions or history.
2. **`inference_horizon` at inference must equal `inference_horizon` at training.**
3. **n ≥ 50 rollouts** before any success-rate claim; n ≥ 200 for a comparison.
4. **Two MuJoCo backends, not interchangeable.** LIBERO/robosuite needs
   `MUJOCO_GL=wgl`; gym-aloha goes through dm_control, which rejects `wgl` and
   needs `MUJOCO_GL=glfw`.
5. **Two conda envs.** `vla` for training, analysis and gym-aloha rollouts;
   `vla_libero` for the LIBERO simulator, which pins numpy 1.22 and would
   downgrade numpy under torch if installed into `vla`.

---

## Quick start

### LIBERO-Goal arm

```bash
python scripts/precompute.py --dataset libero --exp exp06
```

```bash
python scripts/train.py --dataset libero --exp exp06
```

```bash
# Rollouts need the vla_libero env, not vla. 10 tasks x 20 episodes x
# {canonical, swapped}.
MUJOCO_GL=wgl python scripts/eval_libero.py --exp exp06 --episodes 20 --conditions canonical swapped
```

```bash
# Is the harness itself sound? Replays the demos' own actions through it.
MUJOCO_GL=wgl python scripts/validate_harness.py
```

### ALOHA transfer-cube arm

```bash
python scripts/precompute.py --dataset aloha --exp exp05
```

```bash
python scripts/train.py --dataset aloha --exp exp05
```

```bash
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 --episodes 200 --no-video
```

```bash
# Replication on DISJOINT scenes. AlohaEnv.reset derives the cube pose from the
# episode seed alone, so without --seed-offset a rerun replays the same 200
# initial scenes and resamples only the policy's flow noise.
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 --episodes 200 --no-video --seed-offset 1000 --output asset/runs/aloha/exp05_groot_transfer/sim_results_rep.json
```

```bash
# One point of the checkpoint ladder. Hold --seed-offset fixed across every
# snapshot so the curve is paired.
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 --checkpoint asset/runs/aloha/exp05_groot_transfer/checkpoints/epoch_0100.pt --episodes 50 --no-video --seed-offset 2000
```

### Head diagnostics

Inference-only on the cached embeddings — no backbone is loaded, nothing
retrains. Covers adapter probes, cross-attention mass and entropy, text
ablation, PE sensitivity, phase/task loss, per-dimension error, and an
open-loop checkpoint ladder.

```bash
python scripts/analysis/head_diagnostics.py --dataset libero
```

```bash
python scripts/analysis/head_diagnostics.py --dataset aloha
```

```bash
python scripts/analysis/plots_head_diagnostics_aloha.py
```

### Report

```bash
python scripts/report/build_report.py   # -> report/vla_anatomy_report.{html,pdf}
```

---

## Project structure

```
├── OBJECTIVE.md             # Research question and framing
├── RESULTS.md               # Design, hypotheses, results, limitations
├── REFERENCES.md            # Literature this study positions against
│
├── configs/
│   ├── base_config.py       # Shared base; all datasets inherit
│   ├── registry.py          # get_config(dataset, exp_id) factory
│   ├── libero/              # LIBERO-Goal arms
│   │   ├── exp03_groot.py           GR00T N1.7,  1 view   62.5% SR
│   │   ├── exp04_qwen3vl.py         Qwen3-VL-2B, 1 view   68.0% SR
│   │   ├── exp05_groot_2view.py     GR00T N1.7,  2 views  91.5% SR
│   │   └── exp06_qwen3vl_2view.py   Qwen3-VL-2B, 2 views  89.0% SR
│   └── aloha/               # ALOHA transfer-cube, bimanual 14-DOF
│       ├── exp05_groot_transfer.py    GR00T N1.7   61.25% SR
│       └── exp06_qwen3vl_transfer.py  Qwen3-VL-2B  51.75% SR
│
├── data/
│   ├── libero/dataset.py    # LiberoDataset (+ frames.py, paraphrases.py)
│   └── aloha/dataset.py     # AlohaDataset — parquet shards + single mp4
│
├── envs/
│   ├── libero_env.py        # LiberoAgent + LIBERO's 50 fixed init states
│   └── aloha_env.py         # AlohaAgent + seed_offset for replication
│
├── models/
│   ├── vla.py               # VLAModel, VLMTokenAdapter, PerTokenLoRA
│   ├── flow_matching.py     # DiTFlowDecoder
│   └── vla_train.py         # VLATrainModel (adapter + decoder)
│
├── scripts/
│   ├── precompute.py        # 1 · cache frozen-VLM embeddings
│   ├── train.py             # 2 · train the head on the cache
│   ├── evaluate.py          # 3 · rollouts (gym-aloha)
│   ├── eval_libero.py       # 3 · rollouts (LIBERO)
│   ├── validate_harness.py  # oracle-action replay — is the harness sound?
│   ├── analysis/            # head diagnostics, probes, figures
│   └── report/              # HTML + PDF report builder
│
└── asset/                   # NOT in git (large files)
    ├── model/               #   frozen backbone weights
    ├── dataset/             #   LIBERO-Goal, ALOHA
    ├── runs/<dataset>/<exp>/#   caches, checkpoints, rollout results
    └── analysis/            #   diagnostics JSON + figures
```

---

## Baselines

- **[SmolVLA on LIBERO](baselines/smolvla_libero/)** — a full-finetuning VLA
  reference reproduced on LIBERO-Spatial via `lerobot`: **76.0% SR** (100
  rollouts). Documents a reproducibility caveat (the paper's ~87–90% does not
  reproduce ecosystem-wide) and the hf-libero / robosuite setup gotchas.
