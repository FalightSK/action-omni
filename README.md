# VLA anatomy — which components are load-bearing, and which are cargo

A controlled study of **vision-language-action** architecture. One identical
19.2M head (adapter + DiT flow decoder) is trained against several **frozen**
published backbones, layer-matched, on the same data and schedule, and compared
by **closed-loop rollout** rather than offline loss.

**Research question:** *which parts of a VLA actually decide closed-loop success,
and how do the components interact?* Not "is a frozen VLM good enough" — that is
occupied territory (SmolVLA, EF-VLA). The anatomy question subsumes it: frozen-VLM
sufficiency is one cell in the table, robot pretraining is another, and the
contribution is the table. See `OBJECTIVE.md`.

---

## Current Status

| Phase | Status | Best Result |
|---|---|---|
| **PushT** (architecture validation) | ✅ Complete | **56% SR** (Exp2a, n=50) |
| **LIBERO-Goal** (closed-loop component study) | ✅ Complete | **68.0% SR** (Qwen3-VL, 1-view, n=200) |
| **LIBERO 2-camera** (benchmark-matched spec) | ✅ Complete | **91.5% SR** (GR00T, n=200) |
| **ALOHA transfer-cube** (bimanual 14-DOF) | ✅ Complete | **61.25% SR** (GR00T, n=400) |
| **Language Table** (language grounding) | ❌ Dropped | fails the decodability gate (R² ≤ 0.063) |

### Headline results

**Cameras dominate backbones.** On LIBERO-Goal, adding the wrist camera is worth
+21 to +29 points; swapping a robot-pretrained backbone for its stock base is
worth 2.5 points (p = 0.40).

| Arm | Backbone | Views | Val loss | SR (n=200) | Swapped |
|---|---|---|---|---|---|
| exp01 | Pi-0.5 (robot-pretrained) | 1 | 0.0356 | 33.0% | 0.0% |
| exp02 | PaliGemma-3B (its stock base) | 1 | 0.0528 | 42.0% | 0.0% |
| exp03 | GR00T N1.7 (robot-pretrained) | 1 | 0.0316 | 62.5% | 0.0% |
| exp04 | Qwen3-VL-2B (its stock root) | 1 | 0.0332 | 68.0% | 0.0% |
| exp05 | GR00T N1.7 | **2** | 0.0352 | **91.5%** | 0.0% |
| exp06 | Qwen3-VL-2B | **2** | 0.0352 | **89.0%** | 0.0% |

**But the null is bounded — it does not survive a bimanual task.** ALOHA
transfer-cube, n = 400 per arm (two 200-episode runs on disjoint seed ranges):

| Arm | Backbone | Pooled SR | Wilson 95% CI |
|---|---|---|---|
| aloha exp05 | GR00T N1.7 | **61.25%** (245/400) | [56.4, 65.9] |
| aloha exp06 | Qwen3-VL-2B | 51.75% (207/400) | [46.9, 56.6] |

+9.5 pts, z = 2.71, **p = 0.0067** (clears Bonferroni 0.05/6). The entire gap is
one transition — P(handover | lift) 71.6% vs 59.7%, **p = 0.0009** — while touch
and lift show no gap and in fact run slightly against GR00T.

**Pretraining buys ~2× training efficiency, and the gap does not close.**
Closed-loop checkpoint ladder, paired seeds, 50 episodes per point:

| Epoch | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| GR00T | 8.0% | 28.0% | 46.0% | 54.0% | 72.0% | 66.0% |
| Qwen3-VL | **0.0%** | **0.0%** | 22.0% | 28.0% | 48.0% | 46.0% |

The stock backbone scores 0/50 through epoch 50 (McNemar p = 0.0001) and needs
~2× the epochs to reach any given rate.

**Offline metrics do not rank policies.** On LIBERO, validation loss
anti-predicts success (4 of 4 within-pair, r = −0.503). On ALOHA it is worse than
useless — it is *blind*: velocity loss differs by 0.1%, open-loop action error by
−0.4%, PE sensitivity by 1.1%, against a 15.5% relative gap in success rate.

**Text ablation has a floor, and Pi-0.5 falls below it.** ALOHA has one fixed
instruction, so zeroing its text tokens removes no information; the residual
1.14–1.24× is the cost of the perturbation alone. Against that floor, Pi-0.5's
1.023× means it ignores its instruction *more* completely than a model whose
instruction is a constant. LIBERO arms with varying instructions read 2.8–7.4×.

See `OBJECTIVE.md` for the research plan and `CHAPTER2_PLAN.md` for hypotheses,
full results, and limitations (§7 especially — LIBERO and ALOHA differ on six
axes at once, so no single axis can be named as the cause). `RESEARCH.md` and
`MILESTONES.md` were retired when the project refocused on the anatomy question.

---

## Architecture

The diagram below is the PushT configuration. The head is unchanged across every
arm; only the frozen backbone and the token geometry vary — PaliGemma family 272
tokens (256 image), Qwen3-VL family 88 (64 image, 1 view) or 152 (128 image, 2
views), ALOHA 96 (54 image on a 6×9 grid).

```
PIL Image + Task Text
       │
Qwen3.5-0.8B  ❄ FROZEN (853M params, bfloat16)
       │  hidden states (B, 82, 1024)
       │
VLMTokenAdapter  🔥 TRAINABLE (~5.9M params)
  Stage 1 — PerTokenLoRA         rank=16,  h' = h + 0.1·B(A(h))
  Stage 2 — SpatialAwareMLP      DINO-style 2D positional encoding → (B, 82, 512)
  Stage 3 — AttentionReadout     82 tokens → context (B, 512)
       │  context → adaLN (global)
       │  tokens  → DiT cross-attention (local, per denoising step)
       │
DiTFlowDecoder  🔥 TRAINABLE (10.1M params)
  16 action tokens, self-attn + cross-attn to 82 VLM tokens
  OT-CFM: x_t = (1-t)x₀ + tx₁, 3-step Euler inference
       │
16 × 2D delta actions → execute 4 steps, replan
```

**Total trainable: 16M / 869M total = 1.9%**

### Two Conditioning Pathways (Why DiT Beats MLP)

The DiT receives visual information through two parallel channels:
- **Global (adaLN):** AttentionReadout compresses 82→1 token → modulates every LayerNorm
- **Local (cross-attention):** All 82 tokens used as K,V per denoising step — each action step queries the full spatial scene

Removing cross-attention: **+261% validation loss**. This bypass around the readout bottleneck is why DiT (56% SR) beats MLP (30% SR).

---

## PushT Results Summary (n=50, final)

| Exp | Architecture | SR | Key finding |
|---|---|---|---|
| Exp1 | MLP, layer 24 | 30% | Baseline |
| Exp2 | DiT + 6D state | 0% | **Covariate shift** — never use action history |
| **Exp2a** | **DiT + 2D state** | **56% ✅** | Cross-attn is load-bearing mechanism |
| Exp2c | DiT, ih=8 | 0% | **Hard constraint:** ih must match training |
| Exp2d | DiT, ms=500 | 42% | MPS variance (p=0.16) |
| Exp3 | DiT, multi-scale 8/16/24 | 44% | Naive linear fusion invalid |

---

## Mechanistic Findings (Exp2a)

| Component removed | Loss Δ | Interpretation |
|---|---|---|
| **LoRA zeroed** | **+533%** | LoRA is 1024→16→1024 task-projection, not correction |
| **Cross-attention removed** | **+261%** | Per-step spatial querying is load-bearing |
| adaLN zeroed | +231% | Global scene conditioning critical |
| Readout → mean-pool | +160% | Readout still needed for adaLN |
| Spatial PE removed | +12% | Helpful but not the bottleneck |

---

## Hard Constraints (Never Violate)

1. **State = environment observations only** — never action predictions or history
2. **inference_horizon at inference = inference_horizon at training** — cannot change post-training
3. **n_eval ≥ 50** before drawing SR conclusions (MPS non-determinism)
4. **Multi-scale fusion must have less capacity than LoRA** (< 32K params)

---

## Quick Start

### Install

```bash
brew install ffmpeg
pip install -r requirements.txt
```

### Evaluate best model (Exp02a, 56% SR)

```bash
# Requires checkpoint at asset/runs/pusht/exp02a_dit/checkpoints/best.pt
python scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50 --no-video

# Any arm, end-to-end. The pipelines/pusht/ wrappers were retired in favour of
# calling the three stages directly, since each has its own resource profile.
python scripts/precompute.py --dataset pusht --exp exp02a
python scripts/train.py      --dataset pusht --exp exp02a
```

### LIBERO-Goal closed-loop arms

```bash
# Stage 1 — encode the frozen backbone once (cache lands under asset/runs/...)
python scripts/precompute.py --dataset libero --exp exp04

# Stage 2 — train the 19.2M head on the cache
python scripts/train.py --dataset libero --exp exp04

# Stage 3 — rollouts. NOTE: needs the vla_libero env, not vla: the simulator
# stack (libero, robosuite, mujoco) is only installed there, and Windows needs
# MUJOCO_GL=wgl. 10 tasks x 20 episodes x {canonical, swapped}.
MUJOCO_GL=wgl python scripts/eval_libero.py --exp exp04 --episodes 20 \
    --conditions canonical swapped

# Is the harness itself sound? Replays the demos' own actions through it.
MUJOCO_GL=wgl python scripts/validate_harness.py     # ~90% = harness OK
```

### Mechanistic analysis

```bash
python scripts/analysis/mechanistic.py --dataset pusht --exp exp02a   # DiT (recommended)
```

### Head diagnostics — what the trained head reads

Inference-only, on the cached embeddings, so no backbone is loaded and no
retraining is involved. Covers adapter probes, cross-attention mass and entropy,
text ablation, PE sensitivity, phase/task loss, per-dimension error, and an
open-loop checkpoint ladder.

```bash
python scripts/analysis/head_diagnostics.py --dataset libero
```

```bash
# ALOHA. Note the single fixed instruction: task-discriminative analyses
# (silhouette, per-task loss) are undefined here and are dropped rather than
# reported as nulls. Text ablation on this dataset measures the metric's FLOOR.
python scripts/analysis/head_diagnostics.py --dataset aloha
```

```bash
python scripts/analysis/plots_head_diagnostics.py         # LIBERO figures
python scripts/analysis/plots_head_diagnostics_aloha.py   # ALOHA + cross-dataset
```

### ALOHA transfer-cube arms (bimanual, 14-DOF)

```bash
# exp05 = GR00T N1.7, exp06 = stock Qwen3-VL-2B. Same head, same 96-token budget
# on a 6x9 grid, same layer-16 read; the pair differs only in backbone weights.
python scripts/precompute.py --dataset aloha --exp exp05
python scripts/train.py      --dataset aloha --exp exp05
```

```bash
# Rollouts. MUJOCO_GL=glfw, NOT wgl: gym-aloha goes through dm_control, which
# rejects wgl outright (wgl is LIBERO/robosuite's raw-mujoco backend).
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 \
    --episodes 200 --no-video
```

```bash
# Replication on a DISJOINT set of scenes. AlohaEnv.reset derives the cube pose
# from the episode seed alone, so without --seed-offset a rerun replays the same
# 200 initial scenes and resamples only the policy's flow noise.
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 \
    --episodes 200 --no-video --seed-offset 1000 \
    --output asset/runs/aloha/exp05_groot_transfer/sim_results_rep.json
```

```bash
# Closed-loop checkpoint ladder — one point of the speed-vs-skill curve.
# Hold --seed-offset fixed across every snapshot so the comparison is paired.
MUJOCO_GL=glfw python scripts/evaluate.py --dataset aloha --exp exp05 \
    --checkpoint asset/runs/aloha/exp05_groot_transfer/checkpoints/epoch_0100.pt \
    --episodes 50 --no-video --seed-offset 2000
```

### Add a new dataset

```bash
# 1. Create config:  configs/<ds>/exp01_baseline.py  (inherit BaseVLAConfig)
# 2. Register it:    configs/registry.py  (add to _REGISTRY dict)
# 3. Data loader:    data/<ds>/dataset.py
# 4. Env wrapper:    envs/<ds>_env.py
```

---

## Project Structure

```
vla_project/
├── OBJECTIVE.md             # Research plan and framing
├── CHAPTER2_PLAN.md         # Closed-loop component study: hypotheses, results, limits
│
├── configs/                 # Experiment configs, organized by dataset
│   ├── base_config.py       #   Shared base (all datasets inherit from this)
│   ├── registry.py          #   get_config(dataset, exp_id) factory
│   ├── pusht/exp02a_dit.py  #   56% SR (DiT decoder)
│   ├── libero/              #   LIBERO-Goal arms — one head, four backbones
│   │   ├── exp01_goal.py    #     Pi-0.5            33.0% SR
│   │   ├── exp02_paligemma.py #   PaliGemma-3B      42.0% SR
│   │   ├── exp03_groot.py   #     GR00T N1.7        62.5% SR
│   │   ├── exp04_qwen3vl.py #     Qwen3-VL-2B       68.0% SR
│   │   └── exp05/exp06      #     same pair, both cameras — 91.5% / 89.0% SR
│   └── aloha/               #   ALOHA transfer-cube — bimanual 14-DOF
│       ├── exp05_groot_transfer.py    # GR00T N1.7   61.25% SR (n=400)
│       └── exp06_qwen3vl_transfer.py  # Qwen3-VL-2B  51.75% SR (n=400)
│
├── data/                    # Dataset loaders, organized by dataset
│   ├── pusht/dataset.py     #   PushTDataset + PushTEmbeddingDataset
│   ├── libero/dataset.py    #   LiberoDataset (+ frames.py, paraphrases.py)
│   └── aloha/dataset.py     #   AlohaDataset — parquet shards + single mp4
│
├── envs/                    # Environment wrappers (pluggable per dataset)
│   ├── pusht_env.py         #   PushTAgent + run_episode()
│   ├── libero_env.py        #   LiberoAgent + LIBERO's 50 fixed init states
│   └── aloha_env.py         #   AlohaAgent + seed_offset for replication
│
├── models/                  # Architecture (shared across all datasets)
│   ├── vla.py               #   VLAModel, VLMTokenAdapter, PerTokenLoRA
│   ├── flow_matching.py     #   FlowMatchingDecoder (MLP) + DiTFlowDecoder
│   └── vla_train.py         #   VLATrainModel (adapter + decoder combined)
│
├── scripts/                 # Unified entry points
│   ├── precompute.py        #   Step 1: --dataset pusht --exp exp02a
│   ├── train.py             #   Step 2: --dataset pusht --exp exp02a
│   ├── evaluate.py          #   Step 3 (sim): --dataset pusht --exp exp02a --episodes 50
│   ├── offline_eval.py      #   Step 3 (offline metrics): MSE, MAE, L2, directional acc
│   ├── analysis.py          #   Post-training analysis (8 figures)
│   ├── mechanistic_analysis.py  #   Component ablations, attention maps, LoRA contribution
│   ├── mechanistic.py       #   Entry point → dispatches to mechanistic_analysis.py
│   ├── compare.py           #   Head-to-head comparison across experiments
│   └── migrate_assets.sh    #   One-time migration from old asset layout
│
├── pipelines/               # CONCEPT.md only; per-stage scripts live in scripts/
│   ├── run_pipeline.sh      #   Full PushT suite (all experiments)
│   └── run_exp.sh           #   Single experiment end-to-end
│
├── baselines/               # Reference baselines (external policies on benchmarks)
│   └── smolvla_libero/      #   SmolVLA (450M) → 76.0% SR on LIBERO-Spatial
│
├── docs/
│   ├── experiments/pusht/   # Experiment reports (exp01_mlp.md, exp02a_dit.md, …)
│   └── conclusions/pusht.md # Phase conclusion + mechanistic findings
│
# Backward-compat shims (old --exp 1/2/3 commands still work unchanged):
├── config_loader.py  train.py  inference.py  precompute_embeddings.py
│
└── asset/                   # NOT in git (large files)
    ├── model/Qwen3.5-0.8B/  #   VLM weights (~1.6 GB)
    ├── dataset/pusht/        #   PushT parquet + video
    └── runs/pusht/           #   Experiment outputs
        ├── exp01_mlp/        #     checkpoints, analysis, mechanistic, videos
        ├── exp02a_dit/       #     (BEST) checkpoints, analysis, mechanistic
        └── exp03_multiscale/
```

---

## Baselines

Reference baselines (external policies reproduced on standard benchmarks) live under `baselines/`.

- **[SmolVLA on LIBERO](baselines/smolvla_libero/)** — reproduces the SmolVLA (450M) policy on
  LIBERO-Spatial via `lerobot`'s closed-loop eval: **76.0% SR** (100 rollouts, 10 tasks × 10 eps). A
  full-finetuning VLA reference point on a manipulation benchmark, with a documented reproducibility
  caveat (paper's ~87–90% does not reproduce ecosystem-wide) and the hf-libero / robosuite setup
  gotchas. See its [README](baselines/smolvla_libero/README.md).

---

## Key Papers (Related Work)

**Supports frozen backbone concept:** R3M (2022), MVP (2023), DINOv2 for robotics (2023), RoboFlamingo (2023)

**Opposing — full VLM finetuning:** RT-2 (2023), OpenVLA (2024), π0 (2024), Octo (2023)

**Critical baseline to beat:** CLIPort (2021) — CLIP-conditioned Diffusion Policy for language-conditioned manipulation

Our contribution: frozen *language-visual* VLM generalizes OOD better than (1) frozen vision-only (ViT), (2) contrastive-aligned (CLIP), and (3) task-specific (Diffusion Policy). Proven via custom OOD dataset — train on 4 colors, test on 6.
