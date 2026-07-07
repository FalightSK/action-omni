# Frozen VLM as a Generalizable Visual Backbone for Robot Manipulation

A **Vision-Language-Action (VLA) system** using a frozen Qwen3.5-0.8B backbone — no robot-data pretraining of the visual backbone, no language-specific finetuning. Only the lightweight adapter and flow-matching decoder are trained (16M params vs 853M frozen).

**Research goal:** Prove that internet-scale language-visual pretraining generalizes to novel robot manipulation tasks and out-of-distribution visual distributions — without ever finetuning the backbone on robot data.

---

## Current Status

| Phase | Status | Best Result |
|---|---|---|
| **PushT** (architecture validation) | ✅ Complete | **56% SR** (Exp2a, n=50) |
| **ViT Ablation** (language-visual vs vision-only) | ⏳ Priority 1 | — |
| **ALOHA** (physical complexity scaling) | ⏳ Priority 2 | — |
| **Language Table** (language grounding) | ⏳ Priority 3 | — |
| **Custom OOD** (the headline claim) | ⏳ Priority 4 | — |

See `MILESTONES.md` for full progress tracking and `RESEARCH.md` for research plan.

---

## Architecture

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

# Single experiment end-to-end (precompute + train + eval):
bash pipelines/pusht/run_exp.sh exp02a

# Full PushT suite (all experiments):
bash pipelines/pusht/run_pipeline.sh
```

### Mechanistic analysis

```bash
python scripts/mechanistic.py --dataset pusht --exp exp01    # MLP
python scripts/mechanistic.py --dataset pusht --exp exp02a   # DiT (recommended)
python scripts/mechanistic.py --dataset pusht --exp exp03    # multi-scale
```

### Add a new dataset (e.g. ALOHA)

```bash
# 1. Create config:  configs/aloha/exp01_baseline.py  (inherit BaseVLAConfig)
# 2. Register it:    configs/registry.py  (add to _REGISTRY dict)
# 3. Data loader:    data/aloha/dataset.py
# 4. Env wrapper:    envs/aloha_env.py
# 5. Run:
python scripts/precompute.py --dataset aloha --exp exp01
python scripts/train.py      --dataset aloha --exp exp01
python scripts/evaluate.py   --dataset aloha --exp exp01 --episodes 50
```

---

## Project Structure

```
vla_project/
├── RESEARCH.md              # Research plan, hypotheses, contributions, contingencies
├── MILESTONES.md            # Phase-by-phase progress tracking
│
├── configs/                 # Experiment configs, organized by dataset
│   ├── base_config.py       #   Shared base (all datasets inherit from this)
│   ├── registry.py          #   get_config(dataset, exp_id) factory
│   └── pusht/               #   PushT configs
│       ├── exp01_mlp.py     #     30% SR (MLP decoder)
│       ├── exp02a_dit.py    #     56% SR (DiT decoder) ← BEST
│       ├── exp03_multiscale.py  # 44% SR (multi-scale)
│       └── exp04_vit.py     #     ViT ablation ← Priority 1
│
├── data/                    # Dataset loaders, organized by dataset
│   └── pusht/dataset.py     #   PushTDataset + PushTEmbeddingDataset
│
├── envs/                    # Environment wrappers (pluggable per dataset)
│   └── pusht_env.py         #   PushTAgent + run_episode()
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
├── pipelines/pusht/         # Pipeline orchestration
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
