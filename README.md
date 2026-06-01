# VLA PushT — Qwen3.5-0.8B + OT-CFM Diffusion Policy

A **Vision-Language-Action (VLA)** model that learns to push a T-shaped block
onto a target zone using a frozen VLM backbone and a conditional flow-matching
action decoder.

| | |
|---|---|
| **Backbone** | Qwen3.5-0.8B (frozen, bfloat16) |
| **Decoder** | OT-CFM velocity MLP (3 Euler steps at inference) |
| **Dataset** | LeRobot PushT — 206 episodes, 25,650 frames |
| **Hardware** | Apple M1 Mac (MPS backend) |
| **Golden rule** | State input must **never** include block or goal position |

---

## Experiments

| | Exp 1 (baseline) | Exp 2 (B+C) |
|---|---|---|
| **State** | 2D agent pos | 6D: pos + 2 prev deltas |
| **VLM layers** | Layer 28 only | Layers 14, 21, 28 (multi-scale) |
| **cond_dim** | 514 | 518 |
| **Trainable params** | 16.2M | 19.4M |
| **Cache** | `asset/result/vlm_embeddings.pt` (4.0 GB) | `asset/result_exp2/vlm_embeddings.pt` (~12.8 GB) |
| **Best SR** | 25% (5/20 episodes) | TBD |
| **Mean coverage** | 87.2% | TBD |

Switch between experiments with `--exp 1` or `--exp 2` (default) on any script.

---

## Project Structure

```
vla_project/
│
├── config.py              # Experiment 2 config (current)
├── config_exp1.py         # Experiment 1 config (frozen snapshot)
├── config_loader.py       # get_config(exp) factory — used by all scripts
│
├── data/
│   └── pusht_dataset.py   # PushTDataset (raw) + PushTEmbeddingDataset (cached)
│
├── models/
│   ├── vla.py             # MultiScaleFusion, VLMTokenAdapter, VLAModel
│   └── flow_matching.py   # FlowMatchingDecoder (OT-CFM)
│
├── precompute_embeddings.py   # Step 1 — run VLM once, save token sequences
├── train.py                   # Step 2 — train adapter + decoder
├── evaluate.py                # Step 3 — offline metrics
├── inference.py               # Step 4 — simulation (gym_pusht)
├── analysis.py                # Analysis plots (8 figures)
├── architecture_diagram.py    # Full architecture diagram
│
├── EXPERIMENT_01_VLA_PUSHT.md # Experiment 1 report
├── requirements.txt
├── run_pipeline.sh
│
└── asset/                 # ⚠ NOT in git (too large)
    ├── model/             # Qwen3.5-0.8B weights   (~1.6 GB)
    ├── dataset/           # PushT parquet + video
    ├── result/            # Exp1 cache, checkpoints, analysis
    └── result_exp2/       # Exp2 cache, checkpoints
```

---

## Architecture

### VLMTokenAdapter (trainable)

```
Qwen hidden states  (B, n_layers, 82, 1024)
        │
  Stage 0 — MultiScaleFusion          [Exp2 only]
    Linear(n_layers×1024 → 1024) + LayerNorm
        │
  Stage 1 — PerTokenLoRA
    h_i' = h_i + scale * B(A(h_i))   rank=16
        │
  Stage 2 — SpatialAwareMLP  (DINO-style)
    image tokens: cat(h_i', 2D_PE(row,col))
    text  tokens: cat(h_i', 1D_PE(pos))
    → MLP(1152 → 512) per token
        │
  Stage 3 — AttentionReadout
    learnable query cross-attends 82 tokens → (B, 512)
```

### FlowMatchingDecoder (trainable)

```
OT-CFM:  x_t = (1-t)·x_0 + t·x_1,   loss = MSE(v_θ, x_1-x_0)
Inference: 3-step Euler from x_0 ~ N(0,I)
Input:   [x_t (32D), t (sinusoidal emb), cond (518D)]
Network: 6 × ResidualBlock(512)
Output:  velocity field (32D = 16 steps × 2D)
```

---

## Quick Start

### 1. Install dependencies

```bash
brew install ffmpeg      # AV1 codec
pip install -r requirements.txt
```

### 2. Download assets (not in repo)

Place the following in `asset/`:
- `model/Qwen3.5-0.8B/` — Qwen3.5 model weights
- `dataset/pusht_dataset/` — LeRobot-format PushT dataset

### 3. Pre-compute VLM embeddings

```bash
# Experiment 2 (B+C) — requires full re-run (~95 min, ~12.8 GB)
python precompute_embeddings.py --exp 2

# Experiment 1 — if you only need the 6D state update (< 5 min, no VLM re-run)
python precompute_embeddings.py --exp 1 --recompute-states
```

### 4. Train

```bash
python train.py --exp 2       # Experiment 2 (default)
python train.py --exp 1       # Re-train Experiment 1
```

### 5. Evaluate

```bash
python evaluate.py --exp 2
python evaluate.py --exp 1    # Reproduces: val_loss=0.4490, L2=9.33px
```

### 6. Simulate

```bash
python inference.py --exp 2
python inference.py --exp 1   # Reproduces: 25% SR, 87.2% coverage
```

---

## Key Findings (Experiment 1)

| Finding | Evidence |
|---|---|
| Last-mile stall (13/20) | Model reaches 87–94% coverage then stops |
| Single attention query locks to center-left | Attention entropy ~3.0 bits; corners ignored |
| LoRA rank-16 bottleneck | LoRA B gradient 7× LoRA A gradient |
| Text tokens receive 0% attention | img_mean=0.0156, txt_mean=0.0000 |
| Mean coverage gap to success: 10.7% | 8/15 failed episodes within 10% of threshold |

Full analysis: `EXPERIMENT_01_VLA_PUSHT.md`

---

## Configuration

All hyperparameters live in `config.py` (Exp2) and `config_exp1.py` (Exp1).
Use `get_config(exp)` from `config_loader.py` — do not import configs directly.

### State design rule

> **The state vector must only contain agent position and action history.**
> Block position and goal/destination position are **strictly forbidden** as inputs.
