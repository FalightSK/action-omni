# Exp05 — Weighted Layer Fusion (PushT)

**Status:** ✅ Complete  
**Dataset:** PushT  
**Architecture:** DiT decoder + layers 8/16/24 + learnable weighted sum fusion + LoRA rank-16 + 2D state  
**Result:** SR = **36%** (n=50), Wilson 95% CI [24%, 50%], p=0.045 vs Exp02a (significantly worse)  
**Hypothesis:** ✅ **Partially confirmed** — LoRA fully restored (+478%), but early layers 8/16 add noise that hurts SR relative to single-layer Exp02a (56%).  
**Conclusion:** Layer 24 (final Qwen output) is optimal. Intermediate layers degrade performance.

---

## Motivation

Exp03 identified a critical failure: the `MultiScaleFusion` layer (Linear 3072→1024, ~3.1M params) was **96× larger** than LoRA (rank-16, ~32K params). The optimizer found it easier to use this high-capacity projection to memorise the offline embedding distribution, effectively reducing LoRA to near-identity (+2.8% contribution, down from +533% in Exp02a).

The hard constraint from this finding:

> **Fusion capacity must be < LoRA capacity (< 32K params)**

Exp05 tests the minimal fix: replace the large linear projection with **3 learnable scalar logits** (one per layer), softmax-normalised into a weighted sum. With just 3 params, this cannot compete with LoRA for capacity — LoRA should return as the load-bearing mechanism.

| Module | Exp03 | Exp05 |
|---|---|---|
| Fusion | `Linear(3072→1024) + LN` | `softmax(α₀, α₁, α₂)` |
| Fusion params | ~3,147,776 (3.1M) | **3** |
| Fusion / LoRA ratio | **96×** | **0.0001×** |
| LoRA contribution (expected) | +2.8% (displaced) | >> +100% (restored) |

---

## Architecture

Identical to Exp02a except:
- **3 VLM layers extracted**: hidden_states[14] (early), [21] (mid), [28] (final)
- **Fusion**: `WeightedLayerFusion` — learned softmax mixture of the 3 layer outputs
- All other components unchanged (LoRA rank-16, SpatialAwareMLP, AttentionReadout, DiT decoder)

```
Qwen3.5-0.8B (frozen)
  ├── hidden_states[14]  ─┐
  ├── hidden_states[21]  ─┤→  WeightedLayerFusion   (3 params)
  └── hidden_states[28]  ─┘     w = softmax(α₀, α₁, α₂)
                                 out = Σ wᵢ · hᵢ   → (B, 82, 1024)
                           ↓
                     PerTokenLoRA          (~32K params) ← now load-bearing
                           ↓
                     SpatialAwareMLP       (~2.8M params)
                           ↓
                     AttentionReadout      (~2.1M params)
                           ↓
                     DiTFlowDecoder        (~10.1M params)
```

**Total trainable params**: ~16M (same as Exp02a, up from 3 scalar weights — effectively negligible addition)

---

## Embedding Cache

Exp05 **reuses the Exp03 cache** — same layers (14, 21, 28), same extraction code, so no re-precompute is needed if `asset/runs/pusht/exp03_multiscale/vlm_embeddings.pt` exists.

Cache format (v3):
```
embeddings : (N, 3, 82, 1024)  bfloat16
img_masks  : (N, 82)           bool
states     : (N, 2)            float32   ← 2D: agent (x, y) only
actions    : (N, 16, 2)        float32
```

---

## Predictions

| Outcome | Interpretation |
|---|---|
| **SR > 56% AND LoRA ≥ +100%** | ✅ Hypothesis confirmed: weighted fusion is valid, multi-scale adds value |
| **SR ≈ 56% AND LoRA ≥ +100%** | LoRA restored, but layer 28 already captures all task-relevant info — multi-scale adds no marginal gain. Stick with Exp02a. |
| **SR > 56% AND LoRA < +100%** | Unlikely — suggests fusion still has too much capacity somehow |
| **SR < 56%** | Early/mid layers add noise. Inspect learned weights `α` to see which layers dominate. If α₂ ≈ 1.0 (all weight on layer 28), the model learned to ignore early layers. |

**Primary success criterion**: SR > 56% with p < 0.05 vs Exp02a (chi-squared, n=50)  
**Secondary success criterion**: LoRA contribution ≥ +100% in mechanistic ablation

---

## Learned Weight Interpretation

After training, inspect `adapter.fusion.logits`:

```python
import torch
from configs.registry import get_config
from models.vla_train import VLATrainModel

cfg = get_config("pusht", "exp05")
model = VLATrainModel(cfg)
ckpt = torch.load("asset/runs/pusht/exp05_weighted/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

logits = model.adapter.fusion.logits.detach()
weights = torch.softmax(logits, dim=0)
layers = [14, 21, 28]
for l, w in zip(layers, weights.tolist()):
    print(f"  Layer {l:2d}: {w:.3f}")
```

Expected patterns:
- **Uniform** (~0.33 each): All layers contribute equally
- **Late-biased** (layer 28 >> others): Final semantic features dominate
- **Early-biased** (layer 14 >> others): Low-level visual features help most

---

## Commands

```bash
# 1. Train (uses Exp03 cache — no precompute needed)
python scripts/train.py --dataset pusht --exp exp05

# 2. Evaluate (n=50 minimum — MPS non-determinism)
python scripts/evaluate.py --dataset pusht --exp exp05 --episodes 50

# 3. Mechanistic analysis (check LoRA contribution is restored)
python scripts/mechanistic_analysis.py --exp 5

# 4. Inspect learned layer weights (see snippet above)
```

Or run as a full pipeline:

```bash
# Skip precompute — reuse Exp03 cache
bash pipelines/pusht/run_exp.sh exp05 --no-precompute --episodes 50
```

---

## Results

### Success Rate (n=50)
**36%** (18/50) — Wilson 95% CI [24%, 50%]  
**p=0.045** vs Exp02a (56%) — statistically significantly worse.

### Mechanistic Ablations
| Component removed | Val loss | Δ | % increase |
|---|---|---|---|
| Baseline | 0.3774 | — | — |
| **LoRA zeroed** | 2.1806 | +1.803 | **+478%** ✅ (restored) |
| No cross-attention | 1.1179 | +0.741 | +196% |
| adaLN zeroed | 0.9657 | +0.588 | +156% |
| Readout → mean-pool | 0.8027 | +0.425 | +113% |
| No spatial PE | 0.5144 | +0.137 | +36% |

### Learned Fusion Weights
Layer 8: **0.327** | Layer 16: **0.329** | Layer 24: **0.344**  
Near-uniform with slight bias to final layer — model didn't collapse onto one layer, but the equal weighting of early layers appears to dilute the final-layer signal.

## Comparison Table

| Exp | Fusion | Fusion params | LoRA contribution | SR (n=50) |
|---|---|---|---|---|
| Exp02a | None (single layer 24) | — | +533% | **56%** ✅ Best |
| Exp03 | Linear 3072→1024 + LN | 3.1M | +2.8% ⚠ | 44% |
| **Exp05** | softmax(α₀, α₁, α₂) | **3** | **+478%** ✅ | **36%** ⬇ |

## Interpretation

The capacity constraint hypothesis is **confirmed**: with only 3 fusion params (vs 3.1M), LoRA is
fully restored as the dominant mechanism (+478% vs +2.8% in Exp03, vs +533% in Exp02a).

However, SR dropped to 36% vs Exp02a's 56%. The learned weights are nearly uniform
(0.327 / 0.329 / 0.344), meaning early layers 8 and 16 each contribute ~1/3 of the signal.
The final layer (24) slightly dominates but not enough to outweigh the noise from early layers.

**Why early layers hurt:**
- Layer 8 encodes low-level features (edges, colors) — not useful for action prediction
- Layer 16 encodes mid-level features (object parts) — partially useful
- Layer 24 encodes high-level semantic features — most useful for task understanding
- Averaging across all three dilutes the high-quality final-layer signal with earlier noise

---

## File Index

| File | Purpose |
|---|---|
| `configs/pusht/exp05_weighted.py` | Experiment config |
| `asset/runs/pusht/exp03_multiscale/vlm_embeddings.pt` | Shared embedding cache (Exp03 layers 14/21/28) |
| `asset/runs/pusht/exp05_weighted/best_model.pt` | Best checkpoint |
| `asset/runs/pusht/exp05_weighted/training_log.json` | Loss curve |
| `asset/runs/pusht/exp05_weighted/eval_results.json` | SR + Wilson CI |
| `asset/runs/pusht/exp05_weighted/mechanistic/` | LoRA/component ablation plots |

---

## Hard Constraints

1. `state_dim=2` — agent position only. **Never add block or goal position.**
2. `inference_horizon=4` — must match training. **Cannot change post-training.**
3. `fusion_mode="weighted"` — satisfies capacity constraint (3 params << 32K LoRA)
4. `n_eval >= 50` before drawing SR conclusions (MPS non-determinism)
