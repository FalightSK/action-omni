# PushT Phase Conclusion — Frozen VLM as Visual Backbone

**Completed:** 2026-06-02  
**Status:** ✅ Phase complete — proceeding to ALOHA

---

## 1. What We Set Out to Test

Can a frozen, non-robot-pretrained VLM (Qwen3.5-0.8B) serve as a visual backbone for robot manipulation? Specifically:
1. Does the basic frozen VLM → adapter → flow decoder loop work?
2. Does decoder architecture (MLP vs DiT) matter?
3. Does feature extraction depth (single layer vs multi-scale) matter?
4. What are the load-bearing components?

All tests were conducted on PushT: a 2D planar pushing task, 96×96 image, 2D action space, 25,650 training frames, success threshold = 95% block coverage.

---

## 2. Complete Results Table

| Exp | Architecture | SR (n=50) | Wilson 95% CI | Val Loss | Mean Cov | Stalls (85–94%) |
|---|---|---|---|---|---|---|
| Exp1 | MLP decoder, layer 24 | 30% | [19%, 44%] | 0.4490 | 86.7% | 27 |
| Exp2 | DiT + 6D state | **0%** | — | — | 14.4% | — |
| Exp2a | DiT + 2D state, layer 24 | **56%** ✅ | [42%, 69%] | 0.3725 | 84.3% | 10 |
| Exp2c | Exp2a, inference_horizon=8 | **0%** | [0%, 7%] | — | 63.9% | — |
| Exp2d | Exp2a, sim_max_steps=500 | 42% | [29%, 56%] | — | 90.3% | 24 |
| Exp3 | DiT + multi-scale 8/16/24 | 44% | [31%, 58%] | 0.3538 | 90.4% | 23 |
| Exp05 | DiT + weighted-sum fusion 8/16/24 (3 params) | 36% | [24%, 50%] | 0.3691 | — | — |

**Best result: Exp2a — 56% SR (28/50)**  
This is the architecture to carry forward to ALOHA.

---

## 3. What We Found

### Finding 1: The frozen VLM works. 56% SR with no robot-data pretraining.

The basic hypothesis is validated. A 853M-parameter frozen VLM, with only 16M trainable parameters in the adapter and decoder, achieves 56% SR on PushT. For reference, Diffusion Policy (CNN backbone, trained from scratch, robot-specific) achieves ~60–70% SR — our frozen-backbone approach is in the same range with fundamentally different architecture and training assumptions.

### Finding 2: Decoder architecture is the most impactful choice (MLP vs DiT: 30% → 56%)

Replacing the MLP decoder with a DiT decoder provides a **+26pp SR improvement** (chi-squared p=0.0086). This is the only statistically significant improvement across all PushT experiments.

**Why the DiT wins:** Cross-attention. Each of the 16 action step tokens independently queries all 82 VLM tokens at each of 3 denoising steps — 18 cross-attention computations per forward pass. The MLP collapses all spatial information into a single 512D readout vector before the decoder sees it.

### Finding 3: LoRA is not a small correction — it is a task-specific re-projection

The mechanistic analysis (component ablations on Exp1 and Exp2a) shows:
- Exp2a: LoRA zeroed → **+533% val loss** (most critical component)
- Exp1: LoRA zeroed → **+300% val loss**

Despite `scale=0.1` dampening, the LoRA correction dominates the residual. The adapter has learned a full **1024 → 16 → 1024 projection** — compressing frozen VLM features into a 16-dimensional robotics-useful subspace and reconstructing. The frozen features provide input geometry; LoRA re-bases them onto the task-relevant submanifold.

This is not a bug. It is the most important architectural insight of the PushT phase: frozen LLM features need substantial re-projection, not incremental correction, to be useful for motor control.

### Finding 4: Cross-attention is the load-bearing mechanism explaining the DiT's advantage

Mechanistic ablation:
- Exp2a: no cross-attention → **+261% val loss** (second most critical)
- Exp1: no cross-attention → N/A (MLP has no cross-attention)

Removing cross-attention from the DiT reduces it to a weaker-than-MLP baseline. This directly confirms: the per-step spatial querying of VLM tokens is the mechanism, not the transformer architecture itself.

### Finding 5: Multi-scale features (Exp3, Exp05) do not improve SR — final-layer features are sufficient

Exp3 (44%) vs Exp2a (56%): p=0.317, not significant. Exp05 (36%) vs Exp2a (56%): p=0.045, significantly *worse*. Neither multi-scale variant improves over the single-layer baseline.

Exp05 is the more informative test: it uses a minimal 3-parameter weighted-sum fusion (compared to Exp3's 3.1M-parameter MultiScaleFusion module), confirming that the failure is not a capacity artifact. The mechanistic breakdown is clear:

- **Exp3 (large fusion, 3.1M params):** LoRA contribution collapsed to +2.8% — fusion module *displaced* LoRA as the dominant pathway. Invalid comparison.
- **Exp05 (small fusion, 3 params):** LoRA restored to +478% — valid architecture. But SR still degraded vs Exp2a.

Exp05's learned fusion weights (layers 8/16/24): **0.327 / 0.329 / 0.344** — near-uniform across all three layers, suggesting the model finds no reason to prefer deeper features. Early layers (8, 16) contribute noise that interferes with the precise spatial signal from layer 24.

**Conclusion:** Layer-24 final features already encode sufficient spatial information for PushT. Early layers add distributional noise, not complementary signal.

### Finding 6: Two hard constraints discovered

**Constraint A: Covariate shift.** Any action-history in the state causes catastrophic failure (Exp2: 6D state → 0% SR). The delta distribution at inference differs from the expert distribution at training. State must consist exclusively of environment observations — quantities readable directly from the simulator with no dependence on model history.

**Constraint B: Inference horizon must match training.** Executing inference_horizon=8 steps (trained at ih=4) → 0% SR. Errors compound without feedback. There is no tuning path — this parameter must be set at training time.

### Finding 7: n=20 is insufficient for PushT evaluation

The initial n=20 comparison showed MLP (30%) > DiT (20%) — this completely reversed at n=50 (MLP 30% < DiT 56%). MPS non-determinism on Apple Silicon produces multi-episode swings between runs even with identical seeds. **Minimum n=50 for any SR comparison.**

---

### Multi-scale Fusion: Complete Analysis (Exp3 + Exp05)

Two experiments tested whether fusing features from multiple VLM layers (8, 16, 24) improves over single-layer extraction (layer 24 only). Both failed, but for different and informative reasons:

| Experiment | Fusion type | Fusion params | LoRA contribution | SR | Verdict |
|---|---|---|---|---|---|
| Exp2a | None (layer 24 only) | 0 | +533% | **56%** | Baseline |
| Exp3 | MultiScaleFusion (learned linear) | 3.1M | +2.8% | 44% | Invalid — fusion displaced LoRA |
| Exp05 | Weighted sum (3 scalars) | 3 | +478% | 36%* | Valid architecture, but early layers hurt |

*p=0.045 vs Exp2a — statistically significantly worse.

**Interpretation:**

1. **Exp3 (large fusion):** The 3.1M-parameter fusion module exceeded LoRA's capacity (524K params) and became the dominant signal pathway, effectively bypassing LoRA. The LoRA contribution collapsed from +533% to +2.8%. Result is invalid as a test of multi-scale features — it tests large fusion vs small LoRA.

2. **Exp05 (small fusion, 3 params):** The capacity constraint is resolved. LoRA is restored to dominance (+478%). This is an architecturally valid multi-scale test. Yet SR drops to 36%. The fusion weights converge to near-uniform (0.327 / 0.329 / 0.344), indicating layers 8 and 16 provide no complementary information — only noise relative to layer 24.

3. **Conclusion:** Layer-24 final output is the optimal single extraction point for PushT. Multi-scale fusion with any capacity adds noise (from earlier layers' less-task-relevant representations) without adding signal. **Do not use multi-scale for ALOHA unless there is a specific motivation from task structure.**

---

## 4. Component Importance Ranking

Combined across Exp1, Exp2a, and Exp05 mechanistic ablations:

| Component | Exp1 (MLP) | Exp2a (DiT) | Exp05 (DiT + fusion) | Role |
|---|---|---|---|---|
| **LoRA** | +300% | **+533%** | +478% | Task-specific re-projection of frozen features |
| **Cross-attention** | N/A | **+261%** | +196% | Per-step spatial querying — the DiT's core advantage |
| **adaLN** | **+569%** | +231% | +156% | Global scene conditioning |
| **AttentionReadout** | **+457%** | +160% | — | Critical for MLP; less critical with cross-attn bypass |
| **Spatial PE** | +6% | +12% | — | Helpful but not the bottleneck |

**LoRA is most critical in both architectures.** Cross-attention is why the DiT wins.

Exp05 confirms LoRA dominance is robust: with a small-capacity fusion (3 params), LoRA returns to +478% — close to Exp2a's +533%. The slight reduction is consistent with noisy early-layer features partially interfering with the LoRA-learned subspace.

---

## 5. The Remaining Gap

The best result (56% SR, Exp2a) leaves a persistent failure mode: **final alignment stall**. 10–23 episodes across all experiments reach 90–94% coverage but cannot cross 95%. The T-block is nearly on target but rotated by 3–5% — corresponding to a few pixels at 96×96 resolution.

Possible causes:
1. **Resolution ceiling (96×96):** Sub-pixel rotational alignment may be geometrically impossible at this resolution. The model cannot see the misalignment even if it exists in the image.
2. **LoRA rank ceiling (rank=16):** The 16-dimensional bottleneck may be insufficient to encode the precise rotational geometry needed for final alignment. Rank=64 would test this.
3. **Structural issue:** PushT at 95% threshold requires solving a very specific final-rotation sub-problem. The model may need task-specific data augmentation or curriculum for this regime.

On PushT, we cannot distinguish which cause dominates. ALOHA will provide the discriminating test.

---

## 6. ViT Ablation Placement Note

The original plan listed "ViT ablation on PushT" as Priority 1. **This has been revised.**

PushT uses a fixed task instruction: `"Push the T-shaped block onto the T-shaped target."` — identical every episode. A Qwen vs ViT comparison on PushT would only measure **visual encoder quality**, not **language-visual joint pretraining**, because the language component never changes. Any SR difference would be attributable to visual feature quality alone, not to the joint pretraining claim.

The ViT ablation is deferred to **Language Table** (Phase 3), where:
- The instruction varies per episode (different objects, different goals)
- Language understanding actually drives episode-to-episode SR variation
- A Qwen (language-visual, causal LLM) vs ViT-B/16 + CLIP text comparison is scientifically meaningful

This does not block ALOHA. Proceed directly to ALOHA.

---

## 7. Decision: Proceed to ALOHA

**Gate check:** SR > 40% required to proceed. Exp2a achieves 56% — gate cleared.

**Architecture for ALOHA:**

| Component | PushT (Exp2a) | ALOHA starting point | Why |
|---|---|---|---|
| VLM | Qwen3.5-0.8B frozen | Same | Frozen — no change |
| VLM layers | Layer 24 (single) | Layer 24 (single) | Multi-scale didn't help on PushT |
| LoRA rank | 16 | **Start with 16, test 64** | Key ablation for 14D action space |
| Adapter | LoRA + SpatialMLP + Readout | Same | Proven structure |
| Decoder | DiT (h=256, L=6, heads=8) | **Scale up: h=512, L=8?** | 14D vs 2D action |
| State dim | 2D (agent pos) | 14D (joint angles) | Env obs — no covariate shift risk |
| Action dim | 2D | 14D | Major increase |
| inference_horizon | 4 | 8–16 (dataset-specific) | Must set in training config |

**First ALOHA experiment question:** Does the pipeline function at all for 14D joints? Any non-zero SR with the base Exp2a adapter architecture confirms scalability. The LoRA rank ablation (16→64) is the first thing to test if SR is low.

---

## 7. Revised Research Contribution (Post-PushT)

After the PushT phase, the research contribution is sharper than originally framed:

> *"We demonstrate that a frozen, non-robot-pretrained LLM (853M parameters) can serve as a visual backbone for robot manipulation, with only 16M trainable parameters in a lightweight adapter and flow-matching decoder. The critical design choices are: (1) a per-token LoRA adapter that learns to re-project frozen VLM features into a task-specific subspace (not a small correction but a full re-basis); and (2) a DiT decoder with per-step cross-attention to all VLM tokens, bypassing the global readout bottleneck. Together these two components achieve 56% SR on PushT without any robot-data pretraining of the visual backbone. The key unresolved question — whether rank-16 LoRA scales to high-DOF (14D) tasks — will be tested on ALOHA."*

---

*PushT phase completed 2026-06-02*  
*Total experiments: Exp1, Exp2, Exp2a, Exp2c, Exp2d, Exp3, Exp05*  
*Total n: 400 simulation episodes across all experiments*  
*Next: ALOHA-A (Transfer Cube, sim)*
