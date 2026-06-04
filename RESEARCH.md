# Research Plan — Frozen VLM as a Generalizable Visual Backbone for Robot Manipulation

**Project:** VLA Model Without Robot-Data Backbone Pretraining  
**Started:** 2026-06-01  
**Platform:** MacBook Pro M1 (MPS) → CUDA server for larger tasks  
**Status:** 🔄 Active — PushT phase complete ✅, preparing ALOHA

---

## 1. Research Goal

Build a **Vision-Language-Action (VLA) model** — not a task-specific policy — where:

- The visual backbone is a **frozen, non-robot-pretrained VLM** (Qwen3.5-0.8B)
- Language commands condition behavior **without any language-specific finetuning** — the VLM's internet-scale language understanding transfers directly
- The system **generalizes to novel visual distributions** that were never in the robot training demonstrations

The ultimate test is not "does it work on PushT" — it is: **"does it generalize to objects, colors, and instructions it has never seen during robot training, while task-specific baselines fail?"**

This directly challenges the orthodoxy of current VLA research (RT-2, OpenVLA, π0) which requires large-scale VLM finetuning on robot data. Our claim: **finetuning the backbone is unnecessary and potentially harmful to generalization — internet-scale pretraining already provides the representations needed, and only a lightweight adapter needs to be trained.**

---

## 2. The Benchmark Ladder

The benchmark experiments are a **validation scaffold**, not the contribution. Each benchmark gates the next by proving the architecture handles a new class of complexity.

```
PushT  ──────────────────────────────── ✅ COMPLETE
  Simple 2D push, no language needed
  Gate question: Does the architecture function at all?
  Gate threshold: SR > 40%
  Result: 56% SR (Exp2a, n=50) — PASSED

ALOHA (Transfer Cube / Insertion)  ──── ⏳ NEXT
  Bimanual 14-DOF, 3D workspace, contact-rich
  Gate question: Does it scale to physical complexity?
  Gate threshold: Non-trivial SR (any consistent task completion)
  Key architectural test: Does LoRA rank-16 hold for 14D action space?

Language Table  ─────────────────────── ⏳ After ALOHA
  2D push with natural language target selection
  Gate question: Does the VLM's language understanding actually condition behavior?
  Gate threshold: Policy follows different instructions differently
  Key test: Do text tokens contribute > 5% attention weight?

Custom OOD Dataset  ─────────────────── ⏳ The scientific contribution
  Pick & place, train on 4 colors, test on 6 colors
  Headline question: Does the frozen VLM generalize where baselines fail?
  Comparison: vs Diffusion Policy (CNN), vs frozen ViT + CLIP text, vs frozen CLIP
  This is where the paper's central claim is proven or falsified
```

**Why this is not a waste of time:** Each benchmark tests a specific failure mode. If PushT works but ALOHA fails → architecture doesn't scale. If ALOHA works but Language Table fails → language pathway is broken. If Language Table works but OOD fails → the generalization claim is wrong. Each gate has a specific fix if it fails.

---

## 3. Hypotheses

**H1 — Architecture Viability** *(PushT — PROVEN ✅)*
> A frozen VLM encoder + per-token LoRA adapter + DiT flow-matching decoder forms a complete VLA pipeline that achieves competitive performance on manipulation tasks without any backbone finetuning on robot data.

Evidence: 56% SR on PushT, 16M trainable params vs 853M frozen backbone.

**H2 — Physical Complexity Scaling** *(ALOHA — pending)*
> The same architecture scales to high-DOF bimanual manipulation without structural modification. The LoRA adapter's rank may need to scale with action space dimensionality (rank-16 for 2D actions, rank-64 for 14D actions).

What confirms it: Non-trivial ALOHA SR. LoRA rank ablation shows relationship between rank and action DOF.

**H3 — Language Grounding Transfers** *(Language Table — pending)*
> The frozen VLM's internet-scale language-visual alignment naturally conditions robot behavior: the same representations that bind "red" to red objects in web text also condition the policy to select the red object at execution time — without language-specific finetuning.

What confirms it: Policy SR differs significantly between different language commands. Text token attention > 5%.

**H4 — Out-of-Distribution Generalization** *(Custom OOD — the headline claim)*
> Because the backbone was never finetuned on robot data, policies trained with our approach generalize to novel visual categories that were never in the training demonstrations. Task-specific baselines trained from scratch fail on the same distribution shift.

What confirms it: SR on unseen colors/instructions significantly above Diffusion Policy baseline and frozen ViT baseline. SR above or comparable to CLIP-based baseline on compositional language OOD (where CLIP was not designed to generalize).

---

## 4. Contributions

### Primary Contribution: A validated frozen-VLM VLA architecture for OOD generalization

A single architecture that progressively scales from 2D manipulation → high-DOF manipulation → language-conditioned control → out-of-distribution generalization, without ever modifying the visual backbone. The system demonstrates that:
- Internet-scale language-visual pretraining provides transferable representations for manipulation
- Only the adapter (16M params) needs task-specific training
- Language grounding comes from the backbone, not from robot-data finetuning

### Secondary Contribution: Mechanistic understanding of frozen VLM adaptation

From the PushT mechanistic analysis (Exp1 + Exp2a):

1. **LoRA acts as a task-specific projection, not a correction.** The adapter learns a 1024→16→1024 re-basis of frozen features — not a small perturbation but a full subspace projection. LoRA zeroed → +533% validation loss. This reframes why lightweight adapters work for VLMs.

2. **Per-step cross-attention is the load-bearing mechanism.** DiT cross-attention from action tokens to all VLM tokens (bypassing the readout bottleneck) explains the full 30%→56% SR improvement. Cross-attn removed → +261% loss.

3. **Covariate shift is catastrophic for action-history state.** Any action-history in the state vector causes complete policy collapse (0% SR). State must consist exclusively of environment observations.

4. **Multi-scale feature fusion requires capacity-constrained fusion.** Naive linear projection (3M params) displaces LoRA and overfits training distribution. Valid fusion must have less capacity than LoRA (< 32K params) — e.g., learned weighted sum (3 scalars).

### Tertiary Contribution: Benchmark ladder methodology

A structured, statistically validated protocol for testing VLA architectures progressively — from 2D planar → high-DOF → language-conditioned → OOD generalization. With minimum n=50 evaluation requirement and chi-squared significance testing.

---

## 5. Advantages and Research Positioning

| Advantage | Compared To | Why It Matters |
|---|---|---|
| No backbone finetuning on robot data | RT-2, OpenVLA, π0 | Removes most expensive step in VLA development |
| Language grounding at zero cost | Diffusion Policy, ACT | Language commands work without language-specific training |
| OOD visual generalization by design | All task-specific policies | Frozen backbone has seen the full internet — unseen objects are known to it |
| 1.9% trainable parameters | Full VLM finetuning | 16M trained vs 853M frozen — accessible to non-TPU researchers |
| Modular: swap backbone freely | Fixed-architecture systems | Better VLM → better policy, zero re-engineering |

**The structural argument:** Task-specific policies learn visual features from robot demonstration data. A robot dataset of 25,650 frames contains perhaps 50 distinct visual configurations. The internet contains billions. When the test distribution shifts (new object color, different lighting), task-specific features are out of distribution. Frozen VLM features are not — the backbone has already seen the variation.

---

## 6. Related Work

### Supporting Evidence (Frozen Backbone Works)

| Paper | What They Show | Gap vs This Work |
|---|---|---|
| **R3M** (Nair et al., 2022) | Frozen ResNet → robot policies | Vision-only, no language |
| **MVP** (Radosavovic et al., 2023) | Frozen MAE-ViT → manipulation | Vision-only, no language |
| **RoboFlamingo** (Li et al., 2023) | Frozen VLM + finetuned head | Partially finetunes visual encoder |

These prove frozen visual features work — but use vision-only encoders. **None test whether language-visual pretraining (VLM) is better than vision-only (ViT) for generalization.** Our ViT ablation fills this gap.

### Opposing Position (Finetuning Is Necessary)

| Paper | Their Implicit Claim | Our Counter |
|---|---|---|
| **RT-2** (Brohan et al., 2023) | Finetuned VLM → emergent generalization | Frozen VLM + adapter may generalize comparably at 50× lower cost |
| **OpenVLA** (Kim et al., 2024) | 7B finetuned VLM outperforms all | Finetuning may destroy internet-scale representations for unseen distributions |
| **π0** (Black et al., 2024) | VLM + flow matching (finetuned) for dexterous tasks | Same decoder architecture — we test frozen backbone variant |
| **Octo** (Team, 2023) | Large-scale robot data pretraining is the path | Internet-scale VLM pretraining may be a better prior |

**The gap no paper has filled:** None of these papers compare finetuned VLM against frozen VLM on an OOD generalization test. They show finetuned VLMs work — they never test whether finetuning helps or hurts generalization relative to frozen. This is the experiment we run.

### Critical Baseline: CLIP

CLIP (Radford et al., 2021) was trained contrastively on image-text pairs. It already aligns "orange" with orange objects zero-shot. For color OOD tests, CLIP is a strong baseline that cannot be ignored. We must explicitly compare against:

- **CLIP vision + CLIP text + adapter + DiT decoder** — isolates joint VLM encoding vs. separate CLIP encoders
- **Frozen ViT + CLIP text + adapter + DiT decoder** — isolates VLM language understanding vs. CLIP language

If Qwen beats CLIP on compositional/relational language OOD ("pick the cube between the two markers") where CLIP's contrastive objective doesn't apply, the contribution is clean.

---

## 7. Architecture (Current Best: Exp2a)

```
PIL Image + Task Text
       │
Qwen3.5-0.8B (853M, FROZEN, bfloat16)
       │  hidden states from layer 24 (single layer — multi-scale not justified by PushT)
       │
VLMTokenAdapter (~5.9M trainable params)
  Stage 0 — MultiScaleFusion    [Exp3+, use ONLY with capacity-constrained fusion]
  Stage 1 — PerTokenLoRA        rank=16 (PushT) / rank=64 (ALOHA TBD)
  Stage 2 — SpatialAwareMLP     DINO-style 2D positional encoding
  Stage 3 — AttentionReadout    global context (512D) for adaLN
       │                   ↓
       │          per-token tokens (B, 82, 512) for DiT cross-attention
       │
cat(context, state) → cond (B, 514)
       │
DiTFlowDecoder (~10.1M trainable params)
  16 action tokens, self-attn + cross-attn to 82 VLM tokens
  adaLN conditioning per block
  OT-CFM: 3-step Euler inference
       │
Receding-horizon control: execute 4 steps, replan
```

**Two information pathways to DiT:**
1. `context` (readout → adaLN): global scene summary, modulates LayerNorm scale/shift
2. `vlm_tokens` (all 82 tokens → cross-attn): local per-step spatial queries

**Hard constraints (never violate):**
- State = environment observations only. Never action predictions or history.
- inference_horizon at inference = inference_horizon at training.
- n_eval ≥ 50 before drawing SR conclusions (MPS non-determinism).

---

## 8. Multi-Scale Fusion — Status and Next Steps

Exp3 showed that naive `Linear(3072→1024)` fusion is architecturally invalid:
- MultiScaleFusion (3.1M params) displaced LoRA — LoRA became irrelevant (+2.8% loss when zeroed vs +533% in Exp2a)
- The large projection overfits training distribution → larger offline-online gap
- Multi-scale did not improve SR (44% vs Exp2a's 56%, p=0.317)

**Valid fusion alternatives (to test after ALOHA rank ablation):**

| Approach | Capacity | Principle |
|---|---|---|
| Scalar weighted sum | 3 params | Fusion must have less capacity than LoRA |
| Per-token layer gating (small MLP) | ~66K params | Allow per-token layer selection |
| Separate LoRA per layer, no fusion | 3× LoRA | Preserve rank constraint per layer |
| Per-layer project to 64D then concat | ~393K params | Bottleneck per layer before fusion |

**Decision:** Do not retry multi-scale on PushT. Test on ALOHA after rank ablation. Use scalar weighted sum as the first attempt (3 params — cleanest test of hypothesis).

---

## 9. OOD Experiment Design

The custom dataset must test at least three types of OOD to separate what each baseline can and cannot handle:

**OOD Type 1 — Visual category shift (color/texture)**
- Train: pick red, blue, green, yellow cube
- Test: pick orange, purple cube (never in training demos)
- CLIP may handle this (has seen orange things in contrastive training)
- Baseline: CLIP vision + CLIP text policy

**OOD Type 2 — Compositional/relational language** *(CLIP cannot do this)*
- Train: "pick up the cube on the left" / "pick up the cube on the right"
- Test: "pick up the cube closest to the bowl" / "pick up the cube between the two markers"
- CLIP's contrastive training does not teach spatial relational reasoning
- Qwen (causal LLM trained on language describing spatial relationships) should handle this
- This is the decisive comparison where VLM > CLIP is expected

**OOD Type 3 — Novel instruction phrasing**
- Train: "pick up the red cube"
- Test: "get the crimson block" / "grasp the scarlet object" / "retrieve the red one"
- Tests whether language generalization extends to synonyms and paraphrases
- Both CLIP and Qwen may handle color synonyms
- Tests rich language understanding

**OOD Type 2 is the strongest test** — it requires compositional reasoning that contrastive training fundamentally cannot provide. A positive result here distinguishes VLM from CLIP and directly supports H4.

---

## 10. Required Baselines

All baselines must use the **same DiT decoder** and **same adapter structure** (only the backbone changes), to isolate the backbone contribution:

| Baseline | Visual Encoder | Language Encoder | Fusion | Purpose |
|---|---|---|---|---|
| **Ours** | Qwen VLM (joint) | Qwen VLM (joint) | Same transformer | Full system |
| **Frozen ViT-B/16** | ViT-B/16 | CLIP text encoder | Late concatenation | Is LLM backbone better than vision-only? |
| **Frozen CLIP** | CLIP ViT | CLIP text encoder | Contrastive-aligned | Is joint causal LLM better than contrastive? |
| **Diffusion Policy (CNN)** | ResNet | None / CLIP text | N/A | SOTA task-specific baseline |
| **VLM-LoRA backbone** | Qwen + backbone LoRA | Qwen + backbone LoRA | Joint | Is frozen strictly better than lightly finetuned? |

**Most critical:** ViT-B/16 ablation. If Qwen SR ≈ ViT SR → language-visual pretraining provides no advantage → the core claim is weakened. This must be run on PushT first (cheapest test, 8hr training) before committing to the full OOD experiment.

---

## 11. Contingency Plans

### If ALOHA fails (architecture doesn't scale)

**Fix A:** Increase LoRA rank (16→64) — directly tests information capacity hypothesis. Rank scaling with action DOF becomes a publishable finding.

**Fix B:** Scale DiT decoder (hidden=512, more layers) — if the bottleneck is decoder capacity, not adapter.

**Fix C:** Narrow scope — skip ALOHA, design OOD custom dataset as 2D pick-and-place (like Language Table). ALOHA is validation, not contribution.

### If language conditioning never activates (text tokens ignored)

**Fix:** Dedicated text readout — force a separate text context vector alongside the image readout:
```python
text_context = mean_pool(tokens[~img_mask])   # always extracts language
img_context  = self.readout(tokens[img_mask]) # spatial focus
cond = cat([img_context, text_context, state])
```
The finding becomes: *joint VLM encoding does not automatically activate language conditioning — a forced text pathway is necessary, but once added, language generalization transfers from internet pretraining.*

### If OOD generalization fails (frozen Qwen ≤ CLIP or ViT)

**Sub-failure: CLIP beats Qwen on color OOD**
→ Test OOD Type 2 (compositional language). If Qwen wins there, contribution is: *color-level OOD requires only contrastive alignment (CLIP), but relational/compositional language OOD requires causal VLM pretraining.*

**Sub-failure: Nothing generalizes**
→ Reframe as negative result: *frozen backbone generalization has limits — we characterize exactly where those limits are.* Publishable as a findings paper.

**Sub-failure: Qwen and CLIP both generalize, Diffusion Policy fails**
→ Language-visual alignment is the key factor, not VLM specifically. Contribution: *language-visual pretraining (frozen) is necessary and sufficient for OOD manipulation generalization; task-specific visual training is the limiting factor.*

### Universal Backup: VLM-LoRA

If frozen-only is insufficient, add rank-16 LoRA **on the Qwen backbone itself** (~1M params). Ablation becomes:

| Config | Trainable | OOD SR |
|---|---|---|
| Frozen backbone (current) | 16M | X% |
| + VLM backbone LoRA rank-16 | ~17M | ? |
| Full VLM finetuning (RT-2 style) | 853M | ? |

Contribution: *minimum backbone adaptation for generalization is rank-16 LoRA (~1M params), not full finetuning (853M params). Full finetuning does not improve generalization over lightweight LoRA.*

---

## 12. Decision Tree

```
ViT ablation on PushT (8hr, 1 run)
         ↓
Qwen SR >> ViT SR?
  YES → Language-visual pretraining matters. Proceed with full claim.
  NO  → Efficiency claim only. Design OOD for compositional language (where CLIP fails).
         ↓
ALOHA rank-16: non-trivial SR?
  YES → Architecture scales. Proceed to Language Table.
  NO  → Test rank-64. Works? → rank scaling finding. Still fails? → narrow scope to 2D OOD.
         ↓
Language Table: text tokens > 5% attention?
  YES → Language transfers naturally. Proceed.
  NO  → Add dedicated text readout (1-day fix). Retest.
         ↓
Custom OOD Type 1 (color): Qwen > Diffusion Policy?
  YES → Basic generalization claim holds.
  NO  → Fundamental failure. Frozen backbone insufficient for this OOD type.
         ↓
Custom OOD Type 2 (relational language): Qwen > CLIP?
  YES → Strong paper. VLM > contrastive alignment for compositional tasks.
  NO  → CLIP is sufficient. Contribution is efficiency, not generalization quality.
```

---

## 13. Publishability Criteria

**Workshop paper (IROS/CoRL workshop):** PushT + mechanistic analysis alone. Already achievable.

**Conference paper (ICRA/IROS main):** PushT + ALOHA + mechanistic analysis + ViT ablation. Demonstrates architecture scales and language-visual pretraining matters.

**Strong conference paper (CoRL/ICLR):** Full ladder + OOD generalization on custom dataset + CLIP comparison. Demonstrates frozen VLM generalizes where baselines fail.

**Top-tier (NeurIPS/ICLR spotlight):** Full ladder + OOD generalization + VLM-LoRA ablation showing minimum adaptation needed + Language Table language following. Provides a complete characterization of when frozen backbone is sufficient, when minimal finetuning helps, and what baselines it beats and why.

---

## 14. Experiment Roadmap

### PushT Phase ✅ Complete

| Exp | Architecture | SR (n=50) | Finding |
|---|---|---|---|
| Exp1 | MLP, layer 24 | 30% | Baseline |
| Exp2 | DiT, 6D state | 0% | Covariate shift — never use action history |
| Exp2a | DiT, 2D state | **56% BEST** | Cross-attn is load-bearing mechanism |
| Exp2c | DiT, ih=8 | 0% | Inference horizon is a hard training constraint |
| Exp2d | DiT, ms=500 | 42% | MPS variance — 300-step limit not the cause |
| Exp3 | DiT, multi-scale | 44% | Naive linear fusion invalid — displaces LoRA |

### ViT Ablation (Priority 1 — Before ALOHA)

| Exp | Backbone | SR (n=50) | Purpose |
|---|---|---|---|
| VIT-A | Frozen ViT-B/16, same adapter/DiT | TBD | Is language-visual pretraining better than vision-only? |

Run this on PushT first. 1 precompute run + 1 training run. If Qwen >> ViT: H2 evidence. If Qwen ≈ ViT: reframe early.

### ALOHA Phase (Priority 2)

| Exp | Config | Question |
|---|---|---|
| ALOHA-A | Exp2a adapter, rank-16 | Does pipeline function for 14D? |
| ALOHA-B | rank-64 ablation | Does rank matter for high-DOF? |
| ALOHA-C | Multi-camera | Overhead + wrist → better precision? |

### Language Table Phase (Priority 3)

| Exp | Config | Question |
|---|---|---|
| LT-A | Exp2a + forced text pathway | Does language actually condition behavior? |
| LT-B | Multi-instruction test | Does policy differentiate between instructions? |

### Custom OOD Phase (Priority 4 — The Contribution)

| Exp | OOD Type | Question |
|---|---|---|
| OOD-1 | Color shift (4→6 colors) | Basic visual generalization |
| OOD-2 | Relational language | VLM vs CLIP on compositional reasoning |
| OOD-3 | Instruction rephrasing | Language generalization breadth |

### Baselines (Run Alongside Each Phase)

| Baseline | When | Purpose |
|---|---|---|
| Diffusion Policy CNN | With PushT/ALOHA | SOTA task-specific comparison |
| Frozen ViT + CLIP text | With Language Table/OOD | Separate encoding vs joint VLM |
| Frozen CLIP (vision+text) | With OOD | Contrastive vs causal LLM |
| VLM-LoRA (backbone LoRA) | With OOD | Minimum finetuning needed |

---

*Last updated: 2026-06-03*  
*Status: PushT phase complete. ViT ablation next priority before ALOHA.*  
*See docs/conclusions/pusht.md for full PushT findings. See MILESTONES.md for progress tracking.*
