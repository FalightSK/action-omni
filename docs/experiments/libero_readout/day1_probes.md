# Day-1 probes — VLM→action read-out on LIBERO

**Date:** 2026-08-02 · **Hardware:** RunPod RTX 5090 (32 GB), 48-core EPYC 7443P
**Backbone:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, frozen, 32 layers, d=960
**Data:** `HuggingFaceVLA/libero`, 40 tasks × 50 scenes = 2000 scenes; 7 instruction variants = **14,000 forward passes**
**Cost:** ~23 min GPU wall clock for extraction, ~45 min CPU for probes.

Suites: tasks 0–9 = LIBERO-10 (`long`), 10–19 = `goal`, 20–29 = `object`, 30–39 = `spatial`.

---

## 0. Setup facts that turned out to matter

| Fact | Value |
|---|---|
| Prompt layout | `<\|im_start\|>User:<image>{instruction}<end_of_utterance>\nAssistant:` |
| Tokens, `do_image_splitting=False` (SmolVLA's setting) | **T=81** — 64 image, 11 template scaffold, ~6 instruction |
| Tokens, `do_image_splitting=True` (HF **default**) | **T=1141** — 1088 image, ~6 instruction |
| Image block position | positions 5–68, i.e. **strictly before** the instruction |

The splitting flag is not reported in any VLA paper we know of, and it moves the vision:instruction token ratio by **20×**. Any claim about language attention mass is meaningless without stating it.

---

## 1. Result 1 — vision-token states are *structurally* instruction-invariant

The language model is causal and the image block precedes the instruction. Therefore image-token hidden states cannot depend on the instruction. Verified directly, same image, two unrelated instructions:

```
layer  0: max|Δ| over the 64 image tokens = 0.000e+00
layer  8: 0.000e+00     layer 16: 0.000e+00
layer 24: 0.000e+00     layer 32: 0.000e+00
```

Exact zero at every layer. Under batched execution (variable padding → different kernel tiling) it becomes fp16 noise: vision-pooled $\eta^2_{instr} \le 8.6\times10^{-8}$, ~7 orders of magnitude below the scene effect.

**Consequence.** 75 of 81 token states (**92.6%**) carry zero instruction information. Any read-out that pools or cross-attends uniformly over VLM tokens has its instruction sensitivity capped before a single gradient step. This is a *structural* bound, stronger and cheaper to establish than the norm-handicap argument in the protocol's §3.4 — it needs no assumption about norms at all.

This is also a design lever: **put the instruction before the image and all 64 vision states become instruction-conditioned.** Token order is a free, untested read-out variable.

---

## 2. Result 2 — the read-out's token subset decides whether depth matters at all

Ridge $R^2$ (5-fold, episode-grouped, λ by inner CV) predicting the action from pooled features, LIBERO-Goal:

| pooling | min | max | **range across layers** | argmax |
|---|---|---|---|---|
| all tokens | 0.340 | 0.392 | **0.053** | L5 |
| vision only | 0.286 | 0.334 | 0.048 | L29 |
| text only | 0.145 | 0.412 | 0.267 | L25 |
| **instruction only** | 0.151 | **0.415** | **0.264** | **L27** |

Two things:

1. **The 6 instruction tokens are the best linear action read-out — better than all 64 image tokens** (0.415 vs 0.334). They are last in a causal stack, so they have integrated the image. Depth is required to get there: 0.151 at L0 → 0.415 at L27.
2. **With all-token pooling the depth effect nearly vanishes** (range 0.053). With instruction-token pooling it is 5× larger.

**Implication for E2/E3.** The protocol's tap-depth sweep uses `pool(H^(j))` over all tokens. At that pooling there is barely a depth effect to find, so E2 would likely return a null and E3's Spearman test would be underpowered — not because A2 is false, but because the pooling washes it out. Fix the read-out before running the sweep.

---

## 3. Result 3 — the norm imbalance is real, depth-dependent, and driven by scaffold tokens

$\rho_j = \mathbb E\|h_{\text{text}}\|/\mathbb E\|h_{\text{vis}}\|$:

| layer | text/vis | **instr/vis** | implied instruction attention mass |
|---|---|---|---|
| 0 | 0.022 | 0.024 | 0.23% |
| 8 | 5.12 | 0.541 | 4.8% |
| 16 | 5.19 | 0.789 | 6.9% |
| 24 | 3.97 | 1.030 | 8.8% |
| 28 | 2.99 | **1.185** | **10.0%** |
| 32 | 1.06 | 1.018 | 8.7% |

A6 is **confirmed, with a correction**. The 5× "text" advantage at mid-depth is carried by the *template scaffold* tokens (`<|im_start|>`, `User`, `:`, `Assistant`) — classic attention-sink behaviour — not by the instruction. Instruction tokens only reach parity with vision tokens around L24. At L0 they are **45× smaller**.

Implied instruction attention mass at initialisation never exceeds 10%, and is 0.23% at L0. The protocol's §3.4 estimate of 11.1% assumed $T_v/T_t=8$; measured it is $64/6 = 10.7$ with a depth-varying norm ratio.

---

## 4. Result 4 — η² and Λ disagree, and Λ is the one that means something

$\Lambda_j$ = (goal-swap distance)/(paraphrase distance). $\Lambda\gg1$ = encodes meaning; $\Lambda\approx1$ = encodes surface form.

| suite | mean Λ (instr-pooled) | mean $\eta^2_{instr}$ |
|---|---|---|
| long | **3.54** | 0.095 |
| goal | **2.42** | 0.110 |
| object | **0.62** | **0.506** |
| spatial | 1.05 | 0.363 |

The two predictors rank the suites in **opposite order**. `object` has by far the *highest* instruction-variance (η²=0.51) and the *lowest* semantic discriminability (Λ=0.62 — paraphrasing moves the representation further than swapping to a different target object).

η² cannot separate meaning from wording; it counts both as "instruction variance". **P2 alone would actively mislead.** P1 is the discriminating predictor. (This reverses my earlier recommendation to make P2 primary — the measurement says otherwise.)

### ⚠️ Confound — read before citing the cross-suite numbers

The cross-suite Λ comparison is **confounded by lexical edit distance**, and it is my stimulus design that causes it:

- `object` originals share one template; the goal-swap partner changes **one noun** ("ketchup"→"milk") while my paraphrases change **3–4 verbs/prepositions**. Λ<1 is largely forced by that asymmetry.
- `goal`/`long` originals are lexically diverse, so a swap changes many tokens and Λ comes out high.

**Within-suite depth trends are valid. Cross-suite Λ levels are not.** The fix is a matched-edit-distance stimulus set (equal token-level edit distance for paraphrase and swap), which also fixes the same latent problem in the protocol's §3.1. Not yet run.

Scale calibration that is *not* confounded: $d_{\text{swap}}/d_{\text{scene}} \approx 0.88$–$1.19$ across all suites — swapping the instruction moves the instruction-token features about as much as replacing the entire scene.

---

## 5. Result 5 — CKA is not the flat negative control the protocol assumes

Measured CKA-to-final-layer, Goal, all-pooled: **min 0.549, max 1.000, sd 0.124.**

The protocol (§3.2, P6) argues CKA is insensitive by construction, deriving CKA=0.997 after annihilating principal direction 10 under a $\lambda_k \propto k^{-1}$ spectrum. Real spectra here are not power-law, and CKA varies substantially with depth.

CKA is still a poor predictor, but for a different reason than stated: *CKA-to-final-layer is a monotone proxy for distance-to-the-reference-layer*, so its variation is close to tautological. **The §3.2 closed-form argument should be retracted or restricted**; the empirical case against CKA stands on its own.

---

## 6. What this changes in the protocol

| Protocol item | Status |
|---|---|
| A6 (norm imbalance) | **Confirmed**, but depth-dependent and scaffold-driven; instruction tokens reach parity only at L24 |
| A2 (depth matters) | **Supported — conditional on read-out.** Range 0.264 (instr-pool) vs 0.053 (all-pool) |
| §3.2 P6 CKA closed-form | **Refuted as stated.** CKA sd=0.124, not flat |
| §3.2 P1 vs P2 | **P1 primary.** P2 confuses wording with meaning and inverts the suite ranking |
| §3.4 attention mass | Recompute: $T_v/T_t = 10.7$, $\rho$ depth-varying; mass ≤10%, and 0.23% at L0 |
| §3.1 stimulus set | Needs **edit-distance matching** between paraphrase and swap |
| E2/E3 read-out | Must **not** use all-token pooling, or the sweep returns a false null |
| A9 (eval is not GPU-bound) | **Extended:** feature extraction is *also* CPU-bound — 1.2 PFLOP over 1355 s = **0.4% GPU duty cycle** |

---

## 7. Files

```
scripts (on pod, /workspace/omni):
  variants.py      40 tasks x 3 hand-written paraphrases + swap/blank/nonsense
  build_scenes.py  2000-scene sampler, episode-spread
  extract.py       14,000 forwards -> feats.npy (2000,7,33,4,960) fp16
  probes.py        P1-P6 x 4 poolings x 4 suites
  plot1.py         fig1_layerwise.png
outputs (this dir):
  figs/fig1_layerwise.png
  probe_results.npz
```

## 8. Limitations

- Single backbone; cross-backbone replication (PaliGemma-3B) not yet run.
- Paraphrases for `object`/`spatial` (tasks 20–39) are rule-generated from a shared template; `long`/`goal` (0–19) are hand-written. See §4 confound.
- One camera view (`observation.images.image`); wrist camera unused.
- **No closed-loop evidence.** Everything here is frozen-feature geometry. The open-loop→closed-loop gap documented in `docs/experiments/pusht/exp03_multiscale.md` §10.4 applies in full.
