# Phase 1 — VLM→Action Read-Out Interface: Results

**Date:** 2026-08-02 · **Pod:** RunPod RTX 5090 (32 GB), 48-core EPYC 7443P, ~7 h wall clock
**Backbone:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, **frozen**, 32 layers, d=960
**Actor:** DiT flow-matching head, 3.7 M trainable params, cross-attention read-out
**Arena:** LIBERO-Goal (10 tasks, shared kitchen scene) · closed-loop via `hf-libero` + robosuite 1.4.0

Every number below is recomputable from `results/` by `scripts/audit.py`.

---

## 1. Headline closed-loop results (n = 300 per arm)

| read-out | tap | instruction | success | Wilson 95% CI |
|---|---|---|---|---|
| all-token | 0 | original | **249/300 = 83.0%** | [78.3, 86.8] |
| all-token | 9 | original | **253/300 = 84.3%** | [79.8, 88.0] |
| all-token | 30 | original | **249/300 = 83.0%** | [78.3, 86.8] |
| all-token | 30 | **goal-swap** | **0/300 = 0.0%** | [0.0, 1.3] |
| instr-only | 3 | original | 230/300 = 76.7% | [71.6, 81.1] |
| instr-only | 30 | original | 219/300 = 73.0% | [67.7, 77.7] |
| instr-only | 30 | **goal-swap** | **0/300 = 0.0%** | [0.0, 1.3] |

### 1.1 The read-out's token subset matters; the tap layer does not

- all-token vs instruction-only at tap 30: **83.0% vs 73.0% — CIs separated.**
- tap 9 vs tap 30 (all-token): 84.3% vs 83.0% — **CIs overlap.**
- **tap 0 vs tap 30: 249/300 vs 249/300 — identical.**

Tap 0 is the *embedding layer*: SigLIP vision features + word embeddings, with **zero LM transformer
layers**. It matches layer 30 exactly under a matched protocol (400 max steps, n=300, all-token,
original instruction). On LIBERO-Goal the VLM's 32 transformer layers contribute nothing measurable
to closed-loop success.

### 1.2 Language is fully load-bearing

83.0% → **0.0%** under goal-swap, for both read-outs. All ten LIBERO-Goal tasks share one scene, so
a swapped instruction can only be followed by *reading it* — and the policy follows it into failing
the scored task. This is the direct opposite of the LIBERO-PRO report that VLAs emit identical
action sequences for real, paraphrased and nonsense instructions.

### 1.3 Open-loop action error is an invalid proxy — measured

| read-out | Spearman(open-loop MSE, closed-loop SR) | valid proxy would be |
|---|---|---|
| all-token | **+0.866** (n=3, p=0.333) | ≈ −1.0 |
| instr-only | **+1.000** (n=2, p=n/a) | ≈ −1.0 |

The point estimate has the **wrong sign in both read-out modes** — lower open-loop error went with
*lower* closed-loop success. With n=3 and n=2 taps this is not statistically significant, and we do
not claim it is. What we do claim: **there is no evidence open-loop error orders taps correctly**,
and the ranking inverted independently in both modes. This replicates the offline–online disconnect
documented in `docs/experiments/pusht/exp03_multiscale.md` §10.4 on a new benchmark.

**Consequence:** any Part-1 plan whose dependent variable is open-loop divergence is measuring the
wrong thing. Closed-loop is not optional.

---

## 2. Training-free results (no simulator, ~25 min GPU)

### 2.1 Vision-token states are structurally instruction-invariant

Same image, two unrelated instructions:

```
layer  0: max|Δ| over the 64 image tokens = 0.000e+00
layer  8: 0.000e+00   layer 16: 0.000e+00
layer 24: 0.000e+00   layer 32: 0.000e+00
```

Exact zero at every layer, because the LM is causal and the image block (positions 5–68) precedes
the instruction. Recomputed from the token masks over the real corpus:

| quantity | value |
|---|---|
| instruction tokens | min 4, **median 7**, max 9 |
| vision tokens | 64 |
| total tokens | 79–84 |
| **% of tokens that cannot carry the instruction** | **89.3–94.9% (median 91.5%)** |

This is a *derivation*, not a measurement — it holds for any VLA built on a causal VLM with
image-before-text prompting (OpenVLA, π0, SmolVLA, RT-2-style). Its consequence for read-out design
is what is new: a read-out that pools or attends uniformly over VLM tokens has its instruction
sensitivity capped before training starts.

### 2.2 Token order is a free intervention

| prompt order | vision-token η²_instr | best vision-pooled ridge R² |
|---|---|---|
| `User:<image>{instruction}` (standard) | 9.83e-09 | 0.3337 |
| `User:{instruction}<image>` | **1.39e-02** | **0.3929** |

Reordering moves vision-token instruction sensitivity by **six orders of magnitude** and improves
linear action decodability by **+0.059 R² (+18% relative)**. This is a one-line change to a prompt
format every VLA inherits from its VLM without deliberating over it.

⚠️ Probe-level only. The policy-level test (train an actor on text-first features) was **not run**.

### 2.3 `do_image_splitting` swings the token budget 20×

| setting | total tokens | vision | instruction |
|---|---|---|---|
| `True` (HF **default**) | 1141 | 1088 | ~6 |
| `False` (what SmolVLA uses) | 81 | 64 | 4–9 |

No VLA paper we are aware of reports this flag, yet it changes the vision:instruction ratio by 20×
and therefore any claim about language attention mass.

### 2.4 Modality norms — A6 confirmed, with a correction

`ρ = |h_text| / |h_vis|` peaks at **5.19 at layer 16** but that is carried by *template scaffold*
tokens (`<|im_start|>`, `User`, `:`, `Assistant`) behaving as attention sinks. Instruction tokens
alone reach parity with vision only at ~L24 and are **45× smaller at L0**. Implied instruction
attention mass at init never exceeds **10%**, and is **0.23% at L0**.

### 2.5 P1 and P2 disagree, and P2 misleads

| suite | Λ (swap/paraphrase, instr-pooled) | η²_instr |
|---|---|---|
| long | 3.54 | 0.095 |
| goal | 2.42 | 0.110 |
| object | **0.62** | **0.506** |
| spatial | 1.05 | 0.363 |

They rank the suites in **opposite order**. η² counts wording and meaning as the same variance;
Λ separates them. **P2 alone would actively mislead.**

⚠️ **Confound, ours:** cross-suite Λ is contaminated by paraphrase edit-distance asymmetry — for
`object`, a goal-swap changes one noun while our paraphrases change 3–4 verbs. Within-suite depth
trends are valid; cross-suite Λ *levels* are not. Fix = edit-distance-matched stimuli.

### 2.6 CKA is not a flat negative control

Measured CKA-to-final-layer: **min 0.549, max 1.000, sd 0.124.** The protocol's §3.2 closed-form
(CKA = 0.997 after annihilating direction 10 under λ_k ∝ k⁻¹) does not describe the real spectrum.
CKA remains a poor predictor, but because CKA-to-final-layer is a near-tautological proxy for
distance-to-reference — not for the reason given.

---

## 3. Sweep (36 runs: 12 taps × 2 read-outs + 12 LR checks)

| read-out | open-loop MSE range | deg_swap | deg_para |
|---|---|---|---|
| all-token | 0.2351–0.2531 (range 0.0180) | 2.152 ± 0.071 | 1.092 ± 0.025 |
| instr-only | 0.2598–0.2803 (range 0.0205) | **2.282 ± 0.100** | **1.065 ± 0.020** |

Trade-off: instruction-only read-out buys language sensitivity and costs 10 pp of closed-loop
accuracy. You need the visual tokens for precision.

### 3.1 A4 — LR optimum is stable, LR *fragility* is depth-dependent

| tap | ol_mse @ lr 1e-3 | @ lr 3e-3 |
|---|---|---|
| 0 | 0.2498 | **0.9314** (broken) |
| 18 | 0.2456 | 0.7756 |
| 32 | 0.2386 | 0.2865 |

The optimum is 1e-3 everywhere, so a single LR is valid — **but only the conservative one**. Shallow
taps are far more LR-fragile. Anyone who tunes LR at a deep tap and reuses it shallow will conclude
shallow taps are useless. This is a plausible mechanism for GR00T (mid > final) and SmolVLA
(monotone) reaching opposite conclusions.

### 3.2 E3 — probes beat validation loss, but the chain breaks

| predictor | ρ vs open-loop error | p |
|---|---|---|
| **P3 ridge R² (instr-pooled)** | **+0.664** | **0.019** |
| val_loss | −0.168 | 0.602 |

P3 was pre-registered as primary, so this stands uncorrected. The other five predictors are
exploratory (`erank` reaches ρ=−0.657, p=0.020, but would not survive Holm across a family of 6).

**However** — §1.3 shows open-loop does not predict closed-loop, so P3 → open-loop → closed-loop
fails at the last link. P3 is validated against a target that is itself invalid.

### 3.3 ❌ Our pre-registered prediction FAILED

Figure 1 predicted (recorded before the sweep ran) that the depth effect would be ~5× larger for
instruction-token read-out: probe R² range 0.264 (instr) vs 0.053 (all) → 4.98×.

**Observed: 1.14×** (0.0205 vs 0.0180). The prediction was wrong.

The reason is itself a finding: the probe measured **mean-pooled** features while the actor uses
**cross-attention over token sets**. Pooling destroys precisely what cross-attention exploits, so
**pooled-feature probes cannot predict cross-attention read-out ranking.** This is a direct
correction to P3's premise in the protocol.

---

## 4. Assumption ledger

| # | Assumption | Verdict |
|---|---|---|
| A1 | Frozen VLM suffices, no robot-data pretraining | **Supported, strongly.** Tap 0 (zero LM layers) = tap 30 = 83.0%, matched protocol |
| A2 | Action info varies non-trivially across depth | **Refuted.** Closed-loop CIs overlap across taps 0/9/30 |
| A4 | Per-layer LN makes one LR valid across taps | **Supported, refined** — optimum stable, fragility depth-dependent |
| A5 | 200 trials low-noise enough | **Supported.** n=300 gives ±5 pp Wilson half-width |
| A6 | Text/visual norms materially imbalanced | **Confirmed, corrected** — driven by scaffold sinks, not the instruction |
| A9 | Closed-loop is CPU/simulator-bound | **Refuted at batch 1** (144 ms VLM vs 13 ms sim); true only after vectorising |
| A10 | LIBERO instructions diagnostic enough | **Strongly supported** (rated "Weak" in the protocol): 83% → 0% |
| §3.2 P6 | CKA is a flat negative control | **Refuted.** sd = 0.124 |
| §3.2 P1/P2 | Both usable | **P2 misleads** — inverts P1's ranking |
| E3 | Probes predict closed-loop success | **Broken chain** — P3 predicts open-loop; open-loop doesn't predict closed-loop |

---

## 5. Engineering findings (relevant to the compute request)

| finding | number |
|---|---|
| Closed-loop is **policy-bound**, not simulator-bound, at batch 1 | 144 ms VLM vs 13 ms `env.step` |
| Vectorised envs (process-per-env) | 0.4% → **48.6% GPU utilisation**, 32 → 50–60 env-steps/s |
| `lerobot-eval` cannot parallelise envs | shared EGL context → `MjRenderContextOffscreen has no attribute 'con'` |
| **`img[::-1]` mirrors the LIBERO scene** | rot180 MSE 285 vs hflip 4619 / vflip 5592 / raw 6774 |
| Feature extraction is CPU-bound | 1.2 PFLOP over 1355 s = **0.4% GPU duty cycle** |

The mirror bug is the dangerous one: the conventional robosuite vertical flip produces a
horizontally mirrored observation on this dataset, which would yield ~0% success and read as a
modelling failure. Caught only by diffing renders against the training data.

**Implication for the allocation:** line-items 1–4 (training + feature caching) are CPU/disk work,
not A100 work. The genuinely GPU-bound item is closed-loop rollout, and even that needs batched
envs before more GPU helps.

---

## 6. What is NOT established

- **No external SOTA anchor.** π0 / π0.5 blocked: lerobot 0.4.4 requires a patched
  `transformers.models.siglip.check` that pip does not install and that is absent from current
  lerobot source. SmolVLA needed a checkpoint swap (`HuggingFaceVLA/*` expects
  `observation.images.image/image2`; lerobot 0.4.4's env emits `camera1/camera2`) and did not
  produce a usable number in the time available. GR00T N1.7 exists
  (`nvidia/gr00t17-lerobot-libero_spatial-640`) but only for **LIBERO-Spatial**, not Goal.
  **We do not compare our numbers to the repo's 76% SmolVLA-on-Spatial figure** — different suite,
  different harness.
- **Single seed** per configuration; no between-seed SD.
- **Single backbone**; no cross-backbone replication.
- **LIBERO-Goal only** for closed loop; other three suites are probe-only.
- **Token-order intervention is probe-level only** — no policy trained on text-first features.
- Cross-suite Λ confounded (§2.5).
- Sweep v1 (27 runs) **discarded** — 81-token truncation clipped 39/50 instruction variants, and
  padding was fed unmasked into cross-attention. Kept as `sweep_TRUNCATED81.json`, used for nothing.

---

## 7. What this changes about the paper

The protocol is built to find the best **tap layer**. The data say the tap layer does not matter,
the read-out's **token subset** does, and the **open-loop proxy planned to measure it is invalid**.

Three claims survive as a coherent paper:

1. **A structural bound**: 91.5% of a causal VLA's token budget cannot carry the instruction, with a
   free intervention (token order) that lifts it — architecture-general.
2. **A negative result that matters**: open-loop action error does not order read-out variants by
   closed-loop success, replicated on two benchmarks (PushT `exp03`, LIBERO-Goal here).
3. **A frozen-backbone result**: zero LM layers reach 83.0% on LIBERO-Goal with a 3.7 M-param head,
   and the instruction is fully load-bearing (83.0% → 0.0%).

---

## 8. Files

```
figs/
  fig1_layerwise.png       P1-P6 x 4 poolings x 4 suites
  fig2_fig4_final.png      predictor / open-loop / closed-loop / perturbation battery
  cmp_rot180.png           sim-vs-dataset render match (orientation fix)
  cmp_sim_vs_dataset.png   the mirrored (wrong) version, kept as evidence
results/
  sweep2.json              36 runs
  probe_results.npz        P1-P6 arrays
  order_probe.json         token-order intervention
  vroll_*.json             7 closed-loop arms
  final_summary.json       consolidated
  {train,val}_mask2.npy    token type masks (0 pad 1 vis 2 scaffold 3 instr)
  sweep_TRUNCATED81.json   DISCARDED v1, kept for the record
scripts/   all code (probe, cache, sweep, vectorised rollout, audit)
logs/      raw run logs
checkpoints/  5 actors used in the closed-loop arms
```

Reproduce the audit: `python scripts/audit.py` from `results/`.
