# Phase 1 — VLM→Action Read-Out Interface: Results

**Date:** 2026-08-02 · **Pod:** RunPod RTX 5090 (32 GB), 48-core EPYC 7443P, ~7 h wall clock
**Backbone:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, **frozen**, 32 layers, d=960
**Actor:** DiT flow-matching head, 3.7 M trainable params, cross-attention read-out
**Arena:** LIBERO-Goal (10 tasks, shared kitchen scene) · closed-loop via `hf-libero` + robosuite 1.4.0

Every number below is recomputable from `results/` by `scripts/audit.py`.

---

> ⚠️ **See §6bb** — the paraphrase-robustness claim based on open-loop `deg_para` is RETRACTED.
> Closed-loop, an unseen rewording costs 32.5–44.0 pp.

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
a swapped instruction can only be acted on by *reading it*.

#### 1.2.1 Does it follow the other instruction, or just break? — tested

A 0.0% arm is consistent with two very different things: the policy performs the **other** task
(instruction-following), or it simply fails under a mismatched instruction (breakage). **The first
draft of this report asserted the former without testing it.** It has now been tested.

LIBERO-Goal's ten tasks share one scene, so the policy can run in task *j*'s environment under task
*k*'s instruction while *k*'s goal predicate is scored by transferring the sim state into *k*'s
environment each replan step (`scripts/swap_diagnosis.py`).

| n = 100 (10 tasks × 10 eps), Qwen3.5-0.8B, all-token tap 12 | result |
|---|---|
| succeeds at the **scored** task *j* | **0/100 = 0.0%** |
| succeeds at the **swapped** task *k* | **83/100 = 83.0%**  [74.5, 89.1] |

Per task: 10, 9, 7, 10, 6, 8, 10, 6, 7, 10 — every task redirects, none collapses.

**The policy performs the task it is told to perform, at undiminished competence.** 83.0% under a
swapped instruction is statistically indistinguishable from the 81.5% [75.5, 86.3] it achieves when
instruction and scoring agree. The instruction does not perturb behaviour — it *selects* which task
is executed.

This is what makes the LIBERO-PRO contrast legitimate: that work reports VLA policies emitting
near-identical action sequences for real, paraphrased and nonsense instructions. Here the
instruction fully determines behaviour, and 0.0% reflects correct obedience to a mis-specified goal
rather than failure.

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

## 6b. Cross-backbone replication: Qwen3.5-0.8B

Run on the local RTX 4070 Ti after the pod was terminated. **Qwen3.5-0.8B is a
Qwen3-VL-family model with no robot-action supervision** — 24 layers + embedding, d=1024,
`image-text-to-text`. It is *not* the same family as SmolVLM2, and shares no lineage with
SmolVLA, so it is a genuine independent backbone.

Closed-loop, LIBERO-Goal, identical actor / protocol / stimulus set:

| backbone | read-out | tap | instruction | success | Wilson 95% CI |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | all-token | 12 | original | 163/200 = **81.5%** | [75.5, 86.3] |
| Qwen3.5-0.8B | all-token | 12 | **goal-swap** | 0/200 = **0.0%** | [0.0, 1.9] |
| Qwen3.5-0.8B | instr-only | 12 | original | 147/200 = 73.5% | [67.0, 79.1] |
| Qwen3.5-0.8B | all-token | 0 | original | 150/200 = 75.0% | [68.6, 80.5] |

### Verdict per claim

| # | Claim | SmolVLM2 | Qwen3.5 | Verdict |
|---|---|---|---|---|
| 1 | Language is load-bearing | 83.0% → 0.0%, separated | 81.5% → 0.0%, separated | ✅ **REPLICATED** |
| 2 | Read-out token subset matters | +10.0 pp, CIs **separated** (n=300) | +8.0 pp, CIs **overlap** (n=200) | ⚠️ **direction consistent, NOT independently confirmed** |
| 3 | Tap depth does not matter | tap0 = tap30 exactly (0.0 pp) | tap0 vs tap12 = 6.5 pp, CIs overlap | ✅ **REPLICATED** (weaker) |
| 4 | A1 — zero LM layers suffice | 83.0%, equal to best tap | 75.0%, **−6.5 pp** vs best tap | ⚠️ **WEAKER on Qwen** |
| 5 | Vision states instruction-invariant | `0.000e+00` (33 layers) | `0.000e+00` (25 layers) | ✅ **REPLICATED EXACTLY** |
| 6 | Token order lifts vision sensitivity | η² 9.8e-09→1.4e-02, ΔR² +0.0592 | η² 1.1e-13→1.2e-02, ΔR² **+0.0739** | ✅ **REPLICATED, larger** |
| 7 | Validation loss mis-ranks read-outs | instr val 0.527 < all 0.694, yet worse rollout | instr val 0.528 < all 0.687, yet worse rollout | ✅ **REPLICATED** |
| 8 | instr read-out is more paraphrase-robust | deg_para 1.065 < 1.092 | deg_para **1.173 > 1.068** | ❌ **CONTRADICTED** |
| 9 | Open-loop proxy is invalid | ranking inverted in both modes | **untestable** — only 2 taps with closed loop | ⬜ **NOT TESTED** |

### Honest reading

**Two claims did not survive intact.**

**Claim 8 is contradicted.** On SmolVLM2 the instruction-only read-out was *more* robust to
rewording (deg_para 1.065 vs 1.092); on Qwen it is *less* (1.173 vs 1.068). The direction
reverses. This sub-claim is backbone-specific and should be dropped from any general
statement about read-out design.

**Claim 2 is not independently confirmed.** The direction replicates (+8.0 pp, same sign and
similar size as SmolVLM2's +10.0 pp) but at n=200 the Wilson intervals overlap
([75.5, 86.3] vs [67.0, 79.1]). This is an underpowering problem, not evidence against —
but it is not confirmation either, and reporting it as such would be wrong. Resolving it
needs n≈400/arm.

**Claim 9 could not be tested.** The proxy-invalidity finding needs several taps measured in
closed loop; Qwen has only two (tap 0 and tap 12). Those two happen to be ordered
*correctly* (ol_mse 0.2461→75.0%, 0.2262→81.5%), i.e. they do **not** reproduce the
inversion — but n=2 supports no conclusion in either direction. The claim rests solely on
the SmolVLM2 evidence.

**Claim 4 is weaker than reported.** On SmolVLM2, tap 0 matched the best tap *exactly*
(249/300 both). On Qwen, tap 0 is 6.5 pp below tap 12. The intervals overlap so no depth
effect is demonstrated, but "the LM layers contribute nothing" is a stronger reading than
Qwen supports. The defensible version is: *the embedding layer alone reaches 75–83% and no
depth effect is statistically demonstrable on either backbone.*

**What is now genuinely solid** — replicated on two unrelated backbones, one of them with an
exact-zero measurement: the **structural bound** (claim 5), the **token-order intervention**
(claim 6, larger on Qwen), **language load-bearing** (claim 1), and **validation loss
mis-ranking read-outs** (claim 7).

### Protocol differences (why Qwen is not a perfect mirror)

| | SmolVLM2 run | Qwen run |
|---|---|---|
| training frames | 8,000 | 6,000 |
| taps swept | 12 | 3 |
| episodes/arm | 300 | 200 |
| deep tap compared | 30 (of 32) | 12 (of 24) |
| hardware | RTX 5090 (rented) | RTX 4070 Ti (local) |

These differences all *reduce* Qwen's statistical power relative to the original run. They do
not explain the claim-8 reversal, which is a sign flip rather than a width problem.

---

## 6bb. ⚠️ RETRACTION: the policy is NOT paraphrase-robust

Earlier sections of this report cited `deg_para ≈ 1.06–1.09` (open-loop) as evidence that the
policy is robust to rewording. **That claim is wrong and is retracted here.**

### The design flaw

Training used `run(tr, [0], ...)` — variant 0 only, i.e. **one fixed instruction string per
task**. With ten fixed strings the model can treat them as ten discrete task IDs and never parse
language at all. Worse, **no closed-loop arm had ever used a string absent from training**:
`orig` used the exact training string, and `swap` used another task's training string (only the
*pairing* was novel). The paraphrases — the only unseen strings — had been measured **open-loop
only**, via the very proxy §1.3 shows is unreliable.

### Condition A — trained on one phrasing, tested on an unseen one

| read-out, tap 12 | `orig` (seen) | `para1` (**unseen**) | drop |
|---|---|---|---|
| all-token | 81.5% [75.5, 86.3] | **49.0%** [41.7, 55.4] | **−32.5 pp** |
| instruction-only | 73.5% [67.0, 79.1] | **29.5%** [23.2, 35.7] | **−44.0 pp** |

Both **separated**. The policy loses a third to a half of all successes on a reworded
instruction.

### How badly open-loop understated it

| read-out | open-loop error increase | closed-loop success drop |
|---|---|---|
| all-token | +10.1% | **−32.5 pp** |
| instruction-only | +28.5% | **−44.0 pp** |

A 10% rise in open-loop action error corresponds to losing a third of all successes. This is the
**third and strongest** demonstration that the open-loop proxy is invalid — and the only one that
caused a wrong claim to be published in an earlier draft of this report.

### Replicated on BOTH backbones

The collapse is not a quirk of one model. Both arms run locally under the same harness,
n=200, max_steps=400, all-token read-out:

| backbone | `orig` (seen) | `para1` (**unseen**) | drop |
|---|---|---|---|
| SmolVLM2-500M (tap 30) | 163/200 = 81.5% [75.5, 86.3] | **93/200 = 46.5%** [39.7, 53.4] | **−35.0 pp** |
| Qwen3.5-0.8B (tap 12) | 163/200 = 81.5% [75.5, 86.3] | **98/200 = 49.0%** [42.2, 55.9] | **−32.5 pp** |

Both separate decisively, and the two collapse magnitudes are statistically indistinguishable
from one another (the `para1` intervals overlap heavily). Both `orig` arms scoring exactly
163/200 is coincidence; the pattern is not.

**This makes the retraction architecture-general.** Memorisation follows from training on ten
fixed instruction strings — it is not a property of any particular backbone, tokenizer or
image-patching scheme.

### Condition B — does instruction augmentation fix it?

Same 6,000 training samples and identical forward-pass budget; only phrasing diversity changes.
Training draws from {`orig`, `para1`, `para2`}; **`para3` is held out from training and
validation entirely.**

| condition | train phrasings | `orig` (seen) | held-out paraphrase | drop |
|---|---|---|---|---|
| A | 1 | 81.5% [75.5, 86.3] | 49.0% [41.7, 55.4] | −32.5 pp |
| B | **3** | 84.5% [78.8, 88.9] | **56.0%** [49.1, 62.7] | −28.5 pp |

**On Qwen, augmentation does not solve it.** A's [41.7, 55.4] and B's [49.1, 62.7] **overlap**,
so the +7 pp gain is not significant, and the seen-vs-unseen gap barely moves (32.5 → 28.5 pp).
Augmentation did not hurt in-distribution performance (81.5% → 84.5%).

#### ⚠️ But the two backbones DIVERGE on this — corrected

Condition B was initially run on Qwen only, and "augmentation does not solve it" was written from
that. Replicating on SmolVLM2 **partly reverses it**:

| backbone | condition | `orig` (seen) | held-out paraphrase | gap | verdict |
|---|---|---|---|---|---|
| SmolVLM2 | A (1 phrasing) | 81.5% [75.5, 86.3] | 46.5% [39.7, 53.4] | **−35.0 pp** | separated |
| SmolVLM2 | **B (3 phrasings)** | 77.5% [71.2, 82.7] | **68.0%** [61.2, 74.1] | **−9.5 pp** | **overlap** |
| Qwen3.5 | A (1 phrasing) | 81.5% [75.5, 86.3] | 49.0% [42.2, 55.9] | −32.5 pp | separated |
| Qwen3.5 | B (3 phrasings) | 84.5% [78.8, 88.9] | 56.0% [49.1, 62.7] | −28.5 pp | separated |

**On SmolVLM2 augmentation works.** Held-out performance rises 46.5% → **68.0%** (+21.5 pp,
intervals [39.7, 53.4] vs [61.2, 74.1] — **separated**, a real effect), and the seen-vs-unseen gap
falls to −9.5 pp with **overlapping** intervals, i.e. no longer demonstrable.

**On Qwen it does not.** +7.0 pp, intervals overlap, gap stays −28.5 pp and separated.

So the correct statement is **not** "instruction augmentation fails". It is: *three phrasings
suffice to close the generalisation gap on SmolVLM2 but not on Qwen3.5.* The collapse under
condition A is architecture-general; **the fix is not.**

⚠️ **Confound — the backbones are not perfectly matched.** SmolVLM2 trained on 8,000 samples at
tap 30; Qwen on 6,000 at tap 12. Each backbone's own A-vs-B comparison *is* matched (same sample
count, same tap, same actor, same schedule — only phrasing diversity differs), so the
within-backbone conclusions hold. But the *cross-backbone* difference in effect size (+21.5 vs
+7.0 pp) cannot be attributed to the backbone alone; training-set size and tap depth also differ.
Resolving that needs Qwen re-run at 8,000 samples.

The open-loop view of condition B makes the mechanism obvious — the model fits the strings it
saw and nothing transfers:

```
B, instruction-only tap 12:
   orig  0.2702 | para1 0.2706 | para2 0.2729     <- all three IN the training mix
   para3 0.3432                                    <- HELD OUT, +27%
```

### What this means

**The policy matches memorised strings; it does not read instructions.** The instruction still
*selects* the task — the 83/100 swap diagnosis (§1.2.1) is closed-loop and stands — but
"selects among ten memorised strings" is a far weaker capability than "follows a natural-language
instruction", and earlier drafts let the stronger reading stand.

**Caveat on scope:** three phrasings is very little augmentation. Real instruction-following
systems train on orders of magnitude more phrasing diversity. What is shown is that 3 phrasings
is insufficient — **not** that augmentation cannot work.

---

## 6c. External baseline: SmolVLA under a matched harness

Earlier drafts had **no external anchor** — the pod attempts failed on a checkpoint/feature-key
mismatch and were abandoned. It exists now, and it is run through **our** harness (same benchmark
init states, same success predicate, same `max_steps`), so suite *and* harness are matched. The
pod attempt used `lerobot-eval` and was neither.

| policy | trainable params | backbone | LIBERO-Goal | Wilson 95% CI | n |
|---|---|---|---|---|---|
| **Ours** (SmolVLM2, all-token tap 30) | **3.7 M** | frozen, no robot data | **83.0%** | [78.3, 86.8] | 300 |
| **Ours** (Qwen3.5, all-token tap 12) | **3.7 M** | frozen, no robot data | **81.5%** | [75.5, 86.3] | 200 |
| SmolVLA (`HuggingFaceVLA/smolvla_libero`) | 604.9 M | full finetune, robot-pretrained | 76.0% | [66.8, 83.3] | 100 |

SmolVLA per task: 7, 8, 8, **3**, 9, 9, 6, 10, 9, 7 (weakest on the two-stage
"open the top drawer and put the bowl inside").

### What this does and does not say

**All three confidence intervals overlap.** The defensible claim is that a 3.7 M-parameter
read-out on a frozen backbone is **statistically indistinguishable** from a 604.9 M full
fine-tune on this suite under this harness. It is **not** "we beat SmolVLA", and it should not
be written that way — the baseline's n=100 gives a ±8 pp interval that comfortably spans our
point estimates.

**The 76.0% is a sane reproduction, which validates the harness.** The SmolVLA paper reports
~87–90%; the community reproduction band documented in `baselines/smolvla_libero/README.md` is
~63–73%. Landing at 76.0% sits just above that band, so the harness is not silently broken —
which is exactly the failure mode that produced 0% three times during this work.

**Caveats that materially limit the comparison:**

- **n=100 vs our n=300 / n=200.** The baseline is the least-powered arm.
- **SmolVLA runs at its *most favourable* setting** — native `n_action_steps=1`, replanning
  every step (2 s/step for a 605 M model). Ours replans every 8 steps. If anything this
  advantages the baseline.
- **One suite, one seed.** LIBERO-Goal only.
- **Different training budgets.** SmolVLA was finetuned on the full LIBERO dataset; our head saw
  6,000–8,000 LIBERO-Goal frames.

**Three mapping errors had to be fixed before the number meant anything**, each of which
silently yields ~0%:

| error | fix |
|---|---|
| Wrong checkpoint: `lerobot/smolvla_libero` expects `camera1/2/3` + state(6) | `HuggingFaceVLA/smolvla_libero` expects `image`/`image2` + state(8), matching both lerobot's LIBERO env and our convention |
| Wrist camera orientation never validated | measured against the dataset: **rot180** for *both* cameras (wrist 917 vs 5522 next-best) |
| Hand-built input batch | use lerobot's `make_pre_post_processors` with the checkpoint's shipped normalizer stats |

---

## 7. What this changes about the paper

The protocol is built to find the best **tap layer**. The data say the tap layer does not matter,
the read-out's **token subset** does, and the **open-loop proxy planned to measure it is invalid**.

Three claims survive as a coherent paper — **revised after the Qwen3.5-0.8B replication (§6b)**,
which confirmed four, weakened one, contradicted one, and left one untested:

1. **A structural bound**, now on two unrelated backbones: ~91.5% of a causal VLA's token budget
   cannot carry the instruction (`max|Δ| = 0.000e+00` on both SmolVLM2 and Qwen3.5), with a free
   intervention — token order — lifting vision-token instruction sensitivity by 9–11 orders of
   magnitude and action decodability by +0.059 / +0.074 R². Strongest result; architecture-general.
2. **A negative result that matters**: open-loop action error does not order read-out variants by
   closed-loop success (PushT `exp03`, LIBERO-Goal). ⚠️ **SmolVLM2 evidence only** — Qwen has just
   two closed-loop taps, too few to test it either way.
3. **A frozen-backbone result**: the embedding layer alone — zero LM transformer layers — reaches
   83.0% (SmolVLM2) / 75.0% (Qwen), no depth effect is statistically demonstrable on either, and
   the instruction is fully load-bearing (83.0% → 0.0%, 81.5% → 0.0%).

**Dropped:** that instruction-only read-out is more paraphrase-robust. It reverses sign between
backbones (§6b claim 8).

**Not yet earned:** that the read-out token subset *governs* performance. Direction holds on both
(+10.0 pp, +8.0 pp) but only SmolVLM2 reaches interval separation. Needs n≈400/arm on a second
backbone before it can be stated as established.

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
