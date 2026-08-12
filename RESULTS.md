# Results — component interaction in a frozen-backbone VLA

Six arms. One head. Two testbeds. Every arm shares the 19.2M-parameter head
(token adapter + DiT flow-matching decoder), the layer-16 read, the optimiser
schedule and the rollout protocol, so a difference in success rate is
attributable to the factor that moved.

---

## 1. Design

### 1.1 The pair

Stock **Qwen3-VL-2B** → Cosmos-Reason2-2B → **GR00T N1.7**. Both hops are
verified real finetunes (584/625 and 476/493 tensors differ), so the pair spans
the whole robot-pretraining treatment rather than one weak hop. Neither model
performs backbone-level language fusion, so neither arm has fusion that our head
could render redundant — removing an asymmetry that would otherwise confound the
comparison.

### 1.2 Two controls that make the comparison honest

**Layer matching.** Both arms are read at layer 16 — GR00T's own `select_layer`,
intermediate for Qwen3-VL's 28 layers. Reading each at its own last layer would
compare depth 16 against depth 28 and attribute a depth effect to pretraining.

**Final-norm correction.** Because layer 16 is *final* for one arm and
*intermediate* for the other, HF would apply the language stack's final RMSNorm
to one read and not the other. `models/vla.py::encode_vlm` applies it to the
intermediate read. Without this the pair differs by normalisation rather than by
weights — the defect that once made a provably identical pair read cosine 0.22
instead of 1.0.

### 1.3 The factors moved

| Factor | Levels | Testbed |
|---|---|---|
| Backbone pretraining | GR00T N1.7 vs stock Qwen3-VL-2B | both |
| Camera configuration | 1 view vs 2 views | LIBERO-Goal |
| Task structure / embodiment | single-arm 7-DOF vs bimanual 14-DOF | across testbeds |

---

## 2. LIBERO-Goal — cameras dominate backbones

10 goals over one fixed scene; 200 rollouts per arm (10 tasks × 20 episodes),
50 fixed initial states.

| Arm | Backbone | Views | Val loss | Canonical SR | Swapped SR |
|---|---|---|---|---|---|
| exp03 | GR00T N1.7 | 1 | 0.0316 | 62.5% | 0.0% |
| exp04 | Qwen3-VL-2B | 1 | 0.0332 | 68.0% | 0.0% |
| exp05 | GR00T N1.7 | **2** | 0.0352 | **91.5%** | 0.0% |
| exp06 | Qwen3-VL-2B | **2** | 0.0352 | **89.0%** | 0.0% |

**Camera effect: +29.0 pts (GR00T), +21.0 pts (Qwen3-VL), p < 10⁻⁷.**
**Backbone effect at the benchmark's own 2-view spec: 2.5 pts, p = 0.40.**

An order of magnitude separates the observation configuration from the backbone.
Single-view results are reported for completeness but are not the basis of any
claim: single-view puts the robot-pretrained arm outside its trained input
configuration while leaving the stock arm inside its own, an asymmetry pointing
the same direction as the result.

**The instruction is load-bearing for every arm.** 0/200 under a swapped
instruction, all ten tasks. Language is not decoration on this testbed.

**Both 2-view arms learn near-identical routing** — wrist-camera attention 51.6%
vs 54.5%, ablation cost ×11.94 vs ×11.64. Given different backbone weights, the
head converges to the same solution.

---

## 3. ALOHA transfer-cube — the null is bounded

The pre-registered falsifier for the LIBERO backbone null. Bimanual manipulation
is inside GR00T's pretraining distribution, so this venue is biased **toward**
finding a pretraining effect; a null here would have been strong evidence.

It did not return a null.

### 3.1 Headline, n = 400 per arm

Two independent 200-episode runs on disjoint seed ranges. `AlohaEnv.reset`
derives the cube pose from the episode seed alone, so `--seed-offset` is
mandatory for a genuine replication — without it a rerun replays the same 200
scenes and resamples only the policy's flow noise.

| Arm | Run 1 (offset 0) | Run 2 (offset 1000) | Pooled | Wilson 95% CI |
|---|---|---|---|---|
| exp05 GR00T N1.7 | 60.0% | 62.5% | **61.25%** (245/400) | [56.4, 65.9] |
| exp06 Qwen3-VL-2B | 49.0% | 54.5% | 51.75% (207/400) | [46.9, 56.6] |

**+9.5 pts, z = 2.71, p = 0.0067.** Clears the Bonferroni bar for the six
comparisons in this study (0.05/6 = 0.0083). Intervals do not overlap. Both arms
moved in the same direction on replication (+2.5 and +5.5), so the second seed
batch was slightly easier; nothing suggests the first run was a fluke.

### 3.2 The gap is one transition

ALOHA scores {1,2,3,4} = touch / lift / handover / success. `max_reward == 3`
never occurs in 800 episodes — GR00T `{0:43, 1:15, 2:97, 4:245}`, Qwen3-VL
`{0:30, 1:23, 2:140, 4:207}` — so once the receiving gripper contacts the cube
the episode always completes, and the ladder is touch → lift → handover.

| Stage | GR00T | Qwen3-VL | Δ | p |
|---|---|---|---|---|
| P(touch) | 89.2% | 92.5% | −3.3 | 0.11 |
| P(lift \| touch) | 95.8% | 93.8% | +2.0 | 0.22 |
| **P(handover \| lift)** | **71.6%** | **59.7%** | **+12.0** | **0.0009** |

Both early stages run slightly *against* the pretrained arm. This is not a
uniform competence advantage — it is localised to the one stage bimanual
pretraining plausibly covers, and the localisation is tighter than the top-line
result.

### 3.3 Checkpoint ladder — speed or skill?

The n = 400 numbers describe the ceiling only. The claim usually made for robot
pretraining is *sample efficiency*, which needs the whole curve. Six snapshots ×
2 arms × 50 episodes, **paired**: every eval sees the same 50 cube poses
(`--seed-offset 2000`), so scene difficulty cancels between conditions.

| Epoch | GR00T | Qwen3-VL | Gap | McNemar p |
|---|---|---|---|---|
| 25 | 8.0% | **0.0%** | +8.0 | 0.125 |
| 50 | 28.0% | **0.0%** | +28.0 | **0.0001** |
| 100 | 46.0% | 22.0% | +24.0 | 0.017 |
| 150 | 54.0% | 28.0% | +26.0 | 0.035 |
| 200 | 72.0% | 48.0% | +24.0 | 0.036 |
| 300 | 66.0% | 46.0% | +20.0 | 0.053 |

- **The stock backbone cannot do the task early at all** — 0/50 at epochs 25 and
  50 while the pretrained arm is at 28%. The strongest point in the ladder.
- **Stock needs ~2× the training** to reach any given rate: 2.4× / 2.8× / 2.2× /
  1.9× at the 20/30/40/46% targets.
- **The gap does not close by epoch 300.** "Pretraining buys speed, not skill" is
  therefore *not* supported; it buys both, with speed the larger effect.

**Anchor check — why the ladder's magnitude is not quotable.** The 50 paired
scenes are a fixed and evidently slightly unrepresentative sample: epoch 300
reads GR00T 66.0% against the 61.25% anchor (+4.8) and Qwen3-VL 46.0% against
51.75% (−5.8). Each deviation sits inside one SE (~6.7 pts), so nothing is
broken — but the ladder establishes the **shape** and the pooled n = 400 runs
establish the **magnitude (+9.5 pts)**. The +24 pt ladder gap is not the
population gap.

**Flagged, not claimed.** Both arms peak at epoch 200 and drop at 300, while
best-validation-loss selected epochs 285 and 280. If that survives a higher-n
test it is a further instance of offline selection failing. At n = 50, 72% vs
66% is within noise and no paired test was run.

---

## 4. Head diagnostics — what the trained head reads

Inference-only forward passes on 2,000 held-out frames from the existing caches
(`scripts/analysis/head_diagnostics.py`). No backbone is loaded; nothing
retrains. The validation split is reproduced exactly, so the comparison is
paired frame-for-frame across arms.

### 4.1 On ALOHA, offline metrics are blind

Between-arm difference, stock minus pretrained, as % of pretrained:

| Measure | Δ |
|---|---|
| velocity loss (overall) | +0.1% |
| velocity loss (mid phase, contains handover) | +1.9% |
| open-loop action error (nMAE, 14 dims) | −0.4% |
| PE sensitivity | +1.1% |
| attention mass on image | **−5.2%** |
| **closed-loop success rate** | **−15.5%** |

Four accuracy measures are flat against a gap the rollouts resolve at p = 0.0067
with a mechanism at p = 0.0009. This is stronger than "offline ranks them
wrongly" — offline carries **no signal** about the difference.

Loss by episode phase shows the objective *does* know the handover is hard —
0.0295/0.0301 in mid phase against 0.0161/0.0162 late — it simply does not know
that one arm is worse at it.

The one offline quantity that moves is attention **allocation**: the stock arm
spends 5.2% less mass on image tokens and correspondingly more on text tokens
that carry zero information on this task. Correlational, one testbed; the only
surviving mechanistic candidate, not a finding.

### 4.2 Text ablation has a floor, and it must be measured

ALOHA has a single fixed instruction, so zeroing its text tokens removes no task
information. Loss nonetheless rises **1.14×** (GR00T) and **1.24×** (Qwen3-VL).
That residual is the cost of an off-distribution perturbation alone — the
metric's floor, and a control condition an ablation on a multi-instruction
dataset cannot supply for itself.

| Arm | Testbed | Ratio |
|---|---|---|
| GR00T | ALOHA (1 instruction) | 1.14× — *floor* |
| Qwen3-VL | ALOHA (1 instruction) | 1.24× — *floor* |
| Qwen3-VL 2-view | LIBERO-Goal (10 instructions) | 5.39× |
| GR00T 2-view | LIBERO-Goal | 6.12× |
| Qwen3-VL 1-view | LIBERO-Goal | 7.05× |
| GR00T 1-view | LIBERO-Goal | 7.41× |

Reporting an ablation ratio without its floor overstates the effect. Note also
that the 2-view arms depend on text *less* than the 1-view arms — with a wrist
camera available, some of what the instruction supplied is recoverable from
vision.

### 4.3 Action space determines how much the policy uses vision

Zeroing the 2D positional encoding and measuring the resulting action shift:

| Testbed | Control mode | Relative shift |
|---|---|---|
| LIBERO-Goal (4 arms) | end-effector | 0.097 – 0.111 |
| ALOHA (2 arms) | joint, full 14-DOF proprioception | **0.038** |

2.5–3× less. Where the state already determines the arm configuration, the head
sources far less from image position. This was predicted the *wrong* way round
before measurement — the pre-registered expectation was that bimanual handover
would be more position-sensitive, on the reasoning that handover is spatial
registration. It is not; the action space dominates.

### 4.4 Attention is near-uniform

Normalised cross-attention entropy is 0.90–0.97 across all six DiT blocks.
Neither head sharply selects tokens; both broadly average over the full token
set, with the stock arm consistently more diffuse. Worth knowing before building
an interpretability story on "which token the policy attends to."

---

## 5. Is the harness correct?

Every offline number comes from the precomputed cache while rollouts re-encode
live. A mismatch between those paths — image convention, state layout, action
scaling, control mode, horizon — would degrade **only** closed-loop performance.
So the policy was removed from the loop and the demonstrations' own recorded
actions were replayed through the same env construction, success detector and
step budget (`scripts/validate_harness.py`, 5 demos × 10 tasks × 2 conditions).

| Replay condition | SR |
|---|---|
| no settling steps | **90.0%** (45/50) |
| 5 settling steps, as the policy rollout does | **92.0%** (46/50) |

**The harness is correct.** Its replay rate coincides with the ~90% band
published methods report, so this is not a stricter harness. The residual ~10% is
expected: open-loop replay of recorded actions through an OSC controller is
sensitive to MuJoCo version and platform drift since recording. The tasks that
replay worst are also where every policy does worst, so part of the per-task
spread is intrinsic difficulty.

This is a **replay rate, not a policy ceiling** — the 2-view arms exceed it on
two tasks (100% vs 80%, 95% vs 60%).

ALOHA cannot be validated this way: `AlohaEnv.reset` samples the cube pose from a
seed into a module-global, and the dataset records neither the pose nor the seed,
so oracle replay is not constructible. `scripts/validate_harness_aloha.py`
validates action plumbing only.

### What this study removes relative to published recipes

Audited against LeRobot's flow-matching VLA configurations and
`groot_n17_3b/config.json`.

**Deliberate — these *are* the experiment:**

- frozen backbone (published recipes tune the LLM, vision encoder, projector and
  diffusion head)
- action expert discarded; a 19.2M head replaces GR00T's ~403M diffusion head
  (32 layers × dim 1024)
- no image augmentation — a structural consequence of precompute, since the cache
  fixes the encoding (GR00T uses colour jitter + 0.95 random crop)

**Unforced, and worth fixing:**

1. **No action/state normalisation.** Published recipes use MEAN_STD or
   QUANTILES; we train on raw actions while flow matching integrates from N(0,1).
2. **3 denoising steps** against LeRobot's 10 and GR00T's 4. Inference-only, free.
3. **Uniform timestep sampling**; GR00T uses a beta schedule.
4. **No EMA**; resolution 224² against GR00T's 256².

None of these threaten internal validity — every arm shares them identically, so
they lower the whole curve rather than favouring a backbone.

### Training budget is not the constraint

| | This study (per arm) | A published LeRobot-family recipe on LIBERO |
|---|---|---|
| optimizer steps | **53,760** | 30,000 |
| batch | 128 | 32 |
| samples seen | **6,881,280** | 960,000 |
| passes over the training set | **120** | ~3.5 |
| trainable params | 19.2M | ~3B |

7.2× more samples and roughly 34× more passes over our data. Validation flattens
after epoch ~75 and open-loop action correlation reaches 0.979–0.997, so the runs
are optimisation-saturated. "Undertrained" is not an available explanation; the
binding constraints are informational and architectural.

---

## 6. What this study cannot settle

- **Two backbones, one lineage.** Every backbone claim rests on one pair.
  Whether the result generalises to other pretraining corpora is untested.
- **The six-axis confound.** LIBERO-Goal and ALOHA differ simultaneously in
  instruction variation (10 vs 1), DOF (7 vs 14), arm count, action space
  (end-effector vs joint), camera count (2 vs 1), and distribution match to
  GR00T's pretraining data. Two testbeds cannot separate six axes. The DOF
  hypothesis, the bimanual hypothesis and the distribution-match hypothesis all
  predict exactly what was observed, and this study cannot distinguish them.
- **One head architecture.** "Backbone barely matters on LIBERO" may be specific
  to a head with this capacity and this cross-attention design.
- **Head diagnostics are correlational.** They locate where two heads differ on
  frozen checkpoints; they cannot prove that difference causes the handover gap.
  A causal test would swap handover-window tokens between arms.
- **Simulation only.** No physical robot.
- **Nulls are not equivalence.** At n = 200 per LIBERO condition the SE is
  3.3 pts, so a null means "no detectable difference at this readout capacity,"
  never "no difference."
- **`d13` is a lead, not evidence.** The one per-dimension outlier (+9.2%,
  plausibly the receiving gripper) is 1 of 14 dims with no multiple-comparison
  control and unverified column ordering.

---

## 7. Defects found and fixed during the study

Recorded because each would have produced a plausible-looking wrong number.

| Defect | Consequence if unfixed |
|---|---|
| Final RMSNorm not applied to the intermediate read | The pair would differ by normalisation, not weights |
| Token padding mismatch — cache held 78 real + 10 zero tokens, live eval at batch 1 produced 78 | Train/eval input shapes diverge silently |
| 2D positional encoding clamped at `n_img-1`, collapsing view 2 onto view 1's last row | Two-camera arms would encode the second view as garbage |
| `per_dim_error` hard-coded `.reshape(-1, 7)` | On 14-DOF ALOHA this does not raise; it interleaves two joints into every reported column |
| Episode seeding `ep_idx * 137 + 42` with no offset | A "replication" would replay the same 200 scenes and resample only flow noise |
| Cache written in task order, sampled by prefix slice | Diagnostics would cover only the first tasks |
| Per-arm colour scale on a figure captioned as comparable | Visual comparison invalid |
