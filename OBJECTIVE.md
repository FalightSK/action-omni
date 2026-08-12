# Project objective — VLA anatomy

## The question

**Which parts of a vision-language-action model are load-bearing, and which are
cargo — and how do they interact?**

There is existing work on how to choose *data* and how to choose a *backbone*.
The gap this study addresses is the **interaction between components** of current
open-source VLA models: backbone pretraining, camera configuration, action space,
and task structure are chosen jointly in practice but are almost never varied
against each other under a fixed head and a fixed evaluation protocol.

Framed as a table: frozen-VLM sufficiency is one cell, robot pretraining is
another, and the contribution is the table rather than any single cell.

## Two tiers of evidence, never conflated

| Tier | Method | Claims it supports | Cost |
|---|---|---|---|
| **Availability** | linear probes, offline loss, open-loop error on frozen checkpoints | "this information is present and readable" | cheap; works on any released model |
| **Necessity** | change one factor → retrain → closed-loop rollout | "a policy needs this component" | expensive; only on our own stack |

This distinction is load-bearing, not pedantic. Every sentence containing
"needed" must rest on tier 2.

The study hardened the distinction from "the tiers can disagree" to **"tier 1 can
be blind."** On the ALOHA pair, velocity loss differs by 0.1%, open-loop action
error by −0.4% and PE sensitivity by 1.1% — against a closed-loop gap of +9.5
points at p = 0.0067 whose mechanism localises to a single transition at
p = 0.0009. The offline metrics did not rank the policies wrongly; they carried
no signal about the difference at all.

## Design principle

One head, one recipe, one factor at a time.

Every arm shares the 19.2M-parameter head (token adapter + DiT flow decoder), the
read depth, the optimiser schedule, the data, and the rollout protocol. The
backbone is **frozen** in every arm, so what is being compared is what the
backbone makes available, not what a finetune could reshape.

Two controls make the comparison honest:

- **Layer matching.** Both backbones are read at layer 16 — GR00T's own
  `select_layer`, intermediate for Qwen3-VL. Reading each at its own final layer
  would compare depth 16 against depth 28 and attribute a depth effect to
  pretraining.
- **Final-norm correction.** Because layer 16 is final for one arm and
  intermediate for the other, the language stack's final RMSNorm is applied to
  the intermediate read. Without it the pair would differ by normalisation rather
  than by weights.

## Component ledger

| Component | Status | Evidence |
|---|---|---|
| Second camera | **required, and dominant** | +29.0 pts (GR00T) and +21.0 pts (Qwen3-VL) on LIBERO-Goal, p < 10⁻⁷ |
| Robot-pretrained VLM | **task-dependent** | LIBERO-Goal at the benchmark spec: 2.5 pts, p = 0.40. ALOHA bimanual: **+9.5 pts, p = 0.0067** (n = 400/arm), localised to P(handover \| lift) at p = 0.0009, plus ~2× training efficiency |
| Instruction pathway | **load-bearing where instructions vary** | 0/200 under a swapped instruction on every LIBERO arm; text-ablation 5.4–7.4× against a measured floor of 1.14–1.24× |
| Spatial positional encoding | **graded by action space** | zeroing it shifts actions 0.097–0.111 under end-effector control, 0.038 under joint control with full proprioception |
| Offline loss as a selector | **not usable** | anti-predicts on LIBERO; carries no signal at all on ALOHA |

## Testbeds

| Dataset | Role | Why it is in the study |
|---|---|---|
| **LIBERO-Goal** | language anatomy — 10 goals over one fixed scene | isolates the instruction pathway: the scene is constant, so only the goal distinguishes tasks |
| **ALOHA transfer-cube** | motor anatomy — bimanual, 14-DOF, joint-space | the pre-registered falsifier for the LIBERO backbone null. Bimanual manipulation is inside GR00T's pretraining distribution, so this venue is biased *toward* finding a pretraining effect |

**The dataset gate.** Before committing GPU time, check whether the action is
linearly recoverable from a single observation by *any* encoder. A dataset that
fails this gate returns "no effect" for every ablation regardless of what is
ablated. Running the gate costs minutes and can save weeks; one candidate
dataset was rejected by it (action R² ≤ 0.063 across all nine arms tested).

## The confound that bounds every cross-testbed claim

LIBERO-Goal and ALOHA differ on **six axes simultaneously**:

| | LIBERO-Goal | ALOHA |
|---|---|---|
| Instruction variation | 10 | 1 |
| Degrees of freedom | 7 | 14 |
| Arms | single | bimanual |
| Action space | end-effector | joint |
| Cameras | 2 | 1 |
| Inside GR00T's pretraining distribution | arguably | yes |

Two testbeds cannot separate six axes. Any statement naming one axis as *the*
cause of the LIBERO/ALOHA reversal is an interpretation, not a result, and is
written as such throughout `RESULTS.md`.

## Environments

| Env | Purpose | Why separate |
|---|---|---|
| `vla` | training, analysis, probes, gym-aloha rollouts | numpy 2.2 / torch 2.10 / transformers 5.3, Python 3.12 |
| `vla_libero` | LIBERO simulator, closed-loop rollouts | LIBERO pins numpy 1.22 / robosuite 1.4 / mujoco 2.3; installing it into `vla` would downgrade numpy under torch and break the study |

**Two MuJoCo backends, not interchangeable.** LIBERO/robosuite drives raw mujoco
and needs `MUJOCO_GL=wgl`; gym-aloha goes through dm_control, which rejects `wgl`
outright and needs `MUJOCO_GL=glfw`. Using the wrong one fails at env creation.

Simulator setup is scripted and idempotent — `scripts/data/_setup_libero.py`,
verified by `scripts/data/_verify_libero.py`. Four Windows-specific blockers are
handled there and documented in place.

## Standing constraints

- Data and weights live in `asset/`, never in the HuggingFace cache.
- Every pipeline or decision diagram is a Mermaid block.
- Document each step with its reason, not just its effect.
- Long jobs report progress through a log that can be tailed.
- No success-rate claim below n = 50; no between-arm comparison below n = 200.
