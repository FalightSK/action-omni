# Project objective — VLA anatomy

## The question

**Which parts of a vision-language-action model are load-bearing, and which are
cargo?**

Not "is a frozen VLM good enough" and not "does robot pretraining help" — both
are narrower, and the first is already occupied territory (SmolVLA, EF-VLA). The
anatomy question subsumes them: *frozen VLM sufficiency* is one cell in the
table, *robot pretraining* is another, and the contribution is the table.

## Why the direction changed

The study started as a controlled comparison of robot-pretrained against stock
VLMs. Its two strongest results turned out not to be about pretraining at all:

- **SmolVLA's vision-language backbone is bit-identical to stock SmolVLM2** — all
  345 tensors, zero differing. A published, working VLA never finetuned its VLM.
  That is a fact about *which components are needed*, not about pretraining.
- **The study's cleanest "pretraining signature" was a depth artifact.** Two of
  three robot arms truncate their language stack as well as finetuning it, so
  comparing each arm at its own last layer confounded the two. With
  depth-matched controls, 0–3% of the effect survived.

Both were filed under the wrong question. Reframed as anatomy they are direct
answers: one component is optional, and one apparent component does not exist.

## Two tiers of evidence, never conflated

| Tier | Method | Claims it supports | Cost |
|---|---|---|---|
| **Availability** | linear probes on frozen checkpoints | "this information is present and linearly readable" | cheap, works on anyone's released model |
| **Necessity** | ablate → retrain → closed-loop rollout | "a policy needs this component" | expensive, only on our own stack |

This distinction is load-bearing, not pedantic. A closed-loop ALOHA ablation
earlier in this project found image tokens carried the control signal while
**offline loss had overstated the text pathway** — availability and necessity
disagreed on the same component. Any sentence containing "needed" must rest on
tier 2.

## Chapters

| | Scope | Status |
|---|---|---|
| **Ch.1 — Audit of shipped VLAs** | Weight-level dissection of Pi-0.5, SmolVLA, GR00T N1.7 against their own base checkpoints; what their representations make available | **done** — `asset/analysis/latent_compare/` |
| **Ch.2 — Component necessity** | Ablate → retrain → roll out. First axis: language-stack **depth** × the VLM→action **adapter** (AdaLN vs linear projection vs cross-attention) | next |
| **Ch.3 — Data requirements** | What a dataset must supply for the task to be learnable at all; the Language Table negative and the gate that detects it cheaply | partly done |

## Component ledger

What Ch.1 already settles, and what Ch.2 must test:

| Component | Status | Evidence |
|---|---|---|
| Robot-pretrained VLM | **not required** | SmolVLA ships stock and frozen; verified tensor-identical |
| Upper half of the language stack | **likely not required** | action R² peaks at ~50% depth for 9/9 arms on ALOHA, 8/9 on LIBERO-Goal; two teams independently cut to 16 layers |
| Image/text subspace separation | **not a component** | a depth effect, not a pretraining product (0–3% survives depth matching) |
| VLM restructuring | **optional, graded** | Pi-0.5 heavy (median Δ 0.183), GR00T light (0.003), SmolVLA zero |
| Read depth | **untested for necessity** | Ch.2, axis 1 |
| VLM→action adapter | **untested** | Ch.2, axis 2 |
| Instruction pathway | **untested** | needs LIBERO-Goal (fixed scene, varying goal) |

## Testbeds

| Dataset | Role | Action R² |
|---|---|---|
| ALOHA transfer / insertion | motor anatomy | 0.67 – 0.79 |
| LIBERO-Goal | language anatomy — 10 goals, one fixed scene | 0.26 – 0.40 |
| Language Table (curated) | a documented negative, not a testbed | 0.006 – 0.063 |

**The dataset gate.** Before committing GPU time, check whether the action is
linearly recoverable from a single observation by *any* encoder. Language Table
fails it across all nine arms — a single frame plus instruction does not
determine the next action there, so any ablation on it returns "no effect"
regardless of what is ablated. LIBERO-Goal was admitted by this gate, not
assumed. Running it costs minutes and can save weeks.

## Environments

| Env | Purpose | Why separate |
|---|---|---|
| `vla` | training, analysis, probes | numpy 2.2 / torch 2.10 / transformers 5.3, Python 3.12 |
| `vla_libero` | LIBERO simulator, closed-loop rollouts | LIBERO pins numpy 1.22 / robosuite 1.4 / mujoco 2.3; installing it into `vla` would downgrade numpy under torch and break the study |
| `vla_lt_data` | TFDS / Language Table conversion | same reason |

Setup for the simulator is scripted and idempotent —
`scripts/data/_setup_libero.py`, verified by `scripts/data/_verify_libero.py`.
Four Windows-specific blockers are handled there and documented in place.

## Standing constraints

- Data and weights live in `asset/`, never in the HuggingFace cache.
- Every pipeline or decision diagram is a Mermaid block.
- Document each step with its reason, not just its effect.
- Long jobs report progress through a log the user can `tail -f`.
