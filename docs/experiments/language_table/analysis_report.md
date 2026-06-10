# Language Table — analysis report

**Task:** Google `language_table_sim` (2D block-pushing, **per-episode language instruction**), evaluated live in the PyBullet sim (gym 0.23). Success = library-official `reward > 0` (target block within 0.05 of goal; no dwell). All sim eval at **n = 50/task**, episode cap 200 steps. `BLOCK_8` board = 2 of each colour × 2 of each shape (red/green/blue/yellow × moon/cube/star/pentagon).
**Architecture:** frozen **Qwen3.5-0.8B** (853 M, bf16) + **VLMTokenAdapter** (PerTokenLoRA r16 + SpatialAwareMLP + AttentionReadout, 2.79 M) + **DiT flow-matching decoder** (OT-CFM, 256-hidden 6-layer, ~10 M). 2-D effector state, 2-D ee-delta action, action_horizon 16, replan every 4 steps. Image 320×180 → Qwen emits **66 image tokens** (6×11 grid).
**Platform:** Windows 11 · RTX 4070 Ti (12 GB) · conda envs `vla_lt` (sim/train) + `gemma4` (Gemma 4).

---

## 1. Headline

Two training experiments + a root-cause study of *why* the policy fails.

| Experiment | Data / schedule | Eval | Result (n=50) |
|---|---|---|---|
| **exp01 baseline** | ~5k narrow episodes, 150 epochs (cached embeddings) | in-dist vs held-out new-command | in-dist **10%**, OOD **60%** |
| **exp02 balanced (Plan A)** | balanced 40% across 3 tasks (~72k ep), **1 epoch** on-the-fly | seen vs unseen task-gen | seen **5%**, unseen **43%** |

The model **reads the instruction well but cannot place objects precisely.** Both experiments show the same shape: *easy* tasks (reach / push-apart) succeed at 70–86%, *precise placement* tasks fail at 2–12% — **whether seen or unseen, and regardless of 12× more balanced data.** The root cause (§5) is the **frozen visual front-end**: general patch-token VLMs discard per-object colour/identity at the token resolution, and this is **independent of model scale or architecture** (Qwen 0.8B ≈ SmolVLM2 0.5B ≈ Gemma 4 12B, all far below a trivial raw-pixel baseline).

---

## 2. Experiment 1 — baseline + new-command generalization

Trained on a ~5k-episode subset of the precise-placement families, with the `separate` (push-apart) and `point` (reach) verbs **held out** of training, then evaluated on both. See [lt_generalization.png](lt_generalization.png), [lt_why_ood.png](lt_why_ood.png).

| Task | Kind | SR | mean steps |
|---|---|---|---|
| `block2block` | in-dist | **8%** | 188 |
| `block2absolute` | in-dist | **12%** | 187 |
| `separate` (push apart) | **held-out** | **46%** | 147 |
| `point` (reach) | **held-out** | **74%** | 87 |
| **mean in-dist** | | **10%** | |
| **mean held-out (OOD)** | | **60%** | |

**The "OOD beats in-dist" result is a task-difficulty artifact, not genuine extrapolation superiority.** The held-out verbs (`separate`, `point`) only require coarse motion — push two blocks apart, or move the effector to a block — whereas the in-dist families require landing a specific block within 5 cm of a goal. The model generalizes the *language* of unseen verbs fine; it just happens that the unseen tasks are the easy ones. The real axis is **placement precision**, not seen/unseen.

---

## 3. Experiment 2 — balanced Plan A + seen/unseen task-gen

To test whether the in-dist failures were a **data-imbalance / under-training** problem, we built a class-balanced 40% subset (`block2block` / `block2absolute` / `block2relative` capped equally, ~72k episodes) and trained **1 epoch** with the frozen-VLM forward run **on-the-fly** inside the loop (shard-streamed to fit RAM; ~56–64 fps). Eval on the 3 seen families + 4 unseen (`separate`, `point`, `between`, `corner`). See [lt_balanced_plan.png](lt_balanced_plan.png), [lt_taskgen.png](lt_taskgen.png).

| Task | Kind | Skill | SR | mean steps |
|---|---|---|---|---|
| `block2block` | seen | precise | **2%** | 199 |
| `block2absolute` | seen | precise | **6%** | 194 |
| `block2relative` | seen | precise | **8%** | 191 |
| `separate` | unseen | easy | **70%** | 121 |
| `point` | unseen | easy | **86%** | 71 |
| `between` | unseen | precise | **2%** | 198 |
| `corner` | unseen | medium | **12%** | 182 |
| **mean seen** | | | **5.3%** | |
| **mean unseen** | | | **42.5%** | |

**Honest read:** 12× more balanced data + a full epoch did **not** fix precise placement (still 2–8% on seen *and* on unseen `between`). It **did** improve easy-task generalization vs exp01 (`separate` 46 → 70%, `point` 74 → 86%) — more task diversity helps language→skill transfer. The seen/unseen mean gap again reflects **difficulty** (the unseen set is inflated by the easy reach/push verbs), not memorization. Grouped by *skill*, precise-placement is ~2–8% across the board; only easy tasks succeed.

**Eval validity** ([lt_eval_validity.png](lt_eval_validity.png)): success is the library's official `reward>0`; `reset()` guards reject trivial starts; a **no-op baseline scores 0/25**, so the easy-task successes are earned, not free. Goal-overlay projection was verified pixel-accurate.

---

## 4. Diagnosis — the model reads language, but doesn't ground objects

Probes on the trained policy (representation PCA, cross-attention, causal language ablation):

- **Language is read well.** Frozen-Qwen text embeddings cluster cleanly by **command type** (silhouette ≈ 0.37; [lt_pca.png](lt_pca.png)). Forcing a *wrong* instruction drops behaviour sharply (language causally drives the policy; [lt_language_effect.png](lt_language_effect.png)).
- **The action decoder attends broadly, not to the named object.** Qwen text→image attention and the DiT action→image cross-attention both spread over the scene rather than locking onto the referenced block ([lt_attention_qwen_vs_dit.png](lt_attention_qwen_vs_dit.png), [lt_dit_token_attention.png](lt_dit_token_attention.png), [lt_action_token_attention.png](lt_action_token_attention.png)). The balanced model is no sharper ([lt_attention_balanced.png](lt_attention_balanced.png)).
- **The conditioning is command-agnostic on the balanced model.** The adapter context's command-silhouette collapses (0.17 → ~0.00) vs exp01 ([lt_pca_balanced.png](lt_pca_balanced.png)) — consistent with better easy-task transfer but no per-object precision.

This pointed at the **frozen visual front-end** as the bottleneck, tested directly in §5.

---

## 5. The visual-grounding wall — is the frozen VLM "colour-blind"? (two studies)

A direct test of whether the frozen VLM even *encodes which block is which*. For both studies we feed tight single-block **crops** to each frozen VLM and measure how cleanly the pooled image-token embedding separates by the block's true **colour** / **shape** (silhouette on top-10 PCs), against a **raw mean-RGB control** (the colour is trivially in the pixels: control = **0.18**). See [lt_vlm_colorblind.png](lt_vlm_colorblind.png), [lt_smolvla_actual_embed.png](lt_smolvla_actual_embed.png), [lt_gemma4_colorblind.png](lt_gemma4_colorblind.png).

### 5a. Qwen vs SmolVLA's VLM — does a "robotics" VLM help?

First a key fact from `lerobot/smolvla_base/config.json`: SmolVLA sets **`train_expert_only: true` + `freeze_vision_encoder: true`** and loads `HuggingFaceTB/SmolVLM2-500M` (first 16 layers). We verified the shipped checkpoint's VLM is **byte-identical** to base SmolVLM2 (all probed weights mean|Δ| = 0; crop-embedding |Δ| = 0). **So SmolVLA's VLM carries zero robotics learning** — robotics only trained the action expert.

### 5b. Does a *smarter / bigger* VLM escape it? — Gemma 4 scaling

Gemma 4 (encoder-free multimodal) at E2B / E4B / **12B**, 4-bit.

| Encoder | ~size | **VIS colour** | VIS shape | TXT command | TXT colour |
|---|---|---|---|---|---|
| **raw pixels (mean RGB)** | — | **0.18** | 0.00 | — | — |
| SmolVLM2-500M (= SmolVLA's frozen VLM) | 0.5 B | 0.09 | 0.06 | 0.40 | −0.01 |
| Qwen3.5 (ours) | 0.8 B | 0.06 | 0.08 | 0.38 | −0.01 |
| Gemma 4 E2B | ~2 B | 0.05 | −0.01 | 0.34 | −0.03 |
| Gemma 4 E4B | ~4 B | 0.08 | 0.00 | 0.36 | −0.01 |
| **Gemma 4 12B** (encoder-free) | 12 B | **0.00** | 0.02 | 0.31 | 0.02 |

**Findings:**
1. **Every VLM is colour-blind** relative to the raw-pixel ceiling — colour silhouette 0.00–0.09 vs **0.18**. The colour is plainly in the pixels; the VLMs discard most of it.
2. **Scale and architecture don't fix it.** Across 0.5 B → 12 B and three different visual front-ends (Qwen ViT, SmolVLM SigLIP, Gemma 4 encoder-free), colour separability stays flat-to-*declining* — the 12 B encoder-free model is the **worst** (≈ 0.00). Bigger LMs abstract low-level appearance away *more*.
3. **The split is language vs grounding, not "can't read."** All models separate **command type** strongly (TXT command 0.31–0.40) while encoding the *named colour* at ≈ 0 (TXT colour) — exactly the policy's behaviour.

*Caveat:* this measures the **top-layer pooled** representation that an action head consumes (the relevant signal for a VLA); colour likely lives in earlier vision-proximal layers and is abstracted out by the deep LM (consistent with the 12 B being lowest).

---

## 6. Synthesis

The frozen-Qwen VLA **understands the per-episode instruction** (clean command clustering, causal language effect, broad object attention) but **cannot do precise placement** because its frozen visual front-end does not preserve per-object colour/identity at the 66-token resolution. This is:

- **not a data problem** — 12× more balanced data + a full epoch didn't move precise SR (§3);
- **not a seen/unseen problem** — the real axis is placement *difficulty* (§2–3);
- **not a backbone-choice or scale problem** — Qwen, SmolVLM2, and Gemma 4 up to 12 B are all colour-blind on the same probe (§5).

**The fix has to be in the vision pathway** — higher-resolution / colour-preserving visual tokens, or unfreezing the encoder — not a bigger LM, a different "robotics" backbone, or more data.

---

## 7. Figure index (`docs/experiments/language_table/`)

| File | Content |
|---|---|
| `lt_data_budget.png`, `lt_data_coverage.png`, `lt_dataset_diversity.png`, `lt_language_breakdown.png` | dataset: family distribution, budget, instruction diversity |
| `lt_balanced_plan.png` | Plan A balanced-40% design + per-task caps |
| `lt_generalization.png`, `lt_why_ood.png` | exp01 in-dist vs held-out; difficulty-not-extrapolation explainer |
| `lt_taskgen.png` | exp02 seen/unseen SR by task + by skill |
| `lt_eval_validity.png` | eval-validity proof (no-op baseline, reset guards, projection) |
| `lt_pca.png`, `lt_pca_balanced.png`, `lt_pca_unseen.png` | representation PCA (command/colour/shape silhouettes) |
| `lt_image_pca.png`, `lt_image_scatter.png` | image-token representation probes |
| `lt_attention.png`, `lt_attention_balanced.png`, `lt_attention_qwen_vs_dit.png` | Qwen + DiT attention to the scene |
| `lt_dit_token_attention.png`, `lt_action_token_attention.png` | per-token / per-action-step attention |
| `lt_language_effect.png` | causal language-effect (wrong-instruction ablation) |
| `lt_rollout_filmstrip.png` | annotated rollout filmstrip |
| `lt_decision.png` | dataset-budget decision visual |
| **`lt_vlm_colorblind.png`** | Qwen vs SmolVLM2 (=SmolVLA VLM) vs raw-pixel colour/shape separability |
| **`lt_smolvla_actual_embed.png`** | real SmolVLA VLM embedding + byte-identical-to-SmolVLM2 weight proof |
| **`lt_gemma4_colorblind.png`** | Gemma 4 E2B/E4B/12B scaling curve vs raw-pixel ceiling |

*Result JSONs: `lt_vlm_colorblind.json`, `lt_smolvla_actual_embed.json`, `lt_gemma4_colorblind.json`. Eval JSONs under `asset/runs/language_table/{exp01_baseline,exp02_balanced}/` (gitignored).*
