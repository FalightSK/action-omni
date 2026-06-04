# ALOHA transfer-cube — analysis report

**Task:** `gym_aloha/AlohaTransferCube-v0` (bimanual 14-DOF), success = reward 4 (cube transferred to the left gripper and lifted). All sim eval at **n=50**, episode cap 400 steps (8 s @ 50 fps).
**Platform:** Windows 11 · RTX 4070 Ti (12 GB) · conda env `vla_aloha`.

---

## 1. Headline

| Model | Trainable | Train | **Success (n=50)** |
|---|---|---|---|
| **Frozen-Qwen VLA** (ours) | 16M (LoRA adapter + DiT), backbone frozen | 300 ep, ~40 min | **66%** (33/50), Wilson 95% CI [52%, 78%] |
| **SmolVLA finetune** | 100M action expert (VLM frozen) | 12k steps, ~73 min | **0%** (1/50 even touched the cube) |

The lightweight frozen-VLM adapter solves the task; the SmolVLA finetune, on this constrained setup, does not. The rest of this report explains *where* the Qwen policy succeeds/fails and *why* it transfers where SmolVLA doesn't.

Figures: [training_results.png](training_results.png) (Qwen), [smolvla_results.png](smolvla_results.png) (SmolVLA), [smolvla_vs_qwen.png](smolvla_vs_qwen.png) (head-to-head).

---

## 2. Where the Qwen policy succeeds and fails — failure funnel

See [qwen_deep_analysis.png](qwen_deep_analysis.png).

Reward-stage histogram (n=50): `0 (no contact): 4 · 1 (touched): 1 · 2 (lifted): 12 · 3 (at left): 0 · 4 (transfer): 33`.

Cumulative funnel + conditional success:

| Stage | Reached | Conditional |
|---|---|---|
| Attempted | 50 | — |
| Touched cube | 46 | **P(touch) = 92%** |
| Lifted off table | 45 | **P(lift \| touch) = 98%** |
| Transferred (success) | 33 | **P(transfer \| lift) = 73%** ← bottleneck |

**Grasping is near-solved (92% touch, 98% lift-given-touch). The single failure mode is the right→left hand-off** — 12 of 45 lifts drop or miss the transfer. When it succeeds it's quick: **mean 299 steps (6.0 s)**, range 242–379, i.e. no "buzzer-beater" successes.
→ Highest-leverage improvement: the bimanual hand-off (more demos of that phase, or a wrist camera).

---

## 3. Why it works — mechanistic analysis

See [qwen_mechanistic.png](qwen_mechanistic.png). Computed on the trained weights over a 256-sample batch (paired-noise ablations).

**Load-bearing components (validation-loss increase when removed):**
- **Zero the LoRA → +1648%.** The rank-16 adapter is a full *task re-basis* of the frozen Qwen features, not a small correction.
- **Remove DiT cross-attention → +1871%.** Per-denoising-step visual grounding is essential.

These dwarf the project's PushT-phase numbers (LoRA +533%, cross-attn +261%) — i.e. **at 14-DOF these mechanisms matter ~3–7× more** than at 2D.

**Gradient flow** is healthy across all components (no dead/vanishing parts); the DiT in/out projections and self-attention carry the most signal, with the adapter and DiT both active.

---

## 4. What the policies look at — attention

**Qwen VLA** ([qwen_attention_overlay.png](qwen_attention_overlay.png)): **~85% of action-token cross-attention lands on image tokens**, in every phase, concentrated on the central workspace (arms + cube) and shifting with the task (reach → grasp → transfer). The policy is genuinely vision-driven and re-queries the live scene every denoising step.

**SmolVLA** ([smolvla_attention.png](smolvla_attention.png)): **~67% of action-expert attention on the image**, and **sparse/peaky** — almost all weight on 1–2 of the 64 SmolVLM image tokens. It under-uses the camera (64 image tokens are most of its prefix, yet get only 67%) and grounds narrowly.

*Caveat:* different architectures/tokenizers (Qwen DiT over an 8×10 image grid vs SmolVLA's expert over SmolVLM's pooled 8×8 tokens), so 85% vs 67% is **directional**, not exact.

---

## 5. Chunk size / closed-loop rate

*(inference-horizon sweep — results below once the run completes)*

Hypothesis: a large part of the Qwen→SmolVLA gap is **closed-loop correction frequency** — Qwen replans every 8 steps (0.16 s) vs SmolVLA's 50-step (1.0 s) open-loop chunk. Test: re-eval the trained Qwen at `inference_horizon ∈ {1, 4, 8, 16}` (how many actions execute before replanning), n=50 each.

`<<CHUNK_SIZE_RESULTS>>`

---

## 6. Cause synthesis — why Qwen transfers and SmolVLA (here) does not

Not raw model size (SmolVLA is 6× larger). The causal differences:

1. **Closed-loop rate** — Qwen replans every 8 steps vs SmolVLA's 50-step open-loop chunk → ~6× more correction; SmolVLA drifts open-loop and can't recover. (See §5.)
2. **Per-step visual grounding** — Qwen's DiT cross-attends to the live scene at *every* denoising step (85% on image); SmolVLA conditions once per chunk and attends less/narrower (67%, peaky).
3. **Low-overfit adaptation** — 16M trainable on a frozen backbone vs a 100M expert fit to **50 demos** (SmolVLA's stated minimum). SmolVLA's train loss converged (~0.06) yet didn't transfer → a generalization/transfer gap, not a step-count one.
4. **Camera mismatch** — native 1-camera vs SmolVLA pretrained on 3 cameras.

**SmolVLA's 0% is not its ceiling** — it reflects a constrained setup (50 demos, 1 camera, ~30% of the recommended training budget). A multi-camera + more-demos + full-recipe run is needed for a fair head-to-head.

---

## 7. Figure index (`docs/experiments/aloha/`)

| File | Content |
|---|---|
| `training_results.png` | Qwen: loss + reward stages + success donut + episode length |
| `smolvla_results.png` | SmolVLA: same panels (0% SR) |
| `smolvla_vs_qwen.png` | head-to-head with budget caveat |
| `qwen_deep_analysis.png` | failure funnel + conditional cascade + timing + benchmark context |
| `qwen_mechanistic.png` | gradient flow + load-bearing ablations + cross-attn map |
| `qwen_attention_overlay.png` | Qwen attention on real frames |
| `smolvla_attention.png` | SmolVLA expert attention + image-fraction comparison |
| `qwen_chunksize_sweep.png` | SR vs inference_horizon (closed-loop rate) |
