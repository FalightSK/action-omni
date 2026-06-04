# Project Milestones — Frozen VLM VLA

**Goal:** Prove a frozen, non-robot-pretrained VLM can serve as a generalizable VLA backbone.  
**Updated:** 2026-06-03

---

## Phase 0 — Foundation ✅ COMPLETE

- [x] VLA pipeline implemented (Qwen3.5-0.8B + VLMTokenAdapter + FlowMatchingDecoder)
- [x] DiT decoder with cross-attention to VLM tokens (DiTFlowDecoder)
- [x] OT-CFM flow matching training loop
- [x] Receding-horizon inference controller
- [x] Precompute embeddings pipeline
- [x] Mechanistic analysis tool (component ablations, attention maps, LoRA contribution)
- [x] Statistical evaluation framework (n=50, Wilson CI, chi-squared)

---

## Phase 1 — PushT Validation ✅ COMPLETE

**Gate question:** Does the basic frozen VLM → adapter → decoder pipeline function?  
**Gate threshold:** SR > 40%  
**Result:** 56% SR (Exp2a, n=50) — **PASSED**

| Milestone | Status | Result |
|---|---|---|
| Exp1: MLP baseline | ✅ | 30% SR |
| Exp2: 6D state (covariate shift test) | ✅ | 0% SR — covariate shift confirmed fatal |
| Exp2a: DiT decoder | ✅ | **56% SR** — best architecture |
| Exp2c: inference horizon ablation | ✅ | 0% SR — hard constraint confirmed |
| Exp2d: step limit ablation | ✅ | 42% SR — MPS variance, not time limit |
| Exp3: multi-scale layers 8/16/24 | ✅ | 44% SR — naive fusion invalid |
| Mechanistic analysis (Exp1 + Exp2a) | ✅ | LoRA +533%, cross-attn +261% |
| Mechanistic analysis (Exp3) | ✅ | LoRA +2.8% — fusion displaced LoRA |
| PushT conclusion documented | ✅ | docs/conclusions/pusht.md |
| **Exp5: weighted layer fusion (8/16/24)**  | ✅ | 36% SR — LoRA +478% (restored), but layers 8/16 hurt SR |

**Key findings:**
- Best architecture: DiT + layer-28 + LoRA rank-16 + 2D state (Exp02a, 56% SR)
- LoRA acts as 1024→16→1024 task-projection (not correction)
- DiT cross-attention is the load-bearing mechanism (+261% loss when removed)
- Multi-scale fusion with large linear projection (3.1M params) is architecturally invalid —
  violates hard constraint: fusion capacity must be < LoRA capacity (< 32K params)
- **Exp05 (weighted fusion, layers 8/16/24):** 36% SR — LoRA fully restored (+478%), but early
  layers add noise. Layer 28 features (last_hidden_state) are most useful — intermediate
  layers 8/16 actively hurt. Final recommendation: Exp02a (single layer 24) is the best architecture.

---

## Phase 2 — ALOHA 🔄 ALOHA-A COMPLETE (66% SR, n=50)

**Gate question:** Does the architecture scale to physical complexity (14-DOF bimanual)?  
**Key test:** Does LoRA rank-16 hold, or is rank-64 needed for 14D action space?
**Note:** ViT ablation is NOT run here. PushT has a fixed task instruction — it cannot test language-visual pretraining. ViT ablation is deferred to Phase 3 (Language Table), where instruction varies per episode.

| Milestone | Status | Target |
|---|---|---|
| ALOHA data loader (v3.0 parquet + PyAV) | ✅ | raw shards + sequential mp4 decode |
| Config: action_dim=14, state_dim=14, img_grid 8×10 | ✅ | seq_len 103, absolute joint targets |
| ALOHA-A: baseline Exp2a architecture | ✅ | **66% SR (n=50)**, Wilson [52%, 78%] |
| ALOHA-B: rank-64 ablation | ⬜ | Does rank matter for 14D? |
| ALOHA-C: multi-camera | ⬜ | Overhead + wrist cameras |
| Mechanistic analysis on ALOHA | ⬜ | Does LoRA still dominate? |

**Decision gate → TAKEN (2026-06-04):** Non-trivial SR with rank-16 (**66%, n=50**) → **architecture scales.** Rank-16 held for the 14-D action space — rank-64 is not needed as a fix (becomes a confirmatory ablation). Main failure mode is the right→left handoff, not grasping (90% lift, 73% of lifts transfer). Full writeup: [docs/experiments/aloha/exp01_baseline.md](docs/experiments/aloha/exp01_baseline.md).
- Non-trivial SR with rank-16: Architecture scales. Proceed to Language Table.
- Rank-16 fails, rank-64 works: Rank scaling finding — document relationship.
- Both fail: Fix architecture (scale DiT), narrow OOD scope to 2D tasks.

---

## Phase 3 — Language Table ⏳ PRIORITY 2

**Gate question:** Does the VLM's language understanding transfer to language-conditioned control?  
**ViT ablation runs HERE** — Language Table has per-episode instruction variation, making Qwen vs ViT a meaningful test of language-visual joint pretraining vs vision-only pretraining.

| Milestone | Status | Target |
|---|---|---|
| Language Table data loader | ⬜ | — |
| LT-A: baseline (check text token attention) | ⬜ | > 5% text attention? |
| If text ignored: add dedicated text readout | ⬜ | 1-day architectural fix |
| LT-B: multi-instruction test | ⬜ | SR differs across instructions? |
| Ablation: text conditioning OFF vs ON | ⬜ | Quantify language contribution |
| **ViT ablation: ViT-B/16 + CLIP text vs Qwen** | ⬜ | chi-squared, n=50 |
| Statistical comparison Qwen vs ViT on LT | ⬜ | Does language-visual joint pretraining help? |

**Decision gate:**
- Text tokens > 5% and SR differs by instruction: Language transfers. Strong H3 evidence.
- Text tokens ≈ 0%: Add forced text pathway. Retest. If still no effect: language doesn't transfer naturally.
- Qwen SR >> ViT SR (p < 0.05): Language-visual pretraining claim validated. Proceed.
- Qwen SR ≈ ViT SR: Reframe as efficiency paper. Design OOD for compositional language (where CLIP fails).

---

## Phase 4 — Custom OOD Dataset ⏳ PRIORITY 3 (The Contribution)

**Gate question:** Does the frozen VLM generalize out-of-distribution where baselines fail?

| Milestone | Status | Target |
|---|---|---|
| Dataset design: pick-and-place 4-color train / 6-color test | ⬜ | — |
| OOD Type 1: color shift evaluation | ⬜ | Qwen vs Diffusion Policy |
| OOD Type 2: compositional language | ⬜ | Qwen vs CLIP (key comparison) |
| OOD Type 3: instruction rephrasing | ⬜ | Qwen vs CLIP text |
| Baseline: Diffusion Policy (CNN) | ⬜ | SOTA task-specific |
| Baseline: Frozen ViT + CLIP text | ⬜ | Separate encoding |
| Baseline: Frozen CLIP (vision + text) | ⬜ | Contrastive alignment |
| Optional: VLM backbone LoRA | ⬜ | Minimum finetuning needed |
| Statistical analysis across all baselines | ⬜ | — |
| Paper draft: OOD generalization section | ⬜ | — |

**Decision gates:**
- Qwen > all baselines on OOD Type 2: Strong paper. Full H4 confirmed.
- Qwen > Diffusion Policy, ≈ CLIP on color OOD: Partial. Run OOD Type 2 to differentiate.
- Qwen ≤ all baselines: Negative result paper. Characterize limits of frozen backbone.

---

## Phase 5 — Paper Writeup ⏳

| Milestone | Status | Target |
|---|---|---|
| Related work section complete | ⬜ | — |
| Method section: architecture diagram + equations | ⬜ | — |
| PushT section: experiments + mechanistic analysis | ⬜ | — |
| ALOHA section: scaling results | ⬜ | — |
| Language Table section: language grounding | ⬜ | — |
| OOD section: generalization claim + comparisons | ⬜ | — |
| Ablation tables: LoRA rank, fusion variants, decoder | ⬜ | — |
| Submit to venue | ⬜ | CoRL / ICLR / ICRA |

---

## Publishability Gates

| Milestone Set | Venue | Strength |
|---|---|---|
| PushT + mechanistic analysis | Workshop (CoRL/IROS) | ✅ Achievable now |
| + ALOHA + ViT ablation | ICRA / IROS main | Strong system paper |
| + Language Table + OOD Type 1 | CoRL main | Strong empirical paper |
| + OOD Type 2 (beats CLIP) + full baselines | ICLR / NeurIPS | Top-tier claim |

---

## Hard Constraints (Never Violate Across All Experiments)

1. **State = environment observations only** — never action predictions or history
2. **inference_horizon at inference = inference_horizon at training** — cannot change post-training
3. **n_eval ≥ 50** before drawing SR conclusions (MPS non-determinism)
4. **ViT ablation must run on Language Table (not PushT) before claiming "VLM backbone matters"** — PushT has fixed instruction; only a language-conditioned task can validate the language-visual pretraining claim
5. **CLIP baseline must be included in any language-conditioned OOD experiment**
6. **Multi-scale fusion capacity must be < LoRA capacity** (< 32K params)
