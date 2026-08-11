# pipelines/ — concept (rework in progress)

The previous `pipelines/pusht/run_exp.sh` + `run_pipeline.sh` (removed) hardcoded a
fixed sequence of experiments — precompute → train → eval → mechanistic — with no
validity checks in between. A bad dataset or a broken data flow would silently
propagate all the way to a success-rate number before anyone noticed. This file
replaces them with the *shape* the next pipeline should have, before any script
is written.

## Why rework

Two prior runs reached a final SR number that meant nothing:
- Language Table 5-instr subset: 0% SR — caused by a train/eval phrasing mismatch,
  not the model. Discovered only after training + full eval.
- Nothing upstream of `evaluate.py` would have caught it earlier.

The fix is ordering: validity gates run *before* the expensive stages they'd
otherwise waste, and each gate has to pass before the next stage is allowed to run.

## Concept: gated pipeline, one experiment = one directed run through 5 gates

```
Gate 1 — Dataset validity        (cheap, no GPU)
    ↓ pass
Gate 2 — Data flow / architecture validity   (one batch, no training)
    ↓ pass
Gate 3 — Training / gradient validity        (short run, e.g. 1 epoch + overfit-1-batch)
    ↓ pass
Gate 4 — Result vs. benchmark                 (full train + eval, the expensive stage)
    ↓ pass
Gate 5 — Component / bottleneck analysis      (ablations, only run on a Gate-4 pass)
```

Each gate is a separate script with a pass/fail verdict written to
`asset/runs/<dataset>/<exp>/gates/gateN_*.json` (or `.png` for plots), not just a
printed line — so a launcher can check the verdict file and refuse to advance.

| Gate | Question | Cost | Blocks on |
|---|---|---|---|
| 1. Dataset | Is the data clean, diverse, and balanced enough to learn from? | seconds–minutes, CPU only | NaNs, degenerate action std, class imbalance |
| 2. Data flow / architecture | Does raw → embed → adapter → decoder → action run with correct shapes/masks/normalization, frozen VLM confirmed frozen? | one batch, no training | shape mismatch, unfrozen VLM, silent NaN in forward pass |
| 3. Training / gradient | Does loss go down and gradients flow where they should? | short (~1 epoch, overfit-1-batch sanity) | flat/NaN loss, dead gradients, exploding LR |
| 4. Benchmark | Is the trained result competitive (> baseline / > random, ideally >50% SR) with a real CI? | full train + eval — the expensive stage | below threshold |
| 5. Analysis | Which component / which slice of data is the bottleneck? | ablations on the Gate-4 checkpoint | only runs if Gate 4 passed |

Gates 1–2 already exist as standalone scripts for ALOHA insertion and curated
Language Table (`scripts/validation/_val_gate1_aloha.py`,
`scripts/validation/_val_gate1_lt.py`) — both pass. Gate 2 onward has not been
built yet; this file is the plan for what comes next, not an implementation.

## What a rebuilt `pipelines/<dataset>/run_exp.sh` should do differently

- Run Gate 1 first; abort if it fails — never spend GPU time on bad data.
- Run Gate 2 next on a single batch; abort before starting a real training job.
- Only start full training (Gate 3→4) after 1–2 pass.
- Only run Gate 5 ablations after a Gate-4 pass — no more analyzing failed runs.
- Write one `gates/summary.json` per experiment so `--dataset X --exp Y` status is
  a single file read, not a re-derivation from logs.

## Scope note

This is dataset-agnostic — the same 5 gates apply to PushT, ALOHA, and Language
Table. The old `pipelines/pusht/` scripts were PushT-only launchers with the
gate logic absent; the replacement should live at `pipelines/<dataset>/run_exp.sh`
but share one gate-runner module rather than duplicating the sequence per dataset.
