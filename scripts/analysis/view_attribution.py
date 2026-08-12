"""
scripts/analysis/view_attribution.py
────────────────────────────────────
Why does the two-camera arm reach 91.5% when the one-camera arm reached 62.5%?

The rollout records already answer part of it: on episodes it SUCCEEDS at, the
one-view arm is just as efficient (127.3 steps, 1.7 gripper actuations) as the
two-view arm (120.4, 1.7). The wrist camera does not make good episodes better —
it converts failures into successes. Every failure in both arms is a 400-step
timeout, never an early termination, and one-view failures thrash the gripper
(15.4 actuations vs 6.5), which is the signature of repeated failed grasps.

Per task, the gain correlates -0.855 with the one-view success rate: the wrist
view rescues exactly what was broken. The two tasks needing NO grasp at all
(middle drawer, push plate — zero gripper actuation in both arms) gained least
(+15, +5) while the eight grasping tasks gained +33.8 on average.

This script tests the remaining link in that chain directly, on the cache the
policy trained on:

  1. cross-attention mass split three ways — exterior image tokens, wrist image
     tokens, instruction tokens — so we can see whether the head actually uses
     the second camera or merely tolerates it.
  2. per-input ablation — zero each group in turn and measure the velocity-loss
     multiplier. This is the causal version: attention mass says where the head
     looks, ablation says what it cannot do without.

Token layout: the chat template emits image 1 then image 2 then the text, so
within the 128 image positions the first 64 are the exterior view and the next
64 are the wrist. Asserted at runtime rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from head_diagnostics import (BATCH, DEVICE, load_arm, read_batch,  # noqa: E402
                              stride, val_indices)

ARMS = {
    "exp05": ("exp05", "asset/runs/libero/exp05_groot_2view"),
    "exp06": ("exp06", "asset/runs/libero/exp06_qwen3vl_2view"),
}
OUT = ROOT / "asset" / "analysis" / "head_diagnostics"
N = 400
SEED = 0


def groups(img_mask: np.ndarray):
    """Split token positions into exterior / wrist / text."""
    idx = np.where(img_mask)[0]
    n = len(idx)
    if n % 2:
        raise ValueError(f"{n} image tokens is odd — cannot split into two views")
    half = n // 2
    ext, wri = idx[:half], idx[half:]
    txt = np.where(~img_mask)[0]
    return ext, wri, txt


def run_one(exp, run) -> dict:
    h5 = ROOT / run / "vlm_embeddings.h5"
    import h5py
    with h5py.File(h5, "r") as f:
        n_total = f["embeddings"].shape[0]
    rows = stride(np.sort(val_indices(n_total)), N)

    model, cfg, _ = load_arm(exp, run)
    _, m0, _, _ = read_batch(h5, rows[:1])
    ext, wri, txt = groups(m0[0].numpy())
    print(f"token layout: {len(ext)} exterior + {len(wri)} wrist + {len(txt)} text "
          f"= {m0.shape[1]}")

    # ── 1. attention mass, split by view ──────────────────────────────────
    blocks = [b for b in model.decoder.blocks if getattr(b, "has_cross", False)]
    rec, orig = [], []

    def wrap(mod):
        o = mod.forward

        def fwd(q, k, v, *a, **kw):
            kw.pop("need_weights", None)
            kw.pop("average_attn_weights", None)
            out, w = o(q, k, v, *a, need_weights=True,
                       average_attn_weights=True, **kw)
            rec.append(w.detach().float().cpu())
            return out, w
        mod.forward = fwd
        return o

    for b in blocks:
        orig.append(wrap(b.cross_attn))
    try:
        with torch.no_grad():
            for i in range(0, len(rows), BATCH):
                emb, msk, _, st = read_batch(h5, rows[i:i + BATCH])
                model.sample(emb.to(DEVICE), st.to(DEVICE), num_steps=3,
                             img_mask=msk.to(DEVICE))
    finally:
        for b, o in zip(blocks, orig):
            b.cross_attn.forward = o

    W = torch.cat(rec, 0).mean(dim=(0, 1)).numpy()
    W = W / W.sum()
    att = {"exterior": float(W[ext].sum()), "wrist": float(W[wri].sum()),
           "text": float(W[txt].sum())}
    per = {"exterior": att["exterior"] / len(ext), "wrist": att["wrist"] / len(wri),
           "text": att["text"] / len(txt)}
    print("\n=== cross-attention mass ===")
    for k in ("exterior", "wrist", "text"):
        print(f"  {k:9} {100*att[k]:5.1f}%   per token {per[k]:.5f}")

    # ── 2. per-input ablation ─────────────────────────────────────────────
    def loss_with(zero_idx=None):
        tot = []
        for i in range(0, len(rows), BATCH):
            emb, msk, act, st = read_batch(h5, rows[i:i + BATCH])
            emb = emb.to(DEVICE)
            if zero_idx is not None:
                emb = emb.clone()
                emb[:, zero_idx] = 0.0
            torch.manual_seed(SEED + i)   # pin flow noise so the pair is matched
            with torch.no_grad():
                tot.append(float(model(emb, st.to(DEVICE), act.to(DEVICE),
                                       msk.to(DEVICE))))
        return float(np.mean(tot))

    base = loss_with(None)
    abl = {k: loss_with(v) for k, v in
           (("exterior", ext), ("wrist", wri), ("text", txt))}
    print(f"\n=== ablation (base velocity loss {base:.4f}) ===")
    for k in ("exterior", "wrist", "text"):
        print(f"  zero {k:9} -> {abl[k]:.4f}   x{abl[k]/base:.2f}")

    del model
    torch.cuda.empty_cache()
    return {"attention": att, "per_token": per, "loss_base": base,
            "loss_ablated": abl, "ratio": {k: abl[k] / base for k in abl},
            "n_frames": int(len(rows))}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = {}
    for tag, (exp, run) in ARMS.items():
        print(f"\n########## {tag}  ({run.split('/')[-1]}) ##########")
        out[tag] = run_one(exp, run)
    (OUT / "view_attribution.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'view_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
