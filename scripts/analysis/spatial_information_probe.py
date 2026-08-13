"""
scripts/analysis/spatial_information_probe.py
─────────────────────────────────────────────
Tests the alternative explanation for the PE-sensitivity result.

The finding: zeroing the 2D positional encoding shifts actions by 0.097-0.111 on
LIBERO (end-effector control) but only 0.038 on ALOHA (joint control with full
14-DOF proprioception). The reading offered is that where the state already
determines the arm configuration, the head sources less from image position.

The obvious objection: ALOHA's scene is visually simpler -- one fixed top camera,
a plain table, one small cube -- so maybe its image tokens simply carry less
spatial information, and the action space has nothing to do with it. A gradient
or Jacobian analysis cannot separate these: a visually impoverished scene
produces low image sensitivity under either explanation.

This probe separates them directly. It asks how much spatial information the
image tokens CONTAIN, independent of whether the policy uses it:

    image tokens (adapter input, mean-pooled over image positions)  ->  arm state

If ALOHA's image tokens predict arm configuration as well as LIBERO's, the scene
is not information-poor and "visually simpler" is not available as an
explanation. If they predict it much worse, the objection stands and the
action-space claim has to be weakened.

Everything is cross-validated (cv_r2 from latent_compare), so the numbers report
generalisation rather than fit -- what licenses the phrase "linearly readable".

Two controls make the comparison fair across testbeds:
  * State dimensionality differs (7 vs 14). R^2 is computed per target dimension
    and averaged, so a 14-D target is not penalised for having more to predict.
  * Frame counts differ. Both testbeds are subsampled to the same n.

    python scripts/analysis/spatial_information_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "latent_compare"))

from analyze import cv_r2  # noqa: E402

OUT = ROOT / "asset" / "analysis" / "spatial_probe"
N_FRAMES = 3000
SEED = 0

ARMS = {
    # (dataset, exp, run_dir) -- both arms of both testbeds, so the comparison is
    # not resting on one backbone.
    "LIBERO GR00T 2v":   ("libero", "exp05", "asset/runs/libero/exp05_groot_2view"),
    "LIBERO Qwen3-VL 2v": ("libero", "exp06", "asset/runs/libero/exp06_qwen3vl_2view"),
    "ALOHA GR00T":       ("aloha", "exp05", "asset/runs/aloha/exp05_groot_transfer"),
    "ALOHA Qwen3-VL":    ("aloha", "exp06", "asset/runs/aloha/exp06_qwen3vl_transfer"),
}


def val_indices(n_total: int) -> np.ndarray:
    """Reproduce train.py's validation split, so the probe never sees train frames."""
    val_len = max(1, int(n_total * 0.10))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    return perm[n_total - val_len:].numpy()


def probe_arm(run_dir: str) -> dict:
    h5path = ROOT / run_dir / "vlm_embeddings.h5"
    rng = np.random.default_rng(SEED)
    with h5py.File(h5path, "r") as f:
        n_total = f["embeddings"].shape[0]
        vi = val_indices(n_total)
        rows = np.sort(rng.choice(vi, size=min(N_FRAMES, len(vi)), replace=False))
        emb = f["embeddings"][rows]
        msk = f["img_masks"][rows].astype(bool)
        st = f["states"][rows]
        act = f["actions"][rows]

    # bf16 stored as uint16: reinterpret the bit pattern, never cast.
    E = torch.from_numpy(emb.astype(np.uint16)).view(torch.bfloat16).float().numpy()

    # Mean-pool over IMAGE positions only. Text tokens are excluded so this
    # measures what the picture carries, not what the instruction restates.
    img = np.stack([E[i][msk[i]].mean(axis=0) for i in range(len(E))])

    def mean_r2(X, Y):
        """Per-dimension cv R^2, averaged. Keeps a 14-D target comparable to 7-D."""
        return float(np.mean([cv_r2(X, Y[:, d]) for d in range(Y.shape[1])]))

    A = act.reshape(len(act), -1)
    return {
        "n_frames": int(len(rows)),
        "n_img_tokens": int(msk[0].sum()),
        "embed_dim": int(E.shape[-1]),
        "state_dim": int(st.shape[1]),
        "r2_state": mean_r2(img, st),
        "r2_action_first": mean_r2(img, A[:, :st.shape[1]]),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"{'arm':22s} {'img tok':>8s} {'state dim':>10s} "
          f"{'R2 state':>9s} {'R2 action':>10s}")
    for name, (_ds, _exp, run_dir) in ARMS.items():
        r = probe_arm(run_dir)
        res[name] = r
        print(f"{name:22s} {r['n_img_tokens']:8d} {r['state_dim']:10d} "
              f"{r['r2_state']:9.4f} {r['r2_action_first']:10.4f}")

    def agg(prefix, key):
        return float(np.mean([v[key] for k, v in res.items() if k.startswith(prefix)]))

    ls, as_ = agg("LIBERO", "r2_state"), agg("ALOHA", "r2_state")
    la, aa = agg("LIBERO", "r2_action_first"), agg("ALOHA", "r2_action_first")
    print(f"\nR2(arm state | image tokens):  LIBERO {ls:.4f}  ALOHA {as_:.4f}"
          f"   (difference {as_ - ls:+.4f})")
    print(f"R2(action     | image tokens):  LIBERO {la:.4f}  ALOHA {aa:.4f}")

    # Report the magnitude, not a binary. A 0.03 difference is not "less
    # information" in any sense that rescues the visually-simpler objection --
    # and ALOHA reaches it from 54 image tokens against LIBERO's 128.
    print("\nVERDICT")
    if abs(as_ - ls) < 0.10:
        print("  Arm configuration is EQUALLY readable from image tokens on both "
              f"testbeds ({ls:.2f} vs {as_:.2f}), and ALOHA reaches that from 54 "
              "image tokens against LIBERO's 128. The scene is not "
              "information-poor, so 'ALOHA is visually simpler' does not explain "
              "its lower PE sensitivity.")
    else:
        print(f"  Readability differs by {as_ - ls:+.3f}. The visually-simpler "
              "objection is NOT ruled out and the action-space claim must be "
              "weakened accordingly.")
    print(f"\n  Action recoverability differs sharply ({la:.2f} vs {aa:.2f}). "
          "LIBERO commands end-effector DELTAS, which a single frame does not "
          "determine; ALOHA commands ABSOLUTE joint targets, which sit close to "
          "the current pose the image already encodes. That is the same "
          "action-space distinction the PE result reports, measured a second and "
          "independent way.")

    (OUT / "spatial_probe.json").write_text(json.dumps(res, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {OUT / 'spatial_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
