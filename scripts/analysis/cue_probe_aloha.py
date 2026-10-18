"""
scripts/analysis/cue_probe_aloha.py
───────────────────────────────────
Does the PRETRAINED backbone encode the receive cue more accessibly than the
stock one?

The closed-loop result localises the whole ALOHA gap to one event: the stock
policy's receiving gripper never opens in 26.2% of post-lift episodes against
14.5% for the pretrained arm. Three mechanisms produce that same number:

  (1) the cue is not in the stock backbone's tokens,
  (2) the cue is there but the identical 19.2M head cannot extract it,
  (3) the head commands the open and the arm fails to execute it.

The failure-mode analysis cannot separate them, and the paper currently asserts
(1). This probe measures it: it asks how well a LINEAR readout of image tokens
alone can tell that the receiving gripper is about to open.

    image tokens (mean-pooled)  ->  "gripper opens within the next K frames"

Read as: a higher AUC means the cue is more linearly accessible from that
backbone's representation. This is a statement about the representation, not
about the policy -- the trained head is not involved anywhere in this file.

Three controls, because the naive version of this probe measures nothing:

  * TIME baseline. "About to open" is heavily confounded with "we are halfway
    through the episode". A readout that decodes elapsed time gets a high AUC
    with no cue sensitivity at all. Normalised within-episode time is therefore
    probed on its own, and the token probe has to beat it to mean anything.
  * STATE baseline. If 14-DOF proprioception alone predicts the open, the cue
    is not visual and the whole reading collapses. Probed on its own.
  * GROUPED cross-validation. Adjacent frames are near-duplicates, so a random
    split leaks: the same episode lands in train and test and the AUC inflates.
    Folds are split BY EPISODE (GroupKFold), never by frame.

Labels come from the demonstrations, whose gripper polarity is verified here
rather than assumed -- an earlier version of the failure analysis inverted it.

    python scripts/analysis/cue_probe_aloha.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "asset" / "analysis" / "cue_probe"

ARMS = {
    "GR00T N1.7 (pretrained)": "asset/runs/aloha/exp05_groot_transfer",
    "Qwen3-VL-2B (stock)":     "asset/runs/aloha/exp06_qwen3vl_transfer",
}

RECEIVER_GRIPPER_DIM = 6   # verified in verify_polarity(), not assumed
K_FRAMES = 20              # width of the "about to open" window
BUFFER = 3                 # negatives must sit >= BUFFER*K before the open
BLOCK = 500                # h5 rows per read; keeps peak memory near 200 MB
SEED = 0


def episode_index() -> np.ndarray:
    """Per-frame episode id, in the same row order as the embedding cache.

    Rebuilds AlohaDataset's order -- the file-NNN.parquet shards concatenated and
    sorted by `index` -- reading only the episode column, so no video decoding.
    """
    from configs.registry import get_config

    cfg = get_config("aloha", "exp05")
    shards = sorted(glob.glob(str(
        Path(cfg.dataset_root) / "data" / "chunk-000" / "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["index", "episode_index"])
                    for f in shards], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)["episode_index"].to_numpy()


def verify_polarity(states: np.ndarray, ep: np.ndarray) -> dict:
    """Confirm dim 6 is the receiver and that HIGH means open, before using it.

    The receiver's signature is a flat stretch followed by a single sharp rise
    around mid-episode; the picker's gripper instead opens early, closes on the
    cube, and reopens only at the end. Getting this backwards once already
    produced a wrong result, so it is checked rather than trusted.
    """
    g = states[:, RECEIVER_GRIPPER_DIM]
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    thr = lo + 0.6 * (hi - lo)

    first_half, second_half = [], []
    for e in np.unique(ep):
        x = g[ep == e]
        h = len(x) // 2
        first_half.append((x[:h] > thr).mean())
        second_half.append((x[h:] > thr).mean())

    return {"dim": RECEIVER_GRIPPER_DIM, "p5": float(lo), "p95": float(hi),
            "threshold": float(thr),
            "frac_open_first_half": float(np.mean(first_half)),
            "frac_open_second_half": float(np.mean(second_half))}


def build_labels(states: np.ndarray, ep: np.ndarray, k: int):
    """y=1 on the k frames before the receiver opens, y=0 well before that.

    Frames at or after the open are dropped: once the gripper is open the scene
    contains the answer, and predicting it is not the question. The buffer band
    between the two classes is dropped too, so the probe is not rewarded for
    resolving a boundary that the labelling itself smeared.
    """
    g = states[:, RECEIVER_GRIPPER_DIM]
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    thr = lo + 0.6 * (hi - lo)

    y = np.full(len(g), -1, dtype=np.int8)     # -1 = excluded
    t_norm = np.zeros(len(g), dtype=np.float32)
    n_open = 0
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        x = g[idx]
        t_norm[idx] = np.linspace(0, 1, len(idx))
        opened = np.flatnonzero(x > thr)
        if len(opened) == 0:                   # demo never opens: unusable
            continue
        n_open += 1
        t0 = opened[0]
        pos = idx[max(0, t0 - k):t0]
        neg = idx[:max(0, t0 - BUFFER * k)]
        y[neg] = 0
        y[pos] = 1
    return y, t_norm, n_open


def pooled_image_tokens(h5path: Path, rows: np.ndarray) -> np.ndarray:
    """Mean-pool over IMAGE positions only, streamed in blocks.

    Text tokens are excluded: ALOHA has one fixed instruction, so they restate a
    constant and would only add a bias term. bf16 is stored as uint16 -- the bit
    pattern is reinterpreted, never cast.
    """
    import torch

    out = []
    with h5py.File(h5path, "r") as f:
        emb, msk = f["embeddings"], f["img_masks"]
        for s in range(0, len(rows), BLOCK):
            r = rows[s:s + BLOCK]
            E = torch.from_numpy(emb[r].astype(np.uint16)).view(torch.bfloat16)
            E = E.float().numpy()
            M = msk[r].astype(bool)
            out.append(np.stack([E[i][M[i]].mean(0) for i in range(len(r))]))
    return np.concatenate(out, 0)


def grouped_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Out-of-fold AUC with folds split by episode, so no episode spans a split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    ep = episode_index()
    with h5py.File(ROOT / ARMS["GR00T N1.7 (pretrained)"] / "vlm_embeddings.h5",
                   "r") as f:
        states = f["states"][:]
    assert len(ep) == len(states), f"row mismatch {len(ep)} vs {len(states)}"

    pol = verify_polarity(states, ep)
    res["polarity"] = pol
    print("POLARITY CHECK  (receiving gripper, verified not assumed)")
    print(f"  dim {pol['dim']}: p5 {pol['p5']:+.3f}  p95 {pol['p95']:+.3f}  "
          f"threshold {pol['threshold']:+.3f}")
    print(f"  fraction above threshold: first half {pol['frac_open_first_half']:.3f}"
          f"   second half {pol['frac_open_second_half']:.3f}")
    if pol["frac_open_first_half"] >= pol["frac_open_second_half"]:
        print("  [!] the gripper is not predominantly open in the second half -- "
              "the polarity assumption is wrong, stopping")
        return 1
    print("  -> HIGH means open, and the open happens late. Labels are valid.\n")

    y, t_norm, n_open = build_labels(states, ep, K_FRAMES)
    keep = y >= 0
    rows = np.flatnonzero(keep)
    yk, gk = y[keep], ep[keep]
    print(f"LABELS  K={K_FRAMES} frames, buffer {BUFFER}xK")
    print(f"  episodes with a detected open: {n_open}/{len(np.unique(ep))}")
    print(f"  positives {int(yk.sum())}   negatives {int((yk == 0).sum())}   "
          f"excluded {int((~keep).sum())}\n")
    res["labels"] = {"k_frames": K_FRAMES, "n_pos": int(yk.sum()),
                     "n_neg": int((yk == 0).sum()), "n_episodes_open": n_open}

    # ---- baselines: what is obtainable WITHOUT the tokens -------------------
    auc_time = grouped_auc(t_norm[keep][:, None], yk, gk)
    auc_state = grouped_auc(states[keep], yk, gk)
    res["baselines"] = {"time_only": auc_time, "state_only": auc_state}
    print("BASELINES  (the bar the token probe has to clear)")
    print(f"  normalised episode time only : AUC {auc_time:.4f}")
    print(f"  14-DOF proprioception only   : AUC {auc_state:.4f}\n")

    # ---- the probe ----------------------------------------------------------
    print("IMAGE-TOKEN PROBE  (mean-pooled image tokens -> about to open)")
    for name, run_dir in ARMS.items():
        X = pooled_image_tokens(ROOT / run_dir / "vlm_embeddings.h5", rows)
        auc = grouped_auc(X, yk, gk)
        res.setdefault("token_probe", {})[name] = {"auc": auc,
                                                   "embed_dim": int(X.shape[1])}
        print(f"  {name:26s} AUC {auc:.4f}")

    a = res["token_probe"]["GR00T N1.7 (pretrained)"]["auc"]
    b = res["token_probe"]["Qwen3-VL-2B (stock)"]["auc"]
    res["delta_auc"] = float(a - b)
    print(f"\n  delta (pretrained - stock) = {a - b:+.4f}")

    (OUT / "cue_probe.json").write_text(json.dumps(res, indent=2),
                                        encoding="utf-8")
    print(f"\nwrote {OUT / 'cue_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
