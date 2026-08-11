"""
scripts/_val_gate1_aloha.py — Validation Gate 1 (dataset quality & diversity), ALOHA insertion.

Proves the training set is clean and diverse enough to learn from:
  • integrity: frame/episode counts, NaN check, episode-length uniformity
  • per-joint ACTION and STATE distributions (all 14 DOF exercised, non-degenerate)
  • action smoothness (step-to-step |Δa| — teleop should be smooth, no jumps)
  • initial-state diversity across the 50 episodes (varied start configs)
  • visual diversity: sample top-camera frames across episodes (peg/socket placement)

Output: docs/experiments/validation/gate1_aloha_insert.png
"""
from __future__ import annotations
import glob, os
from pathlib import Path

import av
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[2]
DS = ROOT / "asset/data/aloha_sim_insertion_human"
OUT = ROOT / "docs/experiments/validation/gate1_aloha_insert.png"


def load():
    sh = sorted(glob.glob(str(DS / "data/chunk-000/file-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in sh], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    A = np.array(df["action"].tolist(), np.float32)
    S = np.array(df["observation.state"].tolist(), np.float32)
    ep = df["episode_index"].to_numpy()
    return df, A, S, ep


def sample_frames(idxs):
    vid = sorted(glob.glob(str(DS / "videos/observation.images.top/chunk-000/*.mp4")))[0]
    want = sorted(idxs); out = {}
    c = av.open(vid); s = c.streams.video[0]; n = 0
    for packet in c.demux(s):
        for fr in packet.decode():
            if n in want:
                out[n] = fr.to_ndarray(format="rgb24")
            n += 1
            if n > want[-1]:
                break
        if n > want[-1]:
            break
    c.close()
    return [out[i] for i in want]


def main():
    df, A, S, ep = load()
    N = len(df); n_ep = len(np.unique(ep))
    ep_len = df.groupby("episode_index").size().to_numpy()
    nan = bool(np.isnan(A).any() or np.isnan(S).any())
    dA = np.abs(np.diff(A, axis=0)); dA = dA[np.diff(ep) == 0]        # within-episode step deltas
    starts = np.array([np.where(ep == e)[0][0] for e in np.unique(ep)])
    init_states = S[starts]                                           # (50,14) per-episode start

    frame_idx = list(starts[np.linspace(0, n_ep - 1, 6).astype(int)])
    frames = sample_frames([int(i) for i in frame_idx])

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.15, 1.15],
                  hspace=0.42, wspace=0.5, top=0.90, bottom=0.07, left=0.06, right=0.98)

    # Row 0 — sample frames (visual diversity)
    for k, (fr, gi) in enumerate(zip(frames, frame_idx)):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(fr); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ep {int(ep[gi])}", fontsize=9)
    fig.text(0.06, 0.915, "Visual diversity — top camera at 6 episode starts (varied peg/socket placement)",
             fontsize=11, fontweight="bold")

    # Row 1 left — per-joint ACTION violin
    axA = fig.add_subplot(gs[1, 0:3])
    axA.violinplot([A[:, j] for j in range(14)], showextrema=False, widths=0.9)
    axA.set_title("Per-joint ACTION distribution (all 14 DOF exercised)", fontsize=11, fontweight="bold")
    axA.set_xlabel("joint index (1-7 right arm, 8-14 left arm)"); axA.set_ylabel("norm. joint target")
    axA.set_xticks(range(1, 15)); axA.grid(axis="y", alpha=0.3)

    # Row 1 right — per-joint STATE violin
    axS = fig.add_subplot(gs[1, 3:6])
    axS.violinplot([S[:, j] for j in range(14)], showextrema=False, widths=0.9)
    axS.set_title("Per-joint STATE (observation) distribution", fontsize=11, fontweight="bold")
    axS.set_xlabel("joint index"); axS.set_ylabel("norm. joint pos")
    axS.set_xticks(range(1, 15)); axS.grid(axis="y", alpha=0.3)

    # Row 2 left — episode length
    ax1 = fig.add_subplot(gs[2, 0:2])
    ax1.hist(ep_len, bins=20, color="#4C72B0", edgecolor="black")
    ax1.set_title(f"Episode length (all uniform = {int(ep_len[0])})", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("frames per episode"); ax1.set_ylabel("# episodes")

    # Row 2 mid — action smoothness
    ax2 = fig.add_subplot(gs[2, 2:4])
    ax2.hist(dA.mean(1), bins=60, color="#55A868", edgecolor="none")
    ax2.set_title("Action smoothness: step-to-step |Δa|\n(small = smooth teleop, no jumps)",
                  fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("mean |Δaction| per step"); ax2.set_ylabel("count"); ax2.set_yscale("log")

    # Row 2 right — initial-state diversity across episodes
    ax3 = fig.add_subplot(gs[2, 4:6])
    ax3.bar(range(1, 15), init_states.std(0), color="#C44E52", edgecolor="black")
    ax3.set_title("Init-state diversity across 50 eps\n(std of episode-start pose per joint)",
                  fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("joint index"); ax3.set_ylabel("std across episodes"); ax3.set_xticks(range(1, 15))
    ax3.grid(axis="y", alpha=0.3)

    verdict = ("PASS" if (not nan and len(set(ep_len)) == 1 and (A.std(0) > 0.05).all()) else "CHECK")
    fig.suptitle(f"Gate 1 — ALOHA Insertion Dataset Quality & Diversity   [{N:,} frames · {n_ep} episodes · "
                 f"NaN={nan} · verdict: {verdict}]", fontsize=13.5, fontweight="bold", y=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight"); plt.close()
    print(f"verdict={verdict} | NaN={nan} | ep_len_uniform={len(set(ep_len))==1} | "
          f"min_joint_std={A.std(0).min():.3f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
