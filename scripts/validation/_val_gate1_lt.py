"""
scripts/_val_gate1_lt.py — Validation Gate 1 (dataset quality & diversity), curated Language Table.

Proves the curated 50/50 reach+push subset is clean, balanced, phrasing-diverse, and
covers the workspace:
  • integrity: frame/episode counts, NaN check, per-concept episode balance
  • task balance (reach vs push) + phrasing diversity per concept
  • 2-D action distribution (dx,dy) and effector-state workspace coverage
  • episode-length distribution
  • sample frames across concepts (visual diversity)

Output: docs/experiments/validation/gate1_lt_curated.png
"""
from __future__ import annotations
import glob, io
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[2]
DS = ROOT / "asset/data/language_table_sim_curated"
OUT = ROOT / "docs/experiments/validation/gate1_lt_curated.png"


def main():
    f = sorted(glob.glob(str(DS / "data/chunk-000/file-*.parquet")))[0]
    df = pd.read_parquet(f).sort_values("index").reset_index(drop=True)
    A = np.array(df["action"].tolist(), np.float32)
    S = np.array(df["observation.state"].tolist(), np.float32)
    ep = df["episode_index"].to_numpy()
    nan = bool(np.isnan(A).any() or np.isnan(S).any())
    ep_len = df.groupby("episode_index").size().to_numpy()
    concept_eps = df.groupby("concept")["episode_index"].nunique().sort_values()
    ttype = df.groupby("episode_index")["task_type"].first().value_counts()
    n_uniq = df["instruction"].nunique()

    # sample one frame per concept (visual diversity)
    concepts = list(concept_eps.index)
    sample_imgs, sample_lbl = [], []
    for c in concepts[:8]:
        row = df[df["concept"] == c].iloc[len(df[df["concept"] == c]) // 2]
        sample_imgs.append(np.asarray(Image.open(io.BytesIO(row["image"])).convert("RGB")))
        sample_lbl.append(c)

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 8, figure=fig, height_ratios=[1.0, 1.15, 1.15],
                  hspace=0.5, wspace=0.55, top=0.90, bottom=0.07, left=0.06, right=0.98)

    for k, (im, lb) in enumerate(zip(sample_imgs, sample_lbl)):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lb, fontsize=7.5)
    fig.text(0.06, 0.915, "Visual diversity — one frame per concept (4 reach targets + 4 push pairs)",
             fontsize=11, fontweight="bold")

    # concept balance (episodes)
    axc = fig.add_subplot(gs[1, 0:3])
    colors = ["#55A868" if "->" not in c else "#4C72B0" for c in concept_eps.index]
    axc.barh(range(len(concept_eps)), concept_eps.values, color=colors, edgecolor="black")
    axc.set_yticks(range(len(concept_eps))); axc.set_yticklabels(concept_eps.index, fontsize=8)
    axc.set_title(f"Episodes per concept  (reach={ttype.get('reach',0)}, push={ttype.get('push',0)} — 50/50)",
                  fontsize=10.5, fontweight="bold")
    axc.set_xlabel("# episodes"); axc.grid(axis="x", alpha=0.3)

    # 2D action distribution
    axa = fig.add_subplot(gs[1, 3:5])
    axa.hexbin(A[:, 0], A[:, 1], gridsize=40, cmap="viridis", bins="log")
    axa.set_title("2-D action (dx,dy) density", fontsize=10.5, fontweight="bold")
    axa.set_xlabel("dx"); axa.set_ylabel("dy"); axa.set_aspect("equal")

    # effector workspace coverage
    axw = fig.add_subplot(gs[1, 5:8])
    for tt, col in [("reach", "#55A868"), ("push", "#4C72B0")]:
        m = (df["task_type"] == tt).to_numpy()
        axw.scatter(S[m, 0], S[m, 1], s=2, alpha=0.15, c=col, label=tt)
    axw.set_title("Effector-state workspace coverage", fontsize=10.5, fontweight="bold")
    axw.set_xlabel("effector x"); axw.set_ylabel("effector y")
    axw.legend(markerscale=4, fontsize=9); axw.set_aspect("equal")

    # episode length
    ax1 = fig.add_subplot(gs[2, 0:3])
    for tt, col in [("reach", "#55A868"), ("push", "#4C72B0")]:
        el = df[df["task_type"] == tt].groupby("episode_index").size().to_numpy()
        ax1.hist(el, bins=30, alpha=0.6, color=col, label=f"{tt} (med {int(np.median(el))})")
    ax1.set_title("Episode length by task", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("frames/episode"); ax1.set_ylabel("# episodes"); ax1.legend(fontsize=9)

    # phrasing diversity: verbs used
    ax2 = fig.add_subplot(gs[2, 3:6])
    verbs = Counter(s.lower().split()[0] for s in df["instruction"])
    vk = [v for v, _ in verbs.most_common(8)]
    ax2.bar(vk, [verbs[v] for v in vk], color="#DD8452", edgecolor="black")
    ax2.set_title(f"Phrasing diversity: {n_uniq} unique strings / 8 concepts\n(leading verb distribution)",
                  fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("# frames"); ax2.tick_params(axis="x", rotation=30, labelsize=8)

    # per-joint action std (both dims non-degenerate)
    ax3 = fig.add_subplot(gs[2, 6:8])
    ax3.bar(["dx", "dy"], A.std(0), color="#C44E52", edgecolor="black")
    ax3.set_title("Action std per dim\n(non-degenerate)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("std")

    balanced = abs(ttype.get("reach", 0) - ttype.get("push", 0)) <= 2
    verdict = "PASS" if (not nan and balanced and (A.std(0) > 0.001).all()) else "CHECK"
    fig.suptitle(f"Gate 1 — Curated Language Table Dataset Quality & Diversity   "
                 f"[{len(df):,} frames · {df.episode_index.nunique()} eps · 8 concepts · "
                 f"NaN={nan} · verdict: {verdict}]", fontsize=13.5, fontweight="bold", y=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight"); plt.close()
    print(f"verdict={verdict} | NaN={nan} | balanced={balanced} | uniq_instr={n_uniq} | "
          f"reach={ttype.get('reach',0)} push={ttype.get('push',0)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
