"""
scripts/analysis/plots_raw_vs_adapted.py
────────────────────────────────────────
What does the trained adapter actually change?

fig7 showed the adapter's OUTPUT separating the ten LIBERO-Goal tasks very
differently per arm (image-pool silhouette +0.273 for Pi-0.5, +0.071 for
PaliGemma). That is a property of the adapted space — but the raw VLM features
already differ between the arms, so it does not say how much of the structure
the adapter CREATED as opposed to inherited.

This script runs the identical pipeline on the raw cached embeddings — the
frozen VLM's layer-18 output, before LoRA, before the positional concat, before
the MLP — and puts the two side by side.

Making the comparison fair
──────────────────────────
Raw features are 2048-wide, adapted are 512. Silhouette is not
dimension-invariant, so scoring them in their native widths would confound "the
adapter sharpened task structure" with "the adapter reduced dimensionality".
Both are therefore scored in their own top-50 PC space, which is the same
protocol the Chapter 1 analysis uses (N_PC = 50 in analyze_joint.py). The UMAP
panels are drawn from the standardised full-width features, since UMAP is only
being used to look at the data, not to score it.

Outputs
───────
fig9_raw_vs_adapted   8 UMAP panels: 2 arms x {raw, adapted} x {image, text}
fig10_silhouette_shift  the same four (arm, pool) pairs as a slope chart, plus
                        the per-arm change, which is the actual answer
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from head_diagnostics import (ARMS, BATCH, DEVICE, frame_labels,  # noqa: E402
                              read_batch, stride, val_indices)
from plots_head_latents import (EXP, FIG, N_PC, RUN, SHORT,  # noqa: E402
                                SHORT_TASK, TASK_CMAP, adapter_pools,
                                task_silhouette, _style)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
COLORS = {"pi05": "#eb6834", "paligemma": "#1baf7a"}
N_SCATTER = 1500


def raw_pools(arm, rows):
    """Mean-pooled RAW cached embeddings per frame, split by token type.

    The cache stores bf16 bit patterns as uint16 because h5py has no bfloat16;
    they must be reinterpreted with .view, never cast. read_batch handles that.
    """
    h5 = ROOT / RUN[arm] / "vlm_embeddings.h5"
    img, txt = [], []
    for i in range(0, len(rows), BATCH):
        emb, msk, _, _ = read_batch(h5, rows[i:i + BATCH])
        m = msk.unsqueeze(-1).float()
        img.append((emb * m).sum(1).div(m.sum(1)).numpy())
        nm = (~msk).unsqueeze(-1).float()
        txt.append((emb * nm).sum(1).div(nm.sum(1)).numpy())
    return np.concatenate(img, 0), np.concatenate(txt, 0)


def embed(Xz):
    import umap
    return umap.UMAP(n_neighbors=30, min_dist=0.12, random_state=0,
                     n_components=2).fit_transform(Xz)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    task_idx, _, names = frame_labels()
    with h5py.File(ROOT / RUN["pi05"] / "vlm_embeddings.h5", "r") as f:
        n_total = f["embeddings"].shape[0]
    rows = stride(np.sort(val_indices(n_total)), N_SCATTER)
    y = task_idx[rows]

    data: dict = {}
    for arm in ARMS:
        ri, rt = raw_pools(arm, rows)
        ai, at = adapter_pools(arm, rows)
        torch.cuda.empty_cache()
        for stage, (pi, pt) in (("raw", (ri, rt)), ("adapted", (ai, at))):
            for pool, X in (("image", pi), ("text", pt)):
                s, Xz = task_silhouette(X, y)
                data[(arm, stage, pool)] = (s, embed(Xz))
                print(f"  {arm:10} {stage:8} {pool:6} silhouette {s:+.3f} "
                      f"(dim {X.shape[1]})")

    # ── fig 9 — the eight panels ────────────────────────────────────────────
    cols = [("raw", "image"), ("adapted", "image"), ("raw", "text"), ("adapted", "text")]
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 9.0))
    for r, arm in enumerate(ARMS):
        for c, (stage, pool) in enumerate(cols):
            ax = _style(axes[r][c])
            s, Z = data[(arm, stage, pool)]
            for t in np.unique(y):
                m = y == t
                ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.75,
                           color=TASK_CMAP(t % 10),
                           label=SHORT_TASK.get(names[t], names[t][:18]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{stage} · {pool} tokens\nsilhouette {s:+.3f}",
                         fontsize=9.5)
            if c == 0:
                ax.set_ylabel(SHORT[arm], fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=6.2, markerscale=2.4,
                      loc="upper left", ncol=2, handletextpad=0.2,
                      columnspacing=0.5)
    fig.suptitle("Frozen VLM features (raw) vs what the trained adapter hands the DiT.\n"
                 "Scored in each space's own top-50 PCs so the 2048-wide raw and "
                 "512-wide adapted features are comparable.",
                 fontsize=11.5, y=0.98)
    fig.savefig(FIG / "fig9_raw_vs_adapted.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig9_raw_vs_adapted")

    # ── fig 10 — the answer, as numbers ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    ax = _style(axes[0])
    for arm in ARMS:
        for pool, ls, mk in (("image", "-", "o"), ("text", "--", "s")):
            a = data[(arm, "raw", pool)][0]
            b = data[(arm, "adapted", pool)][0]
            ax.plot([0, 1], [a, b], ls, marker=mk, color=COLORS[arm], lw=2, ms=6,
                    label=f"{SHORT[arm].split('  ')[0]} · {pool}")
            ax.annotate(f"{b:+.3f}", (1, b), fontsize=8, color=COLORS[arm],
                        xytext=(6, -3), textcoords="offset points")
            ax.annotate(f"{a:+.3f}", (0, a), fontsize=8, color=COLORS[arm],
                        ha="right", xytext=(-6, -3), textcoords="offset points")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw VLM\n(frozen, layer 18)", "after adapter\n(trained)"])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylabel("task silhouette (top-50 PC space)")
    ax.set_title("What training changes", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax = _style(axes[1])
    w, xs = 0.36, np.arange(2)
    for i, arm in enumerate(ARMS):
        d = [data[(arm, "adapted", p)][0] - data[(arm, "raw", p)][0]
             for p in ("image", "text")]
        b = ax.bar(xs + (i - 0.5) * w, d, w * 0.92, color=COLORS[arm],
                   label=SHORT[arm].split("  ")[0])
        for bb in b:
            h = bb.get_height()
            ax.text(bb.get_x() + bb.get_width() / 2, h + (0.004 if h >= 0 else -0.012),
                    f"{h:+.3f}", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["image pool", "text pool"])
    ax.set_ylabel("Δ silhouette (adapted − raw)")
    ax.set_title("Where each adapter puts its capacity", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.24)

    fig.suptitle("The adapter preserves task structure; it does not create it — and where the "
                 "backbone\nsupplied a lot, the adapter discards some as irrelevant to "
                 "the action objective",
                 fontsize=11, y=1.06)
    fig.savefig(FIG / "fig10_silhouette_shift.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig10_silhouette_shift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
