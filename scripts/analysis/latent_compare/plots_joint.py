"""
Figures for the joint (all-datasets-pooled) analysis.

fig10 — one UMAP per backbone over all 9,536 frames at once, coloured by which
        of the four datasets each frame came from. Model identity is carried by
        panel position, so hue is free to encode dataset alone.

fig11 — REMOVED. It plotted the task/scene ratio and the 3-way silhouette as
        bars. The ratio now appears in every fig13 panel title and is recoverable
        exactly from the fig19 distance matrices, which involve no projection at
        all, so the bar chart was a lossier restatement of something already
        shown twice. The function is kept but no longer called.

fig12 — representational similarity analysis. Models cannot share a UMAP (their
        axes are incomparable), but their frame x frame distance matrices can be
        correlated with each other. That puts the backbones themselves into
        one space.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plots import (COLORS, GRID, INK, INK2, LABELS, SHORT, SURF, _style,  # noqa: E402
                   resolve_label_overlaps)

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

# Imported rather than redeclared: a second copy of this list silently went
# stale when a sixth arm was added, rendering fig10/fig11 with five models
# while fig12 read six from the JSON.
from plots import KEY_LABEL, KEYS, MODELS  # noqa: E402

# Derived from KEYS, never hard-coded. The literal four-entry list outlived the
# removal of Language Table and mislabelled the figure: with three datasets in
# the data, the third cluster is LIBERO-Goal, but the legend still called it
# "Language Table (curated)" and the real LIBERO label went unused. A legend
# indexed by position rather than by name cannot detect that.
DS_LABEL = [KEY_LABEL[k] for k in KEYS]
# Panel position carries model identity, so hue only ever encodes dataset.
# These are the palette slots that clear the all-pairs colour-vision floor.
_DS_COLOR_OF = {"aloha_transfer": "#2a78d6", "aloha_insertion": "#eb6834",
                "language_table": "#1baf7a", "libero_goal": "#8a63d2"}
DS_COLOR = [_DS_COLOR_OF[k] for k in KEYS]


def fig_joint_umap(J: dict):
    E = np.load(DIR / "joint_embed.npz")
    y = E["labels"]
    # Wrap to a grid past four arms: a single row of six is ~19in wide and the
    # panels shrink below the point where the cluster structure is legible.
    ncol = min(len(MODELS), 3) if len(MODELS) > 4 else len(MODELS)
    nrow = -(-len(MODELS) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.9 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(MODELS):]:
        ax.set_visible(False)
    for ax, m in zip(flat, MODELS):
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        emb = E[f"umap_{m}"]
        order = np.random.default_rng(0).permutation(len(emb))  # avoid draw-order bias
        for li in range(len(DS_LABEL)):
            sel = order[y[order] == li]
            ax.scatter(emb[sel, 0], emb[sel, 1], s=2.4, c=DS_COLOR[li],
                       linewidths=0, alpha=0.75, label=DS_LABEL[li])
        r = J["per_model"]["image"][m]["task_scene_ratio"]
        ax.set_title(f"{LABELS[m]}\ntask/scene ratio {r:.2f}", fontsize=8)
    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=len(DS_LABEL), labelcolor=INK,
               markerscale=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.10 / nrow))
    # Frame count computed, not asserted — the literal "9,536" survived the
    # removal of a dataset and understated nothing but was simply wrong.
    fig.suptitle(
        f"All {len(DS_LABEL)} datasets in one latent space per backbone "
        f"(image tokens, {len(y):,} frames)",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig10_joint_umap_datasets.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_task_scene(J: dict):   # noqa: F811  (retained for ad-hoc use, not called)
    """DEPRECATED — fig11. Removed from the report and no longer generated.

    Its two panels were the task/scene ratio and the 3-way silhouette. The ratio
    is now printed in every fig13 panel title and is recoverable exactly from the
    fig19 distance matrices, which involve no projection; the silhouette added
    little beyond it. Kept as a function rather than deleted so the metric can
    still be plotted on demand, but main() no longer calls it.
    """
    pools = ["image", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9), squeeze=False)

    ax = _style(axes[0][0])
    w = 0.38
    for pi, pool in enumerate(pools):
        xs = np.arange(len(MODELS)) + (pi - 0.5) * w
        vs = [J["per_model"][pool][m]["task_scene_ratio"] for m in MODELS]
        cs = [COLORS[m] for m in MODELS]
        b = ax.bar(xs, vs, width=w * 0.9, color=cs, alpha=1.0 if pool == "image" else 0.45,
                   edgecolor=SURF, linewidth=1.3)
        for rect, v in zip(b, vs):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.4, color=INK2)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_ylabel("task / scene separation ratio")
    ax.set_title("Same scene, different task  ÷  different scene\n"
                 "(higher = organised by task, not appearance)", fontsize=9.5)
    hand = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=a) for a in (1.0, 0.45)]
    ax.legend(hand, ["image tokens", "all tokens"], frameon=False, fontsize=7.5,
              labelcolor=INK)

    ax2 = _style(axes[0][1])
    for pi, pool in enumerate(pools):
        xs = np.arange(len(MODELS)) + (pi - 0.5) * w
        vs = [J["per_model"][pool][m]["silhouette_3way"] for m in MODELS]
        cs = [COLORS[m] for m in MODELS]
        b = ax2.bar(xs, vs, width=w * 0.9, color=cs, alpha=1.0 if pool == "image" else 0.45,
                    edgecolor=SURF, linewidth=1.3)
        for rect, v in zip(b, vs):
            ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=6.4, color=INK2)
    ax2.set_xticks(range(len(MODELS)))
    ax2.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax2.set_ylabel("silhouette (3-way dataset clustering)")
    ax2.set_title("How cleanly the three datasets separate overall", fontsize=9.5)

    fig.suptitle("Does the latent organise by task, or by visual appearance?", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_task_vs_scene.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_rsa(J: dict):
    R = np.array(J["rsa"]["matrix"])
    mds = np.array(J["rsa"]["mds"])
    names = J["rsa"]["models"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # sequential single hue, light -> dark (never a rainbow for an ordered value)
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(R, cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([SHORT[m].replace("\n", " ") for m in names], fontsize=7, rotation=30,
                       ha="right")
    ax.set_yticklabels([SHORT[m].replace("\n", " ") for m in names], fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=SURF if R[i, j] > 0.82 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Representational similarity between backbones\n"
                 "(Spearman correlation of frame×frame distance matrices)", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_edgecolor(GRID)

    # MDS placement. Labels go into PRE-ALLOCATED slots rather than being pushed
    # around: points on the left half get a label column on the left, points on
    # the right get one on the right, and within each column the slots are evenly
    # spaced and assigned in vertical order. Overlap is then impossible by
    # construction, which iterative repulsion could not guarantee here — five of
    # the seven backbones sit almost on top of each other (RSA 0.87-0.96), so any
    # scheme that derives label positions from point positions puts five labels
    # in the same place.
    ax2 = _style(axes[1])
    n = len(names)
    xlo, xhi = mds[:, 0].min(), mds[:, 0].max()
    ylo, yhi = mds[:, 1].min(), mds[:, 1].max()
    spanx = max(xhi - xlo, 1e-9)
    spany = max(yhi - ylo, 1e-9)
    mid = (xlo + xhi) / 2

    left = [i for i in range(n) if mds[i, 0] <= mid]
    right = [i for i in range(n) if mds[i, 0] > mid]
    left.sort(key=lambda i: mds[i, 1])
    right.sort(key=lambda i: mds[i, 1])

    lab_x_left = xlo - 0.62 * spanx
    lab_x_right = xhi + 0.62 * spanx

    def slots(k):
        """k evenly spaced y positions spanning the point range, with margin."""
        if k == 0:
            return []
        if k == 1:
            return [(ylo + yhi) / 2]
        return list(np.linspace(ylo - 0.12 * spany, yhi + 0.12 * spany, k))

    for col, xpos, ha in ((left, lab_x_left, "right"), (right, lab_x_right, "left")):
        for idx, ypos in zip(col, slots(len(col))):
            m = names[idx]
            ax2.annotate(
                SHORT[m].replace("\n", " "), xy=(mds[idx, 0], mds[idx, 1]),
                xytext=(xpos, ypos), ha=ha, va="center", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9,
                                shrinkA=0, shrinkB=10),
            )
    for i, m in enumerate(names):
        ax2.scatter(mds[i, 0], mds[i, 1], s=190, color=COLORS[m], edgecolor=SURF,
                    linewidth=2, zorder=3)

    ax2.set_xlim(lab_x_left - 0.42 * spanx, lab_x_right + 0.42 * spanx)
    ax2.set_ylim(ylo - 0.34 * spany, yhi + 0.34 * spany)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"The {len(MODELS)} backbones placed in one space\n"
                  "(MDS on 1 − representational similarity)", fontsize=9.5)

    fig.suptitle("How similarly do these backbones see the world?", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_rsa_models.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    J = json.loads((DIR / "joint_metrics.json").read_text(encoding="utf-8"))
    fig_joint_umap(J)
    fig_rsa(J)
    print("wrote fig10, fig12   (fig11 retired — see fig_task_scene docstring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
