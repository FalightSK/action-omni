"""LIBERO-Goal figures for the language axis.

fig17 is the plot the language question actually needs, and the one the study
did not have. LIBERO-Goal's ten tasks share a fixed scene and object set and
differ only in the goal, so if a backbone separates the ten in its latent space
it is separating them by *instruction*, not by appearance. That is precisely the
contrast Language Table cannot provide — there the instruction co-varies with
what is on the table, so image tokens can carry the separation.

Two panels per backbone, deliberately:
  image tokens — should NOT separate the goals. All ten share a scene, so any
                 clustering here is trajectory/appearance, not language.
  text tokens  — SHOULD separate the goals, and how cleanly is the measurement.

Colour is carried by goal identity. Ten categories is far past the point where
hue alone is separable, so this is one of the rare cases where the legend is
load-bearing and the panel is read as "how many distinct blobs", not "which blob
is which" — the eta-squared value printed per panel is the quantitative answer.

Output: figures/fig17_libero_goal_umap.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plots import GRID, INK, INK2, LABELS, MODELS, SURF  # noqa: E402

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"
KEY = "libero_goal"

# 10 goals: a qualitative ramp is the honest choice — no ordering is implied and
# no 10-hue set is all-pairs separable. tab10 is the standard for exactly this.
GOAL_COLORS = plt.get_cmap("tab10").colors


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    M = json.loads((DIR / "metrics.json").read_text(encoding="utf-8"))
    if KEY not in M:
        print(f"{KEY} not in metrics.json — nothing to plot")
        return 0

    with h5py.File(DIR / f"probe_{KEY}.h5", "r") as f:
        instr = json.loads(f.attrs["instructions"])
        iid = f["instr_id"][:]

    mdls = [m for m in MODELS if (DIR / f"embed_{m}_{KEY}.npz").exists()]
    pools = ["image", "text"]
    fig, axes = plt.subplots(len(pools), len(mdls),
                             figsize=(2.55 * len(mdls), 2.9 * len(pools)),
                             squeeze=False)

    for r, pool in enumerate(pools):
        for c, m in enumerate(mdls):
            ax = axes[r][c]
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            z = np.load(DIR / f"embed_{m}_{KEY}.npz")
            name = f"umap_{pool}"
            if name not in z:
                ax.axis("off"); continue
            emb = z[name]
            order = np.random.default_rng(0).permutation(len(emb))
            for g in range(len(instr)):
                sel = order[iid[order] == g]
                ax.scatter(emb[sel, 0], emb[sel, 1], s=2.2,
                           color=GOAL_COLORS[g % 10], linewidths=0, alpha=0.8,
                           label=instr[g] if (r == 0 and c == 0) else None)
            eta = M[KEY].get(m, {}).get("factors", {}).get(pool, {}).get("eta2_instruction")
            if eta is not None and eta == eta:
                ax.annotate(f"η² = {eta:.2f}", (0.04, 0.95), xycoords="axes fraction",
                            fontsize=7.5, color=INK, fontweight="bold", va="top")
            if r == 0:
                ax.set_title(LABELS[m], fontsize=7.6)
            if c == 0:
                ax.set_ylabel(f"{pool} tokens", fontsize=8.5)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=5, labelcolor=INK,
               markerscale=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        "LIBERO-Goal — ten goals, one fixed scene, coloured by goal\n"
        "the scene is constant across all ten, so separation here is language, not appearance",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig17_libero_goal_umap.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("wrote fig17_libero_goal_umap.png")

    # quantitative companion the figure is read against
    print(f"\n{'model':<12}{'image η²':>10}{'text η²':>10}{'all η²':>10}")
    for m in mdls:
        fa = M[KEY][m]["factors"]
        print(f"{m:<12}" + "".join(
            f"{fa.get(p, {}).get('eta2_instruction', float('nan')):>10.3f}"
            for p in ("image", "text", "all")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
