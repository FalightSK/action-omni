"""The depth-matched control, drawn — the figure behind the Finding 3 retraction.

This is the most consequential analysis in the study and it had no figure: it
overturned a claimed "6 out of 6 replicated signature of robot pretraining" by
showing that most of the effect was the read-depth difference, not the
finetuning. Tables state that; a figure makes it obvious.

The design point is the pairing. Each robot arm appears twice — once against its
stock model read at that model's own last layer (the conventional, CONFOUNDED
comparison) and once against the same stock weights truncated to the robot arm's
exact depth (the CLEAN comparison). Reading down each pair of bars shows how much
of the apparent effect survives once depth is held fixed.

Two panels because the two metrics disagree, and that disagreement is the result:

  left   image/text principal angle — the effect COLLAPSES under depth matching
         (GR00T keeps 3%, SmolVLA 0%). What looked like a signature of robot
         pretraining was subspaces separating as you ascend any language stack.

  right  task/scene ratio — the effect SURVIVES for GR00T (98%) and was never
         there for SmolVLA. So robot finetuning does change task organisation;
         it just does not change image/text geometry.

Pi-0.5 is drawn once, in a different fill, because its pair was depth-matched
from the start (18 layers vs 18) and needs no correction.

Output: figures/fig20_depth_matched_control.png
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
from plots import GRID, INK, INK2, SURF, _style  # noqa: E402

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

# RETIRED alongside depth_matched_control.py — the arms this plots
# (cosmos16, smolvlm2_16) are no longer in backbones.ARMS, because every stock
# control is now read at its robot descendant's documented layer, making the
# whole figure's comparison redundant. Kept for provenance only; running it will
# fail to find its inputs.
KEYS = ["aloha_transfer", "aloha_insertion", "language_table"]
KEY_SHORT = ["ALOHA-T", "ALOHA-I", "Lang.Table"]

# (contrast key, row label, is_depth_matched)
ROWS = [
    ("pi05_vs_paligemma",      "Pi-0.5 vs PaliGemma\n(18 vs 18 — already matched)", True),
    ("groot_vs_cosmos",        "GR00T vs Cosmos\n(16 vs 28 — confounded)",          False),
    ("groot_vs_cosmos16",      "GR00T vs Cosmos-16\n(16 vs 16 — clean)",            True),
    ("smolvla_vs_smolvlm2",    "SmolVLA vs SmolVLM2\n(16 vs 32 — confounded)",      False),
    ("smolvla_vs_smolvlm2_16", "SmolVLA vs SmolVLM2-16\n(16 vs 16 — clean)",        True),
]
# blue = a comparison you can trust, grey-brown = one that is depth-confounded
C_CLEAN, C_CONF = "#2a78d6", "#a89880"


def bars(ax, values, title, xlabel, xlim=None):
    """One horizontal grouped bar per contrast, one bar per dataset."""
    _style(ax)
    h = 0.8 / len(KEYS)
    for ri, (key, label, matched) in enumerate(ROWS):
        col = C_CLEAN if matched else C_CONF
        for ki in range(len(KEYS)):
            v = values.get(key, [np.nan] * len(KEYS))[ki]
            if v != v:
                continue
            y = -(ri + (ki - (len(KEYS) - 1) / 2) * h)
            ax.barh(y, v, height=h * 0.88, color=col,
                    alpha=[1.0, 0.72, 0.45][ki], edgecolor=SURF, linewidth=0.9)
            ax.text(v + (0.012 * (xlim[1] - xlim[0]) if xlim else 0.02) * np.sign(v or 1),
                    y, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.4, color=INK2)
    ax.axvline(0, color=INK2, lw=1.0, zorder=4)
    ax.set_yticks([-i for i in range(len(ROWS))])
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=7.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9.8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    D = json.loads((DIR / "depth_matched.json").read_text(encoding="utf-8"))
    G = json.loads((DIR / "depth_matched_geometry.json").read_text(encoding="utf-8"))

    ang = {k: [D["mean_angle_deg"].get(k, {}).get("per_dataset", {}).get(ds, {}).get("delta", np.nan)
               if D["mean_angle_deg"].get(k, {}).get("per_dataset", {}).get(ds) else np.nan
               for ds in KEYS]
           for k, _, _ in ROWS}
    # geometry has one value per contrast, not one per dataset — repeat it into
    # the first slot only so the bar reads as a single measurement
    ratio = {k: [G.get(k, {}).get("delta", np.nan), np.nan, np.nan] for k, _, _ in ROWS}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    bars(axes[0], ang,
         "Image/text subspace angle — the effect COLLAPSES\n"
         "three bars per row: ALOHA-T · ALOHA-I · Language Table",
         "change in mean principal angle vs the stock model (degrees)",
         xlim=(-3, 25))
    bars(axes[1], ratio,
         "Task/scene organisation — the effect SURVIVES\n"
         "one measurement per row (image tokens, all datasets pooled)",
         "change in task/scene ratio vs the stock model",
         xlim=(-0.06, 0.40))

    for ax, note in zip(axes, [
        "GR00T keeps 3% · SmolVLA keeps 0%",
        "GR00T keeps 98% · SmolVLA 0%",
    ]):
        ax.annotate(note, (0.98, 0.03), xycoords="axes fraction", ha="right",
                    fontsize=8, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec=GRID))

    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_CLEAN),
               plt.Rectangle((0, 0), 1, 1, fc=C_CONF)]
    fig.legend(handles, ["depth-matched — trustworthy",
                         "depth-confounded — finetuning mixed with read depth"],
               frameon=False, fontsize=8.5, ncol=2, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Holding read depth fixed removes most of what was attributed to robot pretraining",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig20_depth_matched_control.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("wrote fig20_depth_matched_control.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
