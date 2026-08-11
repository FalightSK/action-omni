"""Figures for findings 5 (read depth) and 6 (the dataset gate).

Both findings were text-only until now, which is the wrong way round: finding 5
is the most directly actionable result in the study, and its shape — a rise to
mid-stack then a decline — is far easier to read as a curve than as a table of
best-depth labels.

fig15  action R² against relative read depth, one panel per dataset, one line
       per arm. The question it answers at a glance: does control information
       accumulate all the way up the language stack, or peak early?

fig16  the dataset gate. Left: how decodable the action is in each dataset,
       every arm plotted, so the three regimes (ALOHA / LIBERO-Goal /
       Language Table) separate visually. Right: LIBERO-Goal broken out by
       factor, which is where its case rests — its action signal is middling but
       its instruction separation is what makes it the language testbed.

Design follows plots.py: arm identity is carried by hue only on line/bar forms
(never inside a scatter), the palette and short labels are imported rather than
redefined so they cannot drift, and every bar carries a direct value label
because several palette slots fall below the 3:1 contrast floor.
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
from plots import (COLORS, GRID, INK, INK2, KEY_LABEL, LABELS, MODELS, PAIRS,  # noqa: E402
                   SHORT, SURF, _style)

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

sys.path.insert(0, str(HERE))
from backbones import DOC_LAYER, GATE_KEYS  # noqa: E402
from backbones import KEYS  # noqa: E402

DEPTHS = ["d000", "d025", "d050", "d075", "d100"]
DEPTH_X = [0.00, 0.25, 0.50, 0.75, 1.00]

# Fig 15 (the gate) is the ONLY figure that keeps Language Table. Its argument
# is that LT fails the gate while LIBERO-Goal passes, so dropping LT there would
# leave the scale with no negative anchor.
GATE_KEY_LIST = GATE_KEYS

# The 0% point is the vision tower plus the embedding lookup, with no decoder
# block applied — and at that depth the two modalities are still in their native
# scales, so a pooled "all" vector is dominated by whichever has the larger norm.
# Measured: SmolVLM2 reads image 3.09 / text 0.063, while Pi-0.5 reads image
# 1.25 / text 3.40, because Gemma scales embeddings by sqrt(hidden)=45.25. The
# imbalance INVERTS between architectures, so an "all" curve is not comparable
# across arms at 0%. Each pool is therefore drawn as its own panel row.
POOLS_SHOWN = ["all", "image", "text"]

# The truncated controls (cosmos16, smolvlm2_16) are gone from the roster.
# They existed only to depth-match a robot arm against its stock base; now that
# every stock arm is read at its DESCENDANT's documented layer, the comparison
# is depth-matched by construction and a separate truncated arm would duplicate
# it. This list is therefore just the roster.
DEPTH_MODELS = MODELS


def _arm_style(m: str) -> tuple[str, str, str]:
    """(colour, linestyle, legend label) for one arm.

    Stock controls are dashed so each finetuned/stock pair reads as a pair —
    the comparison the figure exists to support.
    """
    stock = {s for _, s in PAIRS}
    return COLORS[m], ("--" if m in stock else "-"), LABELS[m].replace("\n", " ")


# ── Fig 15: depth curves ─────────────────────────────────────────────────────

def fig_depth(D: dict):
    keys = [k for k in KEYS if k in D]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 4.0), squeeze=False)

    for ax, key in zip(axes[0], keys):
        _style(ax)
        for m in DEPTH_MODELS:
            r = D[key].get(m)
            if not r:
                continue
            xs = [x for x, d in zip(DEPTH_X, DEPTHS) if d in r]
            ys = [r[d]["r2_action"] for d in DEPTHS if d in r]
            if not ys:
                continue
            col, ls, lab = _arm_style(m)
            ax.plot(xs, ys, marker="o", ls=ls, ms=4.5, lw=1.9, color=col,
                    mec=SURF, mew=1.0, label=lab)
            # mark the peak — the whole point of the figure is where it sits
            bi = int(np.argmax(ys))
            ax.plot([xs[bi]], [ys[bi]], "o", ms=10, mfc="none",
                    mec=col, mew=1.8, zorder=5)
        # Derived from DEPTH_X, never hard-coded: a literal list silently
        # disagreed with the grid the moment 0% and 25% were added, and
        # matplotlib only caught it because the counts happened to differ.
        ax.set_xticks(DEPTH_X)
        ax.set_xticklabels(
            [{0.0: "0%\n(no decoder)", 1.0: "100%\n(last layer)"}.get(x, f"{x:.0%}")
             for x in DEPTH_X], fontsize=8)
        ax.set_xlabel("relative depth in the language stack")
        ax.set_title(KEY_LABEL[key], fontsize=10)
        if ax is axes[0][0]:
            ax.set_ylabel("action chunk decodability (CV ridge R²)")

    h, l = axes[0][0].get_legend_handles_labels()
    # Nine arms wrap badly on one row at this width; five per row keeps the
    # dashed controls beside their solid parents in reading order.
    fig.legend(h, l, frameon=False, fontsize=8, ncol=5, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        "Where in the language stack does control information live?\n"
        "ring = each arm's peak · dashed = stock control · 0% = vision tower + "
        "text embeddings, before any decoder block\n"
        "action R² peaks in the FIRST QUARTER of the stack, then degrades — and "
        "PaliGemma peaks at 0%, its decoder adding nothing",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig15_depth_curves.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ── Fig 16: the dataset gate ─────────────────────────────────────────────────

def fig_gate(M: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # left — action decodability per dataset, every arm as a point
    ax = _style(axes[0])
    # GATE_KEY_LIST, not KEYS: this is the one figure that keeps Language Table,
    # as the negative anchor that makes the gate a scale rather than two points.
    keys = [k for k in GATE_KEY_LIST if k in M]
    for xi, key in enumerate(keys):
        vals = []
        for m in MODELS:
            fa = M[key].get(m, {}).get("factors", {}).get("all", {})
            v = fa.get("r2_action")
            if v is None:
                continue
            vals.append(v)
            ax.plot(xi, v, "o", ms=7, color=COLORS[m], mec=SURF, mew=1.2, zorder=3)
        if vals:
            ax.plot([xi - 0.26, xi + 0.26], [np.mean(vals)] * 2, "-",
                    color=INK2, lw=1.6, zorder=2)
            ax.annotate(f"{min(vals):.2f}–{max(vals):.2f}", (xi, max(vals)),
                        textcoords="offset points", xytext=(0, 11), ha="center",
                        fontsize=8, color=INK2, fontweight="bold")
    # the gate threshold: below this a dataset cannot test the action pathway
    ax.axhspan(0, 0.10, color="#c9463a", alpha=0.07, zorder=0)
    ax.annotate("below ~0.10: the action is not recoverable\nfrom one frame by any encoder",
                (len(keys) - 0.5, 0.05), ha="right", va="center",
                fontsize=7.6, color="#8c3a31")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([KEY_LABEL[k].replace(" (curated)", "\n(curated)") for k in keys],
                       fontsize=8)
    ax.set_ylabel("action chunk decodability (CV ridge R²)")
    ax.set_ylim(-0.03, 0.92)
    ax.set_title("The dataset gate — is the task learnable at all?\n"
                 "one point per backbone; bar = mean", fontsize=9.8)

    # right — LIBERO-Goal by factor: why it earns its place despite mid action R²
    ax2 = _style(axes[1])
    factors = [("r2_action", "action chunk"), ("r2_state", "proprio state"),
               ("eta2_instruction", "instruction (η²)")]
    w = 0.8 / len(MODELS)
    for mi, m in enumerate(MODELS):
        fa = M.get("libero_goal", {}).get(m, {}).get("factors", {}).get("all", {})
        vs = [fa.get(f, np.nan) for f, _ in factors]
        xs = np.arange(len(factors)) + (mi - (len(MODELS) - 1) / 2) * w
        b = ax2.bar(xs, vs, width=w * 0.9, color=COLORS[m], edgecolor=SURF,
                    linewidth=1.1, label=SHORT[m].replace("\n", " "))
        for rect, v in zip(b, vs):
            if v == v:
                ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=5.8, color=INK2, rotation=90)
    ax2.set_xticks(range(len(factors)))
    ax2.set_xticklabels([n for _, n in factors], fontsize=8.5)
    ax2.set_ylabel("variance explained")
    ax2.set_ylim(0, 1.06)
    ax2.set_title("LIBERO-Goal by factor — why it is the language testbed\n"
                  "10 goals over a fixed scene, so instruction is the only cue",
                  fontsize=9.8)
    ax2.legend(frameon=False, fontsize=6.8, ncol=2, labelcolor=INK, loc="upper left")

    fig.suptitle("Not every dataset can test every component", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig16_dataset_gate.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    D = json.loads((DIR / "depth_sweep.json").read_text(encoding="utf-8"))
    M = json.loads((DIR / "metrics.json").read_text(encoding="utf-8"))
    fig_depth(D)
    fig_gate(M)
    print("wrote fig15_depth_curves.png, fig16_dataset_gate.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
