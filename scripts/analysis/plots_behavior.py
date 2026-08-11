"""
scripts/analysis/plots_behavior.py
──────────────────────────────────
Model-behaviour figures across all four Chapter 2 arms.

The earlier figure set (plots_head_diagnostics.py) covered the Pi-0.5/PaliGemma
pair only. With exp03/exp04 finished there are four arms in two families, and the
routing result now has enough points to be shown as a relationship rather than a
contrast.

fig12_routing_four_arms
    A  text-ablation ratio vs closed-loop SR — the headline. How much each head's
       loss degrades when the 16/24 instruction tokens are zeroed, against how
       well it actually controls. r = +0.984 over four arms.
    B  the same against raw attention MASS, which is a much weaker predictor
       (r = +0.570) — included deliberately, because it is the metric one would
       reach for first and it is the wrong one. The pairs differ in token
       composition (16 of 272 vs 24 of 88), so mass is not comparable across
       families; ablation measures function directly.
    C  where the goal lives: action-decodability of the adapted image pool vs the
       adapted text pool, per arm.

fig13_per_dim_error
    Open-loop error per action dimension, normalised by each dimension's own
    ground-truth standard deviation. The seven OSC dims differ in scale by up to
    17x (rotation std 0.055-0.101, gripper 0.956) and we train an unweighted MSE
    on RAW actions, so this asks whether the low-variance rotation channels are
    being neglected. They are not — the penalty is a uniform ~1.6x, not the
    blindness a naive loss-share argument predicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HD = ROOT / "asset" / "analysis" / "head_diagnostics"
FIG = HD / "figures"

ARMS = ["pi05", "paligemma", "groot", "qwen3vl"]
LAB = {"pi05": "Pi-0.5", "paligemma": "PaliGemma-3B",
       "groot": "GR00T N1.7", "qwen3vl": "Qwen3-VL-2B"}
RUN = {"pi05": "exp01_goal", "paligemma": "exp02_paligemma",
       "groot": "exp03_groot", "qwen3vl": "exp04_qwen3vl"}
# Pretrained arms warm, stock arms cool — the same encoding as fig11.
COL = {"pi05": "#eb6834", "paligemma": "#1baf7a",
       "groot": "#8a63d2", "qwen3vl": "#0f9b8e"}
MARK = {"pi05": "o", "paligemma": "o", "groot": "s", "qwen3vl": "s"}
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


def _style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def load():
    d = json.loads((HD / "head_diagnostics.json").read_text(encoding="utf-8"))
    sr = {}
    for a in ARMS:
        r = json.loads((ROOT / "asset" / "runs" / "libero" / RUN[a]
                        / "eval_libero.json").read_text(encoding="utf-8"))["results"]
        sr[a] = 100.0 * np.mean([x["success"] for x in r
                                 if x["condition"] == "canonical"])
    return d, sr


def _scatter_fit(ax, xs, ys, xlabel, title):
    for a, x, y in zip(ARMS, xs, ys):
        ax.scatter(x, y, s=130, color=COL[a], marker=MARK[a],
                   edgecolor=SURF, lw=1.2, zorder=3)
        ax.annotate(LAB[a], (x, y), fontsize=8, color=INK2,
                    xytext=(8, -4), textcoords="offset points")
    z = np.polyfit(xs, ys, 1)
    xx = np.linspace(min(xs), max(xs), 20)
    ax.plot(xx, np.polyval(z, xx), color=INK2, ls="--", lw=1.3, zorder=1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("canonical success rate (%)")
    ax.set_title(f"{title}\nr = {r:+.3f}  (n = 4)", fontsize=9.5)
    ax.margins(x=0.24, y=0.20)
    return r


def fig12(d, sr):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    ys = [sr[a] for a in ARMS]

    ab = [d[a]["text_ablation"]["ratio"] for a in ARMS]
    _scatter_fit(_style(axes[0]), ab, ys,
                 "loss multiplier when instruction tokens are zeroed",
                 "How much the head NEEDS to read the text")

    tm = [100 * d[a]["cross_attention"]["mean_mass_text"] for a in ARMS]
    _scatter_fit(_style(axes[1]), tm, ys,
                 "% of cross-attention on text tokens",
                 "Attention MASS is the weaker predictor")

    ax = _style(axes[2])
    w, xs = 0.36, np.arange(len(ARMS))
    im = [d[a]["adapter_probes"]["image"]["r2_action"] for a in ARMS]
    tx = [d[a]["adapter_probes"]["text"]["r2_action"] for a in ARMS]
    b1 = ax.bar(xs - w / 2, im, w * 0.92, color="#2a78d6", label="image pool")
    b2 = ax.bar(xs + w / 2, tx, w * 0.92, color="#c98a1b", label="text pool")
    for bb in list(b1) + list(b2):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.006,
                f"{bb.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[a].replace(" ", "\n", 1) for a in ARMS], fontsize=7.6)
    ax.set_ylabel("action R² after adapter")
    ax.set_title("Only PaliGemma's TEXT pool carries the action signal;\n"
                 "for the blind pair, text selects the task instead", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.18)

    fig.suptitle("Model behaviour: the more a head must read the instruction itself, "
                 "the better it controls", fontsize=12, y=1.04)
    fig.savefig(FIG / "fig12_routing_four_arms.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig12_routing_four_arms")


def fig13(d):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))

    ax = _style(axes[0])
    w, xs = 0.20, np.arange(len(DIMS))
    for i, a in enumerate(ARMS):
        v = [d[a]["per_dim_error"][x]["nmae"] for x in DIMS]
        ax.bar(xs + (i - 1.5) * w, v, w * 0.9, color=COL[a], label=LAB[a])
    ax.axvspan(2.5, 5.5, color="#c9a227", alpha=0.10, zorder=0)
    ax.text(4.0, ax.get_ylim()[1] * 0.94, "rotation dims", ha="center",
            fontsize=8.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("MAE ÷ that dimension's own std")
    ax.set_title("Rotation costs ~1.6× more than translation for EVERY arm —\n"
                 "a uniform penalty, not the blindness a loss-share argument predicts",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, ncol=2)

    ax = _style(axes[1])
    gt = [d["qwen3vl"]["per_dim_error"][x]["gt_std"] for x in DIMS]
    b = ax.bar(np.arange(len(DIMS)), gt, 0.62, color="#52514e")
    for bb, g in zip(b, gt):
        ax.text(bb.get_x() + bb.get_width() / 2, g + 0.012, f"{g:.3f}",
                ha="center", fontsize=8)
    ax.axvspan(2.5, 5.5, color="#c9a227", alpha=0.10, zorder=0)
    ax.set_xticks(np.arange(len(DIMS)))
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("ground-truth std")
    ax.set_title("Why it was a live concern: the 7 OSC dims differ in scale by 17×,\n"
                 "and we train unweighted MSE on RAW (unnormalised) actions",
                 fontsize=9.5)
    ax.margins(y=0.16)

    fig.suptitle("Per-dimension action error — testing the missing-normalisation concern",
                 fontsize=12, y=1.05)
    fig.savefig(FIG / "fig13_per_dim_error.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig13_per_dim_error")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    d, sr = load()
    fig12(d, sr)
    fig13(d)
    print(f"\nwrote to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
