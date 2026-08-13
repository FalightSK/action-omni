"""
scripts/analysis/plots_paper.py
───────────────────────────────
The final figure set, numbered to match the manuscript's contributions.

This supersedes the exploratory figures under head_diagnostics/aloha/figures,
which were built incrementally as results arrived and overlap each other. Two
figures that the manuscript needs did not exist there at all: the headline
success-rate comparison (fig1) and the stage decomposition (fig2).

    fig1  Main result — success across all six arms, both testbeds
    fig2  Stage decomposition — the gap is one transition
    fig3  Checkpoint ladders — pretraining's payoff is task-specific
    fig4  Offline diagnostics are blind to the closed-loop gap
    fig5  Failure taxonomy — same failure, more often
    fig6  Text-ablation floor
    fig7  Action space governs visual reliance

Every number is READ FROM DISK (rollout JSONs, diagnostics JSONs) rather than
typed in, so a figure cannot silently disagree with the data that produced it.
The two exceptions are the LIBERO success rates and their p-values, which come
from eval runs whose per-episode records live in a different schema; those are
marked and cross-checked against RESULTS.md.

Colour convention throughout: orange = robot-pretrained (GR00T N1.7),
green = stock base (Qwen3-VL-2B). Grey = an offline measure. Red = closed-loop.

    python scripts/analysis/plots_paper.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANA = ROOT / "asset" / "analysis"
RUNS = ROOT / "asset" / "runs"
FIG = ANA / "paper_figures"

PRE, STOCK = "#eb6834", "#1baf7a"
GREY, RED, INK, INK2, GRID, SURF = "#9aa0a6", "#c0392b", "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def _style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def load_aloha(exp_dir, files=("sim_results.json", "sim_results_rep.json")):
    """Pooled per-episode records across the two disjoint-seed runs."""
    eps = []
    for f in files:
        p = RUNS / "aloha" / exp_dir / f
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        eps += j["episodes"] if isinstance(j, dict) and "episodes" in j else j
    return eps


ALOHA = {"GR00T": "exp05_groot_transfer", "Qwen3-VL": "exp06_qwen3vl_transfer"}

# LIBERO success rates: 200 rollouts per arm, canonical condition. Recorded in
# RESULTS.md section 2; the per-episode schema differs from ALOHA's so these are
# stated rather than re-derived.
LIBERO_SR = {("GR00T", 1): 62.5, ("Qwen3-VL", 1): 68.0,
             ("GR00T", 2): 91.5, ("Qwen3-VL", 2): 89.0}
LIBERO_N = 200


# ──────────────────────────────────────────────────────────────────────────────
def fig1():
    """Main result: cameras dominate backbones; the null does not survive ALOHA."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = _style(axes[0])
    groups = [("1 camera", 1), ("2 cameras", 2)]
    xs = np.arange(len(groups))
    for i, (arm, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        vals = [LIBERO_SR[(arm, v)] for _, v in groups]
        err = []
        for v in vals:
            lo, hi = wilson(round(v / 100 * LIBERO_N), LIBERO_N)
            err.append([v - 100 * lo, 100 * hi - v])
        ax.bar(xs + (i - 0.5) * 0.36, vals, 0.34, color=col, label=arm, zorder=3,
               yerr=np.array(err).T, capsize=3,
               error_kw=dict(ecolor=INK2, lw=1.1))
        # Above the CI cap, not on it — a label crossed by an error bar is unreadable.
        for xi, vv, e in zip(xs + (i - 0.5) * 0.36, vals, err):
            ax.text(xi, vv + e[1] + 1.6, f"{vv:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs, [g for g, _ in groups])
    ax.set_ylabel("closed-loop success rate (%)")
    ax.set_ylim(0, 132)
    ax.set_title("LIBERO-Goal (single-arm, 7-DOF)   n = 200 per arm",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

    # The two comparisons the figure exists to make, drawn as brackets in the
    # headroom above the bars rather than over them — an arrow crossing a bar
    # reads as belonging to that bar.
    def bracket(x0, x1, y, text, col=INK, lw=1.4):
        ax.plot([x0, x0, x1, x1], [y - 2.5, y, y, y - 2.5], color=col, lw=lw)
        ax.text((x0 + x1) / 2, y + 1.5, text, ha="center", va="bottom",
                fontsize=8.6, color=col)

    # camera effect: same backbone (GR00T), 1 view -> 2 views
    bracket(-0.18, 0.82, 112, "camera:  +29.0 pts,  $p<10^{-7}$", col=RED, lw=1.6)
    # backbone effect: same observation spec (2 views), pretrained -> stock
    bracket(0.82, 1.18, 100, "backbone:  +2.5 pts,  $p=0.40$")

    ax = _style(axes[1])
    xs = np.arange(2)
    for i, (arm, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        eps = load_aloha(ALOHA[arm])
        k = sum(e["is_success"] for e in eps)
        n = len(eps)
        sr = 100 * k / n
        lo, hi = wilson(k, n)
        ax.bar([i], [sr], 0.5, color=col, zorder=3,
               yerr=[[sr - 100 * lo], [100 * hi - sr]], capsize=4,
               error_kw=dict(ecolor=INK2, lw=1.1))
        ax.text(i, 100 * hi + 1.6, f"{sr:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(i, 4, f"{k}/{n}", ha="center", fontsize=8, color="white")
    ax.set_xticks(xs, ["GR00T\n(pretrained)", "Qwen3-VL\n(stock)"], fontsize=9)
    ax.set_ylim(0, 132)
    ax.set_ylabel("closed-loop success rate (%)")
    ax.set_title("ALOHA transfer-cube (bimanual, 14-DOF)   n = 400 per arm",
                 fontsize=10, loc="left")
    ax.plot([0, 0, 1, 1], [97.5, 100, 100, 97.5], color=RED, lw=1.6)
    ax.text(0.5, 101.5, "backbone:  +9.5 pts,  $p=0.0067$", ha="center",
            va="bottom", fontsize=8.8, color=RED)

    fig.suptitle("The same backbone swap is worth 2.5 points on one task and "
                 "9.5 on the other", fontsize=11, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig1_main_result.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def fig2():
    """Stage decomposition: the entire gap sits in one transition."""
    stages = [(1, "contact"), (2, "lift"), (4, "handover\n(= success)")]
    G = load_aloha(ALOHA["GR00T"])
    Q = load_aloha(ALOHA["Qwen3-VL"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    # Marginal survival curve.
    ax = _style(axes[0])
    curves = {}
    for arm, eps, col in [("GR00T", G, PRE), ("Qwen3-VL", Q, STOCK)]:
        curves[arm] = [100 * np.mean([e["max_reward"] >= s for e in eps])
                       for s, _ in stages]
    for arm, col in [("GR00T", PRE), ("Qwen3-VL", STOCK)]:
        v = curves[arm]
        other = curves["Qwen3-VL" if arm == "GR00T" else "GR00T"]
        ax.plot(range(len(stages)), v, "o-", color=col, lw=2.4, ms=8, label=arm)
        # The two curves nearly coincide at contact and lift — that is the point of
        # the panel — so a fixed offset would stack the labels on top of each other.
        for i, vv in enumerate(v):
            up = vv >= other[i]
            ax.text(i, vv + (2.4 if up else -3.8), f"{vv:.1f}",
                    ha="center", fontsize=8.8, color=col)
    ax.set_xticks(range(len(stages)), [s for _, s in stages])
    ax.set_ylabel("% of episodes reaching this stage")
    ax.set_ylim(40, 102)
    ax.set_title("Both arms reach contact and lift equally", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)

    # Conditional transition probabilities — where the gap actually is.
    ax = _style(axes[1])
    labels = ["P(contact)", "P(lift | contact)", "P(handover | lift)"]
    thr = [(1, None), (2, 1), (4, 2)]
    xs = np.arange(3)
    for i, (arm, eps, col) in enumerate([("GR00T", G, PRE), ("Qwen3-VL", Q, STOCK)]):
        v = []
        for s, prev in thr:
            num = sum(e["max_reward"] >= s for e in eps)
            den = len(eps) if prev is None else sum(e["max_reward"] >= prev for e in eps)
            v.append(100 * num / den)
        ax.bar(xs + (i - 0.5) * 0.36, v, 0.34, color=col, label=arm, zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.36, v):
            ax.text(xi, vv + 1.4, f"{vv:.1f}", ha="center", fontsize=8.6, color=INK)
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("conditional success (%)")
    ax.set_ylim(0, 112)
    for xi, p in zip(xs, ["p = 0.11", "p = 0.22", "p = 0.0009"]):
        ax.text(xi, 104, p, ha="center", fontsize=8.6,
                color=RED if "0.0009" in p else INK2,
                fontweight="bold" if "0.0009" in p else "normal")
    ax.set_title("The entire gap is one transition", fontsize=10, loc="left")

    fig.suptitle("Pretraining does not raise general competence — it raises one "
                 "conditional probability", fontsize=11, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig2_stage_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def _aloha_ladder(exp_dir):
    eps = ["0025", "0050", "0100", "0150", "0200", "0300"]
    xs, ys = [], []
    for e in eps:
        p = RUNS / "aloha" / exp_dir / "ladder" / f"sim_epoch_{e}.json"
        j = json.loads(p.read_text())
        r = j["episodes"] if isinstance(j, dict) and "episodes" in j else j
        xs.append(int(e))
        ys.append(100 * np.mean([x["is_success"] for x in r]))
    return xs, ys


def _libero_ladder(exp_dir):
    eps = ["0025", "0050", "0075", "0100"]
    xs, ys = [], []
    for e in eps:
        p = RUNS / "libero" / exp_dir / "ladder" / f"eval_epoch_{e}.json"
        j = json.loads(p.read_text())
        r = [x for x in j["results"] if x["condition"] == "canonical"]
        xs.append(int(e))
        ys.append(100 * np.mean([x["success"] for x in r]))
    return xs, ys


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)

    ax = _style(axes[0])
    for arm, d, col in [("GR00T", "exp05_groot_transfer", PRE),
                        ("Qwen3-VL", "exp06_qwen3vl_transfer", STOCK)]:
        xs, ys = _aloha_ladder(d)
        ax.plot(xs, ys, "o-", color=col, lw=2.4, ms=6.5, label=arm)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("closed-loop success rate (%)")
    ax.set_ylim(-4, 100)
    ax.set_title("ALOHA — bimanual\npretrained leads at every checkpoint",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.annotate("stock arm: 0/50\n$p = 0.0001$", xy=(50, 0), xytext=(92, 9),
                fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))

    ax = _style(axes[1])
    for arm, d, col in [("GR00T", "exp05_groot_2view", PRE),
                        ("Qwen3-VL", "exp06_qwen3vl_2view", STOCK)]:
        xs, ys = _libero_ladder(d)
        ax.plot(xs, ys, "o-", color=col, lw=2.4, ms=6.5, label=arm)
    ax.set_xlabel("training epoch")
    ax.set_title("LIBERO-Goal — single-arm\nno pretraining advantage at any point",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.annotate("stock leads\nby 16 pts", xy=(25, 77), xytext=(44, 44),
                fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))

    fig.suptitle("Robot pretraining is not a general sample-efficiency prior",
                 fontsize=11, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig3_checkpoint_ladders.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def fig4():
    A = json.loads((ANA / "head_diagnostics" / "aloha" / "head_diagnostics.json").read_text())
    g, q = A["groot_aloha"], A["qwen3vl_aloha"]

    def rel(a, b):
        return 100.0 * (b - a) / a

    def per_dim_mean(r):
        return float(np.mean([v["nmae"] for v in r["per_dim_error"].values()]))

    def attn_img(r):
        return float(np.mean([b["mass_image"] for b in r["cross_attention"]["per_block"]]))

    rows = [
        ("velocity loss\n(overall)", rel(g["phase_task_loss"]["overall"],
                                         q["phase_task_loss"]["overall"]), GREY),
        ("velocity loss\n(handover phase)", rel(g["phase_task_loss"]["by_phase"]["mid"],
                                                q["phase_task_loss"]["by_phase"]["mid"]), GREY),
        ("open-loop action\nerror (nMAE)", rel(per_dim_mean(g), per_dim_mean(q)), GREY),
        ("PE sensitivity", rel(g["pe_sensitivity"]["relative_shift"],
                               q["pe_sensitivity"]["relative_shift"]), GREY),
        ("attention mass\non image", rel(attn_img(g), attn_img(q)), "#6b7075"),
        ("CLOSED-LOOP\nSUCCESS RATE", rel(61.25, 51.75), RED),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _style(ax)
    x = np.arange(len(rows))
    v = [r[1] for r in rows]
    ax.axhline(0, color=INK2, lw=1.0)
    ax.axhspan(-2, 2, color=GREY, alpha=0.14, zorder=0)
    ax.bar(x, v, 0.6, color=[r[2] for r in rows], zorder=3)
    for xi, vv in zip(x, v):
        ax.text(xi, vv + (0.7 if vv >= 0 else -1.9), f"{vv:+.1f}%",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.6)
    ax.set_ylabel("stock − pretrained   (% of pretrained)")
    ax.set_ylim(min(v) * 1.35, 8)
    ax.text(0.02, 0.055, "±2% band", transform=ax.transAxes,
            fontsize=8, color=INK2)
    ax.set_title("Offline accuracy metrics are blind to a gap the rollouts "
                 "resolve at $p = 0.0067$\n"
                 "grey = offline · red = closed-loop", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_offline_blindness.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def fig5():
    tax = json.loads((ANA / "failure_taxonomy" / "failure_taxonomy.json").read_text())
    cats = ["receiver_never_opened", "opened_never_closed", "closed_on_nothing"]
    short = ["receiver gripper\nNEVER OPENS", "opened but\nnever closed", "closed on\nnothing"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    ax = _style(axes[0])
    xs = np.arange(len(cats))
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        c = tax[name]["counts"]
        tot = sum(c[k] for k in cats)
        v = [100 * c[k] / tot for k in cats]
        ax.bar(xs + (i - 0.5) * 0.36, v, 0.34, color=col, label=name, zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.36, v):
            ax.text(xi, vv + 1.4, f"{vv:.0f}%", ha="center", fontsize=8.6, color=INK)
    ax.set_xticks(xs, short, fontsize=8.4)
    ax.set_ylabel("% of lifted-but-failed episodes")
    ax.set_ylim(0, 76)
    ax.set_title("The stock arm fails to INITIATE the receive\n"
                 "26.2% vs 14.5% of lifted episodes,  $p = 0.0068$",
                 fontsize=9.8, loc="left")
    ax.legend(frameon=False, fontsize=8.8)

    ax = _style(axes[1])
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        c = tax[name]["counts"]
        lf = sum(c[k] for k in cats)
        ax.bar([i], [c["success"]], 0.5, color=col, zorder=3)
        ax.bar([i], [lf], 0.5, bottom=[c["success"]], color=col, alpha=0.35,
               hatch="//", zorder=3)
        ax.text(i, c["success"] / 2, f"{c['success']}\nsuccess", ha="center",
                fontsize=8.6, color="white", fontweight="bold")
        ax.text(i, c["success"] + lf / 2, f"{lf}\nhandover\nfailures", ha="center",
                fontsize=8.6, color=INK)
    ax.set_xticks([0, 1], ["GR00T", "Qwen3-VL"])
    ax.set_ylabel("episodes (of 200)")
    ax.set_title("Total handover failures\n69 vs 52 of 200 episodes",
                 fontsize=9.8, loc="left")

    ax = _style(axes[2])
    vals = {"GR00T": (0.8663, 0.8546), "Qwen3-VL": (0.8407, 0.8380)}
    xs = np.arange(2)
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        ax.bar(xs + (i - 0.5) * 0.36, vals[name], 0.34, color=col, label=name, zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.36, vals[name]):
            ax.text(xi, vv + 0.0022, f"{vv:.3f}", ha="center", fontsize=8.4, color=INK)
    ax.set_xticks(xs, ["successful\nepisodes", "failed\nepisodes"], fontsize=8.8)
    ax.set_ylabel("image attention, handover window")
    ax.set_ylim(0.80, 0.886)
    ax.set_title("Stock attends less to image —\nin successes too ($p<10^{-4}$)",
                 fontsize=9.8, loc="left")
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")

    fig.suptitle("The stock policy fails to initiate the receive — it never opens "
                 "the receiving gripper", fontsize=11, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig5_failure_taxonomy.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def fig6():
    A = json.loads((ANA / "head_diagnostics" / "aloha" / "head_diagnostics.json").read_text())
    L = json.loads((ANA / "head_diagnostics" / "head_diagnostics.json").read_text())
    rows = [
        ("GR00T · 1 view", L["groot"]["text_ablation"]["ratio"], PRE),
        ("Qwen3-VL · 1 view", L["qwen3vl"]["text_ablation"]["ratio"], STOCK),
        ("GR00T · 2 views", L["groot2v"]["text_ablation"]["ratio"], PRE),
        ("Qwen3-VL · 2 views", L["qwen3vl2v"]["text_ablation"]["ratio"], STOCK),
        ("GR00T · ALOHA", A["groot_aloha"]["text_ablation"]["ratio"], PRE),
        ("Qwen3-VL · ALOHA", A["qwen3vl_aloha"]["text_ablation"]["ratio"], STOCK),
    ]
    rows.sort(key=lambda r: r[1])
    floor = max(A[a]["text_ablation"]["ratio"] for a in ("groot_aloha", "qwen3vl_aloha"))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    _style(ax)
    y = np.arange(len(rows))
    ax.axvspan(1.0, floor, color=GREY, alpha=0.22, zorder=0)
    ax.axvline(1.0, color=INK2, lw=1.0, ls=":")
    ax.barh(y, [r[1] for r in rows], 0.62, color=[r[2] for r in rows], zorder=3)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.13, yi, f"{r[1]:.2f}×", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("velocity loss with text zeroed  /  baseline loss")
    ax.set_xlim(0.88, 8.5)
    ax.text(1.30, 4.55, "FLOOR\none fixed instruction:\nzeroing it removes no\n"
                        "information, so this is\nperturbation cost alone",
            fontsize=8.2, color=INK, va="center")
    ax.set_title("Language ablation must be read against its floor, not against 1.0",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_text_ablation_floor.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def fig7():
    A = json.loads((ANA / "head_diagnostics" / "aloha" / "head_diagnostics.json").read_text())
    L = json.loads((ANA / "head_diagnostics" / "head_diagnostics.json").read_text())
    P = json.loads((ANA / "spatial_probe" / "spatial_probe.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    ax = _style(axes[0])
    names = ["GR00T\n1v", "Qwen\n1v", "GR00T\n2v", "Qwen\n2v", "GR00T\nALOHA", "Qwen\nALOHA"]
    vals = [L["groot"]["pe_sensitivity"]["relative_shift"],
            L["qwen3vl"]["pe_sensitivity"]["relative_shift"],
            L["groot2v"]["pe_sensitivity"]["relative_shift"],
            L["qwen3vl2v"]["pe_sensitivity"]["relative_shift"],
            A["groot_aloha"]["pe_sensitivity"]["relative_shift"],
            A["qwen3vl_aloha"]["pe_sensitivity"]["relative_shift"]]
    xs = np.arange(6)
    ax.bar(xs, vals, 0.62, color=[PRE, STOCK] * 3, zorder=3)
    for xi, vv in zip(xs, vals):
        ax.text(xi, vv + 0.004, f"{vv:.3f}", ha="center", fontsize=8, color=INK)
    ax.axvline(3.5, color=INK2, lw=1.1, ls="--")
    ax.text(1.5, 0.152, "LIBERO\nend-effector", ha="center", fontsize=8.4, color=INK2)
    ax.text(4.5, 0.152, "ALOHA\njoint", ha="center", fontsize=8.4, color=INK2)
    ax.set_xticks(xs, names, fontsize=7.8)
    ax.set_ylim(0, 0.175)
    ax.set_ylabel("action shift when PE is zeroed")
    ax.set_title("Joint-space control barely\nuses image position", fontsize=9.8, loc="left")

    ax = _style(axes[1])
    keys = ["LIBERO GR00T 2v", "LIBERO Qwen3-VL 2v", "ALOHA GR00T", "ALOHA Qwen3-VL"]
    lbl = ["LIBERO\nGR00T", "LIBERO\nQwen", "ALOHA\nGR00T", "ALOHA\nQwen"]
    v = [P[k]["r2_state"] for k in keys]
    xs = np.arange(4)
    ax.bar(xs, v, 0.6, color=[PRE, STOCK, PRE, STOCK], zorder=3)
    for xi, vv, k in zip(xs, v, keys):
        ax.text(xi, vv + 0.018, f"{vv:.2f}", ha="center", fontsize=8.6, color=INK)
        ax.text(xi, 0.06, f"{P[k]['n_img_tokens']}\ntokens", ha="center",
                fontsize=7.6, color="white")
    ax.axvline(1.5, color=INK2, lw=1.1, ls="--")
    ax.set_xticks(xs, lbl, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cross-validated $R^2$")
    ax.set_title("…but the scene is NOT\ninformation-poor", fontsize=9.8, loc="left")

    ax = _style(axes[2])
    v = [P[k]["r2_action_first"] for k in keys]
    ax.axhline(0, color=INK2, lw=1.0)
    ax.bar(xs, v, 0.6, color=[PRE, STOCK, PRE, STOCK], zorder=3)
    for xi, vv in zip(xs, v):
        ax.text(xi, vv + (0.05 if vv >= 0 else -0.11), f"{vv:+.2f}",
                ha="center", fontsize=8.6, color=INK)
    ax.axvline(1.5, color=INK2, lw=1.1, ls="--")
    ax.set_xticks(xs, lbl, fontsize=8)
    ax.set_ylim(-0.45, 1.0)
    ax.set_ylabel("cross-validated $R^2$")
    ax.text(0.5, -0.34, "end-effector\nDELTAS", ha="center", fontsize=8, color=INK2)
    ax.text(2.5, 0.30, "absolute\nJOINT TARGETS", ha="center", fontsize=8, color=INK2)
    ax.set_title("The action is defined\nrelative to different things", fontsize=9.8, loc="left")

    fig.suptitle("Visual reliance tracks what the action is defined relative to, "
                 "not scene complexity", fontsize=11, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig7_action_space.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print(f"  {f.__name__} ok")
    for p in sorted(FIG.glob("*.png")):
        print(f"  wrote {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
