"""
scripts/analysis/plots_head_diagnostics_aloha.py
────────────────────────────────────────────────
Figures for the ALOHA head diagnostics, plus the cross-dataset panels that only
become possible once the same battery has been run on both datasets.

What these are for
──────────────────
The ALOHA rollouts give GR00T 61.25% vs Qwen3-VL 51.75% (n=400 each, +9.5 pts,
p=0.0067), with the whole gap localised to P(handover|lift) = 71.6% vs 59.7%
(p=0.0009). The head diagnostics then find the two heads indistinguishable on
every offline measure. These plots exist to show that this is not a weak or
noisy comparison — the offline metrics are not merely poorly correlated with
closed-loop success, they are flat.

Two figures also reinterpret the LIBERO results:

figA1  Text-ablation ratio across all 8 arms with the ALOHA FLOOR marked.
       ALOHA has ONE fixed instruction, so zeroing its text tokens removes zero
       task information; whatever loss increase remains (1.14x, 1.24x) is the
       cost of an off-distribution perturbation alone. That is the control
       condition the LIBERO ablation never had, and it is what licenses the
       sharpest single claim in the study: Pi-0.5's 1.023x sits BELOW the floor,
       so it ignores its instruction more completely than a model whose
       instruction is a constant.

figA2  Every offline diagnostic as a between-arm percentage difference, against
       the closed-loop difference on the same axis. This is the dissociation.

Colour convention follows plots_head_diagnostics.py: orange = robot-pretrained,
green = stock base, so a reader moving between the LIBERO and ALOHA figures
keeps the same mapping.
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
sys.path.insert(0, str(ROOT))

DIAG_ALOHA = ROOT / "asset" / "analysis" / "head_diagnostics" / "aloha" / "head_diagnostics.json"
DIAG_LIBERO = ROOT / "asset" / "analysis" / "head_diagnostics" / "head_diagnostics.json"
FIG = ROOT / "asset" / "analysis" / "head_diagnostics" / "aloha" / "figures"

PRE, STOCK = "#eb6834", "#1baf7a"          # pretrained / stock, as in the LIBERO figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

# Closed-loop numbers, measured today. n=400 per ALOHA arm (two 200-episode runs
# on disjoint seed ranges, pooled); n=200 per LIBERO arm.
SR = {"groot_aloha": 61.25, "qwen3vl_aloha": 51.75}
ARMS = ["groot_aloha", "qwen3vl_aloha"]
NAME = {"groot_aloha": "GR00T N1.7\n(robot-pretrained)",
        "qwen3vl_aloha": "Qwen3-VL-2B\n(stock base)"}
SHORT = {"groot_aloha": "GR00T", "qwen3vl_aloha": "Qwen3-VL"}
COL = {"groot_aloha": PRE, "qwen3vl_aloha": STOCK}


def _style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# figA1 — text ablation across all 8 arms, against the ALOHA floor
# ──────────────────────────────────────────────────────────────────────────────

def figA1(A, L):
    rows = [
        ("GR00T 1-view",     "LIBERO", L["groot"]["text_ablation"]["ratio"],       PRE),
        ("Qwen3-VL 1-view",  "LIBERO", L["qwen3vl"]["text_ablation"]["ratio"],     STOCK),
        ("GR00T 2-view",     "LIBERO", L["groot2v"]["text_ablation"]["ratio"],     PRE),
        ("Qwen3-VL 2-view",  "LIBERO", L["qwen3vl2v"]["text_ablation"]["ratio"],   STOCK),
        ("GR00T",            "ALOHA",  A["groot_aloha"]["text_ablation"]["ratio"], PRE),
        ("Qwen3-VL",         "ALOHA",  A["qwen3vl_aloha"]["text_ablation"]["ratio"], STOCK),
    ]
    rows.sort(key=lambda r: r[2])
    labels = [f"{n}  ({d})" for n, d, _, _ in rows]
    vals = [r[2] for r in rows]
    cols = [r[3] for r in rows]

    floor_lo = min(A[a]["text_ablation"]["ratio"] for a in ARMS)
    floor_hi = max(A[a]["text_ablation"]["ratio"] for a in ARMS)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    _style(ax)
    y = np.arange(len(rows))
    ax.axvspan(1.0, floor_hi, color="#9aa0a6", alpha=0.18, zorder=0)
    ax.axvline(1.0, color=INK2, lw=1.0, ls=":")
    ax.barh(y, vals, 0.66, color=cols, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.12, yi, f"{v:.2f}x", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("velocity loss with text tokens zeroed  /  baseline loss")
    ax.set_xlim(0.9, max(vals) * 1.14)
    ax.set_title("Text ablation against its own floor\n"
                 "grey band = cost of the perturbation alone, measured on ALOHA's "
                 "single fixed instruction", fontsize=10, loc="left")
    ax.annotate("ALOHA has ONE instruction, so zeroing it\n"
                "removes no information. Whatever remains\n"
                "is the metric's floor, not language use.",
                xy=(floor_hi, 0.35), xytext=(2.5, 0.9), fontsize=8.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    fig.tight_layout()
    fig.savefig(FIG / "figA1_text_ablation_floor.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA2 — the dissociation: offline flat, closed-loop not
# ──────────────────────────────────────────────────────────────────────────────

def _rel(a, b):
    """Signed % difference (stock - pretrained) / pretrained."""
    return 100.0 * (b - a) / a


def figA2(A):
    g, q = A["groot_aloha"], A["qwen3vl_aloha"]

    def attn_img(r):
        return float(np.mean([b["mass_image"] for b in r["cross_attention"]["per_block"]]))

    def per_dim_mean(r):
        return float(np.mean([v["nmae"] for v in r["per_dim_error"].values()]))

    metrics = [
        ("velocity loss\n(overall)",      g["phase_task_loss"]["overall"],
                                          q["phase_task_loss"]["overall"]),
        ("velocity loss\n(mid phase)",    g["phase_task_loss"]["by_phase"]["mid"],
                                          q["phase_task_loss"]["by_phase"]["mid"]),
        ("open-loop action\nerror (nMAE)", per_dim_mean(g), per_dim_mean(q)),
        ("attention mass\non image",      attn_img(g), attn_img(q)),
        ("PE sensitivity",                g["pe_sensitivity"]["relative_shift"],
                                          q["pe_sensitivity"]["relative_shift"]),
    ]
    names = [m[0] for m in metrics]
    diffs = [_rel(m[1], m[2]) for m in metrics]
    # Closed-loop, expressed the same way so it shares the axis.
    sr_diff = _rel(SR["groot_aloha"], SR["qwen3vl_aloha"])

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    _style(ax)
    x = np.arange(len(names) + 1)
    vals = diffs + [sr_diff]
    cols = ["#9aa0a6"] * len(diffs) + ["#c0392b"]
    ax.axhline(0, color=INK2, lw=1.0)
    ax.bar(x, vals, 0.62, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.6 if v >= 0 else -1.6), f"{v:+.1f}%",
                ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, names + ["closed-loop\nSUCCESS RATE"], fontsize=8.2)
    ax.set_ylabel("stock minus pretrained  (% of pretrained)")
    ax.set_ylim(min(vals) * 1.35, max(max(vals) * 1.4, 6))
    ax.set_title("Offline diagnostics cannot see a gap the rollouts resolve at p = 0.0067\n"
                 "grey = offline (accuracy measures within +/-2%)   "
                 "red = closed-loop (-15.5%)",
                 fontsize=10, loc="left")
    # Attention mass is the one offline quantity that moves (-5.2%): the two heads
    # allocate differently even though they predict equally well. Called out rather
    # than averaged away, because it is the only surviving mechanistic candidate.
    ax.annotate("only offline metric\nthat moves", xy=(3, diffs[3]),
                xytext=(2.35, -11.5), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    fig.tight_layout()
    fig.savefig(FIG / "figA2_dissociation.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA3 — cross-attention per DiT block, and PE sensitivity vs LIBERO
# ──────────────────────────────────────────────────────────────────────────────

def figA3(A, L):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = _style(axes[0])
    for a in ARMS:
        pb = A[a]["cross_attention"]["per_block"]
        ax.plot([b["block"] for b in pb], [100 * b["mass_image"] for b in pb],
                "o-", color=COL[a], lw=2, ms=5, label=SHORT[a])
    ax.set_xlabel("DiT block"); ax.set_ylabel("attention mass on image tokens (%)")
    ax.set_title("Where the head looks\n(54 image / 42 text tokens)", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    ax = _style(axes[1])
    for a in ARMS:
        pb = A[a]["cross_attention"]["per_block"]
        ax.plot([b["block"] for b in pb], [b["entropy_norm"] for b in pb],
                "o-", color=COL[a], lw=2, ms=5, label=SHORT[a])
    ax.axhline(1.0, color=INK2, lw=1.0, ls=":")
    ax.text(0.05, 0.995, "uniform", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("DiT block"); ax.set_ylabel("normalised attention entropy")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Attention is near-uniform\n(no sharp token selection)", fontsize=9.5, loc="left")

    ax = _style(axes[2])
    lib = {"GR00T 1v": "groot", "Qwen 1v": "qwen3vl",
           "GR00T 2v": "groot2v", "Qwen 2v": "qwen3vl2v"}
    names = list(lib) + ["GR00T\nALOHA", "Qwen\nALOHA"]
    vals = [L[k]["pe_sensitivity"]["relative_shift"] for k in lib.values()] + \
           [A[a]["pe_sensitivity"]["relative_shift"] for a in ARMS]
    cols = [PRE, STOCK, PRE, STOCK, PRE, STOCK]
    xs = np.arange(len(names))
    ax.bar(xs, vals, 0.64, color=cols, zorder=3)
    ax.axvline(3.5, color=INK2, lw=1.0, ls="--")
    ax.text(1.5, max(vals) * 0.99, "LIBERO (end-effector)", fontsize=8, ha="center", color=INK2)
    ax.text(4.5, max(vals) * 0.99, "ALOHA\n(joint)", fontsize=8, ha="center", color=INK2)
    ax.set_xticks(xs, names, fontsize=7.4, rotation=35, ha="right")
    ax.set_ylabel("relative action shift when PE is zeroed")
    ax.set_title("Joint-space control needs\nspatial position far less", fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(FIG / "figA3_routing.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA4 — phase loss and per-dimension error
# ──────────────────────────────────────────────────────────────────────────────

def figA4(A):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))

    ax = _style(axes[0])
    phases = ["early", "mid", "late"]
    xs = np.arange(len(phases))
    for i, a in enumerate(ARMS):
        v = [A[a]["phase_task_loss"]["by_phase"][p] for p in phases]
        ax.bar(xs + (i - 0.5) * 0.34, v, 0.32, color=COL[a], label=SHORT[a], zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.34, v):
            ax.text(xi, vv + 0.0004, f"{vv:.4f}", ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(xs, ["early", "mid\n(contains handover)", "late"])
    ax.set_ylabel("velocity loss")
    ax.set_title("Loss by episode phase — the handover window\n"
                 "is hardest for BOTH arms, and equally so", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    ax = _style(axes[1])
    keys = list(A["groot_aloha"]["per_dim_error"])
    xs = np.arange(len(keys))
    for i, a in enumerate(ARMS):
        v = [A[a]["per_dim_error"][k]["nmae"] for k in keys]
        ax.bar(xs + (i - 0.5) * 0.38, v, 0.36, color=COL[a], label=SHORT[a], zorder=3)
    ax.set_xticks(xs, keys, fontsize=7, rotation=60, ha="right")
    ax.set_ylabel("normalised MAE (error / dim std)")
    ax.set_title("Per-dimension open-loop error, 14 DOF\n"
                 "no arm-specific deficit survives", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG / "figA4_phase_perdim.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA5 — closed-loop checkpoint ladder: does pretraining buy speed or skill?
# ──────────────────────────────────────────────────────────────────────────────

LADDER_EPOCHS = ["0025", "0050", "0100", "0150", "0200", "0300"]
LADDER_DIR = {"groot_aloha": "exp05_groot_transfer",
              "qwen3vl_aloha": "exp06_qwen3vl_transfer"}


def _ladder(arm):
    out = []
    for e in LADDER_EPOCHS:
        p = (ROOT / "asset" / "runs" / "aloha" / LADDER_DIR[arm] / "ladder"
             / f"sim_epoch_{e}.json")
        j = json.loads(p.read_text())
        eps = j["episodes"] if isinstance(j, dict) and "episodes" in j else j
        out.append(100.0 * np.mean([bool(x["is_success"]) for x in eps]))
    return [int(e) for e in LADDER_EPOCHS], out


def figA5():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    ax = _style(axes[0])
    curves = {}
    for a in ARMS:
        xs, ys = _ladder(a)
        curves[a] = (xs, ys)
        ax.plot(xs, ys, "o-", color=COL[a], lw=2.2, ms=6, label=SHORT[a])
    # n=400 anchors: the ladder's 50 paired scenes are a fixed, and evidently
    # slightly unrepresentative, sample. Plotting the anchors keeps the reader
    # from reading the ladder's ~+24 pt gap as the population gap (+9.5).
    for a in ARMS:
        ax.axhline(SR[a], color=COL[a], lw=1.0, ls="--", alpha=0.55)
        ax.text(305, SR[a], f" n=400: {SR[a]:.1f}%", fontsize=7.6,
                color=COL[a], va="center")
    ax.set_xlabel("training epoch"); ax.set_ylabel("closed-loop success rate (%)")
    ax.set_xlim(0, 355); ax.set_ylim(-3, 80)
    ax.set_title("Pretrained arm leads at EVERY checkpoint\n"
                 "stock arm is at 0/50 through epoch 50", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = _style(axes[1])
    targets = [20, 30, 40, 46]

    def reach(ys, xs, t):
        for i in range(1, len(xs)):
            if ys[i] >= t > ys[i - 1]:
                f = (t - ys[i - 1]) / (ys[i] - ys[i - 1])
                return xs[i - 1] + f * (xs[i] - xs[i - 1])
        return None

    xs_t = np.arange(len(targets))
    for i, a in enumerate(ARMS):
        xs, ys = curves[a]
        v = [reach(ys, xs, t) or np.nan for t in targets]
        ax.bar(xs_t + (i - 0.5) * 0.36, v, 0.34, color=COL[a], label=SHORT[a], zorder=3)
    for j, t in enumerate(targets):
        g = reach(*curves["groot_aloha"][::-1], t)
        q = reach(*curves["qwen3vl_aloha"][::-1], t)
        if g and q:
            ax.text(j, max(g, q) + 8, f"{q/g:.1f}x", ha="center",
                    fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(xs_t, [f"{t}%" for t in targets])
    ax.set_xlabel("success-rate target"); ax.set_ylabel("epochs required")
    ax.set_ylim(0, 245)
    ax.set_title("Stock backbone needs ~2x the training\n"
                 "to reach the same success rate", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "figA5_ladder.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA6 — the two ladders side by side: pretraining's payoff is task-specific
# ──────────────────────────────────────────────────────────────────────────────

LIBERO_LADDER = {"groot": ("exp05_groot_2view", ["0025", "0050", "0075", "0100"]),
                 "qwen":  ("exp06_qwen3vl_2view", ["0025", "0050", "0075", "0100"])}


def _libero_ladder(key):
    d, eps = LIBERO_LADDER[key]
    xs, ys = [], []
    for e in eps:
        p = ROOT / "asset" / "runs" / "libero" / d / "ladder" / f"eval_epoch_{e}.json"
        j = json.loads(p.read_text())
        rs = [x for x in j["results"] if x["condition"] == "canonical"]
        xs.append(int(e))
        ys.append(100.0 * np.mean([bool(x["success"]) for x in rs]))
    return xs, ys


def figA6():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)

    ax = _style(axes[0])
    for a in ARMS:
        xs, ys = _ladder(a)
        ax.plot(xs, ys, "o-", color=COL[a], lw=2.2, ms=6, label=SHORT[a])
    ax.set_xlabel("training epoch"); ax.set_ylabel("closed-loop success rate (%)")
    ax.set_ylim(-3, 100)
    ax.set_title("ALOHA — bimanual, 14-DOF\n"
                 "pretrained leads at EVERY checkpoint", fontsize=9.8, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.annotate("stock arm: 0/50", xy=(50, 0), xytext=(88, 11), fontsize=8.2,
                color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    ax = _style(axes[1])
    for key, col, lbl in [("groot", PRE, "GR00T"), ("qwen", STOCK, "Qwen3-VL")]:
        xs, ys = _libero_ladder(key)
        ax.plot(xs, ys, "o-", color=col, lw=2.2, ms=6, label=lbl)
    ax.set_xlabel("training epoch")
    ax.set_title("LIBERO-Goal — single-arm, 2 cameras\n"
                 "STOCK leads at epoch 25, and at 100", fontsize=9.8, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.annotate("no pretraining advantage\nat any checkpoint",
                xy=(25, 77), xytext=(42, 40), fontsize=8.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    fig.suptitle("Robot pretraining's payoff is task-specific, not a general "
                 "sample-efficiency prior", fontsize=10.5, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "figA6_ladder_both.png", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# figA7 — failure taxonomy + time-indexed attention
# ──────────────────────────────────────────────────────────────────────────────

def figA7():
    tax = json.loads((ROOT / "asset" / "analysis" / "failure_taxonomy"
                      / "failure_taxonomy.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # Panel 1 — failure MIX. The point is that the two distributions coincide.
    ax = _style(axes[0])
    cats = ["premature_receiver_close", "receiver_never_engaged",
            "grasp_lost_after_lift"]
    short = ["premature\nreceiver close", "receiver\nnever engaged", "grasp lost\nafter lift"]
    xs = np.arange(len(cats))
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        c = tax[name]["counts"]
        tot = sum(c[k] for k in cats)
        v = [100.0 * c[k] / tot for k in cats]
        ax.bar(xs + (i - 0.5) * 0.36, v, 0.34, color=col, label=name, zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.36, v):
            ax.text(xi, vv + 1.2, f"{vv:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, short, fontsize=8)
    ax.set_ylabel("% of lifted-but-failed episodes")
    ax.set_ylim(0, 78)
    ax.set_title("Failure MIX is identical\n(premature close: 59.6% vs 59.4%, p = 0.98)",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    # Panel 2 — failure COUNT. Same mix, more of it.
    ax = _style(axes[1])
    xs = np.arange(2)
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        c = tax[name]["counts"]
        lifted_fail = sum(c[k] for k in cats)
        ax.bar([i], [c["success"]], 0.5, color=col, zorder=3)
        ax.bar([i], [lifted_fail], 0.5, bottom=[c["success"]],
               color=col, alpha=0.38, zorder=3, hatch="//")
        ax.text(i, c["success"] / 2, f"{c['success']}\nsuccess", ha="center",
                fontsize=8.5, color="white", fontweight="bold")
        ax.text(i, c["success"] + lifted_fail / 2, f"{lifted_fail}\nhandover\nfailures",
                ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs, ["GR00T", "Qwen3-VL"])
    ax.set_ylabel("episodes (of 200)")
    ax.set_title("Failure COUNT differs\nsame kind of failure, 33% more of it",
                 fontsize=9.5, loc="left")

    # Panel 3 — attention in the handover window, split by outcome so the
    # comparison cannot be an artifact of failed episodes running longer.
    ax = _style(axes[2])
    vals = {"GR00T": (0.8663, 0.8546), "Qwen3-VL": (0.8407, 0.8380)}
    xs = np.arange(2)
    for i, (name, col) in enumerate([("GR00T", PRE), ("Qwen3-VL", STOCK)]):
        ax.bar(xs + (i - 0.5) * 0.36, vals[name], 0.34, color=col, label=name, zorder=3)
        for xi, vv in zip(xs + (i - 0.5) * 0.36, vals[name]):
            ax.text(xi, vv + 0.002, f"{vv:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, ["successful\nepisodes", "failed\nepisodes"])
    ax.set_ylabel("image attention mass, handover window")
    ax.set_ylim(0.80, 0.885)
    ax.set_title("Stock arm attends less to image —\nin successes too (p < 10⁻⁴ both)",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "figA7_failure_taxonomy.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    A = json.loads(DIAG_ALOHA.read_text())
    L = json.loads(DIAG_LIBERO.read_text())
    figA1(A, L); figA2(A); figA3(A, L); figA4(A); figA5(); figA6(); figA7()
    for p in sorted(FIG.glob("*.png")):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
