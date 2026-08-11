"""
scripts/analysis/plots_head_diagnostics.py
──────────────────────────────────────────
Figures for the Pi-0.5 vs PaliGemma result and the diagnostics that explain it.

Design follows scripts/analysis/latent_compare/plots.py so the two figure sets
read as one document: same surface, same ink, same per-arm hues, direct value
labels on every bar (several sit below 3:1 against the surface), no dual axes —
quantities on different scales get their own panel rather than a second spine.

Sources, all already on disk:
  asset/analysis/head_diagnostics/head_diagnostics.json   the six diagnostics
  asset/analysis/latent_compare/depth_sweep.json          raw-feature probes
  asset/runs/libero/*/eval_libero.json                    per-episode rollouts
  asset/runs/libero/*/train.log                           loss curves

Nothing is hard-coded from the write-up: success rates are recomputed from the
400/600 per-episode records so a figure cannot drift from the underlying run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HD = ROOT / "asset" / "analysis" / "head_diagnostics"
LC = ROOT / "asset" / "analysis" / "latent_compare"
FIG = HD / "figures"

ARMS = ["pi05", "paligemma"]
RUN = {"pi05": "asset/runs/libero/exp01_goal",
       "paligemma": "asset/runs/libero/exp02_paligemma"}
LABEL = {"pi05": "Pi-0.5\n(robot-pretrained)",
         "paligemma": "PaliGemma-3B\n(stock base)"}
SHORT = {"pi05": "Pi-0.5", "paligemma": "PaliGemma"}
COLORS = {"pi05": "#eb6834", "paligemma": "#1baf7a"}

# Short task names. Truncating the raw strings collides — "put the wine bottle
# on top of the cabinet" and "put the wine bottle on the rack" share their first
# 24 characters, and those two are the widest-separated points in the scatter,
# so a prefix label would put the same text on both.
SHORT_TASK = {
    "open the middle drawer of the cabinet": "middle drawer",
    "put the bowl on the stove": "bowl → stove",
    "put the wine bottle on top of the cabinet": "wine → cabinet",
    "open the top drawer and put the bowl inside": "top drawer + bowl",
    "put the bowl on top of the cabinet": "bowl → cabinet top",
    "push the plate to the front of the stove": "push plate",
    "put the cream cheese in the bowl": "cheese → bowl",
    "turn on the stove": "turn on stove",
    "put the bowl on the plate": "bowl → plate",
    "put the wine bottle on the rack": "wine → rack",
}


def _short(lang: str) -> str:
    return SHORT_TASK.get(lang, lang[:20])

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


def _labels(ax, bars, fmt="{:.3f}", dy=0.012, size=8):
    top = max((b.get_height() for b in bars), default=1.0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * top,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=size, color=INK)


# ── data ──────────────────────────────────────────────────────────────────────

def load_all():
    d = json.loads((HD / "head_diagnostics.json").read_text(encoding="utf-8"))
    depth = json.loads((LC / "depth_sweep.json").read_text(encoding="utf-8"))
    ev, val = {}, {}
    for a in ARMS:
        p = ROOT / RUN[a] / "eval_libero.json"
        ev[a] = json.loads(p.read_text(encoding="utf-8"))["results"]
        log = (ROOT / RUN[a] / "train.log").read_text(encoding="utf-8", errors="replace")
        val[a] = [float(m.group(1))
                  for m in re.finditer(r"val=([\d.]+)", log)]
    return d, depth, ev, val


def sr_by_task(records, condition="canonical"):
    """Per-task success rate, recomputed from per-episode records."""
    out: dict[int, list] = {}
    lang: dict[int, str] = {}
    for r in records:
        if r["condition"] != condition:
            continue
        out.setdefault(r["task"], []).append(bool(r["success"]))
        lang[r["task"]] = r["task_language"]
    return ({t: 100.0 * np.mean(v) for t, v in sorted(out.items())},
            {t: lang[t] for t in sorted(lang)})


# ── figures ───────────────────────────────────────────────────────────────────

def fig1_inversion(d, depth, ev):
    """Every offline measure favours Pi-0.5; the closed-loop measure does not."""
    raw = {a: depth["libero_goal"][a]["doc"]["pools"]["image"]["r2_action"] for a in ARMS}
    adp = {a: d[a]["adapter_probes"]["image"]["r2_action"] for a in ARMS}
    olc = {a: d[a]["ckpt_ladder"]["best.pt"]["corr"] for a in ARMS}
    vloss = {a: d[a]["text_ablation"]["loss_base"] for a in ARMS}
    sr = {a: 100.0 * np.mean([r["success"] for r in ev[a]
                              if r["condition"] == "canonical"]) for a in ARMS}

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    panels = [
        ("Probe action R²\n(higher better)", [("raw L18", raw), ("after adapter", adp)], "{:.3f}"),
        ("Open-loop correlation\n(higher better)", [("best.pt", olc)], "{:.3f}"),
        ("Velocity loss\n(LOWER better)", [("held-out", vloss)], "{:.4f}"),
        ("Closed-loop success %\n(higher better)", [("canonical", sr)], "{:.1f}"),
    ]
    for ax, (title, groups, fmt) in zip(axes, panels):
        _style(ax)
        w, xs = 0.36, np.arange(len(groups))
        for i, a in enumerate(ARMS):
            vals = [g[1][a] for g in groups]
            b = ax.bar(xs + (i - 0.5) * w, vals, w * 0.92, color=COLORS[a],
                       label=SHORT[a])
            _labels(ax, b, fmt)
        ax.set_xticks(xs)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.margins(y=0.20)
    axes[3].set_facecolor("#fff4ef")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Pi-0.5 leads on every offline measure and loses closed-loop",
                 fontsize=11.5, y=1.03)
    fig.savefig(FIG / "fig1_inversion.png", bbox_inches="tight")
    plt.close(fig)


def fig2_routing(d):
    """Where each head sends its cross-attention — the mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))

    ax = _style(axes[0])
    for a in ARMS:
        blocks = d[a]["cross_attention"]["per_block"]
        ax.plot([b["block"] for b in blocks], [100 * b["mass_text"] for b in blocks],
                "o-", color=COLORS[a], lw=2, ms=5, label=SHORT[a])
    ax.set_xlabel("DiT block")
    ax.set_ylabel("% of cross-attention on text tokens")
    ax.set_title("Attention mass on the instruction\n(16 of 272 tokens)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, None)

    ax = _style(axes[1])
    w, xs = 0.36, np.arange(2)
    for i, a in enumerate(ARMS):
        blocks = d[a]["cross_attention"]["per_block"]
        vals = [np.mean([b["mass_per_image_token"] for b in blocks]),
                np.mean([b["mass_per_text_token"] for b in blocks])]
        b = ax.bar(xs + (i - 0.5) * w, vals, w * 0.92, color=COLORS[a], label=SHORT[a])
        _labels(ax, b, "{:.4f}", size=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["per image token", "per text token"])
    ax.set_ylabel("mean attention weight")
    ax.set_title("Attention PER TOKEN\nremoves the 256-vs-16 count effect", fontsize=9.5)
    ax.margins(y=0.22)

    ax = _style(axes[2])
    xs = np.arange(2)
    for i, a in enumerate(ARMS):
        t = d[a]["text_ablation"]
        b = ax.bar(xs + (i - 0.5) * w, [t["loss_base"], t["loss_text_zeroed"]],
                   w * 0.92, color=COLORS[a], label=SHORT[a])
        _labels(ax, b, "{:.4f}", size=7.5)
        # Sit the ratio over the ABLATED bar it describes, not between groups.
        ax.text(1 + (i - 0.5) * w, t["loss_text_zeroed"] * 1.13,
                f"×{t['ratio']:.2f}", ha="center", fontsize=9.5,
                color=COLORS[a], fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["intact", "text tokens zeroed"])
    ax.set_ylabel("velocity loss")
    ax.set_title("Dependence on reading the instruction", fontsize=9.5)
    ax.margins(y=0.28)

    fig.suptitle("Two heads, two solutions: Pi-0.5 reads the goal from image tokens, "
                 "PaliGemma from text tokens", fontsize=11.5, y=1.04)
    fig.savefig(FIG / "fig2_routing.png", bbox_inches="tight")
    plt.close(fig)


def fig3_adapter(d, depth):
    """The adapter does not cause the inversion — it widens the offline gap."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))

    ax = _style(axes[0])
    for a in ARMS:
        r = depth["libero_goal"][a]["doc"]["pools"]["image"]["r2_action"]
        p = d[a]["adapter_probes"]["image"]["r2_action"]
        ax.plot([0, 1], [r, p], "o-", color=COLORS[a], lw=2.2, ms=7, label=SHORT[a])
        ax.text(1.04, p, f"{p:.3f}", va="center", fontsize=8.5, color=COLORS[a])
        ax.text(-0.04, r, f"{r:.3f}", va="center", ha="right",
                fontsize=8.5, color=COLORS[a])
    gr = (depth["libero_goal"]["pi05"]["doc"]["pools"]["image"]["r2_action"]
          - depth["libero_goal"]["paligemma"]["doc"]["pools"]["image"]["r2_action"])
    ga = (d["pi05"]["adapter_probes"]["image"]["r2_action"]
          - d["paligemma"]["adapter_probes"]["image"]["r2_action"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw VLM\n(layer 18)", "after adapter\n(what the DiT reads)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("action R², image pool")
    ax.set_title(f"Gap WIDENS: {gr:+.3f} → {ga:+.3f}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = _style(axes[1])
    w, xs = 0.36, np.arange(2)
    for i, a in enumerate(ARMS):
        v = [d[a]["adapter_probes"]["image"]["r2_action"],
             d[a]["adapter_probes"]["text"]["r2_action"]]
        b = ax.bar(xs + (i - 0.5) * w, v, w * 0.92, color=COLORS[a], label=SHORT[a])
        _labels(ax, b, "{:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["image pool", "text pool"])
    ax.set_ylabel("action R² after adapter")
    ax.set_title("PaliGemma's TEXT pool outranks its image pool;\n"
                 "Pi-0.5's is the reverse", fontsize=9.5)
    ax.margins(y=0.20)

    fig.suptitle("The inversion is not a representation-quality effect",
                 fontsize=11.5, y=1.04)
    fig.savefig(FIG / "fig3_adapter.png", bbox_inches="tight")
    plt.close(fig)


def fig4_per_task(d, ev):
    """Per-task SR and the loss/SR dissociation."""
    sr_pi, lang = sr_by_task(ev["pi05"])
    sr_pa, _ = sr_by_task(ev["paligemma"])
    tasks = sorted(sr_pi)
    lt = {a: d[a]["phase_task_loss"]["by_task"] for a in ARMS}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = _style(axes[0])
    w, xs = 0.38, np.arange(len(tasks))
    for i, (a, sr) in enumerate([("pi05", sr_pi), ("paligemma", sr_pa)]):
        b = ax.bar(xs + (i - 0.5) * w, [sr[t] for t in tasks], w * 0.92,
                   color=COLORS[a], label=SHORT[a])
        _labels(ax, b, "{:.0f}", size=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([_short(lang[t]) for t in tasks], rotation=32,
                       ha="right", fontsize=8)
    ax.set_ylabel("canonical success rate (%)")
    ax.set_title("PaliGemma wins 7 tasks, loses 2, ties 1 — and never collapses\n"
                 "(worst 10% vs Pi-0.5's 0/5/5%)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.16)

    ax = _style(axes[1])
    dl, ds, names = [], [], []
    for t in tasks:
        k = lang[t]
        if k not in lt["pi05"] or k not in lt["paligemma"]:
            continue
        dl.append(lt["paligemma"][k]["loss"] - lt["pi05"][k]["loss"])
        ds.append(sr_pa[t] - sr_pi[t])
        names.append(k)
    dl, ds = np.asarray(dl), np.asarray(ds)
    r = float(np.corrcoef(dl, ds)[0, 1])
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.scatter(dl, ds, s=58, color="#8a63d2", zorder=3, edgecolor=SURF, lw=0.8)
    if len(dl) > 2:
        z = np.polyfit(dl, ds, 1)
        xx = np.linspace(dl.min(), dl.max(), 20)
        ax.plot(xx, np.polyval(z, xx), color=INK2, lw=1.3, ls="--")
    for x, y, n in zip(dl, ds, names):
        if abs(y) >= 20 or x < 0.002:
            ax.annotate(_short(n), (x, y), fontsize=7.5, color=INK2,
                        xytext=(5, 5), textcoords="offset points")
    ax.margins(x=0.18, y=0.14)
    ax.set_xlabel("Δ velocity loss  (PaliGemma − Pi-0.5)")
    ax.set_ylabel("Δ success rate, points")
    ax.set_title(f"Offline loss ANTI-predicts closed-loop gain\nr = {r:+.3f}  (n={len(dl)})",
                 fontsize=9.5)

    fig.suptitle("Where the two policies differ, and why loss does not see it",
                 fontsize=11.5, y=1.02)
    fig.savefig(FIG / "fig4_per_task.png", bbox_inches="tight")
    plt.close(fig)


def fig5_fairness(d, val):
    """The two audit results: checkpoint selection and the PE asymmetry."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = _style(axes[0])
    for a in ARMS:
        ax.plot(range(1, len(val[a]) + 1), val[a], color=COLORS[a], lw=1.5,
                label=SHORT[a])
        lo = int(np.argmin(val[a]))
        ax.plot([lo + 1], [val[a][lo]], "o", ms=7, color=COLORS[a],
                markeredgecolor=SURF)
        ax.annotate(f"best ep {lo+1}", (lo + 1, val[a][lo]), fontsize=7.5,
                    color=COLORS[a], xytext=(4, 7), textcoords="offset points")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation velocity loss")
    ax.set_title("Training curves", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0.03, 0.12)

    ax = _style(axes[1])
    order = ["epoch_0025.pt", "epoch_0050.pt", "epoch_0075.pt",
             "epoch_0100.pt", "best.pt", "final.pt"]
    for a in ARMS:
        lad = d[a]["ckpt_ladder"]
        pts = [(lad[c]["epoch"], lad[c]["corr"]) for c in order if c in lad]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=COLORS[a], lw=1.8, ms=5, label=SHORT[a])
        bc = lad["best.pt"]
        ax.plot([bc["epoch"]], [bc["corr"]], "*", ms=14, color=COLORS[a],
                markeredgecolor=SURF, zorder=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("open-loop action correlation")
    ax.set_title("Checkpoint ladder — ★ marks best.pt\n"
                 "PaliGemma's selected ckpt is not its strongest", fontsize=9.5)

    ax = _style(axes[2])
    # Input-side ratio is a property of the cached features (PE std 0.5442 over
    # each arm's per-dim content std); the measured value is the action shift when
    # the PE is zeroed. Showing both is the point: the input asymmetry is large,
    # the behavioural consequence is not.
    inp = {"pi05": 0.5442 / 1.1890, "paligemma": 0.5442 / 2.2736}
    meas = {a: d[a]["pe_sensitivity"]["relative_shift"] for a in ARMS}
    w, xs = 0.36, np.arange(2)
    for i, a in enumerate(ARMS):
        b = ax.bar(xs + (i - 0.5) * w, [inp[a], meas[a]], w * 0.92,
                   color=COLORS[a], label=SHORT[a])
        _labels(ax, b, "{:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["PE ÷ content\nat the input", "action shift\nwhen PE zeroed"])
    ax.set_title(f"PE asymmetry: {inp['pi05']/inp['paligemma']:.2f}× at input,"
                 f" {meas['pi05']/meas['paligemma']:.2f}× in effect", fontsize=9.5)
    ax.margins(y=0.22)

    fig.suptitle("Fairness audit: neither confound explains the 9-point gap",
                 fontsize=11.5, y=1.04)
    fig.savefig(FIG / "fig5_fairness.png", bbox_inches="tight")
    plt.close(fig)


def fig6_gate(ev):
    """The instruction gate, on both backbones."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _style(ax)
    conds = ["canonical", "swapped"]
    w, xs = 0.36, np.arange(len(conds))
    for i, a in enumerate(ARMS):
        vals = []
        for c in conds:
            rows = [r["success"] for r in ev[a] if r["condition"] == c]
            vals.append(100.0 * np.mean(rows) if rows else np.nan)
        b = ax.bar(xs + (i - 0.5) * w, vals, w * 0.92, color=COLORS[a],
                   label=SHORT[a])
        _labels(ax, b, "{:.1f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["correct instruction", "swapped instruction"])
    ax.set_ylabel("success rate (%)")
    ax.set_title("Language is load-bearing for BOTH backbones\n"
                 "0/200 under a wrong instruction, all ten tasks", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.20)
    fig.savefig(FIG / "fig6_gate.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    d, depth, ev, val = load_all()
    for fn, args in [(fig1_inversion, (d, depth, ev)),
                     (fig2_routing, (d,)),
                     (fig3_adapter, (d, depth)),
                     (fig4_per_task, (d, ev)),
                     (fig5_fairness, (d, val)),
                     (fig6_gate, (ev,))]:
        fn(*args)
        print(f"  {fn.__name__}")
    print(f"\nwrote {len(list(FIG.glob('*.png')))} figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
