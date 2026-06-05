"""
scripts/plot_lt_ood.py
──────────────────────
Phase 5 figure — Language Table language-conditioned generalization.

Panel A: raw success by command type (block2block, block2absolute = in-distribution;
         separate, point = held-out verbs).  These are DIFFERENT tasks of different
         difficulty, so raw SR is NOT a clean interpolation-vs-extrapolation read.
Panel B: difficulty-matched verb-generalization control (from eval_lt_control.py) —
         the SAME task with a trained verb vs a held-out verb.  This isolates whether
         the model follows an unseen word, independent of task difficulty.

Reads:
  asset/runs/language_table/exp01_baseline/ood_results.json       (eval_lt_ood.py)
  asset/runs/language_table/exp01_baseline/control_results.json   (eval_lt_control.py; optional)
Output → docs/experiments/language_table/lt_generalization.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parents[1]
RUN = ROOT / "asset/runs/language_table/exp01_baseline"
OOD = RUN / "ood_results.json"
CTRL = RUN / "control_results.json"
OUT = ROOT / "docs/experiments/language_table/lt_generalization.png"
BLUE, ORANGE = "#4C72B0", "#DD8452"

j = json.loads(OOD.read_text())
by = j["by_reward"]
TASK = {"block2block": "push to block", "block2absolute": "push to region",
        "separate": "push apart", "point": "reach to block"}

order = [r for r in ("block2block", "block2absolute", "separate", "point") if r in by]
labels, srs, colors, examples, msteps, kinds = [], [], [], [], [], []
for r in order:
    v = by[r]
    labels.append(r); srs.append(v["sr"] * 100); kinds.append(v["kind"])
    colors.append(BLUE if v["kind"] == "in_dist" else ORANGE)
    msteps.append(v.get("mean_steps", 0))
    ex = next((e["instruction"] for e in v["episodes"] if e.get("instruction")), "")
    examples.append(ex)

has_ctrl = CTRL.exists()
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.5, 1]})
axA, axB = axes

# ── Panel A : raw SR by command type ───────────────────────────────────────────
x = np.arange(len(labels))
axA.bar(x, srs, color=colors, edgecolor="black", lw=0.9, width=0.66)
for xi, (sr, ex, ms, r) in enumerate(zip(srs, examples, msteps, order)):
    n = by[r]["n"]; h = int(round(by[r]["sr"] * n))
    axA.annotate(f"{sr:.0f}%\n({h}/{n})", (xi, sr), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=11, fontweight="bold")
    axA.text(xi, -6, f"task: {TASK.get(r,'?')}\n⟨{ms:.0f} steps⟩\n{ex[:30]}", ha="center",
             va="top", fontsize=7.6, color="#333")
axA.set_xticks(x)
axA.set_xticklabels([f"{l}\n[{'OOD verb' if k=='ood' else 'in-dist'}]"
                     for l, k in zip(labels, kinds)], fontsize=9.5, fontweight="bold")
axA.set_ylabel("success rate (%)"); axA.set_ylim(-34, 105)
axA.axhline(0, color="black", lw=0.8)
axA.set_title("A. Raw success by command type\n(different TASKS — difficulty varies, not directly comparable)",
              fontsize=11.5, fontweight="bold")
axA.grid(axis="y", alpha=0.3)
axA.legend(handles=[Patch(facecolor=BLUE, edgecolor="black", label="in-distribution verb"),
                    Patch(facecolor=ORANGE, edgecolor="black", label="held-out / new verb")],
           loc="upper left", fontsize=9)

# ── Panel B : difficulty-matched verb-generalization control ────────────────────
if has_ctrl:
    cj = json.loads(CTRL.read_text())
    cond = {c["label"]: c for c in cj["conditions"]}
    # group by task: separate (push apart), reach
    groups = [("separate / push apart",
               next((c for c in cj["conditions"] if c["factory"] == "separate" and c["verb_kind"] == "trained"), None),
               next((c for c in cj["conditions"] if c["factory"] == "separate" and c["verb_kind"] == "held_out"), None)),
              ("point / reach",
               next((c for c in cj["conditions"] if c["factory"] == "point" and c["verb_kind"] == "trained"), None),
               next((c for c in cj["conditions"] if c["factory"] == "point" and c["verb_kind"] == "held_out"), None))]
    gx = np.arange(len(groups)); w = 0.36
    tr = [g[1]["sr"] * 100 if g[1] else 0 for g in groups]
    ho = [g[2]["sr"] * 100 if g[2] else 0 for g in groups]
    axB.bar(gx - w/2, tr, w, color=BLUE, edgecolor="black", lw=0.9, label="trained verb (seen)")
    axB.bar(gx + w/2, ho, w, color=ORANGE, edgecolor="black", lw=0.9, label="held-out verb (unseen)")
    for xi, v in zip(gx - w/2, tr):
        axB.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold")
    for xi, v in zip(gx + w/2, ho):
        axB.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold", color="#8a4b1f")
    for gi, (name, t, h) in enumerate(groups):
        gap = (t["sr"] - h["sr"]) * 100 if (t and h) else 0
        axB.text(gi, -8, f"{name}\nΔ(seen−unseen)={gap:+.0f}pts", ha="center", va="top", fontsize=8.5, color="#333")
    axB.set_xticks(gx); axB.set_xticklabels(["", ""])
    axB.set_ylabel("success rate (%)"); axB.set_ylim(-20, 105)
    axB.axhline(0, color="black", lw=0.8)
    axB.set_title("B. Verb generalization — SAME task, trained vs held-out verb\n"
                  "(\"move away\" vs \"separate\";  \"move arm\" vs \"point\")",
                  fontsize=11.5, fontweight="bold")
    axB.legend(loc="upper right", fontsize=9); axB.grid(axis="y", alpha=0.3)
else:
    axB.text(0.5, 0.5, "control_results.json not found\n(run scripts/eval_lt_control.py)",
             ha="center", va="center", fontsize=11)
    axB.axis("off")

n_per = j.get("n_episodes_per_reward", "?")
fig.suptitle(
    f"Qwen-DiT VLA on Google Language Table — does it follow NEW commands? "
    f"(n={n_per}/condition, ih={j.get('inference_horizon','?')}, 5k-episode / 4.3k-instruction train)\n"
    f"'separate' and 'point' verbs were HELD OUT of training entirely. Panel A: raw SR varies with TASK difficulty "
    f"(precise placement is hard, reach/spread easy). Panel B isolates the verb: same task, seen vs unseen word.",
    fontsize=12, fontweight="bold", y=1.03)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}  (control={'yes' if has_ctrl else 'no'})")
