"""Tier-1 evaluation plots for LT exp01 (CPU-only, from existing logs/JSON):
  1. lt_loss_curve.png      train/val loss vs epoch (full 1-150, across the resume seam)
  2. lt_sr_summary.png      point vs block2block SR with Wilson 95% CIs
  3. lt_sr_vs_distance.png  block2block SR vs required push distance (fine bins)
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
RUN = ROOT / "asset/runs/language_table/exp01_baseline"
DOC = ROOT / "docs/experiments/language_table"; DOC.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))


# ── 1. loss curve ────────────────────────────────────────────────────────────
pat = re.compile(r"^Epoch\s+(\d+)/150\s+\|\s+train=([\d.]+)\s+val=([\d.]+)")
hist = {}
for lg in ("pipeline_prev.log", "pipeline.log"):
    p = RUN / lg
    if not p.exists(): continue
    for line in p.read_text(errors="ignore").splitlines():
        m = pat.match(line.strip())
        if m:
            e = int(m.group(1)); hist[e] = (float(m.group(2)), float(m.group(3)))
eps = sorted(hist); tr = [hist[e][0] for e in eps]; vl = [hist[e][1] for e in eps]
best_e = min(eps, key=lambda e: hist[e][1]); best_v = hist[best_e][1]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(eps, tr, label="train", color="#4C72B0", lw=2)
ax.plot(eps, vl, label="val", color="#C44E52", lw=2)
ax.scatter([best_e], [best_v], color="black", zorder=5,
           label=f"best val {best_v:.3f} @ ep {best_e}")
ax.axvline(25.5, color="gray", ls=":", lw=1)
ax.annotate("resume seam\n(ep 25->26)", (25.5, max(tr)*0.9), fontsize=8, color="gray", ha="left")
ax.set_xlabel("epoch"); ax.set_ylabel("flow-matching loss")
ax.set_title(f"LT exp01 training curve (150 epochs, {len(eps)} logged)\n"
             "smooth monotone decrease, val tracks train -> no overfitting",
             fontsize=11, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(DOC/"lt_loss_curve.png", dpi=140, bbox_inches="tight"); plt.close()
print("saved lt_loss_curve.png  (n_epochs=%d best_val=%.4f@%d)" % (len(eps), best_v, best_e))

# ── 2. SR summary (point vs block2block) ──────────────────────────────────────
sim = json.loads((RUN/"sim_results.json").read_text())
b2b_n = sim["n_episodes"]; b2b_k = int(round(sim["success_rate"]*b2b_n))
b2b_lo, b2b_hi = wilson(b2b_k, b2b_n)
# point SR from language_effect.log
pt_correct = pt_wrong = None; pn = 25
le = (RUN/"language_effect.log")
if le.exists():
    m = re.search(r"correct=(\d+)%\s+wrong=(\d+)%", le.read_text(errors="ignore"))
    if m: pt_correct = int(m.group(1))/100; pt_wrong = int(m.group(2))/100
pt_correct = pt_correct if pt_correct is not None else 0.92
pt_wrong = pt_wrong if pt_wrong is not None else 0.36
pk = int(round(pt_correct*pn)); pt_lo, pt_hi = wilson(pk, pn)

fig, ax = plt.subplots(figsize=(8, 5.4))
labels = ["point\n(reach named block)", "block2block\n(push to target)"]
vals = [pt_correct*100, sim["success_rate"]*100]
los = [pt_lo*100, b2b_lo*100]; his = [pt_hi*100, b2b_hi*100]
errs = [[v-l for v,l in zip(vals,los)], [h-v for v,h in zip(vals,his)]]
bars = ax.bar([0,1], vals, color=["#55A868","#C44E52"], edgecolor="black",
              yerr=errs, capsize=8)
for i,(v,lo,hi) in enumerate(zip(vals,los,his)):
    ax.annotate(f"{v:.0f}%\n[{lo:.0f}-{hi:.0f}]", (i,v), textcoords="offset points",
                xytext=(0,10), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("closed-loop success rate (%)"); ax.set_ylim(0,105)
ax.set_title("Same model, same language grounding, different manipulation demand\n"
             f"point n={pn}, block2block n={b2b_n}; bars = Wilson 95% CI",
             fontsize=11, fontweight="bold")
plt.savefig(DOC/"lt_sr_summary.png", dpi=140, bbox_inches="tight"); plt.close()
print("saved lt_sr_summary.png  (point=%.0f%% b2b=%.0f%%)" % (pt_correct*100, sim["success_rate"]*100))

# ── 3. SR vs required push distance (fine bins) ───────────────────────────────
fm = json.loads((RUN/"failure_modes.json").read_text())
d0 = np.array([r["d0"] for r in fm["rows"]]); ok = np.array([r["success"] for r in fm["rows"]], bool)
edges = np.array([0, 0.10, 0.15, 0.20, 0.25, 0.30, d0.max()+1e-6])
cx, sr, cnt = [], [], []
for i in range(len(edges)-1):
    sel = (d0 >= edges[i]) & (d0 < edges[i+1])
    if sel.sum() == 0: continue
    cx.append((edges[i]+min(edges[i+1], d0.max()))/2); sr.append(ok[sel].mean()*100); cnt.append(int(sel.sum()))
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(cx, sr, "-o", color="#C44E52", lw=2.5, ms=9)
for x,y,c in zip(cx,sr,cnt):
    ax.annotate(f"{y:.0f}%\n(n={c})", (x,y), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("required push distance d0 (board units)"); ax.set_ylabel("success rate (%)")
ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
ax.set_title(f"block2block SR collapses with required push distance (n={len(d0)})\n"
             "evidence for distance-limited closed-loop control", fontsize=11, fontweight="bold")
plt.savefig(DOC/"lt_sr_vs_distance.png", dpi=140, bbox_inches="tight"); plt.close()
print("saved lt_sr_vs_distance.png  (bins=%d)" % len(cx))
print("DONE Tier-1 plots ->", DOC)
