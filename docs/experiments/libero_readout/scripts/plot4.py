"""Figure 4 - closed-loop perturbation battery with Wilson 95% CIs.

The LIBERO-PRO / -Plus / -Para literature reports these numbers with no error
bars at all; supplying them is part of the contribution.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


rows = []
for f in sorted(glob.glob("rollout_tap*_*.json")):
    d = json.load(open(f))
    k = sum(r["success"] for r in d["per_task"])
    n = sum(r["n"] for r in d["per_task"])
    p, lo, hi = wilson(k, n)
    rows.append(dict(tap=d["tap"], variant=d["variant"], k=k, n=n,
                     sr=p, lo=lo, hi=hi, per_task=d["per_task"]))

if not rows:
    raise SystemExit("no rollout_*.json found")

taps = sorted({r["tap"] for r in rows})
order = ["orig", "para1", "swap", "blank", "nonsense"]
vars_ = [v for v in order if any(r["variant"] == v for r in rows)]

print(f"{'tap':>5} {'variant':>9} {'k/n':>8} {'SR':>7}   Wilson 95% CI")
for t in taps:
    for v in vars_:
        r = next((x for x in rows if x["tap"] == t and x["variant"] == v), None)
        if r:
            print(f"{t:>5} {v:>9} {r['k']:>3}/{r['n']:<4} {100*r['sr']:6.1f}%   "
                  f"[{100*r['lo']:5.1f}, {100*r['hi']:5.1f}]")

# non-overlapping CI checks
print("\n=== do the CIs separate? ===")
for t in taps:
    o = next((x for x in rows if x["tap"] == t and x["variant"] == "orig"), None)
    for v in vars_:
        if v == "orig":
            continue
        r = next((x for x in rows if x["tap"] == t and x["variant"] == v), None)
        if o and r:
            sep = "SEPARATED" if (r["hi"] < o["lo"] or o["hi"] < r["lo"]) else "overlap"
            print(f"  tap{t:2d} orig vs {v:8s}: {100*o['sr']:5.1f}% vs {100*r['sr']:5.1f}%  -> {sep}")
if len(taps) >= 2:
    a = next((x for x in rows if x["tap"] == taps[0] and x["variant"] == "orig"), None)
    b = next((x for x in rows if x["tap"] == taps[-1] and x["variant"] == "orig"), None)
    if a and b:
        sep = "SEPARATED" if (a["hi"] < b["lo"] or b["hi"] < a["lo"]) else "overlap"
        print(f"  best vs worst tap (orig): tap{a['tap']}={100*a['sr']:.1f}% "
              f"vs tap{b['tap']}={100*b['sr']:.1f}%  -> {sep}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
w = 0.8 / max(len(taps), 1)
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
a = ax[0]
for i, t in enumerate(taps):
    xs, ys, el, eh = [], [], [], []
    for j, v in enumerate(vars_):
        r = next((x for x in rows if x["tap"] == t and x["variant"] == v), None)
        if r:
            xs.append(j + i * w - 0.4 + w / 2)
            ys.append(100 * r["sr"])
            el.append(100 * (r["sr"] - r["lo"]))
            eh.append(100 * (r["hi"] - r["sr"]))
    a.bar(xs, ys, width=w * 0.9, color=C[i % len(C)], label=f"tap {t}")
    a.errorbar(xs, ys, yerr=[el, eh], fmt="none", ecolor="k", capsize=3, lw=1)
a.set_xticks(range(len(vars_)))
a.set_xticklabels(vars_)
a.set_ylabel("closed-loop success rate (%)")
a.set_title("(a) LIBERO-Goal perturbation battery, Wilson 95% CI")
a.legend(fontsize=8)
a.grid(alpha=.25, axis="y")

a = ax[1]
for i, t in enumerate(taps):
    r = next((x for x in rows if x["tap"] == t and x["variant"] == "orig"), None)
    if not r:
        continue
    pt = [100 * x["success"] / x["n"] for x in r["per_task"]]
    a.plot(range(len(pt)), pt, "o-", ms=4, color=C[i % len(C)], label=f"tap {t}")
a.set_xlabel("LIBERO-Goal task id")
a.set_ylabel("per-task SR (%)")
a.set_title("(b) per-task breakdown (original instruction)")
a.legend(fontsize=8)
a.grid(alpha=.25)
plt.tight_layout()
plt.savefig("fig4_perturbation.png", dpi=150)
print("\nwrote fig4_perturbation.png")
json.dump([{k: v for k, v in r.items() if k != "per_task"} for r in rows],
          open("closed_loop_summary.json", "w"), indent=1)
