"""Final figures + summary for phase 1.

Fig 2 : predictor vs open-loop, and open-loop vs CLOSED-loop (the link that broke)
Fig 4 : closed-loop perturbation battery with Wilson 95% CIs
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

POOLS = ["all", "vis", "text", "instr"]
GOAL = 1
PR = np.load("probe_results.npz")
R = [x for x in json.load(open("sweep2.json")) if not x.get("lr_check")]
taps = sorted({x["tap"] for x in R})


def wilson(k, n, z=1.959963985):
    if n == 0: return 0., 0., 0.
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0., c - h), min(1., c + h)


CL = {}
for f in sorted(glob.glob("vroll_*.json")):
    d = json.load(open(f))
    k = sum(r["success"] for r in d["per_task"]); n = sum(r["n"] for r in d["per_task"])
    CL[(d["mode"], d["tap"], d["variant"])] = (k, n) + wilson(k, n)

ol = {m: np.array([next(x for x in R if x["tap"] == t and x["mode"] == m)["ol_mse_orig"]
                   for t in taps]) for m in ("all", "instr")}
vl = {m: np.array([next(x for x in R if x["tap"] == t and x["mode"] == m)["val_loss"]
                   for t in taps]) for m in ("all", "instr")}

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

a = ax[0, 0]
for m, c in (("all", "#1f77b4"), ("instr", "#d62728")):
    a.plot(taps, ol[m], "o-", c=c, label=f"{m} read-out")
a.set_xlabel("tap layer"); a.set_ylabel("open-loop action MSE")
a.set_title("(a) tap depth vs open-loop error"); a.legend(fontsize=8)

a = ax[0, 1]
for m, c in (("all", "#1f77b4"), ("instr", "#d62728")):
    v = PR["r2"][np.array(taps), POOLS.index("instr"), GOAL]
    rho, p = spearmanr(v, -ol[m])
    a.scatter(v, ol[m], c=c, label=f"{m}: rho={rho:+.2f} p={p:.3f}")
a.set_xlabel(r"P3 ridge $R^2$ (instr-pooled)"); a.set_ylabel("open-loop MSE")
a.set_title("(b) P3 predicts OPEN-loop"); a.legend(fontsize=7)

# --- the link that broke: open-loop vs closed-loop ---
a = ax[0, 2]
pts = []
for (m, t, v), (k, n, p, lo, hi) in CL.items():
    if v != "orig": continue
    o = next((x for x in R if x["tap"] == t and x["mode"] == m), None)
    if o: pts.append((m, t, o["ol_mse_orig"], p, lo, hi))
for m, c in (("all", "#1f77b4"), ("instr", "#d62728")):
    s = [x for x in pts if x[0] == m]
    if not s: continue
    a.errorbar([x[2] for x in s], [100 * x[3] for x in s],
               yerr=[[100 * (x[3] - x[4]) for x in s], [100 * (x[5] - x[3]) for x in s]],
               fmt="o", color=c, capsize=3, label=m)
    for x in s: a.annotate(f"tap{x[1]}", (x[2], 100 * x[3]), fontsize=7)
a.set_xlabel("open-loop action MSE (lower=better)")
a.set_ylabel("closed-loop SR % (higher=better)")
a.set_title("(c) open-loop vs CLOSED-loop\n(monotone decreasing = valid proxy)")
a.legend(fontsize=8)

# --- Fig 4: perturbation battery ---
a = ax[1, 0]
labels, vals, elo, ehi, cols = [], [], [], [], []
for (m, t, v), (k, n, p, lo, hi) in sorted(CL.items()):
    labels.append(f"{m}\ntap{t}\n{v}"); vals.append(100 * p)
    elo.append(100 * (p - lo)); ehi.append(100 * (hi - p))
    cols.append("#2ca02c" if v == "orig" else "#d62728")
a.bar(range(len(vals)), vals, color=cols)
a.errorbar(range(len(vals)), vals, yerr=[elo, ehi], fmt="none", ecolor="k", capsize=3)
a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=6)
a.set_ylabel("closed-loop SR %"); a.set_title("(d) closed-loop, Wilson 95% CI")

a = ax[1, 1]
for m, c in (("all", "#1f77b4"), ("instr", "#d62728")):
    ds = np.array([next(x for x in R if x["tap"] == t and x["mode"] == m)["deg_swap"] for t in taps])
    dp = np.array([next(x for x in R if x["tap"] == t and x["mode"] == m)["deg_para"] for t in taps])
    a.plot(taps, ds, "o-", c=c, label=f"{m} deg_swap")
    a.plot(taps, dp, "s--", c=c, alpha=.5, label=f"{m} deg_para")
a.axhline(1, ls=":", c="k", lw=.8); a.set_xlabel("tap layer")
a.set_title("(e) language grounding (open-loop)"); a.legend(fontsize=7)

a = ax[1, 2]
for m, c in (("all", "#1f77b4"), ("instr", "#d62728")):
    rho, p = spearmanr(-vl[m], -ol[m])
    a.scatter(vl[m], ol[m], c=c, label=f"{m}: rho={rho:+.2f} p={p:.3f}")
a.set_xlabel("flow val loss"); a.set_ylabel("open-loop MSE")
a.set_title("(f) val loss does NOT predict rollout"); a.legend(fontsize=7)

for A in ax.ravel(): A.grid(alpha=.25)
plt.tight_layout(); plt.savefig("fig2_fig4_final.png", dpi=150)
print("wrote fig2_fig4_final.png")

summary = {"closed_loop": {f"{m}|tap{t}|{v}": dict(k=k, n=n, sr=p, lo=lo, hi=hi)
                           for (m, t, v), (k, n, p, lo, hi) in CL.items()},
           "open_loop": {m: {str(t): float(x) for t, x in zip(taps, ol[m])} for m in ol},
           "val_loss": {m: {str(t): float(x) for t, x in zip(taps, vl[m])} for m in vl}}
json.dump(summary, open("final_summary.json", "w"), indent=1)

print("\n=== CLOSED-LOOP TABLE ===")
for (m, t, v), (k, n, p, lo, hi) in sorted(CL.items()):
    print(f"  {m:5s} tap{t:<3} {v:5s}  {k:3d}/{n}  {100*p:5.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]")
print("\n=== IS OPEN-LOOP A VALID PROXY? ===")
for m in ("all", "instr"):
    s = [(t, o, p) for (mm, t, v), (k, n, p, lo, hi) in CL.items() if mm == m and v == "orig"
         for o in [next((x["ol_mse_orig"] for x in R if x["tap"] == t and x["mode"] == m), None)] if o]
    if len(s) >= 2:
        rho, pv = spearmanr([x[1] for x in s], [x[2] for x in s])
        print(f"  {m}: n={len(s)} taps, spearman(open-loop, closed-loop) = {rho:+.3f} (p={pv:.3f})")
        print(f"     valid proxy would give rho close to -1.0 (low MSE -> high SR)")
        for t, o, p in sorted(s, key=lambda z: z[1]):
            print(f"       tap{t:<3} ol={o:.4f}  SR={100*p:.1f}%")
