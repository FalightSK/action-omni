"""Figure 2 + A4 LR-confound analysis.

DV is the OPEN-LOOP ROLLOUT action error (10-step Euler vs ground-truth chunk),
not the flow-matching val loss.  We also test whether val loss predicts it at all.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = json.load(open("sweep.json"))
PR = np.load("probe_results.npz")
POOLS = ["all", "vis", "text", "instr"]
GOAL = 1
TAPS = sorted({r["tap"] for r in R})
LRS = sorted({r["lr"] for r in R})
print(f"taps={TAPS}\nlrs={LRS}\nruns={len(R)}")


def get(tap, lr):
    for r in R:
        if r["tap"] == tap and r["lr"] == lr:
            return r
    return None


# ---------- A4: does the LR optimum move with tap depth? ----------
print("\n=== A4 LR-confound check ===")
best_lr, rows = {}, []
for t in TAPS:
    vals = [(lr, get(t, lr)) for lr in LRS]
    vals = [(lr, v) for lr, v in vals if v]
    if not vals:
        continue
    bl = min(vals, key=lambda x: x[1]["ol_mse_orig"])
    best_lr[t] = bl[0]
    rows.append((t, {lr: v["ol_mse_orig"] for lr, v in vals}, bl[0]))
    print(f" tap{t:2d}  " + "  ".join(f"lr{lr:.0e}={v['ol_mse_orig']:.4f}" for lr, v in vals)
          + f"   best={bl[0]:.0e}")
uniq = sorted(set(best_lr.values()))
print(f"\n -> distinct optimal LRs across taps: {[f'{x:.0e}' for x in uniq]}")
print(f" -> A4 {'VIOLATED (per-tap LR needed)' if len(uniq) > 1 else 'HOLDS (single LR valid)'}")

# best-LR series (removes the LR confound)
B = {t: get(t, best_lr[t]) for t in best_lr}
ol = np.array([B[t]["ol_mse_orig"] for t in TAPS])
vl = np.array([B[t]["val_loss"] for t in TAPS])
dsw = np.array([B[t]["deg_swap"] for t in TAPS])
dpa = np.array([B[t]["deg_para"] for t in TAPS])
sop = np.array([B[t]["swap_over_para"] for t in TAPS])

# ---------- predictor validation ----------
print("\n=== E3: predictors vs OPEN-LOOP action error (lower ol = better) ===")
print("    (sign flipped so positive rho = predictor tracks quality)")
preds = {}
for key, pool, flip in [("r2", "all", +1), ("r2", "instr", +1),
                        ("lambda", "instr", +1), ("eta2", "instr", +1),
                        ("erank", "all", +1), ("cka", "all", +1)]:
    v = PR[key][np.array(TAPS), POOLS.index(pool), GOAL]
    rho, p = spearmanr(v, -ol)          # -ol so higher predictor = better policy
    preds[f"{key}[{pool}]"] = (v, rho, p)
    print(f"  P:{key:7s}[{pool:5s}]  rho={rho:+.3f}  p={p:.4f}"
          + ("  *" if p < 0.05 else ""))

rho_vl, p_vl = spearmanr(-vl, -ol)
print(f"\n  val_loss (the standard DV)   rho={rho_vl:+.3f}  p={p_vl:.4f}"
      + ("  *" if p_vl < 0.05 else ""))
print(f"  critical |rho| at n={len(TAPS)}, alpha=.05 two-tailed = 0.587")

# ---------- figure ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
a = ax[0, 0]
for lr in LRS:
    y = [get(t, lr)["ol_mse_orig"] if get(t, lr) else np.nan for t in TAPS]
    a.plot(TAPS, y, marker="o", ms=3, label=f"lr={lr:.0e}")
a.set_title("(a) A4 control: open-loop error vs tap, per LR")
a.set_xlabel("tap layer"); a.set_ylabel("open-loop action MSE"); a.legend(fontsize=7)

a = ax[0, 1]
a.plot(TAPS, ol, "o-", c="k", label="open-loop MSE (best LR)")
a.set_xlabel("tap layer"); a.set_ylabel("open-loop action MSE")
b = a.twinx(); b.plot(TAPS, vl, "s--", c="#d62728", label="flow val loss")
b.set_ylabel("val loss", color="#d62728")
a.set_title("(b) tap-depth curve: rollout error vs val loss")

a = ax[0, 2]
a.plot(TAPS, dsw, "o-", c="#d62728", label=r"deg$_{swap}$")
a.plot(TAPS, dpa, "s-", c="#2ca02c", label=r"deg$_{para}$")
a.plot(TAPS, sop, "^-", c="#1f77b4", label="swap/para")
a.axhline(1, ls=":", c="k", lw=.8)
a.set_title("(c) language grounding vs depth"); a.set_xlabel("tap layer"); a.legend(fontsize=7)

a = ax[1, 0]
v, rho, p = preds["r2[all]"]
a.scatter(v, ol, c=TAPS, cmap="viridis")
for i, t in enumerate(TAPS):
    a.annotate(str(t), (v[i], ol[i]), fontsize=6)
a.set_xlabel(r"P3 ridge $R^2$ (all-pool)"); a.set_ylabel("open-loop MSE")
a.set_title(f"(d) P3 vs rollout error   " + r"$\rho$" + f"={rho:+.2f}, p={p:.3f}")

a = ax[1, 1]
a.scatter(-vl, -ol, c=TAPS, cmap="viridis")
for i, t in enumerate(TAPS):
    a.annotate(str(t), (-vl[i], -ol[i]), fontsize=6)
a.set_xlabel("-val loss"); a.set_ylabel("-open-loop MSE")
a.set_title(f"(e) val loss vs rollout error   " + r"$\rho$" + f"={rho_vl:+.2f}, p={p_vl:.3f}")

a = ax[1, 2]
names = list(preds)
vals = [preds[n][1] for n in names] + [rho_vl]
a.barh(names + ["val_loss"], vals,
       color=["#1f77b4"] * len(names) + ["#d62728"])
a.axvline(0.587, ls="--", c="k", lw=.8); a.axvline(-0.587, ls="--", c="k", lw=.8)
a.set_xlabel(r"Spearman $\rho$ vs rollout quality"); a.set_title("(f) predictor comparison (n=12)")
a.tick_params(labelsize=7)

for A in ax.ravel():
    A.grid(alpha=.25)
plt.tight_layout(); plt.savefig("fig2_predictor.png", dpi=150)
print("\nwrote fig2_predictor.png")

# ---------- pick arms for closed-loop ----------
order = np.argsort(ol)
best_tap, worst_tap = TAPS[order[0]], TAPS[order[-1]]
print(f"\n=== closed-loop arms ===")
print(f" BEST  tap={best_tap} lr={best_lr[best_tap]:.0e} ol={ol[order[0]]:.4f}")
print(f" WORST tap={worst_tap} lr={best_lr[worst_tap]:.0e} ol={ol[order[-1]]:.4f}")
json.dump(dict(best_tap=int(best_tap), best_lr=best_lr[best_tap],
               worst_tap=int(worst_tap), worst_lr=best_lr[worst_tap],
               taps=[int(t) for t in TAPS],
               ol=[float(x) for x in ol], val_loss=[float(x) for x in vl],
               spearman={k: float(v[1]) for k, v in preds.items()},
               spearman_valloss=float(rho_vl)),
          open("arms.json", "w"), indent=1)
