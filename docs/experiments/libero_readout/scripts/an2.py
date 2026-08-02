import json
import numpy as np
from scipy.stats import spearmanr

R = json.load(open("sweep2.json"))
main = [x for x in R if not x.get("lr_check")]
lrc = [x for x in R if x.get("lr_check")]
PR = np.load("probe_results.npz")
POOLS = ["all", "vis", "text", "instr"]
GOAL = 1
taps = sorted({x["tap"] for x in main})

print("=== corrected sweep: 12 taps x 2 read-outs (96-tok window, masked padding) ===")
print(f"{'tap':>4} | {'ALL ol_mse':>11} {'deg_swap':>9} {'vloss':>7} | "
      f"{'INSTR ol_mse':>13} {'deg_swap':>9} {'vloss':>7}")
rows = {}
for t in taps:
    a = next(x for x in main if x["tap"] == t and x["mode"] == "all")
    i = next(x for x in main if x["tap"] == t and x["mode"] == "instr")
    rows[t] = (a, i)
    print(f"{t:>4} | {a['ol_mse_orig']:11.4f} {a['deg_swap']:9.3f} {a['val_loss']:7.4f} | "
          f"{i['ol_mse_orig']:13.4f} {i['deg_swap']:9.3f} {i['val_loss']:7.4f}")

for mode in ("all", "instr"):
    v = np.array([next(x for x in main if x["tap"] == t and x["mode"] == mode)["ol_mse_orig"]
                  for t in taps])
    print(f"\n{mode:>5}: ol_mse  min={v.min():.4f} max={v.max():.4f} "
          f"range={v.max()-v.min():.4f} sd={v.std():.4f} best=tap{taps[int(v.argmin())]}")

print("\n=== PRE-REGISTERED TEST ===")
print("Figure 1 predicted (before this ran): depth effect for instruction-token")
print("read-out, ~flat for all-token.  probe R^2 range: instr 0.264 vs all 0.053")
ra = np.array([rows[t][0]["ol_mse_orig"] for t in taps])
ri = np.array([rows[t][1]["ol_mse_orig"] for t in taps])
print(f"observed ol_mse range:  all={ra.max()-ra.min():.4f}   instr={ri.max()-ri.min():.4f}")
print(f"ratio instr/all = {(ri.max()-ri.min())/max(ra.max()-ra.min(),1e-9):.2f}x  "
      f"(probe predicted {0.264/0.053:.2f}x)")

print("\n=== predictors vs open-loop rollout error (n=12) ===")
print("  critical |rho| = 0.587 (alpha=.05, two-tailed)")
for mode, ol in (("all", ra), ("instr", ri)):
    print(f"  -- read-out = {mode}")
    for key, pool in [("r2", "all"), ("r2", "instr"), ("lambda", "instr"),
                      ("eta2", "instr"), ("erank", "all"), ("cka", "all")]:
        v = PR[key][np.array(taps), POOLS.index(pool), GOAL]
        rho, p = spearmanr(v, -ol)
        print(f"     {key:6s}[{pool:5s}] rho={rho:+.3f} p={p:.4f}" + ("  *" if p < 0.05 else ""))
    vl = np.array([next(x for x in main if x["tap"] == t and x["mode"] == mode)["val_loss"]
                   for t in taps])
    rho, p = spearmanr(-vl, -ol)
    print(f"     {'val_loss':13s} rho={rho:+.3f} p={p:.4f}" + ("  *" if p < 0.05 else ""))

print("\n=== language grounding ===")
for mode in ("all", "instr"):
    ds = np.array([next(x for x in main if x["tap"] == t and x["mode"] == mode)["deg_swap"]
                   for t in taps])
    dp = np.array([next(x for x in main if x["tap"] == t and x["mode"] == mode)["deg_para"]
                   for t in taps])
    print(f"  {mode:>5}: deg_swap {ds.mean():.3f}+-{ds.std():.3f}   "
          f"deg_para {dp.mean():.3f}+-{dp.std():.3f}")

print("\n=== A4 LR check ===")
for x in sorted(lrc, key=lambda z: (z["mode"], z["tap"], z["lr"])):
    base = next((y for y in main if y["tap"] == x["tap"] and y["mode"] == x["mode"]), None)
    print(f"  {x['mode']:>5} tap{x['tap']:>3} lr={x['lr']:.0e} ol={x['ol_mse_orig']:.4f}"
          + (f"   (lr=1e-03 -> {base['ol_mse_orig']:.4f})" if base else ""))
