"""Evidence audit: recompute every headline number from the saved artifacts.
Anything that cannot be recomputed here is NOT supported and must not be claimed.
"""
import glob, json, os
import numpy as np

OK, MISSING = [], []


def have(p):
    return os.path.exists(p)


def rec(claim, value, src, ok=True):
    (OK if ok else MISSING).append((claim, value, src))


print("=" * 78)
print("EVIDENCE AUDIT")
print("=" * 78)

# --- 1. token budget: how many tokens can carry the instruction? ---
if have("val_mask2.npy"):
    M = np.load("val_mask2.npy")           # (n,5,96) 0pad 1vis 2scaffold 3instr
    ins = (M == 3).sum(-1)
    vis = (M == 1).sum(-1)
    scaf = (M == 2).sum(-1)
    tot = (M > 0).sum(-1)
    o = 0                                   # variant 0 = original instruction
    rec("instruction tokens (orig)",
        f"min={ins[:,o].min()} med={int(np.median(ins[:,o]))} max={ins[:,o].max()}",
        "val_mask2.npy")
    rec("vision tokens", f"{vis[:,o].min()}-{vis[:,o].max()}", "val_mask2.npy")
    rec("total tokens", f"{tot[:,o].min()}-{tot[:,o].max()}", "val_mask2.npy")
    frac = 100 * (1 - ins[:, o] / tot[:, o])
    rec("% tokens that CANNOT carry instruction",
        f"{frac.min():.1f}-{frac.max():.1f}% (median {np.median(frac):.1f}%)",
        "val_mask2.npy")
else:
    rec("token budget", "-", "val_mask2.npy MISSING", ok=False)

# --- 2. token-order intervention ---
if have("order_probe.json"):
    d = json.load(open("order_probe.json"))
    a = max(x["r2"] for x in d["image_first"]); b = max(x["r2"] for x in d["text_first"])
    ea = np.mean([x["eta2"] for x in d["image_first"]])
    eb = np.mean([x["eta2"] for x in d["text_first"]])
    rec("vision-pooled R2 image-first -> text-first", f"{a:.4f} -> {b:.4f} (d={b-a:+.4f})",
        "order_probe.json")
    rec("vision eta2 image-first -> text-first", f"{ea:.3e} -> {eb:.3e}", "order_probe.json")
else:
    rec("token order", "-", "order_probe.json MISSING", ok=False)

# --- 3. sweep: depth effect + read-out effect (open loop) ---
if have("sweep2.json"):
    R = [x for x in json.load(open("sweep2.json")) if not x.get("lr_check")]
    lrc = [x for x in json.load(open("sweep2.json")) if x.get("lr_check")]
    for m in ("all", "instr"):
        v = np.array([x["ol_mse_orig"] for x in R if x["mode"] == m])
        rec(f"open-loop ol_mse range [{m}]",
            f"{v.min():.4f}-{v.max():.4f} (range {v.max()-v.min():.4f})", "sweep2.json")
    for m in ("all", "instr"):
        ds = np.array([x["deg_swap"] for x in R if x["mode"] == m])
        dp = np.array([x["deg_para"] for x in R if x["mode"] == m])
        rec(f"deg_swap / deg_para [{m}]",
            f"{ds.mean():.3f}+-{ds.std():.3f} / {dp.mean():.3f}+-{dp.std():.3f}", "sweep2.json")
    rec("n sweep runs", f"{len(R)} main + {len(lrc)} LR-check = {len(R)+len(lrc)}", "sweep2.json")
    for x in sorted(lrc, key=lambda z: (z["mode"], z["tap"], z["lr"])):
        if x["mode"] == "all" and x["lr"] > 1e-3:
            rec(f"LR fragility all tap{x['tap']} @3e-3", f"ol={x['ol_mse_orig']:.4f}", "sweep2.json")
else:
    rec("sweep", "-", "sweep2.json MISSING", ok=False)

# --- 4. closed loop ---
def wilson(k, n, z=1.959963985):
    if n == 0: return 0, 0, 0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0, c - h), min(1, c + h)


cl = {}
for f in sorted(glob.glob("vroll_*.json")):
    d = json.load(open(f))
    k = sum(r["success"] for r in d["per_task"]); n = sum(r["n"] for r in d["per_task"])
    p, lo, hi = wilson(k, n)
    cl[(d["mode"], d["tap"], d["variant"])] = (k, n, p, lo, hi)
    rec(f"closed-loop {d['mode']} tap{d['tap']} {d['variant']}",
        f"{k}/{n} = {100*p:.1f}%  Wilson[{100*lo:.1f},{100*hi:.1f}]", f)
if not cl:
    rec("closed loop", "-", "no vroll_*.json", ok=False)

print("\n--- SUPPORTED (recomputed from artifacts) ---")
for c, v, s in OK:
    print(f"  [{s:28s}] {c:52s} = {v}")
if MISSING:
    print("\n--- NOT SUPPORTED (artifact missing) ---")
    for c, v, s in MISSING:
        print(f"  [{s}] {c}")

# --- explicit CI comparisons, only where both arms exist ---
print("\n--- CI COMPARISONS (only pairs that actually exist) ---")
def cmp(a, b, label):
    if a in cl and b in cl:
        ka, na, pa, la, ha = cl[a]; kb, nb, pb, lb, hb = cl[b]
        sep = "SEPARATED" if (ha < lb or hb < la) else "OVERLAP"
        print(f"  {label:44s} {100*pa:5.1f}% vs {100*pb:5.1f}%  -> {sep}")
    else:
        miss = [x for x in (a, b) if x not in cl]
        print(f"  {label:44s} PENDING {miss}")

cmp(("all", 30, "orig"), ("instr", 30, "orig"), "read-out: all vs instr (tap30, orig)")
cmp(("all", 9, "orig"), ("all", 30, "orig"), "depth: tap9 vs tap30 (all, orig)")
cmp(("all", 0, "orig"), ("all", 30, "orig"), "A1: tap0 (no LM layers) vs tap30")
cmp(("all", 30, "orig"), ("all", 30, "swap"), "language: orig vs goal-swap (all tap30)")
cmp(("instr", 30, "orig"), ("instr", 30, "swap"), "language: orig vs goal-swap (instr tap30)")

print("\n--- KNOWN LIMITATIONS (not fixable with current artifacts) ---")
for s in ["single seed per config (no between-seed SD)",
          "single backbone (SmolVLM2-500M); no cross-backbone replication",
          "LIBERO-Goal only for closed loop; other 3 suites probe-only",
          "cross-suite Lambda confounded by paraphrase edit-distance asymmetry",
          "no external SOTA anchor yet (pi0/pi05 blocked; SmolVLA running)"]:
    print("  -", s)
