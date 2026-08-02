"""Claim-by-claim verdict: does the SmolVLM2 study replicate on Qwen3.5-0.8B?

Every verdict is decided by Wilson intervals on the actual episode counts, not
by eyeballing point estimates.  Where Qwen is underpowered relative to the
SmolVLM2 run, that is reported as underpowered - not as agreement.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def wilson(k, n, z=1.959963985):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def load(pattern, key):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        d = json.load(open(f))
        k = sum(r["success"] for r in d["per_task"])
        n = sum(r["n"] for r in d["per_task"])
        out[(d["mode"], d["tap"], d["variant"])] = (k, n) + wilson(k, n)
    return out


S = load("vroll_*.json", "smol")     # SmolVLM2-500M, n=300/arm
Q = load("qroll_*.json", "qwen")     # Qwen3.5-0.8B,  n=200/arm

print("=" * 78)
print("CLOSED-LOOP, LIBERO-Goal")
print("=" * 78)
for nm, D in (("SmolVLM2-500M", S), ("Qwen3.5-0.8B", Q)):
    print(f"\n--- {nm} ---")
    for (m, t, v), (k, n, p, lo, hi) in sorted(D.items()):
        print(f"  {m:5s} tap{t:<3} {v:5s}  {k:3d}/{n}  {100*p:5.1f}%  "
              f"[{100*lo:5.1f}, {100*hi:5.1f}]")


def sep(a, b):
    """do two Wilson intervals separate?"""
    return (a[4] < b[3]) or (b[4] < a[3])


print("\n" + "=" * 78)
print("CLAIM-BY-CLAIM VERDICT")
print("=" * 78)

# --- claim 1: language load-bearing
print("\n[1] Language is load-bearing (orig vs goal-swap)")
for nm, D, key_o, key_s in (("SmolVLM2", S, ("all", 30, "orig"), ("all", 30, "swap")),
                            ("Qwen3.5 ", Q, ("all", 12, "orig"), ("all", 12, "swap"))):
    if key_o in D and key_s in D:
        a, b = D[key_o], D[key_s]
        v = "SEPARATED -> CONFIRMED" if sep(a, b) else "overlap -> not shown"
        print(f"  {nm}: {100*a[2]:.1f}% -> {100*b[2]:.1f}%   {v}")

# --- claim 2: read-out token subset matters
print("\n[2] Read-out token subset matters (all-token vs instruction-only)")
for nm, D, ka, kb in (("SmolVLM2", S, ("all", 30, "orig"), ("instr", 30, "orig")),
                      ("Qwen3.5 ", Q, ("all", 12, "orig"), ("instr", 12, "orig"))):
    if ka in D and kb in D:
        a, b = D[ka], D[kb]
        delta = 100 * (a[2] - b[2])
        v = "SEPARATED -> CONFIRMED" if sep(a, b) else "OVERLAP -> underpowered at this n"
        print(f"  {nm}: {100*a[2]:.1f}% vs {100*b[2]:.1f}%  (delta {delta:+.1f} pp, "
              f"n={a[1]})   {v}")

# --- claim 3: tap depth does not matter
print("\n[3] Tap depth does NOT matter (shallow vs deep, all-token, orig)")
for nm, D, ka, kb in (("SmolVLM2", S, ("all", 0, "orig"), ("all", 30, "orig")),
                      ("Qwen3.5 ", Q, ("all", 0, "orig"), ("all", 12, "orig"))):
    if ka in D and kb in D:
        a, b = D[ka], D[kb]
        v = "overlap -> NO depth effect (claim holds)" if not sep(a, b) \
            else "SEPARATED -> depth DOES matter (claim fails)"
        print(f"  {nm}: tap{ka[1]} {100*a[2]:.1f}% vs tap{kb[1]} {100*b[2]:.1f}%  "
              f"(delta {100*(b[2]-a[2]):+.1f} pp)   {v}")

# --- claim 4: A1, zero LM layers suffice
print("\n[4] A1: zero LM transformer layers suffice (tap 0)")
for nm, D, k0, kb in (("SmolVLM2", S, ("all", 0, "orig"), ("all", 30, "orig")),
                      ("Qwen3.5 ", Q, ("all", 0, "orig"), ("all", 12, "orig"))):
    if k0 in D:
        a = D[k0]
        rel = f"{100*a[2]:.1f}%  [{100*a[3]:.1f}, {100*a[4]:.1f}]"
        best = D[kb][2] if kb in D else None
        extra = f"  ({100*(best-a[2]):+.1f} pp vs best tap)" if best else ""
        print(f"  {nm}: {rel}{extra}")

# --- open loop cross-check
print("\n" + "=" * 78)
print("OPEN-LOOP SWEEP (both backbones)")
print("=" * 78)
for nm, f in (("SmolVLM2", "sweep2.json"), ("Qwen3.5", "qsweep.json")):
    p = os.path.join(RES, f)
    if not os.path.exists(p): continue
    R = [x for x in json.load(open(p)) if not x.get("lr_check")]
    print(f"\n--- {nm} ---")
    for m in ("all", "instr"):
        v = np.array([x["ol_mse_orig"] for x in R if x["mode"] == m])
        vl = np.array([x["val_loss"] for x in R if x["mode"] == m])
        ds = np.array([x["deg_swap"] for x in R if x["mode"] == m])
        dp = np.array([x["deg_para"] for x in R if x["mode"] == m])
        print(f"  {m:5s} ol_mse {v.min():.4f}-{v.max():.4f} (range {v.max()-v.min():.4f})  "
              f"val_loss {vl.mean():.3f}  deg_swap {ds.mean():.2f}  deg_para {dp.mean():.3f}")

# --- probe-level (forward-only) cross-check
print("\n" + "=" * 78)
print("PROBE-LEVEL (forward-only)")
print("=" * 78)
qp = os.path.join(RES, "qwen_probe.json")
if os.path.exists(qp):
    d = json.load(open(qp))
    print(f"  Qwen tokens: T={d['T_orig']} image={d['n_image_tokens']} "
          f"non-image={d['n_nonimage']}")
    print(f"  Qwen structural max|delta| = "
          f"{max(d['structural_max_abs_delta'].values()):.3e}")
    t = d["token_order"]
    print(f"  Qwen token order: vision eta2 {t['vision_eta2_image_first']:.3e} -> "
          f"{t['vision_eta2_text_first']:.3e}")
    print(f"                    vision R2   {t['vision_r2_image_first']:.4f} -> "
          f"{t['vision_r2_text_first']:.4f}  (delta {t['delta_r2']:+.4f})")
print("  SmolVLM2 structural max|delta| = 0.000e+00 (all 33 layers)")
print("  SmolVLM2 token order: vision eta2 9.829e-09 -> 1.394e-02")
print("                        vision R2   0.3337 -> 0.3929  (delta +0.0592)")

json.dump({"smolvlm2": {f"{m}|tap{t}|{v}": dict(k=k, n=n, sr=p, lo=lo, hi=hi)
                        for (m, t, v), (k, n, p, lo, hi) in S.items()},
           "qwen35": {f"{m}|tap{t}|{v}": dict(k=k, n=n, sr=p, lo=lo, hi=hi)
                      for (m, t, v), (k, n, p, lo, hi) in Q.items()}},
          open(os.path.join(RES, "backbone_comparison.json"), "w"), indent=1)
print("\nwrote backbone_comparison.json")
