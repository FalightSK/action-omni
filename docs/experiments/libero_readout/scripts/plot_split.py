"""Figure: instruction train/test split, and what it costs at rollout time.

(a) which instruction variants each condition SEES during training
(b) share of the instruction pool used for train vs held out for test
(c) closed-loop success, seen vs unseen phrasing, both backbones
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")

SEEN, UNSEEN = "#2e7d32", "#c62828"
VAR = ["orig", "para1", "para2", "para3", "goal-swap"]
COND = {
    "A: train on 1 phrasing": [1, 0, 0, 0, 0],
    "B: train on 3 phrasings": [1, 1, 1, 0, 0],
}

fig = plt.figure(figsize=(15, 4.6))

# ---- (a) which variants are in training -------------------------------------
ax = fig.add_subplot(1, 3, 1)
for r, (name, mask) in enumerate(COND.items()):
    for c, m in enumerate(mask):
        ax.add_patch(plt.Rectangle((c, -r), 1, -0.85,
                                   color=SEEN if m else UNSEEN, alpha=.85))
        ax.text(c + .5, -r - .42, "TRAIN" if m else "test", ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")
ax.set_xlim(0, len(VAR)); ax.set_ylim(-len(COND), 0.25)
ax.set_xticks(np.arange(len(VAR)) + .5); ax.set_xticklabels(VAR, fontsize=8)
ax.set_yticks([-r - .42 for r in range(len(COND))])
ax.set_yticklabels(list(COND), fontsize=8)
ax.set_title("(a) Which instruction variants the model sees in training", fontsize=9.5)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)

# ---- (b) proportion of the instruction pool ---------------------------------
ax = fig.add_subplot(1, 3, 2)
names = list(COND)
tr = [sum(COND[n]) for n in names]
te = [len(VAR) - t for t in tr]
y = np.arange(len(names))
ax.barh(y, tr, color=SEEN, label="in training")
ax.barh(y, te, left=tr, color=UNSEEN, label="held out (test only)")
for i, (a, b) in enumerate(zip(tr, te)):
    ax.text(a / 2, i, f"{a}/{len(VAR)}\n{100*a/len(VAR):.0f}%", ha="center", va="center",
            color="white", fontsize=8.5, fontweight="bold")
    ax.text(a + b / 2, i, f"{b}/{len(VAR)}\n{100*b/len(VAR):.0f}%", ha="center", va="center",
            color="white", fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("instruction variants per task", fontsize=8.5)
ax.set_title("(b) Train / test share of the instruction pool", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower right"); ax.set_xlim(0, len(VAR))


def wilson(k, n, z=1.959963985):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def load(f):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    k = sum(r["success"] for r in d["per_task"]); n = sum(r["n"] for r in d["per_task"])
    return wilson(k, n)


# ---- (c) closed-loop: seen vs unseen ----------------------------------------
ax = fig.add_subplot(1, 3, 3)
bars = [
    ("SmolVLM2\ncond A", load("sroll_all_tap30_orig.json"), load("sroll_all_tap30_para1.json")),
    ("Qwen3.5\ncond A", load("qroll_all_tap12_orig.json"), load("qroll_all_tap12_para1.json")),
    ("SmolVLM2\ncond B", load("sroll_ck2mix_all_tap30_orig.json"),
     load("sroll_ck2mix_all_tap30_para3.json")),
    ("Qwen3.5\ncond B", load("qroll_qckmix_all_tap12_orig.json"),
     load("qroll_qckmix_all_tap12_para3.json")),
]
x = np.arange(len(bars)); w = 0.36
for i, (lab, s, u) in enumerate(bars):
    if s:
        ax.bar(i - w / 2, s[0], w, color=SEEN,
               yerr=[[s[0] - s[1]], [s[2] - s[0]]], capsize=3, ecolor="k")
        ax.text(i - w / 2, s[0] + 1.5, f"{s[0]:.0f}", ha="center", fontsize=7.5)
    if u:
        ax.bar(i + w / 2, u[0], w, color=UNSEEN,
               yerr=[[u[0] - u[1]], [u[2] - u[0]]], capsize=3, ecolor="k")
        ax.text(i + w / 2, u[0] + 1.5, f"{u[0]:.0f}", ha="center", fontsize=7.5)
    if s and u:
        ax.text(i, 94, f"{u[0]-s[0]:+.1f} pp", ha="center", fontsize=8,
                color="#c62828", fontweight="bold")
    if not (s and u):
        ax.text(i, 40, "pending", ha="center", fontsize=7.5, color="#888", rotation=90)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("closed-loop success rate (%)", fontsize=8.5)
ax.set_ylim(0, 100)
ax.set_title("(c) Cost of an unseen phrasing (Wilson 95% CI)", fontsize=9.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SEEN),
                   plt.Rectangle((0, 0), 1, 1, color=UNSEEN)],
          labels=["instruction SEEN in training", "instruction UNSEEN"],
          fontsize=7.5, loc="upper right")
ax.grid(alpha=.25, axis="y")

plt.tight_layout()
out = os.path.join(BASE, "figs", "fig5_instruction_split.png")
plt.savefig(out, dpi=150)
print("wrote", out)
for lab, s, u in bars:
    if s and u:
        print(f"  {lab.replace(chr(10),' '):20s} seen {s[0]:5.1f}%  unseen {u[0]:5.1f}%  "
              f"delta {u[0]-s[0]:+.1f} pp")
    else:
        print(f"  {lab.replace(chr(10),' '):20s} pending")
