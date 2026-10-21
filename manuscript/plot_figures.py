"""Rebuild camera-ready figures from the verified 2026-10-21 result bundle."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "tmp/experiment_result/action_omni_experiment_results_2026-10-21"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def read_csv(relative):
    with (SOURCE / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


with (SOURCE / "manifest_sha256.csv").open(newline="", encoding="utf-8-sig") as handle:
    manifest = list(csv.DictReader(handle))
for item in manifest:
    path = SOURCE / item["file"]
    assert path.stat().st_size == int(item["bytes"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

aloha = read_csv("ALOHA/development/measured_results.csv")
libero = read_csv("LIBERO/final/measured_results.csv")
tasks = read_csv("LIBERO/final/per_task_results.csv")
losses = json.loads((SOURCE / "ALOHA/development/training_loss_pairs.json").read_text(encoding="utf-8"))["records"]
assert len(aloha) == len(losses) == 37 and len(libero) == 3
assert {(int(r["seed"]), r["exp"], int(r["epoch"])) for r in losses} == {
    (int(r["training_seed"]), "exp05" if r["backbone"] == "groot" else "exp06", int(r["epoch"]))
    for r in aloha
}
for final in libero:
    subset = [r for r in tasks if r["training_seed"] == final["training_seed"] and r["backbone"] == final["backbone"]]
    assert len(subset) == 10 and sum(int(r["successes"]) for r in subset) == int(final["successes"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
# Predefined first and final measured checkpoints: a direct loss/success contrast.
rows = [(3, "groot", "exp05"), (3, "qwen3vl", "exp06"),
        (4, "groot", "exp05"), (4, "qwen3vl", "exp06")]
first, last = [], []
for seed, backbone, exp in rows:
    pair = []
    for epoch in (50, 250):
        loss = next(r for r in losses if r["seed"] == seed and r["exp"] == exp and r["epoch"] == epoch)
        result = next(r for r in aloha if int(r["training_seed"]) == seed
                      and r["backbone"] == backbone and int(r["epoch"]) == epoch)
        assert int(result["episodes"]) == 50
        pair.append((loss["train_loss"], loss["validation_loss"], 100 * float(result["success_rate"])))
    first.append(pair[0])
    last.append(pair[1])

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), sharey=True)
ys = [3, 2, 1, 0]
names = ["GR00T seed 3", "Qwen seed 3", "GR00T seed 4", "Qwen seed 4"]
for j, (ax, title, limit, fmt) in enumerate(zip(
        axes, ("Training loss", "Validation loss", "Development success"),
        (.085, .50, 95), ("{:.3f}", "{:.3f}", "{:.0f}%"))):
    a = [r[j] for r in first]
    b = [r[j] for r in last]
    ax.barh([y + .16 for y in ys], a, height=.28, color="#8a9ba6", label="epoch 50")
    ax.barh([y - .16 for y in ys], b, height=.28, color="#2166ac", label="epoch 250")
    for y, v0, v1 in zip(ys, a, b):
        ax.text(v0 + limit * .012, y + .16, fmt.format(v0), va="center", fontsize=6.5)
        ax.text(v1 + limit * .012, y - .16, fmt.format(v1), va="center", fontsize=6.5)
    ax.axhline(1.5, color="#dddddd", lw=.7)
    ax.set(title=title, xlim=(0, limit), yticks=ys, yticklabels=names)
    ax.grid(axis="x", color="#e5e5e5", lw=.5)
    ax.set_axisbelow(True)
axes[0].tick_params(axis="y", length=0)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(.5, 1.08), ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.0)
fig.savefig(OUT / "aloha_epoch50_vs250.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "aloha_epoch50_vs250.pdf", bbox_inches="tight")
plt.close(fig)

# Task-level held-out contrast for the only shared LIBERO training seed.
seed3 = {(r["backbone"], int(r["task"])): int(r["successes"])
         for r in tasks if int(r["training_seed"]) == 3}
assert len(seed3) == 20
task_diffs = [seed3["Qwen", i] - seed3["GR00T", i] for i in range(10)]
assert sum(task_diffs) == 366 - 337 == 29
fig, ax = plt.subplots(figsize=(3.45, 2.35))
ys = list(range(9, -1, -1))
colors = ["#2166ac" if d > 0 else "#b35806" if d < 0 else "#8a9ba6"
          for d in task_diffs]
ax.barh(ys, task_diffs, height=.65, color=colors)
ax.axvline(0, color="#444444", lw=.8)
ax.set(yticks=ys, yticklabels=[f"Task {i}" for i in range(10)],
       xlim=(-5, 12), xticks=(-4, 0, 4, 8, 12),
       xlabel="Qwen minus GR00T successes (of 40)")
ax.grid(axis="x", color="#e5e5e5", lw=.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
for y, d in zip(ys, task_diffs):
    if d:
        ax.text(d + (.3 if d > 0 else -.3), y, f"{d:+d}",
                ha="left" if d > 0 else "right", va="center", fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT / "libero_seed3_task_difference.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "libero_seed3_task_difference.pdf", bbox_inches="tight")
plt.close(fig)

print("Verified manifest; plotted ALOHA checkpoint pairs and held-out LIBERO seed-3 task differences.")
