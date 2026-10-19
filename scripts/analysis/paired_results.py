"""Paired inference and training-seed summaries for recorded policy evaluations.

CLI input: JSON {"family_size": 6, "pairs": [{"label": "ALOHA backbone",
"left": ".../seed_000/sim_results.json", "right": ".../seed_000/sim_results.json"}, ...]}.
Paths are relative to the input manifest. No historical rates are hard-coded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def wilson(successes, n, z=1.959963984540054):
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (0. if successes == 0 else max(0., center - half),
            1. if successes == n else min(1., center + half))


def holm(pvalues, family_size=None):
    p = np.asarray(pvalues, dtype=float)
    family_size = len(p) if family_size is None else family_size
    if family_size < len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values or planned family size")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (family_size - rank) * p[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def paired_statistics(left, right, strata=None, resamples=20000, seed=20261018):
    a, b = np.asarray(left), np.asarray(right)
    if (a.ndim != 1 or a.shape != b.shape or not len(a)
            or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all()):
        raise ValueError("Need equally sized, nonempty binary paired outcomes")
    if resamples < 1:
        raise ValueError("Bootstrap resample count must be positive")
    a, b = a.astype(bool), b.astype(bool)
    diff = a.astype(float) - b.astype(float)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(only_a, only_a + only_b, p=0.5).pvalue) if only_a + only_b else 1.
    strata = np.zeros(len(a), dtype=int) if strata is None else np.asarray(strata)
    if strata.shape != a.shape:
        raise ValueError("Strata must identify every paired episode")
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    if len({len(g) for g in groups}) != 1:
        raise ValueError("Equal task weights require equal evaluation counts per task")
    rng = np.random.default_rng(seed)
    boot = np.zeros(resamples)
    # Limit temporary memory even when evaluating many starts.
    for group in groups:
        for start in range(0, resamples, 1000):
            count = min(1000, resamples - start)
            indices = rng.choice(group, size=(count, len(group)))
            boot[start:start + count] += diff[indices].mean(1) / len(groups)
    return {"n": len(a), "both_succeed": int((a & b).sum()),
            "both_fail": int((~a & ~b).sum()), "only_left": only_a, "only_right": only_b,
            "left_success_rate": float(a.mean()), "right_success_rate": float(b.mean()),
            "left_wilson95": wilson(int(a.sum()), len(a)),
            "right_wilson95": wilson(int(b.sum()), len(b)),
            "difference_pp": float(100 * diff.mean()),
            "difference_bootstrap95_pp": (100 * np.quantile(boot, [0.025, 0.975])).tolist(),
            "mcnemar_exact_p": p,
            "uncertainty_unit": "paired evaluation starts conditional on these trained policies",
            "bootstrap_resamples": resamples, "bootstrap_seed": seed}


def outcome_map(result):
    dataset = result["dataset"]
    records = result["results"] if dataset == "libero" else result["episodes"]
    mapped = {}
    for r in records:
        if dataset == "libero" and r["condition"] != "canonical":
            continue
        key = (int(r["task"]) if dataset == "libero" else 0, int(r["initial_state_id"]))
        if key in mapped:
            raise ValueError(f"Duplicate paired start {key}")
        success = r["success"] if dataset == "libero" else r["is_success"]
        if not isinstance(success, bool):
            raise ValueError("Success values must be JSON booleans")
        mapped[key] = (success, r["environment_seed"], r["policy_sampling_seed"])
    if not mapped:
        raise ValueError("No canonical evaluation episodes")
    return mapped


def compare_records(left, right, vary="backbone", **kwargs):
    for key in ("dataset", "protocol_id", "training_seed"):
        if left.get(key) is None or left.get(key) != right.get(key):
            raise ValueError(f"Unmatched/missing {key}; legacy results are not eligible")
    for r in (left, right):
        if not all(r.get(k) for k in ("checkpoint_sha256", "run_signature", "cache_id")):
            raise ValueError("Result lacks checkpoint/cache/run provenance")
    if left["checkpoint_sha256"] == right["checkpoint_sha256"]:
        raise ValueError("Cannot compare the same checkpoint with itself")
    if vary not in ("backbone", "camera"):
        raise ValueError("Comparison must vary backbone or camera")
    allowed = {"output_dir", "embeddings_cache", "model_path", "device", "embed_batch_size"}
    allowed |= {"vlm_backbone"} if vary == "backbone" else {"n_views", "img_seq_len"}
    if not left.get("configuration") or not right.get("configuration"):
        raise ValueError("Results lack executed configurations")
    configs = [{k: v for k, v in r["configuration"].items() if k not in allowed}
               for r in (left, right)]
    if configs[0] != configs[1]:
        raise ValueError("Comparison changes training/inference settings beyond its declared factor")
    a, b = outcome_map(left), outcome_map(right)
    if a.keys() != b.keys():
        raise ValueError("Unmatched evaluation starts; refusing truncation")
    keys = sorted(a)
    for key in keys:
        if a[key][1:] != b[key][1:] or a[key][2] is None:
            raise ValueError(f"Unmatched environment/policy sampling seeds at {key}")
    out = paired_statistics([a[k][0] for k in keys], [b[k][0] for k in keys],
                            strata=[k[0] for k in keys], **kwargs)
    out.update(dataset=left["dataset"], training_seed=left["training_seed"],
               protocol_id=left["protocol_id"],
               left_checkpoint=left["checkpoint_sha256"], right_checkpoint=right["checkpoint_sha256"],
               left_arm=(left["configuration"].get("vlm_backbone"), left["configuration"].get("n_views", 1)),
               right_arm=(right["configuration"].get("vlm_backbone"), right["configuration"].get("n_views", 1)))
    return out


def training_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Training summary requires finite per-policy values")
    return {"n_training_seeds": len(values), "individual_values": values.tolist(),
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None}


def export_report(directory, summaries, labels):
    """Optional manuscript-ready table/figure, only from completed result records."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def tex(value):
        replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                        "\\": r"\textbackslash{}"}
        return "".join(replacements.get(c, c) for c in value)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\caption{Completed independently trained policies. Mean and sample SD are across training seeds; intervals for episode comparisons are reported separately.}",
             r"\begin{tabular}{llrr}", r"\toprule", r"Comparison & Arm & Seeds & SR (\%) \\", r"\midrule"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(summaries), figsize=(4*len(summaries), 3.5), squeeze=False)
    for ax, (label, group) in zip(axes[0], summaries.items()):
        arm_labels = labels.get(label, {"left": "Left", "right": "Right"})
        for x, side, color in ((0, "left", "#2878b5"), (1, "right", "#e58b2b")):
            s = group[side + "_success_rate"]
            mean, sd = 100*s["mean"], None if s["sample_sd"] is None else 100*s["sample_sd"]
            display = f"{mean:.2f}" + (f" $\\pm$ {sd:.2f}" if sd is not None else " (SD unavailable)")
            lines.append(f"{tex(label)} & {tex(arm_labels[side])} & {s['n_training_seeds']} & {display} " + r"\\")
            ax.scatter(np.full(s["n_training_seeds"], x), 100*np.asarray(s["individual_values"]),
                       color=color, alpha=.75)
            ax.errorbar(x, mean, yerr=sd, fmt="_", color=color, capsize=4)
        ax.set(title=label, ylabel="Success (%)", xticks=[0, 1],
               xticklabels=[arm_labels["left"], arm_labels["right"]], ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (directory / "generated_results.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    fig.suptitle("Dots: independent training seeds; bars: mean +/- sample SD", fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path,
                    help="Also export generated_results.tex and performance.pdf from these records")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    comparisons, seen, grouped = [], set(), {}
    for entry in manifest["pairs"]:
        paths = [args.manifest.parent / entry[k] for k in ("left", "right")]
        records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        stat = compare_records(*records, vary=entry.get("vary", "backbone"))
        key = (entry["label"], stat["training_seed"])
        if key in seen:
            raise ValueError(f"Repeated training seed for comparison {key}")
        seen.add(key)
        stat["label"] = entry["label"]
        comparisons.append(stat)
        grouped.setdefault(entry["label"], []).append(stat)
    if not comparisons:
        raise ValueError("No completed comparisons provided")
    adjusted = holm([s["mcnemar_exact_p"] for s in comparisons], manifest["family_size"])
    for stat, p in zip(comparisons, adjusted):
        stat["holm_adjusted_p"] = p
    summaries = {}
    for label, group in grouped.items():
        if len({(g["dataset"], g["protocol_id"], g["n"], g["left_arm"], g["right_arm"]) for g in group}) != 1:
            raise ValueError("Cannot summarize heterogeneous protocols/evaluation counts")
        summaries[label] = {key: training_summary([g[key] for g in group]) for key in (
            "left_success_rate", "right_success_rate", "difference_pp")}
    from experiment_protocol import write_json
    write_json(args.output, {"family_size": manifest["family_size"], "pairs": comparisons,
                             "training_summaries": summaries,
                             "note": "Episode tests do not establish an effect across the population of trained policies."})
    if args.report_dir:
        export_report(args.report_dir, summaries, manifest.get("labels", {}))
    print(f"Wrote {len(comparisons)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
