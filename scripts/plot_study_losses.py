"""Plot training/validation loss by optimizer update and align it with rollouts."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def plot_run_losses(run_dir: Path) -> Path | None:
    """Write one per-policy train/validation curve from its epoch history.

    Success-stopping histories also contain rollout-only rows, and a resumed
    run may repeat an epoch. Ignore rows without both losses and keep the last
    valid loss row for each epoch.
    """
    run_dir = Path(run_dir)
    history = run_dir / "history.jsonl"
    if not history.is_file():
        return None

    by_epoch: dict[int, tuple[float, float]] = {}
    with history.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A live reader may observe the final JSONL append mid-write.
                continue
            if not isinstance(row, dict) or not {"epoch", "train_loss", "validation_loss"}.issubset(row):
                continue
            try:
                epoch = int(row["epoch"])
                train_loss = float(row["train_loss"])
                validation_loss = float(row["validation_loss"])
            except (TypeError, ValueError, OverflowError):
                continue
            if epoch < 0 or not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                continue
            by_epoch[epoch] = (train_loss, validation_loss)

    if not by_epoch:
        return None

    epochs = sorted(by_epoch)
    train_losses = [by_epoch[epoch][0] for epoch in epochs]
    validation_losses = [by_epoch[epoch][1] for epoch in epochs]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax.plot(epochs, validation_losses, label="Validation", linewidth=1.8)
    ax.set(title=f"{run_dir.parent.name} / {run_dir.name}: training and validation loss",
           xlabel="Epoch", ylabel="Flow-matching loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output = run_dir / "training_curve.png"
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        fig.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    from configs.registry import get_config
    from experiment_protocol import write_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    ap.add_argument("--run-root", type=Path, default=ROOT / "asset/runs/runpod_update_matched_v1")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    alignment = []
    paired_directions = []
    summaries = {}
    for dataset, settings in plan["datasets"].items():
        series = {}
        rows_by_policy = {}
        for exp in (plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]):
            cfg = get_config(dataset, exp)
            name = Path(cfg.output_dir).name
            for seed in plan["training_seeds"]:
                run = run_root / dataset / name / f"seed_{seed:03d}"
                manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                if manifest.get("status") != "complete" or manifest.get("optimizer_steps") != plan["optimizer_updates"]:
                    raise ValueError(f"Incomplete fixed-update policy: {run}")
                rows = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
                by_step = {row["global_step"]: row for row in rows}
                series[(cfg.vlm_backbone, seed)] = (run, by_step)
                rows_by_policy[(cfg.vlm_backbone, seed)] = by_step

        common_steps = sorted(set.intersection(*(set(v[1]) for v in series.values())))
        if not common_steps or common_steps[-1] != plan["optimizer_updates"]:
            raise ValueError(f"Loss histories do not align on the full update horizon: {dataset}")
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        dataset_summaries = {}
        for ax, field, label in zip(axes, ("train_loss", "validation_loss"),
                                    ("Training loss", "Validation loss")):
            for backbone, pretty, color in (("groot", "GR00T N1.7", "tab:blue"),
                                            ("qwen3vl", "Qwen3-VL-2B", "tab:orange")):
                matrix = np.asarray([[rows_by_policy[(backbone, seed)][step][field]
                                      for step in common_steps]
                                     for seed in plan["training_seeds"]], dtype=float)
                mean = matrix.mean(axis=0)
                sd = matrix.std(axis=0, ddof=1)
                ax.plot(common_steps, mean, color=color, label=pretty)
                ax.fill_between(common_steps, mean - sd, mean + sd, color=color, alpha=0.18)
                dataset_summaries.setdefault(backbone, {})[field] = {
                    "training_seeds": plan["training_seeds"],
                    "updates": common_steps,
                    "mean": mean.tolist(), "sample_sd": sd.tolist(),
                    "per_seed": {str(seed): [rows_by_policy[(backbone, seed)][step][field]
                                               for step in common_steps]
                                 for seed in plan["training_seeds"]}}
            for step in settings["probe_updates"]:
                ax.axvline(step, color="0.5", alpha=0.2, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel("Optimizer update")
        view_text = "two cameras" if dataset == "libero" else "native top camera"
        fig.suptitle(f"{dataset.upper()} ({view_text}): mean ± sample SD across independent seeds")
        fig.savefig(run_root / f"loss_by_update_{dataset}.png", dpi=180)
        plt.close(fig)
        summaries[dataset] = dataset_summaries

        checkpoints = [(step, False) for step in settings["probe_updates"]]
        checkpoints.append((plan["optimizer_updates"], True))
        for step, final in checkpoints:
            per_backbone = {}
            for backbone in ("groot", "qwen3vl"):
                for seed in plan["training_seeds"]:
                    run, history = series[(backbone, seed)]
                    eval_file = run / ("evaluation_final.json" if final else f"evaluation_step_{step:08d}.json")
                    result = json.loads(eval_file.read_text(encoding="utf-8"))
                    if (result.get("training_seed") != seed
                            or result.get("optimizer_step") != step
                            or result.get("protocol_id") != json.loads((run / "run_manifest.json").read_text())["protocol_id"]):
                        raise ValueError(f"Evaluation/loss provenance mismatch: {eval_file}")
                    if dataset == "aloha":
                        successes = sum(bool(ep["is_success"]) for ep in result["episodes"])
                        n = len(result["episodes"])
                    else:
                        successes = sum(bool(ep["success"]) for ep in result["results"])
                        n = len(result["results"])
                    loss = history[step]
                    record = {"dataset": dataset, "backbone": backbone, "training_seed": seed,
                              "optimizer_step": step, "training_loss": loss["train_loss"],
                              "validation_loss": loss["validation_loss"],
                              "successes": successes, "episodes": n, "success_rate": successes / n,
                              "evaluation": str(eval_file)}
                    alignment.append(record)
                    per_backbone.setdefault(backbone, []).append(record)
            for seed in plan["training_seeds"]:
                groot = next(r for r in per_backbone["groot"] if r["training_seed"] == seed)
                qwen = next(r for r in per_backbone["qwen3vl"] if r["training_seed"] == seed)
                # A lower offline validation loss predicts the closed-loop winner only
                # when their pairwise directions agree; this is descriptive per seed.
                loss_gap = qwen["validation_loss"] - groot["validation_loss"]
                success_gap = qwen["success_rate"] - groot["success_rate"]
                pair = {"dataset": dataset, "optimizer_step": step, "training_seed": seed,
                        "qwen_minus_groot_validation_loss": loss_gap,
                        "qwen_minus_groot_success_rate": success_gap,
                        "loss_ranks_closed_loop_winner": (loss_gap * success_gap < 0
                            if loss_gap != 0 and success_gap != 0 else None)}
                paired_directions.append(pair)

    write_json(run_root / "loss_success_alignment.json", {"policy_rows": alignment,
               "paired_directions": paired_directions,
               "note": "Checkpoint rows are repeated observations within each trained policy; seed-pair direction checks are descriptive."})
    write_json(run_root / "loss_summary.json", summaries)


if __name__ == "__main__":
    main()
