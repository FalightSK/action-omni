from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(history: Path, output: Path) -> None:
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    valid = [row["validation_loss"] for row in rows]
    best = min(range(len(valid)), key=valid.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(epochs, train, label="Train", linewidth=1.5)
    ax.plot(epochs, valid, label="Validation", linewidth=1.8)
    ax.scatter([epochs[best]], [valid[best]], color="tab:orange", zorder=3,
               label=f"Best validation: epoch {epochs[best]}")
    ax.set(title="ALOHA GR00T transfer-cube training loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Updated {output} through epoch {epochs[-1]}; best validation epoch {epochs[best]}: {valid[best]:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train and validation losses from history.jsonl")
    parser.add_argument("history", type=Path, help="Path to the run's history.jsonl")
    parser.add_argument("--output", type=Path, help="Output PNG (default: training_loss.png beside history)")
    parser.add_argument("--watch", action="store_true", help="Refresh when new epochs are logged; stop when training ends")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between checks in watch mode")
    args = parser.parse_args()

    history = args.history
    output = args.output or history.with_name("training_loss.png")
    seen_mtime = None
    while True:
        mtime = history.stat().st_mtime if history.exists() else None
        if mtime is not None and mtime != seen_mtime:
            plot(history, output)
            seen_mtime = mtime
        if not args.watch:
            break
        pipeline = history.with_name("pipeline.json")
        if pipeline.exists():
            status = json.loads(pipeline.read_text(encoding="utf-8")).get("status")
            if status not in {"training", "precompute", "running"}:
                break
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
