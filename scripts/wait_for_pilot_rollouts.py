"""Gate production training on verified real rollout results from its pilot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_log import log_event
from scripts.runpod_eval_worker import (discover_tasks, published_result_matches,
                                       task_status_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--study", default="runpod_online_pilot_v2")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    started = time.monotonic()
    previous = None
    while True:
        tasks = [task for _, task in discover_tasks(args.durable_root)
                 if task.get("study_id") == args.study and task.get("dataset") == args.dataset]
        if len(tasks) != 2 or {t["backbone"] for t in tasks} != {"groot", "qwen3vl"}:
            raise ValueError("Pilot must publish exactly the two expected backbone tasks")
        ready = [task["task_id"] for task in tasks
                 if published_result_matches(args.durable_root, task)]
        for task in tasks:
            path = task_status_path(args.durable_root, task)
            if path.is_file():
                status = json.loads(path.read_text())
                if status.get("status") == "failed" and status.get("attempt", 0) >= 3:
                    raise RuntimeError("Pilot evaluator failed repeatedly: " + task["task_id"])
        current = tuple(sorted(ready))
        if previous != current:
            log_event("pilot.rollout_gate_progress", dataset=args.dataset,
                      completed=len(ready), expected=2, task_ids=ready)
            previous = current
        if len(ready) == 2:
            log_event("pilot.rollout_gate_passed", dataset=args.dataset, policies=2,
                      note="Runtime proof only; pilot success rates are not paper results")
            return
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError("Pilot rollout verification timed out; full study was not started")
        time.sleep(15)


if __name__ == "__main__":
    main()
