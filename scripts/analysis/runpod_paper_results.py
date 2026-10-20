"""Incrementally build paper evidence from the exact locked RunPod study.

Only final checkpoints whose task ``plan_id`` equals the supplied plan digest
enter the confirmatory comparison family. Probe checkpoints are inventoried
separately as exploratory. Episode uncertainty is reported within each trained
policy; training-seed variability is summarized separately across seeds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment_protocol import digest, file_digest, write_json
from scripts.analysis.paired_results import compare_records, holm, training_summary
from scripts.success_queue import validate_rollout_result


METRIC_KEYS = ("flow_matching_loss", "action_mse", "action_mae",
               "per_dimension_action_mse", "per_dimension_action_mae",
               "n_samples", "protocol_id", "split", "validation_rng_seed",
               "action_sampling_seed_base")


def summarize_independent_seeds(values: list[float], planned_seeds: list[int],
                                observed_seeds: list[int]) -> dict:
    """Summarize one value per independent policy, withholding SD below n=3."""
    if (len(values) != len(observed_seeds) or len(set(observed_seeds)) != len(observed_seeds)
            or any(seed not in planned_seeds for seed in observed_seeds)):
        raise ValueError("Each value must map to one unique planned training seed")
    observed = set(observed_seeds)
    missing = [seed for seed in planned_seeds if seed not in observed]
    if values:
        summary = training_summary(values)
    else:
        summary = {"n_training_seeds": 0, "individual_values": [],
                   "mean": None, "sample_sd": None}
    # The locked paper table requires all three independent seeds before SD is
    # reported, even though two values would mathematically define a sample SD.
    if len(observed) < 3:
        summary["sample_sd"] = None
    summary.update(status="complete" if not missing else "incomplete",
                   planned_training_seeds=list(planned_seeds),
                   observed_training_seeds=[seed for seed in planned_seeds if seed in observed],
                   missing_training_seeds=missing,
                   sample_sd_status=("available" if summary["sample_sd"] is not None
                                     else "unavailable; fewer than 3 complete independent seeds"))
    return summary


def safe_relative(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative task/result path: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escaped durable root: {relative!r}")
    return resolved


def read_task_results(durable_root: Path, plan: dict, plan_id: str) -> tuple[dict, list[dict], list[str]]:
    indexed: dict[tuple[str, str, int, str], dict] = {}
    probes: list[dict] = []
    errors: list[str] = []
    task_root = durable_root / "evaluation_tasks"
    if not task_root.is_dir():
        return indexed, probes, errors
    for task_path in sorted(task_root.rglob("*.task.json")):
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("plan_id") != plan_id:
                continue
            checkpoint_meta = task.get("checkpoint") or {}
            checkpoint_path = safe_relative(durable_root, checkpoint_meta.get("path", ""))
            checkpoint_valid = (checkpoint_path.is_file()
                                and checkpoint_path.stat().st_size == checkpoint_meta.get("size_bytes"))
            # Final artifacts are few and are the confirmatory evidence. Development
            # checkpoints can be numerous; their evaluator verified the SHA before
            # publishing each result, so avoid rereading every large file on each
            # incremental report refresh.
            if checkpoint_valid and task.get("is_final"):
                checkpoint_valid = file_digest(checkpoint_path) == checkpoint_meta.get("sha256")
            result_path = safe_relative(durable_root, task["result_path"])
            result = (json.loads(result_path.read_text(encoding="utf-8"))
                      if result_path.is_file() and checkpoint_valid else None)
            entry = {"task_id": task.get("task_id"), "task_file": task_path.relative_to(durable_root).as_posix(),
                     "dataset": task.get("dataset"), "experiment": task.get("experiment"),
                     "backbone": task.get("backbone"), "training_seed": task.get("training_seed"),
                     "checkpoint_tag": task.get("checkpoint_tag"), "optimizer_step": task.get("optimizer_step"),
                     "protocol_id": task.get("protocol_id"),
                     "run_signature": task.get("run_signature"),
                     "checkpoint_sha256": (task.get("checkpoint") or {}).get("sha256"),
                     "purpose": task.get("purpose"), "is_final": task.get("is_final"),
                     "selection_epoch": task.get("selection_epoch"),
                     "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256"),
                     "result_path": task.get("result_path"),
                     "validation_metrics_sha256": task.get("validation_metrics_sha256"),
                     "trajectory_path": task.get("trajectory_path"),
                     "trajectory_sha256": None,
                     "status": ("invalid_checkpoint_artifact" if not checkpoint_valid else
                                "pending_evaluation" if result is None else "invalid_result"),
                     "error": None if checkpoint_valid else "published checkpoint missing, wrong size, or wrong SHA-256",
                     "result": None}
            if result is not None:
                expected = {"task_id": task.get("task_id"), "study_id": task.get("study_id"),
                            "plan_id": plan_id, "dataset": task.get("dataset"),
                            "exp_id": task.get("experiment"), "backbone": task.get("backbone"),
                            "training_seed": task.get("training_seed"),
                            "checkpoint_tag": task.get("checkpoint_tag"),
                            "checkpoint_sha256": (task.get("checkpoint") or {}).get("sha256"),
                            "protocol_id": task.get("protocol_id"),
                            "run_signature": task.get("run_signature"),
                            "feature_mode": task.get("feature_mode"),
                            "validation_metrics_sha256": task.get("validation_metrics_sha256"),
                            "purpose": task.get("purpose"), "is_final": task.get("is_final"),
                            "selection_epoch": task.get("selection_epoch"),
                            "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256")}
                if any(result.get(k) != v for k, v in expected.items()):
                    entry["status"] = "invalid_result_identity"
                elif task.get("trajectory_path") and (
                        (result.get("trajectory") or {}).get("path") != task["trajectory_path"]):
                    entry["status"] = "missing_trajectory_pointer"
                elif task.get("trajectory_path") and not _trajectory_matches(
                        durable_root, task["trajectory_path"], result.get("trajectory") or {}):
                    entry["status"] = "invalid_trajectory_artifact"
                elif not task.get("trajectory_path") and result.get("trajectory") is not None:
                    entry["status"] = "unexpected_trajectory_artifact"
                else:
                    try:
                        if task.get("purpose") in {"development_selection", "final_test"}:
                            result = validate_rollout_result(task, result)
                        rate, successes, denominator = final_success_rate(result, task)
                        entry.update(status="complete", result=result,
                                     success_rate=rate, successes=successes,
                                     n_evaluation_episodes=denominator)
                        entry["trajectory_sha256"] = (result.get("trajectory") or {}).get("sha256")
                    except Exception as exc:
                        entry["status"] = "invalid_result_metrics"
                        entry["error"] = f"{type(exc).__name__}: {exc}"
            if not checkpoint_valid:
                errors.append(f"Invalid published checkpoint for task {task.get('task_id')}")
            if task.get("is_final"):
                key = (str(task["dataset"]), str(task["experiment"]),
                       int(task["training_seed"]), str(task["checkpoint_tag"]))
                if key in indexed:
                    errors.append(f"Duplicate final task identity {key}")
                    indexed[key]["status"] = "ambiguous_task_identity"
                    indexed[key]["result"] = None
                else:
                    indexed[key] = {**entry, "task": task}
            else:
                probes.append(entry)
        except Exception as exc:
            errors.append(f"{task_path.relative_to(durable_root).as_posix()}: {type(exc).__name__}: {exc}")
    return indexed, probes, errors


def _trajectory_matches(durable_root: Path, relative_path: str, pointer: dict) -> bool:
    try:
        path = safe_relative(durable_root, relative_path)
        return (path.is_file() and path.stat().st_size == pointer.get("size_bytes")
                and file_digest(path) == pointer.get("sha256"))
    except (OSError, TypeError, ValueError):
        return False


def final_success_rate(result: dict, task: dict) -> tuple[float, int, int]:
    evaluation = task["evaluation"]
    if task["dataset"] == "aloha":
        outcomes = [row.get("is_success") for row in result.get("episodes", [])]
        if len(outcomes) != evaluation["episodes"] or any(not isinstance(x, bool) for x in outcomes):
            raise ValueError("ALOHA result has incomplete or nonboolean outcomes")
        return sum(outcomes) / len(outcomes), sum(outcomes), len(outcomes)
    rows = [row for row in result.get("results", []) if row.get("condition") == "canonical"]
    tasks = evaluation["tasks"]
    by_task = {idx: [] for idx in tasks}
    for row in rows:
        if row.get("task") in by_task and isinstance(row.get("success"), bool):
            by_task[row["task"]].append(row["success"])
    if any(len(by_task[idx]) != evaluation["episodes_per_task"] for idx in tasks):
        raise ValueError("LIBERO result has incomplete canonical task outcomes")
    rate = sum(sum(by_task[idx]) / len(by_task[idx]) for idx in tasks) / len(tasks)
    return rate, sum(sum(by_task[idx]) for idx in tasks), sum(len(by_task[idx]) for idx in tasks)


def valid_metrics(task: dict, result: dict) -> tuple[dict | None, str | None]:
    metrics = result.get("validation_metrics")
    metrics_sha = result.get("validation_metrics_sha256")
    if not isinstance(metrics, dict):
        return None, "checkpoint did not publish validation metrics"
    if metrics != task.get("validation_metrics"):
        return None, "result validation metrics differ from the task's checkpoint metrics"
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        return None, f"validation metrics lack required fields: {missing}"
    if digest(metrics) != metrics_sha or metrics_sha != task.get("validation_metrics_sha256"):
        return None, "validation metrics digest does not match task/checkpoint provenance"
    if (metrics.get("protocol_id") != task.get("protocol_id")
            or metrics.get("split") != "validation_demonstrations"
            or metrics.get("validation_rng_seed") != 4242):
        return None, "validation metrics have wrong protocol/split or no samples"
    if (not isinstance(metrics.get("n_samples"), int)
            or isinstance(metrics.get("n_samples"), bool) or metrics["n_samples"] <= 0
            or not isinstance(metrics.get("action_sampling_seed_base"), int)
            or isinstance(metrics.get("action_sampling_seed_base"), bool)):
        return None, "validation metric counts/seeds have invalid types or values"
    for key in ("flow_matching_loss", "action_mse", "action_mae"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, f"validation metric {key} is not finite numeric data"
    for key in ("per_dimension_action_mse", "per_dimension_action_mae"):
        values = metrics.get(key)
        if (not isinstance(values, list) or not values
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in values)):
            return None, f"validation metric {key} is not a finite numeric list"
    return metrics, None


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                          else value for key, value in row.items()}
            writer.writerow(normalized)


def render_markdown(plan_id: str, inventory: dict, comparisons: list[dict], summaries: dict,
                    rq2: list[dict], probes: list[dict], errors: list[str]) -> str:
    lines = ["# RunPod paper evidence", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Plan digest: `{plan_id}`", "",
             "This report uses only tasks whose `plan_id` exactly matches the supplied locked plan. "
             "Episode intervals/tests are conditional on the evaluated trained policies; they do not "
             "establish an effect across the population of trained policies. Training-seed mean and "
             "sample SD are reported separately. Development and final tests use disjoint simulator reset cohorts "
              "on the same tasks, so this is not unseen-task generalization. Probe checkpoints are exploratory "
              "and excluded from the final six-comparison family. Development-selection snapshots are repeated "
              "evaluations of the same trained policy, not independent runs; they stay separate from final-cohort "
              "success and are not pooled for RQ2 correlations.", "",
             f"Final policies complete: **{inventory['complete_final_policies']}/{inventory['planned_final_policies']}**. "
             f"Paired seed comparisons complete: **{inventory['complete_final_pairs']}/{inventory['planned_final_pairs']}**.", "",
             "## Independent training-seed summaries", "",
              "| Comparison | Arm | Complete seeds / 3 | Missing seeds | Success rate, mean ± sample SD (%) |",
              "|---|---|---:|---|---:|"]
    for label, group in summaries.items():
        for side in ("left", "right"):
            summary = group[side]
            mean = summary.get("mean")
            sd = summary.get("sample_sd")
            display = "pending" if mean is None else f"{100*mean:.2f} ± {100*sd:.2f}" if sd is not None else f"{100*mean:.2f} ± unavailable"
            missing = ", ".join(map(str, summary.get("missing_training_seeds", []))) or "none"
            lines.append(f"| {label} | {summary['arm']} | {summary['n_training_seeds']}/3 | {missing} | {display} |")
    lines += ["", "## Paired differences across independent training seeds", "",
              "| Comparison | Complete paired seeds / 3 | Missing paired seeds | Mean ± sample SD (percentage points) |",
              "|---|---:|---|---:|"]
    for label, group in summaries.items():
        difference = group.get("difference_pp") or {}
        mean, sd = difference.get("mean"), difference.get("sample_sd")
        display = ("pending" if mean is None else f"{mean:.2f} ± {sd:.2f}"
                   if sd is not None else f"{mean:.2f} ± unavailable")
        missing = ", ".join(map(str, difference.get("missing_training_seeds", []))) or "none"
        lines.append(f"| {label} | {group['complete_paired_seeds']}/3 | {missing} | {display} |")
    lines += ["", "## Per-seed paired rollout comparisons", "",
              "Each row uses exact paired starts and reports exact McNemar p and a paired-start bootstrap interval; "
              "the Holm family size is fixed at six planned final comparisons.", "",
              "| Comparison | Seed | Status | Left SR (%) | Right SR (%) | Difference (pp), bootstrap 95% | McNemar exact p | Holm p |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        stats = row.get("statistics") or {}
        ci = stats.get("difference_bootstrap95_pp")
        lines.append(f"| {row['label']} | {row['training_seed']} | {row['status']} | "
                     f"{100*stats['left_success_rate']:.2f} | {100*stats['right_success_rate']:.2f} | "
                     f"{ci[0]:.2f} to {ci[1]:.2f} | {stats['mcnemar_exact_p']:.5g} | "
                     f"{stats['holm_adjusted_p']:.5g} |" if stats else
                     f"| {row['label']} | {row['training_seed']} | {row['status']} | pending | pending | pending | pending | pending |")
    lines += ["", "## RQ2: validation metrics matched to final-checkpoint success", "",
              "Each row reports one final checkpoint per independently trained policy: checkpoint-local validation "
              "metrics beside that checkpoint's final-evaluation-cohort simulator success. The development cohort "
              "was used for checkpoint selection on the same tasks with disjoint reset IDs; its repeated checkpoint "
              "scores are selection evidence, not final outcomes or independent runs. No pooled Pearson inference "
              "or causal claim is made.", "",
              "| Dataset | Backbone | Seed | Final status | Final SR (%) | Selection SR (%) | Flow loss | Action MSE | Action MAE | Samples | Metrics SHA-256 |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rq2:
        success_text = ("pending" if row.get("success_rate") is None
                        else f"{100 * row['success_rate']:.2f}")
        selection_rate = row.get("selection_cohort_success_rate")
        selection_text = ("n/a" if row.get("selection_cohort_status") == "not_applicable" else
                          "pending" if selection_rate is None
                          else f"{100 * selection_rate:.2f}")
        lines.append(f"| {row['dataset']} | {row['backbone']} | {row['training_seed']} | {row['status']} | "
                     f"{success_text} | {selection_text} | "
                     f"{row.get('flow_matching_loss', 'pending')} | {row.get('action_mse', 'pending')} | "
                     f"{row.get('action_mae', 'pending')} | {row.get('n_samples', 'pending')} | "
                     f"{row.get('validation_metrics_sha256') or 'pending'} |")
    lines += ["", "## Pending and exploratory inventory", "",
             f"Development selection tasks found: {inventory['development_task_count']}; these are selection evidence, "
             "not final-test comparisons.",
             f"Other exploratory probe tasks found: {len(probes)}. They are excluded from all final summaries.", ""]
    if inventory["development_selection_inventory"]:
        lines += ["", "| Dataset | Backbone | Seed | Epoch | Status | Development SR (%) | Checkpoint SHA-256 |",
                  "|---|---|---:|---:|---|---:|---|"]
        for item in inventory["development_selection_inventory"]:
            rate = item.get("success_rate")
            lines.append(f"| {item.get('dataset')} | {item.get('experiment')} | {item.get('training_seed')} | "
                         f"{item.get('selection_epoch')} | {item.get('status')} | "
                         f"{'pending' if rate is None else f'{100 * rate:.2f}'} | "
                         f"{item.get('checkpoint_sha256') or 'pending'} |")
    pending = [item for item in inventory["final_inventory"] if item["status"] != "complete"]
    if pending:
        lines.append("Pending final tasks:")
        lines.extend(f"- {item['dataset']} / {item['experiment']} / seed {item['training_seed']}: {item['status']}"
                     for item in pending)
    else:
        lines.append("All planned final tasks have matching evaluation results.")
    if errors:
        lines += ["", "Inventory errors:", *(f"- {error}" for error in errors)]
    return "\n".join(lines) + "\n"


def build_report(durable_root: Path, plan_path: Path, output_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_id = digest(plan)
    seeds = plan["training_seeds"]
    if seeds != [3, 4, 5]:
        raise ValueError(f"Paper report is locked to independent training seeds [3,4,5], got {seeds}")
    indexed, probes, errors = read_task_results(durable_root, plan, plan_id)
    datasets = list(plan["datasets"])
    arms = [plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]]
    if set(datasets) != {"aloha", "libero"} or len(set(arms)) != 2:
        raise ValueError("Paper finalization requires both planned datasets and two distinct backbone arms")
    final_inventory, rq2_rows, comparisons = [], [], []
    is_success_stopped = plan.get("schema") == 2 and plan.get("checkpoint_rule") == \
        "highest_development_success_rate_earliest_tie"
    final_tag = "final_test" if is_success_stopped else "final"
    development_inventory = [entry for entry in probes if entry.get("purpose") == "development_selection"]
    other_probes = [entry for entry in probes if entry.get("purpose") != "development_selection"]
    development_by_policy: dict[tuple[str, str, int, int], dict] = {}
    for entry in development_inventory:
        key = (str(entry.get("dataset")), str(entry.get("experiment")),
               int(entry.get("training_seed") or -1), int(entry.get("selection_epoch") or -1))
        if key in development_by_policy:
            errors.append(f"Duplicate development checkpoint task identity {key}")
            development_by_policy[key] = {**entry, "status": "ambiguous_task_identity", "result": None}
        else:
            development_by_policy[key] = entry
    valid_results: dict[tuple[str, str, int], tuple[dict, dict]] = {}
    policy_success_rates: dict[tuple[str, str, int], float] = {}
    for dataset in datasets:
        for seed in seeds:
            for experiment in arms:
                key = (dataset, experiment, seed, final_tag)
                indexed_entry = indexed.get(key)
                if indexed_entry is None:
                    item = {"dataset": dataset, "experiment": experiment,
                            "backbone": "groot" if experiment == arms[0] else "qwen3vl",
                            "training_seed": seed, "checkpoint_tag": final_tag,
                            "status": "pending_checkpoint_task"}
                    result, task = None, None
                else:
                    item = {key: value for key, value in indexed_entry.items() if key not in {"result", "task"}}
                    task = indexed_entry["task"]
                    result = indexed_entry["result"] if indexed_entry["status"] == "complete" else None
                final_inventory.append(item)
                rq2 = {"dataset": dataset, "experiment": experiment,
                       "backbone": item["backbone"], "training_seed": seed,
                       "checkpoint_tag": item.get("checkpoint_tag"),
                       "purpose": task.get("purpose") if task else None,
                       "evaluation_cohort": "final_evaluation" if task else None,
                       "selection_cohort_success_rate": None,
                       "selection_cohort_status": "pending" if is_success_stopped else "not_applicable",
                       "task_id": item.get("task_id"), "checkpoint_sha256": item.get("checkpoint_sha256"),
                       "validation_metrics_sha256": item.get("validation_metrics_sha256"),
                       "status": ("pending_evaluation" if indexed_entry is None else
                                  "pending_validation_metrics" if indexed_entry["status"] == "complete" else
                                  "invalid_evaluation_result" if indexed_entry["status"].startswith("invalid") else
                                  indexed_entry["status"])}
                if result is not None and task is not None:
                    try:
                        success_rate, successes, n = final_success_rate(result, task)
                        if is_success_stopped:
                            selected_epoch = task.get("selection_epoch")
                            dev = development_by_policy.get((dataset, experiment, seed,
                                                             int(selected_epoch or -1)))
                            if (dev is None or dev.get("status") != "complete"
                                    or dev.get("checkpoint_sha256") != task.get("selected_checkpoint_sha256")
                                    or task.get("selected_checkpoint_sha256") != task.get("checkpoint", {}).get("sha256")):
                                raise ValueError("Final test does not link to the complete selected development checkpoint")
                            if dev.get("success_rate") is None:
                                raise ValueError("Selected development task lacks a verified success rate")
                            item["selection_epoch"] = selected_epoch
                            item["selected_checkpoint_sha256"] = task.get("selected_checkpoint_sha256")
                            item["selected_development_success_rate"] = dev.get("success_rate")
                            item["selected_development_task_id"] = dev.get("task_id")
                            rq2["selection_cohort_success_rate"] = dev.get("success_rate")
                            rq2["selection_cohort_status"] = "complete"
                        item.update(status="complete", success_rate=success_rate,
                                    successes=successes, n_evaluation_episodes=n)
                        valid_results[(dataset, experiment, seed)] = (result, task)
                        policy_success_rates[(dataset, experiment, seed)] = success_rate
                        metrics, metrics_error = valid_metrics(task, result)
                        if metrics is not None:
                            rq2.update(status="complete", success_rate=success_rate,
                                       flow_matching_loss=metrics["flow_matching_loss"],
                                       action_mse=metrics["action_mse"], action_mae=metrics["action_mae"],
                                       per_dimension_action_mse=metrics["per_dimension_action_mse"],
                                       per_dimension_action_mae=metrics["per_dimension_action_mae"],
                                       n_samples=metrics["n_samples"],
                                       validation_rng_seed=metrics["validation_rng_seed"],
                                       action_sampling_seed_base=metrics["action_sampling_seed_base"])
                        else:
                            rq2["status"] = "pending_validation_metrics"
                            rq2["reason"] = metrics_error
                    except Exception as exc:
                        item["status"] = "invalid_result_metrics"
                        item["error"] = f"{type(exc).__name__}: {exc}"
                        rq2["status"] = "invalid_evaluation_result"
                        rq2["reason"] = item["error"]
                rq2_rows.append(rq2)

    rq2_keys = [(row["dataset"], row["experiment"], row["training_seed"]) for row in rq2_rows]
    planned_policy_count = len(datasets) * len(arms) * len(seeds)
    if len(rq2_keys) != planned_policy_count or len(set(rq2_keys)) != planned_policy_count:
        raise RuntimeError("RQ2 must contain exactly one final-checkpoint row per planned trained policy")

    final_policy_status = {(item["dataset"], item["experiment"], item["training_seed"]): item["status"]
                           for item in final_inventory}
    for dataset in datasets:
        label = f"{dataset.upper()} backbone"
        for seed in seeds:
            left_key, right_key = (dataset, arms[0], seed), (dataset, arms[1], seed)
            left = valid_results.get(left_key)
            right = valid_results.get(right_key)
            row = {"label": label, "dataset": dataset, "training_seed": seed,
                   "status": "complete" if left and right else "pending_pair",
                   "left_experiment": arms[0], "right_experiment": arms[1],
                   "left_status": final_policy_status.get((dataset, arms[0], seed), "missing_policy"),
                   "right_status": final_policy_status.get((dataset, arms[1], seed), "missing_policy"),
                   "left_checkpoint_sha256": left[0].get("checkpoint_sha256") if left else None,
                   "right_checkpoint_sha256": right[0].get("checkpoint_sha256") if right else None}
            if left and right:
                try:
                    stats = compare_records(left[0], right[0], vary="backbone")
                    row["statistics"] = stats
                except Exception as exc:
                    row["status"] = "invalid_pair"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            comparisons.append(row)
    completed_stats = [row["statistics"] for row in comparisons if row.get("statistics")]
    adjusted = holm([row["mcnemar_exact_p"] for row in completed_stats], family_size=6)
    for stat, adjusted_p in zip(completed_stats, adjusted):
        stat["holm_adjusted_p"] = adjusted_p

    summaries = {}
    for dataset in datasets:
        label = f"{dataset.upper()} backbone"
        paired_rows = [row for row in comparisons if row["dataset"] == dataset and row.get("statistics")]
        arms_summary = {}
        for side, experiment in (("left", arms[0]), ("right", arms[1])):
            observed_seeds = [seed for seed in seeds
                              if (dataset, experiment, seed) in policy_success_rates]
            values = [policy_success_rates[(dataset, experiment, seed)] for seed in observed_seeds]
            summary = summarize_independent_seeds(values, seeds, observed_seeds)
            arms_summary[side] = {"arm": "GR00T N1.7" if experiment == arms[0] else "Qwen3-VL-2B",
                                  **summary, "complete_across_all_seeds": not summary["missing_training_seeds"],
                                  "n_paired_training_seeds": len(paired_rows)}
        diffs = [row["statistics"]["difference_pp"] for row in paired_rows]
        paired_seeds = [row["training_seed"] for row in paired_rows]
        summaries[label] = {"left": arms_summary["left"], "right": arms_summary["right"],
                            "difference_pp": summarize_independent_seeds(diffs, seeds, paired_seeds),
                            "complete_paired_seeds": len(paired_rows)}

    inventory = {"planned_final_policies": len(datasets) * len(arms) * len(seeds),
                 "complete_final_policies": sum(item["status"] == "complete" for item in final_inventory),
                 "planned_final_pairs": len(datasets) * len(seeds),
                 "complete_final_pairs": sum(bool(row.get("statistics")) for row in comparisons),
                 "final_inventory": final_inventory,
                 "development_task_count": len(development_inventory),
                 "development_selection_inventory": [
                     {key: value for key, value in entry.items() if key not in {"result", "task"}}
                     for entry in development_inventory],
                 "exploratory_probe_inventory": other_probes,
                 "exact_plan_id": plan_id}
    report = {"schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "plan_path": str(plan_path.resolve()), "plan_id": plan_id,
              "protocol_ids": sorted({item.get("protocol_id") for item in final_inventory
                                      if item.get("protocol_id")}),
              "inventory": inventory, "comparisons": comparisons,
              "training_seed_summaries": summaries, "rq2_offline_metrics_vs_success": rq2_rows,
              "rq2_final_policy_row_count": len(rq2_rows),
              "development_selection_task_count": len(development_inventory),
              "exploratory_probe_count": len(other_probes), "inventory_errors": errors,
              "scene_split": plan.get("scene_split"),
              "interpretation": {
                  "episode_tests": "conditional on the evaluated trained policies; not population-level inference",
                  "training_variability": "sample SD is across independent training seeds [3,4,5]",
                  "rq2": "matched checkpoint validation metrics versus same-checkpoint simulator success; descriptive only",
              "development_selection": "selection evidence on the locked development cohort; excluded from final-test pair family",
              "development_snapshots": "repeated checkpoints from one policy are not independent trained runs; no pooled checkpoint correlation is computed",
              "rq2_unit": "one final-evaluation result per independently trained policy, matched to that exact checkpoint's offline validation metrics",
              "probes": "exploratory; excluded from the final six-comparison family",
                  "causality": "no causal claims from captured trajectories"}}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper_results.json", report)
    write_csv(output_dir / "final_inventory.csv", final_inventory,
              ["dataset", "experiment", "backbone", "training_seed", "checkpoint_tag", "purpose", "status",
               "task_id", "optimizer_step", "protocol_id", "run_signature", "checkpoint_sha256",
               "selection_epoch", "selected_checkpoint_sha256", "selected_development_success_rate",
               "selected_development_task_id",
               "validation_metrics_sha256", "result_path", "trajectory_path", "trajectory_sha256", "success_rate",
               "successes", "n_evaluation_episodes", "error"])
    write_csv(output_dir / "final_pairs.csv", comparisons,
              ["label", "dataset", "training_seed", "status", "left_experiment", "right_experiment",
               "left_status", "right_status", "left_checkpoint_sha256", "right_checkpoint_sha256",
               "error", "statistics"])
    write_csv(output_dir / "rq2_offline_metrics_vs_success.csv", rq2_rows,
              ["dataset", "experiment", "backbone", "training_seed", "task_id", "checkpoint_sha256",
               "checkpoint_tag", "purpose", "evaluation_cohort", "selection_cohort_status",
               "selection_cohort_success_rate",
               "validation_metrics_sha256", "status", "reason", "success_rate", "flow_matching_loss",
               "action_mse", "action_mae", "per_dimension_action_mse", "per_dimension_action_mae",
               "n_samples", "validation_rng_seed", "action_sampling_seed_base"])
    write_csv(output_dir / "exploratory_probe_inventory.csv", other_probes,
              ["dataset", "experiment", "backbone", "training_seed", "checkpoint_tag", "status",
               "task_id", "optimizer_step", "protocol_id", "run_signature", "checkpoint_sha256",
               "validation_metrics_sha256", "result_path", "trajectory_path", "trajectory_sha256"])
    write_csv(output_dir / "development_selection_inventory.csv",
              [{key: value for key, value in entry.items() if key not in {"result", "task"}}
               for entry in development_inventory],
              ["dataset", "experiment", "backbone", "training_seed", "checkpoint_tag", "purpose",
               "selection_epoch", "status", "task_id", "optimizer_step", "checkpoint_sha256",
               "success_rate", "successes", "n_evaluation_episodes", "validation_metrics_sha256",
               "result_path", "trajectory_path", "trajectory_sha256", "error"])
    (output_dir / "REPORT.md").write_text(
        render_markdown(plan_id, inventory, comparisons, summaries, rq2_rows, other_probes, errors),
        encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-root", type=Path,
                        default=Path("/workspace-global/action-omni"))
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_success_stopped_study.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "asset/paper_results/runpod_success_stopped_v1")
    args = parser.parse_args(argv)
    report = build_report(args.durable_root.resolve(), args.plan.resolve(), args.output_dir.resolve())
    print(json.dumps({"plan_id": report["plan_id"], "complete_final_policies":
                      report["inventory"]["complete_final_policies"], "planned_final_policies":
                      report["inventory"]["planned_final_policies"], "complete_final_pairs":
                      report["inventory"]["complete_final_pairs"], "planned_final_pairs":
                      report["inventory"]["planned_final_pairs"], "output_dir": str(args.output_dir.resolve())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
