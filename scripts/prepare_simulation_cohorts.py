"""Hash the fixed LIBERO-Goal starts and describe simulator seed cohorts.

This reads the official LIBERO initial-state files only; it does not construct
an environment or advance a simulator. Run with the LIBERO evaluation Python:

    .venv-eval-libero/bin/python scripts/prepare_simulation_cohorts.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def initial_vector_sha256(value) -> str:
    """Match envs.libero_env._array_sha256 for a loaded fixed-start vector."""
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps((array.shape, array.dtype.str),
                        separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(4, "little"))
    digest.update(header)
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def build_manifest(libero_src: Path, study_config: dict,
                   study_config_path: Path) -> dict:
    libero_src = libero_src.resolve()
    if not (libero_src / "libero" / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO source tree not found under {libero_src}")
    sys.path.insert(0, str(libero_src))
    import torch
    from libero.libero import benchmark

    package_root = libero_src / "libero" / "libero"
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    if suite.n_tasks != 10:
        raise ValueError(f"Expected 10 LIBERO-Goal tasks, found {suite.n_tasks}")

    study_libero = study_config["datasets"]["libero"]
    study_aloha = study_config["datasets"]["aloha"]
    scene_split = study_config.get("scene_split", {})
    if scene_split.get("demonstration_scene_absence_verified") is not False:
        raise ValueError("Study config must retain the unverified ALOHA demo-scene limitation")
    dev_cfg = study_libero["development_evaluation"]
    final_cfg = study_libero["final_evaluation"]
    aloha_dev = study_aloha["development_evaluation"]
    aloha_final = study_aloha["final_evaluation"]
    aloha_dev_ids = list(range(aloha_dev["seed_offset"],
                               aloha_dev["seed_offset"] + aloha_dev["episodes"]))
    aloha_final_ids = list(range(aloha_final["seed_offset"],
                                 aloha_final["seed_offset"] + aloha_final["episodes"]))
    if set(aloha_dev_ids) & set(aloha_final_ids):
        raise ValueError("ALOHA development and final reset-seed indices overlap")
    expected_tasks = list(range(suite.n_tasks))
    for name, eval_cfg in (("development", dev_cfg), ("final", final_cfg)):
        if eval_cfg["tasks"] != expected_tasks:
            raise ValueError(f"{name} cohort must cover the same all-task set {expected_tasks}")
        if eval_cfg["conditions"] != ["canonical"]:
            raise ValueError(f"{name} cohort selection must use canonical conditions only")
    dev_offset = int(dev_cfg["initial_state_offset"])
    dev_episodes = int(dev_cfg["episodes_per_task"])
    final_offset = int(final_cfg["initial_state_offset"])
    final_episodes = int(final_cfg["episodes_per_task"])
    dev_ids = list(range(dev_offset, dev_offset + dev_episodes))
    final_ids = list(range(final_offset, final_offset + final_episodes))
    if dev_ids != list(range(0, 10)) or final_ids != list(range(10, 50)):
        raise ValueError("Expected LIBERO development IDs 0-9 and final IDs 10-49")
    task_rows = []
    for task_index in range(suite.n_tasks):
        task = suite.get_task(task_index)
        states_path = package_root / "init_files" / task.problem_folder / task.init_states_file
        if not states_path.is_file():
            raise FileNotFoundError(f"Missing official fixed starts: {states_path}")
        states = np.asarray(torch.load(str(states_path), map_location="cpu",
                                       weights_only=False))
        if len(states) != 50:
            raise ValueError(f"Task {task_index} has {len(states)} starts; expected exactly 50")
        starts = [{"initial_state_id": state_id,
                   "libero_init_state_sha256": initial_vector_sha256(states[state_id])}
                  for state_id in range(50)]
        dev_hashes = {starts[i]["libero_init_state_sha256"] for i in dev_ids}
        final_hashes = {starts[i]["libero_init_state_sha256"] for i in final_ids}
        overlap = dev_hashes & final_hashes
        if overlap:
            raise ValueError(f"Task {task_index} has duplicate init vectors across dev/final cohorts")
        task_rows.append({
            "task_index": task_index,
            "task_language": task.language,
            "problem_folder": task.problem_folder,
            "bddl_file": task.bddl_file,
            "init_states_file": str(states_path.relative_to(libero_src)),
            "starts": starts,
            "dev_final_vector_hashes_disjoint": True,
        })

    if set(dev_ids) & set(final_ids) or sorted(dev_ids + final_ids) != list(range(50)):
        raise AssertionError("LIBERO-Goal start cohorts must partition IDs 0..49")

    return {
        "schema": 1,
        "manifest_type": "simulation_initial_state_cohorts",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hash_spec": (
            "SHA-256 of compact JSON [shape,dtype.str] header (length-prefixed "
            "with 4 little-endian bytes), followed by C-contiguous array bytes; "
            "matches envs.libero_env._array_sha256"
        ),
        "libero": {
            "suite": "libero_goal",
            "task_indices": list(range(10)),
            "task_split": "none; both cohorts use the same ten tasks",
            "start_count_per_task": 50,
            "cohorts": {
                "dev": {"initial_state_offset": dev_offset,
                        "initial_state_ids": dev_ids,
                        "episodes_per_task": dev_episodes},
                "final": {"initial_state_offset": final_offset,
                          "initial_state_ids": final_ids,
                          "episodes_per_task": final_episodes},
            },
            "task_rows": task_rows,
            "cohorts_disjoint_by_id": True,
            "cohorts_disjoint_by_vector_hash_within_each_task": True,
        },
        "source_study_config": str(study_config_path.resolve()),
        "scene_split": scene_split,
        "aloha": {
            "task": "AlohaTransferCube-v0",
            "cohorts": {
                "dev": {
                    "episodes": aloha_dev["episodes"],
                    "seed_offset": aloha_dev["seed_offset"],
                    "reset_seed_indices": aloha_dev_ids,
                },
                "final": {
                    "episodes": aloha_final["episodes"],
                    "seed_offset": aloha_final["seed_offset"],
                    "reset_seed_indices": aloha_final_ids,
                },
            },
            "environment_seed_rule": "(episode_index + seed_offset) * 137 + 42",
            "seed_id_ranges_disjoint": not bool(set(aloha_dev_ids) & set(aloha_final_ids)),
            "training_demo_scene_overlap": scene_split.get("limitation"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-src", type=Path,
                        default=ROOT / "third_party" / "LIBERO",
                        help="Official LIBERO source checkout")
    parser.add_argument("--study-config", type=Path,
                        default=ROOT / "configs" / "runpod_success_stopped_study.json",
                        help="Study config that defines the development and final cohorts")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "asset" / "simulation_cohorts.json")
    args = parser.parse_args()
    study_config = json.loads(args.study_config.read_text(encoding="utf-8"))
    manifest = build_manifest(args.libero_src, study_config, args.study_config)
    atomic_json(args.output, manifest)
    print(f"Wrote {args.output}")
    print("LIBERO-Goal: dev IDs 0-9 and final IDs 10-49 across the same 10 tasks")
    print("ALOHA: seed cohorts documented; overlap with demo scenes is unassessed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
