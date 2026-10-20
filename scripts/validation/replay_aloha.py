"""Bounded real-policy replay gate: same starts twice, then with video.

Compare exact state/action/reward arrays, not success rates on a tiny cohort.
This diagnostic does not retrain a policy or overwrite its original evaluation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["MUJOCO_GL"] = "glfw"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from configs.registry import get_config
from experiment_protocol import checkpoint_config, file_digest, write_json
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from scripts.evaluate import _run_aloha, deterministic_inference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--exp", default="exp05")
    ap.add_argument("--seed-offset", type=int, default=4004)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--render-samples", type=int, choices=(0, 4), default=0)
    args = ap.parse_args()
    if args.episodes < 1 or args.seed_offset < 0:
        ap.error("Episodes must be positive and seed-offset nonnegative")
    deterministic_inference()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_config(get_config("aloha", args.exp), ck, "aloha", verify_live_encoder=True)
    if ck.get("training_seed") is None:
        raise ValueError("Replay requires an explicit training seed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg.output_dir = str(args.output_dir)
    device = cfg.get_device()
    vlm = VLAModel(cfg)
    vlm.vlm.to(device).eval()
    head = VLATrainModel(cfg).to(device).eval()
    head.load_state_dict(ck["state_dict"], strict=True)
    report = {"checkpoint_sha256": file_digest(args.checkpoint), "checkpoint": str(args.checkpoint.resolve()),
              "training_seed": ck["training_seed"], "deterministic_inference": True,
              "render_samples": args.render_samples,
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "initial_state_ids": list(range(args.seed_offset, args.seed_offset + args.episodes)),
              "checks": [], "outcomes": []}
    paths = []
    for index, video in enumerate([False, False, True]):
        path = args.output_dir / f"repeat_{index}.json"
        trace = path.with_suffix(".npz")
        results = _run_aloha(cfg, vlm, head, device, args.episodes, video, path,
                             seed_offset=args.seed_offset, trace_path=trace,
                             training_seed=ck["training_seed"], record_attention=False,
                             render_samples=args.render_samples)
        paths.append(trace)
        report["outcomes"].append(results)
        write_json(path, results)
    with np.load(paths[0], allow_pickle=False) as baseline:
        for index in [1, 2]:
            with np.load(paths[index], allow_pickle=False) as repeat:
                equal = set(baseline.files) == set(repeat.files)
                differences = {}
                for key in baseline.files:
                    if key not in repeat or baseline[key].shape != repeat[key].shape:
                        equal = False
                        differences[key] = None
                    else:
                        equal = equal and np.array_equal(baseline[key], repeat[key])
                        differences[key] = (float(np.max(np.abs(baseline[key].astype(float) - repeat[key].astype(float))))
                                            if baseline[key].size else 0.)
                report["checks"].append({"repeat": index, "video": index == 2,
                                          "all_trace_arrays_equal": bool(equal),
                                          "max_absolute_differences": differences})
    report["passed"] = all(c["all_trace_arrays_equal"] for c in report["checks"])
    write_json(args.output_dir / "replay_check.json", report)
    if not report["passed"]:
        raise RuntimeError("Seeded replay differs; inspect replay_check.json before the study")
    print("PASS: repeated state/action/reward arrays match exactly, including video", flush=True)


if __name__ == "__main__":
    main()
