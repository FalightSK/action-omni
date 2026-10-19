"""Evaluate a saved head on its actual episode holdout; no VLM and no training.

Optional --results checks/classifies the same ALOHA cohort using its saved trace.
Outputs are diagnostics, not causal evidence or population-level training tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.hdf5_embeddings import HDF5EmbeddingDataset
from experiment_protocol import (checkpoint_config, fixed_rng, protocol_arrays,
                                 write_json, file_digest)
from models.vla_train import VLATrainModel


def classify_trace(state, reward, action, threshold, gripper_dim=6):
    if (state.ndim != 2 or len(state) != len(reward) or action.shape != state.shape
            or not len(reward) or not np.isfinite(state).all()
            or not np.isfinite(action).all() or not np.isfinite(reward).all()):
        raise ValueError("Invalid aligned rollout trace")
    maximum = int(np.max(reward))
    if maximum >= 4:
        return "success"
    if maximum < 2:
        return "never_lifted"
    lift = int(np.flatnonzero(reward >= 2)[0])
    g = state[lift:, gripper_dim]
    opened = np.flatnonzero(g > threshold)
    if not len(opened):
        return "receiver_never_opened"
    if (g[opened[0]:] < threshold).any():
        return "opened_then_closed_without_transfer"
    return "opened_not_closed"


def rollout_diagnostics(result, protocol, checkpoint_sha):
    if (result.get("protocol_id") != protocol["protocol_id"]
            or result.get("checkpoint_sha256") != checkpoint_sha
            or result.get("dataset") != "aloha"):
        raise ValueError("Trace/result/checkpoint cohort mismatch")
    trace_path = Path(result["trace"])
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    records = result["episodes"]
    calibration = protocol["receiver_gripper"]
    summaries = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        if (int(trace["n_episodes"][0]) != len(records)
                or len({r["initial_state_id"] for r in records}) != len(records)
                or not np.array_equal(trace["initial_state_ids"], [r["initial_state_id"] for r in records])):
            raise ValueError("Trace index does not match result episode keys")
        for fraction in (0.5, 0.6, 0.7):
            threshold = calibration["low"] + fraction * (calibration["high"] - calibration["low"])
            counts, stages, attention, commands_on_never_opened = {}, {str(i): 0 for i in range(1, 5)}, [], 0
            for i, record in enumerate(records):
                prefix = f"ep{i:04d}_"
                state, reward, action = [trace[prefix+k] for k in ("next_state", "reward", "action")]
                if (int(reward.max()) != record["max_reward"]
                        or bool(reward.max() >= 4) != record["is_success"]):
                    raise ValueError("Trace and headline success/reward disagree")
                category = classify_trace(state, reward, action, threshold, calibration["dimension"])
                counts[category] = counts.get(category, 0) + 1
                for stage in stages:
                    stages[stage] += int(reward.max() >= int(stage))
                if reward.max() >= 2:
                    lift = int(np.flatnonzero(reward >= 2)[0])
                    if category == "receiver_never_opened":
                        commands_on_never_opened += int((action[lift:, calibration["dimension"]] > threshold).any())
                    attn = trace[prefix + "attn_image_mass"]
                    # Reward at step `lift` is observed after that action. The
                    # replan containing it occurred earlier; begin at the next replan.
                    start = (lift + result["inference_horizon"]) // result["inference_horizon"]
                    values = attn[start:start+5]
                    if len(values):
                        attention.append({"initial_state_id": record["initial_state_id"],
                                          "success": record["is_success"], "replans": len(values),
                                          "mean": float(values.mean())})
            lifted = stages["2"]
            summaries[str(fraction)] = {"threshold": threshold, "counts": counts,
                "n_all": len(records), "n_lifted": lifted, "stage_counts": stages,
                "post_lift_failure_rates": {k: v/lifted if lifted else None for k, v in counts.items()
                                             if k not in ("success", "never_lifted")},
                "never_opened_with_commanded_open": commands_on_never_opened,
                "post_lift_attention_per_episode": attention}
    return summaries


def offline_metrics(model, ds, protocol, device, batch_size=64, draws=4):
    ids, val = protocol_arrays(protocol)
    rows = np.flatnonzero(val)
    loader = DataLoader(Subset(ds, rows.tolist()), batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(4242))
    scale = ds.actions[~val, 0].std(dim=0, unbiased=False).clamp_min(1e-8).to(device)
    sums = {int(ep): {"flow_loss": 0., "nmae": 0., "frames": int((ids[val] == ep).sum())}
            for ep in np.unique(ids[val])}
    predicted, targets = [], []
    for draw in range(draws):
        offset = 0
        with fixed_rng(4242 + draw), torch.no_grad():
            for batch in loader:
                embed, state, action, mask = [batch[k].to(device) for k in ("embedding", "state", "actions", "img_mask")]
                context, tokens = model._build_cond(embed, state, mask)
                if not model.use_dit:
                    raise ValueError("Camera-ready diagnostics require the planned DiT head")
                t = torch.rand(len(action), device=device, dtype=action.dtype)
                noise = torch.randn_like(action)
                x = (1-t[:, None, None])*noise + t[:, None, None]*action
                velocity = model.decoder(x, t, context, tokens)
                loss = (velocity - (action-noise)).square().mean((1, 2)).cpu().numpy()
                pred = model.sample(embed, state, num_steps=model.cfg.num_flow_steps, img_mask=mask).reshape_as(action)
                error = ((pred-action).abs()/scale).mean((1, 2)).cpu().numpy()
                for j, ep in enumerate(ids[rows[offset:offset+len(action)]]):
                    sums[int(ep)]["flow_loss"] += float(loss[j]) / draws
                    sums[int(ep)]["nmae"] += float(error[j]) / draws
                predicted.append(pred.cpu().numpy().ravel())
                targets.append(action.cpu().numpy().ravel())
                offset += len(action)
    per_episode = [{"episode_id": ep, "frames": v["frames"],
                    "flow_loss": v["flow_loss"]/v["frames"], "nmae": v["nmae"]/v["frames"]}
                   for ep, v in sums.items()]
    a, b = np.concatenate(predicted), np.concatenate(targets)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite head predictions")
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    episode_uncertainty = {}
    for metric in ("flow_loss", "nmae"):
        values = np.asarray([e[metric] for e in per_episode])
        rng = np.random.default_rng(20261018)
        boot = rng.choice(values, size=(20000, len(values))).mean(1)
        episode_uncertainty[metric] = {
            "equal_episode_mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "episode_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "note": "Few validation episodes; conditional on this trained policy and split."}
    return {"validation_frames": len(rows), "validation_episodes": len(sums),
            "draw_seeds": list(range(4242, 4242+draws)), "per_episode": per_episode,
            "flow_loss_frame_weighted": sum(v["flow_loss"] for v in sums.values())/len(rows),
            "nmae_frame_weighted": sum(v["nmae"] for v in sums.values())/len(rows),
            "nmae_scale": "SD of training-frame first-step cached actions per dimension",
            "open_loop_correlation": corr,
            "episode_uncertainty": episode_uncertainty,
            "note": "Adjacent frames and repeated draws are not independent demonstration episodes."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    if not protocol:
        raise ValueError("Legacy frame-split checkpoints are not eligible")
    cfg = checkpoint_config(checkpoint["cfg"], checkpoint, protocol["dataset"])
    ds = HDF5EmbeddingDataset(args.cache, expect_layers=cfg.vlm_extract_layers, protocol=protocol, config=cfg)
    if checkpoint.get("cache_id") != ds.cache_id:
        raise ValueError("Diagnostics cache differs from training cache")
    model = VLATrainModel(cfg).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    sha = file_digest(args.checkpoint)
    result = {"protocol_id": protocol["protocol_id"], "training_seed": checkpoint["training_seed"],
              "checkpoint_sha256": sha, "offline": offline_metrics(model, ds, protocol, args.device)}
    if args.results:
        result["rollout"] = rollout_diagnostics(json.loads(args.results.read_text(encoding="utf-8")), protocol, sha)
    write_json(args.output, result)
    print(f"Wrote held-out diagnostics to {args.output}")


if __name__ == "__main__":
    main()
