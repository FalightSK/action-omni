"""
scripts/diagnose_smolvla.py
────────────────────────────
Root-cause tests for SmolVLA's 0%.  Two decisive checks:

TEST A  (train/inference consistency):
  Re-run SmolVLA's OWN training loss (policy.forward, flow-matching) on training
  frames through the eval pipeline.
    * loss ~= 0.005 (the logged final train loss)  -> pipeline matches training;
      the low flow-loss is simply non-predictive (fit a low-loss solution that
      doesn't roll out) -> look at TEST B.
    * loss >> 0.005                                -> train != inference pipeline
      (a real preprocessing/normalization mismatch = the bug).

TEST B  (does it use vision?):
  For the first frame of each episode (cube at 50 different places), how much does
  the PREDICTED initial action vary across episodes, vs ground truth?
    * SmolVLA cross-episode std << GT std  -> predictions are ~input-independent
      (ignores the cube) = mode collapse to the average trajectory.
"""

from __future__ import annotations
import glob, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from data.aloha.dataset import _SequentialFrameReader
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = ROOT / "asset/runs/aloha/smolvla_ft/checkpoints/012000/pretrained_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

cfg = get_config("aloha", "exp01")
TASK = cfg.task_text
root = Path(cfg.dataset_root)
shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
states  = np.asarray(df["observation.state"].tolist(), np.float32)
actions = np.asarray(df["action"].tolist(), np.float32)
ep_ids  = df["episode_index"].to_numpy()
N = len(df)

policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(device).eval()
H = policy.config.n_action_steps
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(CKPT),
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {"observation.images.top": "observation.images.camera1"}}})
print(f"loaded SmolVLA (chunk H={H}); train log final loss ~= 0.005\n")

vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]


def gt_chunk(i):
    e = ep_ids[i]; last = actions[i]; out = []
    for k in range(H):
        j = i + k
        if j < N and ep_ids[j] == e:
            last = actions[j]
        out.append(last)
    return np.stack(out)


# ── TEST A: forward (training) loss on training frames ──────────────────────────
idxA = np.arange(60, N, 137)                       # ~145 mid-episode training frames
idxA = idxA[idxA + H < N]
reader = _SequentialFrameReader(vid)
imgsA = [reader.get(int(i)) for i in idxA]
reader.close()

losses = []
B = 8
with torch.no_grad():
    for s in range(0, len(idxA), B):
        ii = idxA[s:s + B]
        img = torch.from_numpy(np.stack([imgsA[s + t] for t in range(len(ii))])).permute(0, 3, 1, 2).float() / 255.0
        st = torch.from_numpy(states[ii])
        act = torch.from_numpy(np.stack([gt_chunk(int(i)) for i in ii]))
        batch = {"observation.images.top": img, "observation.state": st,
                 "action": act, "task": [TASK] * len(ii)}
        batch = pre(batch)
        out = policy.forward(batch)
        loss = out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out)
        losses.append(float(loss))
print(f"TEST A  forward (flow) loss on {len(idxA)} TRAINING frames: "
      f"mean={np.mean(losses):.4f}  (train-log final ~0.005)")
print(f"        -> {'MATCHES train: pipeline OK, low loss non-predictive' if np.mean(losses) < 0.02 else 'MUCH HIGHER: train/inference MISMATCH (bug)'}\n")

# ── TEST B: input-dependence on episode first-frames (cube reach) ───────────────
first = [int(df.index[ep_ids == e][0]) for e in sorted(df["episode_index"].unique())]
reader = _SequentialFrameReader(vid)
imgsB = [reader.get(i) for i in first]
reader.close()

smol_first, gt_first = [], []
with torch.no_grad():
    for s in range(0, len(first), B):
        ii = first[s:s + B]
        img = torch.from_numpy(np.stack([imgsB[s + t] for t in range(len(ii))])).permute(0, 3, 1, 2).float() / 255.0
        st = torch.from_numpy(states[ii])
        batch = {"observation.images.top": img, "observation.state": st, "task": [TASK] * len(ii)}
        batch = pre(batch)
        policy.reset()
        act = post(policy.predict_action_chunk(batch))[..., :14]
        smol_first.append(act.detach().float().cpu().numpy())
        gt_first.append(np.stack([gt_chunk(i) for i in ii]))
smol_first = np.concatenate(smol_first)            # (50,H,14)
gt_first = np.concatenate(gt_first)

# cross-episode std of the FIRST few predicted steps (the reach), averaged over joints
k = 5
gt_var   = gt_first[:, :k, :].std(0).mean()
smol_var = smol_first[:, :k, :].std(0).mean()
print(f"TEST B  cross-episode std of first-{k}-step action (50 episodes, varied cube):")
print(f"        ground truth : {gt_var:.4f}   (how much the correct reach varies with cube)")
print(f"        SmolVLA pred : {smol_var:.4f}   ratio={smol_var/gt_var:.2f}")
print(f"        -> {'IGNORES vision (input-independent)' if smol_var/gt_var < 0.5 else 'does vary with input'}")
