"""
scripts/compute_smolvla_preds.py
─────────────────────────────────
Run the finetuned SmolVLA policy on real dataset frames and cache its predicted
action chunks, so we can compare its prediction distribution against the Qwen
models.  Uses lerobot's own policy + saved processor pipeline (the SAME path as
eval), so normalization / tokenization / image handling match training exactly.

Saves -> asset/runs/aloha/smolvla_ft/pred_cache.npz  (pred, gt : (N,50,14) joint units)
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
OUTNPZ = ROOT / "asset/runs/aloha/smolvla_ft/pred_cache.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_config("aloha", "exp01")
TASK = cfg.task_text
root = Path(cfg.dataset_root)
print(f"device={device}  task={TASK!r}")

# ── raw tabular ─────────────────────────────────────────────────────────────────
shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
states  = np.asarray(df["observation.state"].tolist(), np.float32)   # (N,14) raw
actions = np.asarray(df["action"].tolist(),            np.float32)   # (N,14) raw
ep_ids  = df["episode_index"].to_numpy()
N = len(df)
sample_idx = np.arange(0, N, 50)                                     # 400 frames
print(f"sampling {len(sample_idx)} frames of {N}")

# ── decode the sampled frames (forward-only) ───────────────────────────────────
vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]
reader = _SequentialFrameReader(vid)
imgs = [reader.get(int(i)) for i in sample_idx]                      # list of (480,640,3) uint8
reader.close()
print("frames decoded")

# ── policy + processors (same recipe as lerobot_eval) ──────────────────────────
policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(device).eval()
H = policy.config.n_action_steps
pre, post = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=str(CKPT),
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {"observation.images.top": "observation.images.camera1"}},
    },
)
print(f"policy + processors loaded  (chunk H={H})")


def gt_chunk(i: int) -> np.ndarray:
    e = ep_ids[i]; last = actions[i]; out = []
    for k in range(H):
        j = i + k
        if j < N and ep_ids[j] == e:
            last = actions[j]
        out.append(last)
    return np.stack(out)                                            # (H,14)


preds, gts = [], []
B = 8
with torch.no_grad():
    for s in range(0, len(sample_idx), B):
        idx = sample_idx[s:s + B]
        img_b = np.stack([imgs[s + t] for t in range(len(idx))])     # (b,480,640,3)
        img_t = torch.from_numpy(img_b).permute(0, 3, 1, 2).float() / 255.0   # (b,3,H,W) [0,1]
        st_t = torch.from_numpy(states[idx])                          # (b,14) raw
        batch = {
            "observation.images.top": img_t,
            "observation.state": st_t,
            "task": [TASK] * len(idx),
        }
        batch = pre(batch)
        policy.reset()
        act = policy.predict_action_chunk(batch)                      # (b,H,Dpad)
        act = post(act)
        preds.append(act.detach().float().cpu().numpy())
        gts.append(np.stack([gt_chunk(int(i)) for i in idx]))
        if s == 0:
            print(f"  first batch out: pred {preds[0].shape}  range "
                  f"[{preds[0].min():.3f},{preds[0].max():.3f}]")

pred = np.concatenate(preds)[..., :14]                                # (N,H,14)
gt = np.concatenate(gts)                                              # (N,H,14)
OUTNPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUTNPZ, pred=pred, gt=gt)
print(f"\npred {pred.shape} range [{pred.min():.3f},{pred.max():.3f}]")
print(f"gt   {gt.shape} range [{gt.min():.3f},{gt.max():.3f}]")
print(f"Saved -> {OUTNPZ}")
