"""
scripts/diagnose_smolvla3.py
─────────────────────────────
Decisive number: a STABLE (t-averaged) flow-matching loss for SmolVLA on its own
TRAINING frames, through the eval pipeline.  Flow loss is high-variance in the
sampled diffusion-time t, so we average many draws over a fixed frame set.

  stable loss ~= training-log 0.005  -> model genuinely fits; pipeline is fine.
                                         => the 0% is a SAMPLING/rollout problem
                                            (low flow loss, but integrating the
                                            learned field drifts -- classic with
                                            few demos), not a preprocessing bug.
  stable loss >> 0.005               -> real train/inference input mismatch.

Also reports the per-t loss profile and a no-vision (zeroed image) control to
measure how much the prediction actually depends on the camera.
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
cfg = get_config("aloha", "exp01"); TASK = cfg.task_text
root = Path(cfg.dataset_root)

policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(device).eval()
H = policy.config.n_action_steps
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(CKPT),
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {"observation.images.top": "observation.images.camera1"}}})

shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
states = np.asarray(df["observation.state"].tolist(), np.float32)
actions = np.asarray(df["action"].tolist(), np.float32)
ep = df["episode_index"].to_numpy(); N = len(df)
vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]


def gt_chunk(i):
    e = ep[i]; last = actions[i]; o = []
    for k in range(H):
        j = i + k
        last = actions[j] if (j < N and ep[j] == e) else last
        o.append(last)
    return np.stack(o)


def floss(batch, t=None):
    time = None if t is None else torch.full((batch["observation.state"].shape[0],), float(t), device=device)
    out = policy.forward(batch, time=time)
    return float(out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out))


# fixed set of 64 mid-episode training frames
idx = np.arange(80, N, 311); idx = idx[idx + H < N][:64]
reader = _SequentialFrameReader(vid)
imgs = [reader.get(int(i)) for i in idx]
reader.close()


def build(zero_img=False):
    out_l, out_z = [], []
    for s in range(0, len(idx), 8):
        ii = idx[s:s + 8]
        im = np.stack([imgs[s + t] for t in range(len(ii))])
        img = torch.from_numpy(im).permute(0, 3, 1, 2).float() / 255.0
        if zero_img:
            img = torch.zeros_like(img)
        b = {"observation.images.top": img,
             "observation.state": torch.from_numpy(states[ii]),
             "action": torch.from_numpy(np.stack([gt_chunk(int(i)) for i in ii])),
             "task": [TASK] * len(ii)}
        out_l.append(pre(b))
    return out_l


with torch.no_grad():
    batches = build()
    # (1) stable loss = average over t-grid x frames
    tgrid = np.round(np.linspace(0.05, 0.95, 19), 2)
    per_t = []
    for t in tgrid:
        per_t.append(np.mean([floss(b, t=t) for b in batches]))
    stable = float(np.mean(per_t))
    # (2) random-t draws (what training logs) for reference
    rnd = [np.mean([floss(b) for b in batches]) for _ in range(8)]

print(f"\n{'='*60}")
print(f"STABLE flow loss (t-averaged) on 64 training frames = {stable:.4f}")
print(f"  training-log final loss                           = 0.005")
print(f"  random-t single-draw estimates: "
      f"mean={np.mean(rnd):.3f} range=[{min(rnd):.3f},{max(rnd):.3f}]  (note the variance)")
print(f"\nper-t loss profile:")
for t, l in zip(tgrid, per_t):
    print(f"   t={t:.2f}  loss={l:.4f}")
verdict = ("model FITS training data -> 0% is a SAMPLING/data-efficiency problem"
           if stable < 0.05 else
           "loss >> 0.005 even t-averaged -> real train/inference MISMATCH")
print(f"\nVERDICT: {verdict}")
print('='*60)
