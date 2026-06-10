"""
scripts/diagnose_smolvla2.py
─────────────────────────────
Validate TEST A and localise the train/inference mismatch.

(1) Inspect my hand-built batch AFTER the preprocessor: is the action target and
    state actually normalised?  (If action isn't normalised, the 0.19 was my bug.)
(2) GOLD STANDARD: build the batch via lerobot's OWN LeRobotDataset (the exact
    training data path) and run policy.forward.  This MUST reproduce ~0.005 if the
    checkpoint/processors are fine.  Compare its image to my hand-built image.
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

policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(device).eval()
H = policy.config.n_action_steps
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(CKPT),
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {"observation.images.top": "observation.images.camera1"}}})


def floss(batch):
    out = policy.forward(batch)
    return float(out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out))


# ── (1) inspect my hand-built batch after preprocessing ─────────────────────────
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


reader = _SequentialFrameReader(vid)
idx = [1000, 1001, 5000, 9000]
imgs = [reader.get(i) for i in idx]
reader.close()
img = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
batch = {"observation.images.top": img,
         "observation.state": torch.from_numpy(states[idx]),
         "action": torch.from_numpy(np.stack([gt_chunk(i) for i in idx])),
         "task": [TASK] * len(idx)}
print("RAW  my image:", tuple(img.shape), f"range [{img.min():.3f},{img.max():.3f}]")
pb = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()})
print("\nAfter preprocessor, batch keys:", sorted(pb.keys()))
for key in pb:
    v = pb[key]
    if torch.is_tensor(v) and v.dtype.is_floating_point:
        print(f"  {key:34s} {tuple(v.shape)}  mean={v.float().mean():.3f} std={v.float().std():.3f} "
              f"min={v.float().min():.3f} max={v.float().max():.3f}")
print(f"\nmy-batch forward loss = {floss(pb):.4f}")

# ── (2) GOLD STANDARD via LeRobotDataset (exact training path) ──────────────────
print("\n--- LeRobotDataset (training data path) ---")
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    fps = 50
    dts = {"action": [i / fps for i in range(H)]}
    ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human",
                        root=str(root), delta_timestamps=dts, video_backend="pyav")
    samples = [ds[i] for i in (1000, 1001, 5000, 9000)]
    imkey = "observation.images.top"
    lim = torch.stack([s[imkey] for s in samples])
    print("LeRobot image:", tuple(lim.shape), f"range [{lim.min():.3f},{lim.max():.3f}] dtype={lim.dtype}")
    gbatch = {
        "observation.images.top": lim,
        "observation.state": torch.stack([s["observation.state"] for s in samples]),
        "action": torch.stack([s["action"] for s in samples]),
        "task": [s.get("task", TASK) for s in samples],
    }
    print("action chunk shape from lerobot:", tuple(gbatch["action"].shape))
    gpb = pre(gbatch)
    print(f"LeRobotDataset forward loss = {floss(gpb):.4f}   (should be ~0.005 if pipeline matches training)")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[LeRobotDataset path failed: {e}]")
