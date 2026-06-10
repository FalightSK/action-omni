"""
scripts/diagnose_smolvla4.py
─────────────────────────────
Tie-breaker: across training checkpoints (4k, 8k, 12k), does the inference-path
error actually improve?  Measured two ways on a fixed set of 64 training frames:
  * stable t-averaged flow loss (policy.forward)
  * sampled prediction MAE vs GT (predict_action_chunk -> the real inference path)

If both DROP with training  -> the model genuinely learned; residual error + 0%
                               is a fit-quality / data-efficiency limit.
If FLAT across checkpoints  -> training gains don't reach inference = mismatch.

Also compares to the Qwen exp02 sampled MAE on the same frames as a reference.
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

CKPTS = {
    "4k":  ROOT / "asset/runs/aloha/smolvla_ft/checkpoints/004000/pretrained_model",
    "8k":  ROOT / "asset/runs/aloha/smolvla_ft/checkpoints/008000/pretrained_model",
    "12k": ROOT / "asset/runs/aloha/smolvla_ft/checkpoints/012000/pretrained_model",
}
LOGGED = {"4k": 0.017, "8k": 0.007, "12k": 0.005}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = get_config("aloha", "exp01"); TASK = cfg.task_text
root = Path(cfg.dataset_root)

shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
states = np.asarray(df["observation.state"].tolist(), np.float32)
actions = np.asarray(df["action"].tolist(), np.float32)
ep = df["episode_index"].to_numpy(); N = len(df)
vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]
H = 50


def gt_chunk(i):
    e = ep[i]; last = actions[i]; o = []
    for k in range(H):
        j = i + k
        last = actions[j] if (j < N and ep[j] == e) else last
        o.append(last)
    return np.stack(o)


idx = np.arange(80, N, 311); idx = idx[idx + H < N][:64]
reader = _SequentialFrameReader(vid)
imgs = [reader.get(int(i)) for i in idx]
reader.close()
gt = np.stack([gt_chunk(int(i)) for i in idx])           # (64,50,14)
tgrid = np.round(np.linspace(0.05, 0.95, 10), 2)

print(f"{'ckpt':5s} {'logged':>7s} {'t-avg fwd loss':>15s} {'sampled MAE':>13s}")
for tag, ck in CKPTS.items():
    policy = SmolVLAPolicy.from_pretrained(str(ck)).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(ck),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": {"observation.images.top": "observation.images.camera1"}}})

    def make(s):
        ii = idx[s:s + 8]
        im = torch.from_numpy(np.stack([imgs[s + t] for t in range(len(ii))])).permute(0, 3, 1, 2).float() / 255.0
        return {"observation.images.top": im,
                "observation.state": torch.from_numpy(states[ii]),
                "action": torch.from_numpy(np.stack([gt_chunk(int(i)) for i in ii])),
                "task": [TASK] * len(ii)}

    with torch.no_grad():
        # forward loss (t-averaged)
        fl = []
        for s in range(0, len(idx), 8):
            b = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in make(s).items()})
            B = b["observation.state"].shape[0]
            for t in tgrid:
                out = policy.forward(b, time=torch.full((B,), float(t), device=device))
                fl.append(float(out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out)))
        # sampled prediction MAE
        preds = []
        for s in range(0, len(idx), 8):
            b = pre(make(s)); policy.reset()
            preds.append(post(policy.predict_action_chunk(b))[..., :14].float().cpu().numpy())
        pred = np.concatenate(preds)
        mae = float(np.abs(pred - gt).mean())
    print(f"{tag:5s} {LOGGED[tag]:7.3f} {np.mean(fl):15.4f} {mae:13.4f}")
    del policy

# Qwen exp02 sampled MAE on the same frames, as a reference
from data.aloha.dataset import AlohaEmbeddingDataset
from models.vla_train import VLATrainModel
c2 = get_config("aloha", "exp02")
ds = AlohaEmbeddingDataset(str(ROOT / c2.embeddings_cache))
m = VLATrainModel(c2).to(device)
m.load_state_dict(torch.load(ROOT / c2.output_dir / "checkpoints/best.pt", map_location=device, weights_only=False)["state_dict"])
m.eval()
with torch.no_grad():
    emb = torch.stack([ds.embeddings[i].float() for i in idx]).to(device)
    st = torch.stack([ds.states[i] for i in idx]).to(device)
    im = torch.stack([ds.img_masks[i] for i in idx]).to(device) if ds.img_masks is not None else None
    flat = m.sample(emb, st, num_steps=c2.num_flow_steps, img_mask=im)
    p = flat.view(-1, H, 14).cpu().numpy()
amean = np.array(c2.action_mean, np.float32); astd = np.array(c2.action_std, np.float32)
p = p * astd + amean
gtn = (gt - amean) / astd; pn = (p - amean) / astd      # compare in normalised units (same as SmolVLA MAE)
print(f"{'exp02':5s} {'(qwen)':>7s} {'--':>15s} {np.abs(pn - gtn).mean():13.4f}   <- reference (normalised units)")
