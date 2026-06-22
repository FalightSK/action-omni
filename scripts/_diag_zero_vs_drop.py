"""Diagnostic: is the offline text-removal catastrophe a ZEROING artifact?

Compares the flow-matching val loss under two removal methods:
  ZERO  — set the modality's token vectors to 0 but keep them in the sequence
          (what the offline token_info_probe did)
  DROP  — physically remove those positions from the sequence
          (what the online closed-loop ablation did)

Also captures the adapter's AttentionReadout weight distribution over the 103
token positions, to show where the pooled context vector actually comes from.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla_train import VLATrainModel

cfg = get_config("aloha", "exp01")
device = cfg.get_device()
ckpt = torch.load(Path(cfg.output_dir)/"checkpoints"/"best.pt", map_location=device, weights_only=False)
model = VLATrainModel(cfg).to(device); model.load_state_dict(ckpt["state_dict"]); model.eval()

cache = torch.load(cfg.embeddings_cache, map_location="cpu", weights_only=False)
E = cache["embeddings"].float(); M = cache["img_masks"]; S = cache["states"].float(); A = cache["actions"].float()
N = min(4000, E.shape[0])
g = torch.Generator().manual_seed(7); idx = torch.randperm(E.shape[0], generator=g)[:N]
E, M, S, A = E[idx], M[idx], S[idx], A[idx]
layout = M[0].bool()
assert bool((M.bool() == layout).all()), "img_mask layout not uniform across samples"
print(f"layout: {int(layout.sum())} image + {int((~layout).sum())} text tokens")

@torch.no_grad()
def loss_for(mode):
    tot, nb = 0.0, 0
    for s in range(0, N, 256):
        e = E[s:s+256].to(device); m = M[s:s+256].to(device)
        st = S[s:s+256].to(device); a = A[s:s+256].to(device)
        if mode == "full":
            pass
        elif mode == "zero_text":
            e = e.clone(); e[:, ~layout, :] = 0.0
        elif mode == "zero_image":
            e = e.clone(); e[:, layout, :] = 0.0
        elif mode == "drop_text":
            e = e[:, layout, :]; m = m[:, layout]
        elif mode == "drop_image":
            e = e[:, ~layout, :]; m = m[:, ~layout]
        l = model(e, st, a, m)
        tot += float(l) * e.shape[0]; nb += e.shape[0]
    return tot / nb

modes = ["full", "zero_text", "zero_image", "drop_text", "drop_image"]
res = {m: loss_for(m) for m in modes}
base = res["full"]
print("\nflow-matching val loss (lower=better):")
for m in modes:
    print(f"  {m:12s} {res[m]:.4f}   (+{100*(res[m]-base)/base:6.1f}%)")

# ── Readout attention distribution over tokens ───────────────────────────────
cap = {}
def hook(mod, inp, out):
    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
        cap["w"] = out[1].detach().cpu()   # (B, 1, S)
h = model.adapter.readout.attn.register_forward_hook(hook)
with torch.no_grad():
    e = E[:256].to(device); m = M[:256].to(device); st = S[:256].to(device)
    model.sample(e, st, num_steps=cfg.num_flow_steps, img_mask=m)
h.remove()
w = cap["w"].mean(dim=(0, 1)).numpy()   # (S,)
img_w = float(w[layout.numpy()].sum()); txt_w = float(w[(~layout).numpy()].sum())
print(f"\nReadout attention mass: image={img_w:.3f}  text={txt_w:.3f}  "
      f"(per-token: image={img_w/int(layout.sum()):.4f}  text={txt_w/int((~layout).sum()):.4f})")

out = {"loss": res, "readout_attn_over_tokens": w.tolist(),
       "readout_img_mass": img_w, "readout_txt_mass": txt_w,
       "layout_img_count": int(layout.sum())}
Path(cfg.output_dir, "token_probe", "zero_vs_drop.json").write_text(json.dumps(out, indent=2))
print("\nsaved zero_vs_drop.json")
