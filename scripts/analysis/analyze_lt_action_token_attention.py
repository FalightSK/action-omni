"""
scripts/analyze_lt_action_token_attention.py
────────────────────────────────────────────
What does EACH action token (the 16 denoised steps of the predicted chunk) attend to?

We capture the DiT cross-attention WITHOUT averaging over the action-query dim →
a (16 action tokens × 81 KV tokens) matrix (averaged over heads, blocks, flow steps).
Then:
  • Top row  — spatial image attention for action steps 0,3,6,9,12,15 (shared scale):
               does the spatial focus MOVE across the planned horizon?
  • Bottom-L — heatmap (16 action steps × 15 text tokens): which instruction words each
               action step attends to (do early vs late steps differ?).
  • Bottom-R — per-step image-vs-text attention share across the horizon.

Output → docs/experiments/language_table/lt_action_token_attention.png
"""
from __future__ import annotations
import os, sys
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_action_token_attention.png"


class CrossAttnCapture:
    def __init__(self, decoder):
        self.store = []
        self.h = [b.cross_attn.register_forward_hook(self._h) for b in decoder.blocks]
    def _h(self, m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            self.store.append(o[1].detach())
    def reset(self): self.store = []
    def per_step(self):                         # (Q, KV) — mean over blocks+flow-steps, batch 0
        return torch.stack(self.store, 0).mean(0)[0]
    def remove(self):
        for h in self.h: h.remove()


def overlay(rgb, grid, gmax):
    H, W = rgb.shape[:2]
    g = np.clip(grid / (gmax + 1e-9), 0, 1)
    gi = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    heat = (cm.jet(gi)[..., :3] * 255).astype(np.uint8)
    return (0.55 * rgb.astype(np.float32) + 0.45 * heat).astype(np.uint8)


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    GH, GW, Hh = cfg.img_grid_h, cfg.img_grid_w, cfg.action_horizon
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    cap = CrossAttnCapture(tm.decoder)
    tokz = vlm.processor.tokenizer
    from envs.language_table_env import make_lt_env, get_instruction

    print("[2/3] One rollout, capturing per-action-token attention …")
    env = make_lt_env("block2block", seed=3)
    out = env.reset(); o = out[0] if isinstance(out, tuple) else out
    instr = get_instruction(env, o); rgb = np.asarray(o["rgb"])
    st = (np.asarray(o["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
    inp = vlm.build_vlm_inputs([resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h)], [instr], device)
    tokens, im = vlm.encode_vlm(inp)
    stt = torch.from_numpy(st.astype(np.float32)).unsqueeze(0).to(device)
    cap.reset()
    _ = tm.sample(tokens, stt, num_steps=cfg.num_flow_steps, img_mask=im)
    A = cap.per_step().float().cpu().numpy()        # (Q=16, 81)
    cap.remove(); env.close()
    m = im[0].cpu().numpy(); ids = inp["input_ids"][0].cpu().numpy()
    img_A = A[:, m]                                  # (16, 66)
    txt_A = A[:, ~m]                                 # (16, 15)
    labels = []
    for i in ids[~m]:
        s = tokz.decode([int(i)]).strip() or tokz.convert_ids_to_tokens(int(i))
        labels.append(s.replace("\n", "\\n"))

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.25], hspace=0.32, wspace=0.35)
    steps = [0, 3, 6, 9, 12, 15]
    gmax = img_A.max()
    for i, s in enumerate(steps):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(overlay(rgb, img_A[s].reshape(GH, GW), gmax))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"action step {s}", fontsize=9.5, fontweight="bold")

    axh = fig.add_subplot(gs[1, 0:4])
    im2 = axh.imshow(txt_A, aspect="auto", cmap="viridis")
    axh.set_xticks(range(len(labels))); axh.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axh.set_yticks(range(Hh)); axh.set_yticklabels(range(Hh), fontsize=7)
    axh.set_ylabel("action step"); axh.set_xlabel("text token")
    axh.set_title("Per-action-step attention to each TEXT token", fontsize=10.5, fontweight="bold")
    fig.colorbar(im2, ax=axh, fraction=0.025, pad=0.02)

    axl = fig.add_subplot(gs[1, 4:6])
    img_share = img_A.sum(1) / A.sum(1) * 100; txt_share = txt_A.sum(1) / A.sum(1) * 100
    axl.plot(range(Hh), img_share, "-o", color="#4C72B0", label="image tokens", ms=4)
    axl.plot(range(Hh), txt_share, "-o", color="#C44E52", label="text tokens", ms=4)
    axl.axhline(66 / 81 * 100, color="#4C72B0", ls=":", lw=1, alpha=0.6)
    axl.axhline(15 / 81 * 100, color="#C44E52", ls=":", lw=1, alpha=0.6)
    axl.set_xlabel("action step"); axl.set_ylabel("attention share (%)")
    axl.set_title("Image vs text share per action step\n(dotted = uniform 81/19)", fontsize=10, fontweight="bold")
    axl.legend(fontsize=8.5); axl.grid(alpha=0.3)

    fig.suptitle(
        f"What each ACTION TOKEN attends to — DiT cross-attention per query step (no averaging over the 16 steps)\n"
        f'"{instr}"   (top: spatial image attention per step, shared scale;  bottom: per-step text-token attention + image/text share)',
        fontsize=12, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   image-share per step range: {img_share.min():.0f}-{img_share.max():.0f}%")
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
