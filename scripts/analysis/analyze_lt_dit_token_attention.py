"""
scripts/analyze_lt_dit_token_attention.py
─────────────────────────────────────────
How does the DiT action decoder attend to EACH token it cross-attends to?

The DiT cross-attends to all 81 Qwen tokens (66 image patches + 15 text tokens).
We capture the cross-attention (averaged over heads, decoder blocks, and the action
query steps) → one weight per token, then split it:
  • image tokens → spatial 6x11 heatmap over the frame
  • text  tokens → labelled bar (each decoded instruction/template token)
and report the total image-vs-text attention share (vs the 66:15 uniform baseline).

Output → docs/experiments/language_table/lt_dit_token_attention.png
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
from PIL import Image

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_dit_token_attention.png"


class CrossAttnCapture:
    def __init__(self, decoder):
        self.store = []
        self.h = [b.cross_attn.register_forward_hook(self._h) for b in decoder.blocks]
    def _h(self, m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            self.store.append(o[1].detach())
    def reset(self): self.store = []
    def per_token(self):                       # (B, KV) — mean over blocks(0) and action-query steps(2)
        return torch.stack(self.store, 0).mean(dim=(0, 2))
    def remove(self):
        for h in self.h: h.remove()


def overlay(rgb, grid):
    H, W = rgb.shape[:2]
    g = grid - grid.min(); g = g / (g.max() + 1e-9)
    gi = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    heat = (cm.jet(gi)[..., :3] * 255).astype(np.uint8)
    return (0.55 * rgb.astype(np.float32) + 0.45 * heat).astype(np.uint8)


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    GH, GW = cfg.img_grid_h, cfg.img_grid_w
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    cap = CrossAttnCapture(tm.decoder)
    tok_decoder = vlm.processor.tokenizer
    from envs.language_table_env import make_lt_env, get_instruction

    @torch.no_grad()
    def attn(image, instruction, state_norm):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tokens, im = vlm.encode_vlm(inp)
        st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
        cap.reset()
        _ = tm.sample(tokens, st, num_steps=cfg.num_flow_steps, img_mask=im)
        pt = cap.per_token()[0].float().cpu().numpy()          # (81,)
        ids = inp["input_ids"][0].cpu().numpy(); m = im[0].cpu().numpy()
        return pt, m, ids

    print("[2/3] Collecting samples …")
    env = make_lt_env("block2block", seed=3)
    rows = []
    for _ in range(3):
        out = env.reset(); o = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, o); rgb = np.asarray(o["rgb"])
        st = (np.asarray(o["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
        pt, m, ids = attn(resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h), instr, st)
        rows.append((rgb, instr, pt, m, ids))
    cap.remove(); env.close()

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1, 1.3]})
    for r, (rgb, instr, pt, m, ids) in enumerate(rows):
        img_a = pt[m]; txt_a = pt[~m]
        img_share = img_a.sum() / pt.sum() * 100; txt_share = txt_a.sum() / pt.sum() * 100
        # left: image-token spatial attention
        ax = axes[r, 0]; ax.imshow(overlay(rgb, img_a.reshape(GH, GW))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image tokens (66) — DiT spatial attention\nimage share {img_share:.0f}% vs text {txt_share:.0f}% "
                     f"(uniform: 81/19)", fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f'"{instr}"', fontsize=8.5)
        # right: text-token attention bars (decoded)
        ax = axes[r, 1]
        txt_ids = ids[~m]
        labels = []
        for i in txt_ids:
            s = tok_decoder.decode([int(i)]).strip()
            if not s:
                s = tok_decoder.convert_ids_to_tokens(int(i))
            labels.append(s.replace("\n", "\\n"))
        y = np.arange(len(txt_a))
        top = txt_a >= np.sort(txt_a)[-3] if len(txt_a) >= 3 else np.ones(len(txt_a), bool)
        ax.barh(y, txt_a, color=["#C44E52" if t else "#4C72B0" for t in top], edgecolor="black", lw=0.4)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("DiT cross-attention weight"); ax.set_title("text tokens (15) — per-token attention\n(red = top-3 words)", fontsize=9.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "How the DiT action decoder attends to EACH token (cross-attention, averaged over heads / blocks / action steps)\n"
        "Left: attention over the 66 image patches.  Right: attention over each decoded text token — which instruction words the action looks at.",
        fontsize=12, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
