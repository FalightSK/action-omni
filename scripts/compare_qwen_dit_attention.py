"""
scripts/compare_qwen_dit_attention.py
─────────────────────────────────────
See how the model uses the data at BOTH stages, side by side:

  • QWEN (frozen VLM): the instruction text tokens' attention to the 66 image patches
    (6x11 grid) — "where does the backbone look in the image while reading the command?"
    shown at a middle and a late transformer layer.
  • DiT (action decoder): the cross-attention from the predicted action steps to the
    same image tokens — "where does the action head look when deciding what to do?"

For each scene we render: frame (named block circled) | Qwen mid-layer | Qwen late-layer | DiT.

Output → docs/experiments/language_table/lt_attention_qwen_vs_dit.png
"""
from __future__ import annotations
import os, sys, textwrap
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
from transformers import AutoModel

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_attention_qwen_vs_dit.png"


class CrossAttnCapture:
    def __init__(self, decoder):
        self.store = []
        self.handles = [b.cross_attn.register_forward_hook(self._h) for b in decoder.blocks]
    def _h(self, m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            self.store.append(o[1].detach())
    def reset(self): self.store = []
    def agg(self): return torch.stack(self.store, 0).mean(dim=(0, 2))
    def remove(self):
        for h in self.handles: h.remove()


def cam(env):
    vm, pm, *_ = env.calc_camera_params(env._image_size)
    return (np.array(vm, np.float64).reshape((4, 4), order="F"),
            np.array(pm, np.float64).reshape((4, 4), order="F"),
            env._image_size[1], env._image_size[0])


def project(view, proj, W, H, xy, z=0.02):
    P = np.array([xy[0], xy[1], z, 1.0]); c = proj @ (view @ P); n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1.0 - (n[1] * 0.5 + 0.5)) * H)


def blk(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b]); return np.array(p[:2])


def overlay(rgb, grid):
    H, W = rgb.shape[:2]
    g = grid - grid.min(); g = g / (g.max() + 1e-9)
    gi = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    heat = (cm.jet(gi)[..., :3] * 255).astype(np.uint8)
    return (0.5 * rgb.astype(np.float32) + 0.5 * heat).astype(np.uint8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.data_root:
        os.environ["LT_DATASET_ROOT"] = str(Path(args.data_root).resolve())
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    out_path = Path(args.out) if args.out else OUT
    GH, GW = cfg.img_grid_h, cfg.img_grid_w
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)

    print("[1/3] Loading VLM (eager attention) + checkpoint …")
    vlm = VLAModel(cfg)
    old = vlm.vlm; vlm.vlm = None; del old
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    vlm.vlm = AutoModel.from_pretrained(cfg.model_path, dtype=torch.bfloat16,
                                        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt_path = args.checkpoint or str(Path(cfg.output_dir) / "checkpoints" / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    print(f"   checkpoint: {ckpt_path}")
    cap = CrossAttnCapture(tm.decoder)

    from envs.language_table_env import make_lt_env, get_instruction
    env = make_lt_env("block2block", seed=4)
    view, proj, W, H = cam(env)

    @torch.no_grad()
    def analyse(image, instruction, state_norm):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        out = vlm.vlm(**inp, output_attentions=True, output_hidden_states=True, return_dict=True)
        ids = inp["input_ids"][0]
        img_mask = (ids == cfg.image_token_id)
        img_idx = torch.where(img_mask)[0]; txt_idx = torch.where(~img_mask)[0]
        # Qwen3.5 is a HYBRID linear/full-attention model: only the full-attention
        # layers expose an attention matrix, so out.attentions has 6 (of 24) entries,
        # each (1, 8 heads, S, S) over the full image+text sequence.
        atts = out.attentions
        nL = len(atts)
        early, late = 1, nL - 1                      # 2nd full-attn layer + deepest

        def qwen_grid(L):
            A = atts[L][0].float().mean(0)          # (S,S) head-avg
            v = A[txt_idx][:, img_idx].mean(0)      # text->image, mean over text queries
            return v.cpu().numpy().reshape(GH, GW), L

        qm, Lm = qwen_grid(early); ql, Ll = qwen_grid(late)
        n_full = nL

        tokens = out.last_hidden_state.float()
        cap.reset()
        st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
        _ = tm.sample(tokens, st, num_steps=cfg.num_flow_steps, img_mask=img_mask.unsqueeze(0))
        dit = cap.agg()[0].float().cpu().numpy()[img_mask.cpu().numpy()].reshape(GH, GW)
        return qm, Lm, ql, Ll, dit, n_full

    print("[2/3] Collecting scenes …")
    scenes = []
    for _ in range(3):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs); rc = env._reward_calculator
        rgb = np.asarray(obs["rgb"])
        st = (np.asarray(obs["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
        scenes.append((rgb, instr, rc._start_block, st))

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5))
    col_titles = ["frame (named block circled)", "Qwen text→image (early full-attn)",
                  "Qwen text→image (deepest full-attn)", "DiT cross-attn"]
    for r, (rgb, instr, start, st) in enumerate(scenes):
        qm, Lm, ql, Ll, dit, nF = analyse(resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h), instr, st)
        px, py = project(view, proj, W, H, blk(env, start))
        panels = [rgb, overlay(rgb, qm), overlay(rgb, ql), overlay(rgb, dit)]
        instr_wrapped = "\n".join(textwrap.wrap(f'"{instr}"', width=30))
        labels = [instr_wrapped, f"Qwen full-attn {Lm+1}/{nF}", f"Qwen full-attn {Ll+1}/{nF} (deepest)", "DiT action→image"]
        for c in range(4):
            ax = axes[r, c]; ax.imshow(panels[c]); ax.set_xticks([]); ax.set_yticks([])
            ec = "white" if c else "yellow"
            ax.add_patch(plt.Circle((px, py), 11, fill=False, ec=ec, lw=2.2))
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight="bold")
            ax.set_xlabel(labels[c], fontsize=8.5 if c else 9.5,
                          fontweight="bold" if c == 0 else "normal")
    cap.remove(); env.close()
    fig.suptitle(
        "How the model uses the data — attention at BOTH stages (same scene + instruction; named block circled)\n"
        "Qwen (frozen VLM, hybrid linear/full-attn — only 6 of 24 layers expose attention): the instruction text's attention to the image patches.   "
        "DiT (action decoder): the predicted action's cross-attention to the image tokens.",
        fontsize=12, fontweight="bold", y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[3/3] saved → {out_path}")


if __name__ == "__main__":
    main()
