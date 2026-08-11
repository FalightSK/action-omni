"""
scripts/analyze_lt_attention.py
───────────────────────────────
Attention analysis for the Language Table VLA.

The DiT action decoder cross-attends each predicted action step to the 120 VLM
tokens (66 image patches on a 6x11 grid + the instruction/template text tokens).
We capture those cross-attention weights with a forward hook and ask:

  1. SPATIAL  — overlay the image-token attention on the frame: does the action
     decoder look at the block the instruction names? (named block circled via the
     verified camera projection)
  2. CAUSAL   — same frame, swap which block the instruction names: does the
     attention move to the newly-named block? (the real test of language grounding)
  3. TEXT     — what fraction of cross-attention mass lands on the instruction/text
     tokens vs the image tokens? (MILESTONES Phase-3: > 5% text attention?)

Output → docs/experiments/language_table/lt_attention.png
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

OUT = ROOT / "docs/experiments/language_table/lt_attention.png"


class CrossAttnCapture:
    """Hooks every DiT block's cross-attn to grab head-averaged weights (B,L,S)."""
    def __init__(self, decoder):
        self.store = []
        self.handles = [b.cross_attn.register_forward_hook(self._hook) for b in decoder.blocks]

    def _hook(self, m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            self.store.append(o[1].detach())

    def reset(self):
        self.store = []

    def agg(self):                       # mean over (calls=steps*layers) and query steps → (B,S)
        return torch.stack(self.store, 0).mean(dim=(0, 2))

    def remove(self):
        for h in self.handles:
            h.remove()


def cam(env):
    viewm, projm, *_ = env.calc_camera_params(env._image_size)
    view = np.array(viewm, np.float64).reshape((4, 4), order="F")
    proj = np.array(projm, np.float64).reshape((4, 4), order="F")
    H, W = env._image_size
    return view, proj, W, H


def project(view, proj, W, H, xy, z=0.02):
    P = np.array([xy[0], xy[1], z, 1.0]); clip = proj @ (view @ P); ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * W, (1.0 - (ndc[1] * 0.5 + 0.5)) * H)


def attention_for(vlm, tm, cfg, image, instruction, state_norm, device, cap):
    inputs = vlm.build_vlm_inputs([image], [instruction], device)
    tokens, img_mask = vlm.encode_vlm(inputs)
    cap.reset()
    st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
    _ = tm.sample(tokens, st, num_steps=cfg.num_flow_steps, img_mask=img_mask)
    attn = cap.agg()[0].float().cpu().numpy()          # (S,)
    mask = img_mask[0].cpu().numpy().astype(bool)      # (S,) True=image token
    total = attn.sum() + 1e-9
    img_attn = attn[mask]                              # (66,)
    text_frac = float(attn[~mask].sum() / total)
    grid = img_attn.reshape(cfg.img_grid_h, cfg.img_grid_w)
    return grid, text_frac


def overlay(rgb, grid):
    H, W = rgb.shape[:2]
    g = grid - grid.min(); g = g / (g.max() + 1e-9)
    gimg = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    heat = (cm.jet(gimg)[..., :3] * 255).astype(np.uint8)
    return (0.5 * rgb.astype(np.float32) + 0.5 * heat).astype(np.uint8)


def block_xy(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b])
    return np.array(p[:2])


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    cap = CrossAttnCapture(tm.decoder)

    from envs.language_table_env import make_lt_env, get_instruction
    env = make_lt_env("block2block", seed=3)
    view, proj, W, H = cam(env)
    GH, GW = cfg.img_grid_h, cfg.img_grid_w

    def cell_of(xy):
        px, py = project(view, proj, W, H, xy)
        return (min(GH - 1, max(0, int(py / H * GH))), min(GW - 1, max(0, int(px / W * GW))))

    def grab():
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs)
        rc = env._reward_calculator
        rgb = np.asarray(obs["rgb"])
        st = (np.asarray(obs["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
        return dict(instr=instr, rgb=rgb, st=st, start=rc._start_block,
                    target=rc._target_block, blocks=list(env._blocks_on_table))

    print("[2/3] Collecting frames + attention over N samples …")
    N = 16
    chance = 1.0 / (GH * GW)
    named_v, other_v, bg_v, text_fracs, shown = [], [], [], [], []
    for i in range(N):
        s = grab()
        grid, tf = attention_for(vlm, tm, cfg, resize_frame(s["rgb"], cfg.lt_img_w, cfg.lt_img_h),
                                 s["instr"], s["st"], device, cap)
        gi = grid / (grid.sum() + 1e-9)                       # image-attention as a distribution
        bcells = {b: cell_of(block_xy(env, b)) for b in s["blocks"]}
        nc = bcells[s["start"]]
        named_v.append(gi[nc])
        other_cells = [c for b, c in bcells.items() if b != s["start"]]
        other_v.append(np.mean([gi[c] for c in other_cells]) if other_cells else np.nan)
        occupied = set(bcells.values())
        bg = [gi[r, c] for r in range(GH) for c in range(GW) if (r, c) not in occupied]
        bg_v.append(np.mean(bg) if bg else np.nan)
        text_fracs.append(tf)
        if i < 3:
            shown.append((s, grid))

    # causal pair from the first shown sample
    s0, _ = shown[0]
    alt = next((b for b in s0["blocks"] if b not in (s0["start"], s0["target"])), s0["start"])
    instr_alt = f"push the {alt.replace('_', ' ')} to the {s0['target'].replace('_', ' ')}"

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    # Row 1 — example heatmaps (named block circled)
    for k, (s, grid) in enumerate(shown):
        ax = axes[0, k]; ax.imshow(overlay(s["rgb"], grid))
        px, py = project(view, proj, W, H, block_xy(env, s["start"]))
        ax.add_patch(plt.Circle((px, py), 11, fill=False, ec="white", lw=2.2))
        ax.text(px + 8, py, s["start"].replace("_", " "), color="white", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"“{s['instr'][:40]}”\n(named block circled)", fontsize=9)

    # Row 2 col 0 — grounding metric: attention on named vs other blocks vs background
    ax = axes[1, 0]
    means = [np.nanmean(named_v), np.nanmean(other_v), np.nanmean(bg_v)]
    ax.bar(range(3), [m * 100 for m in means], color=["#55A868", "#4C72B0", "#BBBBBB"], edgecolor="black")
    ax.axhline(chance * 100, color="red", ls="--", lw=1.3, label=f"uniform/chance ({chance*100:.1f}%)")
    for i, m in enumerate(means):
        ax.annotate(f"{m*100:.1f}%", (i, m * 100), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["named\nblock", "other\nblocks", "background"], fontsize=9)
    ax.set_ylabel("image-attention per cell (%)")
    ax.set_title(f"Spatial grounding (N={N})\nattention on the NAMED block vs others", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

    # Row 2 col 1 — text-token attention fraction
    ax = axes[1, 1]
    ax.hist([t * 100 for t in text_fracs], bins=10, color="#DD8452", edgecolor="black")
    ax.axvline(5, color="red", ls="--", lw=1.3, label="5% (Phase-3 bar)")
    ax.axvline(np.mean(text_fracs) * 100, color="black", lw=1.5, label=f"mean {np.mean(text_fracs)*100:.0f}%")
    ax.set_xlabel("text-token attention (%)"); ax.set_ylabel(f"# of {N} samples")
    ax.set_title("Text-token attention fraction\n(decoder reads the instruction)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

    # Row 2 col 2 — causal: same frame, swap named block (attention-on-each-block)
    ax = axes[1, 2]
    pair = [(s0["instr"], s0["start"]), (instr_alt, alt)]
    on_named = []
    for instr, named in pair:
        grid, _ = attention_for(vlm, tm, cfg, resize_frame(s0["rgb"], cfg.lt_img_w, cfg.lt_img_h), instr, s0["st"], device, cap)
        gi = grid / (grid.sum() + 1e-9)
        on_named.append([gi[cell_of(block_xy(env, s0["start"]))] * 100, gi[cell_of(block_xy(env, alt))] * 100])
    xx = np.arange(2); w = 0.36
    ax.bar(xx - w/2, [on_named[0][0], on_named[1][0]], w, color="#55A868", edgecolor="black", label=f"attn on {s0['start'].replace('_',' ')}")
    ax.bar(xx + w/2, [on_named[0][1], on_named[1][1]], w, color="#C44E52", edgecolor="black", label=f"attn on {alt.replace('_',' ')}")
    ax.set_xticks(xx); ax.set_xticklabels([f"name\n{s0['start'].replace('_',' ')}", f"name\n{alt.replace('_',' ')}"], fontsize=8)
    ax.set_ylabel("image-attention on cell (%)")
    ax.set_title("Causal: SAME frame, swap named block\n(does attention follow the instruction?)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    cap.remove(); env.close()
    print(f"   grounding: named={means[0]*100:.2f}% other={means[1]*100:.2f}% bg={means[2]*100:.2f}% chance={chance*100:.2f}%")
    fig.suptitle(
        "Language Table — DiT action-decoder cross-attention (where does it look, and does language steer it?)\n"
        "Row 1: attention heatmap per command (named block circled).  Row 2: SAME frame, swapped instruction → does attention follow the new block?",
        fontsize=12, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[3/3] saved → {OUT}  | text-attn mean={np.mean(text_fracs)*100:.1f}%")


if __name__ == "__main__":
    main()
