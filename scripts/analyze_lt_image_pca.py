"""
scripts/analyze_lt_image_pca.py
───────────────────────────────
PCA of the IMAGE tokens (frozen Qwen patch features), DINO-style.

Qwen is a causal VLM: the 66 image tokens come BEFORE the text, so their hidden states
are instruction-INDEPENDENT — pure visual features of the 6x11 patch grid.  We PCA the
patch features (shared basis over several frames) and map the top-3 PCs to RGB, shown
next to each frame.  Coherent colored regions ⇒ the frozen VLM's visual features
separate things (blocks / arm / table); washed-out ⇒ weak visual structure.

Output → docs/experiments/language_table/lt_image_pca.png
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
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_image_pca.png"


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    GH, GW = cfg.img_grid_h, cfg.img_grid_w
    print("[1/3] Loading VLM …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    from envs.language_table_env import make_lt_env

    @torch.no_grad()
    def patches(image):
        inp = vlm.build_vlm_inputs([image], ["push the blocks together"], device)
        tok, im = vlm.encode_vlm(inp)
        return tok[0][im[0]].float().cpu().numpy()          # (66, 1024) image tokens

    print("[2/3] Collecting frames + image-token features …")
    env = make_lt_env("block2block", seed=8)
    frames, feats = [], []
    for _ in range(8):
        out = env.reset(); o = out[0] if isinstance(out, tuple) else out
        rgb = np.asarray(o["rgb"])
        frames.append(rgb)
        feats.append(patches(resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h)))
    env.close()

    # shared PCA basis over all patches
    allp = np.concatenate(feats, 0)                          # (8*66, 1024)
    mean = allp.mean(0); Xc = allp - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PCs = Vt[:3]                                             # (3, 1024)
    N_PC = 5
    Vpc = Vt[:N_PC]                                          # (N_PC, 1024)
    proj_all = Xc @ Vpc.T                                    # (Npatch, N_PC)
    gmax = np.abs(proj_all).max(0) + 1e-9                    # per-PC symmetric scale
    ev = (S[:N_PC] ** 2 / (S ** 2).sum() * 100)

    show = [0, 1, 2, 3]
    fig, axes = plt.subplots(1 + N_PC, len(show), figsize=(3.4 * len(show), 2.1 * (1 + N_PC)))
    for c, idx in enumerate(show):
        axes[0, c].imshow(frames[idx]); axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        if c == 0:
            axes[0, c].set_ylabel("frame", fontsize=11, fontweight="bold")
        proj = (feats[idx] - mean) @ Vpc.T                  # (66, N_PC)
        for r in range(N_PC):
            ax = axes[1 + r, c]
            ax.imshow(proj[:, r].reshape(GH, GW), cmap="coolwarm", vmin=-gmax[r], vmax=gmax[r],
                      interpolation="bicubic", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"PC{r+1}\n{ev[r]:.0f}% var", fontsize=10, fontweight="bold")

    fig.suptitle(
        "What does the IMAGE embedding encode? — each principal component of the frozen Qwen patch features (6x11 grid) as a spatial map\n"
        f"Instruction-independent (pure vision). Per-PC variance {('/'.join(f'{e:.0f}' for e in ev))}% — "
        "red/blue = +/- along that component. Look for components that light up on the arm, the table, or block regions.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   per-PC variance explained: {ev.round(1).tolist()}")
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
