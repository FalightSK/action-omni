"""
scripts/analyze_lt_image_scatter.py
───────────────────────────────────
Embedding PCA scatter (one dot per scene) at THREE pipeline stages, colored the same way,
so you can see what each stage carries — including AFTER the adapter:

  TEXT emb   = frozen Qwen, mean of instruction(text) tokens   (pre-adapter, instr-dependent)
  IMAGE emb  = frozen Qwen, mean of 66 image patch tokens      (pre-adapter, instr-INDEPENDENT)
  ADAPTER ctx= trained adapter readout over ALL tokens         (post-adapter, image+text pooled)

Rows = stage, columns = coloring (command type / referenced color / referenced shape).
Silhouette in each title = how cleanly that stage separates by that label.

Output → docs/experiments/language_table/lt_image_scatter.png
"""
from __future__ import annotations
import os, sys, re
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_image_scatter.png"
CMDS = ["block2block", "block2absolute", "separate", "point"]
REQUIRE = {"separate": "separate", "point": "point"}
_OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")
CMD_COL = {"block2block": "#4C72B0", "block2absolute": "#6FA8DC", "separate": "#DD8452", "point": "#55A868"}
COL_COL = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
SH_COL = {"cube": "#4C72B0", "moon": "#DD8452", "star": "#55A868", "pentagon": "#8172B3", "crescent": "#937860"}


def pca(X, k=2):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def silh(X, labels):
    labs = np.array(labels); uniq = np.unique(labs)
    if len(uniq) < 2:
        return 0.0
    D = cdist(X, X); s = []
    for i in range(len(X)):
        same = labs == labs[i]; same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labs == u].mean() for u in uniq if u != labs[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    from envs.language_table_env import make_lt_env, get_instruction

    @torch.no_grad()
    def embeds(image, instruction):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        t = tok[0][~im[0]].mean(0).float().cpu().numpy()          # frozen text
        v = tok[0][im[0]].mean(0).float().cpu().numpy()           # frozen image
        context, _ = tm.adapter(tok, im, return_tokens=True)      # adapter readout
        c = context[0].float().cpu().numpy()
        return t, v, c

    print("[2/3] Collecting across command types …")
    K = 28
    TX, IM, CT, cmd, col, sh = [], [], [], [], [], []
    for f in CMDS:
        env = make_lt_env(f, seed=13)
        req = REQUIRE.get(f); n = 0
        while n < K:
            out = env.reset(); o = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, o)
            if req and req not in instr.lower():
                continue
            m = _OBJ.search(instr.lower())
            if not m:
                continue
            t, v, c = embeds(resize_frame(np.asarray(o["rgb"]), cfg.lt_img_w, cfg.lt_img_h), instr)
            TX.append(t); IM.append(v); CT.append(c); cmd.append(f); col.append(m.group(1)); sh.append(m.group(2)); n += 1
        env.close()
    TX, IM, CT = np.array(TX), np.array(IM), np.array(CT)

    stages = [("TEXT emb  (frozen, pre-adapter)", TX),
              ("IMAGE emb  (frozen, pre-adapter)", IM),
              ("ADAPTER context  (trained, post-adapter)", CT)]
    specs = [("command type", cmd, CMD_COL), ("referenced color", col, COL_COL), ("referenced shape", sh, SH_COL)]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 15))
    for r, (sname, X) in enumerate(stages):
        P = pca(X)
        for cidx, (name, lab, pal) in enumerate(specs):
            ax = axes[r, cidx]
            for k in pal:
                m = np.array(lab) == k
                if m.any():
                    ax.scatter(P[m, 0], P[m, 1], s=26, color=pal[k], label=k, edgecolor="black", lw=0.3, alpha=0.85)
            ax.set_title(f"{sname}\nby {name}   (sil {silh(pca(X,10), lab):+.2f})", fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=6.5, loc="best")

    fig.suptitle(
        "Embedding PCA across the pipeline — frozen TEXT → frozen IMAGE → ADAPTER context (after the adapter)\n"
        "Text clusters by command (sil ~0.4); image is intermingled (~0, scene only); the adapter POOLS image+text → command "
        "structure partly survives (~0.2) but is diluted by the task-agnostic image. Object color/shape never separate at any stage.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    for sname, X in stages:
        print(f"   {sname:42s} sil: cmd={silh(pca(X,10),cmd):+.2f} color={silh(pca(X,10),col):+.2f} shape={silh(pca(X,10),sh):+.2f}")
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
