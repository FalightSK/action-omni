"""
scripts/analyze_lt_pca.py
─────────────────────────
Representation (PCA) analysis: how does the model ORGANISE the data internally?

For samples spanning 4 command types (block2block / block2absolute / separate / point)
we extract three representations and PCA them:
  • vlm_text  — frozen Qwen, mean of the instruction(text)-token hidden states (1024-d)
  • context   — the trained adapter's readout vector that conditions the DiT (512-d)
  • action    — the predicted action chunk (32-d)

Panels:
  A  PCA(vlm_text)  coloured by command type
  B  PCA(context)   coloured by command type
  C  PCA(context)   coloured by referenced object COLOR
  D  PCA(context)   coloured by referenced object SHAPE
  E  PCA(action)    coloured by command type
  F  separability (silhouette on top-10 PCs): which attribute each representation
     encodes — command type vs color vs shape, for vlm_text / context / action.

Output → docs/experiments/language_table/lt_pca.png
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

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_pca.png"
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["cube", "moon", "star", "pentagon", "crescent"]
CMDS = ["block2block", "block2absolute", "separate", "point"]
REQUIRE = {"separate": "separate", "point": "point"}
_OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")
CMD_COL = {"block2block": "#4C72B0", "block2absolute": "#6FA8DC", "separate": "#DD8452", "point": "#55A868"}
COL_COL = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
SH_COL = dict(zip(SHAPES, ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]))


def pca(X, k=2):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def silhouette(X, labels):
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="adapter+decoder checkpoint (default: exp01 best.pt)")
    ap.add_argument("--data-root", default=None, help="LT_DATASET_ROOT for stats (default: exp01 5k set)")
    ap.add_argument("--out", default=None, help="output figure path")
    args = ap.parse_args()
    if args.data_root:
        os.environ["LT_DATASET_ROOT"] = str(Path(args.data_root).resolve())
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    out_path = Path(args.out) if args.out else OUT
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt_path = args.checkpoint or str(Path(cfg.output_dir) / "checkpoints" / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    print(f"   checkpoint: {ckpt_path}")
    from envs.language_table_env import make_lt_env, get_instruction

    @torch.no_grad()
    def reps(image, instruction, state_norm):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        txt = ~im[0]
        vlm_text = tok[0][txt].mean(0).float().cpu().numpy()
        st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
        context, _ = tm.adapter(tok, im, return_tokens=True)
        ctx = context[0].float().cpu().numpy()
        acts = []
        for sd in range(4):
            torch.manual_seed(sd);  torch.cuda.manual_seed_all(sd) if torch.cuda.is_available() else None
            acts.append(tm.sample(tok, st, num_steps=cfg.num_flow_steps, img_mask=im)[0].float().cpu().numpy())
        return vlm_text, ctx, np.mean(acts, 0)

    print("[2/3] Collecting samples across command types …")
    K = 28
    VT, CT, AC, cmd, col, sh = [], [], [], [], [], []
    for factory in CMDS:
        env = make_lt_env(factory, seed=13)
        req = REQUIRE.get(factory)
        n = 0
        while n < K:
            out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, obs)
            if req and req not in instr.lower():
                continue
            m = _OBJ.search(instr.lower())
            if not m:
                continue
            image = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
            st = (np.asarray(obs["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
            vt, ct, ac = reps(image, instr, st)
            VT.append(vt); CT.append(ct); AC.append(ac)
            cmd.append(factory); col.append(m.group(1)); sh.append(m.group(2)); n += 1
        env.close()
    VT, CT, AC = np.array(VT), np.array(CT), np.array(AC)
    print(f"   collected {len(VT)} samples")

    # ── figure ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    def scatter(ax, X2, labels, palette, title):
        for lab in palette:
            mask = np.array(labels) == lab
            if mask.any():
                ax.scatter(X2[mask, 0], X2[mask, 1], s=26, color=palette[lab], label=lab, edgecolor="black", lw=0.3, alpha=0.85)
        ax.set_title(title, fontsize=10.5, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="best")

    scatter(axes[0, 0], pca(VT), cmd, CMD_COL, "A. PCA(frozen Qwen text emb)\nby command type")
    scatter(axes[0, 1], pca(CT), cmd, CMD_COL, "B. PCA(trained adapter context)\nby command type")
    scatter(axes[0, 2], pca(AC), cmd, CMD_COL, "E. PCA(predicted action)\nby command type")
    scatter(axes[1, 0], pca(CT), col, COL_COL, "C. PCA(adapter context)\nby referenced COLOR")
    scatter(axes[1, 1], pca(CT), sh, SH_COL, "D. PCA(adapter context)\nby referenced SHAPE")

    # F. silhouette separability (top-10 PCs)
    ax = axes[1, 2]
    reps_map = {"frozen Qwen": VT, "adapter ctx": CT, "action": AC}
    labs_map = {"command type": cmd, "color": col, "shape": sh}
    width = 0.25; x = np.arange(len(reps_map))
    for j, (lname, lab) in enumerate(labs_map.items()):
        vals = [silhouette(pca(reps_map[r], 10), lab) for r in reps_map]
        ax.bar(x + (j - 1) * width, vals, width, label=lname)
        for xi, v in zip(x + (j - 1) * width, vals):
            ax.annotate(f"{v:.02f}", (xi, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(list(reps_map.keys()), fontsize=9)
    ax.set_ylabel("silhouette (separability)")
    ax.set_title("F. Which attribute does each\nrepresentation separate? (↑=cleaner clusters)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Representation (PCA) analysis — how the model organises the data\n"
        "Does the internal representation cluster by command TYPE, by the named object's COLOR, by its SHAPE? "
        "(silhouette in F quantifies which attribute each stage actually encodes)",
        fontsize=12, fontweight="bold", y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    # console summary
    for lname, lab in labs_map.items():
        print(f"   silhouette[{lname}]: " + "  ".join(f"{r}={silhouette(pca(reps_map[r],10), lab):.3f}" for r in reps_map))
    print(f"[3/3] saved → {out_path}")


if __name__ == "__main__":
    main()
