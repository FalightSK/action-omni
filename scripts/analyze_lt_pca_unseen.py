"""
scripts/analyze_lt_pca_unseen.py
────────────────────────────────
Are UNSEEN objects represented like SEEN ones? (representation-level, no retraining)

The 16-block pool splits into 8 objects the policy trained on (SEEN) and 8 it did not
(UNSEEN) — balanced across colors/shapes, so each unseen block is a NOVEL color×shape
combination of seen attributes. NOTE: the BLOCK_8 sim only ever places the 8 SEEN
blocks, so the policy was never *behaviourally* tested on unseen objects; here we probe
the model's REPRESENTATION of each object word, with the scene/template held fixed and
only the object name varied, to ask whether unseen object words land among the seen.

For each object we embed several carrier instructions ("push the <obj> to ...") over a
few fixed frames and average:
  vlm_text — frozen Qwen instruction-token mean    context — trained adapter readout

Panels: A/B PCA(per-object) by COLOR / SHAPE (○=seen ✕=unseen, labelled);
        C PCA(context) seen vs unseen;  D silhouette seen-vs-unseen (≈0 ⇒ intermingled);
        E silhouette by color & shape (does it cluster compositionally?);
        F each UNSEEN object's nearest SEEN object — shares color / shape / neither.

Output → docs/experiments/language_table/lt_pca_unseen.png
"""
from __future__ import annotations
import os, sys
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

OUT = ROOT / "docs/experiments/language_table/lt_pca_unseen.png"
SEEN = ["blue_cube", "blue_moon", "green_cube", "green_star", "red_moon", "red_pentagon", "yellow_pentagon", "yellow_star"]
UNSEEN = ["blue_pentagon", "blue_star", "green_moon", "green_pentagon", "red_cube", "red_star", "yellow_cube", "yellow_moon"]
OBJS = SEEN + UNSEEN
COL_COL = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
SH_COL = {"cube": "#4C72B0", "moon": "#DD8452", "star": "#55A868", "pentagon": "#8172B3"}
TEMPLATES = ["push the {} to the center of the board", "move the {} to the top left",
             "slide the {} next to the other block"]


def pca(X, k=2):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
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
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    from envs.language_table_env import make_lt_env, get_instruction

    # a few fixed frames + states to average over
    env = make_lt_env("block2block", seed=2)
    frames, states = [], []
    for _ in range(3):
        out = env.reset(); o = out[0] if isinstance(out, tuple) else out
        frames.append(resize_frame(np.asarray(o["rgb"]), cfg.lt_img_w, cfg.lt_img_h))
        states.append((np.asarray(o["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8))
    env.close()

    @torch.no_grad()
    def embed(instruction, image, st):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        vt = tok[0][~im[0]].mean(0).float().cpu().numpy()
        stt = torch.from_numpy(st.astype(np.float32)).unsqueeze(0).to(device)
        context, _ = tm.adapter(tok, im, return_tokens=True)
        return vt, context[0].float().cpu().numpy()

    print("[2/3] Embedding each object word (scene/template fixed) …")
    VT, CT = [], []
    for o in OBJS:
        words = o.replace("_", " ")
        vts, cts = [], []
        for t in TEMPLATES:
            for fr, st in zip(frames, states):
                vt, ct = embed(t.format(words), fr, st)
                vts.append(vt); cts.append(ct)
        VT.append(np.mean(vts, 0)); CT.append(np.mean(cts, 0))
    VT, CT = np.array(VT), np.array(CT)
    is_seen = np.array([o in SEEN for o in OBJS])
    color = [o.split("_")[0] for o in OBJS]; shape = [o.split("_")[1] for o in OBJS]
    seen_lab = ["seen" if s else "unseen" for s in is_seen]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    Pv = pca(VT); Pc = pca(CT)

    def obj_map(ax, P, attr, palette, name):
        for i, o in enumerate(OBJS):
            ax.scatter(P[i, 0], P[i, 1], s=150, color=palette[attr[i]],
                       marker="o" if is_seen[i] else "X", edgecolor="black", lw=0.9, zorder=3)
            ax.annotate(o.replace("_", " "), (P[i, 0], P[i, 1]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax.set_title(name, fontsize=10, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])

    obj_map(axes[0, 0], Pv, color, COL_COL, "A. Frozen Qwen object emb — by COLOR\n○=seen  ✕=unseen")
    obj_map(axes[0, 1], Pv, shape, SH_COL, "B. Frozen Qwen object emb — by SHAPE\n○=seen  ✕=unseen")

    # C. seen vs unseen scatter (context)
    ax = axes[0, 2]
    for lab, c in [("seen", "#4C72B0"), ("unseen", "#DD8452")]:
        m = np.array(seen_lab) == lab
        ax.scatter(Pc[m, 0], Pc[m, 1], s=90, color=c, label=lab, edgecolor="black", lw=0.5)
    ax.set_title("C. Adapter context — seen vs unseen", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)

    # D. silhouette seen-vs-unseen
    ax = axes[1, 0]
    sv = [silhouette(pca(VT, 8), seen_lab), silhouette(pca(CT, 8), seen_lab)]
    ax.bar([0, 1], sv, color=["#4C72B0", "#55A868"], edgecolor="black")
    for i, v in enumerate(sv):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen Qwen", "adapter ctx"])
    ax.set_ylabel("silhouette (seen vs unseen)")
    ax.set_title("D. Are unseen objects off-distribution?\n≈0 ⇒ intermingled ⇒ generalises", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # E. silhouette by color & shape (compositional clustering?)
    ax = axes[1, 1]
    sc = [silhouette(pca(VT, 8), color), silhouette(pca(VT, 8), shape)]
    ax.bar([0, 1], sc, color=["#C44E52", "#8172B3"], edgecolor="black")
    for i, v in enumerate(sc):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["by color", "by shape"])
    ax.set_ylabel("silhouette (frozen Qwen)")
    ax.set_title("E. Does the object emb cluster by\ncolor / shape? (compositional axes)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # F. compositional NN
    ax = axes[1, 2]
    D = cdist(VT, VT)
    seen_idx = [i for i in range(len(OBJS)) if is_seen[i]]
    same_c = same_s = neither = 0
    for ui in [i for i in range(len(OBJS)) if not is_seen[i]]:
        nn = min(seen_idx, key=lambda si: D[ui, si])
        if color[nn] == color[ui]:
            same_c += 1
        elif shape[nn] == shape[ui]:
            same_s += 1
        else:
            neither += 1
    nu = len(UNSEEN)
    ax.bar([0, 1, 2], [same_c, same_s, neither], color=["#C44E52", "#8172B3", "#BBBBBB"], edgecolor="black")
    for i, v in enumerate([same_c, same_s, neither]):
        ax.annotate(f"{v}/{nu}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["same\ncolor", "same\nshape", "neither"])
    ax.set_ylabel("# unseen objects"); ax.set_ylim(0, nu)
    ax.set_title("F. Each UNSEEN object's nearest SEEN object\n(frozen Qwen — compositional?)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Representation of SEEN vs UNSEEN objects (8 trained + 8 never-trained, balanced across color/shape)\n"
        "Probing the object WORD (scene fixed). NOTE: BLOCK_8 only spawns the 8 seen blocks, so this is representation-level, not a behavioural unseen-object test.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   silhouette seen-vs-unseen: frozenQwen={sv[0]:.3f} context={sv[1]:.3f}")
    print(f"   silhouette by color={sc[0]:.3f} by shape={sc[1]:.3f}")
    print(f"   unseen NN: same-color={same_c} same-shape={same_s} neither={neither} (of {nu})")
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
