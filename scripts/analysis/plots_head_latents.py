"""
scripts/analysis/plots_head_latents.py
──────────────────────────────────────
Two concrete figures to replace abstract attention averages.

Why these, and not the mean-attention bars
──────────────────────────────────────────
`plots_head_diagnostics.py` fig2 reports that PaliGemma's head puts 23.8% of its
cross-attention on the 16 instruction tokens against Pi-0.5's 4.2%. That is the
right number, but a mean over 272 tokens says nothing about WHERE in the image
either head looks, or about what the adapted space actually looks like. Both are
easy to show directly:

fig7_adapter_latents
    UMAP of the adapter's OUTPUT tokens — the exact tensors the DiT
    cross-attends to — pooled separately over image tokens and over text
    tokens, coloured by task. Four panels: 2 arms x 2 pools.

    This makes the routing result visible rather than inferred. If Pi-0.5's
    backbone fused the instruction into its image tokens, its IMAGE pool must
    separate the ten tasks even though the scene is shared. If PaliGemma's did
    not, its image pool must not — and its task identity must live in the text
    pool instead. Silhouette score is printed per panel so the visual impression
    is backed by a number.

fig8_attention_maps
    The 256 image-token attention weights folded back to the 16x16 patch grid
    and overlaid on the real LIBERO frame the head was looking at, for the same
    frames in both arms. ONE colour scale across both arms so the rows are
    directly comparable, with each arm's text-token share printed alongside —
    that share is mass the map does NOT contain, and is the whole point.

Both read the same cached embeddings the policies trained on, so nothing here
re-encodes anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from head_diagnostics import (ARMS, BATCH, DEVICE, frame_labels,  # noqa: E402
                              load_arm, read_batch, stride, val_indices)

HD = ROOT / "asset" / "analysis" / "head_diagnostics"
FIG = HD / "figures"
RUN = {"pi05": "asset/runs/libero/exp01_goal",
       "paligemma": "asset/runs/libero/exp02_paligemma"}
EXP = {"pi05": "exp01", "paligemma": "exp02"}
SHORT = {"pi05": "Pi-0.5  (robot-pretrained)", "paligemma": "PaliGemma-3B  (stock)"}

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})
# Sequential single hue for attention weight — an ordered quantity never gets a
# rainbow, which would invent boundaries the data does not have.
ATT_CMAP = LinearSegmentedColormap.from_list(
    "att", ["#00000000", "#ffd08a80", "#f2762bcc", "#a3170bff"])
# 10 qualitative hues is past safe separability, so panels also report a
# silhouette score and the scene is shared across tasks by construction.
TASK_CMAP = plt.get_cmap("tab10")

N_SCATTER = 1500
SHORT_TASK = {
    "open the middle drawer of the cabinet": "middle drawer",
    "put the bowl on the stove": "bowl → stove",
    "put the wine bottle on top of the cabinet": "wine → cabinet",
    "open the top drawer and put the bowl inside": "top drawer + bowl",
    "put the bowl on top of the cabinet": "bowl → cabinet top",
    "push the plate to the front of the stove": "push plate",
    "put the cream cheese in the bowl": "cheese → bowl",
    "turn on the stove": "turn on stove",
    "put the bowl on the plate": "bowl → plate",
    "put the wine bottle on the rack": "wine → rack",
}


def _style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# fig 7 — adapter latent space
# ──────────────────────────────────────────────────────────────────────────────

def adapter_pools(arm, rows):
    """Mean-pooled adapter OUTPUT per frame, split by token type."""
    model, _, _ = load_arm(EXP[arm], RUN[arm])
    h5 = ROOT / RUN[arm] / "vlm_embeddings.h5"
    img, txt = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            emb, msk, _, _ = read_batch(h5, rows[i:i + BATCH])
            emb, msk = emb.to(DEVICE), msk.to(DEVICE)
            _, h = model.adapter(emb, msk, return_tokens=True)
            img.append((h * msk.unsqueeze(-1)).sum(1)
                       .div(msk.sum(1, keepdim=True)).cpu().numpy())
            nm = ~msk
            txt.append((h * nm.unsqueeze(-1)).sum(1)
                       .div(nm.sum(1, keepdim=True)).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(img, 0), np.concatenate(txt, 0)


N_PC = 50


def task_silhouette(X, y):
    """Silhouette in the top-50 PC space.

    Scored in PC space, not the native width, so a 2048-wide raw feature and a
    512-wide adapted one are comparable — silhouette is not dimension-invariant,
    and scoring natively would confound "the adapter sharpened task structure"
    with "the adapter reduced dimensionality". 50 components matches the Chapter
    1 protocol (N_PC in analyze_joint.py). Defined here rather than in
    plots_raw_vs_adapted so both files score identically; the two reported
    +0.247 and +0.273 for the same quantity before this was shared.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    Xz = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(N_PC, min(Xz.shape) - 1),
            random_state=0).fit_transform(Xz)
    return float(silhouette_score(Z, y)), Xz


def fig7_adapter_latents(rows, task_idx, names):
    import umap

    y = task_idx[rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.4))
    for r, arm in enumerate(ARMS):
        pools = dict(zip(("image", "text"), adapter_pools(arm, rows)))
        for c, pool in enumerate(("image", "text")):
            ax = _style(axes[r][c])
            sil, Xz = task_silhouette(pools[pool], y)
            Z = umap.UMAP(n_neighbors=30, min_dist=0.12, random_state=0,
                          n_components=2).fit_transform(Xz)
            for t in np.unique(y):
                m = y == t
                ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.75,
                           color=TASK_CMAP(t % 10),
                           label=SHORT_TASK.get(names[t], names[t][:18]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{SHORT[arm]}  —  {pool} tokens\n"
                         f"task silhouette = {sil:+.3f}", fontsize=9.5)
    axes[0][0].legend(frameon=False, fontsize=6.6, markerscale=2.2,
                      loc="upper left", ncol=2, handletextpad=0.2,
                      columnspacing=0.6)
    fig.suptitle("Adapter output — the tensors the DiT cross-attends to.\n"
                 "The ten tasks share one scene, so task structure in the IMAGE "
                 "pool can only come from the instruction.",
                 fontsize=11.5, y=0.985)
    fig.savefig(FIG / "fig7_adapter_latents.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_adapter_latents")


# ──────────────────────────────────────────────────────────────────────────────
# fig 8 — spatial attention over the real frame
# ──────────────────────────────────────────────────────────────────────────────

def load_frames(global_rows):
    """The actual agentview frames behind given cache rows.

    Rebuilds LiberoDataset's index order (sorted files, demos sorted by numeric
    suffix) and reads only the frames asked for. h5py directly rather than
    LiberoDataset, because importing that package pulls data/__init__ -> pusht
    -> av for no reason here.
    """
    from configs.registry import get_config
    from data.libero.frames import resize_frame

    cfg = get_config("libero", "exp01")
    files = sorted(Path(cfg.dataset_root).glob("*_demo.hdf5"))
    index = []
    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as f:
            for dname in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
                T = f["data"][dname]["actions"].shape[0]
                index.extend([(fi, dname, t) for t in range(T)])
    out = []
    for gr in global_rows:
        fi, dname, t = index[gr]
        with h5py.File(files[fi], "r") as f:
            img = np.asarray(f["data"][dname]["obs"]["agentview_rgb"][t])[::-1]
        out.append(np.asarray(resize_frame(img, cfg.lt_img_w, cfg.lt_img_h)))
    return out


def spatial_attention(arm, rows):
    """Mean cross-attention per token, averaged over blocks/queries/denoise steps."""
    model, _, _ = load_arm(EXP[arm], RUN[arm])
    h5 = ROOT / RUN[arm] / "vlm_embeddings.h5"
    blocks = [b for b in model.decoder.blocks if getattr(b, "has_cross", False)]
    rec: list[torch.Tensor] = []
    originals = []

    def wrap(mod):
        orig = mod.forward

        def fwd(q, k, v, *a, **kw):
            kw.pop("need_weights", None)
            kw.pop("average_attn_weights", None)
            o, w = orig(q, k, v, *a, need_weights=True,
                        average_attn_weights=True, **kw)
            rec.append(w.detach().float().cpu())
            return o, w
        mod.forward = fwd
        return orig

    for b in blocks:
        originals.append(wrap(b.cross_attn))
    try:
        with torch.no_grad():
            emb, msk, _, st = read_batch(h5, rows)
            model.sample(emb.to(DEVICE), st.to(DEVICE), num_steps=3,
                         img_mask=msk.to(DEVICE))
    finally:
        for b, orig in zip(blocks, originals):
            b.cross_attn.forward = orig
    del model
    torch.cuda.empty_cache()

    # rec entries are (B, n_q, S); B is constant so stacking is safe.
    W = torch.stack(rec, 0).mean(dim=(0, 2)).numpy()      # (B, S)
    return W, msk.numpy()


def fig8_attention_maps(task_idx, names, phase):
    ENT = {a: float(np.mean([b["entropy_norm"] for b in
                             json.loads((HD / "head_diagnostics.json")
                                        .read_text(encoding="utf-8"))
                             [a]["cross_attention"]["per_block"]]))
           for a in ARMS}
    """One column per example frame, one row per arm, plus the raw frame."""
    # Early-phase frames from four different tasks: at low phase the scene is
    # near-identical across tasks, so any task-specific looking cannot be driven
    # by the visual difference.
    picks = []
    for t in [1, 5, 7, 9]:
        cand = np.where((task_idx == t) & (phase > 0.05) & (phase < 0.25))[0]
        picks.append(int(cand[len(cand) // 2]))
    picks = np.asarray(picks)

    frames = load_frames(picks)
    att = {a: spatial_attention(a, picks) for a in ARMS}

    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(3.05 * n, 9.3))
    for c in range(n):
        ax = axes[0][c]
        ax.imshow(frames[c])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(SHORT_TASK.get(names[task_idx[picks[c]]], "")
                     + f"\nphase {phase[picks[c]]:.2f}", fontsize=9)
        if c == 0:
            ax.set_ylabel("agentview frame", fontsize=9)

    # ONE colour scale across both arms. Normalising per arm would make the two
    # rows look equally peaked, which is exactly the comparison being made.
    #
    # Note on the entropy printed per row: head_diagnostics computes it over all
    # 272 tokens, so PaliGemma's lower value (0.911 vs 0.947) is driven largely
    # by the mass it concentrates on 16 text tokens. It is NOT evidence that its
    # image attention is more peaked, and is labelled accordingly.
    vmax = max(float(att[a][0][:, att[a][1][0]].max()) for a in ARMS)

    for r, arm in enumerate(ARMS):
        W, msk = att[arm]
        imgw = W[:, msk[0]]                              # (B, 256)
        g = int(round(np.sqrt(imgw.shape[1])))
        for c in range(n):
            ax = axes[r + 1][c]
            ax.imshow(frames[c])
            m = imgw[c].reshape(g, g)
            ax.imshow(m, cmap=ATT_CMAP, vmin=0, vmax=vmax,
                      extent=(0, frames[c].shape[1], frames[c].shape[0], 0),
                      interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            txt_share = 100.0 * W[c, ~msk[0]].sum() / W[c].sum()
            ax.set_xlabel(f"text tokens hold {txt_share:.1f}% "
                          f"(not shown)", fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel(SHORT[arm], fontsize=9)
        axes[r + 1][0].text(
            0.02, 0.03, f"entropy over all 272 tokens {ENT[arm]:.3f}", fontsize=7.5,
            color=INK2, transform=axes[r + 1][0].transAxes)
    fig.suptitle("Where each head actually looks — cross-attention over the 16×16 "
                 "patch grid, on the frame the policy saw.\n"
                 "ONE colour scale across both arms; the caption gives the mass "
                 "on instruction tokens, which the map does not contain.",
                 fontsize=11, y=0.965)
    fig.savefig(FIG / "fig8_attention_maps.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_attention_maps")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    task_idx, phase, names = frame_labels()
    with h5py.File(ROOT / RUN["pi05"] / "vlm_embeddings.h5", "r") as f:
        n_total = f["embeddings"].shape[0]
    rows = stride(np.sort(val_indices(n_total)), N_SCATTER)

    fig7_adapter_latents(rows, task_idx, names)
    fig8_attention_maps(task_idx, names, phase)
    print(f"\nwrote to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
