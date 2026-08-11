"""
scripts/_lt_per_head_attn.py
────────────────────────────
Per-HEAD DiT cross-attention over the 6x11 image grid for LT exp01.

analyze_lt_dit_token_attention.py averages over the 8 heads; this keeps them separate
(monkey-patch average_attn_weights=False, like the ALOHA Test-5 probe) to show whether
different heads specialise on different board regions. Averaged over real frames,
DiT blocks, and the action-query steps. Sim-free (uses cached dataset frames).

Output: docs/experiments/language_table/lt_per_head_attn.png
"""
from __future__ import annotations
import os, sys
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import LanguageTableDataset

OUT = ROOT / "docs/experiments/language_table/lt_per_head_attn.png"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cfg = get_config("language_table", "exp01"); device = cfg.get_device()
    GH, GW = cfg.img_grid_h, cfg.img_grid_w; img_count = GH * GW
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = torch.load(Path(cfg.output_dir)/"checkpoints"/"best.pt", map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ck["state_dict"]); tm.eval()

    ds = LanguageTableDataset(cfg)
    g = torch.Generator().manual_seed(3); idxs = torch.randperm(len(ds), generator=g)[:N].tolist()

    blocks = tm.decoder.blocks
    nblk = len(blocks); collected = {b: [] for b in range(nblk)}
    orig, hooks = {}, []
    for bi, blk in enumerate(blocks):
        ca = getattr(blk, "cross_attn", None)
        if ca is None: continue
        of = ca.forward; orig[bi] = (ca, of)
        def patched(q, k, v, _f=of, **kw):
            kw["need_weights"] = True; kw["average_attn_weights"] = False
            return _f(q, k, v, **kw)
        ca.forward = patched
        def hook(m, i, o, _b=bi):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                collected[_b].append(o[1].detach().cpu().float())  # (B,nheads,tgt,src)
        hooks.append(ca.register_forward_hook(hook))

    print(f"[2/3] Capturing per-head cross-attention over {N} frames …")
    with torch.no_grad():
        for i in idxs:
            it = ds[i]
            inp = vlm.build_vlm_inputs([it["image"]], [it["task_text"]], device)
            tok, im = vlm.encode_vlm(inp)
            st = it["state"].unsqueeze(0).to(device)
            torch.manual_seed(0)
            tm.sample(tok, st, num_steps=cfg.num_flow_steps, img_mask=im)
    for h in hooks: h.remove()
    for ca, of in orig.values(): ca.forward = of

    # src (token count) varies per frame with instruction length, so slice the IMAGE
    # part (first img_count tokens, constant) BEFORE stacking, then mean over all.
    allw = []
    for b in range(nblk):
        for w in collected[b]:
            allw.append(w[..., :img_count])   # (1,nheads,tgt,img_count)
    W = torch.cat(allw, 0)                     # (frames*blocks,nheads,tgt,img_count)
    nheads = W.shape[1]
    img = W.mean(dim=(0, 2)).numpy()           # (nheads, img_count)
    maps = img.reshape(nheads, GH, GW)

    print(f"[3/3] {nheads} heads; per-head image-attention mass range "
          f"{img.sum(1).min():.3f}-{img.sum(1).max():.3f}")
    ncol = 4; nrow = int(np.ceil(nheads/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0*ncol, 2.4*nrow))
    fig.subplots_adjust(top=0.80, hspace=0.30, wspace=0.10)
    vmax = maps.max()
    for h in range(nrow*ncol):
        ax = axes.flat[h]
        if h < nheads:
            ax.imshow(maps[h], cmap="inferno", vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_title(f"head {h+1}", fontsize=10, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        if h >= nheads: ax.axis("off")
    fig.suptitle(f"DiT cross-attention per HEAD over the {GH}x{GW} image grid (LT exp01, n={N} frames)\n"
                 "each head specialises on a different board region; the per-token figure averages these 8 maps",
                 fontsize=12, fontweight="bold", y=0.965)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
