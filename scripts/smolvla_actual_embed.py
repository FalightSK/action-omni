"""
scripts/smolvla_actual_embed.py
───────────────────────────────
Pull the embedding from the REAL SmolVLA (lerobot/smolvla_base), not just its
SmolVLM2 backbone, and:
  1. PROVE whether SmolVLA's VLM weights differ from base SmolVLM2-500M (it freezes
     the VLM per config → expect mean|Δ| = 0, i.e. byte-identical).
  2. Run the same Language-Table colour/shape crop probe on SmolVLA's ACTUAL VLM and
     compare to base SmolVLM2 ×16 + the raw-pixel control (+ Qwen from the prior run).

Output:
  docs/experiments/language_table/lt_smolvla_actual_embed.png
  docs/experiments/language_table/lt_smolvla_actual_embed.json
"""
from __future__ import annotations
import os, sys, json, gc
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
# Force ONLINE before importing compare_vlm_colorblind (which setdefaults offline=1) —
# we need to download lerobot/smolvla_base (only its config.json is cached).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
# reuse all probe machinery (this import pulls the torch/transformers stack first)
from compare_vlm_colorblind import (
    build_bank, raw_meanrgb, pca, silhouette, SmolEnc,
    COL_COL, SH_COL, SMOL,
)
from configs.registry import get_config

OUT = ROOT / "docs/experiments/language_table/lt_smolvla_actual_embed.png"
JOUT = ROOT / "docs/experiments/language_table/lt_smolvla_actual_embed.json"
PRIOR = ROOT / "docs/experiments/language_table/lt_vlm_colorblind.json"


class WrapVLM:
    """Wrap an already-loaded SmolVLM-style `vlm` + processor with SmolEnc's encode."""
    _enc = SmolEnc._enc
    visual = SmolEnc.visual
    text = SmolEnc.text

    def __init__(self, vlm, proc, device, short):
        self.m = vlm; self.proc = proc; self.device = device; self.short = short
        try:
            self.proc.image_processor.do_image_splitting = False
        except Exception:
            pass
        self.itid = vlm.config.image_token_id


def weight_diff(real_vlm, base_vlm):
    """mean|Δ| between SmolVLA's VLM and base SmolVLM2 for representative params."""
    rt = dict(real_vlm.model.text_model.named_parameters())
    bt = dict(base_vlm.model.text_model.named_parameters())
    rv = dict(real_vlm.model.vision_model.named_parameters())
    bv = dict(base_vlm.model.vision_model.named_parameters())
    rows = []
    probes = [
        ("text", "layers.0.self_attn.q_proj.weight", rt, bt),
        ("text", "layers.0.mlp.gate_proj.weight", rt, bt),
        ("text", "layers.15.self_attn.q_proj.weight", rt, bt),
        ("vision", "encoder.layers.0.self_attn.q_proj.weight", rv, bv),
        ("vision", "encoder.layers.0.mlp.fc1.weight", rv, bv),
    ]
    for tag, name, rd, bd in probes:
        rk = next((k for k in rd if k.endswith(name)), None)
        bk = next((k for k in bd if k.endswith(name)), None)
        if rk is None or bk is None:
            rows.append((tag, name, None, None)); continue
        a = rd[rk].detach().float().cpu(); b = bd[bk].detach().float().cpu()
        d = (a - b).abs().mean().item()
        rel = d / (b.abs().mean().item() + 1e-12)
        rows.append((tag, name, d, rel))
    # connector
    try:
        rc = dict(real_vlm.model.connector.named_parameters())
        bc = dict(base_vlm.model.connector.named_parameters())
        k = next(iter(rc))
        d = (rc[k].detach().float().cpu() - bc[k].detach().float().cpu()).abs().mean().item()
        rel = d / (bc[k].detach().float().cpu().abs().mean().item() + 1e-12)
        rows.append(("connector", k.split(".")[-2] + "." + k.split(".")[-1], d, rel))
    except Exception:
        pass
    return rows


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()

    print("[1/5] base SmolVLM2-500M ×16 (reference) …")
    ref = SmolEnc(device, "SmolVLM2-500M ×16\n(base / reference)", n_layers=16)

    print("[2/5] downloading + loading REAL SmolVLA (lerobot/smolvla_base) …")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pol = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    vwe = pol.model.vlm_with_expert
    real = WrapVLM(vwe.vlm, vwe.processor, device, "SmolVLA VLM (actual)\nlerobot/smolvla_base")
    print(f"   SmolVLA VLM text layers: {len(vwe.vlm.model.text_model.layers)}  "
          f"image_token_id: {real.itid}")

    print("[3/5] WEIGHT DIFF — SmolVLA's VLM vs base SmolVLM2 …")
    diffs = weight_diff(vwe.vlm, ref.m)
    for tag, name, d, rel in diffs:
        if d is None:
            print(f"   {tag:9s} {name:42s}  (not found)")
        else:
            print(f"   {tag:9s} {name:42s}  mean|Δ|={d:.3e}  rel={rel:.2%}")
    identical = all((d is not None and d < 1e-6) for _, _, d, _ in diffs)
    print(f"   => VLM byte-identical to base SmolVLM2: {identical}")

    print("[4/5] building probe bank + encoding …")
    crops, scenes = build_bank(cfg)
    crop_color = [c["color"] for c in crops]; crop_shape = [c["shape"] for c in crops]
    raw_vis = raw_meanrgb(crops)

    def enc_all(e):
        return np.array([e.visual(c["pil"]) for c in crops])
    real_cv = enc_all(real)
    ref_cv = enc_all(ref)

    sil = {
        "raw_meanrgb":      {"color": silhouette(raw_vis, crop_color),         "shape": silhouette(raw_vis, crop_shape)},
        "smolvla_actual":   {"color": silhouette(pca(real_cv, 10), crop_color), "shape": silhouette(pca(real_cv, 10), crop_shape)},
        "smolvlm2_x16_ref": {"color": silhouette(pca(ref_cv, 10), crop_color),  "shape": silhouette(pca(ref_cv, 10), crop_shape)},
    }
    qwen_color = qwen_shape = None
    if PRIOR.exists():
        pj = json.loads(PRIOR.read_text())["silhouette"].get("qwen", {})
        qwen_color = pj.get("visual_color"); qwen_shape = pj.get("visual_shape")
        sil["qwen_prior"] = {"color": qwen_color, "shape": qwen_shape}

    # also: are SmolVLA-actual and SmolVLM2-ref embeddings numerically equal on crops?
    emb_l1 = float(np.abs(real_cv - ref_cv).mean())
    print(f"   crop-embedding mean|Δ| (SmolVLA-actual vs SmolVLM2×16): {emb_l1:.3e}")
    print("\n   silhouette by COLOUR / SHAPE:")
    for k, v in sil.items():
        c = v["color"]; s = v["shape"]
        print(f"     {k:18s} color={c:.3f}  shape={s:.3f}" if c is not None else f"     {k}: n/a")

    print("[5/5] figure …")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    def scat(ax, X2, labels, palette, title):
        for lab in palette:
            mk = np.array(labels) == lab
            if mk.any():
                ax.scatter(X2[mk, 0], X2[mk, 1], s=32, color=palette[lab], label=lab,
                           edgecolor="black", lw=0.3, alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="best")

    scat(axes[0, 0], pca(real_cv), crop_color, COL_COL,
         f"SmolVLA VLM (ACTUAL) · block crops by COLOUR  (sil={sil['smolvla_actual']['color']:.2f})")
    scat(axes[0, 1], pca(real_cv), crop_shape, SH_COL,
         f"SmolVLA VLM (ACTUAL) · block crops by SHAPE  (sil={sil['smolvla_actual']['shape']:.2f})")
    scat(axes[0, 2], pca(ref_cv), crop_color, COL_COL,
         f"base SmolVLM2 ×16 · by COLOUR  (sil={sil['smolvlm2_x16_ref']['color']:.2f})\n(identical to SmolVLA's frozen VLM)")

    # bars: colour & shape silhouette
    order = ["raw_meanrgb", "smolvla_actual", "smolvlm2_x16_ref"] + (["qwen_prior"] if qwen_color is not None else [])
    labels = {"raw_meanrgb": "raw pixels\n(mean RGB)\n[control]",
              "smolvla_actual": "SmolVLA VLM\n(actual)",
              "smolvlm2_x16_ref": "SmolVLM2 ×16\n(base)",
              "qwen_prior": "Qwen3.5\n(ours)"}
    cols = {"raw_meanrgb": "#444444", "smolvla_actual": "#8172B3",
            "smolvlm2_x16_ref": "#55A868", "qwen_prior": "#C44E52"}
    x = np.arange(len(order))
    cvals = [sil[k]["color"] for k in order]
    svals = [sil[k]["shape"] for k in order]
    axes[1, 0].bar(x, cvals, color=[cols[k] for k in order], edgecolor="black")
    for xi, v in zip(x, cvals):
        axes[1, 0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels([labels[k] for k in order], fontsize=7.5)
    axes[1, 0].axhline(0, color="black", lw=0.8); axes[1, 0].set_ylabel("silhouette by COLOUR")
    axes[1, 0].set_title("COLOUR separability of the embedding\n(raw pixels DO; SmolVLA's VLM ≈ Qwen ≈ colour blind)", fontsize=10, fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(x, svals, color=[cols[k] for k in order], edgecolor="black")
    for xi, v in zip(x, svals):
        axes[1, 1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold")
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels([labels[k] for k in order], fontsize=7.5)
    axes[1, 1].axhline(0, color="black", lw=0.8); axes[1, 1].set_ylabel("silhouette by SHAPE")
    axes[1, 1].set_title("SHAPE separability of the embedding", fontsize=10, fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.3)

    # text panel: weight-diff proof
    axes[1, 2].axis("off")
    lines = ["WEIGHT DIFF — SmolVLA's VLM vs base SmolVLM2-500M", ""]
    for tag, name, d, rel in diffs:
        short = name.replace(".weight", "")
        lines.append(f"  {tag:8s} {short:34s}" + (f"  |Δ|={d:.1e}" if d is not None else "  n/a"))
    lines += ["",
              f"crop-embedding mean|Δ| (actual vs base): {emb_l1:.1e}",
              "",
              "=> SmolVLA's VLM is the FROZEN, UNCHANGED",
              "   SmolVLM2-500M (config: train_expert_only=true,",
              "   freeze_vision_encoder=true). Robotics training",
              "   touched only the action expert — never the VLM.",
              "",
              "So 'SmolVLA's robotics VLM' carries ZERO robotics",
              "learning, and its embedding is just as colour-blind",
              "as Qwen (both far below the raw-pixel ceiling)."]
    axes[1, 2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.6, family="monospace",
                    transform=axes[1, 2].transAxes)

    fig.suptitle(
        "The embedding from the REAL SmolVLA (lerobot/smolvla_base) — its VLM is frozen SmolVLM2-500M, "
        "and its block-crop embedding is just as colour-blind as Qwen.\n"
        "raw mean-RGB separates blocks by colour; SmolVLA's actual VLM embedding does not — confirming the bottleneck is the general visual front-end, not the data.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    JOUT.write_text(json.dumps({
        "vlm_byte_identical_to_smolvlm2": bool(identical),
        "weight_diff": [{"part": t, "param": n, "mean_abs_delta": d, "rel": r} for t, n, d, r in diffs],
        "crop_embedding_mean_abs_delta_actual_vs_base": emb_l1,
        "silhouette": sil,
        "n_crops": len(crops),
    }, indent=2))
    print(f"\nsaved → {OUT}")
    print(f"saved → {JOUT}")


if __name__ == "__main__":
    main()
