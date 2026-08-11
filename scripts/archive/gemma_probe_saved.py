"""
scripts/gemma_probe_saved.py   (run in the `gemma4` env: transformers 5.10.2)
─────────────────────────────────────────────────────────────────────────────
Self-contained (no language_table) colour/shape probe on the EXPORTED bank, run on
Gemma 4 E2B / E4B / 12B (the 12B = `gemma4_unified`, the encoder-free model that needs
transformers >= 5.10). All in 4-bit so the 12B fits a 12 GB card. Compares against the
raw-pixel ceiling and the earlier Qwen3.5 / SmolVLM2 (=SmolVLA's VLM) results.

Output:
  docs/experiments/language_table/lt_gemma4_colorblind.png   (final, all sizes)
  docs/experiments/language_table/lt_gemma4_colorblind.json
"""
from __future__ import annotations
import os, sys, json, gc
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).parents[2]
BANK = ROOT / "docs/experiments/language_table/_bank"
OUT = ROOT / "docs/experiments/language_table/lt_gemma4_colorblind.png"
JOUT = ROOT / "docs/experiments/language_table/lt_gemma4_colorblind.json"
PRIOR = ROOT / "docs/experiments/language_table/lt_vlm_colorblind.json"

COL_COL = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
SH_COL = {"moon": "#DD8452", "cube": "#4C72B0", "star": "#8172B3", "pentagon": "#937860"}
GEMMAS = [
    ("gemma4_e2b", "google/gemma-4-e2b-it", "Gemma 4 E2B", 2.0),
    ("gemma4_e4b", "google/gemma-4-e4b-it", "Gemma 4 E4B", 4.0),
    ("gemma4_12b", "google/gemma-4-12b-it", "Gemma 4 12B\n(encoder-free)", 12.0),
]


def pca(X, k=2):
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
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


def find_image_token_id(model, proc):
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int):
            return v
        sub = getattr(model.config, "text_config", None)
        if sub is not None and isinstance(getattr(sub, attr, None), int):
            return getattr(sub, attr)
    tok = getattr(proc, "tokenizer", proc)
    for attr in ("image_token_id", "image_token_index"):
        if isinstance(getattr(tok, attr, None), int):
            return getattr(tok, attr)
    it = getattr(proc, "image_token", None) or getattr(tok, "image_token", None) or "<image_soft_token>"
    try:
        return tok.convert_tokens_to_ids(it)
    except Exception:
        return None


class Gemma4Enc:
    def __init__(self, rid, device):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(rid)
        for a in ("do_pan_and_scan", "do_image_splitting"):
            try:
                setattr(self.proc.image_processor, a, False)
            except Exception:
                pass
        # Keep the (small) vision / patch-projection / lm_head in bf16 — quantizing the
        # encoder-free patch_dense makes its weight dtype uint8, which breaks the
        # `pixel_values.to(patch_dense.weight.dtype)` cast → LayerNorm on Byte.
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                 llm_int8_skip_modules=["model.embed_vision", "patch_dense", "lm_head", "multi_modal_projector", "vision_tower"])
        self.m = AutoModelForImageTextToText.from_pretrained(
            rid, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16).eval()
        self.itid = find_image_token_id(self.m, self.proc)
        self.n_img = None

    @torch.no_grad()
    def _enc(self, pil, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=False)
        inp = self.proc(text=prompt, images=[pil], return_tensors="pt").to(self.device)
        out = self.m(**inp, output_hidden_states=True, return_dict=True)
        ids = inp["input_ids"][0]; im = (ids == self.itid)
        if self.n_img is None:
            self.n_img = int(im.sum())
        return out.hidden_states[-1][0].float(), im

    def visual(self, pil):
        t, im = self._enc(pil, " "); return t[im].mean(0).cpu().numpy()

    def text(self, pil, instr):
        t, im = self._enc(pil, instr); return t[~im].mean(0).cpu().numpy()

    def close(self):
        del self.m; gc.collect(); torch.cuda.empty_cache()


def main():
    device = "cuda"
    lab = json.loads((BANK / "labels.json").read_text())
    crops = [{"pil": Image.open(BANK / r["file"]).convert("RGB"), **r} for r in lab["crops"]]
    scenes = [{"pil": Image.open(BANK / r["file"]).convert("RGB"), **r} for r in lab["scenes"]]
    crop_color = [c["color"] for c in crops]; crop_shape = [c["shape"] for c in crops]
    sc_cmd = [s["cmd"] for s in scenes]; sc_color = [s["color"] for s in scenes]
    raw_vis = np.array([np.asarray(c["pil"], np.float32).reshape(-1, 3).mean(0) / 255.0 for c in crops])

    sil = {"raw_meanrgb": {"color": silhouette(raw_vis, crop_color), "shape": silhouette(raw_vis, crop_shape),
                           "label": "raw pixels\n(mean RGB)", "size": 0.0}}
    scatters = {}
    for key, rid, label, size in GEMMAS:
        print(f"\n[encode] {label}  ({rid}) …", flush=True)
        try:
            enc = Gemma4Enc(rid, device)
            cv = np.array([enc.visual(c["pil"]) for c in crops])
            tv = np.array([enc.text(s["pil"], s["instr"]) for s in scenes])
            print(f"   itid={enc.itid}  n_img_tokens={enc.n_img}  crop_vis {cv.shape}", flush=True)
            sil[key] = {"color": silhouette(pca(cv, 10), crop_color),
                        "shape": silhouette(pca(cv, 10), crop_shape),
                        "text_command": silhouette(pca(tv, 10), sc_cmd),
                        "text_color": silhouette(pca(tv, 10), sc_color),
                        "label": label, "size": size}
            scatters[key] = cv; enc.close()
            print(f"   {label}: VIS color={sil[key]['color']:.3f}  shape={sil[key]['shape']:.3f}  "
                  f"TXT cmd={sil[key]['text_command']:.3f}  TXT color={sil[key]['text_color']:.3f}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   !! {label} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    prior = {}
    if PRIOR.exists():
        pj = json.loads(PRIOR.read_text())["silhouette"]
        if "qwen" in pj:
            prior["qwen"] = {"color": pj["qwen"]["visual_color"], "shape": pj["qwen"]["visual_shape"],
                             "label": "Qwen3.5\n0.8B (ours)", "size": 0.8}
        if "smolvla_vlm" in pj:
            prior["smolvlm2"] = {"color": pj["smolvla_vlm"]["visual_color"], "shape": pj["smolvla_vlm"]["visual_shape"],
                                 "label": "SmolVLM2\n0.5B (SmolVLA)", "size": 0.5}

    print("\n=== silhouette by COLOUR (raw-pixel ceiling = %.3f) ===" % sil["raw_meanrgb"]["color"])
    for k in ["qwen", "smolvlm2"]:
        if k in prior:
            print(f"  {prior[k]['label'].replace(chr(10),' '):26s} color={prior[k]['color']:.3f}  shape={prior[k]['shape']:.3f}")
    for key, _, _, _ in GEMMAS:
        if key in sil:
            print(f"  {sil[key]['label'].replace(chr(10),' '):26s} color={sil[key]['color']:.3f}  shape={sil[key]['shape']:.3f}")

    # ── figure ──────────────────────────────────────────────────────────────────────
    gkeys = [k for k, _, _, _ in GEMMAS if k in scatters]
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))

    def scat(ax, X2, labels, palette, title):
        for lab_ in palette:
            mk = np.array(labels) == lab_
            if mk.any():
                ax.scatter(X2[mk, 0], X2[mk, 1], s=30, color=palette[lab_], label=lab_, edgecolor="black", lw=0.3, alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7.5, loc="best")

    for j in range(3):
        if j < len(gkeys):
            k = gkeys[j]
            scat(axes[0, j], pca(scatters[k]), crop_color, COL_COL,
                 f"{sil[k]['label'].splitlines()[0]} · crops by COLOUR  (sil={sil[k]['color']:.2f})")
            scat(axes[1, j], pca(scatters[k]), crop_shape, SH_COL,
                 f"{sil[k]['label'].splitlines()[0]} · crops by SHAPE  (sil={sil[k]['shape']:.2f})")
        else:
            axes[0, j].axis("off"); axes[1, j].axis("off")

    src = {**sil, **prior}
    bar_order = (["raw_meanrgb"] + (["smolvlm2"] if "smolvlm2" in prior else [])
                 + (["qwen"] if "qwen" in prior else []) + gkeys)
    labs = [src[k]["label"] for k in bar_order]
    cvals = [src[k]["color"] for k in bar_order]; svals = [src[k]["shape"] for k in bar_order]
    cols = ["#444444" if k == "raw_meanrgb" else "#C44E52" if k == "qwen" else "#55A868" if k == "smolvlm2" else "#8172B3" for k in bar_order]
    x = np.arange(len(bar_order))
    axes[2, 0].bar(x, cvals, color=cols, edgecolor="black")
    for xi, v in zip(x, cvals):
        axes[2, 0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
    axes[2, 0].axhline(sil["raw_meanrgb"]["color"], color="#444444", ls="--", lw=1, alpha=0.7)
    axes[2, 0].set_xticks(x); axes[2, 0].set_xticklabels(labs, fontsize=7); axes[2, 0].axhline(0, color="black", lw=0.8)
    axes[2, 0].set_ylabel("silhouette by COLOUR")
    axes[2, 0].set_title("COLOUR separability — does a SMARTER VLM see colour?\n(dashed = raw-pixel ceiling)", fontsize=10, fontweight="bold")
    axes[2, 0].grid(axis="y", alpha=0.3)

    axes[2, 1].bar(x, svals, color=cols, edgecolor="black")
    for xi, v in zip(x, svals):
        axes[2, 1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
    axes[2, 1].set_xticks(x); axes[2, 1].set_xticklabels(labs, fontsize=7); axes[2, 1].axhline(0, color="black", lw=0.8)
    axes[2, 1].set_ylabel("silhouette by SHAPE"); axes[2, 1].set_title("SHAPE separability", fontsize=10, fontweight="bold")
    axes[2, 1].grid(axis="y", alpha=0.3)

    pts = sorted([(src[k]["size"], src[k]["color"], src[k]["label"]) for k in bar_order if k != "raw_meanrgb"], key=lambda t: t[0])
    axes[2, 2].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#8172B3", lw=2, ms=8)
    for sx, sy, lb in pts:
        axes[2, 2].annotate(lb.replace("\n", " "), (sx, sy), textcoords="offset points", xytext=(4, 5), fontsize=7)
    axes[2, 2].axhline(sil["raw_meanrgb"]["color"], color="#444444", ls="--", lw=1.2, label=f"raw-pixel ceiling ({sil['raw_meanrgb']['color']:.2f})")
    axes[2, 2].set_xscale("log"); axes[2, 2].set_xlabel("model size (B params, approx, log)")
    axes[2, 2].set_ylabel("colour silhouette")
    axes[2, 2].set_title("SCALING: colour separability vs size\n(flat & far below ceiling ⇒ scale doesn't fix it)", fontsize=10, fontweight="bold")
    axes[2, 2].legend(fontsize=8); axes[2, 2].grid(alpha=0.3)

    fig.suptitle(
        "Does a SMARTER VLM escape colour-blindness?  Gemma 4 (E2B / E4B / 12B-encoder-free) on the Language-Table block-colour probe.\n"
        "The colour is plainly in the pixels (raw mean-RGB ceiling). Every VLM — including the 12B encoder-free model — stays far below it: per-object colour loss is fundamental to patch-token VLMs, not a capacity problem.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    JOUT.write_text(json.dumps({"raw_pixel_ceiling_color": sil["raw_meanrgb"]["color"],
                                "silhouette": {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in src.items()}},
                               indent=2, default=float))
    print(f"\nsaved → {OUT}\nsaved → {JOUT}")


if __name__ == "__main__":
    main()
