"""
scripts/compare_vlm_colorblind.py
─────────────────────────────────
Is our frozen Qwen "colour blind", and does SmolVLA's VLM do better?

KEY FACT (from lerobot/smolvla_base/config.json):
  SmolVLA freezes its VLM  ("train_expert_only": true, "freeze_vision_encoder": true)
  and loads it from "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" (first 16 of 32 layers).
  => SmolVLA's VLM carries ZERO robotics learning; robotics only trained the action
     expert.  So "is the robotics-pretrained VLM helpful?" reduces to: is SmolVLM2 a
     less colour-blind visual front-end than Qwen?  We test exactly SmolVLA's VLM
     (SmolVLM2-500M truncated to 16 layers), plus the full 32-layer SmolVLM2 as an
     upper bound, against our Qwen3.5-0.8B.

Two probes on the Language Table sim (BLOCK_8: 2 of each colour, 2 of each shape):
  • VISUAL (single-block CROPS): crop a tight box around each of the 8 known blocks,
    feed it to each VLM, take the mean image-token embedding -> "what colour/shape does
    the vision front-end actually encode?"  Silhouette by COLOUR and by SHAPE.
  • LANGUAGE (full scene + instruction): mean text-token embedding -> does the fused
    representation cluster by command type / referenced colour / referenced shape?

Output:
  docs/experiments/language_table/lt_vlm_colorblind.png
  docs/experiments/language_table/lt_vlm_colorblind.json
"""
from __future__ import annotations
import os, sys, re, gc, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the torch / transformers / torchvision stack BEFORE language_table pulls in
# TensorFlow — otherwise torch._dynamo.trace_rules trips over TF's null __spec__.
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """uint8 (H,W,3) RGB → PIL resized to (width,height). Inlined to avoid the
    data package's pandas/pyarrow import (flaky on this Windows box)."""
    return Image.fromarray(np.ascontiguousarray(rgb_uint8)).resize((width, height))


OUT = ROOT / "docs/experiments/language_table/lt_vlm_colorblind.png"
JOUT = ROOT / "docs/experiments/language_table/lt_vlm_colorblind.json"
SMOL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["moon", "cube", "star", "pentagon"]
CMDS = ["block2block", "block2absolute", "separate", "point"]
REQUIRE = {"separate": "separate", "point": "point"}
_OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")
COL_COL = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
SH_COL = dict(zip(SHAPES, ["#DD8452", "#4C72B0", "#8172B3", "#937860"]))
CMD_COL = {"block2block": "#4C72B0", "block2absolute": "#6FA8DC", "separate": "#DD8452", "point": "#55A868"}
ENC_COL = ["#C44E52", "#55A868", "#8172B3"]


def pca(X, k=2):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
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


# ── camera projection helpers (same as compare_qwen_dit_attention) ────────────────
def cam(env):
    vm, pm, *_ = env.calc_camera_params(env._image_size)
    return (np.array(vm, np.float64).reshape((4, 4), order="F"),
            np.array(pm, np.float64).reshape((4, 4), order="F"),
            env._image_size[1], env._image_size[0])


def project(view, proj, W, H, xy, z=0.02):
    P = np.array([xy[0], xy[1], z, 1.0]); c = proj @ (view @ P); n = c[:3] / c[3]
    return ((n[0] * 0.5 + 0.5) * W, (1.0 - (n[1] * 0.5 + 0.5)) * H)


def blkxy(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b])
    return np.array(p[:2])


# ── sample bank (built once, model-free) ──────────────────────────────────────────
def build_bank(cfg, n_crop_scenes=16, k_scene=24):
    from envs.language_table_env import make_lt_env, get_instruction
    crops, scenes = [], []

    print(f"[bank] crops from {n_crop_scenes} block2block scenes (8 blocks each) …")
    env = make_lt_env("block2block", seed=100)
    view, proj, W, H = cam(env)
    half = int(0.07 * max(W, H))
    for s in range(n_crop_scenes):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        rgb = np.asarray(obs["rgb"]); ih, iw = rgb.shape[:2]
        for b in env._block_to_pybullet_id.keys():
            color, shape = b.split("_")[0], b.split("_")[1]
            px, py = project(view, proj, W, H, blkxy(env, b))
            x0, x1 = int(np.clip(px - half, 0, iw)), int(np.clip(px + half, 0, iw))
            y0, y1 = int(np.clip(py - half, 0, ih)), int(np.clip(py + half, 0, ih))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = rgb[y0:y1, x0:x1]
            pil = resize_frame(crop, cfg.lt_img_w, cfg.lt_img_h)   # PIL
            crops.append({"pil": pil, "color": color, "shape": shape})
    env.close()
    print(f"[bank]   {len(crops)} block crops")

    print(f"[bank] scenes: {k_scene}/command × {len(CMDS)} commands …")
    for factory in CMDS:
        env = make_lt_env(factory, seed=13); req = REQUIRE.get(factory); n = 0; tries = 0
        while n < k_scene and tries < k_scene * 40:
            tries += 1
            out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, obs)
            if req and req not in instr.lower():
                continue
            m = _OBJ.search(instr.lower())
            if not m:
                continue
            pil = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
            scenes.append({"pil": pil, "instr": instr, "cmd": factory,
                           "color": m.group(1), "shape": m.group(2)})
            n += 1
        env.close()
        print(f"[bank]   {factory}: {n}")
    return crops, scenes


# ── encoders ───────────────────────────────────────────────────────────────────────
class QwenEnc:
    short = "Qwen3.5-0.8B\n(our frozen VLM)"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.m = VLAModel(cfg); self.m.vlm.to(device).eval()

    @torch.no_grad()
    def _enc(self, pil, text):
        inp = self.m.build_vlm_inputs([pil], [text], self.device)
        tok, im = self.m.encode_vlm(inp)
        return tok[0].float(), im[0]

    def visual(self, pil):
        t, im = self._enc(pil, " "); return t[im].mean(0).cpu().numpy()

    def text(self, pil, instr):
        t, im = self._enc(pil, instr); return t[~im].mean(0).cpu().numpy()

    def close(self):
        del self.m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


class SmolEnc:
    def __init__(self, device, short, n_layers=None):
        self.short = short; self.device = device
        self.proc = AutoProcessor.from_pretrained(SMOL)
        try:
            self.proc.image_processor.do_image_splitting = False
        except Exception:
            pass
        self.m = AutoModelForImageTextToText.from_pretrained(
            SMOL, dtype="bfloat16", low_cpu_mem_usage=True).to(device).eval()
        if n_layers:
            self.m.model.text_model.layers = self.m.model.text_model.layers[:n_layers]
        self.itid = self.m.config.image_token_id

    @torch.no_grad()
    def _enc(self, pil, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=False)
        inp = self.proc(text=prompt, images=[pil], return_tensors="pt").to(self.device)
        out = self.m(**inp, output_hidden_states=True, return_dict=True)
        ids = inp["input_ids"][0]; im = (ids == self.itid)
        return out.hidden_states[-1][0].float(), im

    def visual(self, pil):
        t, im = self._enc(pil, " "); return t[im].mean(0).cpu().numpy()

    def text(self, pil, instr):
        t, im = self._enc(pil, instr); return t[~im].mean(0).cpu().numpy()

    def close(self):
        del self.m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def raw_meanrgb(crops):
    """Positive control: the mean RGB colour of each crop (3-d). If THIS separates
    blocks by colour but the VLM embeddings don't, the VLM is discarding colour that
    is plainly present in the pixels."""
    return np.array([np.asarray(c["pil"], np.float32).reshape(-1, 3).mean(0) / 255.0 for c in crops])


def run_encoder(enc, crops, scenes):
    cv = np.array([enc.visual(c["pil"]) for c in crops])               # (Ncrop, D)
    tv = np.array([enc.text(s["pil"], s["instr"]) for s in scenes])    # (Nscene, D)
    return cv, tv


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    crops, scenes = build_bank(cfg)
    crop_color = [c["color"] for c in crops]; crop_shape = [c["shape"] for c in crops]
    sc_cmd = [s["cmd"] for s in scenes]; sc_color = [s["color"] for s in scenes]; sc_shape = [s["shape"] for s in scenes]

    encoders = [
        ("qwen", lambda: QwenEnc(cfg, device)),
        ("smolvla_vlm", lambda: SmolEnc(device, "SmolVLM2-500M ×16\n(= SmolVLA's frozen VLM)", n_layers=16)),
        ("smolvlm_full", lambda: SmolEnc(device, "SmolVLM2-500M ×32\n(full general VLM)", n_layers=None)),
    ]
    results = {}; shorts = {}
    for key, make in encoders:
        print(f"\n[encode] {key} …")
        enc = make(); shorts[key] = enc.short
        cv, tv = run_encoder(enc, crops, scenes)
        results[key] = {"crop_vis": cv, "scene_txt": tv}
        enc.close()
        print(f"[encode] {key}: crop_vis {cv.shape}  scene_txt {tv.shape}")

    # ── positive control: raw mean-RGB colour of each crop ───────────────────────
    raw_vis = raw_meanrgb(crops)

    # ── numeric summary ──────────────────────────────────────────────────────────
    summary = {}
    for key in results:
        cv = results[key]["crop_vis"]; tv = results[key]["scene_txt"]
        summary[key] = {
            "visual_color": silhouette(pca(cv, 10), crop_color),
            "visual_shape": silhouette(pca(cv, 10), crop_shape),
            "text_command": silhouette(pca(tv, 10), sc_cmd),
            "text_color":   silhouette(pca(tv, 10), sc_color),
            "text_shape":   silhouette(pca(tv, 10), sc_shape),
        }
    summary["raw"] = {
        "visual_color": silhouette(raw_vis, crop_color),
        "visual_shape": silhouette(raw_vis, crop_shape),
    }
    shorts["raw"] = "raw pixels\n(mean RGB)"
    print(f"\n[control] raw mean-RGB: VIS color={summary['raw']['visual_color']:.3f}  "
          f"VIS shape={summary['raw']['visual_shape']:.3f}")
    print("\n=== silhouette (higher = cleaner clusters) ===")
    hdr = f"{'encoder':14s} {'VIS color':>10s} {'VIS shape':>10s} {'TXT cmd':>9s} {'TXT color':>10s} {'TXT shape':>10s}"
    print(hdr)
    for key in results:
        s = summary[key]
        print(f"{key:14s} {s['visual_color']:10.3f} {s['visual_shape']:10.3f} "
              f"{s['text_command']:9.3f} {s['text_color']:10.3f} {s['text_shape']:10.3f}")

    # ── figure ──────────────────────────────────────────────────────────────────────
    keys = list(results.keys())
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))

    def scat(ax, X2, labels, palette, title):
        for lab in palette:
            mk = np.array(labels) == lab
            if mk.any():
                ax.scatter(X2[mk, 0], X2[mk, 1], s=30, color=palette[lab], label=lab,
                           edgecolor="black", lw=0.3, alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="best")

    # Row 0: crop-visual PCA by COLOUR
    for j, key in enumerate(keys):
        X2 = pca(results[key]["crop_vis"])
        sil = summary[key]["visual_color"]
        scat(axes[0, j], X2, crop_color, COL_COL,
             f"{shorts[key]}\nVISION of block crops · coloured by COLOUR  (sil={sil:.2f})")
    # Row 1: crop-visual PCA by SHAPE
    for j, key in enumerate(keys):
        X2 = pca(results[key]["crop_vis"])
        sil = summary[key]["visual_shape"]
        scat(axes[1, j], X2, crop_shape, SH_COL,
             f"{shorts[key].splitlines()[0]} · VISION · coloured by SHAPE  (sil={sil:.2f})")

    # Row 2 col 0/1: visual COLOUR / SHAPE silhouette bars — raw-pixel control first
    vkeys = ["raw"] + keys
    vlabels = [shorts["raw"].splitlines()[0] + "\n(control)"] + [k.replace("_", "\n") for k in keys]
    vcols = ["#444444"] + ENC_COL
    xv = np.arange(len(vkeys))
    vc = [summary[k]["visual_color"] for k in vkeys]
    vs = [summary[k]["visual_shape"] for k in vkeys]
    axes[2, 0].bar(xv, vc, color=vcols, edgecolor="black")
    for xi, v in zip(xv, vc):
        axes[2, 0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold", fontsize=9)
    axes[2, 0].set_xticks(xv); axes[2, 0].set_xticklabels(vlabels, fontsize=7.5)
    axes[2, 0].set_ylabel("silhouette by COLOUR"); axes[2, 0].axhline(0, color="black", lw=0.8)
    axes[2, 0].set_title("Can the VISION front-end tell block COLOURS apart?\n(raw mean-RGB DOES; every VLM ≈ 0 = colour blind)", fontsize=10, fontweight="bold")
    axes[2, 0].grid(axis="y", alpha=0.3)

    axes[2, 1].bar(xv, vs, color=vcols, edgecolor="black")
    for xi, v in zip(xv, vs):
        axes[2, 1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold", fontsize=9)
    axes[2, 1].set_xticks(xv); axes[2, 1].set_xticklabels(vlabels, fontsize=7.5)
    axes[2, 1].set_ylabel("silhouette by SHAPE"); axes[2, 1].axhline(0, color="black", lw=0.8)
    axes[2, 1].set_title("Can the VISION front-end tell block SHAPES apart?", fontsize=10, fontweight="bold")
    axes[2, 1].grid(axis="y", alpha=0.3)

    # Row 2 col 2: text-emb silhouette (command / colour) grouped per encoder
    x = np.arange(len(keys)); w = 0.38
    tc = [summary[k]["text_command"] for k in keys]
    tcol = [summary[k]["text_color"] for k in keys]
    axes[2, 2].bar(x - w / 2, tc, w, color="#4C72B0", edgecolor="black", label="by command type")
    axes[2, 2].bar(x + w / 2, tcol, w, color="#C44E52", edgecolor="black", label="by referenced colour")
    for xi, v in zip(x - w / 2, tc):
        axes[2, 2].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, tcol):
        axes[2, 2].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axes[2, 2].set_xticks(x); axes[2, 2].set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    axes[2, 2].set_ylabel("silhouette (instruction text emb)"); axes[2, 2].axhline(0, color="black", lw=0.8)
    axes[2, 2].set_title("LANGUAGE pathway: command vs colour\n(all read command; colour is weak = the bottleneck)", fontsize=10, fontweight="bold")
    axes[2, 2].legend(fontsize=8); axes[2, 2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Is Qwen colour-blind, and does SmolVLA's robotics VLM help?  —  SmolVLA FREEZES its VLM "
        "(config: train_expert_only=true, freeze_vision_encoder=true), loading HuggingFaceTB/SmolVLM2-500M (first 16 layers).\n"
        "So SmolVLA's VLM carries ZERO robotics learning — the question reduces to: is SmolVLM2 a less colour-blind visual front-end than Qwen?  "
        "Probe: feed single-block crops + full scenes to each frozen VLM and measure how cleanly its embeddings separate by colour / shape / command.",
        fontsize=11.5, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    JOUT.write_text(json.dumps({
        "note": "SmolVLA freezes its VLM (train_expert_only/freeze_vision_encoder); VLM=SmolVLM2-500M first 16 layers, no robotics learning.",
        "n_crops": len(crops), "n_scenes": len(scenes),
        "shorts": {k: shorts[k].replace(chr(10), " ") for k in shorts},
        "silhouette": summary,
    }, indent=2))
    print(f"\nsaved → {OUT}")
    print(f"saved → {JOUT}")


if __name__ == "__main__":
    main()
