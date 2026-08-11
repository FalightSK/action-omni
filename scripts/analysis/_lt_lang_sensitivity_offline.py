"""
scripts/_lt_lang_sensitivity_offline.py
────────────────────────────────────────
Sim-free language-adaptability probe for Language Table exp01.

Question: does the per-episode INSTRUCTION causally change the model's predicted
action? Uses REAL frames + REAL instructions from the converted dataset (no PyBullet
sim needed). For a fixed scene+state and a fixed flow-matching noise seed:

  noise floor = || a(true instr, seed A) - a(true instr, seed B) ||   (sampling jitter)
  swap        = || a(true instr, seed A) - a(other real instr, seed A) ||
  null        = || a(true instr, seed A) - a("", seed A) ||

If swap/floor >> 1 and null/floor >> 1, language drives the action (good language
adaptability). If ~1, the model ignores the instruction.

Output: prints the ratios + saves a bar chart to
  docs/experiments/language_table/lt_lang_sensitivity_offline.png
"""
from __future__ import annotations
import os, sys, random
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

OUT = ROOT / "docs/experiments/language_table/lt_lang_sensitivity_offline.png"


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()

    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt",
                    map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ck["state_dict"]); tm.eval()
    print(f"   checkpoint epoch={ck.get('epoch','?')} val_loss={ck.get('val_loss', float('nan')):.4f}")

    print("[2/3] Loading dataset (real frames + instructions) …")
    ds = LanguageTableDataset(cfg)
    uniq = sorted(set(ds.instructions))
    print(f"   {len(ds):,} frames, {len(uniq):,} unique instructions")

    @torch.no_grad()
    def action_chunk(image, instruction, state_norm, seed):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        st = state_norm.unsqueeze(0).to(device)
        seed_all(seed)
        flat = tm.sample(tok, st, num_steps=cfg.num_flow_steps, img_mask=im)
        return flat.view(cfg.action_horizon, cfg.action_dim).float().cpu().numpy()

    rng = random.Random(0)
    g = torch.Generator().manual_seed(7)
    idxs = torch.randperm(len(ds), generator=g)[:N].tolist()

    noise_d, swap_d, null_d, rms_a = [], [], [], []
    print(f"[3/3] Probing {N} real frames …")
    for k, i in enumerate(idxs):
        item = ds[i]
        image = item["image"]; state = item["state"]; true_instr = item["task_text"]
        other = rng.choice(uniq)
        while other == true_instr and len(uniq) > 1:
            other = rng.choice(uniq)
        a_true  = action_chunk(image, true_instr, state, seed=1234)
        a_true2 = action_chunk(image, true_instr, state, seed=9999)   # noise floor
        a_swap  = action_chunk(image, other,      state, seed=1234)
        a_null  = action_chunk(image, "",         state, seed=1234)
        noise_d.append(np.sqrt(np.mean((a_true - a_true2) ** 2)))
        swap_d.append(np.sqrt(np.mean((a_true - a_swap) ** 2)))
        null_d.append(np.sqrt(np.mean((a_true - a_null) ** 2)))
        rms_a.append(np.sqrt(np.mean(a_true ** 2)))

    noise = float(np.mean(noise_d)); swap = float(np.mean(swap_d)); null = float(np.mean(null_d))
    rmsa = float(np.mean(rms_a))
    print(f"\n   RMS action magnitude       : {rmsa:.3f}")
    print(f"   noise floor (same instr)   : {noise:.4f}")
    print(f"   swapped instruction        : {swap:.4f}  ({swap/noise:.1f}x floor)")
    print(f"   null instruction           : {null:.4f}  ({null/noise:.1f}x floor)")
    verdict = ("STRONG language effect" if swap/noise >= 3 else
               "MODERATE language effect" if swap/noise >= 1.5 else
               "WEAK language effect")
    print(f"   => {verdict} (swap {swap/noise:.1f}x the sampling-noise floor)")

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    fig.subplots_adjust(top=0.84)
    ax.bar([0, 1, 2], [noise, swap, null],
           color=["#BBBBBB", "#4C72B0", "#DD8452"], edgecolor="black")
    ax.set_ylim(0, max(noise, swap, null) * 1.22)
    for i, v, lab in [(0, noise, "noise floor"), (1, swap, f"{swap/noise:.1f}x floor"),
                      (2, null, f"{null/noise:.1f}x floor")]:
        ax.annotate(f"{v:.3f}\n{lab}", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["same instr\n(noise floor)", "swapped\ninstruction", "null\ninstruction"])
    ax.set_ylabel("RMS action-chunk change (normalised units)")
    ax.set_title(f"Sim-free language sensitivity (LT exp01, n={N})\n"
                 f"swapping the instruction moves the action {swap/noise:.1f}x the noise floor",
                 fontsize=11, fontweight="bold", y=1.0)
    ax.grid(axis="y", alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    print(f"\n   saved -> {OUT}")


if __name__ == "__main__":
    main()
