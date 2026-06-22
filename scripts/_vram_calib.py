"""VRAM calibration: run real train steps at a given batch size, report peak VRAM.

Builds the actual VLATrainModel + AdamW for language_table/exp01 and runs a few
forward+backward+step iterations on synthetic batches of the cache's real shapes,
then prints peak torch-reserved GB and nvidia-smi-equivalent used GB. Each call uses
ONE batch size in a fresh process so the measurement is clean.

Usage: python scripts/_vram_calib.py <batch_size>
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla_train import VLATrainModel


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    cfg = get_config("language_table", "exp01")
    dev = torch.device("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    model = VLATrainModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), weight_decay=cfg.weight_decay)

    S, Hh = cfg.img_seq_len, cfg.vlm_hidden_size
    H, Dd = cfg.action_horizon, cfg.action_dim
    img_count = cfg.img_grid_h * cfg.img_grid_w

    # synthetic batch with the real shapes (values irrelevant for memory)
    embed = torch.randn(B, S, Hh, device=dev)
    state = torch.randn(B, cfg.state_dim, device=dev)
    acts  = torch.randn(B, H, Dd, device=dev)
    mask  = torch.zeros(B, S, dtype=torch.bool, device=dev); mask[:, :img_count] = True

    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(embed, state, acts, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_resv  = torch.cuda.max_memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    # nvidia-smi "used" ~= reserved + CUDA context (~0.4-0.8 GB)
    print(f"B={B:5d}  peak_alloc={peak_alloc:5.2f} GB  peak_reserved={peak_resv:5.2f} GB  "
          f"(~{100*peak_resv/total:4.1f}% of {total:.1f} GB; +~0.6 GB ctx in nvidia-smi)")


if __name__ == "__main__":
    main()
