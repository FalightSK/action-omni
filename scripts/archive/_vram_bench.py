"""Batch-size VRAM + throughput sweep for VLATrainModel (compute only, synthetic data).
Finds the largest batch that fits a VRAM budget and its forward+backward samples/sec,
so we can size --batch-size to actually use the GPU instead of leaving it idle.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla_train import VLATrainModel


def main():
    cfg = get_config("language_table", "exp01")
    dev = torch.device("cuda")
    S, H = cfg.img_seq_len, cfg.vlm_hidden_size
    Hh, Dd = cfg.action_horizon, cfg.action_dim

    model = VLATrainModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters())

    batches = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
               else ["256", "512", "1024", "1536", "2048"])]
    print(f"seq_len={S} hidden={H} horizon={Hh} action_dim={Dd}\n")
    for B in batches:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            embed = torch.randn(B, S, H, device=dev)
            state = torch.randn(B, cfg.state_dim, device=dev)
            acts  = torch.randn(B, Hh, Dd, device=dev)
            mask  = torch.zeros(B, S, dtype=torch.bool, device=dev); mask[:, :66] = True
            # warmup
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss = model(embed, state, acts, mask); loss.backward(); opt.step()
            torch.cuda.synchronize()
            t0 = time.time(); iters = 10
            for _ in range(iters):
                opt.zero_grad(set_to_none=True)
                loss = model(embed, state, acts, mask); loss.backward(); opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            sps = iters * B / dt
            print(f"  batch={B:5d}  peak_VRAM={peak:5.2f} GB  "
                  f"{sps:7.0f} samples/s  ({iters/dt:.1f} step/s)")
            del embed, state, acts, mask
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={B:5d}  OOM")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
