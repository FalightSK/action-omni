"""Find the largest training batch size that fits in VRAM with headroom.
Builds the real VLATrainModel and runs fwd+bwd+optim.step at increasing batch
sizes using random tensors of the correct shape, reporting peak VRAM each.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla_train import VLATrainModel


def main():
    cfg = get_config("language_table", "exp01")
    dev = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU total: {total_gb:.1f} GB")

    model = VLATrainModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), weight_decay=cfg.weight_decay)
    S, H = cfg.img_seq_len, cfg.vlm_hidden_size
    Hor, Ad, Sd = cfg.action_horizon, cfg.action_dim, cfg.state_dim

    for bs in [256, 512, 1024, 2048, 4096, 6144, 8192]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            embed = torch.randn(bs, S, H, device=dev)
            state = torch.randn(bs, Sd, device=dev)
            acts  = torch.randn(bs, Hor, Ad, device=dev)
            mask  = torch.zeros(bs, S, dtype=torch.bool, device=dev); mask[:, :66] = True
            opt.zero_grad(set_to_none=True)
            loss = model(embed, state, acts, mask)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            resv = torch.cuda.max_memory_reserved() / 1e9
            print(f"  bs={bs:5d}  OK   peak_alloc={peak:5.2f} GB  reserved={resv:5.2f} GB  "
                  f"({100*resv/total_gb:.0f}% of VRAM)")
            del embed, state, acts, mask, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  bs={bs:5d}  OOM")
                torch.cuda.empty_cache()
                break
            raise


if __name__ == "__main__":
    main()
