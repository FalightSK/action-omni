"""Quick throughput probe for the HDF5 embedding cache + DataLoader path.
Mimics train.py's loader (8 workers, shuffle) and times batches end-to-end onto GPU.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from data.hdf5_embeddings import HDF5EmbeddingDataset


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else \
        "asset/runs/language_table/exp01_baseline/vlm_embeddings.h5"
    nbatch = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    bs = 256
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = HDF5EmbeddingDataset(cache)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=8, persistent_workers=True,
                    pin_memory=(dev.type == "cuda"), prefetch_factor=4)

    it = iter(dl)
    # warm up workers / prefetch
    b = next(it)
    _ = b["embedding"].to(dev, non_blocking=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    n = 0
    for _ in range(nbatch):
        b = next(it)
        e = b["embedding"].to(dev, non_blocking=True)
        s = b["state"].to(dev, non_blocking=True)
        a = b["actions"].to(dev, non_blocking=True)
        n += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    bps = n / dt
    steps_per_epoch = (len(ds) // bs)
    print(f"\n[probe] {n} batches (bs={bs}) in {dt:.1f}s  -> {bps:.1f} batch/s, "
          f"{bps*bs:.0f} samples/s")
    print(f"[probe] epoch = {steps_per_epoch} steps -> ~{steps_per_epoch/bps/60:.1f} min/epoch (loader only)")
    print(f"[probe] 150 epochs (loader-bound upper estimate): "
          f"~{150*steps_per_epoch/bps/3600:.1f} h")


if __name__ == "__main__":
    main()
