"""
scripts/_h5_decompress.py
─────────────────────────
One-shot transcode of an embedding HDF5 cache from compressed+chunked → contiguous
uncompressed, so shuffled random-access training reads are cheap (no per-sample chunk
decompression). Sequential block read of the source keeps this fast (~one pass),
avoiding a full 3.5h VLM re-precompute.

Usage:
  python scripts/_h5_decompress.py <src.h5> <dst.h5> [--block 4096]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--block", type=int, default=4096, help="rows per copy block")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    assert src.exists(), f"source not found: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        N = int(fin.attrs.get("n_samples", fin["embeddings"].shape[0]))
        print(f"transcoding {N:,} rows  {src.name} -> {dst.name} (contiguous, uncompressed)")

        # contiguous (no chunks/compression) target datasets
        for name in ("embeddings", "img_masks", "states", "actions"):
            d = fin[name]
            fout.create_dataset(name, shape=d.shape, dtype=d.dtype)

        # copy attrs
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        for name in ("embeddings", "img_masks", "states", "actions"):
            din, dout = fin[name], fout[name]
            n = din.shape[0]
            for s in range(0, n, args.block):
                e = min(s + args.block, n)
                dout[s:e] = din[s:e]
            print(f"  copied {name}  shape={tuple(din.shape)} dtype={din.dtype}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    sz = dst.stat().st_size / 1e9
    print(f"done in {(time.time()-t0)/60:.1f} min -> {dst}  ({sz:.1f} GB)")


if __name__ == "__main__":
    main()
