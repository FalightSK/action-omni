"""
scripts/_pt_to_h5_cache.py
──────────────────────────
Convert an existing in-RAM `.pt` VLM-embedding cache to the streamed HDF5 format
(data/hdf5_embeddings.py) WITHOUT re-running the VLM. Lets training lazy-read the
cache off disk (one row per sample) instead of materialising the whole
(N, S, H) tensor in RAM.

Usage:
    python scripts/_pt_to_h5_cache.py asset/runs/aloha/exp03_insertion/vlm_embeddings.pt
    python scripts/_pt_to_h5_cache.py <in.pt> <out.h5>      # explicit output path
    python scripts/_pt_to_h5_cache.py <in.pt> --delete-src  # remove .pt after success

The .pt is loaded once (unavoidable one-time RAM spike); embeddings are then written
to HDF5 in row-batches so the conversion itself adds little overhead. Uncompressed /
contiguous datasets are used (fast shuffled random reads during training).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from data.hdf5_embeddings import HDF5EmbeddingWriter, is_hdf5_path

BATCH = 512  # rows per HDF5 write


def convert(in_pt: str, out_h5: str | None = None, delete_src: bool = False) -> str:
    in_pt = str(in_pt)
    if out_h5 is None:
        out_h5 = str(Path(in_pt).with_suffix(".h5"))
    assert is_hdf5_path(out_h5), f"output must end in .h5/.hdf5: {out_h5}"

    print(f"Loading .pt cache: {in_pt}")
    d = torch.load(in_pt, map_location="cpu", weights_only=False)

    emb = d["embeddings"]                       # (N,S,H) or (N,L,S,H) bf16
    masks = d["img_masks"]                       # (N,S) bool
    states = d["states"].float()                 # (N,state_dim)
    actions = d["actions"].float()               # (N,H,action_dim)
    N = emb.shape[0]
    n_layers = emb.shape[1] if emb.ndim == 4 else 1
    S = emb.shape[-2]
    H = emb.shape[-1]
    horizon = actions.shape[1]
    action_dim = actions.shape[2]
    state_dim = states.shape[1]

    if emb.dtype != torch.bfloat16:
        emb = emb.to(torch.bfloat16)

    attrs = {k: d[k] for k in
             ("format", "dataset", "exp_id", "vlm_extract_layers", "state_dim",
              "full_dataset") if k in d}

    print(f"  N={N:,}  S={S}  H={H}  layers={n_layers}  state_dim={state_dim} "
          f"action=({horizon},{action_dim})  emb.dtype={emb.dtype}")

    w = HDF5EmbeddingWriter(out_h5, N=N, S=S, H=H, state_dim=state_dim,
                            horizon=horizon, action_dim=action_dim,
                            n_layers=n_layers, compression=None, attrs=attrs)
    for ptr in range(0, N, BATCH):
        end = min(ptr + BATCH, N)
        w.write(ptr, emb[ptr:end], masks[ptr:end], states[ptr:end], actions[ptr:end])
        print(f"\r  written {end:,}/{N:,}", end="", flush=True)
    w.close()
    print()

    mb = Path(out_h5).stat().st_size / 1e6
    print(f"  Saved → {out_h5}  ({mb:.0f} MB, HDF5 streamed)")

    if delete_src:
        Path(in_pt).unlink()
        print(f"  Deleted source .pt: {in_pt}")
    return out_h5


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--delete-src"]
    delete_src = "--delete-src" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    in_pt = args[0]
    out_h5 = args[1] if len(args) > 1 else None
    convert(in_pt, out_h5, delete_src=delete_src)


if __name__ == "__main__":
    main()
