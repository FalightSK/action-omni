"""
data/hdf5_embeddings.py
───────────────────────
HDF5-backed embedding cache — drop-in replacement for the in-RAM `.pt` cache used
by PushTEmbeddingDataset, for datasets too large to hold the whole embedding tensor
in memory (e.g. Language Table 10% ≈ ~1M frames × 120 × 1024 bf16 ≈ ~200 GB).

Why HDF5 instead of torch.save(.pt):
  • precompute writes embeddings INCREMENTALLY to disk (chunked datasets), so peak
    RAM is one batch, not the full (N, S, H) tensor — the `.pt` path pre-allocates
    the entire tensor in RAM before saving, which is impossible at this scale.
  • training reads LAZILY per sample (h5py slices straight off disk, like mmap), so
    the trainer never materialises the full cache either.

bfloat16 storage:
  h5py has no native bfloat16 dtype. We store the raw 2-byte bf16 bit-pattern as
  uint16 (lossless, same 2 bytes/elem as bf16) and reinterpret on read with
  torch.Tensor.view(torch.bfloat16). This keeps the cache the same size as the
  bf16 `.pt` cache while staying h5py-compatible.

File schema (single HDF5 file):
  /embeddings  uint16   (N, S, H)        bf16 bits      [chunked, lzf-compressed]
               or       (N, L, S, H)     multi-scale
  /img_masks   uint8    (N, S)           bool as 0/1
  /states      float32  (N, state_dim)
  /actions     float32  (N, H, action_dim)
  attrs: n_samples, format, dataset, exp_id, vlm_extract_layers, state_dim,
         full_dataset, embed_is_bf16_bits=True
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def is_hdf5_path(path: str | Path) -> bool:
    return str(path).lower().endswith((".h5", ".hdf5"))


class HDF5EmbeddingDataset(Dataset):
    """Lazily reads pre-computed VLM token sequences from an HDF5 cache.

    Mirrors PushTEmbeddingDataset's __getitem__ contract:
        {"embedding": float32 (S,H) or (L,S,H), "state": float32, "actions": float32,
         "img_mask": bool (S,)}

    The HDF5 file handle is opened lazily on first access and cached per worker, so
    this is safe with DataLoader num_workers>0 (each worker reopens its own handle).
    """

    def __init__(self, cache_path: str) -> None:
        import h5py  # local import so non-HDF5 paths don't require h5py

        self.cache_path = str(cache_path)
        self._h5: "h5py.File | None" = None

        # Read small tensors + metadata eagerly; keep embeddings/masks on disk.
        with h5py.File(self.cache_path, "r") as f:
            self.n = int(f.attrs.get("n_samples", f["embeddings"].shape[0]))
            self.states  = torch.from_numpy(f["states"][:]).float()
            self.actions = torch.from_numpy(f["actions"][:]).float()
            self._has_mask = "img_masks" in f
            self._embed_is_bf16_bits = bool(f.attrs.get("embed_is_bf16_bits", True))
            self._embed_ndim = f["embeddings"].ndim
            embed_shape = tuple(f["embeddings"].shape[1:])
            fmt = f.attrs.get("format", "hdf5")

        if self._embed_ndim == 4:
            kind = "v3 (multi-scale, HDF5)"
        elif self._embed_ndim == 3:
            kind = "v2 (full sequences, HDF5)"
        else:
            kind = "v1 (mean-pooled, HDF5)"
        print(f"  Loaded embedding cache [{kind}]: {self.n:,} samples  "
              f"| embed shape={embed_shape}  | file={Path(self.cache_path).name}")

        assert self.n == len(self.states) == len(self.actions), \
            "HDF5 cache tensors have mismatched lengths."

    def _file(self):
        import h5py
        if self._h5 is None:
            # swmr-friendly read-only open; one handle per process/worker.
            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        f = self._file()
        raw = f["embeddings"][idx]                     # numpy uint16 (S,H) or (L,S,H)
        if self._embed_is_bf16_bits:
            t = torch.from_numpy(np.ascontiguousarray(raw))   # uint16
            embed = t.view(torch.bfloat16).float()
        else:
            embed = torch.from_numpy(np.ascontiguousarray(raw)).float()

        item = {
            "embedding": embed,
            "state":     self.states[idx],
            "actions":   self.actions[idx],
        }
        if self._has_mask:
            m = f["img_masks"][idx]                    # uint8 (S,)
            item["img_mask"] = torch.from_numpy(np.ascontiguousarray(m)).bool()
        return item

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


# ── Writer helper used by scripts/precompute.py ──────────────────────────────────

class HDF5EmbeddingWriter:
    """Streaming writer — creates chunked datasets and appends batches to disk.

    Usage:
        w = HDF5EmbeddingWriter(path, N, S, H, state_dim, horizon, action_dim,
                                n_layers=1, attrs={...})
        w.write(ptr, embeds_bf16, img_mask_bool, states_f32, actions_f32)
        ...
        w.close()
    """

    def __init__(self, path, N, S, H, state_dim, horizon, action_dim,
                 n_layers=1, compression=None, attrs=None):
        """compression: None (default) → CONTIGUOUS uncompressed datasets, which give
        cheap single-row random reads during shuffled training (one seek + ~S*H*2 B,
        no chunk decompression). Set e.g. "lzf"/"gzip" only for archival caches that
        will be read sequentially — compressed+chunked is ~100× slower for the random
        shuffled access pattern the trainer uses (GPU starves, CPU pegs on decompress).
        """
        import h5py

        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.n_layers = n_layers

        emb_shape = (N, n_layers, S, H) if n_layers > 1 else (N, S, H)

        # uint16 holds the raw bf16 bit-pattern (reinterpreted on read).
        # compression requires chunking; uncompressed → contiguous (best for random reads).
        if compression:
            chunk_rows = 64
            emb_chunk = ((min(chunk_rows, N), n_layers, S, H) if n_layers > 1
                         else (min(chunk_rows, N), S, H))
            self.f.create_dataset("embeddings", shape=emb_shape, dtype=np.uint16,
                                  chunks=emb_chunk, compression=compression)
            self.f.create_dataset("img_masks", shape=(N, S), dtype=np.uint8,
                                  chunks=(min(chunk_rows, N), S), compression=compression)
        else:
            self.f.create_dataset("embeddings", shape=emb_shape, dtype=np.uint16)
            self.f.create_dataset("img_masks", shape=(N, S), dtype=np.uint8)
        self.f.create_dataset("states", shape=(N, state_dim), dtype=np.float32)
        self.f.create_dataset("actions", shape=(N, horizon, action_dim), dtype=np.float32)

        self.f.attrs["n_samples"] = N
        self.f.attrs["embed_is_bf16_bits"] = True
        if attrs:
            for k, v in attrs.items():
                self.f.attrs[k] = v

    def write(self, ptr, embeds_bf16, img_mask_bool, states_f32, actions_f32):
        """embeds_bf16: torch bf16 (B,S,H) or (B,L,S,H). Others torch tensors."""
        B = embeds_bf16.shape[0]
        # reinterpret bf16 → uint16 bits (lossless), move to numpy
        u16 = embeds_bf16.contiguous().view(torch.uint16).cpu().numpy()
        self.f["embeddings"][ptr:ptr+B] = u16
        self.f["img_masks"][ptr:ptr+B]  = img_mask_bool.to(torch.uint8).cpu().numpy()
        self.f["states"][ptr:ptr+B]     = states_f32.cpu().numpy()
        self.f["actions"][ptr:ptr+B]    = actions_f32.cpu().numpy()

    def close(self):
        self.f.close()
