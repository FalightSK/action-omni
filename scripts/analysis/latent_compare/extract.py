"""
Step 2/3 — run every backbone over the frozen probe set.

Each arm sees the identical PIL frame and identical instruction string; only the
prompt wrapper differs, because each family has its own native chat/prompt
convention and evaluating a model outside the format it was trained on would
measure the mismatch rather than the representation.

For every (depth, pool) we store a mean-pooled vector per frame:
  depth — 0.50 / 0.75 / 1.00 of the language stack (relative, not absolute:
          the arms are 16-32 layers deep and absolute indices are incomparable)
  pool  — image tokens only / text tokens only / all non-pad tokens
          (this split is what later lets us ask whether the instruction and the
          scene occupy shared or separate subspaces)

Output: asset/analysis/latent_compare/latents_<model>_<key>.h5
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from backbones import ARMS, DEPTHS, DOC_LAYER, GATE_KEYS, POOLS, load_backbone  # noqa: E402

OUT_DIR = ROOT / "asset" / "analysis" / "latent_compare"

MODELS = ARMS
# The truncated controls (cosmos16, smolvlm2_16) are gone. They existed to
# depth-match a robot arm against its stock base; now that every stock arm is
# read at its descendant's documented layer (backbones.DOC_LAYER), the whole
# comparison is depth-matched by construction and a separate truncated arm would
# only duplicate it.
EXTRA: list[str] = []
# Language Table is still extracted — it is dropped from the per-arm analyses,
# not from the corpus, because the dataset-gate figure needs it as the negative
# anchor. Analyses import backbones.KEYS (three datasets); only the gate imports
# GATE_KEYS.
KEYS = GATE_KEYS

# 3B arms carry 256 image tokens at 224x224; the 500M arms tile 512x512 into
# many more.
#
# These were measured, not guessed. Benchmarking cosmos (the heaviest arm — 28
# layers, and output_hidden_states keeps every one of them):
#     B=16  5.64 GB  56.8 img/s      B=64   7.91 GB  55.5 img/s
#     B=32  6.39 GB  56.1 img/s      B=128 10.90 GB  58.9 img/s
# Throughput is FLAT from B=16 upward while VRAM grows ~0.047 GB/sample, so this
# stage is compute-bound. Pushing the batch until the card is full would buy ~4%
# and spend the headroom that absorbs LIBERO's longer instruction strings. 32 is
# the knee: the full gain, at 6.4 GB of 16.3. The old values (4-16) were the
# real waste — they sat below the knee and left ~40% of throughput unused.
BATCH = {"qwen": 32, "pi05": 32, "paligemma": 32, "smolvla": 16, "smolvlm2": 16,
         "groot": 32, "cosmos": 32, "qwen3vl": 32}

# Measured per arm AFTER two fixes that changed these numbers by ~7x, so the
# earlier values are not comparable:
#   - the LM head and its cross-entropy no longer run (they never fed the probe;
#     on pi05 they cost 69M logit elements per frame and 30 GB peak at B=32)
#   - grafted arms are cast to one dtype (pi05 was 439 bf16 + 164 float32, i.e.
#     10.86 GB of weights against stock PaliGemma's 5.85 GB)
# Post-fix, at B=16:  pi05 6.77 GB / 32.8 img/s  |  paligemma 6.33 / 32.2
#                     groot 3.38 GB / 72.9 img/s |  cosmos ~6.4 / 56
# Throughput is flat from B=16 upward on every arm measured, so 32 is chosen for
# margin rather than speed, and the two 500M arms stay at 16 because they tile
# 512x512 into far more image tokens than the 3B arms carry at 224x224.


def run_one(bb, key: str, batch: int, device: str) -> Path:
    probe = OUT_DIR / f"probe_{key}.h5"
    with h5py.File(probe, "r") as f:
        images = f["images"][:]
        instr_id = f["instr_id"][:]
        import json

        instructions = json.loads(f.attrs["instructions"])

    texts = [instructions[i] for i in instr_id]
    n = len(images)
    # Keyed by whatever taps the backbone emits, not by DEPTHS alone — arms with
    # a documented read layer also emit ("doc", pool), and a preallocated dict
    # over DEPTHS silently KeyErrors on it.
    acc: dict[tuple, list] = collections.defaultdict(list)

    t0 = time.time()
    for s in range(0, n, batch):
        e = min(n, s + batch)
        pil = [Image.fromarray(images[i]) for i in range(s, e)]
        res = bb.encode(pil, texts[s:e], device=device)
        for k, v in res.items():
            acc[k].append(v)
        if s % (batch * 20) == 0:
            done = e / n
            el = time.time() - t0
            eta = el / max(done, 1e-9) - el
            print(f"    {key} {e}/{n} ({done:5.1%}) eta {eta/60:.1f}m", flush=True)

    out = OUT_DIR / f"latents_{bb.name}_{key}.h5"
    with h5py.File(out, "w") as f:
        for (d, p), chunks in acc.items():
            arr = np.concatenate(chunks, axis=0).astype(np.float32)
            f.create_dataset(_tag(d, p), data=arr,
                             compression="gzip", compression_opts=4)
        f.attrs["model"] = bb.name
        f.attrs["key"] = key
        f.attrs["n_layers"] = bb.n_layers
        f.attrs["hidden_size"] = bb.hidden_size
        f.attrs["depths"] = list(DEPTHS)
        # Absolute layer behind the doc_* datasets, recorded so a reader never
        # has to recompute it from DOC_LAYER and risk reading a newer table
        # against older latents.
        if bb.doc_layer is not None:
            f.attrs["doc_layer"] = int(bb.doc_layer)
            f.attrs["doc_rel_depth"] = float(bb.doc_layer) / bb.n_layers
    print(f"  wrote {out.name}  ({time.time()-t0:.0f}s)")
    return out


def _tag(d, pool: str) -> str:
    """Dataset name for one (depth-or-'doc', pool) tap.

    'doc' is kept as its own name rather than folded into the d### scheme
    because it is an absolute layer, not a grid fraction, and for most arms it
    falls between grid points (GR00T: 16/28 = 57%).
    """
    if d in ("doc", "kv"):
        return f"{d}_{pool}"
    return f"d{int(float(d) * 100):03d}_{pool}"


def _stale(path: Path) -> bool:
    """True if `path` must be re-extracted.

    Existence alone is NOT sufficient. When the depth grid changed from
    (0.50, 0.75, 1.00) to (0.00, 0.25, 0.50, 0.75, 1.00), an existence-only
    check skipped every previously extracted arm and left the run holding three
    depths for seven arms and five for the eighth — a mix that is invisible
    downstream because each analysis just reads the tags it wants and silently
    drops arms that lack them. So compare the stored grid against the current
    one and re-extract on any mismatch, the same guard the training embedding
    cache uses on vlm_extract_layers.
    """
    if not path.exists():
        return True
    try:
        with h5py.File(path, "r") as f:
            have = tuple(round(float(x), 4) for x in f.attrs.get("depths", ()))
            arm = str(f.attrs.get("model", ""))
            want = list(DEPTHS) + ["doc"]
            # kv_* exists only for arms whose policy reads K/V, so requiring it
            # everywhere would mark every hidden-state arm permanently stale.
            if DOC_LAYER.get(arm, (None, None, "hidden", None))[2] == "kv":
                want.append("kv")
            missing = [_tag(d, p) for d in want for p in POOLS
                       if _tag(d, p) not in f]
    except OSError:
        return True                      # truncated or half-written
    want = tuple(round(float(d), 4) for d in DEPTHS)
    if have != want:
        print(f"  [stale] {path.name}: depths {have} != {want}")
        return True
    if missing:
        print(f"  [stale] {path.name}: missing {missing[:3]}")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS + EXTRA)
    ap.add_argument("--keys", nargs="+", default=KEYS, choices=KEYS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        todo = [k for k in a.keys if a.overwrite or _stale(OUT_DIR / f"latents_{m}_{k}.h5")]
        if not todo:
            print(f"[{m}] all outputs present — skipping")
            continue
        print(f"\n=== loading {m}", flush=True)
        bb = load_backbone(m, device=a.device)
        for k in todo:
            run_one(bb, k, BATCH[m], a.device)
        del bb
        torch.cuda.empty_cache()
    print("\nextraction complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
