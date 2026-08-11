"""Can a backbone's IMAGE tokens see the instruction at all?

Encode the same frames with different text and measure how far the image-token
vector moves. This separates three things the pooled instruction eta^2 conflates:

  cannot see it   image tokens are produced before the instruction is attended
                  to, so the vectors are BIT-IDENTICAL under any text change.
                  No training can alter this — it is the attention mask.
  sees it         the vector moves, but as much for the same words scrambled as
                  for a genuinely different instruction: token-level reactivity.
  tracks meaning  the vector moves substantially MORE for a changed meaning than
                  for the same words shuffled.

Why this matters: the phase-split result (phase_split.py) shows stock PaliGemma
five times above the noise floor at frame 0 while robot-trained GR00T sits on
it. That is not explicable by training, and this script tests the architectural
explanation directly.

All four conditions are encoded in ONE batch so the tokenizer pads every item to
the same length. Encoding them separately lets token count vary between calls,
and a changed sequence length alone moves a bfloat16 vector by ~1e-3 through
kernel/accumulation differences — indistinguishable from a small real effect.
An earlier version of this test made exactly that mistake and reported a
non-zero response for arms that are in fact bit-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backbones import ARMS, load_backbone  # noqa: E402

DIR = HERE.parents[2] / "asset" / "analysis" / "latent_compare"
KEY = "libero_goal"
POOL = "image"
N = 12                  # early-phase frames; the zero/non-zero split is structural
EARLY = 0.10


def main() -> int:
    with h5py.File(DIR / f"probe_{KEY}.h5", "r") as f:
        phase, task = f["phase"][:], f["instr_id"][:]
        instr = json.loads(f.attrs["instructions"])
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(np.where(phase < EARLY)[0], N, replace=False))
        imgs = [Image.fromarray(f["images"][i]) for i in pick]
        tids = task[pick]

    A = [instr[t] for t in tids]                                   # own instruction
    B = [instr[(t + 5) % len(instr)] for t in tids]                # different meaning
    C = [" ".join(rng.permutation(s.split())) for s in A]          # same words, shuffled
    D = [""] * N                                                   # no text
    batch_imgs, batch_txt = imgs * 4, A + B + C + D

    out: dict = {}
    print(f"{N} early-phase frames, 4 conditions in one batch (uniform padding)")
    print(f"{'arm':<11}{'diff instr':>11}{'scrambled':>11}{'empty':>9}{'ratio':>7}  reading")
    for arm in ARMS:
        try:
            bb = load_backbone(arm)
        except Exception as e:                                     # noqa: BLE001
            print(f"{arm:<11} LOAD FAIL {type(e).__name__}: {str(e)[:45]}")
            continue
        X = bb.encode(batch_imgs, batch_txt)[("doc", POOL)]
        a, b, c, d = (X[i * N:(i + 1) * N] for i in range(4))

        def rel(x, y):
            return float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-9))

        ab, ac, ad = rel(a, b), rel(a, c), rel(a, d)
        ratio = (ab / ac) if ac > 1e-9 else None
        reading = ("cannot see the text" if ab == 0.0 else
                   "tracks meaning" if (ratio or 0) > 1.3 else "token-level only")
        out[arm] = {"diff_instruction": ab, "scrambled": ac, "empty": ad,
                    "ratio_meaning_vs_scrambled": ratio, "reading": reading}
        rs = "--" if ratio is None else f"{ratio:.2f}"
        print(f"{arm:<11}{ab:>11.4f}{ac:>11.4f}{ad:>9.4f}{rs:>7}  {reading}", flush=True)
        del bb
        torch.cuda.empty_cache()

    dst = DIR / "text_visibility_libero.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
