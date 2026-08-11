"""Cache per-layer VLM token features for LIBERO-Goal - CORRECTED.

Fixes the 81-token right-truncation that clipped the instruction tail
(39/50 variants exceeded 81 tokens).  Window = 96, and we now also store a
per-token TYPE MASK so read-out variants (all tokens vs instruction-only)
can be compared on identical cached features.

mask codes: 0=pad  1=vision  2=scaffold  3=instruction
"""
import io, os, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import variants as V

POOL = ThreadPoolExecutor(16)
W = 96                                  # token window (max observed T = 88)
TAPS = list(range(0, 33, 3))
if 32 not in TAPS:
    TAPS.append(32)
H = 8
N_TRAIN, N_VAL = 8000, 1000
M = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

INSTR0 = pickle.load(open("tasks.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))
cols = ["observation.images.image", "observation.state", "action",
        "frame_index", "episode_index", "task_index"]
d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
d = d[(d.task_index >= 10) & (d.task_index < 20)].sort_values(
    ["episode_index", "frame_index"]).reset_index(drop=True)
print("goal frames", len(d), "eps", d.episode_index.nunique(), flush=True)

A = np.stack(d["action"].values).astype("float32")
ep = d.episode_index.values
idx = np.array([i for i in range(len(d)) if i + H < len(d) and ep[i + H] == ep[i]])
rng = np.random.RandomState(0)
eps = rng.permutation(d.episode_index.unique())
val_ep = set(eps[:max(1, len(eps) // 6)])
is_val = np.array([ep[i] in val_ep for i in idx])
tr = rng.permutation(idx[~is_val])[:N_TRAIN]
va = rng.permutation(idx[is_val])[:N_VAL]
print("train", len(tr), "val", len(va), flush=True)
chunks = np.stack([A[i:i + H] for i in idx])
cmap = {v: k for k, v in enumerate(idx)}

proc = AutoProcessor.from_pretrained(M)
proc.image_processor.do_image_splitting = False
proc.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
IMG = model.config.image_token_id
tok = proc.tokenizer
EOU = tok.convert_tokens_to_ids("<end_of_utterance>")
FAKE = tok.convert_tokens_to_ids("<fake_token_around_image>")


def run(rows, variants, out, mout):
    Z = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(len(rows), len(variants), len(TAPS), W, 960))
    MK = np.zeros((len(rows), len(variants), W), dtype=np.int8)
    BS, t0, ntrunc = 96, time.time(), 0
    for vi, v in enumerate(variants):
        for s in range(0, len(rows), BS):
            r = rows[s:s + BS]
            _dec = lambda i: Image.open(io.BytesIO(
                d["observation.images.image"].iloc[i]["bytes"])).convert("RGB")
            pil = list(POOL.map(_dec, r))
            ts = d["task_index"].iloc[r].values

            def ins(t):
                t = int(t)
                if v == 0: return INSTR0[t]
                if v < 4:  return V.PARA[t][v - 1]
                if v == 4: return INSTR0[V.swap_partner(t)]
                if v == 5: return V.BLANK
                return V.NONSENSE

            txt = [proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ins(t)}]}],
                add_generation_prompt=True) for t in ts]
            inp = proc(text=txt, images=[[p] for p in pil], return_tensors="pt",
                       padding=True).to("cuda")
            T = inp["input_ids"].shape[1]
            if T > W:
                ntrunc += len(r)
            with torch.no_grad():
                o = model(**inp, output_hidden_states=True)

            def win(x):
                x = x[:, :W]
                if x.shape[1] < W:
                    x = torch.nn.functional.pad(x, (0, 0, 0, W - x.shape[1]))
                return x

            hs = torch.stack([win(o.hidden_states[l]) for l in TAPS], 1)
            Z[s:s + len(r), vi] = hs.to(torch.float16).cpu().numpy()

            ids = inp["input_ids"]
            am = inp["attention_mask"].bool()
            for b in range(ids.shape[0]):
                row = ids[b].tolist()
                m = np.zeros(W, dtype=np.int8)
                n = min(T, W)
                for j in range(n):
                    if not am[b, j]:
                        continue
                    m[j] = 1 if row[j] == IMG else 2
                try:
                    a_ = len(row) - 1 - row[::-1].index(FAKE)
                    e_ = row.index(EOU)
                    for j in range(a_ + 1, min(e_, W)):
                        m[j] = 3
                except ValueError:
                    pass
                MK[s + b, vi] = m
            if (s // BS) % 20 == 0:
                print(out, "v", v, "row", s, "/", len(rows), "T", T,
                      f"{time.time() - t0:.0f}s", flush=True)
    Z.flush()
    np.save(mout, MK)
    print(out, "DONE truncated_samples=", ntrunc, flush=True)


run(tr, [0], "train_feats2.npy", "train_mask2.npy")
np.save("train_act.npy", chunks[[cmap[i] for i in tr]])
np.save("train_state.npy", np.stack(d["observation.state"].iloc[tr].values).astype("float32"))
run(va, [0, 1, 2, 3, 4], "val_feats2.npy", "val_mask2.npy")
np.save("val_act.npy", chunks[[cmap[i] for i in va]])
np.save("val_state.npy", np.stack(d["observation.state"].iloc[va].values).astype("float32"))
np.save("taps.npy", np.array(TAPS))
np.save("val_task.npy", d["task_index"].iloc[va].values)
print("ALL DONE taps", TAPS, "window", W)
