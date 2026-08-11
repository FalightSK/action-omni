"""Condition B cache for SmolVLM2-500M: train on MIXED phrasings, hold out para3.

Mirrors cache_qwen_mix.py. Only tap 30 is cached (the tap used for the
closed-loop arm), which keeps this to ~1.5 GB instead of 17.7 GB.
"""
import io, os, pickle, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
import variants as V

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TAPS = [30]
W, DIM, H = 96, 960, 8
N_TRAIN = 8000
TRAIN_POOL = [0, 1, 2]          # orig, para1, para2 -- para3 (v=3) HELD OUT
TAG = "smix"

INSTR0 = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
paths = pickle.load(open(os.path.join(RES, "paths.pkl"), "rb"))
cols = ["observation.images.image", "observation.state", "action",
        "frame_index", "episode_index", "task_index"]
d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
d = d[(d.task_index >= 10) & (d.task_index < 20)].sort_values(
    ["episode_index", "frame_index"]).reset_index(drop=True)
A = np.stack(d["action"].values).astype("float32")
ep = d.episode_index.values
idx = np.array([i for i in range(len(d)) if i + H < len(d) and ep[i + H] == ep[i]])
rng = np.random.RandomState(0)
eps = rng.permutation(d.episode_index.unique())
val_ep = set(eps[:max(1, len(eps) // 6)])
is_val = np.array([ep[i] in val_ep for i in idx])
tr = rng.permutation(idx[~is_val])[:N_TRAIN]
chunks = np.stack([A[i:i + H] for i in idx])
cmap = {v: k for k, v in enumerate(idx)}
print("train", len(tr), flush=True)

proc = AutoProcessor.from_pretrained(MODEL)
proc.image_processor.do_image_splitting = False
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
IMG_ID = model.config.image_token_id
EOU = tok.convert_tokens_to_ids("<end_of_utterance>")
FAKE = tok.convert_tokens_to_ids("<fake_token_around_image>")


def instr(t, v):
    t = int(t)
    if v == 0: return INSTR0[t]
    return V.PARA[t][v - 1]


Z = np.lib.format.open_memmap(os.path.join(RES, f"{TAG}_feats.npy"), mode="w+",
                              dtype=np.float16, shape=(len(tr), 1, len(TAPS), W, DIM))
MK = np.zeros((len(tr), 1, W), dtype=np.int8)
t0 = time.time()
for n, i in enumerate(tr):
    v = TRAIN_POOL[(i * 7919 + n) % len(TRAIN_POOL)]        # deterministic mix
    text = instr(d["task_index"].iloc[i], v)
    img = Image.open(io.BytesIO(d["observation.images.image"].iloc[i]["bytes"])).convert("RGB")
    p = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
        add_generation_prompt=True)
    inp = proc(text=[p], images=[[img]], return_tensors="pt")
    ids = inp["input_ids"][0].tolist()
    inp = {k: x.cuda() for k, x in inp.items()}
    with torch.no_grad():
        o = model(**inp, output_hidden_states=True)
    T = len(ids)
    for k, l in enumerate(TAPS):
        h = o.hidden_states[l][0]
        if T < W:
            h = torch.nn.functional.pad(h, (0, 0, 0, W - T))
        Z[n, 0, k] = h[:W].to(torch.float16).cpu().numpy()
    m = np.zeros(W, dtype=np.int8)
    for j in range(min(T, W)):
        m[j] = 1 if ids[j] == IMG_ID else 2
    try:
        a_ = len(ids) - 1 - ids[::-1].index(FAKE); e_ = ids.index(EOU)
        for j in range(a_ + 1, min(e_, W)):
            m[j] = 3
    except ValueError:
        pass
    MK[n, 0] = m
    if n % 1000 == 0:
        print(f"  {n}/{len(tr)}  {time.time()-t0:.0f}s", flush=True)
Z.flush()
np.save(os.path.join(RES, f"{TAG}_mask.npy"), MK)
np.save(os.path.join(RES, f"{TAG}_act.npy"), chunks[[cmap[i] for i in tr]])
np.save(os.path.join(RES, f"{TAG}_state.npy"),
        np.stack(d["observation.state"].iloc[tr].values).astype("float32"))
np.save(os.path.join(RES, f"{TAG}_taps.npy"), np.array(TAPS))
print("SMIX_CACHE_DONE", flush=True)
