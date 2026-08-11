"""Cache Qwen3.5-0.8B token features for LIBERO-Goal (local RTX 4070 Ti).

Mirrors cache2.py (the corrected SmolVLM2 cache) but for Qwen:
  d=1024, 25 hidden states, image-token id discovered at runtime.
Window is measured from the data rather than assumed, so no truncation.

mask codes: 0=pad  1=vision  2=scaffold  3=instruction
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

MODEL = "Qwen/Qwen3.5-0.8B"
TAPS = [0, 12, 24]                       # embedding, mid, final
H = 8
N_TRAIN = int(os.environ.get("N_TRAIN", "6000"))
N_VAL = int(os.environ.get("N_VAL", "800"))

INSTR0 = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
paths = pickle.load(open(os.path.join(RES, "paths.pkl"), "rb"))
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

proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
DIM = model.config.text_config.hidden_size
tok = proc.tokenizer


def build(instr, img):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[txt], images=[img], return_tensors="pt")


def instr_for(t, v):
    t = int(t)
    if v == 0: return INSTR0[t]
    if v < 4:  return V.PARA[t][v - 1]
    if v == 4: return INSTR0[V.swap_partner(t)]
    if v == 5: return V.BLANK
    return V.NONSENSE


# --- discover image token id + max sequence length over ALL variants ---
img0 = Image.fromarray(np.array(Image.open(io.BytesIO(
    d["observation.images.image"].iloc[0]["bytes"])).convert("RGB")))
e0 = build(INSTR0[10], img0)
ids0 = e0["input_ids"][0].tolist()
c = {}
for i in ids0: c[i] = c.get(i, 0) + 1
IMG_ID = max(c, key=c.get)
maxT = 0
for t in range(10, 20):
    for v in range(7):
        s = instr_for(t, v)
        maxT = max(maxT, build(s, img0)["input_ids"].shape[1])
W = int(np.ceil(maxT / 8) * 8)
print(f"IMG_ID={IMG_ID} n_img={c[IMG_ID]} maxT={maxT} -> window W={W} dim={DIM}", flush=True)
np.save(os.path.join(RES, "qwen_meta.npy"),
        np.array([IMG_ID, c[IMG_ID], maxT, W, DIM] + TAPS))


def run(rows, variants, out, mout):
    Z = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(len(rows), len(variants), len(TAPS), W, DIM))
    MK = np.zeros((len(rows), len(variants), W), dtype=np.int8)
    t0 = time.time()
    for vi, v in enumerate(variants):
        for n, i in enumerate(rows):
            img = Image.open(io.BytesIO(
                d["observation.images.image"].iloc[i]["bytes"])).convert("RGB")
            inp = build(instr_for(d["task_index"].iloc[i], v), img)
            ids = inp["input_ids"][0].tolist()
            inp = {k: x.cuda() for k, x in inp.items()}
            with torch.no_grad():
                o = model(**inp, output_hidden_states=True)
            T = len(ids)
            for k, l in enumerate(TAPS):
                h = o.hidden_states[l][0]
                if T < W:
                    h = torch.nn.functional.pad(h, (0, 0, 0, W - T))
                Z[n, vi, k] = h[:W].to(torch.float16).cpu().numpy()
            m = np.zeros(W, dtype=np.int8)
            for j in range(min(T, W)):
                m[j] = 1 if ids[j] == IMG_ID else 2
            # instruction = the run of non-image tokens after the image block,
            # before the trailing chat scaffold
            ipos = [j for j in range(min(T, W)) if ids[j] == IMG_ID]
            if ipos:
                start = ipos[-1] + 1
                txt_ids = tok(instr_for(d["task_index"].iloc[i], v),
                              add_special_tokens=False)["input_ids"]
                for j in range(start, min(start + len(txt_ids), W)):
                    m[j] = 3
            MK[n, vi] = m
            if n % 500 == 0:
                print(f"  {os.path.basename(out)} v{v} {n}/{len(rows)} "
                      f"{time.time()-t0:.0f}s", flush=True)
    Z.flush(); np.save(mout, MK)
    print(out, "DONE", flush=True)


run(tr, [0], os.path.join(RES, "qtrain_feats.npy"), os.path.join(RES, "qtrain_mask.npy"))
np.save(os.path.join(RES, "qtrain_act.npy"), chunks[[cmap[i] for i in tr]])
np.save(os.path.join(RES, "qtrain_state.npy"),
        np.stack(d["observation.state"].iloc[tr].values).astype("float32"))
run(va, [0, 1, 2, 3, 4], os.path.join(RES, "qval_feats.npy"), os.path.join(RES, "qval_mask.npy"))
np.save(os.path.join(RES, "qval_act.npy"), chunks[[cmap[i] for i in va]])
np.save(os.path.join(RES, "qval_state.npy"),
        np.stack(d["observation.state"].iloc[va].values).astype("float32"))
np.save(os.path.join(RES, "qtaps.npy"), np.array(TAPS))
print("ALL DONE taps", TAPS, "W", W)
