"""Token-order intervention.

Standard VLM prompting is  User:<image>{instruction}  -- image BEFORE text.
In a causal LM that makes every image-token state independent of the instruction.
Here we compare it against  User:{instruction}<image>  -- text BEFORE image --
which should make all 64 vision states instruction-conditioned.

Forward-only. Measures, per layer, for BOTH orderings:
  eta2_instr on vision-pooled states   (0 by construction for image-first)
  Lambda = swap-dist / paraphrase-dist on vision-pooled states
  ridge R^2 -> action from vision-pooled states
"""
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from transformers import AutoProcessor, AutoModelForImageTextToText

import variants as V

M = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
INSTR0 = pickle.load(open("tasks.pkl", "rb"))
d = np.load("scenes.npz")
sel = np.where((d["ti"] >= 10) & (d["ti"] < 20))[0]      # LIBERO-Goal only
imgs, ti, ep, act = d["imgs"][sel], d["ti"][sel], d["ep"][sel], d["action"][sel]
N = len(imgs)
print("scenes", N, flush=True)

proc = AutoProcessor.from_pretrained(M)
proc.image_processor.do_image_splitting = False
proc.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
IMG = model.config.image_token_id
L = model.config.text_config.num_hidden_layers + 1
VARIANTS = [0, 1, 2, 3, 4]                                # orig, para x3, swap


def prompt(instr, text_first):
    content = ([{"type": "text", "text": instr}, {"type": "image"}] if text_first
               else [{"type": "image"}, {"type": "text", "text": instr}])
    return proc.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True)


def instr_for(t, v):
    t = int(t)
    if v == 0: return INSTR0[t]
    if v < 4:  return V.PARA[t][v - 1]
    return INSTR0[V.swap_partner(t)]


@torch.no_grad()
def extract(text_first):
    """returns Zvis (N, len(VARIANTS), L, 960) - mean over VISION tokens only"""
    Z = np.zeros((N, len(VARIANTS), L, 960), dtype=np.float16)
    BS = 64
    for vi, v in enumerate(VARIANTS):
        for s in range(0, N, BS):
            sl = slice(s, min(s + BS, N))
            pil = [Image.fromarray(x) for x in imgs[sl]]
            txt = [prompt(instr_for(t, v), text_first) for t in ti[sl]]
            inp = proc(text=txt, images=[[p] for p in pil],
                       return_tensors="pt", padding=True).to("cuda")
            out = model(**inp, output_hidden_states=True)
            vis = (inp["input_ids"] == IMG)
            c = vis.sum(1, keepdim=True).clamp(min=1)
            for li, h in enumerate(out.hidden_states):
                Z[sl, vi, li] = ((h.float() * vis.unsqueeze(-1)).sum(1) / c).half().cpu().numpy()
        print(f"  text_first={text_first} variant {v} done", flush=True)
    return Z


def analyse(Z, tag):
    res = []
    for l in range(L):
        z = Z[:, :, l, :].astype(np.float32)
        o, sw = z[:, 0], z[:, 4]
        dsw = ((o - sw) ** 2).sum(-1).mean()
        dpa = ((o[:, None] - z[:, 1:4]) ** 2).sum(-1).mean()
        g = z.mean((0, 1)); vm = z.mean(0)
        eta2 = (N * ((vm - g) ** 2).sum()) / max(((z - g) ** 2).sum(), 1e-12)
        # ridge R^2 from vision-pooled features, episode-grouped
        gk = GroupKFold(n_splits=5); pr = np.zeros_like(act)
        for tr, te in gk.split(o, act, groups=ep):
            best, bl = -9e9, 1.0
            for lam in (1e1, 1e2, 1e3):
                gi = GroupKFold(n_splits=3); sc = []
                for t2, e2 in gi.split(o[tr], act[tr], groups=ep[tr]):
                    m = Ridge(alpha=lam).fit(o[tr][t2], act[tr][t2])
                    sc.append(r2_score(act[tr][e2], m.predict(o[tr][e2])))
                if np.mean(sc) > best: best, bl = np.mean(sc), lam
            pr[te] = Ridge(alpha=bl).fit(o[tr], act[tr]).predict(o[te])
        res.append(dict(layer=l, eta2=float(eta2), lam=float(dsw / max(dpa, 1e-12)),
                        d_swap=float(dsw), d_para=float(dpa), r2=float(r2_score(act, pr))))
        if l % 8 == 0:
            print(f"  {tag} L{l:2d} eta2={eta2:.3e} Lambda={res[-1]['lam']:.3f} R2={res[-1]['r2']:.4f}",
                  flush=True)
    return res


out = {}
for tf in (False, True):
    tag = "text_first" if tf else "image_first"
    print(f"=== {tag} ===", flush=True)
    out[tag] = analyse(extract(tf), tag)

import json
json.dump(out, open("order_probe.json", "w"), indent=1)

print("\n=== VISION-TOKEN instruction sensitivity: image-first vs text-first ===")
print(f"{'layer':>5} {'eta2 img1st':>13} {'eta2 txt1st':>13} {'R2 img1st':>10} {'R2 txt1st':>10}")
for l in (0, 4, 8, 12, 16, 20, 24, 28, 32):
    a, b = out["image_first"][l], out["text_first"][l]
    print(f"{l:>5} {a['eta2']:13.3e} {b['eta2']:13.3e} {a['r2']:10.4f} {b['r2']:10.4f}")
ai = max(x["r2"] for x in out["image_first"])
bi = max(x["r2"] for x in out["text_first"])
print(f"\nbest vision-pooled R2:  image-first={ai:.4f}   text-first={bi:.4f}   delta={bi-ai:+.4f}")
print(f"mean vision eta2:       image-first={np.mean([x['eta2'] for x in out['image_first']]):.3e}"
      f"   text-first={np.mean([x['eta2'] for x in out['text_first']]):.3e}")
