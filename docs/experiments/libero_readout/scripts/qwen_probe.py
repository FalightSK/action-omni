"""Cross-backbone replication on Qwen3.5-0.8B (local RTX 4070 Ti, 12 GB).

Replicates the TRANSFERABLE findings from the SmolVLM2-500M study:
  1. token budget: how many token states can carry the instruction at all
  2. structural claim: are vision-token states EXACTLY instruction-invariant
  3. token-order intervention: image-first vs text-first
  4. layer-wise probes P1 (Lambda), P2 (eta^2), P3 (ridge R^2), P5 (norms)

Closed-loop is NOT run here - it needs MuJoCo/EGL which requires Linux/WSL2.

Inputs are all local: results/scenes.npz (2000 scenes), results/tasks.pkl,
scripts/variants.py.  No dataset download required.
"""
import io, json, os, pickle, sys, time
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
sys.path.insert(0, HERE)
import variants as V

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-0.8B")
OUT = os.path.join(RES, "qwen_probe.json")
N_SCENES = int(os.environ.get("N_SCENES", "500"))       # LIBERO-Goal subset
BS = int(os.environ.get("BS", "16"))
VARIANTS = [0, 1, 2, 3, 4]                               # orig, para x3, swap

d = np.load(os.path.join(RES, "scenes.npz"))
INSTR0 = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
sel = np.where((d["ti"] >= 10) & (d["ti"] < 20))[0][:N_SCENES]
imgs, ti, ep, act = d["imgs"][sel], d["ti"][sel], d["ep"][sel], d["action"][sel]
N = len(imgs)
print(f"scenes={N}  model={MODEL}", flush=True)

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
try:
    tcfg = model.config.text_config
except AttributeError:
    tcfg = model.config
L = tcfg.num_hidden_layers + 1
D = tcfg.hidden_size
print(f"layers(+emb)={L}  hidden={D}", flush=True)


def prompt(instr, text_first):
    content = ([{"type": "text", "text": instr}, {"type": "image"}] if text_first
               else [{"type": "image"}, {"type": "text", "text": instr}])
    return proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)


def encode(instr, img, text_first=False):
    return proc(text=[prompt(instr, text_first)], images=[img], return_tensors="pt")


def instr_for(t, v):
    t = int(t)
    if v == 0: return INSTR0[t]
    if v < 4:  return V.PARA[t][v - 1]
    return INSTR0[V.swap_partner(t)]


# ---------------------------------------------------------------- 1. token layout
img0 = Image.fromarray(imgs[0])
a = encode("put the bowl on the plate", img0)
b = encode("open the middle drawer of the cabinet", img0)
ids_a = a["input_ids"][0].tolist()
cnt = {}
for i in ids_a:
    cnt[i] = cnt.get(i, 0) + 1
IMG_ID = max(cnt, key=cnt.get)                       # the massively repeated token = image
n_img = cnt[IMG_ID]
pos = [i for i, x in enumerate(ids_a) if x == IMG_ID]
report = {"model": MODEL, "layers": L, "hidden": D,
          "T_orig": len(ids_a), "n_image_tokens": n_img,
          "image_token_id": int(IMG_ID),
          "image_span": [pos[0], pos[-1]],
          "n_nonimage": len(ids_a) - n_img}
print(f"T={len(ids_a)} image_tokens={n_img} span={pos[0]}..{pos[-1]} "
      f"nonimage={len(ids_a)-n_img}", flush=True)

# ---------------------------------------------------------------- 2. structural claim
with torch.no_grad():
    ha = model(**{k: v.cuda() for k, v in a.items()}, output_hidden_states=True).hidden_states
    hb = model(**{k: v.cuda() for k, v in b.items()}, output_hidden_states=True).hidden_states
last = pos[-1]
struct = {}
for l in range(0, L, max(1, L // 6)):
    dmax = (ha[l][0, :last + 1].float() - hb[l][0, :last + 1].float()).abs().max().item()
    struct[f"layer_{l}"] = dmax
    print(f"  layer {l:2d}: max|delta| over image block = {dmax:.3e}", flush=True)
report["structural_max_abs_delta"] = struct
report["image_precedes_instruction"] = bool(pos[-1] < len(ids_a) - 1)

# ---------------------------------------------------------------- 3+4. features
@torch.no_grad()
def extract(text_first):
    """returns Zvis, Zall, Zinstr  each (N, len(VARIANTS), L, D) and norm ratios"""
    Zv = np.zeros((N, len(VARIANTS), L, D), dtype=np.float16)
    Za = np.zeros_like(Zv)
    Zi = np.zeros_like(Zv)
    NR = np.zeros((N, len(VARIANTS), L, 2), dtype=np.float32)   # vis, nonvis mean ||h||
    t0 = time.time()
    for vi, v in enumerate(VARIANTS):
        for s in range(0, N, BS):
            sl = slice(s, min(s + BS, N))
            outs = []
            for k in range(sl.start, sl.stop):
                e = encode(instr_for(ti[k], v), Image.fromarray(imgs[k]), text_first)
                outs.append({kk: vv.cuda() for kk, vv in e.items()})
            for j, inp in enumerate(outs):
                o = model(**inp, output_hidden_states=True)
                ids = inp["input_ids"][0]
                vis = (ids == IMG_ID)
                nvis = ~vis
                for li, h in enumerate(o.hidden_states):
                    hf = h[0].float()
                    Zv[sl.start + j, vi, li] = hf[vis].mean(0).half().cpu().numpy()
                    Za[sl.start + j, vi, li] = hf.mean(0).half().cpu().numpy()
                    Zi[sl.start + j, vi, li] = hf[nvis].mean(0).half().cpu().numpy()
                    n = hf.norm(dim=-1)
                    NR[sl.start + j, vi, li, 0] = n[vis].mean().item()
                    NR[sl.start + j, vi, li, 1] = n[nvis].mean().item()
        print(f"  text_first={text_first} variant {v} done {time.time()-t0:.0f}s", flush=True)
    return Zv, Za, Zi, NR


def analyse(Z, tag):
    out = []
    for l in range(L):
        z = Z[:, :, l, :].astype(np.float32)
        o, sw = z[:, 0], z[:, 4]
        dsw = ((o - sw) ** 2).sum(-1).mean()
        dpa = ((o[:, None] - z[:, 1:4]) ** 2).sum(-1).mean()
        g = z.mean((0, 1)); vm = z.mean(0)
        eta2 = float((N * ((vm - g) ** 2).sum()) / max(((z - g) ** 2).sum(), 1e-12))
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
        out.append(dict(layer=l, eta2=eta2, lam=float(dsw / max(dpa, 1e-12)),
                        r2=float(r2_score(act, pr))))
        if l % max(1, L // 6) == 0:
            print(f"  {tag} L{l:2d} eta2={eta2:.3e} Lambda={out[-1]['lam']:.3f} "
                  f"R2={out[-1]['r2']:.4f}", flush=True)
    return out


print("=== image-first ===", flush=True)
Zv, Za, Zi, NR = extract(False)
report["image_first"] = {"vision_pooled": analyse(Zv, "vis"),
                         "all_pooled": analyse(Za, "all"),
                         "nonimage_pooled": analyse(Zi, "instr")}
report["norm_ratio_nonvis_over_vis"] = [
    float(NR[:, 0, l, 1].mean() / max(NR[:, 0, l, 0].mean(), 1e-9)) for l in range(L)]

print("=== text-first ===", flush=True)
Zv2, _, _, _ = extract(True)
report["text_first"] = {"vision_pooled": analyse(Zv2, "vis-tf")}

vi_if = float(np.mean([x["eta2"] for x in report["image_first"]["vision_pooled"]]))
vi_tf = float(np.mean([x["eta2"] for x in report["text_first"]["vision_pooled"]]))
r2_if = float(max(x["r2"] for x in report["image_first"]["vision_pooled"]))
r2_tf = float(max(x["r2"] for x in report["text_first"]["vision_pooled"]))
report["token_order"] = {"vision_eta2_image_first": vi_if, "vision_eta2_text_first": vi_tf,
                         "vision_r2_image_first": r2_if, "vision_r2_text_first": r2_tf,
                         "delta_r2": r2_tf - r2_if}

json.dump(report, open(OUT, "w"), indent=1)
print("\n=== SUMMARY ===")
print(f" model            {MODEL}  layers={L} hidden={D}")
print(f" tokens           T={report['T_orig']} image={report['n_image_tokens']} "
      f"non-image={report['n_nonimage']}")
print(f" structural max|d| {max(struct.values()):.3e}  (0 => vision states instruction-blind)")
print(f" token order      vision eta2 {vi_if:.3e} -> {vi_tf:.3e}")
print(f"                  vision R2   {r2_if:.4f} -> {r2_tf:.4f}  (delta {r2_tf-r2_if:+.4f})")
print("wrote", OUT)
