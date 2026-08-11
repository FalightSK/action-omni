"""Closed-loop LIBERO-Goal rollout for the SmolVLM2-500M read-out, NATIVE WINDOWS.

Exists to test whether the paraphrase collapse found on Qwen3.5 (RESULTS.md 6bb)
also holds on SmolVLM2 -- the retraction otherwise rests on a single backbone.

Self-contained: the Actor is inlined rather than imported from sweep2.py, which
loads multi-GB caches at module level. Action/state normalisation constants are
regenerated from the local parquets using the SAME split logic and seed as
cache2.py, so they match the constants the checkpoint was trained with.
"""
import argparse, io, json, math, os, pickle, sys, time
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("HF_HOME", "E:/hf_cache")

import numpy as np
import pandas as pd
import torch

_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CKPT_DIR = os.path.join(os.path.dirname(HERE), "checkpoints")
sys.path.insert(0, HERE)
import variants as V

AP = argparse.ArgumentParser()
AP.add_argument("--tap", type=int, default=30)
AP.add_argument("--mode", default="all", choices=["all", "instr"])
AP.add_argument("--variant", default="para1",
                choices=["orig", "para1", "para2", "para3", "swap"])
AP.add_argument("--episodes", type=int, default=20)
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--replan", type=int, default=8)
AP.add_argument("--ckpt-tag", default="ck2", help="ck2 (cond A) | ck2mix (cond B)")
A = AP.parse_args()

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
W, DIM, MAXI, H, DA = 96, 960, 20, 8, 7

# ---- regenerate the exact normalisation constants cache2.py used -------------
tasks_map = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
paths = pickle.load(open(os.path.join(RES, "paths.pkl"), "rb"))
cols = ["observation.state", "action", "frame_index", "episode_index", "task_index"]
d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
d = d[(d.task_index >= 10) & (d.task_index < 20)].sort_values(
    ["episode_index", "frame_index"]).reset_index(drop=True)
Aacts = np.stack(d["action"].values).astype("float32")
ep = d.episode_index.values
idx = np.array([i for i in range(len(d)) if i + H < len(d) and ep[i + H] == ep[i]])
rng = np.random.RandomState(0)                       # same seed as cache2.py
eps = rng.permutation(d.episode_index.unique())
val_ep = set(eps[:max(1, len(eps) // 6)])
is_val = np.array([ep[i] in val_ep for i in idx])
tr = rng.permutation(idx[~is_val])[:8000]            # cache2.py used N_TRAIN=8000
chunks = np.stack([Aacts[i:i + H] for i in idx])
cmap = {v: k for k, v in enumerate(idx)}
Atr = chunks[[cmap[i] for i in tr]]
Str = np.stack(d["observation.state"].iloc[tr].values).astype("float32")
mu, sd = Atr.reshape(-1, DA).mean(0), Atr.reshape(-1, DA).std(0) + 1e-6
smu, ssd = Str.mean(0), Str.std(0) + 1e-6
print(f"norm constants from {len(tr)} train samples | mu[0]={mu[0]:.4f} sd[0]={sd[0]:.4f}",
      flush=True)


from actor_def import Actor  # shared definition


proc = AutoProcessor.from_pretrained(MODEL)
proc.image_processor.do_image_splitting = False
tok = proc.tokenizer
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
IMG_ID = vlm.config.image_token_id
EOU = tok.convert_tokens_to_ids("<end_of_utterance>")
FAKE = tok.convert_tokens_to_ids("<fake_token_around_image>")

actor = Actor().cuda().eval()
ck = os.path.join(CKPT_DIR, f"{A.ckpt_tag}_{A.mode}_tap{A.tap}_lr0.001.pt")
actor.load_state_dict(torch.load(ck, map_location="cuda"))
print("loaded", os.path.basename(ck), flush=True)
bd = benchmark.get_benchmark_dict()["libero_goal"]()


def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0); ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang


def pack(o):
    return (o["agentview_image"][::-1, ::-1].copy(),
            np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                            o["robot0_gripper_qpos"]]).astype("float32"))


def instr_for(ti_, v):
    if v == "orig": return tasks_map[ti_]
    if v.startswith("para"): return V.PARA[ti_][int(v[-1]) - 1]
    return tasks_map[V.swap_partner(ti_)]


@torch.no_grad()
def policy(img, state, text):
    p = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
        add_generation_prompt=True)
    inp = proc(text=[p], images=[[Image.fromarray(img)]], return_tensors="pt")
    ids = inp["input_ids"][0].tolist()
    inp = {k: v.cuda() for k, v in inp.items()}
    hs = vlm(**inp, output_hidden_states=True).hidden_states[A.tap][0]
    T = hs.shape[0]
    if T < W:
        hs = torch.nn.functional.pad(hs, (0, 0, 0, W - T))
    hs = hs[:W]
    if A.mode == "all":
        ctx = hs[None]
        kpm = torch.ones(1, W, dtype=torch.bool, device="cuda")
        kpm[0, :min(T, W)] = False
    else:
        try:
            a_ = len(ids) - 1 - ids[::-1].index(FAKE); e_ = ids.index(EOU)
            sel = list(range(a_ + 1, min(e_, W)))[:MAXI]
        except ValueError:
            sel = []
        ctx = torch.zeros(1, MAXI, DIM, device="cuda", dtype=hs.dtype)
        kpm = torch.ones(1, MAXI, dtype=torch.bool, device="cuda")
        if sel:
            ctx[0, :len(sel)] = hs[sel]; kpm[0, :len(sel)] = False
    st = torch.tensor((state - smu) / ssd, dtype=torch.float32, device="cuda")[None]
    x = torch.randn((1, H, DA), device="cuda")
    for k in range(10):
        t = torch.full((1,), 1 - k / 10, device="cuda")
        x = x - 0.1 * actor(x, t, ctx.float(), st, kpm)
    return x[0].cpu().numpy() * sd + mu


rows, t0 = [], time.time()
for tid in range(bd.n_tasks):
    task = bd.get_task(tid)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=256, camera_widths=256)
    env.seed(0)
    name = task.language.strip().lower()
    ti_ = next((k for k, v in tasks_map.items() if v.strip().lower() == name), 10 + tid)
    text = instr_for(ti_, A.variant)
    inits = bd.get_task_init_states(tid)
    succ = 0
    for e in range(A.episodes):
        env.reset()
        o = env.set_init_state(inits[e % len(inits)])
        for _ in range(10):
            o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        done = False
        for _ in range(0, A.max_steps, A.replan):
            img, stt = pack(o)
            chunk = policy(img, stt, text)
            for k in range(A.replan):
                o, r, done, _ = env.step(np.clip(chunk[k], -1, 1).tolist())
                if done: break
            if done: break
        succ += int(done)
    env.close()
    rows.append(dict(task=name, tid=tid, success=succ, n=A.episodes))
    print(json.dumps(rows[-1]) + f"  [{time.time()-t0:.0f}s]", flush=True)

tot = sum(r["success"] for r in rows); n = sum(r["n"] for r in rows)
print(f"\nSR {tot}/{n} = {100*tot/n:.1f}%")
json.dump(dict(model="SmolVLM2-500M", tap=A.tap, mode=A.mode, variant=A.variant,
               success_rate=tot / n, n=n, per_task=rows),
          open(os.path.join(RES, f"sroll_{A.ckpt_tag}_{A.mode}_tap{A.tap}_{A.variant}.json"), "w"), indent=1)
