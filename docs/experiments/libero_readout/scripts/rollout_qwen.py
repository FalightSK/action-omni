"""Closed-loop LIBERO-Goal rollout for a Qwen3.5-0.8B read-out, NATIVE WINDOWS.

Runs in the `libero-qwen` conda env: mujoco 3.3.1 + robosuite 1.4.0 render via
MUJOCO_GL=wgl (GPU, ~9-10 ms/step), Qwen + actor on the same GPU.

No caching in this loop - observations depend on the policy's own actions, so
every VLM forward is live.  Single environment; run several arms in parallel
processes instead (VRAM is ~2 GB per arm).
"""
import argparse, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("HF_HOME", "E:/hf_cache")

import numpy as np
import torch

# LIBERO init_files are numpy pickles; torch>=2.6 defaults weights_only=True.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
import variants as V
from sweep_qwen import Actor, TAPS, W, DIM, MAXI, H, DA

AP = argparse.ArgumentParser()
AP.add_argument("--tap", type=int, required=True)
AP.add_argument("--mode", default="all", choices=["all", "instr"])
AP.add_argument("--variant", default="orig", choices=["orig", "para1", "swap"])
AP.add_argument("--episodes", type=int, default=20, help="per task")
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--replan", type=int, default=8)
A = AP.parse_args()

nz = np.load(os.path.join(RES, "qnorm.npz"))
mu, sd, smu, ssd = nz["mu"], nz["sd"], nz["smu"], nz["ssd"]
META = np.load(os.path.join(RES, "qwen_meta.npy"))
IMG_ID = int(META[0])

MODEL = "Qwen/Qwen3.5-0.8B"
proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
actor = Actor().cuda().eval()
actor.load_state_dict(torch.load(os.path.join(RES, f"qck_{A.mode}_tap{A.tap}.pt"),
                                 map_location="cuda"))
import pickle
tasks_map = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
bd = benchmark.get_benchmark_dict()["libero_goal"]()


def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang


def pack(o):
    return (o["agentview_image"][::-1, ::-1].copy(),
            np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                            o["robot0_gripper_qpos"]]).astype("float32"))


def instr_for(ti_, v):
    if v == "orig":  return tasks_map[ti_]
    if v == "para1": return V.PARA[ti_][0]
    return tasks_map[V.swap_partner(ti_)]


@torch.no_grad()
def policy(img, state, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[txt], images=[Image.fromarray(img)], return_tensors="pt")
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
        ipos = [j for j in range(min(T, W)) if ids[j] == IMG_ID]
        start = (ipos[-1] + 1) if ipos else 0
        ntxt = len(tok(text, add_special_tokens=False)["input_ids"])
        sel = list(range(start, min(start + ntxt, W)))[:MAXI]
        ctx = torch.zeros(1, MAXI, DIM, device="cuda", dtype=hs.dtype)
        kpm = torch.ones(1, MAXI, dtype=torch.bool, device="cuda")
        if sel:
            ctx[0, :len(sel)] = hs[sel]
            kpm[0, :len(sel)] = False
    st = torch.tensor((state - smu) / ssd, dtype=torch.float32, device="cuda")[None]
    x = torch.randn((1, H, DA), device="cuda")
    for k in range(10):
        t = torch.full((1,), 1 - k / 10, device="cuda")
        x = x - 0.1 * actor(x, t, ctx.float(), st, kpm)
    return x[0].cpu().numpy() * sd + mu


results, t0, nsteps = [], time.time(), 0
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
            img, st = pack(o)
            chunk = policy(img, st, text)
            for k in range(A.replan):
                o, r, done, _ = env.step(np.clip(chunk[k], -1, 1).tolist())
                nsteps += 1
                if done: break
            if done: break
        succ += int(done)
    env.close()
    results.append(dict(task=name, tid=tid, success=succ, n=A.episodes))
    el = time.time() - t0
    print(json.dumps(results[-1]) + f"  [{el:.0f}s {nsteps/el:.0f} steps/s]", flush=True)

tot = sum(r["success"] for r in results); n = sum(r["n"] for r in results)
out = dict(model="Qwen3.5-0.8B", tap=A.tap, mode=A.mode, variant=A.variant,
           success_rate=tot / n, n=n, wall_s=round(time.time() - t0, 1), per_task=results)
print(f"\nSR {tot}/{n} = {100*tot/n:.1f}%")
fn = os.path.join(RES, f"qroll_{A.mode}_tap{A.tap}_{A.variant}.json")
json.dump(out, open(fn, "w"), indent=1)
print("wrote", fn)
