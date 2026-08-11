"""Record side-by-side closed-loop rollouts: ORIGINAL vs GOAL-SWAP instruction.

Same task, same initial state, same policy, same noise seed - only the
instruction text differs.  This is the 83.0% vs 0.0% result, visually.
"""
import os, sys, pickle
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, "/workspace/omni")

import torch
import imageio
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForImageTextToText
import variants as V
from sweep2 import Actor, TAPS, mu, sd, smu, ssd, W

TAP, MODE, CKPT = 30, "all", "ck2_all_tap30_lr0.001.pt"
TASKS = [8, 1, 2]          # bowl->plate, bowl->stove, wine bottle->cabinet top
MAXSTEP, REPLAN, H = 300, 8, 8

tasks_map = pickle.load(open("/workspace/omni/tasks.pkl", "rb"))
bd = benchmark.get_benchmark_dict()["libero_goal"]()
MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
proc = AutoProcessor.from_pretrained(MODEL)
proc.image_processor.do_image_splitting = False
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
tap_i = int(np.where(TAPS == TAP)[0][0])
actor = Actor().cuda().eval()
actor.load_state_dict(torch.load(CKPT, map_location="cuda"))


def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1., 1.); ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang


def obs_pack(o):
    return (o["agentview_image"][::-1, ::-1].copy(),
            np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                            o["robot0_gripper_qpos"]]).astype("float32"))


@torch.no_grad()
def policy(img, state, text, gen):
    p = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
        add_generation_prompt=True)
    inp = proc(text=[p], images=[[Image.fromarray(img)]], return_tensors="pt").to("cuda")
    hs = vlm(**inp, output_hidden_states=True).hidden_states[TAP]
    T = hs.shape[1]
    h = hs[:, :W] if T >= W else torch.nn.functional.pad(hs, (0, 0, 0, W - T))[:, :W]
    kpm = torch.ones(1, W, dtype=torch.bool, device="cuda")
    n = min(T, W); kpm[:, :n] = ~inp["attention_mask"].bool()[:, :n]
    st = torch.tensor((state - smu) / ssd, dtype=torch.float32, device="cuda")[None]
    x = torch.randn((1, H, 7), device="cuda", generator=gen)
    NS = 10
    for k in range(NS):
        t = torch.full((1,), 1 - k / NS, device="cuda")
        x = x - (1 / NS) * actor(x, t, h.float(), st, kpm)
    return x[0].cpu().numpy() * sd + mu


def label(frame, lines, colour):
    im = Image.fromarray(frame).resize((384, 384), Image.NEAREST)
    bar = Image.new("RGB", (384, 62), colour)
    d = ImageDraw.Draw(bar)
    for i, ln in enumerate(lines):
        d.text((6, 4 + i * 18), ln[:52], fill=(255, 255, 255))
    out = Image.new("RGB", (384, 446))
    out.paste(bar, (0, 0)); out.paste(im, (0, 62))
    return np.array(out)


def run(tid, variant):
    task = bd.get_task(tid)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    o = env.set_init_state(bd.get_task_init_states(tid)[0])
    for _ in range(10):
        o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    name = task.language.strip().lower()
    ti_ = next((k for k, v in tasks_map.items() if v.strip().lower() == name), 10 + tid)
    text = tasks_map[ti_] if variant == "orig" else tasks_map[V.swap_partner(ti_)]
    gen = torch.Generator(device="cuda").manual_seed(999)
    frames, done = [], False
    col = (26, 110, 46) if variant == "orig" else (150, 30, 30)
    for _ in range(0, MAXSTEP, REPLAN):
        img, st = obs_pack(o)
        chunk = policy(img, st, text, gen)
        for k in range(REPLAN):
            o, r, done, _ = env.step(np.clip(chunk[k], -1, 1).tolist())
            frames.append(label(obs_pack(o)[0],
                                [f"{'ORIGINAL' if variant=='orig' else 'GOAL-SWAP'} instruction",
                                 f'"{text}"',
                                 f"scored task: {name}"], col))
            if done: break
        if done: break
    env.close()
    return frames, bool(done), text, name


os.makedirs("/workspace/omni/videos", exist_ok=True)
summary = []
for tid in TASKS:
    fo, so, to, nm = run(tid, "orig")
    fs, ss, ts, _ = run(tid, "swap")
    n = max(len(fo), len(fs))
    fo += [fo[-1]] * (n - len(fo)); fs += [fs[-1]] * (n - len(fs))
    comb = [np.concatenate([a, b], axis=1) for a, b in zip(fo, fs)]
    out = f"/workspace/omni/videos/task{tid}_orig_vs_swap.mp4"
    imageio.mimsave(out, comb, fps=20, quality=8)
    summary.append((tid, nm, to, so, ts, ss, len(comb), out))
    print(f"task{tid} '{nm}' orig_success={so} swap_success={ss} frames={len(comb)} -> {out}",
          flush=True)

print("\n=== SUMMARY ===")
for tid, nm, to, so, ts, ss, n, out in summary:
    print(f" task{tid}: {nm}")
    print(f"    ORIGINAL  '{to}' -> success={so}")
    print(f"    GOAL-SWAP '{ts}' -> success={ss}")
print("VIDEOS_DONE")
