"""Closed-loop LIBERO-Goal rollout for a trained tap read-out.

Runs in the venv-libero env (numpy<2, robosuite 1.4.0, hf-libero).
Stage A: verify the rendered observation matches the training dataset (camera + flip).
Stage B: closed-loop rollout, success rate per instruction variant.
"""
import os, sys, json, argparse, pickle
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
sys.path.insert(0, "/workspace/omni")

import torch
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

AP = argparse.ArgumentParser()
AP.add_argument("--check", action="store_true", help="only run the render-vs-dataset check")
AP.add_argument("--tap", type=int, default=24)
AP.add_argument("--ckpt", default="")
AP.add_argument("--episodes", type=int, default=10)
AP.add_argument("--variant", default="orig", choices=["orig", "para1", "swap", "blank", "nonsense"])
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--replan", type=int, default=4)
A = AP.parse_args()

SUITE = "libero_goal"
bd = benchmark.get_benchmark_dict()[SUITE]()
N_TASKS = bd.n_tasks


def make_env(tid, res=256):
    task = bd.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    env.seed(0)
    return env, task


def get_img(obs):
    # Empirically matched against the training dataset: rot180 (MSE 285) beats
    # hflip (4619), vflip (5592), raw (6774). A plain [::-1] gives a MIRRORED
    # scene and would silently collapse success to ~0%.
    return obs["agentview_image"][::-1, ::-1]


def quat2aa(q):
    # robosuite quat is (x,y,z,w)
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    if s < 1e-6:
        return np.zeros(3)
    return np.array([x, y, z]) / s * ang


def get_state(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2aa(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype("float32")


# ---------------- Stage A: does the sim render match the training data? ----------------
if A.check:
    import pandas as pd
    paths = pickle.load(open("/workspace/omni/paths.pkl", "rb"))
    cols = ["observation.images.image", "observation.state", "action",
            "frame_index", "episode_index", "task_index"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    tasks = pickle.load(open("/workspace/omni/tasks.pkl", "rb"))
    report = []
    for tid in range(N_TASKS):
        env, task = make_env(tid)
        env.reset()
        init = bd.get_task_init_states(tid)
        obs = env.set_init_state(init[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        sim_img = get_img(obs).astype(np.float32)
        sim_st = get_state(obs)
        # dataset first frames for the matching task name
        name = task.language.strip().lower()
        ti = [k for k, v in tasks.items() if v.strip().lower() == name]
        row = dict(tid=tid, sim_task=name, matched_task_index=(ti[0] if ti else None))
        if ti:
            sub = d[(d.task_index == ti[0]) & (d.frame_index == 0)]
            if len(sub):
                ds_imgs = np.stack([np.array(Image.open(__import__("io").BytesIO(b["bytes"])).convert("RGB"))
                                    for b in sub["observation.images.image"]]).astype(np.float32)
                mse_f = ((ds_imgs - sim_img[None]) ** 2).mean(axis=(1, 2, 3)).min()
                mse_u = ((ds_imgs - sim_img[::-1][None]) ** 2).mean(axis=(1, 2, 3)).min()
                ds_st = np.stack(sub["observation.state"].values).astype("float32")
                row.update(best_mse_flipped=float(mse_f), best_mse_unflipped=float(mse_u),
                           state_absdiff=float(np.abs(ds_st - sim_st[None]).mean(1).min()),
                           ds_state0=ds_st[0].round(3).tolist(), sim_state=sim_st.round(3).tolist())
        report.append(row)
        print(json.dumps(row), flush=True)
        env.close()
    json.dump(report, open("/workspace/omni/render_check.json", "w"), indent=1)
    ok = [r for r in report if r.get("best_mse_flipped") is not None]
    if ok:
        f = np.mean([r["best_mse_flipped"] for r in ok])
        u = np.mean([r["best_mse_unflipped"] for r in ok])
        print(f"\nMEAN image MSE  flipped={f:.1f}  unflipped={u:.1f}  -> use {'FLIPPED' if f < u else 'UNFLIPPED'}")
        print(f"MEAN state |diff| = {np.mean([r['state_absdiff'] for r in ok]):.4f}")
    sys.exit(0)

# ---------------- Stage B: closed-loop ----------------
from transformers import AutoProcessor, AutoModelForImageTextToText
import variants as V
from train_sweep import Actor, TAPS, mu, sd, smu, ssd  # reuse exact training defs

tasks = pickle.load(open("/workspace/omni/tasks.pkl", "rb"))
MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
proc = AutoProcessor.from_pretrained(MODEL)
proc.image_processor.do_image_splitting = False
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
tap_i = int(np.where(TAPS == A.tap)[0][0])

actor = Actor().cuda().eval()
actor.load_state_dict(torch.load(A.ckpt, map_location="cuda"))
H = 8


def instr_for(ti_, v):
    base = tasks[ti_]
    if v == "orig":  return base
    if v == "para1": return V.PARA[ti_][0]
    if v == "swap":  return tasks[V.swap_partner(ti_)]
    if v == "blank": return V.BLANK
    return V.NONSENSE


@torch.no_grad()
def predict(img, state, text):
    p = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
        add_generation_prompt=True)
    inp = proc(text=[p], images=[[Image.fromarray(img)]], return_tensors="pt").to("cuda")
    hs = vlm(**inp, output_hidden_states=True).hidden_states[A.tap]
    T = hs.shape[1]
    ctx = hs[:, :81] if T >= 81 else torch.nn.functional.pad(hs, (0, 0, 0, 81 - T))[:, :81]
    st = torch.tensor((state - smu) / ssd, dtype=torch.float32, device="cuda")[None]
    x = torch.randn((1, H, 7), device="cuda")
    NS = 10
    for k in range(NS):
        t = torch.full((1,), 1 - k / NS, device="cuda")
        x = x - (1 / NS) * actor(x, t, ctx.float(), st)
    return (x[0].cpu().numpy() * sd + mu)


results = []
for tid in range(N_TASKS):
    env, task = make_env(tid)
    name = task.language.strip().lower()
    ti_ = [k for k, v in tasks.items() if v.strip().lower() == name]
    ti_ = ti_[0] if ti_ else 10 + tid
    text = instr_for(ti_, A.variant)
    init = bd.get_task_init_states(tid)
    succ = 0
    for ep in range(A.episodes):
        env.reset()
        obs = env.set_init_state(init[ep % len(init)])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        done = False
        for step in range(0, A.max_steps, A.replan):
            chunk = predict(get_img(obs), get_state(obs), text)
            for k in range(A.replan):
                a = np.clip(chunk[k], -1, 1)
                obs, r, done, _ = env.step(a.tolist())
                if done: break
            if done: break
        succ += int(done)
    env.close()
    results.append(dict(task=name, tid=tid, success=succ, n=A.episodes))
    print(json.dumps(results[-1]), flush=True)

tot = sum(r["success"] for r in results); n = sum(r["n"] for r in results)
out = dict(tap=A.tap, variant=A.variant, success_rate=tot / n, n=n, per_task=results)
print("\nSR", f"{tot}/{n}", round(100 * tot / n, 1), "%")
json.dump(out, open(f"/workspace/omni/rollout_tap{A.tap}_{A.variant}.json", "w"), indent=1)
