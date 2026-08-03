"""Does the goal-swap arm FOLLOW the swapped instruction, or just break?

0% on the scored task is consistent with both. This distinguishes them.

LIBERO-Goal's 10 tasks share one kitchen scene, so we can run the policy in
task j's env with task k's instruction, and score task k's goal predicate by
transferring the sim state into task k's env each replan step.

  swap_task_success high  -> policy executes the OTHER task (instruction-following)
  swap_task_success ~0    -> policy just fails (the earlier claim was wrong)
"""
import argparse, json, os, pickle, sys, time
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("HF_HOME", "E:/hf_cache")

import numpy as np
import torch
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
from sweep_qwen import Actor, W, DIM, MAXI, H, DA

AP = argparse.ArgumentParser()
AP.add_argument("--tap", type=int, default=12)
AP.add_argument("--mode", default="all")
AP.add_argument("--episodes", type=int, default=10)   # per task
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--replan", type=int, default=8)
A = AP.parse_args()

nz = np.load(os.path.join(RES, "qnorm.npz"))
mu, sd, smu, ssd = nz["mu"], nz["sd"], nz["smu"], nz["ssd"]
IMG_ID = int(np.load(os.path.join(RES, "qwen_meta.npy"))[0])
MODEL = "Qwen/Qwen3.5-0.8B"
proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
vlm = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
actor = Actor().cuda().eval()
actor.load_state_dict(torch.load(os.path.join(RES, f"qck_{A.mode}_tap{A.tap}.pt"),
                                 map_location="cuda"))
tasks_map = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))
bd = benchmark.get_benchmark_dict()["libero_goal"]()

lang2tid = {bd.get_task(i).language.strip().lower(): i for i in range(bd.n_tasks)}


def mk(tid):
    t = bd.get_task(tid)
    return OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    t.problem_folder, t.bddl_file),
        camera_heights=256, camera_widths=256)


def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0); ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang


def pack(o):
    return (o["agentview_image"][::-1, ::-1].copy(),
            np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                            o["robot0_gripper_qpos"]]).astype("float32"))


def check_success(env):
    e = getattr(env, "env", env)
    for attr in ("_check_success", "check_success"):
        if hasattr(e, attr):
            try: return bool(getattr(e, attr)())
            except Exception: pass
    return False


@torch.no_grad()
def policy(img, state, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[txt], images=[Image.fromarray(img)], return_tensors="pt")
    ids = inp["input_ids"][0].tolist()
    inp = {k: v.cuda() for k, v in inp.items()}
    hs = vlm(**inp, output_hidden_states=True).hidden_states[A.tap][0]
    T = hs.shape[0]
    if T < W: hs = torch.nn.functional.pad(hs, (0, 0, 0, W - T))
    hs = hs[:W]
    if A.mode == "all":
        ctx = hs[None]
        kpm = torch.ones(1, W, dtype=torch.bool, device="cuda"); kpm[0, :min(T, W)] = False
    else:
        ipos = [j for j in range(min(T, W)) if ids[j] == IMG_ID]
        start = (ipos[-1] + 1) if ipos else 0
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        sel = list(range(start, min(start + n, W)))[:MAXI]
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
    name = bd.get_task(tid).language.strip().lower()
    ti_j = next((k for k, v in tasks_map.items() if v.strip().lower() == name), 10 + tid)
    ti_k = V.swap_partner(ti_j)
    swap_text = tasks_map[ti_k]
    tid_k = lang2tid.get(swap_text.strip().lower())
    if tid_k is None:
        print("no sim task for", swap_text); continue
    env_j, env_k = mk(tid), mk(tid_k)
    env_j.seed(0); env_k.seed(0); env_k.reset()
    inits = bd.get_task_init_states(tid)
    scored_j, scored_k = 0, 0
    for e in range(A.episodes):
        env_j.reset()
        o = env_j.set_init_state(inits[e % len(inits)])
        for _ in range(10):
            o, _, _, _ = env_j.step([0, 0, 0, 0, 0, 0, -1])
        hit_j = hit_k = False
        for _ in range(0, A.max_steps, A.replan):
            img, stt = pack(o)
            chunk = policy(img, stt, swap_text)
            for kk in range(A.replan):
                o, r, done, _ = env_j.step(np.clip(chunk[kk], -1, 1).tolist())
                if done: hit_j = True
            # score the SWAPPED task by transferring sim state into its env
            try:
                env_k.sim.set_state_from_flattened(env_j.sim.get_state().flatten())
                env_k.sim.forward()
                if check_success(env_k): hit_k = True
            except Exception as ex:
                if e == 0: print("  state-transfer failed:", type(ex).__name__, str(ex)[:70])
            if hit_j or hit_k: break
        scored_j += int(hit_j); scored_k += int(hit_k)
    env_j.close(); env_k.close()
    rows.append(dict(task=name, swap_instruction=swap_text, n=A.episodes,
                     scored_task_success=scored_j, swapped_task_success=scored_k))
    print(json.dumps(rows[-1]) + f"  [{time.time()-t0:.0f}s]", flush=True)

n = sum(r["n"] for r in rows)
sj = sum(r["scored_task_success"] for r in rows)
sk = sum(r["swapped_task_success"] for r in rows)
print(f"\n=== GOAL-SWAP DIAGNOSIS (n={n}) ===")
print(f"  succeeds at the SCORED task (j)   : {sj}/{n} = {100*sj/n:.1f}%")
print(f"  succeeds at the SWAPPED task (k)  : {sk}/{n} = {100*sk/n:.1f}%")
print("  -> FOLLOWS the swapped instruction" if sk > 0.3 * n else
      "  -> does NOT complete the swapped task either (policy is failing, not redirecting)")
json.dump(dict(n=n, scored=sj, swapped=sk, per_task=rows),
          open(os.path.join(RES, "swap_diagnosis.json"), "w"), indent=1)
