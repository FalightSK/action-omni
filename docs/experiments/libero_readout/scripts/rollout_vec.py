"""Vectorised closed-loop LIBERO rollout.

K simulator worker processes step in lockstep; the VLM + actor run ONE BATCHED
forward per replan cycle.  No caching anywhere - observations depend on the
policy's own actions, so every forward is live.

Workers are forked BEFORE CUDA is initialised in the parent (they need MuJoCo
+ their own EGL context, never the GPU).
"""
import argparse, json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, "/workspace/omni")

AP = argparse.ArgumentParser()
AP.add_argument("--tap", type=int, required=True)
AP.add_argument("--ckpt", required=True)
AP.add_argument("--mode", default="all", choices=["all", "instr"])
AP.add_argument("--variant", default="orig",
                choices=["orig", "para1", "para2", "para3", "swap", "blank", "nonsense"])
AP.add_argument("--episodes", type=int, default=50, help="episodes per task")
AP.add_argument("--envs", type=int, default=16)
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--replan", type=int, default=8)
AP.add_argument("--out", default="")
A = AP.parse_args()

SUITE = "libero_goal"


# ---------------------------------------------------------------- worker
def worker(conn, tid):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bd = benchmark.get_benchmark_dict()[SUITE]()
    task = bd.get_task(tid)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=256, camera_widths=256)
    env.seed(0)
    inits = bd.get_task_init_states(tid)

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

    conn.send(("ready", task.language))
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            env.close(); conn.close(); return
        if cmd == "reset":
            env.reset()
            o = env.set_init_state(inits[payload % len(inits)])
            for _ in range(10):
                o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            conn.send(pack(o))
        elif cmd == "step":
            done = False
            for a in payload:                       # execute the whole chunk
                o, r, done, _ = env.step(np.clip(a, -1, 1).tolist())
                if done:
                    break
            conn.send((pack(o), bool(done)))


if __name__ == "__main__":
    mp.set_start_method("fork")
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()[SUITE]()
    N_TASKS = bd.n_tasks
    tasks_map = pickle.load(open("/workspace/omni/tasks.pkl", "rb"))
    import variants as V

    def instr_for(ti_, v):
        if v == "orig":  return tasks_map[ti_]
        if v.startswith("para"): return V.PARA[ti_][int(v[-1]) - 1]
        if v == "swap":  return tasks_map[V.swap_partner(ti_)]
        if v == "blank": return V.BLANK
        return V.NONSENSE

    results, t_start = [], time.time()
    total_steps = 0

    # ---- workers first (pre-CUDA fork), then the model ----
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from sweep2 import Actor, TAPS, mu, sd, smu, ssd, MAXI, W

    MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.do_image_splitting = False
    proc.tokenizer.padding_side = "right"
    vlm = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()
    IMG = vlm.config.image_token_id
    tok = proc.tokenizer
    EOU = tok.convert_tokens_to_ids("<end_of_utterance>")
    FAKE = tok.convert_tokens_to_ids("<fake_token_around_image>")
    tap_i = int(np.where(TAPS == A.tap)[0][0])
    actor = Actor().cuda().eval()
    actor.load_state_dict(torch.load(A.ckpt, map_location="cuda"))
    H = 8

    @torch.no_grad()
    def policy(imgs, states, text):
        pr = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True)
        inp = proc(text=[pr] * len(imgs), images=[[Image.fromarray(i)] for i in imgs],
                   return_tensors="pt", padding=True).to("cuda")
        hs = vlm(**inp, output_hidden_states=True).hidden_states[A.tap]
        B, T, D = hs.shape
        h = hs[:, :W] if T >= W else torch.nn.functional.pad(hs, (0, 0, 0, W - T))[:, :W]
        ids = inp["input_ids"]; am = inp["attention_mask"].bool()
        if A.mode == "all":
            kpm = torch.ones(B, W, dtype=torch.bool, device="cuda")
            n = min(T, W); kpm[:, :n] = ~am[:, :n]
            ctx = h
        else:
            ctx = torch.zeros(B, MAXI, D, device="cuda", dtype=h.dtype)
            kpm = torch.ones(B, MAXI, dtype=torch.bool, device="cuda")
            for b in range(B):
                row = ids[b].tolist()
                try:
                    a_ = len(row) - 1 - row[::-1].index(FAKE); e_ = row.index(EOU)
                    w = [j for j in range(a_ + 1, min(e_, W))][:MAXI]
                except ValueError:
                    w = []
                if w:
                    ctx[b, :len(w)] = h[b, w]; kpm[b, :len(w)] = False
        st = torch.tensor((np.stack(states) - smu) / ssd,
                          dtype=torch.float32, device="cuda")
        x = torch.randn((B, H, 7), device="cuda")
        NS = 10
        for k in range(NS):
            t = torch.full((B,), 1 - k / NS, device="cuda")
            x = x - (1 / NS) * actor(x, t, ctx.float(), st, kpm)
        return x.cpu().numpy() * sd + mu

    for tid in range(N_TASKS):
        K = min(A.envs, A.episodes)
        conns, procs = [], []
        for _ in range(K):
            pc, cc = mp.Pipe()
            p = mp.Process(target=worker, args=(cc, tid), daemon=True)
            p.start(); conns.append(pc); procs.append(p)
        lang = None
        for c in conns:
            _, lang = c.recv()
        ti_ = next((k for k, v in tasks_map.items()
                    if v.strip().lower() == lang.strip().lower()), 10 + tid)
        text = instr_for(ti_, A.variant)

        succ, done_eps, ep_cursor = 0, 0, 0
        while done_eps < A.episodes:
            batch = min(K, A.episodes - done_eps)
            for i in range(batch):
                conns[i].send(("reset", ep_cursor + i))
            obs = [conns[i].recv() for i in range(batch)]
            alive = list(range(batch)); finished = [False] * batch
            for _ in range(0, A.max_steps, A.replan):
                if not alive:
                    break
                imgs = [obs[i][0] for i in alive]
                sts = [obs[i][1] for i in alive]
                chunks = policy(imgs, sts, text)
                total_steps += len(alive) * A.replan
                for n, i in enumerate(alive):
                    conns[i].send(("step", chunks[n][:A.replan]))
                nxt = []
                for i in alive:
                    o, dn = conns[i].recv()
                    obs[i] = o
                    if dn:
                        finished[i] = True
                    else:
                        nxt.append(i)
                alive = nxt
            succ += sum(finished)
            done_eps += batch; ep_cursor += batch
        for c in conns:
            c.send(("close", None))
        for p in procs:
            p.join(timeout=10)
            if p.is_alive(): p.terminate()
        results.append(dict(task=lang, tid=tid, success=succ, n=A.episodes))
        el = time.time() - t_start
        print(json.dumps(results[-1]) + f"  [{el:.0f}s, {total_steps/el:.0f} env-steps/s]",
              flush=True)

    tot = sum(r["success"] for r in results); n = sum(r["n"] for r in results)
    el = time.time() - t_start
    out = dict(tap=A.tap, mode=A.mode, variant=A.variant, success_rate=tot / n, n=n,
               envs=A.envs, wall_s=round(el, 1),
               env_steps_per_s=round(total_steps / el, 1), per_task=results)
    print(f"\nSR {tot}/{n} = {100*tot/n:.1f}%   {el:.0f}s   {total_steps/el:.0f} env-steps/s")
    fn = A.out or f"vroll_{A.mode}_tap{A.tap}_{A.variant}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("wrote", fn)
