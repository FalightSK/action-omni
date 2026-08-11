"""SmolVLA baseline on LIBERO-Goal, run through OUR harness.

Same benchmark init states, same success predicate, same max_steps as
rollout_qwen.py / rollout_vec.py, so the comparison is suite-matched AND
harness-matched (the earlier pod attempt used lerobot-eval, which was neither).

Checkpoint: HuggingFaceVLA/smolvla_libero  (604.9M params)
  observation.images.image  <- agentview,          rot180  (verified)
  observation.images.image2 <- robot0_eye_in_hand, rot180  (verified)
  observation.state (8)     <- eef_pos + eef_axis_angle + gripper_qpos

NOTE: SmolVLA is a FULL FINE-TUNE whose backbone saw robot data. Ours is a
3.7M head on a frozen backbone that never did. This is a reference point, not
a like-for-like contest.
"""
import argparse, json, os, pickle, sys, time
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("HF_HOME", "E:/hf_cache")

import numpy as np
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

AP = argparse.ArgumentParser()
AP.add_argument("--episodes", type=int, default=10)      # per task
AP.add_argument("--max-steps", type=int, default=400)
AP.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero")
AP.add_argument("--variant", default="orig", choices=["orig", "swap"])
AP.add_argument("--task-start", type=int, default=0)
AP.add_argument("--task-end", type=int, default=10)
A = AP.parse_args()

sys.path.insert(0, HERE)
import variants as V
tasks_map = pickle.load(open(os.path.join(RES, "tasks.pkl"), "rb"))

policy = SmolVLAPolicy.from_pretrained(A.ckpt).cuda().eval()
policy.config.device = "cuda"
# The checkpoint ships policy_preprocessor.json + normalizer stats; use lerobot's
# own pipeline (rename -> batch-dim -> newline -> tokenize -> device -> normalize)
# rather than hand-building the batch, which is how the pod attempt went wrong.
preproc, postproc = make_pre_post_processors(policy.config, pretrained_path=A.ckpt)
print("loaded", A.ckpt, "| n_action_steps", policy.config.n_action_steps, flush=True)
bd = benchmark.get_benchmark_dict()["libero_goal"]()


def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0); ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang


def obs_raw(o, text):
    """UNBATCHED - the preprocessor pipeline adds the batch dimension itself."""
    main = o["agentview_image"][::-1, ::-1].copy()
    wrist = o["robot0_eye_in_hand_image"][::-1, ::-1].copy()
    st = np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                         o["robot0_gripper_qpos"]]).astype("float32")

    def img(a):
        return torch.from_numpy(a.astype("float32") / 255.0).permute(2, 0, 1)

    return {"observation.images.image": img(main),
            "observation.images.image2": img(wrist),
            "observation.state": torch.from_numpy(st),
            "task": text}


rows, t0, nsteps = [], time.time(), 0
for tid in range(A.task_start, min(A.task_end, bd.n_tasks)):
    task = bd.get_task(tid)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=256, camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)
    name = task.language.strip().lower()
    ti_ = next((k for k, v in tasks_map.items() if v.strip().lower() == name), 10 + tid)
    text = tasks_map[ti_] if A.variant == "orig" else tasks_map[V.swap_partner(ti_)]
    inits = bd.get_task_init_states(tid)
    succ = 0
    for e in range(A.episodes):
        env.reset()
        o = env.set_init_state(inits[e % len(inits)])
        for _ in range(10):
            o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        policy.reset()
        done = False
        for _ in range(A.max_steps):
            with torch.no_grad():
                batch = preproc(obs_raw(o, text))
                act = postproc(policy.select_action(batch))
            a = np.asarray(act).reshape(-1)[:7]
            o, r, done, _ = env.step(np.clip(a, -1, 1).tolist())
            nsteps += 1
            if done: break
        succ += int(done)
    env.close()
    rows.append(dict(task=name, tid=tid, success=succ, n=A.episodes))
    el = time.time() - t0
    print(json.dumps(rows[-1]) + f"  [{el:.0f}s {nsteps/el:.1f} steps/s]", flush=True)

tot = sum(r["success"] for r in rows); n = sum(r["n"] for r in rows)
out = dict(model="SmolVLA-450M (full finetune)", ckpt=A.ckpt, variant=A.variant,
           harness="ours (matched)", success_rate=tot / n, n=n,
           wall_s=round(time.time() - t0, 1), per_task=rows)
print(f"\nSR {tot}/{n} = {100*tot/n:.1f}%")
json.dump(out, open(os.path.join(RES, f"smolvla_baseline_{A.variant}_{A.task_start}_{A.task_end}.json"), "w"), indent=1)
print("wrote smolvla_baseline_%s.json" % A.variant)
