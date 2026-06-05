"""
Gating smoke test for the Language Table PyBullet sim on Windows.
Confirms: import works, env constructs, reset/step run, rgb renders, and how the
instruction is encoded (so we know how to recover the string for the Qwen VLM).
"""
from __future__ import annotations
import sys, traceback
from pathlib import Path
import numpy as np

OUTDIR = Path(__file__).parents[1] / "docs" / "experiments" / "language_table"
OUTDIR.mkdir(parents=True, exist_ok=True)

# stub tensorflow (utils_pybullet only needs tf.io.gfile) before any LT import
sys.path.insert(0, str(Path(__file__).parents[1]))
from envs.lt_compat import install_tf_stub
install_tf_stub()

print("== importing language_table ==", flush=True)
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block
print("imports OK", flush=True)

# torch must still work in this env (we run the policy here at eval time)
try:
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[WARN] torch import/cuda failed in this env: {e}", flush=True)

print("== constructing env ==", flush=True)
env = language_table.LanguageTable(
    block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
    reward_factory=block2block.BlockToBlockReward,
    seed=0,
)
print("env OK; action_space:", getattr(env, "action_space", "?"), flush=True)

obs = env.reset()
print("== reset OK; observation keys ==", flush=True)
for k, v in obs.items():
    a = np.asarray(v)
    rng = f"[{a.min()}, {a.max()}]" if a.size else "[]"
    print(f"  {k:42s} shape={str(a.shape):14s} dtype={a.dtype} {rng}", flush=True)

# ---- instruction encoding probe ----
if "instruction" in obs:
    inst = np.asarray(obs["instruction"])
    nz = inst[inst != 0]
    print("\ninstruction nonzero values (first 50):", nz[:50].tolist(), flush=True)
    print("max token id:", int(inst.max()), flush=True)
    try:
        s = bytes(nz.astype(np.uint8)).decode("utf-8", errors="replace")
        print("byte-decode ->", repr(s), flush=True)
    except Exception as e:
        print("byte-decode failed:", e, flush=True)
for attr in ("_instruction", "instruction", "_instruction_str", "_current_instruction"):
    if hasattr(env, attr):
        try:
            print(f"env.{attr} = {getattr(env, attr)!r}", flush=True)
        except Exception:
            pass

# ---- rgb render ----
rgb = np.asarray(obs["rgb"])
print(f"\nrgb {rgb.shape} {rgb.dtype} range [{rgb.min()},{rgb.max()}]", flush=True)
try:
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(OUTDIR / "_smoke_frame.png")
    print("saved", OUTDIR / "_smoke_frame.png", flush=True)
except Exception as e:
    print("frame save failed:", e, flush=True)

# ---- step a few ----
print("\n== stepping ==", flush=True)
for t in range(6):
    obs, reward, done, info = env.step(np.array([0.05, -0.02], dtype=np.float32))
    ik = list(info.keys()) if isinstance(info, dict) else info
    print(f"  step {t}: reward={reward} done={done} info={ik}", flush=True)
    if done:
        break

print("\nSMOKE OK", flush=True)
