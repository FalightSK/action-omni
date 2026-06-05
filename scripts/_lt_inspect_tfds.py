"""
One-shot probe of the language_table_sim RLDS/TFDS schema (read from GCS).

Prints: dataset feature spec, episode count, and for the FIRST step of the FIRST
episode -> observation keys/shapes/dtypes, action shape/dtype/range, reward, and
the decoded instruction string (so we know exactly how to write the converter).
"""
from __future__ import annotations
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

GCS = "gs://gresearch/robotics/language_table_sim/0.0.1/"

print("== building from directory ==", flush=True)
b = tfds.builder_from_directory(GCS)
print("\n== FEATURES ==", flush=True)
print(b.info.features, flush=True)
print("\n== SPLITS ==", flush=True)
for name, sp in b.info.splits.items():
    print(f"  {name}: {sp.num_examples} episodes, {sp.num_shards} shards", flush=True)

print("\n== first episode / first step ==", flush=True)
ds = b.as_dataset(split="train").take(1)
for ep in ds:
    print("episode keys:", list(ep.keys()), flush=True)
    steps = ep["steps"]
    n = 0
    for step in steps:
        n += 1
    print("steps in episode 0:", n, flush=True)
    for step in ep["steps"].take(1):
        obs = step["observation"]
        print("\n-- observation --", flush=True)
        for k, v in obs.items():
            a = np.asarray(v)
            rng = f"[{a.min()}, {a.max()}]" if a.size and np.issubdtype(a.dtype, np.number) else ""
            print(f"  {k:30s} shape={str(a.shape):16s} dtype={a.dtype} {rng}", flush=True)
        act = np.asarray(step["action"])
        print(f"\n-- action shape={act.shape} dtype={act.dtype} range=[{act.min()},{act.max()}] vals={act.tolist()}", flush=True)
        for fk in ("reward", "is_first", "is_last", "is_terminal"):
            if fk in step:
                print(f"  {fk} = {np.asarray(step[fk])}", flush=True)
        # decode instruction
        if "instruction" in obs:
            inst = np.asarray(obs["instruction"])
            nz = inst[inst != 0]
            print(f"\ninstruction dtype={inst.dtype} shape={inst.shape} nonzero={nz[:40].tolist()}", flush=True)
            try:
                s = bytes(nz.astype(np.uint8)).decode("utf-8", errors="replace")
                print("byte-decode ->", repr(s), flush=True)
            except Exception as e:
                print("byte-decode failed:", e, flush=True)
        # some LT variants carry the string directly
        for sk in ("instruction_str", "language_instruction"):
            if sk in obs:
                print(f"{sk} ->", np.asarray(obs[sk]), flush=True)
    break

print("\nINSPECT OK", flush=True)
