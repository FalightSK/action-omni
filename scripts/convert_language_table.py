"""
Convert Google `language_table_sim` (RLDS/TFDS on GCS) -> the project's local
format, mirroring the ALOHA/PushT layout so precompute/train/eval work unchanged.

Output layout (under --out):
    data/chunk-000/file-000.parquet, file-001.parquet, ...   # one row per frame
    meta/stats.json          # action + observation.state mean/std/min/max (subset)
    meta/instructions.json    # unique instruction inventory + held-out OOD split
    meta/episodes.json        # per-episode: index, instruction, template, n_steps

Per-frame parquet row:
    index            int     global frame index (0..N-1, write order)
    episode_index    int
    frame_index      int     within-episode step
    observation.state list[2] = effector_translation (current ee xy)
    action           list[2] = stored 2D delta setpoint  (used directly, no integ.)
    reward           float
    instruction      str     decoded per-episode language command
    image            bytes   JPEG (resized to --img_w x --img_h)

Modes:
    --inventory K    scan K episodes, print template distribution + examples, write
                     meta/instructions_inventory.json, write NO frames (fast probe).
    (default)        stream until --episodes episodes are WRITTEN, skipping any whose
                     instruction template is in --holdout (the OOD "new command" set).

Run in the `vla_lt_data` conda env (tfds + tf + pandas + pillow + pyarrow).
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds

GCS = "gs://gresearch/robotics/language_table_sim/0.0.1/"


# ── instruction decode + heuristic template classifier ──────────────────────────
def decode_instruction(arr) -> str:
    a = np.asarray(arr)
    nz = a[a != 0]
    if nz.size == 0:
        return ""
    try:
        return bytes(nz.astype(np.uint8)).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


# board-location words that signal an ABSOLUTE-location command (not block-relative)
_ABS_LOC = ("top left", "top right", "bottom left", "bottom right",
            "upper left", "upper right", "lower left", "lower right",
            "top side", "bottom side", "left side", "right side",
            "center", "centre", "middle", "corner", "edge")
_REL_BLOCK = ("to the left of", "to the right of", "above", "below",
              "next to", "near", "close to", "in front of", "behind", "by the")


def classify_instruction(s: str) -> str:
    """Coarse command-type tag used to define the held-out OOD split.
    Heuristic; refined after the inventory pass shows the real strings."""
    t = s.lower()
    if not t:
        return "empty"
    if "separate" in t:
        return "separate"
    if "point" in t:
        return "point"
    if "in between" in t or " between " in t:
        return "between"
    if any(w in t for w in _REL_BLOCK):
        return "relative_to_block"
    if any(w in t for w in _ABS_LOC):
        return "absolute_location"
    if any(w in t for w in ("push", "move", "slide", "put", "place", "bring")):
        return "block2block"
    return "other"


# ── helpers ─────────────────────────────────────────────────────────────────────
def episodes_iter(builder, take: int | None, sl: str = ""):
    split = f"train[{sl}]" if sl else "train"          # sl like "0:30000" for parallel shards
    ds = builder.as_dataset(split=split)
    if take is not None:
        ds = ds.take(take)
    return ds


def first_instruction(ep) -> str:
    for step in ep["steps"].take(1):
        return decode_instruction(step["observation"]["instruction"].numpy())
    return ""


def jpeg_bytes(rgb: np.ndarray, w: int, h: int, q: int) -> bytes:
    im = Image.fromarray(np.ascontiguousarray(rgb)).resize((w, h))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


# ── inventory mode ────────────────────────────────────────────────────────────────
def run_inventory(builder, k: int, out: Path) -> None:
    print(f"== INVENTORY: scanning {k} episodes ==", flush=True)
    tmpl_counter: Counter = Counter()
    examples: dict[str, list[str]] = {}
    inst_counter: Counter = Counter()
    ep_lens: list[int] = []
    act_min = np.array([np.inf, np.inf]); act_max = np.array([-np.inf, -np.inf])
    st_min = np.array([np.inf, np.inf]); st_max = np.array([-np.inf, -np.inf])
    t0 = time.time()
    for i, ep in enumerate(episodes_iter(builder, k)):
        inst = first_instruction(ep)
        tmpl = classify_instruction(inst)
        tmpl_counter[tmpl] += 1
        inst_counter[inst] += 1
        examples.setdefault(tmpl, [])
        if len(examples[tmpl]) < 6 and inst not in examples[tmpl]:
            examples[tmpl].append(inst)
        n = 0
        for step in ep["steps"]:
            n += 1
            a = step["action"].numpy()
            s = step["observation"]["effector_translation"].numpy()
            act_min = np.minimum(act_min, a); act_max = np.maximum(act_max, a)
            st_min = np.minimum(st_min, s); st_max = np.maximum(st_max, s)
        ep_lens.append(n)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{k} episodes  ({time.time()-t0:.0f}s)", flush=True)

    ep_lens_arr = np.array(ep_lens)
    print("\n== TEMPLATE DISTRIBUTION ==", flush=True)
    for tmpl, c in tmpl_counter.most_common():
        print(f"  {tmpl:20s} {c:5d}  ({100*c/len(ep_lens):.1f}%)", flush=True)
        for ex in examples.get(tmpl, [])[:4]:
            print(f"       e.g. {ex!r}", flush=True)
    print(f"\n== EPISODE LENGTHS ==  n={len(ep_lens)}  min={ep_lens_arr.min()} "
          f"max={ep_lens_arr.max()} mean={ep_lens_arr.mean():.1f} "
          f"median={np.median(ep_lens_arr):.0f}", flush=True)
    print(f"  length percentiles 10/25/50/75/90: "
          f"{np.percentile(ep_lens_arr,[10,25,50,75,90]).round(1).tolist()}", flush=True)
    print(f"\n== ACTION range == min={act_min.tolist()} max={act_max.tolist()}", flush=True)
    print(f"== STATE  range == min={st_min.tolist()} max={st_max.tolist()}", flush=True)
    print(f"\nunique instructions: {len(inst_counter)}", flush=True)
    print("top 15 instructions:", flush=True)
    for inst, c in inst_counter.most_common(15):
        print(f"  {c:4d}  {inst!r}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "instructions_inventory.json").write_text(json.dumps({
        "scanned_episodes": int(len(ep_lens)),
        "template_counts": dict(tmpl_counter),
        "template_examples": {k2: v for k2, v in examples.items()},
        "n_unique_instructions": int(len(inst_counter)),
        "episode_len": {"min": int(ep_lens_arr.min()), "max": int(ep_lens_arr.max()),
                        "mean": float(ep_lens_arr.mean()),
                        "median": float(np.median(ep_lens_arr))},
        "action_range": {"min": act_min.tolist(), "max": act_max.tolist()},
        "state_range": {"min": st_min.tolist(), "max": st_max.tolist()},
        "top_instructions": inst_counter.most_common(50),
    }, indent=2))
    print(f"\nwrote {out/'instructions_inventory.json'}\nINVENTORY OK", flush=True)


# ── convert mode ──────────────────────────────────────────────────────────────────
def run_convert(builder, args) -> None:
    out = Path(args.out)
    data_dir = out / "data" / "chunk-000"
    meta_dir = out / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    holdout = set(t.strip() for t in args.holdout.split(",") if t.strip())
    balance = {}
    for kv in args.balance.split(","):
        kv = kv.strip()
        if ":" in kv:
            t, n = kv.rsplit(":", 1); balance[t.strip()] = int(n)
    task_counts: Counter = Counter()
    print(f"== CONVERT: target {args.episodes} eps  holdout={sorted(holdout)}  "
          f"balance={balance or None}  slice={args.slice or 'all'}  img={args.img_w}x{args.img_h} ==", flush=True)

    rows: list[dict] = []
    episodes_meta: list[dict] = []
    shard_id = 0
    gidx = 0                        # global frame index
    written_eps = 0
    skipped_holdout = 0
    # running stats
    asum = np.zeros(2); asq = np.zeros(2); ssum = np.zeros(2); ssq = np.zeros(2)
    amin = np.array([np.inf, np.inf]); amax = np.array([-np.inf, -np.inf])
    smin = np.array([np.inf, np.inf]); smax = np.array([-np.inf, -np.inf])
    nframes = 0
    t0 = time.time()

    def flush_shard():
        nonlocal rows, shard_id
        if not rows:
            return
        path = data_dir / f"file-{shard_id:03d}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"  wrote {path.name}: {len(rows)} rows  "
              f"({written_eps} eps, {time.time()-t0:.0f}s)", flush=True)
        rows = []
        shard_id += 1

    for ep in episodes_iter(builder, None, args.slice):
        inst = first_instruction(ep)
        tmpl = classify_instruction(inst)
        if balance:
            if tmpl not in balance or task_counts[tmpl] >= balance[tmpl]:
                skipped_holdout += 1
                continue
        elif tmpl in holdout:
            skipped_holdout += 1
            continue
        ep_index = written_eps
        n = 0
        for step in ep["steps"]:
            obs = step["observation"]
            a = np.asarray(step["action"].numpy(), dtype=np.float32)
            s = np.asarray(obs["effector_translation"].numpy(), dtype=np.float32)
            rgb = obs["rgb"].numpy()
            rows.append({
                "index": gidx,
                "episode_index": ep_index,
                "frame_index": n,
                "observation.state": s.tolist(),
                "action": a.tolist(),
                "reward": float(step["reward"].numpy()),
                "instruction": inst,
                "image": jpeg_bytes(rgb, args.img_w, args.img_h, args.jpeg_quality),
            })
            asum += a; asq += a * a; ssum += s; ssq += s * s
            amin = np.minimum(amin, a); amax = np.maximum(amax, a)
            smin = np.minimum(smin, s); smax = np.maximum(smax, s)
            gidx += 1; n += 1; nframes += 1
        episodes_meta.append({"episode_index": ep_index, "instruction": inst,
                              "template": tmpl, "n_steps": n})
        written_eps += 1
        if balance:
            task_counts[tmpl] += 1
        if len(rows) >= args.shard_frames:
            flush_shard()
        if balance:
            if (written_eps % 500 == 0):
                print(f"  balance progress: {dict(task_counts)}  ({time.time()-t0:.0f}s)", flush=True)
            if all(task_counts[t] >= c for t, c in balance.items()):
                break
        elif written_eps >= args.episodes:
            break
    flush_shard()

    # ── stats.json (lerobot-compatible keys read by the config __post_init__) ──
    amean = asum / nframes; astd = np.sqrt(np.maximum(asq / nframes - amean**2, 1e-12))
    smean = ssum / nframes; sstd = np.sqrt(np.maximum(ssq / nframes - smean**2, 1e-12))
    stats = {
        "action": {"mean": amean.tolist(), "std": astd.tolist(),
                   "min": amin.tolist(), "max": amax.tolist()},
        "observation.state": {"mean": smean.tolist(), "std": sstd.tolist(),
                              "min": smin.tolist(), "max": smax.tolist()},
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    tmpl_counts = Counter(e["template"] for e in episodes_meta)
    insts = Counter(e["instruction"] for e in episodes_meta)
    (meta_dir / "instructions.json").write_text(json.dumps({
        "n_episodes": written_eps, "n_frames": nframes,
        "skipped_holdout": skipped_holdout, "holdout_templates": sorted(holdout),
        "template_counts": dict(tmpl_counts),
        "n_unique_instructions": len(insts),
        "top_instructions": insts.most_common(50),
    }, indent=2))
    (meta_dir / "episodes.json").write_text(json.dumps(episodes_meta, indent=2))

    print(f"\n== DONE == {written_eps} episodes, {nframes} frames "
          f"({shard_id} shards); skipped {skipped_holdout} held-out", flush=True)
    print(f"action mean={amean.round(5).tolist()} std={astd.round(5).tolist()}", flush=True)
    print(f"state  mean={smean.round(5).tolist()} std={sstd.round(5).tolist()}", flush=True)
    print(f"wrote stats.json / instructions.json / episodes.json -> {meta_dir}", flush=True)
    print("CONVERT OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "asset" / "data" / "language_table_sim"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--inventory", type=int, default=0, help="if >0: inventory-only over N episodes")
    ap.add_argument("--holdout", default="", help="comma-list of templates to EXCLUDE (OOD split)")
    ap.add_argument("--slice", default="", help="TFDS split slice e.g. '0:30000' (parallel shards)")
    ap.add_argument("--balance", default="", help="per-task caps e.g. 'block2block:24136,absolute_location:24136,relative_to_block:24136'")
    ap.add_argument("--img_w", type=int, default=320)
    ap.add_argument("--img_h", type=int, default=180)
    ap.add_argument("--shard_frames", type=int, default=15000)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    print(f"building from {GCS}", flush=True)
    builder = tfds.builder_from_directory(GCS)
    if args.inventory > 0:
        run_inventory(builder, args.inventory, Path(args.out) / "meta")
    else:
        run_convert(builder, args)


if __name__ == "__main__":
    main()
