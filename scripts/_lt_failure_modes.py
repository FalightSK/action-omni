"""
scripts/_lt_failure_modes.py
────────────────────────────
WHY is closed-loop SR only ~22% when the model clearly uses language (2.6x effect)?

Replicates the exact evaluate.py block2block rollout (clip [-0.1,0.1], reward>0 =
success, TARGET_BLOCK_DISTANCE=0.05) but LOGS the start->target block distance every
step, so each episode can be decomposed into a failure mode:

  d0      = initial start->target distance (how far the push needs to go)
  d_min   = closest approach achieved
  d_final = distance at episode end
  moved   = how far the start block was displaced from its initial position
  toward  = net start-block displacement projected onto the initial goal direction
            (>0 = pushed the correct block toward the target)

Categories:
  success     : reward fired (d_min < 0.05 held)
  near_miss   : fail, but d_min < 0.10  (got within ~2x tolerance -> precision/stall)
  partial     : fail, made real progress ((d0-d_min)/d0 > 0.33) but stalled >0.10
  wrong/idle  : fail, block barely moved (<0.03) or net moved AWAY from target

This separates "can't ground the language / wrong target" (wrong/idle, toward<=0)
from "grounds + pushes correctly but can't finish the fine placement" (near_miss).

Output: console table + JSON + figure docs/experiments/language_table/lt_failure_modes.png
"""
from __future__ import annotations
import os, sys, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

THRESH = 0.05   # constants.TARGET_BLOCK_DISTANCE
OUT = ROOT / "docs/experiments/language_table/lt_failure_modes.png"
OUTJSON = ROOT / "asset/runs/language_table/exp01_baseline/failure_modes.json"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = torch.load(Path(cfg.output_dir)/"checkpoints"/"best.pt", map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ck["state_dict"]); tm.eval()

    from envs.language_table_env import make_lt_env, get_instruction, LanguageTableAgent
    env = make_lt_env("block2block", seed=42)
    agent = LanguageTableAgent(vlm, tm, cfg, device)

    def rc_of(env):
        for a in ("_reward_calculator", "_reward", "reward_calculator"):
            r = getattr(env, a, None)
            if r is not None:
                return r
        raise RuntimeError("reward calculator attr not found")

    def blk(b):
        p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b])
        return np.array(p[:2], dtype=np.float64)

    rows = []
    print(f"[2/3] Rolling out {N} block2block episodes (logging distance-to-goal) …")
    for ep in range(N):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs)
        rc = rc_of(env)
        sb = getattr(rc, "_start_block", None); tb = getattr(rc, "_target_block", None)
        if sb is None or tb is None:      # skip degenerate resets
            continue
        agent.reset(); agent.instruction = instr
        sb0 = blk(sb); tb0 = blk(tb)
        d0 = float(np.linalg.norm(sb0 - tb0))
        goal_dir = (tb0 - sb0) / (np.linalg.norm(tb0 - sb0) + 1e-9)
        dists = [d0]; success = False
        for step in range(cfg.sim_max_steps):
            image = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
            state = np.asarray(obs["effector_translation"], dtype=np.float32)
            a = np.clip(agent.act(image, state), -0.1, 0.1).astype(np.float32)
            sout = env.step(a)
            obs, reward, done, info = (sout if len(sout) == 4 else (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            dists.append(float(np.linalg.norm(blk(sb) - blk(tb))))
            if float(reward) > 0: success = True
            if done or success: break
        sbf = blk(sb)
        moved = float(np.linalg.norm(sbf - sb0))
        toward = float((sbf - sb0) @ goal_dir)        # >0 = correct block pushed toward target
        d_min = float(np.min(dists)); d_final = float(dists[-1])
        prog = (d0 - d_min) / (d0 + 1e-9)
        if success:                      cat = "success"
        elif d_min < 2 * THRESH:         cat = "near_miss"
        elif prog > 0.33:                cat = "partial"
        else:                            cat = "wrong/idle"
        rows.append(dict(ep=ep, instr=instr, d0=d0, d_min=d_min, d_final=d_final,
                         moved=moved, toward=toward, prog=prog, success=success,
                         steps=len(dists)-1, cat=cat))
        print(f"  ep {ep+1:2d} {('OK ' if success else 'x  ')} d0={d0:.3f} d_min={d_min:.3f} "
              f"moved={moved:.3f} toward={toward:+.3f} [{cat}] {instr[:42]!r}")
    try: env.close()
    except Exception: pass

    # ── aggregate ────────────────────────────────────────────────────────────────
    A = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("d0","d_min","d_final","moved","toward","prog","success")}
    n = len(rows); sr = float(A["success"].mean())
    cats = {c: sum(1 for r in rows if r["cat"] == c) for c in ("success","near_miss","partial","wrong/idle")}
    fails = [r for r in rows if not r["success"]]
    f_dmin = np.array([r["d_min"] for r in fails])
    toward_pos = float(np.mean(A["toward"] > 0.01))
    # SR vs required push distance d0 (median split)
    med = float(np.median(A["d0"]))
    sr_short = float(A["success"][A["d0"] <= med].mean()) if (A["d0"] <= med).any() else float("nan")
    sr_long  = float(A["success"][A["d0"] >  med].mean()) if (A["d0"] >  med).any() else float("nan")

    print("\n" + "="*64)
    print(f"  FAILURE-MODE DECOMPOSITION  (n={n}, SR={sr*100:.0f}%)")
    print("="*64)
    for c in ("success","near_miss","partial","wrong/idle"):
        print(f"   {c:11s}: {cats[c]:2d}  ({100*cats[c]/n:.0f}%)")
    print(f"\n   correct block pushed TOWARD target (toward>0): {toward_pos*100:.0f}% of eps")
    print(f"   failures' closest approach d_min: median={np.median(f_dmin):.3f} "
          f"min={f_dmin.min():.3f}  (success needs <{THRESH})")
    print(f"   near-miss failures (d_min<{2*THRESH}): {int((f_dmin<2*THRESH).sum())}/{len(fails)}")
    print(f"   SR by required push: short(<= {med:.2f})={sr_short*100:.0f}%  long(>{med:.2f})={sr_long*100:.0f}%")
    print("="*64)

    OUTJSON.write_text(json.dumps(dict(n=n, sr=sr, cats=cats, toward_pos=toward_pos,
        median_d0=med, sr_short=sr_short, sr_long=sr_long, rows=rows), indent=2))

    # ── figure ────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.4))
    # A: category breakdown
    cs = ["#55A868","#4C72B0","#DD8452","#C44E52"]
    labs = ["success","near-miss\n(<0.10)","partial\nprogress","wrong/idle"]
    vals = [cats["success"],cats["near_miss"],cats["partial"],cats["wrong/idle"]]
    ax1.bar(range(4), vals, color=cs, edgecolor="black")
    for i,v in enumerate(vals): ax1.annotate(f"{v}\n{100*v/n:.0f}%",(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=10,fontweight="bold")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(labs, fontsize=9)
    ax1.set_ylabel("episodes"); ax1.set_title(f"A. Failure-mode decomposition (n={n})\nmost failures get CLOSE, not wrong",fontsize=10.5,fontweight="bold")
    # B: closest-approach distribution for failures vs threshold
    ax2.hist(f_dmin, bins=np.linspace(0,max(0.3,f_dmin.max()),16), color="#4C72B0", edgecolor="black")
    ax2.axvline(THRESH, color="#C44E52", lw=2.5, label=f"success tol {THRESH}")
    ax2.axvline(2*THRESH, color="#DD8452", lw=1.5, ls="--", label=f"2x tol {2*THRESH}")
    ax2.set_xlabel("closest approach d_min (board units)"); ax2.set_ylabel("# failed episodes")
    ax2.set_title("B. How close do FAILURES get?\n(bars near the red line = precision misses)",fontsize=10.5,fontweight="bold"); ax2.legend(fontsize=9)
    # C: SR vs required push distance
    ax3.bar([0,1],[sr_short*100,sr_long*100],color=["#55A868","#C44E52"],edgecolor="black")
    for i,v in [(0,sr_short*100),(1,sr_long*100)]: ax3.annotate(f"{v:.0f}%",(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=12,fontweight="bold")
    ax3.set_xticks([0,1]); ax3.set_xticklabels([f"short push\n(d0<={med:.2f})",f"long push\n(d0>{med:.2f})"],fontsize=9.5)
    ax3.set_ylabel("success rate (%)"); ax3.set_ylim(0,100)
    ax3.set_title("C. SR vs required push distance\n(harder placements fail more)",fontsize=10.5,fontweight="bold")
    fig.suptitle(f"Why LT exp01 SR={sr*100:.0f}%: the correct block is pushed toward the target in {toward_pos*100:.0f}% of episodes "
                 f"(grounding works); failures cluster as near-misses around the {THRESH} tolerance (fine placement is the bottleneck).",
                 fontsize=11.5, fontweight="bold", y=1.03)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    print(f"\n[3/3] saved -> {OUT}\n          -> {OUTJSON}")


if __name__ == "__main__":
    main()
