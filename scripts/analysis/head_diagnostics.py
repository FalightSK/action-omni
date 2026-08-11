"""
scripts/analysis/head_diagnostics.py
────────────────────────────────────
Inference-only diagnostics on the two trained LIBERO-Goal arms.

Why this exists
───────────────
Chapter 2 produced an inversion: PaliGemma reaches 42.0% closed-loop success
against Pi-0.5's 33.0%, while losing on every offline measure taken so far
(probe action R^2 0.282 vs 0.403; best val loss 0.0528 vs 0.0356). Rolling out
more checkpoints would cost 6-8 GPU-hours and would only describe the gap more
precisely. These six analyses instead ask WHERE it comes from, using forward
passes on the cached embeddings that already exist on disk.

Every analysis reads the SAME held-out frames for both arms. The caches index
the same 63,728 LIBERO-Goal frames in the same order, and train.py splits with
`random_split(..., generator=manual_seed(42))` on an identical length, so the
validation set is frame-for-frame identical across arms. That is what makes the
comparison paired rather than merely matched.

The six
───────
1. adapter_probes    Chapter 1's probes rerun on the ADAPTED tokens the DiT
                     actually cross-attends to, not the raw VLM features. If the
                     adapter narrows or reverses the raw-feature gap, the
                     inversion happens inside the adapter and the raw-feature
                     probes were measuring the wrong space for predicting control.
2. cross_attention   Where the 6 DiT blocks send attention mass across the 272
                     tokens, split image vs text, with entropy. Tests directly
                     whether PaliGemma's head leans on the text tokens to supply
                     the fusion its backbone never performed.
3. text_ablation     Zero the 16 text tokens at inference and measure the
                     velocity-loss increase per arm — how much each policy
                     DEPENDS on reading text itself. Retrain-free.
4. pe_sensitivity    Zero the positional encoding and measure the change in
                     sampled actions. The PE is concatenated at fixed scale
                     (std 0.54) before a shared LayerNorm, so its weight relative
                     to content differs by 1.9x between arms (0.458 vs 0.239).
                     This measures whether that asymmetry is material.
5. phase_task_loss   Velocity loss decomposed by episode phase and by task,
                     connecting the offline objective to the wide per-task SR
                     spread (task 9: Pi-0.5 5% vs PaliGemma 70%).
6. ckpt_ladder       Open-loop action correlation across epoch_0025..final for
                     both arms. A screen — not a resolution — for whether
                     best-val checkpoint selection biased the comparison.

Caveat carried into the report: analyses 3 and 4 evaluate the trained heads
off-distribution. They measure DEPENDENCE on an input, not the performance an
arm could reach if trained without it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "latent_compare"))

from analyze import cv_r2, parallel_analysis, _pcs  # noqa: E402

OUT = ROOT / "asset" / "analysis" / "head_diagnostics"
ARMS = {
    # Pair 1 — PaliGemma family, 272 tokens, read at layer 18.
    "pi05":      ("exp01", "asset/runs/libero/exp01_goal"),
    "paligemma": ("exp02", "asset/runs/libero/exp02_paligemma"),
    # Pair 2 — Qwen3-VL family, 88 tokens, read at layer 16. BOTH are text-blind
    # (Chapter 1 text-visibility 0.0000), so neither has backbone fusion for the
    # head to render redundant. Cross-attention text mass is therefore expected
    # to look like PaliGemma's for both, not like Pi-0.5's for either.
    "groot":     ("exp03", "asset/runs/libero/exp03_groot"),
    "qwen3vl":   ("exp04", "asset/runs/libero/exp04_qwen3vl"),
}
N_FRAMES = 2000        # held-out frames used by every analysis
BATCH    = 50
SEED     = 0
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Shared frame selection
# ──────────────────────────────────────────────────────────────────────────────

def val_indices(n_total: int) -> np.ndarray:
    """Reproduce train.py's validation split exactly.

    train.py: val_len = max(1, int(len * 0.10)); random_split(..., seed 42).
    random_split permutes with the generator and takes the TAIL slice for the
    second output, so the validation indices are the last val_len entries of
    randperm(n). Reproducing it here (rather than sampling fresh frames) keeps
    these diagnostics on data neither arm trained on.
    """
    val_len = max(1, int(n_total * 0.10))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    return perm[n_total - val_len:].numpy()


def stride(rows: np.ndarray, n: int) -> np.ndarray:
    """Evenly spread subsample of a SORTED row index.

    `rows` is kept sorted so HDF5 reads stay sequential, but the cache is
    written in dataset order — all of task 0, then task 1, and so on. A prefix
    slice therefore covers only the first task or two. Striding keeps the read
    pattern cheap while spanning every task and every episode phase.
    """
    if len(rows) <= n:
        return rows
    return rows[np.linspace(0, len(rows) - 1, n).astype(int)]


def frame_labels() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-frame task index and within-episode phase, without decoding images.

    Rebuilds LiberoDataset's index order (sorted files, then demos sorted by
    numeric suffix) reading only `actions.shape` from each demo. That is the
    same order precompute wrote the cache in, so row i here is row i there.
    """
    from configs.registry import get_config
    from data.libero.dataset import task_language

    cfg = get_config("libero", "exp01")
    files = sorted(Path(cfg.dataset_root).glob("*_demo.hdf5"))
    task_idx, phase, names = [], [], []
    for fi, fp in enumerate(files):
        names.append(task_language(fp.name))
        with h5py.File(fp, "r") as f:
            for dname in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
                T = f["data"][dname]["actions"].shape[0]
                task_idx.extend([fi] * T)
                phase.extend((np.arange(T) / max(T - 1, 1)).tolist())
    return np.asarray(task_idx), np.asarray(phase, dtype=np.float32), names


# ──────────────────────────────────────────────────────────────────────────────
# Model / data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_arm(exp: str, run_dir: str, ckpt: str = "best.pt"):
    from configs.registry import get_config
    from models.vla_train import VLATrainModel

    cfg = get_config("libero", exp)
    model = VLATrainModel(cfg)
    ck = torch.load(ROOT / run_dir / "checkpoints" / ckpt,
                    map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{exp} {ckpt}: missing={len(missing)} "
                           f"unexpected={len(unexpected)} — refusing to analyse "
                           f"a partially loaded head")
    return model.to(DEVICE).eval(), cfg, ck.get("epoch", -1)


def read_batch(h5path: Path, rows: np.ndarray):
    """Cache rows -> (embed float32, img_mask bool, actions, states).

    h5py has no bfloat16, so precompute stores the raw bit pattern as uint16.
    It must be reinterpreted with .view(torch.bfloat16), NOT cast — casting the
    integers produces garbage that still has the right shape and dtype.
    """
    order = np.argsort(rows)
    srt = rows[order]
    with h5py.File(h5path, "r") as f:
        e = f["embeddings"][srt]
        m = f["img_masks"][srt].astype(bool)
        a = f["actions"][srt]
        s = f["states"][srt]
    inv = np.argsort(order)
    e, m, a, s = e[inv], m[inv], a[inv], s[inv]
    emb = torch.from_numpy(e.astype(np.uint16)).view(torch.bfloat16).float()
    return (emb, torch.from_numpy(m), torch.from_numpy(a), torch.from_numpy(s))


# ──────────────────────────────────────────────────────────────────────────────
# 1 — adapter-output probes
# ──────────────────────────────────────────────────────────────────────────────

def adapter_probes(model, h5path, rows, actions_all, states_all, phase_all):
    """Probe the adapted tokens the DiT cross-attends to.

    Pools mean over image tokens, over text tokens, and over all — matching the
    pooling Chapter 1 used on raw features, so the two are directly comparable.
    """
    pools = {"image": [], "text": [], "all": []}
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            emb, msk, _, _ = read_batch(h5path, rows[i:i + BATCH])
            emb, msk = emb.to(DEVICE), msk.to(DEVICE)
            _, h = model.adapter(emb, msk, return_tokens=True)   # (B,272,512)
            pools["image"].append((h * msk.unsqueeze(-1)).sum(1)
                                  .div(msk.sum(1, keepdim=True)).cpu().numpy())
            nm = ~msk
            pools["text"].append((h * nm.unsqueeze(-1)).sum(1)
                                 .div(nm.sum(1, keepdim=True)).cpu().numpy())
            pools["all"].append(h.mean(1).cpu().numpy())
    out = {}
    A = actions_all.reshape(len(actions_all), -1)
    S = states_all.reshape(len(states_all), -1)
    for k, v in pools.items():
        X = np.concatenate(v, 0)
        pa = parallel_analysis(X)
        Z = _pcs(X, max(2, pa["n_retained"]))
        out[k] = {
            "n_retained": int(pa["n_retained"]),
            "participation_ratio": float(pa["participation_ratio"]),
            "r2_action": cv_r2(Z, A),
            "r2_state": cv_r2(Z, S),
            "r2_phase": cv_r2(Z, phase_all.astype(np.float64)),
            "raw_dim": int(X.shape[1]),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2 — cross-attention mass and entropy
# ──────────────────────────────────────────────────────────────────────────────

def cross_attention(model, h5path, rows):
    """Per-DiT-block attention mass on image vs text tokens, plus entropy.

    nn.MultiheadAttention is called inside DiTBlock without need_weights, so the
    weights are not exposed. Each cross_attn module's forward is wrapped to
    request and record them, then restored.
    """
    blocks = [b for b in model.decoder.blocks if getattr(b, "has_cross", False)]
    recorded: list[list[torch.Tensor]] = [[] for _ in blocks]
    originals = []

    def wrap(mod, slot):
        orig = mod.forward

        def fwd(q, k, v, *a, **kw):
            kw.pop("need_weights", None)
            kw.pop("average_attn_weights", None)
            o, w = orig(q, k, v, *a, need_weights=True,
                        average_attn_weights=True, **kw)
            recorded[slot].append(w.detach().float().cpu())   # (B, n_q, S)
            return o, w
        mod.forward = fwd
        return orig

    for i, b in enumerate(blocks):
        originals.append(wrap(b.cross_attn, i))

    try:
        with torch.no_grad():
            for i in range(0, len(rows), BATCH):
                emb, msk, _, st = read_batch(h5path, rows[i:i + BATCH])
                model.sample(emb.to(DEVICE), st.to(DEVICE),
                             num_steps=3, img_mask=msk.to(DEVICE))
    finally:
        for b, orig in zip(blocks, originals):
            b.cross_attn.forward = orig

    # img_mask is constant across frames here (256 image then 16 text), so one
    # mask is representative; assert rather than assume.
    _, m0, _, _ = read_batch(h5path, rows[:1])
    img_mask = m0[0].numpy()
    n_img = int(img_mask.sum())

    per_block = []
    for slot, chunks in enumerate(recorded):
        W = torch.cat(chunks, 0)                    # (N*steps, n_q, S)
        w = W.mean(dim=(0, 1)).numpy()              # mean over frames & queries
        w = w / max(w.sum(), 1e-9)
        ent = float(-(w * np.log(w + 1e-12)).sum())
        per_block.append({
            "block": slot,
            "mass_image": float(w[img_mask].sum()),
            "mass_text": float(w[~img_mask].sum()),
            "entropy_nats": ent,
            "entropy_max": float(np.log(len(w))),
            "entropy_norm": ent / float(np.log(len(w))),
            "text_tokens": int((~img_mask).sum()),
            "image_tokens": n_img,
            # per-token, so a fair image/text comparison is not just total mass
            "mass_per_text_token": float(w[~img_mask].sum() / max((~img_mask).sum(), 1)),
            "mass_per_image_token": float(w[img_mask].sum() / max(n_img, 1)),
        })
    total_text = float(np.mean([b["mass_text"] for b in per_block]))
    return {"per_block": per_block, "mean_mass_text": total_text}


# ──────────────────────────────────────────────────────────────────────────────
# 3 / 4 — input ablations
# ──────────────────────────────────────────────────────────────────────────────

def _fixed_noise_loss(model, emb, st, act, msk, seed: int) -> float:
    """Velocity loss with the flow-matching noise and t pinned.

    compute_loss draws t and x0 internally, so two calls differ by sampling
    noise alone. Reseeding before each call makes the ablation comparison
    paired: the only thing that changes is the input being ablated.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        return float(model(emb, st, act, msk))


def text_ablation(model, h5path, rows):
    base, ablated = [], []
    for i in range(0, len(rows), BATCH):
        emb, msk, act, st = read_batch(h5path, rows[i:i + BATCH])
        emb, msk = emb.to(DEVICE), msk.to(DEVICE)
        act, st = act.to(DEVICE), st.to(DEVICE)
        base.append(_fixed_noise_loss(model, emb, st, act, msk, SEED + i))
        e2 = emb.clone()
        e2[~msk] = 0.0                       # zero the 16 text tokens
        ablated.append(_fixed_noise_loss(model, e2, st, act, msk, SEED + i))
    b, a = float(np.mean(base)), float(np.mean(ablated))
    return {"loss_base": b, "loss_text_zeroed": a,
            "delta": a - b, "ratio": a / max(b, 1e-9)}


def pe_sensitivity(model, h5path, rows):
    """Zero the positional encoding, measure the change in sampled actions.

    Patches SpatialAwareMLP.forward to substitute a zero PE while keeping the
    concatenation shape, so the MLP still receives 2048+128 inputs and only the
    positional content changes.
    """
    sp = model.adapter.spatial
    orig = sp.forward

    # pos_dim from the LayerNorm's own width minus the token width — no reliance
    # on an attribute name that could drift.
    pos_dim = sp.mlp[0].normalized_shape[0] - int(model.adapter.lora.A.in_features)

    def fwd_nope(tokens, img_mask, grid_h, grid_w):
        pos = torch.zeros(*tokens.shape[:-1], pos_dim,
                          device=tokens.device, dtype=tokens.dtype)
        return sp.mlp(torch.cat([tokens, pos], dim=-1))

    deltas, scales = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            emb, msk, _, st = read_batch(h5path, rows[i:i + BATCH])
            emb, msk, st = emb.to(DEVICE), msk.to(DEVICE), st.to(DEVICE)
            torch.manual_seed(SEED + i)
            a1 = model.sample(emb, st, num_steps=3, img_mask=msk)
            sp.forward = fwd_nope
            try:
                torch.manual_seed(SEED + i)
                a2 = model.sample(emb, st, num_steps=3, img_mask=msk)
            finally:
                sp.forward = orig
            deltas.append(float((a1 - a2).norm(dim=-1).mean()))
            scales.append(float(a1.norm(dim=-1).mean()))
    d, s = float(np.mean(deltas)), float(np.mean(scales))
    return {"mean_action_shift": d, "mean_action_norm": s,
            "relative_shift": d / max(s, 1e-9)}


# ──────────────────────────────────────────────────────────────────────────────
# 5 — loss by phase and task
# ──────────────────────────────────────────────────────────────────────────────

def phase_task_loss(model, h5path, rows, task_idx, phase, names):
    per_frame = []
    for i in range(0, len(rows), BATCH):
        emb, msk, act, st = read_batch(h5path, rows[i:i + BATCH])
        emb, msk = emb.to(DEVICE), msk.to(DEVICE)
        act, st = act.to(DEVICE), st.to(DEVICE)
        # per-sample loss: run the batch one row at a time would be slow, so use
        # the batch mean but bucket by taking small homogeneous batches instead.
        torch.manual_seed(SEED + i)
        with torch.no_grad():
            for j in range(len(emb)):
                torch.manual_seed(SEED + i + j)
                per_frame.append(float(model(emb[j:j + 1], st[j:j + 1],
                                             act[j:j + 1], msk[j:j + 1])))
    per_frame = np.asarray(per_frame)
    ph = phase[rows][:len(per_frame)]
    ti = task_idx[rows][:len(per_frame)]
    bins = [(0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "late")]
    by_phase = {lab: float(per_frame[(ph >= lo) & (ph < hi)].mean())
                for lo, hi, lab in bins
                if ((ph >= lo) & (ph < hi)).sum() > 0}
    by_task = {}
    for t in np.unique(ti):
        sel = ti == t
        by_task[names[int(t)]] = {"loss": float(per_frame[sel].mean()),
                                  "n": int(sel.sum())}
    return {"overall": float(per_frame.mean()),
            "by_phase": by_phase, "by_task": by_task}


# ──────────────────────────────────────────────────────────────────────────────
# 6 — checkpoint ladder, open-loop correlation
# ──────────────────────────────────────────────────────────────────────────────

def per_dim_error(model, h5path, rows):
    """Open-loop action error per action dimension, absolute and scale-relative.

    The dataset's seven OSC dims differ in scale by up to 17x: translation std
    ~0.37-0.48, rotation std 0.055-0.101, gripper 0.956. We train an unweighted
    MSE on RAW actions (no normalisation, unlike pi0's MEAN_STD and pi05's
    QUANTILES), so the rotation channels contribute ~0.3% of the loss. This
    measures whether that shows up as differentially bad rotation prediction —
    reporting error DIVIDED BY each dim's own std, since a raw MAE comparison
    across dims of different scale means nothing.
    """
    P, G = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            emb, msk, act, st = read_batch(h5path, rows[i:i + BATCH])
            torch.manual_seed(SEED + i)
            out = model.sample(emb.to(DEVICE), st.to(DEVICE), num_steps=3,
                               img_mask=msk.to(DEVICE))
            P.append(out.cpu().numpy())
            G.append(act.numpy())
    P = np.concatenate(P, 0).reshape(-1, 7)
    G = np.concatenate(G, 0).reshape(-1, 7)
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    out = {}
    for i, nm in enumerate(names):
        sd = float(G[:, i].std())
        mae = float(np.abs(P[:, i] - G[:, i]).mean())
        out[nm] = {
            "gt_std": sd,
            "mae": mae,
            "nmae": mae / max(sd, 1e-9),        # 1.0 would be as bad as predicting the mean
            "corr": float(np.corrcoef(P[:, i], G[:, i])[0, 1]),
        }
    return out


def ckpt_ladder(exp, run_dir, h5path, rows):
    ladder = ["epoch_0025.pt", "epoch_0050.pt", "epoch_0075.pt",
              "epoch_0100.pt", "best.pt", "final.pt"]
    res = {}
    for name in ladder:
        p = ROOT / run_dir / "checkpoints" / name
        if not p.exists():
            continue
        try:
            model, _, ep = load_arm(exp, run_dir, name)
        except RuntimeError as e:
            res[name] = {"error": str(e)}
            continue
        preds, gts = [], []
        with torch.no_grad():
            for i in range(0, len(rows), BATCH):
                emb, msk, act, st = read_batch(h5path, rows[i:i + BATCH])
                torch.manual_seed(SEED + i)
                out = model.sample(emb.to(DEVICE), st.to(DEVICE),
                                   num_steps=3, img_mask=msk.to(DEVICE))
                preds.append(out.cpu().numpy().reshape(len(out), -1))
                gts.append(act.numpy().reshape(len(act), -1))
        P = np.concatenate(preds, 0).ravel()
        G = np.concatenate(gts, 0).ravel()
        res[name] = {
            "epoch": int(ep),
            "corr": float(np.corrcoef(P, G)[0, 1]),
            "mae": float(np.abs(P - G).mean()),
        }
        del model
        torch.cuda.empty_cache()
    return res


# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[0/6] frame labels + shared validation subsample")
    task_idx, phase, names = frame_labels()
    with h5py.File(ROOT / ARMS["pi05"][1] / "vlm_embeddings.h5", "r") as f:
        n_total = f["embeddings"].shape[0]
        actions_all_src = f["actions"]
        states_all_src = f["states"]
        vi = val_indices(n_total)
        rows = np.sort(rng.choice(vi, size=min(N_FRAMES, len(vi)), replace=False))
        A = actions_all_src[rows]
        S = states_all_src[rows]
    print(f"      total {n_total:,} | val {len(vi):,} | using {len(rows):,} frames")
    if len(task_idx) != n_total:
        print(f"      [!] label length {len(task_idx):,} != cache {n_total:,} — "
              f"skipping analysis 5")

    sub500, sub800 = stride(rows, 500), stride(rows, 800)

    results: dict = {"n_frames": int(len(rows)), "device": DEVICE,
                     "frame_rows": rows[:20].tolist()}

    for arm, (exp, run_dir) in ARMS.items():
        h5path = ROOT / run_dir / "vlm_embeddings.h5"
        print(f"\n=== {arm} ({exp}) ===")
        model, cfg, ep = load_arm(exp, run_dir)
        r: dict = {"best_epoch": ep}

        print("  [1/7] adapter-output probes")
        r["adapter_probes"] = adapter_probes(model, h5path, rows, A, S,
                                             phase[rows] if len(task_idx) == n_total
                                             else np.zeros(len(rows), np.float32))

        print("  [2/7] cross-attention mass + entropy")
        r["cross_attention"] = cross_attention(model, h5path, sub500)

        print("  [3/7] text-token ablation")
        r["text_ablation"] = text_ablation(model, h5path, sub500)

        print("  [4/7] PE sensitivity")
        r["pe_sensitivity"] = pe_sensitivity(model, h5path, sub500)

        print("  [6/7] per-dimension action error")
        r["per_dim_error"] = per_dim_error(model, h5path, sub500)

        if len(task_idx) == n_total:
            print("  [5/7] loss by phase and task")
            r["phase_task_loss"] = phase_task_loss(model, h5path, sub800,
                                                   task_idx, phase, names)

        del model
        torch.cuda.empty_cache()

        print("  [7/7] checkpoint ladder (open-loop)")
        r["ckpt_ladder"] = ckpt_ladder(exp, run_dir, h5path, sub500)

        results[arm] = r

    (OUT / "head_diagnostics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'head_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
