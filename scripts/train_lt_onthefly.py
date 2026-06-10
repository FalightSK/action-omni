"""
scripts/train_lt_onthefly.py
────────────────────────────
On-the-fly (NO embedding cache) 1-epoch trainer for the balanced Language Table set.

The frozen Qwen forward runs INSIDE the train loop (no 1.1 TB cache). To fit 31.8 GB
RAM, data is streamed SHARD-BY-SHARD from the converted parquet (the converter flushes
at episode boundaries, so action chunks are computable within a shard). Per train
batch: decode JPEGs → frozen-VLM forward (sub-batched at --embed-batch for 12 GB VRAM,
no_grad, padded to img_seq_len) → adapter+DiT flow-matching loss → step.

Usage (point at the balanced dataset; separate output dir from the 5k run):
  python scripts/train_lt_onthefly.py \
      --data-root asset/data/language_table_sim_balanced \
      --out asset/runs/language_table/exp02_balanced \
      --batch 32 --embed-batch 16 --lr-scale 0.5 --epochs 1
"""
from __future__ import annotations
import argparse, glob, io, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)        # train mini-batch (small ⇒ more updates)
    ap.add_argument("--embed-batch", type=int, default=16)  # frozen-VLM forward sub-batch (12 GB VRAM)
    ap.add_argument("--lr-scale", type=float, default=0.5)  # peak-LR scale for the small batch
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--val-frames", type=int, default=1500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=12000)
    args = ap.parse_args()

    os.environ["LT_DATASET_ROOT"] = str(Path(args.data_root).resolve())   # before get_config
    from configs.registry import get_config
    from models.vla import VLAModel
    from models.vla_train import VLATrainModel
    from data.language_table import resize_frame

    cfg = get_config("language_table", "exp01")
    cfg.output_dir = args.out
    device = cfg.get_device()
    S, Hsz, Hh = cfg.img_seq_len, cfg.vlm_hidden_size, cfg.action_horizon
    amean = np.array(cfg.action_mean, np.float32); astd = np.array(cfg.action_std, np.float32)
    smean = np.array(cfg.state_mean, np.float32);  sstd = np.array(cfg.state_std, np.float32)
    iw, ih = cfg.lt_img_w, cfg.lt_img_h
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"on-the-fly train | data={args.data_root} | out={args.out} | device={device}")
    print(f"  batch={args.batch} embed_batch={args.embed_batch} lr_scale={args.lr_scale} epochs={args.epochs}")

    shards = sorted(glob.glob(str(Path(args.data_root) / "data" / "chunk-000" / "file-*.parquet")))
    assert shards, f"no shards under {args.data_root}"
    import pyarrow.parquet as pq
    rows_per = [pq.ParquetFile(f).metadata.num_rows for f in shards]
    if len(shards) >= 2:
        val_shard, train_shards = shards[-1], shards[:-1]
        total_frames = sum(rows_per[:-1])
    else:                                            # smoke: single shard reused
        val_shard, train_shards = shards[0], shards
        total_frames = sum(rows_per)
    steps_per_epoch = total_frames // args.batch
    total_steps = max(1, args.epochs * steps_per_epoch)
    print(f"  {len(shards)} shards, {sum(rows_per):,} frames; train {total_frames:,} ({len(train_shards)} shards), val shard={Path(val_shard).name}")
    print(f"  ~{total_steps:,} optimizer steps over {args.epochs} epoch(s)")

    # ── models ──────────────────────────────────────────────────────────────────
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    model = VLATrainModel(cfg).to(device)
    optim = torch.optim.AdamW([{"params": model.adapter.parameters()},
                               {"params": model.decoder.parameters()}], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[cfg.learning_rate * 0.5 * args.lr_scale, cfg.learning_rate * args.lr_scale],
        total_steps=total_steps, pct_start=min(0.3, cfg.warmup_steps / total_steps),
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)

    def load_shard(path):
        df = pd.read_parquet(path).sort_values("index").reset_index(drop=True)
        return (df["image"].tolist(), np.asarray(df["observation.state"].tolist(), np.float32),
                np.asarray(df["action"].tolist(), np.float32), df["episode_index"].to_numpy(np.int64),
                df["instruction"].tolist())

    def make_batch(idxs, imgs, states, acts, eps, instrs):
        images, sout, aout, texts = [], [], [], []
        n = len(eps)
        for i in idxs:
            images.append(resize_frame(np.asarray(Image.open(io.BytesIO(imgs[i])).convert("RGB")), iw, ih))
            sout.append((states[i] - smean) / (sstd + 1e-8))
            ep = eps[i]; last = acts[i]; chunk = []
            for k in range(Hh):
                fi = i + k
                a = acts[fi] if (fi < n and eps[fi] == ep) else last
                last = a if (fi < n and eps[fi] == ep) else last
                chunk.append((a - amean) / (astd + 1e-8))
            aout.append(np.stack(chunk, 0)); texts.append(instrs[i])
        return (images,
                torch.tensor(np.asarray(sout), dtype=torch.float32, device=device),
                torch.tensor(np.asarray(aout), dtype=torch.float32, device=device), texts)

    @torch.no_grad()
    def vlm_embed(images, texts):
        B = len(images)
        emb = torch.zeros(B, S, Hsz, device=device); msk = torch.zeros(B, S, dtype=torch.bool, device=device)
        for i in range(0, B, args.embed_batch):
            vi = vlm.build_vlm_inputs(images[i:i + args.embed_batch], texts[i:i + args.embed_batch], device)
            tok, im = vlm.encode_vlm(vi); b = tok.shape[0]; sl = min(tok.shape[1], S)
            emb[i:i + b, :sl] = tok[:, :sl].float(); msk[i:i + b, :sl] = im[:, :sl]
        return emb, msk

    # ── fixed val set (subset of the held-out val shard) ────────────────────────
    vi_, vs_, va_, ve_, vt_ = load_shard(val_shard)
    vsel = list(range(min(args.val_frames, len(ve_))))
    val_imgs, val_states, val_actions, val_texts = make_batch(vsel, vi_, vs_, va_, ve_, vt_)
    del vi_, vs_, va_, ve_, vt_

    @torch.no_grad()
    def validate():
        model.eval(); tot = 0.0; nb = 0
        for i in range(0, len(val_imgs), args.batch):
            emb, msk = vlm_embed(val_imgs[i:i + args.batch], val_texts[i:i + args.batch])
            tot += model(emb, val_states[i:i + args.batch], val_actions[i:i + args.batch], msk).item(); nb += 1
        model.train(); return tot / max(1, nb)

    print(f"\n[train] {total_steps:,} steps …", flush=True)
    step = 0; best = float("inf"); run_loss = 0.0; t0 = time.time(); seen = 0
    model.train()
    rng = random.Random(0)
    for epoch in range(1, args.epochs + 1):
        sh = train_shards[:]; rng.shuffle(sh)
        for sp in sh:
            imgs, states, acts, eps, instrs = load_shard(sp)
            order = list(range(len(eps))); rng.shuffle(order)
            for bs in range(0, len(order) - args.batch + 1, args.batch):
                idxs = order[bs:bs + args.batch]
                images, sb, ab, texts = make_batch(idxs, imgs, states, acts, eps, instrs)
                emb, msk = vlm_embed(images, texts)
                optim.zero_grad(set_to_none=True)
                loss = model(emb, sb, ab, msk)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step(); sched.step()
                step += 1; run_loss += loss.item(); seen += len(idxs)
                if step % args.log_every == 0:
                    fps = seen / (time.time() - t0)
                    print(f"  step {step:6d}/{total_steps}  loss={run_loss/args.log_every:.4f}  "
                          f"lr={sched.get_last_lr()[1]:.2e}  {fps:.0f} frames/s  "
                          f"eta {(total_steps-step)*args.batch/max(fps,1)/3600:.1f}h", flush=True)
                    run_loss = 0.0
                if step % args.ckpt_every == 0:
                    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg,
                                "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "last.pt")
                if step % args.val_every == 0:
                    v = validate()
                    print(f"  [val] step {step}: {v:.4f}  {'★ best' if v < best else ''}", flush=True)
                    if v < best:
                        best = v
                        torch.save({"step": step, "val_loss": v, "state_dict": model.state_dict(), "cfg": cfg,
                                    "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "best.pt")
            del imgs, states, acts, eps, instrs
    v = validate()
    print(f"\n[done] final val {v:.4f} (best {min(best, v):.4f})", flush=True)
    if v <= best:
        torch.save({"step": step, "val_loss": v, "state_dict": model.state_dict(), "cfg": cfg,
                    "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "best.pt")
    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg,
                "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "final.pt")
    print(f"checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    main()
