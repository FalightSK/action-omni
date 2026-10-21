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
import argparse, glob, io, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_log import log_event, write_json_atomic

_ACTIVE_PROGRESS: dict = {}


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
    if min(args.batch, args.embed_batch, args.epochs, args.val_frames, args.log_every,
           args.ckpt_every, args.val_every) < 1:
        ap.error("batch sizes, epochs, validation frames, and logging/checkpoint intervals must be positive")
    if args.lr_scale <= 0:
        ap.error("lr-scale must be positive")

    run_started = time.perf_counter()
    log_event("online_training.started", dataset="language_table", data_root=args.data_root,
              output_dir=args.out, batch_size=args.batch, embed_batch_size=args.embed_batch,
              epochs=args.epochs, log_every_steps=args.log_every)
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
    progress_path = Path(cfg.output_dir) / "progress.json"
    _ACTIVE_PROGRESS.update(path=progress_path)
    progress = {"status": "starting", "stage": "startup", "dataset": "language_table",
                "started_utc": datetime.now(timezone.utc).isoformat()}

    def update_progress(stage: str, **fields) -> None:
        progress.update(status="running", stage=stage,
                        updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json_atomic(progress_path, progress)

    update_progress("data_scan")
    print(f"on-the-fly train | data={args.data_root} | out={args.out} | device={device}")
    print(f"  batch={args.batch} embed_batch={args.embed_batch} lr_scale={args.lr_scale} epochs={args.epochs}")

    scan_started = time.perf_counter()
    log_event("data_collection.scan_started", dataset="language_table", root=args.data_root)
    shards = sorted(glob.glob(str(Path(args.data_root) / "data" / "chunk-000" / "file-*.parquet")))
    assert shards, f"no shards under {args.data_root}"
    import pyarrow.parquet as pq
    rows_per = []
    for index, shard in enumerate(shards, start=1):
        shard_started = time.perf_counter()
        rows = pq.ParquetFile(shard).metadata.num_rows
        rows_per.append(rows)
        log_event("data_collection.shard_indexed", shard=Path(shard).name,
                  shard_index=index, shard_count=len(shards), rows=rows,
                  bytes=Path(shard).stat().st_size,
                  elapsed_s=round(time.perf_counter() - shard_started, 3))
    if len(shards) >= 2:
        val_shard, train_shards = shards[-1], shards[:-1]
        total_frames = sum(rows_per[:-1])
    else:                                            # smoke: single shard reused
        val_shard, train_shards = shards[0], shards
        total_frames = sum(rows_per)
    train_rows = rows_per[:-1] if len(shards) >= 2 else rows_per
    steps_per_epoch = sum(rows // args.batch for rows in train_rows)
    if steps_per_epoch < 1:
        raise ValueError("Training shards do not contain a full batch")
    total_steps = max(1, args.epochs * steps_per_epoch)
    log_event("data_collection.scan_complete", shards=len(shards), frames=sum(rows_per),
              train_frames=total_frames, train_steps_per_epoch=steps_per_epoch,
              validation_shard=Path(val_shard).name,
              elapsed_s=round(time.perf_counter() - scan_started, 3))
    print(f"  {len(shards)} shards, {sum(rows_per):,} frames; train {total_frames:,} ({len(train_shards)} shards), val shard={Path(val_shard).name}")
    print(f"  ~{total_steps:,} optimizer steps over {args.epochs} epoch(s)")

    # ── models ──────────────────────────────────────────────────────────────────
    model_started = time.perf_counter()
    log_event("online_training.model_load_started", backbone=cfg.vlm_backbone, device=str(device))
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    model = VLATrainModel(cfg).to(device)
    log_event("online_training.model_load_complete", backbone=cfg.vlm_backbone,
              frozen_backbone_parameters=sum(p.numel() for p in vlm.vlm.parameters()),
              trainable_adapter_parameters=sum(p.numel() for p in model.adapter.parameters() if p.requires_grad),
              trainable_decoder_parameters=sum(p.numel() for p in model.decoder.parameters() if p.requires_grad),
              elapsed_s=round(time.perf_counter() - model_started, 3))
    optim = torch.optim.AdamW([{"params": model.adapter.parameters()},
                               {"params": model.decoder.parameters()}], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[cfg.learning_rate * 0.5 * args.lr_scale, cfg.learning_rate * args.lr_scale],
        total_steps=total_steps, pct_start=min(0.3, cfg.warmup_steps / total_steps),
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)

    def load_shard(path, split: str):
        started = time.perf_counter()
        log_event("data_collection.shard_load_started", split=split, shard=Path(path).name)
        df = pd.read_parquet(path).sort_values("index").reset_index(drop=True)
        result = (df["image"].tolist(), np.asarray(df["observation.state"].tolist(), np.float32),
                  np.asarray(df["action"].tolist(), np.float32), df["episode_index"].to_numpy(np.int64),
                  df["instruction"].tolist())
        log_event("data_collection.shard_load_complete", split=split, shard=Path(path).name,
                  frames=len(df), elapsed_s=round(time.perf_counter() - started, 3))
        return result

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
        started = time.perf_counter()
        encoder_s = 0.0
        B = len(images)
        emb = torch.zeros(B, S, Hsz, device=device); msk = torch.zeros(B, S, dtype=torch.bool, device=device)
        for i in range(0, B, args.embed_batch):
            vi = vlm.build_vlm_inputs(images[i:i + args.embed_batch], texts[i:i + args.embed_batch], device)
            tok, im = vlm.encode_vlm(vi); b = tok.shape[0]; sl = min(tok.shape[1], S)
            emb[i:i + b, :sl] = tok[:, :sl].float(); msk[i:i + b, :sl] = im[:, :sl]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_s = time.perf_counter() - started
        return emb, msk, encoder_s

    # ── fixed val set (subset of the held-out val shard) ────────────────────────
    val_started = time.perf_counter()
    log_event("data_collection.validation_load_started", shard=Path(val_shard).name,
              requested_frames=args.val_frames)
    vi_, vs_, va_, ve_, vt_ = load_shard(val_shard, "validation")
    vsel = list(range(min(args.val_frames, len(ve_))))
    if not vsel:
        raise ValueError(f"Validation shard {val_shard} contains no frames")
    val_imgs, val_states, val_actions, val_texts = make_batch(vsel, vi_, vs_, va_, ve_, vt_)
    del vi_, vs_, va_, ve_, vt_
    log_event("data_collection.validation_ready", frames=len(val_imgs),
              elapsed_s=round(time.perf_counter() - val_started, 3))
    update_progress("model_initialization", validation_frames=len(val_imgs), total_steps=total_steps)

    @torch.no_grad()
    def validate(reason="periodic"):
        validation_started = time.perf_counter()
        log_event("online_training.validation_started", reason=reason, step=step,
                  batches=(len(val_imgs) + args.batch - 1) // args.batch)
        update_progress("validation", step=step, total_steps=total_steps, reason=reason)
        model.eval(); tot = 0.0; nb = 0; encoder_total = 0.0
        for i in range(0, len(val_imgs), args.batch):
            emb, msk, encoder_s = vlm_embed(val_imgs[i:i + args.batch], val_texts[i:i + args.batch])
            encoder_total += encoder_s
            tot += model(emb, val_states[i:i + args.batch], val_actions[i:i + args.batch], msk).item(); nb += 1
        model.train()
        result = tot / max(1, nb)
        log_event("online_training.validation_complete", reason=reason, step=step,
                  validation_loss=round(result, 6), encoder_s=round(encoder_total, 3),
                  elapsed_s=round(time.perf_counter() - validation_started, 3))
        return result

    print(f"\n[train] {total_steps:,} steps …", flush=True)
    step = 0; best = float("inf"); run_loss = 0.0; t0 = time.perf_counter(); seen = 0
    window_started = t0
    window_steps = 0
    window_prepare_s = 0.0
    window_encoder_s = 0.0
    window_update_s = 0.0
    window_loss = 0.0
    window_grad_norm = 0.0
    model.train()
    rng = random.Random(0)
    for epoch in range(1, args.epochs + 1):
        log_event("online_training.epoch_started", epoch=epoch, total_epochs=args.epochs,
                  optimizer_step=step, total_steps=total_steps)
        update_progress("training", epoch=epoch, total_epochs=args.epochs,
                        optimizer_step=step, total_steps=total_steps)
        sh = train_shards[:]; rng.shuffle(sh)
        for sp in sh:
            shard_started = time.perf_counter()
            shard_step_start = step
            imgs, states, acts, eps, instrs = load_shard(sp, "train")
            order = list(range(len(eps))); rng.shuffle(order)
            for bs in range(0, len(order) - args.batch + 1, args.batch):
                idxs = order[bs:bs + args.batch]
                prepare_started = time.perf_counter()
                images, sb, ab, texts = make_batch(idxs, imgs, states, acts, eps, instrs)
                prepare_s = time.perf_counter() - prepare_started
                encoder_started = time.perf_counter()
                emb, msk, encoder_s = vlm_embed(images, texts)
                encoder_wall_s = time.perf_counter() - encoder_started
                update_started = time.perf_counter()
                optim.zero_grad(set_to_none=True)
                loss = model(emb, sb, ab, msk)
                loss.backward(); grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step(); sched.step()
                loss_value = float(loss.detach())
                grad_norm_value = float(grad_norm)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                update_s = time.perf_counter() - update_started
                step += 1; run_loss += loss_value; seen += len(idxs)
                window_steps += 1
                window_prepare_s += prepare_s
                window_encoder_s += encoder_wall_s
                window_update_s += update_s
                window_loss += loss_value
                window_grad_norm += grad_norm_value
                if step % args.log_every == 0 or step == total_steps:
                    elapsed = time.perf_counter() - window_started
                    updates_per_s = window_steps / max(elapsed, 1e-9)
                    fps = seen / max(time.perf_counter() - t0, 1e-9)
                    log_event("online_training.step_progress", epoch=epoch, optimizer_step=step,
                              total_optimizer_steps=total_steps, window_updates=window_steps,
                              mean_loss=round(window_loss / max(window_steps, 1), 6),
                              mean_grad_norm=round(window_grad_norm / max(window_steps, 1), 6),
                              learning_rate=sched.get_last_lr()[1],
                              mean_data_prepare_s=round(window_prepare_s / max(window_steps, 1), 4),
                              mean_online_vlm_s=round(window_encoder_s / max(window_steps, 1), 4),
                              mean_update_s=round(window_update_s / max(window_steps, 1), 4),
                              frames_per_s=round(fps, 2), updates_per_s=round(updates_per_s, 3),
                              eta_s=round((total_steps - step) / max(updates_per_s, 1e-9), 1),
                              elapsed_s=round(time.perf_counter() - run_started, 1))
                    update_progress("training", epoch=epoch, optimizer_step=step,
                                    total_steps=total_steps, latest_loss=round(window_loss / max(window_steps, 1), 6),
                                    updates_per_s=round(updates_per_s, 3),
                                    eta_s=round((total_steps - step) / max(updates_per_s, 1e-9), 1))
                    print(f"  step {step:6d}/{total_steps}  loss={window_loss/window_steps:.4f}  "
                          f"lr={sched.get_last_lr()[1]:.2e}  {fps:.0f} frames/s  "
                          f"eta {(total_steps-step)/max(updates_per_s,1e-9)/3600:.1f}h", flush=True)
                    run_loss = 0.0
                    window_started = time.perf_counter()
                    window_steps = 0
                    window_prepare_s = window_encoder_s = window_update_s = window_loss = 0.0
                    window_grad_norm = 0.0
                if step % args.ckpt_every == 0:
                    checkpoint_started = time.perf_counter()
                    log_event("online_training.checkpoint_started", kind="recovery", step=step)
                    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg,
                                "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "last.pt")
                    log_event("online_training.checkpoint_complete", kind="recovery",
                              path=str(ckpt_dir / "last.pt"), bytes=(ckpt_dir / "last.pt").stat().st_size,
                              elapsed_s=round(time.perf_counter() - checkpoint_started, 3))
                if step % args.val_every == 0:
                    v = validate("periodic")
                    print(f"  [val] step {step}: {v:.4f}  {'★ best' if v < best else ''}", flush=True)
                    if v < best:
                        best = v
                        checkpoint_started = time.perf_counter()
                        log_event("online_training.checkpoint_started", kind="best", step=step,
                                  validation_loss=round(v, 6))
                        torch.save({"step": step, "val_loss": v, "state_dict": model.state_dict(), "cfg": cfg,
                                    "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "best.pt")
                        log_event("online_training.checkpoint_complete", kind="best",
                                  path=str(ckpt_dir / "best.pt"), bytes=(ckpt_dir / "best.pt").stat().st_size,
                                  elapsed_s=round(time.perf_counter() - checkpoint_started, 3))
            del imgs, states, acts, eps, instrs
            log_event("online_training.shard_complete", shard=Path(sp).name, epoch=epoch,
                      optimizer_updates=step - shard_step_start,
                      elapsed_s=round(time.perf_counter() - shard_started, 3))
    v = validate("final")
    print(f"\n[done] final val {v:.4f} (best {min(best, v):.4f})", flush=True)
    if v <= best:
        checkpoint_started = time.perf_counter()
        log_event("online_training.checkpoint_started", kind="best", step=step,
                  validation_loss=round(v, 6))
        torch.save({"step": step, "val_loss": v, "state_dict": model.state_dict(), "cfg": cfg,
                    "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "best.pt")
        log_event("online_training.checkpoint_complete", kind="best", path=str(ckpt_dir / "best.pt"),
                  bytes=(ckpt_dir / "best.pt").stat().st_size,
                  elapsed_s=round(time.perf_counter() - checkpoint_started, 3))
    checkpoint_started = time.perf_counter()
    log_event("online_training.checkpoint_started", kind="final", step=step)
    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg,
                "dataset": "language_table", "exp_id": "exp02_balanced"}, ckpt_dir / "final.pt")
    log_event("online_training.checkpoint_complete", kind="final", path=str(ckpt_dir / "final.pt"),
              bytes=(ckpt_dir / "final.pt").stat().st_size,
              elapsed_s=round(time.perf_counter() - checkpoint_started, 3))
    progress.update(status="complete", stage="complete", optimizer_step=step,
                    total_steps=total_steps, final_validation_loss=round(v, 6),
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_s=round(time.perf_counter() - run_started, 3))
    write_json_atomic(progress_path, progress)
    log_event("online_training.complete", optimizer_step=step, total_steps=total_steps,
              validation_loss=round(v, 6), elapsed_s=round(time.perf_counter() - run_started, 3),
              output_dir=cfg.output_dir)
    print(f"checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failed_utc = datetime.now(timezone.utc).isoformat()
        progress_path = _ACTIVE_PROGRESS.get("path")
        if progress_path is not None:
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
                progress.update(status="failed", failed_stage=progress.get("stage", "startup"),
                                stage="failed", error_type=type(exc).__name__, error=str(exc),
                                failed_utc=failed_utc)
                write_json_atomic(progress_path, progress)
            except (OSError, ValueError):
                pass
        log_event("online_training.failed", level="ERROR", error_type=type(exc).__name__,
                  error=str(exc), progress_path=str(progress_path) if progress_path else None)
        raise
