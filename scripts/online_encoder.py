"""Raw-data loading and frozen VLM encoding for online-feature training.

ALOHA stores images in a single video, while training samples shuffled frame
indices. Its source reader is intentionally sequential, so this module stages
only resized RGB frames to local ``.npy`` memory maps; states and action windows
are read/computed from raw labels per sample. It never stores VLM embeddings.
LIBERO is read directly from its per-task HDF5 files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset

from scripts.run_log import log_event


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_files(dataset_root: Path, camera_key: str) -> list[Path]:
    parquet = sorted((dataset_root / "data/chunk-000").glob("file-*.parquet"))
    videos = sorted((dataset_root / "videos" / camera_key / "chunk-000").glob("*.mp4"))
    if not parquet or not videos:
        raise FileNotFoundError(
            f"ALOHA raw data is incomplete under {dataset_root}: "
            f"parquet_shards={len(parquet)}, videos={len(videos)}")
    return parquet + videos


def _identity(cfg, protocol: dict | None, dataset_root: Path,
              source_files: list[Path]) -> dict:
    from experiment_protocol import file_digest

    files = {}
    for path in source_files:
        rel = path.relative_to(dataset_root).as_posix()
        files[rel] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    dataset_code = Path(__file__).resolve().parents[1] / "data/aloha/dataset.py"
    return {
        "schema": 1,
        "dataset": "aloha",
        "protocol_id": protocol["protocol_id"] if protocol else None,
        "source_files": files,
        "source_code_sha256": file_digest(dataset_code),
        "frame_stage_code_sha256": file_digest(Path(__file__).resolve()),
        "preprocessing": {
            "camera_key": cfg.camera_key,
            "width": int(cfg.aloha_img_w),
            "height": int(cfg.aloha_img_h),
            "pil_version": PIL.__version__,
            "action_horizon": int(cfg.action_horizon),
            "state_dim": int(cfg.state_dim),
            "action_dim": int(cfg.action_dim),
            "action_mean": list(cfg.action_mean),
            "action_std": list(cfg.action_std),
            "state_mean": list(cfg.state_mean),
            "state_std": list(cfg.state_std),
        },
    }


def _array_records(directory: Path) -> dict:
    records = {}
    for name in ("frames.npy",):
        path = directory / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        records[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                         "size": path.stat().st_size, "sha256": _sha256(path)}
        del array
    return records


def _validate_aloha_stage(final_dir: Path, identity_id: str, identity: dict,
                          samples: int) -> tuple[dict, dict]:
    manifest_path = final_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Incomplete ALOHA raw-frame stage has no manifest: {final_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("samples") != samples
            or manifest.get("identity_id") != identity_id
            or manifest.get("identity") != identity):
        raise RuntimeError(f"ALOHA raw-frame stage identity mismatch: {final_dir}")
    actual = _array_records(final_dir)
    if actual != manifest.get("arrays"):
        raise RuntimeError(f"ALOHA raw-frame stage integrity check failed: {final_dir}")
    if len(np.load(final_dir / "frames.npy", mmap_mode="r", allow_pickle=False)) != samples:
        raise RuntimeError("ALOHA raw-frame stage sample count differs from source data")
    return manifest, actual


def _publish_aloha_stage(temp_dir: Path, final_dir: Path, identity_id: str,
                         identity: dict, samples: int) -> tuple[bool, dict | None, dict | None]:
    """Publish once; if a concurrent policy won, validate and reuse its cache."""
    try:
        os.replace(temp_dir, final_dir)
    except FileExistsError:
        if not final_dir.is_dir():
            raise
        manifest, arrays = _validate_aloha_stage(final_dir, identity_id, identity, samples)
        return False, manifest, arrays
    return True, None, None


class OnlineAlohaDataset(Dataset):
    """Memory-mapped RGB frames and live action-window labels for ALOHA."""

    def __init__(self, cache_dir: Path, raw_dataset, cfg):
        self.cache_dir = Path(cache_dir)
        self.episode_ids = np.asarray(raw_dataset.episode_ids, dtype=np.int64)
        self.states_raw = raw_dataset.states
        self.actions_raw = raw_dataset.actions
        self.action_mean = np.asarray(cfg.action_mean, dtype=np.float32)
        self.action_std = np.asarray(cfg.action_std, dtype=np.float32)
        self.state_mean = np.asarray(cfg.state_mean, dtype=np.float32)
        self.state_std = np.asarray(cfg.state_std, dtype=np.float32)
        self.action_horizon = int(cfg.action_horizon)
        self.task_text = cfg.task_text
        self.frames = np.load(self.cache_dir / "frames.npy", mmap_mode="r", allow_pickle=False)
        self.validation_mask = None
        if len(self.frames) != len(self.states_raw) or len(self.frames) != len(self.episode_ids):
            raise ValueError("ALOHA staged frames do not match raw labels/episode metadata")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = np.array(self.frames[index], copy=True)
        episode = self.episode_ids[index]
        last_action = self.actions_raw[index]
        actions = []
        for offset in range(self.action_horizon):
            frame_index = index + offset
            if frame_index < len(self.actions_raw) and self.episode_ids[frame_index] == episode:
                last_action = self.actions_raw[frame_index]
            actions.append((last_action - self.action_mean) / (self.action_std + 1e-8))
        state = ((self.states_raw[index] - self.state_mean) / (self.state_std + 1e-8))
        return {
            "image": Image.fromarray(frame),
            "state": torch.from_numpy(np.asarray(state, dtype=np.float32).copy()),
            "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            "task_text": self.task_text,
            "idx": int(index),
        }


class OnlineLiberoDataset(Dataset):
    """Random-access LIBERO rows with bounded HDF5 action-chunk reads."""

    def __init__(self, raw_dataset, cfg):
        self.raw = raw_dataset
        self.cfg = cfg
        self.episode_ids = raw_dataset.episode_ids
        self.validation_mask = getattr(raw_dataset, "validation_mask", None)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> dict:
        from data.libero.dataset import CAMERA, CAMERA_WRIST, resize_frame

        fi, dname, t = self.raw.index[index]
        demo = self.raw._f(fi)["data"][dname]
        episode_length = self.raw._lengths[(fi, dname)]
        image = np.asarray(demo["obs"][CAMERA][t])[::-1]
        if getattr(self.cfg, "n_views", 1) > 1:
            wrist = np.asarray(demo["obs"][CAMERA_WRIST][t])[::-1]
            image = [resize_frame(image, self.raw.w, self.raw.h),
                     resize_frame(wrist, self.raw.w, self.raw.h)]
        else:
            image = resize_frame(image, self.raw.w, self.raw.h)

        end = min(t + self.raw.horizon, episode_length)
        actions = np.asarray(demo["actions"][t:end], dtype=np.float32)
        if len(actions) < self.raw.horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.raw.horizon - len(actions), axis=0)],
                axis=0)
        return {
            "image": image,
            "state": torch.from_numpy(
                np.asarray(demo["robot_states"][t], dtype=np.float32)),
            "actions": torch.from_numpy(actions),
            "task_text": self.raw.task_texts[index],
            "idx": index,
        }

    def close(self) -> None:
        self.raw.close()


def _stage_aloha(raw_dataset, cfg, protocol: dict | None,
                 cache_root: Path) -> tuple[OnlineAlohaDataset, dict]:
    started = time.perf_counter()
    dataset_root = Path(cfg.dataset_root).resolve()
    source_files = _source_files(dataset_root, cfg.camera_key)
    log_event("online_data.source_fingerprint_started", dataset="aloha",
              files=len(source_files), bytes=sum(p.stat().st_size for p in source_files))
    identity = _identity(cfg, protocol, dataset_root, source_files)
    log_event("online_data.source_fingerprint_complete", dataset="aloha",
              files=len(source_files), identity_id=hashlib.sha256(
                  json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
              ).hexdigest(), elapsed_s=round(time.perf_counter() - started, 3))
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    final_dir = cache_root / identity_id
    cache_root.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        manifest, actual = _validate_aloha_stage(final_dir, identity_id, identity, len(raw_dataset))
        log_event("online_data.raw_stage_reused", dataset="aloha", path=str(final_dir),
                  samples=len(raw_dataset), bytes=sum(v["size"] for v in actual.values()),
                  verification_s=round(time.perf_counter() - started, 3))
    else:
        log_event("online_data.raw_stage_started", dataset="aloha", path=str(final_dir),
                  samples=len(raw_dataset), identity_id=identity_id,
                  bytes_estimate=(len(raw_dataset) * int(cfg.aloha_img_h)
                                  * int(cfg.aloha_img_w) * 3))
        with tempfile.TemporaryDirectory(prefix=f".{identity_id}.partial-", dir=cache_root) as temp:
            temp_dir = Path(temp)
            n = len(raw_dataset)
            frames = np.lib.format.open_memmap(
                temp_dir / "frames.npy", mode="w+", dtype=np.uint8,
                shape=(n, int(cfg.aloha_img_h), int(cfg.aloha_img_w), 3))
            last_report = time.perf_counter()
            report_every = max(1, n // 20)
            try:
                from data.aloha.dataset import resize_frame
                for index in range(n):
                    image = resize_frame(raw_dataset._reader.get(index),
                                         int(cfg.aloha_img_w), int(cfg.aloha_img_h))
                    frames[index] = np.asarray(image, dtype=np.uint8)
                    if (index + 1) % report_every == 0 or index + 1 == n:
                        elapsed = time.perf_counter() - started
                        log_event("online_data.raw_stage_progress", dataset="aloha",
                                  completed=index + 1, total=n,
                                  samples_per_s=round((index + 1) / max(elapsed, 1e-9), 2),
                                  eta_s=round((n - index - 1) * elapsed / (index + 1), 1),
                                  interval_s=round(time.perf_counter() - last_report, 3))
                        last_report = time.perf_counter()
            except BaseException:
                log_event("online_data.raw_stage_failed", level="ERROR", dataset="aloha",
                          completed=index if "index" in locals() else 0, total=n,
                          elapsed_s=round(time.perf_counter() - started, 3))
                raise
            finally:
                frames.flush()
            del frames
            arrays = _array_records(temp_dir)
            manifest = {"schema": 1, "identity_id": identity_id, "identity": identity,
                        "samples": n, "arrays": arrays,
                        "created_utc": datetime.now(timezone.utc).isoformat()}
            manifest_tmp = temp_dir / "manifest.json.tmp"
            manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(manifest_tmp, temp_dir / "manifest.json")
            # Publish the complete directory atomically; incomplete stages never
            # appear at the deterministic reusable path.
            published, winner_manifest, winner_arrays = _publish_aloha_stage(
                temp_dir, final_dir, identity_id, identity, n)
            if not published:
                manifest, arrays = winner_manifest, winner_arrays
        if published:
            log_event("online_data.raw_stage_complete", dataset="aloha", path=str(final_dir),
                      samples=len(raw_dataset), bytes=sum(v["size"] for v in arrays.values()),
                      elapsed_s=round(time.perf_counter() - started, 3))
        else:
            log_event("online_data.raw_stage_race_reused", dataset="aloha", path=str(final_dir),
                      samples=len(raw_dataset), bytes=sum(v["size"] for v in arrays.values()),
                      verification_s=round(time.perf_counter() - started, 3))

    staged = OnlineAlohaDataset(final_dir, raw_dataset, cfg)
    staged.validation_mask = getattr(raw_dataset, "validation_mask", None)
    raw_dataset._reader.close()
    return staged, {"identity_id": identity_id, "path": str(final_dir),
                    "samples": len(staged), "arrays": manifest["arrays"]}


def load_online_dataset(dataset_name: str, cfg, protocol: dict | None = None):
    """Load ALOHA/LIBERO raw samples and verify their protocol row order."""
    if dataset_name == "aloha":
        from data.aloha import AlohaDataset
        raw = AlohaDataset(cfg)
    elif dataset_name == "libero":
        from data.libero import LiberoDataset
        raw = LiberoDataset(cfg)
    else:
        raise ValueError("--online-encoder currently supports only aloha and libero")

    if protocol is not None:
        from experiment_protocol import protocol_arrays
        episode_ids, validation_mask = protocol_arrays(protocol)
        if (len(raw) != protocol["n_samples"]
                or not np.array_equal(np.asarray(raw.episode_ids), episode_ids)):
            close = getattr(raw, "close", None)
            if close:
                close()
            raise ValueError("Raw dataset row order differs from the prepared episode protocol")
        raw.validation_mask = validation_mask

    if dataset_name == "aloha":
        stage_root = Path(__file__).resolve().parents[1] / "asset/cache/online_raw_frames/aloha"
        staged, details = _stage_aloha(raw, cfg, protocol, stage_root)
        return staged, {"mode": "online_encoder", "raw_stage": details}
    return OnlineLiberoDataset(raw, cfg), {"mode": "online_encoder", "raw_stage": None}


def raw_collate(batch: list[dict]) -> dict:
    return {
        "images": [item["image"] for item in batch],
        "states": torch.stack([item["state"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch]),
        "task_texts": [item["task_text"] for item in batch],
        "idxs": torch.tensor([item["idx"] for item in batch]),
    }


def load_frozen_encoder(cfg, device: torch.device):
    from models.vla import VLAModel

    started = time.perf_counter()
    log_event("online_encoder.backbone_load_started", backbone=cfg.vlm_backbone)
    model = VLAModel(cfg)
    model.vlm.to(device).eval()
    for parameter in model.vlm.parameters():
        parameter.requires_grad_(False)
    log_event("online_encoder.backbone_load_complete", backbone=cfg.vlm_backbone,
              parameters=sum(p.numel() for p in model.vlm.parameters()),
              trainable_parameters=sum(p.numel() for p in model.vlm.parameters() if p.requires_grad),
              backbone_training=model.vlm.training,
              elapsed_s=round(time.perf_counter() - started, 3))
    return model


@torch.no_grad()
def encode_online_batch(encoder, images: list, task_texts: list[str],
                        device: torch.device, microbatch_size: int,
                        profile: bool = False):
    """Encode a training batch in bounded VLM microbatches, without gradients."""
    if microbatch_size < 1:
        raise ValueError("VLM microbatch size must be positive")
    if not getattr(encoder, "_patch_projection_verified", False):
        from models.vision_patch_projection import (
            find_qwen3vl_patch_embeds, install_qwen3vl_patch_projection,
            uninstall_qwen3vl_patch_projection)
        if find_qwen3vl_patch_embeds(encoder.vlm):
            # Reuse the already loaded model and the first real training image.
            # No extra model initialization or dataset-wide precompute is needed.
            inputs = encoder.build_vlm_inputs(images[:1], task_texts[:1], device)
            uninstall_qwen3vl_patch_projection(encoder.vlm)
            try:
                reference, reference_mask = encoder.encode_vlm(dict(inputs))
                reference = reference.to(torch.bfloat16).float()
            finally:
                install_qwen3vl_patch_projection(encoder.vlm, required=True)
            actual, actual_mask = encoder.encode_vlm(dict(inputs))
            actual = actual.to(torch.bfloat16).float()
            relative_l2 = float(torch.linalg.vector_norm(reference - actual) /
                                torch.linalg.vector_norm(reference).clamp_min(1e-8))
            cosine = float(torch.nn.functional.cosine_similarity(
                reference.reshape(1, -1), actual.reshape(1, -1)).item())
            masks_equal = bool(torch.equal(reference_mask, actual_mask))
            passed = (torch.allclose(reference, actual, atol=0.125, rtol=0.01)
                      and relative_l2 <= 0.003 and cosine >= 0.99999 and masks_equal)
            log_event("online_encoder.patch_projection_verified", passed=bool(passed),
                      real_images=1, relative_l2_error=relative_l2,
                      cosine_similarity=cosine, masks_equal=masks_equal,
                      maximum_absolute_error=float((reference - actual).abs().max()))
            if not passed:
                raise RuntimeError("Optimized vision patch projection failed live feature parity")
        encoder._patch_projection_verified = True
    tokens_out, masks_out = [], []
    for microbatch_index, start in enumerate(range(0, len(images), microbatch_size), start=1):
        end = min(start + microbatch_size, len(images))
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        input_started = time.perf_counter()
        inputs = encoder.build_vlm_inputs(images[start:end], task_texts[start:end], device)
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        input_build_s = time.perf_counter() - input_started
        forward_started = time.perf_counter()
        tokens, mask = encoder.encode_vlm(inputs)
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - forward_started
        if profile:
            log_event("online_encoder.microbatch_timing", microbatch_index=microbatch_index,
                      microbatch_count=(len(images) + microbatch_size - 1) // microbatch_size,
                      samples=end - start, input_build_and_transfer_s=round(input_build_s, 4),
                      frozen_vlm_forward_s=round(forward_s, 4), device=str(device))
        # Match the on-disk cache's bf16 quantization followed by float32 use.
        tokens_out.append(tokens.to(torch.bfloat16).float())
        masks_out.append(mask)
    return torch.cat(tokens_out, dim=0), torch.cat(masks_out, dim=0)
