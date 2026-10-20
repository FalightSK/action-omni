"""Small, shared reproducibility helpers for the opt-in camera-ready protocol.

No training or model download occurs on import. Legacy callers need not use this
module; camera-ready callers must carry the same manifest through every stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def split_episodes(episodes, seed=42):
    """Assign whole episodes, independently within each sorted task."""
    episodes = [dict(e) for e in episodes]
    rng = np.random.default_rng(seed)
    for task in sorted({e["task"] for e in episodes}):
        group = sorted((e for e in episodes if e["task"] == task),
                       key=lambda e: e["id"])
        if len(group) < 2:
            raise ValueError(f"Task {task!r} needs at least two demonstrations")
        count = max(1, int(len(group) * 0.10))
        validation = set(rng.permutation(len(group))[:count].tolist())
        for i, episode in enumerate(group):
            episode["split"] = "validation" if i in validation else "train"
    return episodes


def protocol_arrays(protocol):
    """Reconstruct exact cache row membership; reject gaps and overlaps."""
    n = protocol["n_samples"]
    episode_ids = np.full(n, -1, dtype=np.int64)
    validation = np.zeros(n, dtype=bool)
    seen = set()
    for e in protocol["episodes"]:
        start, stop = e["start"], e["stop"]
        if (e["id"] in seen or not 0 <= start < stop <= n
                or (episode_ids[start:stop] != -1).any()
                or e["split"] not in ("train", "validation")):
            raise ValueError("Invalid or overlapping episode mapping")
        seen.add(e["id"])
        episode_ids[start:stop] = e["id"]
        validation[start:stop] = e["split"] == "validation"
    if (episode_ids < 0).any() or not validation.any() or validation.all():
        raise ValueError("Manifest must cover all rows with nonempty train/validation")
    return episode_ids, validation


def load_protocol(path, dataset=None):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = digest({k: v for k, v in p.items() if k != "protocol_id"})
    if p.get("schema") != 1 or p.get("protocol_id") != expected:
        raise ValueError("Protocol integrity/schema mismatch")
    if dataset is not None and p["dataset"] != dataset:
        raise ValueError(f"Protocol dataset {p['dataset']} != {dataset}")
    protocol_arrays(p)
    return p


def apply_normalization(cfg, protocol):
    # New caches/live inference must not expose batch-dependent processor padding.
    cfg.canonicalize_padding = True
    cfg.vlm_bf16_reduced_precision_reduction = False
    for field, values in protocol.get("normalization", {}).items():
        if field not in ("action_mean", "action_std", "state_mean", "state_std"):
            raise ValueError(f"Unknown normalization field: {field}")
        dim = cfg.action_dim if field.startswith("action") else cfg.state_dim
        if len(values) != dim or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} for configured dimensions")
        if field.endswith("std") and (np.asarray(values) < 0).any():
            raise ValueError(f"Negative scale in {field}")
        setattr(cfg, field, tuple(values))


def cache_config(cfg):
    fields = ("vlm_backbone", "model_path", "vlm_extract_layers", "vlm_hidden_size",
              "img_seq_len", "img_grid_h", "img_grid_w", "n_views", "task_text",
              "aloha_img_h", "aloha_img_w", "lt_img_w", "lt_img_h",
              "action_horizon", "state_dim", "action_dim", "use_relative_actions",
              "action_mean", "action_std", "state_mean", "state_std", "canonicalize_padding",
              "vlm_bf16_reduced_precision_reduction")
    # JSON normalizes tuple/list representations across checkpoints and HDF5.
    return json.loads(json.dumps({k: getattr(cfg, k) for k in fields if hasattr(cfg, k)}))


def encoder_code_id():
    return digest({p: file_digest(ROOT / p) for p in (
        "models/vla.py", "models/backbones.py", "models/vision_patch_projection.py",
        "data/aloha/dataset.py", "data/libero/dataset.py", "data/libero/frames.py",
        "data/hdf5_embeddings.py", "scripts/precompute.py")})


def source_assets(cfg):
    """Hash the released weights plus the processors/configs actually used."""
    directories = {"groot": "groot_n17_3b", "qwen3vl": "qwen3_vl_2b",
                   "cosmos": "cosmos_reason2_2b", "cosmos16": "cosmos_reason2_2b"}
    root = ROOT / "asset/models" / directories[cfg.vlm_backbone]
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No source weights in {root}")
    files = weights + sorted(root.glob("*.json"))
    if cfg.vlm_backbone == "groot":
        files += sorted((ROOT / "asset/models/cosmos_reason2_2b").glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in files}


def seed_everything(seed):
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore CUDA RNG state on a CPU-only host")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def fixed_rng(seed):
    state = rng_state()
    try:
        seed_everything(seed)
        yield
    finally:
        restore_rng(state)


def sampling_seed(dataset, training_seed, task, start):
    dataset_id = {"aloha": 1, "libero": 2}[dataset]
    return int(np.random.SeedSequence([20261018, dataset_id, training_seed, task, start])
               .generate_state(1)[0])


def run_signature(cfg, protocol, seed):
    fields = asdict(cfg)
    for key in ("output_dir", "embeddings_cache", "device", "embed_batch_size"):
        fields.pop(key, None)
    training_code = {p: file_digest(ROOT / p) for p in (
        "scripts/train.py", "experiment_protocol.py", "models/vla_train.py", "models/flow_matching.py")}
    return digest({"cfg": fields, "protocol_id": protocol["protocol_id"], "seed": seed,
                   "training_code": training_code, "encoder_code_id": encoder_code_id()})


def checkpoint_config(requested_cfg, checkpoint, dataset, verify_live_encoder=False):
    """Use saved normalization/settings, but never load another requested arm."""
    saved = checkpoint.get("cfg")
    protocol = checkpoint.get("protocol")
    if protocol:
        if digest({k: v for k, v in protocol.items() if k != "protocol_id"}) != protocol.get("protocol_id"):
            raise ValueError("Embedded checkpoint protocol integrity mismatch")
        protocol_arrays(protocol)
        if protocol["dataset"] != dataset or checkpoint.get("training_seed") is None:
            raise ValueError("Checkpoint lacks compatible protocol/training seed")
        if saved is None:
            raise ValueError("Protocol checkpoint lacks configuration")
    if saved is not None:
        for key in ("vlm_backbone", "n_views", "vlm_extract_layers", "action_dim",
                    "state_dim", "dit_hidden_dim", "action_horizon"):
            if getattr(saved, key, None) != getattr(requested_cfg, key, None):
                raise ValueError(f"Requested arm differs from checkpoint: {key}")
        if protocol:
            if run_signature(saved, protocol, checkpoint["training_seed"]) != checkpoint.get("run_signature"):
                raise ValueError("Checkpoint configuration/signature mismatch")
            if verify_live_encoder:
                provenance = checkpoint.get("encoder_provenance", {})
                if (provenance.get("encoder_code_id") != encoder_code_id()
                        or provenance.get("source_assets") != source_assets(saved)):
                    raise ValueError("Live encoder assets/code differ from the training cache")
        return saved
    return requested_cfg
