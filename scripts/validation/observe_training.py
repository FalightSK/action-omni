"""Bounded CUDA/real-data training check. Outputs are never paper results.

Encode a few real frames, repeat that cache to exercise the configured batch
size, and compare uninterrupted training against checkpoint-interrupted resume.
The production trainer is reused with a three-epoch smoke schedule only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from data.hdf5_embeddings import HDF5EmbeddingWriter
from experiment_protocol import (apply_normalization, encoder_code_id, load_protocol,
                                 protocol_arrays, write_json)
from models.vla import VLAModel
from scripts.precompute import _custom_collate, _get_raw_dataset
import scripts.train as trainer


class SmokeInterruption(RuntimeError):
    """Intentional interruption immediately after last.pt is atomically saved."""


def encode_subset(cfg, dataset, protocol, samples, cache):
    _, validation = protocol_arrays(protocol)
    indices = np.flatnonzero(~validation)[:samples].tolist()
    raw = _get_raw_dataset(dataset, cfg)
    if len(raw) != protocol["n_samples"]:
        raise ValueError("Raw data and prepared protocol lengths differ")
    loader = DataLoader(Subset(raw, indices), batch_size=4, num_workers=0,
                        collate_fn=_custom_collate)
    model = VLAModel(cfg)
    model.vlm.to("cuda").eval()
    writer = HDF5EmbeddingWriter(
        cache, len(indices), cfg.img_seq_len, cfg.vlm_hidden_size,
        cfg.state_dim, cfg.action_horizon, cfg.action_dim,
        n_layers=cfg.n_vlm_layers,
        attrs={"smoke_only": True, "full_dataset": False, "dataset": dataset,
               "vlm_extract_layers": list(cfg.vlm_extract_layers),
               "encoder_code_id": encoder_code_id()})
    writer.f.create_dataset("source_rows", data=indices)
    ptr = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs = model.build_vlm_inputs(batch["images"], batch["task_texts"],
                                            torch.device("cuda"))
            tokens, mask = model.encode_vlm(inputs)
            if tokens.shape[1:] != (cfg.img_seq_len, cfg.vlm_hidden_size):
                raise ValueError(f"Unexpected token geometry: {tuple(tokens.shape)}")
            if not torch.isfinite(tokens).all() or not mask.any():
                raise ValueError("Nonfinite or image-free encoder output")
            writer.write(ptr, tokens.to("cpu", dtype=torch.bfloat16), mask.cpu(),
                         batch["states"], batch["actions"])
            ptr += len(tokens)
            print(f"[encode] {dataset}: {ptr}/{len(indices)} real frames", flush=True)
    writer.close()
    if hasattr(raw, "close"):
        raw.close()
    if hasattr(raw, "_reader"):
        raw._reader.close()
    seconds = time.perf_counter() - started
    del model, inputs, tokens, mask, raw, loader
    gc.collect()
    torch.cuda.empty_cache()
    return {"unique_frames": len(indices), "source_rows": indices,
            "seconds_excluding_model_load": seconds, "cache": str(cache)}


def observe(cfg, dataset, exp, output, workers, seed):
    # Repeat the tiny real cache instead of encoding a full training dataset.
    # Repeated frames and a shortened schedule are infrastructure checks only.
    load_dataset = trainer._get_embedding_dataset
    adam_step = torch.optim.AdamW.step
    replace = Path.replace
    observations = []
    phase = ""

    def repeated_dataset(*args, **kwargs):
        ds = load_dataset(*args, **kwargs)
        n = cfg.batch_size * 4
        return Subset(ds, [i % len(ds) for i in range(n)])

    def checked_step(optim, *args, **kwargs):
        before = []
        record = {"phase": phase, "groups": []}
        for name, group in zip(("adapter", "decoder"), optim.param_groups):
            params = [p for p in group["params"] if p.grad is not None]
            if not params or any(p.device.type != "cuda" for p in params):
                raise ValueError(f"{name}: missing CUDA parameters/gradients")
            norms = torch.stack([p.grad.detach().norm() for p in params])
            if not torch.isfinite(norms).all():
                raise ValueError(f"{name}: nonfinite gradients")
            chosen = params[int(norms.argmax())]
            before.append((chosen, chosen.detach().clone()))
            record["groups"].append({"name": name, "gradient_norm": norms.norm().item(),
                                     "learning_rate": group["lr"]})
        started = time.perf_counter()
        result = adam_step(optim, *args, **kwargs)
        torch.cuda.synchronize()
        for group, (param, old) in zip(record["groups"], before):
            group["parameter_delta_max"] = (param.detach() - old).abs().max().item()
        record["optimizer_seconds"] = time.perf_counter() - started
        observations.append(record)
        print(f"[update] {dataset}/{phase} step {len(observations)}: "
              + json.dumps(record["groups"]), flush=True)
        return result

    def interrupt_after_checkpoint(path, target):
        result = replace(path, target)
        if Path(target).name == "last.pt":
            raise SmokeInterruption("Checkpoint committed; simulating process interruption")
        return result

    phases = []
    for phase, directory, interrupt in (
        ("uninterrupted", output / "baseline", False),
        ("interrupted", output / "resumed", True),
        ("resume", output / "resumed", False),
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with patch.object(trainer, "get_config", side_effect=lambda *_: deepcopy(cfg)), \
             patch.object(trainer, "_get_embedding_dataset", side_effect=repeated_dataset), \
             patch.object(torch.optim.AdamW, "step", checked_step):
            try:
                with (patch.object(Path, "replace", interrupt_after_checkpoint)
                      if interrupt else patch.object(Path, "replace", replace)):
                    trainer.main(["--dataset", dataset, "--exp", exp, "--seed", str(seed),
                                  "--cache", cfg.embeddings_cache, "--output-dir", str(directory),
                                  "--workers", str(workers)])
            except SmokeInterruption:
                if not interrupt:
                    raise
                print("[resume] Intentional interruption after epoch-1 checkpoint", flush=True)
            else:
                if interrupt:
                    raise AssertionError("Checkpoint interruption did not happen")
        checkpoint = torch.load(directory / "checkpoints/last.pt", map_location="cpu",
                                weights_only=False)
        phases.append({"phase": phase, "seconds": time.perf_counter() - started,
                       "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        del checkpoint

    left = torch.load(output / "baseline/checkpoints/last.pt", map_location="cpu", weights_only=False)
    right = torch.load(output / "resumed/checkpoints/last.pt", map_location="cpu", weights_only=False)
    max_delta = 0.0
    for key, value in left["state_dict"].items():
        torch.testing.assert_close(value, right["state_dict"][key], rtol=1e-5, atol=1e-6)
        max_delta = max(max_delta, (value - right["state_dict"][key]).abs().max().item())
    if left["scheduler"] != right["scheduler"] or left["global_step"] != right["global_step"]:
        raise AssertionError("Resumed schedule/step count differs from uninterrupted training")
    for param, state in left["optim"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["optim"]["state"][param][key], rtol=1e-5, atol=1e-6)
    for name in ("adapter", "decoder"):
        groups = [g for row in observations for g in row["groups"] if g["name"] == name]
        if not any(g["gradient_norm"] > 0 and g["parameter_delta_max"] > 0 for g in groups):
            raise AssertionError(f"{name} never received an effective update")
    return {"batch_size": cfg.batch_size, "workers": workers, "head_precision": "float32",
            "epochs": cfg.num_epochs, "warmup_steps": cfg.warmup_steps,
            "repeated_rows": cfg.batch_size * 4, "phases": phases,
            "resume_parameter_max_difference": max_delta, "resume_matches": True,
            "updates": observations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["aloha", "libero"], default=["aloha", "libero"])
    parser.add_argument("--exp", default="exp05")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.samples <= 128 or not 0 <= args.workers <= 6:
        parser.error("Use 2..128 real frames and 0..6 workers for this bounded check")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback for a GPU readiness check")
    output = (args.output_dir or ROOT / "asset/runs/readiness" /
              datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "smoke_only": True, "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "worker_count": args.workers, "datasets": {},
              "limitations": "Tiny real subset repeated for batch sizing; shortened schedule; "
                             "no policy-performance, full-cache, simulator, or long-duration claim."}
    path = output / "observation.json"
    write_json(path, report)
    try:
        for dataset in args.datasets:
            cfg = get_config(dataset, args.exp)
            protocol_ids = {
                "aloha": "97fa699cb307366377cec2eb0f29795d967710f9ffe0fea9d1e1c50108b8d537",
                "libero": "4d1d6267e4270ac86caf491ea9b5fb630eb0e3a8a728822f96e94be9b3e3b0ad",
            }
            protocol = load_protocol(ROOT / "asset/protocols" / dataset /
                                     f"{protocol_ids[dataset]}.json", dataset)
            apply_normalization(cfg, protocol)
            cfg.device = "cuda"
            cfg.num_epochs, cfg.warmup_steps = 3, 2
            cfg.min_epochs = 0
            work = output / f"{dataset}_{args.exp}"
            work.mkdir()
            cfg.embeddings_cache = str(work / "smoke_features.h5")
            encoded = encode_subset(cfg, dataset, protocol, args.samples, cfg.embeddings_cache)
            report["datasets"][dataset] = {"encoding": encoded, "status": "training"}
            write_json(path, report)
            observed = observe(cfg, dataset, args.exp, work, args.workers, args.seed)
            report["datasets"][dataset].update(observed, status="passed")
            write_json(path, report)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(path, report)
        print(f"[observation] {report['status']}: {path}", flush=True)


if __name__ == "__main__":
    main()
