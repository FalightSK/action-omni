"""Bounded parity/performance check for the guarded Qwen3-VL patch projection.

Example: python scripts/verify_vision_patch_projection.py --dataset aloha --exp exp05
Loads the configured frozen backbone and tests batch sizes 1 and 4. It does not
train, write checkpoints, or change global numerical precision settings.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from experiment_protocol import apply_normalization, digest, load_protocol, source_assets
from models.vision_patch_projection import (
    find_qwen3vl_patch_embeds,
    install_qwen3vl_patch_projection,
    linearized_patch_projection,
    uninstall_qwen3vl_patch_projection,
)
from scripts.online_encoder import load_frozen_encoder, load_online_dataset


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(function, device: torch.device, repeats: int) -> dict:
    function()  # warm-up
    _sync(device)
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        function()
        _sync(device)
        values.append(time.perf_counter() - started)
    return {"seconds": values, "median_seconds": statistics.median(values)}


def _images(cfg, count: int):
    width = int(getattr(cfg, "aloha_img_w", getattr(cfg, "libero_img_w", 256)))
    height = int(getattr(cfg, "aloha_img_h", getattr(cfg, "libero_img_h", 256)))
    result = []
    for sample in range(count):
        views = []
        for view in range(int(getattr(cfg, "n_views", 1))):
            color = ((37 * sample + 61 * view + 17) % 256,
                     (83 * sample + 29 * view + 53) % 256,
                     (19 * sample + 97 * view + 101) % 256)
            views.append(Image.new("RGB", (width, height), color))
        result.append(views if len(views) > 1 else views[0])
    return result


def _inputs(encoder, images, device):
    texts = [encoder.config.task_text] * len(images)
    return encoder.build_vlm_inputs(images, texts, device)


def _tokens(encoder, images, device):
    encoded, mask = encoder.encode_vlm(_inputs(encoder, images, device))
    # Training stores/uses these exact BF16-quantized features.
    return encoded.to(torch.bfloat16).float(), mask


def _parity(expected, actual, *, atol: float, rtol: float) -> dict:
    difference = (expected.float() - actual.float()).abs()
    max_abs = float(difference.max().item())
    denominator = expected.float().abs().clamp_min(1e-8)
    max_rel = float((difference / denominator).max().item())
    close = torch.allclose(expected, actual, atol=atol, rtol=rtol)
    relative_l2 = float(torch.linalg.vector_norm(expected.float() - actual.float()) /
                        torch.linalg.vector_norm(expected.float()).clamp_min(1e-8))
    cosine = float(torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), actual.float().reshape(1, -1)).item())
    return {"shape": list(expected.shape), "max_abs_error": max_abs,
            "max_relative_error": max_rel, "atol": atol, "rtol": rtol,
            "relative_l2_error": relative_l2, "cosine_similarity": cosine,
            "allclose": close}


def _reference_patch_projection(module, hidden_states):
    patches = hidden_states.view(-1, module.in_channels, module.temporal_patch_size,
                                 module.patch_size, module.patch_size)
    patches = patches.to(dtype=module.proj.weight.dtype)
    return module.proj(patches).view(-1, module.embed_dim)


def _direct_patch_checks(module) -> dict:
    channels = int(module.in_channels)
    temporal = int(module.temporal_patch_size)
    patch = int(module.patch_size)
    feature_count = channels * temporal * patch * patch
    cpu_module = SimpleNamespace(
        in_channels=channels, temporal_patch_size=temporal, patch_size=patch,
        embed_dim=int(module.embed_dim),
        proj=copy.deepcopy(module.proj).to(device="cpu", dtype=torch.float32),
    )
    cpu_inputs = torch.randn(8, feature_count, dtype=torch.float32)
    cpu_expected = _reference_patch_projection(cpu_module, cpu_inputs)
    cpu_actual = linearized_patch_projection(cpu_module, cpu_inputs)
    cpu_result = _parity(cpu_expected, cpu_actual, atol=1e-5, rtol=1e-4)
    if not cpu_result["allclose"]:
        raise RuntimeError(f"CPU float32 patch projection mismatch: {cpu_result}")

    gpu_result = {"skipped": "CUDA unavailable"}
    if torch.cuda.is_available():
        gpu_module = SimpleNamespace(
            in_channels=channels, temporal_patch_size=temporal, patch_size=patch,
            embed_dim=int(module.embed_dim),
            proj=copy.deepcopy(module.proj).to(device="cuda", dtype=torch.bfloat16),
        )
        gpu_inputs = torch.randn(8, feature_count, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            gpu_expected = _reference_patch_projection(gpu_module, gpu_inputs)
            gpu_actual = linearized_patch_projection(gpu_module, gpu_inputs)
            torch.cuda.synchronize()
        gpu_result = _parity(gpu_expected, gpu_actual, atol=0.125, rtol=0.01)
        if not gpu_result["allclose"]:
            raise RuntimeError(f"CUDA BF16 patch projection mismatch: {gpu_result}")
    return {"cpu_float32_n8": cpu_result, "cuda_bfloat16_n8": gpu_result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    parser.add_argument("--backbone", choices=("groot", "qwen3vl"), required=True)
    parser.add_argument("--exp", help="Override experiment (must match --backbone)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_success_stopped_study.json")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    device = torch.device(args.device)
    exp_id = args.exp or ("exp05" if args.backbone == "groot" else "exp06")
    cfg = get_config(args.dataset, exp_id)
    plan = json.loads(args.plan.read_text())
    protocol = load_protocol(ROOT / plan["datasets"][args.dataset]["split_manifest"], args.dataset)
    apply_normalization(cfg, protocol)
    if cfg.vlm_backbone != args.backbone:
        parser.error(f"{args.dataset}/{exp_id} uses {cfg.vlm_backbone}, not {args.backbone}")
    if cfg.vlm_backbone not in ("groot", "cosmos", "cosmos16", "qwen3vl"):
        parser.error(f"{args.dataset}/{exp_id} uses unsupported backbone {cfg.vlm_backbone!r}")

    print(f"[verify] dataset={args.dataset} exp={exp_id} backbone={cfg.vlm_backbone} "
          f"device={device} torch={torch.__version__} repeats={args.repeats}", flush=True)
    assets = source_assets(cfg)
    dataset, _ = load_online_dataset(args.dataset, cfg, protocol)
    train_rows = [e["start"] for e in protocol["episodes"] if e["split"] == "train"]
    if len(train_rows) < 4:
        raise ValueError("Projection verification requires at least four training demonstrations")
    encoder = load_frozen_encoder(cfg, device)
    model = encoder.vlm
    patch_modules = find_qwen3vl_patch_embeds(model)
    if len(patch_modules) != 1:
        raise RuntimeError(f"Expected one patch embed module, found {len(patch_modules)}")
    patch_module = patch_modules[0]

    result = {
        "schema": 1,
        "dataset": args.dataset,
        "experiment": exp_id,
        "backbone": cfg.vlm_backbone,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "source_assets": assets,
        "source_assets_id": digest(assets),
        "direct_projection_checks": None,
        "batch_results": [],
    }
    result["direct_projection_checks"] = _direct_patch_checks(patch_modules[0])
    print(json.dumps({"direct_projection_checks": result["direct_projection_checks"]},
                     sort_keys=True), flush=True)

    for batch_size in (1, 4):
        images = [dataset[index]["image"] for index in train_rows[:batch_size]]
        captured = []
        hook = patch_module.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach()))

        uninstall_qwen3vl_patch_projection(model)
        reference_tokens, reference_mask = _tokens(encoder, images, device)
        reference_patch_inputs = captured[-1]
        reference_full_time = _timed(lambda: _tokens(encoder, images, device), device,
                                      args.repeats)
        reference_patch_time = _timed(lambda: patch_module(reference_patch_inputs),
                                      device, args.repeats)

        patch_report = install_qwen3vl_patch_projection(model, required=True)
        captured.clear()
        optimized_tokens, optimized_mask = _tokens(encoder, images, device)
        optimized_patch_inputs = captured[-1]
        if (reference_patch_inputs.shape != optimized_patch_inputs.shape
                or not torch.equal(reference_patch_inputs, optimized_patch_inputs)):
            raise RuntimeError("Original and optimized runs did not receive identical patch inputs")
        optimized_full_time = _timed(lambda: _tokens(encoder, images, device), device,
                                     args.repeats)
        optimized_patch_time = _timed(
            lambda: linearized_patch_projection(patch_module, optimized_patch_inputs),
            device, args.repeats)
        hook.remove()

        patch_expected = patch_module.proj(
            reference_patch_inputs.view(-1, patch_module.in_channels,
                                        patch_module.temporal_patch_size,
                                        patch_module.patch_size, patch_module.patch_size)
            .to(dtype=patch_module.proj.weight.dtype)
        ).view(-1, patch_module.embed_dim)
        patch_actual = linearized_patch_projection(patch_module, optimized_patch_inputs)
        patch_check = _parity(patch_expected, patch_actual,
                              atol=0.125 if patch_expected.dtype == torch.bfloat16 else 1e-5,
                              rtol=0.01 if patch_expected.dtype == torch.bfloat16 else 1e-4)
        token_check = _parity(reference_tokens, optimized_tokens, atol=0.125, rtol=0.01)
        mask_equal = bool(torch.equal(reference_mask, optimized_mask))
        entry = {
            "batch_size": batch_size,
            "processor_input_shapes": {key: list(value.shape) for key, value in
                                       _inputs(encoder, images, device).items()
                                       if torch.is_tensor(value)},
            "patch_input_shape": list(optimized_patch_inputs.shape),
            "patch_dtype": str(optimized_patch_inputs.dtype),
            "patch_module_weight_dtype": str(patch_module.proj.weight.dtype),
            "patch_parity": patch_check,
            "full_token_parity": token_check,
            "image_mask_equal": mask_equal,
            "reference_patch_timing": reference_patch_time,
            "linear_patch_timing": optimized_patch_time,
            "reference_full_vlm_timing": reference_full_time,
            "optimized_full_vlm_timing": optimized_full_time,
            "patch_install": {"enabled": patch_report.enabled,
                              "patched_modules": patch_report.patched_modules},
        }
        result["batch_results"].append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if (not patch_check["allclose"] or not token_check["allclose"] or not mask_equal
                or token_check["relative_l2_error"] > 0.003
                or token_check["cosine_similarity"] < 0.99999):
            raise RuntimeError(f"Parity failed at batch size {batch_size}")

    print("[verify] PASS", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
