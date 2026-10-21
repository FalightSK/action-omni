"""Fast equivalent for the Qwen3-VL vision patch Conv3d projection.

Transformers 4.57.6 implements non-overlapping, unpadded 3D patches as a
Conv3d. Since every output position consumes the complete kernel exactly once,
that operation is the same matrix multiplication as a flattened linear layer.
The loader installs this forward only for that exact HF module/version and
Conv3d contract; parameter names and state_dict contents stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_TRANSFORMERS_VERSION = "4.57.6"
SUPPORTED_CLASS_MODULE = "transformers.models.qwen3_vl.modeling_qwen3_vl"
SUPPORTED_CLASS_NAME = "Qwen3VLVisionPatchEmbed"

_ORIGINAL_FORWARDS = WeakKeyDictionary()


@dataclass(frozen=True)
class PatchProjectionReport:
    enabled: bool
    patched_modules: int
    transformers_version: str | None
    reason: str | None = None


def _contract_error(module: nn.Module) -> str | None:
    proj = getattr(module, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        return "proj is not Conv3d"
    try:
        module_channels = int(module.in_channels)
        temporal = int(module.temporal_patch_size)
        patch = int(module.patch_size)
        embed_dim = int(module.embed_dim)
    except (AttributeError, TypeError, ValueError):
        return "module patch dimensions are missing or invalid"
    if proj.in_channels != module_channels or proj.out_channels != embed_dim:
        return "module dimensions do not match Conv3d channels"
    if tuple(proj.kernel_size) != (temporal, patch, patch):
        return "module patch dimensions do not match Conv3d kernel"
    if tuple(proj.kernel_size) != tuple(proj.stride):
        return "kernel_size must equal stride"
    if tuple(proj.padding) != (0, 0, 0):
        return "padding must be zero"
    if tuple(proj.dilation) != (1, 1, 1) or proj.groups != 1:
        return "dilation/groups differ from supported contract"
    if proj.padding_mode != "zeros":
        return "padding_mode must be zeros"
    if proj.weight.shape[1] != proj.in_channels:
        return "weight input channels do not match Conv3d"
    if tuple(proj.weight.shape[2:]) != tuple(proj.kernel_size):
        return "weight kernel shape does not match Conv3d"
    if proj.bias is None:
        return "expected the HF patch projection bias"
    return None


def linearized_patch_projection(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run HF's exact input cast and patch projection using ``F.linear``."""
    error = _contract_error(module)
    if error:
        raise ValueError(f"Unsupported Qwen3-VL patch projection: {error}")
    channels = int(module.in_channels)
    temporal = int(module.temporal_patch_size)
    patch = int(module.patch_size)
    expected = (channels, temporal, patch, patch)
    flattened_width = channels * temporal * patch * patch
    is_flat = hidden_states.ndim == 2 and hidden_states.shape[-1] == flattened_width
    is_5d = hidden_states.ndim == 5 and tuple(hidden_states.shape[1:]) == expected
    if not (is_flat or is_5d):
        raise ValueError(f"Expected flattened patches (N,{flattened_width}) or "
                         f"(N,{','.join(map(str, expected))}), got {tuple(hidden_states.shape)}")

    conv = module.proj
    # Transformers 4.57.6 casts before Conv3d. Preserve that cast explicitly.
    patches = hidden_states.view(-1, *expected).to(dtype=conv.weight.dtype)
    flat = patches.view(patches.shape[0], -1)
    weight = conv.weight.view(conv.out_channels, -1)
    # PyTorch 2.9.1's CUDA Conv3d path computes convolution first, writes the
    # BF16 result, then calls output.add_(bias). Passing bias to a 2D F.linear
    # instead takes the fused addmm path, adding bias before BF16 rounding; that
    # changed full VLM tokens materially. Match Conv3d's separate BF16 add.
    output = F.linear(flat, weight, None)
    output.add_(conv.bias)
    return output.view(-1, int(module.embed_dim))


def _fast_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return linearized_patch_projection(self, hidden_states)


def find_qwen3vl_patch_embeds(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules()
            if type(module).__name__ == SUPPORTED_CLASS_NAME
            and type(module).__module__ == SUPPORTED_CLASS_MODULE]


def install_qwen3vl_patch_projection(model: nn.Module, *, required: bool = False) -> PatchProjectionReport:
    """Install the equivalent linear projection, with strict version/shape guards."""
    try:
        import transformers
    except ImportError as exc:
        if required:
            raise RuntimeError("Transformers is required for the Qwen3-VL patch optimization") from exc
        return PatchProjectionReport(False, 0, None, "transformers is unavailable")

    version = getattr(transformers, "__version__", None)
    if version != SUPPORTED_TRANSFORMERS_VERSION:
        reason = f"requires transformers=={SUPPORTED_TRANSFORMERS_VERSION}; found {version}"
        if required:
            raise RuntimeError(reason)
        return PatchProjectionReport(False, 0, version, reason)

    modules = find_qwen3vl_patch_embeds(model)
    if not modules:
        reason = "no supported Qwen3VLVisionPatchEmbed module found"
        if required:
            raise RuntimeError(reason)
        return PatchProjectionReport(False, 0, version, reason)

    for module in modules:
        error = _contract_error(module)
        if error:
            reason = f"unsupported patch module contract: {error}"
            if required:
                raise RuntimeError(reason)
            return PatchProjectionReport(False, 0, version, reason)

    patched = 0
    for module in modules:
        if module in _ORIGINAL_FORWARDS:
            continue
        # Store the unbound class function: a bound method would strongly hold
        # its weak key and keep discarded GPU models alive through this table.
        _ORIGINAL_FORWARDS[module] = type(module).forward
        module.forward = MethodType(_fast_forward, module)
        patched += 1
    return PatchProjectionReport(True, patched, version)


def uninstall_qwen3vl_patch_projection(model: nn.Module) -> int:
    """Restore original HF forwards; useful for parity diagnostics."""
    restored = 0
    for module in find_qwen3vl_patch_embeds(model):
        original = _ORIGINAL_FORWARDS.pop(module, None)
        if original is not None:
            module.forward = MethodType(original, module)
            restored += 1
    return restored
