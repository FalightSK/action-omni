"""
configs/registry.py
────────────────────
Central config factory for all datasets and experiments.

Usage:
    from configs.registry import get_config

    cfg = get_config("pusht", "exp02a")   # Best PushT architecture
    cfg = get_config("pusht", "exp01")    # MLP baseline
    cfg = get_config("aloha", "exp01")    # First ALOHA experiment (future)

Experiment ID convention:
    exp01   — first experiment for this dataset
    exp02a  — variant a of experiment 2
    exp03   — third experiment
    exp04   — fourth (e.g. ViT ablation for PushT)

Legacy integer shim (for backward compat with old --exp 1/2/3 scripts):
    get_config_legacy(dataset="pusht", exp_int=2) → same as get_config("pusht", "exp02a")
"""

from __future__ import annotations


# ── Registry: (dataset, exp_id) → config class ───────────────────────────────

_REGISTRY: dict[tuple[str, str], str] = {
    # PushT
    ("pusht", "exp01"):       "configs.pusht.exp01_mlp.PushTExp01MLP",
    ("pusht", "exp02"):       "configs.pusht.exp02a_dit.PushTExp02aDiT",
    ("pusht", "exp02a"):      "configs.pusht.exp02a_dit.PushTExp02aDiT",
    ("pusht", "exp03"):       "configs.pusht.exp03_multiscale.PushTExp03MultiScale",
    ("pusht", "exp04"):       "configs.pusht.exp04_vit.PushTExp04ViT",
    ("pusht", "exp04_vit"):   "configs.pusht.exp04_vit.PushTExp04ViT",
    ("pusht", "exp05"):       "configs.pusht.exp05_weighted.PushTExp05Weighted",
    ("pusht", "exp05_weighted"): "configs.pusht.exp05_weighted.PushTExp05Weighted",
    # ALOHA — bimanual 14-DOF transfer cube
    ("aloha", "exp01"):          "configs.aloha.exp01_baseline.AlohaExp01",
    ("aloha", "exp01_baseline"): "configs.aloha.exp01_baseline.AlohaExp01",
    ("aloha", "exp02"):          "configs.aloha.exp02_openloop.AlohaExp02OpenLoop",
    ("aloha", "exp02_openloop"): "configs.aloha.exp02_openloop.AlohaExp02OpenLoop",
    # Language Table — language-conditioned 2D block pushing
    ("language_table", "exp01"):          "configs.language_table.exp01_baseline.LTExp01",
    ("language_table", "exp01_baseline"): "configs.language_table.exp01_baseline.LTExp01",
    # LIBERO — language-conditioned 7-DoF manipulation (libero_spatial)
    ("libero", "exp01"):          "configs.libero.exp01_baseline.LiberoExp01",
    ("libero", "exp01_baseline"): "configs.libero.exp01_baseline.LiberoExp01",
}

# Legacy integer → (dataset, exp_id) mapping for backward compat
_LEGACY_MAP: dict[int, tuple[str, str]] = {
    1: ("pusht", "exp01"),
    2: ("pusht", "exp02a"),
    3: ("pusht", "exp03"),
    5: ("pusht", "exp05"),
}


def get_config(dataset: str, exp_id: str):
    """
    Return a config instance for the given dataset + experiment ID.

    Args:
        dataset: "pusht", "aloha", "language_table", "ood"
        exp_id:  "exp01", "exp02a", "exp03", etc.

    Returns:
        Config dataclass instance (subclass of BaseVLAConfig)

    Raises:
        ValueError if (dataset, exp_id) is not registered.
    """
    key = (dataset.lower(), exp_id.lower())
    if key not in _REGISTRY:
        available = [f"  {d}/{e}" for (d, e) in sorted(_REGISTRY.keys())]
        raise ValueError(
            f"Unknown config ({dataset!r}, {exp_id!r}). "
            f"Available:\n" + "\n".join(available)
        )

    module_path, class_name = _REGISTRY[key].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_config_legacy(exp_int: int, dataset: str = "pusht"):
    """
    Backward-compat shim: accepts old integer experiment IDs.

    get_config_legacy(1) → PushTExp01MLP
    get_config_legacy(2) → PushTExp02aDiT  (best)
    get_config_legacy(3) → PushTExp03MultiScale
    """
    if exp_int not in _LEGACY_MAP:
        raise ValueError(
            f"Legacy exp={exp_int} not found. "
            f"Valid: {sorted(_LEGACY_MAP.keys())}. "
            f"Or use get_config(dataset, exp_id) instead."
        )
    mapped_dataset, exp_id = _LEGACY_MAP[exp_int]
    if mapped_dataset != dataset:
        raise ValueError(
            f"Legacy exp={exp_int} maps to dataset={mapped_dataset!r}, "
            f"but you specified dataset={dataset!r}."
        )
    return get_config(mapped_dataset, exp_id)


def list_configs() -> list[tuple[str, str]]:
    """Return all registered (dataset, exp_id) pairs."""
    return sorted(_REGISTRY.keys())
