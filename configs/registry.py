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
    # PushT — best architecture (DiT)
    ("pusht", "exp02"):       "configs.pusht.exp02a_dit.PushTExp02aDiT",
    ("pusht", "exp02a"):      "configs.pusht.exp02a_dit.PushTExp02aDiT",
    # ALOHA — bimanual 14-DOF; transfer-cube baseline + peg insertion (both kept)
    ("aloha", "exp01"):          "configs.aloha.exp01_baseline.AlohaExp01",
    ("aloha", "exp01_baseline"): "configs.aloha.exp01_baseline.AlohaExp01",
    ("aloha", "exp03"):           "configs.aloha.exp03_insertion.AlohaExp03Insertion",
    ("aloha", "exp03_insertion"): "configs.aloha.exp03_insertion.AlohaExp03Insertion",
    # LIBERO-Goal — 10 goals over one fixed scene; the language testbed
    ("libero", "exp01"):      "configs.libero.exp01_goal.LiberoExp01Goal",
    ("libero", "exp01_goal"): "configs.libero.exp01_goal.LiberoExp01Goal",
    # Exp02 is the H1 control: stock PaliGemma-3B, the checkpoint Pi-0.5 was
    # finetuned from. Identical head/data/read-layer/chunking, so the pair
    # differs only in whether the VLM ever saw robot data.
    ("libero", "exp02"):      "configs.libero.exp02_paligemma.LiberoExp02PaliGemma",
    ("libero", "exp02_paligemma"): "configs.libero.exp02_paligemma.LiberoExp02PaliGemma",
    # Exp03/exp04 repeat H1 in the Qwen3-VL family, where BOTH arms are
    # text-blind (Chapter 1 text-visibility 0.0000) so neither has backbone
    # fusion for our head to render redundant — removing the asymmetry that
    # makes the Pi-0.5/PaliGemma pair arguable.
    ("libero", "exp03"):      "configs.libero.exp03_groot.LiberoExp03Groot",
    ("libero", "exp03_groot"): "configs.libero.exp03_groot.LiberoExp03Groot",
    ("libero", "exp04"):      "configs.libero.exp04_qwen3vl.LiberoExp04Qwen3VL",
    ("libero", "exp04_qwen3vl"): "configs.libero.exp04_qwen3vl.LiberoExp04Qwen3VL",
    # exp05/exp06 repeat the exp03/exp04 pair with BOTH cameras — the observation
    # spec every published LIBERO pipeline uses. Single-view puts the
    # robot-pretrained arm outside its trained input configuration while leaving
    # the stock arm inside its own, an asymmetry pointing the same way as the
    # result, so only this pair can test H1 cleanly.
    # ALOHA transfer-cube: the LIBERO pretraining null retested on a bimanual
    # 14-DOF embodiment with a joint-space action space — a venue inside GR00T's
    # pretraining distribution, so biased toward finding a pretraining effect.
    ("aloha", "exp05"):       "configs.aloha.exp05_groot_transfer.AlohaExp05GrootTransfer",
    ("aloha", "exp05_groot_transfer"): "configs.aloha.exp05_groot_transfer.AlohaExp05GrootTransfer",
    ("aloha", "exp06"):       "configs.aloha.exp06_qwen3vl_transfer.AlohaExp06Qwen3VLTransfer",
    ("aloha", "exp06_qwen3vl_transfer"): "configs.aloha.exp06_qwen3vl_transfer.AlohaExp06Qwen3VLTransfer",
    ("libero", "exp05"):      "configs.libero.exp05_groot_2view.LiberoExp05Groot2View",
    ("libero", "exp05_groot_2view"): "configs.libero.exp05_groot_2view.LiberoExp05Groot2View",
    ("libero", "exp06"):      "configs.libero.exp06_qwen3vl_2view.LiberoExp06Qwen3VL2View",
    ("libero", "exp06_qwen3vl_2view"): "configs.libero.exp06_qwen3vl_2view.LiberoExp06Qwen3VL2View",
    # Language Table — curated 50/50 reach+push subset
    ("language_table", "exp03"):          "configs.language_table.exp03_curated.LTExp03Curated",
    ("language_table", "exp03_curated"):  "configs.language_table.exp03_curated.LTExp03Curated",
}

# Legacy integer → (dataset, exp_id) mapping for backward compat
_LEGACY_MAP: dict[int, tuple[str, str]] = {
    2: ("pusht", "exp02a"),
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
