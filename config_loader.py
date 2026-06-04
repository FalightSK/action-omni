"""
config_loader.py  [BACKWARD-COMPAT SHIM]
─────────────────────────────────────────
Preserved so old scripts that call get_config(exp_int) still work unchanged.
Delegates directly to configs.registry — no longer depends on flat config files.

New code should use:
    from configs.registry import get_config
    cfg = get_config("pusht", "exp02a")
"""
from __future__ import annotations
from configs.registry import get_config_legacy


def get_config(exp: int = 2):
    """Backward-compat integer → config. Delegates to configs.registry."""
    return get_config_legacy(exp, dataset="pusht")
