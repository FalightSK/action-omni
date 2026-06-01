"""
config_loader.py
────────────────
Central config factory.  All scripts call get_config(exp) instead of
instantiating VLAConfig directly.

Usage in scripts:
    parser.add_argument("--exp", type=int, default=2, choices=[1, 2])
    cfg = get_config(args.exp)
"""

from __future__ import annotations


def get_config(exp: int = 2):
    """
    Return the VLAConfig dataclass for the requested experiment.

      exp=1  →  VLAConfigExp1  (read-only historical snapshot)
               cache : asset/result/vlm_embeddings.pt  (v2, 4.0 GB)
               ckpt  : asset/result/checkpoints/best.pt

      exp=2  →  VLAConfig      (current experiment B+C)
               cache : asset/result_exp2/vlm_embeddings.pt  (v3, ~12.8 GB)
               ckpt  : asset/result_exp2/checkpoints/best.pt
    """
    if exp == 1:
        from config_exp1 import VLAConfigExp1
        return VLAConfigExp1()
    elif exp == 2:
        from config import VLAConfig
        return VLAConfig()
    else:
        raise ValueError(f"Unknown experiment: {exp}. Choose 1 or 2.")
