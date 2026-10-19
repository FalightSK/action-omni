"""Compatibility imports; the shared implementation lives in models.backbones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.backbones import (  # noqa: F401,E402
    ARMS, DEPTHS, DOC_LAYER, GATE_KEYS, KEYS, MODELS, PAIRS, POOLS, ROOT,
    Backbone, load_backbone,
)
