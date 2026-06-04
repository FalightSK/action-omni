"""
data/
─────
Dataset loaders for all VLA environments.

Usage:
    from data.pusht import PushTDataset, PushTEmbeddingDataset
    # from data.aloha import AlohaDataset, AlohaEmbeddingDataset  (future)

Legacy imports (backward compat):
    from data import PushTDataset  # still works via pusht subpackage
"""
# Legacy compat — redirect to subpackage
from data.pusht.dataset import PushTDataset, PushTEmbeddingDataset

__all__ = ["PushTDataset", "PushTEmbeddingDataset"]
