"""Optional intermediate-checkpoint control; matches exp05/06 within ALOHA."""
from dataclasses import dataclass
from configs.aloha.exp05_groot_transfer import AlohaExp05GrootTransfer


@dataclass
class AlohaExp07CosmosTransfer(AlohaExp05GrootTransfer):
    vlm_backbone: str = "cosmos"
    output_dir: str = "asset/runs/aloha/exp07_cosmos_transfer"
    embeddings_cache: str = "asset/runs/aloha/exp07_cosmos_transfer/vlm_embeddings.h5"
