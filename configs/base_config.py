"""
configs/base_config.py
──────────────────────
Shared base configuration for all VLA experiments across all datasets.
Dataset-specific configs inherit from BaseVLAConfig and override fields.

Fields that MUST be set in each subclass:
  dataset_path, output_dir, embeddings_cache,
  state_dim, action_dim, action_mean, action_std, state_mean, state_std,
  img_seq_len, img_grid_h, img_grid_w
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class BaseVLAConfig:
    """
    Base VLA config — fields shared across all datasets.
    Subclass and override at minimum: paths, dimensions, normalisation stats.
    """

    # ── Paths (MUST override per dataset/experiment) ──────────────────────────
    dataset_path:     str = ""
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = ""
    embeddings_cache: str = ""

    # ── Task ──────────────────────────────────────────────────────────────────
    task_text: str = ""

    # ── Data dimensions (MUST override per dataset) ───────────────────────────
    state_dim:  int = 2
    action_dim: int = 2

    # ── Action horizons ───────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4   # HARD CONSTRAINT: must match training value

    # ── Action representation ─────────────────────────────────────────────────
    use_relative_actions: bool = True

    # ── Normalisation stats (MUST override per dataset) ───────────────────────
    action_mean: tuple = (0.0,)
    action_std:  tuple = (1.0,)
    state_mean:  tuple = (0.0,)
    state_std:   tuple = (1.0,)

    # ── VLM config (frozen — same for all datasets) ───────────────────────────
    freeze_vlm:      bool = True
    vlm_hidden_size: int  = 1024     # Qwen3.5-0.8B hidden dim
    image_token_id:  int  = 248056   # Qwen image token id

    # Token sequence config (MUST override to match dataset camera resolution)
    # img_seq_len = img_grid_h * img_grid_w + n_text_tokens
    img_seq_len: int = 82   # 8×8 image grid + ~18 text tokens (PushT default)
    img_grid_h:  int = 8
    img_grid_w:  int = 8

    # Which VLM layer(s) to extract. Single = (28,), multi-scale = (14, 21, 28)
    vlm_extract_layers: tuple = (28,)

    @property
    def n_vlm_layers(self) -> int:
        return len(self.vlm_extract_layers)

    # Fusion mode — only used when n_vlm_layers > 1
    #   "linear"   — MultiScaleFusion: Linear(n×d → d) + LN  ⚠ Exp03 only,
    #                  ~3.1M params (96× LoRA) → displaces LoRA (INVALID for new work)
    #   "weighted" — WeightedLayerFusion: softmax(n scalars)  ✅ Exp05+,
    #                  n_layers params (3 params << 32K LoRA) → LoRA stays load-bearing
    # Hard constraint: fusion capacity must be < LoRA capacity (< 32K params)
    fusion_mode: str = "linear"

    # ── Trainable VLM Token Adapter ───────────────────────────────────────────
    vlm_adapter_dim: int   = 512
    lora_rank:       int   = 16
    lora_scale:      float = 0.1
    pos_enc_dim:     int   = 128
    readout_heads:   int   = 8
    adapter_dropout: float = 0.25

    # ── Decoder selection ─────────────────────────────────────────────────────
    use_dit_decoder: bool = True    # True = DiT (Exp2+), False = MLP (Exp1)

    # MLP decoder params
    decoder_hidden_dim: int   = 512
    decoder_num_layers: int   = 6
    decoder_dropout:    float = 0.10

    # DiT decoder params
    dit_hidden_dim:  int = 256
    dit_num_layers:  int = 6
    dit_num_heads:   int = 8

    num_flow_steps: int = 3

    # ── Conditioning dimension (derived) ──────────────────────────────────────
    @property
    def cond_dim(self) -> int:
        return self.vlm_adapter_dim + self.state_dim

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:      int   = 20
    sim_max_steps:     int   = 300
    success_threshold: float = 0.95
    video_fps:         int   = 10

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:          int   = 256
    num_epochs:          int   = 300
    learning_rate:       float = 3e-4
    weight_decay:        float = 1e-2
    grad_clip:           float = 1.0
    warmup_steps:        int   = 300
    log_every:           int   = 100
    save_every:          int   = 25
    early_stop_patience: int   = 50

    # ── Device ────────────────────────────────────────────────────────────────
    device:           str = "auto"
    embed_batch_size: int = 8

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)
