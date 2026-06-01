"""
config_exp1.py — Frozen snapshot of Experiment 1 configuration.

This file is read-only historical record.  Do not modify it.
To reproduce Experiment 1 results, pass --exp 1 to any script:

  python train.py      --exp 1
  python evaluate.py   --exp 1
  python inference.py  --exp 1

Checkpoint : asset/result/checkpoints/best.pt  (epoch=152, val_loss=0.4490)
Cache      : asset/result/vlm_embeddings.pt    (v2, 4.0 GB)
Results    : 25% SR, 87.2% mean max coverage (20 episodes, threshold=0.95)
"""

from dataclasses import dataclass
import torch


@dataclass
class VLAConfigExp1:
    # ── Paths ────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht_dataset"
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = "asset/result"
    embeddings_cache: str = "asset/result/vlm_embeddings.pt"

    # ── Task ─────────────────────────────────────────────────────────────────
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Data dimensions ───────────────────────────────────────────────────────
    state_dim:  int = 2    # agent (x, y) only
    action_dim: int = 2

    # ── Action horizons ───────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4

    # ── Action representation ─────────────────────────────────────────────────
    use_relative_actions: bool = True
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── VLM (frozen) ──────────────────────────────────────────────────────────
    freeze_vlm:      bool = True
    vlm_hidden_size: int  = 1024
    image_token_id:  int  = 248056
    img_seq_len:     int  = 82
    img_grid_h:      int  = 8
    img_grid_w:      int  = 8

    # Single final layer — no multi-scale (Exp1 baseline)
    vlm_extract_layers: tuple = (28,)

    @property
    def n_vlm_layers(self) -> int:
        return 1   # always single-layer for exp1

    # ── Adapter ───────────────────────────────────────────────────────────────
    vlm_adapter_dim: int   = 512
    lora_rank:       int   = 16
    lora_scale:      float = 0.1
    pos_enc_dim:     int   = 128
    readout_heads:   int   = 8
    adapter_dropout: float = 0.25

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder_hidden_dim: int   = 512
    decoder_num_layers: int   = 6
    decoder_dropout:    float = 0.10
    num_flow_steps:     int   = 3

    @property
    def cond_dim(self) -> int:
        return self.vlm_adapter_dim + self.state_dim   # 514

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
