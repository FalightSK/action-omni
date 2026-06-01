from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class VLAConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    dataset_path:      str = "asset/dataset/pusht_dataset"
    model_path:        str = "asset/model/Qwen3.5-0.8B"
    output_dir:        str = "asset/result_exp2"
    # Reuse Exp1 cache — same layer 28, same 2D state, no re-run needed.
    embeddings_cache:  str = "asset/result/vlm_embeddings.pt"

    # ── Task ─────────────────────────────────────────────────────────────────
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Data dimensions ───────────────────────────────────────────────────────
    # Exp2a: 2D state (agent position only) — same as Exp1.
    # Drop delta history to eliminate covariate shift.
    # Golden rule: NEVER add block or goal position to state.
    state_dim:  int = 2    # agent (x, y) only
    action_dim: int = 2

    # ── Action horizons ───────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4

    # ── Action representation: relative delta (action - state) ────────────────
    use_relative_actions: bool = True

    # Normalisation
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── VLM config (always frozen) ────────────────────────────────────────────
    freeze_vlm:      bool = True
    vlm_hidden_size: int  = 1024

    image_token_id: int = 248056
    img_seq_len:    int = 82
    img_grid_h:     int = 8
    img_grid_w:     int = 8

    # Single final layer — same as Exp1 for fair decoder comparison.
    # Multi-scale extraction (14, 21, 28) deferred to Exp3.
    vlm_extract_layers: tuple = (28,)

    @property
    def n_vlm_layers(self) -> int:
        return len(self.vlm_extract_layers)

    # ── Trainable VLM Token Adapter ───────────────────────────────────────────
    vlm_adapter_dim: int   = 512
    lora_rank:       int   = 16
    lora_scale:      float = 0.1
    pos_enc_dim:     int   = 128
    readout_heads:   int   = 8
    adapter_dropout: float = 0.25

    # ── Flow-matching decoder — choose MLP (Exp1) or DiT (Exp2+) ─────────────
    use_dit_decoder: bool = True    # DiT for Exp2; False = MLP (Exp1 baseline)

    # MLP decoder params (kept for fallback / Exp1 reproduction)
    decoder_hidden_dim: int   = 512
    decoder_num_layers: int   = 6
    decoder_dropout:    float = 0.10

    # DiT decoder params
    dit_hidden_dim:  int = 256   # token embedding dim inside DiT
    dit_num_layers:  int = 6     # number of DiT blocks
    dit_num_heads:   int = 8     # attention heads (256 / 8 = 32 per head)
    # cross_attn_dim is derived: = vlm_adapter_dim (512)
    # Each action step cross-attends to all 82 adapted VLM tokens.

    num_flow_steps: int = 3

    # ── Conditioning dimension (derived) ─────────────────────────────────────
    # cond = adapted_embed (vlm_adapter_dim) ‖ norm_state (state_dim)
    @property
    def cond_dim(self) -> int:
        return self.vlm_adapter_dim + self.state_dim   # 512 + 6 = 518

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:       int   = 20
    sim_max_steps:      int   = 300
    success_threshold:  float = 0.95
    video_fps:          int   = 10

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
