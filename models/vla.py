"""
VLA model: Qwen3.5-0.8B (frozen)  +  VLMTokenAdapter  +  FlowMatchingDecoder.

Architecture — Experiment 2 (B+C)
───────────────────────────────────

  PIL Image + task text
        │
  Qwen3.5-0.8B  ❄ FROZEN   (853M params, bfloat16)
        │  hidden_states[14, 21, 28]  →  (B, 3, 82, 1024)   [Exp C]
        │
  ┌─────────────────────────────────────────────────────────┐
  │  VLMTokenAdapter  🔥 TRAINABLE                          │
  │                                                         │
  │  Stage 0 – MultiScaleFusion  (NEW — Exp C)              │
  │    Linear(3×1024 → 1024) + LayerNorm per token          │
  │    fuses early/mid/final VLM representations            │
  │                                                         │
  │  Stage 1 – PerTokenLoRA                                 │
  │    h_i' = h_i + scale * B(A(h_i))   for ALL 82 tokens  │
  │                                                         │
  │  Stage 2 – SpatialAwareMLP  (DINO-style geometric)      │
  │    image tokens : cat([h_i', sincos_2D(row, col)])      │
  │    text  tokens : cat([h_i', sincos_1D(seq_pos)])       │
  │    → MLP(1024+128 → adapter_dim) per token              │
  │                                                         │
  │  Stage 3 – AttentionReadout  (replaces mean-pool)       │
  │    learnable query Q attends to all 82 projected tokens │
  │    → single context vector (B, adapter_dim)             │
  └─────────────────────────────────────────────────────────┘
        │  context (B, 512)
        │
  concat 6D state  →  cond (B, 518)   [Exp B: was 514 with 2D state]
  state = [agent_x, agent_y, dΔx₁, dΔy₁, dΔx₂, dΔy₂]
        │
  FlowMatchingDecoder  🔥 TRAINABLE  (512-dim, 6-layer)
        │
  relative delta actions (B, horizon, 2)

Cache format v3 (precompute_embeddings.py)
───────────────────────────────────────────
  embeddings : (N, 3, 82, 1024)  bfloat16   ← multi-scale (Exp C)
  img_masks  : (N, 82)           bool
  states     : (N, 6)            float32    ← extended state (Exp B)
  actions    : (N, H, 2)         float32
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel

from .flow_matching import FlowMatchingDecoder


# ──────────────────────────────────────────────────────────────────────────────
# Stage 0 — Multi-scale VLM feature fusion (Experiment C)
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleFusion(nn.Module):
    """
    Fuses hidden states from multiple VLM layers into a single per-token vector.

    Input  : (B, n_layers, seq_len, vlm_dim)
    Output : (B, seq_len, vlm_dim)

    Each token gets all n_layers representations concatenated then projected back
    to vlm_dim via a single linear + LayerNorm.  This lets the model use early
    (low-level visual), mid, and final (semantic) Qwen features simultaneously.
    """

    def __init__(self, n_layers: int, vlm_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_layers * vlm_dim, vlm_dim, bias=False)
        self.norm = nn.LayerNorm(vlm_dim)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_layers, seq, vlm_dim)
        B, L, S, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, S, L * D)  # (B, seq, n_layers*vlm_dim)
        return self.norm(self.proj(x))                    # (B, seq, vlm_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Per-token LoRA (correct application: before pooling)
# ──────────────────────────────────────────────────────────────────────────────

class PerTokenLoRA(nn.Module):
    """
    LoRA applied per-token on the full sequence BEFORE any pooling.

    input  : (B, seq_len, dim)   raw Qwen hidden states
    output : (B, seq_len, dim)   corrected hidden states

    Each token is independently updated:
        h_i' = h_i  +  scale * B( A(h_i) )

    This is the correct LoRA application.  Applying LoRA after mean-pooling
    (as in the previous design) only corrects the average, losing all
    per-token information that LoRA is supposed to preserve and refine.
    """

    def __init__(self, dim: int, rank: int = 16, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)   # zero-init → identity at start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, dim)  — works for any leading batch dims
        return x + self.scale * self.B(self.A(x))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — DINO-style geometric position encoding
# ──────────────────────────────────────────────────────────────────────────────

class SpatialPositionEncoding2D(nn.Module):
    """
    2D sinusoidal position encoding — geometric awareness for image patches.

    Follows Meta's DINO / DINOv2 design: each image patch at grid position
    (row, col) receives a dedicated PE that encodes its spatial location.
    This informs the downstream MLP of *where* in the image each token came from,
    preventing the projection from treating all patches identically.

    PE(row, col) = [ sin(row/f_0), cos(row/f_0), …,   ← height encoding (dim/2)
                     sin(col/f_0), cos(col/f_0), … ]   ← width  encoding (dim/2)

    dim must be divisible by 4.
    """

    def __init__(self, dim: int, max_h: int = 16, max_w: int = 16) -> None:
        super().__init__()
        assert dim % 4 == 0, "pos_enc_dim must be divisible by 4"
        half    = dim // 2
        quarter = half // 2
        freqs   = torch.exp(
            -math.log(10_000.0)
            * torch.arange(quarter, dtype=torch.float32) / max(1, quarter - 1)
        )
        h_pos = torch.arange(max_h, dtype=torch.float32)
        w_pos = torch.arange(max_w, dtype=torch.float32)

        # height encoding  (max_h, half)
        h_args = h_pos[:, None] * freqs[None]    # (max_h, quarter)
        h_enc  = torch.cat([h_args.sin(), h_args.cos()], dim=-1)  # (max_h, half)

        # width encoding   (max_w, half)
        w_args = w_pos[:, None] * freqs[None]    # (max_w, quarter)
        w_enc  = torch.cat([w_args.sin(), w_args.cos()], dim=-1)  # (max_w, half)

        self.register_buffer("h_enc", h_enc)   # (max_h, half)
        self.register_buffer("w_enc", w_enc)   # (max_w, half)
        self.max_h = max_h
        self.max_w = max_w

    def forward(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Returns 2D PE for all patches in a (grid_h × grid_w) grid.
        Output shape: (grid_h * grid_w, dim)
        """
        h_pe = self.h_enc[:grid_h]                           # (grid_h, half)
        w_pe = self.w_enc[:grid_w]                           # (grid_w, half)
        h_pe = h_pe.unsqueeze(1).expand(-1, grid_w, -1)     # (grid_h, grid_w, half)
        w_pe = w_pe.unsqueeze(0).expand(grid_h, -1, -1)     # (grid_h, grid_w, half)
        return torch.cat([h_pe, w_pe], dim=-1).reshape(grid_h * grid_w, -1)


class SinusoidalPE1D(nn.Module):
    """1D sinusoidal PE for text tokens (by sequence position)."""

    def __init__(self, dim: int, max_len: int = 128) -> None:
        super().__init__()
        assert dim % 2 == 0
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, dtype=torch.float32) / max(1, half - 1)
        )
        pos  = torch.arange(max_len, dtype=torch.float32)
        args = pos[:, None] * freqs[None]                           # (max_len, half)
        enc  = torch.cat([args.sin(), args.cos()], dim=-1)          # (max_len, dim)
        self.register_buffer("enc", enc)

    def forward(self, n: int) -> torch.Tensor:
        """Returns PE for n consecutive positions. Shape: (n, dim)"""
        return self.enc[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Spatial-aware MLP
# ──────────────────────────────────────────────────────────────────────────────

class SpatialAwareMLP(nn.Module):
    """
    Projects each token to adapter_dim while injecting geometric position info.

    Image tokens : cat( [h_i (1024),  2D_PE(row,col) (pos_dim)] ) → MLP → (adapter_dim,)
    Text  tokens : cat( [h_i (1024),  1D_PE(seq_pos) (pos_dim)] ) → MLP → (adapter_dim,)

    The position encoding is concatenated BEFORE the LayerNorm + Linear stack,
    so the normalisation and projection are aware of the token's spatial origin.
    This is the core of DINO-style geometric awareness applied to our setting.

    All tokens share the same MLP weights — the PE makes each token's projection
    spatially specialised without requiring separate networks.
    """

    def __init__(
        self,
        vlm_dim:     int = 1024,
        adapter_dim: int = 512,
        pos_dim:     int = 128,
        dropout:     float = 0.1,
        max_h:       int = 16,
        max_w:       int = 16,
        max_seq:     int = 128,
    ) -> None:
        super().__init__()
        in_dim = vlm_dim + pos_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )
        self.pos2d = SpatialPositionEncoding2D(pos_dim, max_h, max_w)
        self.pos1d = SinusoidalPE1D(pos_dim, max_seq)

    def forward(
        self,
        tokens:   torch.Tensor,   # (B, seq_len, vlm_dim) — after LoRA
        img_mask: torch.Tensor,   # (B, seq_len) bool — True for image tokens
        grid_h:   int,
        grid_w:   int,
    ) -> torch.Tensor:            # (B, seq_len, adapter_dim)
        B, S, D = tokens.shape
        dev = tokens.device
        dtype = tokens.dtype

        # Build per-token position encodings
        pos_enc = torch.zeros(B, S, self.pos2d.h_enc.shape[-1] * 2,
                              device=dev, dtype=dtype)
        img_pe  = self.pos2d(grid_h, grid_w).to(dtype)   # (n_img, pos_dim)
        txt_pe  = self.pos1d(S).to(dtype)                 # (S, pos_dim)  — upper bound

        for b in range(B):
            img_pos = img_mask[b].nonzero(as_tuple=True)[0]  # image token indices
            txt_pos = (~img_mask[b]).nonzero(as_tuple=True)[0]

            n_img = min(len(img_pos), grid_h * grid_w)
            if n_img > 0:
                pos_enc[b, img_pos[:n_img]] = img_pe[:n_img]

            n_txt = min(len(txt_pos), txt_pe.shape[0])
            if n_txt > 0:
                pos_enc[b, txt_pos[:n_txt]] = txt_pe[:n_txt]

        x = torch.cat([tokens, pos_enc], dim=-1)   # (B, S, vlm_dim + pos_dim)
        return self.mlp(x)                          # (B, S, adapter_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Learned attention readout (replaces mean-pool)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionReadout(nn.Module):
    """
    Replaces mean-pooling with a learned cross-attention aggregation.

    A single learnable query vector attends to all projected tokens,
    producing one context vector that weights tokens by relevance
    rather than treating all 82 positions equally.

    This is analogous to a CLS token but added after the projection
    (so it doesn't need to be trained into Qwen — fully post-hoc).

    Q  = learnable (1, 1, adapter_dim)
    K  = V = projected tokens  (B, seq_len, adapter_dim)
    out = softmax(QK^T/√d) V   → (B, adapter_dim)
    """

    def __init__(self, dim: int, n_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(dim)

    def forward(
        self,
        tokens:           torch.Tensor,          # (B, seq_len, adapter_dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, seq_len) True=ignore
    ) -> torch.Tensor:                            # (B, adapter_dim)
        B = tokens.shape[0]
        Q = self.query.expand(B, -1, -1)          # (B, 1, adapter_dim)
        out, _ = self.attn(Q, tokens, tokens,
                           key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))           # (B, adapter_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Full corrected adapter
# ──────────────────────────────────────────────────────────────────────────────

class VLMTokenAdapter(nn.Module):
    """
    Adapter for frozen VLM token sequences.  Supports single-layer (v2) and
    multi-scale (v3, Experiment C) input formats.

    Input  : tokens (B, seq_len, 1024)          — single layer (v2)
           : tokens (B, n_layers, seq_len, 1024) — multi-scale (v3)
           : img_mask (B, seq_len) bool
    Output : context (B, adapter_dim)

    Stage 0 — MultiScaleFusion  : fuse n_layers → 1 per token  [only if n_layers > 1]
    Stage 1 — PerTokenLoRA      : corrects each token independently
    Stage 2 — SpatialAwareMLP   : projects with geometric position awareness
    Stage 3 — AttentionReadout  : learned aggregation over all tokens
    """

    def __init__(
        self,
        vlm_dim:      int   = 1024,
        adapter_dim:  int   = 512,
        lora_rank:    int   = 16,
        lora_scale:   float = 0.1,
        pos_dim:      int   = 128,
        readout_heads: int  = 8,
        dropout:      float = 0.1,
        img_grid_h:   int   = 8,
        img_grid_w:   int   = 8,
        n_vlm_layers: int   = 1,
    ) -> None:
        super().__init__()
        self.grid_h = img_grid_h
        self.grid_w = img_grid_w

        # Stage 0: multi-scale fusion (only built when n_layers > 1)
        self.fusion  = MultiScaleFusion(n_vlm_layers, vlm_dim) if n_vlm_layers > 1 else None

        self.lora    = PerTokenLoRA(vlm_dim, lora_rank, lora_scale)
        self.spatial = SpatialAwareMLP(vlm_dim, adapter_dim, pos_dim,
                                        dropout, img_grid_h * 2, img_grid_w * 2)
        self.readout = AttentionReadout(adapter_dim, readout_heads)

        n = sum(p.numel() for p in self.parameters())
        print(f"   VLMTokenAdapter: {n:,} params  "
              f"(LoRA rank={lora_rank}, adapter_dim={adapter_dim}, "
              f"pos_dim={pos_dim}, readout_heads={readout_heads}, "
              f"n_vlm_layers={n_vlm_layers})")

    def forward(
        self,
        tokens:       torch.Tensor,        # (B, seq, 1024) or (B, n_layers, seq, 1024)
        img_mask:     torch.Tensor,        # (B, seq) bool
        return_tokens: bool = False,       # also return pre-readout tokens for DiT cross-attn
    ) -> torch.Tensor:                     # (B, adapter_dim)  or  tuple
        # Stage 0: fuse multi-scale layers if present
        if tokens.ndim == 4:
            tokens = self.fusion(tokens)          # (B, seq, 1024)

        # Stage 1: per-token LoRA
        h = self.lora(tokens)                     # (B, seq, 1024)

        # Stage 2: spatial-aware MLP — h is now (B, seq, adapter_dim=512)
        h = self.spatial(h, img_mask, self.grid_h, self.grid_w)

        # Stage 3: attention readout → single context vector
        context = self.readout(h)                 # (B, adapter_dim)

        if return_tokens:
            # h: (B, seq, adapter_dim) — geometrically-aware per-token features
            # Used by DiTFlowDecoder for cross-attention: each action step can
            # query any of the 82 spatially-encoded VLM tokens at every denoising step.
            return context, h
        return context


# ──────────────────────────────────────────────────────────────────────────────
# Full VLA model (used during live inference only — precompute bypasses this)
# ──────────────────────────────────────────────────────────────────────────────

class VLAModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # ── Frozen Qwen3.5-0.8B ──────────────────────────────────────────
        print(f"  Loading VLM from {config.model_path} …")
        self.processor = AutoProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self.vlm = AutoModel.from_pretrained(
            config.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        )
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self.vlm.eval()
        print("  VLM frozen (853M params).")

        # ── Trainable VLM Token Adapter ──────────────────────────────────
        self.vlm_adapter = VLMTokenAdapter(
            vlm_dim      = config.vlm_hidden_size,
            adapter_dim  = config.vlm_adapter_dim,
            lora_rank    = config.lora_rank,
            lora_scale   = config.lora_scale,
            pos_dim      = config.pos_enc_dim,
            readout_heads= config.readout_heads,
            dropout      = config.adapter_dropout,
            img_grid_h   = config.img_grid_h,
            img_grid_w   = config.img_grid_w,
            n_vlm_layers = config.n_vlm_layers,
        )

        # ── Flow-matching decoder ────────────────────────────────────────
        self.flow_decoder = FlowMatchingDecoder(
            action_dim = config.action_dim * config.action_horizon,
            cond_dim   = config.cond_dim,
            hidden_dim = config.decoder_hidden_dim,
            num_layers = config.decoder_num_layers,
            dropout    = config.decoder_dropout,
        )
        n_flow = sum(p.numel() for p in self.flow_decoder.parameters())
        print(f"   FlowMatchingDecoder: {n_flow:,} params")

    # ── VLM encoding ──────────────────────────────────────────────────────

    def build_vlm_inputs(
        self, images: list, task_texts: list[str], device: torch.device
    ) -> dict:
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  txt},
            ]}]
            for img, txt in zip(images, task_texts)
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]
        inp = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inp.items()}

    @torch.no_grad()
    def encode_vlm(self, vlm_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          tokens   : (B, seq_len, vlm_hidden_size)           — single layer (n_vlm_layers==1)
                   : (B, n_layers, seq_len, vlm_hidden_size)  — multi-scale (n_vlm_layers>1)
          img_mask : (B, seq_len) bool
        """
        layers = self.config.vlm_extract_layers
        if len(layers) > 1:
            out = self.vlm(**vlm_inputs, return_dict=True, output_hidden_states=True)
            tokens = torch.stack(
                [out.hidden_states[l].float() for l in layers], dim=1
            )  # (B, n_layers, S, 1024)
        else:
            out    = self.vlm(**vlm_inputs, return_dict=True)
            tokens = out.last_hidden_state.float()                  # (B, S, 1024)
        img_mask = (vlm_inputs["input_ids"] == self.config.image_token_id)
        return tokens, img_mask

    # ── End-to-end inference (used in inference.py) ───────────────────────

    @torch.no_grad()
    def predict(
        self,
        image,
        state:     torch.Tensor,
        device:    torch.device,
        adapter:   "VLMTokenAdapter",
        decoder:   FlowMatchingDecoder,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Returns predicted RELATIVE delta actions (horizon, 2) — normalised.
        adapter and decoder are the trained VLATrainModel sub-modules.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device)

        inputs           = self.build_vlm_inputs([image], [self.config.task_text], device)
        tokens, img_mask = self.encode_vlm(inputs)           # multi-scale or single

        adapted = adapter(tokens, img_mask)                  # (1, 512)
        cond    = torch.cat([adapted, state], dim=-1)        # (1, 514)

        steps = num_steps or self.config.num_flow_steps
        flat  = decoder.sample(cond, num_steps=steps)
        return flat.view(self.config.action_horizon, self.config.action_dim)
