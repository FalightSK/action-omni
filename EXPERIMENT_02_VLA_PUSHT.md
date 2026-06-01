# Experiment 02 — VLA PushT: DiT Decoder + 6D Action-History State

**Date:** 2026-06-01 **Platform:** MacBook Pro M1 (MPS) **Status:** 🔄 In Progress

---

## 1. Objective

Isolate the contribution of two targeted improvements over the Exp1 baseline:

- **Change B** — Expand state input from 2D to 6D by including the last two executed action deltas, giving the decoder a sense of recent motion without violating the golden rule.
- **Change D** — Replace the MLP flat decoder with a Diffusion Transformer (DiT) decoder that treats each action step as a separate token, resolving the information bottleneck in the MLP design.

**Scope boundary (fair comparison):** Multi-scale VLM feature extraction (layers 14/21/28) is intentionally excluded here. Exp2 uses the same single-layer (28) embeddings as Exp1. Multi-scale fusion is deferred to Exp3.

**Golden Rule (permanent):** State input must NEVER include block position or destination/goal position.

---

## 2. Changes vs Experiment 1

ComponentExperiment 1Experiment 2VLM layers28 only**28 only** (same)State input`[agent_x, agent_y]` (2D)`[agent_x, agent_y, Δx₋₁, Δy₋₁, Δx₋₂, Δy₋₂]` (6D)DecoderMLP — flat 32D vector**DiT** — 16 action tokensSelf-attention❌✅ steps attend to each otherCross-attention❌✅ each step queries 82 VLM tokensadaLN conditioning❌✅ time + state modulates every blockCache path`asset/result/vlm_embeddings.ptasset/result_exp2/vlm_embeddings.pt`Cache formatv2 — `(N, 82, 1024)`v2 — `(N, 82, 1024)` + 6D statesTotal params16.2M\~16.1M

---

## 3. Model Architecture

### 3.1 Full Pipeline

```mermaid
flowchart TD
    IMG["🖼 PIL Image\n96×96 RGB"]
    TXT["📝 Task Text\n'Push T-block onto target'"]
    PROC["Qwen3VLProcessor\ninput_ids · pixel_values"]

    subgraph VLM ["❄️ Qwen3.5-0.8B — FROZEN (853M params, bfloat16)"]
        L28["Layer 28 hidden states\n(B, 82, 1024)"]
    end

    subgraph ADAPTER ["🔥 VLMTokenAdapter — TRAINABLE (~5.9M params)"]
        LORA["Stage 1 · PerTokenLoRA\nh_i' = h_i + scale·B(A(h_i))\nrank=16, scale=0.1\n(B, 82, 1024)"]
        SPATIAL["Stage 2 · SpatialAwareMLP\ncat(h_i', 2D_PE(row,col))\n→ MLP(1152→512)\n(B, 82, 512)"]
        READOUT["Stage 3 · AttentionReadout\n1 learnable query × 82 tokens\n→ context (B, 512)"]
    end

    subgraph STATE ["State — Change B"]
        S6D["6D State\n[agent_x, agent_y,\nΔx₋₁, Δy₋₁, Δx₋₂, Δy₋₂]\nnormalised"]
    end

    subgraph COND ["Conditioning"]
        CONCAT["cat(context, state)\n→ cond (B, 518)"]
    end

    subgraph DIT ["🔥 DiTFlowDecoder — TRAINABLE (~10.1M params)  ← Change D"]
        NOISE["Noisy actions x_t\n(B, 16, 2)"]
        APROJ["action_proj + pos_emb\n→ (B, 16, 256)"]
        TEMB["SinusoidalTimeEmb(t)\n→ time_emb (B, 256)"]
        CPROJ["cond_proj(cond)\n→ (B, 256)"]
        CEMB["cond_emb = time_emb + cond_proj\n(B, 256)"]

        subgraph BLOCKS ["× 6 DiT Blocks"]
            ADALN["adaLN\nLinear(256→6×256) zero-init\n→ scale/shift per LayerNorm"]
            SA["Self-Attention\nQ=K=V: (B,16,256)\nstep k attends to all 16 steps"]
            CA["Cross-Attention\nQ: (B,16,256)  KV: (B,82,512)\neach step queries all 82 VLM tokens"]
            FFN["FFN\n256→1024→256 + GELU"]
        end

        OUTNORM["LayerNorm"]
        OUTPROJ["Linear(256→2) zero-init\n→ velocity field (B, 16, 2)"]
    end

    EULER["Euler ODE (3 steps)\nx_{i+1} = x_i + v_θ·Δt"]
    ACT["Actions (B, 16, 2)\nrelative deltas, normalised"]

    IMG --> PROC
    TXT --> PROC
    PROC --> VLM
    VLM --> L28
    L28 --> LORA
    LORA --> SPATIAL
    SPATIAL --> READOUT
    SPATIAL -->|"vlm_tokens (B,82,512)\nfor cross-attention"| CA
    READOUT --> CONCAT
    S6D --> CONCAT
    CONCAT --> CPROJ
    TEMB --> CEMB
    CPROJ --> CEMB
    CEMB --> ADALN
    NOISE --> APROJ
    APROJ --> SA
    ADALN -->|"scale/shift"| SA
    ADALN -->|"scale/shift"| CA
    ADALN -->|"scale/shift"| FFN
    SA --> CA
    CA --> FFN
    FFN --> OUTNORM
    OUTNORM --> OUTPROJ
    OUTPROJ --> EULER
    EULER --> ACT
```

### 3.2 DiT Decoder Detail

```mermaid
flowchart LR
    subgraph INPUT ["Input at denoising step i"]
        XT["x_t: (B, 16, 2)\nnoisy action sequence"]
        T["t ∈ [0,1]\nflow time"]
        COND["cond: (B, 518)\nVLM context + 6D state"]
        VLM2["vlm_tokens: (B, 82, 512)\nspatially-encoded per-token features"]
    end

    subgraph PROJ ["Tokenise"]
        AP["action_proj Linear(2→256)"]
        PE["+ learnable pos_emb\n(1, 16, 256)"]
        H0["h: (B, 16, 256)"]
    end

    subgraph TEMB2 ["Time + Condition"]
        TE["SinusoidalTimeEmb\n+ MLP(256→1024→256)"]
        CE["cond_proj Linear(518→256)"]
        ADD["+ → cond_emb (B, 256)"]
    end

    subgraph BLOCK ["One DiT Block (×6)"]
        A1["adaLN: Linear(256→1536)\n→ (s1,h1, s_ca,h_ca, s2,h2)"]
        N1["LN(x)·(1+s1)+h1"]
        SA2["MultiheadAttn\n8 heads, dim=32/head\nself-attn over 16 steps"]
        N2["LN(x)·(1+s_ca)+h_ca"]
        CA2["MultiheadAttn\nQ=action(256) KV=vlm(512)\n→ direct visual access per step"]
        N3["LN(x)·(1+s2)+h2"]
        FF2["Linear 256→1024→256\n+ GELU + Dropout"]
    end

    OUT["LN → Linear(256→2)\n→ velocity v_θ: (B, 16, 2)"]

    XT --> AP --> PE --> H0
    T --> TE --> ADD
    COND --> CE --> ADD
    ADD --> A1
    H0 --> N1 --> SA2 --> N2 --> CA2
    VLM2 --> CA2
    CA2 --> N3 --> FF2 --> OUT
    A1 -->|"s1,h1"| N1
    A1 -->|"s_ca,h_ca"| N2
    A1 -->|"s2,h2"| N3
```

### 3.3 6D State Construction (Change B)

```mermaid
flowchart LR
    EP["Episode timeline: …t-2, t-1, t"]

    subgraph T2 ["Step t-2"]
        A2["action[t-2, 0]\nfirst step of chunk\n= executed delta Δ₋₂"]
    end

    subgraph T1 ["Step t-1"]
        A1b["action[t-1, 0]\n= executed delta Δ₋₁"]
    end

    subgraph Tnow ["Step t (current)"]
        POS["agent (x, y)\nfrom env state"]
        S6["6D state vector\n[x, y, Δx₋₁, Δy₋₁, Δx₋₂, Δy₋₂]"]
    end

    A2 --> S6
    A1b --> S6
    POS --> S6
    EP --> T2
    EP --> T1
    EP --> Tnow
```

---

## 4. Key Design Decisions

### Why DiT resolves the MLP bottleneck

The Exp1 MLP compressed the entire 16-step action chunk into a single 512D hidden state. Each step's prediction was globally pooled — step 8's velocity had no more direct access to step 7 than step 1 did.

The DiT resolves this in two ways:

1. **Self-attention**: Step 8 can explicitly attend to steps 6, 7, 9, 10 — correlated trajectory planning emerges naturally.
2. **Cross-attention at every denoising step**: Each action token queries all 82 spatially-encoded VLM tokens at every denoising iteration, rather than passing through a single 512D bottleneck once. The ratio changes from 164:1 (82 tokens → 1 pooled vector) to 1:82 (each step queries all tokens directly).

### Why 6D state

The PushT task is continuous: the robot must maintain contact with the block over multiple steps. The 2D state (position only) discards all motion context — the decoder cannot distinguish "approaching from the left" from "approaching from the right" even though they require opposite control strategies.

Adding the 2 most recent executed deltas costs 4 dims and provides strong short-horizon motion context without encoding any privileged information (block/goal position).

### Why same VLM layer (not multi-scale)

Using layers 14/21/28 would conflate two changes at once. Exp2 isolates the decoder architecture improvement. If Exp2 beats Exp1, adding multi-scale features in Exp3 will give a cleaner ablation signal.

---

## 5. Training Configuration

```
Epochs       : 300 (early stop patience = 50)
Batch size   : 256
Optimizer    : AdamW (weight_decay=0.01)
Scheduler    : OneCycleLR
  peak LR    : adapter=1.5e-4  decoder=3e-4
  warmup     : ~1.2% of total steps → cosine decay
Grad clip    : 1.0
Dropout      : 0.10 (decoder), 0.25 (adapter)
Flow steps   : 3 (training & inference)
```

---

## 6. Results

*(To be filled after training and evaluation)*

MetricExp1 (baseline)Exp2Best val loss0.4490 (epoch 152)—Success rate (SR)25% (5/20)—Mean max coverage87.2%—Episodes simulated20—Checkpoint`asset/result/checkpoints/best.ptasset/result_exp2/checkpoints/best.pt`

---

## 7. File Manifest

FileRole`config.py`Exp2 config (state_dim=6, use_dit_decoder=True, layer=28)`config_exp1.py`Frozen Exp1 snapshot (read-only)`models/flow_matching.pyDiTBlock`, `DiTFlowDecoder`, `FlowMatchingDecodermodels/vla.pyVLMTokenAdapter` (MultiScaleFusion stub, PerTokenLoRA, SpatialAwareMLP, AttentionReadout)`train.pyVLATrainModel` — adapter + decoder dispatcher`precompute_embeddings.py`Cache builder (v2 embeddings + 6D states)`evaluate.py`Quantitative eval + analysis plots`inference.pyPushTAgent` with 6D state tracking + video simulation`asset/result_exp2/vlm_embeddings.pt`Cache — `(25650, 82, 1024)` embeds + `(25650, 6)` states

---

## 8. Reproduction

```bash
# Precompute (already done — reuses Exp1 embeddings, adds 6D states):
python3 precompute_embeddings.py --exp 2 --recompute-states

# Train:
python3 train.py --exp 2

# Evaluate:
python3 evaluate.py --exp 2

# Simulate:
python3 inference.py --exp 2
```

To reproduce Exp1 baseline:

```bash
python3 train.py      --exp 1
python3 evaluate.py   --exp 1
python3 inference.py  --exp 1
```
