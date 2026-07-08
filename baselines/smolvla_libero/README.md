# SmolVLA — LIBERO baseline

Reproduces the **SmolVLA (450M)** policy on the **LIBERO-Spatial** benchmark using the official
checkpoint and `lerobot`'s closed-loop simulation eval. This is a reference/anchor baseline for
comparing action models under a fixed local protocol.

## Result

| Policy | Benchmark | Rollouts | Camera | Precision | Success |
|---|---|---|---|---|---|
| SmolVLA (`HuggingFaceVLA/smolvla_libero`) | libero_spatial | 100 (10 tasks × 10 eps) | matched | bf16 | **76.0%** |

### Reproducibility note (read this)

76.0% sits at the **top of the community-reproduction band (~63–73%)**. The paper's reported
~87–90% does **not** reproduce on the current `lerobot` / LIBERO stack ecosystem-wide — see
lerobot issues [#1369](https://github.com/huggingface/lerobot/issues/1369),
[#2354](https://github.com/huggingface/lerobot/issues/2354),
[#3264](https://github.com/huggingface/lerobot/issues/3264). Treat all comparisons as **relative**
under this fixed protocol (hf-libero renderer, `env.fps=10`, benchmark init states); absolute
comparisons to published numbers are out of scope.

## Resources & runtime (measured)

Eval-only baseline — the checkpoint is the official one; no training was run.

| Item | Measured |
|---|---|
| Hardware | 1× consumer GPU, 12 GB VRAM (RTX 4070-class), bf16 |
| Platform | WSL2 (Ubuntu), headless MuJoCo/EGL rendering |
| Eval wall time | **9,770 s ≈ 2.7 h** for 100 episodes |
| Per episode | 97.7 s average (closed-loop: sim step + policy forward every control step) |
| GPU memory | SmolVLA-450M in bf16 fits well under 12 GB |

Wall time is dominated by closed-loop simulation, and scales with success rate — failed episodes
run to the step cap, successful ones terminate early. Budget ~3 GPU-hours per 100-episode suite
on this hardware class; a faster GPU helps less than you'd expect since much of the loop is
CPU-side sim/rendering.

## Critical setup gotchas

These are the non-obvious things that silently collapse success to ~0% if wrong:

1. **Use `hf-libero`, not the upstream LIBERO clone.** The upstream repo renders the Panda arm with
   the wrong (green camo) materials, which do not match the demonstration data — the policy sees an
   out-of-distribution scene and fails every task. `hf-libero` is HF's maintained fork and matches
   the visuals of the `HuggingFaceVLA/libero` dataset. This alone is the difference between 0% and 76%.
2. **`robosuite==1.4.0` exactly** — LIBERO's pin. 1.4.1 also renders wrong textures.
3. **numpy alias shim.** robosuite 1.4.0 uses `np.float`/`np.int`/… which numpy ≥ 1.24 removed;
   `numpy_shim.py` restores them env-wide (installed as `sitecustomize.py`).
4. **`env.fps=10`** — must match the dataset's control rate.

## Quickstart

```bash
# 1. Environment (creates a conda env `smolvla-libero`; override with ENV_NAME=...)
bash setup_env.sh

# 2. Fetch the policy checkpoint + LIBERO dataset into your HF cache
export HF_HOME=/path/to/hf_cache
conda activate smolvla-libero
bash fetch.sh

# 3. Evaluate (writes to runs/smolvla_libero_spatial/)
bash eval.sh
```

Requires a CUDA GPU (SmolVLA-450M in bf16 fits comfortably in <12 GB) and a headless-capable
MuJoCo/EGL setup (Linux/WSL2 recommended).
