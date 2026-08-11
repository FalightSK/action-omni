#!/usr/bin/env bash
# SmolVLA-on-LIBERO baseline environment.
#
# Critical: LIBERO must be the `hf-libero` fork (HF's renderer that produced the
# HuggingFaceVLA/libero dataset). The upstream LIBERO clone renders the Panda arm with
# the wrong (green camo) materials, which collapses eval success to ~0%.
set -euxo pipefail

ENV_NAME=${ENV_NAME:-smolvla-libero}

conda create -y -n "$ENV_NAME" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# lerobot with the libero extra (sim eval harness + policy loaders)
pip install "lerobot[libero]"

# robosuite must be EXACTLY 1.4.0 — LIBERO's pin. 1.4.1 renders wrong textures.
pip install "robosuite==1.4.0" "bddl==1.0.1" easydict future matplotlib "gym==0.25.2"

# hf-libero: HF's maintained LIBERO fork; matches the dataset's visuals.
pip uninstall -y cmake 2>/dev/null || true   # pip cmake 4.x is broken; use the system cmake
pip install "hf-libero>=0.1.3,<0.2.0"

# numpy >= 1.24 removed the aliases robosuite 1.4.0 uses — install the env-wide shim.
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp "$(dirname "$0")/numpy_shim.py" "$SP/sitecustomize.py"

# sanity: the sim must import and construct an offscreen render env
python -c "from libero.libero.envs import OffScreenRenderEnv; print('LIBERO sim OK')"
echo "ENV_READY: $ENV_NAME"
