#!/usr/bin/env bash
# Reproducible simulator runtime setup for the dedicated RunPod evaluation Pod.
# Run from the repository root: bash scripts/setup_runpod_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GLOBAL_ROOT="${ACTIONOMNI_DURABLE_ROOT:-/workspace-global/action-omni}"
if [[ ! -d /workspace-global ]]; then
  echo "RunPod Global Volume is not mounted at /workspace-global" >&2
  exit 2
fi
mkdir -p "$GLOBAL_ROOT/cache/huggingface" "$GLOBAL_ROOT/cache/pip" \
  "$GLOBAL_ROOT/cache/uv" "$GLOBAL_ROOT/assets/data/libero" \
  "$GLOBAL_ROOT/config/libero" /tmp/action-omni-uv-cache
export HF_HOME="$GLOBAL_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export PIP_CACHE_DIR="$GLOBAL_ROOT/cache/pip"
export UV_CACHE_DIR=/tmp/action-omni-uv-cache

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  git curl ca-certificates build-essential pkg-config \
  libegl1 libgl1 libosmesa6 libglfw3 libglew2.2 \
  libx11-6 libxext6 libxrender1 libsm6 ffmpeg xvfb

# Install uv from its official installer if not already available.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p third_party asset/models
if [[ ! -d third_party/LIBERO/.git ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi
LIBERO_REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"
git -C third_party/LIBERO fetch --depth 1 origin "$LIBERO_REVISION"
git -C third_party/LIBERO checkout --detach "$LIBERO_REVISION"
if [[ "$(git -C third_party/LIBERO rev-parse HEAD)" != "$LIBERO_REVISION" ]]; then
  echo "LIBERO checkout does not match the locked revision" >&2
  exit 2
fi

# ALOHA: retain the image's torch 2.9.1+cu128 installation, use PyPI gym-aloha
# (whose simulator dependency is MuJoCo 3), and isolate NumPy/Transformers.
python3 -m venv --system-site-packages .venv-eval-aloha
.venv-eval-aloha/bin/python -m pip install --upgrade pip
.venv-eval-aloha/bin/python -m pip install \
  'numpy==1.26.4' 'pyarrow==17.0.0' 'transformers==4.57.6' 'accelerate>=0.30.0' \
  'Pillow>=10.0.0' 'einops>=0.8.0' 'gym-aloha' 'av>=12.0.0' imageio

# Stage only the locked model files from the shared study asset manifest.
# The Global Volume is the source of truth; no stale study_v1 path fallback.
.venv-eval-aloha/bin/python -u scripts/stage_runpod_assets.py --dataset evaluation \
  --source-root "$GLOBAL_ROOT" --plan configs/runpod_success_stopped_study.json

# LIBERO/robosuite 1.4.1 uses the MuJoCo 2.3 API, so give it a separate Python
# 3.10 environment. Do not run scripts/data/_setup_libero.py: it contains
# Windows-only macros/DLL changes. Configure the Linux source path directly.
uv python install 3.10
uv venv --python 3.10 .venv-eval-libero
LIBERO_PY="$ROOT/.venv-eval-libero/bin/python"
uv pip install --python "$LIBERO_PY" pip
"$LIBERO_PY" -m pip install --upgrade pip
"$LIBERO_PY" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1' 'torchvision==0.24.1'
"$LIBERO_PY" -m pip install \
  'numpy==1.26.4' 'pyarrow==17.0.0' 'transformers==4.57.6' 'accelerate>=0.30.0' \
  'Pillow>=10.0.0' 'einops>=0.8.0' 'h5py>=3.10.0' 'scipy>=1.11.0' 'matplotlib>=3.8.0' \
  'robosuite==1.4.1' 'mujoco==2.3.7' 'gym==0.25.2' \
  bddl easydict cloudpickle termcolor imageio imageio-ffmpeg

# LIBERO has no importable top-level __init__.py; add its pinned source root and
# keep its config/data and download caches on the durable Global Volume.
LIBERO_SRC="$ROOT/third_party/LIBERO"
SITE="$ROOT/.venv-eval-libero/lib/python3.10/site-packages"
printf '%s\n' "$LIBERO_SRC" > "$SITE/libero_src.pth"
mkdir -p "$GLOBAL_ROOT/config/libero" "$GLOBAL_ROOT/assets/data/libero"
cat > "$GLOBAL_ROOT/config/libero/config.yaml" <<EOF
benchmark_root: $LIBERO_SRC/libero/libero
bddl_files: $LIBERO_SRC/libero/libero/bddl_files
init_states: $LIBERO_SRC/libero/libero/init_files
datasets: $GLOBAL_ROOT/assets/data/libero
assets: $LIBERO_SRC/libero/libero/assets
EOF

# The actual simulator smoke sets its rendering backend per environment.
cat > "$ROOT/.runpod-eval-env" <<EOF
export ACTIONOMNI_DURABLE_ROOT="$GLOBAL_ROOT"
export HF_HOME="$GLOBAL_ROOT/cache/huggingface"
export HF_HUB_CACHE="$GLOBAL_ROOT/cache/huggingface/hub"
export HF_ASSETS_CACHE="$GLOBAL_ROOT/cache/huggingface/assets"
export PIP_CACHE_DIR="$GLOBAL_ROOT/cache/pip"
export UV_CACHE_DIR=/tmp/action-omni-uv-cache
export LIBERO_CONFIG_PATH="$GLOBAL_ROOT/config/libero"
EOF

echo "Evaluation simulator runtimes installed under $ROOT"
echo "LIBERO source revision: $LIBERO_REVISION"
echo "Durable caches and simulator data: $GLOBAL_ROOT"
echo "ALOHA interpreter: $ROOT/.venv-eval-aloha/bin/python"
echo "LIBERO interpreter: $ROOT/.venv-eval-libero/bin/python"
echo "Source $ROOT/.runpod-eval-env before starting scripts/runpod_eval_worker.py"
