#!/usr/bin/env bash
# Run inside an existing PyTorch Pod, from an unpacked action-omni checkout.
set -Eeuo pipefail
dataset="${1:?Usage: bash scripts/setup_runpod_train.sh aloha|libero [cloud-asset-root]}"
case "$dataset" in aloha|libero) ;; *) printf 'Expected aloha or libero\n' >&2; exit 2;; esac
source_root="${2:-/workspace-global/action-omni/study_v1}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt \
  'torch==2.9.1' 'torchvision==0.24.1' 'transformers==4.57.6' 'numpy==1.26.4' 'pyarrow==17.0.0'
# B300 is compute capability 10.3 (sm_103); the CUDA 12.8 wheel's NVRTC cannot
# compile kernels for it. Force a CUDA 13.0 wheel into this venv even when the
# base image exposes a version-matching cu128 torch through system-site-packages.
if ! .venv/bin/python -c 'import torch, torchvision; assert torch.__version__ == "2.9.1+cu130"; assert torchvision.__version__ == "0.24.1+cu130"; assert torch.version.cuda == "13.0"'; then
  .venv/bin/python -m pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple \
    'torch==2.9.1' 'torchvision==0.24.1' 'numpy==1.26.4' 'pyarrow==17.0.0' 'filelock<4'
fi
.venv/bin/python -u scripts/stage_runpod_assets.py --dataset "$dataset" \
  --source-root "$source_root" --prepare-protocol
.venv/bin/python -m pip freeze > "asset/runtime_${dataset}.txt"
