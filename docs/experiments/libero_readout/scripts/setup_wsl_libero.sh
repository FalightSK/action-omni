#!/bin/bash
# LIBERO closed-loop environment inside WSL2 Ubuntu (RTX 4070 Ti passthrough).
#
# Mirrors the recipe validated on the RunPod box.  Two WSL-specific wrinkles:
#   * the NVIDIA EGL/GL libs live in /usr/lib/wsl/lib, not the usual paths
#   * /usr/share may be writable here (unlike the pod), but we still use the
#     __EGL_VENDOR_LIBRARY_FILENAMES override so the recipe is portable
set -x
export DEBIAN_FRONTEND=noninteractive
ROOT=/opt/libero
mkdir -p $ROOT

apt-get update -qq
apt-get install -y -qq curl ca-certificates \
    libegl1 libgl1 libglew2.2 libosmesa6 libopengl0 libglx-mesa0 libglu1-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender1

# uv for a pinned python 3.10 (robosuite 1.4.0 predates 3.12+)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv python install 3.10
uv venv --python 3.10 $ROOT/venv
source $ROOT/venv/bin/activate

uv pip install torch --torch-backend=cu126
uv pip install "robosuite==1.4.0" "bddl==1.0.1" easydict future "gym==0.25.2" "numpy<2"
uv pip install "hf-libero>=0.1.3,<0.2.0"
uv pip install transformers pillow imageio imageio-ffmpeg pandas pyarrow num2words
uv pip install torchvision --torch-backend=cu126

# numpy alias shim (robosuite 1.4.0 uses np.float / np.int, removed in numpy>=1.24)
SP=$($ROOT/venv/bin/python -c "import site; print(site.getsitepackages()[0])")
cat > "$SP/sitecustomize.py" <<'SHIM'
import numpy as np
for _n, _t in (("float", float), ("int", int), ("bool", bool),
               ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)
SHIM

# EGL vendor ICD -> point at the WSL-provided NVIDIA driver
cat > $ROOT/egl_nvidia.json <<'JSON'
{
  "file_format_version" : "1.0.0",
  "ICD" : { "library_path" : "libEGL_nvidia.so.0" }
}
JSON

# LIBERO writes a config on first import and prompts interactively without it
P=$($ROOT/venv/bin/python -c "import libero,os;print(os.path.join(os.path.dirname(libero.__file__),'libero'))" 2>/dev/null)
mkdir -p /root/.libero
printf "benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/datasets\nassets: %s/assets\n" \
    "$P" "$P" "$P" "$ROOT" "$P" > /root/.libero/config.yaml
mkdir -p $ROOT/datasets

echo "SETUP_STAGE_DONE"
