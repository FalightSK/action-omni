#!/bin/bash
# Stage 2: the deps that need a compiler.
#   egl-probe 1.0.2 (pulled in by hf-libero -> robomimic) ships a pre-3.5
#   CMakeLists, which CMake 4.x refuses by default -> set the compat floor.
set -x
export PATH="/root/.local/bin:$PATH"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
ROOT=/opt/libero
source $ROOT/venv/bin/activate

uv pip install termcolor
uv pip install "hf-libero>=0.1.3,<0.2.0"

python - <<'PY'
import os, libero
p = os.path.join(os.path.dirname(libero.__file__), "libero")
os.makedirs("/root/.libero", exist_ok=True)
open("/root/.libero/config.yaml", "w").write(
    f"benchmark_root: {p}\n"
    f"bddl_files: {p}/bddl_files\n"
    f"init_states: {p}/init_files\n"
    f"datasets: /opt/libero/datasets\n"
    f"assets: {p}/assets\n")
print("libero at", p)
print(open("/root/.libero/config.yaml").read())
PY

echo "STAGE2_DONE"
