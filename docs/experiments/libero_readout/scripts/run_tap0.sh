#!/bin/bash
cd /workspace/omni
export HF_HOME=/workspace/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/egl_nvidia.json
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/workspace/venv-libero/bin/python rollout_vec.py --tap 0 --mode all \
  --ckpt ck2_all_tap0_lr0.001.pt --episodes 30 --envs 8 --max-steps 400 \
  --replan 8 --variant orig > vlog_all_0_orig.log 2>/dev/null
echo TAP0_DONE
