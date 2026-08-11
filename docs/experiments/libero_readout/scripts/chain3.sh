#!/bin/bash
# Fires when the 36-run sweep completes: Figure 2, then closed-loop at best/worst tap.
cd /workspace/omni
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/egl_nvidia.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

while ! grep -q ALL_DONE chain2.log; do sleep 20; done
echo "=== sweep done -> figure 2 ==="
python plot2.py > plot2.log 2>&1
echo "--- plot2 ---"; cat plot2.log

BEST=$(python -c "import json;print(json.load(open('arms.json'))['best_tap'])")
BLR=$(python -c "import json;print('%g'%json.load(open('arms.json'))['best_lr'])")
WORST=$(python -c "import json;print(json.load(open('arms.json'))['worst_tap'])")
WLR=$(python -c "import json;print('%g'%json.load(open('arms.json'))['worst_lr'])")
echo "ARMS best=tap$BEST/lr$BLR  worst=tap$WORST/lr$WLR"

PY=/workspace/venv-libero/bin/python
for ARM in "$BEST $BLR" "$WORST $WLR"; do
  set -- $ARM; T=$1; L=$2
  for V in orig para1 swap; do
    echo "=== closed-loop tap=$T lr=$L variant=$V ==="
    $PY rollout.py --tap "$T" --ckpt "ckpt_tap${T}_lr${L}.pt" \
        --episodes 10 --max-steps 400 --replan 8 --variant "$V" \
        2>/dev/null | grep -E '^\{|^SR'
  done
done
echo "=== CLOSED_LOOP_DONE ==="
