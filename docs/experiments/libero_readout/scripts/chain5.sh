#!/bin/bash
cd /workspace/omni
export HF_HOME=/workspace/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/egl_nvidia.json
PY=/workspace/venv-libero/bin/python

while ! grep -q SWEEP2_DONE sweep2.log; do sleep 20; done
echo "=== sweep2 done ==="

python - <<'EOF' > arms2.txt
import json, collections
R=[x for x in json.load(open("sweep2.json")) if not x.get("lr_check")]
best=collections.defaultdict(lambda:(9e9,None)); worst=collections.defaultdict(lambda:(-9e9,None))
for x in R:
    m=x["mode"]
    if x["ol_mse_orig"]<best[m][0]: best[m]=(x["ol_mse_orig"],x)
    if x["ol_mse_orig"]>worst[m][0]: worst[m]=(x["ol_mse_orig"],x)
lines=[]
for m in ("all","instr"):
    for lab,d in (("best",best),("worst",worst)):
        x=d[m][1]
        if x: lines.append(f"{m} {x['tap']} {x['lr']:g} {lab} {x['ol_mse_orig']:.4f}")
print("\n".join(lines))
EOF
cat arms2.txt

# depth question: best vs worst tap, each read-out, original instruction
# language question: best tap, each read-out, original vs goal-swap
while read -r MODE TAP LR LAB OL; do
  [ -z "$MODE" ] && continue
  VARS="orig"
  [ "$LAB" = "best" ] && VARS="orig swap"
  for V in $VARS; do
    echo "=== closed-loop mode=$MODE tap=$TAP ($LAB) variant=$V ==="
    $PY rollout_vec.py --tap "$TAP" --mode "$MODE" --ckpt "ck2_${MODE}_tap${TAP}_lr${LR}.pt" \
        --episodes 30 --envs 16 --max-steps 400 --replan 8 --variant "$V" \
        2>/dev/null | grep -E '^\{|^SR|wrote'
  done
done < arms2.txt

echo "=== CHAIN5_DONE ==="
