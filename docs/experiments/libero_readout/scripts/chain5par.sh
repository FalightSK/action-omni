#!/bin/bash
# Closed-loop arms, 2 concurrent (2 x 10 envs = 20 cores), leaving room for the
# 3 concurrent lerobot baselines (3 x 8 = 24 cores) on this 48-core box.
cd /workspace/omni
export HF_HOME=/workspace/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/egl_nvidia.json
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
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
out=[]
for m in ("all","instr"):
    for lab,d in (("best",best),("worst",worst)):
        x=d[m][1]
        if not x: continue
        vs = ["orig","swap"] if lab=="best" else ["orig"]
        for v in vs:
            out.append(f"{m} {x['tap']} {x['lr']:g} {lab} {v}")
print("\n".join(out))
EOF
cat arms2.txt

arm () {
  read -r MODE TAP LR LAB V <<< "$1"
  echo "=== arm mode=$MODE tap=$TAP ($LAB) variant=$V ==="
  $PY rollout_vec.py --tap "$TAP" --mode "$MODE" --ckpt "ck2_${MODE}_tap${TAP}_lr${LR}.pt" \
      --episodes 30 --envs 10 --max-steps 400 --replan 8 --variant "$V" \
      > "vlog_${MODE}_${TAP}_${V}.log" 2>/dev/null
  grep -E '^SR|wrote' "vlog_${MODE}_${TAP}_${V}.log"
}

N=0
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  arm "$LINE" &
  N=$((N+1))
  if [ $((N % 2)) -eq 0 ]; then wait; fi     # 2 concurrent
done < arms2.txt
wait
echo "=== CHAIN5_DONE ==="
