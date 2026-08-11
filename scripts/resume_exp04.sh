#!/usr/bin/env bash
# Resume exp04 (stock Qwen3-VL-2B) after a shutdown.
#
# Precompute is already DONE and its 21.4 GB cache is on disk at
# asset/runs/libero/exp04_qwen3vl/vlm_embeddings.h5, so this deliberately does
# NOT call scripts/precompute.py — that script has no skip-if-exists guard and
# would spend 18 minutes rebuilding a cache that is already correct.
#
# train.py resumes on its own: it looks for checkpoints/last.pt and restores the
# model, optimizer and scheduler, continuing from the next epoch. If no last.pt
# exists (shutdown happened during epoch 1) it simply starts from scratch, which
# costs about a minute. Pass --fresh to force a restart from epoch 1.
#
# Eval runs under vla_libero: the simulator stack is not installed in vla, and
# Qwen3-VL's processor additionally needs torchvision there (installed
# 2026-08-10 as torchvision 0.28.0+cu130, which pins the existing torch 2.13.0).
set -u
cd "F:/work/capstone/action-omni"
PY="C:/Users/SK/miniconda3/envs/vla/python.exe"
PY_SIM="C:/Users/SK/miniconda3/envs/vla_libero/python.exe"
LOG="asset/runs/libero/exp04_qwen3vl"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

if [ ! -f "$LOG/vlm_embeddings.h5" ]; then
  echo "[ERROR] cache missing at $LOG/vlm_embeddings.h5"
  echo "        run: $PY scripts/precompute.py --dataset libero --exp exp04"
  exit 1
fi

echo "STAGE_START train  $(date)"
"$PY" scripts/train.py --dataset libero --exp exp04 >> "$LOG/train.log" 2>&1
rc=$?; echo "STAGE_train_EXIT=$rc  $(date)"
if [ "$rc" -ne 0 ]; then tail -30 "$LOG/train.log"; exit "$rc"; fi
grep -E "Best val" "$LOG/train.log" | tail -1

# Same protocol as exp01/exp02/exp03: 10 tasks x 20 episodes x 2 conditions,
# LIBERO's 50 fixed init states. Do not change it -- the arms are only
# comparable because this is identical across all four.
echo "STAGE_START eval  $(date)"
MUJOCO_GL=wgl "$PY_SIM" scripts/eval_libero.py --exp exp04 --episodes 20 \
    --conditions canonical swapped > "$LOG/gate_eval.log" 2>&1
rc=$?; echo "STAGE_eval_EXIT=$rc  $(date)"
tail -12 "$LOG/gate_eval.log"
exit "$rc"
