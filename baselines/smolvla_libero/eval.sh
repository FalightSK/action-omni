#!/usr/bin/env bash
# Reproduce the SmolVLA baseline on LIBERO-Spatial.
#
# Result on this stack: 76.0% success over 100 rollouts (10 tasks x 10 episodes),
# matched camera, bf16. See README.md for the reproducibility caveat.
set -euxo pipefail

OUT=${OUT:-runs/smolvla_libero_spatial}
TASK=${TASK:-libero_spatial}   # libero_spatial | libero_object | libero_goal | libero_10
mkdir -p "$OUT"

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero --env.task="$TASK" --env.fps=10 \
  --eval.n_episodes=100 --eval.batch_size=1 \
  --policy.device=cuda \
  --output_dir="$OUT"
