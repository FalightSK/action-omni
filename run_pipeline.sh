#!/usr/bin/env bash
# run_pipeline.sh
# ───────────────
# Waits for the full VLM precompute to finish, then automatically:
#   1. Rebuilds action sequences with relative deltas (instant)
#   2. Clears old checkpoints
#   3. Trains the new architecture (adapter + decoder, 300 epochs)
#   4. Runs simulation (20 episodes, saves videos)
#
# Usage:
#   bash run_pipeline.sh           # run all steps
#   bash run_pipeline.sh --skip-precompute  # skip waiting, start from step 1

set -e
cd "$(dirname "$0")"

SKIP_WAIT=false
if [ "$1" = "--skip-precompute" ]; then SKIP_WAIT=true; fi

echo "════════════════════════════════════════════════════════"
echo "  VLA Pipeline  (relative actions + LoRA adapter)"
echo "════════════════════════════════════════════════════════"

# ── Step 0: Wait for precompute ────────────────────────────────────────────
if [ "$SKIP_WAIT" = false ]; then
  echo ""
  echo "[0/5] Waiting for precompute_embeddings.py to finish …"
  echo "      (check progress: tail -f /tmp/precompute_full.txt)"
  while ! grep -q "Saved →\|Saved embeddings" /tmp/precompute_full.txt 2>/dev/null; do
    sleep 30
  done
  echo "      Precompute done!"
fi

# ── Step 1: Recompute action sequences ────────────────────────────────────
echo ""
echo "[1/4] Rebuilding action sequences (relative deltas, horizon=16) …"
python3 precompute_embeddings.py --recompute-actions
echo "      Done."

# ── Step 2: Clear old checkpoints ─────────────────────────────────────────
echo ""
echo "[2/4] Clearing old checkpoints …"
rm -f asset/result/checkpoints/*.pt
echo "      Cleared."

# ── Step 3: Train ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] Training (300 epochs, adapter + decoder) …"
PYTHONUNBUFFERED=1 python3 train.py --epochs 300 2>&1 | tee /tmp/train_final.txt
echo "      Training complete."

# ── Step 4: Simulation ────────────────────────────────────────────────────
echo ""
echo "[4/4] Running simulation (20 episodes) …"
PYTHONUNBUFFERED=1 python3 inference.py --episodes 20 2>&1 | tee /tmp/inference_final.txt
echo "      Simulation complete."

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline finished!  Check:"
echo "  - Simulation results : asset/result/sim_results.json"
echo "  - Videos             : asset/result/videos/"
echo "════════════════════════════════════════════════════════"
