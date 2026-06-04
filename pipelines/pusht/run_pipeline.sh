#!/usr/bin/env bash
# pipelines/pusht/run_pipeline.sh
# ─────────────────────────────────
# Full PushT experimental pipeline.
#
# Runs all PushT experiments in order:
#   1. Exp02a (DiT, best) — precompute + train + eval 50 episodes
#   2. Exp01  (MLP)       — eval only (reuses Exp02a's cache)
#   3. Exp02a ablations   — ih=8 and ms=500
#   4. Exp03 (multi-scale) — precompute + train + eval 50 episodes
#
# To run a single experiment only, see run_exp.sh.
#
# Usage:
#   cd vla_project && bash pipelines/pusht/run_pipeline.sh
#   Monitor: tail -f asset/runs/pusht/logs/pipeline.log
#
# Note: Precompute (~90 min) + Train (~8 hr) + Eval (~18 min) per experiment.
# Total: ~20 hours for full pipeline. Run overnight.

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="asset/runs/pusht/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PIPELINE_LOG"; }
sep() {
    echo "" | tee -a "$PIPELINE_LOG"
    echo "══════════════════════════════════════════════════════" | tee -a "$PIPELINE_LOG"
    log "$*"
    echo "══════════════════════════════════════════════════════" | tee -a "$PIPELINE_LOG"
}

sep "PUSHT PIPELINE START"

# ── 1: Precompute Exp01/02a shared cache ──────────────────────────────────────
sep "[1/7] Precompute Exp01/02a embeddings (layer 28, single-scale)"
log "→ $LOG_DIR/precompute_exp01.log"
log "Est: ~90 min on M1 Mac, ~4.1 GB cache"
python3 -u scripts/precompute.py --dataset pusht --exp exp01 \
    2>&1 | tee "$LOG_DIR/precompute_exp01.log"
log "Step 1 DONE"

# ── 2: Train Exp01 (MLP baseline) ─────────────────────────────────────────────
sep "[2/7] Train Exp01 — MLP decoder"
log "→ $LOG_DIR/train_exp01.log"
python3 -u scripts/train.py --dataset pusht --exp exp01 \
    2>&1 | tee "$LOG_DIR/train_exp01.log"
log "Step 2 DONE"

# ── 3: Train Exp02a (DiT, best) ───────────────────────────────────────────────
sep "[3/7] Train Exp02a — DiT decoder (reuses Exp01 cache)"
log "→ $LOG_DIR/train_exp02a.log"
python3 -u scripts/train.py --dataset pusht --exp exp02a \
    2>&1 | tee "$LOG_DIR/train_exp02a.log"
log "Step 3 DONE"

# ── 4: Eval Exp01 + Exp02a (50 episodes each) ─────────────────────────────────
sep "[4/7] Eval Exp01 — 50 episodes"
python3 -u scripts/evaluate.py --dataset pusht --exp exp01 --episodes 50 --no-video \
    2>&1 | tee "$LOG_DIR/eval_exp01.log"
log "Step 4a DONE"

sep "[4b/7] Eval Exp02a — 50 episodes"
python3 -u scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50 --no-video \
    2>&1 | tee "$LOG_DIR/eval_exp02a.log"
log "Step 4b DONE"

# ── 5: Exp02a ablations ───────────────────────────────────────────────────────
sep "[5/7] Exp02a ablations"
log "  5a: inference_horizon=8 (Exp02c)"
python3 -u scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50 --no-video \
    --inference-horizon 8 \
    --output asset/runs/pusht/exp02a_dit/sim_results_ih8.json \
    2>&1 | tee "$LOG_DIR/eval_exp02c_ih8.log"
log "  5b: sim_max_steps=500 (Exp02d)"
python3 -u scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50 --no-video \
    --max-steps 500 \
    --output asset/runs/pusht/exp02a_dit/sim_results_ms500.json \
    2>&1 | tee "$LOG_DIR/eval_exp02d_ms500.log"
log "Step 5 DONE"

# ── 6: Exp03 multi-scale ──────────────────────────────────────────────────────
sep "[6/7] Precompute Exp03 (multi-scale layers 14,21,28)"
log "Est: ~90 min, ~12.8 GB cache"
python3 -u scripts/precompute.py --dataset pusht --exp exp03 \
    2>&1 | tee "$LOG_DIR/precompute_exp03.log"
log "Step 6a DONE"

sep "[6b/7] Train Exp03 — multi-scale DiT"
python3 -u scripts/train.py --dataset pusht --exp exp03 \
    2>&1 | tee "$LOG_DIR/train_exp03.log"
log "Step 6b DONE"

sep "[6c/7] Eval Exp03 — 50 episodes"
python3 -u scripts/evaluate.py --dataset pusht --exp exp03 --episodes 50 --no-video \
    2>&1 | tee "$LOG_DIR/eval_exp03.log"
log "Step 6c DONE"

# ── 7: Mechanistic analysis ───────────────────────────────────────────────────
sep "[7/7] Mechanistic analysis (Exp01 + Exp02a + Exp03)"
python3 -u scripts/mechanistic.py --dataset pusht --exp exp01 \
    2>&1 | tee "$LOG_DIR/mech_exp01.log"
python3 -u scripts/mechanistic.py --dataset pusht --exp exp02a \
    2>&1 | tee "$LOG_DIR/mech_exp02a.log"
python3 -u scripts/mechanistic.py --dataset pusht --exp exp03 \
    2>&1 | tee "$LOG_DIR/mech_exp03.log"
log "Step 7 DONE"

sep "PUSHT PIPELINE COMPLETE"
log "All results in asset/runs/pusht/"
log "Logs in $LOG_DIR/"
