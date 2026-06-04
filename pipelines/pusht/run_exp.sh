#!/usr/bin/env bash
# pipelines/pusht/run_exp.sh
# ───────────────────────────
# Run a single PushT experiment end-to-end.
#
# Usage:
#   bash pipelines/pusht/run_exp.sh exp02a          # default dataset=pusht
#   bash pipelines/pusht/run_exp.sh exp04 --no-precompute  # skip precompute
#   EXP=exp03 bash pipelines/pusht/run_exp.sh       # via env var
#
# Flags:
#   --no-precompute   Skip precompute (use existing cache)
#   --no-train        Skip training (use existing checkpoint)
#   --no-eval         Skip evaluation
#   --episodes N      Number of eval episodes (default 50)
#   --no-video        Skip video recording

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXP="${1:-exp02a}"
DATASET="${DATASET:-pusht}"
shift 2>/dev/null || true

DO_PRECOMPUTE=true
DO_TRAIN=true
DO_EVAL=true
EPISODES=50
NO_VIDEO="--no-video"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-precompute) DO_PRECOMPUTE=false ;;
        --no-train)      DO_TRAIN=false ;;
        --no-eval)       DO_EVAL=false ;;
        --episodes)      EPISODES="$2"; shift ;;
        --no-video)      NO_VIDEO="--no-video" ;;
        --video)         NO_VIDEO="" ;;
        *) echo "[WARN] Unknown flag: $1" ;;
    esac
    shift
done

LOG_DIR="asset/runs/${DATASET}/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${EXP}_$(date '+%Y%m%d_%H%M%S').log"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "Running experiment: dataset=${DATASET}  exp=${EXP}"
log "Log: $LOG"

if $DO_PRECOMPUTE; then
    log "[1/3] Precompute embeddings …"
    python3 -u scripts/precompute.py --dataset "$DATASET" --exp "$EXP" \
        2>&1 | tee -a "$LOG"
    log "Precompute done."
fi

if $DO_TRAIN; then
    log "[2/3] Training …"
    python3 -u scripts/train.py --dataset "$DATASET" --exp "$EXP" \
        2>&1 | tee -a "$LOG"
    log "Training done."
fi

if $DO_EVAL; then
    log "[3/3] Evaluating ($EPISODES episodes) …"
    python3 -u scripts/evaluate.py --dataset "$DATASET" --exp "$EXP" \
        --episodes "$EPISODES" $NO_VIDEO \
        2>&1 | tee -a "$LOG"
    log "Eval done."
fi

log "Experiment $EXP complete. Results in asset/runs/${DATASET}/${EXP}/"
