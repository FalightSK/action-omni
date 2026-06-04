#!/usr/bin/env bash
# scripts/migrate_assets.sh
# ──────────────────────────
# Migrate asset directories from the old flat layout to the new
# dataset-namespaced layout.
#
# Old layout:       asset/result/       asset/result_exp2/  asset/result_exp3/
# New layout:  asset/runs/pusht/exp01_mlp/   exp02a_dit/    exp03_multiscale/
#
# This script creates the new directories and moves (or symlinks) the files.
# Run once after pulling the restructured repo.
#
# Usage:
#   bash scripts/migrate_assets.sh          # move files (recommended)
#   bash scripts/migrate_assets.sh --symlink  # create symlinks instead

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="move"
if [[ "$1" == "--symlink" ]]; then MODE="symlink"; fi

echo "Asset migration: mode=$MODE"
echo "Project root: $ROOT"

migrate() {
    local OLD="$1"
    local NEW="$2"
    if [ ! -e "$OLD" ]; then
        echo "  [SKIP] Not found: $OLD"
        return
    fi
    if [ -e "$NEW" ] && [ ! -L "$NEW" ]; then
        echo "  [SKIP] Already exists: $NEW"
        return
    fi
    mkdir -p "$(dirname "$NEW")"
    if [ "$MODE" = "symlink" ]; then
        ln -sfn "$(realpath "$OLD")" "$NEW"
        echo "  [LINK] $OLD → $NEW"
    else
        mv "$OLD" "$NEW"
        echo "  [MOVE] $OLD → $NEW"
    fi
}

mkdir -p asset/runs/pusht/logs
mkdir -p asset/dataset

# ── Dataset ───────────────────────────────────────────────────────────────────
# Old: asset/dataset/pusht_dataset/   New: asset/dataset/pusht/
migrate "asset/dataset/pusht_dataset" "asset/dataset/pusht"

# ── Experiment results ────────────────────────────────────────────────────────
# Exp01 (MLP): asset/result/ → asset/runs/pusht/exp01_mlp/
migrate "asset/result" "asset/runs/pusht/exp01_mlp"

# Exp02a (DiT): asset/result_exp2/ → asset/runs/pusht/exp02a_dit/
migrate "asset/result_exp2" "asset/runs/pusht/exp02a_dit"

# Exp03 (multi-scale): asset/result_exp3/ → asset/runs/pusht/exp03_multiscale/
migrate "asset/result_exp3" "asset/runs/pusht/exp03_multiscale"

# ── Logs ──────────────────────────────────────────────────────────────────────
# Old: asset/logs/  New: asset/runs/pusht/logs/
migrate "asset/logs" "asset/runs/pusht/logs"

# ── Comparison results (root-level JSON) ──────────────────────────────────────
migrate "asset/comparison_results.json" "asset/runs/pusht/comparison_results.json"

echo ""
echo "Migration complete."
echo "New structure:"
echo "  asset/runs/pusht/exp01_mlp/       (Exp01 MLP, 30% SR)"
echo "  asset/runs/pusht/exp02a_dit/      (Exp02a DiT, 56% SR — BEST)"
echo "  asset/runs/pusht/exp03_multiscale/ (Exp03 multi-scale, 44% SR)"
echo "  asset/dataset/pusht/               (raw PushT dataset)"
echo ""
echo "Verify with: ls -la asset/runs/pusht/"
