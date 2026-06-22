# =============================================================================
# scripts/run_lt_pipeline.ps1
# Language Table exp01 - full 10% pipeline: convert -> precompute(HDF5) -> train.
#
# WHY each step (capstone gate: test the architecture's LANGUAGE adaptability):
#   1. convert    - pull 10% (18,102 / 181,020 episodes, NO OOD holdout so all
#                   command types are included) from GCS RLDS into local parquet.
#                   We need a large, instruction-diverse slice so the per-episode
#                   language signal is what the adapter must learn, not one fixed task.
#   2. precompute - run the FROZEN Qwen VLM once over every frame and STREAM the token
#                   embeddings to an HDF5 file (uint16 bf16-bits, chunked + lzf). HDF5
#                   not .pt because ~470k frames x 120 x 1024 bf16 (~115 GB) cannot be
#                   held in RAM for a single torch.save; HDF5 writes/reads lazily.
#   3. train      - train the LoRA adapter + DiT flow decoder on the cached embeddings.
#
# convert runs in the vla_lt_data env (TensorFlow/TFDS); precompute+train run in the
# vla env (torch+CUDA). Each step is guarded; a failure halts the chain.
# =============================================================================

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"

$repo     = "F:\work\capstone\action-omni"
$pyData   = "C:\Users\SK\miniconda3\envs\vla_lt_data\python.exe"
$pyVla    = "C:\Users\SK\miniconda3\envs\vla\python.exe"
$dataRoot = "$repo\asset\data\language_table_sim"
$cache    = "$repo\asset\runs\language_table\exp01_baseline\vlm_embeddings.h5"
$episodes = 18102

Set-Location $repo
Write-Output "==== LT PIPELINE START $(Get-Date -Format o) ===="

# Step 1: convert (skip if parquet already present)
if (Test-Path "$dataRoot\data\chunk-000\file-000.parquet") {
    Write-Output "[1/3] convert SKIPPED - parquet already exists at $dataRoot"
} else {
    Write-Output "[1/3] convert: $episodes episodes (no holdout) -> $dataRoot"
    & $pyData scripts/convert_language_table.py --out $dataRoot --episodes $episodes --img_w 320 --img_h 180
    if (-not $?) { Write-Output "[1/3] CONVERT FAILED - halting"; exit 1 }
}

# Step 2: precompute VLM embeddings -> HDF5 (skip if cache exists)
if (Test-Path $cache) {
    Write-Output "[2/3] precompute SKIPPED - cache already exists at $cache"
} else {
    Write-Output "[2/3] precompute -> HDF5 stream: $cache"
    & $pyVla scripts/precompute.py --dataset language_table --exp exp01
    if (-not $?) { Write-Output "[2/3] PRECOMPUTE FAILED - halting"; exit 1 }
}

# Step 3: train (auto-retry + resume from last.pt on GPU-driver TDR / crash)
# batch sized from VRAM calibration (scripts/_vram_calib.py). Do NOT raise the batch
# without recalibrating - overshooting VRAM crashes the PC. train.py auto-resumes from
# checkpoints/last.pt, so a TDR only costs the current epoch; we relaunch up to N times.
$batch = 1024
$maxRetries = 30
for ($try = 1; $try -le $maxRetries; $try++) {
    Write-Output "[3/3] train attempt $try/$maxRetries : language_table/exp01 (150 epochs, batch $batch)"
    & $pyVla scripts/train.py --dataset language_table --exp exp01 --batch-size $batch
    if ($?) { Write-Output "[3/3] train finished cleanly"; break }
    Write-Output "[3/3] train exited non-zero (likely GPU TDR) - waiting 30s for the driver to settle, then resuming from last.pt"
    Start-Sleep -Seconds 30
}

Write-Output "==== LT PIPELINE DONE $(Get-Date -Format o) ===="
