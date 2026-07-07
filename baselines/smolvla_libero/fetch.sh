#!/usr/bin/env bash
# Download the SmolVLA-LIBERO policy checkpoint and the LIBERO dataset into HF_HOME.
set -euxo pipefail
: "${HF_HOME:?set HF_HOME to your cache dir, e.g. export HF_HOME=/path/to/hf_cache}"

# Policy checkpoint (SmolVLA fine-tuned on LIBERO)
python - <<'PY'
from huggingface_hub import snapshot_download
print("policy:", snapshot_download("HuggingFaceVLA/smolvla_libero"))
PY

# LIBERO dataset (lerobot v3 format; caches under $HF_HOME/lerobot)
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("HuggingFaceVLA/libero")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames)
PY

echo "FETCH_DONE"
