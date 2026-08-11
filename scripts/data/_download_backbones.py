"""One-time download of the VLA backbones used by the latent-comparison study.

Everything lands in `asset/models/` inside the repo — NOT the HuggingFace hub
cache — so all large artifacts stay inside the project tree.

Four models, forming a 2x2 of (architecture) x (robot-finetuned vs stock):

  pi05_base    lerobot/pi05_base    14.5 GB  robot-finetuned PaliGemma-3B (gemma_2b)
  paligemma    google/paligemma...  11.7 GB  stock control for pi05
  smolvla_base lerobot/smolvla_base  0.9 GB  robot-finetuned SmolVLM2-500M (16 layers)
  smolvlm2     HuggingFaceTB/Smol...  ~1 GB  stock control for smolvla

The stock repos are needed for two reasons: they are the pre-robot-finetuning
control arm, and they carry the processor/tokenizer configs that the lerobot
checkpoints omit (lerobot ships weights only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

ROOT = Path(__file__).parents[2]
DEST = ROOT / "asset" / "models"

# Skip onnx/gguf/tflite mirrors of the same weights — they inflate SmolVLM2 from
# ~1 GB to ~8 GB and we only ever load the safetensors through transformers.
WEIGHTS_ONLY = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "tokenizer*",
    "preprocessor_config.json",
]

JOBS = [
    ("lerobot/smolvla_base", "smolvla_base", None),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "smolvlm2_500m", WEIGHTS_ONLY),
    ("google/paligemma-3b-pt-224", "paligemma_3b_pt_224", WEIGHTS_ONLY),
    ("lerobot/pi05_base", "pi05_base", None),
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for repo_id, folder, patterns in JOBS:
        target = DEST / folder
        print(f"\n=== {repo_id} -> {target}", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(target),
                allow_patterns=patterns,
                max_workers=8,
            )
            print(f"    done: {repo_id}", flush=True)
        except Exception as exc:  # keep going; report at the end
            print(f"    FAILED {repo_id}: {type(exc).__name__}: {exc}", flush=True)
    print("\nAll download jobs finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
