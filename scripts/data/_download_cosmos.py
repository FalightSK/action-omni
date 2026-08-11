"""Download nvidia/Cosmos-Reason2-2B — the stock control for GR00T N1.7.

Why this matters: without it, GR00T could be measured but not *explained*. A
task/scene ratio or a subspace angle is a joint product of the base model and
whatever finetuning did to it, so a single number identifies neither. Stock
PaliGemma scoring 0.60 on task/scene — higher than GR00T's 0.50, with no robot
data at all — is the concrete proof that levels alone say nothing. With Cosmos
present, every GR00T statement becomes a within-pair difference, and Finding 3's
paired test grows from 6/6 across two families to 9/9 across three.

Destination is asset/models/cosmos_reason2_2b (weights + tokenizer together).
The earlier tokenizer-only directory (…_tok) exists because GR00T ships weights
with no vocab; once this completes, that directory is redundant and the groot
arm reads its processor from here instead.

Xet is disabled for the same reason as _download_groot.py: this account's
transfers stall inside hf_xet's chunk-reconstruction step, while the plain HTTP
downloader works. The stall watchdog from that script is reused unchanged.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi

ROOT = Path(__file__).parents[2]
DEST = ROOT / "asset" / "models" / "cosmos_reason2_2b"
REPO = "nvidia/Cosmos-Reason2-2B"
# Everything except the README's benchmark PNGs.
PATTERNS = ["model.safetensors", "config.json", "generation_config.json",
            "tokenizer*", "vocab.json", "merges.txt", "chat_template.json",
            "*preprocessor_config.json"]

STALL_SECONDS = 90
POLL_SECONDS = 10
MAX_ATTEMPTS = 6

CHILD_CODE = f"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={REPO!r}, repo_type="model", local_dir={str(DEST)!r},
    allow_patterns={PATTERNS!r}, max_workers=2,
)
"""


def matches(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in PATTERNS)


def verify(api: HfApi) -> tuple[bool, list[str]]:
    info = api.model_info(REPO, files_metadata=True)
    problems = []
    for s in info.siblings:
        if not matches(s.rfilename):
            continue
        p = DEST / s.rfilename
        size = s.size or 0
        if not p.exists():
            problems.append(f"missing: {s.rfilename} ({size/1e9:.2f} GB)")
        elif size and p.stat().st_size != size:
            problems.append(f"size mismatch: {s.rfilename} "
                            f"{p.stat().st_size/1e9:.2f} vs {size/1e9:.2f} GB")
    return not problems, problems


def total_bytes() -> int:
    return sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())


def run_attempt() -> bool:
    proc = subprocess.Popen([sys.executable, "-c", CHILD_CODE])
    last, last_growth = total_bytes(), time.monotonic()
    try:
        while True:
            if (ret := proc.poll()) is not None:
                print(f"    child exited, code={ret}", flush=True)
                return ret == 0
            time.sleep(POLL_SECONDS)
            now = total_bytes()
            if now != last:
                last, last_growth = now, time.monotonic()
                print(f"    progress: {now/1e9:.2f} GB", flush=True)
            elif time.monotonic() - last_growth > STALL_SECONDS:
                print(f"    STALLED at {now/1e9:.2f} GB — killing child", flush=True)
                proc.kill(); proc.wait(timeout=15)
                return False
    finally:
        if proc.poll() is None:
            proc.kill()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"=== attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        if not run_attempt():
            for p in DEST.rglob("*.incomplete"):
                print(f"    clearing partial {p.name}", flush=True)
                p.unlink()
        ok, problems = verify(api)
        if ok:
            print("    verified: every file present at its expected size", flush=True)
            print("COSMOS_DOWNLOAD_OK", flush=True)
            return 0
        print(f"    verification failed ({len(problems)}):", flush=True)
        for p in problems[:5]:
            print(f"      {p}", flush=True)
    print("COSMOS_DOWNLOAD_FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
