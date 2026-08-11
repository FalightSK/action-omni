"""Download and verify nvidia/GR00T-N1.7-3B — GR00T only, no stock control.

Cosmos-Reason2-2B's *weights* are deliberately NOT fetched: it is not an arm in
this study, so its 4.9 GB would buy nothing. Its small tokenizer/processor text
files ARE fetched, because GR00T's own repo ships weights with no vocab and its
backbone cannot be tokenized without them.

Note on this file's history: the two big shards were in the end downloaded
manually through the website after the stall workarounds below still could not
finish them. The watchdog logic is kept because it is what makes an unattended
re-run survivable, but if it stalls again, downloading the two shards by hand
into asset/models/groot_n17_3b/ is a legitimate and faster fix — this script
verifies byte sizes against the Hub afterwards either way.

Robustness — two failure modes were observed on this network and both are
worked around explicitly:

  1. A plain snapshot_download call can hang with NO exception raised: not
     "slow", but a `.incomplete` file whose mtime stops advancing entirely and
     never resumes, even left for 10+ minutes. A passive read-timeout inside
     the same process does not help here, because the call that's stuck
     doesn't raise — it just never returns. So downloading happens in a CHILD
     PROCESS, and the parent polls total bytes on disk every 15s; if bytes
     haven't grown for STALL_SECONDS, the parent kills the child outright
     (something a timeout *inside* the hung call cannot do).

  2. Resuming a large partial `.incomplete` file (an HTTP Range request) is
     what actually hangs — a *fresh* full download of the same file does not
     exhibit this. So a stale `.incomplete` (one that triggered the stall
     watchdog) is deleted before the next attempt, trading the lost partial
     bytes for a request shape that this network actually completes.

After every attempt, every expected file's local byte size is verified against
what the Hub itself reports for that file — "the file exists" is not enough,
since an abandoned download leaves a short file that looks present.
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
# ROOT CAUSE, found by manually tracing the request: this repo is stored on
# HF's newer "Xet" chunk/CAS backend, and `hf_xet` was already installed in
# this env — so it silently activated on the very first attempt, before
# hf_transfer was ever added, and both "different backends" above were
# actually the same hung Xet chunk-reconstruction path. A manual HTTP Range
# GET against the plain resolve/CDN URL succeeded immediately (1MB in ~2.7s),
# proving the network and CDN are fine — it's specifically hf_xet's chunk
# manifest / reconstruction step that hangs on this file. Force it off so
# huggingface_hub uses the plain HTTP downloader that's already proven to work.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi

ROOT = Path(__file__).parents[2]
DEST = ROOT / "asset" / "models" / "groot_n17_3b"
REPO = "nvidia/GR00T-N1.7-3B"
PATTERNS = ["config.json", "model*.safetensors", "model.safetensors.index.json",
            "processor_config.json", "embodiment_id.json",
            "experiment_cfg/final_model_config.json",
            "experiment_cfg/final_processor_config.json"]

# GR00T ships weights but no vocab, so its backbone cannot be tokenized without
# these. Text files only — deliberately NOT model.safetensors, which would pull
# 4.9 GB of Cosmos weights we have no use for (Cosmos is not an arm in this
# study; see scripts/analysis/latent_compare/README.md).
TOK_REPO = "nvidia/Cosmos-Reason2-2B"
TOK_DEST = ROOT / "asset" / "models" / "cosmos_reason2_2b_tok"
TOK_PATTERNS = ["tokenizer*", "vocab.json", "merges.txt", "chat_template.json",
                "config.json", "*preprocessor_config.json", "generation_config.json"]

STALL_SECONDS = 90    # no byte growth for this long -> treat the child as hung
POLL_SECONDS = 10
MAX_ATTEMPTS = 6

CHILD_CODE = f"""
import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={REPO!r}, repo_type="model", local_dir={str(DEST)!r},
    allow_patterns={PATTERNS!r}, max_workers=2,
)
"""


def expected_sizes(api: HfApi) -> dict[str, int]:
    info = api.model_info(REPO, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def matches_patterns(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in PATTERNS)


def verify(api: HfApi) -> tuple[bool, list[str]]:
    exp = expected_sizes(api)
    problems = []
    for name, size in exp.items():
        if not matches_patterns(name):
            continue
        p = DEST / name
        if not p.exists():
            problems.append(f"missing: {name} (expected {size/1e9:.2f} GB)")
        elif size and p.stat().st_size != size:
            problems.append(
                f"size mismatch: {name} local={p.stat().st_size/1e9:.2f} GB "
                f"expected={size/1e9:.2f} GB"
            )
    return (len(problems) == 0), problems


def total_bytes() -> int:
    return sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())


def clear_stale_incomplete() -> None:
    """Delete .incomplete files — a fresh GET succeeds where a Range-resume of
    a large partial hangs on this network, so the lost partial bytes are worth
    trading away rather than letting the next attempt retry the same resume."""
    for p in DEST.rglob("*.incomplete"):
        print(f"    clearing stale partial: {p.name} ({p.stat().st_size/1e9:.2f} GB)", flush=True)
        p.unlink()


def run_one_attempt() -> bool:
    """Run the download in a child process; kill it if bytes stop growing.
    Returns True iff the child exited cleanly on its own (not killed)."""
    proc = subprocess.Popen([sys.executable, "-c", CHILD_CODE])
    last_total = total_bytes()
    last_growth = time.monotonic()

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(f"    child exited on its own, code={ret}", flush=True)
                return ret == 0

            time.sleep(POLL_SECONDS)
            now_total = total_bytes()
            if now_total != last_total:
                last_total = now_total
                last_growth = time.monotonic()
                print(f"    progress: {now_total/1e9:.2f} GB on disk", flush=True)
            elif time.monotonic() - last_growth > STALL_SECONDS:
                print(f"    STALLED — no growth for {STALL_SECONDS}s at "
                      f"{now_total/1e9:.2f} GB, killing child", flush=True)
                proc.kill()
                proc.wait(timeout=15)
                return False
    finally:
        if proc.poll() is None:
            proc.kill()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    ok = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"=== attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        clean_exit = run_one_attempt()
        if not clean_exit:
            clear_stale_incomplete()

        ok, problems = verify(api)
        if ok:
            print("    verification passed — every file present at its expected size", flush=True)
            break
        print(f"    verification failed ({len(problems)} problem(s)):", flush=True)
        for p in problems[:10]:
            print(f"      {p}", flush=True)
        if attempt < MAX_ATTEMPTS:
            print("    retrying …", flush=True)

    if ok:
        # Small text files, plain snapshot_download — these are kilobytes and
        # have never exhibited the stall the big shards do, so they don't need
        # the watchdog machinery above.
        from huggingface_hub import snapshot_download
        snapshot_download(TOK_REPO, local_dir=str(TOK_DEST),
                          allow_patterns=TOK_PATTERNS)
        print(f"    fetched tokenizer/processor -> {TOK_DEST}", flush=True)
        print("GROOT_DOWNLOAD_OK", flush=True)
        return 0
    print("GROOT_DOWNLOAD_FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
