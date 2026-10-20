"""Atomic checkpoint file operations that work on restricted shared volumes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_copy_checkpoint(source: Path, destination: Path,
                           expected_sha256: str | None = None) -> str:
    """Copy checkpoint bytes, verify them, then atomically publish the file."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        return file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        # GlobalVolume can reject copy2's metadata/utime step even when file
        # data writes are permitted.
        shutil.copyfile(source, temporary)
        actual_sha = file_sha256(temporary)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise IOError(f"Durable checkpoint copy SHA mismatch for {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return actual_sha
