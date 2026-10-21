"""Verified checkpoint copies and immutable recovery revisions."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
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
    """Copy and verify bytes; rename is atomic only on a supporting filesystem."""
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
        if file_sha256(destination) != actual_sha:
            raise IOError(f"Published checkpoint SHA mismatch for {destination}")
    finally:
        if temporary.exists():
            temporary.unlink()
    return actual_sha


def publish_recovery_revision(source: Path, directory: Path, epoch: int, revision: int) -> Path:
    """Publish a verified SHA-addressed recovery file as the durable checkpoint."""
    checksum = file_sha256(source)
    target = Path(directory) / f"recovery_e{epoch:06d}_r{revision:08d}_{checksum}.pt"
    if not target.is_file() or file_sha256(target) != checksum:
        atomic_copy_checkpoint(source, target, checksum)
    return target


def latest_verified_recovery(directory: Path) -> Path | None:
    candidates = []
    for path in Path(directory).glob("recovery_e*_r*_*.pt"):
        match = re.fullmatch(r"recovery_e(\d+)_r(\d+)_([0-9a-f]{64})\.pt", path.name)
        if match:
            candidates.append((int(match[1]), int(match[2]), match[3], path))
    for _, _, checksum, path in sorted(candidates, reverse=True):
        try:
            if file_sha256(path) == checksum:
                return path
        except OSError:
            pass
    return None
