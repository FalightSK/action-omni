"""Small structured logging helpers shared by the training pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


def log_event(event: str, *, level: str = "INFO", **fields) -> dict:
    """Write one timestamped JSON event to stdout and flush it immediately."""
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    print(json.dumps(record, sort_keys=True, ensure_ascii=True, default=str), flush=True)
    return record


def write_json_atomic(path: str | Path, payload: dict) -> None:
    """Publish a complete JSON status file so readers never see a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
