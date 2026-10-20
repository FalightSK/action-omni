"""Stage revision-locked raw data/models locally; never compute VLM features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_protocol import file_digest, load_protocol, write_json
from scripts.run_log import log_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero", "evaluation"), required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Cloud asset root containing assets_manifest.json and assets/")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    parser.add_argument("--prepare-protocol", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads((args.source_root / "assets_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_revisions") != plan["source_revisions"]:
        raise ValueError("Cloud assets do not match the experiment's pinned source revisions")
    if args.dataset != "evaluation" and args.dataset not in manifest.get("datasets_ready", []):
        raise ValueError(f"Cloud asset manifest does not declare {args.dataset} ready")
    groups = ["models/groot_n17_3b", "models/qwen3_vl_2b", "models/cosmos_reason2_2b"]
    if args.dataset != "evaluation":
        groups.append("data/aloha_sim_transfer_cube_human" if args.dataset == "aloha" else "data/libero")
    source = (args.source_root / "assets").resolve()
    target = ROOT / "asset"
    copied = verified = 0
    for group in groups:
        directory = source / group
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = [p for p in sorted(directory.rglob("*")) if p.is_file() and ".cache" not in p.parts]
        if not files:
            raise ValueError(f"Empty asset group: {directory}")
        log_event("worker.assets_group_started", dataset=args.dataset, group=group, files=len(files))
        for index, src in enumerate(files, 1):
            relative = src.relative_to(source)
            expected = manifest.get("files", {}).get(relative.as_posix())
            dst = target / relative
            matches = dst.is_file() and dst.stat().st_size == src.stat().st_size
            if matches and expected:
                matches = file_digest(dst) == expected["sha256"]
            if not matches:
                dst.parent.mkdir(parents=True, exist_ok=True)
                partial = dst.with_name(dst.name + ".staging")
                shutil.copyfile(src, partial)
                if partial.stat().st_size != src.stat().st_size:
                    raise IOError(f"Incomplete asset copy: {relative}")
                if expected and file_digest(partial) != expected["sha256"]:
                    raise ValueError(f"Asset checksum mismatch: {relative}")
                partial.replace(dst)
                copied += 1
            if expected:
                if dst.stat().st_size != expected["size"]:
                    raise ValueError(f"Manifest size mismatch: {relative}")
                verified += 1
            if index == len(files) or index % 25 == 0:
                log_event("worker.assets_progress", dataset=args.dataset, group=group,
                          files=index, total_files=len(files), copied_files=copied,
                          verified_files=verified, elapsed_s=round(time.perf_counter() - started, 1))
        selected = [name for name in manifest.get("files", {}) if name.startswith(group + "/")]
        if not selected:
            raise ValueError(f"No checksummed manifest entries for {group}")
        for name in selected:
            if not (target / name).is_file():
                raise FileNotFoundError(f"Manifest asset missing after staging: {name}")
    if args.prepare_protocol:
        if args.dataset == "evaluation":
            parser.error("Evaluation needs checkpoint metadata, not a raw-data split")
        from configs.registry import get_config
        from scripts.prepare_protocol import prepare
        protocol_path = ROOT / plan["datasets"][args.dataset]["split_manifest"]
        if protocol_path.exists():
            protocol = load_protocol(protocol_path, args.dataset)
        else:
            protocol = prepare(args.dataset, Path(get_config(args.dataset, "exp05").dataset_root))
            if protocol["protocol_id"] != protocol_path.stem:
                raise ValueError("Prepared episode split differs from the locked experiment")
            write_json(protocol_path, protocol)
        if protocol["protocol_id"] != protocol_path.stem:
            raise ValueError("Local episode split differs from the locked experiment")
        log_event("worker.protocol_ready", dataset=args.dataset, protocol=str(protocol_path),
                  protocol_id=protocol["protocol_id"])
    report = {"dataset": args.dataset, "source_root": str(args.source_root),
              "source_revisions": plan["source_revisions"], "groups": groups,
              "copied_files": copied, "verified_files": verified,
              "elapsed_s": round(time.perf_counter() - started, 1)}
    write_json(target / f"staging_{args.dataset}.json", report)
    log_event("worker.assets_ready", **report)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("worker.assets_failed", level="ERROR", error_type=type(exc).__name__, error=str(exc))
        raise
