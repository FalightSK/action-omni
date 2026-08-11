"""
scripts/mechanistic.py
───────────────────────
Mechanistic analysis: component ablations, attention maps, LoRA contribution.

Usage
─────
  python scripts/mechanistic.py --dataset pusht --exp exp02a
  python scripts/mechanistic.py --dataset pusht --exp exp01
  python scripts/mechanistic.py --dataset pusht --exp exp03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanistic analysis")
    parser.add_argument("--dataset", type=str, default="pusht",
                        help="Dataset name: pusht, aloha, …")
    parser.add_argument("--exp",     type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, …")
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.exp)

    # Delegate to existing mechanistic_analysis.py with legacy integer mapping
    from configs.registry import _LEGACY_MAP
    legacy_int = None
    for k, (d, e) in _LEGACY_MAP.items():
        if d == args.dataset and (e == args.exp or e.replace("exp0", "exp") == args.exp):
            legacy_int = k
            break

    if legacy_int is None:
        print(f"[ERROR] No legacy integer mapping for ({args.dataset}, {args.exp}).")
        print("        mechanistic_analysis.py currently only supports exp=1/2/3.")
        print("        Add support for new experiments in mechanistic_analysis.py.")
        sys.exit(1)

    print(f"Delegating to scripts/mechanistic_analysis.py --exp {legacy_int} …")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mechanistic_analysis.py"), "--exp", str(legacy_int)],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
