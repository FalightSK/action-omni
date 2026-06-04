"""
precompute_embeddings.py  [BACKWARD-COMPAT SHIM]
──────────────────────────────────────────────────
Preserved so `python precompute_embeddings.py --exp 2` still works.

New canonical entry point:
  python scripts/precompute.py --dataset pusht --exp exp02a

Argument mapping:
  --exp 1  →  --dataset pusht --exp exp01
  --exp 2  →  --dataset pusht --exp exp02a
  --exp 3  →  --dataset pusht --exp exp03
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

_EXP_MAP = {1: "exp01", 2: "exp02a", 3: "exp03"}

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--exp",               type=int, default=2, choices=[1, 2, 3])
parser.add_argument("--max-samples",       type=int, default=None)
parser.add_argument("--output",            type=str, default=None)
parser.add_argument("--recompute-actions", action="store_true")
parser.add_argument("--recompute-states",  action="store_true")
parser.add_argument("-h", "--help",        action="store_true")
args, extra = parser.parse_known_args()

if args.help:
    print(__doc__)
    print("Full options: python scripts/precompute.py --help")
    sys.exit(0)

exp_id  = _EXP_MAP[args.exp]
cmd     = [sys.executable, str(ROOT / "scripts" / "precompute.py"),
           "--dataset", "pusht", "--exp", exp_id]

if args.max_samples:       cmd += ["--max-samples", str(args.max_samples)]
if args.output:            cmd += ["--output",       args.output]
if args.recompute_actions: cmd += ["--recompute-actions"]

print(f"[precompute_embeddings.py shim] → python scripts/precompute.py --dataset pusht --exp {exp_id}")
result = subprocess.run(cmd, cwd=str(ROOT))
sys.exit(result.returncode)
