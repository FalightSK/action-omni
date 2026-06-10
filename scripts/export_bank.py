"""Export the Language-Table colour-probe bank (block crops + scenes + labels) to disk
so a model in a DIFFERENT conda env (no language_table) can run the same probe."""
from __future__ import annotations
import os, sys, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_vlm_colorblind import build_bank
from configs.registry import get_config

BANK = ROOT / "docs/experiments/language_table/_bank"


def main():
    cfg = get_config("language_table", "exp01")
    crops, scenes = build_bank(cfg)
    BANK.mkdir(parents=True, exist_ok=True)
    crec, srec = [], []
    for i, c in enumerate(crops):
        fn = f"crop_{i:03d}.png"; c["pil"].save(BANK / fn)
        crec.append({"file": fn, "color": c["color"], "shape": c["shape"]})
    for i, s in enumerate(scenes):
        fn = f"scene_{i:03d}.png"; s["pil"].save(BANK / fn)
        srec.append({"file": fn, "instr": s["instr"], "cmd": s["cmd"], "color": s["color"], "shape": s["shape"]})
    (BANK / "labels.json").write_text(json.dumps({"crops": crec, "scenes": srec}, indent=2))
    print(f"exported {len(crec)} crops + {len(srec)} scenes → {BANK}")


if __name__ == "__main__":
    main()
