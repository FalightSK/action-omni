"""One-time setup for the LIBERO simulator on this machine.

Run with the vla_libero interpreter, NOT the main vla env:
    C:/Users/SK/miniconda3/envs/vla_libero/python.exe scripts/data/_setup_libero.py

Why a separate environment
──────────────────────────
LIBERO pins numpy==1.22.4, robosuite==1.4.0, transformers==4.21.1, gym==0.25.2.
The main `vla` env runs numpy 2.2.3 / torch 2.10 / transformers 5.3 on Python
3.12. Installing LIBERO's stack there would downgrade numpy underneath torch and
break the entire latent study, so the simulator lives in `vla_libero`
(Python 3.10) — the same split already used for `vla_lt_data`/TFDS.

Only the simulator half of LIBERO is installed: robosuite, bddl, mujoco, gym.
Its training stack (robomimic, hydra, wandb, an ancient transformers) is not
needed to create environments or read demonstrations, and pulling it in would
add conflicts for no benefit.

Two Windows/automation problems this fixes
──────────────────────────────────────────
1. `pip install -e` maps nothing. LIBERO's top-level `libero/` directory has no
   __init__.py — it is an implicit namespace package — so setuptools'
   find_packages() returns an empty list and the editable install produces a
   finder with an empty MAPPING. Fixed with a plain .pth pointing at the repo
   root, which lets Python resolve it as a namespace package.

2. `import libero.libero` blocks on input(). On first import it prompts
   "Do you want to specify a custom path for the dataset folder? (Y/N)" and
   raises EOFError under any non-interactive runner. Fixed by writing
   ~/.libero/config.yaml ahead of time.

The config also points `datasets` inside the project tree rather than at
LIBERO's package directory, so downloaded demonstrations land in asset/ with
the rest of the project's data instead of in site-packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LIBERO_SRC = ROOT / "third_party" / "LIBERO"
PKG = LIBERO_SRC / "libero" / "libero"
DATASETS = ROOT / "asset" / "data" / "libero"


def write_pth() -> Path:
    """Put the LIBERO source tree on sys.path for this interpreter."""
    sp = Path(sys.prefix) / "Lib" / "site-packages"
    if not sp.exists():                       # non-Windows layout
        sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    # Remove the broken editable-install artefacts if they are still present.
    for stale in ("__editable__.libero-0.1.0.pth", "__editable___libero_0_1_0_finder.py"):
        p = sp / stale
        if p.exists():
            p.unlink()
            print(f"  removed stale {stale}")
    pth = sp / "libero_src.pth"
    pth.write_text(str(LIBERO_SRC) + "\n", encoding="utf-8")
    print(f"  wrote {pth} -> {LIBERO_SRC}")
    return pth


def write_config() -> Path:
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"
    DATASETS.mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark_root": str(PKG),
        "bddl_files": str(PKG / "bddl_files"),
        "init_states": str(PKG / "init_files"),
        "datasets": str(DATASETS),
        "assets": str(PKG / "assets"),
    }
    cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
    print(f"  wrote {cfg_file}")
    for k, v in cfg.items():
        mark = "ok " if Path(v).exists() else "MISSING"
        print(f"    {mark:8}{k:<16}{v}")
    return cfg_file


MACROS_PRIVATE = '''\
# Written by scripts/data/_setup_libero.py.
#
# robosuite 1.4.1 cannot import on Windows with GPU rendering enabled. In
# utils/binding_utils.py it does:
#
#     if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL") not in ["osmesa", "glx"]:
#         if _SYSTEM == "Darwin": os.environ["MUJOCO_GL"] = "cgl"
#         else:                   os.environ["MUJOCO_GL"] = "egl"
#
# The else branch covers Windows as well as Linux, so MUJOCO_GL is forced to
# "egl" — and the validation immediately below rejects "egl" on Windows, whose
# only accepted backend is "wgl". Setting MUJOCO_GL externally does not help,
# because robosuite overwrites it. Disabling GPU rendering skips that block and
# falls through to GLFWGLContext, which works here.
#
# Offscreen rendering still uses the GPU through GLFW/WGL; this flag only
# controls robosuite's EGL device-selection path, which is Linux-specific.
MUJOCO_GPU_RENDERING = False
'''


def write_macros_private() -> None:
    import robosuite
    p = Path(robosuite.__file__).parent / "macros_private.py"
    p.write_text(MACROS_PRIVATE, encoding="utf-8")
    print(f"  wrote {p}")


def copy_mujoco_dll() -> None:
    """robosuite loads mujoco.dll from its OWN utils/ directory on Windows.

    binding_utils.py does ctypes.WinDLL(os.path.join(os.path.dirname(__file__),
    "mujoco.dll")) — it does not search the mujoco package or PATH — so the DLL
    has to be copied next to it. Re-run this after any mujoco version change,
    or the stale DLL will be loaded against the new Python bindings.
    """
    import filecmp
    import shutil
    import stat

    import mujoco
    import robosuite
    src = Path(mujoco.__file__).parent / "mujoco.dll"
    dst = Path(robosuite.__file__).parent / "utils" / "mujoco.dll"
    if not src.exists():
        print(f"  [warn] {src} not found — skipping DLL copy")
        return
    # Idempotent: re-running setup must not fail on an already-correct DLL.
    # Windows refuses to overwrite it in place (it is read-only once copied, and
    # can be locked if any interpreter still has it loaded), so compare first
    # and only replace when the contents actually differ.
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        print(f"  mujoco.dll already current at {dst}")
        return
    if dst.exists():
        dst.chmod(dst.stat().st_mode | stat.S_IWRITE)
        try:
            dst.unlink()
        except PermissionError:
            print(f"  [ERROR] {dst} is locked — close any running LIBERO/robosuite "
                  "process and re-run")
            return
    shutil.copy2(src, dst)
    print(f"  copied mujoco.dll -> {dst}  (mujoco {mujoco.__version__})")


def check_versions() -> bool:
    """robosuite 1.4.1 targets the MuJoCo 2.3 API.

    Under mujoco 3.x the controller dies with
    `'MjData' object has no attribute 'qM'` only once an environment is
    stepped — long after import succeeds — so pin-checking here is worth more
    than it looks.
    """
    import mujoco
    ok = True
    major = int(mujoco.__version__.split(".")[0])
    if major != 2:
        print(f"  [ERROR] mujoco {mujoco.__version__} is incompatible with "
              f"robosuite 1.4.1 (needs 2.3.x — 3.x removes MjData.qM).")
        print("          fix: pip install 'mujoco==2.3.7' && re-run this script")
        ok = False
    else:
        print(f"  mujoco {mujoco.__version__} ok for robosuite 1.4.1")
    return ok


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("LIBERO setup")
    if not LIBERO_SRC.exists():
        print(f"  ERROR: {LIBERO_SRC} not found — clone it first:")
        print("    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "
              "third_party/LIBERO")
        return 1
    write_pth()
    write_config()
    write_macros_private()
    copy_mujoco_dll()
    ok = check_versions()
    print("\nnext: verify with scripts/data/_verify_libero.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
