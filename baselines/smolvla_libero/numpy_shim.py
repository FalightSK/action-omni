# Installed as `sitecustomize.py` in the environment's site-packages so it loads on
# every interpreter start. numpy >= 1.24 removed these aliases; robosuite 1.4.0
# (LIBERO's pin) still references them. The `dir()` guard avoids triggering numpy-2's
# __getattr__ FutureWarnings.
import numpy as _np

for _name, _type in [("float", float), ("int", int), ("bool", bool),
                     ("object", object), ("str", str), ("unicode", str)]:
    if _name not in dir(_np):
        setattr(_np, _name, _type)
if "infty" not in dir(_np):
    _np.infty = _np.inf
