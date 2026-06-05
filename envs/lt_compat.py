"""
Compatibility shims so Google's Language Table PyBullet sim runs inside our
torch env WITHOUT pulling the heavy/old TensorFlow stack.

`language_table.environments.utils.utils_pybullet` does `import tensorflow as tf`
but only uses `tf.io.gfile.GFile` to (de)serialize pybullet state to disk — which
is never called during a normal rollout. So we register a minimal fake
`tensorflow` module that maps gfile onto builtins, letting the import succeed.

Call `install_tf_stub()` BEFORE importing anything from `language_table`.
"""
from __future__ import annotations
import os
import sys
import types


def install_tf_stub() -> None:
    if "tensorflow" in sys.modules:
        return
    tf = types.ModuleType("tensorflow")
    io = types.ModuleType("tensorflow.io")
    gfile = types.ModuleType("tensorflow.io.gfile")
    gfile.GFile = open                       # local file IO is all utils_pybullet needs
    gfile.exists = os.path.exists
    gfile.makedirs = lambda p, *a, **k: os.makedirs(p, exist_ok=True)
    gfile.glob = lambda p: __import__("glob").glob(p)
    io.gfile = gfile
    tf.io = io
    tf.__version__ = "stub-0.0"
    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.io"] = io
    sys.modules["tensorflow.io.gfile"] = gfile
