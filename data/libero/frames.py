"""LIBERO frame preprocessing — the single source of truth, with no heavy deps.

Both the training dataset and the closed-loop eval agent must preprocess frames
identically; a mismatch makes a working policy look broken, which is
indistinguishable from the hypothesis under test being false.

This lives in its own module rather than in dataset.py because importing
`data.libero.dataset` triggers `data/__init__.py`, which eagerly imports the
PushT dataset and therefore `av` — a video codec the eval path has no use for
and which is not installed in the simulator environment. Keeping the shared
function dependency-free (numpy + PIL only) lets both sides import the same
code without dragging unrelated packages behind it.

Image convention: LIBERO HDF5 demos carry macros_image_convention="opengl", so
frames are stored bottom-up, and the live simulator returns them the same way.
Callers flip once with [::-1] before calling this. That flip is deliberately NOT
done here — the eval agent reads a different observation key than the dataset
does, so the flip belongs at each call site where the array is obtained, and
burying it here would make a double flip easy and silent.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """uint8 (H, W, 3) RGB -> PIL, resized to (width, height).

    Bilinear, matching what the processor would otherwise do at a different
    point in the pipeline; doing it once here keeps train and eval identical.
    """
    img = Image.fromarray(np.ascontiguousarray(rgb_uint8))
    if (img.width, img.height) != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img
