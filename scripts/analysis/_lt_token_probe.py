"""
Measure how many tokens Qwen's processor emits for a 320x180 Language Table frame
+ a representative instruction -> sets img_grid_h/w and img_seq_len in the config.
Processor-only (no model weights loaded).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2]))
from transformers import AutoProcessor
from configs.registry import get_config

cfg = get_config("aloha", "exp01")          # reuse a known-good Qwen model_path
print("model_path:", cfg.model_path, flush=True)
proc = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
IMG_TOK = cfg.image_token_id

img = Image.fromarray((np.random.rand(180, 320, 3) * 255).astype("uint8"))  # 320w x 180h
print("image size (W,H):", img.size, flush=True)

insts = [
    "push the red moon to the blue cube",
    "move the blue moon in between the yellow pentagon and green cube",
    "adjust the green and blue cube slightly to the down to form a horizontal line of blocks",
    "slide the blue moon diagonally top left to the red pentagon then stop",
]

maxlen = 0
for txt in insts:
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": txt}]}]]
    chat = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages]
    inp = proc(text=chat, images=[img], padding=True, return_tensors="pt")
    ids = inp["input_ids"][0]
    n_img = int((ids == IMG_TOK).sum())
    maxlen = max(maxlen, len(ids))
    grid = inp.get("image_grid_thw")
    print(f"  total={len(ids):3d}  img_tokens={n_img:3d}  text/other={len(ids)-n_img:3d}  "
          f"grid_thw={None if grid is None else grid.tolist()}  | {txt[:45]}", flush=True)

print(f"\nMAX total seq len over samples: {maxlen}", flush=True)
print("PROBE OK", flush=True)
