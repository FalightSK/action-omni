"""
Uniform loader for the five backbone arms of the latent-comparison study.

The point of this module is to hide the fact that the three model families keep
their weights in three different shapes, and expose one interface:

    bb = load_backbone("pi05")
    pooled = bb.encode(images, texts)   # {(depth, pool): (B, D) float32}

Arms
────
  qwen       Qwen3.5-0.8B, stock            — the project's own frozen backbone
  pi05       lerobot/pi05_base              — robot-finetuned PaliGemma-3B
  paligemma  google/paligemma-3b-pt-224     — stock control for pi05
  smolvla    lerobot/smolvla_base           — robot-finetuned SmolVLM2-500M (16 layers)
  smolvlm2   HuggingFaceTB/SmolVLM2-500M    — stock control for smolvla
  groot      nvidia/GR00T-N1.7-3B           — robot-finetuned Qwen3-VL (16 layers)
  cosmos     nvidia/Cosmos-Reason2-2B       — stock control for groot (28 layers)
  cosmos16   the same weights truncated to GR00T's first 16 layers — a
             depth-matched control, so GR00T's finetuning is not confounded with
             the fact that it also shortened its own language stack.

The two lerobot arms ship a *policy* state dict (VLM + action expert fused), not
a HF model. We instantiate the matching HF architecture from the stock repo's
config and graft in the VLM subtree, discarding the action expert — we are
studying the perceptual representation, not the action head. The stock repo also
supplies the processor/tokenizer, which the lerobot repos omit entirely.

Depth handling: layers are addressed by *relative* depth, not absolute index,
because the arms have different stack heights (Qwen 28, PaliGemma-3B 18,
SmolVLM2 32, SmolVLA 16). Comparing "layer 24" across them would be meaningless;
comparing "75% of the way up" is at least defensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parents[3]
MODELS = ROOT / "asset" / "models"

# Fractions of the language-model stack at which to read hidden states.
#
# 0.00 is a control, not a filler point: it is hidden_states[0], the vision
# projector output concatenated with the text embedding table, BEFORE any
# language-model block runs. Whatever it already decodes is attributable to the
# image encoder and the embedding lookup alone, so the rise from 0.00 to the
# peak is the part the language stack actually contributes. Without it, a curve
# starting at 0.50 cannot distinguish "the stack builds this" from "the encoder
# already had it".
DEPTHS = (0.00, 0.25, 0.50, 0.75, 1.00)
POOLS = ("image", "text", "all")

# ── Where each policy ACTUALLY reads, and how ────────────────────────────────
#
# Single source of truth for the report's methods table. Every entry is sourced
# from the shipped config or checkpoint, never from the paper text alone, and
# the provenance string names the file it came from so a reader can re-check it.
#
# The important consequence: a stock control must be read at the layer its
# ROBOT DESCENDANT reads, not at its own last layer. SmolVLM2 has 32 layers but
# SmolVLA consumes layer 16, so probing SmolVLM2 at 32 compares layer 16 against
# layer 32 and measures depth, not finetuning. That confound is what retracted
# the study's original "robot pretraining separates image and text" finding.
#
# read_mode distinguishes what the real policy consumes:
#   "hidden"  a single layer's hidden state, exactly as the probe takes it
#   "kv"      the action expert attends to per-layer key/value tensors at EVERY
#             layer, so no single hidden state reproduces its input — the probe
#             is an approximation and the report must say so.
DOC_LAYER = {
    # arm          layer  n_layers  read_mode  provenance
    "qwen":        (24, 24, "hidden", "ours — configs read the last layer"),
    "pi05":        (18, 18, "kv",     "pi0/pi0.5 action expert attends VLM KV at every layer"),
    "paligemma":   (18, 18, "kv",     "stock base of pi05 — matched to pi05's depth"),
    # SmolVLA's expert interleaves with the VLM (self_attn_every_n_layers=2), so
    # like Pi-0.5 it consumes per-layer K/V, not one hidden state. Earlier rows
    # said "hidden" here, which contradicted the config this table cites.
    "smolvla":     (16, 16, "kv",     "smolvla_base/config.json num_vlm_layers=16, self_attn_every_n_layers=2"),
    "smolvlm2":    (16, 32, "kv",     "matched to smolvla's num_vlm_layers=16 (not its own 32)"),
    "groot":       (16, 16, "hidden", "groot_n17_3b/config.json select_layer=16"),
    "cosmos":      (16, 28, "hidden", "matched to groot's select_layer=16 (not its own 28)"),
    "qwen3vl":     (16, 28, "hidden", "matched to groot's select_layer=16 (not its own 28)"),
}

# Robot arm -> the checkpoint it was initialised from. Used for every paired
# claim; an arm with no entry cannot enter a controlled finetuned-vs-stock
# comparison. Every pair is verified by direct tensor comparison, not taken from
# a model card — one published claim already failed that check:
#   smolvla vs smolvlm2   345/345 identical      -> VLM frozen (train_expert_only)
#   groot   vs cosmos     476/493 differ, ~1.8%  -> VLM finetuned (tune_llm=True)
#   cosmos  vs qwen3vl    584/625 differ, ~1.6%  -> VLM finetuned
# The GR00T row is why "GR00T just IS Cosmos" is false: had its backbone been
# frozen, it would read 493/493 identical like the SmolVLA row does.
PAIRS = [("pi05", "paligemma"), ("smolvla", "smolvlm2"),
         ("groot", "cosmos"), ("cosmos", "qwen3vl")]

# ── The roster, and the datasets ─────────────────────────────────────────────
#
# Named ARMS, not MODELS, because MODELS is already the models *directory* in
# this module. Every consumer imports these two lists instead of keeping its own
# copy — the previous per-file copies are exactly how fig15 came to plot seven
# curves under a caption claiming nine.
ARMS = ["qwen", "pi05", "paligemma", "smolvla", "smolvlm2",
        "groot", "cosmos", "qwen3vl"]

# Language Table is excluded from every per-arm analysis: its action signal is
# absent for all arms (R^2 <= 0.063), so any ablation on it returns "no effect"
# regardless of what is ablated, and including it inflates cross-dataset
# distances with a dataset nothing can learn.
KEYS = ["aloha_transfer", "aloha_insertion", "libero_goal"]

# The one place Language Table is still used. The dataset gate's whole argument
# is that LT fails it while LIBERO-Goal passes, so deleting LT here would remove
# the negative anchor that makes the gate a usable scale rather than two points.
GATE_KEYS = ["aloha_transfer", "aloha_insertion", "libero_goal", "language_table"]


def _final_norm(model) -> torch.nn.Module | None:
    """The language stack's final normalisation module, or None.

    Needed because HF applies this norm ONLY to the last entry of
    hidden_states. Reading a full model at an intermediate layer therefore
    returns a PRE-norm vector, while reading a truncated model at its own last
    layer returns a POST-norm one — so a stock control read at its descendant's
    depth is not the same object the descendant reads, even when the weights are
    bit-identical.

    Measured on the frozen SmolVLA/SmolVLM2 pair (345/345 tensors identical,
    both read at layer 16): mean norm 25.35 vs 297.76 and cosine similarity
    0.22. RMSNorm carries learned per-channel gains, so it is a diagonal map
    that rotates as well as rescales — not something a downstream
    standardisation undoes.
    """
    for path in ("model.language_model.norm", "language_model.norm",
                 "model.text_model.norm", "text_model.norm",
                 "model.norm", "norm"):
        m = model
        try:
            for part in path.split("."):
                m = getattr(m, part)
        except AttributeError:
            continue
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _resolve_final_norm(model, hidden_size: int, name: str) -> torch.nn.Module:
    """_final_norm, but it must succeed and must be the right module.

    A silent None here is the worst possible failure: that arm alone would keep
    pre-norm intermediate reads while every other arm got post-norm ones, and
    nothing downstream could tell. So this raises instead, and additionally
    checks the module's weight width against the language stack — a vision-tower
    norm would otherwise satisfy the name lookup and be silently wrong.
    """
    m = _final_norm(model)
    if m is None:
        raise RuntimeError(
            f"[{name}] final norm module not found. Add its path to _final_norm; "
            f"do NOT extract without it — intermediate reads would be pre-norm "
            f"while last-layer reads are post-norm, which is not comparable."
        )
    w = getattr(m, "weight", None)
    if w is None or w.shape[-1] != hidden_size:
        raise RuntimeError(
            f"[{name}] resolved final norm {type(m).__name__} has width "
            f"{None if w is None else tuple(w.shape)}, expected {hidden_size}. "
            f"Wrong module — likely a vision-tower norm."
        )
    return m


def _cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (keys, values) from whatever cache object this version returns.

    transformers has moved this API more than once — legacy tuple-of-tuples,
    then key_cache/value_cache lists, now Cache.layers[i].keys/.values — and the
    probe should not break on a library upgrade, so all three are accepted.
    """
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return [(k, v) for k, v in cache]
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(l.keys, l.values) for l in layers]
    kc, vc = getattr(cache, "key_cache", None), getattr(cache, "value_cache", None)
    if kc is not None and vc is not None:
        return list(zip(kc, vc))
    to_legacy = getattr(cache, "to_legacy_cache", None)
    if to_legacy is not None:
        return [(k, v) for k, v in to_legacy()]
    return []


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    processor: object
    image_token_id: int
    n_layers: int
    hidden_size: int
    build_inputs: object = field(repr=False, default=None)
    # Absolute index of the layer this arm's real policy consumes, from
    # DOC_LAYER. None only if the arm has no documented read.
    doc_layer: int | None = None
    # "hidden" or "kv" — what the real policy consumes. Drives whether the KV
    # tap is computed at all.
    read_mode: str = "hidden"
    # The language stack's final norm, resolved once at load and required.
    final_norm: torch.nn.Module | None = field(repr=False, default=None)

    def layer_index(self, frac: float) -> int:
        """hidden_states[0] is the embedding output, so layer k lives at k.

        frac == 0 must return 0, not 1. The previous max(1, ...) clamp existed
        when 0.50 was the shallowest depth sampled and would silently relabel
        the pre-stack control as layer 1 — reporting an encoder-only baseline
        that had in fact been through a transformer block.
        """
        if frac <= 0.0:
            return 0
        return max(1, int(round(frac * self.n_layers)))

    @torch.no_grad()
    def encode(self, images: list, texts: list[str], device: str = "cuda") -> dict:
        inp = self.build_inputs(self.processor, images, texts)
        inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}

        # Never run the LM head or the loss. Both are pure waste here — the probe
        # reads hidden states only — and on PaliGemma they dominate everything
        # else: logits are (B, 269, 257216), i.e. 69M elements PER FRAME, and
        # some processors also emit `labels`, which triggers a full
        # cross-entropy on top. Measured on pi05 at B=32 that was 30.3 GB peak
        # on a 16.3 GB card, so extraction paged through host memory and ran at
        # 4.6 img/s against Cosmos's 56.
        #
        # HF convention: XForConditionalGeneration.model is the backbone without
        # the head. Fall back to the full module if an arm does not follow it.
        inp.pop("labels", None)
        core = getattr(self.model, "model", self.model)
        # use_cache only when this arm's policy actually reads K/V; it costs a
        # little memory and nothing else.
        want_kv = self.read_mode == "kv"
        out = core(**inp, output_hidden_states=True, return_dict=True,
                   use_cache=want_kv)
        hs = out.hidden_states

        ids = inp["input_ids"]
        attn = inp.get("attention_mask", torch.ones_like(ids))
        img_m = (ids == self.image_token_id) & attn.bool()
        txt_m = (~(ids == self.image_token_id)) & attn.bool()
        all_m = attn.bool()
        masks = {"image": img_m, "text": txt_m, "all": all_m}

        res: dict[tuple, np.ndarray] = {}
        # The relative grid, plus "doc": the ABSOLUTE layer this arm's real
        # policy reads. Those are not the same point and must not be conflated —
        # GR00T reads layer 16 of Cosmos's 28, i.e. relative depth 0.571, which
        # is not on the grid at all. Snapping it to the nearest grid point would
        # probe layer 14 and label it as the documented read.
        taps: list[tuple] = [(f, self.layer_index(f)) for f in DEPTHS]
        if self.doc_layer is not None:
            taps.append(("doc", self.doc_layer))

        # Apply the final norm to intermediate reads so every tap means the same
        # thing: "what this stack outputs if truncated here". Without it, an arm
        # read at its own last layer is post-norm while a deeper arm read at the
        # same absolute layer is pre-norm, and the two are not comparable — which
        # silently confounded the GR00T/Cosmos and SmolVLA/SmolVLM2 pairs, the
        # two where the stock control is deeper than its descendant.
        fin = self.final_norm
        for tag, li in taps:
            h = hs[li]
            if fin is not None and li < len(hs) - 1:
                h = fin(h)
            h = h.float()  # (B, S, D)
            for pool, m in masks.items():
                w = m.unsqueeze(-1).float()
                denom = w.sum(1).clamp(min=1.0)
                # Mean pooling over the tokens selected by `m`, with the
                # attention mask applied so padding never enters the average.
                res[(tag, pool)] = ((h * w).sum(1) / denom).cpu().numpy()

        # ── the KV tap ───────────────────────────────────────────────────────
        # For Pi-0.5 and SmolVLA the action expert never sees the residual
        # stream: it attends to per-layer keys and values. Those are a LINEAR
        # map of the hidden state (K = W_k h), and under multi-query attention a
        # much narrower one -- PaliGemma has 8 attention heads but only 1 KV
        # head, so each layer exposes 2*256 dims out of a 2048-wide stream.
        #
        # Two consequences, in opposite directions, which is why neither the
        # hidden tap nor this one alone is honest:
        #   per layer   a linear probe on h upper-bounds a linear probe on K,V
        #   all layers  the concatenation is not bounded by any single h
        #
        # Layers 1..doc_layer only: SmolVLA's expert sees SmolVLM2's first 16 of
        # 32, so including the rest would credit it with input it never gets.
        if want_kv:
            kv = _cache_layers(out.past_key_values)
            if kv:
                for pool, m in masks.items():
                    w = m.unsqueeze(-1).float()          # (B, S, 1)
                    parts = []
                    for k_t, v_t in kv[: self.doc_layer or len(kv)]:
                        for t in (k_t, v_t):
                            # (B, n_kv_heads, S, head_dim) -> (B, S, n*d)
                            x = t.float().permute(0, 2, 1, 3).flatten(2)
                            denom = w.sum(1).clamp(min=1.0)
                            parts.append((x * w).sum(1) / denom)
                    res[("kv", pool)] = torch.cat(parts, dim=-1).cpu().numpy()
        return res


# ── input builders (each family has its own prompt convention) ────────────────

def _qwen_inputs(processor, images, texts):
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": txt},
        ]}]
        for img, txt in zip(images, texts)
    ]
    chat = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    return processor(text=chat, images=images, padding=True, return_tensors="pt")


def _paligemma_inputs(processor, images, texts):
    # Pi-0/Pi-0.5 feed the bare instruction with a trailing newline; mirror that
    # so the finetuned arm is evaluated in the format it was trained on.
    prompts = [f"{t}\n" for t in texts]
    return processor(text=prompts, images=images, padding=True, return_tensors="pt")


def _qwen3vl_inputs(processor, images, texts):
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": txt},
        ]}]
        for img, txt in zip(images, texts)
    ]
    chat = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    return processor(text=chat, images=images, padding=True, return_tensors="pt")


def _smolvlm_inputs(processor, images, texts):
    messages = [
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": txt},
        ]}]
        for txt in texts
    ]
    chat = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    return processor(text=chat, images=[[i] for i in images], padding=True, return_tensors="pt")


# ── graft helpers ────────────────────────────────────────────────────────────

def _graft(model: torch.nn.Module, ckpt: Path, prefix: str, n_keep: int | None = None):
    """Load the VLM subtree of a lerobot policy checkpoint into a HF model.

    `prefix` is stripped from every key; keys outside it (the action expert) are
    dropped. When `n_keep` is set, language layers with index >= n_keep are also
    dropped so the tensor set matches a truncated config.
    """
    # ckpt may be a single .safetensors file (lerobot arms) or a directory of
    # shards plus an index (GR00T ships two shards).
    if ckpt.is_dir():
        sd = {}
        for shard in sorted(ckpt.glob("*.safetensors")):
            sd.update(load_file(str(shard)))
    else:
        sd = load_file(str(ckpt))
    ref = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        nk = k[len(prefix):]
        if n_keep is not None:
            m = re.search(r"layers\.(\d+)\.", nk)
            if m and int(m.group(1)) >= n_keep:
                continue
        # Pi-0.5 trims PaliGemma's padded vocab (257216 -> 257152), which removes
        # the <image> row. Keep the padded matrix and fill only the leading rows:
        # the image row is masked-scattered over with vision features downstream,
        # so its content is never read.
        if nk in ref and ref[nk].shape != v.shape:
            tgt = ref[nk]
            if tgt.dim() == v.dim() and tgt.shape[1:] == v.shape[1:] and tgt.shape[0] > v.shape[0]:
                padded = tgt.clone()
                padded[: v.shape[0]] = v.to(padded.dtype)
                print(f"    row-padded {nk}: {tuple(v.shape)} -> {tuple(tgt.shape)}")
                v = padded
            else:
                print(f"    SKIP shape-mismatch {nk}: ckpt{tuple(v.shape)} vs model{tuple(tgt.shape)}")
                continue
        # Cast to the dtype the model was BUILT with. Checkpoints here store
        # F32; without this, load_state_dict leaves those tensors float32 and
        # the result is a mixed-precision model. Measured on pi05: 439 bf16 +
        # 164 float32 tensors, 10.86 GB of weights against stock PaliGemma's
        # 5.85 GB, and 8.8 img/s against 32.2 — same architecture, same inputs.
        # It also means a grafted arm and its from_pretrained control were never
        # matched on numerical precision.
        out[nk] = v.to(ref[nk].dtype) if nk in ref else v
    missing, unexpected = model.load_state_dict(out, strict=False)
    # Belt and braces: any parameter the graft did not touch, or that
    # load_state_dict assigned rather than copied, is forced to one dtype here.
    ref_dtype = next(iter(ref.values())).dtype
    model.to(ref_dtype)
    hard = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    print(f"    grafted {len(out)} tensors | missing={len(hard)} unexpected={len(unexpected)}")
    if hard[:5]:
        print(f"      e.g. missing: {hard[:5]}")
    if unexpected[:5]:
        print(f"      e.g. unexpected: {unexpected[:5]}")
    return model


def _img_token_id(cfg, processor) -> int:
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    tok = getattr(processor, "tokenizer", processor)
    for s in ("<image>", "<|image_pad|>", "<|vision_pad|>"):
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i >= 0:
            return i
    raise RuntimeError("could not resolve image token id")


# ── public loader ────────────────────────────────────────────────────────────

def load_backbone(name: str, device: str = "cuda", dtype=torch.bfloat16) -> Backbone:
    from transformers import (
        AutoConfig, AutoModel, AutoModelForImageTextToText, AutoProcessor,
        PaliGemmaForConditionalGeneration,
    )

    if name == "qwen":
        src = "Qwen/Qwen3.5-0.8B"
        proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        model = AutoModel.from_pretrained(src, dtype=dtype, trust_remote_code=True)
        cfg = model.config
        txt_cfg = getattr(cfg, "text_config", cfg)
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      txt_cfg.num_hidden_layers, txt_cfg.hidden_size, _qwen_inputs)

    elif name in ("paligemma", "pi05"):
        src = MODELS / "paligemma_3b_pt_224"
        proc = AutoProcessor.from_pretrained(str(src))
        if name == "paligemma":
            model = PaliGemmaForConditionalGeneration.from_pretrained(str(src), dtype=dtype)
        else:
            cfg = AutoConfig.from_pretrained(str(src))
            model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
            _graft(model, MODELS / "pi05_base" / "model.safetensors",
                   prefix="paligemma_with_expert.paligemma.")
        cfg = model.config
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size,
                      _paligemma_inputs)

    elif name in ("smolvlm2", "smolvla", "smolvlm2_16"):
        src = MODELS / "smolvlm2_500m"
        # The stock SmolVLM2 processor upscales to 2048px and tiles the image
        # (do_image_splitting=True). lerobot's SmolVLA does neither: it feeds one
        # padded 512x512 frame per camera. Tiling would both evaluate SmolVLA
        # outside its training format and inflate the sequence ~5x, so turn it
        # off for BOTH SmolVLM arms — they must stay mutually comparable.
        proc = AutoProcessor.from_pretrained(
            str(src), do_image_splitting=False, size={"longest_edge": 512}
        )
        if name == "smolvlm2":
            model = AutoModelForImageTextToText.from_pretrained(str(src), dtype=dtype)
        elif name == "smolvlm2_16":
            # Depth-matched control for SmolVLA, mirroring cosmos16 for GR00T.
            # SmolVLA truncates 32 -> 16 layers, so pairing it against full-depth
            # SmolVLM2 confounds robot finetuning with reading half as deep. The
            # GR00T pair showed that confound was doing ALL the work there, so
            # this pair must be checked the same way rather than assumed clean.
            cfg = AutoConfig.from_pretrained(str(src))
            cfg.text_config.num_hidden_layers = 16
            model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
            _graft(model, src / "model.safetensors", prefix="", n_keep=16)
        else:
            cfg = AutoConfig.from_pretrained(str(src))
            cfg.text_config.num_hidden_layers = 16  # SmolVLA truncates the LM
            model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
            _graft(model, MODELS / "smolvla_base" / "model.safetensors",
                   prefix="model.vlm_with_expert.vlm.", n_keep=16)
        cfg = model.config
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size,
                      _smolvlm_inputs)
    elif name == "groot":
        # GR00T N1.7's backbone is Cosmos-Reason2-2B, a stock
        # Qwen3VLForConditionalGeneration — but the GR00T repo ships weights
        # only (no tokenizer/vocab files), so its config/tokenizer/image
        # processor must come from the Cosmos repo either way. Prefer the full
        # Cosmos download when present so both arms of the pair read byte-identical
        # processor config; fall back to the tokenizer-only directory that
        # existed before Cosmos was fetched as a control arm.
        src = MODELS / "cosmos_reason2_2b"
        if not src.exists():
            src = MODELS / "cosmos_reason2_2b_tok"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found. The 'groot' backbone needs "
                "Cosmos-Reason2-2B's tokenizer/processor files (its own repo "
                "ships weights only) — fetch those before loading this arm."
            )
        # Qwen3-VL is a dynamic-resolution model: left alone the processor sizes
        # by the input frame and the token count drifts per dataset. GR00T feeds
        # a fixed 256x256 (image_target_size in its config), so pin the same
        # budget here — 256/16 patches merged 2x2 = 64 image tokens, constant
        # across all three datasets and close to Qwen's 80.
        proc = AutoProcessor.from_pretrained(
            str(src), size={"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
        )
        cfg = AutoConfig.from_pretrained(str(src))
        # GR00T keeps only the first 16 LM layers — its config's select_layer
        # is 16, so everything above that was dropped from the checkpoint. This
        # is verifiable, not assumed: the shards contain language_model.layers
        # 0-15 and nothing above.
        cfg.text_config.num_hidden_layers = 16
        model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
        _graft(model, MODELS / "groot_n17_3b", prefix="backbone.model.", n_keep=16)
        cfg = model.config
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size,
                      _qwen3vl_inputs)

    elif name in ("cosmos", "cosmos16"):
        # Stock control for GR00T. Two variants, because GR00T changed both the
        # weights AND the depth (28 -> 16 layers), and those must not be
        # confounded:
        #   cosmos    full 28 layers — the model as published. Pairs with GR00T
        #             under the study's relative-depth convention (each arm read
        #             at its own last layer), same as smolvlm2 vs smolvla.
        #   cosmos16  first 16 layers — GR00T's exact architecture minus the
        #             robot finetuning. This is a strictly better control than
        #             either other pair has: identical width, identical depth,
        #             identical processor, so the only remaining difference is
        #             the finetuning itself.
        src = MODELS / "cosmos_reason2_2b"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found — run scripts/data/_download_cosmos.py first."
            )
        proc = AutoProcessor.from_pretrained(
            str(src), size={"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
        )
        if name == "cosmos":
            model = AutoModelForImageTextToText.from_pretrained(str(src), dtype=dtype)
        else:
            cfg = AutoConfig.from_pretrained(str(src))
            cfg.text_config.num_hidden_layers = 16
            model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
            _graft(model, src / "model.safetensors", prefix="", n_keep=16)
        cfg = model.config
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size,
                      _qwen3vl_inputs)

    elif name == "qwen3vl":
        # The stock root of the GR00T chain. Cosmos-Reason2-2B declares
        # architectures=["Qwen3VLForConditionalGeneration"] with 28 text layers
        # at hidden 2048 — i.e. it IS a finetuned Qwen3-VL-2B — so this is the
        # only checkpoint in the study that has seen neither robot data nor
        # physical-reasoning data, and it completes the three-level chain:
        #   qwen3vl (stock) -> cosmos (physical AI) -> groot (robot actions)
        # Same processor budget as cosmos/groot so token counts match exactly.
        src = MODELS / "qwen3_vl_2b"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found — download Qwen/Qwen3-VL-2B-Instruct into it."
            )
        proc = AutoProcessor.from_pretrained(
            str(src), size={"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
        )
        model = AutoModelForImageTextToText.from_pretrained(str(src), dtype=dtype)
        cfg = model.config
        bb = Backbone(name, model, proc, _img_token_id(cfg, proc),
                      cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size,
                      _qwen3vl_inputs)

    else:
        raise ValueError(f"unknown backbone: {name}")

    bb.model.to(device).eval()
    for p in bb.model.parameters():
        p.requires_grad_(False)
    # Resolved here, not lazily inside encode(), so a missing or wrong module
    # fails before any GPU time is spent rather than after.
    bb.final_norm = _resolve_final_norm(bb.model, bb.hidden_size, name)
    print(f"    final norm: {type(bb.final_norm).__name__}"
          f"(width={tuple(bb.final_norm.weight.shape)})")

    # Attach the documented read layer, and assert the table matches the model
    # that actually loaded. DOC_LAYER is hand-transcribed from shipped configs,
    # so it is exactly the kind of table that silently rots when a checkpoint is
    # swapped; failing loudly here is far cheaper than discovering afterwards
    # that a whole extraction probed the wrong depth.
    if name in DOC_LAYER:
        layer, n_expect, mode, prov = DOC_LAYER[name]
        if n_expect != bb.n_layers:
            raise ValueError(
                f"DOC_LAYER['{name}'] says {n_expect} layers but the loaded model "
                f"has {bb.n_layers}. Fix the table, do not adjust the model."
            )
        if not (0 <= layer <= bb.n_layers):
            raise ValueError(
                f"DOC_LAYER['{name}'] layer {layer} outside 0..{bb.n_layers}"
            )
        bb.doc_layer = layer
        bb.read_mode = mode
        print(f"  [{name}] layers={bb.n_layers} hidden={bb.hidden_size} "
              f"img_tok={bb.image_token_id} | doc read: layer {layer} "
              f"({layer / bb.n_layers:.0%}, {mode}) — {prov}")
    else:
        print(f"  [{name}] layers={bb.n_layers} hidden={bb.hidden_size} "
              f"img_tok={bb.image_token_id} | no documented read layer")
    return bb
