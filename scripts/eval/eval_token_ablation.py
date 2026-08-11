"""
scripts/eval_token_ablation.py
──────────────────────────────
Closed-loop (in-sim) token-ablation study for ALOHA exp01.

Purpose
───────
The offline token-information probe (scripts/token_info_probe.py) showed that
zeroing TEXT token positions destroys action-prediction loss (+3834%) far more
than zeroing image positions (+15%), and that text tokens linearly predict joint
state almost as well as image tokens (R² 0.895 vs 0.914). That is offline
evidence. This script turns it into a CLOSED-LOOP causal test and adds the
control needed to rule out the obvious alternative explanation.

Two questions this answers
──────────────────────────
Q1  Is the visual information that lives in TEXT tokens actually sufficient to
    *drive the policy to task success* — not just lower a regression loss?
      → condition `text_only`: zero all image-token positions at every replan,
        keep the (live) text tokens, run 50 sim episodes.

Q2  Is the 72% success simply because the adapter + DiT decoder are strong
    enough to solve transfer-cube from PROPRIOCEPTION (joint state) + learned
    motion priors, regardless of what the VLM provides?  If so, corrupting the
    VLM tokens should NOT hurt.  Controls:
      → `no_vlm`        : zero ALL VLM tokens. Decoder sees state + noise only.
                          If SR stays high ⇒ decoder solves it alone (VLM irrelevant).
      → `shuffle_all`   : replace the live tokens with a RANDOM mismatched frame's
                          tokens (identical statistics/norms, wrong content). If SR
                          stays high ⇒ decoder ignores VLM *content*.
      → `text_only_mismatch`: zero image positions AND swap in a mismatched frame's
                          text tokens. Isolates whether it is THIS frame's leaked
                          visual content in text (vs. generic text structure).

The agent ALWAYS receives the true joint state — only the VLM token pathway is
manipulated. This isolates the visual channel from the proprioceptive one.

Interpretation gate
───────────────────
  text_only ≫ no_vlm   AND   text_only ≫ shuffle_all
      ⇒ the policy genuinely uses visual information carried by text tokens;
        the decoder is NOT solving the task from state alone.
  text_only ≈ no_vlm ≈ full
      ⇒ proprioception + priors dominate; VLM contribution is marginal.

Mismatched-frame tokens are drawn from the precomputed embedding cache
(vlm_embeddings.pt) — 20k real VLM token sequences, so statistics are exact.

Usage
─────
  python scripts/eval_token_ablation.py --dataset aloha --exp exp01 --episodes 50
  python scripts/eval_token_ablation.py --episodes 50 \
      --conditions full,text_only,no_vlm,shuffle_all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from envs.aloha_env import AlohaAgent, run_episode


ALL_CONDITIONS = [
    "full",                 # sanity — should reproduce the headline SR
    "text_only",            # zero image positions  (Q1: text sufficiency)
    "image_only",           # zero text positions   (image sufficiency)
    "no_vlm",               # zero all tokens       (Q2: decoder-from-state alone)
    "shuffle_all",          # mismatched frame      (Q2: decoder ignores VLM content)
    "text_only_mismatch",   # zero image + foreign text (isolates frame-specific leak)
]


class AblationAgent(AlohaAgent):
    """AlohaAgent whose VLM tokens are ablated each replan according to `condition`.

    The true joint state is always passed through unchanged — only the VLM token
    tensor (and its image/text mask) is manipulated, so proprioception is never
    removed. That is deliberate: the controls must leave state intact to test
    whether the *visual* pathway matters on top of proprioception.
    """

    def __init__(self, vlm, train_model, cfg, device, condition: str,
                 token_bank: torch.Tensor, mask_bank: torch.Tensor,
                 rng: np.random.Generator) -> None:
        super().__init__(vlm, train_model, cfg, device)
        self.condition  = condition
        self.token_bank = token_bank      # (K, S, 1024) cpu, float
        self.mask_bank  = mask_bank        # (K, S)      cpu, bool
        self.rng        = rng

    def _draw_bank(self, dtype, device):
        """Return (tokens, img_mask) for a random mismatched frame from the cache."""
        j = int(self.rng.integers(0, self.token_bank.shape[0]))
        t = self.token_bank[j:j+1].to(device=device, dtype=dtype)   # (1, S, 1024)
        m = self.mask_bank[j:j+1].to(device=device)                  # (1, S) bool
        return t, m

    @staticmethod
    def _drop(tokens, img_mask, keep):
        """Keep only positions where `keep` (per-position bool, B=1) is True, and
        return the sliced (tokens, img_mask). True masking — dropped tokens are
        physically removed from the sequence, so they never reach the adapter
        readout or the DiT cross-attention (not merely zeroed)."""
        k = keep[0]                                  # (S,) bool
        return tokens[:, k, :], img_mask[:, k]

    def _ablate(self, tokens: torch.Tensor, img_mask: torch.Tensor):
        """Apply the condition to live (tokens, img_mask). img_mask True = image position.

        NOTE on text_only: the VLM was ALREADY run on [image]+[text] in _replan, so the
        text-token hidden states have absorbed image information via Qwen's causal
        attention. Here we DROP the image-token positions and feed only the text tokens
        to the DiT — testing whether that leaked visual info is task-sufficient. We never
        run the VLM on text alone.
        """
        c = self.condition
        if c == "full":
            return tokens, img_mask

        if c == "text_only":                 # drop image positions → only text → DiT
            return self._drop(tokens, img_mask, ~img_mask)

        if c == "image_only":                # drop text positions → only image → DiT
            return self._drop(tokens, img_mask, img_mask)

        if c == "no_vlm":                    # zero everything (state-only control)
            return torch.zeros_like(tokens), img_mask

        if c == "shuffle_all":               # mismatched real frame (both modalities wrong)
            return self._draw_bank(tokens.dtype, tokens.device)

        if c == "text_only_mismatch":        # foreign frame's text only (drop its image)
            t, m = self._draw_bank(tokens.dtype, tokens.device)
            return self._drop(t, m, ~m)

        raise ValueError(f"Unknown condition: {c}")

    @torch.no_grad()
    def _replan(self, image, state) -> None:
        state_t = torch.from_numpy(
            self._norm_state(state).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        inputs           = self.vlm.build_vlm_inputs([image], [self.cfg.task_text], self.device)
        tokens, img_mask = self.vlm.encode_vlm(inputs)
        tokens, img_mask = self._ablate(tokens, img_mask)

        acts_flat = self.train_model.sample(
            tokens, state_t, num_steps=self.cfg.num_flow_steps, img_mask=img_mask
        )
        acts = acts_flat.view(
            self.cfg.action_horizon, self.cfg.action_dim
        ).cpu().float().numpy()
        acts = acts[: self.cfg.inference_horizon]
        self._buffer       = [self._denorm_action(a).astype(np.float32) for a in acts]
        self.replan_count += 1


def _load_token_bank(cfg, k: int, seed: int):
    """Sample k real VLM token sequences from the precomputed cache as a mismatch bank."""
    cache_path = Path(cfg.embeddings_cache)
    if not cache_path.exists():
        print(f"[ERROR] Embedding cache not found: {cache_path}")
        sys.exit(1)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    embeds = cache["embeddings"]      # (N, S, 1024) bf16
    masks  = cache["img_masks"]       # (N, S) bool
    n = embeds.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(k, n), replace=False)
    bank_t = embeds[idx].float()      # (k, S, 1024)
    bank_m = masks[idx]               # (k, S)
    print(f"  Mismatch bank: {bank_t.shape[0]} sequences  shape={tuple(bank_t.shape[1:])}")
    return bank_t, bank_m


def _run_condition(cfg, vlm_model, train_model, device, condition,
                   n_ep, token_bank, mask_bank, seed):
    import gymnasium as gym
    import gym_aloha  # noqa

    rng   = np.random.default_rng(seed + 1)
    agent = AblationAgent(vlm_model, train_model, cfg, device,
                          condition, token_bank, mask_bank, rng)
    env = gym.make(cfg.env_id, obs_type="pixels_agent_pos",
                   max_episode_steps=cfg.sim_max_steps, disable_env_checker=True)

    print(f"\n{'─'*60}\n  Condition: {condition}   ({n_ep} episodes)\n{'─'*60}")
    results = []
    for ep in range(n_ep):
        results.append(run_episode(env, agent, cfg, ep, n_ep, False, None))
    env.close()

    succs = [r["is_success"]   for r in results]
    covs  = [r["max_coverage"] for r in results]
    sr    = float(np.mean(succs))
    print(f"  → {condition}: SR={sr*100:.1f}%  ({sum(succs)}/{n_ep})  "
          f"mean_progress={np.mean(covs)*100:.1f}%")
    return {
        "condition": condition,
        "success_rate": round(sr, 4),
        "n_success": int(sum(succs)),
        "n_episodes": n_ep,
        "mean_progress": round(float(np.mean(covs)), 4),
        "episodes": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop token-ablation study")
    ap.add_argument("--dataset",    type=str, default="aloha")
    ap.add_argument("--exp",        type=str, default="exp01")
    ap.add_argument("--episodes",   type=int, default=20)
    # Default excludes `full` — we already have that SR from the standard eval (72%).
    ap.add_argument("--conditions", type=str,
                    default="text_only,image_only,no_vlm,shuffle_all,text_only_mismatch",
                    help="Comma-separated subset of: " + ",".join(ALL_CONDITIONS))
    ap.add_argument("--bank-size",  type=int, default=256,
                    help="Mismatched-frame token bank size")
    ap.add_argument("--full-sr",    type=float, default=0.72,
                    help="Known full-condition SR baseline (from standard eval) for the "
                         "Q2 comparison when 'full' is not re-run")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     type=str, default=None)
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"[ERROR] Unknown condition {c!r}. Choose from {ALL_CONDITIONS}")
            sys.exit(1)

    cfg    = get_config(args.dataset, args.exp)
    device = cfg.get_device()
    n_ep   = args.episodes
    if n_ep < 50:
        print(f"[WARN] n_ep={n_ep} < 50 — SR conclusions need n>=50.")

    print(f"Dataset: {args.dataset}  Exp: {args.exp}  Device: {device}")
    print(f"Conditions: {conditions}")

    print("\n[1/3] Loading VLM (frozen) …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "best.pt"
    print(f"[2/3] Loading adapter+decoder from {ckpt_path} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()
    print(f"   Epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    token_bank, mask_bank = _load_token_bank(cfg, args.bank_size, args.seed)

    print("\n[3/3] Running ablation conditions …")
    all_results = []
    for cond in conditions:
        all_results.append(
            _run_condition(cfg, vlm_model, train_model, device, cond,
                           n_ep, token_bank, mask_bank, args.seed))

    # ── Summary table + interpretation gate ────────────────────────────────────
    by = {r["condition"]: r for r in all_results}
    print(f"\n{'═'*60}\n  ABLATION SUMMARY  [{args.dataset}/{args.exp}, n={n_ep}]\n{'═'*60}")
    print(f"  {'condition':<22}{'SR':>8}{'progress':>12}")
    for r in all_results:
        print(f"  {r['condition']:<22}{r['success_rate']*100:>7.1f}%{r['mean_progress']*100:>11.1f}%")
    print('═'*60)

    # Two independent verdicts: (Q2) is the decoder solving alone? (Q1) which
    # modality actually carries the task-critical signal in closed loop?
    def sr(c):
        return by[c]["success_rate"] if c in by else None

    full = sr("full") if "full" in by else args.full_sr
    nv   = sr("no_vlm")
    sh   = sr("shuffle_all")
    to   = sr("text_only")
    io   = sr("image_only")

    q2 = q1 = None
    # Q2 — rule out "adapter/DiT strong enough to solve from proprioception alone".
    if full is not None and (nv is not None or sh is not None):
        baselines = [x for x in (nv, sh) if x is not None]
        worst_corrupt = max(baselines)            # most favourable corrupted case
        if full - worst_corrupt >= 0.30:
            q2 = (f"DECODER-ALONE RULED OUT — corrupting the VLM collapses SR "
                  f"(full={full*100:.0f}% vs no_vlm={nv*100 if nv is not None else float('nan'):.0f}%"
                  f"{f', shuffle={sh*100:.0f}%' if sh is not None else ''}). "
                  f"VLM content is causally necessary; proprioception + priors do NOT suffice.")
        else:
            q2 = (f"NOT RULED OUT — corrupted-VLM SR stays high "
                  f"(no_vlm={nv*100 if nv is not None else float('nan'):.0f}%); "
                  f"decoder may be solving largely from state.")
    # Q1 — which modality carries the closed-loop signal?
    if to is not None and io is not None:
        if io - to >= 0.15:
            q1 = (f"IMAGE-CARRIED — image_only={io*100:.0f}% >> text_only={to*100:.0f}%. "
                  f"Spatial image tokens drive control; offline loss OVERSTATED text "
                  f"importance (pooled-context artefact).")
        elif to - io >= 0.15:
            q1 = (f"TEXT-CARRIED — text_only={to*100:.0f}% >> image_only={io*100:.0f}%. "
                  f"Visual info leaked into text registers is task-sufficient.")
        else:
            q1 = (f"SHARED/BOTH-NEEDED — text_only={to*100:.0f}% ≈ image_only={io*100:.0f}%; "
                  f"neither modality alone reproduces full SR.")
    verdict = "  ".join(v for v in (q2, q1) if v)

    summary = {
        "dataset": args.dataset, "exp_id": args.exp,
        "n_episodes": n_ep, "bank_size": args.bank_size, "seed": args.seed,
        "results": all_results,
        "verdict": verdict,
    }
    out = Path(args.output) if args.output else \
        Path(cfg.output_dir) / "token_probe" / "ablation_sim_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    if verdict:
        print("\n  VERDICT:", verdict)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
