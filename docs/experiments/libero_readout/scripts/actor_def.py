"""Shared DiT flow-matching actor definition.

Extracted so training and rollout scripts import the same class instead of
duplicating it (or importing a module that executes a rollout at import time).
Context width `dc` differs per backbone: 960 for SmolVLM2-500M, 1024 for
Qwen3.5-0.8B.
"""
import math
import torch
import torch.nn as nn

H, DA = 8, 7          # action chunk length, action dim


class Block(nn.Module):
    def __init__(s, dm, dc, nh=6):
        super().__init__()
        s.n1, s.n2, s.n3 = (nn.LayerNorm(dm, elementwise_affine=False) for _ in range(3))
        s.sa = nn.MultiheadAttention(dm, nh, batch_first=True)
        s.ca = nn.MultiheadAttention(dm, nh, kdim=dc, vdim=dc, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm))
        s.ada = nn.Sequential(nn.SiLU(), nn.Linear(dm, 6 * dm))
        nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)

    def forward(s, x, c, g, kpm):
        a, b, cc, dd, e, f = s.ada(g).chunk(6, -1)
        h = s.n1(x) * (1 + a.unsqueeze(1)) + b.unsqueeze(1)
        x = x + cc.unsqueeze(1) * s.sa(h, h, h, need_weights=False)[0]
        h = s.n2(x)
        x = x + s.ca(h, c, c, key_padding_mask=kpm, need_weights=False)[0]
        h = s.n3(x) * (1 + dd.unsqueeze(1)) + e.unsqueeze(1)
        return x + f.unsqueeze(1) * s.mlp(h)


class Actor(nn.Module):
    def __init__(s, dm=384, dc=960, nb=6):
        super().__init__()
        s.ln = nn.LayerNorm(dc); s.proj = nn.Linear(dc, dc)
        s.inp = nn.Linear(DA, dm); s.pos = nn.Parameter(torch.zeros(1, H, dm))
        s.tau = nn.Sequential(nn.Linear(256, dm), nn.SiLU(), nn.Linear(dm, dm))
        s.st = nn.Linear(8, dm)
        s.blocks = nn.ModuleList([Block(dm, dc) for _ in range(nb)])
        s.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, DA))
        nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)

    def temb(s, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device) / 128)
        a = t[:, None] * f[None] * 1000
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(s, x, t, ctx, state, kpm):
        c = s.proj(s.ln(ctx))
        g = s.tau(s.temb(t)) + s.st(state)
        h = s.inp(x) + s.pos
        for b in s.blocks:
            h = b(h, c, g, kpm)
        return s.out(h)
