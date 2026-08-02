"""Corrected tap-depth sweep.

Fixes vs v1:
  * 96-token window (v1 truncated at 81, clipping 39/50 instruction variants)
  * key_padding_mask on cross-attention (v1 attended to padded zeros)
  * READ-OUT VARIANT as an IV: ctx = all tokens  vs  instruction tokens only,
    on identical cached features.

Pre-registered prediction (from Figure 1, recorded BEFORE this ran):
  ridge-R^2 range across layers = 0.264 instruction-pooled vs 0.053 all-pooled,
  so the depth effect should appear for ctx=instr and be ~flat for ctx=all.

DV = open-loop rollout action error vs ground-truth chunk (not flow val loss).
"""
import json, math, sys, time
import numpy as np
import torch
import torch.nn as nn

dev = "cuda"
TAPS = np.load("taps.npy")
H, DA, W = 8, 7, 96
MAXI = 20                                   # max instruction tokens kept

Xtr = np.load("train_feats2.npy", mmap_mode="r")     # (N,1,12,96,960)
Mtr = np.load("train_mask2.npy")                     # (N,1,96) 0pad 1vis 2scaffold 3instr
Xva = np.load("val_feats2.npy", mmap_mode="r")       # (n,5,12,96,960)
Mva = np.load("val_mask2.npy")
Atr, Str = np.load("train_act.npy"), np.load("train_state.npy")
Ava, Sva = np.load("val_act.npy"), np.load("val_state.npy")

mu, sd = Atr.reshape(-1, DA).mean(0), Atr.reshape(-1, DA).std(0) + 1e-6
smu, ssd = Str.mean(0), Str.std(0) + 1e-6
Atr_n = torch.tensor((Atr - mu) / sd, dtype=torch.float32)
Ava_n = torch.tensor((Ava - mu) / sd, dtype=torch.float32)
Str_n = torch.tensor((Str - smu) / ssd, dtype=torch.float32)
Sva_n = torch.tensor((Sva - smu) / ssd, dtype=torch.float32)


def gather_idx(mask, mode):
    """-> (N, T') int64 index array + (N, T') bool pad mask"""
    N = mask.shape[0]
    if mode == "all":
        keep = mask > 0
        T = W
        idx = np.zeros((N, T), dtype=np.int64)
        pad = np.ones((N, T), dtype=bool)
        for i in range(N):
            w = np.where(keep[i])[0]
            idx[i, :len(w)] = w
            pad[i, :len(w)] = False
        return idx, pad
    keep = mask == 3
    idx = np.zeros((N, MAXI), dtype=np.int64)
    pad = np.ones((N, MAXI), dtype=bool)
    for i in range(N):
        w = np.where(keep[i])[0][:MAXI]
        idx[i, :len(w)] = w
        pad[i, :len(w)] = False
    return idx, pad


class Block(nn.Module):
    def __init__(s, d, dc, nh=6):
        super().__init__()
        s.n1, s.n2, s.n3 = (nn.LayerNorm(d, elementwise_affine=False) for _ in range(3))
        s.sa = nn.MultiheadAttention(d, nh, batch_first=True)
        s.ca = nn.MultiheadAttention(d, nh, kdim=dc, vdim=dc, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)

    def forward(s, x, c, g, kpm):
        a, b, cc, dd, e, f = s.ada(g).chunk(6, -1)
        h = s.n1(x) * (1 + a.unsqueeze(1)) + b.unsqueeze(1)
        x = x + cc.unsqueeze(1) * s.sa(h, h, h, need_weights=False)[0]
        h = s.n2(x)
        x = x + s.ca(h, c, c, key_padding_mask=kpm, need_weights=False)[0]
        h = s.n3(x) * (1 + dd.unsqueeze(1)) + e.unsqueeze(1)
        x = x + f.unsqueeze(1) * s.mlp(h)
        return x


class Actor(nn.Module):
    def __init__(s, d=384, dc=960, nb=6):
        super().__init__()
        s.ln = nn.LayerNorm(dc); s.proj = nn.Linear(dc, dc)
        s.inp = nn.Linear(DA, d); s.pos = nn.Parameter(torch.zeros(1, H, d))
        s.tau = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.st = nn.Linear(8, d)
        s.blocks = nn.ModuleList([Block(d, dc) for _ in range(nb)])
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, DA))
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


def run(tap_i, mode, lr, steps=2000, bs=128, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    itr, ptr = gather_idx(Mtr[:, 0], mode)
    Xt = torch.from_numpy(np.asarray(Xtr[:, 0, tap_i])).float()          # (N,96,960)
    Ct = torch.gather(Xt, 1, torch.from_numpy(itr)[:, :, None].expand(-1, -1, 960))
    Pt = torch.from_numpy(ptr)
    del Xt
    m = Actor().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.1)
    N = len(Atr_n); rs = np.random.RandomState(seed)
    for it in range(steps):
        b = rs.randint(0, N, bs)
        a = Atr_n[b].to(dev); ctx = Ct[b].to(dev); kpm = Pt[b].to(dev); st = Str_n[b].to(dev)
        t = torch.rand(bs, device=dev); e = torch.randn_like(a)
        xt = t[:, None, None] * e + (1 - t[:, None, None]) * a
        loss = ((m(xt, t, ctx, st, kpm) - (e - a)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    del Ct
    # ---- eval: open-loop rollout per instruction variant, identical noise ----
    m.eval(); n = Ava_n.shape[0]; preds = []; vl = 0.0
    with torch.no_grad():
        for v in range(5):
            iv, pv = gather_idx(Mva[:, v], mode)
            Xv = torch.from_numpy(np.asarray(Xva[:, v, tap_i])).float()
            Cv = torch.gather(Xv, 1, torch.from_numpy(iv)[:, :, None].expand(-1, -1, 960))
            Pv = torch.from_numpy(pv); del Xv
            out = []
            for s in range(0, n, 256):
                ctx = Cv[s:s + 256].to(dev); kpm = Pv[s:s + 256].to(dev)
                st = Sva_n[s:s + 256].to(dev)
                if v == 0:
                    a = Ava_n[s:s + 256].to(dev)
                    g = torch.Generator(device=dev).manual_seed(1234 + s)
                    t = torch.rand(a.shape[0], device=dev, generator=g)
                    e = torch.randn(a.shape, device=dev, generator=g)
                    xt = t[:, None, None] * e + (1 - t[:, None, None]) * a
                    vl += ((m(xt, t, ctx, st, kpm) - (e - a)) ** 2).mean().item() * a.shape[0]
                gg = torch.Generator(device=dev).manual_seed(999)
                x = torch.randn((ctx.shape[0], H, DA), device=dev, generator=gg)
                NS = 10
                for k in range(NS):
                    tt = torch.full((ctx.shape[0],), 1 - k / NS, device=dev)
                    x = x - (1 / NS) * m(x, tt, ctx, st, kpm)
                out.append(x.cpu())
            preds.append(torch.cat(out)); del Cv
    vl /= n
    P = torch.stack(preds)
    G = Ava_n
    ol = {}
    for i, nm in enumerate(["orig", "para1", "para2", "para3", "swap"]):
        ol[f"ol_mse_{nm}"] = float(((P[i] - G) ** 2).mean())
    ol["ol_mse_para"] = float(np.mean([ol[f"ol_mse_para{k}"] for k in (1, 2, 3)]))
    ol["deg_swap"] = ol["ol_mse_swap"] / max(ol["ol_mse_orig"], 1e-12)
    ol["deg_para"] = ol["ol_mse_para"] / max(ol["ol_mse_orig"], 1e-12)
    ol["swap_over_para"] = ol["ol_mse_swap"] / max(ol["ol_mse_para"], 1e-12)
    ol["d_para"] = float(((P[0][None] - P[1:4]) ** 2).sum(-1).mean())
    ol["d_swap"] = float(((P[0] - P[4]) ** 2).sum(-1).mean())
    torch.save(m.state_dict(), f"ck2_{mode}_tap{int(TAPS[tap_i])}_lr{lr:g}.pt")
    return dict(tap=int(TAPS[tap_i]), mode=mode, lr=lr, val_loss=vl, **ol)


if __name__ == "__main__":
    R = []
    for mode in ("all", "instr"):
        for i in range(len(TAPS)):
            t0 = time.time(); r = run(i, mode, 1e-3); r["sec"] = round(time.time() - t0, 1)
            R.append(r); print(json.dumps(r), flush=True)
            json.dump(R, open("sweep2.json", "w"), indent=1)
    # A4 re-check at 3 LRs on the shallowest / mid / deepest tap, both modes
    for mode in ("all", "instr"):
        for i in (0, len(TAPS) // 2, len(TAPS) - 1):
            for lr in (3e-4, 3e-3):
                r = run(i, mode, lr); r["lr_check"] = True
                R.append(r); print(json.dumps(r), flush=True)
                json.dump(R, open("sweep2.json", "w"), indent=1)
    print("SWEEP2_DONE")
