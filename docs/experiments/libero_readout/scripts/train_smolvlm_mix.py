"""Train the SmolVLM2 read-out on MIXED phrasings (condition B), tap 30, all-token.

Same actor/optimiser/schedule as sweep2.py so condition A and B differ only in
which instruction strings the training frames carried.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CKPT = os.path.join(os.path.dirname(HERE), "checkpoints")
sys.path.insert(0, HERE)
from actor_def import Actor, H, DA
W, DIM, MAXI = 96, 960, 20

dev = "cuda"
X = np.load(os.path.join(RES, "smix_feats.npy"), mmap_mode="r")   # (N,1,1,W,DIM)
M = np.load(os.path.join(RES, "smix_mask.npy"))
Atr = np.load(os.path.join(RES, "smix_act.npy"))
Str = np.load(os.path.join(RES, "smix_state.npy"))
mu, sd = Atr.reshape(-1, DA).mean(0), Atr.reshape(-1, DA).std(0) + 1e-6
smu, ssd = Str.mean(0), Str.std(0) + 1e-6
Atr_n = torch.tensor((Atr - mu) / sd, dtype=torch.float32)
Str_n = torch.tensor((Str - smu) / ssd, dtype=torch.float32)
np.savez(os.path.join(RES, "smix_norm.npz"), mu=mu, sd=sd, smu=smu, ssd=ssd)

keep = M[:, 0] > 0
N = len(Atr_n)
idx = np.zeros((N, W), dtype=np.int64); pad = np.ones((N, W), dtype=bool)
for i in range(N):
    w = np.where(keep[i])[0]
    idx[i, :len(w)] = w; pad[i, :len(w)] = False

Xt = torch.from_numpy(np.asarray(X[:, 0, 0])).float()
Ct = torch.gather(Xt, 1, torch.from_numpy(idx)[:, :, None].expand(-1, -1, DIM))
Pt = torch.from_numpy(pad); del Xt
print("train tensor", tuple(Ct.shape), flush=True)

torch.manual_seed(0); np.random.seed(0)
m = Actor().to(dev)
lr, steps, bs = 1e-3, 2000, 128
opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.1)
rs = np.random.RandomState(0); t0 = time.time()
for it in range(steps):
    b = rs.randint(0, N, bs)
    a = Atr_n[b].to(dev); ctx = Ct[b].to(dev); kpm = Pt[b].to(dev); st = Str_n[b].to(dev)
    t = torch.rand(bs, device=dev); e = torch.randn_like(a)
    xt = t[:, None, None] * e + (1 - t[:, None, None]) * a
    loss = ((m(xt, t, ctx, st, kpm) - (e - a)) ** 2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if it % 500 == 0:
        print(f"  step {it} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
out = os.path.join(CKPT, "ck2mix_all_tap30_lr0.001.pt")
torch.save(m.state_dict(), out)
print("saved", out, flush=True)
print("SMIX_TRAIN_DONE")
