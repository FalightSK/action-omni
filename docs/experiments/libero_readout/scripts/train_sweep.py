"""Tap-depth sweep. Read-out R^tap_j = W.LN(H^(j)); DiT flow-matching actor.
DV: val flow loss + open-loop divergence RATIO  D_swap / D_para  (scale-free, see notes)."""
import torch, torch.nn as nn, numpy as np, json, sys, time, math
dev="cuda"
TAPS=np.load("taps.npy"); H,DA=8,7
Xtr=np.load("train_feats.npy",mmap_mode="r"); Atr=np.load("train_act.npy"); Str=np.load("train_state.npy")
Xva=np.load("val_feats.npy",mmap_mode="r");   Ava=np.load("val_act.npy");   Sva=np.load("val_state.npy")
mu,sd=Atr.reshape(-1,DA).mean(0),Atr.reshape(-1,DA).std(0)+1e-6
smu,ssd=Str.mean(0),Str.std(0)+1e-6
Atr_n=torch.tensor((Atr-mu)/sd,dtype=torch.float32); Ava_n=torch.tensor((Ava-mu)/sd,dtype=torch.float32)
Str_n=torch.tensor((Str-smu)/ssd,dtype=torch.float32); Sva_n=torch.tensor((Sva-smu)/ssd,dtype=torch.float32)

class Block(nn.Module):
    def __init__(s,d,dc,nh=6):
        super().__init__()
        s.n1,s.n2,s.n3=(nn.LayerNorm(d,elementwise_affine=False) for _ in range(3))
        s.sa=nn.MultiheadAttention(d,nh,batch_first=True)
        s.ca=nn.MultiheadAttention(d,nh,kdim=dc,vdim=dc,batch_first=True)
        s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        s.ada=nn.Sequential(nn.SiLU(),nn.Linear(d,6*d)); nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)
    def forward(s,x,c,g):
        a,b,cc,dd,e,f=s.ada(g).chunk(6,-1)
        h=s.n1(x)*(1+a.unsqueeze(1))+b.unsqueeze(1); x=x+cc.unsqueeze(1)*s.sa(h,h,h,need_weights=False)[0]
        h=s.n2(x); x=x+s.ca(h,c,c,need_weights=False)[0]
        h=s.n3(x)*(1+dd.unsqueeze(1))+e.unsqueeze(1); x=x+f.unsqueeze(1)*s.mlp(h)
        return x
class Actor(nn.Module):
    def __init__(s,d=384,dc=960,nb=6):
        super().__init__()
        s.ln=nn.LayerNorm(dc)                       # per-layer LN on the tap  (methodological requirement)
        s.proj=nn.Linear(dc,dc)
        s.inp=nn.Linear(DA,d); s.pos=nn.Parameter(torch.zeros(1,H,d))
        s.tau=nn.Sequential(nn.Linear(256,d),nn.SiLU(),nn.Linear(d,d))
        s.st=nn.Linear(8,d)
        s.blocks=nn.ModuleList([Block(d,dc) for _ in range(nb)])
        s.out=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,DA)); nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)
    def temb(s,t):
        f=torch.exp(-math.log(10000)*torch.arange(128,device=t.device)/128)
        a=t[:,None]*f[None]*1000
        return torch.cat([a.sin(),a.cos()],-1)
    def forward(s,x,t,ctx,state):
        c=s.proj(s.ln(ctx))
        g=s.tau(s.temb(t))+s.st(state)
        h=s.inp(x)+s.pos
        for b in s.blocks: h=b(h,c,g)
        return s.out(h)

def train_one(tap_i,lr,steps=2000,bs=128,seed=0,log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m=Actor().to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=.1)
    N=len(Atr_n); rs=np.random.RandomState(seed)
    Xt=torch.from_numpy(np.asarray(Xtr[:,0,tap_i])).float()      # (N,81,960) in RAM
    for it in range(steps):
        b=rs.randint(0,N,bs)
        a=Atr_n[b].to(dev); ctx=Xt[b].to(dev); st=Str_n[b].to(dev)
        t=torch.rand(bs,device=dev); e=torch.randn_like(a)
        xt=t[:,None,None]*e+(1-t[:,None,None])*a
        loss=((m(xt,t,ctx,st)-(e-a))**2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
    # ---- eval
    m.eval(); Xv=torch.from_numpy(np.asarray(Xva[:,:,tap_i])).float()   # (n,5,81,960)
    n=Xv.shape[0]; vl=0.0
    with torch.no_grad():
        g=torch.Generator(device=dev).manual_seed(1234)
        for s in range(0,n,256):
            a=Ava_n[s:s+256].to(dev); ctx=Xv[s:s+256,0].to(dev); st=Sva_n[s:s+256].to(dev)
            t=torch.rand(a.shape[0],device=dev,generator=g); e=torch.randn(a.shape,device=dev,generator=g)
            xt=t[:,None,None]*e+(1-t[:,None,None])*a
            vl+=((m(xt,t,ctx,st)-(e-a))**2).mean().item()*a.shape[0]
        vl/=n
        # deterministic 3-step Euler rollout under each instruction variant, SAME noise
        preds=[]
        for v in range(5):
            out=[]
            for s in range(0,n,256):
                ctx=Xv[s:s+256,v].to(dev); st=Sva_n[s:s+256].to(dev)
                gg=torch.Generator(device=dev).manual_seed(999)
                x=torch.randn((ctx.shape[0],H,DA),device=dev,generator=gg)
                NS=10
                for k in range(NS):
                    t=torch.full((ctx.shape[0],),1-k/NS,device=dev)
                    x=x-(1/NS)*m(x,t,ctx,st)
                out.append(x.cpu())
            preds.append(torch.cat(out))
    P=torch.stack(preds)                     # (5,n,H,DA)  normalised action chunks
    d_para=((P[0]-P[1:4])**2).sum(-1).mean().item()
    d_swap=((P[0]-P[4])**2).sum(-1).mean().item()
    # ---- OPEN-LOOP ACTION ERROR vs ground-truth demo chunk (the real DV) ----
    G=Ava_n                                   # (n,H,DA) ground truth, normalised
    sdt=torch.tensor(sd,dtype=torch.float32)  # de-normalise to raw action units
    ol={}
    names=["orig","para1","para2","para3","swap"]
    for i,nm in enumerate(names):
        e=P[i]-G
        ol[f"ol_mse_{nm}"]=float((e**2).mean())
        ol[f"ol_l2_{nm}"]=float((e**2).sum(-1).sqrt().mean())
        ol[f"ol_l2raw_{nm}"]=float(((e*sdt)**2).sum(-1).sqrt().mean())
    ol["ol_mse_para"]=float(np.mean([ol[f"ol_mse_para{k}"] for k in (1,2,3)]))
    # degradation ratios: ~1.0 == the policy is ignoring the instruction
    ol["deg_swap"]=ol["ol_mse_swap"]/max(ol["ol_mse_orig"],1e-12)
    ol["deg_para"]=ol["ol_mse_para"]/max(ol["ol_mse_orig"],1e-12)
    # per-task-conditional check: does the swap hurt more than a reword?
    ol["swap_over_para"]=ol["ol_mse_swap"]/max(ol["ol_mse_para"],1e-12)
    torch.save(m.state_dict(), f"ckpt_tap{int(TAPS[tap_i])}_lr{lr:g}.pt")
    return dict(tap=int(TAPS[tap_i]),lr=lr,val_loss=vl,d_para=d_para,d_swap=d_swap,
                ratio=d_swap/max(d_para,1e-12),**ol)
if __name__=="__main__":
    LRS=[3e-4,1e-3,3e-3]
    R=[]
    for i in range(len(TAPS)):
        for lr in LRS:
            t0=time.time(); r=train_one(i,lr); r["sec"]=round(time.time()-t0,1); R.append(r)
            print(json.dumps(r),flush=True)
            json.dump(R,open("sweep.json","w"),indent=1)
    print("DONE")
