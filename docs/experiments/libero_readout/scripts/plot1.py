import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=np.load("probe_results.npz")
POOLS=["all","vis","text","instr"]; SUITES=["long","goal","object","spatial"]
L=R["lambda"].shape[0]; x=np.arange(L)
fig,ax=plt.subplots(2,3,figsize=(15,7.5))
C=dict(zip(POOLS,["#444","#1f77b4","#ff7f0e","#d62728"]))
S=dict(zip(SUITES,["#999","#d62728","#1f77b4","#2ca02c"]))

# (a) modality norm ratio + implied attention mass
rho=R["rho"]  # (L,4,2)
a=ax[0,0]
a.plot(x,rho[:,:,0].mean(1),c="#ff7f0e",label=r"$\rho$ = |h_text| / |h_vis|")
a.plot(x,rho[:,:,1].mean(1),c="#d62728",label=r"$\rho$ = |h_instr| / |h_vis|")
a.axhline(1,ls=":",c="k",lw=.8); a.set_title("(a) P5  modality norm ratio"); a.set_xlabel("layer"); a.legend(fontsize=7)
# implied attention mass at init, Tv=64, Ti=6
Tv,Ti=64,6
M=1/(1+(Tv/Ti)*rho[:,:,1].mean(1)**-1)
a2=a.twinx(); a2.plot(x,100*M,c="#888",ls="--",lw=1); a2.set_ylabel("implied instr. attn mass %",fontsize=7,color="#888")

# (b) P1 lambda, per pooling (goal suite)
gi=SUITES.index("goal"); a=ax[0,1]
for p,nm in enumerate(POOLS):
    a.plot(x,R["lambda"][:,p,gi],c=C[nm],label=nm)
a.axhline(1,ls=":",c="k",lw=.8); a.set_yscale("log")
a.set_title(r"(b) P1  $\Lambda$ = swap-dist / paraphrase-dist  [Goal]"); a.set_xlabel("layer"); a.legend(fontsize=7)

# (c) P2 eta2 per pooling (goal)
a=ax[0,2]
for p,nm in enumerate(POOLS): a.plot(x,R["eta2"][:,p,gi],c=C[nm],label=nm)
a.set_title(r"(c) P2  $\eta^2_{instr}$  [Goal]"); a.set_xlabel("layer"); a.set_yscale("log"); a.legend(fontsize=7)

# (d) P2 per suite, instr pooling  <- the conditional-MI prediction
a=ax[1,0]; pi=POOLS.index("instr")
for s,nm in enumerate(SUITES): a.plot(x,R["eta2"][:,pi,s],c=S[nm],label=nm)
a.set_title(r"(d) P2 $\eta^2_{instr}$ by suite (instr-pooled)"); a.set_xlabel("layer"); a.set_yscale("log"); a.legend(fontsize=7)

# (e) P3 ridge R2
a=ax[1,1]
for p,nm in enumerate(POOLS): a.plot(x,R["r2"][:,p,gi],c=C[nm],label=nm)
a.set_title("(e) P3  ridge $R^2$ -> action  [Goal]"); a.set_xlabel("layer"); a.legend(fontsize=7)

# (f) P4 erank + P6 CKA (negative control)
a=ax[1,2]
a.plot(x,R["erank"][:,0,gi],c="#2ca02c",label="P4 erank (all-pool)")
a.set_ylabel("erank",color="#2ca02c"); a.set_xlabel("layer")
b=a.twinx(); b.plot(x,R["cka"][:,0,gi],c="#9467bd",ls="--",label="P6 CKA to final")
b.set_ylim(0,1.05); b.set_ylabel("CKA",color="#9467bd")
a.set_title("(f) P4 erank / P6 CKA [negative control]")
for A in ax.ravel(): A.grid(alpha=.25)
plt.tight_layout(); plt.savefig("fig1_layerwise.png",dpi=150)
print("wrote fig1_layerwise.png")

# numeric summary
np.set_printoptions(precision=3,suppress=True)
print("\n=== P5 norm ratios (mean over suites) ===")
for l in [0,4,8,12,16,20,24,28,32]:
    print(f" L{l:2d}  text/vis={rho[l,:,0].mean():.3f}  instr/vis={rho[l,:,1].mean():.3f}  implied attn mass={100/(1+(64/6)/max(rho[l,:,1].mean(),1e-9)):.2f}%")
print("\n=== best layer per predictor (Goal suite) ===")
for k in ["lambda","eta2","r2","erank"]:
    for p,nm in enumerate(POOLS):
        v=R[k][:,p,gi]; print(f" {k:7s} {nm:6s} argmax=L{int(np.nanargmax(v)):2d}  max={np.nanmax(v):.4f}  L32={v[-1]:.4f}")
print("\n=== eta2 (instr pool) by suite: mean over layers ===")
for s,nm in enumerate(SUITES): print(f" {nm:8s} {R[chr(101)+chr(116)+chr(97)+chr(50)][:,pi,s].mean():.4f}")
print("\n=== CKA range (should be flat/high = insensitive) ===")
print(" CKA min",R["cka"][:,0,gi].min().round(3)," max",R["cka"][:,0,gi].max().round(3))
