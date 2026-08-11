import numpy as np, json
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
d=np.load("scenes.npz"); ti=d["ti"]; ep=d["ep"]; act=d["action"]; st=d["state"]
Z=np.load("feats.npy",mmap_mode="r")          # (N,7,L,4,D)
NRM=np.load("norms.npy")                      # (N,7,L,3) vis,text,instr
N,NV,L,NP,D=Z.shape
POOLS=["all","vis","text","instr"]; SUITES=["long","goal","object","spatial"]
suite=ti//10
ORIG,PARA,SWAP,BLANK,NONS=0,[1,2,3],4,5,6

def sq(a,b): return ((a-b)**2).sum(-1)

res={"n":int(N),"layers":int(L),"pools":POOLS,"suites":SUITES}
P={k:np.zeros((L,NP,4)) for k in ["lambda","eta2","r2","erank","cka","d_swap","d_para","d_scene"]}
rho=np.zeros((L,4,2))

for si,sname in enumerate(SUITES):
    m=np.where(suite==si)[0]
    for p in range(NP):
        Zp=np.asarray(Z[m,:,:,p,:],dtype=np.float32)   # (n,7,L,D)
        for l in range(L):
            z=Zp[:,:,l,:]                               # (n,7,D)
            o=z[:,ORIG]; sw=z[:,SWAP]; pa=z[:,PARA].mean(1)
            dsw=sq(o,sw).mean(); dpa=sq(o[:,None,:],z[:,PARA]).mean()
            # scene distance: shuffle within suite, same instruction
            r=np.random.RandomState(0).permutation(len(m)); dsc=sq(o,o[r]).mean()
            P["d_swap"][l,p,si]=dsw; P["d_para"][l,p,si]=dpa; P["d_scene"][l,p,si]=dsc
            P["lambda"][l,p,si]=dsw/max(dpa,1e-12)
            # eta2: variant main effect / total
            zz=z[:,[ORIG]+PARA+[SWAP]]          # 5 well-defined levels (blank has no instr tokens)
            g=zz.mean((0,1)); vm=zz.mean(0)
            P["eta2"][l,p,si]=(len(m)*((vm-g)**2).sum())/max(((zz-g)**2).sum(),1e-12)
            # erank on orig features
            c=o-o.mean(0); s=np.linalg.svd(c,compute_uv=False); s=s/max(s.sum(),1e-12)
            P["erank"][l,p,si]=float(np.exp(-(s*np.log(s+1e-12)).sum()))
            # linear CKA to final layer
            y=Zp[:,ORIG,L-1,:]; x=o
            xc=x-x.mean(0); yc=y-y.mean(0)
            hsic=np.linalg.norm(xc.T@yc,"fro")**2
            P["cka"][l,p,si]=hsic/max(np.linalg.norm(xc.T@xc,"fro")*np.linalg.norm(yc.T@yc,"fro"),1e-12)
            # ridge R2, grouped by episode
            Y=act[m]; gk=GroupKFold(n_splits=5); pr=np.zeros_like(Y)
            for tr,te in gk.split(x,Y,groups=ep[m]):
                best,bl=-9e9,1.0
                for lam in [1e0,1e1,1e2,1e3,1e4]:
                    gi=GroupKFold(n_splits=3); sc=[]
                    for tr2,te2 in gi.split(x[tr],Y[tr],groups=ep[m][tr]):
                        r_=Ridge(alpha=lam).fit(x[tr][tr2],Y[tr][tr2])
                        sc.append(r2_score(Y[tr][te2],r_.predict(x[tr][te2])))
                    if np.mean(sc)>best: best,bl=np.mean(sc),lam
                pr[te]=Ridge(alpha=bl).fit(x[tr],Y[tr]).predict(x[te])
            P["r2"][l,p,si]=r2_score(Y,pr)
        print(sname,POOLS[p],"done",flush=True)
    for l in range(L):
        rho[l,si,0]=NRM[m,ORIG,l,1].mean()/max(NRM[m,ORIG,l,0].mean(),1e-9)   # text/vis
        rho[l,si,1]=NRM[m,ORIG,l,2].mean()/max(NRM[m,ORIG,l,0].mean(),1e-9)   # instr/vis
np.savez("probe_results.npz",rho=rho,**P)
json.dump(res,open("probe_meta.json","w"),indent=1)
print("SAVED")
