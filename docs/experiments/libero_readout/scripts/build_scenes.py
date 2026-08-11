import pickle, io, numpy as np, pandas as pd
from PIL import Image
N_PER_TASK=50
paths=pickle.load(open("paths.pkl","rb"))
cols=["observation.images.image","observation.state","action","frame_index","episode_index","task_index"]
rows=[]
for p in paths:
    d=pd.read_parquet(p,columns=cols); rows.append(d)
d=pd.concat(rows,ignore_index=True)
print("frames",len(d),"tasks",d.task_index.nunique(),flush=True)
rng=np.random.RandomState(0)
keep=[]
for t,g in d.groupby("task_index"):
    eps=g.episode_index.unique()
    per=max(1,N_PER_TASK//len(eps)+1)
    picks=[]
    for e in eps:
        ge=g[g.episode_index==e]
        # spread across the episode, skip first/last 10% (idle frames)
        n=len(ge); lo,hi=int(.1*n),int(.9*n)
        if hi<=lo: continue
        idx=np.linspace(lo,hi-1,min(per,hi-lo)).astype(int)
        picks.append(ge.iloc[idx])
    c=pd.concat(picks)
    c=c.sample(n=min(N_PER_TASK,len(c)),random_state=0)
    keep.append(c)
k=pd.concat(keep,ignore_index=True)
print("scenes",len(k),k.groupby("task_index").size().min(),k.groupby("task_index").size().max(),flush=True)
imgs=np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((256,256))) for b in k["observation.images.image"]])
np.savez_compressed("scenes.npz",
    imgs=imgs,
    state=np.stack(k["observation.state"].values).astype("float32"),
    action=np.stack(k["action"].values).astype("float32"),
    ti=k["task_index"].values.astype("int64"),
    ep=k["episode_index"].values.astype("int64"),
    fi=k["frame_index"].values.astype("int64"))
print("saved",imgs.shape)
