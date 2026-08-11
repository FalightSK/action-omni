import pickle, collections
from huggingface_hub import hf_hub_download
m=pickle.load(open("filemap.pkl","rb"))
need=collections.defaultdict(lambda:900)
tot=collections.Counter()
for c in m.values(): tot.update(c)
for t in tot: need[t]=900
sel=[]
# prefer files that are "pure-ish" for still-needed tasks, most useful first
while sum(need.values())>0:
    best,score=None,-1
    for p,c in m.items():
        if p in sel: continue
        s=sum(min(n,need[t]) for t,n in c.items())
        if s>score: best,score=p,s
    if score<=0: break
    sel.append(best)
    for t,n in m[best].items(): need[t]=max(0,need[t]-n)
print("files",len(sel),"deficit",sum(need.values()),flush=True)
paths=[]
for i,p in enumerate(sel):
    rel=p.split("HuggingFaceVLA/libero/")[1]
    paths.append(hf_hub_download("HuggingFaceVLA/libero",rel,repo_type="dataset"))
    if i%10==0: print(i,flush=True)
pickle.dump(paths,open("paths.pkl","wb")); print("DONE",len(paths))
