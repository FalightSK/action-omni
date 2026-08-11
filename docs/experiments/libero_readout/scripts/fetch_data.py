from huggingface_hub import hf_hub_download
import pandas as pd, numpy as np, os, io, pickle
R="HuggingFaceVLA/libero"
e=pd.read_parquet(hf_hub_download(R,"meta/episodes/chunk-000/file-000.parquet",repo_type="dataset"),
                  columns=["episode_index","data/chunk_index","data/file_index","tasks","length"])
e["task"]=e["tasks"].str[0]
tk=pd.read_parquet(hf_hub_download(R,"meta/tasks.parquet",repo_type="dataset")).reset_index()
t2i=dict(zip(tk["index"],tk["task_index"])); i2t={v:k for k,v in t2i.items()}
e["ti"]=e["task"].map(t2i)
pickle.dump(i2t,open("tasks.pkl","wb"))
# pick minimal files covering all 40 tasks with >=4 episodes each
need={t:4 for t in range(40)}; files=[]
for f,g in e.groupby("data/file_index"):
    use=False
    for t in g["ti"].unique():
        if need[t]>0: use=True
    if use:
        for t in g["ti"]: need[t]=max(0,need[t]-1)
        files.append(int(f))
    if sum(need.values())==0: break
print("files",len(files),flush=True)
paths=[]
for f in files:
    c=int(e[e["data/file_index"]==f]["data/chunk_index"].iloc[0])
    paths.append(hf_hub_download(R,f"data/chunk-{c:03d}/file-{f:03d}.parquet",repo_type="dataset"))
    print("got",f,flush=True)
pickle.dump(paths,open("paths.pkl","wb"))
print("DONE",len(paths))
