from huggingface_hub import hf_hub_download
import pandas as pd, pickle
R="HuggingFaceVLA/libero"
sel=pickle.load(open("sel_files.pkl","rb"))
e=pd.read_parquet(hf_hub_download(R,"meta/episodes/chunk-000/file-000.parquet",repo_type="dataset"),
                  columns=["data/chunk_index","data/file_index"]).drop_duplicates()
c={int(r["data/file_index"]):int(r["data/chunk_index"]) for _,r in e.iterrows()}
paths=[]
for i,f in enumerate(sel):
    paths.append(hf_hub_download(R,f"data/chunk-{c[f]:03d}/file-{f:03d}.parquet",repo_type="dataset"))
    print(i+1,"/",len(sel),flush=True)
pickle.dump(paths,open("paths.pkl","wb")); print("DONE",len(paths))
