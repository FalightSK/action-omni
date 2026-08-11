"""Cache per-layer VLM token features for LIBERO-Goal (tasks 10-19).
train: original instruction only. val: 5 instruction variants (for the divergence ratio)."""
import torch, numpy as np, pickle, io, time, sys
from concurrent.futures import ThreadPoolExecutor
POOL=ThreadPoolExecutor(16)
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import variants as V
TAPS=list(range(0,33,3))            # 11 strided taps + final
if 32 not in TAPS: TAPS.append(32)
H=8                                  # action chunk
N_TRAIN,N_VAL=8000,1000
M="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
INSTR0=pickle.load(open("tasks.pkl","rb"))
paths=pickle.load(open("paths.pkl","rb"))
cols=["observation.images.image","observation.state","action","frame_index","episode_index","task_index"]
d=pd.concat([pd.read_parquet(p,columns=cols) for p in paths],ignore_index=True)
d=d[(d.task_index>=10)&(d.task_index<20)].sort_values(["episode_index","frame_index"]).reset_index(drop=True)
print("goal frames",len(d),"eps",d.episode_index.nunique(),flush=True)
# action chunks within episode
A=np.stack(d["action"].values).astype("float32"); ep=d.episode_index.values
idx=[]
for i in range(len(d)):
    if i+H<len(d) and ep[i+H]==ep[i]: idx.append(i)
idx=np.array(idx)
rng=np.random.RandomState(0)
eps=rng.permutation(d.episode_index.unique())
val_ep=set(eps[:max(1,len(eps)//6)])                # episode-level split, no leakage
is_val=np.array([ep[i] in val_ep for i in idx])
tr=rng.permutation(idx[~is_val])[:N_TRAIN]; va=rng.permutation(idx[is_val])[:N_VAL]
print("train",len(tr),"val",len(va),flush=True)
chunks=np.stack([A[i:i+H] for i in idx]); cmap={v:k for k,v in enumerate(idx)}
proc=AutoProcessor.from_pretrained(M); proc.image_processor.do_image_splitting=False
proc.tokenizer.padding_side="right"
model=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16).cuda().eval()
IMG=model.config.image_token_id
def run(rows,variants,out):
    Z=np.lib.format.open_memmap(out,mode="w+",dtype=np.float16,shape=(len(rows),len(variants),len(TAPS),81,960))
    BS=96; t0=time.time()
    for vi,v in enumerate(variants):
        for s in range(0,len(rows),BS):
            r=rows[s:s+BS]
            _dec=lambda i: Image.open(io.BytesIO(d["observation.images.image"].iloc[i]["bytes"])).convert("RGB")
            pil=list(POOL.map(_dec, r))
            ts=d["task_index"].iloc[r].values
            def ins(t):
                if v==0: return INSTR0[int(t)]
                if v<4: return V.PARA[int(t)][v-1]
                if v==4: return INSTR0[V.swap_partner(int(t))]
                if v==5: return V.BLANK
                return V.NONSENSE
            txt=[proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":ins(t)}]}],add_generation_prompt=True) for t in ts]
            inp=proc(text=txt,images=[[p] for p in pil],return_tensors="pt",padding=True).to("cuda")
            with torch.no_grad(): o=model(**inp,output_hidden_states=True)
            T=inp["input_ids"].shape[1]
            _pad=lambda l: (o.hidden_states[l] if T>=81 else torch.nn.functional.pad(o.hidden_states[l],(0,0,0,81-T)))[:,:81]
            hs=torch.stack([_pad(l) for l in TAPS],1)
            Z[s:s+len(r),vi]=hs.to(torch.float16).cpu().numpy()
            if (s//BS)%20==0: print(out,"v",v,"row",s,"/",len(rows),f"{time.time()-t0:.0f}s",flush=True)
    Z.flush(); return Z
run(tr,[0],"train_feats.npy")
np.save("train_act.npy",chunks[[cmap[i] for i in tr]]); np.save("train_state.npy",np.stack(d["observation.state"].iloc[tr].values).astype("float32"))
run(va,[0,1,2,3,4],"val_feats.npy")
np.save("val_act.npy",chunks[[cmap[i] for i in va]]); np.save("val_state.npy",np.stack(d["observation.state"].iloc[va].values).astype("float32"))
np.save("taps.npy",np.array(TAPS)); np.save("val_task.npy",d["task_index"].iloc[va].values)
print("DONE taps",TAPS)
