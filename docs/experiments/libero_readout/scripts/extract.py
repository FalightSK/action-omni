import torch, numpy as np, pickle, sys, time
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import variants as V

M="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 0
BS=int(sys.argv[2]) if len(sys.argv)>2 else 40
d=np.load("scenes.npz"); imgs=d["imgs"]; ti=d["ti"]
if LIMIT: imgs=imgs[:LIMIT]; ti=ti[:LIMIT]
N=len(imgs)
proc=AutoProcessor.from_pretrained(M); proc.image_processor.do_image_splitting=False
proc.tokenizer.padding_side="right"
model=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16).cuda().eval()
L=model.config.text_config.num_hidden_layers+1; D=model.config.text_config.hidden_size
IMG_ID=model.config.image_token_id
tok=proc.tokenizer
EOU=tok.convert_tokens_to_ids("<end_of_utterance>"); FAKE=tok.convert_tokens_to_ids("<fake_token_around_image>")
VNAMES=["orig","para1","para2","para3","swap","blank","nonsense"]
def instr_for(t,v):
    if v==0: return V.PARA and list(V.PARA.keys()) and INSTR0[t]
    if v in (1,2,3): return V.PARA[t][v-1]
    if v==4: return INSTR0[V.swap_partner(t)]
    if v==5: return V.BLANK
    return V.NONSENSE
INSTR0=pickle.load(open("tasks.pkl","rb"))  # task_index -> original instruction
POOLS=["all","vis","text","instr"]
Z=np.lib.format.open_memmap("feats.npy",mode="w+",dtype=np.float16,shape=(N,7,L,4,D))
NRM=np.zeros((N,7,L,3),dtype=np.float32)   # vis, text, instr mean ||h||
t0=time.time()
for v in range(7):
    for s in range(0,N,BS):
        sl=slice(s,min(s+BS,N))
        pil=[Image.fromarray(x) for x in imgs[sl]]
        txt=[proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":instr_for(int(t),v)}]}],add_generation_prompt=True) for t in ti[sl]]
        inp=proc(text=txt,images=[[p] for p in pil],return_tensors="pt",padding=True).to("cuda")
        with torch.no_grad(): out=model(**inp,output_hidden_states=True)
        ids=inp["input_ids"]; am=inp["attention_mask"].bool()
        vis=(ids==IMG_ID)
        txtm=am&(~vis)
        # instruction span: between LAST fake-token and end_of_utterance
        ins=torch.zeros_like(vis)
        for b in range(ids.shape[0]):
            r=ids[b].tolist()
            try:
                a=len(r)-1-r[::-1].index(FAKE); e=r.index(EOU)
                if e>a+1: ins[b,a+1:e]=True
            except ValueError: pass
        for li,h in enumerate(out.hidden_states):
            h=h.float()
            for pi,m in enumerate([am,vis,txtm,ins]):
                c=m.sum(1,keepdim=True).clamp(min=1)
                Z[sl,v,li,pi]=((h*m.unsqueeze(-1)).sum(1)/c).half().cpu().numpy()
            n=h.norm(dim=-1)
            for ni,m in enumerate([vis,txtm,ins]):
                NRM[sl,v,li,ni]=((n*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy()
        if s==0: print(f"v={v} T={ids.shape[1]} vis={vis[0].sum().item()} txt={txtm[0].sum().item()} ins={ins[0].sum().item()} elapsed={time.time()-t0:.1f}s",flush=True)
    print(f"variant {VNAMES[v]} done {time.time()-t0:.1f}s",flush=True)
Z.flush(); np.save("norms.npy",NRM)
np.savez("meta.npz",ti=ti,pools=np.array(POOLS),variants=np.array(VNAMES))
print("DONE",Z.shape,time.time()-t0)
