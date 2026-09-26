"""Trace the activation dtype through the whole forward."""
import sys
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
seen=[]
def hook(name):
    def post(mod,inp,out):
        d=out[0] if isinstance(out,(tuple,list)) else out
        seen.append((name,str(getattr(d,'dtype',type(d).__name__)),tuple(d.shape)))
    return post
for n,mod in [('encoder',m.encoder),('block0',m.blocks[0]),('block1',m.blocks[1]),
              ('block2',m.blocks[2]),('out_norm',m.out_norm)]:
    mod.register_forward_hook(hook(n))
ids=torch.randint(0,32768,(2,64),device=dev)
with torch.no_grad():
    out=m(ids,None)
for n,dt,sh in seen: print(f"{n:10s} {dt:16s} {sh}")
print("hidden out dtype:", out.hidden.dtype if hasattr(out,'hidden') else '?')
# а что реально внутри блока
h=torch.randn(2,64,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
inner=[]
for n,mod in [('b0.attn',m.blocks[0].attn),('b0.mixer',m.blocks[0].mixer),
              ('b0.attn_norm',m.blocks[0].attn.attn_norm)]:
    mod.register_forward_hook(hook(n))
with torch.no_grad(): m.blocks[0](h,None,None)
for n,dt,sh in seen[-3:]: print(f"{n:12s} {dt:16s} {sh}")
