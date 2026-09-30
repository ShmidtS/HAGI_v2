"""What dtype does each block submodule actually run in?"""
import sys
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
blk=m.blocks[0]
seen=[]
def hook(name):
    def f(mod, inp, out):
        d = out[0] if isinstance(out,(tuple,list)) else out
        dt = getattr(d,'dtype',type(d).__name__)
        seen.append((name, str(dt)))
    return f
for n,mod in [('attn_norm',blk.attn.attn_norm),('attn',blk.attn),
              ('mixer_norm',blk.mixer.norm),('mixer',blk.mixer)]:
    mod.register_forward_hook(hook(n))
with torch.no_grad():
    blk(h,None,None)
for n,dt in seen: print(f"{n:12s} out={dt}")
print("вход:", h.dtype)
# BitLinear внутри mixer — какие у него веса
for n,p in blk.mixer.named_parameters():
    print(f"  {n:32s} {p.dtype} {tuple(p.shape)}")
    break
