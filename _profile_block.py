"""Inside one block: attention vs mixer vs norms, forward vs backward."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
B,T=8,1024
h=torch.randn(B,T,1152,device=dev,requires_grad=True)
blk=m.blocks[0]
def t(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000

with torch.no_grad():
    print(f"block fwd only          {t(lambda: blk(h,None,None)):8.1f} ms")
def fb():
    blk(h,None,None).sum().backward()
print(f"block fwd+bwd           {t(fb):8.1f} ms")

# attention отдельно
attn=blk.attn
def afb():
    y=attn(h,None,None); y.sum().backward()
print(f"  attn fwd+bwd          {t(afb):8.1f} ms")
# mixer отдельно
mx=blk.mixer
def mfb():
    y=mx(h); y.sum().backward()
print(f"  mixer fwd+bwd         {t(mfb):8.1f} ms")
# norm отдельно
with torch.no_grad():
    print(f"  attn_norm fwd         {t(lambda: attn.attn_norm(h)):8.1f} ms")
# fp32 против bf16 у блока
h2=h.detach().to(torch.bfloat16).requires_grad_(True)
try:
    def bfb16():
        y=blk(h2,None,None); y.sum().backward()
    print(f"block bf16 fwd+bwd      {t(bfb16):8.1f} ms")
except Exception as e:
    print("bf16:", type(e).__name__, str(e)[:70])
