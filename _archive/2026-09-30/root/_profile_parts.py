"""Split the backward cost: which module owns it."""
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
ids=torch.randint(0,32768,(B,T),device=dev); tgt=torch.randint(0,32768,(B,T),device=dev)
def sync(): torch.cuda.synchronize()
def t(fn,n=8,warm=3):
    for _ in range(warm): fn()
    sync(); t0=time.time()
    for _ in range(n): fn()
    sync(); return (time.time()-t0)/n*1000

# 1. полный шаг
def full():
    out=m(ids,tgt); out.loss.backward()
print(f"full fwd+bwd            {t(full):8.1f} ms")

# 2. только embed+head (без блоков)
emb=m.embed if hasattr(m,'embed') else None
print("модули:", [n for n,_ in m.named_children()][:12])

# 3. по слоям: каждый блок отдельно
with torch.no_grad():
    pass
h=torch.randn(B,T,1152,device=dev,requires_grad=True)
for i,blk in enumerate(m.blocks):
    def f(blk=blk,h=h):
        y=blk(h,None,None); y.sum().backward()
    print(f"block[{i}] fwd+bwd      {t(f):8.1f} ms")

# 4. head (vocab projection) — suspects the 32k softmax
logits=torch.randn(B,T,32768,device=dev,requires_grad=True)
def head():
    l=logits.float(); l.sum().backward()
print(f"vocab head 32k fwd+bwd  {t(head):8.1f} ms")
# 5. fp32 softmax против bf16
def head_bf16():
    l=logits.to(torch.bfloat16); l.sum().backward()
print(f"vocab head bf16         {t(head_bf16):8.1f} ms")
