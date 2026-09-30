"""The head never appeared in hooks. Time the loss and its backward."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
print("head type:", type(m.head).__name__)
B,T,V=8,1024,32768
h=torch.randn(B,T,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def bench(fn,n=5,warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
with torch.no_grad():
    print(f"head fwd                 {bench(lambda: m.head(h)):8.1f} ms")
lg=m.head(h)
print("logits shape", tuple(lg.shape) if hasattr(lg,'shape') else type(lg), "dtype", getattr(lg,'dtype','?'))
tgt=torch.randint(0,V,(B,T),device=dev)
def fb():
    m.head.weight.grad=None
    o=m.head(h)
    l=torch.nn.functional.cross_entropy(o.float().reshape(-1,V), tgt.reshape(-1))
    l.backward()
print(f"head fwd+CE+backward      {bench(fb):8.1f} ms")
