"""Time the head and its loss through the real API."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
hd=m.head
print("chunk_rows", getattr(hd,'chunk_rows','?'), "| V", hd.vocab_size, "| weight", hd.weight.dtype)
def bench(fn,n=5,warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
N,H=8*1024,1152
hid=torch.randn(N,H,device=dev,dtype=torch.bfloat16,requires_grad=True)
tgt=torch.randint(0,32768,(N,),device=dev)
def fb():
    hd.weight.grad=None
    l=hd.exact_loss(hid,tgt)
    l.backward()
print(f"exact_loss fwd+bwd      {bench(fb):8.1f} ms")
with torch.no_grad():
    print(f"exact_loss fwd          {bench(lambda: hd.exact_loss(hid.detach(),tgt)):8.1f} ms")
    lg=hid.detach()@hd.weight.to(torch.bfloat16).T
    print(f"наивный matmul fwd      {bench(lambda: hid.detach()@hd.weight.to(torch.bfloat16).T):8.1f} ms  logits {tuple(lg.shape)}")
