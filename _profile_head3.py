"""Cost of full-vocab CE vs sampled, at training shape."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev); hd=m.head
N,H=8*1024,1152
hid=torch.randn(N,H,device=dev,dtype=torch.bfloat16,requires_grad=True)
tgt=torch.randint(0,32768,(N,),device=dev)
def bench(fn,n=5,warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
def fbe():
    hd.weight.grad=None
    hd.exact_loss(hid,tgt).backward()
print(f"exact_loss fwd+bwd      {bench(fbe):8.1f} ms")
with torch.no_grad():
    print(f"exact_loss fwd          {bench(lambda: hd.exact_loss(hid.detach(),tgt)):8.1f} ms")
    # сколько стоит сам matmul в bf16
    w=hd.weight.to(torch.bfloat16)
    h=hid.detach()
    print(f"matmul bf16 fwd         {bench(lambda: h@w.T):8.1f} ms")
    wf=hd.weight
    print(f"matmul fp32 fwd         {bench(lambda: h.float()@wf.T):8.1f} ms")
    lg=h@w.T
    lgt=tgt
    def ce():
        return torch.nn.functional.cross_entropy(lg.float(), lgt)
    print(f"CE на готовых logits    {bench(ce):8.1f} ms")
