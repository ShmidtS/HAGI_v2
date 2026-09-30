"""After the bf16-master fix, where is the step time?"""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
ids=torch.randint(0,32768,(8,1024),device=dev); tgt=torch.randint(0,32768,(8,1024),device=dev)
def bench(fn,n=6,warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
def fwd():
    with torch.no_grad(): m(ids,tgt)
print(f"forward only        {bench(fwd):8.1f} ms")
def fb():
    for p in m.parameters(): p.grad=None
    out=m(ids,tgt)
    (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
print(f"fwd+bwd             {bench(fb):8.1f} ms")
# только энкодер
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def enc():
    y=m.encoder(ids); y.float().sum().backward()
print(f"encoder fwd+bwd     {bench(enc):8.1f} ms")
# блоки по одному, с отключенными grad весов
for i,b in enumerate(m.blocks):
    for p in b.parameters(): p.requires_grad_(False)
    def fb2(b=b):
        y=b(h,None,None); y.float().sum().backward()
    print(f"block[{i}] без grad  {bench(fb2):8.1f} ms")
    for p in b.parameters(): p.requires_grad_(True)
def blk_grad():
    for p in m.blocks[0].parameters(): p.grad=None
    y=m.blocks[0](h,None,None); y.float().sum().backward()
print(f"block[0] с grad    {bench(blk_grad):8.1f} ms")
