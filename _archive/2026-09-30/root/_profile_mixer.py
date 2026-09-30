"""Inside the mixer: which of its parts owns 872ms?"""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
mx=m.blocks[0].mixer
print("mixer children:", [n for n,_ in mx.named_children()])
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
for name,mod in mx.named_children():
    def fb(mod=mod):
        y=mod(h); y.float().sum().backward()
    try: print(f"  {name:16s} fwd+bwd {bench(fb):8.1f} ms")
    except Exception as e: print(f"  {name:16s} err {type(e).__name__} {str(e)[:50]}")
# вручную fwd+bwd целиком
def whole():
    y=mx(h); y.float().sum().backward()
print(f"  {'WHOLE':16s} fwd+bwd {bench(whole):8.1f} ms")
with torch.no_grad():
    print(f"  {'WHOLE':16s} fwd     {bench(lambda: mx(h)):8.1f} ms")
