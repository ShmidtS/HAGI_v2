"""Does the existing ternary step-cache recover the fp32-master cost?"""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
from hagi.model.ternary import cache_ternary_weights, clear_ternary_weights
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
ids=torch.randint(0,32768,(8,1024),device=dev); tgt=torch.randint(0,32768,(8,1024),device=dev)
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
def step(cache):
    def f():
        for p in m.parameters(): p.grad=None
        if cache: cache_ternary_weights(m)
        out=m(ids,tgt)
        (out.loss if hasattr(out,"loss") else out[0].float().pow(2).mean()).backward()
        if cache: clear_ternary_weights(m)
    return f
print("без step-cache :", round(bench(step(False)),1),"ms")
print("со step-cache :", round(bench(step(True)),1),"ms")
