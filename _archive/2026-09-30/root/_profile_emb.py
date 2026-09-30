"""Encoder embedding is 47% of params in fp32. What does it cost per step?"""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
from hagi.model.ternary import BitLinear
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
def step(grad_emb=True, cache=True):
    def f():
        for p in m.parameters(): p.grad=None
        if not grad_emb:
            m.encoder.embedding.weight.requires_grad_(False)
        out=m(ids,tgt)
        (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
        if not grad_emb:
            m.encoder.embedding.weight.requires_grad_(True)
    return f
print("эмбеддинг с grad  :", round(bench(step(True)),1),"ms")
print("эмбеддинг без grad :", round(bench(step(False)),1),"ms")
# и просто операция сбора градиента по эмбеддингу
w=m.encoder.embedding.weight
def g():
    w.grad=None
    out=m(ids,tgt); (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
print("то же, с градиентом:", round(bench(g),1),"ms")
