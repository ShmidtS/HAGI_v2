"""Does the cost come from grads flowing into ternary weights?"""
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
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
# A: как сейчас — все параметры требуют grad
def a():
    for p in m.parameters(): p.grad=None
    y=mx(h); y.float().sum().backward()
print("все параметры с grad :", round(bench(a),1),"ms")
# B: только вход
for p in m.parameters(): p.requires_grad_(False)
def b():
    y=mx(h); y.float().sum().backward()
print("только вход          :", round(bench(b),1),"ms")
# C: без grad вообще
h2=h.detach()
with torch.no_grad():
    print("только forward       :", round(bench(lambda: mx(h2)),1),"ms")
# D: сколько параметров у миксера и их dtype
for p_ in m.parameters(): p_.requires_grad_(True)
ps=[(n,p_.dtype,tuple(p_.shape)) for n,p_ in mx.named_parameters()]
print("параметров миксера:", len(ps))
for n,d,s in ps[:4]: print("  ",n,d,s)
