"""Same mixer, different call signature. Which one costs 872ms?"""
import sys, time, inspect
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
mx=m.blocks[0].mixer
print("signature mixer.forward:", inspect.signature(mx.forward))
print("signature mixer.mixer :", inspect.signature(mx.mixer.forward))
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
# variant A: как в прошлом тесте
def a():
    y=mx(h); y.float().sum().backward()
print("A fwd+bwd (float().sum()):", round(bench(a),1),"ms")
# variant B: без .float()
def b():
    y=mx(h); y.sum().backward()
print("B fwd+bwd (sum()):        ", round(bench(b),1),"ms")
# variant C: с loss как в реальном train
def c():
    y=mx(h); loss=y.float().pow(2).mean(); loss.backward()
print("C fwd+bwd (mean loss):    ", round(bench(c),1),"ms")
