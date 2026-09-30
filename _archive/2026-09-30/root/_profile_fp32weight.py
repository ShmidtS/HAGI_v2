"""Is the fp32 weight the whole story?"""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
dev="cuda"
cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
m=build_model_for_config(cfg).to(dev)
tgt=torch.randint(0,32768,(8,1024),device=dev)
ids=torch.randint(0,32768,(8,1024),device=dev)
h=torch.randn(8,1024,1152,device=dev,dtype=torch.bfloat16,requires_grad=True)
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
def full():
    for p in m.parameters(): p.grad=None
    out=m(ids, tgt); (out.loss if hasattr(out,"loss") else out[0].float().pow(2).mean()).backward()
print("веса fp32 (сейчас)  :", round(bench(full),1),"ms")
# перевести ВСЕ веса в bf16
m=m.to(torch.bfloat16)
for p in m.parameters(): p.grad=None
print("веса bf16           :", round(bench(full),1),"ms")
# только блоки
m2=build_model_for_config(cfg).to(dev)
for p in m2.blocks.parameters(): p.requires_grad_(False)
def f2():
    y=m2.blocks[0](h,None,None); y.float().sum().backward()
print("блок, веса без grad :", round(bench(f2),1),"ms")
