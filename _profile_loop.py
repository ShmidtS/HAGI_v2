"""Find the missing 7.9s: hook every submodule and time it."""
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
acc={}
def mk(name):
    def pre(mod,inp):
        torch.cuda.synchronize(); acc[name]=acc.get(name,0.0)+time.time()
    def post(mod,inp,out):
        torch.cuda.synchronize(); acc[name]=acc.get(name,0.0)-(time.time()-acc.pop(name) if False else 0)
    return pre
times={}
def pre_hook(name):
    def pre(mod,inp):
        torch.cuda.synchronize(); times[name]=time.time()
    return pre
def post_hook(name):
    def post(mod,inp,out):
        torch.cuda.synchronize(); d=time.time()-times[name]
        acc[name]=acc.get(name,0.0)+d
    return post
for name,mod in m.named_modules():
    if name and len(name.split('.'))<=2:
        mod.register_forward_pre_hook(pre_hook(name))
        mod.register_forward_hook(post_hook(name))
for _ in range(2):
    for p in m.parameters(): p.grad=None
    out=m(ids,tgt)
    (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
tot=sum(acc.values())
print(f"сумма submodule fwd: {tot*1000/2:.0f} ms за шаг")
for k,v in sorted(acc.items(), key=lambda kv:-kv[1])[:10]:
    print(f"  {k:24s} {v*1000/2:8.1f} ms")
