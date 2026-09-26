"""Is the full-vocab CE the real bottleneck? Compare against sampled."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config
from hagi.train.loop import cast_model
dev="cuda"
def bench(fn,n=8,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
def run(k, keep=1.0, chunk=4096):
    cfg=Config(); cfg.model.vocab_size=32768; cfg.model.hidden_size=1152
    cfg.model.num_layers=3; cfg.model.attention.num_query_heads=18; cfg.model.attention.num_kv_heads=6
    cfg.model.head.sampled_softmax_k=k; cfg.model.head.ce_chunk_rows=chunk
    cfg.train.ce_keep_rate=keep; cfg.train.ce_keep_mode="bernoulli"
    m=build_model_for_config(cfg).to(dev); cast_model(m,"bf16")
    ids=torch.randint(0,32768,(8,1024),device=dev); tgt=torch.randint(0,32768,(8,1024),device=dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    def step():
        for p in m.parameters(): p.grad=None
        out=m(ids,tgt)
        (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    ms=bench(step)
    return ms, 1000/ms
print("k=0 (полный CE)      :", " ".join(f"{v:7.1f}" if i==0 else f"{v:7.2f} шаг/с" for i,v in enumerate(run(0))))
for k in (2048, 4096):
    print(f"k={k} (sampled)      :", " ".join(f"{v:7.1f}" if i==0 else f"{v:7.2f} шаг/с" for i,v in enumerate(run(k))))
print("k=0, keep=0.25       :", " ".join(f"{v:7.1f}" if i==0 else f"{v:7.2f} шаг/с" for i,v in enumerate(run(0,0.25))))
print("k=0, chunk=8192      :", " ".join(f"{v:7.1f}" if i==0 else f"{v:7.2f} шаг/с" for i,v in enumerate(run(0,1.0,8192))))
