"""Post-fix: what does a real step cost now, and does quality hold?"""
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
dt=set(p.dtype for n,p in m.named_parameters() if any(n.startswith(pre+'.') for pre,mod in m.named_modules() if isinstance(mod,BitLinear)))
print("dtype BitLinear-весов:", dt)
print("dtype эмбеддинга   :", m.encoder.embedding.weight.dtype)
print("dtype норм         :", m.blocks[0].attn.attn_norm.weight.dtype)
ids=torch.randint(0,32768,(8,1024),device=dev); tgt=torch.randint(0,32768,(8,1024),device=dev)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
def step():
    for p in m.parameters(): p.grad=None
    out=m(ids,tgt)
    (out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()).backward()
    opt.step(); opt.zero_grad(set_to_none=True)
for _ in range(3): step()
torch.cuda.synchronize(); t0=time.time(); n=8
for _ in range(n): step()
torch.cuda.synchronize()
ms=(time.time()-t0)/n*1000
print(f"полный шаг: {ms:.1f} ms -> {1000/ms:.2f} шаг/с, {8*1024*1000/ms/1e6:.2f} Mток/с")
