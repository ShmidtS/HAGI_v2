"""Where does a training step actually go? Measure, do not guess."""
import sys, time
sys.path.insert(0,'src')
import torch
from hagi.config import Config
from hagi.model.factory import build_model_for_config

dev = "cuda"
cfg = Config()
cfg.model.vocab_size = 32768          # the real budget, not the 262k default
cfg.model.hidden_size = 1152
cfg.model.num_layers = 3
cfg.model.attention.num_query_heads = 18
cfg.model.attention.num_kv_heads = 6
m = build_model_for_config(cfg).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
B, T, H = 8, 1024, cfg.model.hidden_size
ids = torch.randint(0, 32768, (B, T), device=dev)
tgt = torch.randint(0, 32768, (B, T), device=dev)

def timeit(fn, n=12, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time()-t0)/n*1000

def fwd():
    return m(ids, tgt)
def fwd_bwd():
    out = m(ids, tgt)
    loss = out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()
    loss.backward()
    return loss
def fwd_bwd_step():
    out = m(ids, tgt)
    loss = out.loss if hasattr(out,'loss') else out[0].float().pow(2).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

print(f"модель: {m.param_summary()['total']/1e6:.1f}M, H={H}, layers={cfg.model.num_layers}")
print(f"batch={B} seq={T} => {B*T} токенов/шаг")
print(f"forward          {timeit(fwd):8.2f} ms")
print(f"forward+backward {timeit(fwd_bwd):8.2f} ms")
full = timeit(fwd_bwd_step)
print(f"полный шаг       {full:8.2f} ms  -> {1000/full:.1f} шаг/с, {B*T*1000/full/1e6:.2f} Mток/с")
