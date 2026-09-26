"""One BitLinear: fp32 weight vs bf16 weight, same activations."""
import sys, time
sys.path.insert(0,'src')
import torch, torch.nn.functional as F
from hagi.model.ternary import BitLinear
dev="cuda"
def bench(fn,n=20,warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
B,T,I,O=8,1024,1152,3072
for wdt in (torch.float32, torch.bfloat16):
    lin=BitLinear(I,O,bias=False).to(dev)
    lin.weight.data=lin.weight.data.to(wdt)
    x=torch.randn(B,T,I,device=dev,dtype=wdt,requires_grad=True)
    def f():
        for p in lin.parameters(): p.grad=None
        y=lin(x); y.float().sum().backward()
    print(f"weight={str(wdt):16s} fwd+bwd {bench(f):8.2f} ms")
    with torch.no_grad():
        print(f"weight={str(wdt):16s} fwd     {bench(lambda: lin(x)):8.2f} ms")
