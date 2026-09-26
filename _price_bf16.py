"""What does an fp32 master buy? Measure the quantization error directly."""
import sys
sys.path.insert(0,'src')
import torch
from hagi.model.ternary import ternarize
dev="cuda"
torch.manual_seed(0)
# тренировочный масштаб: веса инициализируются и живут в fp32
for shape in ((3072,1152),(1152,3072),(1152,1152)):
    w32=torch.randn(*shape,device=dev)*0.02
    w16=w32.to(torch.bfloat16)
    q32,_=ternarize(w32,1e-6)
    q16,_=ternarize(w16,1e-6)
    q16f=q16.to(torch.float32)
    # ошибка квантования каждого
    e32=(q32-w32).pow(2).mean().sqrt().item()
    e16=(q16f-w32.to(torch.float32)).pow(2).mean().sqrt().item()
    rel32=e32/w32.pow(2).mean().sqrt().item()
    rel16=e16/w32.pow(2).mean().sqrt().item()
    print(f"{str(shape):14s} rms err fp32={e32:.3e} bf16={e16:.3e}  rel fp32={rel32*100:.2f}% bf16={rel16*100:.2f}%  (хуже в {e16/max(e32,1e-12):.2f}x)")
