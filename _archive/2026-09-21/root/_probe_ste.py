"""Verify single-expert STE training with BitLinear + step cache."""
import sys, os, time
os.chdir(r'C:/HAGI_v2')
sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F
from torch import nn
import numpy as np
from src.hagi.model.ternary import BitLinear, cache_ternary_weights, clear_ternary_weights

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
from safetensors import safe_open
import json
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
D, I = 4096, 2048

def ld(p):
    k = f'model.language_model.layers.3.mlp.experts.0.{p}.weight'
    with safe_open(f'{HF}/{_wm[k]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(k).float()
        s = sf.get_tensor(k.replace('.weight', '.weight_scale_inv')).float()
    return (w.reshape(s.shape[0], 128, s.shape[1], 128) * s[:, None, :, None]).reshape(w.shape).numpy()

# gate/up: (2048, 4096) = (I, D), down: (4096, 2048) = (D, I)
print('gate', ld('gate_proj').shape, 'up', ld('up_proj').shape, 'down', ld('down_proj').shape)

dev = 'cuda'
# BitLinear(in, out): weight is [out, in]. gate: in=D, out=I -> (I, D)=(2048, 4096) ✓
gl = BitLinear(D, I); gl.weight.data.copy_(torch.from_numpy(ld('gate_proj')))
ul = BitLinear(D, I); ul.weight.data.copy_(torch.from_numpy(ld('up_proj')))
dl = BitLinear(I, D); dl.weight.data.copy_(torch.from_numpy(ld('down_proj')))
gl, ul, dl = gl.to(dev), ul.to(dev), dl.to(dev)

z = torch.randn(64, D, device=dev)
with torch.no_grad():
    cache_ternary_weights(nn.ModuleList([gl, ul, dl]))
    teacher = dl(F.silu(gl(z)) * ul(z))
    clear_ternary_weights(nn.ModuleList([gl, ul, dl]))
print(f'teacher: {tuple(teacher.shape)} norm={teacher.norm().item():.3f}')

opt = torch.optim.AdamW([gl.weight, ul.weight, dl.weight], lr=1e-4)
for st in range(5):
    t = time.time()
    opt.zero_grad(set_to_none=True)
    cache_ternary_weights(nn.ModuleList([gl, ul, dl]))
    s = dl(F.silu(gl(z)) * ul(z))
    loss = F.mse_loss(s, teacher)
    loss.backward()
    clear_ternary_weights(nn.ModuleList([gl, ul, dl]))
    gn = torch.cat([gl.weight.grad.flatten(), ul.weight.grad.flatten(), dl.weight.grad.flatten()]).norm()
    opt.step()
    print(f'  st{st}: loss={loss.item():.6e} gnorm={gn:.2e} dt={time.time()-t:.3f}s', flush=True)
print('OK')
