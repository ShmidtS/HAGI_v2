"""GLM-5 STE: train one layer ternary, verify improvement vs skel8.
One expert at a time, 200 steps AdamW, MSE vs fp16 teacher on cascade acts.
"""
import os,sys,io,time,json,glob,gc
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch,torch.nn.functional as F
from torch import nn
import numpy as np

dev='cuda' if torch.cuda.is_available() else 'cpu'
D,I,EA=4096,2048,288
HF='//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm=json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
ST,LR,NU=200,1e-4,1500

class Trn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w,e=1e-8):
        s=w.abs().mean(1,keepdim=True).clamp_min(e);return(w/s).clamp(-1.,1.).round()*s
    @staticmethod
    def backward(ctx,g):return g,None

# Preload all 288 experts for one layer into CPU RAM
from safetensors import safe_open
def preload(il):
    wg={};wu={};wd={}
    for e in range(EA):
        for p,o,i,n in [('gate_proj',I,D,'g'),('up_proj',I,D,'u'),('down_proj',D,I,'d')]:
            k=f'model.language_model.layers.{il}.mlp.experts.{e}.{p}.weight'
            with safe_open(f'{HF}/{_wm[k]}','pt','cpu') as sf:
                w=sf.get_tensor(k).float();s=sf.get_tensor(k.replace('.weight','.weight_scale_inv')).float()
            W=(w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o,i)
            if n=='g':wg[e]=W; continue
            if n=='u':wu[e]=W; continue
            wd[e]=W
    return wg,wu,wd

def pk(W):
    o,i=W.shape;nb=i//256;out=np.empty((o,nb*54),dtype=np.uint8)
    for r in range(o):
        b=W[r].reshape(nb,256);s=np.abs(b).max(1,keepdims=True).clip(1e-8,None)
        x=np.clip(np.round(b/s).astype(np.int64)+1,0,2).astype(np.uint32)
        qs=np.zeros((nb,48),dtype=np.uint32);qh=np.zeros((nb,4),dtype=np.uint32)
        for n,c in enumerate([81,27,9,3,1]):
            qs[:,:32]+=x[:,n*32:(n+1)*32]*np.uint32(c);qs[:,32:]+=x[:,160+n*16:160+(n+1)*16]*np.uint32(c)
        for m,c in enumerate([81,27,9,3]):qh[:]+=x[:,240+m*4:244+m*4]*np.uint32(c)
        pb=np.empty((nb,54),dtype=np.uint8)
        pb[:,:48]=((qs.astype(np.uint64)*256+242)//243).astype(np.uint8)
        pb[:,48:52]=((qh.astype(np.uint64)*256+242)//243).astype(np.uint8)
        pb[:,52:54]=s.astype(np.float16).view(np.uint8).reshape(nb,2)
        out[r]=pb.reshape(-1)
    return out

il=int(sys.argv[1]) if len(sys.argv)>1 else 3
print(f"LAYER L{il}",flush=True)

# Load cascade input
m=[]
for f in sorted(glob.glob(f'glm5_gguf/dump_moe_in_skel8/moe_in_L{il}_*.f32')):
    a=np.fromfile(f,dtype=np.float32);n=len(a)//D
    if n:m.append(a[:n*D].reshape(n,D))
z=np.concatenate(m)[:NU].astype(np.float32)
zt=torch.from_numpy(z[:1200]).to(dev);zv=torch.from_numpy(z[1200:]).to(dev)
print(f"  {z.shape[0]} tok",flush=True)

# Preload all experts
t0=time.time()
wg_all,wu_all,wd_all=preload(il)
print(f"  preloaded {EA} experts {time.time()-t0:.0f}s",flush=True)

# Output buffers
gb=np.empty((EA,I,864),dtype=np.uint8)
ub=np.empty((EA,I,864),dtype=np.uint8)
db=np.empty((EA,D,432),dtype=np.uint8)
ratios=[]

for e in range(EA):
    te=time.time()
    Wg=nn.Parameter(wg_all[e].to(dev))
    Wu=nn.Parameter(wu_all[e].to(dev))
    Wd=nn.Parameter(wd_all[e].to(dev))
    # Teacher
    with torch.no_grad():
        tg=F.silu(zt@Wg.T);tt=(tg*(zt@Wu.T))@Wd.T
        tg=F.silu(zv@Wg.T);tv=(tg*(zv@Wu.T))@Wd.T
    # Fast-skip if teacher ~0
    if tt.abs().mean().item()<1e-6:
        print(f"  e{e:03d} SKIP (zero teacher)")
        # Use skel8 baseline
        from gguf import GGUFReader
        rr=GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
        by={t.name:t for t in rr.tensors}
        for tn,buf in [(f'blk.{il}.ffn_gate_exps.weight',gb),(f'blk.{il}.ffn_up_exps.weight',ub),(f'blk.{il}.ffn_down_exps.weight',db)]:
            buf[e]=np.array(by[tn].data)[e]
        continue
    # Train
    opt=torch.optim.AdamW([Wg,Wu,Wd],lr=LR)
    for st in range(ST):
        opt.zero_grad()
        g=F.silu(zt@Trn.apply(Wg).T)
        loss=F.mse_loss((g*(zt@Trn.apply(Wu).T))@Trn.apply(Wd).T,tt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Wg,Wu,Wd],1.0)
        opt.step()
    # Measure
    with torch.no_grad():
        g=F.silu(zv@Trn.apply(Wg).T)
        so=(g*(zv@Trn.apply(Wu).T))@Trn.apply(Wd).T
        vm=F.mse_loss(so,tv).item()
        bv=F.mse_loss(torch.zeros_like(tv),tv).item()
        r=vm/bv*100 if bv>1e-10 else 0
    ratios.append(r)
    print(f"  e{e:03d} val_r={r:.1f}% {time.time()-te:.0f}s")
    # Pack
    with torch.no_grad():
        gb[e]=pk(Trn.apply(Wg).cpu().numpy().astype(np.float32))
        ub[e]=pk(Trn.apply(Wu).cpu().numpy().astype(np.float32))
        db[e]=pk(Trn.apply(Wd).cpu().numpy().astype(np.float32))
    del Wg,Wu,Wd,tt,tv;gc.collect();torch.cuda.empty_cache()

# Save
np.save(f'glm5_pod/tac/L{il}_g.npy',gb)
np.save(f'glm5_pod/tac/L{il}_u.npy',ub)
np.save(f'glm5_pod/tac/L{il}_d.npy',db)
valid=[r for r in ratios if 0<r<500]
print(f"\nL{il} DONE: mean_val_r={np.mean(valid):.1f}% "
      f"min={min(valid):.1f}% max={max(valid):.1f}% "
      f"n_skip={EA-len(ratios)} n_valid={len(valid)}",flush=True)