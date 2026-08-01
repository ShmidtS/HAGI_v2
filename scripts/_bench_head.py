from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch

def bench(fn, name, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:55s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

H=1152; T=1024; V=32768
x = torch.randn(30, T, H, device="cuda", dtype=torch.bfloat16)
w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)

# fullVotes
def full(): torch.mm(x.view(-1,H), w.t())
bench(full, "full head bf16 [30*1024,1152]@[1152,32768]")

# sampled: top 4096 tokens by unigram + uniform
sel_idx = torch.randperm(V, device="cuda")[:4096]
def sampled():
    w_sel = w[sel_idx]                    # [4096, H] gather = reads 4096*1152*2 = 9.4MB
    torch.mm(x.view(-1,H), w_sel.t())
bench(sampled, "sampled head 4096/V")

# rope precompute est: rope cos/sin cached? check attention.py: buffers vs computed live
# torch.compile overhead test on CE
import torch.nn.functional as F
logits = torch.randn(30*T, V, device="cuda", dtype=torch.bfloat16)
tgt = torch.randint(0, V, (30*T,), device="cuda")
bench(lambda: F.cross_entropy(logits.float(), tgt), "CE fp32 (current with .float())")
bench(lambda: F.cross_entropy(logits, tgt), "CE bf16 (safe?)")
logit32 = logits.float()
bench(lambda: F.cross_entropy(logit32, tgt), "CE fp32 (pre-cast)")
bench(lambda: logits.float(), "cast logits->fp32 alone")

# test correctness: does bf16 logits break CE on rare tokens?
torch.manual_seed(0)
lg = torch.randn(1000, V, device="cuda", dtype=torch.bfloat16) * 5
tg = torch.randint(0, V, (1000,), device="cuda")
ce32 = F.cross_entropy(lg.float(), tg)
ce16 = F.cross_entropy(lg, tg)
print(f"\nce32={ce32.item():.5f} ce16={ce16.item():.5f} diff={abs(ce32-ce16).item():.5f}")
# max bias?
lg_b = lg.clone(); lg_b[:, 0] += 100  # outlier
print(f"outlier: ce32={F.cross_entropy(lg_b.float(), tg).item():.5f} ce16={F.cross_entropy(lg_b, tg).item():.5f}")
