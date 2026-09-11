"""Full-stack decode projection: all 43 layers, banks resident, triton path.
Simulates per-token MoE cost of the whole model with real bank data."""
import sys
import time

import torch

sys.path.insert(0, "scripts")
from ternary_bank_kernels import k_h13t, trit_lut
import triton_bank_kernels as tbk
from safetensors.torch import load_file

D, I, GS, NL = 4096, 2048, 128, 43


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t = load_file(f"dsv4_release/layers/layer{layer}.safetensors", device="cuda")
    t["s2h"] = t["s2"].half()
    lut = trit_lut()
    torch.manual_seed(0)
    ids = torch.randperm(256, device="cuda")[:6].to(torch.int32)
    wts = torch.rand(6, device="cuda")
    z = (torch.randn(D, device="cuda") * 3).contiguous()
    h = torch.empty(6, I, device="cuda")
    y = torch.empty(6, D, device="cuda")
    out = torch.zeros(D, device="cuda")

    def step():
        for li in range(NL):
            out.zero_()
            k_h13t[(6, I // 64)](z, ids, t["w13t"], t["s13"], t["b13"], lut, h,
                                 D=D, I=I, GS=GS, BI=64, num_warps=2)
            tbk.k_yb[(6, D // 512)](h, ids, wts, t["w2a"], t["s2h"], y, out,
                                    D=D, I=I, GS=GS, BD=512, BK=256, num_warps=8)
        return out

    step(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"43-layer MoE token step: {dt*1e3:.1f} ms -> {1/dt:.1f} tok/s (MoE-bound ceiling)")
    print("(+ attention/embeddings overhead on top; target >=5 tok/s e2e)")


if __name__ == "__main__":
    main()
