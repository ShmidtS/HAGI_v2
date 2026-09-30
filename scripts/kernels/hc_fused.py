"""Fused mHC helpers: cut ~45 tiny torch launches per hc site to 2 kernels.

k_hc_proj: one program per output j (24 = pre/post/comb logits):
  w[j] = sum_d rmsnorm(x)[d] * fn[j, d];  rms computed by program 0 and
  shared via a tiny scratch (single-wave sync via atomic counter is
  overkill: instead EVERY program computes the 1-pass rms itself - D=20480
  loads x24 programs = 0.5MB reads, trivial).

k_hc_sinkhorn: softmax + eps + alternating row/col normalization, ITERS
iterations, on a [4,4] matrix - one program, everything in registers.

Together: hc forward = proj (1 launch) + sinkhorn (1 launch) + 2 tiny
torch ops for pre/post heads + stream collapse. ~45 launches -> ~6.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_hc_rms(x_ptr, rms_ptr, HD: tl.constexpr, BD: tl.constexpr, EPS: tl.constexpr):
    ssq = 0.0
    for d0 in range(0, HD, BD):
        xv = tl.load(x_ptr + d0 + tl.arange(0, BD)).to(tl.float32)
        ssq += tl.sum(xv * xv, axis=0)
    tl.store(rms_ptr, tl.rsqrt(ssq / HD + EPS))


@triton.jit
def k_hc_proj(x_ptr, rms_ptr, fn_ptr, w_ptr,
              HD: tl.constexpr, MIX: tl.constexpr, BD: tl.constexpr):
    j = tl.program_id(0)
    rms = tl.load(rms_ptr)
    acc = 0.0
    for d0 in range(0, HD, BD):
        xv = (tl.load(x_ptr + d0 + tl.arange(0, BD)).to(tl.float32)) * rms
        fv = tl.load(fn_ptr + j * HD + d0 + tl.arange(0, BD))
        acc += tl.sum(xv * fv, axis=0)
    tl.store(w_ptr + j, acc)


@triton.jit
def k_hc_sinkhorn(w_ptr, base_ptr, scale_ptr, comb_ptr,
                  H: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr):
    # w layout: [pre(4), post(4), comb(16)] for ONE token
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    j = tl.arange(0, H)[:, None] * H + tl.arange(0, H)[None, :]
    # pre & post heads (scalars per index) - store sigmoid results
    pj = tl.arange(0, H)
    pre_w = tl.load(w_ptr + pj)
    pre_b = tl.load(base_ptr + pj)
    pre = tl.sigmoid(pre_w * s0 + pre_b) + EPS
    tl.store(comb_ptr + 16 + pj, pre)          # temp area after comb (16+4)
    post_w = tl.load(w_ptr + H + pj)
    post_b = tl.load(base_ptr + H + pj)
    post = 2.0 * tl.sigmoid(post_w * s1 + post_b)
    tl.store(comb_ptr + 16 + H + pj, post)     # 20..23
    # comb
    cb = tl.load(base_ptr + 2 * H + j)
    cl = tl.load(w_ptr + 2 * H + j) * s2 + cb
    mx = tl.max(cl, axis=1)[:, None]
    e = tl.exp(cl - mx)
    comb = e / tl.sum(e, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + j, comb)


class HCFused:
    """Reusable scratch + launches for one hc site at decode shape."""

    def __init__(self, hc_module, D=4096):
        self.fn = hc_module.fn.data.cuda().float().contiguous()
        self.base = hc_module.base.data.cuda().float().contiguous()
        self.scale = hc_module.scale.data.cuda().float().contiguous()
        self.H = int(hc_module.hc_mult)
        self.iters = int(hc_module.hc_sinkhorn_iters)
        self.eps = float(hc_module.hc_eps)
        self.epsn = float(hc_module.input_norm.variance_epsilon) \
            if hasattr(hc_module.input_norm, "variance_epsilon") else 1e-5
        self.HD = self.H * D
        self.D = D
        self.rms = torch.zeros(1, device="cuda", dtype=torch.float32)
        self.w = torch.zeros(2 * self.H + self.H * self.H, device="cuda", dtype=torch.float32)
        self.out = torch.zeros(16 + 2 * self.H, device="cuda", dtype=torch.float32)

    def forward(self, hidden_streams):
        """hidden_streams [1,1,H,D] -> (post[1,1,H], comb[1,1,H,H],
        collapsed[1,1,D]) - same order as the reference forward."""
        x = hidden_streams.reshape(1, -1)          # flatten(2); S==1 decode
        xf = x.float().contiguous()
        k_hc_rms[(1,)](xf, self.rms, HD=self.HD, BD=1024, EPS=self.epsn)
        MIXN = 2 * self.H + self.H * self.H
        k_hc_proj[(MIXN,)](xf, self.rms, self.fn, self.w, HD=self.HD,
                           MIX=MIXN, BD=1024)
        k_hc_sinkhorn[(1,)](self.w, self.base, self.scale, self.out,
                            H=self.H, ITERS=self.iters, EPS=self.eps)
        H = self.H
        pre = self.out[16:16 + H].reshape(1, 1, H, 1)
        post = self.out[16 + H:16 + 2 * H].reshape(1, 1, H)
        comb = self.out[:H * H].reshape(1, 1, H, H)
        collapsed = (pre * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
