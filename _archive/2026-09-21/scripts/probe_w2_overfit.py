"""Decisive test: is the W2 ridge solve overfitting thin rows?

For one hot expert at one layer:
  - collect rows on stream A (fit) and stream B (validation), both random
    tokens through the compressed prefix (fit-time conditions);
  - solve W2 ridge with several lambdas (base, x10, x100) and an anchored
    variant (shrink toward the CD-snapped W2 from ptq_closed_form);
  - report out-of-sample resid for each.

No checkpoints are written.
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401
import dsv4_experts as de
from dsv4_experts import (HASH_LAYERS, ROUTER_BIAS, ROUTER_TID, ROUTER_W,
                          TOP_K, ffn, load_router)
import dsv4_generate_ttt as gen
from dsv4_generate_ttt import get_dequant, get_int4x

L = int(os.environ.get("PL", "4"))
K = int(os.environ.get("PK", "8"))
N_TOK = int(os.environ.get("PN", "8192"))
COMP = [int(x) for x in os.environ.get("I4X_LAYERS", "").split(",") if x != ""]
CUR_IDS = None
ROWS = []  # x rows (cpu fp32)
HB = []  # (h, y) per stream


def main():
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(
        "C:/HAGI_v2/dsv4_shared_only", torch_dtype=torch.bfloat16)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    global CUR_IDS

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()
            import torch.nn.functional as F
            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CUR_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)

            flatb = flat.to(torch.bfloat16)
            sw = gen.get_shared_dequant(li)
            out = ffn(flatb, sw["w1"], sw["w3"], sw["w2"]).float() if False else ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()

            if li == L:
                m_any = (indices == K).any(dim=1)
                if m_any.any():
                    ROWS.append(flat[m_any].detach().cpu())
            if li in COMP:
                _z = None
                for k in indices.unique().tolist():
                    d = get_int4x(li, k)
                    m_any = (indices == k).any(dim=1)
                    if _z is None and d is not None:
                        _z = (flatb - d["mu"]) @ d["P"]
                    if d is None:
                        w1, w2, w3 = get_dequant(li, k)
                        out_k = ffn(flatb, w1, w2, w3).float()
                    else:
                        h = gen.int4x_forward_z(d, _z[m_any])
                        out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * out_k[pos[m]]
            else:
                for k in indices.unique().tolist():
                    w1, w2, w3 = get_dequant(li, k)
                    m_any = (indices == k).any(dim=1)
                    y_k = ffn(flatb, w1, w2, w3).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * y_k[pos[m]]
            return out.to(x.dtype).reshape(B, S, D)
        return hook

    for li in range(43):
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True)

    def run_stream(seed):
        global CUR_IDS, ROWS
        ROWS = []
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            ids = torch.randint(0, 129280, (1, N_TOK), generator=g)
            CH = 4096
            for c0 in range(0, ids.shape[1], CH):
                chunk = ids[:, c0:c0 + CH]
                CUR_IDS = chunk.reshape(-1)
                model(chunk.cuda())
        x = torch.cat(ROWS).float().cuda()
        d = get_int4x(L, K)
        h = gen.int4x_forward(d, x)
        w1, w2, w3 = get_dequant(L, K)
        y = ffn(x.to(torch.bfloat16), w1, w2, w3).float()
        return x, h.float(), y

    print(f"stream A (fit) ...", flush=True)
    xA, hA, yA = run_stream(1234)
    print(f"stream B (val) ...", flush=True)
    xB, hB, yB = run_stream(999)
    print(f"rows: A={hA.shape[0]} B={hB.shape[0]}", flush=True)

    d = get_int4x(L, K)
    w2q = d["w2b"].float()  # quantized readout from the checkpoint

    def resid(w2, h, y):
        return (((h @ w2.T - y) ** 2).sum() / (y ** 2).sum().clamp_min(1e-30)).item()

    print(f"checkpoint w2b:      in={resid(w2q, hA, yA)*100:6.2f}%  out={resid(w2q, hB, yB)*100:6.2f}%")

    G = hA.T @ hA
    rhs = hA.T @ yA
    for lam_mult in (1, 10, 100, 1000):
        lam = G.diagonal().mean() * 1e-2 * lam_mult
        W2 = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device="cuda"), rhs).T.contiguous()
        print(f"ridge lam x{lam_mult:<5d}   in={resid(W2, hA, yA)*100:6.2f}%  out={resid(W2, hB, yB)*100:6.2f}%")

    # anchored: shrink toward the checkpoint's quantized W2
    for lam_mult in (1, 10, 100):
        lam = G.diagonal().mean() * 1e-2 * lam_mult
        W2 = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device="cuda"),
                                rhs + lam * w2q.T).T.contiguous()
        print(f"anchored lam x{lam_mult:<4d}  in={resid(W2, hA, yA)*100:6.2f}%  out={resid(W2, hB, yB)*100:6.2f}%")

    # CONTROL: continuous W13 features (original weights through the same
    # mu/P fold) - if these also fail out-of-sample, the probe is broken.
    P = d["P"].float()
    mu = d["mu"].float()
    w1, w2o, w3 = get_dequant(L, K)
    w1r, w3r = (w1.float() @ P), (w3.float() @ P)
    b1 = mu.reshape(-1) @ w1.float().T
    b3 = mu.reshape(-1) @ w3.float().T

    def h_cont(x):
        z = (x.to(torch.bfloat16) - d["mu"]) @ d["P"]
        g = gen.soft_lim(z.float() @ w1r.T + b1[None, :])
        u = gen.soft_lim(z.float() @ w3r.T + b3[None, :])
        return (torch.nn.functional.silu(g) * u)



    def h_cont(x):
        z = ((x.to(torch.bfloat16) - d["mu"]) @ d["P"]).float()
        g = gen.soft_lim(z @ w1r.T + b1[None, :])
        u = gen.soft_lim(z @ w3r.T + b3[None, :])
        return torch.nn.functional.silu(g) * u

    hcA, hcB = h_cont(xA), h_cont(xB)
    # feature-level attribution: how bad are ternary W13 features per se?
    feA = ((hA - hcA) ** 2).sum() / hcA.pow(2).sum().clamp_min(1e-30)
    feB = ((hB - hcB) ** 2).sum() / hcB.pow(2).sum().clamp_min(1e-30)
    print(f"FEATURE err (tern W13 vs cont W13): A={feA.item()*100:.2f}%  B={feB.item()*100:.2f}%")
    # RTN-W2: int4 round-to-nearest toward the ORIGINAL W2 (A-independent)
    w2o_q = (w2o.float() / (w2o.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7)).round().clamp(-7, 7)
    w2o_s = w2o.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7
    # one LS pass on scales
    num = (w2o_q * w2o.float()).sum(dim=1, keepdim=True)
    den = (w2o_q * w2o_q).sum(dim=1, keepdim=True).clamp_min(1e-9)
    w2o_s = (num / den).clamp_min(1e-9)
    w2rtn = w2o_q * w2o_s
    print(f"RTN-W2 (no ridge, no A):  in={resid(w2rtn, hA, yA)*100:6.2f}%  out={resid(w2rtn, hB, yB)*100:6.2f}%")
    print(f"(teacher-self check: cont-W13 + orig W2: out={resid(w2o.float().T.contiguous().T, hcB, yB)*100:.2f}%)")

    # SHERRY-style: re-ternarize W13 with weighted-LS scales (importance =
    # per-column energy of the FIT z rows), same 128-groups, same grid.
    zA_rows = ((xA.to(torch.bfloat16) - d["mu"]) @ d["P"]).float()
    imp = (zA_rows ** 2).sum(0).clamp_min(1e-12)  # [K]

    def tern_wls(W, gs=128, iters=4):
        out_, in_ = W.shape
        ng_ = in_ // gs
        Wg = W.view(out_, ng_, gs)
        ig = imp.view(1, ng_, gs)
        s = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-9)
        for _ in range(iters):
            q = (Wg / s).round().clamp(-1, 1)
            num = (q * Wg * ig).sum(-1, keepdim=True)
            den = (q * q * ig).sum(-1, keepdim=True).clamp_min(1e-9)
            s = (num / den).clamp_min(1e-9)
        return ((Wg / s).round().clamp(-1, 1) * s).view(out_, in_)

    w1t, w3t = tern_wls(w1r), tern_wls(w3r)

    def h_wls(x):
        z = ((x.to(torch.bfloat16) - d["mu"]) @ d["P"]).float()
        g = gen.soft_lim(z @ w1t.T + b1[None, :])
        u = gen.soft_lim(z @ w3t.T + b3[None, :])
        return torch.nn.functional.silu(g) * u

    hwA, hwB = h_wls(xA), h_wls(xB)
    fwA = ((hwA - hcA) ** 2).sum() / hcA.pow(2).sum().clamp_min(1e-30)
    fwB = ((hwB - hcB) ** 2).sum() / hcB.pow(2).sum().clamp_min(1e-30)
    print(f"FEATURE err wLS-scale tern: A={fwA.item()*100:.2f}%  B={fwB.item()*100:.2f}%  (was: A=13.66 B=29.76)")

    # SHERRY 3:4: per group of 4 weights force ONE zero at the least
    # important position (importance = z-column energy), others sign(w);
    # scale d = weighted-LS per scale-group.
    def tern_34(W, gs_scale=128, iters=4):
        out_, in_ = W.shape
        n4 = in_ // 4
        W4 = W.view(out_, n4, 4)
        i4 = imp.view(n4, 4)
        sgn4 = W4.sign()
        zero_mask = torch.zeros_like(sgn4, dtype=torch.bool)
        zero_mask.scatter_(2, i4.argmin(dim=1, keepdim=True).unsqueeze(0).expand(out_, n4, 1), True)
        sgn4 = sgn4.masked_fill(zero_mask, 0.0)
        ngs = in_ // gs_scale
        sgn = sgn4.reshape(out_, ngs, gs_scale)
        Wg = W.reshape(out_, ngs, gs_scale)
        ig = imp.view(1, ngs, gs_scale)
        s = Wg.abs().mean(-1, keepdim=True).clamp_min(1e-9)
        for _ in range(iters):
            num = (sgn * Wg * ig).sum(-1, keepdim=True)
            den = (sgn * sgn * ig).sum(-1, keepdim=True).clamp_min(1e-9)
            s = (num / den).clamp_min(1e-9)
        return (sgn * s).view(out_, in_)

    def h_34(x, w1q, w3q):
        z = ((x.to(torch.bfloat16) - d["mu"]) @ d["P"]).float()
        g = gen.soft_lim(z @ w1q.T + b1[None, :])
        u = gen.soft_lim(z @ w3q.T + b3[None, :])
        return torch.nn.functional.silu(g) * u

    for tag, gs_s in (("g128", 128), ("g16", 16)):
        w134, w334 = tern_34(w1r, gs_s), tern_34(w3r, gs_s)
        h34A, h34B = h_34(xA, w134, w334), h_34(xB, w134, w334)
        f34A = ((h34A - hcA) ** 2).sum() / hcA.pow(2).sum().clamp_min(1e-30)
        f34B = ((h34B - hcB) ** 2).sum() / hcB.pow(2).sum().clamp_min(1e-30)
        print(f"FEATURE err 3:4-{tag}:          A={f34A.item()*100:.2f}%  B={f34B.item()*100:.2f}%")
        G34 = h34A.T @ h34A
        lam34 = G34.diagonal().mean() * 1e-2 * 10
        W234 = torch.linalg.solve(G34 + lam34 * torch.eye(G34.shape[0], device="cuda"), h34A.T @ yA).T.contiguous()
        print(f"3:4-{tag} + ridge x10:      in={resid(W234, h34A, yA)*100:6.2f}%  out={resid(W234, h34B, yB)*100:6.2f}%")

    # full expert with ridge lam x10 on A (wLS)
    Gw = hwA.T @ hwA
    lam = Gw.diagonal().mean() * 1e-2 * 10
    W2w = torch.linalg.solve(Gw + lam * torch.eye(Gw.shape[0], device="cuda"), hwA.T @ yA).T.contiguous()
    print(f"wLS-tern + ridge x10:    in={resid(W2w, hwA, yA)*100:6.2f}%  out={resid(W2w, hwB, yB)*100:6.2f}%  (was out=32.38)")
    Gc = hcA.T @ hcA
    rhsc = hcA.T @ yA
    for lam_mult in (1, 10):
        lam = Gc.diagonal().mean() * 1e-2 * lam_mult
        W2c = torch.linalg.solve(Gc + lam * torch.eye(Gc.shape[0], device="cuda"), rhsc).T.contiguous()
        print(f"CONTROL cont-W13 lam x{lam_mult:<3d} in={resid(W2c, hcA, yA)*100:6.2f}%  out={resid(W2c, hcB, yB)*100:6.2f}%")


if __name__ == "__main__":
    main()
