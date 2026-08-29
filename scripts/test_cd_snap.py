# -*- coding: utf-8 -*-
"""Test: does coordinate-descent scale fitting (AngelSlim STQ1_0 trick) let the
int4 snap HOLD the continuous TTT adaptation gain on honest validation?

Compares on real expert (L=0,k=0), real activation stream:
  r_static  - checkpoint readout (baseline)
  r_cont    - adapted continuous w2 (the gain we want to keep)
  r_snap    - current single-pass snap (amax init + one LS pass)
  r_cd      - CD snap (rounds of LS scale <-> pattern requant)
  r_cdw     - CD snap weighted by feature energy (imatrix analog: c_j = sum h_j^2)
"""
import sys, torch
sys.path.insert(0, "scripts")
import dsv4_generate_ttt as G

torch.manual_seed(0)


def snap_cd(w2, c=None, rounds=3):
    """Coordinate descent: alternate LS scale <-> pattern requant; optional
    per-column weights c [F] (weighted error == the true functional resid)."""
    cw = c if c is not None else torch.ones(w2.shape[1], device=w2.device)
    sg = w2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q = (w2 / sg).round().clamp(-7, 7)
    for _ in range(rounds):
        num = (cw * q * w2).sum(dim=1, keepdim=True)
        den = (cw * q * q).sum(dim=1, keepdim=True).clamp_min(1e-9)
        sg = num / den
        sg = torch.where(sg.abs() < 1e-9, torch.full_like(sg, 1e-9), sg)  # keep magnitude!
        q = (w2 / sg).round().clamp(-7, 7)
    num = (cw * q * w2).sum(dim=1)
    den = (cw * q * q).sum(dim=1).clamp_min(1e-9)
    a = (num / den)
    a = a.sign() * a.abs().clamp_min(1e-6)
    return q, a


L, k = 0, 0
d = G.get_int4x(L, k)
from dsv4_ttt_probe import load_expert
h_all, y_all, _ = load_expert(L, k)

# original checkpoint readout (static baseline)
from dsv4_experts import unpack_int4
e = torch.load(f"dsv4_reduced/layer_{L}/expert_{k}.pt", map_location="cpu", weights_only=False)
w2_static = unpack_int4(e["w2a"]).float().cuda() * e["w2a_scale"].float().cuda()[:, None]

# 1) adapt on train stream (same as evolve flush would)
for t in range(0, 1200, 16):
    G.ttt_update(L, k, h_all[t:t+16], y_all[t:t+16])
w2 = G.get_w2_fp32(L, k)

# 2) honest validation = rows never fed into G/C = Hva
st = G.TTT_STATE[(L, k)]
Hva, Yva = torch.cat(st["Hva"]), torch.cat(st["Yva"])
# fresh tail (extra honesty)
Hf, Yf = h_all[1200:1500], y_all[1200:1500]

res = lambda w, H, Y: (((H @ w.T - Y) ** 2).sum() / (Y ** 2).sum().clamp_min(1e-12)).item()
r_static = res(w2_static, Hva, Yva)
r_cont = res(w2, Hva, Yva)

q0, a0 = G.snap_int4(w2)
r_snap = res(q0 * a0[:, None], Hva, Yva)

q1, a1 = snap_cd(w2)
r_cd = res(q1 * a1[:, None], Hva, Yva)

# feature-energy weights from the TRAIN buffer (what the functional error weights)
Htr = torch.cat(st["H"])
c = (Htr ** 2).sum(dim=0)
c = c / c.mean().clamp_min(1e-12)
q2, a2 = snap_cd(w2, c=c)
r_cdw = res(q2 * a2[:, None], Hva, Yva)

print(f"VAL (never trained on):   static={r_static*100:.3f}%  cont={r_cont*100:.3f}%")
print(f"  snap(1-pass LS): {r_snap*100:.3f}%   CD: {r_cd*100:.3f}%   CD+imatrix: {r_cdw*100:.3f}%")
print(f"  kept-gain vs cont: 1pass {100*(r_static-r_snap)/max(r_static-r_cont,1e-12):.0f}%  "
      f"CD {100*(r_static-r_cd)/max(r_static-r_cont,1e-12):.0f}%  "
      f"CDw {100*(r_static-r_cdw)/max(r_static-r_cont,1e-12):.0f}%")
# fresh-tail cross-check
print(f"FRESH tail:              static={res(w2_static,Hf,Yf)*100:.3f}%  cont={res(w2,Hf,Yf)*100:.3f}%  "
      f"1pass={res(q0*a0[:,None],Hf,Yf)*100:.3f}%  CD={res(q1*a1[:,None],Hf,Yf)*100:.3f}%  "
      f"CDw={res(q2*a2[:,None],Hf,Yf)*100:.3f}%")
