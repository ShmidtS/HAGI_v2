"""CCA-based joint P/Q alignment test (crossbar / коммутация).

Key claim: in the CCA basis, the linear cross-covariance cov(z,g) is
exactly diagonal with entries rho_i. So the linear part of x->y is
captured by a DIAGONAL layer, and the ternary core only needs to model
the nonlinear residual -> inter can shrink drastically.

Test on aggregated layer-0 acts (enough tokens for stable CCA).
"""
import torch, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_experts as de  # noqa

D = 4096
LAM = 0.01


def cca(X, Y, lam=LAM):
    """Ridge CCA. X,Y centered [N,D]. Returns P,Q [D,D], rho [D]."""
    N = X.shape[0]
    Cxx = (X.T @ X) / N
    Cyy = (Y.T @ Y) / N
    Cxy = (X.T @ Y) / N
    exx, Vxx = torch.linalg.eigh(Cxx + lam * torch.eye(D, device=X.device, dtype=X.dtype))
    eyy, Vyy = torch.linalg.eigh(Cyy + lam * torch.eye(D, device=X.device, dtype=X.dtype))
    Wx = Vxx @ torch.diag(1.0 / torch.sqrt(exx.clamp_min(1e-12))) @ Vxx.T
    Wy = Vyy @ torch.diag(1.0 / torch.sqrt(eyy.clamp_min(1e-12))) @ Vyy.T
    M = Wx @ Cxy @ Wy.T
    U, S, Vh = torch.linalg.svd(M)
    P = Wx @ U
    Q = Wy @ Vh.T
    return P, Q, S.clamp(0, 1)


def main():
    d = 'checkpoints_dsv4/pod_accurate'
    acts = torch.load(os.path.join(d, 'acts_layer0.pt'), map_location='cpu', weights_only=False)
    xs, ys = [], []
    for k, (x, y) in acts.items():
        xs.append(x); ys.append(y)
    X = torch.cat(xs).to('cuda').double()   # float64 for stability
    Y = torch.cat(ys).to('cuda').double()
    N = X.shape[0]
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)

    P, Q, rho = cca(Xc, Yc)

    # 1. verify diagonality: cov(z,g) = P^T Cxy Q should be diag(rho)
    Cxy = (Xc.T @ Yc) / N
    cov = P.T @ Cxy @ Q
    offdiag = cov - torch.diag(torch.diag(cov))
    print(f'cov(z,g) off-diagonal max abs: {offdiag.abs().max().item():.2e}')
    print(f'cov(z,g) diagonal vs rho max err: {(torch.diag(cov) - rho).abs().max().item():.2e}')

    # 2. rho decay + linear reconstruction with r diagonal terms
    tr_y = (Yc ** 2).mean().item()  # total y variance (trace/norm)
    print(f'N={N}  trace(y-var)={tr_y:.1f}  sum rho^2={ (rho**2).sum().item():.1f}')
    for r in [128, 256, 384, 512]:
        Pr = P[:, :r]
        Qr = Q[:, :r]
        z = Xc @ Pr                  # [N,r]
        g = z * rho[:r].unsqueeze(0)  # diagonal link
        y_hat = g @ Qr.T             # [N,D]
        resid = ((Yc - y_hat) ** 2).mean().item() / tr_y
        print(f'  CCA diag r={r}: linear residual = {resid*100:.2f}%')

    # 3. compare: PCA top-r of x, PCA top-r of y, linear regression between
    Ux, Sx, Vhx = torch.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, Vhy = torch.linalg.svd(Yc, full_matrices=False)
    for r in [256, 384, 512]:
        Px = Vhx.T[:, :r]  # PCA basis = right singular vectors
        Qy = Vhy.T[:, :r]
        # best linear map between z and g
        z = Xc @ Px
        g = Yc @ Qy
        W = torch.linalg.lstsq(z, g).solution  # [r,r]
        y_hat = (z @ W) @ Qy.T
        resid = ((Yc - y_hat) ** 2).mean().item() / tr_y
        print(f'  PCA-regress r={r}: linear residual = {resid*100:.2f}%')


if __name__ == '__main__':
    main()
