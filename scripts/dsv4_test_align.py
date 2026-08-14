"""Fast cross-covariance alignment (коммутация) test.

Instead of full CCA (slow eigh/SVD 4096x4096 in double), project onto the
existing PCA bases first (P: 512, Q: 384), then SVD the small 512x384
cross-covariance to align P and Q so the linear link is diagonal.
"""
import torch, os

d = 'checkpoints_dsv4/pod_accurate'
acts = torch.load(os.path.join(d, 'acts_layer0.pt'), map_location='cpu', weights_only=False)
xs, ys = [], []
for k, (x, y) in acts.items():
    xs.append(x); ys.append(y)
X = torch.cat(xs).cuda().float()
Y = torch.cat(ys).cuda().float()
Xc = X - X.mean(0)
Yc = Y - Y.mean(0)
N = Xc.shape[0]
tr_y = (Yc ** 2).mean().item()
print(f'N={N}  total y energy={tr_y:.1f}', flush=True)

# fast PCA via eigh of covariance (float32)
Cxx = Xc.T @ Xc / N
Cyy = Yc.T @ Yc / N
evx, Vx = torch.linalg.eigh(Cxx)
evy, Vy = torch.linalg.eigh(Cyy)
P = Vx[:, -512:].contiguous()
Q = Vy[:, -384:].contiguous()

z = Xc @ P   # [N,512]
g = Yc @ Q   # [N,384]
Czg = z.T @ g / N   # [512,384]

# full linear regression residual (no alignment) = current capability
W = torch.linalg.lstsq(z, g).solution   # [512,384]
y_hat_full = (z @ W) @ Q.T
resid_full = ((Yc - y_hat_full) ** 2).mean().item() / tr_y
print(f'full linear residual (512->384, no align): {resid_full*100:.2f}%', flush=True)

# cross-cov alignment (коммутация): SVD of 512x384
U, S, Vh = torch.linalg.svd(Czg)
P_al = P @ U          # [4096,512]
Q_al = Q @ Vh.T       # [4096,384]
z_al = Xc @ P_al
g_al = Yc @ Q_al
rmin = min(z_al.shape[1], g_al.shape[1])
cov = z_al[:, :rmin].T @ g_al / N   # [rmin, rmin] should be diag(S[:rmin])
offdiag = cov - torch.diag(torch.diag(cov))
print(f'aligned cov(z,g) [{rmin}x{rmin}]: off-diag max={offdiag.abs().max().item():.2e}, '
      f'diag-vs-rho err={(torch.diag(cov)-S[:rmin]).abs().max().item():.2e}', flush=True)

# diagonal reconstruction with r terms (linear part)
for r in [128, 256, 384, 512]:
    y_hat = (z_al[:, :r] * S[:r]) @ Q_al[:, :r].T
    resid = ((Yc - y_hat) ** 2).mean().item() / tr_y
    print(f'diag(commutated) r={r}: residual={resid*100:.2f}%', flush=True)

# rho decay (how many diagonal links matter)
for r in [128, 256, 384]:
    print(f'  rho[{r}]={S[r].item():.4f}', flush=True)
