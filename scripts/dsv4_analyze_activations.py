"""Activation inference analysis (lightweight, no CCA).

Two questions from the original idea:
1. In-expert: how predictable is each expert's output y_k from its input x_k?
   -> linear explained variance via SVD of x_k (small matrix [n_k, 4096]).
2. Between layers: how predictable is x_{L+1} from x_L?
   -> linear explained variance via SVD of x_layer{L}.

Both are single small SVDs, cheap on GPU.
"""
import torch, os, glob

POD = 'checkpoints_dsv4/pod_accurate'
N_LAYERS = 43


def lin_explained(X, Y):
    """Fraction of centered Y variance explained by linear function of centered X."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Xc = Xc.float().cuda()
    Yc = Yc.float().cuda()
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    # range(X) spanned by Vh.T (right singular vectors); project Y onto it
    coef = Yc @ Vh.T          # [N, r]
    proj = coef @ Vh          # [N, D]  (Vh.T @ Vh = I, rank r)
    explained = (proj ** 2).sum() / (Yc ** 2).sum().clamp_min(1e-12)
    return explained.item()


def main():
    # 1. in-expert linearity
    residuals = []
    per_layer = []
    for L in range(N_LAYERS):
        fp = os.path.join(POD, f'acts_layer{L}.pt')
        if not os.path.exists(fp):
            continue
        acts = torch.load(fp, map_location='cpu', weights_only=False)
        layer_res = []
        for k, (x_k, y_k) in acts.items():
            if x_k.shape[0] < 2:
                continue
            expl = lin_explained(x_k, y_k)
            layer_res.append(1.0 - expl)
        if layer_res:
            per_layer.append(torch.tensor(layer_res).mean().item())
            residuals.extend(layer_res)
        print(f'layer {L}: mean in-expert linear residual = '
              f'{torch.tensor(layer_res).mean().item()*100:.2f}% '
              f'({len(layer_res)} experts)', flush=True)

    r = torch.tensor(residuals)
    print(f'\\nIN-EXPERT linear residual: mean={r.mean()*100:.2f}% '
          f'median={r.median()*100:.2f}% p10={r.quantile(0.1)*100:.2f}% '
          f'p90={r.quantile(0.9)*100:.2f}%', flush=True)

    # 2. between-layer evolution
    print('\\nBETWEEN-LAYER linear predictability (x_L -> x_{L+1}):', flush=True)
    prev = None
    for L in range(N_LAYERS):
        fp = os.path.join(POD, f'x_layer{L}.pt')
        if not os.path.exists(fp):
            prev = None
            continue
        X = torch.load(fp, map_location='cpu', weights_only=False)
        if prev is not None:
            expl = lin_explained(prev, X)
            print(f'  L{L-1}->L{L}: explained = {expl*100:.2f}%', flush=True)
        prev = X


if __name__ == '__main__':
    main()
