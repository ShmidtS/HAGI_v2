"""Automated inter x kp sweep to find the minimal expert size at a residual threshold.

Reuses train_sweep/resid_full/size_mb from dsv4_bench_compress2. Sweeps a grid of
inter x kp, reports size vs residual, then prints:
  - the minimal-size config whose residual is <= --threshold
  - the Pareto frontier (size -> residual tradeoff)

Usage:
  python scripts/dsv4_bench_autosize.py --layer 0 --n-experts 4 --steps 800 \
      --q-dtype bf16 --threshold 0.015 \
      --inter-list 32,64,128,256,512,1024 --kp-list 2,4,8,16,32,64
"""
import torch, os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_bench_compress2 import train_sweep, resid_full, size_mb, POD, REDUCED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--n-experts', type=int, default=4)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--q-dtype', choices=['int8', 'bf16', 'fp32'], default='bf16')
    ap.add_argument('--inter-list', default='32,64,128,256,512,1024')
    ap.add_argument('--kp-list', default='2,4,8,16,32,64')
    ap.add_argument('--threshold', type=float, default=0.015, help='max residual %%')
    args = ap.parse_args()

    inters = [int(x) for x in args.inter_list.split(',')]
    kps = [int(x) for x in args.kp_list.split(',')]
    q_bytes = {'int8': 1, 'bf16': 2, 'fp32': 4}[args.q_dtype]

    acts = torch.load(os.path.join(POD, f'acts_layer{args.layer}.pt'),
                      map_location='cpu', weights_only=False)
    P = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', 'P.pt'),
                   map_location='cuda').float()
    mu = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', 'mu.pt'),
                    map_location='cuda').float()

    keys = list(acts.keys())[:args.n_experts]
    pairs_full = []
    for k in keys:
        x_k, y_k = acts[k]
        n_k = x_k.shape[0]
        if n_k > 1024:
            idx = torch.randperm(n_k)[:1024]
            x_k = x_k[idx]
            y_k = y_k[idx]
        z = (x_k.float().cuda() - mu) @ P
        y_full = y_k.float().cuda()
        e = torch.load(os.path.join(REDUCED, f'layer_{args.layer}', f'expert_{k}.pt'),
                       map_location='cpu', weights_only=False)
        Qfull = e['Q'].float() * e['Q_scale'].float()[None, :]
        pairs_full.append((k, z, y_full, Qfull))

    print(f'layer {args.layer}, {len(pairs_full)} experts, {args.steps} steps, '
          f'q={args.q_dtype}, threshold={args.threshold}%', flush=True)
    print(f'{"inter":>6} {"kp":>5} {"MB":>7} {"resid%":>10}', flush=True)

    results = []
    for inter in inters:
        for kp in kps:
            pairs = [(z, y, Qfull[:, :kp]) for _, z, y, Qfull in pairs_full]
            t0 = time.time()
            out = train_sweep(pairs, inter, kp, args.steps, args.q_dtype,
                              check_every=args.steps + 1)
            meds = [resid_full(z, y, *r) for (_, z, y, _), r in zip(pairs_full, out)]
            med = torch.tensor(meds).median().item() * 100
            mb = size_mb(inter, kp, q_bytes)
            results.append((inter, kp, mb, med))
            print(f'{inter:>6} {kp:>5} {mb:>7.2f} {med:>10.4f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
            torch.cuda.empty_cache()

    ok = [r for r in results if r[3] <= args.threshold]
    print('', flush=True)
    if ok:
        best = min(ok, key=lambda r: r[2])
        print(f'OPTIMAL: inter={best[0]} kp={best[1]} size={best[2]:.2f} MB '
              f'resid={best[3]:.4f}% (min size at <= {args.threshold}%)', flush=True)
    else:
        print(f'No config meets threshold {args.threshold}%', flush=True)

    frontier = []
    for r in sorted(results, key=lambda x: x[2]):
        if not frontier or r[3] < frontier[-1][3]:
            frontier.append(r)
    print('Pareto frontier (size -> residual):', flush=True)
    for inter, kp, mb, med in frontier:
        print(f'  {inter:>6} {kp:>5} {mb:>7.2f} MB  {med:.4f}%', flush=True)


if __name__ == '__main__':
    main()
