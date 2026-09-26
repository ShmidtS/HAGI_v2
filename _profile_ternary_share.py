"""Which parameters are actually ternary, and which are ordinary fp32?

Motivation: a full step costs 5.0s with fp32 weights and 0.53s with bf16
weights, while one transformer block in isolation costs 55ms with no weight
gradients. Something is computing in fp32 that does not need to. Before
changing any dtype, find out which parameters are BitLinear masters (which
must stay fp32, the optimizer needs them) and which are plain weights that
happen to be fp32 by default and dominate the backward cost.
"""
import sys

sys.path.insert(0, "src")
import torch

from hagi.config import Config
from hagi.model.factory import build_model_for_config
from hagi.model.ternary import BitLinear


def main() -> None:
    cfg = Config()
    cfg.model.vocab_size = 32768
    cfg.model.hidden_size = 1152
    cfg.model.num_layers = 3
    cfg.model.attention.num_query_heads = 18
    cfg.model.attention.num_kv_heads = 6
    model = build_model_for_config(cfg)

    bit_params = {
        name
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
        for name, _ in module.named_parameters(recurse=False)
    }
    prefixes = [
        name for name, module in model.named_modules() if isinstance(module, BitLinear)
    ]

    total = sum(p.numel() for p in model.parameters())
    in_bit = sum(
        p.numel()
        for name, p in model.named_parameters()
        if any(name.startswith(pre + ".") for pre in prefixes)
    )
    print(f"total        {total/1e6:8.1f}M")
    print(f"BitLinear    {in_bit/1e6:8.1f}M ({in_bit/total*100:.1f}%)")
    print(f"plain fp32   {(total-in_bit)/1e6:8.1f}M ({(total-in_bit)/total*100:.1f}%)")

    by_group: dict[str, float] = {}
    for name, param in model.named_parameters():
        group = name.split(".")[0]
        by_group[group] = by_group.get(group, 0.0) + param.numel()
    print("\nby top-level module:")
    for key, value in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {key:12s} {value/1e6:7.1f}M")

    print("\nplain (non-BitLinear) parameters by name:")
    shown = 0
    for name, param in model.named_parameters():
        if any(name.startswith(pre + ".") for pre in prefixes):
            continue
        print(f"  {name:44s} {str(param.dtype):16s} {tuple(param.shape)}")
        shown += 1
        if shown >= 14:
            print("  ...")
            break


if __name__ == "__main__":
    main()
