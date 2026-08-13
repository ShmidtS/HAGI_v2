"""Recursive F3 tree merge with LOSSLESS compression between ternary levels.

Pipeline per layer:
    level 0: 128 triples (E2k, S, E2k+1) -> sync channel + renorm to S
    LOSSLESS compress the 128 super-experts (table + packed indices)
    level 1: group into triples -> sync + renorm
    LOSSLESS compress
    ... until 1 expert remains.

Lossless encoding: after a merge the super-expert values are a finite set
(sum of fp4-grid values), so we store torch.unique values (float32 table) +
int16 indices. Dequant is bit-exact (torch.equal).

Reports per-level compressed size + compression ratio and verifies bit-exact
roundtrip at every level.
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import dsv4_experts as de
import dsv4_merge_f3 as dm

PROJ = ("w1", "w2", "w3")
DEVICE = "cuda"


def lossless_bits(t: torch.Tensor) -> int:
    """Entropy lower bound in bits/element: ceil(log2(#unique values)). CPU."""
    tc = t.detach().to("cpu")
    u = torch.unique(tc)
    return math.ceil(math.log2(max(u.numel(), 2)))


def lossless_compress(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a tensor as (unique values table, int16 indices). Bit-exact."""
    tc = t.detach().to("cpu")
    u, inv = torch.unique(tc, return_inverse=True)
    return u, inv.to(torch.int16 if u.numel() <= 32767 else torch.int32)


def lossless_decompress(u: torch.Tensor, inv: torch.Tensor, shape: tuple) -> torch.Tensor:
    return u[inv.long()].reshape(shape)


def expert_compressed_size(u: dict[str, torch.Tensor], inv: dict[str, torch.Tensor]) -> int:
    """Bytes needed for the lossless representation (table fp32 + int16 indices)."""
    return sum(u[p].numel() * 4 + inv[p].numel() * 2 for p in PROJ)


def expert_raw_size(t: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() * v.element_size() for v in t.values())


def merge_layer_lossless(
    lossless_file: str,
    layer_prefix: str,
    out_dir: str,
    progress: bool = True,
    renorm: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Recursive tree merge with lossless compression of each intermediate level."""
    S = {p: de.load_shared_file(lossless_file, layer_prefix)[p].to(DEVICE) for p in PROJ}
    t0 = time.time()

    # Level 0: 128 triples.
    level: list[dict[str, torch.Tensor]] = []
    for k in range(128):
        a = de.load_expert_file(lossless_file, layer_prefix, 2 * k)
        c = de.load_expert_file(lossless_file, layer_prefix, 2 * k + 1)
        m = {}
        for p in PROJ:
            m[p] = (a[p].to(DEVICE) + S[p] + c[p].to(DEVICE)) / math.sqrt(3.0)
        del a, c
        if renorm:
            dm._renorm_to_ref(m, S)
        level.append(m)

    report: list[dict] = []
    lvl = 0
    while True:
        # Lossless-compress the current level and verify bit-exact roundtrip.
        raw_bytes = 0
        comp_bytes = 0
        max_bits = 0
        for ex in level:
            for p in PROJ:
                uu, ii = lossless_compress(ex[p])
                rec = lossless_decompress(uu, ii, ex[p].shape)
                assert torch.equal(rec, ex[p].to("cpu")), f"level{lvl} {p} NOT bit-exact"
                raw_bytes += ex[p].numel() * ex[p].element_size()
                comp_bytes += uu.numel() * 4 + ii.numel() * ii.element_size()
                max_bits = max(max_bits, math.ceil(math.log2(max(uu.numel(), 2))))
        report.append({
            "level": lvl,
            "n_experts": len(level),
            "raw_bytes": raw_bytes,
            "comp_bytes": comp_bytes,
            "ratio": raw_bytes / comp_bytes,
            "bits": max_bits,
        })
        if progress:
            print(f"  level{lvl}: {len(level)} experts, raw {raw_bytes/1e9:.2f} GB -> "
                  f"lossless {comp_bytes/1e9:.2f} GB ({raw_bytes/comp_bytes:.2f}x), "
                  f"bits/elem={max_bits}, bit-exact", flush=True)

        if len(level) == 1:
            break

        # Next ternary level.
        lvl += 1
        nxt: list[dict[str, torch.Tensor]] = []
        for i in range(0, len(level), 3):
            group = level[i:i + 3]
            nxt.append(dm._sync_merge(group, S, renorm=renorm))
        level = nxt

    return level[0], report


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "layers.3.ffn"
    lossless_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lossless_layers"))
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "merge_lossless_report"
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints_dsv4"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    lossless_file = os.path.join(lossless_dir, prefix.replace(".", "_") + ".safetensors")
    t0 = time.time()
    sup, report = merge_layer_lossless(lossless_file, prefix, out_dir, renorm=False)
    print(f"merged {prefix} in {time.time() - t0:.1f}s (no renorm)", flush=True)
    for p in PROJ:
        print(f"  {p}: {tuple(sup[p].shape)} std={sup[p].std():.5f} norm={sup[p].norm():.3f}")
    # save the final super-expert (bf16) for model rebuild
    out = os.path.join(ckpt_dir, prefix.replace(".", "_") + ".pt")
    torch.save({k: v.to(torch.bfloat16) for k, v in sup.items()}, out)
    print(f"saved super-expert -> {out}", flush=True)
    # summary: total raw vs total compressed across all levels
    raw = sum(r["raw_bytes"] for r in report)
    comp = sum(r["comp_bytes"] for r in report)
    print(f"TOTAL across levels: raw {raw/1e9:.2f} GB -> lossless {comp/1e9:.2f} GB ({raw/comp:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
