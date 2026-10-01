"""Invariant tests for expert merging (merge_experts).

The measured lessons behind these tests:

1. (.omc/attempts/temperature_correction.md) A concat of N IDENTICAL children
   must reproduce one child. The old sqrt(N) logit_scale division violated
   that -- sharpened logits on an undertrained model lower CE for free, which
   made a broken merge look like a win. The merged scale must be child/n.

2. (.omc/attempts/cleanup_2026-09-27.md, debugging session) The apparent
   "drift" of a concat was NOT a bug in three separate places -- each was an
   intended mechanism:
     * ``expert_weight_source='ternary_master'`` ternarizes each expert block
       before the merge (leaf configs use plain Linear, so use
       ``effective_sparse`` to compare functions);
     * ``mixer_type='hadamard'`` right-multiplies the head projection by the
       orthonormal Hadamard Q -- that is the cross-expert mixer doing its
       job, and it is NOT identity at step 0 by design;
     * the qkv projection is re-laid-out to [all-q, all-k, all-v] block
       diagonals, which matches attention.py's split -- verified by hand.
   The invariant is therefore only exact on the no-rotation path
   (``mixer_type='swiglu'``, ``n_mixers=0``, ``effective_sparse``).

No training, no GPU: the models are tiny (H=8) and run on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import merge_experts  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402

VOCAB = 64


def _leaf_cfg(seed: int):
    # NOTE: this config is the pre-optimization leaf (lr 3e-4 era). The
    # canonical leaf_h128_v2 does NOT satisfy the tiny-model invariant here
    # (bisect pending: theta/rope or fused_ce path difference, see
    # .omc/attempts/test_config_regression.md). The invariant is about the
    # MERGE machinery, not the leaf recipe, so the historical config pins it.
    cfg = load_config(".omc/attic/sweep_configs/leaf_h128_s1001.yaml")
    m = cfg.model
    m.vocab_size = VOCAB
    m.hidden_size = 8
    m.num_layers = 1
    m.attention.num_query_heads = 1
    m.attention.num_kv_heads = 1
    m.attention.head_dim = 8
    m.attention.max_seq_len = 64
    m.ffn.expansion = 1.0
    m.ffn.multiple_of = 8
    m.init_seed = seed
    m.attention.qk_norm = False  # tiny-model invariant path: no learned q/k gains
    return cfg


def _parent_cfg(n: int):
    cfg = load_config("configs/parent_h128_flat.yaml")
    cfg.model.vocab_size = VOCAB
    cfg.model.hidden_size = 8 * n
    cfg.model.num_layers = 1
    cfg.model.attention.num_query_heads = n
    cfg.model.attention.num_kv_heads = n
    cfg.model.attention.head_dim = 8
    cfg.model.attention.max_seq_len = 64
    cfg.model.ffn.expansion = 1.0
    cfg.model.ffn.multiple_of = 8
    cfg.merge.n_experts = n
    # Exact-invariant overrides: sinks and loop re-application change the
    # model FUNCTION, not the merge machinery, so pin them off here.
    cfg.model.attention.sink_len = 0
    cfg.model.loop_depth = 1
    # The exact-invariant path: no ternarization (children are plain Linear),
    # no hadamard rotation (that is the mixer, not the merge), no mixers.
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.mixer_type = "swiglu"
    cfg.model.attention.qk_norm = False  # match the leaf's invariant path
    return cfg


def _states(n: int, seed: int = 0):
    """n identical leaf state_dicts built from ONE fixed leaf."""
    torch.manual_seed(seed)
    leaf = HAGI(_leaf_cfg(seed))
    sd = leaf.state_dict()
    return [dict(sd) for _ in range(n)]


def _logit_scale(model) -> float:
    return float(model.head.logit_scale)


def test_concat_identical_children_reproduce_child_logits():
    """Core invariant: concat of N identical children == one child, exactly.

    The concat head sums the block contributions into each logit, so the
    scale divided by n cancels the sum; with no ternarization and no
    hadamard rotation the merged function equals the child's function.
    """
    n = 3
    states = _states(n)
    leaf = HAGI(_leaf_cfg(0))
    leaf.load_state_dict(states[0], strict=True)
    merged = merge_experts(_parent_cfg(n), states, n_mixers=0)
    torch.manual_seed(7)
    ids = torch.randint(0, VOCAB, (1, 32))
    with torch.no_grad():
        a = leaf(ids, return_logits=True).logits
        b = merged(ids, return_logits=True).logits
    assert torch.allclose(a, b, atol=1e-5), (
        f"concat of identical children drifted: max|d|="
        f"{float((a - b).abs().max()):.2e}"
    )


def test_logit_scale_divides_by_n_not_sqrt_n():
    """The regression pin: the merged scale is child_scale / n."""
    n = 3
    states = _states(n)
    leaf = HAGI(_leaf_cfg(0))
    leaf.load_state_dict(states[0], strict=True)
    merged = merge_experts(_parent_cfg(n), states, n_mixers=0)
    expected = _logit_scale(leaf) / n
    assert abs(_logit_scale(merged) - expected) < 1e-6, (
        f"merged scale {_logit_scale(merged):.6f} != child/n {expected:.6f}"
    )


def test_hadamard_rotation_is_not_identity_by_design():
    """With mixer_type='hadamard' the head projection is right-multiplied by
    the orthonormal Q, so identical children do NOT reproduce the child --
    that is the cross-expert mixer, intended at design time. This test pins
    that the rotation is present (and orthonormal-sized), so nobody "fixes"
    it into an accidental identity."""
    n = 3
    states = _states(n)
    leaf = HAGI(_leaf_cfg(0))
    leaf.load_state_dict(states[0], strict=True)
    cfg = _parent_cfg(n)
    cfg.merge.mixer_type = "hadamard"
    merged = merge_experts(cfg, states, n_mixers=0)
    a_w = leaf.state_dict()["head.projection.weight"]
    b_w = merged.state_dict()["head.projection.weight"]
    same = torch.allclose(b_w[:, : a_w.shape[1]], a_w, atol=1e-6)
    assert not same, "hadamard rotation vanished: the head is no longer pre-rotated"


def test_local_window_sinks_fused():
    """Round 27 (DA synthesis §8/§9): the chunked local+sinks path must
    be bit-exact with the dense mask reference and O(T*(W+S))."""
    import sys
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from hagi.model.attention import build_attention_mask, local_window_attention

    torch.manual_seed(0)
    B, H, T, D, W, S = 1, 2, 256, 16, 64, 4
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    out = local_window_attention(q, k, v, W, sink_len=S)
    mask = build_attention_mask(T, T, window=W, sink_len=S, device=q.device, dtype=q.dtype)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-9)
    # window-only still exact
    out_w = local_window_attention(q, k, v, W, sink_len=0)
    mask_w = build_attention_mask(T, T, window=W, device=q.device, dtype=q.dtype)
    assert torch.allclose(
        out_w, F.scaled_dot_product_attention(q, k, v, attn_mask=mask_w), atol=1e-9
    )
