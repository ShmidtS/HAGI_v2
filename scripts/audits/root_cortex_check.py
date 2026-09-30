"""Root-mode cortex vs the parent-preserving invariant -- MEASURED, not argued.

``docs/ARCHITECTURE_V2.md`` claims the F3 fence was real for a DENSE cortex
and too wide for a ROOT-mode one. This script measures both halves on a real
depth-1 parent assembled from three identical child states:

1. root cortex  -- the parent-preserving invariant holds (the tree's logits
   equal the single child's logits within fp32 tolerance) and the leaves stay
   equal after the bark wakes;
2. dense cortex -- the leaves stop being equal as soon as the edges carry
   gradient, so ``BlockTreeNorm`` normalizes siblings by different statistics
   and the invariant is gone. That is what ``config.py`` forbids, and the
   measurement is the proof the fence was not mere caution.

The dense parent is assembled by hand (a dense cortex in a ternary tree is
rejected by ``validate_config`` on purpose). At the exact init instant a dense
cortex also preserves the invariant -- its links are zero-initialized, so it
injects nothing -- which is why the decisive measurement is taken after one
real optimizer step, not at init.

Run:  python scripts/root_cortex_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from hagi.config import Config, validate_config  # noqa: E402
from hagi.model.cortex import PyramidalCortex  # noqa: E402
from hagi.model.merge import merge_recursive_f3  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402

# fp32 tolerance. The parent and the child take different arithmetic paths
# (BlockTreeNorm vs RMSNorm, a lifted head vs a plain one), so exact bitwise
# equality is not the claim; equality to fp32 round-off is.
TOL = 1e-4

CHILD_HIDDEN = 24
PARENT_HIDDEN = 3 * CHILD_HIDDEN
HEAD_DIM = 8
NUM_LAYERS = 2
N_LEAVES = 3  # depth-1 parent: 3**1 leaves of CHILD_HIDDEN each
RANK = 8
LR = 0.5
STEPS = 5


def _base_model(cfg: Config, hidden: int, q_heads: int, kv_heads: int, inter: int) -> None:
    """Tiny effective-sparse body shared by child and parent."""
    m = cfg.model
    m.vocab_size = 64
    m.hidden_size = hidden
    m.num_layers = NUM_LAYERS
    m.loop_depth = 2  # the fixed technical self-improve seam ternary_f3 demands
    m.attention.num_query_heads = q_heads
    m.attention.num_kv_heads = kv_heads
    m.attention.head_dim = HEAD_DIM
    m.attention.max_seq_len = 16
    m.sliding.window = 0
    m.ffn.intermediate_size = inter
    m.ffn.multiple_of = 8
    m.ternary.enabled = False
    m.embedding.tie_lm_head = False
    m.embedding.conv_kernel = 1
    m.head.unigram_prior = False
    m.head.unigram_path = ""
    m.head.sampled_proposal = "uniform"
    m.head.sampled_softmax_k = 16
    m.head.ce_chunk_rows = 64
    m.adapters.enabled = False
    m.cortex.enabled = False
    m.decision.enabled = False
    cfg.train.precision = "fp32"
    cfg.train.ternary_fp32_master = False
    cfg.train.ternary_step_cache = False
    cfg.train.data.seq_len = 16


def child_config(seed: int = 7) -> Config:
    cfg = Config()
    _base_model(cfg, CHILD_HIDDEN, 3, 1, CHILD_HIDDEN)
    cfg.model.init_seed = seed
    validate_config(cfg)
    return cfg


def parent_config(cortex_mode: str | None, seed: int = 7) -> Config:
    """Depth-1 parent config. ``cortex_mode``: ``"root"``, ``"dense"`` or None.

    A dense cortex is only ever requested here, for the measurement that shows
    why the config fence rejects it; no supported config produces one.
    """
    cfg = Config()
    _base_model(cfg, PARENT_HIDDEN, 9, 3, PARENT_HIDDEN)
    cfg.model.init_seed = seed
    mg = cfg.merge
    mg.enabled = True
    mg.mixer_type = "ternary_f3"
    mg.n_experts = 3
    mg.expert_hidden = CHILD_HIDDEN
    mg.ternary_depth = 1
    mg.ternary_tree_schema_version = 1
    mg.ternary_lift_mode = "parent_preserving"
    mg.expert_weight_source = "effective_sparse"
    if cortex_mode is not None:
        cx = cfg.model.cortex
        cx.enabled = True
        cx.mode = cortex_mode
        cx.num_levels = 2
        cx.rank = RANK
        cx.link_strides = (1,)
        cx.residual_scale = 0.1
    validate_config(cfg)
    return cfg


def leaf_spread(h: torch.Tensor, n_leaves: int = N_LEAVES) -> float:
    """Max abs deviation of any leaf from leaf 0 -- the invariant's observable.

    ``h`` is ``[B, T, n_leaves * leaf_hidden]``. Zero means the stream is a
    pure lift of one parent state (all leaves equal); anything above zero means
    the tree carries leaf-dependent content the per-leaf norms will distort.
    """
    leaves = h.detach().reshape(*h.shape[:-1], n_leaves, -1)
    return float((leaves - leaves[..., :1, :]).abs().max())


def logits_and_hidden(model: nn.Module, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(ids, return_logits=True)
    if out.logits is None or out.hidden is None:
        raise RuntimeError("forward returned no logits/hidden")
    return out.logits.detach(), out.hidden.detach()


def wake_cortex(model: nn.Module, ids: torch.Tensor, targets: torch.Tensor) -> None:
    """One real optimizer step on the cortex parameters only.

    Init is not the interesting instant: both modes inject exactly zero while
    the directed links are zero. Training moves the links on the very first
    step, and that is where the two modes part. A plain SGD step with an
    deliberately large rate stands in for the run's Muon/AdamW update -- the
    magnitude does not matter, only that the edges are no longer zero.
    """
    model.train()
    params = [p for p in model.cortex.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR)
    opt.zero_grad(set_to_none=True)
    model(ids, targets=targets).loss.backward()
    with torch.no_grad():
        for p in params:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError("cortex gradient is non-finite")
    opt.step()
    model.eval()

def reference_dense_boundary(
    cortex: PyramidalCortex, x: torch.Tensor, level: int, states: list[torch.Tensor | None]
) -> torch.Tensor:
    """The documented formula, written straight from the module docstring.

    ``h' = h + residual_scale * U_l(sum_{k<l} P_{k,l} z_k)`` with
    ``z_k = R_k(h_k)``. Used to prove the shipped dense path is bit-identical
    to the pre-change arithmetic, i.e. no regression on the flat path.
    """
    if level in cortex._down_index:
        down = cortex.down[cortex._down_index[level]]
        states[level] = down(x)
    sources = cortex._sources_by_level[level]
    if not sources:
        return x
    total = None
    for source in sources:
        link = cortex.links[cortex._edge_index[(source, level)]]
        piece = nn.functional.linear(states[source], link.weight)
        total = piece if total is None else total + piece
    up = cortex.up[cortex._target_index[level]]
    delta = nn.functional.linear(total, up.weight)
    return x + cortex.residual_scale * delta


def trace(
    model: nn.Module, ids: torch.Tensor, targets: torch.Tensor, child_logits: torch.Tensor, steps: int
) -> list[tuple[int, float, float]]:
    """Return ``(step, leaf spread, |deltalogits vs child|)`` for step 0 .. ``steps``.

    Step 0 is the init instant; steps 1..N are real optimizer steps. Printing
    the whole schedule instead of one number is the point: root's spread is flat
    at zero while dense's grows every step, and a single snapshot could be
    mistaken for luck.
    """
    rows: list[tuple[int, float, float]] = []
    for step in range(steps + 1):
        if step:
            wake_cortex(model, ids, targets)
        with torch.no_grad():
            logits, hidden = logits_and_hidden(model, ids)
        rows.append((step, leaf_spread(hidden), float((logits - child_logits).abs().max())))
    return rows


def main() -> int:
    torch.manual_seed(0)
    child_cfg = child_config()
    child = HAGI(child_cfg).eval()
    state = {k: v.detach().clone() for k, v in child.state_dict().items()}

    ids = torch.randint(0, child_cfg.model.vocab_size, (2, 8))
    targets = torch.randint(0, child_cfg.model.vocab_size, (2, 8))

    with torch.no_grad():
        child_logits, child_hidden = logits_and_hidden(child, ids)

    print("=" * 74)
    print("HAGI root-cortex check -- depth-1 F3 parent, three identical children")
    print("=" * 74)
    print(f"child:  hidden {CHILD_HIDDEN}, leaves x {CHILD_HIDDEN}, vocab {child_cfg.model.vocab_size}")
    print(f"parent: hidden {PARENT_HIDDEN} = {N_LEAVES} x {CHILD_HIDDEN}, rank {RANK}, "
          f"levels 2, strides (1,), residual_scale 0.1")
    print(f"tolerance: {TOL:g} (fp32, max abs logit deviation)")

    failures: list[str] = []

    # ---------------------------------------------------------------- root
    root_cfg = parent_config("root")
    root_parent = merge_recursive_f3(
        root_cfg, [state, state, state], child_configs=[child_cfg, child_cfg, child_cfg],
        cross_parent_transform="parent_preserving",
    ).eval()

    # ---------------------------------------------------------------- dense
    # A dense cortex cannot be assembled through merge_recursive_f3 (the fence
    # is the point), so build the same parent without a cortex and attach one.
    plain_cfg = parent_config(None)
    dense_parent = merge_recursive_f3(
        plain_cfg, [state, state, state], child_configs=[child_cfg, child_cfg, child_cfg],
        cross_parent_transform="parent_preserving",
    ).eval()
    torch.manual_seed(11)
    dense_parent.cortex = PyramidalCortex(NUM_LAYERS, PARENT_HIDDEN, _dense_cortex_cfg())

    root_rows = trace(root_parent, ids, targets, child_logits, STEPS)
    dense_rows = trace(dense_parent, ids, targets, child_logits, STEPS)

    print("\n[1] leaf spread (max |leaf_i - leaf_0| of the final hidden state) and")
    print("    max |logits(parent) - logits(child)|, per optimizer step")
    print()
    print("    step |   root spread  |  dense spread   |  root |dlogit| | dense |dlogit|")
    print("    -----+----------------+-----------------+----------------+---------------")
    for (rs, rsp, rd), (_, dsp, dd) in zip(root_rows, dense_rows):
        label = "init" if rs == 0 else str(rs)
        print(f"    {label:>4} | {rsp:14.4e} | {dsp:15.4e} | {rd:14.4e} | {dd:14.4e}")

    # --- the init instant
    r0_spread, r0_diff = root_rows[0][1], root_rows[0][2]
    d0_spread, d0_diff = dense_rows[0][1], dense_rows[0][2]
    print()
    print(f"    root  @ init: |dlogit| = {r0_diff:.3e} (tolerance {TOL:g}), spread {r0_spread:.1e} -> PASS")
    print(f"    dense @ init: |dlogit| = {d0_diff:.3e}, spread {d0_spread:.1e} -> holds too")
    print("    Both hold at init: the directed links are zero-initialized, so a fresh")
    print("    cortex injects nothing. The fence is not about this instant.")
    if not (r0_diff <= TOL and r0_spread == 0.0):
        failures.append("root invariant at init")
    if not (d0_diff <= TOL and d0_spread == 0.0):
        failures.append("dense at init broke unexpectedly (zero-init property lost?)")

    # --- the trained instants
    root_broke = [(s, v) for s, v, _ in root_rows if s and v != 0.0]
    dense_first = next(((s, v) for s, v, _ in dense_rows if s and v > 0.0), None)
    print()
    if root_broke:
        print(f"    ROOT  FAIL: leaf equality broken at steps {[s for s, _ in root_broke]}")
        failures.append("root leaf equality after wake")
    else:
        print(f"    ROOT  PASS: spread stayed exactly 0.0 for all {STEPS} steps with awake edges")
        print("          the injected term is equal in every leaf, so BlockTreeNorm sees")
        print("          no distortion and the residual stream carries it untouched.")
    if dense_first is None:
        print("    DENSE unexpected: spread never broke -- the fence would be unjustified")
        failures.append("dense leaf spread did not break (fence unjustified?)")
    else:
        step, value = dense_first
        print(f"    DENSE FAIL: spread > 0 from step {step} onward, {value:.4e} at step {STEPS}")
        print("          the additive term is leaf-dependent, BlockTreeNorm normalizes each")
        print("          sibling by its own statistic and the parent-preserving invariant")
        print("          is gone. This is what config.py's ternary_f3 fence prevents.")
        print(f"          root {root_rows[-1][1]:.1e} vs dense {dense_rows[-1][1]:.4e} at step {STEPS}.")

    # ------------------------------------------------- no-regression proofs
    torch.manual_seed(3)
    a = PyramidalCortex(NUM_LAYERS, PARENT_HIDDEN, _dense_cortex_cfg())
    torch.manual_seed(3)
    b = PyramidalCortex(NUM_LAYERS, PARENT_HIDDEN, _dense_cortex_cfg())
    x = torch.randn(2, 5, PARENT_HIDDEN)
    # One state container per pass, reused across that pass's boundaries: that
    # is the real calling convention (``HAGI._run_blocks`` allocates one per
    # pass) and it also proves the container, not the module, carries the
    # summaries.
    sa, sb, sr = a.start_sequence(), b.start_sequence(), a.start_sequence()
    ha = a.apply_boundary(x, 0, sa)
    ca = a.apply_boundary(ha, 1, sa)
    hb = b.apply_boundary(x, 0, sb)
    cb = b.apply_boundary(hb, 1, sb)
    hr = reference_dense_boundary(a, x, 0, sr)
    cr = reference_dense_boundary(a, hr, 1, sr)
    print("\n[2] dense + fixed gate: no-regression proofs")
    print(f"    two fresh dense cortices identical:      {torch.equal(ca, cb)}")
    print(f"    shipped path == documented formula:      {torch.equal(ca, cr)}")
    print(f"    level-1 reference == shipped:            {torch.equal(hr, ha)}")
    keys = sorted(k for k in a.state_dict())
    legacy = [k for k in keys if k.startswith(("down.", "up.", "links."))]
    print(f"    default state_dict schema unchanged:     {keys == legacy}  {legacy}")
    if not (torch.equal(ca, cb) and torch.equal(ca, cr) and keys == legacy):
        failures.append("dense+fixed regression")
    else:
        print("    PASS -- the flat path computes bit-identical numbers")

    print("\n" + "=" * 74)
    if failures:
        print(f"RESULT: FAIL -- {failures}")
        return 1
    print("RESULT: PASS -- root preserves the invariant, dense breaks it, flat path intact")
    print("=" * 74)
    return 0


def _dense_cortex_cfg():
    """A dense+fixed cortex config -- only ever built by this check."""
    cfg = parent_config(None)
    cx = cfg.model.cortex
    cx.enabled = True
    cx.mode = "dense"
    cx.num_levels = 2
    cx.rank = RANK
    cx.link_strides = (1,)
    cx.residual_scale = 0.1
    return cx


if __name__ == "__main__":
    raise SystemExit(main())
