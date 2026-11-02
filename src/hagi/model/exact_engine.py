"""Exact-engine construction kit — compiled logic inside HAGI weights.

Adapted (Apache-2.0) from voidash/qwenbend ``bendvm/engine_circuit.py``
(the Mlp AND-gadget, one-hot nibble word encoding, energy ballast and
latch-head discipline) and divagr18/qwenputer's DeltaNet-state-as-RAM
technique (MIT; both cited upstream by qwenbend itself). Also informed
by vixhal-baraiya/qwenGPU (MIT) — GDN one-hot rows as registers/memory.

The three ported primitives, proven in the sources bit-for-bit against
reference interpreters:

1. ``ExactAND`` — a SwiGLU unit that fires (silu(g)*up == 1.0 exactly)
   iff all its features are 1.0. Gadget (qwenbend Mlp.unit):
       gate_logit = 256 * (sum(features) - n) + 128,   up = 1/128
   silu(128) = 128 exactly in fp32/bf16 (sigmoid(128) == 1.0);
   silu(x) <= -128*sigmoid(-128) ~ 0 exactly when x <= -128.
   A feature is one residual dim constrained to {0, 1}, or a one-hot
   GROUP of dims (at most one is 1) — an OR over the group.
   Negative features (must be 0) count as (1 - f).
2. ``NibbleWord`` — a 32-bit word as 8 one-hot nibbles (128 dims at
   1.0); encode/decode with exact one-hot validation (qwenbend
   decode_word: any non-one-hot nibble -> None, no silent garbage).
3. ``LatchHead`` — a GDN head configured as an n-row register
   (qwenputter technique): key row r hot at write, query row r hot
   always, beta opens only on the write token kind; the head's state
   holds one 128-dim word per row until overwritten. With our
   GatedDeltaNet (ram mode, g=1 beta=1, one-hot keys) this is exactly
   the column-write contract the Lean gdnUpdate/gdnBridge formalizes.

Everything here is construction-only (no learning): weights are written
by these builders the way a compiler emits them. Use_ternary must be
OFF for exact layers (ternary quantization destroys exact values).
"""
from __future__ import annotations

import torch


class ExactAND:
    """Allocate SwiGLU units of one layer as exact AND gates (qwenbend
    ``Mlp``), against HAGI's SwiGLU layout (gate/up/down Linear modules,
    ``down(silu(gate(x)) * up(x))``, hagi.model.ffn.SwiGLU).

    Feature spec (same semantics as upstream):
    - an int: a single residual dim that must be 1.0;
    - a list[int]: a one-hot GROUP — fires if ANY dim in the group is 1.0
      (an OR; at most one is 1 by the encoder's discipline);
    - pair it through ``negs`` for features that must be 0 (each
      contributes its complement (1 - f) to the gate sum).
    """

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        one_dim: int,
    ) -> None:
        """one_dim: index of the constant-1.0 residual dim every engine
        token carries (qwenbend ``ONE`` field) — the gadget's anchor."""
        if gate_weight.shape != up_weight.shape:
            raise ValueError("gate/up weights must have identical shapes")
        if down_weight.shape[1] != gate_weight.shape[0]:
            raise ValueError("down weight must map intermediate -> hidden")
        self.gate = gate_weight
        self.up = up_weight
        self.down = down_weight
        self.one = int(one_dim)
        self.n_units = 0

    def unit(
        self,
        feats: list[int | list[int]],
        outs: dict[int, float],
        negs: list[int | list[int]] = (),
        up: dict[int, float] | None = None,
    ) -> None:
        """Add one AND unit: fires iff all ``feats`` are 1 and all ``negs``
        are 0; writes ``outs`` {hidden dim: weight} through ``down``.
        ``up`` overrides the constant 1/128 pass-through (a gated scaling:
        output = 128 * w exactly when fired — qwenbend's ``up=`` form).
        """
        if not isinstance(outs, dict):
            outs = {d: 1.0 for d in outs}
        i = self.n_units
        self.n_units += 1
        if i >= self.gate.shape[0]:
            raise ValueError("out of MLP units")
        # qwenputer q35 ``t_and`` half-unit margins (robust form): the gate is
        # 512*(sum - n_pos) + 256 over must-be-1 features (+512 each) and
        # -512 over must-be-0 features (each complement (1-f) shifts the sum
        # the same way). All-fire -> +256 (silu == 256, up 1/256 -> exactly 1);
        # ANY single violation shifts the gate by 512 -> <= -256, and
        # silu(<= -256) == -0 EXACTLY even when literals carry fp32 noise.
        # This doubles the noise slack of the earlier 256/128 form.
        for f in feats:
            dims = [f] if isinstance(f, int) else list(f)
            self.gate[i, dims] += 512.0
        for f in negs:
            dims = [f] if isinstance(f, int) else list(f)
            self.gate[i, dims] -= 512.0
        self.gate[i, self.one] += 256.0 - 512.0 * len(feats)
        if up is None:
            self.up[i, self.one] = 1.0 / 256.0
        else:
            for dim, w in up.items():
                self.up[i, dim] = w
        for dim, v in outs.items():
            self.down[dim, i] += v


class NibbleWord:
    """32-bit word as 8 one-hot nibbles (128 dims, qwenbend layout)."""

    WIDE = 8
    NIB = 16
    DIMS = WIDE * NIB  # 128

    @staticmethod
    def encode(value: int) -> list[int]:
        """Value -> hot dim indices (one per nibble)."""
        if not 0 <= value < 2**32:
            raise ValueError(f"value out of u32 range: {value}")
        hot = []
        for j in range(NibbleWord.WIDE):
            hot.append(16 * j + ((value >> (4 * j)) & 15))
        return hot

    @staticmethod
    def nibbles(value: int, n: int) -> list[int]:
        """First n nibbles, least significant first (qwenbend nibbles())."""
        return [(value >> (4 * j)) & 15 for j in range(n)]

    @staticmethod
    def decode(x: torch.Tensor, dim_base: int = 0) -> int | None:
        """Read the word at dims [dim_base, dim_base+128): None unless every
        nibble is exactly one-hot 1.0 (qwenbend decode_word — no garbage)."""
        if x.shape[-1] < dim_base + NibbleWord.DIMS:
            raise ValueError("tensor too small for a NibbleWord")
        v = 0
        for j in range(NibbleWord.WIDE):
            nib = x[dim_base + 16 * j : dim_base + 16 * j + 16].tolist()
            if sorted(nib) != [0.0] * 15 + [1.0]:
                return None
            v += int(nib.index(1.0)) << (4 * j)
        return v


def latch_head_config(
    n_rows: int,
    key_dims: list[int],
    query_dims: list[int],
    value_dims: list[int],
    write_dim: int,
) -> dict[str, torch.Tensor]:
    """Build the four projections of a GDN head acting as an n-row register
    (qwenputer/qwenbend latch-head technique, adapted to our GatedDeltaNet
    projections k/v/gate/beta from in_proj).

    Contract (Lean gdnUpdate, ram mode): on a token whose ``write_dim`` is
    hot, the head writes the value word into the row selected by the key
    one-hot; every other token is a pure read (beta 0). Returns the k, v,
    gate, beta weight tensors shaped for one head of ``n_rows`` channels.

    - key: [n_rows] rows, hot at (key_dims[r], write_dim) — AND via two
      contributions multiplied later by beta's gating: we give key_dim r a
      plain 1.0 from key_dims[r] and let beta do the write-gating.
    - query: query_dims[r] -> row r (read anywhere).
    - value: value word dims -> value channels.
    - gate/beta: beta hot exactly on write_dim.
    """
    if not (len(key_dims) == len(query_dims) == n_rows):
        raise ValueError("key/query dim lists must have n_rows entries")
    k = torch.zeros(n_rows, max(key_dims) + 1)
    for r, d in enumerate(key_dims):
        k[r, d] = 1.0
    q = torch.zeros(n_rows, max(query_dims) + 1)
    for r, d in enumerate(query_dims):
        q[r, d] = 1.0
    v = torch.zeros(max(value_dims) + 1, n_rows)
    for d in value_dims:
        # broadcast the word into every row channel; the delta rule writes
        # only the keyed row
        v[d, :] = 1.0
    beta = torch.zeros(max([write_dim]) + 1)
    beta[write_dim] = 1.0
    return {"k": k, "q": q, "v": v, "beta": beta}


def energy_ballast(embedding_row: torch.Tensor, target_energy: float) -> None:
    """Top a token embedding's squared norm up to ``target_energy`` with
    powers of two (qwenbend ballast(): base-4 digits of the remainder on
    spare dims) so RMSNorm energy is identical across engine tokens."""
    rest = target_energy - float((embedding_row**2).sum())
    if rest < 0:
        raise ValueError("row already exceeds the target energy")
    d, j = 0, 0
    while rest > 1e-12:
        take = rest % 4
        for _ in range(int(take)):
            if d >= embedding_row.shape[0]:
                raise ValueError("out of ballast dims")
            embedding_row[d] += 2.0**j
            d += 1
        rest = int(rest) // 4
        j += 1
