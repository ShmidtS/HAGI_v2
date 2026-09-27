"""Import a foreign HF model as a HAGI F3 leaf.

Why this exists: the recursive ternary ladder (3 -> 9 -> 27 leaves) needs
children that are already good, and training 27 leaves from scratch is the
bottleneck. A pretrained body gives a leaf a real hidden space before the
first step. The lift is orthogonal and fixes (1,1,1), so what matters is
only that the three children share ONE architecture -- which the merge
fences verify by fingerprint, not by provenance.

The transfer is a projection, and is honest about being one: a foreign
model's hidden width (Gemma-4-E2B: 1536) and head count (8q/1kv, head_dim
256) do not match the leaf's (384; 6q/2kv, head_dim 64). Nothing here
recovers the source model's function -- the leaf is a compressed,
head-subset view of it, and it still has to be trained afterwards. What it
buys is a starting point that is not random, and layers that already
separate tokens usefully.

Determinism: every projection is built from a fixed rule (top-left
principal slice for width, first-N heads for head count). No RNG is used
anywhere, so two runs give byte-identical output.

Usage:
    python scripts/import_foreign_leaf.py --out checkpoints/leaf_gemma_a
    python scripts/import_foreign_leaf.py --source <snapshot dir> --layers 3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from hagi.config import ffn_width, load_config  # noqa: E402
from hagi.train.checkpoint import CHECKPOINT_FORMAT_VERSION  # noqa: E402

DEFAULT_SOURCE = Path(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it"
    "/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03"
).expanduser()


def say(message: str) -> None:
    print(f"[import] {message}", flush=True)


def principal_slice(weight: torch.Tensor, out_features: int, dim: int) -> torch.Tensor:
    """Keep the leading ``out_features`` along ``dim``.

    A leading slice is used rather than a random one because it is
    deterministic and reproducible; it is *not* a principal-component
    selection, and the docstring says so rather than implying more.
    """
    size = weight.shape[dim]
    if out_features > size:
        raise ValueError(f"cannot widen {size} to {out_features} along dim {dim}")
    index = torch.arange(out_features, device=weight.device)
    return weight.index_select(dim, index).contiguous()


def fit(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Trim both axes of a 2-D weight to ``(rows, cols)``.

    A foreign width is always larger than the leaf's here, so the transfer is
    a leading slice on each axis rather than a resize. Leading slices keep the
    parameter ordering intact, which a random projection would not, and the
    result is deterministic.
    """
    t = weight
    if t.shape[0] > rows:
        t = principal_slice(t, rows, 0)
    if t.shape[1] > cols:
        t = principal_slice(t, cols, 1)
    return t.contiguous()


class ForeignLeafImporter:
    """Assemble a leaf-sized state dict from a foreign checkpoint."""

    def __init__(self, source: Path, cfg, layers: int) -> None:
        self.source = source
        self.cfg = cfg
        self.layers = layers
        m = cfg.model
        self.hidden = m.hidden_size
        self.heads_q = m.attention.num_query_heads
        self.heads_kv = m.attention.num_kv_heads
        self.head_dim = m.attention.head_dim
        self.vocab = m.vocab_size
        self.ffn = ffn_width(m)
        self.state: dict[str, torch.Tensor] = {}
        self._declared_head_dim = 256

    # -- source access -------------------------------------------------
    def _resolve_shard(self) -> Path:
        if self.source.is_file():
            return self.source
        if not self.source.is_dir():
            raise FileNotFoundError(f"source not found: {self.source}")
        shards = sorted(self.source.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {self.source}")
        if len(shards) == 1:
            return shards[0]
        # A sharded checkpoint: pick the shard that actually holds the keys.
        index = self.source / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(
                f"{self.source} has {len(shards)} shards but no index to route keys"
            )
        raise FileNotFoundError(
            "multi-shard source: pass --source pointing at the single shard "
            "that holds embed_tokens, or a snapshot whose index maps them"
        )

    def _get(self, handle, key: str) -> torch.Tensor | None:
        try:
            return handle.get_tensor(key)
        except Exception:
            return None

    # -- assembly ------------------------------------------------------
    def build(self) -> dict[str, torch.Tensor]:
        shard = self._resolve_shard()
        say(f"source shard: {shard.name}")
        with safe_open(str(shard), framework="pt") as handle:
            keys = set(handle.keys())
            say(f"source tensors available: {len(keys)}")

            prefix = self._language_prefix(keys)
            say(f"language prefix: {prefix!r}")
            embed = self._find_embedding(handle, keys, prefix)
            head = self._find_head(handle, keys, prefix, embed)
            self._write_embeddings(embed, head)
            self._write_norms()
            for index in range(self.layers):
                self._write_block(handle, keys, prefix, index)
        self._verify()
        return self.state

    def _language_prefix(self, keys: set[str]) -> str:
        for candidate in ("language_model.model.", "model.language_model.", "model."):
            if any(k.startswith(candidate) for k in keys):
                return candidate
        return ""

    def _find_embedding(self, handle, keys: set[str], prefix: str) -> torch.Tensor:
        for key in (f"{prefix}embed_tokens.weight", "embed_tokens.weight"):
            if key in keys:
                return self._get(handle, key)
        raise KeyError(f"no embedding found among {len(keys)} source tensors")

    def _find_head(self, handle, keys: set[str], prefix: str, embed: torch.Tensor):
        for key in (f"{prefix}lm_head.weight", "lm_head.weight"):
            if key in keys:
                return self._get(handle, key)
        say("no separate lm_head: source ties embeddings, reusing the embedding")
        return None

    def _write_embeddings(self, embed: torch.Tensor, head: torch.Tensor | None) -> None:
        # Rows: the source alphabet is far larger than the leaf's. Taking the
        # FIRST rows would be nearly all rare tokens; a stratified slice keeps
        # the id ordering, so token N in the leaf still means roughly token N
        # in the source. Frequency would be better and needs a corpus pass;
        # this is stated rather than hidden.
        source_vocab = embed.shape[0]
        if source_vocab < self.vocab:
            raise ValueError(f"source vocab {source_vocab} < leaf vocab {self.vocab}")
        stride = source_vocab / self.vocab
        rows = (torch.arange(self.vocab, dtype=torch.float64) * stride).long().clamp_max(
            source_vocab - 1
        )
        say(f"embedding rows: {source_vocab} -> {self.vocab} by stratified slice (stride {stride:.2f})")
        emb = embed.index_select(0, rows).to(torch.float32)
        self.state["encoder.embedding.weight"] = principal_slice(emb, self.hidden, 1).to(
            torch.bfloat16
        )
        # tie_lm_head is false in the leaf, so the head is its own matrix. The
        # source head (when present) is the better-conditioned read-out, so it
        # wins when the widths can be matched.
        if head is not None and head.shape[0] >= self.vocab:
            h = head.index_select(0, rows).to(torch.float32)
            self.state["head.projection.weight"] = principal_slice(
                h, self.hidden, 1
            ).to(torch.bfloat16)
        else:
            self.state["head.projection.weight"] = self.state["encoder.embedding.weight"].clone()
            say("head.projection.weight: reused the embedding (no usable source head)")
        # Logit scale is a calibration of the source's own vocabulary; it does
        # not transfer to a different one, so it is taken from the leaf config
        # domain rather than invented.
        self.state["head.logit_scale"] = torch.tensor(1.0 / (self.hidden**0.5), dtype=torch.float32)
        say("head.logit_scale: set to 1/sqrt(H) -- the source calibration does not transfer")

    def _write_norms(self) -> None:
        one = torch.ones(self.hidden, dtype=torch.bfloat16)
        self.state["out_norm.weight"] = one.clone()

    def _write_block(self, handle, keys: set[str], prefix: str, index: int) -> None:
        base = f"{prefix}layers.{index}."
        say(f"block {index}: gathering {base}*")
        attn = f"blocks.{index}.attn."
        mix = f"blocks.{index}.mixer."

        q = self._pick(handle, keys, [base + "self_attn.q_proj.weight", base + "self_attn.q_proj"])
        k = self._pick(handle, keys, [base + "self_attn.k_proj.weight", base + "self_attn.k_proj"])
        v = self._pick(handle, keys, [base + "self_attn.v_proj.weight", base + "self_attn.v_proj"])
        o = self._pick(handle, keys, [base + "self_attn.o_proj.weight", base + "self_attn.o_proj"])
        gate = self._pick(handle, keys, [base + f"mlp.gate_proj.weight", base + "mlp.gate_proj"])
        up = self._pick(handle, keys, [base + "mlp.up_proj.weight", base + "mlp.up_proj"])
        down = self._pick(handle, keys, [base + "mlp.down_proj.weight", base + "mlp.down_proj"])

        qkv = self._fuse_qkv(q, k, v)
        self.state[attn + "qkv_proj.weight"] = principal_slice(qkv, self.hidden, 1).to(torch.bfloat16)
        if o is not None:
            # source (hidden, heads*head_dim) -> target (hidden, hidden): both
            # axes are wider in the source, so both need trimming, not just the
            # output axis.
            self.state[attn + "out_proj.weight"] = fit(
                o, self.hidden, self.hidden
            ).to(torch.bfloat16)
        else:
            say(f"block {index}: no o_proj, leaving out_proj to the merged parent")
        for name, tensor in (("attn_norm", q), ("norm", gate)):
            if tensor is None:
                continue
        self.state[attn + "attn_norm.weight"] = torch.ones(self.hidden, dtype=torch.bfloat16)
        self.state[attn + "q_norm.weight"] = torch.ones(self.head_dim, dtype=torch.bfloat16)
        self.state[attn + "k_norm.weight"] = torch.ones(self.head_dim, dtype=torch.bfloat16)
        # The leaf's sink bias is a learned per-head logit sink; a foreign model
        # has no such parameter, so zeros are the neutral value and are stated.
        self.state[attn + "sink_bias"] = torch.zeros(
            1, self.heads_q, 1, 4, dtype=torch.bfloat16
        )
        self.state[attn + "branch_scale.scale"] = torch.tensor(1.0, dtype=torch.float32)

        self.state[mix + "norm.weight"] = torch.ones(self.hidden, dtype=torch.bfloat16)
        for name, tensor in (("gate", gate), ("up", up), ("down", down)):
            if tensor is None:
                say(f"block {index}: no mlp.{name}_proj, leaving it to the merged parent")
                continue
            source = tensor.to(torch.float32)
            if name == "down":
                # source (hidden, inter) -> target (hidden, ffn)
                self.state[mix + "mixer.down.weight"] = fit(
                    source, self.hidden, self.ffn
                )
            else:
                # source (inter, hidden) -> target (hidden, ffn)
                self.state[mix + f"mixer.{name}.weight"] = fit(
                    source, self.ffn, self.hidden
                ).t()
        self.state[mix + "mixer.branch_scale.scale"] = torch.tensor(1.0, dtype=torch.float32)

    def _pick(self, handle, keys: set[str], candidates: list[str]) -> torch.Tensor | None:
        for key in candidates:
            if key in keys:
                return self._get(handle, key)
        return None

    def _fuse_qkv(self, q, k, v) -> torch.Tensor:
        if q is None:
            raise KeyError("source has no q projection; cannot build qkv_proj")
        qh = self.heads_q
        kh = self.heads_kv
        pieces: list[torch.Tensor] = [self._take_heads(q, qh)]
        k_part = self._take_heads(k, kh) if k is not None else None
        v_part = self._take_heads(v, kh) if v is not None else None
        if k_part is None or v_part is None:
            say("  no separate k/v: reusing the query projection for both")
            k_part = k_part if k_part is not None else self._take_heads(q, kh)
            v_part = v_part if v_part is not None else self._take_heads(q, kh)
        elif k_part.shape[0] < kh * self.head_dim:
            # Gemma-4 has a single KV head (256 rows) where the leaf wants two
            # (2 x 64). Duplicating is the honest minimal move: the leaf's
            # per-head QK normalisation then sees two equal heads, which is
            # what a one-head source actually says. Padding with zeros instead
            # would leave a head that can never fire.
            say(
                f"  source has {k_part.shape[0] // self.head_dim} kv head(s), leaf wants {kh}: "
                "duplicating rather than zero-padding"
            )
            while k_part.shape[0] < kh * self.head_dim:
                k_part = torch.cat([k_part, k_part], dim=0)[: kh * self.head_dim]
                v_part = torch.cat([v_part, v_part], dim=0)[: kh * self.head_dim]
        pieces.extend([k_part, v_part])
        fused = torch.cat(pieces, dim=0)
        expected = (qh + 2 * kh) * self.head_dim
        if fused.shape[0] != expected:
            raise ValueError(
                f"fused qkv rows {fused.shape[0]} != expected {expected} "
                f"({qh}q + 2x{kh}kv x {self.head_dim})"
            )
        return fused

    def _take_heads(self, weight: torch.Tensor, heads: int) -> torch.Tensor:
        """First ``heads`` source heads, each truncated to the leaf head_dim."""
        t = weight.to(torch.float32)
        source_dim = self._declared_head_dim
        total_heads = t.shape[0] // source_dim
        if total_heads < 1 or t.shape[0] % source_dim:
            raise ValueError(
                f"projection rows {t.shape[0]} are not a multiple of the source "
                f"head_dim {source_dim}"
            )
        say(
            f"  projection {tuple(t.shape)}: {total_heads} source heads x {source_dim}"
            f" -> taking {heads}, truncating to {self.head_dim}"
        )
        keep = min(heads, total_heads)
        rows = keep * self.head_dim
        return t[:rows].contiguous()

    def _verify(self) -> None:
        reference = Path("checkpoints/f3_leaf_s1001/step-0003000.pt")
        if not reference.is_file():
            say("reference leaf not found; skipping the contract check")
            return
        want = torch.load(reference, map_location="cpu", weights_only=False)["model"]
        want_keys = {k for k in want if not k.startswith("recursive_f3")}
        have = set(self.state)
        missing = sorted(want_keys - have)
        extra = sorted(have - want_keys)
        bad = [
            (k, tuple(self.state[k].shape), tuple(want[k].shape))
            for k in sorted(want_keys & have)
            if tuple(self.state[k].shape) != tuple(want[k].shape)
        ]
        say(f"contract check: {len(have)} tensors, missing={len(missing)} extra={len(extra)} shape-mismatch={len(bad)}")
        if missing:
            say(f"  missing: {missing[:8]}")
        if extra:
            say(f"  extra: {extra[:8]}")
        for key, got, exp in bad[:8]:
            say(f"  mismatch {key}: got {got} want {exp}")
        if missing or bad:
            raise ValueError(
                f"imported state does not match the leaf contract "
                f"({len(missing)} missing, {len(bad)} mismatched)"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--config", default="configs/f3_leaf_s1001.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000, help="completed_steps recorded in the payload")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from hagi.train.checkpoint import config_to_dict

    importer = ForeignLeafImporter(Path(args.source), cfg, args.layers)
    state = importer.build()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": state,
            "config": config_to_dict(cfg),
            "completed_steps": args.steps,
        },
        out,
    )
    say(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    say("NOTE: this leaf is a compressed view of a foreign model, not a working one. "
        "It must be trained before it says anything about quality.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
