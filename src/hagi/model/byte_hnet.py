"""Byte-level H-Net path — raw UTF-8 bytes in, next-byte logits out.

Reference-first:
- Cartesia "From Tokens to Bytes" parts 1-2 (H-Net): two-stage hierarchical
  byte processing — a learned dynamic chunker compresses the byte stream into
  semantic chunks, a backbone runs at chunk rate, a dechunker expands back to
  byte resolution with a residual from the encoder stage. No BPE, no regex:
  the vocabulary is exactly 256 values, killing the tokenizer boundary problem
  (measured 1.5-1.9x downstream compute penalty of BPE disappears).
- Lean SSOT: primes ``BytePatch`` (patchCount = ceil(N/p), patch_attn_saving)
  for the fixed-rate ceiling; the dynamic boundary predictor is the runtime
  analog of the learned M in AdaptiveChunking (adaptive_le_static: a dynamic
  chunker can only win vs the fixed-rate bound).

Design (KISS, runtime-honest):
- ByteEncoder: embedding(256) + the existing SourceEncoder pulse-shaper path
  is reused (conv_kernel applies at byte rate, cheap).
- DynamicChunker: per-byte boundary logit p_t from a tiny causal GRU-lite
  (linear + sigmoid); boundary where p_t > threshold; straight-through
  estimator (STE) carries gradients through the binary boundary mask;
  min_chunk/max_chunk guards keep the compute bounded (never degenerate to
  1-byte chunks: adaptive_lt_of_strict requires p >= 2 to save anything).
  Fixed-rate fallback (chunk = p bytes) is the static comparison point.
- Dechunker: each chunk vector broadcasts over its byte span; a small MLP +
  the encoder's byte-level residual produce byte-rate hidden states, so the
  LM head still emits 256-way next-byte logits and the exact-CE calibration
  channel (V42 head) works unchanged.

The module owns only the byte<->chunk interface; the backbone (GDN/attention
blocks) is untouched — ChunkedHNET wraps any existing causal body.
"""
from __future__ import annotations

import torch
from torch import nn

from hagi.model.norms import RMSNorm


class DynamicChunker(nn.Module):
    """Learned boundary predictor with straight-through gradients.

    Args:
        hidden_size: byte-rate embedding width.
        min_chunk: smallest emitted chunk (>= 2 keeps compute savings legal).
        max_chunk: hard cap on chunk length (bounds worst-case compute).
        threshold: boundary firing threshold for the boundary logit.
    """

    def __init__(
        self,
        hidden_size: int,
        min_chunk: int = 2,
        max_chunk: int = 8,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if min_chunk < 2:
            # adaptive_lt_of_strict: p >= 2 is the precondition of any saving
            raise ValueError("min_chunk must be >= 2 (fixed-rate p=1 saves nothing)")
        if max_chunk < min_chunk:
            raise ValueError("max_chunk must be >= min_chunk")
        self.min_chunk = int(min_chunk)
        self.max_chunk = int(max_chunk)
        self.threshold = float(threshold)
        self.boundary_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.ones_(self.boundary_head.bias)  # start: fire at every opportunity

    def boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Hard boundary decisions [B, T] in {0,1} (1 = chunk ends here)."""
        logits = self.boundary_head(x).squeeze(-1)
        return (torch.sigmoid(logits) > self.threshold).to(x.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[list[slice]]]:
        """Pool byte-rate states into chunk states.

        Args:
            x: [B, T, H] byte-rate hidden states.

        Returns:
            (chunk_states [B, C, H] mean-pooled, boundary mask [B, T] with the
            STE gradient, per-batch chunk spans as slices of the byte axis).
        """
        B, T, H = x.shape
        raw = self.boundaries(x)  # [B, T] hard mask
        # STE: forward uses the hard decision, backward flows through sigmoid
        logits = self.boundary_head(x).squeeze(-1)
        soft = torch.sigmoid(logits)
        hard = raw + soft - soft.detach()

        chunks: list[list[slice]] = []
        pooled_rows = []
        mask_rows = []
        for b in range(B):
            rb = raw[b].detach().tolist()
            spans: list[slice] = []
            start = 0
            while start < T:
                # fire boundary at the first position >= min_chunk where the
                # hard mask says so, otherwise force one at max_chunk
                fire = start + self.min_chunk
                end = min(start + self.max_chunk, T)
                boundary_pos = end
                for tt in range(fire, end):
                    if rb[tt] == 1.0:
                        boundary_pos = tt + 1
                        break
                spans.append(slice(start, boundary_pos))
                start = boundary_pos
            chunks.append(spans)
            pooled = x[b].new_zeros(len(spans), H)
            for i, sp in enumerate(spans):
                pooled[i] = x[b, sp].mean(dim=0)
            pooled_rows.append(pooled)
            # chunk-rate STE mask: 1 at each chunk's last byte
            mask_row = torch.zeros(T, device=x.device, dtype=x.dtype)
            for sp in spans:
                mask_row[sp.stop - 1] = 1.0
            mask_rows.append(mask_row)
        C = max(len(s) for s in chunks)
        chunk_states = x.new_zeros(B, C, H)
        for b in range(B):
            chunk_states[b, : len(chunks[b])] = pooled_rows[b]
        ste_mask = torch.stack(mask_rows) * hard
        return chunk_states, ste_mask, chunks


class ByteHNet(nn.Module):
    """Two-stage byte path: byte encoder -> dynamic chunker -> (backbone by
    caller) -> dechunker back to byte rate.

    The backbone runs on chunk_states; call :meth:`dechunk` after it.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int = 256,
        min_chunk: int = 2,
        max_chunk: int = 8,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.byte_embed = nn.Embedding(self.vocab_size, hidden_size)
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)
        self.chunker = DynamicChunker(hidden_size, min_chunk=min_chunk, max_chunk=max_chunk)
        self.dechunk_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Small-random (NOT zero) init: a zero final layer kills the gradient
        # to the whole backbone at start (d loss/d chunk = W2^T @ 0 = 0) —
        # measured cold-start trap (test_target_architecture). At std 0.02 the
        # dechunk delta starts negligible-but-alive, like the encoder filter.
        nn.init.normal_(self.dechunk_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dechunk_mlp[-1].bias)
        self.out_norm = RMSNorm(hidden_size, eps=norm_eps)

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[list[slice]]]:
        """bytes [B, T] -> (chunk_states [B, C, H], ste_mask [B, T], spans).

        The STE mask carries the chunker gradient: add it to any scalar
        surrogate (e.g. ``(ste_mask * byte_logits_entropy).sum()``) or use
        :meth:`chunker_surrogate` after the backbone pass.
        """
        if input_ids.min() < 0 or input_ids.max() >= self.vocab_size:
            raise ValueError("input_ids out of byte range")
        byte_states = self.byte_embed(input_ids)
        chunk_states, ste_mask, spans = self.chunker(byte_states)
        return chunk_states, ste_mask, spans

    @staticmethod
    def chunker_surrogate(ste_mask: torch.Tensor) -> torch.Tensor:
        """Differentiable surrogate the trainer adds to the loss so the
        boundary predictor receives gradient: mean mask value (the STE path
        makes d(sum mask)/d(logits) well-defined). Minimizing it pushes
        toward FEWER boundaries (longer chunks) — pair it with the CE gain
        of the backbone to get the adaptive chunking trade-off.
        """
        return ste_mask.mean()

    def dechunk(
        self,
        chunk_states: torch.Tensor,
        spans: list[list[slice]],
        byte_states: torch.Tensor,
    ) -> torch.Tensor:
        """Expand chunk-rate backbone output to byte rate with encoder residual.

        chunk_states: [B, C, H] backbone output at chunk rate; byte_states:
        [B, T, H] the encoder's byte-rate states (residual path).
        Returns [B, T, H] byte-rate states ready for the LM head.
        """
        B, T, H = byte_states.shape
        out = byte_states.clone()
        for b in range(B):
            spans_b = spans[b]
            for i, sp in enumerate(spans_b):
                if i >= chunk_states.shape[1]:
                    break
                pooled = chunk_states[b, i].expand(sp.stop - sp.start, H)
                combined = torch.cat([pooled, byte_states[b, sp]], dim=-1)
                out[b, sp] = out[b, sp] + self.dechunk_mlp(combined)
        return self.out_norm(out)
