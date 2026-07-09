"""MSA — Memory Sparse Attention with pure-tensor ring buffer + MLA.

V7: DFE (read) + HARQ buffer (write) for channel memory in turbo loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.config import MSAConfig
from hagi_v4.model.norms import RMSNorm


class TensorSlotRegistry(nn.Module):
    slot_keys: torch.Tensor
    slot_kv: torch.Tensor
    write_ptr: torch.Tensor

    def __init__(self, max_slots: int, key_dim: int, compress_dim: int):
        super().__init__()
        self.max_slots = max_slots
        self.key_dim = key_dim
        self.compress_dim = compress_dim
        self.register_buffer("slot_keys", torch.zeros(max_slots, key_dim), persistent=False)
        self.register_buffer("slot_kv", torch.zeros(max_slots, compress_dim), persistent=False)
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("num_written", torch.zeros(1, dtype=torch.long), persistent=False)

    def write(self, keys: torch.Tensor, kv_compressed: torch.Tensor) -> None:
        """Write keys and compressed KV into the ring buffer.

        .detach() on keys/kv prevents gradients from flowing through the
        memory write path — slots are treated as fixed context, not as
        differentiable parameters of the current forward pass.
        """
        n = keys.shape[0]
        ptr = int(self.write_ptr.item())
        idx = torch.arange(n, device=keys.device)
        indices = (idx + ptr) % self.max_slots
        self.slot_keys.index_copy_(0, indices, keys.detach().to(self.slot_keys.dtype))
        self.slot_kv.index_copy_(0, indices, kv_compressed.detach().to(self.slot_kv.dtype))
        new_ptr = torch.tensor([(ptr + n) % self.max_slots], dtype=torch.long, device=keys.device)
        self.write_ptr.copy_(new_ptr)
        self.num_written.copy_(torch.tensor([min(ptr + n, self.max_slots)], dtype=torch.long, device=keys.device))

    def read_topk(self, query: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.slot_keys, dim=-1)
        scores = q_norm @ k_norm.T
        k = min(top_k, self.max_slots)
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        return top_indices, top_scores

    def gather_kv(self, indices: torch.Tensor) -> torch.Tensor:
        return self.slot_kv[indices]

    def clear(self) -> None:
        self.slot_keys.zero_()
        self.slot_kv.zero_()
        self.write_ptr.zero_()
        self.num_written.zero_()


class MSAModule(nn.Module):
    def __init__(self, cfg: MSAConfig, hidden_size: int = 576):
        super().__init__()
        self.cfg = cfg
        self.route_proj = nn.Linear(hidden_size, cfg.routing_key_dim, bias=False)
        self.mla_compress = nn.Linear(hidden_size, cfg.mla_compress_dim, bias=False)
        self.mla_up_k = nn.Linear(cfg.mla_compress_dim, cfg.mla_up_dim, bias=False)
        self.mla_up_v = nn.Linear(cfg.mla_compress_dim, cfg.mla_up_dim, bias=False)
        self.q_proj = nn.Linear(hidden_size, cfg.mla_up_dim, bias=False)
        self.o_proj = nn.Linear(cfg.mla_up_dim, hidden_size, bias=False)
        self.attn_norm = RMSNorm(hidden_size)
        self.registry = TensorSlotRegistry(cfg.max_slots, cfg.routing_key_dim, cfg.mla_compress_dim)
        self._n_kv_heads = cfg.n_kv_heads
        self._head_dim = cfg.head_dim
        self._scalar_slice = (0, cfg.grade_dims[0])
        self._bivector_slice = (sum(cfg.grade_dims[:2]), sum(cfg.grade_dims[:2]) + cfg.grade_dims[2])
        self._adaptive_chunk = cfg.use_adaptive_chunk_size
        self._chunk_low = cfg.chunk_size_low_entropy
        self._chunk_high = cfg.chunk_size_high_entropy
        self._default_chunk = cfg.slot_chunk_size

    def _select_chunk_size(self, h: torch.Tensor) -> int:
        return self._default_chunk

    def write(self, h: torch.Tensor) -> None:
        B, T, _ = h.shape
        chunk = self._select_chunk_size(h)
        flat_h = h.reshape(B * T, -1)
        n = flat_h.shape[0]
        n_slots = n // chunk
        if n_slots == 0:
            return
        usable = flat_h[: n_slots * chunk]
        chunked = usable.view(n_slots, chunk, -1).mean(dim=1)
        keys = self.route_proj(chunked)
        kv = self.mla_compress(chunked)
        self.registry.write(keys, kv)

    def read(self, h: torch.Tensor, top_k: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = h.shape
        flat_h = h.reshape(B * T, -1)
        query = self.route_proj(flat_h)
        top_indices, top_scores = self.registry.read_topk(query, top_k)
        kv_compressed = self.registry.gather_kv(top_indices)
        k = self.mla_up_k(kv_compressed).view(B * T, top_k, self._n_kv_heads, self._head_dim)
        v = self.mla_up_v(kv_compressed).view(B * T, top_k, self._n_kv_heads, self._head_dim)
        h_proj = self.q_proj(flat_h).view(B * T, 1, self._n_kv_heads, self._head_dim)
        q_4d = h_proj.permute(0, 2, 1, 3)
        k_4d = k.permute(0, 2, 1, 3)
        v_4d = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B * T, -1)
        msa_out = self.o_proj(attn_out)
        msa_out = msa_out.view(B, T, -1)
        lb = self._load_balance_loss(top_indices, top_scores)
        return msa_out, lb

    def _load_balance_loss(self, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss encouraging uniform slot usage.

        f = slot visit frequency, P = mean routing probability (softmaxed scores
        averaged across queries, then across slots). The double mean on P
        first averages over queries (dim=0) yielding per-slot probabilities,
        then averages over slots to produce a scalar — this is intentional
        to match the Switch Transformer load-balancing formulation.
        """
        counts = torch.bincount(indices.reshape(-1), minlength=self.cfg.max_slots).float()
        total = counts.sum()
        f = counts / total if total > 0 else counts
        P = (scores.softmax(dim=-1).mean(dim=0)).mean()
        return self.cfg.load_balance_weight * (f * P).sum()

    def clear(self) -> None:
        self.registry.clear()
