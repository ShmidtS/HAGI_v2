"""MSA — Memory Sparse Attention with pure-tensor ring buffer + MLA.

V7: DFE (read) + HARQ buffer (write) for channel memory in turbo loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.codec_contracts import MSADecodeConfig


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
        n = keys.shape[0]
        ptr = int(self.write_ptr.item())
        idx = torch.arange(n, device=keys.device)
        indices = (idx + ptr) % self.max_slots
        # Routing keys stay detached: topk indices are non-differentiable anyway,
        # so write-side keys only serve address lookup. route_proj learns via the
        # read-side query path + load-balance loss.
        self.slot_keys = torch.index_copy(
            self.slot_keys, 0, indices, keys.detach().to(self.slot_keys.dtype)
        )
        # KV values keep grad_fn: read gathers top-k and feeds mla_up_k/v, so
        # gradient flows back into mla_compress through the slots actually read.
        # Top-k (<=6) bounds the BPTT fan-in per query, not the full ring buffer.
        self.slot_kv = torch.index_copy(self.slot_kv, 0, indices, kv_compressed.to(self.slot_kv.dtype))
        new_ptr = (ptr + n) % self.max_slots
        self.write_ptr.fill_(new_ptr)
        self.num_written.fill_(min(int(self.num_written.item()) + n, self.max_slots))

    def read_topk(self, query: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_valid = int(self.num_written.item())
        if num_valid == 0:
            dummy_idx = torch.zeros(query.shape[0], top_k, dtype=torch.long, device=query.device)
            dummy_scores = torch.zeros(query.shape[0], top_k, dtype=query.dtype, device=query.device)
            return dummy_idx, dummy_scores
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.slot_keys[:num_valid], dim=-1)
        scores = q_norm @ k_norm.T
        k = min(top_k, num_valid)
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        if k < top_k:
            pad_idx = torch.zeros(query.shape[0], top_k - k, dtype=torch.long, device=query.device)
            pad_scores = torch.full((query.shape[0], top_k - k), -1e9, dtype=query.dtype, device=query.device)
            top_indices = torch.cat([top_indices, pad_idx], dim=-1)
            top_scores = torch.cat([top_scores, pad_scores], dim=-1)
        return top_indices, top_scores

    def gather_kv(self, indices: torch.Tensor) -> torch.Tensor:
        return self.slot_kv[indices]

    def clear(self) -> None:
        # Reassign (not in-place) so non-leaf slot_kv from a prior forward
        # graph doesn't raise "a leaf Variable that requires grad is used in
        # an in-place operation" / "modified by an inplace operation".
        self.slot_keys = self.slot_keys.detach().zero_()
        self.slot_kv = self.slot_kv.detach().zero_()
        self.write_ptr.zero_()
        self.num_written.zero_()


class MSAModule(nn.Module):
    def __init__(self, cfg: MSADecodeConfig, hidden_size: int = 576):
        super().__init__()
        self.cfg = cfg
        self.route_proj = nn.Linear(hidden_size, cfg.routing_key_dim, bias=False)
        self.mla_compress = nn.Linear(hidden_size, cfg.mla_compress_dim, bias=False)
        self.mla_up_k = nn.Linear(cfg.mla_compress_dim, cfg.mla_up_dim, bias=False)
        self.mla_up_v = nn.Linear(cfg.mla_compress_dim, cfg.mla_up_dim, bias=False)
        self.q_proj = nn.Linear(hidden_size, cfg.mla_up_dim, bias=False)
        self.o_proj = nn.Linear(cfg.mla_up_dim, hidden_size, bias=False)
        self.registry = TensorSlotRegistry(cfg.max_slots, cfg.routing_key_dim, cfg.mla_compress_dim)
        self._n_kv_heads = cfg.n_kv_heads
        self._head_dim = cfg.head_dim
        self._default_chunk = cfg.slot_chunk_size

    def write(self, h: torch.Tensor) -> None:
        B, T, _ = h.shape
        chunk = self._default_chunk
        flat_h = h.reshape(B * T, -1)
        n = flat_h.shape[0]
        n_slots = n // chunk
        remainder = n % chunk
        chunks = []
        if n_slots > 0:
            chunks.append(flat_h[: n_slots * chunk].view(n_slots, chunk, -1).mean(dim=1))
        if remainder > 0:
            chunks.append(flat_h[n_slots * chunk :].mean(dim=0, keepdim=True))
        if not chunks:
            return
        chunked = torch.cat(chunks, dim=0)
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
        num_valid = int(self.registry.num_written.item())
        if num_valid == 0:
            return scores.new_zeros(())
        counts = torch.bincount(indices.reshape(-1), minlength=self.cfg.max_slots).float()
        counts = counts[:num_valid]
        total = counts.sum()
        if total == 0:
            return scores.new_zeros(())
        f = counts / total
        P_per_slot = scores.softmax(dim=-1).mean(dim=0)[:num_valid]
        if P_per_slot.shape[0] != f.shape[0]:
            n = min(f.shape[0], P_per_slot.shape[0])
            return (f[:n] * P_per_slot[:n]).sum()
        return (f * P_per_slot).sum()

    def clear(self) -> None:
        self.registry.clear()

    def serialize_feedback(self) -> torch.Tensor:
        return torch.cat(
            [
                self.registry.slot_keys.flatten(),
                self.registry.slot_kv.flatten(),
                self.registry.write_ptr,
                self.registry.num_written,
            ]
        ).detach()

    def restore_feedback(self, feedback: torch.Tensor) -> None:
        key_size = self.registry.slot_keys.numel()
        kv_size = self.registry.slot_kv.numel()
        if feedback.numel() != key_size + kv_size + 2:
            raise ValueError("Invalid MSA feedback state")
        flat = feedback.to(device=self.registry.slot_keys.device, dtype=self.registry.slot_keys.dtype)
        self.registry.slot_keys = flat[:key_size].view_as(self.registry.slot_keys).clone()
        self.registry.slot_kv = flat[key_size : key_size + kv_size].view_as(self.registry.slot_kv).clone()
        self.registry.write_ptr.copy_(flat[-2:-1].to(self.registry.write_ptr.dtype))
        self.registry.num_written.copy_(flat[-1:].to(self.registry.num_written.dtype))
