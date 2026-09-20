"""Minimal native HAGI head-LoRA for post-answer online adaptation.

Only the LM receiver is adapted in the MVP. Base HAGI weights remain frozen.
The adapter is factorized B @ A, with B initialized to zero so attaching it is
an exact no-op at initialization.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class HeadLoRA(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, rank: int = 4, alpha: float | None = None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scale = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(self.rank, hidden_size))
        self.B = nn.Parameter(torch.zeros(vocab_size, self.rank))
        nn.init.normal_(self.A, std=hidden_size ** -0.5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        z = F.linear(hidden, self.A)
        return F.linear(z, self.B) * self.scale

    @property
    def trainable_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def save(self, path: str | Path) -> None:
        payload = {
            "rank": self.rank,
            "alpha": self.alpha,
            "A": self.A.detach().cpu(),
            "B": self.B.detach().cpu(),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, hidden_size: int, vocab_size: int, device: torch.device):
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        adapter = cls(hidden_size, vocab_size, int(payload["rank"]), float(payload["alpha"]))
        adapter.A.data.copy_(payload["A"])
        adapter.B.data.copy_(payload["B"])
        return adapter.to(device)


class HeadLoRATrainer:
    """Small sampled-softmax update for prompt->continuation pseudo labels."""

    def __init__(self, *, lr: float = 5e-2, negatives: int = 128, weight_decay: float = 0.0):
        self.lr = lr
        self.negatives = max(1, negatives)
        self.weight_decay = weight_decay

    def attach(self, model: nn.Module, rank: int = 4, alpha: float | None = None) -> HeadLoRA:
        adapter = HeadLoRA(model.cfg.model.hidden_size, model.cfg.model.vocab_size, rank, alpha).to(
            next(model.parameters()).device
        )
        model.head.attach_lora(adapter)
        return adapter

    @torch.no_grad()
    def _sample_negatives(self, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
        return torch.randint(0, vocab_size, (self.negatives,), device=targets.device)

    def loss(self, model: nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adapter = getattr(model.head, "adaptive_lora", None)
        if adapter is None:
            raise RuntimeError("attach a HeadLoRA before training")
        head = model.head
        weight = head.weight
        neg = self._sample_negatives(targets, head.vocab_size)

        z = F.linear(hidden, adapter.A)
        target_base = (hidden * head.logit_scale.to(hidden.dtype)) @ weight.index_select(0, targets).t()
        target_base = target_base.diagonal()
        target_delta = F.linear(z, adapter.B.index_select(0, targets)) * adapter.scale
        target_logits = target_base + target_delta.to(target_base.dtype)

        neg_weight = weight.index_select(0, neg)
        neg_base = (hidden * head.logit_scale.to(hidden.dtype)) @ neg_weight.t()
        neg_delta = (z @ adapter.B.index_select(0, neg).t()) * adapter.scale
        neg_logits = neg_base + neg_delta.to(neg_base.dtype)

        if head.log_prior is not None:
            target_logits = target_logits + head.log_prior.index_select(0, targets).to(target_logits.dtype)
            neg_logits = neg_logits + head.log_prior.index_select(0, neg).to(neg_logits.dtype)

        # Target is class 0 for every row. Remove accidental target duplicates
        # from the negative bank.
        neg_logits = neg_logits.masked_fill(targets.unsqueeze(1).eq(neg), float("-inf"))
        logits = torch.cat([target_logits.unsqueeze(1), neg_logits], dim=1)
        return F.cross_entropy(logits.float(), torch.zeros(targets.shape[0], dtype=torch.long, device=targets.device))

    def step(self, model: nn.Module, input_ids: torch.Tensor, targets: torch.Tensor) -> float:
        adapter = getattr(model.head, "adaptive_lora", None)
        if adapter is None:
            raise RuntimeError("attach a HeadLoRA before training")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        adapter.A.requires_grad_(True)
        adapter.B.requires_grad_(True)

        with torch.no_grad():
            out = model(input_ids, return_logits=False)
        hidden = out.hidden.reshape(-1, out.hidden.shape[-1]).detach()
        targets = targets.reshape(-1)
        if hidden.shape[0] != targets.shape[0]:
            raise ValueError("input/target token count mismatch")

        opt = torch.optim.AdamW([adapter.A, adapter.B], lr=self.lr, weight_decay=self.weight_decay)
        opt.zero_grad(set_to_none=True)
        loss = self.loss(model, hidden, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([adapter.A, adapter.B], 1.0)
        opt.step()
        return float(loss.detach())

    def save_version(self, adapter: HeadLoRA, directory: str | Path, version: int) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"head_lora_v{version:04d}.pt"
        adapter.save(path)
        return path
