"""Multimodal source coding: per-modality coders + fixed-rate bridge.

Separable source coding. Each modality has its own statistics, so each gets its
own source coder: text a codebook, image a patch projection with 2D-RoPE, audio
a mel-frame projection with 1D-RoPE. Then a **fixed-rate bridge** compresses any
modality to exactly ``n_bridge_queries`` tokens, which are prepended to the text
sequence.

Fixed rate is the scalability property. Early fusion appends one token per patch,
so a 512x512 image at patch 16 costs 1024 sequence positions and the attention
cost grows quadratically in image area. The bridge makes the cost a constant
chosen by configuration, independent of resolution or audio duration.

Position uses RoPE rather than a learned table: 2D-RoPE rotates a patch by
``(row, col)`` with the head dimension split into a row band and a column band,
so the query-key inner product depends on ``(Δrow, Δcol)``. A learned table of
fixed size cannot represent a resolution it was not trained on; a rotation
extrapolates by construction.

Cross-modal grounding (:class:`Grounding`) is an auxiliary objective on the
*joint embedding*, and it stays off the language-modelling path:

* **InfoNCE** maximizes a lower bound on ``I(text; image)``. In coding terms
  this is Slepian-Wolf: learn the correlation between two sources so that one
  can serve as side information when decoding the other.
* **Variance / covariance hinge** prevents the representation from collapsing to
  a constant (which trivially maximizes nothing but makes InfoNCE degenerate)
  and decorrelates dimensions so the embedding does not waste axes on redundant
  copies of the same feature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import Config
from hagi.model.norms import RMSNorm
from hagi.model.rope import rope_cos_sin, rope_cos_sin_2d, rotate_pairs


class BridgeLayer(nn.Module):
    """One bridge layer: query self-attention, query-to-token cross-attention, FFN.

    Args:
        hidden_size: H.
        n_heads: attention heads (must divide H).
        norm_eps: RMSNorm epsilon.
    """

    def __init__(self, hidden_size: int, n_heads: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}")
        self.h = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.self_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.self_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.self_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.cross_norm_q = RMSNorm(hidden_size, eps=norm_eps)
        self.cross_norm_kv = RMSNorm(hidden_size, eps=norm_eps)
        self.cross_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.cross_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_up = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ffn_down = nn.Linear(4 * hidden_size, hidden_size, bias=False)

        # Zero-init every branch output: the bridge starts as an exact identity on
        # its queries, so enabling the multimodal path cannot perturb a text-only
        # model at step 0.
        for module in (self.self_out, self.cross_out, self.ffn_down):
            nn.init.zeros_(module.weight)
        for module in (self.self_qkv, self.cross_q, self.cross_kv, self.ffn_up):
            nn.init.normal_(module.weight, std=hidden_size**-0.5)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).reshape(b, t, self.h)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Refine ``[B, Q, H]`` queries against ``[B, T, H]`` modality tokens."""
        q_in = self.self_norm(queries)
        q, k, v = self.self_qkv(q_in).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v))
        queries = queries + self.self_out(self._merge(attn))

        q = self.cross_q(self.cross_norm_q(queries))
        k, v = self.cross_kv(self.cross_norm_kv(tokens)).chunk(2, dim=-1)
        attn = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v))
        queries = queries + self.cross_out(self._merge(attn))

        return queries + self.ffn_down(F.silu(self.ffn_up(self.ffn_norm(queries))))


class Grounding(nn.Module):
    """Off-path cross-modal objective: InfoNCE plus anti-collapse regularizers.

    Args:
        hidden_size: H.
        temperature: InfoNCE temperature.
        variance_gamma: per-dimension standard-deviation floor.
    """

    def __init__(self, hidden_size: int, temperature: float = 0.07, variance_gamma: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.variance_gamma = float(variance_gamma)
        self.text_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.modal_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _regularize(self, z: torch.Tensor) -> torch.Tensor:
        """Variance hinge + off-diagonal covariance penalty on ``[B, D]``."""
        if z.shape[0] < 2:
            return z.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        std = (z.var(dim=0, unbiased=False) + 1e-8).sqrt()
        var_loss = F.relu(self.variance_gamma - std).mean()
        cov = (z.t() @ z) / (z.shape[0] - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return var_loss + 0.04 * off_diag / z.shape[1]

    def forward(self, text_pooled: torch.Tensor, modal_pooled: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE plus regularizers over a batch of paired embeddings.

        Args:
            text_pooled: ``[B, H]`` mean-pooled text states.
            modal_pooled: ``[B, H]`` mean-pooled non-text states.

        Returns:
            Scalar loss; zero when the batch has fewer than 2 pairs (InfoNCE
            needs at least one negative).
        """
        if text_pooled.shape[0] < 2:
            return text_pooled.new_zeros(())
        zt = self.text_proj(text_pooled.float())
        zm = self.modal_proj(modal_pooled.float())
        logits = F.normalize(zt, dim=-1) @ F.normalize(zm, dim=-1).t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        infonce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return infonce + self._regularize(zt) + self._regularize(zm)


class MultimodalBridge(nn.Module):
    """Per-modality source coders plus a shared fixed-rate bridge.

    Args:
        cfg: top-level config (reads ``model.multimodal`` and ``model.hidden_size``).
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        mm = m.multimodal
        h = m.hidden_size
        self.h = h
        self.n_bridge = mm.n_bridge_queries
        self.patch = mm.image_patch_size
        self.mel_bins = mm.audio_mel_bins
        self.rope_theta = m.attention.rope_theta
        self.head_dim = h // mm.bridge_heads
        if self.head_dim % 4:
            raise ValueError(
                f"bridge head_dim ({self.head_dim}) must be divisible by 4: 2D-RoPE "
                "splits it into a row band and a column band, each of even width"
            )

        self.image_proj = nn.Linear(mm.image_channels * self.patch**2, h, bias=False)
        self.audio_proj = nn.Linear(self.mel_bins, h, bias=False)
        nn.init.normal_(self.image_proj.weight, std=(mm.image_channels * self.patch**2) ** -0.5)
        nn.init.normal_(self.audio_proj.weight, std=self.mel_bins**-0.5)

        self.queries = nn.Parameter(torch.randn(mm.n_bridge_queries, h) * 0.02)
        self.image_layers = nn.ModuleList(
            BridgeLayer(h, mm.bridge_heads, m.norm_eps) for _ in range(mm.bridge_layers)
        )
        self.audio_layers = nn.ModuleList(
            BridgeLayer(h, mm.bridge_heads, m.norm_eps) for _ in range(mm.bridge_layers)
        )
        self.out_norm = RMSNorm(h, eps=m.norm_eps)
        self.grounding = Grounding(h, mm.infonce_temperature, mm.variance_gamma)
        self.modality_dropout = float(mm.modality_dropout)
        self.n_bridge = int(mm.n_bridge_queries)

    def _apply_rope(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate ``[B, T, H]`` viewed as ``[B, n_heads, T, head_dim]``."""
        b, t, h = tokens.shape
        x = tokens.view(b, t, h // self.head_dim, self.head_dim).transpose(1, 2)
        x = rotate_pairs(x, cos, sin)
        return x.transpose(1, 2).reshape(b, t, h)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify ``[B, C, H_img, W_img]`` and apply 2D-RoPE.

        Returns:
            ``[B, n_patches, H]``.
        """
        b, c, h_img, w_img = images.shape
        p = self.patch
        if h_img % p or w_img % p:
            raise ValueError(f"image size ({h_img}, {w_img}) must be divisible by patch size {p}")
        n_h, n_w = h_img // p, w_img // p
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, n_h * n_w, c * p * p)
        tokens = self.image_proj(patches)
        rows = torch.arange(n_h, device=tokens.device).repeat_interleave(n_w).float()
        cols = torch.arange(n_w, device=tokens.device).repeat(n_h).float()
        cos, sin = rope_cos_sin_2d(rows, cols, self.head_dim, self.rope_theta, tokens.device, tokens.dtype)
        return self._apply_rope(tokens, cos, sin)

    def encode_audio(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Project ``[B, n_mels, T_frames]`` frames and apply 1D-RoPE.

        Returns:
            ``[B, T_frames, H]``.
        """
        tokens = self.audio_proj(spectrograms.transpose(1, 2))
        pos = torch.arange(tokens.shape[1], device=tokens.device).float()
        cos, sin = rope_cos_sin(pos, self.head_dim, self.rope_theta, tokens.device, tokens.dtype)
        return self._apply_rope(tokens, cos, sin)

    def bridge(self, tokens: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        """Compress ``[B, T, H]`` modality tokens to the fixed bridge rate."""
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(tokens.dtype)
        for layer in layers:
            q = layer(q, tokens)
        return self.out_norm(q)

    def forward(
        self, images: torch.Tensor | None = None, spectrograms: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Encode and bridge whichever modalities are present.

        Train-time modality dropout independently drops each non-text source
        with probability ``modality_dropout`` so the text channel remains
        self-sufficient. Every surviving modality uses the same fixed rate.

        Returns:
            ``(prefix, modal_pooled)`` — ``prefix`` is ``[B, sum n_i, H]`` for
            the language stack, ``modal_pooled`` is ``[B, H]`` for grounding.
            Both None when no modality is supplied (or all were dropped).
        """
        drop_p = self.modality_dropout if self.training else 0.0
        keep_image = images is not None and (drop_p <= 0 or torch.rand(()) >= drop_p)
        keep_audio = spectrograms is not None and (drop_p <= 0 or torch.rand(()) >= drop_p)

        encoded: list[tuple[torch.Tensor, nn.ModuleList]] = []
        if keep_image:
            encoded.append((self.encode_image(images), self.image_layers))
        if keep_audio:
            encoded.append((self.encode_audio(spectrograms), self.audio_layers))
        if not encoded:
            return None, None

        parts = [self.bridge(tokens, layers) for tokens, layers in encoded]
        prefix = torch.cat(parts, dim=1)
        return prefix, prefix.mean(dim=1)
