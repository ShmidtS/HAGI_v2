"""Multimodal source coding + Q-Former bridge + grounded infomax.

Separable source-coding theorem -> one factorized source encoder per modality
(text reuses the shared ``ConvEmbedding``; image = ViT patch linear + 2D-RoPE;
audio = mel linear + 1D-RoPE). A **Q-Former bridge** (Flamingo/BLIP-2
bottleneck queries) compresses each modality to a FIXED number of fused tokens
(``n_bridge_queries``) that are prepended as a prefix to the text sequence.
This is O(1) multimodal tokens regardless of image size — the scalability fix
vs the V25 early-fusion concatenation (which was O(image_patches) in length).

V28 replaces V27's fixed-1024-entry learned positional tables (which were
neither 2D nor scalable past 32x32 patches) with genuine 2D-RoPE (image
patches rotate by (row, col)) and 1D-RoPE (audio frames by frame index) from
``rope.py``. Position is now a continuous rotation that generalizes to any
image/audio size.

Grounded infomax (VICReg + InfoNCE) is applied on the per-modality pooled
embeddings to align the joint space — see ``grounded.py``.

All parameters here are FP32 source-codebook / 1D heads: they route to AdamW
(the optimizer picks them by type, not name — none are BitLinear). The Q-Former
cross-attention is FP because it is a source-side compressor, not a channel
weight.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import Config
from hagi.model.rope import rope_cos_sin, rope_cos_sin_2d


def _inv_var_gate(h: torch.Tensor, log_var_head: nn.Module) -> torch.Tensor:
    """Learned inverse-variance gate in (0, 1).

    ``sigmoid(-logit)`` keeps the modality-specific residual when the modality
    is confident (low variance -> gate -> 1) and suppresses it when uncertain.
    """
    logit = log_var_head(h).squeeze(-1)
    return torch.sigmoid(-logit.float()).to(h.dtype)


class _InvVarHead(nn.Module):
    """Per-position scalar log-variance estimator. Zero-bias init -> gate 0.5."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.log_var = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.log_var.weight, std=1.0 / max(hidden_size, 1))
        nn.init.zeros_(self.log_var.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.log_var(h)


class QFormerLayer(nn.Module):
    """One Q-Former layer: self-attn over queries, cross-attn queries->tokens, FFN."""

    def __init__(self, hidden_size: int, n_heads: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cross_norm_q = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cross_norm_kv = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.self_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """One Q-Former refinement pass.

        Args:
            queries: ``[B, Q, H]`` learnable bridge queries.
            tokens: ``[B, T, H]`` modality tokens to attend to.

        Returns:
            ``[B, Q, H]`` refined queries.
        """
        q = self.q_norm(queries)
        sa, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + sa
        cq = self.cross_norm_q(queries)
        ckv = self.cross_norm_kv(tokens)
        ca, _ = self.cross_attn(cq, ckv, ckv, need_weights=False)
        queries = queries + ca
        fq = self.ffn_norm(queries)
        queries = queries + self.ffn(fq)
        return queries


class MultimodalFusion(nn.Module):
    """Per-modality source encoders + Q-Former bridge + grounded infomax.

    Args:
        cfg: top-level config.
        text_encoder: the model's shared ``ConvEmbedding`` (one source codebook
            per model). Falls back to a standalone ``ConvEmbedding``.
    """

    NUM_MODALITIES = 3

    def __init__(self, cfg: Config, text_encoder: nn.Module | None = None) -> None:
        super().__init__()
        m = cfg.model
        H = m.hidden_size
        mm = m.multimodal
        self.H = H
        self.n_bridge = mm.n_bridge_queries
        self.image_patch_size = mm.image_patch_size
        self.audio_n_mels = mm.audio_mel_bins

        if text_encoder is not None:
            self.text_embed = text_encoder
            self._text_shared = True
        else:
            from hagi.model.conv_embedding import ConvEmbedding

            self.text_embed = ConvEmbedding(
                vocab_size=m.vocab_size,
                hidden_size=H,
                factor_rank=m.embeddings.factor_rank,
                kernel_size=m.embeddings.kernel_size,
                norm_eps=m.norm_eps,
            )
            self._text_shared = False

        # Per-modality image/audio source encoders + inv-var gating.
        self.image_embed = nn.Linear(mm.image_channels * self.image_patch_size**2, H, bias=False)
        self.audio_embed = nn.Linear(self.audio_n_mels, H, bias=False)
        self.image_unc = _InvVarHead(H)
        self.audio_unc = _InvVarHead(H)
        r_shared = max(8, H // 4)
        self.shared_down = nn.Linear(H, r_shared, bias=False)
        self.shared_up = nn.Linear(r_shared, H, bias=False)
        nn.init.normal_(self.shared_down.weight, std=1.0 / max(H, 1))
        nn.init.zeros_(self.shared_up.weight)
        self.modality_embeds = nn.Parameter(torch.zeros(self.NUM_MODALITIES, H))
        nn.init.normal_(self.modality_embeds, std=mm.modality_embed_std)

        # 2D/1D-RoPE for image patches / audio frames (replaces V27 fixed-1024
        # tables). Position is a continuous rotation, generalizing to any size.
        # The rotary band lives at the modality head_dim = H // bridge_n_heads.
        self._rope_head_dim = H // mm.bridge_n_heads
        if self._rope_head_dim % 4 != 0:
            raise ValueError(
                f"multimodal bridge head_dim (H // bridge_n_heads = {self._rope_head_dim}) "
                f"must be divisible by 4 for 2D-RoPE"
            )
        self._rope_theta = m.attention.rope_theta
        self._image_patch_size = self.image_patch_size

        # Q-Former bridge: shared learnable queries + per-modality cross-attn layers.
        self.bridge_queries = nn.Parameter(torch.zeros(self.n_bridge, H))
        nn.init.normal_(self.bridge_queries, std=0.02)
        self.image_qformer = nn.ModuleList(
            QFormerLayer(H, mm.bridge_n_heads, m.norm_eps) for _ in range(mm.bridge_layers)
        )
        self.audio_qformer = nn.ModuleList(
            QFormerLayer(H, mm.bridge_n_heads, m.norm_eps) for _ in range(mm.bridge_layers)
        )

        from hagi.model.grounded import GroundedInfomax

        self.grounded = GroundedInfomax(H, m.grounded, self.NUM_MODALITIES)

    def _fuse(self, h: torch.Tensor, unc: nn.Module) -> torch.Tensor:
        """Shared/specific split + inverse-variance gating of the specific residual."""
        z_shared = self.shared_up(self.shared_down(h))
        z_specific = h - z_shared
        gate = _inv_var_gate(h, unc).unsqueeze(-1)
        return z_shared + gate * z_specific

    def _apply_modality_rope(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply 2D/1D-RoPE to modality tokens ``[B, T, H]``.

        Reshapes to ``[B, n_heads, T, head_dim]`` and rotates via the half-split
        scheme used by ``rope.apply_rope_2d`` / ``apply_rope``. ``head_dim`` =
        ``H // bridge_n_heads`` so the rotary band matches the bridge heads.
        """
        b, t, hh = h.shape
        hd = self._rope_head_dim
        nh = hh // hd
        x = h.view(b, t, nh, hd).permute(0, 2, 1, 3)  # [B, nh, T, hd]
        cos4 = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
        sin4 = sin.unsqueeze(0).unsqueeze(0)
        half = hd // 2
        from hagi.model.rope import rotate_half

        def _rot2d(z: torch.Tensor) -> torch.Tensor:
            z1, z2 = z[..., :half], z[..., half:]
            r1 = (z1 * cos4[..., :half]) + (rotate_half(z1) * sin4[..., :half])
            r2 = (z2 * cos4[..., half:]) + (rotate_half(z2) * sin4[..., half:])
            return torch.cat([r1, r2], dim=-1)

        x = _rot2d(x)
        return x.permute(0, 2, 1, 3).reshape(b, t, hh)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.text_embed(input_ids)
        return h + self.modality_embeds[0]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h_img, w_img = images.shape
        p = self.image_patch_size
        n_h, n_w = h_img // p, w_img // p
        t_i = n_h * n_w
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, t_i, c * p * p)
        h = self.image_embed(patches)
        # 2D-RoPE: (row, col) per patch.
        rows = torch.arange(n_h, device=h.device).view(n_h, 1).expand(n_h, n_w).reshape(-1).to(torch.float32)
        cols = torch.arange(n_w, device=h.device).view(1, n_w).expand(n_h, n_w).reshape(-1).to(torch.float32)
        cos, sin = rope_cos_sin_2d(rows, cols, self._rope_head_dim, self._rope_theta, h.device, h.dtype)
        h = self._apply_modality_rope(h, cos, sin)
        h = h + self.modality_embeds[1]
        return self._fuse(h, self.image_unc)

    def encode_audio(self, spectrograms: torch.Tensor) -> torch.Tensor:
        b, _n_mels, t_frames = spectrograms.shape
        frames = spectrograms.transpose(1, 2)
        h = self.audio_embed(frames)
        # 1D-RoPE: frame index.
        pos = torch.arange(t_frames, device=h.device, dtype=torch.float32)
        cos, sin = rope_cos_sin(pos, self._rope_head_dim, self._rope_theta, h.device, h.dtype)
        h = self._apply_modality_rope(h, cos, sin)
        h = h + self.modality_embeds[2]
        return self._fuse(h, self.audio_unc)

    def _bridge(self, tokens: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        """Q-Former: compress ``tokens`` to ``n_bridge`` fused tokens."""
        b = tokens.shape[0]
        q = self.bridge_queries.unsqueeze(0).expand(b, -1, -1).contiguous()
        for layer in layers:
            q = layer(q, tokens)
        return q  # [B, n_bridge, H]

    def forward(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode text + bridge each present modality to a fixed prefix.

        Returns:
            ``(h, modality_ids, info)`` where ``h`` is ``[B, T_text + n_bridge*nm, H]``
            (prefix = Q-Former tokens, then text) and ``modality_ids`` is the
            matching long tensor. ``info`` carries 'prefix_len' (prefix length)
            and 'bridge_tokens' (count) for prefix-LM attention.
        """
        h_text = self.text_embed(text_ids)
        h_text = h_text + self.modality_embeds[0]
        b, t_text, _ = h_text.shape

        parts: list[torch.Tensor] = []
        mod_ids: list[torch.Tensor] = []
        # Modality-pooled source embeddings for grounded infomax are taken from
        # the pre-bridge fused tokens (see the model forward that calls us).
        for tokens, mod_idx, layers in (
            (self.encode_image(images) if images is not None else None, 1, self.image_qformer),
            (self.encode_audio(spectrograms) if spectrograms is not None else None, 2, self.audio_qformer),
        ):
            if tokens is None:
                continue
            bridge = self._bridge(tokens, layers)  # [B, n_bridge, H]
            parts.append(bridge)
            mod_ids.append(torch.full((b, self.n_bridge), mod_idx, dtype=torch.long, device=h_text.device))

        if parts:
            prefix = torch.cat(parts, dim=1)  # [B, n_bridge*nm, H]
            prefix_mod = torch.cat(mod_ids, dim=1)
            h = torch.cat([prefix, h_text], dim=1)
            modality_ids = torch.cat(
                [prefix_mod, torch.zeros(b, t_text, dtype=torch.long, device=h_text.device)], dim=1
            )
            prefix_len = prefix.shape[1]
        else:
            h = h_text
            modality_ids = torch.zeros(b, t_text, dtype=torch.long, device=h_text.device)
            prefix_len = 0

        info = {"prefix_len": prefix_len, "bridge_tokens": self.n_bridge}
        return h, modality_ids, info
