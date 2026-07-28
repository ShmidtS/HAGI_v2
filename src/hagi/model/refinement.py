"""Off-path predictive refinement (HEP extrinsic error highway) — opt-in.

Rehabilitated from the V25 ``predictive_v25.py`` (deleted in the V8->V25
collapse) and re-placed CORRECTLY. The original idea was sound:

  * **Highway Error Propagation (HEP)**: a zero-init linear feedback keeps the
    refinement correction depth-independent so deep error highways train (vanilla
    predictive coding decays exponentially with depth).
  * **Extrinsic-only refinement**: each iteration adds ONLY the new gated
    innovation, never re-broadcasts the estimate (prevents information
    recycling / belief amplification).
  * **Identity cold-start**: the update path is zero/small-init so the module is
    a no-op at init (does not disturb from-scratch training).

Its V25 FAILURE was being **in the main LM path** (`context -> IB(z) -> PD ->
LM head`), which deadlocked from-scratch training (CE stalled at ~ln(V)).

The V28 fix: refinement runs on a COPY of the context hidden, strictly OFF the
main LM path. The main logits come from the un-refined ``h_ctx`` (exactly as
V27). An auxiliary loss ``refinement_loss`` asks the refined hidden to predict
the next token better than the un-refined one (KL to the un-refined posterior
plus a tiny CE) — this is the only gradient into the refinement branch, and it
can never intercept the LM signal. EXIT-chart novelty halts the iterations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import RefinementConfig
from hagi.model.norms import RMSNorm
from hagi.model.ternary import BitLinear


def _proj(in_f: int, out_f: int, use_ternary: bool) -> nn.Module:
    return BitLinear(in_f, out_f, bias=False) if use_ternary else nn.Linear(in_f, out_f, bias=False)


def _kl_main_to_ref(ref_logits: torch.Tensor, main_logits: torch.Tensor, row_chunk: int = 1024) -> torch.Tensor:
    """KL(main || ref) averaged over rows, chunked over the row axis.

    The fused form materializes two fp32 ``[N, V]`` log-softmax tensors; at
    V=262144 and long context (N up to ~seq_len) that is ~17 GB and OOMs the
    refinement branch. Chunking over rows keeps only ``[row_chunk, V]`` fp32
    live at once. The short-context path (N <= row_chunk) is the original
    fused expression, so dry-run / short-sequence behavior is unchanged.
    """
    n = ref_logits.shape[0]
    if n == 0:
        return ref_logits.new_zeros(())
    main_det = main_logits.detach()
    if n <= row_chunk:
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        main_logp = F.log_softmax(main_det.float(), dim=-1)
        return (main_logp.exp() * (main_logp - ref_logp)).sum(dim=-1).mean()
    kl_per_row = ref_logits.new_zeros(n, dtype=torch.float32, device=ref_logits.device)
    for i in range(0, n, row_chunk):
        r = ref_logits[i : i + row_chunk].float()
        m = main_det[i : i + row_chunk].float()
        ref_logp = F.log_softmax(r, dim=-1)
        main_logp = F.log_softmax(m, dim=-1)
        kl_per_row[i : i + r.shape[0]] = (main_logp.exp() * (main_logp - ref_logp)).sum(dim=-1)
    return kl_per_row.mean()


class PredictiveRefiner(nn.Module):
    """Off-path iterative extrinsic refinement of the context hidden.

    Runs on a clone of ``h_ctx``; the main LM logits are computed from the
    ORIGINAL ``h_ctx``. The refinement branch receives gradient only through
    :meth:`auxiliary_loss`, so it cannot deadlock from-scratch training.

    Args:
        hidden_size: H.
        cfg: refinement config.
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize the 2D refinement masters via BitLinear.
    """

    def __init__(self, hidden_size: int, cfg: RefinementConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cfg = cfg
        self.iterations = max(1, int(cfg.iterations))
        uh = cfg.update_hidden or hidden_size
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        # Innovation MLP: h -> update. Zero-init W -> identity cold start.
        self.up_proj = _proj(hidden_size, uh, use_ternary)
        self.gate_proj = _proj(hidden_size, uh, use_ternary)
        self.update_out = _proj(uh, hidden_size, use_ternary)
        nn.init.normal_(self.update_out.weight, std=0.02)
        # HEP feedback: depth-independent extrinsic error highway (zero-init).
        self.hep = _proj(hidden_size, hidden_size, use_ternary) if cfg.hep_enabled else None
        if self.hep is not None:
            if isinstance(self.hep, BitLinear):
                nn.init.normal_(self.hep.weight, std=0.02)
            else:
                assert isinstance(self.hep, nn.Linear)
                nn.init.zeros_(self.hep.weight)
        self._last_ext_before: torch.Tensor | None = None
        self._last_ext_after: torch.Tensor | None = None

    def forward(self, h_ctx: torch.Tensor) -> torch.Tensor:
        """Refine a CLONE of ``h_ctx``; return the refined hidden (off-path).

        Extrinsic-only: each iteration adds only the new gated innovation.
        ``ẑ_{t+1} = ẑ_t + g_t * (u_t)`` plus the HEP feedback ``V_t * innovation``.

        Args:
            h_ctx: ``[B, T, H]`` context hidden (main-path, never mutated).

        Returns:
            ``[B, T, H]`` refined hidden (gradient flows only to this branch).
        """
        z = h_ctx.clone()
        innovation_accum = z.new_zeros(())
        ext_before = (z - h_ctx).detach()
        for _ in range(self.iterations):
            h_n = self.norm(z)
            gate = torch.sigmoid(self.gate_proj(h_n))
            update = self.update_out(F.silu(self.up_proj(h_n)) * gate)
            if self.hep is not None:
                update = update + self.hep(update)
            z = z + update
            innovation_accum = innovation_accum + update.detach().float().pow(2).mean()
        ext_after = (z - h_ctx).detach()
        self._last_ext_before = ext_before
        self._last_ext_after = ext_after
        return z

    def novelty(self) -> float:
        """Last refinement pass's extrinsic novelty (for EXIT halt).

        Returns 1.0 (maximal novelty = "not converged") when the refinement is
        still identity at init (ext_before ~ 0): a 0/0 ratio would clamp to a
        huge sentinel, which would mislead the EXIT halt. As soon as the branch
        starts moving, the ratio is well-defined.
        """
        if self._last_ext_before is None or self._last_ext_after is None:
            return 1.0
        eb_norm = self._last_ext_before.float().reshape(-1).norm().item()
        if eb_norm < 1e-8:
            return 1.0
        return float((self._last_ext_after.float().reshape(-1).norm() / (eb_norm + 1e-8)).clamp(0.0, 1e6).item())


class RefinementHead(nn.Module):
    """Wraps :class:`PredictiveRefiner` + an off-path prediction head.

    Computes the auxiliary refinement loss and reports the refinement novelty
    so the training loop's EXIT halt can freeze the distortion beta-anneal on
    convergence.

    Args:
        hidden_size: H.
        vocab_size: V (for the auxiliary CE on refined hidden).
        cfg: refinement config.
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize refiner + head 2D weights.
    """

    def __init__(self, hidden_size: int, vocab_size: int, cfg: RefinementConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        self.refiner = PredictiveRefiner(hidden_size, cfg, norm_eps, use_ternary)
        # Off-path prediction head: refined hidden -> logits. Small-init.
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.to_logits = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.to_logits.weight, std=1.0 / (hidden_size**0.5))

    def forward(
        self,
        h_ctx: torch.Tensor,
        *,
        main_logits: torch.Tensor | None,
        targets: torch.Tensor | None,
        prediction_indices: torch.Tensor | None,
        refine_weight: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute the off-path refinement auxiliary loss + refined hidden.

        The refinement loss asks the refined hidden to (a) predict the next
        token (CE on refined logits) and (b) stay close to the main posterior
        (KL refinement||main). It is the ONLY gradient into the refinement
        branch; the main LM logits are untouched.

        Returns:
            ``(refinement_loss, h_refined)``; loss is None if the branch is off
            or the targets/indices are absent.
        """
        h_refined = self.refiner(h_ctx)
        if refine_weight <= 0.0 or targets is None or prediction_indices is None or main_logits is None:
            return None, h_refined
        # Gather refined logits at the prediction positions.
        idx = prediction_indices.to(h_refined.device)
        h_sel = h_refined.flatten(0, 1).index_select(0, idx)
        ref_logits = self.to_logits(self.norm(h_sel))
        sel_targets = targets.flatten().index_select(0, idx.to(targets.device)).to(ref_logits.device)
        ce_refined = F.cross_entropy(ref_logits, sel_targets)
        # KL(refined || main): encourage the refined posterior to track the main
        # one (stability) while the CE pulls it toward the truth (improvement).
        # Chunked over rows: the fused form holds two fp32 [N, V] log-softmax
        # tensors (~17 GB at long context, V=262144) and OOMs the refinement
        # branch. See _kl_main_to_ref.
        kl = _kl_main_to_ref(ref_logits, main_logits)
        loss = ce_refined + 0.1 * kl
        return loss, h_refined
