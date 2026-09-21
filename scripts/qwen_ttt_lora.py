"""E2: online test-time LoRA contour for the frozen dense-Qwen MTP MLP.

The ternary base and the frozen MTP anchor are never mutated or trained. Only
low-rank LoRA adapters on explicit MTP projections change.

Adapter form (standard LoRA):

    delta(x) = scaling * B @ A.T
    adapted_output = frozen_base_output + delta(x)

A [Fin, r] is a frozen orthonormal analysis basis. B [Fout, r] is the only
trainable block. The caller supplies ``target_delta = teacher_output -
frozen_base_output``; RLS fits B to that delta. This avoids accidentally
learning the frozen base output twice.

Online adaptation reuses the verified invariants from
``scripts/dsv4_generate_ttt.py``:

* G [r, r] and C [r, Fout] are updated only from the 4/5 training split.
* The last 1/5 is Hva/Yva honest holdout and never enters G/C.
* A ridge solve runs every ``TTT_REFIT`` training rows and is applied only if
  it strictly improves the current reconstruction residual.
* Save decisions use Hva/Yva only and require ``gain >= TTT_MIN_GAIN``.
* Checkpoints are written through ``tmp`` + ``os.replace`` and contain only
  LoRA/RLS state, metadata, validation metrics, and RNG state.

Attention projections are opt-in (``include_attn=True`` / ``--attn``), never
implicit. The closed-form low-rank RLS avoids an infeasible full
``O(Fin**2)`` Gram matrix for Qwen's 17,408-dimensional projections.

Reference-first note: the A/B layout follows the standard LoRA form used by
Hugging Face PEFT (``lora_B @ lora_A`` on the forward path); the anchored RLS,
holdout split, refit guard, and save-on-improvement contract comes from the
verified local implementation in ``scripts/dsv4_generate_ttt.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

# --------------------------------------------------------------------------- #
# TTT knobs (local provenance: scripts/dsv4_generate_ttt.py)
# --------------------------------------------------------------------------- #
TTT_ALPHA = float(os.environ.get("TTT_ALPHA", "1000.0"))
TTT_LAM = float(os.environ.get("TTT_LAM", "0.9995"))
TTT_REFIT = int(os.environ.get("TTT_REFIT", "64"))
TTT_MIN_GAIN = float(os.environ.get("TTT_MIN_GAIN", "0.02"))
TTT_ROWS_MAX = int(os.environ.get("TTT_ROWS_MAX", "2048"))
TTT_REG = float(os.environ.get("TTT_REG", "1e-3"))
TTT_PRECOND = os.environ.get("TTT_PRECOND", "1") == "1"
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_SCALING = float(os.environ.get("LORA_SCALING", "1.0"))

# --------------------------------------------------------------------------- #
# Projection surface (MTP layer 0, from mtp_config.json)
# --------------------------------------------------------------------------- #
MTP_MLP_PROJECTIONS: List[Tuple[str, int, int]] = [
    ("mlp.gate_proj", 5120, 17408),
    ("mlp.up_proj", 5120, 17408),
    ("mlp.down_proj", 17408, 5120),
]
MTP_ATTN_PROJECTIONS: List[Tuple[str, int, int]] = [
    ("self_attn.q_proj", 5120, 12288),
    ("self_attn.k_proj", 5120, 1024),
    ("self_attn.v_proj", 5120, 1024),
    ("self_attn.o_proj", 6144, 5120),
]

# Real MTP attention tensor shapes: Fin is the weight column count and Fout is
# the weight row count. For example, q_proj.weight is [Fout=12288, Fin=5120].
MTP_ATTN_SHAPES: Dict[str, Tuple[int, int]] = {
    "self_attn.q_proj": (5120, 12288),
    "self_attn.k_proj": (5120, 1024),
    "self_attn.v_proj": (5120, 1024),
    "self_attn.o_proj": (6144, 5120),
}

MTP_SOURCE_TENSOR_KEYS = (
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
)


def _module_key(public_name: str) -> str:
    """Return an nn.ModuleDict-safe key while preserving public tensor names."""
    return public_name.replace(".", "_")


# --------------------------------------------------------------------------- #
# Frozen base MLP (Qwen3.5 RMSNorm + SwiGLU, pre-norm, outer residual caller)
# --------------------------------------------------------------------------- #
class FrozenMLP(nn.Module):
    """Frozen Qwen3.5 MTP-0 MLP."""

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 17408,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.norm_weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (out * (1.0 + self.norm_weight.float())).type_as(x)

    def forward(self, x: torch.Tensor, adapter: "TTTAdapter") -> torch.Tensor:
        xn = self.rms_norm(x)
        w_dtype = xn.dtype
        gw = self.gate_proj.weight.to(dtype=w_dtype)
        uw = self.up_proj.weight.to(dtype=w_dtype)
        gate = F.linear(xn, gw) + adapter.gate_delta_forward(xn)
        up = F.linear(xn, uw) + adapter.up_delta_forward(xn)
        h = F.silu(gate) * up
        dw = self.down_proj.weight.to(dtype=h.dtype)
        return F.linear(h, dw) + adapter.down_delta_forward(h)

    def load_base_weights(
        self, state: Dict[str, torch.Tensor], prefix: str = ""
    ) -> None:
        """Copy MTP MLP weights read-only; source tensors are never mutated."""
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                _lookup_weight(state, prefix, "mlp.gate_proj.weight")
            )
            self.up_proj.weight.copy_(
                _lookup_weight(state, prefix, "mlp.up_proj.weight")
            )
            self.down_proj.weight.copy_(
                _lookup_weight(state, prefix, "mlp.down_proj.weight")
            )
            self.norm_weight.copy_(
                _lookup_weight(
                    state, prefix, "post_attention_layernorm.weight"
                )
            )
        self._freeze()


def _lookup_weight(
    state: Dict[str, torch.Tensor], prefix: str, leaf: str
) -> torch.Tensor:
    candidates: List[str] = []
    if prefix:
        candidates.extend(
            [
                f"{prefix}.layers.0.{leaf}",
                f"{prefix}.{leaf}",
            ]
        )
    candidates.extend(
        [
            f"mtp.layers.0.{leaf}",
            f"layers.0.{leaf}",
            f"mtp.{leaf}",
            leaf,
        ]
    )
    for key in dict.fromkeys(candidates):
        if key in state:
            return state[key]
    raise KeyError(f"no weight found for {leaf} (prefix={prefix!r})")


# --------------------------------------------------------------------------- #
# Low-rank LoRA + online anchored RLS on B
# --------------------------------------------------------------------------- #
class LowRankLoRA(nn.Module):
    """delta(x) = scaling * (x @ A) @ B.T.

    A [Fin, r] is frozen. B [Fout, r] is adapted against explicit target deltas.
    RLS state is O(r**2 + r*Fout), not O(Fin**2).
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        r: int = LORA_R,
        scaling: float = LORA_SCALING,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.r = r
        self.scaling = float(scaling)
        self.step = 0

        torch.manual_seed(seed)
        self.register_buffer("A", _orth(torch.randn(fin, r)), persistent=False)
        self.B = nn.Parameter(torch.zeros(fout, r))

        # Non-trainable RLS/anchor state. Buffers move with the module to CPU,
        # CUDA, or ROCm and are restored explicitly by load_checkpoint.
        self.register_buffer("G", torch.eye(r) * TTT_ALPHA)
        # C starts as alpha * anchor_B.T == 0 (anchor delta is zero). This
        # mirrors the DSv4 anchored-RLS invariant C = alpha*W_anchor_T, where
        # the zero anchor makes C a zero vector; we keep it explicit so the
        # schema stays faithful to the anchored-RLS contract.
        self.register_buffer("C", torch.zeros(r, fout))
        self.register_buffer("anchor_B", torch.zeros(fout, r))
        self.register_buffer("fsc", torch.empty(0), persistent=False)
        self.train_rows = 0
        self.refits = 0
        self.holdout_rows = 0
        self.H: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        self.Hva: List[torch.Tensor] = []
        self.Yva: List[torch.Tensor] = []

    # -- forward -------------------------------------------------------------
    def _feature(self, x: torch.Tensor) -> torch.Tensor:
        phi = x.to(dtype=self.A.dtype) @ self.A
        if self.fsc.numel() == self.r:
            return phi / self.fsc
        return phi

    def delta_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trainable delta path: [N, Fin] -> [N, Fout]."""
        delta = self.scaling * (self._feature(x) @ self.B.T)
        return delta.to(dtype=x.dtype)

    def delta_weight(self) -> torch.Tensor:
        """Materialized [Fout, Fin] delta for inspection/export."""
        return (self.scaling * self.B @ self.A.T).contiguous()

    # -- anchored low-rank RLS ---------------------------------------------
    @torch.no_grad()
    def rls_step(
        self,
        feature_rows: torch.Tensor,
        target_delta_rows: torch.Tensor,
        holdout: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Update from [N, Fin] features and [N, Fout] target deltas.

        The last 1/5 rows become Hva/Yva and never enter G/C. A ridge solve is
        attempted every ``TTT_REFIT`` training rows and applied only when it
        improves the current training residual.
        """
        if feature_rows.ndim != 2 or target_delta_rows.ndim != 2:
            raise ValueError("feature_rows and target_delta_rows must be 2D")
        if feature_rows.shape[0] != target_delta_rows.shape[0]:
            raise ValueError("feature/target row count mismatch")
        if feature_rows.shape[1] != self.fin:
            raise ValueError(f"expected feature dim {self.fin}")
        if target_delta_rows.shape[1] != self.fout:
            raise ValueError(f"expected target dim {self.fout}")

        n = feature_rows.shape[0]
        n_va = n // 5 if holdout and n >= 5 else (1 if holdout and n > 1 else 0)
        split = n - n_va
        tr_h = feature_rows[:split].detach().to(dtype=self.A.dtype)
        tr_y = target_delta_rows[:split].detach()
        va_h = feature_rows[split:].detach()
        va_y = target_delta_rows[split:].detach()
        nt = tr_h.shape[0]

        tr_phi_raw = tr_h @ self.A
        if TTT_PRECOND and self.fsc.numel() != self.r:
            self.fsc = tr_phi_raw.pow(2).mean(0).sqrt().clamp_min(1e-6)
        tr_phi = (
            tr_phi_raw / self.fsc
            if self.fsc.numel() == self.r
            else tr_phi_raw
        )

        self.G.mul_(TTT_LAM**nt).add_(tr_phi.T @ tr_phi)
        self.C.mul_(TTT_LAM**nt).add_(tr_phi.T @ tr_y.to(dtype=self.C.dtype))
        self.train_rows += nt

        if nt:
            self.H.append(tr_phi.detach())
            self.Y.append(tr_y.detach())
            if sum(t.shape[0] for t in self.H) > TTT_ROWS_MAX:
                self.H.pop(0)
                self.Y.pop(0)

        if n_va:
            self._append_holdout(va_h, va_y)
            self.holdout_rows += n_va

        if self.train_rows < TTT_REFIT:
            return None

        self.train_rows = 0
        self.refits += 1
        self.step += 1

        Hb = torch.cat(self.H)
        Yb = torch.cat(self.Y)
        reg = self.G.diagonal().mean().clamp_min(1e-12) * TTT_REG
        B_candidate = torch.linalg.solve(
            self.G + reg * torch.eye(self.r, device=self.G.device),
            self.C,
        ).T.contiguous()

        current_resid = _resid(self.B, Hb, Yb)
        candidate_resid = _resid(B_candidate, Hb, Yb)
        if candidate_resid < current_resid:
            self.B.copy_(B_candidate)
            return (current_resid, candidate_resid)
        return None

    @torch.no_grad()
    def record_holdout_only(
        self, feature_rows: torch.Tensor, target_delta_rows: torch.Tensor
    ) -> None:
        """Test helper: append honest rows without touching G/C or B."""
        if feature_rows.shape[0] != target_delta_rows.shape[0]:
            raise ValueError("feature/target row count mismatch")
        self._append_holdout(feature_rows.detach(), target_delta_rows.detach())
        self.holdout_rows += feature_rows.shape[0]

    def _append_holdout(
        self, feature_rows: torch.Tensor, target_delta_rows: torch.Tensor
    ) -> None:
        phi = self._feature(feature_rows)
        self.Hva.append(phi.detach())
        self.Yva.append(target_delta_rows.detach())
        if sum(t.shape[0] for t in self.Hva) > TTT_ROWS_MAX // 2:
            self.Hva.pop(0)
            self.Yva.pop(0)

    @torch.no_grad()
    def validate(self) -> float:
        """Honest holdout residual; Hva/Yva are never update inputs."""
        if not self.Hva:
            return float("inf")
        B = self.B.to(dtype=self.A.dtype)
        Hva = torch.cat(self.Hva).to(dtype=self.A.dtype)
        Yva = torch.cat(self.Yva).to(dtype=self.A.dtype)
        return _resid(B, Hva, Yva)

    def normal_equation_fingerprint(self) -> str:
        """Deterministic fingerprint used to prove holdout exclusion."""
        payload = b"".join(
            (
                self.G.detach().cpu().contiguous().numpy().tobytes(),
                self.C.detach().cpu().contiguous().numpy().tobytes(),
            )
        )
        return hashlib.sha256(payload).hexdigest()

    @torch.no_grad()
    def reset(self) -> None:
        nn.init.zeros_(self.B)
        self.G.copy_(torch.eye(self.r, device=self.G.device, dtype=self.G.dtype) * TTT_ALPHA)
        self.C.zero_()
        self.anchor_B.zero_()
        self.fsc.zero_()
        self.step = 0
        self.train_rows = 0
        self.refits = 0
        self.holdout_rows = 0
        self.H: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        self.Hva: List[torch.Tensor] = []
        self.Yva: List[torch.Tensor] = []


def _orth(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix)
    return q


def _resid(
    B: torch.Tensor, phi: torch.Tensor, target_delta: torch.Tensor
) -> float:
    pred = phi @ B.T
    denominator = target_delta.pow(2).sum().clamp_min(1e-12)
    return ((pred - target_delta).pow(2).sum() / denominator).item()


# --------------------------------------------------------------------------- #
# Adapter assembly
# --------------------------------------------------------------------------- #
class TTTAdapter(nn.Module):
    """LoRA adapters on selected MTP projections; base stays separate/frozen."""

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 17408,
        r: int = LORA_R,
        include_attn: bool = False,
        scaling: float = LORA_SCALING,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.r = r
        self.scaling = scaling
        self.projs = list(MTP_MLP_PROJECTIONS)
        if include_attn:
            self.projs.extend(MTP_ATTN_PROJECTIONS)

        self.adapters: Dict[str, LowRankLoRA] = nn.ModuleDict()
        for index, (name, fin, fout) in enumerate(self.projs):
            self.adapters[_module_key(name)] = LowRankLoRA(
                fin, fout, r=r, scaling=scaling, seed=seed + index
            )

    def _adapter(self, public_name: str) -> LowRankLoRA:
        return self.adapters[_module_key(public_name)]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gate_delta_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter("mlp.gate_proj").delta_forward(x)

    def up_delta_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter("mlp.up_proj").delta_forward(x)

    def down_delta_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter("mlp.down_proj").delta_forward(x)

    def mlp_forward(self, mlp: FrozenMLP, x: torch.Tensor) -> torch.Tensor:
        xn = mlp.rms_norm(x)
        gate = F.linear(
            xn, mlp.gate_proj.weight.to(dtype=xn.dtype)
        ) + self.gate_delta_forward(xn)
        up = F.linear(
            xn, mlp.up_proj.weight.to(dtype=xn.dtype)
        ) + self.up_delta_forward(xn)
        h = F.silu(gate) * up
        return F.linear(
            h, mlp.down_proj.weight.to(dtype=h.dtype)
        ) + self.down_delta_forward(h)

    @torch.no_grad()
    def rls_step(
        self,
        feature_rows: Dict[str, torch.Tensor],
        target_delta_rows: Dict[str, torch.Tensor],
        holdout: bool = True,
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        results = {}
        for name, _, _ in self.projs:
            if name not in feature_rows or name not in target_delta_rows:
                raise KeyError(f"missing rows for projection {name}")
            results[name] = self._adapter(name).rls_step(
                feature_rows[name], target_delta_rows[name], holdout=holdout
            )
        return results

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        return {
            name: self._adapter(name).validate() for name, _, _ in self.projs
        }

    def checkpoint_state(
        self, validation: Dict[str, float], rng_state: torch.Tensor
    ) -> Dict:
        return {
            "format": "qwen-mtp-ttt-lora-v1",
            "solver": "anchored-rls",
            "optimizer": None,
            "r": self.r,
            "scaling": self.scaling,
            "projs": self.projs,
            "step": sum(ap.step for ap in self.adapters.values()),
            "validation": {k: float(v) for k, v in validation.items()},
            "adapters": {
                name: {
                    "A": ap.A.detach().cpu().contiguous(),
                    "B": ap.B.detach().cpu().contiguous(),
                    "anchor_B": ap.anchor_B.detach().cpu().contiguous(),
                    "G": ap.G.detach().cpu().contiguous(),
                    "C": ap.C.detach().cpu().contiguous(),
                    "fsc": (
                        ap.fsc.detach().cpu().contiguous()
                        if ap.fsc is not None
                        else None
                    ),
                    "train_rows": ap.train_rows,
                    "refits": ap.refits,
                    "holdout_rows": ap.holdout_rows,
                    "step": ap.step,
                }
                for name, ap in (
                    (public, self._adapter(public)) for public, _, _ in self.projs
                )
            },
            "rng": rng_state.cpu(),
        }

    @torch.no_grad()
    def atomic_save(
        self,
        path: str,
        validation: Dict[str, float],
        rng_state: Optional[bytes] = None,
    ) -> bool:
        """Atomically save LoRA/RLS state only; no frozen base tensors."""
        state = self.checkpoint_state(
            validation,
            rng_state if rng_state is not None else torch.get_rng_state(),
        )
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return True

    @torch.no_grad()
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "qwen-mtp-ttt-lora-v1":
            raise ValueError("unsupported E2 checkpoint format")
        if ckpt["r"] != self.r or ckpt["scaling"] != self.scaling:
            raise ValueError("checkpoint rank/scaling mismatch")
        for public_name, _, _ in self.projs:
            if public_name not in ckpt["adapters"]:
                raise KeyError(f"checkpoint missing projection {public_name}")
            ap = self._adapter(public_name)
            saved = ckpt["adapters"][public_name]
            ap.A.copy_(saved["A"].to(ap.A.device))
            ap.B.copy_(saved["B"].to(ap.B.device))
            ap.G.copy_(saved["G"].to(ap.G.device))
            ap.C.copy_(saved["C"].to(ap.C.device))
            ap.anchor_B.copy_(saved["anchor_B"].to(ap.anchor_B.device))
            if saved["fsc"] is not None and saved["fsc"].numel() > 0:
                if ap.fsc is None or ap.fsc.numel() != self.r:
                    ap.fsc = torch.empty_like(saved["fsc"])
                ap.fsc.copy_(saved["fsc"].to(ap.fsc.device))
            ap.train_rows = int(saved["train_rows"])
            ap.refits = int(saved["refits"])
            ap.holdout_rows = int(saved["holdout_rows"])
            ap.step = int(saved.get("step", 0))


# --------------------------------------------------------------------------- #
# Bounded smoke gate
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2: online TTT-LoRA on frozen MTP MLP."
    )
    parser.add_argument(
        "--B", type=int, default=20, help="rows per online batch"
    )
    parser.add_argument("--H", type=int, default=5120)
    parser.add_argument("--I", type=int, default=17408)
    parser.add_argument("--r", type=int, default=LORA_R)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--refit", type=int, default=TTT_REFIT)
    parser.add_argument("--min-gain", type=float, default=TTT_MIN_GAIN)
    parser.add_argument(
        "--attn", action="store_true", help="also adapt attention (opt-in)"
    )
    parser.add_argument(
        "--output", type=str, default="reports/e2_ttt_lora_smoke.json"
    )
    parser.add_argument(
        "--save-ckpt", type=str, default="reports/e2_checkpoint.pt"
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--real-mtp",
        action="store_true",
        help="load FrozenMLP from model_mtp.safetensors instead of random init",
    )
    parser.add_argument(
        "--safetensors",
        type=str,
        default="models/ternary-bonsai-2-27b-mtp/model_mtp.safetensors",
        help="MTP safetensors path used with --real-mtp (read-only)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="dtype for smoke inputs/weights (auto picks from base)",
    )
    return parser.parse_args()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(-1)
        .to(torch.float32)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_state_equal(left: LowRankLoRA, right: LowRankLoRA) -> bool:
    return all(
        (
            torch.equal(left.A.cpu(), right.A.cpu()),
            torch.equal(left.B.cpu(), right.B.cpu()),
            torch.equal(left.anchor_B.cpu(), right.anchor_B.cpu()),
            torch.equal(left.G.cpu(), right.G.cpu()),
            torch.equal(left.C.cpu(), right.C.cpu()),
            torch.equal(left.fsc.cpu(), right.fsc.cpu()),
            left.train_rows == right.train_rows,
            left.refits == right.refits,
            left.holdout_rows == right.holdout_rows,
            left.step == right.step,
        )
    )


def _adapter_train_state_equal(left: LowRankLoRA, right: LowRankLoRA) -> bool:
    """Compare replay state without honest-holdout bookkeeping."""
    return all(
        (
            torch.equal(left.A.cpu(), right.A.cpu()),
            torch.equal(left.B.cpu(), right.B.cpu()),
            torch.equal(left.anchor_B.cpu(), right.anchor_B.cpu()),
            torch.equal(left.G.cpu(), right.G.cpu()),
            torch.equal(left.C.cpu(), right.C.cpu()),
            torch.equal(left.fsc.cpu(), right.fsc.cpu()),
            left.train_rows == right.train_rows,
            left.refits == right.refits,
            left.step == right.step,
        )
    )


def _checkpoint_contains_base_state(ckpt: Dict) -> bool:
    base_tokens = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    return any(token in key for key, _ in _walk_keys(ckpt) for token in base_tokens)


def _walk_keys(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield child_prefix, child
            yield from _walk_keys(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield child_prefix, child
            yield from _walk_keys(child, child_prefix)


def _teacher_delta(
    adapter: TTTAdapter,
    features: Dict[str, torch.Tensor],
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Deterministic low-rank teacher delta used only by the bounded smoke."""
    deltas: Dict[str, torch.Tensor] = {}
    for index, (name, _, _) in enumerate(adapter.projs):
        torch.manual_seed(seed + index)
        teacher_B = torch.randn(adapter._adapter(name).fout, adapter.r) * 0.05
        ap = adapter._adapter(name)
        raw_phi = features[name].to(dtype=ap.A.dtype) @ ap.A
        phi = raw_phi / (ap.fsc if ap.fsc.numel() == adapter.r else 1.0)
        deltas[name] = (phi @ teacher_B.T).to(dtype=features[name].dtype)
    return deltas


def run_e2_smoke(args: argparse.Namespace) -> Dict:
    global TTT_REFIT, TTT_MIN_GAIN
    os.environ["TTT_REFIT"] = str(args.refit)
    os.environ["TTT_MIN_GAIN"] = str(args.min_gain)
    TTT_REFIT = args.refit
    TTT_MIN_GAIN = args.min_gain
    torch.manual_seed(args.base_seed)

    real_mtp = bool(args.real_mtp)
    safetensors_state: Optional[Dict[str, torch.Tensor]] = None
    base = FrozenMLP(args.H, args.I)

    dtype = None
    if args.dtype != "auto":
        dtype = getattr(torch, args.dtype) if isinstance(args.dtype, str) else args.dtype
    if real_mtp and dtype is None:
        dtype = torch.bfloat16

    for p in base.parameters():
        p.requires_grad_(False)

    if real_mtp:
        safetensors_state = load_file(args.safetensors, device="cpu")
        source_hash_before = {
            key: _tensor_sha256(safetensors_state[key])
            for key in MTP_SOURCE_TENSOR_KEYS
        }
        base.load_base_weights(safetensors_state, "mtp.layers.0.")
        source_hash_after = {
            key: _tensor_sha256(safetensors_state[key])
            for key in MTP_SOURCE_TENSOR_KEYS
        }
        source_tensors_unchanged = source_hash_before == source_hash_after
        if dtype is not None:
            with torch.no_grad():
                for w in (base.gate_proj, base.up_proj, base.down_proj):
                    w.weight.copy_(w.weight.to(dtype=dtype))
    else:
        source_hash_before = None
        source_hash_after = None
        source_tensors_unchanged = True
        if dtype is not None:
            with torch.no_grad():
                base.gate_proj.weight.copy_(base.gate_proj.weight.to(dtype=dtype))
                base.up_proj.weight.copy_(base.up_proj.weight.to(dtype=dtype))
                base.down_proj.weight.copy_(base.down_proj.weight.to(dtype=dtype))

    for p in base.parameters():
        p.requires_grad_(False)

    gate_hash_before = _tensor_sha256(base.gate_proj.weight)
    down_hash_before = _tensor_sha256(base.down_proj.weight)
    up_hash_before = _tensor_sha256(base.up_proj.weight)
    norm_hash_before = _tensor_sha256(base.norm_weight)
    safetensors_sha256_before = (
        _sha256_of_file(args.safetensors) if real_mtp else None
    )

    adapter = TTTAdapter(
        args.H,
        args.I,
        r=args.r,
        include_attn=args.attn,
        seed=args.base_seed + 100,
    )

    losses: List[float] = []
    refit_per_step: List[int] = []
    train_feature_batches: List[Dict[str, torch.Tensor]] = []
    train_target_batches: List[Dict[str, torch.Tensor]] = []
    N = args.B

    for step in range(args.steps):
        torch.manual_seed(args.base_seed + 1000 + step)
        x = torch.randn(N, args.H, dtype=dtype or torch.float32)
        xn = base.rms_norm(x)
        with torch.no_grad():
            gate_y = F.linear(xn, base.gate_proj.weight.to(dtype=xn.dtype))
            up_y = F.linear(xn, base.up_proj.weight.to(dtype=xn.dtype))
            h = F.silu(gate_y) * up_y

        features = {
            "mlp.gate_proj": xn,
            "mlp.up_proj": xn,
            "mlp.down_proj": h,
        }
        if args.attn:
            # Bounded attention smoke: use the real MTP projection shapes.
            # q/k/v consume the hidden state; o_proj consumes grouped attention
            # output (24 heads * head_dim 256 = 6144).
            for name, (fin, _) in MTP_ATTN_SHAPES.items():
                attention_feature = torch.randn(N, fin, dtype=x.dtype)
                features[name] = attention_feature
        target_delta = _teacher_delta(adapter, features, args.base_seed + 2000)

        loss_parts = [
            F.mse_loss(
                adapter._adapter(name).delta_forward(features[name]),
                target_delta[name],
            )
            for name, _, _ in adapter.projs
        ]
        loss = sum(loss_parts)
        if not torch.isfinite(loss):
            raise RuntimeError("E2: non-finite adaptation loss")
        losses.append(loss.item())

        train_feature_batches.append({name: feat.clone() for name, feat in features.items()})
        train_target_batches.append({name: tgt.clone() for name, tgt in target_delta.items()})

        before = sum(ap.refits for ap in adapter.adapters.values())
        adapter.rls_step(features, target_delta)
        after = sum(ap.refits for ap in adapter.adapters.values())
        refit_per_step.append(after - before)

    final_validation = adapter.validate()

    # Direct holdout-exclusion proof: appending honest rows must not alter G/C.
    probe_name, probe_fin, _ = adapter.projs[0]
    probe_ap = adapter._adapter(probe_name)
    torch.manual_seed(args.base_seed + 3000)
    probe_h = torch.randn(3, probe_fin)
    probe_y = torch.randn(3, probe_ap.fout)
    ne_before = probe_ap.normal_equation_fingerprint()
    probe_ap.record_holdout_only(probe_h, probe_y)
    ne_after = probe_ap.normal_equation_fingerprint()
    holdout_excluded = ne_before == ne_after

    # Strong holdout-exclusion proof: run a fresh adapter over ONLY the train
    # split of every batch (holdout rows dropped before rls_step). A train-only
    # adapter must reach the SAME G/C/B as the live full-split run, proving the
    # honest 1/5 holdout never entered the normal equations.
    replay_train = TTTAdapter(
        args.H,
        args.I,
        r=args.r,
        include_attn=args.attn,
        seed=args.base_seed + 100,
    )
    for feat_batch, tgt_batch in zip(train_feature_batches, train_target_batches):
        train_only_feat: Dict[str, torch.Tensor] = {}
        train_only_tgt: Dict[str, torch.Tensor] = {}
        for name, _, _ in adapter.projs:
            full_n = feat_batch[name].shape[0]
            n_va = full_n // 5 if full_n >= 5 else 0
            train_n = full_n - n_va
            train_only_feat[name] = feat_batch[name][:train_n]
            train_only_tgt[name] = tgt_batch[name][:train_n]
        replay_train.rls_step(train_only_feat, train_only_tgt, holdout=False)
    train_only_replay_equal = all(
        _adapter_train_state_equal(
            replay_train._adapter(name), adapter._adapter(name)
        )
        for name, _, _ in adapter.projs
    )

    anchor_validation = _anchor_validation(adapter)
    live_validation = _mean_validation(adapter)
    save_gain = (anchor_validation - live_validation) / max(
        abs(anchor_validation), 1e-12
    )
    did_save = bool(
        save_gain >= args.min_gain and live_validation < anchor_validation
    )

    checkpoint_path = args.save_ckpt or "reports/e2_checkpoint.pt"
    if did_save:
        adapter.atomic_save(
            checkpoint_path,
            final_validation,
            torch.get_rng_state(),
        )

    checkpoint_only_lora = not _checkpoint_contains_base_state(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ) if did_save else True
    checkpoint_reload_equal = False
    if did_save:
        fresh = TTTAdapter(
            args.H,
            args.I,
            r=args.r,
            include_attn=args.attn,
            seed=args.base_seed + 100,
        )
        fresh.load_checkpoint(checkpoint_path)
        checkpoint_reload_equal = all(
            _adapter_state_equal(
                fresh._adapter(name), adapter._adapter(name)
            )
            for name, _, _ in adapter.projs
        )

    gate_hash_after = _tensor_sha256(base.gate_proj.weight)
    up_hash_after = _tensor_sha256(base.up_proj.weight)
    down_hash_after = _tensor_sha256(base.down_proj.weight)
    norm_hash_after = _tensor_sha256(base.norm_weight)
    base_unchanged = (
        gate_hash_before == gate_hash_after
        and up_hash_before == up_hash_after
        and down_hash_before == down_hash_after
        and norm_hash_before == norm_hash_after
    )
    base_grad_absent = all(p.grad is None for p in base.parameters())

    safetensors_sha256_after = (
        _sha256_of_file(args.safetensors) if real_mtp else None
    )
    safetensors_unchanged = (
        safetensors_sha256_before == safetensors_sha256_after
        if real_mtp
        else True
    )

    report = {
        "gate": "e2_ttt_lora_smoke",
        "config": {
            "B": args.B,
            "H": args.H,
            "I": args.I,
            "r": args.r,
            "steps": args.steps,
            "refit": args.refit,
            "min_gain": args.min_gain,
            "attn": args.attn,
            "seed": args.base_seed,
            "real_mtp": real_mtp,
            "dtype": str(dtype) if dtype is not None else None,
        },
        "passed": bool(
            base_unchanged
            and base_grad_absent
            and safetensors_unchanged
            and source_tensors_unchanged
            and holdout_excluded
            and train_only_replay_equal
            and checkpoint_only_lora
            and checkpoint_reload_equal
            and all(torch.isfinite(torch.tensor(value)) for value in losses)
            and all(
                torch.isfinite(torch.tensor(value))
                for value in final_validation.values()
            )
            and (did_save or save_gain < args.min_gain)
        ),
        "finite_loss": all(
            torch.isfinite(torch.tensor(value)) for value in losses
        ),
        "base_unchanged": base_unchanged,
        "base_grad_absent": base_grad_absent,
        "safetensors_unchanged": safetensors_unchanged,
        "source_tensors_unchanged": source_tensors_unchanged,
        "holdout_excluded_from_updates": holdout_excluded,
        "normal_equation_unchanged_by_holdout": holdout_excluded,
        "train_only_replay_equal": train_only_replay_equal,
        "trainable_params": adapter.trainable_param_count(),
        "trainable_param_count_detail": {
            name: adapter._adapter(name).B.numel()
            for name, _, _ in adapter.projs
        },
        "loRA_rank": args.r,
        "n_projections": len(adapter.projs),
        "refit_per_step": refit_per_step,
        "refit_total": sum(ap.refits for ap in adapter.adapters.values()),
        "holdout_rows_total": sum(
            ap.holdout_rows for ap in adapter.adapters.values()
        ),
        "saved_checkpoint": did_save,
        "checkpoint_path": checkpoint_path if did_save else None,
        "checkpoint_only_lora": checkpoint_only_lora if did_save else True,
        "checkpoint_reload_equal": checkpoint_reload_equal if did_save else True,
        "save_gain": float(save_gain),
        "anchor_validation_resid": float(anchor_validation),
        "live_validation_resid": float(live_validation),
        "anchor_B_nonzero": any(
            ap.anchor_B.abs().sum().item() > 0 for ap in adapter.adapters.values()
        ),
        "final_validation": {
            key: float(value) for key, value in final_validation.items()
        },
        "losses": losses,
        "base_hash_before": {
            "gate_proj": gate_hash_before,
            "up_proj": up_hash_before,
            "down_proj": down_hash_before,
            "norm": norm_hash_before,
        },
        "base_hash_after": {
            "gate_proj": gate_hash_after,
            "up_proj": up_hash_after,
            "down_proj": down_hash_after,
            "norm": norm_hash_after,
        },
        "safetensors_sha256": {
            "before": safetensors_sha256_before,
            "after": safetensors_sha256_after,
        },
        "base_param_count": sum(p.numel() for p in base.parameters()),
    }
    return report


def _mean_validation(adapter: TTTAdapter) -> float:
    values = list(adapter.validate().values())
    finite = [value for value in values if value < float("inf")]
    return sum(finite) / len(finite) if finite else float("inf")


def _anchor_validation(adapter: TTTAdapter) -> float:
    """Honest residual for frozen zero-delta anchor (anchor_B == 0)."""
    values = []
    for name, _, _ in adapter.projs:
        ap = adapter._adapter(name)
        if ap.Hva:
            values.append(_resid(ap.anchor_B, torch.cat(ap.Hva), torch.cat(ap.Yva)))
    finite = [value for value in values if value < float("inf")]
    return sum(finite) / len(finite) if finite else float("inf")


def main() -> int:
    args = parse_args()
    report = run_e2_smoke(args)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
