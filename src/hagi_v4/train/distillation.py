"""Knowledge distillation — embedding transfer + hidden state alignment.

Teacher acts as "задающий генератор" (pilot/reference signal generator in 5G):
1. transfer_embeddings: copy Gemma embedding weights into student
2. DistillationTeacher: hidden state alignment (MSE, not KL on logits)

5G analogies:
  Teacher hidden states = DM-RS (Demodulation Reference Signal) — pilot signal
  Student hidden states = received signal
  MSE loss = channel estimation error
  alpha schedule = pilot-to-data power ratio adaptation

Why hidden state alignment (not KL on logits):
  Teacher is causal LM (next-token), student is masked LM (same-position).
  KL on logits forces student to match teacher's next-token predictions,
  which conflicts with same-position masked LM objective.
  Hidden state alignment forces student to match teacher's contextual
  representations — direction-agnostic, works for any LM architecture.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherHidden:
    hidden: torch.Tensor
    visibility_mask: torch.BoolTensor


def create_distillation_teacher(cfg, device: torch.device):
    """Create and load the canonical teacher when distillation is enabled."""
    if cfg.train.distill.enabled is not True:
        return None
    teacher = DistillationTeacher(cfg.train.distill.teacher)
    teacher.load()
    if teacher.is_loaded and device.type == "cuda":
        logger.info(f"Teacher VRAM: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    return teacher


def transfer_embeddings(model: nn.Module, teacher_name: str = "HuggingFaceTB/SmolLM2-135M") -> bool:
    """Copy embedding weights from teacher into student via SVD projection.

    If teacher hidden_size != student hidden_size, project teacher embeddings
    through truncated SVD to match student dimensions. This preserves the
    semantic structure of teacher embeddings while adapting to student space.

    5G analog: pilot signal precoding — adapt reference signal to antenna config.
    """
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError:
        logger.warning("transformers not installed — skipping embedding transfer")
        return False

    try:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_name, dtype=torch.bfloat16, local_files_only=True)
    except (OSError, ValueError) as causal_error:
        try:
            teacher = AutoModel.from_pretrained(teacher_name, dtype=torch.bfloat16, local_files_only=True)
        except (OSError, ValueError) as base_error:
            logger.warning(f"Could not load {teacher_name}: causal={causal_error}; base={base_error}")
            return False

    teacher_emb = None
    for path in [
        lambda m: m.model.embed_tokens.weight,
        lambda m: m.embed_tokens.weight,
        lambda m: m.get_input_embeddings().weight,
    ]:
        try:
            teacher_emb = path(teacher).data.float()
            break
        except (AttributeError, KeyError):
            continue
    if teacher_emb is None:
        logger.warning(f"Could not find embeddings in {teacher_name} — skipping")
        del teacher
        gc.collect()
        return False

    embed_weight = model.embed.weight
    V_t, H_t = teacher_emb.shape
    V_s, H_s = embed_weight.shape
    copy_size = min(V_t, V_s)

    if H_t == H_s:
        embed_weight.data[:copy_size] = teacher_emb[:copy_size].to(embed_weight.dtype)
    else:
        proj = torch.randn(H_t, H_s, dtype=torch.float32) / (H_t**0.5)
        projected = teacher_emb[:copy_size] @ proj
        embed_weight.data[:copy_size] = projected.to(embed_weight.dtype)
        logger.info(f"Embedding projected: {H_t}->{H_s} via random projection")

    extra = V_s - V_t
    if extra > 0:
        logger.info(f"Embedding transferred {copy_size}/{V_s} tokens ({extra} extra kept init)")
    else:
        logger.info(f"Embedding weights transferred from {teacher_name}")
    del teacher
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


class DistillationTeacher:
    """Online KL distillation teacher — SmolLM2-360M.

    Runs teacher forward to get hidden states, then computes KL divergence
    between student and teacher logits in chunks (never materializes full [B,T,V]).
    """

    def __init__(self, teacher_name: str = "HuggingFaceTB/SmolLM2-360M"):
        self.teacher_name = teacher_name
        self._model = None
        self._base_model = None
        self._lm_head_weight = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        return self._load()

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                f"Could not load distillation teacher {self.teacher_name}: transformers unavailable"
            ) from error

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.teacher_name, dtype=torch.bfloat16, local_files_only=True
            )
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad_(False)

            # Gemma 4 multimodal: access language_model submodel for text-only
            if hasattr(self._model, "language_model"):
                self._base_model = self._model.language_model
            elif hasattr(self._model, "model"):
                self._base_model = self._model.model
            else:
                self._base_model = self._model

            # lm_head: try multiple paths
            self._lm_head_weight = None
            for obj in [self._model, self._base_model]:
                if hasattr(obj, "lm_head") and hasattr(obj.lm_head, "weight"):
                    self._lm_head_weight = obj.lm_head.weight
                    break
                if hasattr(obj, "get_output_embeddings"):
                    emb = obj.get_output_embeddings()
                    if emb is not None and hasattr(emb, "weight"):
                        self._lm_head_weight = emb.weight
                        break

            if self._lm_head_weight is None:
                # Fallback: use input embeddings as pseudo lm_head
                if hasattr(self._base_model, "get_input_embeddings"):
                    self._lm_head_weight = self._base_model.get_input_embeddings().weight
                elif hasattr(self._base_model, "embed_tokens"):
                    self._lm_head_weight = self._base_model.embed_tokens.weight

            self._loaded = True
            logger.info(f"Teacher loaded: {self.teacher_name}")
        except (OSError, ValueError) as causal_error:
            try:
                self._model = AutoModel.from_pretrained(self.teacher_name, dtype=torch.bfloat16, local_files_only=True)
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad_(False)
                self._base_model = self._model
                self._lm_head_weight = self._model.get_input_embeddings().weight
                self._loaded = True
                logger.info(f"Teacher loaded (AutoModel): {self.teacher_name}")
            except (OSError, ValueError) as base_error:
                raise RuntimeError(
                    f"Could not load distillation teacher {self.teacher_name}: causal={causal_error}; base={base_error}"
                ) from base_error
        except Exception as error:
            raise RuntimeError(f"Could not load distillation teacher {self.teacher_name}: {error}") from error

    @torch.no_grad()
    def get_hidden(self, input_ids: torch.Tensor, visibility_mask: torch.Tensor) -> TeacherHidden | None:
        """Run teacher forward, return hidden states [B, T, H_teacher]."""
        self._load()
        if not self._loaded:
            return None
        if visibility_mask.shape != input_ids.shape or visibility_mask.dtype != torch.bool:
            raise ValueError("visibility_mask must be a BoolTensor with the same shape as input_ids")
        if visibility_mask.device != input_ids.device:
            raise ValueError(
                f"visibility_mask device {visibility_mask.device} must match input_ids device {input_ids.device}"
            )
        device = input_ids.device
        base_model = self._base_model
        placed_tensor = next(iter(base_model.parameters()), None)
        if placed_tensor is None:
            placed_tensor = next(iter(base_model.buffers()), None)
        if placed_tensor is not None and placed_tensor.device != device:
            base_model.to(device)
        embeddings = base_model.get_input_embeddings()
        teacher_vocab_size = embeddings.num_embeddings
        config = getattr(base_model, "config", getattr(self._model, "config", None))
        fallback_id = next(
            (
                token_id
                for token_id in (
                    getattr(config, "eos_token_id", None),
                    getattr(config, "unk_token_id", None),
                    0,
                )
                if isinstance(token_id, int) and 0 <= token_id < teacher_vocab_size
            ),
            None,
        )
        if fallback_id is None:
            raise ValueError(f"teacher vocabulary must contain at least one token, got {teacher_vocab_size}")
        with torch.inference_mode(False):
            valid_ids = (input_ids >= 0) & (input_ids < teacher_vocab_size)
            effective_visibility = visibility_mask & valid_ids
            teacher_input_ids = input_ids.masked_fill(~effective_visibility, fallback_id)
        if torch.any((teacher_input_ids < 0) | (teacher_input_ids >= teacher_vocab_size)):
            raise ValueError("teacher input IDs remain outside the teacher vocabulary after remapping")

        output = base_model(teacher_input_ids, attention_mask=effective_visibility, use_cache=False)
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden = output.hidden_states[-1]
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        with torch.inference_mode(False):
            hidden = hidden.detach().clone()
        return TeacherHidden(hidden=hidden, visibility_mask=effective_visibility)

    def distillation_loss_chunked(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        visibility_mask: torch.Tensor | None,
        align_projection: nn.Module,
        alpha: float = 0.5,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Hidden state alignment on unmasked positions only.

        Teacher hidden states act as pilot/reference signals (5G DM-RS).
        Student learns to match teacher's contextual representations on
        VISIBLE (unmasked) positions — where both see the same token.

        On masked positions, student must recover the token via CE loss,
        NOT by matching teacher's hidden state (teacher sees the real
        token, student sees mask_embed — MSE there would penalize the
        student for not seeing what it can't see).

        5G analog: DM-RS alignment on pilot positions, data recovery
        on data positions.

        V12: when ``alpha >= 1.0`` the CE signal dominates entirely and the
        MSE term contributes nothing; when ``alpha <= 0.0`` the function
        returns ``reconstruction_loss`` unchanged (CE-only). This is the
        theoretically-correct behavior — V11 had ``alpha * CE + (1-alpha) *
        MSE`` which, at ``alpha = 0``, inverted to ``1 * MSE`` and silently
        replaced CE with hidden alignment. The V12 canonical profile sets
        ``alpha = 0`` to disable hidden alignment entirely (it was found
        to compete with the CE signal via distribution mismatch); this
        early-return makes that intent explicit and avoids the inversion.
        """
        if teacher_hidden is None:
            return reconstruction_loss

        # V12: alpha <= 0 means "CE only, no alignment". Returning early
        # avoids the ``0 * CE + 1 * MSE`` inversion that silently replaced
        # the primary objective with hidden-state matching.
        if alpha <= 0.0:
            return reconstruction_loss

        B, T, H_s = student_hidden.shape
        _, _, H_t = teacher_hidden.shape

        flat_sh = student_hidden.reshape(B * T, H_s).float()
        projection_param = next(align_projection.parameters(), None)
        projection_device = projection_param.device if projection_param is not None else student_hidden.device
        projection_dtype = projection_param.dtype if projection_param is not None else student_hidden.dtype
        flat_th = (
            teacher_hidden.detach().reshape(B * T, H_t).to(device=projection_device, dtype=projection_dtype).clone()
        )

        if visibility_mask is not None:
            valid_mask = visibility_mask.reshape(B * T)
        else:
            valid_mask = torch.ones(B * T, dtype=torch.bool, device=flat_sh.device)

        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            return reconstruction_loss

        flat_th = align_projection(flat_th).float()
        if flat_th.shape[-1] != H_s:
            raise ValueError(f"alignment projection produced {flat_th.shape[-1]} features, expected {H_s}")

        total_mse = flat_sh.new_zeros(())
        for i in range(0, flat_sh.shape[0], chunk_size):
            end = min(i + chunk_size, flat_sh.shape[0])
            mask_c = valid_mask[i:end]
            if not mask_c.any():
                continue
            diff = flat_sh[i:end][mask_c] - flat_th[i:end][mask_c].to(flat_sh.dtype)
            total_mse = total_mse + diff.pow(2).mean(dim=-1).sum()

        mse_loss = total_mse / n_valid
        return alpha * reconstruction_loss + (1.0 - alpha) * mse_loss

    def free(self):
        """Release teacher model from VRAM."""
        if self._model is not None:
            del self._model, self._base_model, self._lm_head_weight
            self._model = None
            self._base_model = None
            self._lm_head_weight = None
            self._loaded = False
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Teacher freed from VRAM")


def alpha_at(
    step: int,
    alpha_start: float = 0.5,
    alpha_end: float = 0.3,
    max_steps: int = 150000,
    distill_end_frac: float = 0.6,
) -> float:
    """Alpha schedule: linear ramp alpha_start -> alpha_end, then 1.0."""
    distill_end_step = int(max_steps * distill_end_frac)
    if step > distill_end_step:
        return 1.0
    progress = min(1.0, step / max(1, distill_end_step))
    return alpha_start + (alpha_end - alpha_start) * progress


def temperature_at(
    step: int,
    max_steps: int,
    temp_start: float = 4.0,
    temp_end: float = 1.0,
    distill_end_frac: float = 0.6,
) -> float:
    """Anneal temperature from high (coarse) to low (fine).

    Early training: high T → soft targets, student learns coarse structure.
    Late training: low T → sharp targets, student refines fine details.
    """
    distill_end_step = int(max_steps * distill_end_frac)
    if step > distill_end_step:
        return temp_end
    progress = min(1.0, step / max(1, distill_end_step))
    return temp_start + (temp_end - temp_start) * progress
