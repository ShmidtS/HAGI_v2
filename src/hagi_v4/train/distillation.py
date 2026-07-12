"""Knowledge distillation — embedding transfer + online KL distillation.

Ported from HAGI v1's distillation.py. Two mechanisms:
1. transfer_embeddings: copy SmolLM2-135M embedding weights into student
2. DistillationTeacher: online KL distillation from SmolLM2-360M

KL loss: L = alpha * CE + (1 - alpha) * T^2 * KL(softmax(s/T) || softmax(t/T))
Alpha schedule: linear ramp alpha_start -> alpha_end over distill phase, then 1.0.
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def transfer_embeddings(model: nn.Module, teacher_name: str = "HuggingFaceTB/SmolLM2-135M") -> bool:
    """Copy embedding weights from SmolLM2-135M into student model.

    Requires vocab_size == 49152 and hidden_size == 576 (SmolLM2-135M match).
    Teacher is loaded, embeddings copied, then immediately freed.
    """
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError:
        logger.warning("transformers not installed — skipping embedding transfer")
        return False

    try:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.bfloat16, local_files_only=True)
    except Exception:
        try:
            teacher = AutoModel.from_pretrained(teacher_name, torch_dtype=torch.bfloat16, local_files_only=True)
        except Exception as e:
            logger.warning(f"Could not load {teacher_name}: {e}")
            return False

    # Try common embedding attribute paths
    teacher_emb = None
    for path in [
        lambda m: m.model.embed_tokens.weight,
        lambda m: m.embed_tokens.weight,
        lambda m: m.get_input_embeddings().weight,
    ]:
        try:
            teacher_emb = path(teacher).data
            break
        except (AttributeError, KeyError):
            continue
    if teacher_emb is None:
        logger.warning(f"Could not find embeddings in {teacher_name} — skipping")
        del teacher
        gc.collect()
        return False

    embed_weight = model.embed.weight  # [V_student, H]

    if teacher_emb.shape[1] != embed_weight.shape[1]:
        logger.warning(f"Hidden size mismatch: student {embed_weight.shape} vs teacher {teacher_emb.shape} — skipping")
        del teacher
        gc.collect()
        return False

    copy_size = min(teacher_emb.shape[0], embed_weight.shape[0])
    embed_weight.data[:copy_size] = teacher_emb[:copy_size].to(embed_weight.dtype)
    extra = embed_weight.shape[0] - teacher_emb.shape[0]
    if extra > 0:
        logger.info(f"Embedding transferred {copy_size}/{embed_weight.shape[0]} tokens ({extra} extra kept)")
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
        except ImportError:
            logger.warning("transformers not installed — distillation disabled")
            return

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.teacher_name, torch_dtype=torch.bfloat16, local_files_only=True
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
        except Exception:
            try:
                self._model = AutoModel.from_pretrained(
                    self.teacher_name, torch_dtype=torch.bfloat16, local_files_only=True
                )
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad_(False)
                self._base_model = self._model
                self._lm_head_weight = self._model.get_input_embeddings().weight
                self._loaded = True
                logger.info(f"Teacher loaded (AutoModel): {self.teacher_name}")
            except Exception as e:
                logger.warning(f"Could not load teacher {self.teacher_name}: {e}")

    @torch.no_grad()
    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Run teacher forward, return hidden states [B, T, H_teacher]."""
        self._load()
        if not self._loaded:
            return None
        device = input_ids.device
        base_model = self._base_model.to(device)
        try:
            output = base_model(input_ids)
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                return output.hidden_states[-1]
            if isinstance(output, tuple):
                return output[0]
            return output
        except Exception as e:
            logger.warning(f"Teacher forward failed: {e}")
            return None

    def distillation_loss_chunked(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        ce_loss: torch.Tensor,
        mask: torch.Tensor | None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Compute alpha * CE + (1 - alpha) * T^2 * KL(student || teacher).

        teacher_hidden must be pre-computed (under inference_mode by caller).
        Chunks over sequence to avoid materializing full [B*T, V] logits.
        """
        if teacher_hidden is None:
            return ce_loss

        B, T, _ = student_hidden.shape
        flat_sh = student_hidden.reshape(B * T, -1)
        flat_th = teacher_hidden.reshape(B * T, -1).to(student_hidden.dtype)

        if mask is not None:
            flat_mask = mask.reshape(B * T)
        else:
            flat_mask = torch.ones(B * T, dtype=torch.bool, device=student_hidden.device)

        valid = flat_mask.sum()
        if valid == 0:
            return ce_loss

        total_kl = flat_sh.new_zeros((), dtype=torch.float32)
        for i in range(0, flat_sh.size(0), chunk_size):
            end = min(i + chunk_size, flat_sh.size(0))
            sh_c = flat_sh[i:end]
            th_c = flat_th[i:end]
            mask_c = flat_mask[i:end]

            s_logits = F.linear(sh_c, student_lm_head_weight.to(sh_c.dtype))
            t_logits = F.linear(th_c, self._lm_head_weight.to(th_c.device, dtype=th_c.dtype))

            # Align vocab sizes: use min(s, t) dimensions
            v_min = min(s_logits.shape[-1], t_logits.shape[-1])
            s_logits = s_logits[..., :v_min]
            t_logits = t_logits[..., :v_min]

            s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)
            t_soft = F.softmax(t_logits / temperature, dim=-1)

            kl = F.kl_div(s_log_soft, t_soft, reduction="none").sum(dim=-1)
            kl = kl * mask_c.float()
            total_kl = total_kl + kl.sum()

        kl_loss = total_kl / valid
        return alpha * ce_loss + (1.0 - alpha) * (temperature * temperature) * kl_loss

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
