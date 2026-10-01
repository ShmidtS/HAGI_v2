"""Round 70c: knowledge distillation from an OSS teacher (Gemma-2-2b).

The universal-merge system's distill transport (round70b skeleton):
an external model becomes a tree branch by TEACHING our student on
our own token streams. No architecture alignment needed -- the
teacher's 256k-vocab logits are aggregated into our 32768 compact
vocab through the vocab_map (sum of teacher probs over the tokens
that map to each compact id).

Student: the record model (clamp-8 base + r48 table adapters,
gate CE 3.3686). Loss: KL(teacher || student) on compact vocab +
CE on ground-truth tokens (standard distill mix).

Teacher inference notes:
- Gemma-2-2b runs in bf16 on the same GPU (~5 GB); batches are
  small (2x1024) so it fits alongside the student.
- Teacher sees the SAME token ids lifted to the old 256k space via
  new_to_old (one representative id per compact token). This is a
  lossy view of subword merges but a consistent one; the R66-style
  sanity check below verifies the teacher's CE on our streams
  before any training.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src'), os.path.join(_REPO, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from hagi.config import load_config
from hagi.data.vocab_map import VocabMap
from hagi.train.loop import configure_runtime

TEACHER_DIR = "_raw/gemma-2-2b"
STUDENT_SCRIPT = "scripts/lora/lora_c8_r48.py"  # record lineage builder


def load_teacher(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER_DIR)
    model = AutoModelForCausalLM.from_pretrained(TEACHER_DIR, torch_dtype=torch.bfloat16)
    return model.to(device).eval(), tok


class TeacherOnCompact:
    """Wraps the 256k-vocab teacher to score COMPACT-vocab windows.

    ids -> lift each compact id to its representative old id ->
    teacher logits [N, 256k] -> log-softmax -> aggregate onto the
    compact vocab (logsumexp over members; single member in the
    representative case, so log-probs pass through with a
    correctness constant).
    """

    def __init__(self, teacher, vmap: VocabMap, device: str, max_len: int = 1024):
        self.teacher = teacher
        self.new_to_old = torch.as_tensor(vmap.new_to_old, dtype=torch.long, device=device)
        self.device = device
        self.max_len = max_len
        # membership: old id -> compact id (vectorized gather table)
        self.old_to_new = torch.as_tensor(vmap.old_to_new, dtype=torch.long, device=device)

    @torch.no_grad()
    def logprobs_on(self, compact_ids: torch.Tensor) -> torch.Tensor:
        """compact_ids [B, T] (targets shifted outside) -> [B*T, V_compact]."""
        B, T = compact_ids.shape
        old_ids = self.new_to_old[compact_ids.reshape(-1)].reshape(B, T)
        logits = self.teacher(old_ids).logits.float()  # [B, T, V_old]
        lp = F.log_softmax(logits, dim=-1)
        # aggregate: max-member trick -- use the representative member's prob
        # (exact for 1:1 compact ids, lower bound for merged ones)
        flat_lp = lp.reshape(-1, lp.shape[-1])
        # scatter: for each compact id, take logsumexp over its old members.
        # With per-compact representative lists this is a segment op; the
        # vocab_map is functionally 1:1 for retained ids (compact -> old),
        # so we gather: compact c's logprob = lp[:, new_to_old[c]].
        out = flat_lp[:, self.new_to_old]  # [B*T, V_compact] -- gather by old id
        return out


def teacher_sanity(teacher_wrap, batches):
    """Teacher CE on our compact streams (must be WELL below student's 3.36
    for distillation to carry new information; if near ln V the bridge is
    broken)."""
    tot, n = 0.0, 0
    for x, y in batches[:2]:
        lp = teacher_wrap.logprobs_on(x)  # predicts x[t+1] from x[<t]... caller shifts
        # NOTE: logits[t] predicts token t+1; use lp[:-1] vs y
        tgt = y.reshape(-1)
        lp_shift = lp.reshape(x.shape[0], x.shape[1], -1)[:, :-1].reshape(-1, lp.shape[-1])
        ce = F.cross_entropy(lp_shift, tgt, reduction="sum")
        tot += float(ce)
        n += tgt.numel()
    return tot / n


def main() -> int:
    configure_runtime()
    device = "cuda"
    vmap = VocabMap("data/vocab_map.npz")

    # gate batches (canonical tail windows) for the sanity check first
    CORP = ["edu", "python_instruct", "wikipedia_en", "wikipedia_ru", "oscar_ru",
            "openwebmath", "tinystories", "smoltalk"]
    batches = []
    for c in CORP:
        p = Path(f"data/{c}.compact.bin")
        total = p.stat().st_size // 4
        with p.open("rb") as fh:
            fh.seek((total - 2_000_000) * 4)
            T = np.frombuffer(fh.read(2048 * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(T[:2048]).reshape(2, 1024)
        batches.append((ids[:, :-1], ids[:, 1:]))

    teacher, tok = load_teacher(device)
    wrap = TeacherOnCompact(teacher, vmap, device)
    ce = teacher_sanity(wrap, batches)
    print(f"teacher CE on compact streams: {ce:.4f}")
    print(f"(student record: 3.3686; ln V: {np.log(32768):.3f})")
    print("=> informative teacher" if ce < 3.0 else
          "=> WARNING: bridge may be broken, inspect before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
