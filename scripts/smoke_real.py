"""One-shot smoke: real corpus pipeline, tiny geometry, a few optimizer steps.

Not part of the test suite — this needs data/*.bin on disk and a real unigram
prior. Verifies the full path: pack -> doc_ids -> forward/backward -> optimizer
-> controller -> diagnostics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.model import HAGI
from hagi.train.loop import Trainer

cfg = load_config(
    "configs/v31_1b.yaml",
    **{
        "model.hidden_size": 128,
        "model.num_layers": 4,
        "model.vocab_size": 262144,
        "model.attention.num_query_heads": 4,
        "model.attention.num_kv_heads": 2,
        "model.attention.head_dim": 32,
        "model.attention.max_seq_len": 128,
        "model.sliding.window": 32,
        "model.sliding.full_every": 4,
        "model.ffn.multiple_of": 32,
        "model.head.ce_chunk_rows": 256,
        "train.batch_size": 2,
        "train.grad_accum_steps": 1,
        "train.data.seq_len": 64,
        "train.data.num_workers": 0,
        "train.max_steps": 4,
        "train.schedule.warmup_steps": 2,
        "train.grad_checkpointing": False,
        "train.logging.log_interval": 1,
        "train.logging.diag_interval": 1,
        "train.checkpoint_interval": 0,
    },
)
cfg.model.head.unigram_prior = True

model = HAGI(cfg)
trainer = Trainer(model, cfg)
dataloader = build_dataloader(cfg)

step = 0
for batch in dataloader:
    if step >= 4:
        break
    metrics = trainer.train_step([batch])
    print(
        f"step {metrics['step']} ce={metrics['ce']:.4f} "
        f"qk={metrics.get('qk_gain', -1):.3f} lts={metrics.get('logit_scale', -1):.4f} "
        f"tok={metrics['tokens']}"
    )
    step += 1

print("baseline = 8.05 (unigram entropy at 262144 vocab)")
