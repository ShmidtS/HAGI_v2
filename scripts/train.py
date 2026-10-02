"""Training entry point. Every parameter comes from the YAML config.

Usage:
    python scripts/train.py --config configs/level0_ab/ru_baseline.yaml
    python scripts/train.py --config configs/level0_ab/ru_baseline.yaml --dry-run
    python scripts/train.py --config configs/level0_ab/ru_baseline.yaml --resume
    python scripts/train.py --config configs/level0_ab/ru_baseline.yaml --steps 500 --profile 3
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `hagi` importable when run directly, before any hagi import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "level0_ab" / "ru_baseline.yaml"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Must precede `import torch`: the allocator and the ROCm kernel selection read
# these at initialization.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402

# ROCm Windows torch ships without torch._C._distributed_c10d, so transformers'
# eager FSDP/DTensor imports crash at import time. No-op elsewhere.
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train._rocm_fsdp_stub import install as install_rocm_stub  # noqa: E402

install_rocm_stub()

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> str:
    """Log to stderr and to a timestamped file. Returns the file path."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = f"{log_dir}/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dry_run(model, cfg, device: torch.device) -> int:
    """One forward/backward on synthetic data: shapes, loss sanity, peak memory.

    The check that matters is ``ce`` against the right baseline, and the baseline
    depends on how the targets were drawn:

    * Targets drawn **uniformly** — an untrained model without a source prior sits
      at ``ln V``. With a unigram prior it sits far *above* ``ln V``, because a
      uniformly drawn target is by construction improbable under the corpus
      marginal. That is correct behaviour, not a failure.
    * Targets drawn from the **unigram distribution** — the model sits at the
      unigram entropy, which is the honest zero-order baseline and the number the
      real corpus will start from.

    So the targets are sampled from the model's own prior when one is loaded. A
    value materially above the corresponding baseline means the head or the target
    alignment is wrong, and catching that here costs seconds instead of hours.
    """
    import math

    from hagi.train.loop import cast_model

    cast_model(
        model,
        cfg.train.precision,
        ternary_fp32_master=cfg.train.ternary_fp32_master,
    )
    batch, seq = 2, min(cfg.train.data.seq_len, 256)
    ids = torch.randint(0, cfg.model.vocab_size, (batch, seq), device=device)
    doc_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)
    doc_ids[:, seq // 2 :] = 1

    prior = getattr(model.head, "log_prior", None)
    if prior is not None:
        probs = prior.detach().float().exp()
        targets = torch.multinomial(probs, batch * seq, replacement=True).view(batch, seq).to(device)
        baseline = float(-(probs * prior.detach().float()).sum())
        baseline_name = "unigram entropy"
    else:
        targets = torch.randint(0, cfg.model.vocab_size, (batch, seq), device=device)
        baseline = math.log(cfg.model.vocab_size)
        baseline_name = "ln V"

    model.train()
    output = model(ids, targets, doc_ids=doc_ids)
    ce = float(output.ce.detach())
    receiver = "nce" if cfg.model.head.sampled_softmax_k > 0 else "ce"
    exact_ce = None
    if cfg.model.head.sampled_softmax_k > 0:
        rows = min(int(cfg.train.logging.exact_ce_rows), batch * seq)
        flat_hidden = output.hidden.detach().reshape(-1, output.hidden.shape[-1])[:rows]
        flat_targets = targets.reshape(-1)[:rows]
        with torch.no_grad():
            exact_ce = float(model.head.exact_loss(flat_hidden, flat_targets).detach())
    output.loss.backward()

    logger.info(
        "dry run: loss=%.4f %s=%.4f exact_ce=%s (%s baseline %.4f) z=%.4f tokens=%d",
        float(output.loss.detach()),
        receiver,
        ce,
        f"{exact_ce:.4f}" if exact_ce is not None else "n/a",
        baseline_name,
        baseline,
        float(output.z_loss.detach()),
        output.n_tokens,
    )
    measured_ce = exact_ce if exact_ce is not None else ce
    if measured_ce > baseline * 1.10:
        logger.warning("exact CE exceeds the %s baseline by >10%% at init — check target alignment", baseline_name)

    without_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if without_grad:
        logger.error("parameters received no gradient: %s", without_grad[:20])
        return 1
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        logger.error("no trainable parameters; the run cannot train")
        return 1
    logger.info("all %d trainable parameter tensors received gradient "
                "(of %d total)", len(trainable), len(list(model.parameters())))

    if device.type == "cuda":
        logger.info("peak VRAM: %.3f GB", torch.cuda.max_memory_allocated() / 1e9)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the HAGI channel language model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=None, help="overrides train.data.data_dir")
    parser.add_argument("--steps", type=int, default=None, help="overrides train.max_steps")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, help="path, or bare flag for the latest"
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="load weights as initialization (step 0, fresh optimizer) instead of resuming",
    )
    parser.add_argument(
        "--no-optimizer-state",
        action="store_true",
        help="resume model weights but skip the optimizer state (e.g. Phase 1 mixer-only -> Phase 2 unfreeze, where the parameter-group count differs)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="token offset into the corpus to begin reading from (recursive growth: the next expert continues where the previous one stopped)",
    )
    parser.add_argument("--profile", type=int, default=0, help="profile the first N steps; 0 = off")
    return parser


def default_config_path_overrides(config_path: str | Path) -> dict[str, object]:
    """Resolve bundled data paths when the built-in config is used off-repo."""
    if Path(config_path).resolve() != DEFAULT_CONFIG.resolve():
        return {}
    return {
        "model.head.unigram_path": str(_REPO_ROOT / "data" / "unigram.compact.npy"),
        "train.data.data_dir": str(_REPO_ROOT / "data"),
        "train.checkpoint_dir": str(_REPO_ROOT / "checkpoints_l0_ab" / "ru_baseline"),
    }


def main() -> int:
    args = build_parser().parse_args()

    log_path = setup_logging(args.log_dir)
    logger.info("log file: %s", log_path)

    from hagi.config import describe, load_config
    from hagi.model.model import HAGI
    from hagi.version import __architecture__, __version__

    overrides: dict[str, object] = default_config_path_overrides(args.config)
    if args.steps is not None:
        overrides["train.max_steps"] = args.steps
    if args.checkpoint_dir is not None:
        overrides["train.checkpoint_dir"] = args.checkpoint_dir
    if args.data_dir is not None:
        overrides["train.data.data_dir"] = args.data_dir
    cfg = load_config(args.config, **overrides)

    device = resolve_device(args.device)
    logger.info("hagi %s (%s) | device %s | config %s", __version__, __architecture__, device, args.config)
    for line in describe(cfg).splitlines():
        logger.info("  %s", line)

    # Same-initialization for expert merge: every expert must start from the
    # same random weights so the block-diagonal merge is a faithful
    # concatenation of N subspaces grown from one shared prior. Seed the RNG
    # immediately before constructing the model. init_seed=0 leaves the current
    # RNG state untouched (no seeding).
    resume_path = None
    if args.resume is not None:
        from hagi.train.checkpoint import latest_checkpoint, load_payload

        resume_path = args.resume if args.resume != "latest" else latest_checkpoint(
            cfg.train.checkpoint_dir
        )
        if resume_path is None:
            raise FileNotFoundError(f"--resume: no step-*.pt in {cfg.train.checkpoint_dir}")

    # Seed unconditionally when the config carries one. The old guard was
    # ``if cfg.model.init_seed:``, which silently skipped the value 0 -- and 0
    # is the default in every F3 config, so no leaf was ever seeded. With the
    # guard, three leaves that differed only in init_seed still received the
    # same RNG stream and their weights came out byte-identical: the expert
    # symmetry that Net2Net and the MoE-upcycling literature both warn about,
    # and which the F3 lift cannot break on its own.
    if cfg.model.init_seed is not None:
        torch.manual_seed(cfg.model.init_seed)
    if resume_path is not None:
        from hagi.model.merge import build_model_from_payload
        from hagi.train.checkpoint import config_from_dict, load_payload

        resume_payload = load_payload(resume_path, str(device))
        model = build_model_from_payload(
            config_from_dict(resume_payload["config"]),
            resume_payload["model"],
            n_mixers=1,
            mixer_init_scale=cfg.merge.mixer_init_scale,
            device=device,
        )
    else:
        if cfg.merge.enabled and cfg.merge.mixer_type == "ternary_f3":
            # Recursive ternary tree. Children are themselves trained bodies --
            # leaves at depth 1, lifted parents deeper in. merge_recursive_f3
            # fingerprints each child's config to prove the three share one
            # architecture, so the config is read back from the checkpoint it
            # was trained with rather than from this YAML: this YAML describes
            # the parent, which is three times wider than any child.
            from hagi.model.merge import merge_recursive_f3
            from hagi.train.checkpoint import config_from_dict, load_payload

            payloads = [
                load_payload(p, str(device)) for p in cfg.merge.expert_checkpoints
            ]
            model = merge_recursive_f3(
                cfg,
                [pl["model"] for pl in payloads],
                child_configs=[config_from_dict(pl["config"]) for pl in payloads],
                drop_expert_mixers=False,
            ).to(device)
        elif cfg.merge.enabled:
            from hagi.model.merge import merge_experts

            if cfg.merge.expert_checkpoints:
                from hagi.train.checkpoint import load_payload

                states = [
                    load_payload(p, str(device))["model"]
                    for p in cfg.merge.expert_checkpoints
                ]
                model = merge_experts(
                    cfg,
                    states,
                    n_mixers=1,
                    mixer_init_scale=cfg.merge.mixer_init_scale,
                    # Hierarchical merge: the experts are themselves merged models
                    # with their own cross-block mixers, which must be dropped and
                    # replaced by a fresh level-N mixer. Harmless for plain experts
                    # (they have no ``mixers.*`` keys).
                    drop_expert_mixers=True,
                ).to(device)
                # Step-0 checkpoint: the joint configs of the winning line load
                # ``step-0000000.pt`` — the exact merged state (the Jensen
                # step-0 identity) BEFORE any training. Saving it here keeps
                # the merge artifact even if the merged run is interrupted.
                from hagi.train.checkpoint import save_checkpoint

                save_checkpoint(model, cfg, 0, cfg.train.checkpoint_dir)
            else:
                # No expert checkpoints configured: build the merged body from the
                # current (random) weights so the machinery is exercised. With
                # ``scratch_block_norm`` this is arm S: the merged arm's
                # architecture with random weights, which is the only way to
                # separate merging from conditioning (see
                # .omc/plans/clean_merge_test_20260926.md).
                # The class choice lives in hagi.model.factory so the evaluator
                # rebuilds the same architecture for a checkpoint written here.
                model = build_model_for_config(cfg).to(device)
        else:
            model = HAGI(cfg).to(device)
        # Optional zero-init of the residual projections (HAGI_ZERO_INIT_PROJ=1):
        # the NanoGPT speedrun's validated trick -- each block's output
        # projection starts at zero so the residual stream begins as an exact
        # pass-through and the blocks fade in from identity. On the speedrun's
        # 12-layer model this measurably helps; on a 3-layer leaf it may be
        # neutral, which is exactly what the flag lets us measure.
        if os.environ.get("HAGI_ZERO_INIT_PROJ", "") or bool(getattr(cfg.train, "zero_init_proj", False)):
            if args.init_from or (cfg.train.init_from or None):
                logger.warning(
                    "train.zero_init_proj is a no-op with init_from: the "
                    "prior's weights are loaded AFTER this zeroing (train.py "
                    "init-from path) -- remove the flag or rely on the prior."
                )
            with torch.no_grad():
                zeroed = 0
                for name, p in model.named_parameters():
                    if name.endswith("attn.out_proj.weight") or name.endswith("mixer.mixer.down.weight"):
                        p.zero_()
                        zeroed += 1
            logger.info("zero-init residual projections: %d tensors", zeroed)
        # Optional vocab slicing (``HAGI_VOCAB_SLICE="i:N"``): mask the
        # log_prior so the leaf only ever predicts tokens in its slice. The
        # input stays full-vocabulary -- every leaf sees the whole context;
        # only the OUTPUT side is sliced, which is what keeps concat assembly
        # valid (each joint logit row is owned by exactly one leaf block).
        # Implemented as a bias, not by deleting rows, so the parameter
        # count and the state dict stay unchanged and the merged parent can
        # be built by the ordinary concat path.
        slice_spec = os.environ.get("HAGI_VOCAB_SLICE", "")
        if slice_spec:
            index_str, count_str = slice_spec.split(":")
            si, sn = int(index_str), int(count_str)
            if sn < 1 or not 0 <= si < sn:
                raise ValueError(f"bad HAGI_VOCAB_SLICE: {slice_spec!r}")
            v = int(cfg.model.vocab_size)
            lo = si * v // sn
            hi = (si + 1) * v // sn
            logger.info("vocab slice %d/%d: tokens [%d, %d)", si, sn, lo, hi)
    counts = model.param_summary()
    logger.info(
        "parameters: total %.1fM | body %.1fM | embedding %.1fM | active body %.1fM",
        counts["total"] / 1e6,
        counts["body"] / 1e6,
        counts["embedding"] / 1e6,
        counts["active_body"] / 1e6,
    )

    # "Train only the Cross-Expert Mixer" mode: freeze every parameter except
    # the cross-block mixers. The experts' weights stay frozen; only the mixer
    # connections learn. Requires a merged model (merge.enabled).
    if cfg.merge.enabled and cfg.merge.freeze_experts:
        frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith("mixers."):
                param.requires_grad = False
                frozen += 1
        logger.info(
            "freeze_experts: froze %d parameter tensors; only mixers.* trainable",
            frozen,
        )

    start_step = 0
    optimizer_state = None
    # --resume takes precedence over any initialization. A resume continues a
    # run; an initialization starts a new one. The two are not alternatives at
    # the same priority: a config that sets train.init_from (every sibling
    # config does, to inherit the parent's shared prior) would otherwise
    # SILENTLY swallow --resume, and the supervisor -- which resumes from
    # whatever checkpoint is on disk -- would discard every trained step and
    # restart from the parent. Observed on gen-4: a checkpoint at step 1300
    # was on disk, the supervisor correctly passed --resume, and the run
    # reported "initialized weights from ... (fresh optimizer, step 0)".
    if args.resume is not None:
        from hagi.train.checkpoint import load_model

        path = resume_path
        start_step, _ = load_model(path, model, str(device))
        if args.no_optimizer_state:
            optimizer_state = None
        else:
            optimizer_state = load_payload(path, str(device)).get("optimizer")
        logger.info(
            "resumed %s at step %d (optimizer state: %s)",
            path,
            start_step,
            "yes" if optimizer_state else "no — expect a loss spike",
        )
    else:
        # --init-from takes precedence; otherwise fall back to the config's
        # train.init_from (the recursive-growth shared prior).
        init_from = args.init_from or (cfg.train.init_from or None)
        if init_from is not None:
            # Load weights as an *initialization*, not a resume: completed_steps
            # resets to 0 and the optimizer starts fresh. This is the recursive
            # growth primitive — a level-N expert starts from the merged level-(N-1)
            # model (the shared prior) and specializes on its own domain.
            from hagi.train.checkpoint import load_model

            path = init_from
            if not Path(path).exists():
                raise FileNotFoundError(f"--init-from: no checkpoint at {path}")
            # Skip the cross-expert mixers (``mixers.*``): the target model's mixers
            # may have a different geometry than the source (e.g. a fresh Hadamard
            # mixer initialized from a SwiGLU-merged prior). The body/embed/head
            # transfer; the mixers stay fresh so the Hadamard geometry is kept.
            # Also skip ``head.log_prior``: the unigram prior is a deterministic
            # function of the corpus (unigram_path), not learned state -- a target
            # that enables the prior rebuilds it from its own config, and a prior
            # without it must not inherit a stale one (round-28 NCE A/B).
            _, _ = load_model(
                path, model, str(device), skip_prefixes=("mixers.", "head.log_prior"),
                lenient_config=True,
            )
            logger.info("initialized weights from %s (fresh optimizer, step 0)", path)

    if args.dry_run:
        return dry_run(model, cfg, device)

    from hagi.data.dataset import build_dataloader
    from hagi.train.loop import format_metrics, train

    dataloader = build_dataloader(cfg, args.data_dir, start_offset=args.start_offset)
    tokens_per_step = cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.train.data.seq_len
    logger.info(
        "training %d steps | %d tokens/step | %.2fB tokens total",
        cfg.train.max_steps,
        tokens_per_step,
        tokens_per_step * cfg.train.max_steps / 1e9,
    )

    if args.profile > 0:
        from torch.profiler import ProfilerActivity, profile, schedule

        accum = cfg.train.grad_accum_steps
        trace = f"{args.log_dir}/profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.info("profiling %d steps (step 0 skipped: kernel autotune) -> %s", args.profile, trace)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=accum, active=max(accum * args.profile, 1), repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(trace),
        ) as prof:
            for index, metrics in enumerate(
                train(model, dataloader, cfg, start_step=start_step, optimizer_state=optimizer_state, start_offset=args.start_offset)
            ):
                logger.info(format_metrics(metrics))
                prof.step()
                if index >= args.profile:
                    break
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        return 0

    for metrics in train(model, dataloader, cfg, start_step=start_step, optimizer_state=optimizer_state, start_offset=args.start_offset):
        logger.info(format_metrics(metrics))
    logger.info("training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
