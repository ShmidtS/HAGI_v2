"""Smoke-test the ternary_f3 parent build: three trained leaves, no training.

Why this exists: `configs/f3_parent_d1.yaml` drives a code path that, until
2026-09-26, had never been scored end to end. `merge_recursive_f3` is not
reachable from `build_model_for_config` (factory.py:41-57 raises for
`ternary_f3` by design), so a crash here would only ever surface as a failure
inside a multi-hour GPU run. This script builds the parent, runs one forward
pass, and prints what actually happened -- including failures.

Two suspicions from code reading are checked explicitly, because neither is
visible from a log line:
  * `BlockTreeNorm` / `RecursiveBranchScale` are attached in
    `from_state_dict` (merge.py:1185-1205), not in `__init__`. A build that
    goes straight to forward would silently normalise wrongly rather than
    crash, so their presence is printed, not assumed.
  * `cortex.gate` is unset in the YAML, so the config default applies
    (config.py:389). The gate mode is printed, not assumed to be gumbel.

Usage:
    python scripts/f3_smoke.py
    python scripts/f3_smoke.py --config configs/f3_parent_d1_nocortex.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import merge_recursive_f3  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402


def say(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def count_named(model: torch.nn.Module, needle: str) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (name, tuple(module.weight.shape) if hasattr(module, "weight") else ())
        for name, module in model.named_modules()
        if needle in type(module).__name__
    ]


def run_one(config_path: str, leaf_ckpt: str, device: str, seq_len: int) -> bool:
    say(f"config={config_path}")
    cfg = load_config(config_path)
    checkpoints = list(cfg.merge.expert_checkpoints)
    say(f"children={len(checkpoints)} ternary_depth={cfg.merge.ternary_depth} "
        f"lift={cfg.merge.ternary_lift_mode}")
    say(f"cortex.enabled={cfg.model.cortex.enabled} mode={cfg.model.cortex.mode} "
        f"gate={cfg.model.cortex.gate!r}")

    payloads = [load_payload(path, device) for path in checkpoints]
    child_configs = [config_from_dict(pl["config"]) for pl in payloads]
    say(f"child hidden={[c.model.hidden_size for c in child_configs]} "
        f"parent hidden={cfg.model.hidden_size}")

    model = merge_recursive_f3(
        cfg,
        [pl["model"] for pl in payloads],
        child_configs=child_configs,
        drop_expert_mixers=False,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    say(f"model={type(model).__name__} params={params:,}")

    for needle in ("BlockTreeNorm", "RecursiveBranchScale", "Cortex"):
        found = count_named(model, needle)
        say(f"modules[{needle}]={len(found)} shapes={found[:4]}")

    ids = torch.randint(0, cfg.model.vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        out = model(ids, return_logits=True)
    logits = out.logits
    say(f"logits shape={tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} "
        f"min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f}")
    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bool(torch.isfinite(logits).all())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None,
                    help="repeatable; defaults to both parent configs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    configs = args.config or [
        "configs/f3_parent_d1.yaml",
        "configs/f3_parent_d1_nocortex.yaml",
    ]
    results: dict[str, bool] = {}
    for config_path in configs:
        try:
            results[config_path] = run_one(config_path, None, args.device, args.seq_len)
        except Exception:
            say(f"FAILED {config_path}")
            traceback.print_exc()
            results[config_path] = False

    say("summary: " + " | ".join(f"{k}={'ok' if v else 'fail'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
