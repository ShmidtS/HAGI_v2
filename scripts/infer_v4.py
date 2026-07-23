"""HAGI V4 inference — load checkpoint and generate text.

Usage:
    python scripts/infer_v4.py --checkpoint checkpoints/step-001000.pt
    python scripts/infer_v4.py --checkpoint checkpoints/step-001000.pt --prompt "Once upon a time"
    python scripts/infer_v4.py --checkpoint checkpoints/step-001000.pt --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load model + config from checkpoint. Keeps embedding/lm_head on CPU to save VRAM.

    Embedding table (83% of params, tied with lm_head) stays on CPU.
    Only 230M non-embed params go to GPU (~0.5 GB VRAM in bf16).
    Token IDs are moved to CPU for embedding lookup, hidden states back to GPU.
    """
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.checkpoint import cfg_from_dict, load_checkpoint_payload, load_model_checkpoint

    target = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    ckpt_payload = load_checkpoint_payload(checkpoint_path, "cpu")
    cfg = cfg_from_dict(ckpt_payload["config"])
    ckpt_state = ckpt_payload.get("model", {})
    v23_keys = [
        k
        for k in ckpt_state
        if any(x in k for x in ("kalman", "harq", "l_transition", "h_transition", "z_h_to_hidden", "z_l_to_hidden"))
    ]
    if not v23_keys:
        cfg.model.kalman = None
        cfg.model.msa = None
        cfg.model.hrm = None
        logger.info("V22 checkpoint detected — disabling V23 modules (Kalman/HARQ/HRM)")
    else:
        logger.info(f"V23 checkpoint detected ({len(v23_keys)} V23 module keys)")
    dev = torch.device(target)
    model = HAGIv4(cfg)
    step, cfg = load_model_checkpoint(checkpoint_path, model, "cpu")
    if cfg.train.precision == "bf16":
        model.to(torch.bfloat16)

    if dev.type == "cuda":
        for name, param in model.named_parameters():
            if "embed.weight" in name or "lm_head.weight" in name:
                continue
            param.data = param.data.to(dev)
        for name, buf in model.named_buffers():
            if "embed" in name or "lm_head" in name:
                continue
            buf.data = buf.data.to(dev)
    else:
        model = model.to(dev)

    model.eval()
    return model, cfg, step, dev


def tokens_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """Decode token IDs through the checkpoint tokenizer."""
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def text_to_tokens(text: str, tokenizer, device: str = "cuda") -> torch.Tensor:
    """Encode text to token IDs (no BOS — model trained on raw chunks without special tokens)."""
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    return ids.to(device)


def build_generation_kwargs(
    args, inference_cfg, eos_token_id: int, pad_token_id: int, forbidden_token_ids: tuple[int, ...]
):
    """Build generation settings from checkpoint config plus explicit CLI overrides."""
    return {
        "max_new_tokens": args.max_tokens if args.max_tokens is not None else inference_cfg.max_new_tokens,
        "max_iterations": args.iterations if args.iterations is not None else inference_cfg.max_iterations,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "forbidden_token_ids": forbidden_token_ids,
        "min_new_tokens": inference_cfg.min_new_tokens,
        "temperature": args.temperature if args.temperature is not None else inference_cfg.temperature,
        "top_k": args.top_k if args.top_k is not None else inference_cfg.top_k,
        "repetition_penalty": inference_cfg.repetition_penalty,
        "repetition_window": inference_cfg.repetition_window,
        "no_repeat_ngram_size": inference_cfg.no_repeat_ngram_size,
    }


def main() -> int:
    from hagi_v4.data.tokenizer import load_tokenizer
    from hagi_v4.inference.generate import generate

    parser = argparse.ArgumentParser(description="HAGI V4 inference")
    parser.add_argument("--checkpoint", default="checkpoints/step-001000.pt")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--config", default="configs/8gb_google.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override config hard cap on generated tokens")
    parser.add_argument("--iterations", type=int, default=None, help="Override config refinement iterations")
    parser.add_argument("--temperature", type=float, default=None, help="Override config temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override config top_k")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name (auto-detected from checkpoint config)")
    parser.add_argument("--speculative", action="store_true", help="Use speculative block decoding")
    args = parser.parse_args()

    model, cfg, step, dev = load_model_from_checkpoint(args.checkpoint, args.device)
    if args.speculative:
        cfg.inference.speculative.enabled = True
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded checkpoint: step {step} | {n_params / 1e6:.1f}M params | device: {dev}")

    tokenizer_name = args.tokenizer or cfg.train.tokenizer
    logger.info(f"Using tokenizer: {tokenizer_name}")

    tokenizer = load_tokenizer(tokenizer_name, local_files_only=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = cfg.train.pad_token_id
    if eos_token_id != cfg.train.eos_token_id:
        raise ValueError(f"tokenizer eos_token_id {eos_token_id} != checkpoint {cfg.train.eos_token_id}")
    if pad_token_id != cfg.train.pad_token_id:
        raise ValueError(f"pad_token_id {pad_token_id} != checkpoint {cfg.train.pad_token_id}")
    forbidden_token_ids = tuple(
        token_id for token_id in tokenizer.all_special_ids if token_id not in (eos_token_id, pad_token_id)
    )

    gen_kwargs = build_generation_kwargs(
        args,
        cfg.inference,
        eos_token_id,
        pad_token_id,
        forbidden_token_ids,
    )

    if args.interactive:
        logger.info("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            prompt_ids = text_to_tokens(prompt, tokenizer, str(dev))
            if prompt_ids.shape[1] == 0:
                continue
            output = generate(model, prompt_ids, **gen_kwargs)
            generated = output.token_ids[0, prompt_ids.shape[1] : prompt_ids.shape[1] + output.generated_lengths[0]]
            response = tokens_to_text(generated, tokenizer)
            print(f"HAGI: {response}")
        return 0

    prompt_ids = text_to_tokens(args.prompt, tokenizer, str(dev))
    logger.info(f"Prompt: {args.prompt} ({prompt_ids.shape[1]} tokens)")
    output = generate(model, prompt_ids, **gen_kwargs)
    generated_length = output.generated_lengths[0].item()
    generated = output.token_ids[0, prompt_ids.shape[1] : prompt_ids.shape[1] + generated_length]
    response = tokens_to_text(generated, tokenizer)
    logger.info(f"Generated {generated_length} tokens:")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
