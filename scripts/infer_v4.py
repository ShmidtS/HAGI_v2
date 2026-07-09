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
    """Load model + config from checkpoint (single load)."""
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.checkpoint import cfg_from_dict

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(state["config"])

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model = HAGIv4(cfg).to(dev)
    if cfg.train.precision == "bf16":
        model.to(torch.bfloat16)
    model.load_state_dict(state["model"])
    step = state["step"]
    model.eval()
    return model, cfg, step, dev


def tokens_to_text(token_ids: torch.Tensor, tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M") -> str:
    """Decode token IDs to text using the model's tokenizer."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return tok.decode(token_ids.tolist(), skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer '{tokenizer_name}': {e}")
        return " ".join(str(t) for t in token_ids.tolist())


def text_to_tokens(text: str, tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M", device: str = "cuda") -> torch.Tensor:
    """Encode text to token IDs."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        ids = tok.encode(text, return_tensors="pt")
        return ids.to(device)
    except Exception as e:
        logger.warning(f"Could not load tokenizer '{tokenizer_name}': {e}")
        tokens = [sum(ord(c) for c in w) % 49152 for w in text.split()]
        return torch.tensor([tokens], dtype=torch.long, device=device)


def main() -> int:
    from hagi_v4.inference.generate import generate

    parser = argparse.ArgumentParser(description="HAGI V4 inference")
    parser.add_argument("--checkpoint", default="checkpoints/step-001000.pt")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--config", default="configs/8gb_canonical.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=128, help="Hard cap on generated tokens")
    parser.add_argument(
        "--iterations", type=int, default=4, help="Refinement iterations (more = better quality, slower)"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override config temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override config top_k")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name (auto-detected from checkpoint config)")
    args = parser.parse_args()

    model, cfg, step, dev = load_model_from_checkpoint(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded checkpoint: step {step} | {n_params / 1e6:.1f}M params | device: {dev}")

    tokenizer_name = args.tokenizer or cfg.train.tokenizer
    logger.info(f"Using tokenizer: {tokenizer_name}")

    mask_token_id = cfg.model.masking.mask_token_id
    eos_token_id = cfg.model.vocab_size - 2

    icfg = cfg.inference
    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        max_iterations=args.iterations,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        temperature=args.temperature if args.temperature is not None else icfg.temperature,
        top_k=args.top_k if args.top_k is not None else icfg.top_k,
        block_size=icfg.block_size,
        refine_passes=icfg.refine_passes,
        repetition_penalty=icfg.repetition_penalty,
        repetition_window=icfg.repetition_window,
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
            prompt_ids = text_to_tokens(prompt, tokenizer_name, str(dev))
            if prompt_ids.shape[1] == 0:
                continue
            gen_ids = generate(model, prompt_ids, **gen_kwargs)
            generated = gen_ids[0, prompt_ids.shape[1] :]
            response = tokens_to_text(generated, tokenizer_name)
            print(f"HAGI: {response}")
        return 0

    prompt_ids = text_to_tokens(args.prompt, tokenizer_name, str(dev))
    logger.info(f"Prompt: {args.prompt} ({prompt_ids.shape[1]} tokens)")
    gen_ids = generate(model, prompt_ids, **gen_kwargs)
    generated = gen_ids[0, prompt_ids.shape[1] :]
    response = tokens_to_text(generated, tokenizer_name)
    logger.info(f"Generated {generated.shape[0]} tokens:")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
