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
    """Decode token IDs to text using SmolLM2 tokenizer."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return tok.decode(token_ids.tolist(), skip_special_tokens=True)
    except Exception:
        return " ".join(str(t.item()) for t in token_ids)


def text_to_tokens(text: str, tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M", device: str = "cuda") -> torch.Tensor:
    """Encode text to token IDs."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        ids = tok.encode(text, return_tensors="pt")
        return ids.to(device)
    except Exception:
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
    parser.add_argument("--max-tokens", type=int, default=512, help="Hard cap on generated tokens")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    model, cfg, step, dev = load_model_from_checkpoint(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded checkpoint: step {step} | {n_params / 1e6:.1f}M params | device: {dev}")

    mask_token_id = cfg.model.masking.mask_token_id
    eos_token_id = cfg.model.vocab_size - 2

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
            prompt_ids = text_to_tokens(prompt, args.tokenizer, str(dev))
            if prompt_ids.shape[1] == 0:
                continue
            gen_ids = generate(
                model,
                prompt_ids,
                max_new_tokens=args.max_tokens,
                max_iterations=args.iterations,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            generated = gen_ids[0, prompt_ids.shape[1] :]
            response = tokens_to_text(generated, args.tokenizer)
            print(f"HAGI: {response}")
        return 0

    prompt_ids = text_to_tokens(args.prompt, args.tokenizer, str(dev))
    logger.info(f"Prompt: {args.prompt} ({prompt_ids.shape[1]} tokens)")
    gen_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        max_iterations=args.iterations,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generated = gen_ids[0, prompt_ids.shape[1] :]
    response = tokens_to_text(generated, args.tokenizer)
    logger.info(f"Generated {generated.shape[0]} tokens:")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
