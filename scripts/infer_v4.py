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
    """Load model + config from checkpoint."""
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.checkpoint import _cfg_from_dict, load_checkpoint

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_dict(state["config"])

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model = HAGIv4(cfg).to(dev)
    if cfg.train.precision == "bf16":
        model.to(torch.bfloat16)
    step, _ = load_checkpoint(checkpoint_path, model, device=str(dev))
    model.eval()
    return model, cfg, step, dev


def generate_text(
    model,
    cfg,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    max_iterations: int = 4,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate text via progressive unmasking."""

    B, T_prompt = prompt_ids.shape
    V = cfg.model.vocab_size
    mask_token_id = cfg.model.masking.mask_token_id
    eos_token_id = V - 2  # second-to-last token

    # Start with prompt + mask tokens for generation
    total_len = T_prompt + max_new_tokens
    full_ids = torch.full((B, total_len), mask_token_id, dtype=torch.long, device=device)
    full_ids[:, :T_prompt] = prompt_ids

    # Track which positions are still masked
    still_masked = torch.ones(B, total_len, dtype=torch.bool, device=device)
    still_masked[:, :T_prompt] = False

    with torch.inference_mode():
        for iteration in range(max_iterations):
            if not still_masked.any():
                break

            # Forward: predict all positions
            logits = model(full_ids, targets=None, mask=still_masked).logits  # [B, total_len, V]

            # For masked positions: sample tokens
            for b in range(B):
                masked_positions = still_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue

                for pos in masked_positions:
                    logit = logits[b, pos] / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logit, min(top_k, V))
                        logit[logit < v[-1]] = float("-inf")
                    probs = torch.softmax(logit, dim=-1)
                    sampled = torch.multinomial(probs, 1)
                    full_ids[b, pos] = sampled
                    still_masked[b, pos] = False

                    if sampled.item() == eos_token_id:
                        # Stop generating after EOS
                        still_masked[b, pos + 1 :] = False
                        break

            # Re-mask lowest-confidence positions for next iteration (except last iteration)
            if iteration < max_iterations - 1 and still_masked.any():
                with torch.inference_mode():
                    logits = model(full_ids, targets=None, mask=still_masked).logits
                for b in range(B):
                    masked_positions = still_masked[b].nonzero(as_tuple=True)[0]
                    if len(masked_positions) <= 1:
                        continue
                    confidences = torch.softmax(logits[b, masked_positions], dim=-1).max(dim=-1).values
                    # Re-mask bottom 50% by confidence
                    n_remask = len(masked_positions) // 2
                    if n_remask > 0:
                        _, low_conf_idx = torch.topk(confidences, n_remask, largest=False)
                        for idx in low_conf_idx:
                            pos = masked_positions[idx]
                            full_ids[b, pos] = mask_token_id
                            still_masked[b, pos] = True

    # Cut at EOS
    result = full_ids[0, T_prompt:]
    for i, tok in enumerate(result):
        if tok.item() == eos_token_id:
            result = result[:i]
            break

    return result


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
        # Fallback: simple hash-based
        tokens = [hash(w) % 49152 for w in text.split()]
        return torch.tensor([tokens], dtype=torch.long, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGI V4 inference")
    parser.add_argument("--checkpoint", default="checkpoints/step-001000.pt")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--config", default="configs/8gb_canonical.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    model, cfg, step, dev = load_model_from_checkpoint(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded checkpoint: step {step} | {n_params / 1e6:.1f}M params | device: {dev}")

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
            gen_ids = generate_text(
                model,
                cfg,
                prompt_ids,
                max_new_tokens=args.max_tokens,
                max_iterations=args.iterations,
                temperature=args.temperature,
                top_k=args.top_k,
                device=str(dev),
            )
            response = tokens_to_text(gen_ids, args.tokenizer)
            print(f"HAGI: {response}")
        return 0

    # Single prompt
    prompt_ids = text_to_tokens(args.prompt, args.tokenizer, str(dev))
    logger.info(f"Prompt: {args.prompt} ({prompt_ids.shape[1]} tokens)")
    gen_ids = generate_text(
        model,
        cfg,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        max_iterations=args.iterations,
        temperature=args.temperature,
        top_k=args.top_k,
        device=str(dev),
    )
    response = tokens_to_text(gen_ids, args.tokenizer)
    logger.info(f"Generated {gen_ids.shape[0]} tokens:")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
