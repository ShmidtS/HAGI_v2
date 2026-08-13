"""Generate with the merged DSV4 model (1 super-expert per layer).

Loads the compact bf16 checkpoint, tokenizes with gigatoken, and runs
autoregressive sampling (argmax) for a small number of tokens.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401

from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dsv4_shared_only"))
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"
BOS_ID = 0
EOS_ID = 1


def main() -> None:
    import gigatoken

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = tok.encode(prompt)
    ids = [BOS_ID] + list(ids)
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)

    print("loading model...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    print(f"model loaded in {time.time()-t0:.1f}s, embed dtype={model.model.embed_tokens.weight.dtype}, experts={model.config._experts_implementation}", flush=True)

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    print(f"generating {max_new} tokens...", flush=True)

    t0 = time.time()
    generated = list(ids)
    with torch.no_grad():
        for step in range(max_new):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1]
            nxt = int(logits.argmax().item())
            generated.append(nxt)
            if nxt == EOS_ID:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[nxt]], device="cuda", dtype=torch.long)], dim=1
            )
            if (step + 1) % 5 == 0:
                print(f"  {step+1} tokens, {time.time()-t0:.1f}s", flush=True)

    print(f"generated {len(generated)-len(ids)} new tokens in {time.time()-t0:.1f}s", flush=True)
    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
