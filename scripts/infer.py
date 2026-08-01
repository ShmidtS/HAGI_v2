"""
Интерактивный диалог с HAGI v31.

- Устройство выбирается автоматически: cuda (ROCm/NVIDIA), иначе cpu.
- Непрерывный цикл: после ответа снова спрашивает prompt.
- История диалога сохраняется и подставляется в контекст каждого следующего
  запроса (с обрезкой до attention.max_seq_len).

Примеры:
  python scripts/infer.py --checkpoint checkpoints_v31_1b/step-0030000.pt
  python scripts/infer.py --checkpoint_dir checkpoints_v31_1b --max_tokens 256
  python scripts/infer.py --checkpoint ... --prompt "Привет" --device cuda

Выход из диалога: пустая строка, "exit", "quit" или Ctrl+C.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import os

# Включить экспериментальный flash-attention на AMD ROCm (как в train.py),
# иначе SDPA падает на медленный math-бэкенд.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch

from hagi.inference.generate import generate
from hagi.model.model import HAGI
from hagi.train.loop import cast_model

try:
    import gigatoken as gt
except ImportError:
    print("Требуется gigatoken: pip install gigatoken")
    sys.exit(1)


def find_latest_ckpt(ckpt_dir: str) -> Path | None:
    pts = sorted(Path(ckpt_dir).glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    return pts[-1] if pts else None


def resolve_ckpt(args) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if args.checkpoint_dir is not None:
        path = find_latest_ckpt(args.checkpoint_dir)
        if path is None:
            print(f"В {args.checkpoint_dir} нет checkpoint-ов")
            sys.exit(1)
        return path
    print("Укажите --checkpoint или --checkpoint_dir")
    sys.exit(1)


def decode_text(tokenizer, ids, vocab_map=None) -> str:
    """Если модель в компактном пространстве, маппим обратно в старое для токенизатора."""
    if vocab_map is not None:
        ids = vocab_map.to_old(ids).tolist()
    raw = tokenizer.decode(ids)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description="Интерактивный диалог HAGI v31")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--prompt", default=None, help="первый prompt (иначе интерактив)")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    # The checkpoint carries its own config; read it so the model always matches
    # the weights (a CLI --config that differs silently corrupts every shape).
    ckpt_path = resolve_ckpt(args)
    from hagi.train.checkpoint import config_from_dict, load_payload

    payload = load_payload(ckpt_path, device)
    cfg = config_from_dict(payload["config"])
    # If the corpus was compacted (vocab_map.npz present), the model operates in
    # the compact id space; the tokenizer emits old-space ids and the map bridges
    # the two. Dropped ids fall back to UNK (id 3, always reserved).
    vocab_map = None
    map_path = Path(cfg.train.data.data_dir) / "vocab_map.npz"
    if map_path.exists() and cfg.model.vocab_size < 262144:
        from hagi.data.vocab_map import VocabMap

        vocab_map = VocabMap(map_path)
        print(f"compact vocabulary: model={cfg.model.vocab_size} tokens, "
              f"map={map_path.name} (old {vocab_map.old_vocab} -> new {vocab_map.new_vocab})")

    max_seq_len = cfg.model.attention.max_seq_len
    budget = max_seq_len - args.max_tokens  # сколько токенов истории влезает
    if budget < 8:
        print(f"max_tokens {args.max_tokens} почти равен max_seq_len {max_seq_len}; уменьшите max_tokens")
        return 1

    model = HAGI(cfg).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    cast_model(model, cfg.train.precision)  # bf16 + fp32-гейны, как в обучении
    model.eval()

    tokenizer = gt.Tokenizer(cfg.train.tokenizer)
    history: list[int] = []  # все токены диалога в компактном id-пространстве

    def encode(text: str) -> list[int]:
        """Токенизируем в старое пространство, затем маппим в компактное."""
        raw = tokenizer.encode(text)
        old = list(raw) if not isinstance(raw, list) else raw
        if vocab_map is not None:
            return vocab_map.to_compact(old).tolist()
        return old

    def run_turn(user_ids: list[int]) -> list[int]:
        """Один запрос с полной историей в контексте. Возвращает новые токены."""
        context = (history + user_ids)[-budget:]
        prompt_tensor = torch.tensor([context], dtype=torch.long, device=device)
        output = generate(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            eos_token_id=cfg.train.data.eos_token_id,
            pad_token_id=cfg.train.data.pad_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        full = output.token_ids[0].tolist()
        return full[len(context):]

    try:
        if args.prompt is not None:
            user_ids = encode(args.prompt)
            new = run_turn(user_ids)
            history.extend(user_ids + new)
            print(decode_text(tokenizer, new, vocab_map))
        while True:
            try:
                text = input("\n> ")
            except EOFError:
                break
            if not text.strip() or text.strip().lower() in {"exit", "quit", "q"}:
                break
            user_ids = encode(text)
            new = run_turn(user_ids)
            history.extend(user_ids + new)
            print(decode_text(tokenizer, new, vocab_map))
    except KeyboardInterrupt:
        print("\n(прервано)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
