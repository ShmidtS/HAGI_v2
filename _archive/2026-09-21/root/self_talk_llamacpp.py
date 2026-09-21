"""Bounded self-talk on the llama.cpp server (bonsai2-mtp).

Мотивация: PyT-путь (scripts/self_evolve.py -> dsv4_generate_ttt.py --evolve)
требует dsv4_reduced (~72GB) и TTT/RLS-адаптацию, которой llama.cpp runtime
не располагает — он inference-only и не пишет адаптацию. Поэтому self-talk
здесь реализуется как bounded conversational loop через /v1/chat/completions:
модель получает собственный ответ как контекст, а качество проверяется через
trigram-repetition gate (как в scripts/self_evolve.py::gate), без изменения весов.

Ограничения: никогда не более HAGI_SELF_SEEDS seeds * HAGI_SELF_ROUNDS раундов
(по умолчанию 4*3=12), каждый раунд — max_tokens=48. Сервер управляется
извне, long-running процессов нет.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

SERVER = os.environ.get("HAGI_SERVER", "http://127.0.0.1:8090")
MODEL = "bonsai2-mtp"

SYS = (
    "Ты — русскоязычная языковая модель. Веди себя как self-chat: "
    "отвечай кратко, одним содержательным предложением, и продолжай "
    "собственный разговор, не повторяя фразы."
)

SEEDS = [
    "Объясни, как языковая модель может улучшать собственную генерацию без изменения весов.",
    "Почему self-talk помогает структурировать рассуждение?",
    "В чём разница между TTT и обычным fine-tuning?",
    "Как pyramid KV-cache меняет длину контекста?",
]


def trigram_rep(text: str) -> float:
    """Доля повторов trigram: 1.0 = всё повторяется (плохо), 0.0 = уникально."""
    toks = text.split()
    if len(toks) < 6:
        return 1.0
    tri = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def completion(messages, max_tokens=48, temperature=0.9):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.5,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_seed(seed: str, rounds: int):
    """Один self-chat: seed -> model ответ -> feedback loop."""
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": seed}]
    transcript = [{"seed": seed}]
    for i in range(rounds):
        t0 = time.time()
        r = completion(messages, max_tokens=48, temperature=0.9)
        choice = r["choices"][0]["message"]["content"].strip()
        usage = r.get("usage", {})
        dt = time.time() - t0
        rep = trigram_rep(choice)
        status = "DEGENERATE" if rep >= 0.5 else "OK"
        print(
            f"  [seed #{len(transcript)} / round {i+1}/{rounds}] "
            f"t={dt:.1f}s rep={rep:.2f} {status} "
            f"tokens={usage.get('completion_tokens', '?')}/{usage.get('total_tokens', '?')}"
        )
        print(f"    seed: {seed[:80]}")
        print(f"    answ: {choice[:120]}")
        transcript.append({"round": i + 1, "answer": choice, "rep": rep, "dt": dt})
        messages.append({"role": "assistant", "content": choice})
        messages.append({"role": "user", "content": "Продолжи кратко."})
    return transcript, min(tr["rep"] for tr in transcript if "rep" in tr)


def main():
    rounds = int(os.environ.get("HAGI_SELF_ROUNDS", "3"))
    max_seeds = int(os.environ.get("HAGI_SELF_SEEDS", "4"))
    seeds = SEEDS[:max_seeds]
    os.makedirs("self_talk_logs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"self_talk_logs/self_talk_{stamp}.jsonl"

    all_transcripts = []
    worst_rep = 1.0
    n_ok = 0
    for seed in seeds:
        print(f"=== seed: {seed[:70]!r} ===")
        transcript, rep = run_seed(seed, rounds)
        all_transcripts.append(transcript)
        if rep < 0.5:
            n_ok += 1
        worst_rep = min(worst_rep, rep)
        sys.stdout.flush()

    with open(log_path, "w", encoding="utf-8") as f:
        for tr in all_transcripts:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    passed = n_ok == len(seeds)
    print("\n=== self-talk finished ===")
    print(f"  seeds={len(seeds)} rounds={rounds} ok={n_ok} pass_rate={passed:.2f} worst_rep={worst_rep:.2f}")
    print(f"  log: {log_path}")
    print(f"  gate: {'PASS' if passed else 'ROLLBACK (все весы unchanged, просто не удалось улучшить self-talk)'}")


if __name__ == "__main__":
    main()
