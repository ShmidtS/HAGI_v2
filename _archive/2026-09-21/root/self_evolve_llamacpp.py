"""Bounded self-evolution loop on llama.cpp server (bonsai2-mtp, 8090).

Ограниченная версия self-talk: N раундов, каждый следующий раунд получает
ответ модели как контекст. Никаких long-running процессов — сервер
управляется извне через HTTP /v1/chat/completions.
"""
import json
import os
import sys
import time
import urllib.request

SERVER = os.environ.get("HAGI_SERVER", "http://127.0.0.1:8090")
MODEL = "bonsai2-mtp"

SYS = (
    "Ты — русскоязычная языковая модель. Веди себя как self-chat: "
    "отвечай кратко, одним предложением, и продолжай собственный разговор."
)


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


def main():
    rounds = int(os.environ.get("HAGI_SELF_ROUNDS", "3"))
    seed = os.environ.get("HAGI_SELF_SEED", "Как модель может улучшать собственную генерацию без изменения весов?")
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": seed},
    ]
    for i in range(rounds):
        t0 = time.time()
        r = completion(messages, max_tokens=48, temperature=0.9)
        choice = r["choices"][0]["message"]["content"]
        usage = r.get("usage", {})
        dt = time.time() - t0
        print(f"[round {i+1}/{rounds}] t={dt:.1f}s tokens={usage.get('completion_tokens','?')}/{usage.get('total_tokens','?')}")
        print(f"  {choice}")
        sys.stdout.flush()
        messages.append({"role": "assistant", "content": choice})
        messages.append({"role": "user", "content": "Продолжи кратко."})
    print("=== self-talk finished ===")


if __name__ == "__main__":
    main()
