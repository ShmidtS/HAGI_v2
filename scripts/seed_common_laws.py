"""WAVE 238: seed well-known laws VIA THE CHAT INBOX — write .chat
files into checkpoints/inbox/, the live training process ingests
them between chunks (single-owner weight writes, no races).

Each .chat contains: human-readable law description + a demo byte
stream (hex) instantiating the law. The training process stores
the description in semantic memory and learns the stream as a
domain; the symbolic synthesizer can then register laws in
law_registry.
"""
import os
import sys

INBOX = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "checkpoints", "inbox")


def laws() -> list[tuple[str, str, bytes]]:
    out = []
    # 1. y = x + 3 (mod 256)
    s = bytearray(); x = 10
    for _ in range(1200):
        s.append(x); s.append((x + 3) % 256); x = (x + 7) % 256
    out.append(("law_plus3",
                "Закон: следующий байт = предыдущий + 3 (mod 256).",
                bytes(s)))
    # 2. y = 2x (mod 256)
    s = bytearray(); x = 5
    for _ in range(1200):
        s.append(x); s.append((2 * x) % 256); x = (x + 11) % 256
    out.append(("law_double",
                "Закон: следующий байт = 2 * предыдущий (mod 256).",
                bytes(s)))
    # 3. y = x XOR 0xFF
    s = bytearray(); x = 0
    for _ in range(1200):
        s.append(x); s.append(x ^ 0xFF); x = (x + 13) % 256
    out.append(("law_xor_ff",
                "Закон: следующий байт = инверсия предыдущего "
                "(XOR 0xFF).",
                bytes(s)))
    # 4. y = x1 + x2 (mod 256), Фибоначчи
    s = bytearray(); a, b = 3, 7
    for _ in range(1200):
        c = (a + b) % 256
        s += bytes([a, b, c]); a, b = b, c
    out.append(("law_fib_sum",
                "Закон: третий байт = сумма двух предыдущих "
                "(mod 256), рекуррентность Фибоначчи.",
                bytes(s)))
    # 5. алфавит: b -> следующий по алфавиту
    s = bytearray()
    for _ in range(600):
        s += b"abcdefghijklmnopqrstuvwxyz. "
    out.append(("law_alphabet",
                "Закон текста: после буквы алфавита идёт следующая "
                "буква; после 'z' — '. ' и цикл повторяется.",
                bytes(s)))
    # 6. чётность: y = x XOR 1
    s = bytearray(); x = 100
    for _ in range(1200):
        s.append(x); s.append(x ^ 1); x = (x + 5) % 256
    out.append(("law_parity",
                "Закон: следующий байт отличается младшим битом "
                "(XOR 1) — переключение чётности.",
                bytes(s)))
    return out


def main() -> None:
    os.makedirs(INBOX, exist_ok=True)
    for name, desc, stream in laws():
        hexs = stream.hex()
        # bounded message: описание + первые ~2000 байт потока hex
        msg = desc + "\n\n[demo stream hex, first 2000 bytes]\n" \
            + hexs[:4000]
        path = os.path.join(INBOX, name + ".chat")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(msg)
        print("queued:", path, flush=True)


if __name__ == "__main__":
    main()
