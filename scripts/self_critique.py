"""WAVE 240: SELF-CRITIQUE loop — the model grades its own law
reasoning with EXACT ground truth.

For each known law:
  1. generate a continuation from a law-consistent seed context
  2. compare byte-by-byte with ground truth
  3. mismatches -> positions + expected bytes become the critique
  4. queue a corrective .chat into the training inbox (the live
     cycle picks it up and learns from the CORRECT stream, weighted
     toward the model's failure positions)

Metric: law_accuracy per law + global. This is a measurable
reasoning-progress metric (not vibes).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from hagi.unified import Unified

INBOX = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "checkpoints", "inbox")


def known_laws() -> list[dict]:
    """Law name, seed context (chronological bytes), step function
    producing the TRUE next byte from the last k bytes."""
    laws = []
    # y = x + 3
    laws.append({"name": "plus3",
                 "seed": [10, 13, 16],
                 "step": lambda hist: (hist[-1] + 3) % 256})
    # y = 2x
    laws.append({"name": "double",
                 "seed": [5, 10],
                 "step": lambda hist: (2 * hist[-1]) % 256})
    # y = x XOR 0xFF
    laws.append({"name": "xor_ff",
                 "seed": [0, 255],
                 "step": lambda hist: hist[-1] ^ 0xFF})
    # y = x1 + x2 (Fibonacci)
    laws.append({"name": "fib_sum",
                 "seed": [3, 7],
                 "step": lambda hist: (hist[-1] + hist[-2]) % 256})
    # parity toggle
    laws.append({"name": "parity",
                 "seed": [100, 101],
                 "step": lambda hist: hist[-1] ^ 1})
    # alphabet: next letter in 'abcdefghijklmnopqrstuvwxyz'
    alpha = b"abcdefghijklmnopqrstuvwxyz"
    laws.append({"name": "alphabet",
                 "seed": list(alpha[:10]),
                 "step": lambda hist: alpha[
                     (alpha.index(hist[-1]) + 1) % 26]})
    return laws


def grade_law(u: Unified, law: dict, n_steps: int = 32,
              temperature: float = 0.0) -> dict:
    """Generate greedily from seed, compare with ground truth."""
    import torch as _t
    hist = list(law["seed"])
    preds, truths = [], []
    for _ in range(n_steps):
        true_next = law["step"](hist)
        ctx_list = hist[-u.model.k:][::-1]  # newest-first
        x = _t.as_tensor([ctx_list], dtype=_t.long,
                         device=u.model.device)
        with _t.no_grad():
            logits = u.model.logits(x)[0]
        pred = int(logits.argmax())
        preds.append(pred)
        truths.append(true_next)
        # teacher-forced history: follow TRUE path so errors don't
        # cascade (we measure per-position accuracy, not rollout)
        hist.append(true_next)
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    # symbolic law prediction as second opinion
    sym = u.predict_next_byte(bytes(hist[-4:]))
    return {"name": law["name"],
            "accuracy": correct / n_steps,
            "n": n_steps,
            "first_errors": [(i, p, t) for i, (p, t)
                             in enumerate(zip(preds, truths))
                             if p != t][:8],
            "symbolic_byte": sym.get("byte"),
            "symbolic_status": sym.get("status")}


def critique_message(results: list[dict]) -> str:
    """Compose the corrective chat message from worst laws."""
    worst = sorted(results, key=lambda r: r["accuracy"])[:3]
    lines = ["САМОКРИТИКА: модель ошибается на этих законах. "
             "Повтори точные правила."]
    for w in worst:
        lines.append(f"Закон {w['name']}: точность "
                     f"{w['accuracy']:.0%}. Примеры ошибок "
                     f"(позиция, предсказано, верно): "
                     f"{w['first_errors'][:4]}")
    lines.append("Повтори закономерности на новых примерах "
                 "ниже (правильные продолжения).")
    return "\n".join(lines)


def main() -> None:
    u = Unified.load() if Unified.exists() else Unified()
    results = []
    print("=== SELF-CRITIQUE (isolated grading) ===", flush=True)
    for law in known_laws():
        r = grade_law(u, law)
        results.append(r)
        sym = f"sym={r['symbolic_byte']}({r['symbolic_status']})" \
            if r.get("symbolic_byte") is not None else "sym=none"
        print(f"  {r['name']:9}: {r['accuracy']:5.1%} "
              f"({r['n']} шагов) {sym}", flush=True)
    avg = sum(r["accuracy"] for r in results) / len(results)
    print(f"=== средняя точность законов: {avg:.1%} ===", flush=True)
    # corrective message for the live cycle (only if accuracy low)
    weak = [r for r in results if r["accuracy"] < 0.6]
    if weak:
        msg = critique_message(results)
        os.makedirs(INBOX, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        path = os.path.join(INBOX, f"critique_{stamp}.chat")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(msg)
        print(f"critique queued: {path} ({len(weak)} слабых законов)",
              flush=True)
    else:
        print("все законы >= 60% — критика не нужна", flush=True)
    return results


if __name__ == "__main__":
    main()
