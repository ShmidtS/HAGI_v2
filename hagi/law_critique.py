"""Automatic self-critique from laws already accepted into HAGI's law registry.

The registry is the source of truth. No hand-written seed laws are used.
Only accepted registry entries with stored, law-consistent byte examples are
eligible. PARTIAL_LAW remains PARTIAL_LAW; this module never upgrades a law's
proof status.

The neural metric is deliberately named teacher_forced_replay_accuracy: each
model prediction receives the true chronological context. It is not a free
autoregressive rollout and is not an out-of-sample generalization claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable


_ACCEPTED_STATUSES = {"PROVEN_EXACT", "INVENTED", "PARTIAL_LAW"}
_DEFAULT_REEVALUATE_EVERY = 4
_DEFAULT_MAX_LAWS_PER_RUN = 8
_DEFAULT_MAX_EVIDENCE = 256
_DEFAULT_MAX_ERROR_EXAMPLES = 64
_DEFAULT_MAX_GOOD_EXAMPLES = 16


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _blueprint_payload(entry: dict[str, Any]) -> Any:
    bp = entry.get("bp")
    if bp is None:
        return entry.get("blueprint", {})
    if hasattr(bp, "to_dict"):
        try:
            return bp.to_dict()
        except Exception:
            pass
    return {
        "q_name": getattr(bp, "q_name", None),
        "q_param": getattr(bp, "q_param", None),
        "basis": getattr(bp, "basis", None),
        "payload": getattr(bp, "payload", None),
    }


def law_fingerprint(domain: str, entry: dict[str, Any]) -> str:
    """Stable identity for one accepted law commit."""
    explicit = entry.get("commit_id") or entry.get("law_id")
    if explicit:
        return str(explicit)
    material = {
        "domain": domain,
        "feature": entry.get("feature"),
        "verify": entry.get("verify"),
        "support": entry.get("support"),
        "blueprint": _blueprint_payload(entry),
    }
    raw = json.dumps(material, sort_keys=True, ensure_ascii=False,
                     default=_json_default, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _accepted(entry: dict[str, Any]) -> bool:
    """A registry entry is committed-to-registry; verification stays separate."""
    if entry.get("accepted_to_registry", True) is not True:
        return False
    return str(entry.get("verify", "")).upper() in _ACCEPTED_STATUSES


def iter_accepted_laws(
    law_registry: dict[str, list[dict[str, Any]]],
) -> Iterable[tuple[str, dict[str, Any]]]:
    for domain in sorted(law_registry):
        for entry in law_registry.get(domain, []) or []:
            if isinstance(entry, dict) and _accepted(entry):
                yield domain, entry


def _normalize_evidence(
    entry: dict[str, Any],
    model_context_length: int | None,
    max_examples: int = _DEFAULT_MAX_EVIDENCE,
) -> list[dict[str, Any]]:
    """Return only evidence that agrees with the accepted symbolic law.

    `context` is the full, chronological LM context. `feature_context` is the
    short tuple used by the symbolic blueprint (normally the last four bytes).
    """
    bp = entry.get("bp")
    raw = entry.get("evidence") or []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            context = [int(v) for v in item["context"]]
            feature_context = [int(v) for v in item["feature_context"]]
            target = int(item["target"])
        except (KeyError, TypeError, ValueError):
            continue
        if not context or not feature_context or not 0 <= target <= 255:
            continue
        # HAGI may intentionally use a shorter domain context (Unified.k)
        # than the maximum attention window (model.k); both are valid.
        if model_context_length and len(context) > model_context_length:
            continue
        if any(v < 0 or v > 255 for v in context + feature_context):
            continue
        # Re-check law consistency before using an example for self-training.
        if bp is not None:
            try:
                if int(bp.apply(tuple(feature_context))) != target:
                    continue
            except Exception:
                continue
        rows.append({
            "context": context,
            "feature_context": feature_context,
            "target": target,
            "source_index": int(item.get("source_index", len(rows))),
        })
    if len(rows) <= max_examples:
        return rows
    # Deterministic, evenly spaced sample rather than a biased prefix.
    last = len(rows) - 1
    indices = [round(i * last / (max_examples - 1))
               for i in range(max_examples)] if max_examples > 1 else [0]
    return [rows[i] for i in indices]


def _model_predictions(model: Any, contexts: list[list[int]],
                       batch_size: int = 64) -> list[int]:
    """Greedy model predictions with chronological context order preserved."""
    import torch

    if not contexts:
        return []
    device = getattr(model, "device", "cpu")
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start:start + batch_size]
            x = torch.as_tensor(batch, dtype=torch.long, device=device)
            logits = model.logits(x)
            predictions.extend(int(v) for v in logits.argmax(dim=-1).detach().cpu().tolist())
    return predictions


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:48] or "law"


def _hex(values: list[int]) -> str:
    return bytes(values).hex()


def _render_critique(domain: str, entry: dict[str, Any], law_id: str,
                     rows: list[dict[str, Any]], predictions: list[int],
                     errors: list[int], generation: int) -> str:
    status = str(entry.get("verify", "UNKNOWN")).upper()
    support = entry.get("support")
    support_text = "не указан" if support is None else f"{float(support):.3f}"
    feature = str(entry.get("feature", "unnamed"))
    lines = [
        "АВТОСАМОКРИТИКА HAGI: обучающий материал построен по закону,",
        "уже принятому в law_registry. Это не новый закон и не новое доказательство.",
        f"Закон: {feature}",
        f"Домен: {domain}",
        f"Статус проверки закона: {status}",
        f"Поддержка закона: {support_text}",
        f"Идентификатор коммита закона: {law_id}",
        f"Поколение обучения: {generation}",
        "Пары ниже повторно проверены самой символьной программой закона.",
        "Контекст записан в хронологическом порядке; после стрелки — верный следующий байт.",
        "Исправляй нейросетевое предсказание по этим подтверждённым парам.",
        "",
        "ОШИБКИ МОДЕЛИ (context_hex -> target_byte_hex ; predicted_byte_hex):",
    ]
    for i in errors[:_DEFAULT_MAX_ERROR_EXAMPLES]:
        row = rows[i]
        lines.append(
            f"{_hex(row['context'])} -> {row['target']:02x} ; "
            f"predicted={predictions[i]:02x}"
        )
    error_set = set(errors)
    good = [i for i in range(len(rows)) if i not in error_set]
    if good:
        lines.extend(["", "КОНТРОЛЬНЫЕ ВЕРНЫЕ ПАРЫ:"])
        for i in good[:_DEFAULT_MAX_GOOD_EXAMPLES]:
            row = rows[i]
            lines.append(f"{_hex(row['context'])} -> {row['target']:02x}")
    lines.extend([
        "",
        "Метрика выше — teacher-forced replay accuracy: каждый контекст подан",
        "в исходном порядке с правильной историей. Это не свободная генерация",
        "и не доказательство обобщения за пределы поддержки закона.",
    ])
    return "\n".join(lines) + "\n"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            default=_json_default) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    try:
        from hagi.util import CKPT_DIR
        root = Path(CKPT_DIR)
    except Exception:
        root = Path(os.environ.get("HAGI_CKPT_DIR", "checkpoints"))
    return (root / "inbox", root / "law_critique_state.json",
            root / "law_critique_ledger.jsonl")


def run_law_self_critique(
    u: Any,
    *,
    inbox_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    reevaluate_every: int = _DEFAULT_REEVALUATE_EVERY,
    max_laws_per_run: int = _DEFAULT_MAX_LAWS_PER_RUN,
    max_evidence: int = _DEFAULT_MAX_EVIDENCE,
    predict_fn: Callable[[Any, list[list[int]]], list[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate newly committed laws immediately and revisit them periodically.

    On model errors, atomically queue a `.chat` into the existing live-training
    inbox. The live cycle ingests that file on its normal single-writer path.
    Repeated identical error sets are not queued again.
    """
    default_inbox, default_state, default_ledger = _default_paths()
    inbox = Path(inbox_dir) if inbox_dir is not None else default_inbox
    state_file = Path(state_path) if state_path is not None else default_state
    ledger_file = Path(ledger_path) if ledger_path is not None else default_ledger
    inbox.mkdir(parents=True, exist_ok=True)

    registry = getattr(u, "law_registry", {}) or {}
    model = getattr(u, "model", None)
    if model is None:
        return {"status": "SKIP", "reason": "no_model", "evaluated": 0,
                "queued": 0}
    model_k = getattr(model, "k", None)
    # WAVE 242 fix: critique-generated chat domains must NOT count
    # as "new training domains" — otherwise each corrective chat
    # advances the re-evaluation clock and the loop self-accelerates
    # (measured: generations 79->91 in one session, critique on
    # nearly every chunk, starving real disk chunks).
    _doms = getattr(u, "domains", {}) or {}
    generation = max(0, sum(1 for d in _doms
                            if not str(d).startswith("chat_")))

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError, TypeError):
        state = {}
    if not isinstance(state.get("laws"), dict):
        state["laws"] = {}

    evaluated = 0
    queued = 0
    skipped = 0
    reports: list[dict[str, Any]] = []
    laws = list(iter_accepted_laws(registry))
    laws.sort(key=lambda pair: (pair[0], str(pair[1].get("feature", "")),
                                law_fingerprint(pair[0], pair[1])))

    for domain, entry in laws:
        if evaluated >= max_laws_per_run:
            break
        law_id = law_fingerprint(domain, entry)
        previous = state["laws"].get(law_id, {})
        if not isinstance(previous, dict):
            previous = {}
        last_generation = int(previous.get("last_generation", -10**9))
        if generation - last_generation < max(1, reevaluate_every):
            skipped += 1
            continue

        evidence = _normalize_evidence(entry, model_k, max_evidence)
        if not evidence:
            skipped += 1
            state["laws"][law_id] = {
                **previous,
                "domain": domain,
                "feature": entry.get("feature"),
                "verify": entry.get("verify"),
                "reason": "no_law_consistent_full_context_evidence",
                "last_generation": generation,
            }
            continue

        contexts = [row["context"] for row in evidence]
        try:
            if predict_fn is None:
                predictions = _model_predictions(model, contexts)
            else:
                predictions = [int(v) for v in predict_fn(model, contexts)]
            if len(predictions) != len(evidence):
                raise ValueError("predict_fn returned a different number of predictions")
        except Exception as exc:
            skipped += 1
            state["laws"][law_id] = {
                **previous,
                "domain": domain,
                "feature": entry.get("feature"),
                "verify": entry.get("verify"),
                "last_generation": generation,
                "error": repr(exc),
            }
            reports.append({"law_id": law_id, "domain": domain,
                            "feature": entry.get("feature"),
                            "status": "EVAL_ERROR", "error": repr(exc)})
            continue

        targets = [int(row["target"]) for row in evidence]
        errors = [i for i, (pred, target) in enumerate(zip(predictions, targets))
                  if pred != target]
        accuracy = 1.0 - len(errors) / max(1, len(evidence))
        signature_material = [(i, predictions[i], targets[i]) for i in errors]
        signature = hashlib.sha256(json.dumps(
            signature_material, separators=(",", ":")).encode("ascii")
        ).hexdigest()[:16]
        report = {
            "law_id": law_id,
            "domain": domain,
            "feature": entry.get("feature"),
            "verification_status": str(entry.get("verify", "UNKNOWN")),
            "accepted_to_registry": True,
            "support": entry.get("support"),
            "generation": generation,
            "metric": "teacher_forced_replay_accuracy",
            "n": len(evidence),
            "correct": len(evidence) - len(errors),
            "errors": len(errors),
            "accuracy": round(accuracy, 6),
            "error_signature": signature,
        }
        reports.append(report)
        _append_jsonl(ledger_file, report)

        old_signature = previous.get("error_signature")
        queued_this_law = False
        if errors and signature != old_signature:
            message = _render_critique(domain, entry, law_id, evidence,
                                       predictions, errors, generation)
            name = _safe_name(domain)
            chat_path = inbox / f"critique_law_{name}_{law_id}_g{generation}.chat"
            if not chat_path.exists():
                tmp = chat_path.with_suffix(chat_path.suffix + ".tmp")
                tmp.write_text(message, encoding="utf-8")
                os.replace(tmp, chat_path)
                queued += 1
                queued_this_law = True

        state["laws"][law_id] = {
            "domain": domain,
            "feature": entry.get("feature"),
            "verify": entry.get("verify"),
            "support": entry.get("support"),
            "last_generation": generation,
            "error_signature": signature,
            "accuracy": round(accuracy, 6),
            "n": len(evidence),
            "queued": queued_this_law,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        evaluated += 1

    _atomic_json(state_file, state)
    return {
        "status": "OK" if evaluated or queued else "NO_PENDING_LAWS",
        "registry_laws": len(laws),
        "evaluated": evaluated,
        "queued": queued,
        "skipped": skipped,
        "reports": reports,
        "state_path": str(state_file),
        "ledger_path": str(ledger_file),
    }
