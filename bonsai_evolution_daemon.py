"""Непрерывный self-evolution daemon на llama.cpp server (bonsai2-mtp).

Контракт (логическая непрерывность без бесконечного KV):
  generate → JEV-style self-critique → confidence (entropy logprobs) →
  correct/branch → verifier (trigram + quality) → durable commit
  (state.json) → compress (rolling summary).

Физически KV-cache не растёт: длинный контекст заменяется rolling
summary + bounded recent-window. Логическая непрерывность через
persistent state.json, который resume восстанавливает после падения.

Вес модели локально не меняется (inference-only runtime).
Self-improvement = behavioral: коррекция следующего хода через
confidence gate, branch/rollback и self-critique.
"""
from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

SERVER = os.environ.get("HAGI_SERVER", "http://127.0.0.1:8090")
MODEL = "bonsai2-mtp"
STATE_DIR = os.environ.get("HAGI_STATE_DIR",
                           os.path.join(os.path.dirname(__file__), "evolve_state"))
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "daemon.log")

MAX_TOKENS = max(1, int(os.environ.get("HAGI_DAEMON_TOK", "96")))
TEMPERATURE = float(os.environ.get("HAGI_DAEMON_TEMP", "0.7"))
TOP_P = float(os.environ.get("HAGI_DAEMON_TOP_P", "0.95"))
TOP_K = int(os.environ.get("HAGI_DAEMON_TOP_K", "20"))
MIN_NEW = int(os.environ.get("HAGI_DAEMON_MIN_NEW", "8"))
HORIZON_TOKENS = int(os.environ.get("HAGI_DAEMON_HORIZON", "60000"))  # 0 == continuous
MAX_TURNS = int(os.environ.get("HAGI_MAX_TURNS", "0"))  # 0 = continuous
MAX_CONSECUTIVE_ERRORS = max(1, int(
    os.environ.get("HAGI_MAX_CONSECUTIVE_ERRORS", "5")))
REQUEST_TIMEOUT = max(1.0, float(os.environ.get("HAGI_REQUEST_TIMEOUT", "120")))

ENTROPY_WARN = float(os.environ.get("HAGI_ENT_WARN", "2.2"))
ENTROPY_LOW = float(os.environ.get("HAGI_ENT_LOW", "0.4"))
RETRY_BOOST = float(os.environ.get("HAGI_RETRY_BOOST", "0.4"))
REP_PENALTY = float(os.environ.get("HAGI_REP_PEN", "1.5"))
TRIGRAM_LIMIT = float(os.environ.get("HAGI_TRIGRAM_LIMIT", "0.55"))

ALLOWED_PROBES = {"date", "uname", "wc", "ls", "cat", "head", "tail",
                  "grep"}

SYSTEM_PROMPT = (
    "Ты — русскоязычная языковая модель-самоэволюционирующий системный "
    "агент. Ты можешь вызывать локальные read-only утилиты (date, uname, wc, "
    "ls, cat, head, tail, grep) через tool 'system_probe'. Ты коротко отвечаешь на "
    "русском, критикуешь свои ответы и улучшаешь их. Ты знаешь, что inference-"
    "only: не меняешь веса, а корректируешь поведение через confidence/branch/"
    "rollback. Всегда называй сегодняшнюю дату через system_probe."
)

SYSTEM_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "system_probe",
        "description": "Read-only local probe: date/uname/wc/ls/cat/head/tail/grep only.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "One safe read-only shell command.",
                },
            },
            "required": ["command"],
        },
    },
}

_RUN = {"stop": False, "start": time.time()}


class ShutdownRequested(Exception):
    """Interrupt a blocking request when SIGINT/SIGTERM is received."""


@dataclass
class DaemonState:
    started: float = field(default_factory=time.time)
    turn: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    confidence_history: list[float] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rejected_branches: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    last_slot: str = ""
    status: str = "running"
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "started": self.started, "turn": self.turn,
            "transcript": self.transcript, "summary": self.summary,
            "summary_tokens": self.summary_tokens,
            "confidence_history": self.confidence_history,
            "accepted": self.accepted, "rejected": self.rejected,
            "rejected_branches": self.rejected_branches,
            "checkpoints": self.checkpoints,
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "last_slot": self.last_slot, "status": self.status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DaemonState:
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in f})


def log(msg: str) -> None:
    # File logging is always UTF-8; console output is wrapped so that a
    # Windows cp1251 stdout cannot raise UnicodeEncodeError on glyphs like ->.
    # Log rotation: cap log file size to avoid unbounded growth.
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 2_000_000:
            bak = LOG_FILE + ".bak"
            if os.path.exists(bak):
                os.remove(bak)
            os.rename(LOG_FILE, bak)
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + chr(10))
    except Exception:
        pass
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(line.encode("utf-8", "replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def ensure_state() -> DaemonState:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                st = DaemonState.from_dict(json.load(f))
            log(f"resume: turn={st.turn} gen={st.total_generated} "
                f"acc={st.total_accepted} rej={st.total_rejected} "
                f"summary_tokens={st.summary_tokens}")
            return st
        except Exception as e:
            log(f"state corrupt ({e}); starting fresh")
    st = DaemonState()
    st.transcript.append({"idx": 0, "role": "system", "content": SYSTEM_PROMPT})
    save_state(st)
    return st


def save_state(st: DaemonState) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st.as_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def completion(messages: list[dict], *, max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE, top_p: float = TOP_P,
               top_k: int = TOP_K, repeat_penalty: float = REP_PENALTY,
               logprobs: bool = True, top_logprobs: int = 0,
               tools: list | None = None) -> dict:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": 0.0,
        "repeat_penalty": repeat_penalty,
        "min_new_tokens": MIN_NEW,
        "logprobs": logprobs,
        "stream": False,
    }
    # llama.cpp rejects top_logprobs when logprobs is disabled.
    if logprobs:
        payload["top_logprobs"] = max(1, top_logprobs)
    if tools is not None:
        payload["tools"] = tools
    return post("/v1/chat/completions", payload)


__all__ = ["main", "run_probe", "completion"]


def mean_entropy(probs: list[float]) -> float:
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log(p)
    return s


def needs_review(result: dict) -> bool:
    """Single source of truth for "this candidate deserves a second look".

    One cheap predicate over extractable features (confidence, entropy,
    trigram repetition) — no extra LLM generation. It drives BOTH the
    self-critique call and the branch-retry gate, so the two can never
    disagree about whether a candidate was flagged.

    Returning False is the fast path: the candidate skips the ~160-token
    self-critique generation entirely.
    """
    return bool(result["content"]) and (
        result["confidence"] < ENTROPY_LOW
        or result["entropy"] > ENTROPY_WARN
        or result["rep"] >= TRIGRAM_LIMIT
    )


def visible_content(msg: dict) -> str:
    """Return the model's visible statement, including reasoning-only replies."""
    return ((msg.get("content") or "") or
            (msg.get("reasoning_content") or ""))[:2000]


def confidence_from_logprobs(logprobs: list) -> tuple[float, float]:
    """(mean_entropy_nats, confidence∈[0,1] = exp(-entropy))."""
    if not logprobs:
        return ENTROPY_WARN, 0.5
    entropies: list[float] = []
    for entry in logprobs:
        toks = entry.get("top_logprobs", [])
        if not toks:
            continue
        dist = [math.exp(min(0.0, float(tp.get("logprob", 0.0)))) for tp in toks]
        total = sum(dist)
        if total > 0:
            dist = [p / total for p in dist]
        else:
            continue
        entropies.append(mean_entropy(dist))
    if not entropies:
        return ENTROPY_WARN, 0.5
    mean_e = sum(entropies) / len(entropies)
    return mean_e, math.exp(-mean_e)


def trigram_rep(text: str) -> float:
    toks = text.split()
    if len(toks) < 6:
        return 0.0
    tri = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def candidate_score(content: str, confidence: float, rep: float) -> tuple[bool, float, float]:
    """Rank non-empty, non-repetitive, high-confidence candidates first."""
    return bool(content.strip()), -float(rep), float(confidence)


def run_probe(command: str) -> str:
    parts = shlex.split(command)
    if not parts or parts[0] not in ALLOWED_PROBES:
        return f"DENIED: '{parts[0] if parts else ''}' not allowed"
    try:
        r = subprocess.run(parts, capture_output=True, text=True,
                           timeout=25, check=False, shell=False)
        out = (r.stdout or "")[:2000]
        if r.stderr:
            out += "\n[stderr] " + (r.stderr or "")[:500]
        return out.strip()
    except Exception as e:
        return f"ERROR: {e}"


def handle_tool_calls_detail(msg: dict) -> tuple[str | None, dict[str, int]]:
    """Execute allowed probes and return text plus honest call accounting."""
    tc = msg.get("tool_calls")
    status = {"present": 0, "executed": 0, "denied": 0, "rejected": 0,
              "errors": 0}
    if not tc:
        return None, status
    status["present"] = len(tc)
    results: list[str] = []
    for call in tc:
        if (not isinstance(call, dict) or call.get("type") != "function"
                or "function" not in call):
            status["rejected"] += 1
            continue
        fn = call["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name != "system_probe":
            status["denied"] += 1
            results.append(f"<error tool={name} denied</error>")
            continue
        cmd = (args or {}).get("command", "") if isinstance(args, dict) else ""
        output = run_probe(cmd)
        if output.startswith("DENIED:"):
            status["denied"] += 1
        elif output.startswith("ERROR:"):
            status["errors"] += 1
        else:
            status["executed"] += 1
        results.append(f"<tool=system_probe>{output}</tool>")
    return " ".join(results) if results else None, status


def handle_tool_calls(msg: dict) -> str | None:
    """Backward-compatible text wrapper used by callers that do not need stats."""
    text, _ = handle_tool_calls_detail(msg)
    return text


def critique(messages: list[dict], answer: str) -> str:
    """JEV discriminator: model critiques its own answer.

    Ensure the request ends with a user turn before appending the candidate
    assistant answer and the critique prompt. This keeps llama.cpp's chat
    template valid even when the bounded history ends in assistant.
    """
    sys = next((m for m in messages if m.get("role") == "system"), None)
    body = [m for m in messages if m.get("role") != "system"]
    crit_msgs: list[dict] = []
    if sys is not None:
        crit_msgs.append(sys)
    crit_msgs.extend(body)
    if not crit_msgs or crit_msgs[-1]["role"] != "user":
        crit_msgs.append({"role": "user", "content": "Продолжи."})
    crit_msgs.append({"role": "assistant", "content": answer})
    crit_msgs.append({"role": "user",
                      "content": (
                          "Critique the answer briefly: truth, repetition, "
                          "confidence 1-5. Then give an improved version."
                      )})
    try:
        r = completion(
            crit_msgs, max_tokens=160, temperature=0.5, top_p=0.9, top_k=10,
            repeat_penalty=1.3, logprobs=False, top_logprobs=0, tools=None,
        )
        return r["choices"][0]["message"].get("content", "") or ""
    except Exception as e:
        return f"[critique error: {e}]"


def branch_retry(messages: list[dict], prev: str) -> tuple[str, dict]:
    """Retry generation with higher temperature; re-fetch metrics."""
    try:
        retry_msgs = list(messages)
        if not retry_msgs or retry_msgs[-1]["role"] != "user":
            retry_msgs.append({"role": "user", "content": "Продолжи."})
        retry_msgs.extend([
            {"role": "assistant", "content": prev},
            {"role": "user",
             "content": (
                 "Your answer was rejected for repetition or low "
                 "confidence. Give a better, more specific answer."
             )},
        ])
        r = completion(
            retry_msgs,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE + RETRY_BOOST,
            top_p=0.95, repeat_penalty=1.8, logprobs=True, top_logprobs=5,
        )
        msg = r["choices"][0]["message"]
        content = visible_content(msg)
        lp = r["choices"][0].get("logprobs", {})
        ent, conf = confidence_from_logprobs(lp.get("content", []) if lp else [])
        rep = trigram_rep(content)
        ntok = len(lp.get("content", []) or []) or (1 if content else 0)
        return content, {"ent": ent, "conf": conf, "rep": rep, "ntok": ntok}
    except Exception:
        return prev, {"ent": ENTROPY_WARN, "conf": 0.5, "rep": 1.0, "ntok": 0}


def gen_one(messages: list[dict], with_tools: bool = True) -> dict:
    """Generate one assistant turn; return message + metrics."""
    tools = [SYSTEM_PROBE_TOOL] if with_tools else None
    r = completion(messages, tools=tools, logprobs=True, top_logprobs=5)
    msg = r["choices"][0]["message"]
    lp = r["choices"][0].get("logprobs", {}) or {}
    tok_lp = lp.get("content", []) if isinstance(lp, dict) else []
    entropy, confidence = confidence_from_logprobs(tok_lp)
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    # Reasoning-enabled templates may expose the visible answer only as
    # reasoning_content while `content` remains empty. Use reasoning as the
    # visible payload when content is empty so the self-critique and confidence
    # gate operate on the model's actual statement, not on an empty string.
    if not content and reasoning:
        content = reasoning[:2000]
    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": r["choices"][0].get("finish_reason", ""),
        "entropy": entropy,
        "confidence": confidence,
        "rep": trigram_rep(content),
        "ntok": (len(tok_lp) if isinstance(tok_lp, list) else 0)
               or (1 if content else 0),
    }


def active_messages(st: DaemonState, window: int = 32) -> list[dict]:
    """Build bounded context: system + last `window` non-system turns.

    llama.cpp's chat template accepts only system/user/assistant roles. Tool
    results and internal critique records are therefore normalized before
    every request instead of leaking unsupported roles into the HTTP API.
    """
    role_map = {"tool": "user", "assistant_critique": "user"}
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if st.summary:
        msgs.append({"role": "user",
                     "content": "Контекст из rolling summary: " + st.summary})
    recent = [t for t in st.transcript
              if t.get("role") != "system"][-window:]
    have_user = bool(st.summary)
    for t in recent:
        role = role_map.get(t.get("role", ""), t.get("role", ""))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = t.get("content", "")
        if t.get("role") == "assistant_critique":
            content = ("Self-critique: " + (content or "[empty]"))
        msgs.append({"role": role, "content": content})
        if role == "user":
            have_user = True
    if not have_user:
        msgs.append({"role": "user", "content": (
            "Привет. Ты — непрерывно самоэволюционирующая языковая система. "
            "Назови сегодняшнюю дату через system_probe, затем коротко "
            "расскажи о self-improvement в LLM.")})
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": "Продолжи."})
    return msgs


def commit(st: DaemonState, idx: int, role: str, content: str,
           entropy: float, confidence: float, rep: float, ntok: int,
           accepted: bool, ts: float, branch: str = "main") -> None:
    rec = {"idx": idx, "role": role, "content": content,
           "entropy": round(entropy, 4), "confidence": round(confidence, 4),
           "rep": round(rep, 4), "ntok": ntok, "ts": round(ts, 3),
           "branch": branch, "accepted": accepted}
    st.transcript.append(rec)
    if role == "assistant":
        if accepted:
            st.accepted.append(rec)
            st.total_accepted += 1
        else:
            st.rejected.append(rec)
            st.total_rejected += 1
        st.confidence_history.append(confidence)
        st.total_generated += ntok
    st.turn = idx
    save_state(st)


def compress(st: DaemonState, max_keep: int = 40) -> None:
    """Rolling summary keeps logical continuity without unbounded KV."""
    non_sys = [t for t in st.transcript if t.get("role") != "system"]
    if len(non_sys) <= max_keep:
        return
    verbatim = non_sys[-max_keep:]
    chunk = json.dumps(
        [{"role": t.get("role", ""), "content": t.get("content", "")}
         for t in verbatim[-16:]], ensure_ascii=False
    )[:2500]
    try:
        r = completion(
            [{"role": "user",
              "content": (
                  "Сделай краткое резюме истории на русском (до 6 "
                  "предложений): " + chunk
              )}],
            max_tokens=80, temperature=0.2, top_p=0.9, top_k=8,
            repeat_penalty=1.2, logprobs=False,
        )
        summary_new = (r["choices"][0]["message"].get("content", "") or "").strip()
        if summary_new:
            st.summary = summary_new
            st.summary_tokens += max(0, len(non_sys) - max_keep)
    except Exception as e:
        log(f"summ: {e}")
    sys_msg = next((t for t in st.transcript if t.get("role") == "system"), None)
    kept_ids = {id(t) for t in verbatim}
    st.transcript = ([sys_msg] if sys_msg else []) + [
        t for t in st.transcript
        if (id(t) in kept_ids) or (t.get("role") == "system")
    ]
    st.transcript = st.transcript[-max_keep - (1 if sys_msg else 0):]
    save_state(st)


def checkpoint(st: DaemonState, label: str = "periodic") -> None:
    st.last_slot = f"turn_{st.turn}_{int(time.time())}"
    st.checkpoints.append({
        "turn": st.turn, "ts": time.time(), "label": label,
        "generated": st.total_generated,
        "accepted": st.total_accepted,
        "rejected": st.total_rejected,
    })
    save_state(st)
    log(f"checkpoint[{label}] turn={st.turn} gen={st.total_generated} "
        f"acc={st.total_accepted} rej={st.total_rejected} "
        f"slot={st.last_slot}")


def signal_handler(signum, frame):
    _RUN["stop"] = True
    raise ShutdownRequested()


class TimeoutGuard:
    """Cooperative timeout around a single blocking HTTP call.

    The timer sets ``_RUN["timeout"]`` so the main loop can stop scheduling new
    work after the deadline. It does NOT interrupt an in-flight ``urlopen``:
    the hard ceiling for each HTTP call remains the socket timeout passed to
    ``urlopen`` (``REQUEST_TIMEOUT``). Lower that env var to get faster aborts.
    """

    def __init__(self, seconds: float):
        self.seconds = max(1.0, seconds)
        self._timer = threading.Timer(self.seconds, self._fire)
        self._timer.daemon = True

    @staticmethod
    def _fire():
        _RUN["stop"] = True
        _RUN["timeout"] = True

    def __enter__(self):
        _RUN["timeout"] = False
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self._timer.cancel()
        _RUN["timeout"] = False
        return False


def _finalize(st: DaemonState, reason: str, exit_code: int) -> int:
    """Single exit path used by signal, horizon, error, and max_turns cases.

    ``checkpoint`` already calls ``save_state`` for persistence, so we do NOT
    call ``save_state`` here again — a duplicate write is harmless in isolation
    but can mask state mutation that happens between the two calls.
    """
    if not st.last_error:
        if reason == "signal":
            st.last_error = "shutdown"
        elif reason == "max_turns":
            st.last_error = "max_turns"
        elif reason == "horizon":
            st.last_error = "horizon"
        else:
            st.last_error = reason
    st.status = "stopped"
    try:
        log(
            f"daemon stopped: gen={st.total_generated} turn={st.turn} "
            f"acc={st.total_accepted} rej={st.total_rejected} "
            f"reason={reason} exit={exit_code}"
        )
    except Exception:
        pass
    checkpoint(st, label="final")
    return exit_code


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    st = ensure_state()
    if st.status != "running":
        st.status = "running"
        if st.last_error and st.last_error != "shutdown":
            st.last_error = ""
        save_state(st)
    log("daemon: starting continuous self-evolution (bonsai2-mtp)")
    log(
        f"limits: horizon={HORIZON_TOKENS}t max_tokens={MAX_TOKENS}t "
        f"max_turns={MAX_TURNS} max_errors={MAX_CONSECUTIVE_ERRORS} "
        f"req_timeout={REQUEST_TIMEOUT}s rep_gate={TRIGRAM_LIMIT} "
        f"ent_warn={ENTROPY_WARN} ent_low={ENTROPY_LOW}"
    )

    turn_idx = st.turn
    consecutive_errors = 0

    while (
        not _RUN["stop"]
        and (
            HORIZON_TOKENS == 0
            or st.total_generated < HORIZON_TOKENS
        )
    ):
        if MAX_TURNS > 0 and turn_idx >= MAX_TURNS:
            log(f"max_turns ({MAX_TURNS}) reached; halting")
            return _finalize(st, "max_turns", 0)
        turn_idx += 1
        tool_in_turn = False
        tool_stats = {
            "present": 0,
            "executed": 0,
            "denied": 0,
            "rejected": 0,
            "errors": 0,
        }
        msgs = active_messages(st)

        try:
            with TimeoutGuard(REQUEST_TIMEOUT):
                result = gen_one(msgs, with_tools=True)
        except ShutdownRequested:
            return _finalize(st, "signal", 0)
        except Exception as e:
            consecutive_errors += 1
            st.last_error = f"gen:{e!r}"
            log(
                f"generate error ({consecutive_errors}/"
                f"{MAX_CONSECUTIVE_ERRORS}): {e}"
            )
            save_state(st)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log("too many consecutive errors; halting")
                return _finalize(st, "error", 1)
            time.sleep(4)
            continue
        if _RUN.get("timeout"):
            st.last_error = "timeout"
            log("request-timeout; halting turn")
            save_state(st)
            return _finalize(st, "timeout", 1)
        consecutive_errors = 0
        st.last_error = ""  # Clear stale transient errors after a successful
        # generation so the next real stop reason is recorded accurately.

        # --- TOOL CALL EXECUTION (honest accounting + safe allowlist) ---
        if result["tool_calls"]:
            tool_text, tool_stats = handle_tool_calls_detail(
                {"tool_calls": result["tool_calls"]}
            )
            tool_in_turn = True
            commit(
                st,
                turn_idx,
                "tool",
                tool_text or "[no tool result]",
                result["entropy"],
                result["confidence"],
                result["rep"],
                0,
                True,
                time.time(),
                "tool",
            )
            log(
                f"tool_call present={tool_stats['present']} "
                f"executed={tool_stats['executed']} "
                f"denied={tool_stats['denied']} "
                f"rejected={tool_stats['rejected']} "
                f"errors={tool_stats['errors']} "
                f"out={tool_text[:80]!r}"
            )
            follow_msgs = active_messages(st)
            follow_msgs.append({"role": "user", "content": "Продолжи."})
            try:
                with TimeoutGuard(REQUEST_TIMEOUT):
                    result = gen_one(follow_msgs, with_tools=False)
            except ShutdownRequested:
                return _finalize(st, "signal", 0)
            except Exception as e:
                consecutive_errors += 1
                st.last_error = f"tool-followup:{e!r}"
                log(
                    f"tool-followup error ({consecutive_errors}/"
                    f"{MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                save_state(st)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return _finalize(st, "error", 1)
                time.sleep(2)
                continue

        # --- FEATURE GATE: one predicate drives critique + branch retry ---
        # Budget-constrained routing over extractable features only (ACL 2026
        # #1329: "activates input-dependent subset of rank directions"). When
        # it stays quiet the candidate takes the fast path and skips the
        # ~160-token self-critique generation (~1ms gate vs ~12-15s).
        content = result["content"]
        conf = result["confidence"]
        ent = result["entropy"]
        rep = result["rep"]
        needs_retry = needs_review(result)
        if needs_retry:
            crit = critique(msgs, content)
            commit(
                st,
                turn_idx,
                "assistant_critique",
                crit,
                result["entropy"],
                result["confidence"],
                result["rep"],
                0,
                True,
                time.time(),
                "critique",
            )
        else:
            log(
                f"turn {turn_idx}: fast-path (conf={conf:.3f} ent={ent:.3f} "
                f"rep={rep:.3f}) — skipped LLM critique"
            )

        # --- BRANCH RETRY on the same flag ---
        if needs_retry:
            log(
                f"turn {turn_idx}: GATED conf={result['confidence']:.3f} "
                f"ent={result['entropy']:.3f} rep={result['rep']:.3f} -> retry"
            )
            prev_content = result["content"]
            prev_metrics = {
                "ent": result["entropy"],
                "conf": result["confidence"],
                "rep": result["rep"],
                "ntok": result["ntok"],
            }
            new_content, m = branch_retry(msgs, prev_content)
            # Compare measurable quality: non-empty, lower repetition,
            # then higher confidence. If retry is not better, keep original.
            orig_score = candidate_score(
                prev_content, prev_metrics["conf"], prev_metrics["rep"]
            )
            retry_score = candidate_score(
                new_content, m["conf"], m["rep"]
            )
            if retry_score >= orig_score and new_content.strip():
                result["content"] = new_content
                result["entropy"] = m["ent"]
                result["confidence"] = m["conf"]
                result["rep"] = m["rep"]
                result["ntok"] = m["ntok"]
                branch = "retry"
                log(
                    f"turn {turn_idx}: retry accepted "
                    f"rep={m['rep']:.3f} conf={m['conf']:.3f}"
                )
            else:
                st.rejected_branches.append(
                    {
                        "turn": turn_idx,
                        "reason": "retry_worse",
                        "branch": "retry_rejected",
                        "prev_conf": prev_metrics["conf"],
                        "retry_conf": m["conf"],
                        "prev_rep": prev_metrics["rep"],
                        "retry_rep": m["rep"],
                        "ts": time.time(),
                    }
                )
                log(f"turn {turn_idx}: retry rejected (rollback to original)")
        else:
            branch = "main"

        accepted = (
            bool(result["content"].strip())
            and result["rep"] < TRIGRAM_LIMIT
            and result["confidence"] > 0.0
        )
        consecutive_errors = 0
        commit(
            st,
            turn_idx,
            "assistant",
            result["content"],
            result["entropy"],
            result["confidence"],
            result["rep"],
            result["ntok"],
            accepted,
            time.time(),
            branch,
        )

        if not accepted:
            log(
                f"turn {turn_idx}: REJECTED rep={result['rep']:.3f} "
                f"conf={result['confidence']:.3f} tool={tool_in_turn} "
                f"texec={tool_stats['executed']}/{tool_stats['denied']}"
            )

        if turn_idx % 8 == 0:
            checkpoint(st)
            compress(st, max_keep=40)
            save_state(st)
            elapsed = time.time() - st.started
            rate = st.total_generated / elapsed if elapsed > 0 else 0
            log(
                f"turn {turn_idx} stats: gen={st.total_generated} "
                f"acc={st.total_accepted} rej={st.total_rejected} "
                f"rate={rate:.1f}t/s conf={st.confidence_history[-3:]}"
            )

        time.sleep(0.3)

    reason = "signal" if _RUN["stop"] else "horizon"
    return _finalize(st, reason, 0)


if __name__ == "__main__":
    sys.exit(main())
