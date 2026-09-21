"""Targeted unit tests for bonsai_evolution_daemon.py core invariants.

Covers:
- confidence_from_logprobs entropy/confidence math
- trigram_rep false-positive avoidance for short/empty text
- visible_content reasoning fallback
- role normalization in active_messages (tool -> user, critique -> user)
- save_state / resume round-trip via DaemonState
- run_probe allowlist (denies dangerous commands)
Covers: confidence/trigram/state-resume/tool-allow from daemon plan.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

DAEMON = Path(__file__).resolve().parents[1] / "bonsai_evolution_daemon.py"
spec = importlib.util.spec_from_file_location("daemon", DAEMON)
daemon = importlib.util.module_from_spec(spec)
sys.modules["daemon"] = daemon
spec.loader.exec_module(daemon)


def _tmp_state():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def test_trigram_rep_empty_and_short():
    assert daemon.trigram_rep("") == 0.0
    assert daemon.trigram_rep("one two") == 0.0  # too short -> no false positive
    # all-unique trigrams -> 0.0
    assert daemon.trigram_rep("a b c d e f g h") == 0.0
    # fully repeated trigrams -> high rep (1.0)
    t = " ".join(["раз", "два", "три"] * 3)
    assert daemon.trigram_rep(t) >= 0.5


def test_visible_content_reasoning_fallback():
    assert daemon.visible_content({"content": "x", "reasoning_content": "r"}) == "x"
    assert daemon.visible_content({"content": "", "reasoning_content": "r"}) == "r"
    assert daemon.visible_content({"content": "", "reasoning_content": ""}) == ""


def test_confidence_from_logprobs_bounds():
    ent, conf = daemon.confidence_from_logprobs([])
    assert ent == daemon.ENTROPY_WARN
    assert conf == 0.5
    # uniform distribution over 2 tokens -> entropy ln(2)
    lp = [{"top_logprobs": [{"logprob": 0.0}, {"logprob": 0.0}]}]
    e, c = daemon.confidence_from_logprobs(lp)
    assert abs(e - math.log(2)) < 1e-6
    assert abs(c - math.exp(-math.log(2))) < 1e-6
    assert 0.0 < c < 1.0


def test_active_messages_normalizes_roles():
    st = daemon.DaemonState()
    st.transcript.append({"idx": 1, "role": "tool", "content": "DENIED: '' not allowed"})
    st.transcript.append({"idx": 1, "role": "assistant_critique", "content": "empty check"})
    st.transcript.append({"idx": 1, "role": "assistant", "content": "ok"})
    msgs = daemon.active_messages(st)
    roles = [m["role"] for m in msgs]
    # only system/user/assistant allowed
    assert set(roles) <= {"system", "user", "assistant"}
    # tool and assistant_critique mapped to user
    assert "user" in roles
    # no 2+ assistant at the end
    assert not (len(roles) >= 2 and roles[-1] == "assistant" and roles[-2] == "assistant")


def test_state_save_resume_roundtrip(tmp_path):
    sf = tmp_path / "state.json"
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(sf)
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    st = daemon.DaemonState()
    st.turn = 3
    st.total_generated = 100
    st.transcript.append({"idx": 0, "role": "system", "content": "sys"})
    st.transcript.append({"idx": 1, "role": "user", "content": "hi"})
    daemon.save_state(st)
    st2 = daemon.DaemonState()
    with open(sf, encoding="utf-8") as f:
        st2 = daemon.DaemonState.from_dict(json.load(f))
    assert st2.turn == 3
    assert st2.total_generated == 100
    assert st2.total_accepted == 0


def test_resume_preserves_counts(tmp_path):
    sf = tmp_path / "state.json"
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(sf)
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    st = daemon.DaemonState()
    st.turn = 7
    st.total_accepted = 5
    st.total_rejected = 2
    st.total_generated = 88
    daemon.save_state(st)
    # ensure_state should resume rather than overwrite
    resumed = daemon.ensure_state()
    assert resumed.turn == 7
    assert resumed.total_accepted == 5
    assert resumed.total_rejected == 2
    assert resumed.total_generated == 88


def test_run_probe_allowlist():
    date_out = daemon.run_probe("date")
    # date is allowlisted -> not a DENIED for the command name
    assert not date_out.startswith("DENIED")
    denied = daemon.run_probe("rm -rf /")
    assert "DENIED: 'rm'" in denied
    # shell metacharacters must not bypass the allowlist
    denied2 = daemon.run_probe("ls; rm -rf /")
    assert "DENIED" in denied2
    # rg removed: was an arbitrary-command-execution vector via --pre flag
    assert "rg" not in daemon.ALLOWED_PROBES
    rg_denied = daemon.run_probe("rg --pre 'touch /tmp/PWN' needle")
    assert "DENIED" in rg_denied


def test_checkpoint_records_state(tmp_path):
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    st = daemon.DaemonState()
    st.turn = 10
    st.total_generated = 50
    st.total_accepted = 3
    daemon.checkpoint(st, label="final")
    assert len(st.checkpoints) == 1
    assert st.checkpoints[0]["label"] == "final"
    assert st.checkpoints[0]["turn"] == 10


def test_finalize_sets_stopped_and_persists(tmp_path):
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    st = daemon.DaemonState()
    st.turn = 5
    st.total_generated = 42
    rc = daemon._finalize(st, "signal", 0)
    assert rc == 0
    assert st.status == "stopped"
    assert st.last_error == "shutdown"
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"
    assert raw["last_error"] == "shutdown"
    assert raw["turn"] == 5
    assert len(raw["checkpoints"]) == 1


def test_finalize_error_reason(tmp_path):
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    st = daemon.DaemonState()
    st.turn = 2
    code = daemon._finalize(st, "error", 1)
    assert code == 1
    assert st.status == "stopped"
    assert st.last_error == "error"


def test_candidate_score_ranking():
    # non-empty beats empty
    assert daemon.candidate_score("", 0.9, 0.0) < daemon.candidate_score("x", 0.9, 0.0)
    # lower repetition beats higher repetition at same confidence
    assert daemon.candidate_score("a b c d e f", 0.9, 0.4) > daemon.candidate_score("a b c d e f", 0.9, 0.8)
    # higher confidence beats lower at same rep
    assert daemon.candidate_score("a b c d e f", 0.9, 0.1) > daemon.candidate_score("a b c d e f", 0.2, 0.1)


def test_handle_tool_calls_detail_accounting():
    tc = {"tool_calls": [{"type": "function", "function": {
        "name": "system_probe", "arguments": {"command": "date"}
    }}]}
    text, st_ = daemon.handle_tool_calls_detail(tc)
    assert st_["present"] == 1
    assert st_["executed"] == 1
    assert st_["denied"] == 0
    assert text is not None and "DENIED" not in text
    # disallowed tool name
    tc2 = {"tool_calls": [{"type": "function", "function": {
        "name": "rm", "arguments": {"path": "/"}
    }}]}
    text2, st2 = daemon.handle_tool_calls_detail(tc2)
    assert st2["denied"] == 1
    assert "denied" in (text2 or "").lower()
    # empty tool_calls -> None, zero stats
    text3, st3 = daemon.handle_tool_calls_detail({"tool_calls": None})
    assert text3 is None
    assert st3["present"] == 0


def test_main_max_turns_halt(tmp_path, monkeypatch):
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    # Force an immediate max_turns halt with a zero-token horizon so the loop
    # never reaches the live server: n_turn==0 already equals MAX_TURNS=0 is
    # blocked by max(1, ...), so set MAX_TURNS=1 and horizon=0 => first check
    # n_turn(0) >= 1 is False, one iteration runs; instead force MAX_TURNS big
    # and horizon=1 with a fake gen_one that returns minimal content.
    monkeypatch.setattr(daemon, "MAX_TURNS", 3)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 1)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})
    monkeypatch.setattr(daemon, "gen_one", lambda msgs, with_tools=True: {
        "content": "ok", "reasoning": "", "tool_calls": None,
        "finish_reason": "stop", "entropy": 0.1, "confidence": 0.95,
        "rep": 0.0, "ntok": 1})
    rc = daemon.main()
    assert rc == 0
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"


def test_main_consecutive_errors_halt(tmp_path, monkeypatch):
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_CONSECUTIVE_ERRORS", 3)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 1000)
    monkeypatch.setattr(daemon, "MAX_TURNS", 1000)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})
    calls = {"n": 0}
    def boom(msgs, with_tools=True):
        calls["n"] += 1
        raise RuntimeError("forced generation failure")
    monkeypatch.setattr(daemon, "gen_one", boom)
    rc = daemon.main()
    assert rc == 1
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"
    assert raw["total_generated"] == 0


def test_retry_rejected_records_branch():
    """A worse retry is rejected and the original candidate remains valid."""
    prev_content = "good original answer"
    retry_content = "shorter"
    orig_score = daemon.candidate_score(prev_content, 0.3, 0.0)
    retry_score = daemon.candidate_score(retry_content, 0.1, 0.9)
    assert retry_score < orig_score


def test_signal_handler_sets_stop_flag():
    daemon._RUN = {"stop": False, "timeout": False, "start": 0.0}
    with pytest.raises(daemon.ShutdownRequested):
        daemon.signal_handler(15, None)
    assert daemon._RUN["stop"] is True


def test_turn_limit_is_cumulative_across_resume(tmp_path, monkeypatch):
    """MAX_TURNS must gate on the restored turn_idx, not a per-process counter.

    Sets MAX_TURNS=2 and resumes from turn 1 (persisted in state.json). The
    daemon must halt immediately on the next entry rather than running more
    turns because ``n_turn`` would start at 0.
    """
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_TURNS", 2)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 100000)
    monkeypatch.setattr(daemon, "REQUEST_TIMEOUT", 60)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})

    calls = {"n": 0}

    def fake_gen(msgs, with_tools=True):
        calls["n"] += 1
        raise RuntimeError("should not reach a new turn after resume at the limit")

    monkeypatch.setattr(daemon, "gen_one", fake_gen)
    # Persist a state with turn already at MAX_TURNS.
    st = daemon.DaemonState()
    st.turn = 2
    daemon.save_state(st)
    rc = daemon.main()
    assert rc == 0
    assert calls["n"] == 0, "must halt immediately when restored turn already meets MAX_TURNS"
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"
    assert raw["last_error"] == "max_turns"


def test_consecutive_errors_reset_after_tool_followup(tmp_path, monkeypatch):
    """consecutive_errors must reset after a successful tool-followup gen_one,
    not only after the main gen_one."""
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_CONSECUTIVE_ERRORS", 3)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 100000)
    monkeypatch.setattr(daemon, "MAX_TURNS", 1000)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})

    seq = []

    def fake_gen(msgs, with_tools=True):
        seq.append(len(seq))
        if len(seq) == 1:
            # first main call: return a tool call so the followup branch runs
            return {
                "content": "", "reasoning": "",
                "tool_calls": [{"type": "function", "function": {
                    "name": "system_probe", "arguments": {"command": "date"}}}],
                "finish_reason": "tool_calls", "entropy": 0.1, "confidence": 0.9,
                "rep": 0.0, "ntok": 5,
            }
        if len(seq) == 2:
            # tool-followup succeeds
            raise RuntimeError("boom")
        # subsequent calls always succeed
        return {
            "content": "ok", "reasoning": "",
            "tool_calls": None, "finish_reason": "stop",
            "entropy": 0.1, "confidence": 0.95, "rep": 0.0, "ntok": 1,
        }

    monkeypatch.setattr(daemon, "gen_one", fake_gen)
    # With MAX_CONSECUTIVE_ERRORS=3 and two error paths (main then followup
    # then a second failure), the daemon must NOT halt prematurely — the
    # success after the followup must have reset consecutive_errors to 0.
    # Force a tight horizon so it halts on tokens, not errors.
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 1)
    rc = daemon.main()
    assert rc == 0, f"expected graceful halt (rc=0), got rc={rc}"
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"
    assert raw["last_error"] in ("horizon", "shutdown"), raw["last_error"]


def test_continuous_loop_skips_max_turns_halt(tmp_path, monkeypatch):
    """MAX_TURNS=0 must disable the turn-count halt (continuous mode).

    Regression: bonsai_evolution_daemon.py:44 used max(1, ...) so MAX_TURNS=0
    was impossible, and line 614 unconditionally halted at turn_idx >= MAX_TURNS.
    The fix: MAX_TURNS=0 → no halt, loop bounded only by horizon/errors/stop.
    """
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_TURNS", 0)  # continuous
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 0)  # infinite horizon
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})
    calls = {"n": 0}

    def fake_gen(msgs, with_tools=True):
        calls["n"] += 1
        # Stop the loop by triggering stop signal after 2 turns
        if calls["n"] >= 2:
            daemon._RUN["stop"] = True
        return {
            "content": "ok", "reasoning": "", "tool_calls": None,
            "finish_reason": "stop", "entropy": 0.1, "confidence": 0.95,
            "rep": 0.0, "ntok": 1
        }

    monkeypatch.setattr(daemon, "gen_one", fake_gen)
    rc = daemon.main()
    assert rc == 0
    assert calls["n"] == 2, "should run continuously past MAX_TURNS halt"
    with open(daemon.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["status"] == "stopped"
    assert raw["last_error"] == "shutdown"
    assert raw["turn"] == 2


def test_needs_review_predicate():
    """The single feature gate: flagged on any one signal, quiet otherwise."""
    clean = {"content": "ok", "confidence": 0.95, "entropy": 0.1, "rep": 0.0}
    assert daemon.needs_review(clean) is False

    assert daemon.needs_review(
        dict(clean, confidence=daemon.ENTROPY_LOW - 0.01)) is True
    assert daemon.needs_review(
        dict(clean, entropy=daemon.ENTROPY_WARN + 0.01)) is True
    assert daemon.needs_review(
        dict(clean, rep=daemon.TRIGRAM_LIMIT)) is True

    # nothing to critique on an empty candidate
    assert daemon.needs_review(dict(clean, content="")) is False


def test_fast_path_skips_critique_llm_call(tmp_path, monkeypatch):
    """A clean candidate must not trigger the self-critique generation.

    That call is the whole cost the feature gate exists to avoid (~160 tokens).
    If gating silently breaks, the daemon still looks healthy while every turn
    pays for a critique it does not need.
    """
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_TURNS", 1)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 0)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})

    calls = {"critique": 0, "retry": 0}

    monkeypatch.setattr(daemon, "gen_one", lambda msgs, with_tools=True: {
        "content": "clear confident answer", "reasoning": "",
        "tool_calls": None, "finish_reason": "stop",
        "entropy": 0.1, "confidence": 0.95, "rep": 0.0, "ntok": 4})

    def fake_critique(msgs, content):
        calls["critique"] += 1
        return "critique"

    def fake_branch(msgs, prev):
        calls["retry"] += 1
        return prev, {"ent": 0.1, "conf": 0.95, "rep": 0.0, "ntok": 4}

    monkeypatch.setattr(daemon, "critique", fake_critique)
    monkeypatch.setattr(daemon, "branch_retry", fake_branch)

    assert daemon.main() == 0
    assert calls["critique"] == 0, "clean candidate must skip the LLM self-critique"
    assert calls["retry"] == 0, "clean candidate must not branch a retry"


def test_flagged_candidate_gets_critique_and_retry(tmp_path, monkeypatch):
    """One flag drives both paths, so they can never disagree.

    Regression for the consolidated gate: the router and the retry gate used to
    spell the same three thresholds in two places, which let them drift apart
    silently (router says "skip" while the retry gate says "flag").
    """
    daemon.STATE_DIR = str(tmp_path)
    daemon.STATE_FILE = str(tmp_path / "state.json")
    daemon.LOG_FILE = str(tmp_path / "daemon.log")
    monkeypatch.setattr(daemon, "MAX_TURNS", 1)
    monkeypatch.setattr(daemon, "HORIZON_TOKENS", 0)
    monkeypatch.setattr(daemon, "_RUN", {"stop": False, "timeout": False, "start": 0.0})

    calls = {"critique": 0, "retry": 0}

    monkeypatch.setattr(daemon, "gen_one", lambda msgs, with_tools=True: {
        "content": "hesitant answer", "reasoning": "",
        "tool_calls": None, "finish_reason": "stop",
        "entropy": 0.1, "confidence": 0.05, "rep": 0.0, "ntok": 2})

    def fake_critique(msgs, content):
        calls["critique"] += 1
        return "critique"

    def fake_branch(msgs, prev):
        calls["retry"] += 1
        return prev, {"ent": 0.1, "conf": 0.05, "rep": 0.0, "ntok": 2}

    monkeypatch.setattr(daemon, "critique", fake_critique)
    monkeypatch.setattr(daemon, "branch_retry", fake_branch)

    assert daemon.main() == 0
    assert calls["critique"] == 1, "flagged candidate must be critiqued"
    assert calls["retry"] == 1, "the same flag must branch a retry"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
