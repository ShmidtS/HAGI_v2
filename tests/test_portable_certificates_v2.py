from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import hagi.certificates as certs


ROOT = Path(__file__).resolve().parents[1]
V1_CERT = ROOT / "reports" / "hagi_v9_portable_certificates_v1.json"


def _tiny_v1_artifact() -> dict:
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 0, 0, 0]
    artifact = {
        "format": certs.FORMAT_V1,
        "format_version": 1,
        "hagi_version": "test-fixture",
        "scope_notice": "finite-domain test only; provenance not authenticated",
        "model_version": 1,
        "total_commits": 1,
        "task_count": 1,
        "expert_library": {
            "BOOL_FALSE": {
                "family": "BoolVenn",
                "program": {"arity": 2, "expr": None, "venn_mask": 0},
                "domains": [[0, 1], [0, 1]],
                "valid_rule": "all",
                "code_length": 10,
                "expert_index": 0,
            }
        },
        "tasks": [{
            "name": "BOOL_FALSE",
            "family": "BoolVenn",
            "arity": 2,
            "domains": [[0, 1], [0, 1]],
            "valid_rule": "all",
            "domain_size": 4,
            "expected_outputs": outputs,
            "labels_sha256": certs._sha256({"inputs": inputs, "outputs": outputs}),
            "action": "COMMIT",
            "program_source": "BOOL_FALSE",
            "program_bits": 10,
            "reported_code_length": 10,
            "reason": "synthetic-test-fixture",
        }],
    }
    artifact["artifact_sha256"] = certs._sha256(artifact)
    return artifact


def _rehash(artifact: dict) -> dict:
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = certs._sha256(artifact)
    return artifact


def _tiny_v2_artifact() -> dict:
    return certs.upgrade_portable_certificate_v1_to_v2(_tiny_v1_artifact())["artifact"]


def test_v1_remains_verifiable_and_v2_migration_is_semantics_preserving(tmp_path):
    original = json.loads(V1_CERT.read_text(encoding="utf-8"))
    old_result = certs.verify_portable_certificate(original)
    assert old_result["valid"] is True
    assert old_result["format"] == certs.FORMAT_V1

    target = tmp_path / "migrated-v2.json"
    migrated = certs.upgrade_portable_certificate_v1_to_v2(original, target)
    upgraded = json.loads(target.read_text(encoding="utf-8"))
    assert migrated["valid"] is True
    assert upgraded["format"] == certs.FORMAT
    assert upgraded["format_version"] == 2
    assert upgraded["ast_schema"] == certs.AST_SCHEMA
    assert upgraded["ast_schema_sha256"] == certs.AST_SCHEMA_SHA256
    assert upgraded["tasks"] == original["tasks"]
    assert upgraded["expert_library"] == original["expert_library"]
    assert certs.verify_portable_certificate(upgraded)["domain_rows_checked"] == 145607


def test_v2_unknown_ast_schema_fails_closed_even_with_new_outer_hash():
    artifact = _tiny_v2_artifact()
    artifact["ast_schema"] = "hagi.expr.v999"
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "unsupported AST schema" in result["error"]


def test_v2_rejects_bool_as_format_version():
    artifact = _tiny_v2_artifact()
    artifact["format_version"] = True
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "must be an integer" in result["error"]


def test_duplicate_expert_indices_are_rejected_even_after_rehash():
    artifact = _tiny_v2_artifact()
    artifact["expert_library"]["ID_X"] = {
        "family": "unary",
        "program": {"arity": 1, "expr": {"op": "var", "args": [], "value": 0}, "venn_mask": None},
        "domains": [[0, 1]],
        "valid_rule": "all",
        "code_length": 7,
        "expert_index": 0,
    }
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "must be unique" in result["error"]


def test_recursive_unused_expert_is_rejected_as_cycle():
    artifact = _tiny_v2_artifact()
    name = "UNUSED_SELF_RECURSIVE"
    expression = {
        "op": "call",
        "args": [{"op": "var", "args": [], "value": 0}],
        "value": name,
    }
    program = {"arity": 1, "expr": expression, "venn_mask": None}
    library = artifact["expert_library"]
    indices = {k: v["expert_index"] for k, v in library.items()}
    indices[name] = len(indices)
    library[name] = {
        "family": "adversarial-test",
        "program": program,
        "domains": [[0]],
        "valid_rule": "all",
        "code_length": certs._program_bits(program, indices),
        "expert_index": indices[name],
    }
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "recursive expert call cycle" in result["error"]


def test_oversized_ast_payload_is_rejected_before_domain_evaluation(monkeypatch):
    artifact = _tiny_v2_artifact()
    monkeypatch.setattr(certs, "MAX_AST_PAYLOAD_ITEMS", 2)
    name = "OVERSIZED_SET"
    program = {
        "arity": 1,
        "expr": {
            "op": "in_set",
            "args": [{"op": "var", "args": [], "value": 0}],
            "value": [0, 1, 2],
        },
        "venn_mask": None,
    }
    artifact["expert_library"][name] = {
        "family": "adversarial-test",
        "program": program,
        "domains": [[0]],
        "valid_rule": "all",
        "code_length": 1,
        "expert_index": 1,
    }
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "oversized in_set payload" in result["error"]


def test_self_hash_is_not_a_signature_and_provenance_notice_is_explicit():
    artifact = _tiny_v2_artifact()
    task = artifact["tasks"][0]
    expert = artifact["expert_library"]["BOOL_FALSE"]
    # Change the program and all embedded labels consistently, then recompute
    # the unkeyed SHA-256. This remains internally consistent by design.
    expert["program"]["venn_mask"] = 1
    task["expected_outputs"] = [1, 0, 0, 0]
    task["labels_sha256"] = certs._sha256({
        "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "outputs": [1, 0, 0, 0],
    })
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is True
    assert "provenance not authenticated" in result["scope"]


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate-keys.json"
    path.write_text('{"format":"bad","format":"bad2"}', encoding="utf-8")
    result = certs.verify_portable_certificate(path)
    assert result["valid"] is False
    assert "duplicate JSON object key" in result["error"]


def test_standalone_verifier_runs_under_isolated_python_for_v1_and_v2(tmp_path):
    v2_path = tmp_path / "certificate-v2.json"
    original = json.loads(V1_CERT.read_text(encoding="utf-8"))
    certs.upgrade_portable_certificate_v1_to_v2(original, v2_path)
    for path, expected_format in ((V1_CERT, certs.FORMAT_V1), (v2_path, certs.FORMAT)):
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "hagi" / "certificates.py"), str(path)],
            check=False, capture_output=True, text=True, timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["valid"] is True
        assert payload["format"] == expected_format
        assert payload["tasks"] == 62


def test_detached_signature_verifies_with_pure_stdlib_public_key():
    artifact_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.json"
    signature_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.sig.json"
    public_key_path = ROOT / "reports" / "hagi_v6_demo_public_key.json"
    result = certs.verify_portable_signature(artifact_path, signature_path, public_key_path)
    assert result["valid"] is True
    assert result["algorithm"] == "RSA_PKCS1_V1_5_SHA256"
    assert "out-of-band trust" in result["trust_basis"]


def test_signed_cli_verification_works_in_isolated_python():
    artifact_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.json"
    signature_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.sig.json"
    public_key_path = ROOT / "reports" / "hagi_v6_demo_public_key.json"
    result = subprocess.run(
        [
            sys.executable, "-I", str(ROOT / "hagi" / "certificates.py"),
            str(artifact_path), "--signature", str(signature_path),
            "--trusted-public-key", str(public_key_path),
        ],
        check=False, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["signature_verification"]["valid"] is True


def test_detached_signature_detects_consistent_artifact_tampering():
    artifact_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.json"
    signature_path = ROOT / "reports" / "hagi_v9_portable_certificates_v2.sig.json"
    public_key_path = ROOT / "reports" / "hagi_v6_demo_public_key.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    task = next(t for t in artifact["tasks"] if t["name"] == "BOOL_FALSE")
    expert = artifact["expert_library"]["BOOL_FALSE"]
    expert["program"]["venn_mask"] = 1
    task["expected_outputs"] = [1, 0, 0, 0]
    task["labels_sha256"] = certs._sha256({
        "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "outputs": [1, 0, 0, 0],
    })
    _rehash(artifact)
    assert certs.verify_portable_certificate(artifact)["valid"] is True
    result = certs.verify_portable_signature(artifact, signature_path, public_key_path)
    assert result["valid"] is False
    assert "different artifact digest" in result["error"]


def test_ast_schema_digest_covers_every_supported_operator():
    descriptor = certs._AST_SCHEMA_DESCRIPTOR["operators"]
    assert set(descriptor) == set(certs._AST_OP_ARITY)
    assert descriptor["call"]["arity"] == "callee-arity"
    assert descriptor["if_cond"]["semantics"].startswith("lazy")
    assert descriptor["bit_not8"]["semantics"] == "(~x) & 255"
    assert len(certs.AST_SCHEMA_SHA256) == 64


def test_cumulative_domain_row_budget_is_enforced(monkeypatch):
    artifact = _tiny_v2_artifact()
    monkeypatch.setattr(certs, "MAX_TOTAL_DOMAIN_ROWS", 3)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "total declared domain rows" in result["error"]


def test_total_ast_node_budget_is_enforced(monkeypatch):
    artifact = _tiny_v2_artifact()
    artifact["expert_library"]["ID_X"] = {
        "family": "unary",
        "program": {"arity": 1, "expr": {"op": "var", "args": [], "value": 0}, "venn_mask": None},
        "domains": [[0, 1]],
        "valid_rule": "all",
        "code_length": 7,
        "expert_index": 1,
    }
    monkeypatch.setattr(certs, "MAX_TOTAL_AST_NODES", 0)
    _rehash(artifact)
    result = certs.verify_portable_certificate(artifact)
    assert result["valid"] is False
    assert "total expert AST node limit" in result["error"]
