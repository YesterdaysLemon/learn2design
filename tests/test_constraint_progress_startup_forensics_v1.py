from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


from experiments.local_lab import constraint_progress_startup_forensics_v1 as probe


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "experiments/local_lab/constraint_progress_startup_forensics_v1.py"


def test_probe_is_standard_library_only_and_separate_from_failed_study() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__",
        "argparse",
        "ctypes",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "typing",
    }
    assert "constraint_aware_progress_toy" not in source
    assert "submission" not in source
    assert "numpy" not in source.lower()
    assert "run_local_lab" not in source
    module_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert module_calls == []


def test_frozen_plan_identity_cases_and_environment_are_exact() -> None:
    assert probe.PLAN_REVISION == "59f3c5a5ab5a985b1a67477b68eb6eb9976c2f3f"
    assert probe.CASE_IDS == (
        "gate_only_empty",
        "nested_empty",
        "nested_fragmented",
        "nested_large",
        "nested_wrong_gate",
        "nested_truncated_gate",
        "nested_no_gate",
        "nested_short_length",
        "nested_short_payload",
        "nested_oversized_length",
        "nested_trailing_input",
    )
    assert sum(item[1] for item in probe.CASE_SPECS) == 10
    assert sum(item[4] == "accepted" for item in probe.CASE_SPECS) == 4
    assert sum(item[4] == "rejected" for item in probe.CASE_SPECS) == 7
    environment = probe._probe_environment({"PATH": "x", "SECRET": "forbidden"})
    assert "SECRET" not in environment
    assert environment["PATH"] == "x"
    assert {
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "LEARN2DESIGN_LOCAL_LAB_NETWORK",
        "PYTHONHASHSEED",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    } <= set(environment)


def test_payload_and_all_frames_are_exact_and_bounded() -> None:
    assert probe._payload(0) == b""
    expected_first = hashlib.sha256(
        b"L2D-startup-forensics-v1/payload\0" + (0).to_bytes(4, "little")
    ).digest()
    assert probe._payload(32) == expected_first
    assert probe._frame("gate_only_empty") == probe.CHILD_GATE + b"\0\0\0\0"
    assert probe._frame("nested_wrong_gate").startswith(probe.CHILD_GATE[:-1] + b"!")
    assert probe._frame("nested_truncated_gate") == probe.CHILD_GATE[:-1]
    assert probe._frame("nested_no_gate") == b""
    assert len(probe._frame("nested_large")) == len(probe.CHILD_GATE) + 4 + 32_768
    assert probe._frame("nested_trailing_input").endswith(b"\0\0\0\0\x01")


@pytest.mark.parametrize("case_id", probe.CASE_IDS)
def test_child_frame_consumer_exercises_every_frozen_case(case_id: str) -> None:
    environment = probe._probe_environment()
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-P",
            str(SOURCE),
            "--mode",
            "child",
            "--case",
            case_id,
        ],
        cwd=ROOT,
        env=environment,
        input=probe._frame(case_id),
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    receipt = probe._loads(completed.stdout)
    _pairs, environment_sha256 = probe._environment_receipt(environment)
    probe._validate_child_receipt(receipt, case_id, environment_sha256)
    expected_status = probe.CASE_BY_ID[case_id][4]
    expected_error = probe.CASE_BY_ID[case_id][5]
    expected_payload = (
        probe._payload(probe.CASE_BY_ID[case_id][3])
        if expected_status == "accepted"
        else b""
    )
    assert receipt["status"] == expected_status
    assert receipt["error_code"] == expected_error
    assert receipt["payload_bytes"] == len(expected_payload)
    assert receipt["payload_sha256"] == hashlib.sha256(expected_payload).hexdigest()


def test_duplicate_keys_and_mutated_receipts_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate-key"):
        probe._loads('{"x":1,"x":2}')
    environment = probe._probe_environment()
    _pairs, environment_sha256 = probe._environment_receipt(environment)
    receipt = {
        "schema_version": 1,
        "case_id": "gate_only_empty",
        "environment_sha256": environment_sha256,
        "status": "accepted",
        "error_code": None,
        "payload_bytes": 0,
        "payload_sha256": hashlib.sha256(b"").hexdigest(),
    }
    probe._validate_child_receipt(receipt, "gate_only_empty", environment_sha256)
    mutated = json.loads(json.dumps(receipt))
    mutated["case_id"] = "nested_empty"
    with pytest.raises(probe.ProbeError, match="child_schema"):
        probe._validate_child_receipt(mutated, "gate_only_empty", environment_sha256)


def test_partial_parent_observation_contract_is_closed() -> None:
    environment = probe._probe_environment()
    _pairs, environment_sha256 = probe._environment_receipt(environment)
    observation = probe._empty_observation("nested_empty")
    observation["error_code"] = "job_create"
    probe._validate_observation(observation, "nested_empty", environment_sha256)
    observation["surviving_descendants"] = -1
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_observation(observation, "nested_empty", environment_sha256)

    query_failure = probe._empty_observation("nested_empty")
    query_failure.update(
        {
            "inner_job_assigned": True,
            "membership_before_gate": True,
            "stdout_bytes": 3,
            "stderr_bytes": 2,
            "return_code": 1,
            "surviving_descendants": -1,
            "error_code": "job_query",
        }
    )
    probe._validate_observation(query_failure, "nested_empty", environment_sha256)
    query_failure["error_code"] = "timeout"
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_observation(query_failure, "nested_empty", environment_sha256)


def test_child_relation_failure_requires_an_actual_mismatch() -> None:
    environment = probe._probe_environment()
    _pairs, environment_sha256 = probe._environment_receipt(environment)
    receipt = {
        "schema_version": 1,
        "case_id": "nested_empty",
        "environment_sha256": environment_sha256,
        "status": "accepted",
        "error_code": None,
        "payload_bytes": 0,
        "payload_sha256": hashlib.sha256(b"").hexdigest(),
    }
    observation = probe._empty_observation("nested_empty")
    observation.update(
        {
            "inner_job_assigned": True,
            "membership_before_gate": True,
            "child_receipt": receipt,
            "stdout_bytes": len(probe._canonical(receipt)),
            "return_code": 0,
            "error_code": "child_relation",
        }
    )
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_observation(observation, "nested_empty", environment_sha256)

    mismatched = copy.deepcopy(observation)
    mismatched_receipt = mismatched["child_receipt"]
    mismatched_receipt["status"] = "rejected"
    mismatched_receipt["error_code"] = "gate_read"
    mismatched["stdout_bytes"] = len(probe._canonical(mismatched_receipt))
    probe._validate_observation(mismatched, "nested_empty", environment_sha256)


def _passing_parent() -> dict[str, object]:
    environment = probe._probe_environment()
    pairs, environment_sha256 = probe._environment_receipt(environment)
    children = []
    for case_id in probe.CASE_IDS:
        expected_status = probe.CASE_BY_ID[case_id][4]
        expected_error = probe.CASE_BY_ID[case_id][5]
        expected_payload = (
            probe._payload(probe.CASE_BY_ID[case_id][3])
            if expected_status == "accepted"
            else b""
        )
        receipt = {
            "schema_version": 1,
            "case_id": case_id,
            "environment_sha256": environment_sha256,
            "status": expected_status,
            "error_code": expected_error,
            "payload_bytes": len(expected_payload),
            "payload_sha256": hashlib.sha256(expected_payload).hexdigest(),
        }
        observation = probe._empty_observation(case_id)
        inner_required = probe.CASE_BY_ID[case_id][1]
        observation.update(
            {
                "inner_job_assigned": inner_required,
                "membership_before_gate": inner_required,
                "child_receipt": receipt,
                "stdout_bytes": len(probe._canonical(receipt)),
                "return_code": 0,
                "passed": True,
            }
        )
        children.append(observation)
    value = {
        "schema_version": 1,
        "checkpoint_id": probe.CHECKPOINT_ID,
        "environment_pairs": pairs,
        "environment_sha256": environment_sha256,
        "outer_membership_before_gate": True,
        "children": children,
        "child_launches": 11,
        "inner_assignments": 10,
        "inner_memberships_before_gate": 10,
        "accepted_frames": 4,
        "rejected_frames": 7,
        "child_stderr_bytes": 0,
        "surviving_descendants": 0,
        "passed": True,
        "error_code": None,
    }
    probe._validate_parent(value, pairs, environment_sha256, True)
    return value


def _runner_value(
    monkeypatch: pytest.MonkeyPatch,
    runs: list[dict[str, object]],
    *,
    passed: bool,
    error_code: str | None,
) -> dict[str, object]:
    revision = "a" * 40
    monkeypatch.setattr(probe, "_git", lambda *_args: revision)
    plan_bytes = probe._normalized_bytes(probe.PLAN_PATH)
    source_bytes = probe._normalized_bytes(probe.SOURCE_PATH)
    bodies = [
        probe._canonical(run["body"]) if isinstance(run["body"], dict) else b""
        for run in runs
    ]
    first = bodies[0] if bodies else b""
    second = bodies[1] if len(bodies) > 1 else b""
    return {
        "schema_version": 1,
        "checkpoint_id": probe.CHECKPOINT_ID,
        "plan_revision": probe.PLAN_REVISION,
        "probe_revision": revision,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "probe_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "contract_sha256": hashlib.sha256(
            b"L2D-startup-forensics-v1/contract\0"
            + plan_bytes
            + b"\0"
            + source_bytes
        ).hexdigest(),
        **probe.EXPECTED_RUNTIME_IDENTITY,
        "runs": runs,
        "runs_equal": len(bodies) == 2 and first == second and first != b"",
        "passed": passed,
        "error_code": error_code,
        "receipt_root_sha256": hashlib.sha256(
            b"L2D-startup-forensics-v1/receipt\0" + first + b"\0" + second
        ).hexdigest(),
    }


def _empty_outer_run(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "body_sha256": probe.EMPTY_SHA256,
        "body_bytes": 0,
        "stderr_bytes": 0,
        "return_code": None,
        "surviving_processes": 0,
        "body": None,
    }
    value.update(updates)
    return value


def test_parent_consumes_outer_gate_before_membership_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class RecordingBytesIO(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            events.append("gate-read")
            return super().read(size)

    class Input:
        buffer = RecordingBytesIO(probe.OUTER_GATE)

    projected = probe._probe_environment({})

    def projected_environment(*_args, **_kwargs):
        events.append("environment")
        return projected

    def membership() -> bool:
        events.append("membership")
        raise probe.ProbeError("outer_gate_read")

    monkeypatch.setattr(probe.sys, "stdin", Input())
    monkeypatch.setattr(probe, "_probe_environment", projected_environment)
    monkeypatch.setattr(probe, "_current_process_in_job", membership)
    receipt = probe._parent_receipt()
    assert events[:2] == ["gate-read", "gate-read"]
    assert events[2:] == ["environment", "membership"]
    assert receipt["error_code"] == "outer_gate_read"
    pairs, environment_sha256 = probe._environment_receipt(projected)
    probe._validate_parent(receipt, pairs, environment_sha256, False)


def test_runner_pass_relations_and_receipt_root_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _passing_parent()
    body_bytes = probe._canonical(body)
    run = {
        "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "body_bytes": len(body_bytes),
        "stderr_bytes": 0,
        "return_code": 0,
        "surviving_processes": 0,
        "body": body,
    }
    value = _runner_value(
        monkeypatch, [copy.deepcopy(run), copy.deepcopy(run)], passed=True, error_code=None
    )
    probe._validate_runner_envelope(value)
    mutated = copy.deepcopy(value)
    mutated["receipt_root_sha256"] = "0" * 64
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated)
    mutated = copy.deepcopy(value)
    mutated["runs"][0]["body_bytes"] += 1
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated)


def test_runner_failure_partials_are_error_conditioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_failure = _runner_value(
        monkeypatch, [], passed=False, error_code="runtime_identity"
    )
    probe._validate_runner_envelope(runtime_failure)
    mutated_runtime = copy.deepcopy(runtime_failure)
    mutated_runtime["python_version"] = "0.0.0"
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated_runtime)

    spawn_failure = _runner_value(
        monkeypatch,
        [_empty_outer_run()],
        passed=False,
        error_code="outer_spawn",
    )
    probe._validate_runner_envelope(spawn_failure)
    mutated_spawn = copy.deepcopy(spawn_failure)
    mutated_spawn["runs"][0]["body_bytes"] = 1
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated_spawn)

    stderr_failure = _runner_value(
        monkeypatch,
        [_empty_outer_run(stderr_bytes=1, return_code=0)],
        passed=False,
        error_code="outer_stderr",
    )
    probe._validate_runner_envelope(stderr_failure)
    mutated_stderr = copy.deepcopy(stderr_failure)
    mutated_stderr["runs"][0]["stderr_bytes"] = 0
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated_stderr)

    cap_failure = _runner_value(
        monkeypatch,
        [
            _empty_outer_run(
                body_bytes=probe.MAX_OUTPUT_BYTES + 1,
                return_code=1,
            )
        ],
        passed=False,
        error_code="outer_output_cap",
    )
    probe._validate_runner_envelope(cap_failure)
    mutated_cap = copy.deepcopy(cap_failure)
    mutated_cap["runs"][0]["body_bytes"] = probe.MAX_OUTPUT_BYTES
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated_cap)

    query_failure = _runner_value(
        monkeypatch,
        [_empty_outer_run(return_code=1, surviving_processes=-1)],
        passed=False,
        error_code="outer_job_query",
    )
    probe._validate_runner_envelope(query_failure)
    mutated_query = copy.deepcopy(query_failure)
    mutated_query["runs"][0]["surviving_processes"] = 0
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._validate_runner_envelope(mutated_query)


def test_runner_owns_result_sidecar_verification_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _runner_value(
        monkeypatch, [], passed=False, error_code="runtime_identity"
    )
    temporary_root = Path(probe.tempfile.gettempdir()).resolve()
    before = set(temporary_root.glob(f"{probe.RESULT_DIRECTORY_PREFIX}*"))
    encoded = probe._write_runner_result(value)
    after = set(temporary_root.glob(f"{probe.RESULT_DIRECTORY_PREFIX}*"))
    assert probe._loads(encoded) == value
    assert encoded == probe._canonical(value)
    assert after == before


def test_scratch_containment_and_outer_writer_are_explicit() -> None:
    with pytest.raises(probe.ProbeError, match="outer_relation"):
        probe._checked_scratch_directory(ROOT, "outer_relation")
    source = SOURCE.read_text(encoding="utf-8")
    assert 'name="startup-forensics-outer-gate"' in source
    assert source.count("daemon=False") == 2


def test_runner_unexpected_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe.sys, "argv", [str(SOURCE), "--mode", "runner"])

    def fail_before_result() -> dict[str, object]:
        raise OSError("private exception text")

    monkeypatch.setattr(probe, "_runner_envelope", fail_before_result)
    with pytest.raises(SystemExit) as stopped:
        probe.main()
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
