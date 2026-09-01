from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from experiments.local_lab import (
    constraint_progress_isolated_runtime_forensics_v1 as probe,
)
from tools import run_local_lab as controller


probe._late_imports()


def _child_receipt(
    case_id: str,
    *,
    failure: str | None = None,
) -> dict[str, object]:
    target_index = probe.STAGES.index(case_id)
    if failure is None:
        reached = case_id
        status = "passed"
        reached_index = target_index
    else:
        failure_index = probe.STAGES.index(failure)
        reached = None if failure_index == 0 else probe.STAGES[failure_index - 1]
        status = "failed"
        reached_index = failure_index - 1
    site_hash = (
        probe._expected_site_commitment()
        if reached_index >= probe.STAGES.index("site_discovery")
        else None
    )
    if reached_index >= probe.STAGES.index("runtime_identity"):
        identity_hash = probe._expected_identity_commitment()
        identity_matches = True
    elif failure == "runtime_identity":
        identity_hash = "c" * 64
        identity_matches = False
    else:
        identity_hash = None
        identity_matches = None
    return {
        "schema_version": 1,
        "checkpoint_id": probe.CHECKPOINT_ID,
        "case_id": case_id,
        "target_stage": case_id,
        "reached_stage": reached,
        "status": status,
        "error_code": failure,
        "environment_keys_sha256": probe._environment_keys_sha256(
            probe._probe_environment()
        ),
        "site_commitment_sha256": site_hash,
        "identity_sha256": identity_hash,
        "identity_matches": identity_matches,
        "network_attempts_rejected": (
            probe.NETWORK_PROBES
            if reached_index >= probe.STAGES.index("network_denial")
            else 0
        ),
    }


def _observation(case_id: str, *, failure: str | None = None) -> dict[str, object]:
    receipt = _child_receipt(case_id, failure=failure)
    return {
        "case_id": case_id,
        "stdout_bytes": len(probe._canonical_line(receipt)),
        "stderr_bytes": 0,
        "return_code": 0,
        "child_receipt": receipt,
        "surviving_processes": 0,
        "error_code": None,
    }


def test_frozen_identity_and_case_order() -> None:
    assert probe.PLAN_REVISION == "b6efe5cfaca849fdab4531fb4dcdea04823f0a2a"
    assert probe.FAILED_ATTEMPT_REVISION == "9413cd4982cab74887fa8c7dc3dd4bf9c4d8508a"
    assert probe.CASE_IDS == probe.STAGES
    assert len(probe.CASE_IDS) == 9
    assert probe.MAX_OUTPUT_BYTES == 16_384
    assert probe.TIMEOUT_SECONDS == 60.0


def test_probe_environment_is_closed_and_exact() -> None:
    observed = probe._probe_environment(
        {
            "Path": "safe-path",
            "TEMP": "safe-temp",
            "AWS_SECRET_ACCESS_KEY": "forbidden",
            "L2D_UNEXPECTED": "forbidden",
        }
    )
    assert observed["PATH"] == "safe-path"
    assert observed["TEMP"] == "safe-temp"
    assert "AWS_SECRET_ACCESS_KEY" not in observed
    assert "L2D_UNEXPECTED" not in observed
    assert observed["L2D_PLAN_REVISION"] == probe.FAILED_PLAN_REVISION
    assert observed["L2D_STUDY_REVISION"] == probe.FAILED_ATTEMPT_REVISION
    assert observed["L2D_CONTRACT_SHA256"] == probe.FAILED_CONTRACT_SHA256
    assert set(observed).issubset(probe.ALLOWED_ENVIRONMENT)


@pytest.mark.parametrize("case_id", probe.CASE_IDS)
def test_passed_child_receipt_contract(case_id: str) -> None:
    receipt = _child_receipt(case_id)
    assert probe._validate_child_receipt(receipt, case_id) is receipt


def test_runtime_identity_failure_receipt_is_closed() -> None:
    receipt = _child_receipt("composite", failure="runtime_identity")
    assert receipt["reached_stage"] == "numpy_import"
    assert receipt["identity_matches"] is False
    assert probe._validate_child_receipt(receipt, "composite") is receipt


def test_child_receipt_rejects_late_or_inconsistent_failure() -> None:
    late = _child_receipt("contract_environment")
    late["status"] = "failed"
    late["error_code"] = "numpy_import"
    with pytest.raises(probe.ProbeFailure):
        probe._validate_child_receipt(late, "contract_environment")

    inconsistent = _child_receipt("composite", failure="site_discovery")
    inconsistent["reached_stage"] = "network_denial"
    inconsistent["site_commitment_sha256"] = "b" * 64
    with pytest.raises(probe.ProbeFailure):
        probe._validate_child_receipt(inconsistent, "composite")


def test_observation_rejects_false_byte_count() -> None:
    observation = _observation("composite")
    observation["stdout_bytes"] += 1
    with pytest.raises(probe.ProbeFailure):
        probe._validate_observation(observation, "composite")


def test_prefix_accepts_all_passed_cases() -> None:
    observations = [_observation(case_id) for case_id in probe.CASE_IDS]
    assert probe._derive_prefix(observations) == (True, None)


def test_prefix_accepts_one_deterministic_earliest_stage() -> None:
    failure = "numpy_import"
    failure_index = probe.STAGES.index(failure)
    observations = [
        _observation(case_id, failure=failure if index >= failure_index else None)
        for index, case_id in enumerate(probe.CASE_IDS)
    ]
    assert probe._derive_prefix(observations) == (True, failure)


def test_prefix_rejects_nonmonotone_or_disagreeing_receipts() -> None:
    observations = [_observation(case_id) for case_id in probe.CASE_IDS]
    observations[5] = _observation("numpy_import", failure="numpy_import")
    assert probe._derive_prefix(observations) == (False, None)

    failure_index = probe.STAGES.index("site_discovery")
    observations = [
        _observation(
            case_id,
            failure=("site_discovery" if index >= failure_index else None),
        )
        for index, case_id in enumerate(probe.CASE_IDS)
    ]
    observations[-1] = _observation("composite", failure="runtime_identity")
    assert probe._derive_prefix(observations) == (False, None)


def test_run_validator_recomputes_body_and_relations() -> None:
    observations = [_observation(case_id) for case_id in probe.CASE_IDS]
    core = {
        "cases": observations,
        "first_failure": None,
        "child_launches": len(probe.CASE_IDS),
        "stderr_bytes": 0,
        "surviving_processes": 0,
        "prefix_valid": True,
        "passed": True,
    }
    value = {
        **core,
        "body_sha256": probe._sha256_domain(
            b"L2D-runtime-forensics-v1/run",
            probe._canonical_line(core),
        ),
    }
    assert probe._validate_run(value) is value
    value["body_sha256"] = "0" * 64
    with pytest.raises(probe.ProbeFailure):
        probe._validate_run(value)


def test_json_parser_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(probe.ProbeFailure):
        probe._loads(b'{"a":1,"a":2}\n')
    with pytest.raises(probe.ProbeFailure):
        probe._loads(b'{"a":NaN}\n')


def test_source_is_import_inert_and_excludes_retired_science() -> None:
    source_path = Path(probe.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module)
    assert top_level_imports == ["__future__", "os", "sys"]
    assert "constraint_aware_progress_toy" not in source
    assert "run_local_lab" not in source
    assert "experiments.local_lab" not in source
    assert "import_module(\"submission\")" not in source


def test_controller_quarantines_both_failed_constraint_progress_ids() -> None:
    assert controller.CONSTRAINT_PROGRESS_V1 in controller.QUARANTINED_STUDIES
    assert controller.CONSTRAINT_PROGRESS_V2 in controller.QUARANTINED_STUDIES


def test_controller_refuses_v2_study_before_output_validation(monkeypatch) -> None:
    output_called = False

    def forbidden_output(_path):
        nonlocal output_called
        output_called = True
        raise AssertionError("output validation became reachable")

    monkeypatch.setattr(controller, "_validate_output", forbidden_output)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local_lab.py", "--study", controller.CONSTRAINT_PROGRESS_V2, "--output", "x"],
    )
    with pytest.raises(controller.QuarantinedStudyError):
        controller.main()
    assert output_called is False


def test_controller_refuses_v2_resume_before_resume_function(monkeypatch) -> None:
    resume_called = False

    def forbidden_resume():
        nonlocal resume_called
        resume_called = True
        raise AssertionError("historical resume became reachable")

    monkeypatch.setattr(controller, "_resume_constraint_progress_v2", forbidden_resume)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local_lab.py", "--resume-constraint-progress-v2"],
    )
    with pytest.raises(controller.QuarantinedStudyError):
        controller.main()
    assert resume_called is False
