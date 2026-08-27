from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.run_local_lab as lab_controller

jax = pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
pytest.importorskip("dfbench")
pytest.importorskip("numpy")

pytestmark = pytest.mark.integration

from experiments.local_lab.anchor_lane_stability import (
    REPOSITORY_ROOT,
    run_study,
)
from experiments.local_lab.feasible_progress_clock import (
    run_study as run_feasible_progress_study,
)
from experiments.local_lab.infeasible_prefix_indistinguishability import (
    run_study as run_prefix_boundary_study,
)
from tools.run_local_lab import (
    DuplicateStudyError,
    EXPECTED_SUBMISSION_SOURCE_SHA256,
    EXPECTED_SUBMISSION_TREE_OID,
    _acquire_lease,
    _begin_cycle,
    _git,
    _git_bytes,
    _load_state,
    _release_lease,
    _run_worker,
    _validate_study_result,
    _write_atomic,
)


def test_anchor_lane_stability_frozen_study_passes() -> None:
    result = run_study(include_process_isolation=True)

    assert result["study_id"] == "anchor-lane-stability-v1"
    assert result["status"] == "passed"
    assert result["action"] == "anchor_lane_mechanics_confirmed"
    assert set(result["cases"]) == {
        "diagnostics_disabled_control",
        "exact_twin",
        "exceptional_arithmetic_partial_tail",
        "forced_shared_state_boundary",
        "process_isolation",
        "suffix_invariance",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["environment"]["platform"] == "cpu"
    assert result["fixture"]["population_size"] == 8
    assert result["fixture"]["case_contract"]["forced_shared_state_boundary"][
        "max_evals"
    ] == 32
    assert result["cases"]["suffix_invariance"]["raw_random_hashes_equal"]
    assert result["cases"]["exceptional_arithmetic_partial_tail"][
        "batch_sizes"
    ] == [8, 8, 2]


def test_anchor_lane_result_is_sanitized() -> None:
    result = run_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None

    forbidden = (
        str(REPOSITORY_ROOT),
        "coverage-triage-v1",
        "coverage-robustness-v1",
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
    )
    assert all(value not in encoded for value in forbidden)


def test_feasible_progress_clock_frozen_study_passes() -> None:
    result = run_feasible_progress_study(include_process_isolation=True)

    assert result["study_id"] == "feasible-progress-clock-v1"
    assert result["status"] == "passed"
    assert result["action"] == (
        "finite_infeasible_progress_resets_clock_confirmed"
    )
    assert set(result["cases"]) == {
        "diagnostics_disabled_control",
        "finite_infeasible_descent",
        "finite_infeasible_improve_then_plateau",
        "finite_infeasible_plateau_control",
        "late_feasibility_crossing",
        "mixed_member_clock",
        "process_isolation",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["finite_infeasible_descent"][
        "finite_infeasible_observations"
    ] == 32
    assert result["cases"]["mixed_member_clock"][
        "changed_members_after_boundary"
    ] == [1, 3, 5, 7]
    assert result["environment"]["platform"] == "cpu"


def test_feasible_progress_clock_result_is_sanitized() -> None:
    result = run_feasible_progress_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_infeasible_prefix_boundary_frozen_study_passes() -> None:
    result = run_prefix_boundary_study(include_process_isolation=True)

    assert result["study_id"] == "infeasible-prefix-indistinguishability-v1"
    assert result["status"] == "passed"
    assert result["action"] == (
        "synthetic_identical_prefix_obstruction_confirmed"
    )
    assert set(result["cases"]) == {
        "boundary_sweep",
        "extra_signal_positive_control",
        "action_vector_exhaustion",
        "process_isolation",
        "shared_prefix_identity",
        "witness_partition",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["shared_prefix_identity"]["prefixes_identical"]
    assert result["cases"]["shared_prefix_identity"][
        "forever_prefix_sha256"
    ] == result["cases"]["shared_prefix_identity"]["late_prefix_sha256"]
    assert result["cases"]["action_vector_exhaustion"] == {
        "bound": 13,
        "joint_satisfiers": 0,
        "passed": True,
        "total_action_vectors": 8192,
    }
    assert result["cases"]["witness_partition"] == {
        "bounded_only_policies": 8191,
        "joint_satisfiers": 0,
        "partition_total": 8192,
        "passed": True,
        "preserve_only_policies": 1,
    }
    assert result["cases"]["boundary_sweep"][
        "joint_satisfiers_by_bound"
    ] == [0] * 6
    assert not result["cases"]["extra_signal_positive_control"][
        "prefixes_identical"
    ]
    assert result["environment"]["platform"] == "cpu"


def test_infeasible_prefix_boundary_result_is_sanitized() -> None:
    result = run_prefix_boundary_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_result_validator_requires_exact_sanitized_contract() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "feasible-progress-clock-v1")
    result = run_feasible_progress_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "finite_infeasible_progress_resets_clock_confirmed"

    _validate_study_result("feasible-progress-clock-v1", entry, result)
    result["cases"]["mixed_member_clock"]["parameter_values"] = [[1.0]]
    with pytest.raises(RuntimeError, match="forbidden field"):
        _validate_study_result("feasible-progress-clock-v1", entry, result)


def test_prefix_boundary_validator_requires_exact_fixture_identity() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(
        registry, "infeasible-prefix-indistinguishability-v1"
    )
    result = run_prefix_boundary_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "synthetic_identical_prefix_obstruction_confirmed"

    _validate_study_result(
        "infeasible-prefix-indistinguishability-v1", entry, result
    )
    result["fixture"]["max_bound"] = 12
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(
            "infeasible-prefix-indistinguishability-v1", entry, result
        )


def test_protected_submission_canonical_digest_matches_pin() -> None:
    committed_source = _git_bytes("show", "HEAD:submission/submission.py")

    assert hashlib.sha256(committed_source).hexdigest() == (
        EXPECTED_SUBMISSION_SOURCE_SHA256
    )
    assert _git("rev-parse", "HEAD:submission") == EXPECTED_SUBMISSION_TREE_OID


def test_local_lab_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    first_digest = _write_atomic(output, {"status": "passed"})

    assert len(first_digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "passed"
    }
    with pytest.raises(FileExistsError):
        _write_atomic(output, {"status": "failed"})


def test_local_lab_lease_is_single_writer_and_identity_bound(tmp_path: Path) -> None:
    lock_directory, lease_id = _acquire_lease(
        tmp_path,
        cycle_id="cycle-test",
        revision="a" * 40,
        study="anchor-lane-stability-v1",
    )

    with pytest.raises(RuntimeError, match="lease already exists"):
        _acquire_lease(
            tmp_path,
            cycle_id="cycle-test-2",
            revision="a" * 40,
            study="anchor-lane-stability-v1",
        )
    with pytest.raises(RuntimeError, match="identity changed"):
        _release_lease(lock_directory, "different")

    _release_lease(lock_directory, lease_id)
    assert not lock_directory.exists()


def test_local_lab_state_refuses_overlap(tmp_path: Path) -> None:
    snapshot = {"revision": "a" * 40}
    output = tmp_path / "cycles" / "cycle-test" / "result.json"

    state = _begin_cycle(
        tmp_path,
        cycle_id="cycle-test",
        output=output,
        snapshot=snapshot,
        study="anchor-lane-stability-v1",
    )

    assert state["status"] == "active"
    assert _load_state(tmp_path)["active_cycle"]["cycle_id"] == "cycle-test"
    with pytest.raises(RuntimeError, match="state is not idle"):
        _begin_cycle(
            tmp_path,
            cycle_id="cycle-test-2",
            output=tmp_path / "cycles" / "cycle-test-2" / "result.json",
            snapshot=snapshot,
            study="anchor-lane-stability-v1",
        )


def test_completed_study_refusal_does_not_park_state(tmp_path: Path) -> None:
    original = lab_controller._default_state()
    original["status"] = "awaiting_study"
    original["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "status": "passed",
        }
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", original)

    with pytest.raises(DuplicateStudyError, match="terminal record"):
        _begin_cycle(
            tmp_path,
            cycle_id="duplicate-test",
            output=tmp_path / "cycles" / "duplicate-test" / "result.json",
            snapshot={"revision": "a" * 40},
            study="anchor-lane-stability-v1",
        )

    after = _load_state(tmp_path)
    assert after == original


def test_worker_policy_probe_blocks_network_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-cross-worker-boundary")

    result, _receipt = _run_worker(
        "policy-probe",
        cycle_id="policy-probe",
        heartbeat=lambda _pid, _elapsed: None,
    )

    assert result == {
        "network_blocked": True,
        "sensitive_environment_names": [],
    }


def test_worker_hard_timeout_terminates_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(lab_controller, "CYCLE_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(TimeoutError, match="exceeded"):
        _run_worker(
            "anchor-lane-stability-v1",
            cycle_id="timeout-probe",
            heartbeat=lambda _pid, _elapsed: None,
        )

    assert not list((tmp_path / "worker-tmp").glob("timeout-probe.*"))


def test_controller_end_to_end_terminal_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "anchor-lane-stability-v1")
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "a" * 40,
    }
    output = tmp_path / "cycles" / "controller-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "anchor-lane-stability-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    sidecar = output.with_name("result.json.sha256").read_text(encoding="ascii")
    state = _load_state(tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / "lab-events.jsonl").read_text().splitlines()
    ]
    assert payload["result"]["status"] == "passed"
    assert sidecar.startswith(
        hashlib.sha256(output.read_bytes()).hexdigest() + "  result.json"
    )
    assert state["status"] == "idle"
    assert "anchor-lane-stability-v1" in state["completed_studies"]
    assert {event["event"] for event in events} >= {
        "cycle_completed",
        "cycle_started",
        "heartbeat",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_second_study_end_to_end_leaves_third_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "feasible-progress-clock-v1")
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "b" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "awaiting_study"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "prior-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        }
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "second-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "b" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "feasible-progress-clock-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_third_study_end_to_end_closes_pending_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(
        registry, "infeasible-prefix-indistinguishability-v1"
    )
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "e" * 40,
    }
    prior_anchor = {
        "cycle_id": "anchor-cycle",
        "result_sha256": "a" * 64,
        "revision": "b" * 40,
        "status": "passed",
    }
    prior_clock = {
        "cycle_id": "clock-cycle",
        "result_sha256": "c" * 64,
        "revision": "d" * 40,
        "status": "passed",
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": prior_anchor,
        "feasible-progress-clock-v1": prior_clock,
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "third-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "e" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "infeasible-prefix-indistinguishability-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "awaiting_study"
    assert state["stop_reason"] == "no_approved_study_pending"
    assert state["completed_studies"]["anchor-lane-stability-v1"] == prior_anchor
    assert state["completed_studies"]["feasible-progress-clock-v1"] == prior_clock
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
        "infeasible-prefix-indistinguishability-v1",
    }
    assert not (tmp_path / "lab.lock").exists()
