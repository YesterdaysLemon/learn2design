from __future__ import annotations

import copy
import json
import subprocess
import sys

from experiments.local_lab.multistep_td_action_prefix_v3 import (
    CASE_REQUIRED_FIELDS,
    EXPECTED_FAMILY_SHA256,
    EXPECTED_RANDOM_STREAM_SHA256,
    run_study,
)
from tools import run_local_lab as lab_controller


def _completed_result() -> dict[str, object]:
    result = copy.deepcopy(run_study(include_process_isolation=False))
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = (
        "synthetic_four_step_synchronous_td_propagation_confirmed_for_harness"
    )
    return result


def test_v3_focused_projection_passes_exact_frozen_contract() -> None:
    result = run_study(include_process_isolation=False)

    assert result["study_id"] == "multistep-td-action-prefix-v3"
    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    assert all(
        case["passed"]
        for name, case in result["cases"].items()
        if name != "process_isolation"
    )
    assert {
        name: set(case)
        for name, case in result["cases"].items()
    } == {
        name: set(fields) for name, fields in CASE_REQUIRED_FIELDS.items()
    }
    assert result["fixture"]["expected_family_sha256"] == EXPECTED_FAMILY_SHA256
    assert (
        result["fixture"]["expected_random_stream_sha256"]
        == EXPECTED_RANDOM_STREAM_SHA256
    )
    assert result["cases"]["complete_family_replay"]["rows"] == 122880
    assert result["cases"]["synchronous_td_order"][
        "positive_cells_by_sweep"
    ] == [2, 4, 6, 8]
    assert result["cases"]["multistep_td_recovery"][
        "postfit_test_macro_return"
    ] == 1.0
    assert result["cases"]["transition_target_control"][
        "test_macro_return"
    ] == 0.5
    assert result["cases"]["reward_origin_control"][
        "test_macro_return"
    ] == 0.5
    assert result["cases"]["signal_attribution_control"][
        "refit_test_macro_return"
    ] == 0.5


def test_v3_completed_projection_passes_controller_validator() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "multistep-td-action-prefix-v3")
    lab_controller._validate_study_result(
        "multistep-td-action-prefix-v3", entry, _completed_result()
    )


def test_v3_strict_attack_boundaries_reject() -> None:
    result = run_study(include_process_isolation=False)
    timing = result["cases"]["lazy_information_boundary"]
    pending = result["cases"]["pending_transition_authentication"]
    trace = result["cases"]["keyed_trace_authentication"]

    assert timing["passed"]
    assert timing["attacks_rejected"] == timing["attack_classes"] == 20
    assert pending["passed"]
    assert pending["pending_cleared_after_rejection"]
    assert trace["passed"]
    assert trace["attacks_rejected"] == trace["attack_classes"] == 20


def test_v3_fresh_workers_match_complete_local_projection() -> None:
    result = run_study(include_process_isolation=True)

    assert result["status"] == "passed"
    assert result["cases"]["process_isolation"]["passed"] is True
    assert result["action"] == (
        "synthetic_four_step_synchronous_td_propagation_confirmed_for_harness"
    )


def test_v3_result_is_bounded_and_sanitized() -> None:
    result = run_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    recursive_keys: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            recursive_keys.update(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result)
    assert len(encoded) < 100_000
    assert recursive_keys.isdisjoint(
        {
            "raw_actions",
            "raw_observations",
            "raw_rewards",
            "q_table",
            "policy_state",
            "trajectories",
            "returns",
            "target_values",
            "q_values",
            "policy_parameters",
            "private_evidence",
        }
    )
    assert str(lab_controller.ROOT).lower() not in encoded.lower()


def test_v3_worker_network_gate_is_installed_before_fixture_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from experiments.local_lab.multistep_td_action_prefix_v3_worker "
                "import _disable_network; _disable_network(); import socket; "
                "\ntry:\n socket.create_connection(('127.0.0.1', 9))"
                "\nexcept RuntimeError:\n print('disabled')"
                "\nelse:\n raise SystemExit(2)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "disabled"
