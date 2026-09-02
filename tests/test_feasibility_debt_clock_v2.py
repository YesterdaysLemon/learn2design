from __future__ import annotations

import builtins
import inspect
import io
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

pytest.importorskip("dfbench")
pytest.importorskip("jax")

from experiments.candidates.feasibility_debt_clock_v2 import (
    FeasibilityDebtBatchedRestartAdamV2,
)
from experiments.candidates import feasibility_debt_clock_v2_fixture as fixture
from experiments.candidates.feasibility_debt_clock_v2_source import (
    EXPECTED_HUNKS,
    verify_source_boundary,
)


def _synthetic_child_payload(*, passed: bool = True) -> dict[str, object]:
    outcomes = {key: passed for key in fixture.CASE_KEYS}
    payload: dict[str, object] = {
        "study_id": fixture.STUDY_ID,
        "invocation_revision": "a" * 40,
        "plan_revision": fixture.PLAN_REVISION,
        "plan_sha256": fixture.PLAN_SHA256,
        "candidate_source_sha256": "b" * 64,
        "fixture_source_sha256": "c" * 64,
        "protected_source_sha256": "d" * 64,
        "case_count": 10,
        "case_outcomes": outcomes,
        "case_roots": {key: "e" * 64 for key in fixture.CASE_KEYS},
        "all_cases_passed": passed,
        "source_boundary_root_sha256": "f" * 64,
    }
    payload["core_root_sha256"] = fixture._sha256(
        fixture._canonical_json(payload)
    )
    return payload


@pytest.mark.integration
def test_v2_candidate_source_delta_is_exactly_pinned() -> None:
    projection = verify_source_boundary()

    assert projection["valid"] is True
    assert projection["exact_method_set"] is True
    assert projection["methods_equal"] is True
    assert projection["region_labels"] == [name for name, _ in EXPECTED_HUNKS]


@pytest.mark.integration
def test_v2_progress_mode_is_required_and_closed() -> None:
    parameters = inspect.signature(
        FeasibilityDebtBatchedRestartAdamV2.optimize
    ).parameters

    assert parameters["progress_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["progress_mode"].default is inspect.Parameter.empty
    smoke = fixture._batches(
        1,
        lambda _batch, lane: 4.0 + lane,
        lambda _batch, lane: 2.0 + lane,
        lambda _batch, _lane: False,
    )
    with pytest.raises(ValueError, match="progress_mode"):
        fixture._run_optimizer(
            "candidate",
            "not-a-mode",
            smoke,
            seed=94001,
            patience=4,
        )


@pytest.mark.integration
def test_v2_total_loss_smoke_matches_protected_projection() -> None:
    smoke = fixture._batches(
        3,
        lambda batch, lane: 15.0 - batch - lane * 0.25,
        lambda batch, lane: 6.0 - batch * 0.5 + lane * 0.1,
        lambda batch, lane: batch == 2 and lane % 2 == 0,
    )
    protected = fixture._run_optimizer(
        "protected", None, smoke, seed=94003, patience=6
    )
    candidate = fixture._run_optimizer(
        "candidate", "total_loss", smoke, seed=94003, patience=6
    )

    assert protected["root"] == candidate["root"]


@pytest.mark.integration
def test_v2_post_feasibility_smoke_uses_total_loss_when_infeasible() -> None:
    smoke = [
        [(10.0 + lane, 3.0 + lane, False) for lane in range(4)],
        [(9.0 + lane, 0.0, True) for lane in range(4)],
        [(8.0 + lane, 50.0 + lane, False) for lane in range(4)],
    ]
    run = fixture._run_optimizer(
        "candidate",
        "feasibility_debt",
        smoke,
        seed=94009,
        patience=5,
    )

    assert [
        event["observed_member_improved"].astype(bool).tolist()
        for event in run["capture"].raw_events
    ] == [[True] * 4, [True] * 4, [True] * 4]
    assert run["capture"].raw_events[-1][
        "progress_ever_feasible_after"
    ].astype(bool).tolist() == [True] * 4


@pytest.mark.integration
def test_v2_auxiliary_extra_power_leaf_fails_before_transition() -> None:
    smoke = fixture._batches(
        1,
        lambda _batch, lane: 7.0 + lane,
        lambda _batch, lane: 1.0 + lane,
        lambda _batch, _lane: False,
    )

    with pytest.raises(ValueError, match="unexpected power leaves"):
        fixture._run_optimizer(
            "candidate",
            "feasibility_debt",
            smoke,
            seed=94021,
            patience=5,
            attack="extra-power-value",
        )


def test_v2_closed_parent_failure_has_exact_valid_schema() -> None:
    payload = fixture._closed_parent_failure("1" * 40)

    assert fixture._validate_parent_payload(payload) is True
    assert tuple(payload["case_outcomes"]) == fixture.CASE_KEYS
    assert tuple(payload["transport_outcomes"]) == fixture.TRANSPORT_KEYS
    assert not any(payload["case_outcomes"].values())
    assert not any(payload["transport_outcomes"].values())
    assert payload["action"] == "park_feasibility_debt_v2"


def test_v2_parent_pass_requires_all_relational_hashes() -> None:
    payload = fixture._closed_parent_failure("1" * 40)
    payload.update(
        {
            "candidate_source_sha256": "a" * 64,
            "fixture_source_sha256": "b" * 64,
            "protected_source_sha256": "c" * 64,
            "case_outcomes": {key: True for key in fixture.CASE_KEYS},
            "transport_outcomes": {
                key: True for key in fixture.TRANSPORT_KEYS
            },
            "all_cases_passed": True,
            "runs_equal": True,
            "source_boundary_root_sha256": "d" * 64,
            "process_replay_root_sha256": "e" * 64,
            "action": "approve_feasibility_debt_v2_for_fresh_panel_planning",
        }
    )

    assert fixture._validate_parent_payload(payload) is True

    missing_replay = dict(payload)
    missing_replay["process_replay_root_sha256"] = None
    assert fixture._validate_parent_payload(missing_replay) is False


def test_v2_child_schema_is_canonical_and_relational() -> None:
    payload = _synthetic_child_payload()
    raw = fixture._canonical_json(payload) + b"\n"

    assert fixture._validate_child_payload(payload, raw) is True

    inconsistent = dict(payload)
    inconsistent["all_cases_passed"] = False
    inconsistent_raw = fixture._canonical_json(inconsistent) + b"\n"
    assert fixture._validate_child_payload(inconsistent, inconsistent_raw) is False

    extra = dict(payload)
    extra["unexpected"] = True
    extra_raw = fixture._canonical_json(extra) + b"\n"
    assert fixture._validate_child_payload(extra, extra_raw) is False


def test_v2_run_child_preserves_failed_scientific_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _synthetic_child_payload(passed=False)
    raw = fixture._canonical_json(payload) + b"\n"

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout=raw, stderr=b""
        )

    monkeypatch.setattr(fixture.subprocess, "run", fake_run)
    receipt, observed_raw, observed = fixture._run_child("2" * 40, 1)

    assert all(receipt.values())
    assert observed_raw == raw
    assert observed == payload
    assert observed["all_cases_passed"] is False


def test_v2_child_environment_scrubs_secrets_and_forces_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "do-not-forward")
    monkeypatch.setenv("SAFE_MARKER", "retained")

    environment = fixture._scrubbed_environment()

    assert "EXAMPLE_API_KEY" not in environment
    assert environment["SAFE_MARKER"] == "retained"
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["NO_PROXY"] == "*"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.integration
def test_v2_fresh_process_guard_smoke_is_read_only() -> None:
    code = (
        "from experiments.candidates.feasibility_debt_clock_v2_fixture "
        "import _install_child_guards; "
        "_install_child_guards(); "
        "import jax.numpy as jnp; "
        "from experiments.candidates.feasibility_debt_clock_v2_source "
        "import verify_source_boundary; "
        "assert float(jnp.sum(jnp.asarray([1.0, 2.0]))) == 3.0; "
        "assert verify_source_boundary()['valid'] is True"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=fixture._scrubbed_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_v2_child_guards_allow_reads_and_deny_effects(tmp_path: Path) -> None:
    path = tmp_path / "read-only.txt"
    path.write_text("fixture", encoding="utf-8")
    originals = {
        "builtins_open": builtins.open,
        "io_open": io.open,
        "os_open": os.open,
        "socket": socket.socket,
        "create_connection": socket.create_connection,
        "popen": subprocess.Popen,
    }
    try:
        fixture._install_child_guards()
        assert path.read_text(encoding="utf-8") == "fixture"
        with pytest.raises(RuntimeError, match="operation denied"):
            builtins.open(path, "w", encoding="utf-8")
        with pytest.raises(RuntimeError, match="operation denied"):
            socket.socket()
        with pytest.raises(RuntimeError, match="operation denied"):
            subprocess.Popen(["python", "-V"])
    finally:
        builtins.open = originals["builtins_open"]
        io.open = originals["io_open"]
        os.open = originals["os_open"]
        socket.socket = originals["socket"]
        socket.create_connection = originals["create_connection"]
        subprocess.Popen = originals["popen"]
