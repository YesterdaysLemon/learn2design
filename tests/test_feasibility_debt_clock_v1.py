from __future__ import annotations

import inspect

import pytest

pytest.importorskip("dfbench")
pytest.importorskip("jax")

from experiments.candidates.feasibility_debt_clock_v1 import (
    FeasibilityDebtBatchedRestartAdam,
)
from experiments.candidates.feasibility_debt_clock_v1_fixture import (
    _restart_batches,
    _run_optimizer,
    _scrubbed_environment,
)
from experiments.candidates.feasibility_debt_clock_v1_source import (
    EXPECTED_HUNKS,
    verify_source_boundary,
)


@pytest.mark.integration
def test_candidate_source_delta_is_exactly_pinned() -> None:
    projection = verify_source_boundary()

    assert projection["valid"] is True
    assert projection["exact_method_set"] is True
    assert projection["methods_equal"] is True
    assert projection["region_labels"] == [name for name, _ in EXPECTED_HUNKS]


@pytest.mark.integration
def test_progress_mode_is_required_and_closed() -> None:
    parameters = inspect.signature(
        FeasibilityDebtBatchedRestartAdam.optimize
    ).parameters

    assert parameters["progress_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["progress_mode"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="progress_mode"):
        _run_optimizer(
            "candidate",
            "not-a-mode",
            [(2.0, 1.0, False)],
            seed=7001,
            patience=4,
        )


@pytest.mark.integration
def test_total_loss_smoke_matches_protected_projection() -> None:
    rows = [(5.0, 2.0, False), (4.0, 2.0, False), (3.0, 1.0, False)]
    protected = _run_optimizer(
        "protected", None, rows, seed=7003, patience=5
    )
    candidate = _run_optimizer(
        "candidate", "total_loss", rows, seed=7003, patience=5
    )

    assert protected["root"] == candidate["root"]


@pytest.mark.integration
def test_debt_smoke_changes_only_the_expected_restart_clock() -> None:
    rows = [(6.0, 1.5, False), (5.0, 1.5, False), (4.0, 1.5, False)]
    protected = _run_optimizer(
        "protected", None, rows, seed=7009, patience=2
    )
    treatment = _run_optimizer(
        "candidate", "feasibility_debt", rows, seed=7009, patience=2
    )

    assert _restart_batches(protected) == []
    assert _restart_batches(treatment) == [2]
    assert treatment["projection"]["inputs"] == protected["projection"]["inputs"]
    assert (
        treatment["projection"]["random_draws"][0]
        == protected["projection"]["random_draws"][0]
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "attack",
    [
        "missing:penalty",
        "integer-feasibility",
        "leading:violations",
        "negative-penalty",
        "scalar-penalty",
    ],
)
def test_treatment_auxiliary_smoke_fails_closed(attack: str) -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        _run_optimizer(
            "candidate",
            "feasibility_debt",
            [(2.0, 1.0, False)],
            seed=7013,
            patience=4,
            attack=attack,
        )


def test_child_environment_scrubs_secret_and_forces_cpu(monkeypatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "do-not-forward")
    monkeypatch.setenv("SAFE_MARKER", "retained")

    environment = _scrubbed_environment()

    assert "EXAMPLE_API_KEY" not in environment
    assert environment["SAFE_MARKER"] == "retained"
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
