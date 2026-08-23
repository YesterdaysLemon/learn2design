"""Fail-closed comparison gates for the submission-like evidence workflow."""

from __future__ import annotations

import math

from experiments.uifo_paired.results_ingestion import StudyValidationError


TOLERANCE = 1e-12


def _close(left: object, right: object, label: str) -> None:
    if left is None or right is None:
        if left is not right:
            raise StudyValidationError(f"submission-like replay mismatch: {label}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise StudyValidationError(f"submission-like replay mismatch: {label}")
        if set(left) != set(right):
            raise StudyValidationError(f"submission-like replay mismatch: {label}.keys")
        for key in sorted(left):
            _close(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, list) or isinstance(right, list):
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != len(right)
        ):
            raise StudyValidationError(f"submission-like replay mismatch: {label}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _close(left_item, right_item, f"{label}[{index}]")
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(
            float(left), float(right), rel_tol=TOLERANCE, abs_tol=TOLERANCE
        ):
            raise StudyValidationError(f"submission-like replay mismatch: {label}")
        return
    if type(left) is not type(right):
        raise StudyValidationError(f"submission-like replay mismatch: {label}.type")
    if left != right:
        raise StudyValidationError(f"submission-like replay mismatch: {label}")


def compare_submission_like_replays(
    production: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    """Require production and history-first results to agree before summary access."""
    production_runs = {
        str(seed_row["run_id"]): seed_row
        for topology in production.get("topology_rows", [])
        for seed_row in topology.get("seed_rows", [])
    }
    reference_runs = {
        str(row["run_id"]): row for row in reference.get("run_rows", [])
    }
    if set(production_runs) != set(reference_runs) or len(production_runs) != 20:
        raise StudyValidationError("submission-like replay run hierarchy mismatch")
    for run_id in sorted(production_runs):
        left = production_runs[run_id]
        right = reference_runs[run_id]
        _close(
            left.get("physical_feasible"),
            right.get("physical_feasible"),
            f"{run_id}.physical",
        )
        _close(left.get("finite_feasible"), right.get("finite_feasible"), f"{run_id}.finite")
        _close(
            left.get("best_feasible_loss"),
            right.get("best_feasible_loss"),
            f"{run_id}.best_loss",
        )

    production_topologies = {
        str(row["topology_sha256"]): row
        for row in production.get("topology_rows", [])
    }
    reference_topologies = {
        str(row["topology_sha256"]): row
        for row in reference.get("topology_rows", [])
    }
    if (
        set(production_topologies) != set(reference_topologies)
        or len(production_topologies) != 10
    ):
        raise StudyValidationError("submission-like replay topology hierarchy mismatch")
    for topology_hash in sorted(production_topologies):
        left = production_topologies[topology_hash]
        right = reference_topologies[topology_hash]
        _close(
            left.get("topology_mean_best_feasible_loss"),
            right.get("topology_mean_best_feasible_loss"),
            f"{topology_hash}.mean_loss",
        )
        _close(
            left.get("absolute_seed_gap"),
            right.get("absolute_seed_gap"),
            f"{topology_hash}.seed_gap",
        )

    aggregate_fields = {
        "completed_runs": "completed_runs",
        "physical_feasible_runs": "physical_feasible_runs",
        "finite_feasible_runs": "finite_feasible_runs",
        "complete_topologies": "complete_topology_blocks",
        "topology_arithmetic_mean_best_feasible_loss": (
            "topology_macro_mean_best_feasible_loss"
        ),
        "topology_median_best_feasible_loss": (
            "topology_macro_median_best_feasible_loss"
        ),
        "topology_p90_best_feasible_loss": (
            "topology_macro_p90_best_feasible_loss"
        ),
        "topology_p90_absolute_seed_gap": "topology_p90_absolute_seed_gap",
    }
    for production_field, reference_field in aggregate_fields.items():
        _close(
            production.get(production_field),
            reference.get(reference_field),
            production_field,
        )
    production_ci = production.get("topology_bootstrap_mean_loss_ci_95")
    reference_ci = reference.get(
        "topology_bootstrap_mean_best_feasible_loss_ci_95"
    )
    if production_ci is None or reference_ci is None:
        if production_ci is not None or reference_ci is not None:
            raise StudyValidationError("submission-like replay bootstrap mismatch")
    else:
        if (
            not isinstance(production_ci, list)
            or len(production_ci) != 2
            or not isinstance(reference_ci, dict)
            or set(reference_ci)
            != {"lower", "upper", "seed", "resamples", "method"}
            or reference_ci.get("method")
            != "percentile bootstrap over complete topology blocks"
        ):
            raise StudyValidationError("submission-like replay bootstrap is malformed")
        _close(production_ci[0], reference_ci["lower"], "bootstrap.lower")
        _close(production_ci[1], reference_ci["upper"], "bootstrap.upper")
    _close(
        production.get("bootstrap_seed"),
        reference.get("bootstrap_seed"),
        "bootstrap.seed",
    )
    _close(
        production.get("bootstrap_resamples"),
        reference.get("bootstrap_resamples"),
        "bootstrap.resamples",
    )
    _close(
        production.get("target_hitting"),
        reference.get("target_hitting"),
        "target_hitting",
    )

    production_decision = production.get("predeclared_decision")
    reference_decision = reference.get("predeclared_decision")
    if not isinstance(production_decision, dict) or not isinstance(
        reference_decision, dict
    ):
        raise StudyValidationError("submission-like replay decision is missing")
    production_criteria = production_decision.get("criteria")
    reference_criteria = reference_decision.get("criteria")
    if not isinstance(production_criteria, dict) or not isinstance(
        reference_criteria, dict
    ):
        raise StudyValidationError("submission-like replay criteria are missing")
    if set(production_criteria) != set(reference_criteria) or len(production_criteria) != 5:
        raise StudyValidationError("submission-like replay criterion set mismatch")
    for key in sorted(production_criteria):
        reference_value = reference_criteria[key]
        if not isinstance(reference_value, dict) or "passed" not in reference_value:
            raise StudyValidationError("reference criterion schema mismatch")
        _close(production_criteria[key], reference_value["passed"], f"criterion.{key}")
    for key in ("status", "passed", "action"):
        _close(production_decision.get(key), reference_decision.get(key), f"decision.{key}")
    return {
        "status": "matched",
        "runs_compared": 20,
        "topology_values_compared": 10,
        "target_thresholds_compared": 4,
        "frozen_criteria_compared": 5,
        "absolute_tolerance": TOLERANCE,
        "relative_tolerance": TOLERANCE,
    }


def compare_submission_like_archived_summary(
    production: dict[str, object],
    archived: dict[str, object],
) -> dict[str, object]:
    """Require the runner-written summary to match the fresh production replay."""
    if set(archived) != set(production):
        raise StudyValidationError("archived submission-like summary schema mismatch")
    for field in sorted(production):
        _close(production[field], archived[field], f"archived.{field}")
    return {
        "status": "matched",
        "archived_summary_compared": True,
        "fields_compared": len(production),
        "absolute_tolerance": TOLERANCE,
        "relative_tolerance": TOLERANCE,
    }
