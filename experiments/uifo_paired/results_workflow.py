"""Fail-closed comparison of production, reference, and archived summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiments.uifo_paired.results_ingestion import StudyValidationError


@dataclass(frozen=True)
class NumericalTolerance:
    relative: float = 1e-12
    absolute: float = 1e-12


def _assert_close(
    left: object,
    right: object,
    path: str,
    tolerance: NumericalTolerance,
) -> None:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(
            float(left),
            float(right),
            rel_tol=tolerance.relative,
            abs_tol=tolerance.absolute,
        ):
            raise StudyValidationError(
                f"three-way numerical mismatch at {path}: {left!r} != {right!r}"
            )
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise StudyValidationError(f"three-way mapping-key mismatch at {path}")
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}", tolerance)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise StudyValidationError(f"three-way list-length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_close(left_item, right_item, f"{path}[{index}]", tolerance)
        return
    if left != right:
        raise StudyValidationError(
            f"three-way value mismatch at {path}: {left!r} != {right!r}"
        )


def compare_production_and_reference(
    production: dict[str, object],
    reference: dict[str, object],
    tolerance: NumericalTolerance = NumericalTolerance(),
) -> dict[str, object]:
    paired = production["semantic_prior_vs_no_prior"]
    scalar_pairs = {
        "completed_runs": (production["completed_runs"], reference["completed_runs"]),
        "complete_optimizer_seed_pairs": (
            paired["complete_optimizer_seed_pairs"],
            reference["complete_optimizer_seed_pairs"],
        ),
        "complete_topologies": (
            paired["complete_topologies"],
            reference["complete_topologies"],
        ),
        "wins_ties_losses": (
            paired["wins_ties_losses"],
            reference["wins_ties_losses"],
        ),
        "topology_mean_difference": (
            paired["topology_macro_mean_difference"],
            reference["topology_mean_difference"],
        ),
        "topology_median_difference": (
            paired["topology_macro_median_difference"],
            reference["topology_median_difference"],
        ),
        "topology_p90_regret": (
            paired["topology_p90_regret"],
            reference["topology_p90_regret"],
        ),
        "decision_status": (
            paired["predeclared_decision"]["status"],
            reference["predeclared_decision"]["status"],
        ),
        "decision_action": (
            paired["predeclared_decision"]["action"],
            reference["predeclared_decision"]["action"],
        ),
        "decision_route": (
            paired["predeclared_decision"]["selected_route"],
            reference["predeclared_decision"]["selected_route"],
        ),
    }
    for name, (left, right) in scalar_pairs.items():
        _assert_close(left, right, f"production_reference.{name}", tolerance)

    production_ci = paired["topology_bootstrap_mean_difference_ci_95"]
    reference_ci = reference["topology_bootstrap_mean_difference_ci_95"]
    for key in ("confidence_level", "lower", "upper", "resamples", "seed", "inference_unit"):
        _assert_close(
            production_ci[key],
            reference_ci[key],
            f"production_reference.topology_bootstrap_mean_difference_ci_95.{key}",
            tolerance,
        )

    production_topologies = {
        str(row["topology_sha256"]): row["mean_seed_difference_treatment_minus_control"]
        for row in paired["topology_differences"]
    }
    reference_topologies = {
        str(row["topology_sha256"]): row[
            "mean_seed_difference_semantic_minus_no_prior"
        ]
        for row in reference["topology_rows"]
    }
    _assert_close(
        production_topologies,
        reference_topologies,
        "production_reference.topology_values",
        tolerance,
    )

    production_targets = paired["target_hitting_time_inference"]["targets"]
    reference_targets = reference["target_hitting"]
    if production_targets.keys() != reference_targets.keys():
        raise StudyValidationError("production/reference target set mismatch")
    for target in production_targets:
        production_target = production_targets[target]
        reference_target = reference_targets[target]
        _assert_close(
            production_target["seed_pair_outcomes"],
            reference_target["seed_pair_outcomes"],
            f"production_reference.targets.{target}.outcomes",
            tolerance,
        )
        _assert_close(
            production_target["topology_inference_ready"],
            reference_target["topology_inference_ready"],
            f"production_reference.targets.{target}.inference_ready",
            tolerance,
        )
        _assert_close(
            production_target["order_of_magnitude_claim_ready"],
            reference_target["order_of_magnitude_claim_ready"],
            f"production_reference.targets.{target}.order_of_magnitude",
            tolerance,
        )
    return {
        "status": "matched",
        "topology_values_compared": len(production_topologies),
        "target_thresholds_compared": len(production_targets),
        "relative_tolerance": tolerance.relative,
        "absolute_tolerance": tolerance.absolute,
    }


def compare_archived_summary(
    production: dict[str, object],
    archived: dict[str, object],
    tolerance: NumericalTolerance = NumericalTolerance(),
) -> dict[str, object]:
    paired_production = production["semantic_prior_vs_no_prior"]
    paired_archived = archived["semantic_prior_vs_no_prior"]
    for key in (
        "completed_runs",
        "error_runs",
        "interrupted_runs",
        "arm_summary",
        "run_ids",
    ):
        _assert_close(
            production[key], archived[key], f"production_archived.{key}", tolerance
        )
    for key in (
        "complete_optimizer_seed_pairs",
        "finite_comparable_optimizer_seed_pairs",
        "observed_or_planned_topologies",
        "complete_topologies",
        "incomplete_topologies",
        "censored_topologies",
        "panel_execution_complete",
        "promotion_inference_ready",
        "finite_comparable_topologies",
        "wins_ties_losses",
        "physical_feasibility_discordance",
        "finite_feasibility_discordance",
        "topology_macro_mean_difference",
        "topology_macro_median_difference",
        "topology_p90_regret",
        "observed_topology_p90_regret_guard",
        "topology_bootstrap_mean_difference_ci_95",
        "predeclared_decision",
        "target_hitting_time_inference",
        "topology_differences",
        "optimizer_seed_pair_diagnostics",
    ):
        _assert_close(
            paired_production[key],
            paired_archived[key],
            f"production_archived.semantic_prior_vs_no_prior.{key}",
            tolerance,
        )
    return {
        "status": "matched",
        "relative_tolerance": tolerance.relative,
        "absolute_tolerance": tolerance.absolute,
        "note": (
            "Tolerance covers last-bit Python 3.12/3.13 differences in log10 target "
            "summaries; frozen loss results match at displayed precision."
        ),
    }
