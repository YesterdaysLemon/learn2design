"""Independent history-first replay for the H100 coverage screen.

This module intentionally does not import production analysis, metric, runner,
or decision helpers.  The validated pickle-free histories are its only outcome
input; record metrics are checked as claimed outputs, never trusted as inputs.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Protocol


class CoverageReferenceError(RuntimeError):
    """Independent replay failure, defined without production imports."""


# Keep the replay implementation readable while ensuring its error type and
# study interface are defined locally rather than imported through production.
StudyValidationError = CoverageReferenceError


class ValidatedStudy(Protocol):
    """Narrow structural interface consumed by the independent replay."""

    integrity: dict[str, object]
    plan: dict[str, object]
    configs: dict[str, dict[str, object]]
    records: list[dict[str, object]]
    history_rows: dict[str, list[dict[str, object]]]


PROFILE = "coverage-robustness-screen-v1"
CONTROL_ARM = "no_prior"
TREATMENT_ARM = "coverage_balanced"
RUNS = 48
TOPOLOGIES = 12
SEEDS = {37, 41}
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
TIE_TOLERANCE = 1e-12
FROZEN_POLICY = {
    "study_profile": "coverage-robustness-screen-v1",
    "stage": "optimizer_development_screen",
    "policy_id": "coverage-robustness-development-screen-v1",
    "action_if_passed": "freeze_official_budget_coverage_confirmation",
    "action_if_failed": "retain_random_start_candidate",
    "action_if_not_evaluable": "retain_candidate_attempt_not_evaluable",
    "maximum_topology_median_difference": -0.05,
    "maximum_topology_p90_regret": 0.5,
    "minimum_coverage_topology_wins": 9,
    "minimum_overall_median_evaluation_ratio": 0.95,
    "minimum_topology_evaluation_ratio": 0.90,
    "require_all_pairs_finite_comparable": True,
    "require_both_arm_order_mean_differences_below_zero": True,
    "require_both_seed_mean_differences_below_zero": True,
    "require_complete_uncensored_panel": True,
    "require_topology_mean_difference_below_zero": True,
    "inference_unit": "topology",
    "optimizer_seeds_are_repeated_measurements": True,
    "changes_packaged_candidate_default": False,
    "official_budget_claim_allowed": False,
}


def _topology_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise StudyValidationError("reference percentile has no observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_ci(values: list[float]) -> list[float]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = [
        mean(generator.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _same_number(claimed: object, observed: float | None) -> bool:
    if observed is None:
        return claimed is None
    return (
        not isinstance(claimed, bool)
        and isinstance(claimed, (int, float))
        and math.isfinite(float(claimed))
        and math.isclose(float(claimed), observed, rel_tol=1e-12, abs_tol=1e-12)
    )


def _history_outcome(
    run_id: str,
    rows: list[dict[str, object]],
    record: dict[str, object],
) -> dict[str, object]:
    if not rows:
        raise StudyValidationError(f"reference history is empty: {run_id}")
    physical = False
    finite_losses: list[float] = []
    prior_time = -math.inf
    prior_evals = 0
    for row in rows:
        feasible = row.get("is_feasible")
        if type(feasible) is not bool:
            raise StudyValidationError(
                f"reference feasibility is not strict boolean: {run_id}"
            )
        time_seconds = row.get("time_seconds")
        eval_count = row.get("eval_count_after_call")
        if (
            isinstance(time_seconds, bool)
            or not isinstance(time_seconds, (int, float))
            or not math.isfinite(float(time_seconds))
            or float(time_seconds) < prior_time
        ):
            raise StudyValidationError(f"reference history time drift: {run_id}")
        if (
            isinstance(eval_count, bool)
            or not isinstance(eval_count, int)
            or eval_count < 1
            or eval_count < prior_evals
        ):
            raise StudyValidationError(f"reference evaluation count drift: {run_id}")
        prior_time = float(time_seconds)
        prior_evals = eval_count
        if not feasible:
            continue
        physical = True
        loss = row.get("loss")
        if loss is None:
            continue
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise StudyValidationError(f"reference loss is invalid: {run_id}")
        if math.isfinite(float(loss)):
            finite_losses.append(float(loss))

    finite = bool(finite_losses)
    best = min(finite_losses) if finite else None
    metrics = record.get("metrics")
    accounting = record.get("objective_accounting")
    if not isinstance(metrics, dict) or not isinstance(accounting, dict):
        raise StudyValidationError(f"reference record claims are missing: {run_id}")
    if metrics.get("has_feasible") is not physical:
        raise StudyValidationError(f"reference physical feasibility drift: {run_id}")
    if metrics.get("has_finite_feasible") is not finite:
        raise StudyValidationError(f"reference finite feasibility drift: {run_id}")
    if not _same_number(metrics.get("best_feasible_loss"), best):
        raise StudyValidationError(f"reference best-loss drift: {run_id}")
    if accounting.get("eval_count") != prior_evals:
        raise StudyValidationError(f"reference Objective accounting drift: {run_id}")
    return {
        "physically_feasible": physical,
        "finite_feasible": finite,
        "best_feasible_loss": best,
        "eval_count": prior_evals,
    }


def _pair_integrity(
    pair_id: str,
    control: dict[str, object],
    treatment: dict[str, object],
) -> None:
    control_config = control.get("config")
    treatment_config = treatment.get("config")
    if not isinstance(control_config, dict) or not isinstance(treatment_config, dict):
        raise StudyValidationError(f"reference pair lacks config: {pair_id}")
    for field in (
        "pair_id",
        "optimizer_seed",
        "topology",
        "max_evals",
        "max_time_seconds",
        "population_size",
        "n_frequencies",
        "target_losses",
        "allow_cpu",
        "evaluation_chunk_size",
        "require_a100",
        "require_h100",
        "required_gpu_name",
        "preclock_warmup",
        "jax_compilation_cache_policy",
        "study_profile",
        "decision_policy",
        "candidate_package_evidence",
        "provider_stop_utc",
        "provider_deadline_maximum_horizon_seconds",
        "provider_evacuation_reserve_seconds",
    ):
        if control_config.get(field) != treatment_config.get(field):
            raise StudyValidationError(f"reference pair disagrees on {field}: {pair_id}")
    if (
        control_config.get("arm") != CONTROL_ARM
        or treatment_config.get("arm") != TREATMENT_ARM
        or control_config.get("initial_population_mode") != "random"
        or treatment_config.get("initial_population_mode") != "coverage_balanced"
    ):
        raise StudyValidationError(f"reference pair arm/mode mismatch: {pair_id}")
    if control.get("problem") != treatment.get("problem"):
        raise StudyValidationError(f"reference pair problem mismatch: {pair_id}")
    if control.get("raw_suffix_parameter_hashes") != treatment.get(
        "raw_suffix_parameter_hashes"
    ):
        raise StudyValidationError(f"reference pair raw draw mismatch: {pair_id}")
    control_hashes = control.get("initial_parameter_hashes")
    treatment_hashes = treatment.get("initial_parameter_hashes")
    if (
        not isinstance(control_hashes, list)
        or not isinstance(treatment_hashes, list)
        or not control_hashes
        or not treatment_hashes
        or control_hashes[0] != treatment_hashes[0]
    ):
        raise StudyValidationError(f"reference pair anchor mismatch: {pair_id}")


def reference_coverage_screen(study: ValidatedStudy) -> dict[str, object]:
    """Recompute the frozen topology-level comparison from sealed histories."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("reference replay requires the summary sealed")
    configuration = study.plan.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("study_profile") != PROFILE:
        raise StudyValidationError("reference coverage configuration mismatch")
    if configuration.get("decision_policy") != FROZEN_POLICY:
        raise StudyValidationError("reference frozen decision policy mismatch")
    if len(study.configs) != RUNS or len(study.records) != RUNS:
        raise StudyValidationError("reference replay requires exactly 48 runs")
    if set(study.history_rows) != set(study.configs):
        raise StudyValidationError("reference history/config hierarchy mismatch")

    records_by_id: dict[str, dict[str, object]] = {}
    outcomes: dict[str, dict[str, object]] = {}
    by_pair: dict[str, dict[str, str]] = defaultdict(dict)
    for record in study.records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or run_id in records_by_id:
            raise StudyValidationError("reference run ID is invalid or duplicated")
        config = study.configs.get(run_id)
        if not isinstance(config, dict) or record.get("config") != config:
            raise StudyValidationError(f"reference record/config mismatch: {run_id}")
        arm = str(config.get("arm"))
        pair_id = str(config.get("pair_id"))
        if arm not in {CONTROL_ARM, TREATMENT_ARM} or arm in by_pair[pair_id]:
            raise StudyValidationError(f"reference pair hierarchy mismatch: {run_id}")
        records_by_id[run_id] = record
        outcomes[run_id] = _history_outcome(
            run_id, study.history_rows[run_id], record
        )
        by_pair[pair_id][arm] = run_id

    if len(by_pair) != RUNS // 2 or any(
        set(arms) != {CONTROL_ARM, TREATMENT_ARM} for arms in by_pair.values()
    ):
        raise StudyValidationError("reference replay found incomplete arm pairs")

    pair_rows: list[dict[str, object]] = []
    for pair_id in sorted(by_pair):
        control_id = by_pair[pair_id][CONTROL_ARM]
        treatment_id = by_pair[pair_id][TREATMENT_ARM]
        control = records_by_id[control_id]
        treatment = records_by_id[treatment_id]
        _pair_integrity(pair_id, control, treatment)
        control_outcome = outcomes[control_id]
        treatment_outcome = outcomes[treatment_id]
        difference = None
        if control_outcome["finite_feasible"] and treatment_outcome["finite_feasible"]:
            difference = float(treatment_outcome["best_feasible_loss"]) - float(
                control_outcome["best_feasible_loss"]
            )
        control_evals = int(control_outcome["eval_count"])
        treatment_evals = int(treatment_outcome["eval_count"])
        control_config = control["config"]
        treatment_config = treatment["config"]
        problem = control["problem"]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(control_config["optimizer_seed"]),
                "configured_topology": control_config["topology"],
                "topology_sha256": str(problem["topology_sha256"]),
                "control_physically_feasible": bool(
                    control_outcome["physically_feasible"]
                ),
                "treatment_physically_feasible": bool(
                    treatment_outcome["physically_feasible"]
                ),
                "control_finite_feasible": bool(control_outcome["finite_feasible"]),
                "treatment_finite_feasible": bool(
                    treatment_outcome["finite_feasible"]
                ),
                "difference_coverage_minus_random": difference,
                "coverage_first": int(treatment_config["run_order_within_pair"])
                < int(control_config["run_order_within_pair"]),
                "control_eval_count": control_evals,
                "treatment_eval_count": treatment_evals,
                "evaluation_ratio_coverage_over_random": (
                    treatment_evals / control_evals if control_evals > 0 else None
                ),
            }
        )

    expected_topologies: dict[str, set[int]] = defaultdict(set)
    for config in study.configs.values():
        if config["arm"] == CONTROL_ARM:
            expected_topologies[_topology_key(config["topology"])].add(
                int(config["optimizer_seed"])
            )
    if len(expected_topologies) != TOPOLOGIES or any(
        seeds != SEEDS for seeds in expected_topologies.values()
    ):
        raise StudyValidationError("reference topology/seed hierarchy mismatch")
    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_topology[_topology_key(row["configured_topology"])].append(row)

    topology_rows: list[dict[str, object]] = []
    for topology_key, expected_seeds in sorted(expected_topologies.items()):
        rows = sorted(
            by_topology.get(topology_key, []),
            key=lambda row: int(row["optimizer_seed"]),
        )
        observed_seeds = {int(row["optimizer_seed"]) for row in rows}
        differences = [
            float(row["difference_coverage_minus_random"])
            for row in rows
            if row["difference_coverage_minus_random"] is not None
        ]
        replication_complete = observed_seeds == expected_seeds
        inference_complete = replication_complete and len(differences) == len(
            expected_seeds
        )
        control_evals = [int(row["control_eval_count"]) for row in rows]
        treatment_evals = [int(row["treatment_eval_count"]) for row in rows]
        aggregate_ratio = (
            sum(treatment_evals) / sum(control_evals)
            if replication_complete and sum(control_evals) > 0
            else None
        )
        topology_rows.append(
            {
                "configured_topology": rows[0]["configured_topology"] if rows else None,
                "topology_sha256": rows[0]["topology_sha256"] if rows else None,
                "optimizer_seeds": sorted(observed_seeds),
                "replication_complete": replication_complete,
                "inference_complete": inference_complete,
                "control_finite_feasible_seeds": sum(
                    bool(row["control_finite_feasible"]) for row in rows
                ),
                "treatment_finite_feasible_seeds": sum(
                    bool(row["treatment_finite_feasible"]) for row in rows
                ),
                "mean_seed_difference_coverage_minus_random": (
                    mean(differences) if inference_complete else None
                ),
                "aggregate_evaluation_ratio_coverage_over_random": aggregate_ratio,
                "seed_pair_rows": rows,
            }
        )

    macro_values = [
        float(row["mean_seed_difference_coverage_minus_random"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    inference_ready = len(macro_values) == TOPOLOGIES
    macro_mean = mean(macro_values) if inference_ready else None
    macro_median = median(macro_values) if inference_ready else None
    p90_regret = _percentile(macro_values, 0.9) if inference_ready else None
    wins = sum(value < -TIE_TOLERANCE for value in macro_values)
    ties = sum(abs(value) <= TIE_TOLERANCE for value in macro_values)
    losses = sum(value > TIE_TOLERANCE for value in macro_values)

    seed_means = {}
    for seed in sorted(SEEDS):
        values = [
            float(row["difference_coverage_minus_random"])
            for row in pair_rows
            if int(row["optimizer_seed"]) == seed
            and row["difference_coverage_minus_random"] is not None
        ]
        seed_means[str(seed)] = mean(values) if len(values) == TOPOLOGIES else None
    order_means = {}
    for coverage_first in (False, True):
        values = [
            float(row["difference_coverage_minus_random"])
            for row in pair_rows
            if bool(row["coverage_first"]) is coverage_first
            and row["difference_coverage_minus_random"] is not None
        ]
        key = "coverage_first" if coverage_first else "random_first"
        order_means[key] = mean(values) if len(values) == TOPOLOGIES else None

    pair_ratios = [
        float(row["evaluation_ratio_coverage_over_random"])
        for row in pair_rows
        if row["evaluation_ratio_coverage_over_random"] is not None
    ]
    overall_ratio = median(pair_ratios) if len(pair_ratios) == RUNS // 2 else None
    topology_ratios = [
        float(row["aggregate_evaluation_ratio_coverage_over_random"])
        for row in topology_rows
        if row["aggregate_evaluation_ratio_coverage_over_random"] is not None
    ]
    minimum_topology_ratio = (
        min(topology_ratios) if len(topology_ratios) == TOPOLOGIES else None
    )
    all_feasible = all(
        row["control_physically_feasible"]
        and row["treatment_physically_feasible"]
        and row["control_finite_feasible"]
        and row["treatment_finite_feasible"]
        for row in pair_rows
    )
    criteria = {
        "panel_execution_complete": len(pair_rows) == RUNS // 2,
        "complete_records_revalidated": True,
        "inference_ready": inference_ready,
        "all_runs_physically_and_finite_feasible": all_feasible,
        "all_pairs_finite_comparable": len(pair_rows) == RUNS // 2
        and all(row["difference_coverage_minus_random"] is not None for row in pair_rows),
        "minimum_topology_wins_met": wins
        >= int(FROZEN_POLICY["minimum_coverage_topology_wins"]),
        "median_difference_at_most_negative_0_05": bool(
            macro_median is not None
            and macro_median
            <= float(FROZEN_POLICY["maximum_topology_median_difference"])
        ),
        "mean_difference_below_zero": bool(macro_mean is not None and macro_mean < 0),
        "both_seed_mean_differences_below_zero": set(seed_means) == {"37", "41"}
        and all(value is not None and value < 0 for value in seed_means.values()),
        "both_arm_order_mean_differences_below_zero": set(order_means)
        == {"coverage_first", "random_first"}
        and all(value is not None and value < 0 for value in order_means.values()),
        "p90_regret_at_most_0_5": bool(
            p90_regret is not None
            and p90_regret <= float(FROZEN_POLICY["maximum_topology_p90_regret"])
        ),
        "overall_median_evaluation_ratio_at_least_0_95": bool(
            overall_ratio is not None
            and overall_ratio
            >= float(FROZEN_POLICY["minimum_overall_median_evaluation_ratio"])
        ),
        "every_topology_evaluation_ratio_at_least_0_90": bool(
            minimum_topology_ratio is not None
            and minimum_topology_ratio
            >= float(FROZEN_POLICY["minimum_topology_evaluation_ratio"])
        ),
    }
    passed = all(criteria.values())
    status = "passed" if passed else "failed"
    action = str(
        FROZEN_POLICY["action_if_passed"]
        if passed
        else FROZEN_POLICY["action_if_failed"]
    )
    return {
        "format_version": 1,
        "study_profile": PROFILE,
        "completed_runs": RUNS,
        "error_runs": 0,
        "interrupted_runs": 0,
        "complete_optimizer_seed_pairs": len(pair_rows),
        "finite_comparable_optimizer_seed_pairs": sum(
            row["difference_coverage_minus_random"] is not None for row in pair_rows
        ),
        "complete_topologies": sum(
            bool(row["replication_complete"]) for row in topology_rows
        ),
        "finite_comparable_topologies": len(macro_values),
        "wins_ties_losses": {
            "coverage_balanced_wins": wins,
            "ties": ties,
            "coverage_balanced_losses": losses,
        },
        "topology_macro_mean_difference": macro_mean,
        "topology_macro_median_difference": macro_median,
        "topology_p90_regret": p90_regret,
        "optimizer_seed_mean_differences": seed_means,
        "arm_order_mean_differences": order_means,
        "overall_median_evaluation_ratio_coverage_over_random": overall_ratio,
        "minimum_topology_evaluation_ratio_coverage_over_random": (
            minimum_topology_ratio
        ),
        "topology_bootstrap_mean_difference_ci_95": (
            _bootstrap_mean_ci(macro_values) if inference_ready else None
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "predeclared_decision": {
            "status": status,
            "passed": passed,
            "action": action,
            "criteria": criteria,
        },
        "topology_differences": topology_rows,
        "optimizer_seed_pair_rows": pair_rows,
        "run_ids": sorted(str(record.get("run_id")) for record in study.records),
        "note": (
            "Topology is the inference unit; seeds are repeated paired "
            "measurements. Negative differences favor midpoint Latin-hypercube "
            "coverage balancing. Bootstrap output cannot override the frozen "
            "decision criteria."
        ),
    }
