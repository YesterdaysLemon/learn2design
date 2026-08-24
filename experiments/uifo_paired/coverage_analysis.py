"""Topology-level decision summary for the coverage-robustness screen."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, median

from experiments.uifo_paired.coverage_profiles import (
    coverage_profile_names,
    coverage_profile_spec,
)

CONTROL_ARM = "no_prior"
TREATMENT_ARM = "coverage_balanced"
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
TIE_TOLERANCE = 1e-12


def summarize_coverage_records(
    records: list[dict[str, object]],
    expected_configs: dict[str, dict[str, object]],
    *,
    compute_bootstrap: bool = True,
) -> dict[str, object]:
    """Apply the frozen coverage-screen estimand and promotion criteria."""
    profiles = {
        str(config.get("study_profile")) for config in expected_configs.values()
    }
    if len(profiles) != 1 or not profiles <= coverage_profile_names():
        raise ValueError("coverage configurations disagree on the frozen profile")
    profile = next(iter(profiles))
    specification = coverage_profile_spec(profile)
    expected_seed_labels = {str(seed) for seed in specification.seeds}
    policy = _decision_policy(expected_configs)
    complete = [record for record in records if record.get("status") == "complete"]
    errors = [record for record in records if record.get("status") == "error"]
    interrupted = [
        record for record in records if record.get("status") == "interrupted"
    ]

    by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in complete:
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError("complete coverage record is missing its config")
        pair_id = str(config.get("pair_id"))
        arm = str(config.get("arm"))
        if arm in by_pair[pair_id]:
            raise ValueError(f"duplicate coverage arm {arm!r} for pair {pair_id!r}")
        by_pair[pair_id][arm] = record

    expected_pairs = {
        str(config["pair_id"]) for config in expected_configs.values()
    }
    pair_rows = []
    for pair_id in sorted(expected_pairs):
        arms = by_pair.get(pair_id, {})
        if set(arms) != {CONTROL_ARM, TREATMENT_ARM}:
            continue
        control = arms[CONTROL_ARM]
        treatment = arms[TREATMENT_ARM]
        _assert_pair_integrity(pair_id, control, treatment)
        control_metrics = control["metrics"]
        treatment_metrics = treatment["metrics"]
        if not isinstance(control_metrics, dict) or not isinstance(
            treatment_metrics, dict
        ):
            raise ValueError(f"coverage pair {pair_id!r} is missing metrics")
        control_physical = bool(control_metrics.get("has_feasible"))
        treatment_physical = bool(treatment_metrics.get("has_feasible"))
        control_finite = bool(control_metrics.get("has_finite_feasible"))
        treatment_finite = bool(treatment_metrics.get("has_finite_feasible"))
        control_loss = control_metrics.get("best_feasible_loss")
        treatment_loss = treatment_metrics.get("best_feasible_loss")
        difference = None
        if control_finite and treatment_finite:
            if not isinstance(control_loss, (int, float)) or not isinstance(
                treatment_loss, (int, float)
            ):
                raise ValueError(
                    f"coverage pair {pair_id!r} has invalid feasible losses"
                )
            difference = float(treatment_loss) - float(control_loss)
        control_evals = _eval_count(control)
        treatment_evals = _eval_count(treatment)
        evaluation_ratio = (
            treatment_evals / control_evals
            if control_evals is not None
            and treatment_evals is not None
            and control_evals > 0
            else None
        )
        control_config = control["config"]
        problem = control["problem"]
        assert isinstance(control_config, dict)
        assert isinstance(problem, dict)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(control_config["optimizer_seed"]),
                "configured_topology": control_config["topology"],
                "topology_sha256": str(problem["topology_sha256"]),
                "control_physically_feasible": control_physical,
                "treatment_physically_feasible": treatment_physical,
                "control_finite_feasible": control_finite,
                "treatment_finite_feasible": treatment_finite,
                "difference_coverage_minus_random": difference,
                "coverage_first": int(
                    treatment["config"]["run_order_within_pair"]
                )
                < int(control["config"]["run_order_within_pair"]),
                "control_eval_count": control_evals,
                "treatment_eval_count": treatment_evals,
                "evaluation_ratio_coverage_over_random": evaluation_ratio,
            }
        )

    expected_topologies: dict[str, set[int]] = defaultdict(set)
    for config in expected_configs.values():
        if config["arm"] == CONTROL_ARM:
            expected_topologies[_topology_key(config["topology"])].add(
                int(config["optimizer_seed"])
            )
    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_topology[_topology_key(row["configured_topology"])].append(row)

    topology_rows = []
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
        control_evals = [
            int(row["control_eval_count"])
            for row in rows
            if row["control_eval_count"] is not None
        ]
        treatment_evals = [
            int(row["treatment_eval_count"])
            for row in rows
            if row["treatment_eval_count"] is not None
        ]
        aggregate_evaluation_ratio = (
            sum(treatment_evals) / sum(control_evals)
            if replication_complete
            and len(control_evals) == len(rows)
            and len(treatment_evals) == len(rows)
            and sum(control_evals) > 0
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
                "aggregate_evaluation_ratio_coverage_over_random": (
                    aggregate_evaluation_ratio
                ),
                "seed_pair_rows": rows,
            }
        )

    macro_values = [
        float(row["mean_seed_difference_coverage_minus_random"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    wins = sum(value < -TIE_TOLERANCE for value in macro_values)
    ties = sum(abs(value) <= TIE_TOLERANCE for value in macro_values)
    losses = sum(value > TIE_TOLERANCE for value in macro_values)
    seed_means = {}
    for seed in sorted({int(row["optimizer_seed"]) for row in pair_rows}):
        values = [
            float(row["difference_coverage_minus_random"])
            for row in pair_rows
            if int(row["optimizer_seed"]) == seed
            and row["difference_coverage_minus_random"] is not None
        ]
        seed_means[str(seed)] = (
            mean(values) if len(values) == len(expected_topologies) else None
        )
    order_means = {}
    for coverage_first in (False, True):
        values = [
            float(row["difference_coverage_minus_random"])
            for row in pair_rows
            if bool(row["coverage_first"]) is coverage_first
            and row["difference_coverage_minus_random"] is not None
        ]
        key = "coverage_first" if coverage_first else "random_first"
        order_means[key] = (
            mean(values) if len(values) == len(expected_topologies) else None
        )

    panel_execution_complete = (
        len(records) == len(expected_configs)
        and len(complete) == len(expected_configs)
        and not errors
        and not interrupted
        and len(pair_rows) == len(expected_pairs)
        and all(row["replication_complete"] for row in topology_rows)
    )
    inference_ready = panel_execution_complete and len(macro_values) == len(
        expected_topologies
    )
    macro_mean = mean(macro_values) if inference_ready else None
    macro_median = median(macro_values) if inference_ready else None
    p90_regret = (
        _percentile(sorted(macro_values), 0.9) if inference_ready else None
    )
    pair_evaluation_ratios = [
        float(row["evaluation_ratio_coverage_over_random"])
        for row in pair_rows
        if row["evaluation_ratio_coverage_over_random"] is not None
    ]
    overall_median_evaluation_ratio = (
        median(pair_evaluation_ratios)
        if len(pair_evaluation_ratios) == len(expected_pairs)
        else None
    )
    topology_evaluation_ratios = [
        float(row["aggregate_evaluation_ratio_coverage_over_random"])
        for row in topology_rows
        if row["aggregate_evaluation_ratio_coverage_over_random"] is not None
    ]
    minimum_topology_evaluation_ratio = (
        min(topology_evaluation_ratios)
        if len(topology_evaluation_ratios) == len(expected_topologies)
        else None
    )
    all_runs_physically_and_finite_feasible = len(pair_rows) == len(
        expected_pairs
    ) and all(
        row["control_physically_feasible"]
        and row["treatment_physically_feasible"]
        and row["control_finite_feasible"]
        and row["treatment_finite_feasible"]
        for row in pair_rows
    )
    criteria = {
        "panel_execution_complete": panel_execution_complete,
        "complete_records_revalidated": bool(compute_bootstrap),
        "inference_ready": inference_ready,
        "all_runs_physically_and_finite_feasible": (
            all_runs_physically_and_finite_feasible
        ),
        "all_pairs_finite_comparable": len(pair_rows) == len(expected_pairs)
        and all(
            row["difference_coverage_minus_random"] is not None
            for row in pair_rows
        ),
        "minimum_topology_wins_met": wins
        >= int(policy["minimum_coverage_topology_wins"]),
        "median_difference_at_most_negative_0_05": bool(
            macro_median is not None
            and macro_median
            <= float(policy["maximum_topology_median_difference"])
        ),
        "mean_difference_below_zero": bool(
            macro_mean is not None and macro_mean < 0
        ),
        "both_seed_mean_differences_below_zero": bool(
            set(seed_means) == expected_seed_labels
            and all(value is not None and value < 0 for value in seed_means.values())
        ),
        "both_arm_order_mean_differences_below_zero": bool(
            set(order_means) == {"coverage_first", "random_first"}
            and all(value is not None and value < 0 for value in order_means.values())
        ),
        "p90_regret_at_most_0_5": bool(
            p90_regret is not None
            and p90_regret <= float(policy["maximum_topology_p90_regret"])
        ),
        "overall_median_evaluation_ratio_at_least_0_95": bool(
            overall_median_evaluation_ratio is not None
            and overall_median_evaluation_ratio
            >= float(policy["minimum_overall_median_evaluation_ratio"])
        ),
        "every_topology_evaluation_ratio_at_least_0_90": bool(
            minimum_topology_evaluation_ratio is not None
            and minimum_topology_evaluation_ratio
            >= float(policy["minimum_topology_evaluation_ratio"])
        ),
    }
    if "maximum_harmful_topology_difference" in policy:
        criteria["maximum_harmful_topology_difference_at_most_0_5"] = bool(
            inference_ready
            and max(macro_values)
            <= float(policy["maximum_harmful_topology_difference"])
        )

    status = "pending"
    passed = False
    action = None
    if errors or interrupted:
        status = "not_evaluable"
        action = str(policy["action_if_not_evaluable"])
    elif panel_execution_complete and compute_bootstrap:
        passed = all(criteria.values())
        status = "passed" if passed else "failed"
        action = str(
            policy["action_if_passed"] if passed else policy["action_if_failed"]
        )

    bootstrap_ci = (
        _bootstrap_mean_ci(macro_values)
        if inference_ready and compute_bootstrap
        else None
    )
    return {
        "format_version": 1,
        "study_profile": profile,
        "completed_runs": len(complete),
        "error_runs": len(errors),
        "interrupted_runs": len(interrupted),
        "complete_optimizer_seed_pairs": len(pair_rows),
        "finite_comparable_optimizer_seed_pairs": sum(
            row["difference_coverage_minus_random"] is not None
            for row in pair_rows
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
        "overall_median_evaluation_ratio_coverage_over_random": (
            overall_median_evaluation_ratio
        ),
        "minimum_topology_evaluation_ratio_coverage_over_random": (
            minimum_topology_evaluation_ratio
        ),
        "topology_bootstrap_mean_difference_ci_95": bootstrap_ci,
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
        "run_ids": sorted(str(record.get("run_id")) for record in records),
        "note": (
            "Topology is the inference unit; seeds are repeated paired "
            "measurements. Negative differences favor midpoint Latin-hypercube "
            "coverage balancing. Bootstrap output cannot override the frozen "
            "decision criteria."
        ),
    }


def _assert_pair_integrity(
    pair_id: str,
    control: dict[str, object],
    treatment: dict[str, object],
) -> None:
    control_config = control.get("config")
    treatment_config = treatment.get("config")
    if not isinstance(control_config, dict) or not isinstance(
        treatment_config, dict
    ):
        raise ValueError(f"coverage pair {pair_id!r} is missing config")
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
            raise ValueError(f"coverage pair {pair_id!r} disagrees on {field}")
    if control_config.get("arm") != CONTROL_ARM:
        raise ValueError(f"coverage pair {pair_id!r} has invalid control arm")
    if treatment_config.get("arm") != TREATMENT_ARM:
        raise ValueError(f"coverage pair {pair_id!r} has invalid treatment arm")
    if control_config.get("initial_population_mode") != "random":
        raise ValueError(f"coverage pair {pair_id!r} control mode is not random")
    if treatment_config.get("initial_population_mode") != "coverage_balanced":
        raise ValueError(
            f"coverage pair {pair_id!r} treatment mode is not coverage_balanced"
        )
    if control.get("problem") != treatment.get("problem"):
        raise ValueError(f"coverage pair {pair_id!r} problem mismatch")

    control_algorithm = control.get("algorithm")
    treatment_algorithm = treatment.get("algorithm")
    if not isinstance(control_algorithm, dict) or not isinstance(
        treatment_algorithm, dict
    ):
        raise ValueError(f"coverage pair {pair_id!r} is missing algorithm metadata")
    for field in ("module", "class", "algorithm_str"):
        if control_algorithm.get(field) != treatment_algorithm.get(field):
            raise ValueError(
                f"coverage pair {pair_id!r} algorithm disagrees on {field}"
            )
    control_kwargs = dict(control_algorithm.get("kwargs", {}))
    treatment_kwargs = dict(treatment_algorithm.get("kwargs", {}))
    if control_kwargs.pop("initial_population_mode", None) != "random":
        raise ValueError(f"coverage pair {pair_id!r} control kwargs are invalid")
    if (
        treatment_kwargs.pop("initial_population_mode", None)
        != "coverage_balanced"
    ):
        raise ValueError(f"coverage pair {pair_id!r} treatment kwargs are invalid")
    if control_kwargs != treatment_kwargs:
        raise ValueError(
            f"coverage pair {pair_id!r} differs beyond initialization mode"
        )
    if control_kwargs.get("preclock_warmup") is not True:
        raise ValueError(f"coverage pair {pair_id!r} did not use common warmup")
    if control_kwargs.get("use_semantic_prior") is not False:
        raise ValueError(f"coverage pair {pair_id!r} enabled the semantic prior")

    expected_control_roles = ["anchor"] + ["random"] * (
        int(control_config["population_size"]) - 1
    )
    expected_treatment_roles = ["anchor"] + ["coverage_balanced"] * (
        int(control_config["population_size"]) - 1
    )
    if control.get("initial_population_roles") != expected_control_roles:
        raise ValueError(f"coverage pair {pair_id!r} control roles are invalid")
    if treatment.get("initial_population_roles") != expected_treatment_roles:
        raise ValueError(f"coverage pair {pair_id!r} treatment roles are invalid")
    control_hashes = control.get("initial_parameter_hashes")
    treatment_hashes = treatment.get("initial_parameter_hashes")
    if not isinstance(control_hashes, list) or not isinstance(
        treatment_hashes, list
    ):
        raise ValueError(f"coverage pair {pair_id!r} is missing initial hashes")
    if len(control_hashes) != len(expected_control_roles) or len(
        treatment_hashes
    ) != len(expected_treatment_roles):
        raise ValueError(f"coverage pair {pair_id!r} initial hash count is invalid")
    if control_hashes[0] != treatment_hashes[0]:
        raise ValueError(f"coverage pair {pair_id!r} anchor hash differs")
    control_raw_hashes = control.get("raw_suffix_parameter_hashes")
    treatment_raw_hashes = treatment.get("raw_suffix_parameter_hashes")
    expected_raw_count = int(control_config["population_size"]) - 1
    if (
        not isinstance(control_raw_hashes, list)
        or not isinstance(treatment_raw_hashes, list)
        or len(control_raw_hashes) != expected_raw_count
        or len(treatment_raw_hashes) != expected_raw_count
    ):
        raise ValueError(
            f"coverage pair {pair_id!r} is missing pre-transform draw evidence"
        )
    if control_raw_hashes != treatment_raw_hashes:
        raise ValueError(
            f"coverage pair {pair_id!r} pre-transform random draw differs"
        )


def _decision_policy(
    expected_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    policies = [config.get("decision_policy") for config in expected_configs.values()]
    if not policies or not all(isinstance(policy, dict) for policy in policies):
        raise ValueError("coverage study is missing its frozen decision policy")
    first = policies[0]
    assert isinstance(first, dict)
    if any(policy != first for policy in policies[1:]):
        raise ValueError("coverage configurations disagree on decision policy")
    return first


def _eval_count(record: dict[str, object]) -> int | None:
    accounting = record.get("objective_accounting")
    if not isinstance(accounting, dict):
        return None
    value = accounting.get("eval_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _topology_key(topology: object) -> str:
    import json

    return json.dumps(topology, sort_keys=True, separators=(",", ":"))


def _percentile(sorted_values: list[float], probability: float) -> float:
    import math

    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(values: list[float]) -> list[float]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = [
        mean(generator.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    ordered = sorted(samples)
    return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]
