"""Transparent topology-level summaries for the bounded restart screen."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from datetime import datetime
from statistics import mean, median

from experiments.uifo_paired.optimizer_settings import validate_batched_settings


RESTART_CONTROL_ARM = "no_prior_p600"
RESTART_TREATMENT_ARM = "no_prior_p200"
RESTART_SCREEN_BOOTSTRAP_SEED = 20260821
RESTART_SCREEN_BOOTSTRAP_RESAMPLES = 10_000


def summarize_restart_records(
    records: list[dict[str, object]],
    expected_configs: dict[str, dict[str, object]],
    *,
    compute_bootstrap: bool = True,
    include_exploratory: bool = True,
) -> dict[str, object]:
    profiles = {
        str(config.get("study_profile")) for config in expected_configs.values()
    }
    if len(profiles) != 1:
        raise ValueError("restart study configurations disagree on profile")
    profile = next(iter(profiles))
    if profile == "restart-mechanics-v1":
        return _summarize_mechanics(
            records,
            expected_configs,
            validation_complete=compute_bootstrap,
        )
    if profile == "restart-screen-v1":
        return _summarize_screen(
            records,
            expected_configs,
            compute_bootstrap=compute_bootstrap,
            include_exploratory=include_exploratory,
        )
    raise ValueError(f"unsupported restart study profile: {profile!r}")


def _summarize_mechanics(
    records, expected_configs, *, validation_complete: bool
) -> dict[str, object]:
    policy = _decision_policy(expected_configs)
    complete = [record for record in records if record.get("status") == "complete"]
    errors = [record for record in records if record.get("status") == "error"]
    interrupted = [
        record for record in records if record.get("status") == "interrupted"
    ]
    checks = {
        "one_planned_run": len(expected_configs) == 1,
        "one_complete_run": len(complete) == 1,
        "zero_errors": not errors and not interrupted,
        "complete_record_revalidated": bool(validation_complete),
        "telemetry_present": False,
        "restart_observed": False,
        "post_restart_evaluation_observed": False,
        "within_operational_wall_seconds": False,
    }
    telemetry_summary = None
    wall_seconds = None
    if len(complete) == 1:
        record = complete[0]
        telemetry = record.get("optimizer_telemetry")
        if isinstance(telemetry, dict) and isinstance(telemetry.get("summary"), dict):
            telemetry_summary = telemetry["summary"]
            checks["telemetry_present"] = telemetry.get("mode") == "member-v1"
            checks["restart_observed"] = int(
                telemetry_summary.get("restart_rows", 0)
            ) >= int(policy["minimum_restart_rows"])
            checks["post_restart_evaluation_observed"] = int(
                telemetry_summary.get("post_restart_evaluation_rows", 0)
            ) >= int(policy["minimum_post_restart_evaluation_rows"])
        process = record.get("worker_process")
        if isinstance(process, dict):
            wall_seconds = process.get("full_wall_seconds")
            checks["within_operational_wall_seconds"] = bool(
                isinstance(wall_seconds, (int, float))
                and math.isfinite(float(wall_seconds))
                and float(wall_seconds)
                <= float(policy["maximum_worker_wall_seconds"])
            )
    execution_complete = len(records) == len(expected_configs)
    status = "pending"
    passed = False
    action = None
    if errors or interrupted:
        status = "failed"
        action = str(policy["action_if_failed"])
    elif execution_complete and validation_complete:
        passed = all(checks.values())
        status = "passed" if passed else "failed"
        action = str(
            policy["action_if_passed"] if passed else policy["action_if_failed"]
        )
    return {
        "format_version": 1,
        "study_profile": "restart-mechanics-v1",
        "completed_runs": len(complete),
        "error_runs": len(errors),
        "interrupted_runs": len(interrupted),
        "mechanics": {
            "checks": checks,
            "telemetry_summary": telemetry_summary,
            "worker_wall_seconds": wall_seconds,
            "loss_excluded_from_inference": bool(
                policy["exclude_loss_from_inference"]
            ),
        },
        "predeclared_decision": {
            "status": status,
            "passed": passed,
            "action": action,
        },
        "run_ids": [str(record["run_id"]) for record in records],
    }


def _summarize_screen(
    records,
    expected_configs,
    *,
    compute_bootstrap: bool,
    include_exploratory: bool,
):
    policy = _decision_policy(expected_configs)
    complete = [record for record in records if record.get("status") == "complete"]
    errors = [record for record in records if record.get("status") == "error"]
    interrupted = [
        record for record in records if record.get("status") == "interrupted"
    ]
    start_times = [
        _timestamp(record.get("started_utc"))
        for record in complete
        if _timestamp(record.get("started_utc")) is not None
    ]
    session_origin = min(start_times) if start_times else None
    by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in complete:
        config = record["config"]
        pair_id = str(config["pair_id"])
        arm = str(config["arm"])
        if arm in by_pair[pair_id]:
            raise ValueError(f"duplicate restart arm {arm!r} in pair {pair_id!r}")
        by_pair[pair_id][arm] = record

    expected_pairs = {
        str(config["pair_id"]) for config in expected_configs.values()
    }
    pair_rows = []
    for pair_id in sorted(expected_pairs):
        arms = by_pair.get(pair_id, {})
        if set(arms) != {RESTART_CONTROL_ARM, RESTART_TREATMENT_ARM}:
            continue
        control = arms[RESTART_CONTROL_ARM]
        treatment = arms[RESTART_TREATMENT_ARM]
        _assert_pair_integrity(pair_id, control, treatment)
        control_metrics = control["metrics"]
        treatment_metrics = treatment["metrics"]
        control_finite = bool(control_metrics.get("has_finite_feasible"))
        treatment_finite = bool(treatment_metrics.get("has_finite_feasible"))
        control_loss = control_metrics.get("best_feasible_loss")
        treatment_loss = treatment_metrics.get("best_feasible_loss")
        difference = (
            float(treatment_loss) - float(control_loss)
            if control_finite and treatment_finite
            else None
        )
        config = control["config"]
        problem = control["problem"]
        control_evals = _eval_count(control)
        treatment_evals = _eval_count(treatment)
        control_started = _timestamp(control.get("started_utc"))
        treatment_started = _timestamp(treatment.get("started_utc"))
        session_midpoint = (
            mean((control_started, treatment_started)) - session_origin
            if session_origin is not None
            and control_started is not None
            and treatment_started is not None
            else None
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(config["optimizer_seed"]),
                "configured_topology": config["topology"],
                "topology_sha256": str(problem["topology_sha256"]),
                "control_finite_feasible": control_finite,
                "treatment_finite_feasible": treatment_finite,
                "difference_p200_minus_p600": difference,
                "p200_first": int(treatment["config"]["run_order_within_pair"])
                < int(control["config"]["run_order_within_pair"]),
                "planned_run_index_midpoint": mean(
                    (
                        int(control["config"]["planned_run_index"]),
                        int(treatment["config"]["planned_run_index"]),
                    )
                ),
                "session_start_midpoint_seconds": session_midpoint,
                "control_eval_count": control_evals,
                "treatment_eval_count": treatment_evals,
                "log10_evaluation_ratio_p200_over_p600": (
                    math.log10(treatment_evals / control_evals)
                    if control_evals is not None
                    and treatment_evals is not None
                    and control_evals > 0
                    and treatment_evals > 0
                    else None
                ),
            }
        )

    expected_topologies: dict[str, set[int]] = defaultdict(set)
    for config in expected_configs.values():
        if config["arm"] == RESTART_CONTROL_ARM:
            key = _topology_key(config["topology"])
            expected_topologies[key].add(int(config["optimizer_seed"]))
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
            float(row["difference_p200_minus_p600"])
            for row in rows
            if row["difference_p200_minus_p600"] is not None
        ]
        replication_complete = observed_seeds == expected_seeds
        inference_complete = replication_complete and len(differences) == len(
            expected_seeds
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
                "mean_seed_difference_p200_minus_p600": (
                    mean(differences) if inference_complete else None
                ),
                "mean_planned_run_index": (
                    mean(float(row["planned_run_index_midpoint"]) for row in rows)
                    if replication_complete
                    else None
                ),
                "mean_session_start_seconds": (
                    mean(
                        float(row["session_start_midpoint_seconds"])
                        for row in rows
                    )
                    if replication_complete
                    and all(
                        row["session_start_midpoint_seconds"] is not None
                        for row in rows
                    )
                    else None
                ),
                "mean_log10_evaluation_ratio_p200_over_p600": (
                    mean(
                        float(row["log10_evaluation_ratio_p200_over_p600"])
                        for row in rows
                    )
                    if replication_complete
                    and all(
                        row["log10_evaluation_ratio_p200_over_p600"] is not None
                        for row in rows
                    )
                    else None
                ),
                "seed_pair_rows": rows,
            }
        )

    macro_values = [
        float(row["mean_seed_difference_p200_minus_p600"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    tolerance = 1e-12
    wins = sum(value < -tolerance for value in macro_values)
    ties = sum(abs(value) <= tolerance for value in macro_values)
    losses = sum(value > tolerance for value in macro_values)
    control_only_pairs = sum(
        row["control_finite_feasible"] and not row["treatment_finite_feasible"]
        for row in pair_rows
    )
    lower_treatment_topologies = sum(
        row["treatment_finite_feasible_seeds"]
        < row["control_finite_feasible_seeds"]
        for row in topology_rows
        if row["replication_complete"]
    )
    seed_means = {}
    for seed in sorted(
        {int(row["optimizer_seed"]) for row in pair_rows}
    ):
        values = [
            float(row["difference_p200_minus_p600"])
            for row in pair_rows
            if int(row["optimizer_seed"]) == seed
            and row["difference_p200_minus_p600"] is not None
        ]
        seed_means[str(seed)] = mean(values) if len(values) == 8 else None

    panel_execution_complete = (
        len(records) == len(expected_configs)
        and len(complete) == len(expected_configs)
        and not errors
        and not interrupted
        and len(pair_rows) == len(expected_pairs)
        and all(row["replication_complete"] for row in topology_rows)
    )
    inference_ready = panel_execution_complete and len(macro_values) == 8
    macro_mean = mean(macro_values) if inference_ready else None
    macro_median = median(macro_values) if inference_ready else None
    p90_regret = _percentile(sorted(macro_values), 0.9) if inference_ready else None
    criteria = {
        "panel_execution_complete": panel_execution_complete,
        "complete_records_revalidated": bool(compute_bootstrap),
        "inference_ready": inference_ready,
        "all_pairs_finite_comparable": len(pair_rows) == 16
        and all(
            row["difference_p200_minus_p600"] is not None for row in pair_rows
        ),
        "zero_control_only_finite_feasible_pairs": control_only_pairs == 0,
        "zero_topologies_with_lower_treatment_feasibility": (
            lower_treatment_topologies == 0
        ),
        "minimum_topology_wins_met": wins
        >= int(policy["minimum_patience_200_topology_wins"]),
        "median_difference_at_most_negative_0_05": bool(
            macro_median is not None
            and macro_median
            <= float(policy["maximum_topology_median_difference"])
        ),
        "mean_difference_below_zero": bool(
            macro_mean is not None and macro_mean < 0
        ),
        "p90_regret_at_most_0_5": bool(
            p90_regret is not None
            and p90_regret <= float(policy["maximum_topology_p90_regret"])
        ),
        "both_seed_mean_differences_below_zero": bool(
            set(seed_means) == {"19", "23"}
            and all(value is not None and value < 0 for value in seed_means.values())
        ),
    }
    status = "pending"
    passed = False
    action = None
    if errors or interrupted:
        status = "failed"
        action = str(policy["action_if_failed"])
    elif panel_execution_complete and compute_bootstrap:
        passed = all(criteria.values())
        status = "passed" if passed else "failed"
        action = str(
            policy["action_if_passed"] if passed else policy["action_if_failed"]
        )

    bootstrap_ci = None
    sign_flip_p = None
    sign_p = None
    if inference_ready and compute_bootstrap:
        bootstrap_ci = _bootstrap_mean_ci(macro_values)
    if inference_ready and compute_bootstrap and include_exploratory:
        sign_flip_p = _exact_sign_flip_mean_pvalue(macro_values)
        sign_p = _exact_sign_pvalue(wins, losses)
    exploratory = (
        _exploratory_sensitivity(
            topology_rows,
            macro_values,
            inference_ready=inference_ready,
            compute_bootstrap=compute_bootstrap,
        )
        if include_exploratory
        else {
            "ready": False,
            "deferred_until_frozen_replay_match": True,
            "changes_frozen_decision": False,
        }
    )
    return {
        "format_version": 1,
        "study_profile": "restart-screen-v1",
        "completed_runs": len(complete),
        "error_runs": len(errors),
        "interrupted_runs": len(interrupted),
        "complete_optimizer_seed_pairs": len(pair_rows),
        "finite_comparable_optimizer_seed_pairs": sum(
            row["difference_p200_minus_p600"] is not None for row in pair_rows
        ),
        "complete_topologies": sum(
            bool(row["replication_complete"]) for row in topology_rows
        ),
        "finite_comparable_topologies": len(macro_values),
        "wins_ties_losses": {
            "p200_wins": wins,
            "ties": ties,
            "p200_losses": losses,
        },
        "control_only_finite_feasible_pairs": control_only_pairs,
        "topologies_with_lower_treatment_feasibility": lower_treatment_topologies,
        "topology_macro_mean_difference": macro_mean,
        "topology_macro_median_difference": macro_median,
        "topology_p90_regret": p90_regret,
        "optimizer_seed_mean_differences": seed_means,
        "topology_bootstrap_mean_difference_ci_95": bootstrap_ci,
        "exact_sign_flip_mean_pvalue_two_sided": sign_flip_p,
        "exact_sign_flip_assignments": 2**8,
        "exact_sign_test_pvalue_two_sided": sign_p,
        "bootstrap_seed": RESTART_SCREEN_BOOTSTRAP_SEED,
        "bootstrap_resamples": RESTART_SCREEN_BOOTSTRAP_RESAMPLES,
        "predeclared_decision": {
            "status": status,
            "passed": passed,
            "action": action,
            "criteria": criteria,
        },
        "topology_differences": topology_rows,
        "optimizer_seed_pair_rows": pair_rows,
        "exploratory_sensitivity": exploratory,
        "run_ids": [str(record["run_id"]) for record in records],
        "note": (
            "Topology is the inference unit; optimizer seeds are repeated paired "
            "measurements. Negative differences favor patience 200. Exploratory "
            "p-values cannot override the frozen decision criteria."
        ),
    }


def _assert_pair_integrity(pair_id, control, treatment) -> None:
    control_config = control["config"]
    treatment_config = treatment["config"]
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
        "jax_compilation_cache_policy",
        "study_profile",
        "decision_policy",
        "mechanics_evidence",
        "provider_stop_utc",
        "provider_deadline_maximum_horizon_seconds",
        "provider_evacuation_reserve_seconds",
    ):
        if control_config.get(field) != treatment_config.get(field):
            raise ValueError(f"restart pair {pair_id!r} disagrees on {field}")
    if control["problem"] != treatment["problem"]:
        raise ValueError(f"restart pair {pair_id!r} problem mismatch")
    control_settings = control_config["optimizer_settings"]
    treatment_settings = treatment_config["optimizer_settings"]
    control_settings = validate_batched_settings(control_settings)
    treatment_settings = validate_batched_settings(treatment_settings)
    differing = {
        key
        for key in control_settings
        if control_settings[key] != treatment_settings[key]
    }
    if differing != {"patience"}:
        raise ValueError(
            f"restart pair {pair_id!r} settings differ beyond patience"
        )
    if control_settings["patience"] != 600 or treatment_settings["patience"] != 200:
        raise ValueError(f"restart pair {pair_id!r} patience values are not frozen")


def _decision_policy(
    expected_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    policies = [config.get("decision_policy") for config in expected_configs.values()]
    if not policies or not all(isinstance(policy, dict) for policy in policies):
        raise ValueError("restart study is missing its frozen decision policy")
    first = policies[0]
    assert isinstance(first, dict)
    if any(policy != first for policy in policies[1:]):
        raise ValueError("restart study configurations disagree on decision policy")
    return first


def _topology_key(topology: object) -> str:
    import json

    return json.dumps(topology, sort_keys=True, separators=(",", ":"))


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(values: list[float]) -> list[float]:
    generator = random.Random(RESTART_SCREEN_BOOTSTRAP_SEED)
    samples = []
    for _ in range(RESTART_SCREEN_BOOTSTRAP_RESAMPLES):
        samples.append(mean(generator.choice(values) for _ in values))
    ordered = sorted(samples)
    return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]


def _exact_sign_flip_mean_pvalue(values: list[float]) -> float:
    observed = abs(mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def _exact_sign_pvalue(wins: int, losses: int) -> float | None:
    nonzero = wins + losses
    if nonzero == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(nonzero, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**nonzero))


def _exploratory_sensitivity(
    topology_rows: list[dict[str, object]],
    macro_values: list[float],
    *,
    inference_ready: bool,
    compute_bootstrap: bool,
) -> dict[str, object]:
    if not inference_ready:
        return {
            "ready": False,
            "leave_one_topology_out": [],
            "arm_first": None,
            "serial_order": None,
            "evaluation_throughput": None,
        }
    leave_one_out = []
    for omitted_index, omitted in enumerate(topology_rows):
        retained = [
            value for index, value in enumerate(macro_values) if index != omitted_index
        ]
        tolerance = 1e-12
        leave_one_out.append(
            {
                "omitted_topology_sha256": omitted["topology_sha256"],
                "mean_difference": mean(retained),
                "median_difference": median(retained),
                "p200_wins": sum(value < -tolerance for value in retained),
                "ties": sum(abs(value) <= tolerance for value in retained),
                "p200_losses": sum(value > tolerance for value in retained),
                "p90_regret": _percentile(sorted(retained), 0.9),
            }
        )

    order_contrasts = []
    for row in topology_rows:
        seed_rows = row["seed_pair_rows"]
        p200_first = [
            float(seed_row["difference_p200_minus_p600"])
            for seed_row in seed_rows
            if seed_row["p200_first"]
        ]
        p600_first = [
            float(seed_row["difference_p200_minus_p600"])
            for seed_row in seed_rows
            if not seed_row["p200_first"]
        ]
        if len(p200_first) != 1 or len(p600_first) != 1:
            raise ValueError("restart topology does not have opposite seed arm order")
        order_contrasts.append(p200_first[0] - p600_first[0])

    run_indexes = [float(row["mean_planned_run_index"]) for row in topology_rows]
    session_times = [
        row["mean_session_start_seconds"] for row in topology_rows
    ]
    throughput = [
        row["mean_log10_evaluation_ratio_p200_over_p600"]
        for row in topology_rows
    ]
    throughput_ready = all(value is not None for value in throughput)
    throughput_values = [float(value) for value in throughput if value is not None]
    return {
        "ready": True,
        "leave_one_topology_out": leave_one_out,
        "arm_first": {
            "estimand": (
                "topology p200-minus-p600 difference when p200 ran first minus "
                "the difference when p600 ran first"
            ),
            "topology_contrasts": order_contrasts,
            "mean_contrast": mean(order_contrasts),
            "median_contrast": median(order_contrasts),
            "exact_sign_flip_mean_pvalue_two_sided": (
                _exact_sign_flip_mean_pvalue(order_contrasts)
                if compute_bootstrap
                else None
            ),
        },
        "serial_order": {
            "spearman_planned_run_index_vs_difference": _spearman(
                run_indexes, macro_values
            ),
            "spearman_session_start_vs_difference": (
                _spearman([float(value) for value in session_times], macro_values)
                if all(value is not None for value in session_times)
                else None
            ),
        },
        "evaluation_throughput": (
            {
                "estimand": "topology mean log10(p200 evals / p600 evals)",
                "topology_values": throughput_values,
                "mean": mean(throughput_values),
                "median": median(throughput_values),
                "topology_bootstrap_mean_ci_95": (
                    _bootstrap_mean_ci(throughput_values)
                    if compute_bootstrap
                    else None
                ),
            }
            if throughput_ready
            else None
        ),
        "changes_frozen_decision": False,
    }


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _eval_count(record: dict[str, object]) -> int | None:
    accounting = record.get("objective_accounting")
    if not isinstance(accounting, dict):
        return None
    value = accounting.get("eval_count")
    return int(value) if isinstance(value, int) and value > 0 else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_centered = [value - mean(left_ranks) for value in left_ranks]
    right_centered = [value - mean(right_ranks) for value in right_ranks]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for ordered_index in ordered[index:end]:
            result[ordered_index] = rank
        index = end
    return result
