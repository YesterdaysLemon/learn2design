"""Small transparent aggregates for completed paired runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from statistics import mean, median


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260819


def summarize_records(
    records: list[dict[str, object]],
    expected_configs: dict[str, dict[str, object]] | None = None,
    *,
    compute_bootstrap: bool = True,
) -> dict[str, object]:
    complete = [record for record in records if record.get("status") == "complete"]
    errors = [record for record in records if record.get("status") == "error"]
    interrupted = [
        record for record in records if record.get("status") == "interrupted"
    ]

    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in complete:
        config = _required_mapping(record, "config")
        arm = str(_required(config, "arm", "config"))
        pair_id = str(_required(config, "pair_id", "config"))
        if arm in by_pair[pair_id]:
            raise ValueError(f"duplicate complete arm {arm!r} for pair {pair_id!r}")
        by_arm[arm].append(record)
        by_pair[pair_id][arm] = record

    arm_summary = {}
    for arm, arm_records in sorted(by_arm.items()):
        finite_losses = [
            float(record["metrics"]["best_feasible_loss"])
            for record in arm_records
            if record["metrics"]["best_feasible_loss"] is not None
        ]
        arm_summary[arm] = {
            "runs": len(arm_records),
            "physically_feasible_runs": sum(
                bool(record["metrics"]["has_feasible"])
                for record in arm_records
            ),
            "finite_feasible_runs": len(finite_losses),
            "conditional_mean_best_feasible_loss": (
                mean(finite_losses) if finite_losses else None
            ),
        }

    paired = _paired_prior_summary(by_pair, expected_configs, compute_bootstrap)
    return {
        "format_version": 1,
        "completed_runs": len(complete),
        "error_runs": len(errors),
        "interrupted_runs": len(interrupted),
        "arm_summary": arm_summary,
        "semantic_prior_vs_no_prior": paired,
        "run_ids": [str(record["run_id"]) for record in records],
    }


def _paired_prior_summary(
    by_pair,
    expected_configs: dict[str, dict[str, object]] | None,
    compute_bootstrap: bool,
) -> dict[str, object]:
    seed_pair_rows = []
    for pair_id, arms in sorted(by_pair.items()):
        if "no_prior" not in arms or "semantic_prior" not in arms:
            continue
        control = arms["no_prior"]
        treatment = arms["semantic_prior"]
        _assert_pair_integrity(pair_id, control, treatment)

        control_metrics = _required_mapping(control, "metrics")
        treatment_metrics = _required_mapping(treatment, "metrics")
        control_feasible = bool(
            _required(control_metrics, "has_feasible", "control metrics")
        )
        treatment_feasible = bool(
            _required(treatment_metrics, "has_feasible", "treatment metrics")
        )
        control_loss = control_metrics.get("best_feasible_loss")
        treatment_loss = treatment_metrics.get("best_feasible_loss")
        control_finite_feasible = control_loss is not None
        treatment_finite_feasible = treatment_loss is not None
        difference = None
        if control_loss is not None and treatment_loss is not None:
            control_value = _finite_float(control_loss, "control best feasible loss")
            treatment_value = _finite_float(
                treatment_loss, "treatment best feasible loss"
            )
            difference = treatment_value - control_value

        config = _required_mapping(control, "config")
        problem = _required_mapping(control, "problem")
        seed_pair_rows.append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(
                    _required(config, "optimizer_seed", "control config")
                ),
                "topology_sha256": str(
                    _required(problem, "topology_sha256", "control problem")
                ),
                "configured_topology_key": _canonical(config["topology"]),
                "control_physically_feasible": control_feasible,
                "treatment_physically_feasible": treatment_feasible,
                "control_finite_feasible": control_finite_feasible,
                "treatment_finite_feasible": treatment_finite_feasible,
                "difference_treatment_minus_control": difference,
                "target_hitting": _paired_target_hits(
                    pair_id,
                    control_metrics,
                    treatment_metrics,
                    list(config.get("target_losses", [])),
                ),
            }
        )

    seed_pair_diagnostics = _optimizer_seed_pair_diagnostics(seed_pair_rows)
    expected_by_topology = _expected_causal_pairs(expected_configs)
    topology_differences = _collapse_seed_pairs_by_topology(
        seed_pair_rows, expected_by_topology
    )
    macro_values = [
        float(row["mean_seed_difference_treatment_minus_control"])
        for row in topology_differences
        if row["inference_complete"]
        and row["mean_seed_difference_treatment_minus_control"] is not None
    ]
    observed_topology_values = [
        float(row["mean_seed_difference_treatment_minus_control"])
        for row in topology_differences
        if row["replication_complete"]
        and row["mean_seed_difference_treatment_minus_control"] is not None
    ]

    complete_topology_rows = [
        row for row in topology_differences if row["replication_complete"]
    ]

    topology_wins_ties_losses = {
        "semantic_prior_wins": sum(value < 0 for value in macro_values),
        "ties": sum(value == 0 for value in macro_values),
        "semantic_prior_losses": sum(value > 0 for value in macro_values),
    }
    topology_feasibility_discordance = {
        "semantic_prior_higher_seed_feasibility_rate": sum(
            row["treatment_physical_feasibility_rate"]
            > row["control_physical_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_physical_feasibility_rate"] is not None
        ),
        "equal_seed_feasibility_rate": sum(
            row["treatment_physical_feasibility_rate"]
            == row["control_physical_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_physical_feasibility_rate"] is not None
        ),
        "no_prior_higher_seed_feasibility_rate": sum(
            row["treatment_physical_feasibility_rate"]
            < row["control_physical_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_physical_feasibility_rate"] is not None
        ),
    }
    topology_finite_feasibility_discordance = {
        "semantic_prior_higher_seed_finite_feasibility_rate": sum(
            row["treatment_finite_feasibility_rate"]
            > row["control_finite_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_finite_feasibility_rate"] is not None
        ),
        "equal_seed_finite_feasibility_rate": sum(
            row["treatment_finite_feasibility_rate"]
            == row["control_finite_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_finite_feasibility_rate"] is not None
        ),
        "no_prior_higher_seed_finite_feasibility_rate": sum(
            row["treatment_finite_feasibility_rate"]
            < row["control_finite_feasibility_rate"]
            for row in complete_topology_rows
            if row["treatment_finite_feasibility_rate"] is not None
        ),
    }

    incomplete_topologies = [
        {
            "configured_topology": row["configured_topology"],
            "topology_sha256": row["topology_sha256"],
            "expected_optimizer_seed_pairs": row[
                "expected_optimizer_seed_pairs"
            ],
            "complete_optimizer_seed_pairs": row["optimizer_seed_pairs"],
        }
        for row in topology_differences
        if not row["replication_complete"]
    ]
    censored_topologies = [
        {
            "configured_topology": row["configured_topology"],
            "topology_sha256": row["topology_sha256"],
            "expected_optimizer_seed_pairs": row[
                "expected_optimizer_seed_pairs"
            ],
            "finite_comparable_optimizer_seed_pairs": row[
                "finite_comparable_optimizer_seed_pairs"
            ],
        }
        for row in topology_differences
        if row["replication_complete"] and not row["inference_complete"]
    ]

    bootstrap_ci = _topology_bootstrap_ci(macro_values) if compute_bootstrap else None
    topology_p90_regret = (
        _percentile(sorted(macro_values), 0.9) if macro_values else None
    )
    observed_topology_p90_regret_guard = (
        _percentile(sorted(observed_topology_values), 0.9)
        if observed_topology_values
        else None
    )
    panel_execution_complete = (
        not incomplete_topologies and bool(expected_by_topology)
        if expected_configs is not None
        else None
    )
    promotion_inference_ready = (
        panel_execution_complete is True and not censored_topologies
        if expected_configs is not None
        else None
    )
    decision_policy = _common_decision_policy(expected_configs)
    predeclared_decision = _evaluate_predeclared_decision(
        decision_policy,
        panel_execution_complete,
        promotion_inference_ready,
        topology_wins_ties_losses,
        topology_finite_feasibility_discordance,
        seed_pair_diagnostics,
        median(macro_values) if macro_values else None,
        topology_p90_regret,
        observed_topology_p90_regret_guard,
        bootstrap_ci,
    )

    return {
        "complete_optimizer_seed_pairs": len(seed_pair_rows),
        "finite_comparable_optimizer_seed_pairs": sum(
            row["difference_treatment_minus_control"] is not None
            for row in seed_pair_rows
        ),
        "observed_or_planned_topologies": len(topology_differences),
        "complete_topologies": len(complete_topology_rows),
        "incomplete_topologies": incomplete_topologies,
        "censored_topologies": censored_topologies,
        "panel_execution_complete": panel_execution_complete,
        "promotion_inference_ready": promotion_inference_ready,
        "finite_comparable_topologies": len(macro_values),
        "wins_ties_losses": topology_wins_ties_losses,
        "physical_feasibility_discordance": topology_feasibility_discordance,
        "finite_feasibility_discordance": (
            topology_finite_feasibility_discordance
        ),
        "topology_macro_mean_difference": mean(macro_values) if macro_values else None,
        "topology_macro_median_difference": (
            median(macro_values) if macro_values else None
        ),
        "topology_p90_regret": topology_p90_regret,
        "observed_topology_p90_regret_guard": (
            observed_topology_p90_regret_guard
        ),
        "topology_bootstrap_mean_difference_ci_95": bootstrap_ci,
        "predeclared_decision": predeclared_decision,
        "target_hitting_time_inference": _target_hitting_summary(
            seed_pair_rows,
            expected_by_topology,
            expected_configs,
            compute_bootstrap,
        ),
        "topology_differences": topology_differences,
        "optimizer_seed_pair_diagnostics": seed_pair_diagnostics,
        "note": (
            "Primary wins, ties, losses, feasibility discordance, and confidence "
            "intervals use topology as the inference unit after collapsing optimizer "
            "seeds. Primary loss aggregates require every predeclared seed pair to "
            "be present and finite-feasible in both arms; the separate guarded "
            "finite-feasibility dominance route handles semantic-only outcomes. "
            "Inspect incomplete and censored topology lists, feasibility rates, "
            "decision routes, and seed diagnostics."
        ),
    }


def _assert_pair_integrity(
    pair_id: str,
    control: dict[str, object],
    treatment: dict[str, object],
) -> None:
    control_config = _required_mapping(control, "config")
    treatment_config = _required_mapping(treatment, "config")
    if control_config.get("arm") != "no_prior":
        raise ValueError(f"pair {pair_id!r} control arm is not no_prior")
    if treatment_config.get("arm") != "semantic_prior":
        raise ValueError(f"pair {pair_id!r} treatment arm is not semantic_prior")

    paired_config_fields = (
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
    )
    for field in paired_config_fields:
        control_value = _required(control_config, field, "control config")
        treatment_value = _required(treatment_config, field, "treatment config")
        if _canonical(control_value) != _canonical(treatment_value):
            raise ValueError(f"pair {pair_id!r} config mismatch for {field}")
    for field in ("study_profile", "decision_policy"):
        if field in control_config or field in treatment_config:
            control_value = _required(control_config, field, "control config")
            treatment_value = _required(treatment_config, field, "treatment config")
            if _canonical(control_value) != _canonical(treatment_value):
                raise ValueError(f"pair {pair_id!r} config mismatch for {field}")
    if str(control_config["pair_id"]) != pair_id:
        raise ValueError(f"pair {pair_id!r} record has inconsistent pair_id")

    population_size = int(control_config["population_size"])
    if population_size < 2:
        raise ValueError(f"pair {pair_id!r} population must contain member 1")

    control_problem = _required_mapping(control, "problem")
    treatment_problem = _required_mapping(treatment, "problem")
    for field in ("topology_sha256", "spec", "n_params"):
        control_value = _required(control_problem, field, "control problem")
        treatment_value = _required(treatment_problem, field, "treatment problem")
        if _canonical(control_value) != _canonical(treatment_value):
            raise ValueError(f"pair {pair_id!r} problem mismatch for {field}")
    _assert_topology_digest(pair_id, control_problem, "control")
    _assert_topology_digest(pair_id, treatment_problem, "treatment")

    control_environment = _required_mapping(control, "environment")
    treatment_environment = _required_mapping(treatment, "environment")
    if _canonical(control_environment) != _canonical(treatment_environment):
        raise ValueError(f"pair {pair_id!r} runtime environment mismatch")

    control_objective = _required_mapping(control, "objective_configuration")
    treatment_objective = _required_mapping(treatment, "objective_configuration")
    if _canonical(control_objective) != _canonical(treatment_objective):
        raise ValueError(f"pair {pair_id!r} Objective configuration mismatch")

    control_algorithm = _algorithm_without_prior_flag(
        pair_id, control, expected_flag=False
    )
    treatment_algorithm = _algorithm_without_prior_flag(
        pair_id, treatment, expected_flag=True
    )
    if _canonical(control_algorithm) != _canonical(treatment_algorithm):
        raise ValueError(
            f"pair {pair_id!r} algorithm settings differ beyond use_semantic_prior"
        )

    expected_control_roles = ["anchor"] + ["random"] * (population_size - 1)
    expected_treatment_roles = ["anchor", "semantic_prior"] + [
        "random"
    ] * (population_size - 2)
    control_roles = _required(control, "initial_population_roles", "control record")
    treatment_roles = _required(
        treatment, "initial_population_roles", "treatment record"
    )
    if control_roles != expected_control_roles:
        raise ValueError(f"pair {pair_id!r} has unexpected control population roles")
    if treatment_roles != expected_treatment_roles:
        raise ValueError(
            f"pair {pair_id!r} has unexpected treatment population roles"
        )

    control_hashes = _required(
        control, "initial_parameter_hashes", "control record"
    )
    treatment_hashes = _required(
        treatment, "initial_parameter_hashes", "treatment record"
    )
    if not isinstance(control_hashes, list) or not isinstance(treatment_hashes, list):
        raise ValueError(f"pair {pair_id!r} initial hashes must be lists")
    if len(control_hashes) != population_size or len(treatment_hashes) != population_size:
        raise ValueError(f"pair {pair_id!r} initial hash count disagrees with population")
    if any(not isinstance(value, str) or not value for value in control_hashes):
        raise ValueError(f"pair {pair_id!r} has invalid control initial hashes")
    if any(not isinstance(value, str) or not value for value in treatment_hashes):
        raise ValueError(f"pair {pair_id!r} has invalid treatment initial hashes")
    if control_hashes[0] != treatment_hashes[0]:
        raise ValueError(f"pair {pair_id!r} anchor member hash differs")
    if control_hashes[1] == treatment_hashes[1]:
        raise ValueError(f"pair {pair_id!r} replaced member 1 hash did not differ")
    if control_hashes[2:] != treatment_hashes[2:]:
        raise ValueError(f"pair {pair_id!r} unaffected random member hashes differ")


def _assert_topology_digest(
    pair_id: str, problem: dict[str, object], arm_name: str
) -> None:
    topology = str(_required(problem, "topology_string", f"{arm_name} problem"))
    recorded = str(
        _required(problem, "topology_sha256", f"{arm_name} problem")
    )
    actual = hashlib.sha256(topology.encode()).hexdigest()
    if recorded != actual:
        raise ValueError(f"pair {pair_id!r} {arm_name} topology digest is invalid")


def _algorithm_without_prior_flag(
    pair_id: str, record: dict[str, object], *, expected_flag: bool
) -> dict[str, object]:
    algorithm = copy.deepcopy(_required_mapping(record, "algorithm"))
    kwargs = _required_mapping(algorithm, "kwargs")
    flag = _required(kwargs, "use_semantic_prior", "algorithm kwargs")
    if type(flag) is not bool or flag is not expected_flag:
        arm = "semantic_prior" if expected_flag else "no_prior"
        raise ValueError(
            f"pair {pair_id!r} {arm} algorithm has invalid use_semantic_prior"
        )
    del kwargs["use_semantic_prior"]
    return algorithm


def _paired_target_hits(
    pair_id: str,
    control_metrics: dict[str, object],
    treatment_metrics: dict[str, object],
    configured_targets: list[object],
) -> dict[str, dict[str, object]]:
    if not configured_targets:
        return {}
    control_targets = _required_mapping(control_metrics, "targets")
    treatment_targets = _required_mapping(treatment_metrics, "targets")
    result = {}
    for target in configured_targets:
        target_key = format(_finite_float(target, "configured target loss"), ".12g")
        control_hit = _target_hit(
            _required_mapping(control_targets, target_key),
            f"pair {pair_id!r} no-prior target {target_key}",
        )
        treatment_hit = _target_hit(
            _required_mapping(treatment_targets, target_key),
            f"pair {pair_id!r} semantic-prior target {target_key}",
        )
        if control_hit["reached"] and treatment_hit["reached"]:
            outcome = "both_reached"
            ratio_evidence = "observed_hits"
            time_log10_ratio = math.log10(
                treatment_hit["time_seconds"] / control_hit["time_seconds"]
            )
            eval_log10_ratio = math.log10(
                treatment_hit["eval_count"] / control_hit["eval_count"]
            )
        elif treatment_hit["reached"]:
            outcome = "semantic_prior_only"
            ratio_evidence = "conservative_upper_bound_from_no_prior_censoring"
            control_horizon = _target_censor_horizon(
                control_metrics, f"pair {pair_id!r} no-prior censoring"
            )
            time_log10_ratio = math.log10(
                treatment_hit["time_seconds"] / control_horizon["time_seconds"]
            )
            eval_log10_ratio = math.log10(
                treatment_hit["eval_count"] / control_horizon["eval_count"]
            )
        elif control_hit["reached"]:
            outcome = "no_prior_only"
            ratio_evidence = None
            time_log10_ratio = None
            eval_log10_ratio = None
        else:
            outcome = "neither_reached"
            ratio_evidence = None
            time_log10_ratio = None
            eval_log10_ratio = None
        result[target_key] = {
            "outcome": outcome,
            "no_prior": control_hit,
            "semantic_prior": treatment_hit,
            "ratio_evidence": ratio_evidence,
            "log10_time_ratio_semantic_over_no_prior": time_log10_ratio,
            "log10_eval_ratio_semantic_over_no_prior": eval_log10_ratio,
        }
    return result


def _target_hit(payload: dict[str, object], label: str) -> dict[str, object]:
    time_value = payload.get("time_seconds")
    eval_value = payload.get("eval_count")
    if time_value is None and eval_value is None:
        return {"reached": False, "time_seconds": None, "eval_count": None}
    if time_value is None or eval_value is None:
        raise ValueError(f"{label} has a partial target hit")
    time_seconds = _finite_float(time_value, f"{label} time")
    eval_count = int(eval_value)
    if time_seconds <= 0 or eval_count <= 0:
        raise ValueError(f"{label} hit coordinates must be positive")
    return {
        "reached": True,
        "time_seconds": time_seconds,
        "eval_count": eval_count,
    }


def _target_censor_horizon(
    metrics: dict[str, object], label: str
) -> dict[str, float | int]:
    time_seconds = _finite_float(
        _required(metrics, "last_logged_time_seconds", label), f"{label} time"
    )
    eval_count = int(_required(metrics, "last_logged_eval_count", label))
    if time_seconds <= 0 or eval_count <= 0:
        raise ValueError(f"{label} coordinates must be positive")
    return {"time_seconds": time_seconds, "eval_count": eval_count}


def _optimizer_seed_pair_diagnostics(
    seed_pair_rows: list[dict[str, object]],
) -> dict[str, object]:
    finite_rows = [
        row
        for row in seed_pair_rows
        if row["difference_treatment_minus_control"] is not None
    ]
    differences = [
        float(row["difference_treatment_minus_control"]) for row in finite_rows
    ]
    discordance = {
        "both_physically_feasible": 0,
        "neither_physically_feasible": 0,
        "semantic_prior_only": 0,
        "no_prior_only": 0,
    }
    finite_discordance = {
        "both_finite_feasible": 0,
        "neither_finite_feasible": 0,
        "semantic_prior_only": 0,
        "no_prior_only": 0,
    }
    for row in seed_pair_rows:
        control_feasible = bool(row["control_physically_feasible"])
        treatment_feasible = bool(row["treatment_physically_feasible"])
        if control_feasible and treatment_feasible:
            discordance["both_physically_feasible"] += 1
        elif treatment_feasible:
            discordance["semantic_prior_only"] += 1
        elif control_feasible:
            discordance["no_prior_only"] += 1
        else:
            discordance["neither_physically_feasible"] += 1
        control_finite = bool(row["control_finite_feasible"])
        treatment_finite = bool(row["treatment_finite_feasible"])
        if control_finite and treatment_finite:
            finite_discordance["both_finite_feasible"] += 1
        elif treatment_finite:
            finite_discordance["semantic_prior_only"] += 1
        elif control_finite:
            finite_discordance["no_prior_only"] += 1
        else:
            finite_discordance["neither_finite_feasible"] += 1
    return {
        "complete_optimizer_seed_pairs": len(seed_pair_rows),
        "finite_comparable_optimizer_seed_pairs": len(finite_rows),
        "optimizer_seed_pair_wins_ties_losses": {
            "semantic_prior_wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "semantic_prior_losses": sum(value > 0 for value in differences),
        },
        "optimizer_seed_pair_physical_feasibility_discordance": discordance,
        "optimizer_seed_pair_finite_feasibility_discordance": finite_discordance,
        "optimizer_seed_pair_rows": seed_pair_rows,
    }


def _collapse_seed_pairs_by_topology(
    seed_pair_rows: list[dict[str, object]],
    expected_by_topology: dict[str, set[str]],
) -> list[dict[str, object]]:
    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in seed_pair_rows:
        by_topology[str(row["topology_sha256"])].append(row)

    result = []
    observed_configured_topologies = set()
    for topology, rows in sorted(by_topology.items()):
        rows = sorted(
            rows, key=lambda row: (int(row["optimizer_seed"]), row["pair_id"])
        )
        optimizer_seeds = [int(row["optimizer_seed"]) for row in rows]
        if len(set(optimizer_seeds)) != len(optimizer_seeds):
            raise ValueError(
                f"topology {topology!r} has duplicate optimizer-seed pairs"
            )
        configured_keys = {str(row["configured_topology_key"]) for row in rows}
        if len(configured_keys) != 1:
            raise ValueError(
                f"resolved topology {topology!r} combines distinct planned topologies"
            )
        configured_key = next(iter(configured_keys))
        observed_configured_topologies.add(configured_key)
        completed_pair_ids = {str(row["pair_id"]) for row in rows}
        expected_pair_ids = expected_by_topology.get(
            configured_key, completed_pair_ids
        )
        replication_complete = completed_pair_ids == expected_pair_ids
        differences = [
            float(row["difference_treatment_minus_control"])
            for row in rows
            if row["difference_treatment_minus_control"] is not None
        ]
        control_feasible = sum(
            bool(row["control_physically_feasible"]) for row in rows
        )
        treatment_feasible = sum(
            bool(row["treatment_physically_feasible"]) for row in rows
        )
        control_finite = sum(bool(row["control_finite_feasible"]) for row in rows)
        treatment_finite = sum(
            bool(row["treatment_finite_feasible"]) for row in rows
        )
        result.append(
            {
                "topology_sha256": topology,
                "configured_topology": json.loads(configured_key),
                "optimizer_seed_pairs": len(rows),
                "expected_optimizer_seed_pairs": len(expected_pair_ids),
                "replication_complete": replication_complete,
                "finite_comparable_optimizer_seed_pairs": len(differences),
                "mean_seed_difference_treatment_minus_control": (
                    mean(differences) if differences else None
                ),
                "control_physically_feasible_optimizer_seeds": control_feasible,
                "treatment_physically_feasible_optimizer_seeds": treatment_feasible,
                "control_physical_feasibility_rate": control_feasible / len(rows),
                "treatment_physical_feasibility_rate": (
                    treatment_feasible / len(rows)
                ),
                "control_finite_feasible_optimizer_seeds": control_finite,
                "treatment_finite_feasible_optimizer_seeds": treatment_finite,
                "control_finite_feasibility_rate": control_finite / len(rows),
                "treatment_finite_feasibility_rate": treatment_finite / len(rows),
                "optimizer_seeds": optimizer_seeds,
            }
        )
        result[-1]["inference_complete"] = (
            replication_complete and len(differences) == len(expected_pair_ids)
        )
    for configured_key, expected_pair_ids in sorted(expected_by_topology.items()):
        if configured_key in observed_configured_topologies:
            continue
        result.append(
            {
                "topology_sha256": None,
                "configured_topology": json.loads(configured_key),
                "optimizer_seed_pairs": 0,
                "expected_optimizer_seed_pairs": len(expected_pair_ids),
                "replication_complete": False,
                "inference_complete": False,
                "finite_comparable_optimizer_seed_pairs": 0,
                "mean_seed_difference_treatment_minus_control": None,
                "control_physically_feasible_optimizer_seeds": 0,
                "treatment_physically_feasible_optimizer_seeds": 0,
                "control_physical_feasibility_rate": None,
                "treatment_physical_feasibility_rate": None,
                "control_finite_feasible_optimizer_seeds": 0,
                "treatment_finite_feasible_optimizer_seeds": 0,
                "control_finite_feasibility_rate": None,
                "treatment_finite_feasibility_rate": None,
                "optimizer_seeds": [],
            }
        )
    return result


def _expected_causal_pairs(
    expected_configs: dict[str, dict[str, object]] | None,
) -> dict[str, set[str]]:
    if expected_configs is None:
        return {}
    arms_by_pair: dict[str, set[str]] = defaultdict(set)
    topology_by_pair: dict[str, str] = {}
    for config in expected_configs.values():
        arm = str(config.get("arm"))
        if arm not in {"no_prior", "semantic_prior"}:
            continue
        pair_id = str(_required(config, "pair_id", "expected config"))
        topology_key = _canonical(
            _required(config, "topology", "expected config")
        )
        previous = topology_by_pair.setdefault(pair_id, topology_key)
        if previous != topology_key:
            raise ValueError(f"planned pair {pair_id!r} mixes topologies")
        arms_by_pair[pair_id].add(arm)

    expected: dict[str, set[str]] = defaultdict(set)
    for pair_id, arms in arms_by_pair.items():
        if arms == {"no_prior", "semantic_prior"}:
            expected[topology_by_pair[pair_id]].add(pair_id)
    return dict(expected)


def _common_decision_policy(
    expected_configs: dict[str, dict[str, object]] | None,
) -> dict[str, object] | None:
    if expected_configs is None:
        return None
    policies = {
        _canonical(config.get("decision_policy"))
        for config in expected_configs.values()
        if config.get("arm") in {"no_prior", "semantic_prior"}
    }
    if not policies or policies == {"null"}:
        return None
    if "null" in policies or len(policies) != 1:
        raise ValueError("planned causal runs do not share one decision policy")
    policy = json.loads(next(iter(policies)))
    if not isinstance(policy, dict):
        raise TypeError("decision policy must be a mapping")
    return policy


def _evaluate_predeclared_decision(
    policy: dict[str, object] | None,
    panel_execution_complete: bool | None,
    inference_ready: bool | None,
    wins_ties_losses: dict[str, int],
    topology_finite_feasibility_discordance: dict[str, int],
    seed_pair_diagnostics: dict[str, object],
    median_difference: float | None,
    p90_regret: float | None,
    observed_p90_regret_guard: float | None,
    bootstrap_ci: dict[str, object] | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    seed_discordance = seed_pair_diagnostics[
        "optimizer_seed_pair_finite_feasibility_discordance"
    ]
    assert isinstance(seed_discordance, dict)
    required_wins = int(policy["minimum_semantic_prior_topology_wins"])
    minimum_reduction = float(policy["minimum_practical_median_loss_reduction"])
    maximum_no_prior_only = int(policy["maximum_no_prior_only_seed_pairs"])
    maximum_neither_finite = int(
        policy.get("maximum_neither_finite_feasible_seed_pairs", 0)
    )
    minimum_semantic_higher_topologies = int(
        policy.get(
            "minimum_semantic_prior_higher_finite_feasibility_topologies", 1
        )
    )
    maximum_no_prior_higher_topologies = int(
        policy.get("maximum_no_prior_higher_finite_feasibility_topologies", 0)
    )
    maximum_p90_regret = float(policy["maximum_topology_p90_regret"])
    require_bootstrap = bool(
        policy["require_bootstrap_mean_ci_upper_below_zero"]
    )
    bootstrap_upper = None if bootstrap_ci is None else float(bootstrap_ci["upper"])
    no_prior_only_passed = (
        int(seed_discordance["no_prior_only"]) <= maximum_no_prior_only
    )
    neither_finite_passed = (
        int(seed_discordance["neither_finite_feasible"])
        <= maximum_neither_finite
    )
    common_criteria = {
        "panel_execution_complete": {
            "observed": panel_execution_complete,
            "required": True,
            "passed": panel_execution_complete is True,
        },
        "no_prior_only_seed_pairs": {
            "observed": int(seed_discordance["no_prior_only"]),
            "required_maximum": maximum_no_prior_only,
            "passed": no_prior_only_passed,
        },
        "neither_finite_feasible_seed_pairs": {
            "observed": int(seed_discordance["neither_finite_feasible"]),
            "required_maximum": maximum_neither_finite,
            "passed": neither_finite_passed,
        },
    }
    loss_criteria = {
        "complete_uncensored_panel": {
            "observed": inference_ready,
            "required": bool(policy["require_complete_uncensored_panel"]),
            "passed": inference_ready is True,
        },
        "semantic_prior_topology_wins": {
            "observed": wins_ties_losses["semantic_prior_wins"],
            "required_minimum": required_wins,
            "passed": wins_ties_losses["semantic_prior_wins"] >= required_wins,
        },
        "median_loss_reduction": {
            "observed_treatment_minus_control": median_difference,
            "required_maximum": -minimum_reduction,
            "passed": (
                median_difference is not None
                and median_difference <= -minimum_reduction
            ),
        },
        "topology_p90_regret": {
            "observed": p90_regret,
            "required_maximum": maximum_p90_regret,
            "passed": p90_regret is not None and p90_regret <= maximum_p90_regret,
        },
        "bootstrap_mean_ci_upper": {
            "observed": bootstrap_upper,
            "required_below": 0.0 if require_bootstrap else None,
            "passed": (
                bootstrap_upper is not None and bootstrap_upper < 0.0
                if require_bootstrap
                else True
            ),
        },
    }
    loss_route_passed = (
        panel_execution_complete is True
        and no_prior_only_passed
        and all(bool(item["passed"]) for item in loss_criteria.values())
    )

    semantic_higher_topologies = int(
        topology_finite_feasibility_discordance[
            "semantic_prior_higher_seed_finite_feasibility_rate"
        ]
    )
    no_prior_higher_topologies = int(
        topology_finite_feasibility_discordance[
            "no_prior_higher_seed_finite_feasibility_rate"
        ]
    )
    feasibility_criteria = {
        "semantic_prior_strictly_higher_finite_feasibility_topologies": {
            "observed": semantic_higher_topologies,
            "required_minimum": minimum_semantic_higher_topologies,
            "passed": semantic_higher_topologies
            >= minimum_semantic_higher_topologies,
        },
        "no_prior_higher_finite_feasibility_topologies": {
            "observed": no_prior_higher_topologies,
            "required_maximum": maximum_no_prior_higher_topologies,
            "passed": no_prior_higher_topologies
            <= maximum_no_prior_higher_topologies,
        },
        "no_prior_only_seed_pairs": common_criteria[
            "no_prior_only_seed_pairs"
        ],
        "neither_finite_feasible_seed_pairs": common_criteria[
            "neither_finite_feasible_seed_pairs"
        ],
        "observed_topology_p90_regret_guard": {
            "observed": observed_p90_regret_guard,
            "required_maximum_when_observed": maximum_p90_regret,
            "passed": observed_p90_regret_guard is None
            or observed_p90_regret_guard <= maximum_p90_regret,
        },
    }
    feasibility_route_passed = (
        panel_execution_complete is True
        and all(bool(item["passed"]) for item in feasibility_criteria.values())
    )

    evaluable = panel_execution_complete is True
    passed = evaluable and (feasibility_route_passed or loss_route_passed)
    selected_route = None
    if feasibility_route_passed:
        selected_route = "finite_feasibility_dominance"
    elif loss_route_passed:
        selected_route = "paired_loss"
    if not evaluable:
        action = "collect_complete_predeclared_panel"
        status = "not_evaluable"
    elif passed:
        action = str(policy["action_if_passed"])
        status = "passed"
    else:
        action = str(policy["action_if_failed"])
        status = "failed"
    return {
        "policy": copy.deepcopy(policy),
        "status": status,
        "action": action,
        "all_criteria_passed": passed,
        "selected_route": selected_route,
        "criteria": common_criteria,
        "routes": {
            "finite_feasibility_dominance": {
                "passed": feasibility_route_passed,
                "criteria": feasibility_criteria,
            },
            "paired_loss": {
                "passed": loss_route_passed,
                "criteria": loss_criteria,
            },
        },
        "note": (
            "Finite-feasibility dominance is lexicographically primary, requires "
            "no reverse seed or topology disadvantage, and retains the observed "
            "upper-tail regret guard. Otherwise the complete paired-loss route "
            "applies. This deterministic rule was bound into the plan before "
            "execution; do not alter it in response to study outcomes."
        ),
    }
def _target_hitting_summary(
    seed_pair_rows: list[dict[str, object]],
    expected_by_topology: dict[str, set[str]],
    expected_configs: dict[str, dict[str, object]] | None,
    compute_bootstrap: bool,
) -> dict[str, object]:
    target_keys = _planned_target_keys(seed_pair_rows, expected_configs)
    summaries = {}
    for target_key in target_keys:
        topology_rows = _collapse_target_by_topology(
            seed_pair_rows, expected_by_topology, target_key
        )
        complete = [row for row in topology_rows if row["inference_complete"]]
        time_values = [
            float(row["mean_seed_log10_time_ratio_semantic_over_no_prior"])
            for row in complete
        ]
        eval_values = [
            float(row["mean_seed_log10_eval_ratio_semantic_over_no_prior"])
            for row in complete
        ]
        time_ci = (
            _topology_bootstrap_ci(
                time_values, "topology-level mean log10 time ratios"
            )
            if compute_bootstrap
            else None
        )
        eval_ci = (
            _topology_bootstrap_ci(
                eval_values, "topology-level mean log10 evaluation ratios"
            )
            if compute_bootstrap
            else None
        )
        inference_ready = (
            bool(expected_by_topology)
            and len(topology_rows) == len(expected_by_topology)
            and all(row["inference_complete"] for row in topology_rows)
            if expected_configs is not None
            else None
        )
        order_of_magnitude_ready = (
            inference_ready is True
            and time_ci is not None
            and eval_ci is not None
            and float(time_ci["upper"]) <= -1.0
            and float(eval_ci["upper"]) <= -1.0
        )
        outcome_counts = {name: 0 for name in (
            "both_reached",
            "semantic_prior_only",
            "no_prior_only",
            "neither_reached",
        )}
        for seed_row in seed_pair_rows:
            target = seed_row["target_hitting"].get(target_key)
            if target is not None:
                outcome_counts[str(target["outcome"])] += 1
        summaries[target_key] = {
            "seed_pair_outcomes": outcome_counts,
            "topology_inference_ready": inference_ready,
            "finite_comparable_topologies": len(complete),
            "censored_topologies": [
                row["configured_topology"]
                for row in topology_rows
                if not row["inference_complete"]
            ],
            "right_censored_upper_bound_topologies": [
                row["configured_topology"]
                for row in topology_rows
                if row["uses_right_censored_upper_bound"]
            ],
            "topology_macro_mean_log10_time_ratio": (
                mean(time_values) if time_values else None
            ),
            "topology_macro_mean_log10_eval_ratio": (
                mean(eval_values) if eval_values else None
            ),
            "topology_bootstrap_time_log10_ratio_ci_95": time_ci,
            "topology_bootstrap_eval_log10_ratio_ci_95": eval_ci,
            "order_of_magnitude_claim_ready": order_of_magnitude_ready,
            "topology_rows": topology_rows,
        }
    return {
        "targets": summaries,
        "order_of_magnitude_rule": (
            "A threshold supports a ten-times-faster statement only when every "
            "predeclared topology and seed pair supplies either two observed hits "
            "or a conservative semantic-hit/no-prior-right-censor upper bound, and "
            "the upper 95% topology-bootstrap bounds for both log10(time ratio) "
            "and log10(evaluation ratio) are at most -1. No-prior-only and "
            "neither-reached pairs remain censored and cannot pass."
        ),
    }


def _planned_target_keys(
    seed_pair_rows: list[dict[str, object]],
    expected_configs: dict[str, dict[str, object]] | None,
) -> list[str]:
    target_sets = {
        tuple(sorted(str(key) for key in row["target_hitting"]))
        for row in seed_pair_rows
    }
    if expected_configs is not None:
        target_sets.update(
            tuple(
                sorted(
                    format(float(target), ".12g")
                    for target in config.get("target_losses", [])
                )
            )
            for config in expected_configs.values()
            if config.get("arm") in {"no_prior", "semantic_prior"}
        )
    if not target_sets:
        return []
    if len(target_sets) != 1:
        raise ValueError("causal runs do not share one target-loss set")
    return list(next(iter(target_sets)))


def _collapse_target_by_topology(
    seed_pair_rows: list[dict[str, object]],
    expected_by_topology: dict[str, set[str]],
    target_key: str,
) -> list[dict[str, object]]:
    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in seed_pair_rows:
        by_topology[str(row["topology_sha256"])].append(row)

    result = []
    observed_configured_topologies = set()
    for topology_sha256, rows in sorted(by_topology.items()):
        configured_keys = {str(row["configured_topology_key"]) for row in rows}
        if len(configured_keys) != 1:
            raise ValueError(
                f"resolved topology {topology_sha256!r} combines planned topologies"
            )
        configured_key = next(iter(configured_keys))
        observed_configured_topologies.add(configured_key)
        expected_pair_ids = expected_by_topology.get(
            configured_key, {str(row["pair_id"]) for row in rows}
        )
        target_rows = [row["target_hitting"][target_key] for row in rows]
        completed_pair_ids = {str(row["pair_id"]) for row in rows}
        replication_complete = completed_pair_ids == expected_pair_ids
        ratios_available = all(
            row["log10_time_ratio_semantic_over_no_prior"] is not None
            and row["log10_eval_ratio_semantic_over_no_prior"] is not None
            for row in target_rows
        )
        inference_complete = replication_complete and ratios_available
        time_ratios = [
            float(row["log10_time_ratio_semantic_over_no_prior"])
            for row in target_rows
            if row["log10_time_ratio_semantic_over_no_prior"] is not None
        ]
        eval_ratios = [
            float(row["log10_eval_ratio_semantic_over_no_prior"])
            for row in target_rows
            if row["log10_eval_ratio_semantic_over_no_prior"] is not None
        ]
        result.append(
            {
                "configured_topology": json.loads(configured_key),
                "topology_sha256": topology_sha256,
                "replication_complete": replication_complete,
                "inference_complete": inference_complete,
                "optimizer_seed_pairs": len(rows),
                "expected_optimizer_seed_pairs": len(expected_pair_ids),
                "mean_seed_log10_time_ratio_semantic_over_no_prior": (
                    mean(time_ratios) if inference_complete else None
                ),
                "mean_seed_log10_eval_ratio_semantic_over_no_prior": (
                    mean(eval_ratios) if inference_complete else None
                ),
                "seed_pair_outcomes": [str(row["outcome"]) for row in target_rows],
                "uses_right_censored_upper_bound": any(
                    row["outcome"] == "semantic_prior_only" for row in target_rows
                ),
            }
        )
    for configured_key, expected_pair_ids in sorted(expected_by_topology.items()):
        if configured_key in observed_configured_topologies:
            continue
        result.append(
            {
                "configured_topology": json.loads(configured_key),
                "topology_sha256": None,
                "replication_complete": False,
                "inference_complete": False,
                "optimizer_seed_pairs": 0,
                "expected_optimizer_seed_pairs": len(expected_pair_ids),
                "mean_seed_log10_time_ratio_semantic_over_no_prior": None,
                "mean_seed_log10_eval_ratio_semantic_over_no_prior": None,
                "seed_pair_outcomes": [],
                "uses_right_censored_upper_bound": False,
            }
        )
    return result


def _topology_bootstrap_ci(
    values: list[float],
    estimand: str = "topology-level mean differences",
) -> dict[str, object] | None:
    if not values:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    sample_size = len(values)
    bootstrap_means = sorted(
        mean(generator.choice(values) for _ in range(sample_size))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "confidence_level": 0.95,
        "lower": _percentile(bootstrap_means, 0.025),
        "upper": _percentile(bootstrap_means, 0.975),
        "method": f"percentile bootstrap over {estimand}",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "inference_unit": "topology",
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[
        upper_index
    ] * fraction


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("paired record contains non-canonical JSON data") from error


def _required(mapping: dict[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _required_mapping(
    mapping: dict[str, object], key: str
) -> dict[str, object]:
    value = _required(mapping, key, "record")
    if not isinstance(value, dict):
        raise ValueError(f"record field {key!r} must be an object")
    return value
