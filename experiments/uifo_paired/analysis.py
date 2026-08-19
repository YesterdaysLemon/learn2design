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

    paired = _paired_prior_summary(by_pair, expected_configs)
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
                "difference_treatment_minus_control": difference,
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
        "promotion_inference_ready": (
            not incomplete_topologies
            and not censored_topologies
            and bool(expected_by_topology)
            if expected_configs is not None
            else None
        ),
        "finite_comparable_topologies": len(macro_values),
        "wins_ties_losses": topology_wins_ties_losses,
        "physical_feasibility_discordance": topology_feasibility_discordance,
        "topology_macro_mean_difference": mean(macro_values) if macro_values else None,
        "topology_macro_median_difference": (
            median(macro_values) if macro_values else None
        ),
        "topology_bootstrap_mean_difference_ci_95": _topology_bootstrap_ci(
            macro_values
        ),
        "topology_differences": topology_differences,
        "optimizer_seed_pair_diagnostics": seed_pair_diagnostics,
        "note": (
            "Primary wins, ties, losses, feasibility discordance, and confidence "
            "intervals use topology as the inference unit after collapsing optimizer "
            "seeds. Primary loss aggregates require every predeclared seed pair to "
            "be present and finite-feasible in both arms. Inspect incomplete and "
            "censored topology lists, feasibility rates, and seed diagnostics."
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
    )
    for field in paired_config_fields:
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
    return {
        "complete_optimizer_seed_pairs": len(seed_pair_rows),
        "finite_comparable_optimizer_seed_pairs": len(finite_rows),
        "optimizer_seed_pair_wins_ties_losses": {
            "semantic_prior_wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "semantic_prior_losses": sum(value > 0 for value in differences),
        },
        "optimizer_seed_pair_physical_feasibility_discordance": discordance,
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


def _topology_bootstrap_ci(values: list[float]) -> dict[str, object] | None:
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
        "method": "percentile bootstrap over topology-level mean differences",
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
