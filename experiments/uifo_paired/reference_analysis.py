"""Small independent statistical replay for validated paired UIFO evidence.

This module deliberately does not import ``experiments.uifo_paired.analysis``.
It rebuilds run outcomes from history rows and implements pairing, topology
collapse, bootstrap, and the frozen decision rule directly.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from statistics import mean, median

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)


BOOTSTRAP_SEED = 20260819
BOOTSTRAP_RESAMPLES = 10_000


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    values: list[float],
    statistic,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    generator = random.Random(seed)
    size = len(values)
    estimates = [
        float(statistic([generator.choice(values) for _ in range(size)]))
        for _ in range(resamples)
    ]
    return {
        "confidence_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "method": "percentile bootstrap over complete topology blocks",
        "resamples": resamples,
        "seed": seed,
        "inference_unit": "topology",
    }


def _run_from_history(
    rows: list[dict[str, object]], targets: list[float]
) -> dict[str, object]:
    if not rows:
        raise StudyValidationError("independent replay received an empty history")
    best = math.inf
    physically_feasible = False
    target_hits = {
        format(float(target), ".12g"): {"time_seconds": None, "eval_count": None}
        for target in targets
    }
    previous_call = -1
    previous_time = -math.inf
    previous_evals = -1
    last_time = None
    last_evals = None
    for row in rows:
        call = int(row["call_index"])
        time_seconds = float(row["time_seconds"])
        eval_count = int(row["eval_count_after_call"])
        if call < previous_call or time_seconds < previous_time or eval_count < previous_evals:
            raise StudyValidationError("history chronology is not monotone")
        previous_call = call
        previous_time = time_seconds
        previous_evals = eval_count
        last_time = time_seconds
        last_evals = eval_count
        if bool(row["is_feasible"]):
            physically_feasible = True
            loss = row["loss"]
            if loss is not None:
                value = float(loss)
                if not math.isfinite(value):
                    raise StudyValidationError("finite history loss became non-finite")
                best = min(best, value)
        if math.isfinite(best):
            for target in targets:
                key = format(float(target), ".12g")
                if best <= float(target) and target_hits[key]["time_seconds"] is None:
                    target_hits[key] = {
                        "time_seconds": time_seconds,
                        "eval_count": eval_count,
                    }
    return {
        "has_feasible": physically_feasible,
        "best_feasible_loss": best if math.isfinite(best) else None,
        "targets": target_hits,
        "last_logged_time_seconds": last_time,
        "last_logged_eval_count": last_evals,
    }


def _target_pair(control: dict[str, object], treatment: dict[str, object]) -> dict[str, object]:
    control_time = control["time_seconds"]
    treatment_time = treatment["time_seconds"]
    if control_time is not None and treatment_time is not None:
        outcome = "both_reached"
        time_ratio = math.log10(float(treatment_time) / float(control_time))
        eval_ratio = math.log10(float(treatment["eval_count"]) / float(control["eval_count"]))
        uses_censor_bound = False
    elif control_time is None and treatment_time is not None:
        outcome = "semantic_prior_only"
        time_ratio = math.log10(
            float(treatment_time) / float(control["censor_time_seconds"])
        )
        eval_ratio = math.log10(
            float(treatment["eval_count"]) / float(control["censor_eval_count"])
        )
        uses_censor_bound = True
    elif control_time is not None:
        outcome = "no_prior_only"
        time_ratio = None
        eval_ratio = None
        uses_censor_bound = False
    else:
        outcome = "neither_reached"
        time_ratio = None
        eval_ratio = None
        uses_censor_bound = False
    return {
        "outcome": outcome,
        "log10_time_ratio_semantic_over_no_prior": time_ratio,
        "log10_eval_ratio_semantic_over_no_prior": eval_ratio,
        "uses_right_censored_upper_bound": uses_censor_bound,
    }


def _evaluate_frozen_decision(
    policy: dict[str, object],
    wins: int,
    median_difference: float,
    p90_regret: float,
    bootstrap_ci: dict[str, object],
    seed_feasibility: dict[str, int],
    topology_feasibility: dict[str, int],
    observed_p90_regret: float,
) -> dict[str, object]:
    panel_complete = True
    inference_ready = True
    common = {
        "panel_execution_complete": {
            "observed": panel_complete,
            "required": True,
            "passed": panel_complete,
        },
        "no_prior_only_seed_pairs": {
            "observed": seed_feasibility["no_prior_only"],
            "required_maximum": int(policy["maximum_no_prior_only_seed_pairs"]),
            "passed": seed_feasibility["no_prior_only"]
            <= int(policy["maximum_no_prior_only_seed_pairs"]),
        },
        "neither_finite_feasible_seed_pairs": {
            "observed": seed_feasibility["neither_finite_feasible"],
            "required_maximum": int(
                policy["maximum_neither_finite_feasible_seed_pairs"]
            ),
            "passed": seed_feasibility["neither_finite_feasible"]
            <= int(policy["maximum_neither_finite_feasible_seed_pairs"]),
        },
    }
    require_bootstrap = bool(policy["require_bootstrap_mean_ci_upper_below_zero"])
    paired_loss_criteria = {
        "complete_uncensored_panel": {
            "observed": inference_ready,
            "required": bool(policy["require_complete_uncensored_panel"]),
            "passed": inference_ready,
        },
        "semantic_prior_topology_wins": {
            "observed": wins,
            "required_minimum": int(policy["minimum_semantic_prior_topology_wins"]),
            "passed": wins >= int(policy["minimum_semantic_prior_topology_wins"]),
        },
        "median_loss_reduction": {
            "observed_treatment_minus_control": median_difference,
            "required_maximum": -float(
                policy["minimum_practical_median_loss_reduction"]
            ),
            "passed": median_difference
            <= -float(policy["minimum_practical_median_loss_reduction"]),
        },
        "topology_p90_regret": {
            "observed": p90_regret,
            "required_maximum": float(policy["maximum_topology_p90_regret"]),
            "passed": p90_regret <= float(policy["maximum_topology_p90_regret"]),
        },
        "bootstrap_mean_ci_upper": {
            "observed": float(bootstrap_ci["upper"]),
            "required_below": 0.0 if require_bootstrap else None,
            "passed": (
                float(bootstrap_ci["upper"]) < 0.0 if require_bootstrap else True
            ),
        },
    }
    feasibility_criteria = {
        "semantic_prior_strictly_higher_finite_feasibility_topologies": {
            "observed": topology_feasibility["semantic_prior_higher"],
            "required_minimum": int(
                policy["minimum_semantic_prior_higher_finite_feasibility_topologies"]
            ),
            "passed": topology_feasibility["semantic_prior_higher"]
            >= int(
                policy["minimum_semantic_prior_higher_finite_feasibility_topologies"]
            ),
        },
        "no_prior_higher_finite_feasibility_topologies": {
            "observed": topology_feasibility["no_prior_higher"],
            "required_maximum": int(
                policy["maximum_no_prior_higher_finite_feasibility_topologies"]
            ),
            "passed": topology_feasibility["no_prior_higher"]
            <= int(policy["maximum_no_prior_higher_finite_feasibility_topologies"]),
        },
        "no_prior_only_seed_pairs": common["no_prior_only_seed_pairs"],
        "neither_finite_feasible_seed_pairs": common[
            "neither_finite_feasible_seed_pairs"
        ],
        "observed_topology_p90_regret_guard": {
            "observed": observed_p90_regret,
            "required_maximum_when_observed": float(
                policy["maximum_topology_p90_regret"]
            ),
            "passed": observed_p90_regret
            <= float(policy["maximum_topology_p90_regret"]),
        },
    }
    paired_passed = common["no_prior_only_seed_pairs"]["passed"] and all(
        item["passed"] for item in paired_loss_criteria.values()
    )
    feasibility_passed = all(item["passed"] for item in feasibility_criteria.values())
    passed = paired_passed or feasibility_passed
    return {
        "policy": json.loads(json.dumps(policy)),
        "status": "passed" if passed else "failed",
        "action": str(
            policy["action_if_passed"] if passed else policy["action_if_failed"]
        ),
        "all_criteria_passed": passed,
        "selected_route": (
            "finite_feasibility_dominance"
            if feasibility_passed
            else "paired_loss" if paired_passed else None
        ),
        "criteria": common,
        "routes": {
            "finite_feasibility_dominance": {
                "passed": feasibility_passed,
                "criteria": feasibility_criteria,
            },
            "paired_loss": {
                "passed": paired_passed,
                "criteria": paired_loss_criteria,
            },
        },
    }


def reference_replay(study: ValidatedStudy) -> dict[str, object]:
    """Recompute the frozen study from histories using topology as n=16."""
    configuration = study.manifest["configuration"]
    targets = [float(target) for target in configuration["target_losses"]]
    run_outcomes: dict[str, dict[str, object]] = {}
    records_by_id = {str(record["run_id"]): record for record in study.records}
    for run_id, rows in study.history_rows.items():
        run_outcomes[run_id] = _run_from_history(rows, targets)
        archived = records_by_id[run_id]["metrics"]
        if run_outcomes[run_id]["best_feasible_loss"] != archived["best_feasible_loss"]:
            raise StudyValidationError(
                f"independent best feasible loss mismatch for {run_id}"
            )
        if run_outcomes[run_id]["has_feasible"] != archived["has_feasible"]:
            raise StudyValidationError(f"independent feasibility mismatch for {run_id}")

    by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in study.records:
        config = record["config"]
        pair_id = str(config["pair_id"])
        arm = str(config["arm"])
        if arm in by_pair[pair_id]:
            raise StudyValidationError(f"duplicate arm {arm} in pair {pair_id}")
        by_pair[pair_id][arm] = record
    if len(by_pair) != 32 or any(
        set(arms) != {"no_prior", "semantic_prior"} for arms in by_pair.values()
    ):
        raise StudyValidationError("independent replay found broken arm pairing")

    seed_pairs = []
    feasibility_counts = {
        "semantic_prior_only": 0,
        "both_finite_feasible": 0,
        "no_prior_only": 0,
        "neither_finite_feasible": 0,
    }
    for pair_id, arms in sorted(by_pair.items()):
        control_record = arms["no_prior"]
        treatment_record = arms["semantic_prior"]
        control = run_outcomes[str(control_record["run_id"])]
        treatment = run_outcomes[str(treatment_record["run_id"])]
        c_loss = control["best_feasible_loss"]
        t_loss = treatment["best_feasible_loss"]
        if c_loss is not None and t_loss is not None:
            outcome = "both_finite_feasible"
            difference = float(t_loss) - float(c_loss)
        elif c_loss is None and t_loss is not None:
            outcome = "semantic_prior_only"
            difference = None
        elif c_loss is not None:
            outcome = "no_prior_only"
            difference = None
        else:
            outcome = "neither_finite_feasible"
            difference = None
        feasibility_counts[outcome] += 1
        target_outcomes = {}
        for target in targets:
            key = format(target, ".12g")
            target_outcomes[key] = _target_pair(
                {
                    **control["targets"][key],
                    "censor_time_seconds": control["last_logged_time_seconds"],
                    "censor_eval_count": control["last_logged_eval_count"],
                },
                {
                    **treatment["targets"][key],
                    "censor_time_seconds": treatment["last_logged_time_seconds"],
                    "censor_eval_count": treatment["last_logged_eval_count"],
                },
            )
        seed_pairs.append(
            {
                "pair_id": pair_id,
                "topology_sha256": control_record["problem"]["topology_sha256"],
                "configured_topology": control_record["config"]["topology"],
                "optimizer_seed": int(control_record["config"]["optimizer_seed"]),
                "no_prior_finite_feasible": c_loss is not None,
                "semantic_prior_finite_feasible": t_loss is not None,
                "difference_semantic_minus_no_prior": difference,
                "target_outcomes": target_outcomes,
            }
        )

    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in seed_pairs:
        by_topology[str(pair["topology_sha256"])].append(pair)
    if len(by_topology) != 16:
        raise StudyValidationError("independent replay did not recover n=16 topologies")

    topology_rows = []
    topology_feasibility = {
        "semantic_prior_higher": 0,
        "equal": 0,
        "no_prior_higher": 0,
    }
    for topology_hash, pairs in sorted(by_topology.items()):
        pairs.sort(key=lambda row: int(row["optimizer_seed"]))
        if [pair["optimizer_seed"] for pair in pairs] != [7, 11]:
            raise StudyValidationError("independent replay found incorrect seed hierarchy")
        differences = [pair["difference_semantic_minus_no_prior"] for pair in pairs]
        if any(value is None for value in differences):
            raise StudyValidationError("independent paired-loss replay is censored")
        control_rate = sum(bool(pair["no_prior_finite_feasible"]) for pair in pairs) / 2
        treatment_rate = sum(
            bool(pair["semantic_prior_finite_feasible"]) for pair in pairs
        ) / 2
        if treatment_rate > control_rate:
            topology_feasibility["semantic_prior_higher"] += 1
        elif treatment_rate < control_rate:
            topology_feasibility["no_prior_higher"] += 1
        else:
            topology_feasibility["equal"] += 1
        topology_rows.append(
            {
                "topology_sha256": topology_hash,
                "configured_topology": pairs[0]["configured_topology"],
                "optimizer_seeds": [7, 11],
                "seed_differences_semantic_minus_no_prior": [
                    float(value) for value in differences
                ],
                "mean_seed_difference_semantic_minus_no_prior": mean(
                    float(value) for value in differences
                ),
                "no_prior_finite_feasibility_rate": control_rate,
                "semantic_prior_finite_feasibility_rate": treatment_rate,
            }
        )

    topology_values = [
        float(row["mean_seed_difference_semantic_minus_no_prior"])
        for row in topology_rows
    ]
    wins = sum(value < 0 for value in topology_values)
    ties = sum(value == 0 for value in topology_values)
    losses = sum(value > 0 for value in topology_values)
    bootstrap_mean = _bootstrap(topology_values, mean)
    mean_difference = mean(topology_values)
    median_difference = median(topology_values)
    p90_regret = _percentile(topology_values, 0.9)
    policy = configuration["decision_policy"]
    decision = _evaluate_frozen_decision(
        policy,
        wins,
        median_difference,
        p90_regret,
        bootstrap_mean,
        feasibility_counts,
        topology_feasibility,
        p90_regret,
    )

    target_summaries = {}
    for target in targets:
        key = format(target, ".12g")
        outcome_counts = {
            "both_reached": 0,
            "semantic_prior_only": 0,
            "no_prior_only": 0,
            "neither_reached": 0,
        }
        topology_target_rows = []
        for topology_hash, pairs in sorted(by_topology.items()):
            pair_targets = [pair["target_outcomes"][key] for pair in pairs]
            for row in pair_targets:
                outcome_counts[str(row["outcome"])] += 1
            complete = all(
                row["log10_time_ratio_semantic_over_no_prior"] is not None
                and row["log10_eval_ratio_semantic_over_no_prior"] is not None
                for row in pair_targets
            )
            topology_target_rows.append(
                {
                    "topology_sha256": topology_hash,
                    "inference_complete": complete,
                    "seed_pair_outcomes": [row["outcome"] for row in pair_targets],
                    "mean_seed_log10_time_ratio": (
                        mean(
                            float(row["log10_time_ratio_semantic_over_no_prior"])
                            for row in pair_targets
                        )
                        if complete
                        else None
                    ),
                    "mean_seed_log10_eval_ratio": (
                        mean(
                            float(row["log10_eval_ratio_semantic_over_no_prior"])
                            for row in pair_targets
                        )
                        if complete
                        else None
                    ),
                    "uses_right_censored_upper_bound": any(
                        bool(row["uses_right_censored_upper_bound"])
                        for row in pair_targets
                    ),
                }
            )
        complete_rows = [row for row in topology_target_rows if row["inference_complete"]]
        inference_ready = len(complete_rows) == 16
        time_ci = (
            _bootstrap(
                [float(row["mean_seed_log10_time_ratio"]) for row in complete_rows],
                mean,
            )
            if inference_ready
            else None
        )
        eval_ci = (
            _bootstrap(
                [float(row["mean_seed_log10_eval_ratio"]) for row in complete_rows],
                mean,
            )
            if inference_ready
            else None
        )
        target_summaries[key] = {
            "seed_pair_outcomes": outcome_counts,
            "topology_inference_ready": inference_ready,
            "topology_rows": topology_target_rows,
            "topology_bootstrap_time_log10_ratio_ci_95": time_ci,
            "topology_bootstrap_eval_log10_ratio_ci_95": eval_ci,
            "order_of_magnitude_claim_ready": (
                inference_ready
                and float(time_ci["upper"]) <= -1.0
                and float(eval_ci["upper"]) <= -1.0
            ),
        }

    return {
        "format_version": 1,
        "implementation": (
            "Independent history-row evaluator; no imports from "
            "experiments.uifo_paired.analysis."
        ),
        "completed_runs": len(run_outcomes),
        "complete_optimizer_seed_pairs": len(seed_pairs),
        "complete_topologies": len(topology_rows),
        "finite_feasible_seed_pairs": feasibility_counts,
        "finite_feasibility_discordance": topology_feasibility,
        "wins_ties_losses": {
            "semantic_prior_wins": wins,
            "ties": ties,
            "semantic_prior_losses": losses,
        },
        "topology_mean_difference": mean_difference,
        "topology_median_difference": median_difference,
        "topology_p90_regret": p90_regret,
        "topology_bootstrap_mean_difference_ci_95": bootstrap_mean,
        "predeclared_decision": decision,
        "topology_rows": topology_rows,
        "seed_pair_rows": seed_pairs,
        "target_hitting": target_summaries,
    }
