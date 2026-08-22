"""Small independent reference calculation for ``restart-screen-v1``.

This module intentionally does not import production aggregation, decision, or
metrics helpers.  It scans authenticated candidate histories, pairs arms, and
collapses optimizer seeds within topology directly.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from statistics import mean, median

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)


CONTROL_ARM = "no_prior_p600"
TREATMENT_ARM = "no_prior_p200"
EXPECTED_SEEDS = {19, 23}
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_RESAMPLES = 10_000
TIE_TOLERANCE = 1e-12

FROZEN_POLICY = {
    "policy_id": "patience-200-development-screen-v1",
    "action_if_passed": "plan_untouched_submission_like_gate",
    "action_if_failed": "retain_patience_600",
    "maximum_topology_median_difference": -0.05,
    "maximum_topology_p90_regret": 0.5,
    "minimum_patience_200_topology_wins": 6,
    "require_all_pairs_finite_comparable": True,
    "require_both_seed_mean_differences_below_zero": True,
    "require_no_control_only_finite_feasible_pairs": True,
    "require_no_topology_lower_treatment_feasibility": True,
    "require_topology_mean_difference_below_zero": True,
    "inference_unit": "topology",
    "optimizer_seeds_are_repeated_measurements": True,
    "stage": "optimizer_development_screen",
    "study_profile": "restart-screen-v1",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise StudyValidationError("reference percentile requires observations")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_mean_ci(values: list[float]) -> list[float]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(mean(generator.choice(values) for _ in values))
    samples.sort()
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _exact_sign_flip_mean_pvalue(values: list[float]) -> float:
    observed = abs(mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def _exact_sign_test(wins: int, losses: int) -> float | None:
    nonzero = wins + losses
    if nonzero == 0:
        return None
    tail = min(wins, losses)
    numerator = sum(math.comb(nonzero, index) for index in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**nonzero))


def _history_outcome(
    run_id: str, rows: list[dict[str, object]], record: dict[str, object]
) -> dict[str, object]:
    if not rows:
        raise StudyValidationError(f"reference history is empty: {run_id}")
    physical = False
    finite_losses: list[float] = []
    for row in rows:
        feasible = row.get("is_feasible")
        if type(feasible) is not bool:
            raise StudyValidationError(
                f"reference feasibility is not strict boolean: {run_id}"
            )
        if feasible is not True:
            continue
        physical = True
        loss = row.get("loss")
        if loss is None:
            continue
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise StudyValidationError(f"reference loss is not numeric: {run_id}")
        if not math.isfinite(float(loss)):
            raise StudyValidationError(f"reference history contains nonfinite loss: {run_id}")
        finite_losses.append(float(loss))
    finite = bool(finite_losses)
    best = min(finite_losses) if finite else None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise StudyValidationError(f"reference record metrics are missing: {run_id}")
    if metrics.get("has_feasible") is not physical:
        raise StudyValidationError(f"history/record physical feasibility mismatch: {run_id}")
    if metrics.get("has_finite_feasible") is not finite:
        raise StudyValidationError(f"history/record finite feasibility mismatch: {run_id}")
    if metrics.get("best_feasible_loss") != best:
        raise StudyValidationError(f"history/record best loss mismatch: {run_id}")
    return {
        "physical_feasible": physical,
        "finite_feasible": finite,
        "best_feasible_loss": best,
        "eval_count": max(int(row["eval_count_after_call"]) for row in rows),
    }


def _assert_pair(
    pair_id: str,
    control: dict[str, object],
    treatment: dict[str, object],
) -> None:
    control_config = control["config"]
    treatment_config = treatment["config"]
    if not isinstance(control_config, dict) or not isinstance(treatment_config, dict):
        raise StudyValidationError(f"reference pair config missing: {pair_id}")
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
            raise StudyValidationError(f"reference pair disagrees on {field}: {pair_id}")
    if control.get("problem") != treatment.get("problem"):
        raise StudyValidationError(f"reference pair problem mismatch: {pair_id}")
    control_settings = control_config.get("optimizer_settings")
    treatment_settings = treatment_config.get("optimizer_settings")
    if not isinstance(control_settings, dict) or not isinstance(treatment_settings, dict):
        raise StudyValidationError(f"reference optimizer settings missing: {pair_id}")
    if set(control_settings) != set(treatment_settings) or "patience" not in control_settings:
        raise StudyValidationError(f"reference optimizer schema mismatch: {pair_id}")
    differing = {
        key
        for key in control_settings
        if control_settings[key] != treatment_settings[key]
    }
    if (
        differing != {"patience"}
        or control_settings["patience"] != 600
        or treatment_settings["patience"] != 200
    ):
        raise StudyValidationError(f"reference patience contrast mismatch: {pair_id}")


def reference_restart_screen(
    study: ValidatedStudy, *, include_exploratory: bool = True
) -> dict[str, object]:
    """Recompute the restart-screen result using histories as primary input."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("reference evaluator requires a sealed summary")
    if (
        set(study.configs) != {str(record.get("run_id")) for record in study.records}
        or set(study.configs) != set(study.history_rows)
        or len(study.configs) != 32
    ):
        raise StudyValidationError("reference run/config/history hierarchy mismatch")
    complete_count = sum(record.get("status") == "complete" for record in study.records)
    if complete_count != 32:
        raise StudyValidationError("reference requires exactly 32 complete runs")

    policies = {
        _canonical(config.get("decision_policy")) for config in study.configs.values()
    }
    if policies != {_canonical(FROZEN_POLICY)}:
        raise StudyValidationError("reference frozen policy mismatch")
    if {
        str(config.get("study_profile")) for config in study.configs.values()
    } != {"restart-screen-v1"}:
        raise StudyValidationError("reference study profile mismatch")

    by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    outcomes: dict[str, dict[str, object]] = {}
    for record in study.records:
        run_id = str(record.get("run_id"))
        config = record.get("config")
        if not isinstance(config, dict) or config != study.configs.get(run_id):
            raise StudyValidationError(f"reference record/config mismatch: {run_id}")
        seed = config.get("optimizer_seed")
        arm = config.get("arm")
        if seed not in EXPECTED_SEEDS or arm not in {CONTROL_ARM, TREATMENT_ARM}:
            raise StudyValidationError(f"reference seed or arm mismatch: {run_id}")
        pair_id = str(config.get("pair_id"))
        if str(arm) in by_pair[pair_id]:
            raise StudyValidationError(f"reference duplicate arm in pair: {pair_id}")
        by_pair[pair_id][str(arm)] = record
        outcomes[run_id] = _history_outcome(
            run_id, study.history_rows[run_id], record
        )

    if len(by_pair) != 16:
        raise StudyValidationError("reference requires exactly 16 seed pairs")
    pair_rows: list[dict[str, object]] = []
    topology_hash_to_config: dict[str, str] = {}
    topology_config_to_hash: dict[str, str] = {}
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        if set(arms) != {CONTROL_ARM, TREATMENT_ARM}:
            raise StudyValidationError(f"reference incomplete arm pair: {pair_id}")
        control = arms[CONTROL_ARM]
        treatment = arms[TREATMENT_ARM]
        _assert_pair(pair_id, control, treatment)
        control_id = str(control["run_id"])
        treatment_id = str(treatment["run_id"])
        control_outcome = outcomes[control_id]
        treatment_outcome = outcomes[treatment_id]
        control_finite = bool(control_outcome["finite_feasible"])
        treatment_finite = bool(treatment_outcome["finite_feasible"])
        difference = None
        if control_finite and treatment_finite:
            difference = float(treatment_outcome["best_feasible_loss"]) - float(
                control_outcome["best_feasible_loss"]
            )
        control_config = control["config"]
        assert isinstance(control_config, dict)
        topology_key = _canonical(control_config["topology"])
        topology_hash = str(control["problem"]["topology_sha256"])
        if topology_hash_to_config.setdefault(topology_hash, topology_key) != topology_key:
            raise StudyValidationError("reference topology hash collision")
        if topology_config_to_hash.setdefault(topology_key, topology_hash) != topology_hash:
            raise StudyValidationError("reference topology resolution mismatch")
        pair_rows.append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(control_config["optimizer_seed"]),
                "configured_topology": control_config["topology"],
                "topology_sha256": topology_hash,
                "control_finite_feasible": control_finite,
                "treatment_finite_feasible": treatment_finite,
                "difference_p200_minus_p600": difference,
                "p200_first": int(treatment["config"]["run_order_within_pair"])
                < int(control["config"]["run_order_within_pair"]),
                "control_eval_count": int(control_outcome["eval_count"]),
                "treatment_eval_count": int(treatment_outcome["eval_count"]),
            }
        )

    if len(topology_hash_to_config) != 8 or len(topology_config_to_hash) != 8:
        raise StudyValidationError("reference requires exactly 8 topology identities")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[_canonical(row["configured_topology"])].append(row)

    topology_rows: list[dict[str, object]] = []
    for topology_key in sorted(grouped):
        rows = sorted(grouped[topology_key], key=lambda row: int(row["optimizer_seed"]))
        if {int(row["optimizer_seed"]) for row in rows} != EXPECTED_SEEDS or len(rows) != 2:
            raise StudyValidationError("reference topology seed replication mismatch")
        differences = [
            float(row["difference_p200_minus_p600"])
            for row in rows
            if row["difference_p200_minus_p600"] is not None
        ]
        topology_rows.append(
            {
                "configured_topology": rows[0]["configured_topology"],
                "topology_sha256": rows[0]["topology_sha256"],
                "optimizer_seeds": [19, 23],
                "replication_complete": True,
                "inference_complete": len(differences) == 2,
                "control_finite_feasible_seeds": sum(
                    bool(row["control_finite_feasible"]) for row in rows
                ),
                "treatment_finite_feasible_seeds": sum(
                    bool(row["treatment_finite_feasible"]) for row in rows
                ),
                "mean_seed_difference_p200_minus_p600": (
                    mean(differences) if len(differences) == 2 else None
                ),
                "seed_pair_rows": rows,
            }
        )

    topology_values = [
        float(row["mean_seed_difference_p200_minus_p600"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    inference_ready = len(topology_values) == 8
    topology_mean = mean(topology_values) if inference_ready else None
    topology_median = median(topology_values) if inference_ready else None
    topology_p90 = (
        _percentile(sorted(topology_values), 0.9) if inference_ready else None
    )
    wins = sum(value < -TIE_TOLERANCE for value in topology_values)
    ties = sum(abs(value) <= TIE_TOLERANCE for value in topology_values)
    losses = sum(value > TIE_TOLERANCE for value in topology_values)
    control_only = sum(
        bool(row["control_finite_feasible"])
        and not bool(row["treatment_finite_feasible"])
        for row in pair_rows
    )
    treatment_only = sum(
        bool(row["treatment_finite_feasible"])
        and not bool(row["control_finite_feasible"])
        for row in pair_rows
    )
    neither = sum(
        not bool(row["treatment_finite_feasible"])
        and not bool(row["control_finite_feasible"])
        for row in pair_rows
    )
    both = len(pair_rows) - control_only - treatment_only - neither
    lower_treatment = sum(
        int(row["treatment_finite_feasible_seeds"])
        < int(row["control_finite_feasible_seeds"])
        for row in topology_rows
    )
    seed_means: dict[str, float | None] = {}
    for seed in sorted(EXPECTED_SEEDS):
        values = [
            float(row["difference_p200_minus_p600"])
            for row in pair_rows
            if row["optimizer_seed"] == seed
            and row["difference_p200_minus_p600"] is not None
        ]
        seed_means[str(seed)] = mean(values) if len(values) == 8 else None

    criteria = {
        "panel_execution_complete": len(study.records) == 32,
        "complete_records_revalidated": True,
        "inference_ready": inference_ready,
        "all_pairs_finite_comparable": both == 16,
        "zero_control_only_finite_feasible_pairs": control_only == 0,
        "zero_topologies_with_lower_treatment_feasibility": lower_treatment == 0,
        "minimum_topology_wins_met": wins >= 6,
        "median_difference_at_most_negative_0_05": bool(
            topology_median is not None and topology_median <= -0.05
        ),
        "mean_difference_below_zero": bool(
            topology_mean is not None and topology_mean < 0
        ),
        "p90_regret_at_most_0_5": bool(
            topology_p90 is not None and topology_p90 <= 0.5
        ),
        "both_seed_mean_differences_below_zero": bool(
            set(seed_means) == {"19", "23"}
            and all(value is not None and value < 0 for value in seed_means.values())
        ),
    }
    passed = all(criteria.values())
    decision = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "action": (
            FROZEN_POLICY["action_if_passed"]
            if passed
            else FROZEN_POLICY["action_if_failed"]
        ),
        "criteria": criteria,
    }
    return {
        "format_version": 1,
        "study_profile": "restart-screen-v1",
        "completed_runs": complete_count,
        "error_runs": 0,
        "interrupted_runs": 0,
        "complete_optimizer_seed_pairs": len(pair_rows),
        "finite_comparable_optimizer_seed_pairs": both,
        "complete_topologies": len(topology_rows),
        "finite_comparable_topologies": len(topology_values),
        "wins_ties_losses": {
            "p200_wins": wins,
            "ties": ties,
            "p200_losses": losses,
        },
        "feasibility_pair_outcomes": {
            "both_finite_feasible": both,
            "p200_only_finite_feasible": treatment_only,
            "p600_only_finite_feasible": control_only,
            "neither_finite_feasible": neither,
        },
        "control_only_finite_feasible_pairs": control_only,
        "topologies_with_lower_treatment_feasibility": lower_treatment,
        "topology_macro_mean_difference": topology_mean,
        "topology_macro_median_difference": topology_median,
        "topology_p90_regret": topology_p90,
        "optimizer_seed_mean_differences": seed_means,
        "topology_bootstrap_mean_difference_ci_95": (
            _bootstrap_mean_ci(topology_values) if inference_ready else None
        ),
        "exact_sign_flip_mean_pvalue_two_sided": (
            _exact_sign_flip_mean_pvalue(topology_values)
            if inference_ready and include_exploratory
            else None
        ),
        "exact_sign_flip_assignments": 2**8,
        "exact_sign_test_pvalue_two_sided": (
            _exact_sign_test(wins, losses)
            if inference_ready and include_exploratory
            else None
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "predeclared_decision": decision,
        "topology_differences": topology_rows,
        "optimizer_seed_pair_rows": pair_rows,
        "note": (
            "Independent history-only replay. Topology is n=8; seeds are repeated "
            "measurements. The 1e-12 tie tolerance is numerical, not equivalence."
        ),
    }
