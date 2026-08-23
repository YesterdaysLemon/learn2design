"""Production summary for the frozen no-prior submission-like screen."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from statistics import mean, median


PROFILE = "submission-like-screen-v1"
ARM = "no_prior"
EXPECTED_SEEDS = {29, 31}
EXPECTED_TOPOLOGIES = 10
EXPECTED_RUNS = 20
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_RESAMPLES = 10_000
TARGET_LOSSES = (4.0, 1.0, 0.5, 0.0)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(values: list[float]) -> list[float]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = [
        mean(generator.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _common_policy(
    expected_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    policies = [config.get("decision_policy") for config in expected_configs.values()]
    if not policies or not all(isinstance(policy, dict) for policy in policies):
        raise ValueError("submission-like study is missing its decision policy")
    first = policies[0]
    assert isinstance(first, dict)
    if any(policy != first for policy in policies[1:]):
        raise ValueError("submission-like configurations disagree on decision policy")
    return first


def _target_hitting(topology_rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in TARGET_LOSSES:
        key = format(target, ".12g")
        run_counts = {"reached": 0, "right_censored": 0, "incomplete": 0}
        topology_counts = {
            "both_seeds_reached": 0,
            "one_seed_reached": 0,
            "neither_seed_reached": 0,
            "incomplete": 0,
        }
        target_rows = []
        for topology in topology_rows:
            seed_hits: dict[str, object] = {}
            reached = 0
            complete = bool(topology["replication_complete"])
            for row in topology["seed_rows"]:
                targets = row.get("target_hits")
                hit = targets.get(key) if isinstance(targets, dict) else None
                seed_hits[str(row["optimizer_seed"])] = hit
                if not isinstance(hit, dict):
                    run_counts["incomplete"] += 1
                    complete = False
                elif hit.get("time_seconds") is None:
                    run_counts["right_censored"] += 1
                else:
                    run_counts["reached"] += 1
                    reached += 1
            if not complete:
                category = "incomplete"
            elif reached == 2:
                category = "both_seeds_reached"
            elif reached == 1:
                category = "one_seed_reached"
            else:
                category = "neither_seed_reached"
            topology_counts[category] += 1
            target_rows.append(
                {
                    "topology": topology["configured_topology"]["value"],
                    "category": category,
                    "seed_hits": seed_hits,
                }
            )
        result[key] = {
            "target_loss": target,
            "run_categories": run_counts,
            "topology_categories": topology_counts,
            "topology_rows": target_rows,
            "censoring_preserved": True,
        }
    return result


def summarize_submission_like_records(
    records: list[dict[str, object]],
    expected_configs: dict[str, dict[str, object]],
    *,
    compute_bootstrap: bool = True,
) -> dict[str, object]:
    """Summarize one no-prior arm with topology as the inference unit."""
    policy = _common_policy(expected_configs)
    if len(expected_configs) != EXPECTED_RUNS:
        raise ValueError("submission-like profile requires exactly 20 planned runs")
    if {
        str(config.get("study_profile")) for config in expected_configs.values()
    } != {PROFILE}:
        raise ValueError("submission-like study profile mismatch")
    if {str(config.get("arm")) for config in expected_configs.values()} != {ARM}:
        raise ValueError("submission-like profile requires only no_prior")
    package_evidence = {
        _canonical(config.get("candidate_package_evidence"))
        for config in expected_configs.values()
    }
    candidate_package_bound = len(package_evidence) == 1 and all(
        isinstance(config.get("candidate_package_evidence"), dict)
        for config in expected_configs.values()
    )

    by_id: dict[str, dict[str, object]] = {}
    errors = 0
    interrupted = 0
    for record in records:
        run_id = str(record.get("run_id"))
        if run_id in by_id:
            raise ValueError(f"duplicate submission-like run ID: {run_id}")
        if run_id not in expected_configs:
            raise ValueError(f"unexpected submission-like run ID: {run_id}")
        if record.get("config") != expected_configs[run_id]:
            raise ValueError(f"submission-like record/config mismatch: {run_id}")
        status = record.get("status")
        errors += status == "error"
        interrupted += status == "interrupted"
        if status not in {"complete", "error", "interrupted"}:
            raise ValueError(f"invalid submission-like run status: {run_id}")
        by_id[run_id] = record

    expected_by_topology: dict[str, dict[int, str]] = defaultdict(dict)
    for run_id, config in expected_configs.items():
        seed = config.get("optimizer_seed")
        if seed not in EXPECTED_SEEDS:
            raise ValueError("submission-like optimizer seed mismatch")
        key = _canonical(config.get("topology"))
        if int(seed) in expected_by_topology[key]:
            raise ValueError("duplicate topology/seed in submission-like plan")
        expected_by_topology[key][int(seed)] = run_id
    if len(expected_by_topology) != EXPECTED_TOPOLOGIES or any(
        set(rows) != EXPECTED_SEEDS for rows in expected_by_topology.values()
    ):
        raise ValueError("submission-like topology/seed hierarchy mismatch")

    topology_rows: list[dict[str, object]] = []
    topology_hashes: dict[str, str] = {}
    hash_topologies: dict[str, str] = {}
    for topology_key, seed_ids in expected_by_topology.items():
        seed_rows = []
        for seed in sorted(EXPECTED_SEEDS):
            run_id = seed_ids[seed]
            record = by_id.get(run_id)
            if record is None or record.get("status") != "complete":
                continue
            metrics = record.get("metrics")
            problem = record.get("problem")
            if not isinstance(metrics, dict) or not isinstance(problem, dict):
                raise ValueError(f"submission-like evidence is incomplete: {run_id}")
            has_physical = metrics.get("has_feasible")
            has_finite = metrics.get("has_finite_feasible")
            loss = metrics.get("best_feasible_loss")
            if type(has_physical) is not bool:
                raise ValueError(f"invalid physical-feasibility flag: {run_id}")
            if type(has_finite) is not bool:
                raise ValueError(f"invalid finite-feasibility flag: {run_id}")
            if has_finite and not has_physical:
                raise ValueError(f"finite feasibility lacks physical feasibility: {run_id}")
            if has_finite:
                if (
                    isinstance(loss, bool)
                    or not isinstance(loss, (int, float))
                    or not math.isfinite(float(loss))
                ):
                    raise ValueError(f"invalid finite best loss: {run_id}")
                loss = float(loss)
            elif loss is not None:
                raise ValueError(f"censored run has a finite score: {run_id}")
            topology_hash = str(problem.get("topology_sha256"))
            if len(topology_hash) != 64:
                raise ValueError(f"invalid topology hash: {run_id}")
            if topology_hashes.setdefault(topology_key, topology_hash) != topology_hash:
                raise ValueError("one topology resolved to multiple identities")
            if hash_topologies.setdefault(topology_hash, topology_key) != topology_key:
                raise ValueError("distinct topologies resolved to one identity")
            seed_rows.append(
                {
                    "optimizer_seed": seed,
                    "run_id": run_id,
                    "physical_feasible": has_physical,
                    "finite_feasible": has_finite,
                    "best_feasible_loss": loss,
                    "planned_run_index": int(record["config"]["planned_run_index"]),
                    "target_hits": metrics.get("targets"),
                }
            )
        complete = len(seed_rows) == 2
        losses = [
            float(row["best_feasible_loss"])
            for row in seed_rows
            if row["finite_feasible"]
        ]
        inference_complete = complete and len(losses) == 2
        topology_rows.append(
            {
                "configured_topology": json.loads(topology_key),
                "topology_sha256": topology_hashes.get(topology_key),
                "optimizer_seeds": [int(row["optimizer_seed"]) for row in seed_rows],
                "replication_complete": complete,
                "physically_feasible_seeds": sum(
                    bool(row["physical_feasible"]) for row in seed_rows
                ),
                "finite_feasible_seeds": len(losses),
                "inference_complete": inference_complete,
                "topology_mean_best_feasible_loss": (
                    mean(losses) if inference_complete else None
                ),
                "absolute_seed_gap": (
                    abs(losses[0] - losses[1]) if inference_complete else None
                ),
                "seed_rows": seed_rows,
            }
        )

    complete_records = [
        record for record in records if record.get("status") == "complete"
    ]
    panel_complete = (
        len(records) == EXPECTED_RUNS
        and len(complete_records) == EXPECTED_RUNS
        and errors == 0
        and interrupted == 0
        and all(row["replication_complete"] for row in topology_rows)
    )
    topology_values = [
        float(row["topology_mean_best_feasible_loss"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    seed_gaps = [
        float(row["absolute_seed_gap"])
        for row in topology_rows
        if row["inference_complete"]
    ]
    inference_ready = panel_complete and len(topology_values) == EXPECTED_TOPOLOGIES
    p90_seed_gap = _percentile(seed_gaps, 0.9) if inference_ready else None
    criteria = {
        "panel_execution_complete": panel_complete,
        "complete_records_revalidated": bool(compute_bootstrap),
        "complete_topology_blocks": panel_complete,
        "all_runs_finite_feasible": inference_ready,
        "candidate_package_bound": candidate_package_bound,
    }

    status = "pending"
    action = None
    passed = False
    if errors or interrupted:
        status = "not_evaluable"
        action = str(policy["action_if_not_evaluable"])
    elif panel_complete and compute_bootstrap:
        passed = all(criteria.values())
        status = "passed" if passed else "failed"
        action = str(policy["action_if_passed"] if passed else policy["action_if_failed"])

    return {
        "format_version": 1,
        "study_profile": PROFILE,
        "completed_runs": len(complete_records),
        "error_runs": errors,
        "interrupted_runs": interrupted,
        "complete_topologies": sum(
            bool(row["replication_complete"]) for row in topology_rows
        ),
        "physical_feasible_runs": sum(
            int(row["physically_feasible_seeds"]) for row in topology_rows
        ),
        "finite_feasible_runs": sum(
            int(row["finite_feasible_seeds"]) for row in topology_rows
        ),
        "finite_feasible_topologies": len(topology_values),
        "topology_arithmetic_mean_best_feasible_loss": (
            mean(topology_values) if inference_ready else None
        ),
        "topology_median_best_feasible_loss": (
            median(topology_values) if inference_ready else None
        ),
        "topology_p90_best_feasible_loss": (
            _percentile(topology_values, 0.9) if inference_ready else None
        ),
        "topology_p90_absolute_seed_gap": p90_seed_gap,
        "topology_bootstrap_mean_loss_ci_95": (
            _bootstrap_mean_ci(topology_values)
            if inference_ready and compute_bootstrap
            else None
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "predeclared_decision": {
            "status": status,
            "passed": passed,
            "action": action,
            "criteria": criteria,
        },
        "topology_rows": topology_rows,
        "target_hitting": _target_hitting(topology_rows),
        "run_ids": sorted(by_id),
        "note": (
            "No-prior-only readiness screen. Topology is n=10; optimizer seeds "
            "are repeated measurements. The decision assesses evidence completeness "
            "and seed reliability, not superiority or official-budget performance."
        ),
    }
