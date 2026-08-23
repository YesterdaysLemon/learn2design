"""Small history-first reference evaluator for ``submission-like-screen-v1``.

This module deliberately avoids the production analysis, aggregation, and
metrics helpers.  Authenticated normalized history rows are its computational
input; record metrics are comparison targets only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)


PROFILE_ID = "submission-like-screen-v1"
PANEL_ID = "submission-like-v1"
ARM = "no_prior"
EXPECTED_SEEDS = (29, 31)
TARGET_LOSSES = (4.0, 1.0, 0.5, 0.0)
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_RESAMPLES = 10_000
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

SUBMISSION_LIKE_TOPOLOGIES = (
    "CBCHCHDAG-SLSLSSLLLLLD",
    "BAFBCFGBC-HLLLSLLSSLLL",
    "CFGFGBECB-SSSSLDLLSLLL",
    "HCCBDFFFH-SSLHSLSSLLLL",
    "DHEHGGGCB-LDLSSSSLSSLS",
    "FBCFHEGCF-SLLSLLSSSSSH",
    "AGBFGFGAC-LLLSLSLLDLLS",
    "FBABHECAG-LLLSLLLHSLLS",
    "DHHGGEAEH-SSSSLLSSLLDL",
    "FDEFCDFGE-HLSSSLSSLSLL",
)

FROZEN_POLICY = {
    "policy_id": "no-prior-submission-like-screen-v1",
    "action_if_passed": "candidate_evidence_complete_for_submission_review",
    "action_if_failed": "retain_candidate_and_investigate_submission_like_reliability",
    "action_if_not_evaluable": "retain_candidate_attempt_not_evaluable",
    "require_all_runs_finite_feasible": True,
    "require_candidate_package_bound": True,
    "require_complete_topology_blocks": True,
    "inference_unit": "topology",
    "optimizer_seeds_are_repeated_measurements": True,
    "changes_packaged_candidate": False,
    "official_budget_claim_allowed": False,
}


def _mean(values: list[float]) -> float:
    if not values:
        raise StudyValidationError("reference mean requires observations")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _linear_percentile(values: list[float], probability: float) -> float:
    if not values:
        raise StudyValidationError("reference percentile requires observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(values: list[float]) -> dict[str, object]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(_mean(sample))
    return {
        "lower": _linear_percentile(samples, 0.025),
        "upper": _linear_percentile(samples, 0.975),
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "method": "percentile bootstrap over complete topology blocks",
    }


def _target_key(target: float) -> str:
    return format(target, ".12g")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _relative_source_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return False
    return all(part not in {"", ".", ".."} for part in normalized.split("/"))


def _basename(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
        and re.match(r"^[A-Za-z]:", value) is None
    )


def _candidate_package_evidence(
    study: ValidatedStudy, plan_configuration: dict[str, object]
) -> dict[str, object]:
    evidence = plan_configuration.get("candidate_package_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "format_version",
        "archive_name",
        "archive_sha256",
        "builder_manifest_name",
        "builder_manifest_sha256",
        "project_revision",
        "source_files",
        "upstream_reference",
    }:
        raise StudyValidationError("reference candidate package evidence schema mismatch")
    if evidence.get("format_version") != 1:
        raise StudyValidationError("reference candidate package evidence version mismatch")
    for key in ("archive_name", "builder_manifest_name"):
        if not _basename(evidence.get(key)):
            raise StudyValidationError(
                f"reference candidate package {key} is not a basename"
            )
    for key in ("archive_sha256", "builder_manifest_sha256"):
        value = evidence.get(key)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise StudyValidationError(
                f"reference candidate package {key} is invalid"
            )
    revision = evidence.get("project_revision")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise StudyValidationError(
            "reference candidate package project revision is invalid"
        )
    if revision != study.manifest.get("project_revision"):
        raise StudyValidationError(
            "reference candidate package project revision is not manifest-bound"
        )
    upstream = evidence.get("upstream_reference")
    if not isinstance(upstream, str) or not upstream:
        raise StudyValidationError(
            "reference candidate package upstream reference is invalid"
        )
    source_files = evidence.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise StudyValidationError("reference candidate package source files are missing")
    observed_paths: set[str] = set()
    for item in source_files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise StudyValidationError(
                "reference candidate package source-file schema mismatch"
            )
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if not _relative_source_path(path):
            raise StudyValidationError(
                "reference candidate package source path is unsafe"
            )
        if path in observed_paths:
            raise StudyValidationError(
                "reference candidate package contains duplicate source paths"
            )
        observed_paths.add(str(path))
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise StudyValidationError(
                "reference candidate package source digest is invalid"
            )
        if type(size) is not int or size < 0:
            raise StudyValidationError(
                "reference candidate package source size is invalid"
            )
    return evidence


def _configuration(study: ValidatedStudy) -> dict[str, object]:
    plan_configuration = study.plan.get("configuration")
    manifest_configuration = study.manifest.get("configuration")
    if not isinstance(plan_configuration, dict) or not isinstance(
        manifest_configuration, dict
    ):
        raise StudyValidationError("reference study configuration is missing")
    if plan_configuration != manifest_configuration:
        raise StudyValidationError("reference plan/manifest configuration mismatch")
    if plan_configuration.get("study_profile") != PROFILE_ID:
        raise StudyValidationError("reference study profile mismatch")
    if plan_configuration.get("arms") != [ARM]:
        raise StudyValidationError("reference study must contain exactly the no_prior arm")
    if plan_configuration.get("optimizer_seeds") != list(EXPECTED_SEEDS):
        raise StudyValidationError("reference optimizer seeds are not exactly 29/31")
    if plan_configuration.get("target_losses") != list(TARGET_LOSSES):
        raise StudyValidationError("reference target losses mismatch")
    if plan_configuration.get("optimizer_telemetry") is not None:
        raise StudyValidationError("reference scored study contains optimizer telemetry")
    if plan_configuration.get("execution_mode") != "serial":
        raise StudyValidationError("reference execution mode is not serial")
    if plan_configuration.get("resource_budget") != {
        "currency": "USD",
        "gpu_count": 1,
        "maximum_gpu_hourly_price": 1.6,
        "maximum_provider_charge": 16.0,
        "maximum_provider_hours": 10.0,
        "planned_runs": 20,
        "scored_objective_seconds": 24_000,
    }:
        raise StudyValidationError("reference resource budget mismatch")
    panel = plan_configuration.get("topology_panel")
    if (
        not isinstance(panel, dict)
        or panel.get("panel_id") != PANEL_ID
        or panel.get("topology_count") != 10
    ):
        raise StudyValidationError("reference topology panel identity mismatch")
    expected_specs = [
        {"kind": "string", "value": topology}
        for topology in SUBMISSION_LIKE_TOPOLOGIES
    ]
    if plan_configuration.get("topologies") != expected_specs:
        raise StudyValidationError("reference topology membership/order mismatch")
    policy = plan_configuration.get("decision_policy")
    if not isinstance(policy, dict):
        raise StudyValidationError("reference frozen policy is missing")
    for key, expected in FROZEN_POLICY.items():
        if policy.get(key) != expected:
            raise StudyValidationError(f"reference frozen policy mismatch for {key}")
    return plan_configuration


def _cell_from_config(
    run_id: str, config: dict[str, object]
) -> tuple[str, int, str]:
    if config.get("run_id") != run_id:
        raise StudyValidationError(f"reference config/run ID mismatch: {run_id}")
    if config.get("arm") != ARM:
        raise StudyValidationError(f"reference unexpected arm: {run_id}")
    seed = config.get("optimizer_seed")
    if type(seed) is not int or seed not in EXPECTED_SEEDS:
        raise StudyValidationError(f"reference unexpected optimizer seed: {run_id}")
    topology_spec = config.get("topology")
    if not isinstance(topology_spec, dict) or topology_spec.get("kind") != "string":
        raise StudyValidationError(f"reference topology config is invalid: {run_id}")
    topology = topology_spec.get("value")
    if topology not in SUBMISSION_LIKE_TOPOLOGIES:
        raise StudyValidationError(f"reference unexpected topology: {run_id}")
    pair_id = config.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise StudyValidationError(f"reference pair ID is invalid: {run_id}")
    return str(topology), seed, pair_id


def _history_outcome(
    run_id: str,
    rows: list[dict[str, object]],
    record: dict[str, object],
) -> dict[str, object]:
    if not rows:
        raise StudyValidationError(f"reference history is empty: {run_id}")
    physical_feasible = False
    finite_losses: list[float] = []
    target_hits = {
        _target_key(target): {"time_seconds": None, "eval_count": None}
        for target in TARGET_LOSSES
    }
    running_best: float | None = None
    previous_time = -math.inf
    previous_evals = -1
    for row in rows:
        feasible = row.get("is_feasible")
        if type(feasible) is not bool:
            raise StudyValidationError(
                f"reference feasibility is not strict boolean: {run_id}"
            )
        time_value = row.get("time_seconds")
        eval_value = row.get("eval_count_after_call")
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, (int, float))
            or not math.isfinite(float(time_value))
            or float(time_value) < previous_time
        ):
            raise StudyValidationError(f"reference history time is invalid: {run_id}")
        if type(eval_value) is not int or eval_value < previous_evals:
            raise StudyValidationError(
                f"reference history evaluation count is invalid: {run_id}"
            )
        previous_time = float(time_value)
        previous_evals = eval_value
        if not feasible:
            continue
        physical_feasible = True
        loss = row.get("loss")
        if loss is None:
            continue
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise StudyValidationError(
                f"reference history contains invalid feasible loss: {run_id}"
            )
        loss_value = float(loss)
        finite_losses.append(loss_value)
        running_best = (
            loss_value if running_best is None else min(running_best, loss_value)
        )
        for target in TARGET_LOSSES:
            key = _target_key(target)
            if running_best <= target and target_hits[key]["time_seconds"] is None:
                target_hits[key] = {
                    "time_seconds": float(time_value),
                    "eval_count": eval_value,
                }

    finite_feasible = bool(finite_losses)
    best_loss = min(finite_losses) if finite_feasible else None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not {
        "has_feasible",
        "has_finite_feasible",
        "best_feasible_loss",
        "targets",
    } <= set(metrics):
        raise StudyValidationError(f"reference record metrics are missing: {run_id}")
    if metrics["has_feasible"] is not physical_feasible:
        raise StudyValidationError(
            f"history/record physical feasibility mismatch: {run_id}"
        )
    if metrics["has_finite_feasible"] is not finite_feasible:
        raise StudyValidationError(
            f"history/record finite feasibility mismatch: {run_id}"
        )
    if metrics["best_feasible_loss"] != best_loss:
        raise StudyValidationError(f"history/record best loss mismatch: {run_id}")
    if metrics["targets"] != target_hits:
        raise StudyValidationError(f"history/record target-hit mismatch: {run_id}")
    return {
        "run_id": run_id,
        "physical_feasible": physical_feasible,
        "finite_feasible": finite_feasible,
        "best_feasible_loss": best_loss,
        "target_hits": target_hits,
    }


def _target_hitting(
    outcomes: dict[tuple[str, int], dict[str, object] | None]
) -> dict[str, object]:
    result = {}
    for target in TARGET_LOSSES:
        key = _target_key(target)
        run_counts = {"reached": 0, "right_censored": 0, "incomplete": 0}
        topology_counts = {
            "both_seeds_reached": 0,
            "one_seed_reached": 0,
            "neither_seed_reached": 0,
            "incomplete": 0,
        }
        topology_rows = []
        for topology in SUBMISSION_LIKE_TOPOLOGIES:
            seed_hits: dict[str, dict[str, object] | None] = {}
            complete = True
            reached = 0
            for seed in EXPECTED_SEEDS:
                outcome = outcomes.get((topology, seed))
                if outcome is None:
                    run_counts["incomplete"] += 1
                    seed_hits[str(seed)] = None
                    complete = False
                    continue
                hit = outcome["target_hits"][key]
                seed_hits[str(seed)] = hit
                if hit["time_seconds"] is None:
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
            topology_rows.append(
                {
                    "topology": topology,
                    "category": category,
                    "seed_hits": seed_hits,
                }
            )
        result[key] = {
            "target_loss": target,
            "run_categories": run_counts,
            "topology_categories": topology_counts,
            "topology_rows": topology_rows,
            "censoring_preserved": True,
        }
    return result


def reference_submission_like_screen(study: ValidatedStudy) -> dict[str, object]:
    """Recompute the frozen single-arm evidence from sealed history rows."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("reference calculation requires the summary sealed")
    plan_configuration = _configuration(study)
    package_evidence = _candidate_package_evidence(study, plan_configuration)

    expected_cells = {
        (topology, seed)
        for topology in SUBMISSION_LIKE_TOPOLOGIES
        for seed in EXPECTED_SEEDS
    }
    configs_by_cell: dict[tuple[str, int], tuple[str, dict[str, object]]] = {}
    pair_cells: dict[str, tuple[str, int]] = {}
    for run_id, config in study.configs.items():
        if not isinstance(config, dict):
            raise StudyValidationError(f"reference config is missing: {run_id}")
        if _canonical(config.get("candidate_package_evidence")) != _canonical(
            package_evidence
        ):
            raise StudyValidationError(
                f"reference config candidate package evidence mismatch: {run_id}"
            )
        topology, seed, pair_id = _cell_from_config(run_id, config)
        cell = (topology, seed)
        if cell in configs_by_cell:
            raise StudyValidationError(f"reference duplicate topology/seed cell: {cell}")
        previous_cell = pair_cells.setdefault(pair_id, cell)
        if previous_cell != cell:
            raise StudyValidationError(f"reference pair ID crosses cells: {pair_id}")
        configs_by_cell[cell] = (run_id, config)
    if set(configs_by_cell) - expected_cells:
        raise StudyValidationError("reference contains unexpected topology/seed cells")

    records_by_id: dict[str, dict[str, object]] = {}
    for record in study.records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise StudyValidationError("reference run ID is invalid")
        if run_id in records_by_id:
            raise StudyValidationError(f"reference duplicate run ID: {run_id}")
        if run_id not in study.configs:
            raise StudyValidationError(f"reference record is outside the plan: {run_id}")
        if record.get("config") != study.configs[run_id]:
            raise StudyValidationError(f"reference record/config mismatch: {run_id}")
        records_by_id[run_id] = record
    unexpected_histories = set(study.history_rows) - set(study.configs)
    if unexpected_histories:
        raise StudyValidationError("reference histories contain runs outside the plan")

    outcomes: dict[tuple[str, int], dict[str, object] | None] = {
        cell: None for cell in expected_cells
    }
    run_rows = []
    complete_runs = 0
    finite_feasible_runs = 0
    physical_feasible_runs = 0
    for cell in sorted(expected_cells):
        planned = configs_by_cell.get(cell)
        if planned is None:
            continue
        run_id, config = planned
        record = records_by_id.get(run_id)
        if record is None or record.get("status") != "complete":
            continue
        rows = study.history_rows.get(run_id)
        if rows is None:
            raise StudyValidationError(f"reference complete run lacks history: {run_id}")
        topology, seed = cell
        problem = record.get("problem")
        digest = hashlib.sha256(topology.encode()).hexdigest()
        if (
            not isinstance(problem, dict)
            or problem.get("topology_string") != topology
            or problem.get("topology_sha256") != digest
            or config.get("topology") != {"kind": "string", "value": topology}
        ):
            raise StudyValidationError(f"reference topology identity mismatch: {run_id}")
        if record.get("optimizer_telemetry") is not None:
            raise StudyValidationError(f"reference run contains optimizer telemetry: {run_id}")
        outcome = _history_outcome(run_id, rows, record)
        outcomes[cell] = outcome
        complete_runs += 1
        physical_feasible_runs += int(bool(outcome["physical_feasible"]))
        finite_feasible_runs += int(bool(outcome["finite_feasible"]))
        run_rows.append({"topology": topology, "optimizer_seed": seed, **outcome})

    topology_rows = []
    complete_blocks = 0
    finite_blocks = 0
    topology_means: list[float] = []
    seed_gaps: list[float] = []
    for topology in SUBMISSION_LIKE_TOPOLOGIES:
        seed_outcomes = [outcomes[(topology, seed)] for seed in EXPECTED_SEEDS]
        complete = all(outcome is not None for outcome in seed_outcomes)
        complete_blocks += int(complete)
        finite = complete and all(
            bool(outcome["finite_feasible"]) for outcome in seed_outcomes if outcome
        )
        finite_blocks += int(finite)
        values = (
            [float(outcome["best_feasible_loss"]) for outcome in seed_outcomes if outcome]
            if finite
            else []
        )
        topology_mean = _mean(values) if finite else None
        seed_gap = abs(values[0] - values[1]) if finite else None
        if topology_mean is not None and seed_gap is not None:
            topology_means.append(topology_mean)
            seed_gaps.append(seed_gap)
        topology_rows.append(
            {
                "topology": topology,
                "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                "complete": complete,
                "finite_feasible": finite,
                "seed_best_feasible_loss": {
                    str(seed): (
                        outcomes[(topology, seed)]["best_feasible_loss"]
                        if outcomes[(topology, seed)] is not None
                        else None
                    )
                    for seed in EXPECTED_SEEDS
                },
                "topology_mean_best_feasible_loss": topology_mean,
                "absolute_seed_gap": seed_gap,
            }
        )

    complete_panel = complete_runs == 20 and complete_blocks == 10
    all_finite = finite_feasible_runs == 20 and finite_blocks == 10
    aggregate_ready = complete_panel and all_finite
    p90_gap = _linear_percentile(seed_gaps, 0.9) if aggregate_ready else None
    criteria = {
        "panel_execution_complete": {
            "observed_runs": complete_runs,
            "observed_topology_blocks": complete_blocks,
            "passed": complete_panel,
        },
        "complete_records_revalidated": {
            "observed": complete_runs,
            "required_runs": 20,
            "passed": complete_runs == 20,
        },
        "complete_topology_blocks": {
            "observed": complete_blocks,
            "required_topology_blocks": 10,
            "passed": complete_blocks == 10,
        },
        "all_runs_finite_feasible": {
            "observed": finite_feasible_runs,
            "required": 20,
            "passed": all_finite,
        },
        "candidate_package_bound": {
            "passed": True,
        },
    }
    if not complete_panel:
        status = "not_evaluable"
        action = FROZEN_POLICY["action_if_not_evaluable"]
    elif all(item["passed"] for item in criteria.values()):
        status = "passed"
        action = FROZEN_POLICY["action_if_passed"]
    else:
        status = "failed"
        action = FROZEN_POLICY["action_if_failed"]

    return {
        "format_version": 1,
        "study_profile": PROFILE_ID,
        "policy": dict(FROZEN_POLICY),
        "summary_content_opened": False,
        "inference_unit": "topology",
        "optimizer_seeds_are_repeated_measurements": True,
        "completed_runs": complete_runs,
        "complete_topology_blocks": complete_blocks,
        "physical_feasible_runs": physical_feasible_runs,
        "finite_feasible_runs": finite_feasible_runs,
        "run_rows": run_rows,
        "topology_rows": topology_rows,
        "topology_macro_mean_best_feasible_loss": (
            _mean(topology_means) if aggregate_ready else None
        ),
        "topology_macro_median_best_feasible_loss": (
            _median(topology_means) if aggregate_ready else None
        ),
        "topology_macro_p90_best_feasible_loss": (
            _linear_percentile(topology_means, 0.9) if aggregate_ready else None
        ),
        "topology_p90_absolute_seed_gap": p90_gap,
        "topology_bootstrap_mean_best_feasible_loss_ci_95": (
            _bootstrap_mean_ci(topology_means) if aggregate_ready else None
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "target_hitting": _target_hitting(outcomes),
        "predeclared_decision": {
            "policy_id": FROZEN_POLICY["policy_id"],
            "status": status,
            "passed": status == "passed",
            "action": action,
            "criteria": criteria,
        },
    }
