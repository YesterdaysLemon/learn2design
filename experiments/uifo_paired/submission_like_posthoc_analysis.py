"""Aggregate-only reliability analysis for ``submission-like-screen-v1``.

This module runs only after the frozen production/reference replay agrees.  It
does not alter the predeclared decision and never treats seeds or history rows
as independent experimental units.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)
from experiments.uifo_paired.submission_like_trajectory_analysis import (
    analyze_submission_like_trajectories,
    safe_submission_like_trajectories,
)


EXPLORATORY_SEED = 20260823
EXPLORATORY_RESAMPLES = 10_000
EXPECTED_SEEDS = (29, 31)


def _utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise StudyValidationError(f"post-hoc {field} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StudyValidationError(
            f"post-hoc {field} timestamp is malformed"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StudyValidationError(
            f"post-hoc {field} timestamp must be timezone-aware"
        )
    return parsed.astimezone(UTC)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise StudyValidationError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _finite_test_result(result: object) -> dict[str, float | None]:
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "statistic": statistic if math.isfinite(statistic) else None,
        "two_sided_p_value": pvalue if math.isfinite(pvalue) else None,
    }


def _correlation(left: list[float], right: list[float], implementation) -> dict[str, object]:
    if len(left) != len(right) or len(left) < 3:
        raise StudyValidationError("correlation inputs are incomplete")
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {
            "statistic": None,
            "two_sided_p_value": None,
            "reason": "constant input",
        }
    return _finite_test_result(implementation(left, right))


def _bootstrap_median(topology_means: list[float]) -> dict[str, object]:
    generator = random.Random(EXPLORATORY_SEED)
    estimates = []
    for _ in range(EXPLORATORY_RESAMPLES):
        sample = [generator.choice(topology_means) for _ in topology_means]
        estimates.append(float(median(sample)))
    return {
        "confidence_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "resamples": EXPLORATORY_RESAMPLES,
        "seed": EXPLORATORY_SEED,
        "method": "percentile bootstrap over complete topology blocks",
        "changes_frozen_decision": False,
    }


def _topology_label_map(reference: dict[str, object]) -> dict[str, str]:
    hashes = sorted(str(row["topology_sha256"]) for row in reference["topology_rows"])
    if len(hashes) != 10 or len(set(hashes)) != 10:
        raise StudyValidationError("post-hoc analysis requires ten unique topologies")
    return {digest: f"T{index:02d}" for index, digest in enumerate(hashes, start=1)}


def _terminal_run_rows(
    study: ValidatedStudy,
    reference: dict[str, object],
    label_by_hash: dict[str, str],
) -> list[dict[str, object]]:
    outcomes = {
        (str(row["topology"]), int(row["optimizer_seed"])): row
        for row in reference["run_rows"]
    }
    records = {str(record.get("run_id")): record for record in study.records}
    if len(records) != 20:
        raise StudyValidationError("post-hoc run record IDs are not unique")
    session_started = _utc_timestamp(
        study.session.get("started_utc"), "session.started_utc"
    )
    session_completed = _utc_timestamp(
        study.session.get("completed_utc"), "session.completed_utc"
    )
    if session_completed <= session_started:
        raise StudyValidationError("post-hoc session timestamps are reversed")
    rows = []
    for run_id, config in study.configs.items():
        topology_spec = config.get("topology")
        if not isinstance(topology_spec, dict):
            raise StudyValidationError("post-hoc topology config is invalid")
        topology = topology_spec.get("value")
        seed = config.get("optimizer_seed")
        if not isinstance(topology, str) or type(seed) is not int:
            raise StudyValidationError("post-hoc topology/seed cell is invalid")
        outcome = outcomes.get((topology, seed))
        history = study.history_rows.get(run_id)
        record = records.get(run_id)
        if outcome is None or not history or record is None:
            raise StudyValidationError("post-hoc run/history cell is incomplete")
        started = _utc_timestamp(record.get("started_utc"), "run.started_utc")
        completed = _utc_timestamp(
            record.get("completed_utc"), "run.completed_utc"
        )
        if completed <= started:
            raise StudyValidationError("post-hoc run completion precedes its start")
        if started < session_started or completed > session_completed:
            raise StudyValidationError("post-hoc run timestamp is outside the session")
        terminal_time = max(float(item["time_seconds"]) for item in history)
        terminal_evals = max(int(item["eval_count_after_call"]) for item in history)
        if terminal_time <= 0 or terminal_evals <= 0:
            raise StudyValidationError("post-hoc terminal run values are invalid")
        digest = hashlib.sha256(topology.encode()).hexdigest()
        if digest not in label_by_hash:
            raise StudyValidationError("post-hoc topology binding is invalid")
        planned_index = config.get("planned_run_index")
        if type(planned_index) is not int:
            raise StudyValidationError("post-hoc run order is invalid")
        rows.append(
            {
                "topology_label": label_by_hash[digest],
                "topology_sha256": digest,
                "optimizer_seed": seed,
                "planned_run_index": planned_index,
                "best_feasible_loss": float(outcome["best_feasible_loss"]),
                "terminal_time_seconds": terminal_time,
                "terminal_eval_count": terminal_evals,
                "evaluations_per_second": terminal_evals / terminal_time,
                "session_start_offset_seconds": (
                    started - session_started
                ).total_seconds(),
                "worker_wall_seconds_from_utc": (completed - started).total_seconds(),
            }
        )
    rows.sort(key=lambda row: int(row["planned_run_index"]))
    if [int(row["planned_run_index"]) for row in rows] != list(range(20)):
        raise StudyValidationError("post-hoc serial run order is not exact")
    for earlier, later in zip(rows, rows[1:]):
        if float(later["session_start_offset_seconds"]) <= float(
            earlier["session_start_offset_seconds"]
        ):
            raise StudyValidationError(
                "post-hoc run timestamps are nonmonotone in serial order"
            )
        if float(later["session_start_offset_seconds"]) < (
            float(earlier["session_start_offset_seconds"])
            + float(earlier["worker_wall_seconds_from_utc"])
        ):
            raise StudyValidationError("post-hoc serial run timestamps overlap")
    return rows


def _topology_rows(
    run_rows: list[dict[str, object]],
    reference: dict[str, object],
    label_by_hash: dict[str, str],
) -> list[dict[str, object]]:
    by_cell = {
        (str(row["topology_sha256"]), int(row["optimizer_seed"])): row
        for row in run_rows
    }
    rows = []
    for frozen in reference["topology_rows"]:
        digest = str(frozen["topology_sha256"])
        seed_rows = {seed: by_cell[(digest, seed)] for seed in EXPECTED_SEEDS}
        earlier = min(seed_rows.values(), key=lambda row: int(row["planned_run_index"]))
        later = max(seed_rows.values(), key=lambda row: int(row["planned_run_index"]))
        earlier_throughput = float(earlier["evaluations_per_second"])
        later_throughput = float(later["evaluations_per_second"])
        rows.append(
            {
                "topology_label": label_by_hash[digest],
                "topology_sha256": digest,
                "seed_29_best_feasible_loss": float(seed_rows[29]["best_feasible_loss"]),
                "seed_31_best_feasible_loss": float(seed_rows[31]["best_feasible_loss"]),
                "topology_mean_best_feasible_loss": float(
                    frozen["topology_mean_best_feasible_loss"]
                ),
                "absolute_seed_gap": float(frozen["absolute_seed_gap"]),
                "earlier_seed": int(earlier["optimizer_seed"]),
                "later_seed": int(later["optimizer_seed"]),
                "earlier_planned_run_index": int(earlier["planned_run_index"]),
                "later_planned_run_index": int(later["planned_run_index"]),
                "pair_gap_runs": int(later["planned_run_index"])
                - int(earlier["planned_run_index"]),
                "actual_start_gap_seconds": float(
                    later["session_start_offset_seconds"]
                )
                - float(earlier["session_start_offset_seconds"]),
                "later_minus_earlier_loss": float(later["best_feasible_loss"])
                - float(earlier["best_feasible_loss"]),
                "later_over_earlier_throughput_ratio": later_throughput
                / earlier_throughput,
            }
        )
    rows.sort(key=lambda row: str(row["topology_label"]))
    return rows


def _target_hitting(
    reference: dict[str, object],
    label_by_hash: dict[str, str],
    run_rows: list[dict[str, object]],
) -> dict[str, object]:
    run_by_cell = {
        (str(row["topology_label"]), int(row["optimizer_seed"])): row
        for row in run_rows
    }
    output = {}
    for key, payload in reference["target_hitting"].items():
        both_reached_topology_times = []
        both_reached_topology_evaluations = []
        topology_rows = []
        for row in payload["topology_rows"]:
            topology = str(row["topology"])
            digest = hashlib.sha256(topology.encode()).hexdigest()
            seed_hits = row["seed_hits"]
            safe_hits = {}
            reached_times = []
            reached_evaluations = []
            for seed in EXPECTED_SEEDS:
                hit = seed_hits[str(seed)]
                terminal = run_by_cell[(label_by_hash[digest], seed)]
                safe_hits[str(seed)] = {
                    "event_reached": hit["time_seconds"] is not None,
                    "observed_or_censor_time_seconds": (
                        hit["time_seconds"]
                        if hit["time_seconds"] is not None
                        else terminal["terminal_time_seconds"]
                    ),
                    "observed_or_censor_eval_count": (
                        hit["eval_count"]
                        if hit["eval_count"] is not None
                        else terminal["terminal_eval_count"]
                    ),
                }
                if hit["time_seconds"] is not None:
                    reached_times.append(float(hit["time_seconds"]))
                    reached_evaluations.append(int(hit["eval_count"]))
            if len(reached_times) == 2:
                both_reached_topology_times.append(mean(reached_times))
                both_reached_topology_evaluations.append(mean(reached_evaluations))
            topology_rows.append(
                {
                    "topology_label": label_by_hash[digest],
                    "topology_sha256": digest,
                    "category": row["category"],
                    "seed_hits": safe_hits,
                }
            )
        seed_29_only = sum(
            row["seed_hits"]["29"]["event_reached"]
            and not row["seed_hits"]["31"]["event_reached"]
            for row in topology_rows
        )
        seed_31_only = sum(
            row["seed_hits"]["31"]["event_reached"]
            and not row["seed_hits"]["29"]["event_reached"]
            for row in topology_rows
        )
        output[str(key)] = {
            "target_loss": float(payload["target_loss"]),
            "run_categories": dict(payload["run_categories"]),
            "topology_categories": dict(payload["topology_categories"]),
            "seed_29_only_topologies": seed_29_only,
            "seed_31_only_topologies": seed_31_only,
            "both_reached_topology_time_seconds_median": (
                median(both_reached_topology_times)
                if both_reached_topology_times
                else None
            ),
            "both_reached_topology_time_seconds_p90": (
                _percentile(both_reached_topology_times, 0.9)
                if both_reached_topology_times
                else None
            ),
            "both_reached_topology_eval_count_median": (
                median(both_reached_topology_evaluations)
                if both_reached_topology_evaluations
                else None
            ),
            "both_reached_topology_eval_count_p90": (
                _percentile(both_reached_topology_evaluations, 0.9)
                if both_reached_topology_evaluations
                else None
            ),
            "topology_rows": topology_rows,
            "censoring_rule": (
                "Unreached targets retain their terminal time/evaluation censor "
                "bounds. Time and evaluation summaries use only complete "
                "both-seeds-reached topology blocks and are reported separately."
            ),
            "conditional_summary_warning": (
                "Both-reached topology summaries are selection-biased and "
                "descriptive only."
            ),
            "changes_frozen_decision": False,
        }
    return output


def analyze_submission_like_posthoc(
    study: ValidatedStudy,
    production: dict[str, object],
    reference: dict[str, object],
    *,
    agreement: dict[str, object],
) -> dict[str, object]:
    """Diagnose single-arm reliability after the three-way frozen gate passes."""
    if (
        agreement.get("status") != "matched"
        or agreement.get("archived_summary_compared") is not True
        or agreement.get("fields_compared") != 21
        or agreement.get("absolute_tolerance") != 1e-12
        or agreement.get("relative_tolerance") != 1e-12
    ):
        raise StudyValidationError("post-hoc analysis requires three-way agreement")
    if production.get("completed_runs") != 20 or reference.get("completed_runs") != 20:
        raise StudyValidationError("post-hoc analysis requires the complete panel")
    if production.get("predeclared_decision", {}).get("status") != "passed":
        raise StudyValidationError("post-hoc analysis requires the passed frozen gate")
    if production["predeclared_decision"].get("action") != (
        "candidate_evidence_complete_for_submission_review"
    ):
        raise StudyValidationError("post-hoc analysis requires the exact frozen action")
    if reference.get("predeclared_decision", {}).get("status") != "passed":
        raise StudyValidationError("post-hoc reference decision is not passed")
    if reference["predeclared_decision"].get("action") != (
        "candidate_evidence_complete_for_submission_review"
    ):
        raise StudyValidationError("post-hoc reference action is not exact")

    from scipy.stats import pearsonr, spearmanr

    labels = _topology_label_map(reference)
    run_rows = _terminal_run_rows(study, reference, labels)
    topology_rows = _topology_rows(run_rows, reference, labels)
    trajectory_alignment = analyze_submission_like_trajectories(study)
    means = [float(row["topology_mean_best_feasible_loss"]) for row in topology_rows]
    gaps = [float(row["absolute_seed_gap"]) for row in topology_rows]
    seed_29 = [float(row["seed_29_best_feasible_loss"]) for row in topology_rows]
    seed_31 = [float(row["seed_31_best_feasible_loss"]) for row in topology_rows]
    later_differences = [float(row["later_minus_earlier_loss"]) for row in topology_rows]
    throughput_ratios = [
        float(row["later_over_earlier_throughput_ratio"]) for row in topology_rows
    ]
    pair_gaps = [float(row["pair_gap_runs"]) for row in topology_rows]
    actual_start_gaps = [
        float(row["actual_start_gap_seconds"]) for row in topology_rows
    ]
    log_throughput_ratios = [math.log10(value) for value in throughput_ratios]
    ties = sum(abs(value) <= 1e-12 for value in later_differences)
    topology_terminal_evals = []
    topology_eval_rates = []
    for topology in topology_rows:
        label = str(topology["topology_label"])
        cells = [
            row for row in run_rows if str(row["topology_label"]) == label
        ]
        topology_terminal_evals.append(
            mean(float(row["terminal_eval_count"]) for row in cells)
        )
        topology_eval_rates.append(
            mean(float(row["evaluations_per_second"]) for row in cells)
        )

    loo = []
    for omitted in topology_rows:
        remaining = [row for row in topology_rows if row is not omitted]
        remaining_means = [
            float(row["topology_mean_best_feasible_loss"]) for row in remaining
        ]
        remaining_gaps = [float(row["absolute_seed_gap"]) for row in remaining]
        loo.append(
            {
                "omitted_topology_label": omitted["topology_label"],
                "topology_mean_best_feasible_loss": mean(remaining_means),
                "topology_median_best_feasible_loss": median(remaining_means),
                "topology_p90_best_feasible_loss": _percentile(remaining_means, 0.9),
                "topology_p90_absolute_seed_gap": _percentile(remaining_gaps, 0.9),
                "topologies_remaining": 9,
                "decision_recomputed": False,
            }
        )

    return {
        "format_version": 1,
        "study_profile": "submission-like-screen-v1",
        "inference_unit": "topology (n=10)",
        "seeds_are_repeated_measurements": True,
        "changes_frozen_decision": False,
        "no_new_action_authorized": True,
        "run_rows": run_rows,
        "topology_rows": topology_rows,
        "descriptive_loss": {
            "topology_mean_best_feasible_loss": mean(means),
            "topology_median_best_feasible_loss": median(means),
            "topology_p90_best_feasible_loss": _percentile(means, 0.9),
            "topology_p90_absolute_seed_gap": _percentile(gaps, 0.9),
            "exploratory_median_bootstrap_ci_95": _bootstrap_median(means),
        },
        "seed_consistency": {
            "seed_29_mean_best_feasible_loss": mean(seed_29),
            "seed_31_mean_best_feasible_loss": mean(seed_31),
            "seed_31_minus_seed_29_mean_loss": mean(seed_31) - mean(seed_29),
            "seed_31_minus_seed_29_median_loss": median(
                b - a for a, b in zip(seed_29, seed_31)
            ),
            "absolute_seed_gap_mean": mean(gaps),
            "absolute_seed_gap_median": median(gaps),
            "absolute_seed_gap_p90": _percentile(gaps, 0.9),
            "pearson_across_topologies": _correlation(seed_29, seed_31, pearsonr),
            "spearman_across_topologies": _correlation(seed_29, seed_31, spearmanr),
            "seed_29_lower": sum(a < b - 1e-12 for a, b in zip(seed_29, seed_31)),
            "seed_31_lower": sum(b < a - 1e-12 for a, b in zip(seed_29, seed_31)),
            "ties": sum(abs(a - b) <= 1e-12 for a, b in zip(seed_29, seed_31)),
            "seed_and_sweep_phase_confounded": True,
            "interpretation": (
                "Repeated-seed consistency is descriptive; the two seeds are not "
                "independent topology observations."
            ),
        },
        "leave_one_topology_out": loo,
        "serial_drift": {
            "topology_pairs": 10,
            "pair_gap_runs_vs_later_minus_earlier_loss": _correlation(
                pair_gaps, later_differences, spearmanr
            ),
            "actual_start_gap_vs_later_minus_earlier_loss": _correlation(
                actual_start_gaps, later_differences, spearmanr
            ),
            "pair_gap_runs_vs_log10_throughput_ratio": _correlation(
                pair_gaps, log_throughput_ratios, spearmanr
            ),
            "mirrored_topology_contrasts": {
                "mean_later_minus_earlier_loss": mean(later_differences),
                "median_later_minus_earlier_loss": median(later_differences),
                "later_lower": sum(value < -1e-12 for value in later_differences),
                "later_higher": sum(value > 1e-12 for value in later_differences),
                "ties": ties,
                "mean_log10_later_over_earlier_throughput": mean(
                    log_throughput_ratios
                ),
            },
            "arm_first_order": "not applicable: the frozen screen has one arm",
            "session_time": (
                "Strict timezone-aware run timestamps were validated inside the "
                "completed session and in non-overlapping serial order. Actual "
                "run-start gaps include controller and compilation intervals."
            ),
            "caveat": (
                "Seed 29 is the first sweep and seed 31 the second, so seed and "
                "sweep phase are not identifiable. All diagnostics use ten complete "
                "topology blocks and cannot identify a causal session-time effect."
            ),
            "changes_frozen_decision": False,
            "causal_effect_identified": False,
        },
        "evaluation_throughput": {
            "topology_macro_terminal_eval_count": mean(topology_terminal_evals),
            "topology_macro_evaluations_per_second": mean(topology_eval_rates),
            "mean_log10_later_over_earlier_throughput": mean(
                log_throughput_ratios
            ),
            "median_log10_later_over_earlier_throughput": median(
                log_throughput_ratios
            ),
            "loss_contrast_vs_log10_throughput_contrast": _correlation(
                later_differences, log_throughput_ratios, spearmanr
            ),
            "time_and_evaluation_counts_reported_separately": True,
            "changes_frozen_decision": False,
        },
        "trajectory_alignment": trajectory_alignment,
        "target_hitting": _target_hitting(reference, labels, run_rows),
    }


def safe_submission_like_posthoc(posthoc: dict[str, object]) -> dict[str, object]:
    """Return an allowlisted aggregate view without run rows or topology hashes."""
    targets = {}
    for key, payload in posthoc["target_hitting"].items():
        targets[str(key)] = {
            name: payload[name]
            for name in (
                "target_loss",
                "run_categories",
                "topology_categories",
                "seed_29_only_topologies",
                "seed_31_only_topologies",
                "both_reached_topology_time_seconds_median",
                "both_reached_topology_time_seconds_p90",
                "both_reached_topology_eval_count_median",
                "both_reached_topology_eval_count_p90",
                "censoring_rule",
                "conditional_summary_warning",
                "changes_frozen_decision",
            )
        }
    loo = posthoc["leave_one_topology_out"]
    loo_fields = (
        "topology_mean_best_feasible_loss",
        "topology_median_best_feasible_loss",
        "topology_p90_best_feasible_loss",
        "topology_p90_absolute_seed_gap",
    )
    loo_ranges = {
        field: {
            "minimum": min(float(row[field]) for row in loo),
            "maximum": max(float(row[field]) for row in loo),
        }
        for field in loo_fields
    }
    return {
        "format_version": posthoc["format_version"],
        "study_profile": posthoc["study_profile"],
        "inference_unit": posthoc["inference_unit"],
        "seeds_are_repeated_measurements": True,
        "changes_frozen_decision": False,
        "no_new_action_authorized": True,
        "descriptive_loss": posthoc["descriptive_loss"],
        "seed_consistency": posthoc["seed_consistency"],
        "leave_one_topology_out": {
            "omissions": len(loo),
            "topologies_remaining_each": 9,
            "decision_recomputed": False,
            "ranges": loo_ranges,
        },
        "serial_drift": posthoc["serial_drift"],
        "evaluation_throughput": posthoc["evaluation_throughput"],
        "trajectory_alignment": safe_submission_like_trajectories(
            posthoc["trajectory_alignment"]
        ),
        "target_hitting": targets,
    }


def create_submission_like_plots(
    posthoc: dict[str, object], output_dir: Path
) -> list[Path]:
    """Render the five private aggregate diagnostic figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    topology_rows = posthoc["topology_rows"]
    labels = [str(row["topology_label"]) for row in topology_rows]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for index, row in enumerate(topology_rows):
        values = [row["seed_29_best_feasible_loss"], row["seed_31_best_feasible_loss"]]
        ax.plot([index, index], values, color="#94a3b8", linewidth=1.6, zorder=1)
    ax.scatter(
        x,
        [row["seed_29_best_feasible_loss"] for row in topology_rows],
        label="seed 29",
        color="#2563eb",
        s=42,
        zorder=2,
    )
    ax.scatter(
        x,
        [row["seed_31_best_feasible_loss"] for row in topology_rows],
        label="seed 31",
        color="#dc2626",
        s=42,
        zorder=2,
    )
    ax.scatter(
        x,
        [row["topology_mean_best_feasible_loss"] for row in topology_rows],
        label="topology mean",
        marker="D",
        color="#111827",
        s=34,
        zorder=3,
    )
    ax.axhline(
        4.0,
        color="#64748b",
        linestyle="--",
        linewidth=1.2,
        label="target 4.0",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Best finite feasible loss (lower is better)")
    ax.set_title("Submission-like screen: topology outcomes and seed spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    topology_path = figures_dir / "topology_seed_outcomes.png"
    fig.savefig(topology_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    targets = sorted(
        posthoc["target_hitting"].values(),
        key=lambda item: float(item["target_loss"]),
        reverse=True,
    )
    target_labels = [format(float(item["target_loss"]), ".12g") for item in targets]
    both = [item["topology_categories"]["both_seeds_reached"] for item in targets]
    one = [item["topology_categories"]["one_seed_reached"] for item in targets]
    neither = [
        item["topology_categories"]["neither_seed_reached"] for item in targets
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(target_labels, both, label="both seeds reached", color="#059669")
    ax.bar(target_labels, one, bottom=both, label="one seed reached", color="#f59e0b")
    ax.bar(
        target_labels,
        neither,
        bottom=[a + b for a, b in zip(both, one)],
        label="neither reached",
        color="#94a3b8",
    )
    ax.set_xlabel("Target loss")
    ax.set_ylabel("Topology count (n=10)")
    ax.set_ylim(0, 10.5)
    ax.set_title("Censor-aware target attainment")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    target_path = figures_dir / "target_hitting_outcomes.png"
    fig.savefig(target_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    loo = posthoc["leave_one_topology_out"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    full_mean = posthoc["descriptive_loss"]["topology_mean_best_feasible_loss"]
    full_gap = posthoc["descriptive_loss"]["topology_p90_absolute_seed_gap"]
    axes[0].plot(
        [row["omitted_topology_label"] for row in loo],
        [row["topology_mean_best_feasible_loss"] for row in loo],
        marker="o",
        color="#0f766e",
    )
    axes[0].set_ylabel("Leave-one-out mean loss")
    axes[0].set_title("Mean stability")
    axes[0].axhline(full_mean, color="#475569", linestyle="--", linewidth=1.0)
    axes[1].plot(
        [row["omitted_topology_label"] for row in loo],
        [row["topology_p90_absolute_seed_gap"] for row in loo],
        marker="o",
        color="#b45309",
    )
    axes[1].set_ylabel("Leave-one-out p90 seed gap")
    axes[1].set_title("Seed-gap stability")
    axes[1].axhline(full_gap, color="#475569", linestyle="--", linewidth=1.0)
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.2)
    fig.suptitle("Leave-one-topology-out sensitivity")
    fig.tight_layout()
    loo_path = figures_dir / "leave_one_topology_out.png"
    fig.savefig(loo_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].scatter(
        [row["pair_gap_runs"] for row in topology_rows],
        [row["later_minus_earlier_loss"] for row in topology_rows],
        color="#7c3aed",
        s=42,
    )
    axes[0].axhline(0.0, color="#475569", linewidth=1.0)
    axes[0].set_xlabel("Planned run-position gap within topology")
    axes[0].set_ylabel("Later minus earlier loss")
    axes[0].set_title("Loss phase contrast vs gap")
    axes[1].scatter(
        [row["actual_start_gap_seconds"] for row in topology_rows],
        [
            math.log10(row["later_over_earlier_throughput_ratio"])
            for row in topology_rows
        ],
        color="#0891b2",
        s=42,
    )
    axes[1].axhline(0.0, color="#475569", linewidth=1.0)
    axes[1].set_xlabel("Actual run-start gap (seconds)")
    axes[1].set_ylabel("log10 later/earlier eval throughput")
    axes[1].set_title("Throughput phase contrast vs start gap")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Post-hoc phase diagnostics; seed and sweep phase are confounded")
    fig.tight_layout()
    drift_path = figures_dir / "run_order_and_throughput.png"
    fig.savefig(drift_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    trajectory = posthoc["trajectory_alignment"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for axis, basis, x_label in (
        (axes[0], trajectory["evaluation_aligned"], "Matched evaluation count"),
        (axes[1], trajectory["wall_time_aligned"], "Matched wall time (seconds)"),
    ):
        checkpoints = [
            row
            for row in basis["checkpoints"]
            if row["complete_topologies"] > 0
        ]
        x_values = [row["checkpoint"] for row in checkpoints]
        axis.plot(
            x_values,
            [row["topology_macro_seed_29_mean_loss"] for row in checkpoints],
            marker="o",
            color="#2563eb",
            label="seed 29 / first sweep",
        )
        axis.plot(
            x_values,
            [row["topology_macro_seed_31_mean_loss"] for row in checkpoints],
            marker="o",
            color="#dc2626",
            label="seed 31 / second sweep",
        )
        axis.axhline(4.0, color="#64748b", linestyle="--", linewidth=1.0)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Topology-macro best feasible loss")
        axis.grid(alpha=0.2)
    axes[0].set_title("Evaluation-aligned progress")
    axes[1].set_title("Wall-time-aligned progress")
    axes[0].legend(fontsize=8)
    fig.suptitle("History-only trajectory alignment; seed and sweep remain confounded")
    fig.tight_layout()
    trajectory_path = figures_dir / "trajectory_alignment.png"
    fig.savefig(trajectory_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return [topology_path, drift_path, target_path, loo_path, trajectory_path]
