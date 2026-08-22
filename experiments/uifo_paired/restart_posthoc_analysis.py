"""Post-hoc sensitivity and exploratory analysis for the restart screen."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)


EXPLORATORY_SEED = 20260821
EXPLORATORY_RESAMPLES = 10_000
TIE_TOLERANCE = 1e-12


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_blocks(
    blocks: list[list[float]],
    statistic,
    *,
    seed: int,
) -> dict[str, object]:
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(EXPLORATORY_RESAMPLES):
        sample = [generator.choice(blocks) for _ in blocks]
        topology_means = [mean(block) for block in sample]
        estimates.append(float(statistic(topology_means)))
    return {
        "confidence_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "resamples": EXPLORATORY_RESAMPLES,
        "seed": seed,
        "method": (
            "percentile bootstrap over complete topology blocks; both optimizer "
            "seeds remain within every resampled topology"
        ),
        "inference_unit": "topology",
    }


def _sign_flip(values: list[float]) -> dict[str, object]:
    observed = abs(mean(values))
    extreme = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        extreme += statistic >= observed - 1e-15
        assignments += 1
    return {
        "statistic": "absolute topology-level mean after sign assignment",
        "observed_absolute_mean": observed,
        "assignments_enumerated": assignments,
        "extreme_assignments": extreme,
        "two_sided_p_value": extreme / assignments,
        "definition": (
            f"Enumerate every s in {{-1,+1}}^{len(values)} and compare "
            "abs(mean(s_i*d_i)) with abs(mean(d_i))."
        ),
        "inference_unit": "topology",
    }


def _sign_test(wins: int, losses: int, ties: int) -> dict[str, object]:
    nonzero = wins + losses
    tail = min(wins, losses)
    p_value = (
        min(
            1.0,
            2.0
            * sum(math.comb(nonzero, index) for index in range(tail + 1))
            / (2**nonzero),
        )
        if nonzero
        else None
    )
    return {
        "p200_wins": wins,
        "p200_losses": losses,
        "ties_excluded": ties,
        "nonzero_topologies": nonzero,
        "two_sided_p_value": p_value,
        "estimand": "direction only; magnitudes are discarded",
        "inference_unit": "topology",
    }


def _finite_correlation(result: object) -> dict[str, float | None]:
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    return {
        "statistic": statistic if math.isfinite(statistic) else None,
        "two_sided_p_value": p_value if math.isfinite(p_value) else None,
    }


def _safe_spearman(
    left: list[float], right: list[float], implementation
) -> dict[str, float | None]:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {"statistic": None, "two_sided_p_value": None}
    return _finite_correlation(implementation(left, right))


def _target_key(value: float) -> str:
    return format(float(value), ".12g")


def _target_hitting(
    study: ValidatedStudy,
    label_by_hash: dict[str, str],
) -> dict[str, object]:
    pairs: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in study.records:
        config = record["config"]
        pairs[str(config["pair_id"])][str(config["arm"])] = record
    targets = list(next(iter(study.configs.values()))["target_losses"])
    output: dict[str, object] = {}
    for target in targets:
        key = _target_key(float(target))
        outcomes = {
            "both_reached": 0,
            "p200_only": 0,
            "p600_only": 0,
            "neither_reached": 0,
        }
        topology_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        for arms in pairs.values():
            control = arms["no_prior_p600"]
            treatment = arms["no_prior_p200"]
            control_hit = control["metrics"]["targets"][key]
            treatment_hit = treatment["metrics"]["targets"][key]
            control_reached = control_hit["time_seconds"] is not None
            treatment_reached = treatment_hit["time_seconds"] is not None
            if control_reached and treatment_reached:
                outcomes["both_reached"] += 1
                c_time = float(control_hit["time_seconds"])
                t_time = float(treatment_hit["time_seconds"])
                c_evals = int(control_hit["eval_count"])
                t_evals = int(treatment_hit["eval_count"])
                if min(c_time, t_time, c_evals, t_evals) <= 0:
                    raise StudyValidationError("target ratios require positive observations")
                topology_hash = str(control["problem"]["topology_sha256"])
                topology_rows[topology_hash].append(
                    {
                        "log10_time_ratio_p200_over_p600": math.log10(t_time / c_time),
                        "log10_eval_ratio_p200_over_p600": math.log10(t_evals / c_evals),
                    }
                )
            elif treatment_reached:
                outcomes["p200_only"] += 1
            elif control_reached:
                outcomes["p600_only"] += 1
            else:
                outcomes["neither_reached"] += 1
        complete = {
            topology_hash: rows
            for topology_hash, rows in topology_rows.items()
            if len(rows) == 2
        }
        time_blocks = [
            [row["log10_time_ratio_p200_over_p600"] for row in rows]
            for _, rows in sorted(complete.items())
        ]
        eval_blocks = [
            [row["log10_eval_ratio_p200_over_p600"] for row in rows]
            for _, rows in sorted(complete.items())
        ]
        output[key] = {
            "target_loss": float(target),
            "seed_pair_outcomes": outcomes,
            "finite_comparable_topologies": len(complete),
            "topology_inference_ready": len(complete) == 8,
            "topology_macro_mean_log10_time_ratio": (
                mean(mean(block) for block in time_blocks) if time_blocks else None
            ),
            "topology_macro_mean_log10_eval_ratio": (
                mean(mean(block) for block in eval_blocks) if eval_blocks else None
            ),
            "topology_bootstrap_time_log10_ratio_ci_95": (
                _bootstrap_blocks(time_blocks, mean, seed=EXPLORATORY_SEED + 20)
                if len(time_blocks) == 8
                else None
            ),
            "topology_bootstrap_eval_log10_ratio_ci_95": (
                _bootstrap_blocks(eval_blocks, mean, seed=EXPLORATORY_SEED + 21)
                if len(eval_blocks) == 8
                else None
            ),
            "complete_topology_labels": [
                label_by_hash[topology_hash] for topology_hash in sorted(complete)
            ],
            "censoring_rule": (
                "Unreached targets remain censored. No finite time or evaluation "
                "count is imputed. Ratios use only both-reached seed pairs."
            ),
            "changes_frozen_decision": False,
        }
    return output


def analyze_restart_posthoc(
    study: ValidatedStudy,
    production: dict[str, object],
    reference: dict[str, object],
    *,
    frozen_agreement: dict[str, object],
) -> dict[str, object]:
    if (
        frozen_agreement.get("status") != "matched"
        or frozen_agreement.get("archived_summary_compared") is not True
        or frozen_agreement.get("topology_values_compared") != 8
        or frozen_agreement.get("seed_pairs_compared") != 16
        or frozen_agreement.get("frozen_criteria_compared") != 11
    ):
        raise StudyValidationError(
            "post-hoc analysis requires matched frozen production, reference, "
            "and archived-summary evidence"
        )
    for record in study.records:
        metrics = record.get("metrics")
        best = metrics.get("best_feasible_loss") if isinstance(metrics, dict) else None
        if (
            isinstance(best, bool)
            or not isinstance(best, (int, float))
            or not math.isfinite(float(best))
        ):
            raise StudyValidationError(
                "post-hoc analysis requires a finite feasible loss for every run"
            )
    """Calculate sensitivity analyses after the three-way result gate opens."""
    from scipy import __version__ as scipy_version
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    topology_rows = sorted(
        reference["topology_differences"], key=lambda row: str(row["topology_sha256"])
    )
    if len(topology_rows) != 8:
        raise StudyValidationError("restart exploratory analysis requires 8 topologies")
    labels = {
        str(row["topology_sha256"]): f"T{index:02d}"
        for index, row in enumerate(topology_rows, start=1)
    }
    blocks = [
        [
            float(seed_row["difference_p200_minus_p600"])
            for seed_row in row["seed_pair_rows"]
        ]
        for row in topology_rows
    ]
    values = [mean(block) for block in blocks]
    wins = sum(value < -TIE_TOLERANCE for value in values)
    ties = sum(abs(value) <= TIE_TOLERANCE for value in values)
    losses = sum(value > TIE_TOLERANCE for value in values)
    zeros = sum(value == 0.0 for value in values)
    absolute_ties = {
        value: count
        for value, count in Counter(abs(value) for value in values).items()
        if count > 1
    }
    method = "exact" if zeros == 0 and not absolute_ties else "asymptotic"
    wilcoxon_result = wilcoxon(
        values,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method=method,
    )

    seed_19 = [block[0] for block in blocks]
    seed_23 = [block[1] for block in blocks]
    seed_patterns = Counter()
    for left, right in zip(seed_19, seed_23):
        if left < 0 and right < 0:
            seed_patterns["both_help"] += 1
        elif left > 0 and right > 0:
            seed_patterns["both_harm"] += 1
        elif left == 0 or right == 0:
            seed_patterns["includes_zero"] += 1
        else:
            seed_patterns["opposite_signs"] += 1

    leave_one_out = []
    for index, value in enumerate(values):
        retained = values[:index] + values[index + 1 :]
        leave_one_out.append(
            {
                "omitted_topology": f"T{index + 1:02d}",
                "mean_difference": mean(retained),
                "median_difference": median(retained),
                "p200_wins": sum(item < -TIE_TOLERANCE for item in retained),
                "ties": sum(abs(item) <= TIE_TOLERANCE for item in retained),
                "p200_losses": sum(item > TIE_TOLERANCE for item in retained),
                "p90_regret": _percentile(retained, 0.9),
            }
        )

    records_by_pair: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in study.records:
        config = record["config"]
        records_by_pair[str(config["pair_id"])][str(config["arm"])] = record
    session_start = datetime.fromisoformat(str(study.session["started_utc"]))
    drift_rows: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for pair in records_by_pair.values():
        control = pair["no_prior_p600"]
        treatment = pair["no_prior_p200"]
        control_rows = study.history_rows[str(control["run_id"])]
        treatment_rows = study.history_rows[str(treatment["run_id"])]
        c_evals = max(int(row["eval_count_after_call"]) for row in control_rows)
        t_evals = max(int(row["eval_count_after_call"]) for row in treatment_rows)
        c_time = max(float(row["time_seconds"]) for row in control_rows)
        t_time = max(float(row["time_seconds"]) for row in treatment_rows)
        c_started = datetime.fromisoformat(str(control["started_utc"]))
        t_started = datetime.fromisoformat(str(treatment["started_utc"]))
        topology_hash = str(control["problem"]["topology_sha256"])
        drift_rows[topology_hash].append(
            {
                "difference": float(treatment["metrics"]["best_feasible_loss"])
                - float(control["metrics"]["best_feasible_loss"]),
                "p200_first": int(treatment["config"]["run_order_within_pair"])
                < int(control["config"]["run_order_within_pair"]),
                "planned_midpoint": mean(
                    (
                        int(control["config"]["planned_run_index"]),
                        int(treatment["config"]["planned_run_index"]),
                    )
                ),
                "session_midpoint_hours": mean(
                    (
                        (c_started - session_start).total_seconds(),
                        (t_started - session_start).total_seconds(),
                    )
                )
                / 3600.0,
                "log10_eval_ratio": math.log10(t_evals / c_evals),
                "log10_eval_rate_ratio": math.log10((t_evals / t_time) / (c_evals / c_time)),
            }
        )
    topology_drift = []
    arm_first_contrasts = []
    for topology_hash, rows in sorted(drift_rows.items()):
        if len(rows) != 2 or {bool(row["p200_first"]) for row in rows} != {False, True}:
            raise StudyValidationError("arm-first balance is broken within topology")
        first = next(row for row in rows if row["p200_first"])
        second = next(row for row in rows if not row["p200_first"])
        contrast = float(first["difference"]) - float(second["difference"])
        arm_first_contrasts.append(contrast)
        topology_drift.append(
            {
                "topology": labels[topology_hash],
                "mean_difference": mean(float(row["difference"]) for row in rows),
                "planned_run_midpoint": mean(
                    float(row["planned_midpoint"]) for row in rows
                ),
                "session_midpoint_hours": mean(
                    float(row["session_midpoint_hours"]) for row in rows
                ),
                "arm_first_contrast": contrast,
                "mean_log10_evaluation_ratio": mean(
                    float(row["log10_eval_ratio"]) for row in rows
                ),
                "mean_log10_evaluation_rate_ratio": mean(
                    float(row["log10_eval_rate_ratio"]) for row in rows
                ),
            }
        )
    order = [float(row["planned_run_midpoint"]) for row in topology_drift]
    session = [float(row["session_midpoint_hours"]) for row in topology_drift]
    drift_values = [float(row["mean_difference"]) for row in topology_drift]
    eval_ratios = [
        float(row["mean_log10_evaluation_ratio"]) for row in topology_drift
    ]
    rate_ratios = [
        float(row["mean_log10_evaluation_rate_ratio"]) for row in topology_drift
    ]
    topology_output = [
        {
            "topology": labels[str(row["topology_sha256"])],
            "seed_19_difference": blocks[index][0],
            "seed_23_difference": blocks[index][1],
            "mean_difference": values[index],
        }
        for index, row in enumerate(topology_rows)
    ]
    standardized = mean(values) / stdev(values) if stdev(values) else 0.0
    return {
        "format_version": 1,
        "phase": "Phase 3 — post-hoc sensitivity and exploratory analysis",
        "primary_estimand": (
            "topology-level mean over optimizer seeds of p200 best feasible loss "
            "minus p600 best feasible loss; negative favors patience 200"
        ),
        "inference_unit": "topology",
        "topology_count": 8,
        "exact_mean_sign_flip": _sign_flip(values),
        "exact_direction_sign_test": _sign_test(wins, losses, ties),
        "wilcoxon_signed_rank_sensitivity": {
            "statistic": float(wilcoxon_result.statistic),
            "two_sided_p_value": float(wilcoxon_result.pvalue),
            "zero_method": "wilcox (discard zeros)",
            "zeros": zeros,
            "absolute_rank_ties": len(absolute_ties),
            "method": method,
            "continuity_correction": False,
            "implementation": f"scipy.stats.wilcoxon in SciPy {scipy_version}",
            "assumption": (
                "Sensitivity analysis only; signed-rank inference assumes a "
                "symmetric topology-difference distribution."
            ),
        },
        "effect_sizes": {
            "mean_difference": mean(values),
            "mean_bootstrap_ci_95": _bootstrap_blocks(
                blocks, mean, seed=EXPLORATORY_SEED
            ),
            "median_difference": median(values),
            "median_bootstrap_ci_95": _bootstrap_blocks(
                blocks, median, seed=EXPLORATORY_SEED + 1
            ),
            "standardized_mean_difference": standardized,
            "wins_ties_losses": production["wins_ties_losses"],
        },
        "leave_one_topology_out": leave_one_out,
        "seed_consistency": {
            "seed_19_mean_difference": mean(seed_19),
            "seed_23_mean_difference": mean(seed_23),
            "pearson_seed_19_vs_23": _finite_correlation(
                pearsonr(seed_19, seed_23)
            ),
            "pattern_counts": dict(seed_patterns),
        },
        "drift_diagnostics": {
            "topology_rows": topology_drift,
            "serial_run_order": _safe_spearman(order, drift_values, spearmanr),
            "session_time": _safe_spearman(session, drift_values, spearmanr),
            "arm_first_order": {
                "mean_contrast": mean(arm_first_contrasts),
                "median_contrast": median(arm_first_contrasts),
                "exact_mean_sign_flip": _sign_flip(arm_first_contrasts),
            },
            "evaluation_throughput": {
                "mean_topology_log10_evaluation_ratio_p200_over_p600": mean(
                    eval_ratios
                ),
                "median_topology_log10_evaluation_ratio_p200_over_p600": median(
                    eval_ratios
                ),
                "mean_topology_log10_evaluation_rate_ratio": mean(rate_ratios),
                "bootstrap_mean_log10_evaluation_ratio_ci_95": _bootstrap_blocks(
                    [[value, value] for value in eval_ratios],
                    mean,
                    seed=EXPLORATORY_SEED + 3,
                ),
            },
            "changes_frozen_decision": False,
        },
        "target_hitting": _target_hitting(study, labels),
        "heterogeneity": {
            "topology_rows": topology_output,
            "minimum_difference": min(values),
            "maximum_difference": max(values),
            "interquartile_range": [
                _percentile(values, 0.25),
                _percentile(values, 0.75),
            ],
            "topologies_with_at_least_0_05_help": sum(
                value <= -0.05 for value in values
            ),
            "topologies_with_at_least_0_05_harm": sum(
                value >= 0.05 for value in values
            ),
            "interpretation": (
                "Anonymized topology patterns are descriptive and post-hoc; they "
                "are not confirmed subgroups."
            ),
        },
        "changes_frozen_decision": False,
    }


def create_restart_plots(posthoc: dict[str, object], output_dir: Path) -> list[Path]:
    """Render the four required private diagnostic figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = posthoc["heterogeneity"]["topology_rows"]
    labels = [str(row["topology"]) for row in rows]
    means = [float(row["mean_difference"]) for row in rows]
    seed_19 = [float(row["seed_19_difference"]) for row in rows]
    seed_23 = [float(row["seed_23_difference"]) for row in rows]
    y = list(range(len(rows)))
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, (left, right) in enumerate(zip(seed_19, seed_23)):
        ax.plot([left, right], [index, index], color="#9fb3c8", linewidth=1.5)
    ax.scatter(seed_19, y, label="seed 19", color="#2a9d8f", marker="o")
    ax.scatter(seed_23, y, label="seed 23", color="#e76f51", marker="s")
    ax.scatter(means, y, label="topology mean", color="#102a43", marker="D")
    ax.axvline(0.0, color="#52616b", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("best feasible loss difference: p200 - p600")
    ax.set_title("Topology effects with both optimizer seeds")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "topology_differences.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    ci = posthoc["effect_sizes"]["mean_bootstrap_ci_95"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar(
        [float(posthoc["effect_sizes"]["mean_difference"])],
        [0],
        xerr=[[
            float(posthoc["effect_sizes"]["mean_difference"]) - float(ci["lower"])
        ], [
            float(ci["upper"]) - float(posthoc["effect_sizes"]["mean_difference"])
        ]],
        fmt="D",
        color="#264653",
        capsize=6,
    )
    ax.axvline(0.0, color="#52616b", linestyle="--", linewidth=1)
    ax.set_yticks([0], ["topology-block mean"])
    ax.set_xlabel("p200 - p600 loss difference (95% topology bootstrap interval)")
    ax.set_title("Mean-effect uncertainty")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "mean_uncertainty.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(seed_19, seed_23, color="#6a4c93")
    for label, left, right in zip(labels, seed_19, seed_23):
        ax.annotate(label, (left, right), xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.0, color="#52616b", linewidth=1)
    ax.axvline(0.0, color="#52616b", linewidth=1)
    ax.set_xlabel("seed 19 difference")
    ax.set_ylabel("seed 23 difference")
    ax.set_title("Within-topology seed consistency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "seed_consistency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    targets = posthoc["target_hitting"]
    target_labels = list(targets)
    categories = (
        ("both_reached", "both", "#2a9d8f"),
        ("p200_only", "p200 only", "#457b9d"),
        ("p600_only", "p600 only", "#e9c46a"),
        ("neither_reached", "neither", "#e76f51"),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottoms = [0] * len(target_labels)
    for key, label, color in categories:
        counts = [int(targets[target]["seed_pair_outcomes"][key]) for target in target_labels]
        ax.bar(target_labels, counts, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]
    ax.set_xlabel("target loss")
    ax.set_ylabel("optimizer-seed pairs (n=16)")
    ax.set_title("Censor-aware target-hitting outcomes")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path = output_dir / "target_hitting_outcomes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths
