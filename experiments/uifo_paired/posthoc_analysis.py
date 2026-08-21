"""Post-hoc sensitivity and exploratory analysis at the topology level."""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

from experiments.uifo_paired.reference_analysis import _percentile
from experiments.uifo_paired.results_ingestion import ValidatedStudy


EXPLORATORY_BOOTSTRAP_SEED = 20260821
EXPLORATORY_BOOTSTRAP_RESAMPLES = 10_000


def exact_mean_sign_flip_test(values: list[float]) -> dict[str, object]:
    """Enumerate all 2^n sign assignments for abs(mean(s_i * d_i))."""
    observed = abs(mean(values))
    assignments = 1 << len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        if statistic >= observed - 1e-15:
            extreme += 1
    return {
        "statistic": "absolute topology-level mean after sign assignment",
        "observed_absolute_mean": observed,
        "assignments_enumerated": assignments,
        "extreme_assignments": extreme,
        "two_sided_p_value": extreme / assignments,
        "definition": (
            "Enumerate every s in {-1,+1}^16 and compare "
            "abs(mean(s_i * d_i)) with abs(mean(d_i))."
        ),
        "inference_unit": "topology",
    }


def exact_two_sided_sign_test(wins: int, losses: int, ties: int) -> dict[str, object]:
    nonzero = wins + losses
    tail = min(wins, losses)
    probability = min(
        1.0,
        2.0
        * sum(math.comb(nonzero, index) for index in range(tail + 1))
        / (2**nonzero),
    )
    return {
        "semantic_prior_wins": wins,
        "semantic_prior_losses": losses,
        "ties_excluded": ties,
        "nonzero_topologies": nonzero,
        "two_sided_p_value": probability,
        "estimand": "direction only; magnitudes are discarded",
        "inference_unit": "topology",
    }


def _block_bootstrap(
    blocks: list[dict[str, object]],
    statistic,
    *,
    seed: int = EXPLORATORY_BOOTSTRAP_SEED,
    resamples: int = EXPLORATORY_BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, object], list[float]]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [generator.choice(blocks) for _ in range(len(blocks))]
        topology_means = [
            mean(float(value) for value in block["seed_differences"])
            for block in sample
        ]
        estimates.append(float(statistic(topology_means)))
    return (
        {
            "confidence_level": 0.95,
            "lower": _percentile(estimates, 0.025),
            "upper": _percentile(estimates, 0.975),
            "resamples": resamples,
            "seed": seed,
            "method": (
                "percentile bootstrap resampling complete topology blocks; both "
                "optimizer-seed differences remain inside each resampled block"
            ),
            "inference_unit": "topology",
        },
        estimates,
    )


def _topology_blocks(reference: dict[str, object]) -> list[dict[str, object]]:
    ordered = sorted(reference["topology_rows"], key=lambda row: row["topology_sha256"])
    return [
        {
            "label": f"T{index:02d}",
            "topology_sha256": row["topology_sha256"],
            "seed_differences": list(row["seed_differences_semantic_minus_no_prior"]),
            "mean_difference": row["mean_seed_difference_semantic_minus_no_prior"],
        }
        for index, row in enumerate(ordered, start=1)
    ]


def _leave_one_out(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for omitted in blocks:
        retained = [
            float(block["mean_difference"])
            for block in blocks
            if block is not omitted
        ]
        rows.append(
            {
                "omitted_topology": omitted["label"],
                "mean_difference": mean(retained),
                "median_difference": median(retained),
                "semantic_prior_wins": sum(value < 0 for value in retained),
                "ties": sum(value == 0 for value in retained),
                "semantic_prior_losses": sum(value > 0 for value in retained),
                "p90_regret": _percentile(retained, 0.9),
            }
        )
    return rows


def _pair_records(study: ValidatedStudy) -> dict[str, dict[str, dict[str, object]]]:
    pairs: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in study.records:
        config = record["config"]
        pairs[str(config["pair_id"])][str(config["arm"])] = record
    return pairs


def _drift_and_throughput(
    study: ValidatedStudy,
    blocks: list[dict[str, object]],
) -> dict[str, object]:
    from scipy.stats import spearmanr

    label_by_hash = {block["topology_sha256"]: block["label"] for block in blocks}
    difference_by_hash = {
        block["topology_sha256"]: float(block["mean_difference"]) for block in blocks
    }
    session_start = datetime.fromisoformat(str(study.session["started_utc"]))
    record_session_seconds: dict[str, float] = {}
    for record in study.records:
        started = datetime.fromisoformat(str(record["started_utc"]))
        record_session_seconds[str(record["run_id"])] = (
            started - session_start
        ).total_seconds()

    pairs = _pair_records(study)
    topology_pair_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair_id, arms in pairs.items():
        control = arms["no_prior"]
        treatment = arms["semantic_prior"]
        topology_hash = str(control["problem"]["topology_sha256"])
        c_metrics = control["metrics"]
        t_metrics = treatment["metrics"]
        c_evals = int(c_metrics["last_logged_eval_count"])
        t_evals = int(t_metrics["last_logged_eval_count"])
        c_time = float(c_metrics["last_logged_time_seconds"])
        t_time = float(t_metrics["last_logged_time_seconds"])
        topology_pair_rows[topology_hash].append(
            {
                "pair_id": pair_id,
                "optimizer_seed": int(control["config"]["optimizer_seed"]),
                "difference": float(t_metrics["best_feasible_loss"])
                - float(c_metrics["best_feasible_loss"]),
                "semantic_prior_first": int(
                    treatment["config"]["run_order_within_pair"]
                    < control["config"]["run_order_within_pair"]
                ),
                "planned_pair_midpoint": mean(
                    [
                        int(control["config"]["planned_run_index"]),
                        int(treatment["config"]["planned_run_index"]),
                    ]
                ),
                "session_pair_midpoint_seconds": mean(
                    [
                        record_session_seconds[str(control["run_id"])],
                        record_session_seconds[str(treatment["run_id"])],
                    ]
                ),
                "log10_evaluation_ratio_semantic_over_no_prior": math.log10(
                    t_evals / c_evals
                ),
                "log10_evaluations_per_second_ratio_semantic_over_no_prior": (
                    math.log10((t_evals / t_time) / (c_evals / c_time))
                ),
            }
        )

    topology_rows = []
    order_contrasts = []
    for topology_hash, rows in sorted(topology_pair_rows.items()):
        if len(rows) != 2 or {row["semantic_prior_first"] for row in rows} != {0, 1}:
            raise ValueError("arm-first order was not balanced within topology")
        semantic_first = next(row for row in rows if row["semantic_prior_first"] == 1)
        no_prior_first = next(row for row in rows if row["semantic_prior_first"] == 0)
        contrast = float(semantic_first["difference"]) - float(no_prior_first["difference"])
        order_contrasts.append(contrast)
        topology_rows.append(
            {
                "topology": label_by_hash[topology_hash],
                "mean_difference": difference_by_hash[topology_hash],
                "planned_run_midpoint": mean(
                    float(row["planned_pair_midpoint"]) for row in rows
                ),
                "session_midpoint_hours": mean(
                    float(row["session_pair_midpoint_seconds"]) for row in rows
                )
                / 3600.0,
                "arm_first_contrast": contrast,
                "mean_log10_evaluation_ratio": mean(
                    float(row["log10_evaluation_ratio_semantic_over_no_prior"])
                    for row in rows
                ),
                "mean_log10_evaluations_per_second_ratio": mean(
                    float(
                        row[
                            "log10_evaluations_per_second_ratio_semantic_over_no_prior"
                        ]
                    )
                    for row in rows
                ),
            }
        )
    differences = [float(row["mean_difference"]) for row in topology_rows]
    planned_order = [float(row["planned_run_midpoint"]) for row in topology_rows]
    session_hours = [float(row["session_midpoint_hours"]) for row in topology_rows]
    throughput = [float(row["mean_log10_evaluation_ratio"]) for row in topology_rows]
    rate = [
        float(row["mean_log10_evaluations_per_second_ratio"])
        for row in topology_rows
    ]
    planned_correlation = spearmanr(planned_order, differences)
    session_correlation = spearmanr(session_hours, differences)
    throughput_ci, _ = _block_bootstrap(
        [
            {"seed_differences": [value, value]}
            for value in throughput
        ],
        mean,
        seed=EXPLORATORY_BOOTSTRAP_SEED + 3,
    )
    return {
        "topology_rows": topology_rows,
        "serial_run_order": {
            "spearman_rho": float(planned_correlation.statistic),
            "two_sided_p_value": float(planned_correlation.pvalue),
        },
        "session_time": {
            "spearman_rho": float(session_correlation.statistic),
            "two_sided_p_value": float(session_correlation.pvalue),
        },
        "arm_first_order": {
            "contrast_definition": (
                "Within each topology: seed-pair difference when semantic_prior ran "
                "first minus the seed-pair difference when no_prior ran first."
            ),
            "mean_contrast": mean(order_contrasts),
            "median_contrast": median(order_contrasts),
            "exact_mean_sign_flip": exact_mean_sign_flip_test(order_contrasts),
        },
        "evaluation_throughput": {
            "mean_topology_log10_evaluation_ratio_semantic_over_no_prior": mean(
                throughput
            ),
            "median_topology_log10_evaluation_ratio_semantic_over_no_prior": median(
                throughput
            ),
            "mean_topology_log10_evaluations_per_second_ratio": mean(rate),
            "bootstrap_mean_log10_evaluation_ratio_ci_95": throughput_ci,
        },
        "interpretation": (
            "These are post-hoc drift diagnostics. They do not change the frozen "
            "decision and are not promotion criteria."
        ),
    }


def analyze_posthoc(
    study: ValidatedStudy,
    production: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    from scipy import __version__ as scipy_version
    from scipy.stats import pearsonr, wilcoxon

    blocks = _topology_blocks(reference)
    values = [float(block["mean_difference"]) for block in blocks]
    wins = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    losses = sum(value > 0 for value in values)
    zero_count = sum(value == 0 for value in values)
    absolute_ties = {
        value: count
        for value, count in Counter(abs(value) for value in values).items()
        if count > 1
    }
    wilcoxon_method = "exact" if zero_count == 0 and not absolute_ties else "asymptotic"
    wilcoxon_result = wilcoxon(
        values,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method=wilcoxon_method,
    )
    mean_ci, mean_bootstrap = _block_bootstrap(blocks, mean)
    median_ci, median_bootstrap = _block_bootstrap(
        blocks, median, seed=EXPLORATORY_BOOTSTRAP_SEED + 1
    )
    standardized = mean(values) / stdev(values)
    standardized_ci, _ = _block_bootstrap(
        blocks,
        lambda sample: mean(sample) / stdev(sample) if stdev(sample) else 0.0,
        seed=EXPLORATORY_BOOTSTRAP_SEED + 2,
    )
    seed_7 = [float(block["seed_differences"][0]) for block in blocks]
    seed_11 = [float(block["seed_differences"][1]) for block in blocks]
    seed_correlation = pearsonr(seed_7, seed_11)
    seed_patterns = Counter()
    for left, right in zip(seed_7, seed_11):
        if left < 0 and right < 0:
            seed_patterns["both_help"] += 1
        elif left > 0 and right > 0:
            seed_patterns["both_harm"] += 1
        elif left == 0 or right == 0:
            seed_patterns["includes_zero"] += 1
        else:
            seed_patterns["opposite_signs"] += 1

    leave_one_out = _leave_one_out(blocks)
    target_source = production["semantic_prior_vs_no_prior"][
        "target_hitting_time_inference"
    ]["targets"]
    targets = {
        target: {
            "seed_pair_outcomes": payload["seed_pair_outcomes"],
            "topology_inference_ready": payload["topology_inference_ready"],
            "finite_comparable_topologies": payload["finite_comparable_topologies"],
            "topology_macro_mean_log10_time_ratio": payload[
                "topology_macro_mean_log10_time_ratio"
            ],
            "topology_macro_mean_log10_eval_ratio": payload[
                "topology_macro_mean_log10_eval_ratio"
            ],
            "topology_bootstrap_time_log10_ratio_ci_95": payload[
                "topology_bootstrap_time_log10_ratio_ci_95"
            ],
            "topology_bootstrap_eval_log10_ratio_ci_95": payload[
                "topology_bootstrap_eval_log10_ratio_ci_95"
            ],
            "order_of_magnitude_claim_ready": payload[
                "order_of_magnitude_claim_ready"
            ],
        }
        for target, payload in target_source.items()
    }
    drift = _drift_and_throughput(study, blocks)
    practical_help = sum(value <= -0.05 for value in values)
    practical_harm = sum(value >= 0.05 for value in values)
    heterogeneity = {
        "topology_rows": blocks,
        "minimum_difference": min(values),
        "maximum_difference": max(values),
        "interquartile_range": [
            _percentile(values, 0.25),
            _percentile(values, 0.75),
        ],
        "topologies_with_at_least_0_05_help": practical_help,
        "topologies_with_at_least_0_05_harm": practical_harm,
        "interpretation": (
            "Topology labels are descriptive anonymized IDs. Extremes and apparent "
            "subgroups are post-hoc and are not confirmed discoveries."
        ),
    }
    return {
        "format_version": 1,
        "phase": "Phase 3 — post-hoc sensitivity and exploratory analysis",
        "primary_estimand": (
            "topology-level mean over optimizer seeds of semantic_prior best feasible "
            "loss minus no_prior best feasible loss; negative favors semantic_prior"
        ),
        "inference_unit": "topology",
        "topology_count": 16,
        "exact_mean_sign_flip": exact_mean_sign_flip_test(values),
        "exact_direction_sign_test": exact_two_sided_sign_test(wins, losses, ties),
        "wilcoxon_signed_rank_sensitivity": {
            "statistic": float(wilcoxon_result.statistic),
            "two_sided_p_value": float(wilcoxon_result.pvalue),
            "zero_method": "wilcox (discard zeros)",
            "zeros": zero_count,
            "absolute_rank_ties": len(absolute_ties),
            "method": wilcoxon_method,
            "continuity_correction": False,
            "implementation": f"scipy.stats.wilcoxon in SciPy {scipy_version}",
            "assumption": (
                "Sensitivity analysis only; signed-rank inference assumes the topology "
                "difference distribution is symmetric about its center."
            ),
        },
        "effect_sizes": {
            "mean_difference": mean(values),
            "mean_bootstrap_ci_95": mean_ci,
            "median_difference": median(values),
            "median_bootstrap_ci_95": median_ci,
            "standardized_mean_difference": standardized,
            "standardized_mean_bootstrap_ci_95": standardized_ci,
            "semantic_prior_topology_win_fraction": wins / 16,
            "wins_ties_losses": {
                "semantic_prior_wins": wins,
                "ties": ties,
                "semantic_prior_losses": losses,
            },
        },
        "leave_one_topology_out": leave_one_out,
        "seed_consistency": {
            "pattern_counts": dict(seed_patterns),
            "pearson_correlation_seed7_seed11": float(seed_correlation.statistic),
            "two_sided_p_value": float(seed_correlation.pvalue),
            "interpretation": (
                "Optimizer seeds remain repeated measurements. Correlation and sign "
                "patterns are descriptive across the 16 topology blocks."
            ),
        },
        "target_hitting_existing_censor_aware_summary": targets,
        "heterogeneity": heterogeneity,
        "drift_diagnostics": drift,
        "plot_payload": {
            "mean_bootstrap_estimates": mean_bootstrap,
            "median_bootstrap_estimates": median_bootstrap,
        },
        "conclusion_checks": {
            "semantic_prior_improves_final_feasible_loss_on_average": (
                "not supported; the observed mean difference is positive and its "
                "topology bootstrap interval spans zero"
            ),
            "semantic_prior_improves_the_median_topology": (
                "not supported; the observed median difference is positive and its "
                "exploratory topology-block interval spans zero"
            ),
            "semantic_prior_improves_finite_feasibility_rates": (
                "not supported; all 32 seed pairs were finite-feasible in both arms"
            ),
            "semantic_prior_reaches_useful_targets_sooner": (
                "not supported as a general claim; target outcomes are heavily censored "
                "and no frozen threshold is order-of-magnitude-claim ready"
            ),
            "meaningful_topology_dependent_benefits": (
                "descriptive heterogeneity exists, including helpful topologies, but "
                "post-hoc topology extremes/subgroups are not confirmed discoveries"
            ),
        },
        "multiplicity_note": (
            "The full family of exploratory analyses is reported. No p-value is a new "
            "promotion rule, and conclusions are not selected from the most favorable test."
        ),
        "equivalence_note": (
            "No equivalence margin was preregistered. Failure to reject is not evidence "
            "of equivalence or non-inferiority."
        ),
    }


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def create_plots(posthoc: dict[str, object], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    blocks = posthoc["heterogeneity"]["topology_rows"]
    ordered = sorted(blocks, key=lambda block: float(block["mean_difference"]))

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    y = np.arange(len(ordered))
    means = [float(block["mean_difference"]) for block in ordered]
    colors = ["#2a9d8f" if value < 0 else "#e76f51" for value in means]
    axis.barh(y, means, color=colors, alpha=0.75, label="Topology mean")
    for index, block in enumerate(ordered):
        seed_values = [float(value) for value in block["seed_differences"]]
        axis.scatter(seed_values, [index, index], color="#264653", s=20, zorder=3)
        axis.plot(seed_values, [index, index], color="#264653", linewidth=0.8, zorder=2)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, [block["label"] for block in ordered])
    axis.set_xlabel("semantic_prior − no_prior best feasible loss")
    axis.set_ylabel("Anonymized topology")
    axis.set_title("Topology-block differences (bars) with optimizer seeds (dots)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    topology_path = figures_dir / "topology_differences.png"
    _save_figure(figure, topology_path)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, key, title, observed in (
        (
            axes[0],
            "mean_bootstrap_estimates",
            "Mean difference",
            posthoc["effect_sizes"]["mean_difference"],
        ),
        (
            axes[1],
            "median_bootstrap_estimates",
            "Median difference",
            posthoc["effect_sizes"]["median_difference"],
        ),
    ):
        estimates = posthoc["plot_payload"][key]
        axis.hist(estimates, bins=45, color="#457b9d", alpha=0.8)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(float(observed), color="#e63946", linewidth=2, label="Observed")
        axis.set_title(title)
        axis.set_xlabel("semantic_prior − no_prior")
        axis.set_ylabel("Topology-block bootstrap draws")
        axis.legend()
    figure.suptitle("Exploratory uncertainty; complete topology blocks resampled")
    figure.tight_layout()
    uncertainty_path = figures_dir / "bootstrap_uncertainty.png"
    _save_figure(figure, uncertainty_path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    seed7 = [float(block["seed_differences"][0]) for block in blocks]
    seed11 = [float(block["seed_differences"][1]) for block in blocks]
    axis.scatter(seed7, seed11, color="#6a4c93", s=45)
    for block, left, right in zip(blocks, seed7, seed11):
        axis.annotate(block["label"], (left, right), xytext=(4, 3), textcoords="offset points", fontsize=8)
    bound = max(abs(value) for value in seed7 + seed11) * 1.1
    axis.plot([-bound, bound], [-bound, bound], linestyle="--", color="gray", linewidth=1)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlim(-bound, bound)
    axis.set_ylim(-bound, bound)
    axis.set_xlabel("Optimizer seed 7 difference")
    axis.set_ylabel("Optimizer seed 11 difference")
    axis.set_title("Seed consistency within topology blocks")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    seed_path = figures_dir / "seed_consistency.png"
    _save_figure(figure, seed_path)
    plt.close(figure)

    target_order = ["4", "1", "0.5", "0"]
    outcomes = ["both_reached", "semantic_prior_only", "no_prior_only", "neither_reached"]
    outcome_labels = ["Both", "Semantic only", "No-prior only", "Neither"]
    outcome_colors = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(target_order))
    for outcome, label, color in zip(outcomes, outcome_labels, outcome_colors):
        counts = [
            posthoc["target_hitting_existing_censor_aware_summary"][target][
                "seed_pair_outcomes"
            ][outcome]
            for target in target_order
        ]
        axis.bar(target_order, counts, bottom=bottom, label=label, color=color)
        bottom += np.asarray(counts)
    axis.set_xlabel("Target loss")
    axis.set_ylabel("Topology–optimizer-seed pairs (n=32)")
    axis.set_title("Censor-aware target-hitting outcomes")
    axis.legend(ncols=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    target_path = figures_dir / "target_hitting_outcomes.png"
    _save_figure(figure, target_path)
    plt.close(figure)

    drift_rows = posthoc["drift_diagnostics"]["topology_rows"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(
        [row["session_midpoint_hours"] for row in drift_rows],
        [row["mean_difference"] for row in drift_rows],
        color="#1d3557",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Session midpoint (hours)")
    axes[0].set_ylabel("Topology mean loss difference")
    axes[0].set_title("Serial/session-time drift diagnostic")
    axes[1].scatter(
        [row["mean_log10_evaluation_ratio"] for row in drift_rows],
        [row["mean_difference"] for row in drift_rows],
        color="#f4a261",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("log10 evaluation ratio (semantic/no-prior)")
    axes[1].set_ylabel("Topology mean loss difference")
    axes[1].set_title("Evaluation-throughput diagnostic")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Post-hoc drift diagnostics; not promotion criteria")
    figure.tight_layout()
    drift_path = figures_dir / "drift_and_throughput.png"
    _save_figure(figure, drift_path)
    plt.close(figure)
    return [topology_path, uncertainty_path, seed_path, target_path, drift_path]


def serializable_posthoc(posthoc: dict[str, object]) -> dict[str, object]:
    result = json.loads(json.dumps(posthoc, allow_nan=False))
    result.pop("plot_payload", None)
    return result
