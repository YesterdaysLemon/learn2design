"""History-only trajectory alignment for ``submission-like-screen-v1``.

The frozen decision is already complete before this module runs.  This
diagnostic uses authenticated history rows to compare the two repeated seeds
at common evaluation counts and common wall times.  Topology remains the
inference unit and missing feasible values are never imputed.
"""

from __future__ import annotations

import hashlib
import math
from statistics import mean, median

from experiments.uifo_paired.results_ingestion import (
    StudyValidationError,
    ValidatedStudy,
)


EXPECTED_SEEDS = (29, 31)
CHECKPOINT_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
TIE_TOLERANCE = 1e-12


def _history_curve(
    rows: list[dict[str, object]], *, label: str
) -> list[dict[str, float | int | None]]:
    if not rows:
        raise StudyValidationError(f"trajectory history is empty: {label}")

    calls: list[list[dict[str, object]]] = []
    current_call: int | None = None
    current_rows: list[dict[str, object]] = []
    for row in rows:
        call_index = row.get("call_index")
        if type(call_index) is not int:
            raise StudyValidationError(f"trajectory call index is invalid: {label}")
        if current_call is None or call_index != current_call:
            if current_rows:
                calls.append(current_rows)
            if current_call is not None and call_index <= current_call:
                raise StudyValidationError(
                    f"trajectory call order is not strictly increasing: {label}"
                )
            current_call = call_index
            current_rows = []
        current_rows.append(row)
    calls.append(current_rows)

    curve: list[dict[str, float | int | None]] = []
    running_best: float | None = None
    previous_time = -math.inf
    previous_evals = -1
    for call_rows in calls:
        times = {float(row["time_seconds"]) for row in call_rows}
        evals = {int(row["eval_count_after_call"]) for row in call_rows}
        candidates = [row.get("candidate_index") for row in call_rows]
        if len(times) != 1 or len(evals) != 1:
            raise StudyValidationError(
                f"trajectory rows disagree inside an objective call: {label}"
            )
        if any(type(value) is not int for value in candidates) or len(
            set(candidates)
        ) != len(candidates):
            raise StudyValidationError(
                f"trajectory candidate indexes are invalid: {label}"
            )
        time_value = next(iter(times))
        eval_value = next(iter(evals))
        if (
            not math.isfinite(time_value)
            or time_value < previous_time
            or eval_value <= previous_evals
        ):
            raise StudyValidationError(
                f"trajectory resource coordinates are invalid: {label}"
            )
        previous_time = time_value
        previous_evals = eval_value

        feasible_losses = []
        for row in call_rows:
            feasible = row.get("is_feasible")
            if type(feasible) is not bool:
                raise StudyValidationError(
                    f"trajectory feasibility is not strict boolean: {label}"
                )
            loss = row.get("loss")
            if loss is not None and (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise StudyValidationError(
                    f"trajectory contains an invalid finite loss: {label}"
                )
            if feasible and loss is not None:
                feasible_losses.append(float(loss))
        if feasible_losses:
            call_best = min(feasible_losses)
            running_best = (
                call_best if running_best is None else min(running_best, call_best)
            )
        curve.append(
            {
                "time_seconds": time_value,
                "eval_count": eval_value,
                "best_feasible_loss": running_best,
            }
        )
    return curve


def _snapshot(
    curve: list[dict[str, float | int | None]], coordinate: str, point: float | int
) -> float | None:
    best = None
    for row in curve:
        coordinate_value = row[coordinate]
        assert isinstance(coordinate_value, (int, float))
        if coordinate_value > point:
            break
        best = row["best_feasible_loss"]
    return None if best is None else float(best)


def _evaluation_checkpoints(
    curves: dict[tuple[str, int], list[dict[str, float | int | None]]]
) -> list[tuple[float, int]]:
    shared: set[int] | None = None
    for curve in curves.values():
        values = {int(row["eval_count"]) for row in curve}
        shared = values if shared is None else shared & values
    if not shared:
        raise StudyValidationError("trajectory histories have no common evaluation count")
    ordered = sorted(shared)
    maximum = ordered[-1]
    selected = []
    for fraction in CHECKPOINT_FRACTIONS:
        eligible = [value for value in ordered if value <= maximum * fraction]
        value = eligible[-1] if eligible else ordered[0]
        selected.append((fraction, value))
    return selected


def _time_checkpoints(
    curves: dict[tuple[str, int], list[dict[str, float | int | None]]]
) -> list[tuple[float, float]]:
    common_terminal = min(float(curve[-1]["time_seconds"]) for curve in curves.values())
    if not math.isfinite(common_terminal) or common_terminal <= 0:
        raise StudyValidationError("trajectory common terminal time is invalid")
    return [
        (fraction, round(common_terminal * fraction, 9))
        for fraction in CHECKPOINT_FRACTIONS
    ]


def _checkpoint_summary(
    *,
    basis: str,
    coordinate: str,
    checkpoint: float | int,
    progress_fraction: float,
    labels: list[str],
    curves: dict[tuple[str, int], list[dict[str, float | int | None]]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paired = []
    for label in labels:
        seed_29 = _snapshot(curves[(label, 29)], coordinate, checkpoint)
        seed_31 = _snapshot(curves[(label, 31)], coordinate, checkpoint)
        difference = (
            None if seed_29 is None or seed_31 is None else seed_31 - seed_29
        )
        paired.append(
            {
                "basis": basis,
                "progress_fraction": progress_fraction,
                "checkpoint": checkpoint,
                "topology_label": label,
                "seed_29_best_feasible_loss": seed_29,
                "seed_31_best_feasible_loss": seed_31,
                "seed_31_minus_seed_29_loss": difference,
                "complete_pair": difference is not None,
            }
        )

    complete = [row for row in paired if row["complete_pair"]]
    differences = [float(row["seed_31_minus_seed_29_loss"]) for row in complete]
    seed_29_values = [float(row["seed_29_best_feasible_loss"]) for row in complete]
    seed_31_values = [float(row["seed_31_best_feasible_loss"]) for row in complete]
    summary = {
        "progress_fraction": progress_fraction,
        "checkpoint": checkpoint,
        "complete_topologies": len(complete),
        "incomplete_topologies": len(labels) - len(complete),
        "topology_macro_seed_29_mean_loss": (
            mean(seed_29_values) if seed_29_values else None
        ),
        "topology_macro_seed_31_mean_loss": (
            mean(seed_31_values) if seed_31_values else None
        ),
        "topology_mean_seed_31_minus_seed_29_loss": (
            mean(differences) if differences else None
        ),
        "topology_median_seed_31_minus_seed_29_loss": (
            median(differences) if differences else None
        ),
        "seed_29_lower": sum(value > TIE_TOLERANCE for value in differences),
        "seed_31_lower": sum(value < -TIE_TOLERANCE for value in differences),
        "ties": sum(abs(value) <= TIE_TOLERANCE for value in differences),
        "unreached_feasible_values_imputed": False,
    }
    return summary, paired


def _sign(value: float) -> int:
    if value > TIE_TOLERANCE:
        return 1
    if value < -TIE_TOLERANCE:
        return -1
    return 0


def analyze_submission_like_trajectories(study: ValidatedStudy) -> dict[str, object]:
    """Compare history-derived progress under matched evaluations and wall time."""
    if len(study.configs) != 20 or len(study.history_rows) != 20:
        raise StudyValidationError("trajectory diagnostic requires all 20 histories")

    digests = []
    config_cells: dict[tuple[str, int], tuple[str, int]] = {}
    for run_id, config in study.configs.items():
        topology_spec = config.get("topology")
        seed = config.get("optimizer_seed")
        planned_index = config.get("planned_run_index")
        if (
            not isinstance(topology_spec, dict)
            or not isinstance(topology_spec.get("value"), str)
            or type(seed) is not int
            or seed not in EXPECTED_SEEDS
            or type(planned_index) is not int
        ):
            raise StudyValidationError("trajectory topology/seed/order cell is invalid")
        digest = hashlib.sha256(topology_spec["value"].encode()).hexdigest()
        digests.append(digest)
        cell = (digest, seed)
        if cell in config_cells:
            raise StudyValidationError("trajectory contains a duplicate topology/seed cell")
        config_cells[cell] = (run_id, planned_index)

    unique_digests = sorted(set(digests))
    if len(unique_digests) != 10:
        raise StudyValidationError("trajectory diagnostic requires ten topology blocks")
    label_by_digest = {
        digest: f"T{index:02d}" for index, digest in enumerate(unique_digests, start=1)
    }
    curves: dict[tuple[str, int], list[dict[str, float | int | None]]] = {}
    seed_29_first = True
    for digest in unique_digests:
        if not all((digest, seed) in config_cells for seed in EXPECTED_SEEDS):
            raise StudyValidationError("trajectory topology block is incomplete")
        label = label_by_digest[digest]
        for seed in EXPECTED_SEEDS:
            run_id, _planned_index = config_cells[(digest, seed)]
            curves[(label, seed)] = _history_curve(
                study.history_rows[run_id], label=f"{label}/seed-{seed}"
            )
        seed_29_first &= (
            config_cells[(digest, 29)][1] < config_cells[(digest, 31)][1]
        )
    if not seed_29_first:
        raise StudyValidationError(
            "trajectory profile no longer has seed 29 as the first sweep"
        )

    labels = [label_by_digest[digest] for digest in unique_digests]
    evaluation_summaries = []
    time_summaries = []
    private_rows = []
    for fraction, checkpoint in _evaluation_checkpoints(curves):
        summary, rows = _checkpoint_summary(
            basis="evaluation_count",
            coordinate="eval_count",
            checkpoint=checkpoint,
            progress_fraction=fraction,
            labels=labels,
            curves=curves,
        )
        evaluation_summaries.append(summary)
        private_rows.extend(rows)
    for fraction, checkpoint in _time_checkpoints(curves):
        summary, rows = _checkpoint_summary(
            basis="wall_time_seconds",
            coordinate="time_seconds",
            checkpoint=checkpoint,
            progress_fraction=fraction,
            labels=labels,
            curves=curves,
        )
        time_summaries.append(summary)
        private_rows.extend(rows)

    paired_axes = []
    for evaluation, wall_time in zip(
        evaluation_summaries, time_summaries, strict=True
    ):
        evaluation_difference = evaluation[
            "topology_mean_seed_31_minus_seed_29_loss"
        ]
        time_difference = wall_time["topology_mean_seed_31_minus_seed_29_loss"]
        if evaluation_difference is None or time_difference is None:
            continue
        paired_axes.append(
            {
                "progress_fraction": evaluation["progress_fraction"],
                "evaluation_mean_difference": evaluation_difference,
                "wall_time_mean_difference": time_difference,
                "same_direction": _sign(float(evaluation_difference))
                == _sign(float(time_difference)),
                "absolute_difference_between_axes": abs(
                    float(evaluation_difference) - float(time_difference)
                ),
            }
        )

    return {
        "format_version": 1,
        "study_profile": "submission-like-screen-v1",
        "inference_unit": "topology (n=10)",
        "checkpoint_fractions": list(CHECKPOINT_FRACTIONS),
        "seed_29_is_first_sweep": True,
        "seed_and_sweep_phase_confounded": True,
        "evaluation_aligned": {"checkpoints": evaluation_summaries},
        "wall_time_aligned": {"checkpoints": time_summaries},
        "axis_comparison": {
            "comparable_fractions": len(paired_axes),
            "same_direction_fractions": sum(
                bool(row["same_direction"]) for row in paired_axes
            ),
            "mean_absolute_contrast_difference": (
                mean(
                    float(row["absolute_difference_between_axes"])
                    for row in paired_axes
                )
                if paired_axes
                else None
            ),
            "rows": paired_axes,
            "warning": (
                "Equal progress fractions on different resource axes are a "
                "diagnostic comparison, not interchangeable units or a causal test."
            ),
        },
        "private_topology_checkpoint_rows": private_rows,
        "unreached_feasible_values_imputed": False,
        "changes_frozen_decision": False,
        "no_new_action_authorized": True,
        "interpretation": (
            "Read evaluation-aligned and wall-time-aligned topology summaries "
            "together. Persistence under evaluation alignment is inconsistent "
            "with a throughput-only explanation, but the fixed seed/sweep order "
            "still prevents causal attribution."
        ),
    }


def safe_submission_like_trajectories(
    trajectory: dict[str, object],
) -> dict[str, object]:
    """Return aggregate trajectory diagnostics without topology-level rows."""
    return {
        key: value
        for key, value in trajectory.items()
        if key != "private_topology_checkpoint_rows"
    }
