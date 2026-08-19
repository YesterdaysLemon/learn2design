"""Pure-Python scoring helpers for dfbench histories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import groupby


def _values(entry) -> list:
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        return list(entry)
    return [entry]


def flatten_histories(
    loss_history: list,
    feasible_history: list,
    time_steps: list[float],
    sensitivity_history: list | None = None,
    penalty_history: list | None = None,
) -> list[dict[str, object]]:
    """Flatten scalar or batched call histories without inventing chronology."""
    n_calls = len(loss_history)
    if len(feasible_history) != n_calls or len(time_steps) != n_calls:
        raise ValueError("loss, feasibility, and time histories must align")
    if sensitivity_history is not None and len(sensitivity_history) != n_calls:
        raise ValueError("sensitivity history must align with loss history")
    if penalty_history is not None and len(penalty_history) != n_calls:
        raise ValueError("penalty history must align with loss history")

    rows: list[dict[str, object]] = []
    evaluations = 0
    previous_time = -math.inf
    for call_index, (loss_entry, feasible_entry, time_seconds) in enumerate(
        zip(loss_history, feasible_history, time_steps)
    ):
        losses = _values(loss_entry)
        feasible = _values(feasible_entry)
        if len(losses) != len(feasible):
            raise ValueError(f"call {call_index} has mismatched batch shapes")
        if any(not isinstance(value, bool) for value in feasible):
            raise ValueError(f"call {call_index} has missing or non-boolean feasibility")
        time_value = float(time_seconds)
        if not math.isfinite(time_value) or time_value < previous_time:
            raise ValueError("time history must be finite and non-decreasing")
        previous_time = time_value

        sensitivity = (
            [None] * len(losses)
            if sensitivity_history is None
            else _values(sensitivity_history[call_index])
        )
        penalties = (
            [None] * len(losses)
            if penalty_history is None
            else _values(penalty_history[call_index])
        )
        if len(sensitivity) != len(losses) or len(penalties) != len(losses):
            raise ValueError(f"call {call_index} has mismatched aux batch shapes")

        evaluations += len(losses)
        for candidate_index, loss in enumerate(losses):
            loss_value = float(loss)
            rows.append(
                {
                    "call_index": call_index,
                    "candidate_index": candidate_index,
                    "eval_count_after_call": evaluations,
                    "time_seconds": time_value,
                    "loss": loss_value if math.isfinite(loss_value) else None,
                    "sensitivity_loss": _finite_or_none(sensitivity[candidate_index]),
                    "penalty": _finite_or_none(penalties[candidate_index]),
                    "is_feasible": bool(feasible[candidate_index]),
                }
            )
    return rows


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def summarize_rows(
    rows: list[dict[str, object]],
    targets: list[float] | None = None,
    time_grid: list[float] | None = None,
    eval_grid: list[int] | None = None,
) -> dict[str, object]:
    """Compute competition-aligned feasible metrics from candidate rows."""
    targets = sorted(set(float(target) for target in (targets or [])), reverse=True)
    best = math.inf
    best_time = None
    best_evals = None
    first_time = None
    first_evals = None
    first_loss = None
    first_finite_time = None
    first_finite_evals = None
    physically_feasible_candidates = 0
    finite_feasible_candidates = 0
    feasible_calls: set[int] = set()
    curve: list[dict[str, object]] = []
    hits = {
        _target_key(target): {"time_seconds": None, "eval_count": None}
        for target in targets
    }

    call_indices = []
    for call_index, call_group in groupby(
        rows, key=lambda row: int(row["call_index"])
    ):
        call_indices.append(call_index)
        call_rows = list(call_group)
        call_time = float(call_rows[0]["time_seconds"])
        call_evals = int(call_rows[0]["eval_count_after_call"])
        physical_rows = [row for row in call_rows if bool(row["is_feasible"])]
        finite_losses = [
            float(row["loss"])
            for row in physical_rows
            if row["loss"] is not None
        ]
        physically_feasible_candidates += len(physical_rows)
        finite_feasible_candidates += len(finite_losses)
        if physical_rows:
            feasible_calls.add(call_index)
            if first_time is None:
                first_time = call_time
                first_evals = call_evals
                first_loss = min(finite_losses) if finite_losses else None
        if finite_losses and first_finite_time is None:
            first_finite_time = call_time
            first_finite_evals = call_evals

        call_best = min(finite_losses) if finite_losses else math.inf
        if call_best < best:
            best = call_best
            best_time = call_time
            best_evals = call_evals
        curve.append(
            {
                "time_seconds": call_time,
                "eval_count": call_evals,
                "best_feasible_loss": best if math.isfinite(best) else None,
            }
        )
        _record_hits(hits, targets, best, call_time, call_evals)

    logged_calls = len(call_indices)
    return {
        "has_feasible": physically_feasible_candidates > 0,
        "has_finite_feasible": math.isfinite(best),
        "best_feasible_loss": best if math.isfinite(best) else None,
        "time_to_first_feasible_seconds": first_time,
        "evals_to_first_feasible": first_evals,
        "first_feasible_loss": first_loss,
        "time_to_first_finite_feasible_seconds": first_finite_time,
        "evals_to_first_finite_feasible": first_finite_evals,
        "time_to_best_feasible_seconds": best_time,
        "evals_to_best_feasible": best_evals,
        "physically_feasible_candidate_fraction": (
            physically_feasible_candidates / len(rows) if rows else None
        ),
        "finite_feasible_candidate_fraction": (
            finite_feasible_candidates / len(rows) if rows else None
        ),
        "calls_with_feasible_member": len(feasible_calls),
        "logged_calls": logged_calls,
        "logged_candidates": len(rows),
        "targets": hits,
        "anytime_grid": {
            "time_seconds": _grid_snapshot(curve, "time_seconds", time_grid or []),
            "eval_count": _grid_snapshot(curve, "eval_count", eval_grid or []),
        },
    }


def _record_hits(hits, targets, best, time_seconds, eval_count):
    if not math.isfinite(best):
        return
    for target in targets:
        hit = hits[_target_key(target)]
        if hit["time_seconds"] is None and best <= target:
            hit["time_seconds"] = time_seconds
            hit["eval_count"] = eval_count


def _target_key(target: float) -> str:
    return format(target, ".12g")


def _grid_snapshot(curve, coordinate: str, grid: list) -> dict[str, float | None]:
    result = {}
    for point in sorted(set(grid)):
        best = None
        for item in curve:
            if item[coordinate] > point:
                break
            best = item["best_feasible_loss"]
        result[format(point, ".12g")] = best
    return result
