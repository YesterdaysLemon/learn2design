"""Guarded parent and fresh deterministic cases for feasibility-debt-clock-v3.

Importing this module is stdlib-only and never executes a frozen case.  The
single ``--run`` parent calls the sealed worker twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STUDY_ID = "feasibility-debt-clock-v3"
PLAN_REVISION = "a61ba6003ec7cc5de5f41fc0c4349e62364ebd89"
PLAN_SHA256 = "1bf96ddd42c95dd9aa4ea516b1813929b6835f3949c4feb516fd2d7db62f57b8"
ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "research" / "2026-09-01-feasibility-debt-clock-v3-plan.md"
MAX_WORKER_BYTES = 262_144

CASE_SPECS = (
    ("protected_composite_trace_identity", 93503),
    ("pre_feasible_penalty_routing", 93521),
    ("first_feasible_irreversible_handoff", 93529),
    ("post_handoff_infeasible_reentry", 93553),
    ("masked_restart_state_rng_alignment", 93559),
    ("chunk_partition_trace_equivalence", 93581),
    ("partial_tail_transactionality", 93607),
    ("auxiliary_nonfinite_fail_closed", 93629),
    ("source_delta_and_process_seal", 93637),
)
CASE_KEYS = tuple(name for name, _seed in CASE_SPECS)
TRACE_KEYS = (
    "case",
    "seed",
    "batch",
    "admitted_count",
    "complete_population",
    "loss",
    "penalty",
    "feasible",
    "selected_progress",
    "improvement_mask",
    "stall_before",
    "stall_after",
    "restart_mask",
    "generation_before",
    "generation_after",
    "latch_before",
    "latch_after",
    "best_before",
    "best_after",
    "adam_age_before",
    "adam_age_after",
    "rng_before",
    "rng_after",
    "callback_count",
    "objective_eval_count",
    "update_applied",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _strict_json_loads(value: bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    return json.loads(value.decode("utf-8"), object_pairs_hook=pairs_hook)


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(token in "0123456789abcdef" for token in value)
    )


def _state_root(rng) -> str:
    return _sha256(_canonical_json(rng.bit_generator.state))


def _host(value) -> list[Any]:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind == "f":
        return [float(item) for item in array.reshape(-1)]
    if array.dtype.kind in "iu":
        return [int(item) for item in array.reshape(-1)]
    if array.dtype.kind == "b":
        return [bool(item) for item in array.reshape(-1)]
    raise TypeError("unsupported synthetic array dtype")


def _aux(losses, penalties, feasible):
    import jax.numpy as jnp

    count = int(losses.shape[0])
    return {
        "is_feasible": jnp.asarray(feasible, dtype=bool),
        "penalty": jnp.asarray(penalties, dtype=losses.dtype),
        "sensitivity_loss": jnp.asarray(losses - penalties),
        "violations": jnp.zeros((count, 2), dtype=losses.dtype),
        "power_values": {
            "hard": jnp.zeros((count, 1), dtype=losses.dtype),
            "soft": jnp.zeros((count, 1), dtype=losses.dtype),
            "detector": jnp.zeros((count, 1), dtype=losses.dtype),
        },
    }


def _simulate(
    *,
    case: str,
    seed: int,
    loss_rows: list[list[float]],
    penalty_rows: list[list[float]],
    feasible_rows: list[list[bool]],
    patience: int,
    progress_mode: str,
    engine: str = "candidate",
    partial_count: int | None = None,
    chunk_pattern: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    import jax.numpy as jnp
    import numpy as np

    from experiments.candidates.feasibility_debt_clock_v3 import (
        FeasibilityDebtBatchedRestartAdamV3,
    )
    from submission.submission import BatchedRestartAdam

    population = 5
    initial = [((seed % 64) + lane) / 128 for lane in range(population)]
    params = jnp.asarray(initial, dtype=jnp.float32)[:, None]
    first = jnp.zeros_like(params)
    second = jnp.zeros_like(params)
    steps = jnp.zeros((population,), dtype=jnp.int32)
    best = jnp.full((population,), jnp.inf, dtype=params.dtype)
    stalled = jnp.zeros((population,), dtype=jnp.int32)
    latch = jnp.zeros((population,), dtype=bool)
    generation = jnp.zeros((population,), dtype=jnp.int32)
    learning_rates = jnp.geomspace(0.03, 0.15, population)[:, None]
    rng = np.random.default_rng(seed)
    rng_calls = 0
    callback_count = 0
    eval_count = 0
    rows: list[dict[str, Any]] = []
    restart_batches = {str(lane): [] for lane in range(population)}
    physical_call_count = 0

    for batch, (loss_values, penalty_values, feasible_values) in enumerate(
        zip(loss_rows, penalty_rows, feasible_rows, strict=True)
    ):
        active = population
        complete = True
        if partial_count is not None and batch == len(loss_rows) - 1:
            active = partial_count
            complete = active == population
        if chunk_pattern is None:
            physical_chunks = (active,)
        else:
            physical_chunks = chunk_pattern
            if active != population or sum(physical_chunks) != active:
                raise ValueError("chunk pattern must cover one complete population")
        physical_call_count += len(physical_chunks)
        loss_chunks = []
        penalty_chunks = []
        feasible_chunks = []
        gradient_chunks = []
        auxiliary_chunks = []
        offset = 0
        for width in physical_chunks:
            next_offset = offset + width
            chunk_losses = jnp.asarray(
                loss_values[offset:next_offset], dtype=params.dtype
            )
            chunk_penalties = jnp.asarray(
                penalty_values[offset:next_offset], dtype=params.dtype
            )
            chunk_feasible = jnp.asarray(
                feasible_values[offset:next_offset], dtype=bool
            )
            loss_chunks.append(chunk_losses)
            penalty_chunks.append(chunk_penalties)
            feasible_chunks.append(chunk_feasible)
            gradient_chunks.append(
                jnp.asarray(
                    [
                        [(1 + batch + 2 * lane) / 64]
                        for lane in range(offset, next_offset)
                    ],
                    dtype=params.dtype,
                )
            )
            auxiliary_chunks.append(
                _aux(chunk_losses, chunk_penalties, chunk_feasible)
            )
            offset = next_offset
        losses = jnp.concatenate(loss_chunks, axis=0)
        penalties = jnp.concatenate(penalty_chunks, axis=0)
        feasible = jnp.concatenate(feasible_chunks, axis=0)
        gradient = jnp.concatenate(gradient_chunks, axis=0)
        aux = {
            "is_feasible": jnp.concatenate(
                [item["is_feasible"] for item in auxiliary_chunks], axis=0
            ),
            "penalty": jnp.concatenate(
                [item["penalty"] for item in auxiliary_chunks], axis=0
            ),
            "sensitivity_loss": jnp.concatenate(
                [item["sensitivity_loss"] for item in auxiliary_chunks], axis=0
            ),
            "violations": jnp.concatenate(
                [item["violations"] for item in auxiliary_chunks], axis=0
            ),
            "power_values": {
                name: jnp.concatenate(
                    [item["power_values"][name] for item in auxiliary_chunks],
                    axis=0,
                )
                for name in ("hard", "soft", "detector")
            },
        }
        rng_before = _state_root(rng)
        latch_before = latch[:active]
        best_before = best[:active]
        stalled_before = stalled[:active]
        generation_before = generation[:active]
        steps_before = steps[:active]
        eval_count += active

        if engine == "protected":
            selected = losses
            improved = jnp.isfinite(losses) & (
                losses < best_before
            )
            next_latch = latch_before
            next_best = jnp.where(improved, selected, best_before)
            next_stalled = jnp.where(improved, 0, stalled_before + 1)
        else:
            (
                selected,
                improved,
                next_latch,
                next_best,
                next_stalled,
            ) = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
                losses,
                aux,
                latch_before,
                best_before,
                stalled_before,
                0.0,
                progress_mode,
            )

        if complete:
            adam = (
                BatchedRestartAdam._adam_step
                if engine == "protected"
                else FeasibilityDebtBatchedRestartAdamV3._adam_step
            )
            params, first, second, steps = adam(
                params,
                gradient,
                first,
                second,
                steps,
                learning_rates,
                0.9,
                0.999,
                1e-8,
            )
            latch = next_latch
            best = next_best
            stalled = next_stalled
            restart = stalled >= patience
            if bool(jnp.any(restart)):
                fresh = jnp.asarray(
                    rng.standard_normal((population, 1)), dtype=params.dtype
                )
                rng_calls += 1
                params = jnp.where(restart[:, None], fresh, params)
                first = jnp.where(restart[:, None], 0.0, first)
                second = jnp.where(restart[:, None], 0.0, second)
                steps = jnp.where(restart, 0, steps)
                best = jnp.where(restart, jnp.inf, best)
                stalled = jnp.where(restart, 0, stalled)
                latch = jnp.where(restart, False, latch)
                generation = generation + restart.astype(jnp.int32)
                for lane, fired in enumerate(_host(restart)):
                    if fired:
                        restart_batches[str(lane)].append(batch)
            callback_count += 1
            update_applied = True
        else:
            restart = jnp.zeros((active,), dtype=bool)
            update_applied = False

        row = {
            "case": case,
            "seed": seed,
            "batch": batch,
            "admitted_count": active,
            "complete_population": complete,
            "loss": _host(losses),
            "penalty": _host(penalties),
            "feasible": _host(feasible),
            "selected_progress": _host(selected),
            "improvement_mask": _host(improved),
            "stall_before": _host(stalled_before),
            "stall_after": _host(stalled[:active]),
            "restart_mask": _host(restart),
            "generation_before": _host(generation_before),
            "generation_after": _host(generation[:active]),
            "latch_before": _host(latch_before),
            "latch_after": _host(latch[:active]),
            "best_before": [
                None if math.isinf(value) else value
                for value in _host(best_before)
            ],
            "best_after": [
                None if math.isinf(value) else value
                for value in _host(best[:active])
            ],
            "adam_age_before": _host(steps_before),
            "adam_age_after": _host(steps[:active]),
            "rng_before": rng_before,
            "rng_after": _state_root(rng),
            "callback_count": callback_count,
            "objective_eval_count": eval_count,
            "update_applied": update_applied,
        }
        if tuple(row) != TRACE_KEYS:
            raise RuntimeError("synthetic trace schema drifted")
        rows.append(row)

    final = {
        "params": _host(params),
        "first": _host(first),
        "second": _host(second),
        "steps": _host(steps),
        "best": [None if math.isinf(value) else value for value in _host(best)],
        "stalled": _host(stalled),
        "latch": _host(latch),
        "generation": _host(generation),
        "rng_root": _state_root(rng),
        "rng_calls": rng_calls,
        "callback_count": callback_count,
        "eval_count": eval_count,
        "restart_batches": restart_batches,
    }
    return {
        "rows": rows,
        "final": final,
        "physical_call_count": physical_call_count,
    }


def _default_rows(batches: int) -> tuple[list[list[float]], list[list[float]], list[list[bool]]]:
    losses = [
        [32 - 2 * batch - lane / 2 for lane in range(5)]
        for batch in range(batches)
    ]
    penalties = [
        [16 - batch - lane / 4 for lane in range(5)]
        for batch in range(batches)
    ]
    feasible = [[False] * 5 for _batch in range(batches)]
    return losses, penalties, feasible


def _case_protected_identity() -> dict[str, Any]:
    lane_losses = (
        [8, 7, 6, 5, 4, 3, 2, 1],
        [4, 4, 4, 4, 3, 3, 3, 3],
        [6, 5, 5, 5, 4, 4, 4, 3],
        [3, 4, 3, 4, 3, 4, 3, 4],
        [9, 8, 8, 7, 7, 6, 6, 5],
    )
    losses = [[lane_losses[lane][batch] for lane in range(5)] for batch in range(8)]
    penalties = [[32 + 4 * batch + lane for lane in range(5)] for batch in range(8)]
    feasible = [[(batch + lane) % 3 == 0 for lane in range(5)] for batch in range(8)]
    common = dict(
        case="protected_composite_trace_identity",
        seed=93503,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="total_loss",
    )
    protected = _simulate(**common, engine="protected")
    candidate = _simulate(**common, engine="candidate")
    return {"passed": protected == candidate, "projection": candidate}


def _case_penalty_routing() -> dict[str, Any]:
    losses, _penalties, feasible = _default_rows(6)
    penalty_lanes = (
        [5, 4, 4, 4, 3, 3],
        [2, 2, 2, 2, 2, 2],
        [6, 5, 4, 3, 2, 1],
        [3, 3, 2, 2, 2, 1],
        [1, 1, 1, 0.5, 0.5, 0.5],
    )
    penalties = [[penalty_lanes[lane][batch] for lane in range(5)] for batch in range(6)]
    treatment = _simulate(
        case="pre_feasible_penalty_routing",
        seed=93521,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="feasibility_debt",
    )
    control = _simulate(
        case="pre_feasible_penalty_routing",
        seed=93521,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="total_loss",
    )
    expected = {"0": [3], "1": [2, 5], "2": [], "3": [4], "4": [2, 5]}
    passed = (
        treatment["final"]["restart_batches"] == expected
        and control["final"]["restart_batches"] == {str(i): [] for i in range(5)}
    )
    return {"passed": passed, "treatment": treatment, "control": control}


def _case_first_feasible() -> dict[str, Any]:
    losses = []
    penalties = []
    feasible = []
    for batch in range(7):
        losses.append([
            24 - batch - lane / 2 if batch < lane else 10 + lane
            for lane in range(5)
        ])
        penalties.append([20 + lane - batch for lane in range(5)])
        feasible.append([batch >= lane for lane in range(5)])
    result = _simulate(
        case="first_feasible_irreversible_handoff",
        seed=93529,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="feasibility_debt",
    )
    first_restart = {
        str(lane): result["final"]["restart_batches"][str(lane)][0]
        if result["final"]["restart_batches"][str(lane)]
        else None
        for lane in range(5)
    }
    expected = {str(lane): lane + 2 for lane in range(5)}
    boundaries_hold = all(
        result["rows"][lane]["latch_before"][lane] is False
        and result["rows"][lane]["latch_after"][lane] is True
        and result["rows"][lane]["improvement_mask"][lane] is True
        for lane in range(5)
    )
    return {
        "passed": first_restart == expected and boundaries_hold,
        "first_restart": first_restart,
        "projection": result,
    }


def _case_infeasible_reentry() -> dict[str, Any]:
    loss_levels = [20, 10, 10, 10, 9, 8]
    losses = [[loss_levels[batch] + lane for lane in range(5)] for batch in range(6)]
    penalties = [[6 + lane - batch for lane in range(5)] for batch in range(6)]
    flags = [False, True, False, False, True, True]
    feasible = [[flags[batch]] * 5 for batch in range(6)]
    result = _simulate(
        case="post_handoff_infeasible_reentry",
        seed=93553,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="feasibility_debt",
    )
    expected = {str(lane): [3] for lane in range(5)}
    row4 = result["rows"][4]
    passed = (
        result["final"]["restart_batches"] == expected
        and all(row4["latch_after"])
        and row4["best_after"] == [9.0 + lane for lane in range(5)]
    )
    return {"passed": passed, "projection": result}


def _case_masked_rng() -> dict[str, Any]:
    import jax.numpy as jnp
    import numpy as np

    from submission.submission import BatchedRestartAdam

    losses, _penalties, feasible = _default_rows(5)
    penalties = [
        [([4, 4, 4, 3, 3] if lane % 2 == 0 else [5, 4, 3, 2, 1])[batch] for lane in range(5)]
        for batch in range(5)
    ]
    result = _simulate(
        case="masked_restart_state_rng_alignment",
        seed=93559,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=2,
        progress_mode="feasibility_debt",
    )
    expected = {"0": [2], "1": [], "2": [2], "3": [], "4": [2]}
    mask = jnp.asarray([True, False, True, False, True])
    initial = jnp.asarray(
        [[((93559 % 64) + lane) / 128] for lane in range(5)],
        dtype=jnp.float32,
    )
    params = initial
    first = jnp.zeros_like(params)
    second = jnp.zeros_like(params)
    steps = jnp.zeros((5,), dtype=jnp.int32)
    learning_rates = jnp.geomspace(0.03, 0.15, 5)[:, None]
    replay_rng = np.random.default_rng(93559)
    for batch in range(5):
        gradient = jnp.asarray(
            [[(1 + batch + 2 * lane) / 64] for lane in range(5)],
            dtype=params.dtype,
        )
        params, first, second, steps = BatchedRestartAdam._adam_step(
            params,
            gradient,
            first,
            second,
            steps,
            learning_rates,
            0.9,
            0.999,
            1e-8,
        )
        if batch == 2:
            fresh = jnp.asarray(
                replay_rng.standard_normal((5, 1)), dtype=params.dtype
            )
            params = jnp.where(mask[:, None], fresh, params)
            first = jnp.where(mask[:, None], 0.0, first)
            second = jnp.where(mask[:, None], 0.0, second)
            steps = jnp.where(mask, 0, steps)
    independent = {
        "params": _host(params),
        "first": _host(first),
        "second": _host(second),
        "steps": _host(steps),
        "rng_root": _state_root(replay_rng),
    }
    projected = {
        key: result["final"][key]
        for key in ("params", "first", "second", "steps", "rng_root")
    }
    restart_row = result["rows"][2]
    passed = (
        result["final"]["restart_batches"] == expected
        and result["final"]["rng_calls"] == 1
        and projected == independent
        and restart_row["restart_mask"] == [True, False, True, False, True]
        and restart_row["generation_after"] == [1, 0, 1, 0, 1]
        and restart_row["adam_age_after"] == [0, 3, 0, 3, 0]
    )
    return {
        "passed": passed,
        "independent_mask_replay": independent,
        "projection": result,
    }


def _chunk_rows() -> tuple[list[list[float]], list[list[float]], list[list[bool]]]:
    losses = [[8 - batch + lane / 2 for lane in range(5)] for batch in range(3)]
    penalties = [[5 - batch + lane / 4 for lane in range(5)] for batch in range(3)]
    feasible = [[(batch + lane) % 3 == 0 for lane in range(5)] for batch in range(3)]
    return losses, penalties, feasible


def _case_chunk_equivalence() -> dict[str, Any]:
    losses, penalties, feasible = _chunk_rows()
    left = _simulate(
        case="chunk_partition_trace_equivalence",
        seed=93581,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=3,
        progress_mode="feasibility_debt",
    )
    right = _simulate(
        case="chunk_partition_trace_equivalence",
        seed=93581,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=3,
        progress_mode="feasibility_debt",
        chunk_pattern=(2, 1, 2),
    )
    left_logical = {"rows": left["rows"], "final": left["final"]}
    right_logical = {"rows": right["rows"], "final": right["final"]}
    passed = (
        left_logical == right_logical
        and left["physical_call_count"] == 3
        and right["physical_call_count"] == 9
    )
    return {
        "passed": passed,
        "chunks": [2, 1, 2],
        "full": left,
        "partitioned": right,
    }


def _case_partial_tail() -> dict[str, Any]:
    losses, penalties, feasible = _chunk_rows()
    control = _simulate(
        case="partial_tail_transactionality",
        seed=93607,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=3,
        progress_mode="feasibility_debt",
    )
    losses.append([1.0, 1.5, 0.0, 0.0, 0.0])
    penalties.append([0.5, 0.75, 0.0, 0.0, 0.0])
    feasible.append([True, False, False, False, False])
    result = _simulate(
        case="partial_tail_transactionality",
        seed=93607,
        loss_rows=losses,
        penalty_rows=penalties,
        feasible_rows=feasible,
        patience=3,
        progress_mode="feasibility_debt",
        partial_count=2,
    )
    before = result["rows"][2]
    tail = result["rows"][3]
    immutable_final_keys = (
        "params",
        "first",
        "second",
        "steps",
        "best",
        "stalled",
        "latch",
        "generation",
        "rng_root",
        "rng_calls",
        "callback_count",
        "restart_batches",
    )
    passed = (
        tail["admitted_count"] == 2
        and tail["complete_population"] is False
        and tail["update_applied"] is False
        and tail["callback_count"] == before["callback_count"]
        and tail["rng_after"] == before["rng_after"]
        and result["final"]["eval_count"] == 17
        and control["final"]["eval_count"] == 15
        and all(
            result["final"][key] == control["final"][key]
            for key in immutable_final_keys
        )
    )
    return {"passed": passed, "control": control, "projection": result}


def _case_aux_nonfinite() -> dict[str, Any]:
    import copy
    import jax.numpy as jnp
    import numpy as np

    from experiments.candidates.feasibility_debt_clock_v3 import (
        FeasibilityDebtBatchedRestartAdamV3,
    )

    losses = jnp.arange(5, dtype=jnp.float32)
    valid = _aux(losses, jnp.ones((5,), dtype=losses.dtype), jnp.zeros((5,), dtype=bool))
    attacks = []
    missing = copy.deepcopy(valid)
    del missing["penalty"]
    attacks.append(missing)
    extra = copy.deepcopy(valid)
    extra["extra"] = jnp.zeros((5,))
    attacks.append(extra)
    attacks.extend([None, [], "bad"])
    wrong_bool = copy.deepcopy(valid)
    wrong_bool["is_feasible"] = jnp.zeros((5,), dtype=jnp.int32)
    attacks.append(wrong_bool)
    wrong_penalty = copy.deepcopy(valid)
    wrong_penalty["penalty"] = jnp.zeros((5,), dtype=jnp.int32)
    attacks.append(wrong_penalty)
    bool_penalty = copy.deepcopy(valid)
    bool_penalty["penalty"] = jnp.zeros((5,), dtype=bool)
    attacks.append(bool_penalty)
    string_penalty = copy.deepcopy(valid)
    string_penalty["penalty"] = np.asarray(["bad"] * 5)
    attacks.append(string_penalty)
    scalar = copy.deepcopy(valid)
    scalar["penalty"] = jnp.asarray(1.0)
    attacks.append(scalar)
    wrong_rank = copy.deepcopy(valid)
    wrong_rank["penalty"] = jnp.ones((5, 1), dtype=losses.dtype)
    attacks.append(wrong_rank)
    wrong_leading = copy.deepcopy(valid)
    wrong_leading["penalty"] = jnp.ones((4,), dtype=losses.dtype)
    attacks.append(wrong_leading)
    negative = copy.deepcopy(valid)
    negative["penalty"] = jnp.asarray([1, 1, -1, 1, 1], dtype=losses.dtype)
    attacks.append(negative)
    rejected = 0
    for payload in attacks:
        try:
            FeasibilityDebtBatchedRestartAdamV3._validate_feasibility_debt_aux(payload, 5)
        except (TypeError, ValueError):
            rejected += 1

    best = jnp.asarray([4, 4, 4, 4, 4], dtype=losses.dtype)
    stalled = jnp.zeros((5,), dtype=jnp.int32)
    latch = jnp.zeros((5,), dtype=bool)
    nonfinite_results = []
    for value in (jnp.nan, jnp.inf):
        payload = copy.deepcopy(valid)
        payload["penalty"] = jnp.full((5,), value, dtype=losses.dtype)
        transition = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
            losses,
            payload,
            latch,
            best,
            stalled,
            0.0,
            "feasibility_debt",
        )
        nonfinite_results.append(
            not any(_host(transition[1]))
            and _host(transition[3]) == _host(best)
            and _host(transition[4]) == [1] * 5
        )
    latched = jnp.ones((5,), dtype=bool)
    for value in (jnp.nan, jnp.inf):
        nonfinite_losses = jnp.full((5,), value, dtype=losses.dtype)
        transition = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
            nonfinite_losses,
            valid,
            latched,
            best,
            stalled,
            0.0,
            "feasibility_debt",
        )
        nonfinite_results.append(
            not any(_host(transition[1]))
            and _host(transition[2]) == [True] * 5
            and _host(transition[3]) == _host(best)
            and _host(transition[4]) == [1] * 5
        )
    passed = rejected == len(attacks) and all(nonfinite_results)
    return {
        "passed": passed,
        "allowed_nonfinite_count": len(nonfinite_results),
        "rejected": rejected,
        "attack_count": len(attacks),
    }


def _case_source_seal(stdout_sealed: bool) -> dict[str, Any]:
    from experiments.candidates.feasibility_debt_clock_v3_source import (
        source_projection,
    )

    projection = source_projection()
    return {
        "passed": stdout_sealed
        and _hex(projection["source_boundary_root_sha256"], 64),
        "projection": projection,
    }


def _case_projection(stdout_sealed: bool) -> dict[str, Any]:
    builders = (
        _case_protected_identity,
        _case_penalty_routing,
        _case_first_feasible,
        _case_infeasible_reentry,
        _case_masked_rng,
        _case_chunk_equivalence,
        _case_partial_tail,
        _case_aux_nonfinite,
    )
    results: dict[str, dict[str, Any]] = {}
    for (name, _seed), builder in zip(CASE_SPECS[:-1], builders, strict=True):
        results[name] = builder()
    results[CASE_KEYS[-1]] = _case_source_seal(stdout_sealed)
    outcomes = {name: bool(results[name]["passed"]) for name in CASE_KEYS}
    roots = {name: _sha256(_canonical_json(results[name])) for name in CASE_KEYS}
    return {"case_outcomes": outcomes, "case_roots": roots}


WORKER_KEYS = {
    "study_id",
    "invocation_revision",
    "plan_revision",
    "plan_sha256",
    "protected_source_sha256",
    "candidate_source_sha256",
    "fixture_source_sha256",
    "worker_source_sha256",
    "case_count",
    "case_outcomes",
    "case_roots",
    "all_cases_passed",
    "stdout_sealed",
    "source_boundary_root_sha256",
    "core_root_sha256",
}


def _worker_projection(stdout_sealed: bool = True) -> dict[str, Any]:
    from experiments.candidates.feasibility_debt_clock_v3_source import (
        source_projection,
    )

    invocation_revision = os.environ.get("FDC_V3_INVOCATION_REVISION", "")
    if not _hex(invocation_revision, 40):
        raise RuntimeError("missing invocation revision")
    if _sha256(PLAN_PATH.read_bytes()) != PLAN_SHA256:
        raise RuntimeError("plan hash mismatch")
    cases = _case_projection(stdout_sealed)
    sources = source_projection()
    payload: dict[str, Any] = {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "protected_source_sha256": sources["protected_source_sha256"],
        "candidate_source_sha256": sources["candidate_source_sha256"],
        "fixture_source_sha256": sources["fixture_source_sha256"],
        "worker_source_sha256": sources["worker_source_sha256"],
        "case_count": len(CASE_KEYS),
        "case_outcomes": cases["case_outcomes"],
        "case_roots": cases["case_roots"],
        "all_cases_passed": all(cases["case_outcomes"].values()),
        "stdout_sealed": stdout_sealed,
        "source_boundary_root_sha256": sources["source_boundary_root_sha256"],
    }
    payload["core_root_sha256"] = _sha256(_canonical_json(payload))
    return payload


def _validate_worker_payload(payload: Any, raw: bytes) -> bool:
    if not isinstance(payload, dict) or set(payload) != WORKER_KEYS:
        return False
    if raw != _canonical_json(payload) + b"\n":
        return False
    if payload.get("study_id") != STUDY_ID:
        return False
    if not _hex(payload.get("invocation_revision"), 40):
        return False
    if payload.get("plan_revision") != PLAN_REVISION or payload.get("plan_sha256") != PLAN_SHA256:
        return False
    for key in (
        "protected_source_sha256",
        "candidate_source_sha256",
        "fixture_source_sha256",
        "worker_source_sha256",
        "source_boundary_root_sha256",
        "core_root_sha256",
    ):
        if not _hex(payload.get(key), 64):
            return False
    if type(payload.get("case_count")) is not int or payload["case_count"] != 9:
        return False
    outcomes = payload.get("case_outcomes")
    roots = payload.get("case_roots")
    if not isinstance(outcomes, dict) or tuple(outcomes) != tuple(sorted(CASE_KEYS)):
        return False
    if set(outcomes) != set(CASE_KEYS) or any(type(value) is not bool for value in outcomes.values()):
        return False
    if not isinstance(roots, dict) or set(roots) != set(CASE_KEYS) or any(not _hex(value, 64) for value in roots.values()):
        return False
    if type(payload.get("stdout_sealed")) is not bool or payload["stdout_sealed"] is not True:
        return False
    if type(payload.get("all_cases_passed")) is not bool or payload["all_cases_passed"] != all(outcomes.values()):
        return False
    without_root = {key: value for key, value in payload.items() if key != "core_root_sha256"}
    return payload["core_root_sha256"] == _sha256(_canonical_json(without_root))


TRANSPORT_SUFFIXES = (
    "process_started",
    "exit_code_zero",
    "stderr_empty",
    "stdout_within_cap",
    "json_parsed",
    "schema_valid",
    "study_identity_valid",
)
TRANSPORT_KEYS = tuple(
    f"worker_{number}_{suffix}"
    for number in (1, 2)
    for suffix in TRANSPORT_SUFFIXES
)


def _scrubbed_environment() -> dict[str, str]:
    forbidden = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
    result = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in forbidden)
    }
    result.pop("FDC_V3_IMPORT_NOISE_MODULE", None)
    result.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "NO_PROXY": "*",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return result


def _worker_identity_valid(
    payload: Any,
    invocation_revision: str,
    expected_sources: dict[str, str],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("study_id") != STUDY_ID:
        return False
    if payload.get("invocation_revision") != invocation_revision:
        return False
    if payload.get("plan_revision") != PLAN_REVISION:
        return False
    if payload.get("plan_sha256") != PLAN_SHA256:
        return False
    for key in (
        "protected_source_sha256",
        "candidate_source_sha256",
        "fixture_source_sha256",
        "worker_source_sha256",
        "source_boundary_root_sha256",
    ):
        if payload.get(key) != expected_sources.get(key):
            return False
    return True


def _run_worker(
    invocation_revision: str,
    number: int,
    expected_sources: dict[str, str],
):
    prefix = f"worker_{number}_"
    receipt = {f"{prefix}{suffix}": False for suffix in TRANSPORT_SUFFIXES}
    environment = _scrubbed_environment()
    environment["FDC_V3_INVOCATION_REVISION"] = invocation_revision
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.candidates.feasibility_debt_clock_v3_worker",
                "--child",
            ],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        receipt[f"{prefix}process_started"] = True
        return receipt, None, None
    except (OSError, ValueError):
        return receipt, None, None
    receipt[f"{prefix}process_started"] = True
    receipt[f"{prefix}exit_code_zero"] = completed.returncode == 0
    receipt[f"{prefix}stderr_empty"] = completed.stderr == b""
    receipt[f"{prefix}stdout_within_cap"] = len(completed.stdout) <= MAX_WORKER_BYTES
    payload = None
    if receipt[f"{prefix}stdout_within_cap"]:
        try:
            parsed = _strict_json_loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            receipt[f"{prefix}json_parsed"] = True
            receipt[f"{prefix}schema_valid"] = _validate_worker_payload(parsed, completed.stdout)
            receipt[f"{prefix}study_identity_valid"] = (
                receipt[f"{prefix}schema_valid"]
                and _worker_identity_valid(
                    parsed,
                    invocation_revision,
                    expected_sources,
                )
            )
            if all(receipt.values()):
                payload = parsed
    return receipt, completed.stdout if payload is not None else None, payload


def _current_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "0" * 40
    value = completed.stdout.decode("ascii", errors="ignore").strip()
    return value if completed.returncode == 0 and _hex(value, 40) else "0" * 40


PARENT_KEYS = {
    "study_id",
    "invocation_revision",
    "plan_revision",
    "plan_sha256",
    "protected_source_sha256",
    "candidate_source_sha256",
    "fixture_source_sha256",
    "worker_source_sha256",
    "case_count",
    "case_outcomes",
    "transport_outcomes",
    "all_cases_passed",
    "runs_equal",
    "source_boundary_root_sha256",
    "process_replay_root_sha256",
    "action",
}


def _closed_failure(invocation_revision: str | None = None) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision or _current_revision(),
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "protected_source_sha256": None,
        "candidate_source_sha256": None,
        "fixture_source_sha256": None,
        "worker_source_sha256": None,
        "case_count": len(CASE_KEYS),
        "case_outcomes": {key: False for key in CASE_KEYS},
        "transport_outcomes": {key: False for key in TRANSPORT_KEYS},
        "all_cases_passed": False,
        "runs_equal": False,
        "source_boundary_root_sha256": None,
        "process_replay_root_sha256": None,
        "action": "park_feasibility_debt_v3",
    }


def _validate_parent(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != PARENT_KEYS:
        return False
    if payload.get("study_id") != STUDY_ID or not _hex(payload.get("invocation_revision"), 40):
        return False
    if payload.get("plan_revision") != PLAN_REVISION or payload.get("plan_sha256") != PLAN_SHA256:
        return False
    outcomes = payload.get("case_outcomes")
    transport = payload.get("transport_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(CASE_KEYS) or any(type(value) is not bool for value in outcomes.values()):
        return False
    if not isinstance(transport, dict) or set(transport) != set(TRANSPORT_KEYS) or any(type(value) is not bool for value in transport.values()):
        return False
    expected = all(outcomes.values()) and all(transport.values()) and payload.get("runs_equal") is True
    if payload.get("all_cases_passed") is not expected:
        return False
    action = "approve_feasibility_debt_v3_for_fresh_candidate_screen_planning" if expected else "park_feasibility_debt_v3"
    if payload.get("action") != action:
        return False
    hashes = (
        "protected_source_sha256",
        "candidate_source_sha256",
        "fixture_source_sha256",
        "worker_source_sha256",
        "source_boundary_root_sha256",
        "process_replay_root_sha256",
    )
    if expected and any(not _hex(payload.get(key), 64) for key in hashes):
        return False
    if not expected and any(payload.get(key) is not None and not _hex(payload.get(key), 64) for key in hashes):
        return False
    return type(payload.get("case_count")) is int and payload["case_count"] == 9 and type(payload.get("runs_equal")) is bool


def run_terminal_projection() -> dict[str, Any]:
    invocation_revision = _current_revision()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _closed_failure(invocation_revision)
    if status.returncode != 0 or status.stdout or status.stderr or _sha256(PLAN_PATH.read_bytes()) != PLAN_SHA256:
        return _closed_failure(invocation_revision)

    try:
        from experiments.candidates.feasibility_debt_clock_v3_source import (
            source_projection,
        )

        expected_sources = source_projection()
    except Exception:
        return _closed_failure(invocation_revision)

    first_receipt, first_bytes, first = _run_worker(
        invocation_revision, 1, expected_sources
    )
    second_receipt, second_bytes, second = _run_worker(
        invocation_revision, 2, expected_sources
    )
    transport = {**first_receipt, **second_receipt}
    valid = first if first is not None else second
    outcomes = dict(valid["case_outcomes"]) if valid is not None else {key: False for key in CASE_KEYS}
    runs_equal = first is not None and second is not None and first_bytes == second_bytes and first == second
    passed = all(outcomes.values()) and all(transport.values()) and runs_equal
    payload = {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "protected_source_sha256": valid["protected_source_sha256"] if valid else None,
        "candidate_source_sha256": valid["candidate_source_sha256"] if valid else None,
        "fixture_source_sha256": valid["fixture_source_sha256"] if valid else None,
        "worker_source_sha256": valid["worker_source_sha256"] if valid else None,
        "case_count": len(CASE_KEYS),
        "case_outcomes": outcomes,
        "transport_outcomes": transport,
        "all_cases_passed": passed,
        "runs_equal": runs_equal,
        "source_boundary_root_sha256": valid["source_boundary_root_sha256"] if valid else None,
        "process_replay_root_sha256": _sha256(first_bytes) if runs_equal and first_bytes is not None else None,
        "action": "approve_feasibility_debt_v3_for_fresh_candidate_screen_planning" if passed else "park_feasibility_debt_v3",
    }
    return payload if _validate_parent(payload) else _closed_failure(invocation_revision)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    try:
        payload = run_terminal_projection()
    except Exception:
        payload = _closed_failure()
    if not _validate_parent(payload):
        payload = _closed_failure(payload.get("invocation_revision") if isinstance(payload, dict) else None)
    sys.stdout.buffer.write(_canonical_json(payload) + b"\n")
    return 0 if payload["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
