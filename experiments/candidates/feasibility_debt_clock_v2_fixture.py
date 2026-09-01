"""Frozen deterministic projection for feasibility-debt-clock-v2.

Importing this module never executes a result-bearing case. The ``--run``
entry point is the one terminal parent. It launches exactly two guarded CPU
children and always emits the closed sanitized parent schema on a handled
outcome.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from experiments.candidates.feasibility_debt_clock_v2 import (
    FeasibilityDebtBatchedRestartAdamV2,
)
from experiments.candidates.feasibility_debt_clock_v2_source import (
    verify_source_boundary,
)
from submission.submission import BatchedRestartAdam


STUDY_ID = "feasibility-debt-clock-v2"
PLAN_REVISION = "47efe9d7a55c6f308291b2faa12f160933dce8a5"
PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "2026-09-01-feasibility-debt-clock-v2-plan.md"
)
PLAN_SHA256 = "fac7a4f5a0c6624685f761d95be4f13d8b4f3db1695695fbd808cf8ff8c84df9"
POPULATION_SIZE = 4
DIMENSION = 3
MINIMUM_IMPROVEMENT = 1.0e-7
MAX_CHILD_BYTES = 524_288
COMMON_SETTINGS = {
    "population_size": POPULATION_SIZE,
    "learning_rate_low": 0.04,
    "learning_rate_high": 0.12,
    "minimum_improvement": MINIMUM_IMPROVEMENT,
    "gradient_clip_norm": 1.0,
    "restart_noise_scale": 0.25,
    "safety_seconds": 0.0,
}
SEEDS = {
    "compatibility_no_restart": 92617,
    "compatibility_mixed_restart": 92639,
    "pre_feasibility_lane_routing": 92657,
    "post_feasibility_handoff": 92669,
    "restart_state_isolation": 92681,
    "partial_tail_and_chunking": 92707,
    "auxiliary_attacks": 92723,
    "nonfinite_semantics": 92737,
}
CASE_KEYS = (
    "total_loss_no_restart_identity",
    "total_loss_mixed_restart_identity",
    "pre_feasibility_lane_routing",
    "post_feasibility_total_loss_handoff",
    "restart_state_isolation",
    "partial_tail_no_transition",
    "chunk_projection_equivalence",
    "auxiliary_schema_rejection",
    "nonfinite_progress_semantics",
    "source_delta_boundary",
)
TRANSPORT_KEYS = tuple(
    f"child_{child}_{suffix}"
    for child in (1, 2)
    for suffix in (
        "process_started",
        "exit_code_zero",
        "stderr_empty",
        "stdout_within_cap",
        "json_parsed",
        "schema_valid",
        "study_identity_valid",
    )
)
GRADIENTS = np.asarray(
    [
        [0.50, -0.50, 0.25],
        [-0.40, 0.30, -0.20],
        [0.10, 0.20, -0.30],
        [-0.25, -0.25, 0.50],
    ],
    dtype=np.float32,
)

Row = tuple[float, float, bool]
Batch = list[Row]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_commitment(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256(array.tobytes(order="C")),
    }


def _project_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _array_commitment(value)
        for key, value in sorted(event.items())
        if key not in {"time_seconds", "evaluation_batch_seconds"}
    }


def _batches(
    count: int,
    loss: Callable[[int, int], float],
    penalty: Callable[[int, int], float],
    feasible: Callable[[int, int], bool],
) -> list[Batch]:
    return [
        [
            (
                float(loss(batch, lane)),
                float(penalty(batch, lane)),
                bool(feasible(batch, lane)),
            )
            for lane in range(POPULATION_SIZE)
        ]
        for batch in range(count)
    ]


class ScriptedObjective:
    """Four-lane public-API stand-in with one committed row family."""

    def __init__(
        self,
        batches: list[Batch],
        *,
        max_evals: int,
        attack: str | None = None,
    ) -> None:
        if not batches or any(
            len(batch) != POPULATION_SIZE for batch in batches
        ):
            raise ValueError("script must contain complete four-lane batches")
        self.batches = tuple(tuple(row for row in batch) for batch in batches)
        self.n_params = DIMENSION
        self.max_evals = int(max_evals)
        self.eval_count = 0
        self.algorithm_str = ""
        self.unbounded = False
        self.optimization_pairs = [["component", "tuning"]] * DIMENSION
        self._key = jax.random.PRNGKey(0)
        self._started = False
        self.attack = attack
        self.random_draws: list[dict[str, Any]] = []
        self.input_fragments: list[tuple[int, int, np.ndarray]] = []
        self.best_feasible_loss = math.inf
        self.first_feasible_loss = math.inf

    def set_space_mode(self, unbounded: bool) -> None:
        self.unbounded = bool(unbounded)

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.PRNGKey(int(seed))

    def random_params_unbounded(self, n_samples: int = 1):
        self._key, sample_key = jax.random.split(self._key)
        values = jax.random.normal(sample_key, (int(n_samples), self.n_params))
        self.random_draws.append(_array_commitment(values))
        return values

    @property
    def budget_exceeded(self) -> bool:
        return self.eval_count >= self.max_evals

    @property
    def evals_left(self) -> int:
        return max(0, self.max_evals - self.eval_count)

    @property
    def budget_progress_fraction(self) -> float:
        return self.eval_count / self.max_evals

    @property
    def time_left(self):
        return None

    @property
    def time_elapsed(self) -> float:
        return 0.0

    def start_logging(self) -> None:
        self._started = True

    def _make_aux(
        self,
        penalties: np.ndarray,
        feasible: np.ndarray,
        losses: np.ndarray,
    ) -> dict[str, Any]:
        count = len(penalties)
        penalty = jnp.asarray(penalties, dtype=jnp.float32)
        feasible_values = jnp.asarray(feasible, dtype=bool)
        aux: dict[str, Any] = {
            "is_feasible": feasible_values,
            "penalty": penalty,
            "sensitivity_loss": jnp.asarray(losses, dtype=jnp.float32)
            - penalty,
            "violations": penalty[:, None],
            "power_values": {
                "hard": penalty[:, None],
                "soft": penalty[:, None],
                "detector": penalty[:, None],
            },
        }
        attack = self.attack
        if attack is None:
            return aux
        if attack.startswith("missing:"):
            path = attack.split(":", 1)[1]
            if path.startswith("power_values."):
                del aux["power_values"][path.split(".", 1)[1]]
            else:
                del aux[path]
        elif attack.startswith("leading:"):
            path = attack.split(":", 1)[1]
            if path.startswith("power_values."):
                aux["power_values"][path.split(".", 1)[1]] = jnp.zeros(
                    (count + 1, 1), dtype=jnp.float32
                )
            elif path == "violations":
                aux[path] = jnp.zeros((count + 1, 1), dtype=jnp.float32)
            elif path == "is_feasible":
                aux[path] = jnp.zeros((count + 1,), dtype=bool)
            else:
                aux[path] = jnp.zeros((count + 1,), dtype=jnp.float32)
        elif attack == "integer-feasibility":
            aux["is_feasible"] = jnp.zeros((count,), dtype=jnp.int32)
        elif attack == "scalar-penalty":
            aux["penalty"] = jnp.asarray(1.0, dtype=jnp.float32)
        elif attack == "complex-penalty":
            aux["penalty"] = jnp.ones((count,), dtype=jnp.complex64)
        elif attack == "list-power-values":
            aux["power_values"] = [0.0] * count
        elif attack == "negative-penalty":
            aux["penalty"] = jnp.full((count,), -1.0, dtype=jnp.float32)
        elif attack == "extra-power-value":
            aux["power_values"]["unexpected"] = jnp.zeros(
                (count, 1), dtype=jnp.float32
            )
        else:
            raise AssertionError(f"unknown attack: {attack}")
        return aux

    def _evaluate(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        count = int(params.shape[0])
        logical_batch = self.eval_count // POPULATION_SIZE
        lane_offset = self.eval_count % POPULATION_SIZE
        if logical_batch >= len(self.batches):
            logical_batch = len(self.batches) - 1
        if lane_offset + count > POPULATION_SIZE:
            raise RuntimeError("chunk crossed a logical batch boundary")
        rows = self.batches[logical_batch][lane_offset : lane_offset + count]
        losses_np = np.asarray([row[0] for row in rows], dtype=np.float32)
        penalties_np = np.asarray([row[1] for row in rows], dtype=np.float32)
        feasible_np = np.asarray([row[2] for row in rows], dtype=bool)
        gradients_np = GRADIENTS[lane_offset : lane_offset + count]
        self.input_fragments.append(
            (logical_batch, lane_offset, np.asarray(jax.device_get(params)))
        )
        losses = jnp.asarray(losses_np)
        gradients = jnp.asarray(gradients_np)
        aux = self._make_aux(penalties_np, feasible_np, losses_np)
        self.eval_count += count
        for loss, feasible_value in zip(losses_np, feasible_np, strict=True):
            if bool(feasible_value) and math.isfinite(float(loss)):
                if self.first_feasible_loss == math.inf:
                    self.first_feasible_loss = float(loss)
                self.best_feasible_loss = min(
                    self.best_feasible_loss, float(loss)
                )
        return losses, gradients, aux

    def vmap_value_and_grad_aux(self, params):
        return self._evaluate(params)

    def value_and_grad_aux(self, params):
        losses, gradients, aux = self._evaluate(params[None, :])
        return (
            losses[0],
            gradients[0],
            jax.tree.map(lambda value: value[0], aux),
        )

    def logical_input_arrays(self) -> list[np.ndarray]:
        grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
        for batch, offset, value in self.input_fragments:
            grouped.setdefault(batch, []).append((offset, value))
        result: list[np.ndarray] = []
        for batch in sorted(grouped):
            fragments = sorted(grouped[batch], key=lambda item: item[0])
            expected_offset = 0
            arrays: list[np.ndarray] = []
            for offset, value in fragments:
                if offset != expected_offset:
                    raise RuntimeError("input fragments are not contiguous")
                arrays.append(value)
                expected_offset += len(value)
            result.append(np.concatenate(arrays, axis=0))
        return result

    def logical_input_commitments(self) -> list[dict[str, Any]]:
        return [
            _array_commitment(value) for value in self.logical_input_arrays()
        ]


class Capture:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.raw_events: list[dict[str, np.ndarray]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        hosted = {
            key: np.asarray(jax.device_get(value)) for key, value in event.items()
        }
        self.raw_events.append(hosted)
        self.events.append(_project_event(event))


def _run_optimizer(
    algorithm: str,
    mode: str | None,
    batches: list[Batch],
    *,
    seed: int,
    patience: int,
    max_evals: int | None = None,
    chunk_size: int | None = None,
    attack: str | None = None,
) -> dict[str, Any]:
    objective = ScriptedObjective(
        batches,
        max_evals=(
            max_evals
            if max_evals is not None
            else len(batches) * POPULATION_SIZE
        ),
        attack=attack,
    )
    capture = Capture()
    settings = {
        **COMMON_SETTINGS,
        "random_seed": seed,
        "patience": patience,
        "evaluation_chunk_size": chunk_size,
        "optimizer_telemetry_callback": capture,
    }
    if algorithm == "protected":
        BatchedRestartAdam().optimize(objective, **settings)
    elif algorithm == "candidate":
        if mode is None:
            raise AssertionError("candidate mode is required")
        FeasibilityDebtBatchedRestartAdamV2().optimize(
            objective, progress_mode=mode, **settings
        )
    else:
        raise AssertionError(f"unknown algorithm: {algorithm}")
    projection = {
        "algorithm_str": objective.algorithm_str,
        "best_feasible_loss": (
            None
            if objective.best_feasible_loss == math.inf
            else objective.best_feasible_loss
        ),
        "eval_count": objective.eval_count,
        "events": capture.events,
        "inputs": objective.logical_input_commitments(),
        "random_draws": objective.random_draws,
    }
    return {
        "objective": objective,
        "capture": capture,
        "projection": projection,
        "root": _sha256(_canonical_json(projection)),
    }


def _restart_masks(run: dict[str, Any]) -> list[list[bool]]:
    return [
        event["restart_triggered"].astype(bool).tolist()
        for event in run["capture"].raw_events
    ]


def _restart_batches(run: dict[str, Any]) -> list[int]:
    return [
        index
        for index, mask in enumerate(_restart_masks(run))
        if any(mask)
    ]


def _case_root(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _event_lane_projection(
    events: list[dict[str, np.ndarray]], lanes: tuple[int, ...]
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for event in events:
        row: dict[str, Any] = {}
        for key, value in sorted(event.items()):
            if key in {"time_seconds", "evaluation_batch_seconds"}:
                continue
            array = np.asarray(value)
            if array.ndim >= 1 and array.shape[0] == POPULATION_SIZE:
                row[key] = _array_commitment(array[list(lanes)])
        projected.append(row)
    return projected


def _attack_rejected_before_transition(attack: str) -> bool:
    batches = _batches(
        1,
        lambda _batch, lane: 5.0 + lane,
        lambda _batch, lane: 2.0 + lane,
        lambda _batch, _lane: False,
    )
    objective = ScriptedObjective(
        batches, max_evals=POPULATION_SIZE, attack=attack
    )
    capture = Capture()
    try:
        FeasibilityDebtBatchedRestartAdamV2().optimize(
            objective,
            progress_mode="feasibility_debt",
            **COMMON_SETTINGS,
            random_seed=SEEDS["auxiliary_attacks"],
            patience=8,
            optimizer_telemetry_callback=capture,
        )
    except (KeyError, TypeError, ValueError):
        return (
            objective.eval_count == POPULATION_SIZE
            and len(objective.input_fragments) == 1
            and len(objective.random_draws) == 1
            and capture.events == []
        )
    return False


def _case_projection() -> dict[str, Any]:
    outcomes: dict[str, bool] = {}
    roots: dict[str, str] = {}

    no_restart = _batches(
        6,
        lambda batch, lane: 20.0 - 1.25 * lane - 0.50 * batch,
        lambda batch, lane: 4.0 + 0.10 * lane - 0.20 * batch,
        lambda batch, lane: batch >= 3 + (lane % 2),
    )
    protected = _run_optimizer(
        "protected",
        None,
        no_restart,
        seed=SEEDS["compatibility_no_restart"],
        patience=9,
    )
    compatible = _run_optimizer(
        "candidate",
        "total_loss",
        no_restart,
        seed=SEEDS["compatibility_no_restart"],
        patience=9,
    )
    outcomes["total_loss_no_restart_identity"] = (
        protected["root"] == compatible["root"]
        and _restart_batches(protected) == []
        and _restart_batches(compatible) == []
    )
    roots["total_loss_no_restart_identity"] = _case_root(
        {"protected": protected["root"], "candidate": compatible["root"]}
    )

    mixed_losses = [
        [9.0, 9.0, 9.0, 9.0],
        [9.0, 8.5, 9.0, 8.0],
        [9.0, 8.0, 8.8, 8.0],
        [9.0, 7.5, 8.8, 7.8],
        [8.5, 7.0, 8.6, 7.8],
        [8.5, 6.5, 8.6, 7.6],
        [8.5, 6.0, 8.4, 7.6],
    ]
    mixed_restart = [
        [
            (float(value), 3.0 + 0.1 * lane, False)
            for lane, value in enumerate(row)
        ]
        for row in mixed_losses
    ]
    protected_mixed = _run_optimizer(
        "protected",
        None,
        mixed_restart,
        seed=SEEDS["compatibility_mixed_restart"],
        patience=2,
    )
    compatible_mixed = _run_optimizer(
        "candidate",
        "total_loss",
        mixed_restart,
        seed=SEEDS["compatibility_mixed_restart"],
        patience=2,
    )
    outcomes["total_loss_mixed_restart_identity"] = (
        protected_mixed["root"] == compatible_mixed["root"]
        and any(any(mask) for mask in _restart_masks(protected_mixed))
    )
    roots["total_loss_mixed_restart_identity"] = _case_root(
        {
            "protected": protected_mixed["root"],
            "candidate": compatible_mixed["root"],
        }
    )

    routing_losses = [
        [12, 8, 11, 10],
        [11, 9, 10, 9],
        [10, 10, 9, 8],
        [9, 11, 8, 7],
        [8, 12, 7, 6],
    ]
    routing_penalties = [
        [3, 5, 6, 4],
        [3, 4, 5, 4],
        [3, 3, 4, 4],
        [3, 2, 3, 4],
        [3, 1, 2, 4],
    ]
    routing = [
        [
            (
                float(routing_losses[batch][lane]),
                float(routing_penalties[batch][lane]),
                False,
            )
            for lane in range(POPULATION_SIZE)
        ]
        for batch in range(5)
    ]
    routing_control = _run_optimizer(
        "protected",
        None,
        routing,
        seed=SEEDS["pre_feasibility_lane_routing"],
        patience=3,
    )
    routing_treatment = _run_optimizer(
        "candidate",
        "feasibility_debt",
        routing,
        seed=SEEDS["pre_feasibility_lane_routing"],
        patience=3,
    )
    outcomes["pre_feasibility_lane_routing"] = (
        _restart_masks(routing_control)[3] == [False, True, False, False]
        and _restart_masks(routing_treatment)[3]
        == [True, False, False, True]
        and routing_control["projection"]["inputs"][:4]
        == routing_treatment["projection"]["inputs"][:4]
        and routing_control["projection"]["random_draws"][:2]
        == routing_treatment["projection"]["random_draws"][:2]
    )
    roots["pre_feasibility_lane_routing"] = _case_root(
        {
            "control": routing_control["root"],
            "treatment": routing_treatment["root"],
        }
    )

    handoff_lanes = [
        [
            (10, 4, False),
            (9, 3, False),
            (8, 0, True),
            (7, 2, False),
            (6, 3, False),
            (6, 1, True),
        ],
        [
            (12, 5, False),
            (10, 0, True),
            (10, 1, False),
            (10, 1, False),
            (9, 2, False),
            (8, 0, True),
        ],
        [
            (14, 6, False),
            (13, 5, False),
            (12, 4, False),
            (11, 0, True),
            (10, 4, False),
            (9, 4, False),
        ],
        [
            (16, 8, False),
            (15, 7, False),
            (14, 6, False),
            (13, 5, False),
            (12, 4, False),
            (11, 3, False),
        ],
    ]
    handoff = [
        [
            (
                float(handoff_lanes[lane][batch][0]),
                float(handoff_lanes[lane][batch][1]),
                bool(handoff_lanes[lane][batch][2]),
            )
            for lane in range(POPULATION_SIZE)
        ]
        for batch in range(6)
    ]
    handoff_run = _run_optimizer(
        "candidate",
        "feasibility_debt",
        handoff,
        seed=SEEDS["post_feasibility_handoff"],
        patience=3,
    )
    expected_improved = [
        [True, True, True, True],
        [True, True, True, True],
        [True, False, True, True],
        [True, False, True, True],
        [True, True, True, True],
        [False, True, True, True],
    ]
    expected_ever = [
        [False, False, False, False],
        [False, True, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
    ]
    observed_improved = [
        event["observed_member_improved"].astype(bool).tolist()
        for event in handoff_run["capture"].raw_events
    ]
    observed_ever = [
        event["progress_ever_feasible_after"].astype(bool).tolist()
        for event in handoff_run["capture"].raw_events
    ]
    outcomes["post_feasibility_total_loss_handoff"] = (
        observed_improved == expected_improved
        and observed_ever == expected_ever
        and _restart_batches(handoff_run) == []
    )
    roots["post_feasibility_total_loss_handoff"] = handoff_run["root"]

    isolation = _batches(
        8,
        lambda batch, lane: 30.0 - 2.0 * lane - batch,
        lambda batch, lane: (
            5.0 - math.floor(batch / 3)
            if lane == 0
            else 8.0 - batch
            if lane == 1
            else 7.0 - math.floor(batch / 3)
            if lane == 2
            else max(0.0, 4.0 - batch)
        ),
        lambda batch, lane: lane == 3 and batch >= 2,
    )
    isolation_counterfactual = _batches(
        8,
        lambda batch, lane: 30.0 - 2.0 * lane - batch,
        lambda batch, lane: (
            8.0 - batch
            if lane in (0, 1, 2)
            else max(0.0, 4.0 - batch)
        ),
        lambda batch, lane: lane == 3 and batch >= 2,
    )
    isolation_run = _run_optimizer(
        "candidate",
        "feasibility_debt",
        isolation,
        seed=SEEDS["restart_state_isolation"],
        patience=2,
    )
    counterfactual_run = _run_optimizer(
        "candidate",
        "feasibility_debt",
        isolation_counterfactual,
        seed=SEEDS["restart_state_isolation"],
        patience=2,
    )
    masks = _restart_masks(isolation_run)
    mask_ok = (
        [index for index, mask in enumerate(masks) if any(mask)] == [2, 5]
        and masks[2] == [True, False, True, False]
        and masks[5] == [True, False, True, False]
    )
    unchanged_lane_inputs = all(
        np.array_equal(left[[1, 3]], right[[1, 3]])
        for left, right in zip(
            isolation_run["objective"].logical_input_arrays(),
            counterfactual_run["objective"].logical_input_arrays(),
            strict=True,
        )
    )
    unchanged_lane_events = _event_lane_projection(
        isolation_run["capture"].raw_events, (1, 3)
    ) == _event_lane_projection(
        counterfactual_run["capture"].raw_events, (1, 3)
    )
    reset_ok = True
    for next_batch in (3, 6):
        event = isolation_run["capture"].raw_events[next_batch]
        for lane in (0, 2):
            reset_ok = reset_ok and (
                not bool(event["progress_ever_feasible_before"][lane])
                and math.isinf(
                    float(
                        event["progress_best_infeasible_debt_before"][lane]
                    )
                )
                and math.isinf(
                    float(
                        event[
                            "progress_best_post_feasibility_total_loss_before"
                        ][lane]
                    )
                )
                and int(event["adam_age_before"][lane]) == 0
            )
    oracle_ok = True
    inputs = isolation_run["objective"].logical_input_arrays()
    learning_rates = np.geomspace(0.04, 0.12, POPULATION_SIZE).astype(
        np.float32
    )
    for restart_next, proposal_batch in ((3, 4), (6, 7)):
        for lane in (0, 2):
            params = jnp.asarray(inputs[restart_next][lane : lane + 1])
            expected_params, _, _, _ = (
                FeasibilityDebtBatchedRestartAdamV2._adam_step(
                    params,
                    jnp.asarray(GRADIENTS[lane : lane + 1]),
                    jnp.zeros_like(params),
                    jnp.zeros_like(params),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.asarray([[learning_rates[lane]]]),
                    0.9,
                    0.999,
                    1.0e-8,
                )
            )
            oracle_ok = oracle_ok and np.array_equal(
                np.asarray(jax.device_get(expected_params))[0],
                inputs[proposal_batch][lane],
            )
    outcomes["restart_state_isolation"] = (
        mask_ok
        and unchanged_lane_inputs
        and unchanged_lane_events
        and reset_ok
        and oracle_ok
    )
    roots["restart_state_isolation"] = _case_root(
        {
            "canonical": isolation_run["root"],
            "counterfactual": counterfactual_run["root"],
            "oracle_ok": oracle_ok,
        }
    )

    partial = _batches(
        4,
        lambda batch, lane: 20.0 - batch - 0.1 * lane,
        lambda _batch, lane: 5.0 + lane,
        lambda batch, lane: batch == 3 and lane < 2,
    )
    partial_run = _run_optimizer(
        "candidate",
        "feasibility_debt",
        partial,
        seed=SEEDS["partial_tail_and_chunking"],
        patience=2,
        max_evals=14,
    )
    tail = partial_run["capture"].raw_events[-1]
    outcomes["partial_tail_no_transition"] = (
        partial_run["objective"].eval_count == 14
        and tail["update_applied"].tolist() == [False, False]
        and tail["restart_triggered"].tolist() == [False, False]
        and tail["stalled_steps_before"].tolist() == [0, 0]
        and tail["stalled_steps_after"].tolist() == [0, 0]
        and len(partial_run["projection"]["random_draws"]) == 2
    )
    roots["partial_tail_no_transition"] = partial_run["root"]

    chunk_roots: dict[str, list[str]] = {}
    chunk_ok = True
    for label, script, patience, max_evals in (
        ("handoff", handoff, 3, None),
        ("partial", partial, 2, 14),
    ):
        runs = [
            _run_optimizer(
                "candidate",
                "feasibility_debt",
                script,
                seed=SEEDS["partial_tail_and_chunking"],
                patience=patience,
                max_evals=max_evals,
                chunk_size=chunk,
            )
            for chunk in (None, 1, 2, 3)
        ]
        logical = [
            {
                "events": run["projection"]["events"],
                "inputs": run["projection"]["inputs"],
                "random_draws": run["projection"]["random_draws"],
                "eval_count": run["projection"]["eval_count"],
            }
            for run in runs
        ]
        observed_roots = [_case_root(item) for item in logical]
        chunk_roots[label] = observed_roots
        chunk_ok = chunk_ok and len(set(observed_roots)) == 1
    outcomes["chunk_projection_equivalence"] = chunk_ok
    roots["chunk_projection_equivalence"] = _case_root(chunk_roots)

    attacks = [
        *[
            f"missing:{name}"
            for name in (
                "is_feasible",
                "penalty",
                "sensitivity_loss",
                "violations",
                "power_values",
                "power_values.hard",
                "power_values.soft",
                "power_values.detector",
            )
        ],
        *[
            f"leading:{name}"
            for name in (
                "is_feasible",
                "penalty",
                "sensitivity_loss",
                "violations",
                "power_values.hard",
                "power_values.soft",
                "power_values.detector",
            )
        ],
        "integer-feasibility",
        "scalar-penalty",
        "complex-penalty",
        "list-power-values",
        "negative-penalty",
    ]
    rejected = sum(
        _attack_rejected_before_transition(attack) for attack in attacks
    )
    outcomes["auxiliary_schema_rejection"] = rejected == 20 == len(attacks)
    roots["auxiliary_schema_rejection"] = _case_root(
        {"attacks": attacks, "rejected": rejected}
    )

    nonfinite = [
        [
            (8.0, math.nan, False),
            (8.0, math.inf, False),
            (8.0, -math.inf, False),
            (8.0, 5.0, False),
        ],
        [(7.0, 4.0, False)] * POPULATION_SIZE,
        [(10.0, 0.0, True)] * POPULATION_SIZE,
        [
            (math.nan, 1.0, False),
            (math.inf, 1.0, False),
            (-math.inf, 1.0, False),
            (9.0, 1.0, False),
        ],
        [(9.0, 2.0, False)] * POPULATION_SIZE,
    ]
    nonfinite_run = _run_optimizer(
        "candidate",
        "feasibility_debt",
        nonfinite,
        seed=SEEDS["nonfinite_semantics"],
        patience=8,
    )
    nonfinite_improved = [
        event["observed_member_improved"].astype(bool).tolist()
        for event in nonfinite_run["capture"].raw_events
    ]
    outcomes["nonfinite_progress_semantics"] = nonfinite_improved == [
        [False, False, False, True],
        [True, True, True, True],
        [True, True, True, True],
        [False, False, False, True],
        [True, True, True, False],
    ]
    roots["nonfinite_progress_semantics"] = nonfinite_run["root"]

    source = verify_source_boundary()
    outcomes["source_delta_boundary"] = bool(source["valid"])
    roots["source_delta_boundary"] = source["boundary_root_sha256"]

    if tuple(outcomes) != CASE_KEYS or tuple(roots) != CASE_KEYS:
        raise RuntimeError("case projection order does not match frozen keys")
    return {
        "case_outcomes": outcomes,
        "case_roots": roots,
        "candidate_source_sha256": source["candidate_text_sha256"],
        "protected_source_sha256": source["protected_text_sha256"],
        "source_boundary_root_sha256": source["boundary_root_sha256"],
    }


def _install_child_guards() -> None:
    def deny(*_args, **_kwargs):
        raise RuntimeError("operation denied")

    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(marker in str(mode) for marker in ("w", "a", "x", "+")):
            return deny(file, mode)
        return original_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if any(marker in str(mode) for marker in ("w", "a", "x", "+")):
            return deny(file, mode)
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        write_flags = (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        if int(flags) & write_flags:
            return deny(path, flags)
        return original_os_open(path, flags, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    socket.socket = deny
    socket.create_connection = deny
    subprocess.Popen = deny


def _child_projection() -> dict[str, Any]:
    invocation_revision = os.environ.get("FDC_V2_INVOCATION_REVISION", "")
    if not _is_hex(invocation_revision, 40):
        raise RuntimeError("missing invocation revision")
    if _sha256(PLAN_PATH.read_bytes()) != PLAN_SHA256:
        raise RuntimeError("plan hash mismatch")
    _install_child_guards()
    projection = _case_projection()
    outcomes = projection["case_outcomes"]
    payload: dict[str, Any] = {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "candidate_source_sha256": projection["candidate_source_sha256"],
        "fixture_source_sha256": _sha256(Path(__file__).read_bytes()),
        "protected_source_sha256": projection["protected_source_sha256"],
        "case_count": len(CASE_KEYS),
        "case_outcomes": outcomes,
        "case_roots": projection["case_roots"],
        "all_cases_passed": all(outcomes.values()),
        "source_boundary_root_sha256": projection[
            "source_boundary_root_sha256"
        ],
    }
    payload["core_root_sha256"] = _sha256(_canonical_json(payload))
    return payload


def _strict_json_loads(value: bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    return json.loads(value.decode("utf-8"), object_pairs_hook=pairs_hook)


CHILD_KEYS = {
    "study_id",
    "invocation_revision",
    "plan_revision",
    "plan_sha256",
    "candidate_source_sha256",
    "fixture_source_sha256",
    "protected_source_sha256",
    "case_count",
    "case_outcomes",
    "case_roots",
    "all_cases_passed",
    "source_boundary_root_sha256",
    "core_root_sha256",
}


def _validate_child_payload(payload: Any, raw: bytes) -> bool:
    if not isinstance(payload, dict) or set(payload) != CHILD_KEYS:
        return False
    if raw != _canonical_json(payload) + b"\n":
        return False
    if not _is_hex(payload.get("invocation_revision"), 40):
        return False
    if payload.get("plan_revision") != PLAN_REVISION:
        return False
    if payload.get("plan_sha256") != PLAN_SHA256:
        return False
    for key in (
        "candidate_source_sha256",
        "fixture_source_sha256",
        "protected_source_sha256",
        "source_boundary_root_sha256",
        "core_root_sha256",
    ):
        if not _is_hex(payload.get(key), 64):
            return False
    if type(payload.get("case_count")) is not int or payload["case_count"] != 10:
        return False
    outcomes = payload.get("case_outcomes")
    roots = payload.get("case_roots")
    if (
        not isinstance(outcomes, dict)
        or set(outcomes) != set(CASE_KEYS)
        or any(type(value) is not bool for value in outcomes.values())
        or not isinstance(roots, dict)
        or set(roots) != set(CASE_KEYS)
        or any(not _is_hex(value, 64) for value in roots.values())
    ):
        return False
    if type(payload.get("all_cases_passed")) is not bool:
        return False
    if payload["all_cases_passed"] != all(outcomes.values()):
        return False
    without_root = {
        key: value for key, value in payload.items() if key != "core_root_sha256"
    }
    return payload["core_root_sha256"] == _sha256(
        _canonical_json(without_root)
    )


def _scrubbed_environment() -> dict[str, str]:
    forbidden = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
    retained = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in forbidden)
    }
    retained.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "*",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return retained


def _run_child(
    invocation_revision: str, child_number: int
) -> tuple[dict[str, bool], bytes | None, dict[str, Any] | None]:
    prefix = f"child_{child_number}_"
    receipt = {
        f"{prefix}{key}": False
        for key in (
            "process_started",
            "exit_code_zero",
            "stderr_empty",
            "stdout_within_cap",
            "json_parsed",
            "schema_valid",
            "study_identity_valid",
        )
    }
    environment = _scrubbed_environment()
    environment["FDC_V2_INVOCATION_REVISION"] = invocation_revision
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.candidates.feasibility_debt_clock_v2_fixture",
                "--child",
            ],
            cwd=Path(__file__).resolve().parents[2],
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
    receipt[f"{prefix}stdout_within_cap"] = (
        len(completed.stdout) <= MAX_CHILD_BYTES
    )
    payload = None
    if receipt[f"{prefix}stdout_within_cap"]:
        try:
            parsed = _strict_json_loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            receipt[f"{prefix}json_parsed"] = True
            receipt[f"{prefix}schema_valid"] = _validate_child_payload(
                parsed, completed.stdout
            )
            receipt[f"{prefix}study_identity_valid"] = (
                isinstance(parsed, dict) and parsed.get("study_id") == STUDY_ID
            )
            if all(receipt.values()):
                payload = parsed
    return receipt, completed.stdout if payload is not None else None, payload


def _current_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "0" * 40
    value = completed.stdout.decode("ascii", errors="ignore").strip()
    return (
        value
        if completed.returncode == 0 and _is_hex(value, 40)
        else "0" * 40
    )


def _closed_parent_failure(
    invocation_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision or _current_revision(),
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "candidate_source_sha256": None,
        "fixture_source_sha256": None,
        "protected_source_sha256": None,
        "case_count": len(CASE_KEYS),
        "case_outcomes": {key: False for key in CASE_KEYS},
        "transport_outcomes": {key: False for key in TRANSPORT_KEYS},
        "all_cases_passed": False,
        "runs_equal": False,
        "source_boundary_root_sha256": None,
        "process_replay_root_sha256": None,
        "action": "park_feasibility_debt_v2",
    }


PARENT_KEYS = {
    "study_id",
    "invocation_revision",
    "plan_revision",
    "plan_sha256",
    "candidate_source_sha256",
    "fixture_source_sha256",
    "protected_source_sha256",
    "case_count",
    "case_outcomes",
    "transport_outcomes",
    "all_cases_passed",
    "runs_equal",
    "source_boundary_root_sha256",
    "process_replay_root_sha256",
    "action",
}


def _validate_parent_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != PARENT_KEYS:
        return False
    if payload.get("study_id") != STUDY_ID:
        return False
    if not _is_hex(payload.get("invocation_revision"), 40):
        return False
    if (
        payload.get("plan_revision") != PLAN_REVISION
        or payload.get("plan_sha256") != PLAN_SHA256
    ):
        return False
    for key in (
        "candidate_source_sha256",
        "fixture_source_sha256",
        "protected_source_sha256",
        "source_boundary_root_sha256",
        "process_replay_root_sha256",
    ):
        if payload.get(key) is not None and not _is_hex(payload[key], 64):
            return False
    if type(payload.get("case_count")) is not int or payload["case_count"] != 10:
        return False
    outcomes = payload.get("case_outcomes")
    transport = payload.get("transport_outcomes")
    if (
        not isinstance(outcomes, dict)
        or set(outcomes) != set(CASE_KEYS)
        or any(type(value) is not bool for value in outcomes.values())
        or not isinstance(transport, dict)
        or set(transport) != set(TRANSPORT_KEYS)
        or any(type(value) is not bool for value in transport.values())
    ):
        return False
    if (
        type(payload.get("all_cases_passed")) is not bool
        or type(payload.get("runs_equal")) is not bool
    ):
        return False
    expected_pass = (
        all(outcomes.values())
        and all(transport.values())
        and payload["runs_equal"]
    )
    required_pass_hashes = (
        "candidate_source_sha256",
        "fixture_source_sha256",
        "protected_source_sha256",
        "source_boundary_root_sha256",
        "process_replay_root_sha256",
    )
    if expected_pass and any(payload[key] is None for key in required_pass_hashes):
        return False
    if payload["runs_equal"] != (
        payload["process_replay_root_sha256"] is not None
    ):
        return False
    if payload["all_cases_passed"] != expected_pass:
        return False
    expected_action = (
        "approve_feasibility_debt_v2_for_fresh_panel_planning"
        if expected_pass
        else "park_feasibility_debt_v2"
    )
    return payload.get("action") == expected_action


def run_terminal_projection() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    invocation_revision = _current_revision()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _closed_parent_failure(invocation_revision)
    if (
        status.returncode != 0
        or status.stderr
        or status.stdout
        or invocation_revision == "0" * 40
        or _sha256(PLAN_PATH.read_bytes()) != PLAN_SHA256
    ):
        return _closed_parent_failure(invocation_revision)

    first_receipt, first_bytes, first = _run_child(invocation_revision, 1)
    second_receipt, second_bytes, second = _run_child(invocation_revision, 2)
    transport = {**first_receipt, **second_receipt}
    valid_payload = first if first is not None else second
    outcomes = (
        dict(valid_payload["case_outcomes"])
        if valid_payload is not None
        else {key: False for key in CASE_KEYS}
    )
    runs_equal = (
        first is not None
        and second is not None
        and first_bytes == second_bytes
        and first == second
    )
    all_passed = (
        all(outcomes.values()) and all(transport.values()) and runs_equal
    )
    payload = {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": PLAN_REVISION,
        "plan_sha256": PLAN_SHA256,
        "candidate_source_sha256": (
            valid_payload["candidate_source_sha256"] if valid_payload else None
        ),
        "fixture_source_sha256": (
            valid_payload["fixture_source_sha256"] if valid_payload else None
        ),
        "protected_source_sha256": (
            valid_payload["protected_source_sha256"] if valid_payload else None
        ),
        "case_count": len(CASE_KEYS),
        "case_outcomes": outcomes,
        "transport_outcomes": transport,
        "all_cases_passed": all_passed,
        "runs_equal": runs_equal,
        "source_boundary_root_sha256": (
            valid_payload["source_boundary_root_sha256"]
            if valid_payload
            else None
        ),
        "process_replay_root_sha256": (
            _sha256(first_bytes)
            if runs_equal and first_bytes is not None
            else None
        ),
        "action": (
            "approve_feasibility_debt_v2_for_fresh_panel_planning"
            if all_passed
            else "park_feasibility_debt_v2"
        ),
    }
    return (
        payload
        if _validate_parent_payload(payload)
        else _closed_parent_failure(invocation_revision)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--child", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.child:
        try:
            payload = _child_projection()
            raw = _canonical_json(payload) + b"\n"
            if not _validate_child_payload(payload, raw):
                return 2
            sys.stdout.buffer.write(raw)
            return 0
        except Exception:
            return 2

    try:
        payload = run_terminal_projection()
    except Exception:
        payload = _closed_parent_failure()
    if not _validate_parent_payload(payload):
        revision = (
            payload.get("invocation_revision")
            if isinstance(payload, dict)
            else None
        )
        payload = _closed_parent_failure(revision)
    sys.stdout.buffer.write(_canonical_json(payload) + b"\n")
    return 0 if payload["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
