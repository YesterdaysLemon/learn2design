"""Executed full-surface prefix information-boundary fixture."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import submission.submission as submission_module
from submission.submission import BatchedRestartAdam


STUDY_ID = "full-surface-prefix-indistinguishability-v1"
SCHEMA_VERSION = 1
SEED = 20260827
POPULATION_SIZE = 4
PARAMETER_DIMENSION = 3
MAX_BOUND = 8
LATE_CROSSING_STEP = MAX_BOUND + 1
MAX_EVALS = POPULATION_SIZE * LATE_CROSSING_STEP
PATIENCE = 64
LEARNING_RATE = 0.05
HYPERPARAMETERS = {
    "batch_time_safety_factor": 1.5,
    "batch_time_window": 8,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
    "learning_rate": LEARNING_RATE,
    "minimum_improvement": 1e-7,
    "patience": PATIENCE,
    "population_size": POPULATION_SIZE,
    "restart_noise_scale": 0.35,
    "safety_seconds": 0.0,
}
CLAIM_BOUNDARY = "executed_synthetic_normal_path_next_feasibility_only"
TOP_LEVEL_FIELDS = [
    "candidate",
    "loss",
    "total_gradient",
    "aux",
    "optimizer_state",
    "rng_transcript",
    "budget_counters",
    "incumbent_state",
]
AUX_LEAF_PATHS = [
    "aux.is_feasible",
    "aux.penalty",
    "aux.power_values.detector",
    "aux.power_values.hard",
    "aux.power_values.soft",
    "aux.sensitivity_loss",
    "aux.violations",
]
TELEMETRY_FIELDS = {
    "adam_age_after",
    "adam_age_before",
    "batch_index",
    "budget_progress_fraction",
    "eval_count_after_batch",
    "evaluated_generation",
    "evaluation_batch_seconds",
    "feasible",
    "finite_loss",
    "global_feasible_improvement",
    "gradient_clip_scale",
    "gradient_nonfinite_count",
    "gradient_norm",
    "learning_rate",
    "loss_float_bits",
    "member_index",
    "next_generation",
    "observed_member_best_loss",
    "observed_member_improved",
    "restart_kind",
    "restart_noise_scale",
    "restart_round",
    "restart_triggered",
    "stalled_steps_after",
    "stalled_steps_before",
    "time_seconds",
    "update_applied",
}
TELEMETRY_DTYPES = {
    "adam_age_after": "<i4",
    "adam_age_before": "<i4",
    "batch_index": "<i4",
    "budget_progress_fraction": "<f8",
    "eval_count_after_batch": "<i4",
    "evaluated_generation": "<i4",
    "evaluation_batch_seconds": "<f8",
    "feasible": "|b1",
    "finite_loss": "|b1",
    "global_feasible_improvement": "|b1",
    "gradient_clip_scale": "<f4",
    "gradient_nonfinite_count": "<i4",
    "gradient_norm": "<f4",
    "learning_rate": "<f8",
    "loss_float_bits": "<i2",
    "member_index": "<i2",
    "next_generation": "<i4",
    "observed_member_best_loss": "<f4",
    "observed_member_improved": "|b1",
    "restart_kind": "|i1",
    "restart_noise_scale": "<f8",
    "restart_round": "<i4",
    "restart_triggered": "|b1",
    "stalled_steps_after": "<i4",
    "stalled_steps_before": "<i4",
    "time_seconds": "<f8",
    "update_applied": "|b1",
}
SIGNAL_CONTROL_PATHS = [
    ("candidate", "candidate.sha256"),
    ("loss", "loss.sha256"),
    ("total_gradient", "total_gradient.sha256"),
    (
        "optimizer_state",
        "optimizer_state.transition_state_sha256",
    ),
    ("rng_transcript", "rng_transcript.transcript_sha256"),
    ("budget_counters", "budget_counters.eval_count_after"),
    ("incumbent_state", "incumbent_state.present"),
]
SIGNAL_CLASSES = [name for name, _path in SIGNAL_CONTROL_PATHS]
FORBIDDEN_EXTENSION_CLASSES = [
    "unlogged_callable",
    "extra_evaluation",
    "hessian",
    "manual_log",
    "private_attribute",
    "saved_record",
    "structural_metadata",
]
CASE_CONTRACT = {
    "normal_path_execution": {
        "batches": LATE_CROSSING_STEP,
        "max_evals": MAX_EVALS,
        "parameter_dimension": PARAMETER_DIMENSION,
        "population_size": POPULATION_SIZE,
        "worlds": ["forever_infeasible", "late_crossing"],
    },
    "adapter_schema": {
        "array_projection_fields": ["dtype", "shape", "sha256"],
        "aux_leaf_paths": AUX_LEAF_PATHS,
        "exact_schema_required": True,
        "top_level_fields": TOP_LEVEL_FIELDS,
    },
    "shared_full_surface_prefix": {
        "bound": MAX_BOUND,
        "decision_timing": "observe_then_decide",
        "late_crossing_step": LATE_CROSSING_STEP,
        "next_primary_difference": "aux.is_feasible",
    },
    "signal_class_negative_controls": {
        "control_names": SIGNAL_CLASSES,
        "exact_diff_paths_per_control": 1,
    },
    "aux_leaf_negative_controls": {
        "aux_leaf_paths": AUX_LEAF_PATHS,
        "exact_diff_paths_per_control": 1,
    },
    "typed_array_metadata_boundary": {
        "controls": ["dtype", "shape"],
        "raw_bytes_held_equal": True,
    },
    "forbidden_extension_rejection": {
        "extension_classes": FORBIDDEN_EXTENSION_CLASSES,
        "scope": "exact_schema_sentinel_only",
    },
    "action_vector_exhaustion": {
        "action_vector_count": 1 << MAX_BOUND,
        "bound": MAX_BOUND,
        "scope": "abstract_corollary_on_shared_executed_prefix",
    },
    "process_isolation": {
        "source_projection": "complete_non_process_cases",
        "workers": 2,
    },
}
REPOSITORY_ROOT = Path(__file__).parents[2]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _array_bytes(value: object) -> bytes:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + array.tobytes()


def _array_projection(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(_array_bytes(array)).hexdigest(),
    }


def _tree_projection(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(name): _tree_projection(child)
            for name, child in sorted(value.items())
        }
    return _array_projection(value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_array_projection(
    value: object,
    *,
    expected_dtype: str | None = None,
    expected_shape: list[int] | None = None,
) -> bool:
    if not (
        isinstance(value, dict)
        and set(value) == {"dtype", "shape", "sha256"}
        and isinstance(value["dtype"], str)
        and isinstance(value["shape"], list)
        and all(
            isinstance(dimension, int) and dimension >= 0
            for dimension in value["shape"]
        )
        and isinstance(value["sha256"], str)
        and SHA256_PATTERN.fullmatch(value["sha256"]) is not None
    ):
        return False
    try:
        dtype = np.dtype(value["dtype"])
    except TypeError:
        return False
    if dtype.str != value["dtype"]:
        return False
    if expected_dtype is not None and value["dtype"] != expected_dtype:
        return False
    return expected_shape is None or value["shape"] == expected_shape


def _valid_snapshot(snapshot: object) -> bool:
    if not isinstance(snapshot, dict) or set(snapshot) != set(TOP_LEVEL_FIELDS):
        return False
    budget_hint = snapshot["budget_counters"]
    batch_hint = (
        budget_hint.get("batch_index") if isinstance(budget_hint, dict) else None
    )
    if type(batch_hint) is not int or not 0 <= batch_hint < LATE_CROSSING_STEP:
        return False
    expected_candidate_dtype = "<f4" if batch_hint == 0 else "<f8"
    if not _valid_array_projection(
        snapshot["candidate"],
        expected_dtype=expected_candidate_dtype,
        expected_shape=[POPULATION_SIZE, PARAMETER_DIMENSION],
    ) or not _valid_array_projection(
        snapshot["loss"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE],
    ) or not _valid_array_projection(
        snapshot["total_gradient"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE, PARAMETER_DIMENSION],
    ):
        return False

    aux = snapshot["aux"]
    if not isinstance(aux, dict) or set(aux) != {
        "is_feasible",
        "penalty",
        "power_values",
        "sensitivity_loss",
        "violations",
    }:
        return False
    power_values = aux["power_values"]
    if not isinstance(power_values, dict) or set(power_values) != {
        "detector",
        "hard",
        "soft",
    }:
        return False
    if not _valid_array_projection(
        aux["is_feasible"],
        expected_dtype="|b1",
        expected_shape=[POPULATION_SIZE],
    ) or not _valid_array_projection(
        aux["penalty"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE],
    ) or not _valid_array_projection(
        aux["sensitivity_loss"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE],
    ) or not _valid_array_projection(
        aux["violations"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE, PARAMETER_DIMENSION],
    ) or not _valid_array_projection(
        power_values["detector"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE, 2],
    ) or not _valid_array_projection(
        power_values["hard"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE, 1],
    ) or not _valid_array_projection(
        power_values["soft"],
        expected_dtype="<f4",
        expected_shape=[POPULATION_SIZE, 2],
    ):
        return False

    optimizer_state = snapshot["optimizer_state"]
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
        "completed_batches",
        "telemetry",
        "hyperparameters_sha256",
        "prior_state_sha256",
        "timing_window_micros",
        "transition_state_sha256",
    }:
        return False
    telemetry = optimizer_state["telemetry"]
    if (
        not isinstance(telemetry, dict)
        or set(telemetry) != TELEMETRY_FIELDS
        or not all(
            _valid_array_projection(
                value,
                expected_dtype=TELEMETRY_DTYPES[name],
                expected_shape=[POPULATION_SIZE],
            )
            for name, value in telemetry.items()
        )
    ):
        return False
    if not (
        type(optimizer_state["completed_batches"]) is int
        and 1 <= optimizer_state["completed_batches"] <= LATE_CROSSING_STEP
    ) or not (
        isinstance(optimizer_state["timing_window_micros"], list)
        and all(
            isinstance(value, int) and value >= 0
            for value in optimizer_state["timing_window_micros"]
        )
        and len(optimizer_state["timing_window_micros"]) <= 8
    ):
        return False
    if any(
        not isinstance(optimizer_state[name], str)
        or SHA256_PATTERN.fullmatch(optimizer_state[name]) is None
        for name in (
            "hyperparameters_sha256",
            "prior_state_sha256",
            "transition_state_sha256",
        )
    ):
        return False
    expected_hyperparameters_sha256 = _json_sha256(HYPERPARAMETERS)
    if optimizer_state["hyperparameters_sha256"] != expected_hyperparameters_sha256:
        return False

    rng = snapshot["rng_transcript"]
    if not isinstance(rng, dict) or set(rng) != {
        "calls_before_batch",
        "draw_count",
        "transcript_sha256",
    }:
        return False
    if (
        type(rng["calls_before_batch"]) is not int
        or type(rng["draw_count"]) is not int
        or rng["calls_before_batch"] != 1
        or rng["draw_count"] != 1
        or not isinstance(rng["transcript_sha256"], str)
        or SHA256_PATTERN.fullmatch(rng["transcript_sha256"]) is None
    ):
        return False

    budget = snapshot["budget_counters"]
    if not isinstance(budget, dict) or set(budget) != {
        "batch_index",
        "budget_exceeded_after",
        "budget_progress_ppm_after",
        "eval_count_after",
        "eval_count_before",
        "evaluation_limit",
        "evals_left_after",
        "evals_left_before",
        "time_elapsed_micros_after",
        "time_left_micros_before",
    }:
        return False
    integer_budget_fields = set(budget) - {"budget_exceeded_after"}
    if not isinstance(budget["budget_exceeded_after"], bool) or not all(
        type(budget[name]) is int for name in integer_budget_fields
    ):
        return False
    if not (
        0 <= budget["batch_index"] < LATE_CROSSING_STEP
        and budget["evaluation_limit"] == MAX_EVALS
        and budget["eval_count_before"] == POPULATION_SIZE * budget["batch_index"]
        and budget["eval_count_after"]
        == budget["eval_count_before"] + POPULATION_SIZE
        and budget["evals_left_before"]
        == budget["evaluation_limit"] - budget["eval_count_before"]
        and budget["evals_left_after"]
        == budget["evaluation_limit"] - budget["eval_count_after"]
        and budget["time_left_micros_before"] >= 0
        and budget["time_elapsed_micros_after"] >= 0
        and 0 <= budget["budget_progress_ppm_after"] <= 1_000_000
        and budget["budget_exceeded_after"]
        == (budget["eval_count_after"] >= budget["evaluation_limit"])
    ):
        return False

    incumbent = snapshot["incumbent_state"]
    if not isinstance(incumbent, dict) or set(incumbent) != {
        "present",
        "best_loss",
        "candidate_sha256",
    }:
        return False
    if not isinstance(incumbent["present"], bool):
        return False
    if incumbent["present"]:
        if not _valid_array_projection(
            incumbent["best_loss"], expected_dtype="<f4", expected_shape=[1]
        ):
            return False
        if (
            not isinstance(incumbent["candidate_sha256"], str)
            or SHA256_PATTERN.fullmatch(incumbent["candidate_sha256"]) is None
        ):
            return False
    elif (
        incumbent["best_loss"] is not None
        or incumbent["candidate_sha256"] is not None
    ):
        return False
    transition_inputs = {
        "completed_batches": optimizer_state["completed_batches"],
        "evaluation": {
            name: snapshot[name]
            for name in TOP_LEVEL_FIELDS
            if name != "optimizer_state"
        },
        "hyperparameters_sha256": optimizer_state["hyperparameters_sha256"],
        "prior_state_sha256": optimizer_state["prior_state_sha256"],
        "telemetry": optimizer_state["telemetry"],
        "timing_window_micros": optimizer_state["timing_window_micros"],
    }
    if optimizer_state["transition_state_sha256"] != _json_sha256(
        transition_inputs
    ):
        return False
    return True


def _valid_snapshot_sequence(snapshots: object) -> bool:
    if not isinstance(snapshots, list) or not snapshots:
        return False
    prior_transition = None
    for batch_index, snapshot in enumerate(snapshots):
        if not _valid_snapshot(snapshot):
            return False
        optimizer_state = snapshot["optimizer_state"]
        budget = snapshot["budget_counters"]
        if (
            optimizer_state["completed_batches"] != batch_index + 1
            or budget["batch_index"] != batch_index
            or optimizer_state["timing_window_micros"]
            != [4_000] * min(batch_index, 8)
        ):
            return False
        if (
            prior_transition is not None
            and optimizer_state["prior_state_sha256"] != prior_transition
        ):
            return False
        prior_transition = optimizer_state["transition_state_sha256"]
    return True


class _DeterministicClock:
    """Supply repeatable synthetic batch durations to the protected path."""

    def __init__(self) -> None:
        self._ticks = -1

    def perf_counter(self) -> float:
        self._ticks += 1
        return self._ticks * 0.004


class _OnlineSurfaceAdapter:
    """Project only the current evaluation and public telemetry callbacks."""

    def __init__(self) -> None:
        self.snapshots: list[dict[str, object]] = []
        self.restart_count = 0
        self.rng_draws: list[dict[str, object]] = []
        self.initial_population: dict[str, object] | None = None
        self.pending_evaluation: dict[str, object] | None = None
        self.timing_window_micros: list[int] = []
        self.prefix_all_finite = True
        self.prefix_all_infeasible = True
        self.prefix_strictly_improving = True
        self.incumbent_loss_value = float("inf")
        self.incumbent_state: dict[str, object] = {
            "present": False,
            "best_loss": None,
            "candidate_sha256": None,
        }
        self.hyperparameters_sha256 = _json_sha256(HYPERPARAMETERS)
        self.prior_state_sha256: str | None = None

    def observe_rng_draw(self, sample: object) -> None:
        self.rng_draws.append(
            {
                "call_index": len(self.rng_draws),
                "sample": _array_projection(sample),
            }
        )

    def observe_initial_population(self, value: object) -> None:
        if self.initial_population is not None:
            raise RuntimeError("initial population was observed twice")
        self.initial_population = _array_projection(value)
        self.prior_state_sha256 = _json_sha256(
            {
                "hyperparameters_sha256": self.hyperparameters_sha256,
                "initial_population": self.initial_population,
                "rng_transcript_sha256": _json_sha256(self.rng_draws),
            }
        )

    def observe_evaluation(
        self,
        *,
        batch_index: int,
        params: object,
        losses: object,
        grads: object,
        aux: dict[str, object],
        budget: dict[str, object],
    ) -> None:
        if self.pending_evaluation is not None:
            raise RuntimeError("telemetry did not close the prior evaluation")
        host_params = np.asarray(jax.device_get(params))
        host_losses = np.asarray(jax.device_get(losses))
        host_feasible = np.asarray(
            jax.device_get(aux["is_feasible"]), dtype=bool
        )
        if batch_index < MAX_BOUND:
            self.prefix_all_finite = self.prefix_all_finite and bool(
                np.isfinite(host_losses).all()
            )
            self.prefix_all_infeasible = self.prefix_all_infeasible and not bool(
                host_feasible.any()
            )
            if batch_index > 0:
                previous_losses = (
                    100.0
                    - float(batch_index - 1)
                    + np.arange(POPULATION_SIZE, dtype=np.float32) / 8.0
                )
                self.prefix_strictly_improving = (
                    self.prefix_strictly_improving
                    and bool(np.all(host_losses < previous_losses))
                )

        if bool(host_feasible.any()):
            feasible_indices = np.flatnonzero(host_feasible)
            local_index = int(np.argmin(host_losses[feasible_indices]))
            best_index = int(feasible_indices[local_index])
            best_loss = float(host_losses[best_index])
            if best_loss < self.incumbent_loss_value:
                self.incumbent_loss_value = best_loss
                self.incumbent_state = {
                    "present": True,
                    "best_loss": _array_projection(host_losses[best_index]),
                    "candidate_sha256": hashlib.sha256(
                        _array_bytes(host_params[best_index])
                    ).hexdigest(),
                }

        self.pending_evaluation = {
            "candidate": _array_projection(params),
            "loss": _array_projection(losses),
            "total_gradient": _array_projection(grads),
            "aux": _tree_projection(aux),
            "rng_transcript": {
                "calls_before_batch": len(self.rng_draws),
                "draw_count": len(self.rng_draws),
                "transcript_sha256": _json_sha256(self.rng_draws),
            },
            "budget_counters": budget,
            "incumbent_state": copy.deepcopy(self.incumbent_state),
        }

    def observe_telemetry(self, event: dict[str, object]) -> None:
        if self.pending_evaluation is None or self.prior_state_sha256 is None:
            raise RuntimeError("telemetry arrived outside a complete evaluation")
        if set(event) != TELEMETRY_FIELDS:
            raise RuntimeError("the protected optimizer telemetry surface drifted")
        hosted = {
            name: np.asarray(jax.device_get(value))
            for name, value in event.items()
        }
        if {int(value.shape[0]) for value in hosted.values()} != {POPULATION_SIZE}:
            raise RuntimeError("malformed full-surface telemetry event")
        batch_indices = np.unique(hosted["batch_index"])
        durations = np.unique(hosted["evaluation_batch_seconds"])
        if batch_indices.size != 1 or durations.size != 1:
            raise RuntimeError("telemetry members disagree on batch metadata")
        batch_index = int(batch_indices[0])
        duration_micros = int(round(float(durations[0]) * 1_000_000))
        if batch_index > 0:
            self.timing_window_micros.append(duration_micros)
            self.timing_window_micros = self.timing_window_micros[-8:]

        telemetry = {
            name: _array_projection(value) for name, value in hosted.items()
        }
        transition_inputs = {
            "completed_batches": batch_index + 1,
            "evaluation": self.pending_evaluation,
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "telemetry": telemetry,
            "timing_window_micros": list(self.timing_window_micros),
        }
        transition_state_sha256 = _json_sha256(transition_inputs)
        snapshot = dict(self.pending_evaluation)
        snapshot["optimizer_state"] = {
            "completed_batches": batch_index + 1,
            "telemetry": telemetry,
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "timing_window_micros": list(self.timing_window_micros),
            "transition_state_sha256": transition_state_sha256,
        }
        self.snapshots.append(
            {name: snapshot[name] for name in TOP_LEVEL_FIELDS}
        )
        self.prior_state_sha256 = transition_state_sha256
        self.pending_evaluation = None
        self.restart_count += int(np.asarray(hosted["restart_triggered"]).sum())


class _ScriptedObjective:
    """Public-shape synthetic objective executed by the protected optimizer."""

    def __init__(self, *, world: str, adapter: _OnlineSurfaceAdapter) -> None:
        if world not in {"forever_infeasible", "late_crossing"}:
            raise ValueError(f"unknown full-surface world: {world}")
        self.world = world
        self.adapter = adapter
        self.n_params = PARAMETER_DIMENSION
        self.max_evals = MAX_EVALS
        self.eval_count = 0
        self.algorithm_str = ""
        self.unbounded = False
        self._key = jax.random.PRNGKey(0)
        self._started = False
        self.start_logging_count = 0
        self.scalar_calls = 0
        self.rng_call_count = 0
        self.batch_count = 0
        self.optimization_pairs = [
            [f"synthetic_full_surface_coordinate_{index}", "tuning"]
            for index in range(self.n_params)
        ]

    def set_space_mode(self, unbounded: bool) -> None:
        self.unbounded = bool(unbounded)

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.PRNGKey(seed)

    def random_params_unbounded(self, n_samples: int = 1):
        self._key, sample_key = jax.random.split(self._key)
        sample = jax.random.uniform(
            sample_key,
            shape=(n_samples, self.n_params),
            minval=-1.5,
            maxval=1.5,
            dtype=jnp.float32,
        )
        self.adapter.observe_rng_draw(sample)
        self.rng_call_count += 1
        return sample

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
    def time_left(self) -> float:
        return 1000.0 - self.time_elapsed

    @property
    def time_elapsed(self) -> float:
        return self.eval_count / POPULATION_SIZE * 0.01

    def start_logging(self) -> None:
        if self._started:
            raise RuntimeError("scripted objective logging started twice")
        self._started = True
        self.start_logging_count += 1

    def _script(self, batch: int, active_members: int):
        members = jnp.arange(active_members, dtype=jnp.float32)
        losses = 100.0 - jnp.asarray(batch, dtype=jnp.float32) + members / 8.0
        gradient_row = jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float32)
        grads = jnp.broadcast_to(gradient_row, (active_members, self.n_params))
        feasible = jnp.zeros((active_members,), dtype=bool)
        if self.world == "late_crossing" and batch == MAX_BOUND:
            feasible = feasible.at[0].set(True)
        penalty = 20.0 + members
        sensitivity_loss = losses - penalty
        violations = jnp.stack(
            (members + 1.0, members + 2.0, members + 3.0), axis=1
        )
        power_values = {
            "detector": jnp.stack((members + 10.0, members + 11.0), axis=1),
            "hard": (members + 4.0)[:, None],
            "soft": jnp.stack((members + 6.0, members + 7.0), axis=1),
        }
        aux = {
            "is_feasible": feasible,
            "penalty": penalty,
            "power_values": power_values,
            "sensitivity_loss": sensitivity_loss,
            "violations": violations,
        }
        return losses, grads, aux

    def _budget_projection_before(self, batch: int) -> dict[str, object]:
        return {
            "batch_index": batch,
            "eval_count_before": int(self.eval_count),
            "evals_left_before": int(self.evals_left),
            "evaluation_limit": int(self.max_evals),
            "time_left_micros_before": int(round(self.time_left * 1_000_000)),
        }

    def vmap_value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        active_members = int(params.shape[0])
        if active_members != POPULATION_SIZE:
            raise RuntimeError("the full-surface fixture forbids partial tails")
        batch = self.batch_count
        budget = self._budget_projection_before(batch)
        losses, grads, aux = self._script(batch, active_members)
        self.eval_count += active_members
        budget.update(
            {
                "budget_exceeded_after": bool(self.budget_exceeded),
                "budget_progress_ppm_after": int(
                    round(self.budget_progress_fraction * 1_000_000)
                ),
                "eval_count_after": int(self.eval_count),
                "evals_left_after": int(self.evals_left),
                "time_elapsed_micros_after": int(
                    round(self.time_elapsed * 1_000_000)
                ),
            }
        )
        self.adapter.observe_evaluation(
            batch_index=batch,
            params=params,
            losses=losses,
            grads=grads,
            aux=aux,
            budget=budget,
        )
        self.batch_count += 1
        return losses, grads, aux

    def value_and_grad_aux(self, params):
        del params
        self.scalar_calls += 1
        raise RuntimeError("the frozen full-surface fixture requires full batches")


def _initial_population() -> np.ndarray:
    return np.linspace(
        -1.0,
        1.0,
        POPULATION_SIZE * PARAMETER_DIMENSION,
        dtype=np.float32,
    ).reshape(POPULATION_SIZE, PARAMETER_DIMENSION)


def _run_trace(world: str) -> dict[str, object]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the full-surface study requires a CPU backend")
    adapter = _OnlineSurfaceAdapter()
    objective = _ScriptedObjective(world=world, adapter=adapter)

    original_time = submission_module.time
    submission_module.time = _DeterministicClock()
    try:
        with (
            jax.default_device(cpu_devices[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            BatchedRestartAdam().optimize(
                objective,
                init_params=_initial_population(),
                random_seed=SEED,
                population_size=POPULATION_SIZE,
                learning_rate_low=LEARNING_RATE,
                learning_rate_high=LEARNING_RATE,
                patience=PATIENCE,
                safety_seconds=0.0,
                initial_population_callback=adapter.observe_initial_population,
                optimizer_telemetry_callback=adapter.observe_telemetry,
            )
    finally:
        submission_module.time = original_time

    if adapter.initial_population is None:
        raise RuntimeError("initial population was not captured exactly once")
    if not (
        objective.batch_count
        == len(adapter.snapshots)
        == LATE_CROSSING_STEP
    ):
        raise RuntimeError("the executed full-surface trace has the wrong length")

    return {
        "batch_sizes": [POPULATION_SIZE] * objective.batch_count,
        "eval_count": objective.eval_count,
        "initial_population": adapter.initial_population,
        "prefix_all_finite": adapter.prefix_all_finite,
        "prefix_all_infeasible": adapter.prefix_all_infeasible,
        "prefix_strictly_improving": adapter.prefix_strictly_improving,
        "restart_count": adapter.restart_count,
        "rng_calls": objective.rng_call_count,
        "scalar_calls": objective.scalar_calls,
        "snapshots": adapter.snapshots,
        "start_logging_count": objective.start_logging_count,
        "state_commitments": len(adapter.snapshots),
        "telemetry_events": len(adapter.snapshots),
    }


def _diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for name in set(left) | set(right):
            path = f"{prefix}.{name}" if prefix else name
            if name not in left or name not in right:
                paths.add(path)
            else:
                paths.update(_diff_paths(left[name], right[name], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = set()
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.add(path)
            else:
                paths.update(_diff_paths(left[index], right[index], path))
        return paths
    return set() if left == right else {prefix}


def _mutate_path(value: object, path: str) -> object:
    changed = copy.deepcopy(value)
    cursor = changed
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise TypeError("negative-control path left the frozen object schema")
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise TypeError("negative-control path did not end in an object")
    leaf = parts[-1]
    original = cursor[leaf]
    if isinstance(original, bool):
        cursor[leaf] = not original
    elif isinstance(original, int):
        cursor[leaf] = original + 1
    elif isinstance(original, str):
        cursor[leaf] = ("0" if original[0] != "0" else "1") + original[1:]
    else:
        raise TypeError("unsupported frozen negative-control leaf")
    return changed


def _policy_partition() -> dict[str, int]:
    total = 1 << MAX_BOUND
    bounded_only = 0
    preserve_only = 0
    joint = 0
    for action_mask in range(total):
        restarts_by_bound = action_mask != 0
        preserves_late_crossing = action_mask == 0
        if restarts_by_bound and preserves_late_crossing:
            joint += 1
        elif restarts_by_bound:
            bounded_only += 1
        elif preserves_late_crossing:
            preserve_only += 1
    return {
        "bounded_only": bounded_only,
        "joint": joint,
        "preserve_only": preserve_only,
        "total": total,
    }


def _proof_projection() -> dict[str, object]:
    forever = _run_trace("forever_infeasible")
    late = _run_trace("late_crossing")
    forever_prefix = forever["snapshots"][:MAX_BOUND]
    late_prefix = late["snapshots"][:MAX_BOUND]
    forever_next = forever["snapshots"][MAX_BOUND]
    late_next = late["snapshots"][MAX_BOUND]
    baseline_snapshot = forever_prefix[-1]

    missing_aux = copy.deepcopy(baseline_snapshot)
    del missing_aux["aux"]["violations"]
    extra_field = copy.deepcopy(baseline_snapshot)
    extra_field["extension"] = True

    signal_control_exact = {
        name: _diff_paths(
            baseline_snapshot,
            _mutate_path(baseline_snapshot, path),
        )
        == {path}
        for name, path in SIGNAL_CONTROL_PATHS
    }
    aux_control_exact = {
        path: _diff_paths(
            baseline_snapshot,
            _mutate_path(baseline_snapshot, f"{path}.sha256"),
        )
        == {f"{path}.sha256"}
        for path in AUX_LEAF_PATHS
    }
    extension_rejections = {}
    for extension in FORBIDDEN_EXTENSION_CLASSES:
        extended = copy.deepcopy(baseline_snapshot)
        extended[extension] = True
        extension_rejections[extension] = not _valid_snapshot(extended)

    base = np.asarray([1, 2], dtype=np.int16)
    shape_changed = base.reshape(1, 2)
    dtype_changed = base.view(np.uint16)
    metadata_controls = {
        "dtype": (
            base.tobytes() == dtype_changed.tobytes()
            and _array_projection(base) != _array_projection(dtype_changed)
        ),
        "shape": (
            base.tobytes() == shape_changed.tobytes()
            and _array_projection(base) != _array_projection(shape_changed)
        ),
    }

    forever_evaluation = {
        name: forever_next[name]
        for name in ("candidate", "loss", "total_gradient", "aux")
    }
    late_evaluation = {
        name: late_next[name]
        for name in ("candidate", "loss", "total_gradient", "aux")
    }
    next_evaluation_diff_paths = sorted(
        _diff_paths(forever_evaluation, late_evaluation)
    )
    return {
        "adapter": {
            "baseline_valid": _valid_snapshot(baseline_snapshot),
            "extra_field_rejected": not _valid_snapshot(extra_field),
            "missing_aux_rejected": not _valid_snapshot(missing_aux),
            "schema_sha256": _json_sha256(
                {
                    "array_projection_fields": ["dtype", "shape", "sha256"],
                    "aux_leaf_paths": AUX_LEAF_PATHS,
                    "telemetry_fields": sorted(TELEMETRY_FIELDS),
                    "top_level_fields": TOP_LEVEL_FIELDS,
                }
            ),
        },
        "aux_control_exact": aux_control_exact,
        "extension_rejections": extension_rejections,
        "forever_prefix_sha256": _json_sha256(forever_prefix),
        "late_prefix_sha256": _json_sha256(late_prefix),
        "metadata_controls": metadata_controls,
        "next_evaluation_diff_paths": next_evaluation_diff_paths,
        "next_full_snapshots_differ": forever_next != late_next,
        "prefix_all_finite": bool(
            forever["prefix_all_finite"] and late["prefix_all_finite"]
        ),
        "prefix_all_infeasible": bool(
            forever["prefix_all_infeasible"]
            and late["prefix_all_infeasible"]
        ),
        "prefix_strictly_improving": bool(
            forever["prefix_strictly_improving"]
            and late["prefix_strictly_improving"]
        ),
        "normal_path": {
            "forever": {
                name: value for name, value in forever.items() if name != "snapshots"
            },
            "late": {
                name: value for name, value in late.items() if name != "snapshots"
            },
        },
        "partition": _policy_partition(),
        "prefix_snapshots_valid": (
            _valid_snapshot_sequence(forever["snapshots"])
            and _valid_snapshot_sequence(late["snapshots"])
        ),
        "prefixes_identical": forever_prefix == late_prefix,
        "signal_control_exact": signal_control_exact,
    }


def isolated_worker_trace() -> dict[str, object]:
    """Return the timing-free projection used by isolation checks."""
    return _proof_projection()


def _isolated_trace() -> dict[str, object]:
    safe_names = {
        "COMSPEC",
        "LD_LIBRARY_PATH",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "VIRTUAL_ENV",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name.upper() in safe_names
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.local_lab.full_surface_prefix_worker",
            "--mode",
            "full-surface-prefix-trace",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def run_study(*, include_process_isolation: bool = True) -> dict[str, object]:
    """Execute the complete frozen case set and return a sanitized result."""
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the full-surface study requires a CPU backend")

    proof = _proof_projection()
    adapter = proof["adapter"]
    normal_path = proof["normal_path"]
    partition = proof["partition"]
    assert isinstance(adapter, dict)
    assert isinstance(normal_path, dict)
    assert isinstance(partition, dict)

    expected_path_projection = {
        "batch_sizes": [POPULATION_SIZE] * LATE_CROSSING_STEP,
        "eval_count": MAX_EVALS,
        "prefix_all_finite": True,
        "prefix_all_infeasible": True,
        "prefix_strictly_improving": True,
        "restart_count": 0,
        "rng_calls": 1,
        "scalar_calls": 0,
        "start_logging_count": 1,
        "state_commitments": LATE_CROSSING_STEP,
        "telemetry_events": LATE_CROSSING_STEP,
    }
    path_checks = []
    initial_population_digests = []
    for world in ("forever", "late"):
        projection = normal_path[world]
        initial_population_digests.append(projection["initial_population"])
        comparable = {
            name: value
            for name, value in projection.items()
            if name != "initial_population"
        }
        path_checks.append(comparable == expected_path_projection)
    normal_path_passed = all(path_checks) and (
        initial_population_digests[0] == initial_population_digests[1]
    )
    adapter_passed = all(
        bool(adapter[name])
        for name in ("baseline_valid", "extra_field_rejected", "missing_aux_rejected")
    )
    prefix_passed = (
        bool(proof["prefix_snapshots_valid"])
        and bool(proof["prefix_all_finite"])
        and bool(proof["prefix_all_infeasible"])
        and bool(proof["prefix_strictly_improving"])
        and bool(proof["prefixes_identical"])
        and proof["forever_prefix_sha256"] == proof["late_prefix_sha256"]
        and proof["next_evaluation_diff_paths"] == ["aux.is_feasible.sha256"]
        and bool(proof["next_full_snapshots_differ"])
    )
    signal_controls = proof["signal_control_exact"]
    aux_controls = proof["aux_control_exact"]
    metadata_controls = proof["metadata_controls"]
    extension_controls = proof["extension_rejections"]
    signal_passed = set(signal_controls) == set(SIGNAL_CLASSES) and all(
        bool(value) for value in signal_controls.values()
    )
    aux_passed = set(aux_controls) == set(AUX_LEAF_PATHS) and all(
        bool(value) for value in aux_controls.values()
    )
    metadata_passed = set(metadata_controls) == {"dtype", "shape"} and all(
        bool(value) for value in metadata_controls.values()
    )
    extension_passed = set(extension_controls) == set(
        FORBIDDEN_EXTENSION_CLASSES
    ) and all(bool(value) for value in extension_controls.values())
    action_passed = (
        partition["total"] == 1 << MAX_BOUND
        and partition["bounded_only"] == (1 << MAX_BOUND) - 1
        and partition["preserve_only"] == 1
        and partition["joint"] == 0
    )

    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = isolated_left == isolated_right == proof
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = "not-run-in-focused-test"

    cases = {
        "normal_path_execution": {
            "batches_per_world": LATE_CROSSING_STEP,
            "evaluations_per_world": MAX_EVALS,
            "passed": normal_path_passed,
            "restart_events": 0,
            "rng_draws_per_world": 1,
            "scalar_calls": 0,
            "state_commitments_per_world": LATE_CROSSING_STEP,
            "telemetry_events_per_world": LATE_CROSSING_STEP,
            "worlds": 2,
        },
        "adapter_schema": {
            "array_projection_fields": ["dtype", "shape", "sha256"],
            "aux_leaf_paths": AUX_LEAF_PATHS,
            "exact_schema_required": True,
            "passed": adapter_passed,
            "snapshot_schema_sha256": adapter["schema_sha256"],
            "top_level_fields": TOP_LEVEL_FIELDS,
        },
        "shared_full_surface_prefix": {
            "bound": MAX_BOUND,
            "forever_prefix_sha256": proof["forever_prefix_sha256"],
            "late_prefix_sha256": proof["late_prefix_sha256"],
            "next_evaluation_difference_paths": proof[
                "next_evaluation_diff_paths"
            ],
            "next_full_snapshots_differ": proof["next_full_snapshots_differ"],
            "passed": prefix_passed,
            "prefix_all_finite": proof["prefix_all_finite"],
            "prefix_all_infeasible": proof["prefix_all_infeasible"],
            "prefix_strictly_improving": proof["prefix_strictly_improving"],
            "prefixes_identical": proof["prefixes_identical"],
        },
        "signal_class_negative_controls": {
            "control_names": SIGNAL_CLASSES,
            "controls_checked": len(signal_controls),
            "exact_single_path_controls": sum(
                bool(value) for value in signal_controls.values()
            ),
            "passed": signal_passed,
        },
        "aux_leaf_negative_controls": {
            "aux_leaf_paths": AUX_LEAF_PATHS,
            "controls_checked": len(aux_controls),
            "exact_single_path_controls": sum(
                bool(value) for value in aux_controls.values()
            ),
            "passed": aux_passed,
        },
        "typed_array_metadata_boundary": {
            "controls": ["dtype", "shape"],
            "controls_checked": len(metadata_controls),
            "passed": metadata_passed,
            "typed_identity_changes": sum(
                bool(value) for value in metadata_controls.values()
            ),
        },
        "forbidden_extension_rejection": {
            "extensions_checked": len(extension_controls),
            "extensions_rejected": sum(
                bool(value) for value in extension_controls.values()
            ),
            "passed": extension_passed,
            "scope": "exact_schema_sentinel_only",
        },
        "action_vector_exhaustion": {
            "bound": MAX_BOUND,
            "bounded_only_transcripts": partition["bounded_only"],
            "joint_satisfiers": partition["joint"],
            "passed": action_passed,
            "preserve_only_transcripts": partition["preserve_only"],
            "scope": "abstract_corollary_on_shared_executed_prefix",
            "total_action_vectors": partition["total"],
        },
        "process_isolation": {
            "passed": isolation_passed,
            "trace_sha256": isolation_digest,
        },
    }
    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    return {
        "action": (
            "synthetic_full_surface_prefix_twin_confirmed"
            if passed
            else (
                "park_full_surface_prefix_research"
                if completed
                else "no_decision_incomplete_study"
            )
        ),
        "cases": cases,
        "environment": {
            "device_kind": str(cpu_devices[0].device_kind),
            "jax_version": str(jax.__version__),
            "platform": str(cpu_devices[0].platform),
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
        },
        "fixture": {
            "case_contract": CASE_CONTRACT,
            "aux_leaf_paths": AUX_LEAF_PATHS,
            "claim_boundary": CLAIM_BOUNDARY,
            "decision_timing": "observe_then_decide",
            "execution_adapter": "protected_optimizer_scripted_public_surface",
            "forbidden_extension_classes": FORBIDDEN_EXTENSION_CLASSES,
            "late_crossing_offset": 1,
            "max_bound": MAX_BOUND,
            "parameter_dimension": PARAMETER_DIMENSION,
            "policy_projection": "binary_actions_on_one_shared_executed_prefix",
            "population_size": POPULATION_SIZE,
            "signal_classes": SIGNAL_CLASSES,
            "snapshot_top_level_fields": TOP_LEVEL_FIELDS,
            "typed_array_projection_fields": ["dtype", "shape", "sha256"],
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
