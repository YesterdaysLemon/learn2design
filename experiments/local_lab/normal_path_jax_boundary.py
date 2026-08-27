"""Frozen CPU mechanics for the protected normal-path JAX boundary."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import dfbench.core.objective as objective_module
import submission.submission as submission_module
from dfbench.core.objective import Objective
from dfbench.core.problem import ContinuousProblem
from submission.submission import BatchedRestartAdam


STUDY_ID = "normal-path-jax-boundary-v1"
SCHEMA_VERSION = 1
DFBENCH_VERSION = "0.3.3"
DFBENCH_WHEEL_SHA256 = (
    "1f96d75b813ea42f93992da5c1f50d6a4f59dd7a507bcf561676b0e416378c43"
)
OBJECTIVE_SOURCE_SHA256 = (
    "9e2c2bb54517f59efacf4c2a59908ffd55c7fb2e15089d53263ece796e71daa2"
)
SUBMISSION_SOURCE_SHA256 = (
    "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
)
JAX_VERSION = "0.9.0.1"
JAXLIB_VERSION = "0.9.0.1"
POPULATION_SIZE = 4
PARAMETER_DIMENSION = 3
SEED = 20260827
MAX_EVALS = 4
PATIENCE = 64
LEARNING_RATE_LOW = 0.05
LEARNING_RATE_HIGH = 0.11
MINIMUM_IMPROVEMENT = 1e-7
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
GRADIENT_CLIP_NORM = 1.0
BATCH_MICROSECONDS = 4000
EXPECTED_TYPED_MATCHES = 46
CLAIM_BOUNDARY = "synthetic_cpu_one_batch_no_restart_exact_typed_equivalence"

AUX_LEAF_PATHS = [
    "is_feasible",
    "penalty",
    "power_values.detector",
    "power_values.hard",
    "power_values.soft",
    "sensitivity_loss",
    "violations",
]
TELEMETRY_LEAVES = [
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
]

CASE_CONTRACT = {
    "dependency_source_identity": {
        "dfbench_version": DFBENCH_VERSION,
        "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
        "jax_version": JAX_VERSION,
        "jaxlib_version": JAXLIB_VERSION,
        "objective_source_sha256": OBJECTIVE_SOURCE_SHA256,
        "submission_source_sha256": SUBMISSION_SOURCE_SHA256,
    },
    "source_boundary_inventory": {
        "objective_logging_device_host_sites": 5,
        "optimizer_callback_sites": 4,
        "optimizer_clock_sites": 2,
        "optimizer_device_scalar_sites": 3,
        "optimizer_jit_sites": 0,
        "optimizer_ready_sites": 2,
    },
    "normal_path_boundary_trace": {
        "batch_shape": [POPULATION_SIZE, PARAMETER_DIMENSION],
        "batch_microseconds": BATCH_MICROSECONDS,
        "evaluations": MAX_EVALS,
        "explicit_barriers": 1,
        "performance_clock_reads": 2,
        "wall_clock_reads": 3,
    },
    "pure_jax_transition_equivalence": {
        "adam_state_leaves": 4,
        "aux_leaf_paths": AUX_LEAF_PATHS,
        "expected_typed_matches": EXPECTED_TYPED_MATCHES,
        "telemetry_leaves": TELEMETRY_LEAVES,
    },
    "explicit_jit_lowering": {
        "compiled_calls": 1,
        "forbidden_callback_primitives": [
            "host_callback",
            "io_callback",
            "pure_callback",
        ],
        "required_exact_modes": ["eager", "jit", "compiled"],
    },
    "boundary_negative_controls": {
        "callback_sentinel": "pure_callback",
        "single_path_controls": [
            "budget.eval_count",
            "state.candidate.sha256",
            "telemetry.feasible.sha256",
            "timing.batch_micros",
        ],
        "typed_controls": ["dtype", "shape"],
    },
    "process_isolation": {
        "source_projection": "complete_non_process_cases",
        "workers": 2,
    },
}

FIXTURE_IDENTITY = {
    "aux_leaf_paths": AUX_LEAF_PATHS,
    "batch_microseconds": BATCH_MICROSECONDS,
    "beta1": BETA1,
    "beta2": BETA2,
    "claim_boundary": CLAIM_BOUNDARY,
    "dfbench_version": DFBENCH_VERSION,
    "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
    "epsilon": EPSILON,
    "gradient_clip_norm": GRADIENT_CLIP_NORM,
    "jax_version": JAX_VERSION,
    "jaxlib_version": JAXLIB_VERSION,
    "learning_rate_high": LEARNING_RATE_HIGH,
    "learning_rate_low": LEARNING_RATE_LOW,
    "max_evals": MAX_EVALS,
    "minimum_improvement": MINIMUM_IMPROVEMENT,
    "objective_source_sha256": OBJECTIVE_SOURCE_SHA256,
    "parameter_dimension": PARAMETER_DIMENSION,
    "patience": PATIENCE,
    "population_size": POPULATION_SIZE,
    "seed": SEED,
    "submission_source_sha256": SUBMISSION_SOURCE_SHA256,
    "telemetry_leaves": TELEMETRY_LEAVES,
}

REPOSITORY_ROOT = Path(__file__).parents[2]
PRIVATE_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT.with_name(f"{REPOSITORY_ROOT.name}-local-lab")
    / "worker-tmp"
    / "normal-path-jax-boundary-checkpoints"
)


def _normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_projection(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _tree_projection(value: object) -> object:
    if isinstance(value, dict):
        return {key: _tree_projection(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_tree_projection(item) for item in value]
    return _array_projection(value)


def _typed_leaf_count(value: object) -> int:
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "sha256"}:
        return 1
    if isinstance(value, dict):
        return sum(_typed_leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_typed_leaf_count(item) for item in value)
    return 0


def _diff_paths(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            return [prefix]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _diff_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        return paths
    return [] if left == right else [prefix]


def _locked_dfbench_projection() -> dict[str, str]:
    with (REPOSITORY_ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    packages = [
        package
        for package in lock.get("package", [])
        if package.get("name") == "dfbench"
    ]
    if len(packages) != 1:
        raise RuntimeError("dependency lock has no unique dfbench package")
    wheel_hashes = [
        wheel.get("hash", "").removeprefix("sha256:")
        for wheel in packages[0].get("wheels", [])
    ]
    if len(wheel_hashes) != 1:
        raise RuntimeError("dependency lock has no unique dfbench wheel")
    return {
        "dfbench_version": str(packages[0].get("version")),
        "dfbench_wheel_sha256": wheel_hashes[0],
    }


class _SyntheticBoundaryProblem(ContinuousProblem):
    """Small differentiable problem with the current public aux shape."""

    name = "synthetic_normal_path_boundary"

    def __init__(self) -> None:
        self.optimization_pairs = [
            ("synthetic", "x0"),
            ("synthetic", "x1"),
            ("synthetic", "x2"),
        ]

        def objective_function_aux(params):
            center = jnp.asarray([0.25, -0.5, 0.75], dtype=params.dtype)
            sensitivity_loss = jnp.sum(jnp.square(params - center))
            violations = jnp.abs(params)
            penalty = jnp.asarray(0.125, dtype=params.dtype) * jnp.sum(violations)
            aux = {
                "is_feasible": jnp.all(params == jnp.zeros_like(params)),
                "penalty": penalty,
                "power_values": {
                    "detector": jnp.square(params[2:3]) + 0.75,
                    "hard": jnp.square(params[:1]) + 0.5,
                    "soft": jnp.square(params[1:2]) + 0.25,
                },
                "sensitivity_loss": sensitivity_loss,
                "violations": violations,
            }
            return sensitivity_loss + penalty, aux

        def objective_function(params):
            value, _ = objective_function_aux(params)
            return value

        self.objective_function = jax.jit(objective_function)
        self.objective_function_aux = jax.jit(objective_function_aux)

    @property
    def bounds(self):
        return jnp.asarray(
            [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=jnp.float32
        )

    def to_spec(self) -> dict[str, object]:
        return {"type": type(self).__name__}


class _TraceObjective(Objective):
    """Public Objective path with label-only boundary hooks."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.trace_enabled = False
        self.public_call_count = 0
        self.scalar_call_count = 0
        self.warmup_call_count = 0
        self.rng_call_count = 0
        self.public_batch_shape: list[int] | None = None
        self.evaluated_candidate = None
        self.observed_losses = None
        self.observed_grads = None
        self.observed_aux = None
        self.in_public_evaluation = False
        self.in_budget_progress = False
        super().__init__(
            _SyntheticBoundaryProblem(),
            unbounded=False,
            max_evals=MAX_EVALS,
            max_time=None,
            save_time_steps=False,
            save_params_history=False,
            save=[],
            verbose=0,
            checkpoint_dir=PRIVATE_CHECKPOINT_ROOT,
        )
        self.trace_enabled = True

    def _event(self, name: str) -> None:
        if self.trace_enabled:
            self.events.append(name)

    def set_space_mode(self, *args, **kwargs):
        self._event("set_space_mode")
        return super().set_space_mode(*args, **kwargs)

    def set_seed(self, seed: int) -> None:
        self._event("set_seed")
        super().set_seed(seed)

    def random_params_unbounded(self, *args, **kwargs):
        self._event("rng_draw")
        self.rng_call_count += 1
        return super().random_params_unbounded(*args, **kwargs)

    def start_logging(self) -> None:
        self._event("start_logging")
        super().start_logging()

    def vmap_value_and_grad_aux(self, params):
        self._event("public_evaluation_enter")
        self.public_call_count += 1
        self.public_batch_shape = list(params.shape)
        self.evaluated_candidate = params
        self.in_public_evaluation = True
        try:
            losses, grads, aux = super().vmap_value_and_grad_aux(params)
        finally:
            self.in_public_evaluation = False
        self.observed_losses = losses
        self.observed_grads = grads
        self.observed_aux = aux
        self._event("public_evaluation_return")
        return losses, grads, aux

    def value_and_grad_aux(self, params):
        self._event("scalar_evaluation")
        self.scalar_call_count += 1
        return super().value_and_grad_aux(params)

    def warmup_value_and_grad_aux(self):
        self._event("scalar_warmup")
        self.warmup_call_count += 1
        return super().warmup_value_and_grad_aux()

    def warmup_vmap_value_and_grad_aux(self, population_size: int):
        self._event("batch_warmup")
        self.warmup_call_count += 1
        return super().warmup_vmap_value_and_grad_aux(population_size)

    @property
    def budget_exceeded(self) -> bool:
        self._event("budget_exceeded_read")
        return Objective.budget_exceeded.fget(self)

    @property
    def time_left(self):
        self._event("time_left_read")
        return Objective.time_left.fget(self)

    @property
    def evals_left(self):
        self._event("evals_left_read")
        return Objective.evals_left.fget(self)

    @property
    def eval_count(self) -> int:
        self._event(
            "budget_progress_eval_count_read"
            if self.in_budget_progress
            else "eval_count_read"
        )
        return Objective.eval_count.fget(self)

    @property
    def time_elapsed(self) -> float:
        self._event(
            "objective_log_time_elapsed_read"
            if self.in_public_evaluation
            else "optimizer_time_elapsed_read"
        )
        return Objective.time_elapsed.fget(self)

    @property
    def budget_progress_fraction(self) -> float:
        self._event("budget_progress_read")
        self.in_budget_progress = True
        try:
            return Objective.budget_progress_fraction.fget(self)
        finally:
            self.in_budget_progress = False


class _Callbacks:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.raw_initial = None
        self.final_initial = None
        self.telemetry = None

    def raw_initial_callback(self, params) -> None:
        self.events.append("raw_initial_callback")
        self.raw_initial = params

    def final_initial_callback(self, params) -> None:
        self.events.append("final_initial_callback")
        self.final_initial = params

    def telemetry_callback(self, event) -> None:
        self.events.append("telemetry_callback")
        self.telemetry = event


class _DeterministicWallClock:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.values = [1000.0, 1000.25, 1000.5]
        self.calls = 0

    def time(self) -> float:
        self.events.append("wall_clock_read")
        if self.calls >= len(self.values):
            raise RuntimeError("the frozen wall clock received an extra read")
        value = self.values[self.calls]
        self.calls += 1
        return value


class _DeterministicPerformanceClock:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.values = [0.0, BATCH_MICROSECONDS / 1_000_000]
        self.calls = 0

    def perf_counter(self) -> float:
        label = "performance_clock_start" if self.calls == 0 else "performance_clock_stop"
        self.events.append(label)
        if self.calls >= len(self.values):
            raise RuntimeError("the frozen performance clock received an extra read")
        value = self.values[self.calls]
        self.calls += 1
        return value


def pure_jax_transition(
    params,
    losses,
    grads,
    aux,
    first_moment,
    second_moment,
    member_steps,
    member_best_loss,
    stalled_steps,
    learning_rates,
    global_feasible_loss,
    global_feasible_params,
    member_generations,
    batch_index,
    eval_count,
    time_seconds,
    evaluation_batch_seconds,
    budget_progress_fraction,
    minimum_improvement,
    beta1,
    beta2,
    epsilon,
    gradient_clip_norm,
    patience,
):
    """Pure no-restart form of the protected full-batch state transition."""
    population_size = params.shape[0]
    feasible = aux["is_feasible"]
    finite_loss = jnp.isfinite(losses)
    feasible_losses = jnp.where(
        finite_loss & jnp.asarray(feasible, dtype=bool), losses, jnp.inf
    )
    feasible_index = jnp.argmin(feasible_losses)
    feasible_loss = feasible_losses[feasible_index]
    global_feasible_improved = feasible_loss < global_feasible_loss
    next_global_feasible_loss = jnp.where(
        global_feasible_improved, feasible_loss, global_feasible_loss
    )
    next_global_feasible_params = jnp.where(
        global_feasible_improved,
        params[feasible_index],
        global_feasible_params,
    )

    improved = finite_loss & (
        losses < member_best_loss - minimum_improvement
    )
    next_member_best_loss = jnp.where(improved, losses, member_best_loss)
    next_stalled_steps = jnp.where(improved, 0, stalled_steps + 1)

    gradient_nonfinite_count = jnp.sum(
        ~jnp.isfinite(grads), axis=1, dtype=jnp.int32
    )
    sanitized_grads = jnp.nan_to_num(
        grads, nan=0.0, posinf=0.0, neginf=0.0
    )
    gradient_norms = jnp.linalg.norm(sanitized_grads, axis=1)
    gradient_clip_scales = jnp.minimum(
        1.0, gradient_clip_norm / (gradient_norms + 1e-12)
    )
    clipped_grads = sanitized_grads * gradient_clip_scales[:, None]

    next_member_steps = member_steps + 1
    next_first_moment = (
        beta1 * first_moment + (1.0 - beta1) * clipped_grads
    )
    next_second_moment = (
        beta2 * second_moment
        + (1.0 - beta2) * jnp.square(clipped_grads)
    )
    member_ages = next_member_steps[:, None]
    corrected_first = next_first_moment / (
        1.0 - jnp.power(beta1, member_ages)
    )
    corrected_second = next_second_moment / (
        1.0 - jnp.power(beta2, member_ages)
    )
    next_params = params - learning_rates * corrected_first / (
        jnp.sqrt(corrected_second) + epsilon
    )

    restart_mask = next_stalled_steps >= patience
    member_ids = jnp.arange(population_size, dtype=jnp.int16)
    restart_kind = jnp.full((population_size,), -1, dtype=jnp.int8)
    restart_scales = jnp.full((population_size,), jnp.nan)
    telemetry = {
        "adam_age_after": next_member_steps,
        "adam_age_before": member_steps,
        "batch_index": jnp.full(
            (population_size,), batch_index, dtype=jnp.int32
        ),
        "budget_progress_fraction": jnp.full(
            (population_size,), budget_progress_fraction
        ),
        "eval_count_after_batch": jnp.full(
            (population_size,), eval_count, dtype=jnp.int32
        ),
        "evaluated_generation": member_generations,
        "evaluation_batch_seconds": jnp.full(
            (population_size,), evaluation_batch_seconds
        ),
        "feasible": jnp.asarray(feasible, dtype=bool),
        "finite_loss": finite_loss,
        "global_feasible_improvement": (
            member_ids == feasible_index
        ) & global_feasible_improved,
        "gradient_clip_scale": gradient_clip_scales,
        "gradient_nonfinite_count": gradient_nonfinite_count,
        "gradient_norm": gradient_norms,
        "learning_rate": learning_rates[:, 0],
        "loss_float_bits": jnp.full(
            (population_size,), losses.dtype.itemsize * 8, dtype=jnp.int16
        ),
        "member_index": member_ids,
        "next_generation": member_generations,
        "observed_member_best_loss": next_member_best_loss,
        "observed_member_improved": improved,
        "restart_kind": restart_kind,
        "restart_noise_scale": restart_scales,
        "restart_round": jnp.full(
            (population_size,), -1, dtype=jnp.int32
        ),
        "restart_triggered": restart_mask,
        "stalled_steps_after": next_stalled_steps,
        "stalled_steps_before": stalled_steps,
        "time_seconds": jnp.full((population_size,), time_seconds),
        "update_applied": jnp.ones((population_size,), dtype=bool),
    }
    state = {
        "candidate": next_params,
        "first_moment": next_first_moment,
        "global_feasible_loss": next_global_feasible_loss,
        "global_feasible_params": next_global_feasible_params,
        "member_best_loss": next_member_best_loss,
        "member_generations": member_generations,
        "member_steps": next_member_steps,
        "second_moment": next_second_moment,
        "stalled_steps": next_stalled_steps,
    }
    observation = {
        "aux": aux,
        "candidate": params,
        "loss": losses,
        "total_gradient": grads,
    }
    return observation, state, telemetry


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _pure_source_host_calls() -> list[str]:
    source = textwrap.dedent(inspect.getsource(pure_jax_transition))
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if (
            name in {"bool", "float", "int"}
            or name == "jax.device_get"
            or name.startswith("jax.random.")
            or name.startswith("numpy.")
            or name.startswith("np.")
            or name.startswith("time.")
            or "callback" in name
        ):
            forbidden.append(name)
    return sorted(forbidden)


def _primitive_names(closed_jaxpr) -> list[str]:
    names: list[str] = []
    visited: set[int] = set()

    def visit(value) -> None:
        identity = id(value)
        if identity in visited:
            return
        if hasattr(value, "jaxpr"):
            visited.add(identity)
            visit(value.jaxpr)
            return
        if hasattr(value, "eqns"):
            visited.add(identity)
            for equation in value.eqns:
                names.append(equation.primitive.name)
                for parameter in equation.params.values():
                    visit(parameter)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(closed_jaxpr)
    return names


def _callback_primitives(closed_jaxpr) -> list[str]:
    return sorted({name for name in _primitive_names(closed_jaxpr) if "callback" in name})


def _run_protected_batch() -> dict[str, object]:
    events: list[str] = []
    callbacks = _Callbacks(events)
    objective = _TraceObjective(events)
    optimizer = BatchedRestartAdam()
    wall_clock = _DeterministicWallClock(events)
    performance_clock = _DeterministicPerformanceClock(events)

    original_objective_time = objective_module.time
    original_submission_time = submission_module.time
    original_ready = jax.block_until_ready

    def traced_ready(value):
        events.append("explicit_ready_enter")
        ready = original_ready(value)
        events.append("explicit_ready_return")
        return ready

    objective_module.time = wall_clock
    submission_module.time = performance_clock
    jax.block_until_ready = traced_ready
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer.optimize(
                objective,
                random_seed=SEED,
                population_size=POPULATION_SIZE,
                learning_rate_low=LEARNING_RATE_LOW,
                learning_rate_high=LEARNING_RATE_HIGH,
                patience=PATIENCE,
                minimum_improvement=MINIMUM_IMPROVEMENT,
                beta1=BETA1,
                beta2=BETA2,
                epsilon=EPSILON,
                gradient_clip_norm=GRADIENT_CLIP_NORM,
                safety_seconds=0.0,
                initial_population_mode="random",
                evaluation_chunk_size=None,
                preclock_warmup=False,
                raw_initial_population_callback=callbacks.raw_initial_callback,
                initial_population_callback=callbacks.final_initial_callback,
                optimizer_telemetry_callback=callbacks.telemetry_callback,
            )
    finally:
        jax.block_until_ready = original_ready
        submission_module.time = original_submission_time
        objective_module.time = original_objective_time

    required = (
        callbacks.raw_initial,
        callbacks.final_initial,
        callbacks.telemetry,
        objective.evaluated_candidate,
        objective.observed_losses,
        objective.observed_grads,
        objective.observed_aux,
    )
    if any(value is None for value in required):
        raise RuntimeError("the protected boundary trace did not capture every reference")
    telemetry = callbacks.telemetry
    if not isinstance(telemetry, dict):
        raise RuntimeError("the protected boundary trace returned malformed telemetry")
    sanitized_grads = jnp.nan_to_num(
        objective.observed_grads, nan=0.0, posinf=0.0, neginf=0.0
    )
    clipped_grads = sanitized_grads * telemetry["gradient_clip_scale"][:, None]
    params = objective.evaluated_candidate
    learning_rates = jnp.geomspace(
        LEARNING_RATE_LOW, LEARNING_RATE_HIGH, POPULATION_SIZE
    )[:, None]
    adam_output = BatchedRestartAdam._adam_step(
        params,
        clipped_grads,
        jnp.zeros_like(params),
        jnp.zeros_like(params),
        jnp.zeros((POPULATION_SIZE,), dtype=jnp.int32),
        learning_rates,
        BETA1,
        BETA2,
        EPSILON,
    )
    objective.trace_enabled = False
    evaluation_count_after = objective.eval_count

    return {
        "adam_output": adam_output,
        "events": events,
        "evaluation_count_after": evaluation_count_after,
        "final_initial": callbacks.final_initial,
        "objective": objective,
        "performance_clock_calls": performance_clock.calls,
        "raw_initial": callbacks.raw_initial,
        "telemetry": callbacks.telemetry,
        "wall_clock_calls": wall_clock.calls,
    }


def _transition_arguments(protected: dict[str, object]) -> tuple[object, ...]:
    objective = protected["objective"]
    if not isinstance(objective, _TraceObjective):
        raise RuntimeError("wrong protected Objective type")
    params = objective.evaluated_candidate
    losses = objective.observed_losses
    grads = objective.observed_grads
    aux = objective.observed_aux
    telemetry = protected["telemetry"]
    if not isinstance(aux, dict) or not isinstance(telemetry, dict):
        raise RuntimeError("malformed protected observation")
    learning_rates = jnp.geomspace(
        LEARNING_RATE_LOW, LEARNING_RATE_HIGH, POPULATION_SIZE
    )[:, None]
    return (
        params,
        losses,
        grads,
        aux,
        jnp.zeros_like(params),
        jnp.zeros_like(params),
        jnp.zeros((POPULATION_SIZE,), dtype=jnp.int32),
        jnp.full((POPULATION_SIZE,), jnp.inf),
        jnp.zeros((POPULATION_SIZE,), dtype=jnp.int32),
        learning_rates,
        jnp.asarray(jnp.inf, dtype=losses.dtype),
        params[0],
        jnp.zeros((POPULATION_SIZE,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        telemetry["eval_count_after_batch"][0],
        telemetry["time_seconds"][0],
        telemetry["evaluation_batch_seconds"][0],
        telemetry["budget_progress_fraction"][0],
        jnp.asarray(MINIMUM_IMPROVEMENT, dtype=losses.dtype),
        jnp.asarray(BETA1, dtype=losses.dtype),
        jnp.asarray(BETA2, dtype=losses.dtype),
        jnp.asarray(EPSILON, dtype=losses.dtype),
        jnp.asarray(GRADIENT_CLIP_NORM, dtype=losses.dtype),
        jnp.asarray(PATIENCE, dtype=jnp.int32),
    )


def _reference_projection(protected: dict[str, object]) -> dict[str, object]:
    objective = protected["objective"]
    telemetry = protected["telemetry"]
    adam_output = protected["adam_output"]
    if not isinstance(objective, _TraceObjective) or not isinstance(telemetry, dict):
        raise RuntimeError("malformed protected trace")
    aux = objective.observed_aux
    if not isinstance(aux, dict):
        raise RuntimeError("malformed protected aux")
    feasible = np.asarray(jax.device_get(aux["is_feasible"]), dtype=bool)
    losses = np.asarray(jax.device_get(objective.observed_losses))
    feasible_losses = np.where(feasible & np.isfinite(losses), losses, np.inf)
    feasible_index = int(np.argmin(feasible_losses))
    state = {
        "candidate": adam_output[0],
        "first_moment": adam_output[1],
        "global_feasible_loss": objective.observed_losses[feasible_index],
        "global_feasible_params": objective.evaluated_candidate[feasible_index],
        "member_best_loss": telemetry["observed_member_best_loss"],
        "member_generations": telemetry["next_generation"],
        "member_steps": adam_output[3],
        "second_moment": adam_output[2],
        "stalled_steps": telemetry["stalled_steps_after"],
    }
    observation = {
        "aux": aux,
        "candidate": objective.evaluated_candidate,
        "loss": objective.observed_losses,
        "total_gradient": objective.observed_grads,
    }
    return {
        "observation": _tree_projection(observation),
        "state": _tree_projection(state),
        "telemetry": _tree_projection(telemetry),
    }


def _transition_projection(output) -> dict[str, object]:
    observation, state, telemetry = output
    return {
        "observation": _tree_projection(observation),
        "state": _tree_projection(state),
        "telemetry": _tree_projection(telemetry),
    }


def _dependency_case() -> dict[str, object]:
    objective_path = inspect.getsourcefile(Objective)
    if objective_path is None:
        raise RuntimeError("installed Objective source is unavailable")
    locked = _locked_dfbench_projection()
    projection = {
        **locked,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "objective_source_sha256": _normalized_sha256(Path(objective_path)),
        "submission_source_sha256": _normalized_sha256(
            REPOSITORY_ROOT / "submission" / "submission.py"
        ),
    }
    passed = (
        projection
        == {
            "dfbench_version": DFBENCH_VERSION,
            "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
            "jax_version": JAX_VERSION,
            "jaxlib_version": JAXLIB_VERSION,
            "objective_source_sha256": OBJECTIVE_SOURCE_SHA256,
            "submission_source_sha256": SUBMISSION_SOURCE_SHA256,
        }
        and importlib.metadata.version("dfbench") == DFBENCH_VERSION
        and jax.default_backend() == "cpu"
        and bool(jax.config.x64_enabled)
    )
    return {
        **projection,
        "passed": bool(passed),
        "platform": jax.default_backend(),
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def _source_inventory_case() -> dict[str, object]:
    optimize_source = inspect.getsource(BatchedRestartAdam.optimize)
    adam_source = inspect.getsource(BatchedRestartAdam._adam_step)
    bind_source = inspect.getsource(Objective._bind_evaluation_functions)
    nanargmin_source = inspect.getsource(Objective._nanargmin_or_none)
    log_evals_source = inspect.getsource(Objective._log_evals)
    compact_optimize = re.sub(r"\s+", "", optimize_source)
    compact_bind = re.sub(r"\s+", "", bind_source)
    compact_nanargmin = re.sub(r"\s+", "", nanargmin_source)
    compact_log_evals = re.sub(r"\s+", "", log_evals_source)

    optimizer_ready_sites = optimize_source.count("jax.block_until_ready(")
    optimizer_clock_sites = optimize_source.count("time.perf_counter(")
    optimize_tree = ast.parse(textwrap.dedent(optimize_source))
    callback_names = [
        _call_name(node.func)
        for node in ast.walk(optimize_tree)
        if isinstance(node, ast.Call)
    ]
    optimizer_callback_sites = sum(
        callback_names.count(name)
        for name in (
            "raw_initial_population_callback",
            "initial_population_callback",
            "optimizer_telemetry_callback",
        )
    )
    optimizer_jit_sites = (
        optimize_source.count("jax.jit(") + adam_source.count("jax.jit(")
    )
    device_scalar_tokens = [
        "int(jnp.argmin(feasible_losses))",
        "float(feasible_losses[feasible_index])",
        "bool(jnp.any(restart_mask))",
    ]
    optimizer_device_scalar_sites = sum(
        token in compact_optimize for token in device_scalar_tokens
    )
    public_aux_transform_bound = all(
        token in compact_bind
        for token in (
            "self._value_and_grad_aux_func=jax.jit(",
            "jax.value_and_grad(self._func_aux,has_aux=True)",
            "self._vmap_value_and_grad_aux_func=jax.vmap(self._value_and_grad_aux_func)",
        )
    )
    objective_host_tokens = [
        "jnp.all(jnp.isnan(arr))",
        "int(jnp.nanargmin(arr))",
        "jnp.all(jnp.isnan(loss))",
        "int(jnp.nanargmin(loss))",
        "ifmin_loss<self._best_loss:",
    ]
    objective_logging_device_host_sites = sum(
        token in compact_nanargmin or token in compact_log_evals
        for token in objective_host_tokens
    )
    projection = {
        "objective_logging_device_host_sites": objective_logging_device_host_sites,
        "optimizer_callback_sites": optimizer_callback_sites,
        "optimizer_clock_sites": optimizer_clock_sites,
        "optimizer_device_scalar_sites": optimizer_device_scalar_sites,
        "optimizer_jit_sites": optimizer_jit_sites,
        "optimizer_ready_sites": optimizer_ready_sites,
        "public_aux_transform_bound": public_aux_transform_bound,
    }
    expected = {
        **CASE_CONTRACT["source_boundary_inventory"],
        "public_aux_transform_bound": True,
    }
    return {
        **projection,
        "passed": projection == expected,
        "source_projection_sha256": _json_sha256(projection),
    }


def _runtime_case(protected: dict[str, object]) -> dict[str, object]:
    objective = protected["objective"]
    events = protected["events"]
    telemetry = protected["telemetry"]
    if (
        not isinstance(objective, _TraceObjective)
        or not isinstance(events, list)
        or not isinstance(telemetry, dict)
    ):
        raise RuntimeError("malformed protected trace")

    def count(name: str) -> int:
        return events.count(name)

    ordering = [
        events.index("raw_initial_callback")
        < events.index("final_initial_callback")
        < events.index("start_logging"),
        events.index("performance_clock_start")
        < events.index("public_evaluation_enter")
        < events.index("public_evaluation_return")
        < events.index("explicit_ready_enter")
        < events.index("explicit_ready_return")
        < events.index("performance_clock_stop")
        < events.index("telemetry_callback"),
    ]
    restart_events = int(
        np.asarray(jax.device_get(telemetry["restart_triggered"]), dtype=np.int32).sum()
    )
    batch_microseconds = int(
        round(
            float(
                np.asarray(
                    jax.device_get(telemetry["evaluation_batch_seconds"])
                )[0]
            )
            * 1_000_000
        )
    )
    budget_counts = {
        "budget_exceeded": count("budget_exceeded_read"),
        "budget_progress": count("budget_progress_read"),
        "budget_progress_eval_count": count("budget_progress_eval_count_read"),
        "eval_count": count("eval_count_read"),
        "evals_left": count("evals_left_read"),
        "objective_time_elapsed": count("objective_log_time_elapsed_read"),
        "optimizer_time_elapsed": count("optimizer_time_elapsed_read"),
        "time_left": count("time_left_read"),
    }
    passed = all(
        (
            objective.public_batch_shape == [POPULATION_SIZE, PARAMETER_DIMENSION],
            batch_microseconds == BATCH_MICROSECONDS,
            protected["evaluation_count_after"] == MAX_EVALS,
            objective.public_call_count == 1,
            objective.scalar_call_count == 0,
            objective.warmup_call_count == 0,
            objective.rng_call_count == 1,
            count("raw_initial_callback") == 1,
            count("final_initial_callback") == 1,
            count("start_logging") == 1,
            count("explicit_ready_enter") == 1,
            count("explicit_ready_return") == 1,
            protected["performance_clock_calls"] == 2,
            protected["wall_clock_calls"] == 3,
            count("telemetry_callback") == 1,
            restart_events == 0,
            budget_counts
            == {
                "budget_exceeded": 2,
                "budget_progress": 1,
                "budget_progress_eval_count": 1,
                "eval_count": 1,
                "evals_left": 2,
                "objective_time_elapsed": 1,
                "optimizer_time_elapsed": 1,
                "time_left": 1,
            },
            all(ordering),
        )
    )
    return {
        "batch_microseconds": batch_microseconds,
        "batch_shape": objective.public_batch_shape,
        "budget_exceeded_reads": budget_counts["budget_exceeded"],
        "budget_progress_reads": budget_counts["budget_progress"],
        "budget_progress_eval_count_reads": budget_counts[
            "budget_progress_eval_count"
        ],
        "eval_count_reads": budget_counts["eval_count"],
        "evals_left_reads": budget_counts["evals_left"],
        "evaluation_count_after": int(protected["evaluation_count_after"]),
        "evaluation_count_before": 0,
        "explicit_barriers": count("explicit_ready_return"),
        "final_initial_callbacks": count("final_initial_callback"),
        "objective_time_elapsed_reads": budget_counts["objective_time_elapsed"],
        "optimizer_time_elapsed_reads": budget_counts["optimizer_time_elapsed"],
        "order_sha256": _json_sha256(events),
        "passed": bool(passed),
        "performance_clock_reads": int(protected["performance_clock_calls"]),
        "public_evaluation_calls": objective.public_call_count,
        "raw_initial_callbacks": count("raw_initial_callback"),
        "restart_events": restart_events,
        "rng_draws": objective.rng_call_count,
        "scalar_calls": objective.scalar_call_count,
        "start_logging_calls": count("start_logging"),
        "telemetry_callbacks": count("telemetry_callback"),
        "time_left_reads": budget_counts["time_left"],
        "wall_clock_reads": int(protected["wall_clock_calls"]),
        "warmup_calls": objective.warmup_call_count,
    }


def _pure_and_lowering_cases(
    protected: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    arguments = _transition_arguments(protected)
    reference = _reference_projection(protected)
    eager_output = pure_jax_transition(*arguments)
    eager_projection = _transition_projection(eager_output)

    closed_jaxpr = jax.make_jaxpr(pure_jax_transition)(*arguments)
    clean_callback_primitives = _callback_primitives(closed_jaxpr)
    pure_source_host_calls = _pure_source_host_calls()
    typed_matches = _typed_leaf_count(reference)
    restart_events = int(
        np.asarray(
            jax.device_get(eager_output[2]["restart_triggered"]), dtype=np.int32
        ).sum()
    )
    pure_passed = all(
        (
            reference == eager_projection,
            typed_matches == EXPECTED_TYPED_MATCHES,
            sorted(eager_output[2]) == TELEMETRY_LEAVES,
            pure_source_host_calls == [],
            clean_callback_primitives == [],
            restart_events == 0,
        )
    )
    pure_case = {
        "adam_state_leaves_checked": 4,
        "aux_leaves_checked": len(AUX_LEAF_PATHS),
        "callback_primitives": clean_callback_primitives,
        "exact_typed_matches": typed_matches,
        "passed": bool(pure_passed),
        "projection_sha256": _json_sha256(eager_projection),
        "pure_source_host_calls": pure_source_host_calls,
        "restart_events": restart_events,
        "telemetry_leaves_checked": len(TELEMETRY_LEAVES),
    }

    jitted = jax.jit(pure_jax_transition)
    jit_projection = _transition_projection(jitted(*arguments))
    lowered = jitted.lower(*arguments)
    stablehlo = lowered.as_text(dialect="stablehlo")
    compiled = lowered.compile()
    compiled_projection = _transition_projection(compiled(*arguments))
    jaxpr_text = str(closed_jaxpr)
    lowering_case = {
        "callback_primitives": clean_callback_primitives,
        "compiled_calls": 1,
        "eager_compiled_exact": eager_projection == compiled_projection,
        "eager_jit_exact": eager_projection == jit_projection,
        "jaxpr_sha256": hashlib.sha256(jaxpr_text.encode("utf-8")).hexdigest(),
        "passed": bool(
            eager_projection == jit_projection == compiled_projection
            and clean_callback_primitives == []
        ),
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode("utf-8")).hexdigest(),
    }

    negative_base = {
        "budget": {"eval_count": MAX_EVALS},
        "state": copy.deepcopy(reference["state"]),
        "telemetry": copy.deepcopy(reference["telemetry"]),
        "timing": {"batch_micros": BATCH_MICROSECONDS},
    }
    controls = {
        "budget.eval_count": MAX_EVALS + 1,
        "state.candidate.sha256": "0" * 64,
        "telemetry.feasible.sha256": "1" * 64,
        "timing.batch_micros": BATCH_MICROSECONDS + 1,
    }
    exact_single_path_controls = 0
    for expected_path, replacement in controls.items():
        changed = copy.deepcopy(negative_base)
        cursor = changed
        parts = expected_path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = replacement
        if _diff_paths(negative_base, changed) == [expected_path]:
            exact_single_path_controls += 1

    raw_words = np.asarray([0x3F800000, 0x40000000, 0x40400000, 0x40800000], dtype=np.uint32)
    raw_projection = _array_projection(raw_words)
    dtype_projection = _array_projection(raw_words.view(np.float32))
    shape_projection = _array_projection(raw_words.reshape(2, 2))
    typed_identity_changes = int(raw_projection != dtype_projection) + int(
        raw_projection != shape_projection
    )

    sentinel_input = jnp.ones((2,), dtype=jnp.float32)

    def callback_sentinel(value):
        shape = jax.ShapeDtypeStruct(value.shape, value.dtype)
        return jax.pure_callback(lambda item: item, shape, value)

    sentinel_jaxpr = jax.make_jaxpr(callback_sentinel)(sentinel_input)
    sentinel_callbacks = _callback_primitives(sentinel_jaxpr)
    callback_sentinel_detected = "pure_callback" in sentinel_callbacks
    negative_case = {
        "callback_sentinel_detected": callback_sentinel_detected,
        "clean_callback_primitives": len(clean_callback_primitives),
        "controls_checked": len(controls) + 3,
        "exact_single_path_controls": exact_single_path_controls,
        "passed": bool(
            exact_single_path_controls == len(controls)
            and typed_identity_changes == 2
            and callback_sentinel_detected
            and clean_callback_primitives == []
        ),
        "typed_identity_changes": typed_identity_changes,
    }
    return pure_case, lowering_case, negative_case


def _proof_projection() -> dict[str, object]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices or jax.default_backend() != "cpu":
        raise RuntimeError("the normal-path JAX boundary study requires CPU")
    protected = _run_protected_batch()
    pure_case, lowering_case, negative_case = _pure_and_lowering_cases(protected)
    cases = {
        "boundary_negative_controls": negative_case,
        "dependency_source_identity": _dependency_case(),
        "explicit_jit_lowering": lowering_case,
        "normal_path_boundary_trace": _runtime_case(protected),
        "pure_jax_transition_equivalence": pure_case,
        "source_boundary_inventory": _source_inventory_case(),
    }
    return {
        "cases": cases,
        "projection_sha256": _json_sha256(cases),
    }


def isolated_worker_trace() -> dict[str, object]:
    """Return the deterministic non-process projection for isolation checks."""
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
            "experiments.local_lab.normal_path_jax_boundary_worker",
            "--mode",
            "normal-path-jax-boundary-trace",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return json.loads(completed.stdout)


def run_study(*, include_process_isolation: bool = True) -> dict[str, object]:
    """Run the complete frozen contract and return sanitized aggregate evidence."""
    parent = _proof_projection()
    cases = copy.deepcopy(parent["cases"])
    if include_process_isolation:
        workers = [_isolated_trace(), _isolated_trace()]
        process_passed = workers[0] == parent and workers[1] == parent
        process_sha256 = parent["projection_sha256"]
    else:
        process_passed = None
        process_sha256 = "0" * 64
    cases["process_isolation"] = {
        "passed": process_passed,
        "trace_sha256": process_sha256,
    }

    terminal = include_process_isolation
    passed = terminal and all(case["passed"] for case in cases.values())
    status = "passed" if passed else ("failed" if terminal else "incomplete")
    action = (
        "synthetic_normal_path_jax_boundary_equivalent"
        if passed
        else (
            "park_normal_path_jax_boundary_research"
            if terminal
            else "no_decision_incomplete_study"
        )
    )
    cpu_device = jax.devices("cpu")[0]
    return {
        "action": action,
        "cases": cases,
        "environment": {
            "device_kind": cpu_device.device_kind,
            "jax_version": jax.__version__,
            "platform": cpu_device.platform,
            "python": platform.python_version(),
        },
        "fixture": {**FIXTURE_IDENTITY, "case_contract": CASE_CONTRACT},
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "study_id": STUDY_ID,
    }
