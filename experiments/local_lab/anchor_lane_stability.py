"""Deterministic CPU fixtures for the frozen anchor-lane stability study."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from submission.submission import BatchedRestartAdam


STUDY_ID = "anchor-lane-stability-v1"
SCHEMA_VERSION = 1
SEED = 20260826
POPULATION_SIZE = 8
N_PARAMS = 5
REPOSITORY_ROOT = Path(__file__).parents[2]
CASE_CONTRACT = {
    "diagnostics_disabled_control": {
        "initial_variants": ["suffix-a", "suffix-a"],
        "max_evals": 18,
        "mode": "quadratic",
        "patience": 100,
    },
    "exact_twin": {
        "initial_variants": [None, None],
        "max_evals": 24,
        "mode": "quadratic",
        "patience": 100,
    },
    "exceptional_arithmetic_partial_tail": {
        "initial_variants": ["exceptional-a", "exceptional-b"],
        "max_evals": 18,
        "mode": "exceptional",
        "patience": 100,
    },
    "forced_shared_state_boundary": {
        "initial_variants": ["boundary-a", "boundary-b"],
        "max_evals": 32,
        "mode": "shared-boundary",
        "patience": 2,
    },
    "process_isolation": {
        "source_case": "exact_twin",
        "workers": 2,
    },
    "suffix_invariance": {
        "initial_variants": ["suffix-a", "suffix-b"],
        "max_evals": 24,
        "mode": "quadratic",
        "patience": 100,
    },
}
TIMING_TELEMETRY_FIELDS = {
    "time_seconds",
    "evaluation_batch_seconds",
}


def _array_bytes(value: object) -> bytes:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + array.tobytes()


def _array_sha256(value: object) -> str:
    return hashlib.sha256(_array_bytes(value)).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@jax.custom_jvp
def _finite_value_with_exceptional_gradient(value):
    """Return a finite square while exposing a NaN derivative for positives."""
    return jnp.square(value)


@_finite_value_with_exceptional_gradient.defjvp
def _finite_value_with_exceptional_gradient_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    derivative = jnp.where(value > 0.25, jnp.nan, 2.0 * value)
    return jnp.square(value), derivative * tangent


class _TelemetryTrace:
    """Keep stable scalar telemetry in memory without retaining vectors."""

    def __init__(self) -> None:
        self.events: list[dict[str, np.ndarray]] = []

    def __call__(self, event: dict[str, object]) -> None:
        hosted = {
            name: np.asarray(jax.device_get(value))
            for name, value in event.items()
        }
        lengths = {int(value.shape[0]) for value in hosted.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise RuntimeError("malformed local-lab telemetry event")
        self.events.append(hosted)

    def summary(self) -> dict[str, object]:
        digest = hashlib.sha256()
        restart_events = []
        global_improvement_events = []
        partial_events = []
        nonfinite_gradient_values = 0
        partial_state_violations = 0
        sanitizer_violations = 0

        for event in self.events:
            for name in sorted(set(event) - TIMING_TELEMETRY_FIELDS):
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(_array_bytes(event[name]))

            rows = len(event["member_index"])
            partial_members = []
            for row in range(rows):
                batch = int(event["batch_index"][row])
                member = int(event["member_index"][row])
                nonfinite_gradient_values += int(
                    event["gradient_nonfinite_count"][row]
                )
                gradient_norm = float(event["gradient_norm"][row])
                clip_scale = float(event["gradient_clip_scale"][row])
                if not (
                    np.isfinite(gradient_norm)
                    and gradient_norm >= 0.0
                    and np.isfinite(clip_scale)
                    and 0.0 <= clip_scale <= 1.0
                ):
                    sanitizer_violations += 1
                if bool(event["restart_triggered"][row]):
                    restart_events.append(
                        {
                            "batch": batch,
                            "kind": int(event["restart_kind"][row]),
                            "member": member,
                            "round": int(event["restart_round"][row]),
                        }
                    )
                if bool(event["global_feasible_improvement"][row]):
                    global_improvement_events.append(
                        {
                            "batch": batch,
                            "feasible": bool(event["feasible"][row]),
                            "finite": bool(event["finite_loss"][row]),
                            "member": member,
                        }
                    )
                if not bool(event["update_applied"][row]):
                    partial_members.append(member)
                    if not (
                        int(event["adam_age_before"][row])
                        == int(event["adam_age_after"][row])
                        and int(event["stalled_steps_before"][row])
                        == int(event["stalled_steps_after"][row])
                        and int(event["evaluated_generation"][row])
                        == int(event["next_generation"][row])
                        and not bool(event["restart_triggered"][row])
                        and int(event["restart_kind"][row]) == -1
                        and int(event["restart_round"][row]) == -1
                        and np.isnan(float(event["restart_noise_scale"][row]))
                    ):
                        partial_state_violations += 1
            if partial_members:
                partial_events.append(
                    {
                        "batch": int(event["batch_index"][0]),
                        "members": partial_members,
                    }
                )

        return {
            "global_improvement_events": global_improvement_events,
            "nonfinite_gradient_values": nonfinite_gradient_values,
            "partial_events": partial_events,
            "partial_state_violations": partial_state_violations,
            "restart_events": restart_events,
            "sanitizer_violations": sanitizer_violations,
            "stable_sha256": digest.hexdigest(),
        }


class _AnalyticObjective:
    """Small deterministic public-Objective stand-in with hashed traces."""

    def __init__(self, *, mode: str, max_evals: int) -> None:
        if mode not in {"quadratic", "shared-boundary", "exceptional"}:
            raise ValueError(f"unknown analytic objective mode: {mode}")
        self.mode = mode
        self.n_params = N_PARAMS
        self.max_evals = int(max_evals)
        self.eval_count = 0
        self.algorithm_str = ""
        self.unbounded = False
        self._key = jax.random.PRNGKey(0)
        self._started = False
        self.batches: list[dict[str, object]] = []
        self.optimization_pairs = [
            [f"synthetic_component_{index}", "tuning"]
            for index in range(self.n_params)
        ]

    def set_space_mode(self, unbounded: bool) -> None:
        self.unbounded = bool(unbounded)

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.PRNGKey(seed)

    def random_params_unbounded(self, n_samples: int = 1):
        self._key, sample_key = jax.random.split(self._key)
        return jax.random.uniform(
            sample_key,
            shape=(n_samples, self.n_params),
            minval=-1.5,
            maxval=1.5,
            dtype=jnp.float32,
        )

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
        if self._started:
            raise RuntimeError("analytic objective logging started twice")
        self._started = True

    def _value(self, params):
        if self.mode == "quadratic":
            target = jnp.linspace(-0.20, 0.30, self.n_params)
            loss = jnp.sum(jnp.square(params - target))
            feasible = jnp.asarray(True)
        elif self.mode == "shared-boundary":
            loss = jax.lax.stop_gradient(params[0])
            feasible = jnp.asarray(True)
        else:
            loss = jnp.sum(_finite_value_with_exceptional_gradient(params))
            feasible = params[0] <= 0.25
        return loss, {"is_feasible": feasible}

    def _record(self, params, losses, grads, feasible) -> None:
        host_params = np.asarray(jax.device_get(params))
        host_losses = np.asarray(jax.device_get(losses))
        host_grads = np.asarray(jax.device_get(grads))
        host_feasible = np.asarray(jax.device_get(feasible), dtype=bool)
        self.batches.append(
            {
                "feasible": host_feasible.tolist(),
                "finite_loss": np.isfinite(host_losses).tolist(),
                "gradient_nonfinite_count": np.sum(
                    ~np.isfinite(host_grads), axis=1, dtype=np.int64
                ).tolist(),
                "loss_hashes": [
                    _array_sha256(host_losses[index])
                    for index in range(len(host_losses))
                ],
                "parameter_hashes": [
                    _array_sha256(host_params[index])
                    for index in range(len(host_params))
                ],
            }
        )

    def vmap_value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        (losses, aux), grads = jax.vmap(
            jax.value_and_grad(self._value, has_aux=True)
        )(params)
        self._record(params, losses, grads, aux["is_feasible"])
        self.eval_count += int(params.shape[0])
        return losses, grads, aux

    def value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        (loss, aux), grad = jax.value_and_grad(self._value, has_aux=True)(params)
        self._record(
            params[None, :],
            loss[None],
            grad[None, :],
            aux["is_feasible"][None],
        )
        self.eval_count += 1
        return loss, grad, aux


def _initial_population(variant: str) -> np.ndarray:
    population = np.zeros((POPULATION_SIZE, N_PARAMS), dtype=np.float32)
    if variant == "suffix-a":
        population[1:] = np.asarray(
            [
                [-0.75, -0.50, -0.25, 0.25, 0.50],
                [0.60, 0.45, 0.30, 0.15, -0.10],
                [1.20, 0.90, 0.60, 0.30, 0.10],
                [-1.10, -0.85, -0.55, -0.20, 0.15],
                [0.20, 0.40, 0.60, 0.80, 1.00],
                [-0.30, 0.10, 0.50, 0.90, 1.30],
                [1.40, 1.00, 0.50, 0.00, -0.50],
            ],
            dtype=np.float32,
        )
    elif variant == "suffix-b":
        population[1:] = np.asarray(
            [
                [1.35, -1.10, 0.80, -0.65, 0.40],
                [-1.25, 1.00, -0.75, 0.55, -0.35],
                [0.35, -0.30, 0.25, -0.20, 0.15],
                [1.10, 0.70, 0.30, -0.10, -0.50],
                [-0.90, -0.45, 0.00, 0.45, 0.90],
                [0.55, -0.15, -0.85, 0.35, 1.05],
                [-1.40, -0.95, -0.50, -0.05, 0.40],
            ],
            dtype=np.float32,
        )
    elif variant == "boundary-a":
        population[1:] = np.asarray(
            [
                [-4.0, -0.8, -0.6, -0.4, -0.2],
                [1.0, 0.8, 0.6, 0.4, 0.2],
                [2.0, 1.6, 1.2, 0.8, 0.4],
                [2.5, 2.0, 1.5, 1.0, 0.5],
                [3.0, 2.4, 1.8, 1.2, 0.6],
                [3.5, 2.8, 2.1, 1.4, 0.7],
                [4.0, 3.2, 2.4, 1.6, 0.8],
            ],
            dtype=np.float32,
        )
    elif variant == "boundary-b":
        population[1:] = np.asarray(
            [
                [-2.0, 0.7, 0.5, 0.3, 0.1],
                [-5.0, -0.8, -0.6, -0.4, -0.2],
                [2.0, -1.6, -1.2, -0.8, -0.4],
                [2.5, -2.0, -1.5, -1.0, -0.5],
                [3.0, -2.4, -1.8, -1.2, -0.6],
                [3.5, -2.8, -2.1, -1.4, -0.7],
                [4.0, -3.2, -2.4, -1.6, -0.8],
            ],
            dtype=np.float32,
        )
    elif variant == "exceptional-a":
        population[1:] = np.asarray(
            [
                [0.75, 0.50, 0.40, 0.30, 0.20],
                [-0.80, -0.60, -0.40, -0.20, -0.10],
                [-0.50, 0.10, -0.20, 0.15, -0.10],
                [0.55, 0.35, 0.15, -0.05, -0.25],
                [-0.65, -0.45, -0.25, -0.05, 0.15],
                [0.95, -0.30, 0.45, -0.20, 0.35],
                [-1.00, -0.75, -0.50, -0.25, 0.00],
            ],
            dtype=np.float32,
        )
    elif variant == "exceptional-b":
        population[1:] = np.asarray(
            [
                [1.25, 0.90, 0.70, 0.50, 0.30],
                [-1.10, -0.70, -0.50, -0.30, -0.15],
                [-0.35, 0.20, -0.15, 0.10, -0.05],
                [0.85, 0.65, 0.45, 0.25, 0.05],
                [-0.90, -0.55, -0.35, -0.15, 0.05],
                [1.40, -0.50, 0.80, -0.35, 0.60],
                [-1.30, -0.95, -0.65, -0.35, -0.10],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unknown initial-population variant: {variant}")
    return population


def _run_trace(
    *,
    mode: str,
    max_evals: int,
    patience: int,
    initial_population: np.ndarray | None,
    telemetry_enabled: bool,
) -> dict[str, object]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the local mechanics study requires a CPU backend")
    objective = _AnalyticObjective(mode=mode, max_evals=max_evals)
    telemetry = _TelemetryTrace() if telemetry_enabled else None
    captured_raw_hashes = []
    captured_initial_hashes = []

    def capture_hashes(destination: list[list[str]], value: object) -> None:
        population = np.asarray(jax.device_get(value))
        destination.append([_array_sha256(member) for member in population])

    with jax.default_device(cpu_devices[0]), contextlib.redirect_stdout(io.StringIO()):
        BatchedRestartAdam().optimize(
            objective,
            init_params=initial_population,
            random_seed=SEED,
            population_size=POPULATION_SIZE,
            learning_rate_low=0.05,
            learning_rate_high=0.05,
            patience=patience,
            safety_seconds=0.0,
            raw_initial_population_callback=lambda value: capture_hashes(
                captured_raw_hashes, value
            ),
            initial_population_callback=lambda value: capture_hashes(
                captured_initial_hashes, value
            ),
            optimizer_telemetry_callback=telemetry,
        )

    if len(captured_raw_hashes) != 1 or len(captured_initial_hashes) != 1:
        raise RuntimeError("initial populations were not captured exactly once")
    trace = {
        "batch_sizes": [len(batch["parameter_hashes"]) for batch in objective.batches],
        "batches": objective.batches,
        "eval_count": objective.eval_count,
        "initial_population_hashes": captured_initial_hashes[0],
        "raw_random_hashes": captured_raw_hashes[0],
    }
    if telemetry is not None:
        trace["telemetry"] = telemetry.summary()
    return trace


def _objective_projection(trace: dict[str, object]) -> dict[str, object]:
    return {
        "batch_sizes": trace["batch_sizes"],
        "batches": trace["batches"],
        "eval_count": trace["eval_count"],
        "initial_population_hashes": trace["initial_population_hashes"],
        "raw_random_hashes": trace["raw_random_hashes"],
    }


def _run_declared_trace(
    case: str,
    *,
    initial_population: np.ndarray | None,
    telemetry_enabled: bool,
) -> dict[str, object]:
    contract = CASE_CONTRACT[case]
    return _run_trace(
        mode=str(contract["mode"]),
        max_evals=int(contract["max_evals"]),
        patience=int(contract["patience"]),
        initial_population=initial_population,
        telemetry_enabled=telemetry_enabled,
    )


def _lane_series(
    trace: dict[str, object], field: str, member: int = 0
) -> list[object]:
    values = []
    for batch in trace["batches"]:
        rows = batch[field]
        if len(rows) > member:
            values.append(rows[member])
    return values


def _first_difference(left: list[object], right: list[object]) -> int | None:
    for index, (left_value, right_value) in enumerate(
        zip(left, right, strict=True)
    ):
        if left_value != right_value:
            return index
    return None


def isolated_worker_trace() -> dict[str, object]:
    """Return the sanitized exact-twin trace used by child processes."""
    return _run_declared_trace(
        "exact_twin",
        initial_population=None,
        telemetry_enabled=True,
    )


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
            "experiments.local_lab.worker",
            "--mode",
            "exact-trace",
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
    """Execute every predeclared fixture and return a sanitized result."""
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the local mechanics study requires a CPU backend")

    exact_left = isolated_worker_trace()
    exact_right = isolated_worker_trace()
    exact_passed = exact_left == exact_right

    suffix_left = _run_declared_trace(
        "suffix_invariance",
        initial_population=_initial_population("suffix-a"),
        telemetry_enabled=True,
    )
    suffix_right = _run_declared_trace(
        "suffix_invariance",
        initial_population=_initial_population("suffix-b"),
        telemetry_enabled=True,
    )
    suffix_anchor_params_equal = _lane_series(
        suffix_left, "parameter_hashes"
    ) == _lane_series(suffix_right, "parameter_hashes")
    suffix_anchor_losses_equal = _lane_series(
        suffix_left, "loss_hashes"
    ) == _lane_series(suffix_right, "loss_hashes")
    suffixes_differ = (
        suffix_left["initial_population_hashes"][1:]
        != suffix_right["initial_population_hashes"][1:]
    )
    raw_random_hashes_equal = (
        suffix_left["raw_random_hashes"] == suffix_right["raw_random_hashes"]
    )
    suffix_passed = (
        suffix_anchor_params_equal
        and suffix_anchor_losses_equal
        and suffixes_differ
        and raw_random_hashes_equal
    )

    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolated_passed = isolated_left == isolated_right
        isolated_digest = _json_sha256(isolated_left)
    else:
        isolated_passed = None
        isolated_digest = "not-run-in-focused-test"

    boundary_left = _run_declared_trace(
        "forced_shared_state_boundary",
        initial_population=_initial_population("boundary-a"),
        telemetry_enabled=True,
    )
    boundary_right = _run_declared_trace(
        "forced_shared_state_boundary",
        initial_population=_initial_population("boundary-b"),
        telemetry_enabled=True,
    )
    boundary_left_params = _lane_series(boundary_left, "parameter_hashes")
    boundary_right_params = _lane_series(boundary_right, "parameter_hashes")
    boundary_left_losses = _lane_series(boundary_left, "loss_hashes")
    boundary_right_losses = _lane_series(boundary_right, "loss_hashes")
    first_parameter_difference = _first_difference(
        boundary_left_params, boundary_right_params
    )
    first_loss_difference = _first_difference(
        boundary_left_losses, boundary_right_losses
    )
    expected_restarts = [
        {
            "batch": 2,
            "kind": 1 if member % 2 == 0 else 0,
            "member": member,
            "round": 0,
        }
        for member in range(POPULATION_SIZE)
    ]
    expected_left_incumbent = {
        "batch": 0,
        "feasible": True,
        "finite": True,
        "member": 1,
    }
    expected_right_incumbent = {
        "batch": 0,
        "feasible": True,
        "finite": True,
        "member": 2,
    }
    boundary_passed = (
        boundary_left_params[:3] == boundary_right_params[:3]
        and boundary_left_losses[:3] == boundary_right_losses[:3]
        and first_parameter_difference == 3
        and first_loss_difference == 3
        and boundary_left["telemetry"]["restart_events"] == expected_restarts
        and boundary_right["telemetry"]["restart_events"] == expected_restarts
        and expected_left_incumbent
        in boundary_left["telemetry"]["global_improvement_events"]
        and expected_right_incumbent
        in boundary_right["telemetry"]["global_improvement_events"]
    )

    exceptional_left = _run_declared_trace(
        "exceptional_arithmetic_partial_tail",
        initial_population=_initial_population("exceptional-a"),
        telemetry_enabled=True,
    )
    exceptional_right = _run_declared_trace(
        "exceptional_arithmetic_partial_tail",
        initial_population=_initial_population("exceptional-b"),
        telemetry_enabled=True,
    )
    exceptional_global_events = (
        exceptional_left["telemetry"]["global_improvement_events"]
        + exceptional_right["telemetry"]["global_improvement_events"]
    )
    exceptional_finite_infeasible = {
        "variant_a": sum(
            int(finite and not feasible)
            for batch in exceptional_left["batches"]
            for finite, feasible in zip(
                batch["finite_loss"], batch["feasible"], strict=True
            )
        ),
        "variant_b": sum(
            int(finite and not feasible)
            for batch in exceptional_right["batches"]
            for finite, feasible in zip(
                batch["finite_loss"], batch["feasible"], strict=True
            )
        ),
    }
    exceptional_passed = (
        exceptional_left["batch_sizes"] == [8, 8, 2]
        and exceptional_right["batch_sizes"] == [8, 8, 2]
        and exceptional_left["eval_count"] == 18
        and exceptional_right["eval_count"] == 18
        and _lane_series(exceptional_left, "parameter_hashes")
        == _lane_series(exceptional_right, "parameter_hashes")
        and _lane_series(exceptional_left, "loss_hashes")
        == _lane_series(exceptional_right, "loss_hashes")
        and exceptional_left["telemetry"]["nonfinite_gradient_values"] > 0
        and exceptional_right["telemetry"]["nonfinite_gradient_values"] > 0
        and all(value > 0 for value in exceptional_finite_infeasible.values())
        and exceptional_left["telemetry"]["sanitizer_violations"] == 0
        and exceptional_right["telemetry"]["sanitizer_violations"] == 0
        and exceptional_left["telemetry"]["partial_state_violations"] == 0
        and exceptional_right["telemetry"]["partial_state_violations"] == 0
        and exceptional_left["telemetry"]["restart_events"] == []
        and exceptional_right["telemetry"]["restart_events"] == []
        and exceptional_left["telemetry"]["partial_events"]
        == [{"batch": 2, "members": [0, 1]}]
        and exceptional_right["telemetry"]["partial_events"]
        == [{"batch": 2, "members": [0, 1]}]
        and all(
            event["finite"] and event["feasible"]
            for event in exceptional_global_events
        )
    )

    diagnostics_off = _run_declared_trace(
        "diagnostics_disabled_control",
        initial_population=_initial_population("suffix-a"),
        telemetry_enabled=False,
    )
    diagnostics_on = _run_declared_trace(
        "diagnostics_disabled_control",
        initial_population=_initial_population("suffix-a"),
        telemetry_enabled=True,
    )
    diagnostics_passed = _objective_projection(
        diagnostics_off
    ) == _objective_projection(diagnostics_on)

    cases = {
        "diagnostics_disabled_control": {
            "passed": diagnostics_passed,
            "trace_sha256": _json_sha256(_objective_projection(diagnostics_off)),
        },
        "exact_twin": {
            "batches": exact_left["batch_sizes"],
            "passed": exact_passed,
            "trace_sha256": _json_sha256(exact_left),
        },
        "exceptional_arithmetic_partial_tail": {
            "batch_sizes": exceptional_left["batch_sizes"],
            "global_improvement_events": exceptional_global_events,
            "finite_infeasible_observations": exceptional_finite_infeasible,
            "nonfinite_gradient_values": {
                "variant_a": exceptional_left["telemetry"][
                    "nonfinite_gradient_values"
                ],
                "variant_b": exceptional_right["telemetry"][
                    "nonfinite_gradient_values"
                ],
            },
            "passed": exceptional_passed,
            "partial_events": exceptional_left["telemetry"]["partial_events"],
            "partial_state_violations": {
                "variant_a": exceptional_left["telemetry"][
                    "partial_state_violations"
                ],
                "variant_b": exceptional_right["telemetry"][
                    "partial_state_violations"
                ],
            },
            "sanitizer_violations": {
                "variant_a": exceptional_left["telemetry"][
                    "sanitizer_violations"
                ],
                "variant_b": exceptional_right["telemetry"][
                    "sanitizer_violations"
                ],
            },
        },
        "forced_shared_state_boundary": {
            "first_lane_zero_loss_difference_batch": first_loss_difference,
            "first_lane_zero_parameter_difference_batch": first_parameter_difference,
            "incumbent_events": {
                "variant_a": expected_left_incumbent,
                "variant_b": expected_right_incumbent,
            },
            "passed": boundary_passed,
            "restart_events": expected_restarts,
        },
        "process_isolation": {
            "passed": isolated_passed,
            "trace_sha256": isolated_digest,
        },
        "suffix_invariance": {
            "anchor_loss_hashes_equal": suffix_anchor_losses_equal,
            "anchor_parameter_hashes_equal": suffix_anchor_params_equal,
            "passed": suffix_passed,
            "raw_random_hashes_equal": raw_random_hashes_equal,
            "suffix_initial_hashes_differ": suffixes_differ,
        },
    }
    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    return {
        "action": (
            "anchor_lane_mechanics_confirmed"
            if passed
            else (
                "park_initializer_and_restart_research"
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
            "n_params": N_PARAMS,
            "population_size": POPULATION_SIZE,
            "seed": SEED,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
