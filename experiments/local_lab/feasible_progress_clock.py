"""Deterministic CPU fixtures for the frozen feasible-progress clock study."""

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


STUDY_ID = "feasible-progress-clock-v1"
SCHEMA_VERSION = 1
SEED = 20260826
POPULATION_SIZE = 8
N_PARAMS = 3
MAX_EVALS = 32
PATIENCE = 2
LEARNING_RATE = 0.05
MINIMUM_IMPROVEMENT = 1e-7
CLAIM_BOUNDARY = "mechanics_only_no_candidate_or_performance_claim"
CASE_CONTRACT = {
    "diagnostics_disabled_control": {
        "max_evals": MAX_EVALS,
        "mode": "mixed",
        "patience": PATIENCE,
        "telemetry_variants": [False, True],
    },
    "finite_infeasible_descent": {
        "feasible_batches": [],
        "max_evals": MAX_EVALS,
        "mode": "descent",
        "patience": PATIENCE,
    },
    "finite_infeasible_improve_then_plateau": {
        "feasible_batches": [],
        "improvement_batches": [0, 1],
        "max_evals": 40,
        "mode": "improve-then-plateau",
        "patience": PATIENCE,
    },
    "finite_infeasible_plateau_control": {
        "feasible_batches": [],
        "max_evals": MAX_EVALS,
        "mode": "plateau",
        "patience": PATIENCE,
    },
    "late_feasibility_crossing": {
        "feasible_member": 0,
        "first_feasible_batch": 3,
        "max_evals": MAX_EVALS,
        "mode": "late-crossing",
        "patience": PATIENCE,
    },
    "mixed_member_clock": {
        "descending_members": [0, 2, 4, 6],
        "max_evals": MAX_EVALS,
        "mode": "mixed",
        "patience": PATIENCE,
        "plateau_members": [1, 3, 5, 7],
    },
    "process_isolation": {
        "source_case": "mixed_member_clock",
        "workers": 2,
    },
}
REPOSITORY_ROOT = Path(__file__).parents[2]
TIMING_TELEMETRY_FIELDS = {
    "evaluation_batch_seconds",
    "time_seconds",
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


class _ClockTelemetry:
    """Retain timing-free scalar clock observations, never parameter vectors."""

    def __init__(self) -> None:
        self.events: list[dict[str, np.ndarray]] = []

    def __call__(self, event: dict[str, object]) -> None:
        hosted = {
            name: np.asarray(jax.device_get(value))
            for name, value in event.items()
        }
        row_counts = {int(value.shape[0]) for value in hosted.values()}
        if row_counts != {POPULATION_SIZE}:
            raise RuntimeError("malformed feasible-progress telemetry event")
        self.events.append(hosted)

    def summary(self) -> dict[str, object]:
        stable_digest = hashlib.sha256()
        improved_by_batch = []
        stalled_before_by_batch = []
        stalled_after_by_batch = []
        feasible_by_batch = []
        adam_age_before_by_batch = []
        adam_age_after_by_batch = []
        evaluated_generation_by_batch = []
        next_generation_by_batch = []
        restart_events = []
        global_improvement_events = []
        nonfinite_derivative_values = 0
        sanitizer_violations = 0

        for event in self.events:
            for name in sorted(set(event) - TIMING_TELEMETRY_FIELDS):
                stable_digest.update(name.encode("utf-8") + b"\0")
                stable_digest.update(_array_bytes(event[name]))

            improved_by_batch.append(
                np.asarray(event["observed_member_improved"], dtype=bool).tolist()
            )
            stalled_before_by_batch.append(
                np.asarray(event["stalled_steps_before"], dtype=np.int64).tolist()
            )
            stalled_after_by_batch.append(
                np.asarray(event["stalled_steps_after"], dtype=np.int64).tolist()
            )
            feasible_by_batch.append(
                np.asarray(event["feasible"], dtype=bool).tolist()
            )
            adam_age_before_by_batch.append(
                np.asarray(event["adam_age_before"], dtype=np.int64).tolist()
            )
            adam_age_after_by_batch.append(
                np.asarray(event["adam_age_after"], dtype=np.int64).tolist()
            )
            evaluated_generation_by_batch.append(
                np.asarray(event["evaluated_generation"], dtype=np.int64).tolist()
            )
            next_generation_by_batch.append(
                np.asarray(event["next_generation"], dtype=np.int64).tolist()
            )
            for row in range(POPULATION_SIZE):
                batch = int(event["batch_index"][row])
                member = int(event["member_index"][row])
                nonfinite_derivative_values += int(
                    event["gradient_nonfinite_count"][row]
                )
                gradient_norm = float(event["gradient_norm"][row])
                clip_scale = float(event["gradient_clip_scale"][row])
                if not (
                    np.isfinite(gradient_norm)
                    and gradient_norm == 0.0
                    and np.isfinite(clip_scale)
                    and clip_scale == 1.0
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
                            "member": member,
                        }
                    )

        return {
            "adam_age_after_by_batch": adam_age_after_by_batch,
            "adam_age_before_by_batch": adam_age_before_by_batch,
            "evaluated_generation_by_batch": evaluated_generation_by_batch,
            "feasible_by_batch": feasible_by_batch,
            "global_improvement_events": global_improvement_events,
            "improved_by_batch": improved_by_batch,
            "restart_events": restart_events,
            "next_generation_by_batch": next_generation_by_batch,
            "nonfinite_derivative_values": nonfinite_derivative_values,
            "sanitizer_violations": sanitizer_violations,
            "stable_sha256": stable_digest.hexdigest(),
            "stalled_after_by_batch": stalled_after_by_batch,
            "stalled_before_by_batch": stalled_before_by_batch,
        }


class _ScriptedObjective:
    """Public-Objective stand-in with fixed loss and feasibility scripts."""

    def __init__(self, *, mode: str, max_evals: int) -> None:
        if mode not in {
            "descent",
            "improve-then-plateau",
            "late-crossing",
            "mixed",
            "plateau",
        }:
            raise ValueError(f"unknown feasible-progress mode: {mode}")
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
            [f"synthetic_clock_coordinate_{index}", "tuning"]
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
            raise RuntimeError("scripted objective logging started twice")
        self._started = True

    def _script(self, batch: int, active_members: int):
        members = jnp.arange(active_members, dtype=jnp.float32)
        base = jnp.asarray(100.0, dtype=jnp.float32) + members / 8.0
        descending = base - jnp.asarray(batch, dtype=jnp.float32)
        if self.mode in {"descent", "late-crossing"}:
            losses = descending
        elif self.mode == "plateau":
            losses = base
        elif self.mode == "improve-then-plateau":
            losses = base - jnp.asarray(min(batch, 1), dtype=jnp.float32)
        else:
            member_ids = jnp.arange(active_members, dtype=jnp.int32)
            losses = jnp.where(member_ids % 2 == 0, descending, base)

        feasible = jnp.zeros((active_members,), dtype=bool)
        if self.mode == "late-crossing" and batch == 3:
            feasible = feasible.at[0].set(True)
        return losses, feasible

    def _record(self, params, losses, feasible) -> None:
        host_params = np.asarray(jax.device_get(params))
        host_losses = np.asarray(jax.device_get(losses))
        host_feasible = np.asarray(jax.device_get(feasible), dtype=bool)
        self.batches.append(
            {
                "feasible": host_feasible.tolist(),
                "finite_loss": np.isfinite(host_losses).tolist(),
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
        active_members = int(params.shape[0])
        batch = len(self.batches)
        losses, feasible = self._script(batch, active_members)
        grads = jnp.zeros_like(params)
        self._record(params, losses, feasible)
        self.eval_count += active_members
        return losses, grads, {"is_feasible": feasible}

    def value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        batch = len(self.batches)
        losses, feasible = self._script(batch, 1)
        grads = jnp.zeros_like(params)
        self._record(params[None, :], losses, feasible)
        self.eval_count += 1
        return losses[0], grads, {"is_feasible": feasible[0]}


def _initial_population() -> np.ndarray:
    return np.linspace(
        -1.0,
        1.0,
        POPULATION_SIZE * N_PARAMS,
        dtype=np.float32,
    ).reshape(POPULATION_SIZE, N_PARAMS)


def _run_trace(
    *, mode: str, telemetry_enabled: bool, max_evals: int = MAX_EVALS
) -> dict[str, object]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the feasible-progress study requires a CPU backend")
    objective = _ScriptedObjective(mode=mode, max_evals=max_evals)
    telemetry = _ClockTelemetry() if telemetry_enabled else None
    captured_initial_hashes: list[list[str]] = []

    def capture_initial(value: object) -> None:
        population = np.asarray(jax.device_get(value))
        captured_initial_hashes.append(
            [_array_sha256(member) for member in population]
        )

    with jax.default_device(cpu_devices[0]), contextlib.redirect_stdout(io.StringIO()):
        BatchedRestartAdam().optimize(
            objective,
            init_params=_initial_population(),
            random_seed=SEED,
            population_size=POPULATION_SIZE,
            learning_rate_low=LEARNING_RATE,
            learning_rate_high=LEARNING_RATE,
            patience=PATIENCE,
            minimum_improvement=MINIMUM_IMPROVEMENT,
            safety_seconds=0.0,
            initial_population_callback=capture_initial,
            optimizer_telemetry_callback=telemetry,
        )

    if len(captured_initial_hashes) != 1:
        raise RuntimeError("initial population was not captured exactly once")
    trace: dict[str, object] = {
        "batch_sizes": [len(batch["parameter_hashes"]) for batch in objective.batches],
        "batches": objective.batches,
        "eval_count": objective.eval_count,
        "initial_population_hashes": captured_initial_hashes[0],
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
    }


def _changed_members(
    trace: dict[str, object], *, before_batch: int, after_batch: int
) -> list[int]:
    before = trace["batches"][before_batch]["parameter_hashes"]
    after = trace["batches"][after_batch]["parameter_hashes"]
    return [
        member
        for member, (left, right) in enumerate(zip(before, after, strict=True))
        if left != right
    ]


def _stalled_by_member(trace: dict[str, object]) -> dict[str, list[int]]:
    by_batch = trace["telemetry"]["stalled_after_by_batch"]
    return {
        str(member): [int(batch[member]) for batch in by_batch]
        for member in range(POPULATION_SIZE)
    }


def _restart_state_reset_members(
    trace: dict[str, object], *, restart_batch: int
) -> list[int]:
    telemetry = trace["telemetry"]
    return [
        member
        for member in range(POPULATION_SIZE)
        if (
            telemetry["adam_age_after_by_batch"][restart_batch][member] == 0
            and telemetry["evaluated_generation_by_batch"][restart_batch][member]
            + 1
            == telemetry["next_generation_by_batch"][restart_batch][member]
            and telemetry["stalled_before_by_batch"][restart_batch + 1][member]
            == 0
            and telemetry["adam_age_before_by_batch"][restart_batch + 1][member]
            == 0
            and telemetry["evaluated_generation_by_batch"][restart_batch + 1][
                member
            ]
            == 1
        )
    ]


def isolated_worker_trace() -> dict[str, object]:
    """Return the frozen timing-free mixed-member trace for child processes."""
    return _run_trace(mode="mixed", telemetry_enabled=True)


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
            "experiments.local_lab.feasible_progress_clock_worker",
            "--mode",
            "feasible-progress-clock-trace",
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
        raise RuntimeError("the feasible-progress study requires a CPU backend")

    descent = _run_trace(mode="descent", telemetry_enabled=True)
    descent_telemetry = descent["telemetry"]
    descent_finite_infeasible = sum(
        int(finite and not feasible)
        for batch in descent["batches"]
        for finite, feasible in zip(
            batch["finite_loss"], batch["feasible"], strict=True
        )
    )
    descent_improved_rows = sum(
        int(improved)
        for batch in descent_telemetry["improved_by_batch"]
        for improved in batch
    )
    descent_passed = (
        descent["batch_sizes"] == [8, 8, 8, 8]
        and descent["eval_count"] == 32
        and descent_finite_infeasible == 32
        and descent_improved_rows == 32
        and descent_telemetry["stalled_after_by_batch"] == [[0] * 8] * 4
        and descent_telemetry["restart_events"] == []
        and descent_telemetry["global_improvement_events"] == []
        and descent_telemetry["nonfinite_derivative_values"] == 0
        and descent_telemetry["sanitizer_violations"] == 0
    )

    plateau = _run_trace(mode="plateau", telemetry_enabled=True)
    plateau_telemetry = plateau["telemetry"]
    expected_all_restarts = [
        {"batch": 2, "kind": 0, "member": member, "round": 0}
        for member in range(POPULATION_SIZE)
    ]
    expected_plateau_improved = [
        [True] * 8,
        [False] * 8,
        [False] * 8,
        [True] * 8,
    ]
    expected_plateau_stalls = [[0] * 8, [1] * 8, [2] * 8, [0] * 8]
    plateau_changed_members = _changed_members(
        plateau, before_batch=2, after_batch=3
    )
    plateau_reset_members = _restart_state_reset_members(
        plateau, restart_batch=2
    )
    plateau_passed = (
        plateau["batch_sizes"] == [8, 8, 8, 8]
        and plateau["eval_count"] == 32
        and plateau_telemetry["improved_by_batch"] == expected_plateau_improved
        and plateau_telemetry["stalled_after_by_batch"] == expected_plateau_stalls
        and plateau_telemetry["restart_events"] == expected_all_restarts
        and plateau_changed_members == list(range(POPULATION_SIZE))
        and plateau_reset_members == list(range(POPULATION_SIZE))
        and plateau_telemetry["global_improvement_events"] == []
        and plateau_telemetry["nonfinite_derivative_values"] == 0
        and plateau_telemetry["sanitizer_violations"] == 0
    )

    improve_then_plateau = _run_trace(
        mode="improve-then-plateau",
        telemetry_enabled=True,
        max_evals=40,
    )
    improve_then_plateau_telemetry = improve_then_plateau["telemetry"]
    expected_delayed_restarts = [
        {"batch": 3, "kind": 0, "member": member, "round": 0}
        for member in range(POPULATION_SIZE)
    ]
    expected_delayed_improved = [
        [True] * 8,
        [True] * 8,
        [False] * 8,
        [False] * 8,
        [True] * 8,
    ]
    expected_delayed_stalls = [
        [0] * 8,
        [0] * 8,
        [1] * 8,
        [2] * 8,
        [0] * 8,
    ]
    improve_then_plateau_changed_members = _changed_members(
        improve_then_plateau, before_batch=3, after_batch=4
    )
    improve_then_plateau_reset_members = _restart_state_reset_members(
        improve_then_plateau, restart_batch=3
    )
    improve_then_plateau_passed = (
        improve_then_plateau["batch_sizes"] == [8, 8, 8, 8, 8]
        and improve_then_plateau["eval_count"] == 40
        and improve_then_plateau_telemetry["improved_by_batch"]
        == expected_delayed_improved
        and improve_then_plateau_telemetry["stalled_after_by_batch"]
        == expected_delayed_stalls
        and improve_then_plateau_telemetry["restart_events"]
        == expected_delayed_restarts
        and improve_then_plateau_changed_members
        == list(range(POPULATION_SIZE))
        and improve_then_plateau_reset_members
        == list(range(POPULATION_SIZE))
        and improve_then_plateau_telemetry["global_improvement_events"] == []
        and improve_then_plateau_telemetry["nonfinite_derivative_values"] == 0
        and improve_then_plateau_telemetry["sanitizer_violations"] == 0
    )

    crossing = _run_trace(mode="late-crossing", telemetry_enabled=True)
    crossing_telemetry = crossing["telemetry"]
    expected_crossing_feasible = [
        [False] * 8,
        [False] * 8,
        [False] * 8,
        [True] + [False] * 7,
    ]
    expected_global_event = [{"batch": 3, "member": 0}]
    crossing_passed = (
        crossing["batch_sizes"] == [8, 8, 8, 8]
        and crossing["eval_count"] == 32
        and crossing_telemetry["feasible_by_batch"] == expected_crossing_feasible
        and crossing_telemetry["stalled_after_by_batch"] == [[0] * 8] * 4
        and crossing_telemetry["restart_events"] == []
        and crossing_telemetry["global_improvement_events"]
        == expected_global_event
        and crossing_telemetry["nonfinite_derivative_values"] == 0
        and crossing_telemetry["sanitizer_violations"] == 0
    )

    mixed = _run_trace(mode="mixed", telemetry_enabled=True)
    mixed_telemetry = mixed["telemetry"]
    expected_mixed_improved = [
        [True] * 8,
        [member % 2 == 0 for member in range(POPULATION_SIZE)],
        [member % 2 == 0 for member in range(POPULATION_SIZE)],
        [True] * 8,
    ]
    expected_mixed_stalls = [
        [0] * 8,
        [0 if member % 2 == 0 else 1 for member in range(POPULATION_SIZE)],
        [0 if member % 2 == 0 else 2 for member in range(POPULATION_SIZE)],
        [0] * 8,
    ]
    expected_mixed_restarts = [
        {"batch": 2, "kind": 0, "member": member, "round": 0}
        for member in (1, 3, 5, 7)
    ]
    mixed_changed_members = _changed_members(
        mixed, before_batch=2, after_batch=3
    )
    mixed_reset_members = _restart_state_reset_members(mixed, restart_batch=2)
    mixed_passed = (
        mixed["batch_sizes"] == [8, 8, 8, 8]
        and mixed["eval_count"] == 32
        and mixed_telemetry["improved_by_batch"] == expected_mixed_improved
        and mixed_telemetry["stalled_after_by_batch"] == expected_mixed_stalls
        and mixed_telemetry["restart_events"] == expected_mixed_restarts
        and mixed_changed_members == [1, 3, 5, 7]
        and mixed_reset_members == [1, 3, 5, 7]
        and mixed_telemetry["global_improvement_events"] == []
        and mixed_telemetry["nonfinite_derivative_values"] == 0
        and mixed_telemetry["sanitizer_violations"] == 0
    )

    diagnostics_off = _run_trace(mode="mixed", telemetry_enabled=False)
    diagnostics_on = _run_trace(mode="mixed", telemetry_enabled=True)
    diagnostics_projection = _objective_projection(diagnostics_off)
    diagnostics_passed = diagnostics_projection == _objective_projection(
        diagnostics_on
    )

    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = isolated_left == isolated_right
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = "not-run-in-focused-test"

    cases = {
        "diagnostics_disabled_control": {
            "passed": diagnostics_passed,
            "trace_sha256": _json_sha256(diagnostics_projection),
        },
        "finite_infeasible_descent": {
            "finite_infeasible_observations": descent_finite_infeasible,
            "improved_rows": descent_improved_rows,
            "nonfinite_derivative_values": descent_telemetry[
                "nonfinite_derivative_values"
            ],
            "passed": descent_passed,
            "restart_events": descent_telemetry["restart_events"],
            "stalled_steps_by_member": _stalled_by_member(descent),
            "trace_sha256": _json_sha256(descent),
        },
        "finite_infeasible_improve_then_plateau": {
            "changed_members_after_restart": (
                improve_then_plateau_changed_members
            ),
            "passed": improve_then_plateau_passed,
            "restart_events": improve_then_plateau_telemetry[
                "restart_events"
            ],
            "restart_state_reset_members": (
                improve_then_plateau_reset_members
            ),
            "stalled_steps_by_member": _stalled_by_member(
                improve_then_plateau
            ),
        },
        "finite_infeasible_plateau_control": {
            "changed_members_after_restart": plateau_changed_members,
            "passed": plateau_passed,
            "restart_events": plateau_telemetry["restart_events"],
            "restart_state_reset_members": plateau_reset_members,
            "stalled_steps_by_member": _stalled_by_member(plateau),
        },
        "late_feasibility_crossing": {
            "first_feasible_batch": 3,
            "global_improvement_events": crossing_telemetry[
                "global_improvement_events"
            ],
            "passed": crossing_passed,
            "restart_events": crossing_telemetry["restart_events"],
            "stalled_steps_by_member": _stalled_by_member(crossing),
        },
        "mixed_member_clock": {
            "changed_members_after_boundary": mixed_changed_members,
            "passed": mixed_passed,
            "restart_events": mixed_telemetry["restart_events"],
            "restart_state_reset_members": mixed_reset_members,
            "stalled_steps_by_member": _stalled_by_member(mixed),
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
            "finite_infeasible_progress_resets_clock_confirmed"
            if passed
            else (
                "park_feasible_progress_clock_research"
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
            "claim_boundary": CLAIM_BOUNDARY,
            "initial_population_sha256": _array_sha256(_initial_population()),
            "learning_rate": LEARNING_RATE,
            "loss_dtype": "float32",
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "n_params": N_PARAMS,
            "patience": PATIENCE,
            "population_size": POPULATION_SIZE,
            "seed": SEED,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
