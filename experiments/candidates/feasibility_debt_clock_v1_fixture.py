"""Frozen deterministic projection for feasibility-debt-clock-v1.

The ``--run`` entry point is the terminal study orchestrator.  It launches two
credential-scrubbed CPU children, requires byte-identical sanitized outputs,
and emits one bounded JSON object.  Importing this module does not execute a
study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experiments.candidates.feasibility_debt_clock_v1 import (
    FeasibilityDebtBatchedRestartAdam,
)
from experiments.candidates.feasibility_debt_clock_v1_source import (
    verify_source_boundary,
)
from submission.submission import BatchedRestartAdam


STUDY_ID = "feasibility-debt-clock-v1"
PLAN_REVISION = "197a7433c235ef9cf2e160e8a3bd4a8889d33029"
PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "2026-09-01-feasibility-debt-clock-v1-plan.md"
)
PLAN_SHA256 = "f312663798b558dd0592aba8cb795d046529ca8e5f92a52754cae867a4c0e895"
POPULATION_SIZE = 3
DIMENSION = 2
MINIMUM_IMPROVEMENT = 1.0e-7
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
    "compatibility_no_restart": 91021,
    "compatibility_restart": 91031,
    "flat_debt": 91033,
    "falling_debt": 91079,
    "feasibility_switch": 91121,
    "partial_and_chunks": 91139,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


class ScriptedObjective:
    """Public-API stand-in with one committed scalar batch script."""

    def __init__(
        self,
        rows: list[tuple[float, float, bool]],
        *,
        max_evals: int,
        attack: str | None = None,
    ) -> None:
        self.rows = tuple(rows)
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
        self.input_fragments: list[tuple[int, np.ndarray]] = []
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

    def _logical_batch(self) -> int:
        return min(self.eval_count // POPULATION_SIZE, len(self.rows) - 1)

    def _make_aux(
        self, count: int, penalty: float, feasible: bool
    ) -> dict[str, Any]:
        penalties = jnp.full((count,), penalty, dtype=jnp.float32)
        feasible_values = jnp.full((count,), feasible, dtype=bool)
        aux: dict[str, Any] = {
            "is_feasible": feasible_values,
            "penalty": penalties,
            "sensitivity_loss": jnp.zeros((count,), dtype=jnp.float32),
            "violations": penalties[:, None],
            "power_values": {
                "hard": penalties[:, None],
                "soft": penalties[:, None],
                "detector": penalties[:, None],
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
                name = path.split(".", 1)[1]
                aux["power_values"][name] = jnp.zeros(
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
        elif attack == "negative-penalty":
            aux["penalty"] = jnp.full((count,), -1.0, dtype=jnp.float32)
        elif attack == "scalar-penalty":
            aux["penalty"] = jnp.asarray(1.0, dtype=jnp.float32)
        elif attack == "nan-penalty":
            aux["penalty"] = jnp.full((count,), jnp.nan, dtype=jnp.float32)
        else:
            raise AssertionError(f"unknown attack: {attack}")
        return aux

    def _evaluate(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        count = int(params.shape[0])
        logical_batch = self._logical_batch()
        self.input_fragments.append(
            (logical_batch, np.asarray(jax.device_get(params)))
        )
        loss, penalty, feasible = self.rows[logical_batch]
        losses = jnp.full((count,), loss, dtype=jnp.float32)
        grads = jnp.tile(
            jnp.asarray([[0.6, -0.8]], dtype=jnp.float32), (count, 1)
        )
        aux = self._make_aux(count, penalty, feasible)
        if "sensitivity_loss" in aux:
            aux["sensitivity_loss"] = jnp.full(
                (count,), loss - penalty, dtype=jnp.float32
            )
        self.eval_count += count
        if feasible and math.isfinite(loss):
            if self.first_feasible_loss == math.inf:
                self.first_feasible_loss = loss
            self.best_feasible_loss = min(self.best_feasible_loss, loss)
        return losses, grads, aux

    def vmap_value_and_grad_aux(self, params):
        return self._evaluate(params)

    def value_and_grad_aux(self, params):
        losses, grads, aux = self._evaluate(params[None, :])
        return (
            losses[0],
            grads[0],
            jax.tree.map(lambda value: value[0], aux),
        )

    def logical_input_commitments(self) -> list[dict[str, Any]]:
        grouped: dict[int, list[np.ndarray]] = {}
        for batch, value in self.input_fragments:
            grouped.setdefault(batch, []).append(value)
        return [
            _array_commitment(np.concatenate(grouped[index], axis=0))
            for index in sorted(grouped)
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
    rows: list[tuple[float, float, bool]],
    *,
    seed: int,
    patience: int,
    max_evals: int | None = None,
    chunk_size: int | None = None,
    attack: str | None = None,
) -> dict[str, Any]:
    objective = ScriptedObjective(
        rows,
        max_evals=max_evals if max_evals is not None else len(rows) * POPULATION_SIZE,
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
        assert mode is not None
        FeasibilityDebtBatchedRestartAdam().optimize(
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


def _restart_batches(run: dict[str, Any]) -> list[int]:
    batches: set[int] = set()
    for event in run["capture"].raw_events:
        triggered = event["restart_triggered"].astype(bool)
        if np.any(triggered):
            batches.update(int(value) for value in event["batch_index"][triggered])
    return sorted(batches)


def _attack_rejected_before_transition(attack: str) -> bool:
    objective = ScriptedObjective(
        [(3.0, 1.0, False)], max_evals=POPULATION_SIZE, attack=attack
    )
    capture = Capture()
    try:
        FeasibilityDebtBatchedRestartAdam().optimize(
            objective,
            progress_mode="feasibility_debt",
            **COMMON_SETTINGS,
            random_seed=SEEDS["partial_and_chunks"],
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


def _common_event(event: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: _array_commitment(value)
        for key, value in sorted(event.items())
        if not key.startswith("progress_")
        and key not in {"time_seconds", "evaluation_batch_seconds"}
    }


def _case_projection() -> dict[str, Any]:
    case_outcomes: dict[str, bool] = {}
    case_roots: dict[str, str] = {}

    no_restart_rows = [(9.0, 3.0, False), (8.0, 2.5, False),
                       (7.0, 2.0, False), (6.0, 1.5, False),
                       (5.0, 1.0, False)]
    protected = _run_optimizer(
        "protected", None, no_restart_rows,
        seed=SEEDS["compatibility_no_restart"], patience=8,
    )
    compatible = _run_optimizer(
        "candidate", "total_loss", no_restart_rows,
        seed=SEEDS["compatibility_no_restart"], patience=8,
    )
    case_outcomes["protected_compatibility_no_restart"] = (
        protected["root"] == compatible["root"]
    )
    case_roots["protected_compatibility_no_restart"] = protected["root"]

    restart_rows = [(4.0, 4.0, False)] * 5
    protected_restart = _run_optimizer(
        "protected", None, restart_rows,
        seed=SEEDS["compatibility_restart"], patience=2,
    )
    compatible_restart = _run_optimizer(
        "candidate", "total_loss", restart_rows,
        seed=SEEDS["compatibility_restart"], patience=2,
    )
    case_outcomes["protected_compatibility_restart"] = (
        protected_restart["root"] == compatible_restart["root"]
    )
    case_roots["protected_compatibility_restart"] = protected_restart["root"]

    flat_rows = [(8.0, 2.0, False), (7.0, 2.0, False),
                 (6.0, 2.0, False), (5.0, 2.0, False),
                 (4.0, 2.0, False)]
    flat_control = _run_optimizer(
        "protected", None, flat_rows,
        seed=SEEDS["flat_debt"], patience=3,
    )
    flat_treatment = _run_optimizer(
        "candidate", "feasibility_debt", flat_rows,
        seed=SEEDS["flat_debt"], patience=3,
    )
    stalls = [
        event["stalled_steps_after"].tolist()
        for event in flat_treatment["capture"].raw_events
    ]
    pre_restart_equal = (
        flat_control["projection"]["inputs"][:4]
        == flat_treatment["projection"]["inputs"][:4]
        and flat_control["projection"]["random_draws"][:1]
        == flat_treatment["projection"]["random_draws"][:1]
        and all(
            _common_event(control_event) == _common_event(treatment_event)
            for control_event, treatment_event in zip(
                flat_control["capture"].raw_events[:3],
                flat_treatment["capture"].raw_events[:3],
                strict=True,
            )
        )
    )
    case_outcomes["flat_debt_divergence"] = (
        _restart_batches(flat_control) == []
        and _restart_batches(flat_treatment) == [3]
        and stalls == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]
        and pre_restart_equal
    )
    case_roots["flat_debt_divergence"] = _sha256(
        _canonical_json({
            "control": flat_control["root"],
            "treatment": flat_treatment["root"],
        })
    )

    falling_rows = [(8.0, 3.0, False), (7.0, 2.5, False),
                    (6.0, 2.0, False), (5.0, 1.5, False),
                    (4.0, 1.0, False)]
    falling = _run_optimizer(
        "candidate", "feasibility_debt", falling_rows,
        seed=SEEDS["falling_debt"], patience=3,
    )
    falling_stalls = [
        event["stalled_steps_after"].tolist()
        for event in falling["capture"].raw_events
    ]
    case_outcomes["falling_debt_control"] = (
        _restart_batches(falling) == []
        and falling_stalls == [[0, 0, 0]] * 5
    )
    case_roots["falling_debt_control"] = falling["root"]

    switch_rows = [(8.0, 3.0, False), (7.0, 2.0, False),
                   (6.0, 0.0, True), (5.0, 0.0, True),
                   (5.0, 0.0, True), (5.0, 1.0, False)]
    switch = _run_optimizer(
        "candidate", "feasibility_debt", switch_rows,
        seed=SEEDS["feasibility_switch"], patience=3,
    )
    improvements = [
        bool(event["observed_member_improved"][0])
        for event in switch["capture"].raw_events
    ]
    ever_feasible = [
        bool(event["progress_ever_feasible_after"][0])
        for event in switch["capture"].raw_events
    ]
    case_outcomes["feasibility_switch_control"] = (
        improvements == [True, True, True, True, False, False]
        and ever_feasible == [False, False, True, True, True, True]
        and _restart_batches(switch) == []
    )
    case_roots["feasibility_switch_control"] = switch["root"]

    partial_rows = [(4.0, 2.0, False)] * 4
    partial = _run_optimizer(
        "candidate", "feasibility_debt", partial_rows,
        seed=SEEDS["partial_and_chunks"], patience=2, max_evals=11,
    )
    partial_event = partial["capture"].raw_events[-1]
    case_outcomes["partial_tail_control"] = (
        partial["objective"].eval_count == 11
        and partial_event["update_applied"].tolist() == [False, False]
        and partial_event["restart_triggered"].tolist() == [False, False]
        and partial_event["stalled_steps_before"].tolist() == [0, 0]
        and partial_event["stalled_steps_after"].tolist() == [0, 0]
    )
    case_roots["partial_tail_control"] = partial["root"]

    chunk_roots: dict[str, list[str]] = {}
    chunk_ok = True
    for label, rows, patience, max_evals in (
        ("falling", falling_rows, 3, None),
        ("partial", partial_rows, 2, 11),
    ):
        runs = [
            _run_optimizer(
                "candidate", "feasibility_debt", rows,
                seed=SEEDS["partial_and_chunks"], patience=patience,
                max_evals=max_evals, chunk_size=chunk,
            )
            for chunk in (None, 1, 2)
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
        roots = [_sha256(_canonical_json(item)) for item in logical]
        chunk_roots[label] = roots
        chunk_ok = chunk_ok and len(set(roots)) == 1
    case_outcomes["chunk_projection_equivalence"] = chunk_ok
    case_roots["chunk_projection_equivalence"] = _sha256(
        _canonical_json(chunk_roots)
    )

    attacks = [
        *[f"missing:{name}" for name in (
            "is_feasible", "penalty", "sensitivity_loss", "violations",
            "power_values", "power_values.hard", "power_values.soft",
            "power_values.detector",
        )],
        *[f"leading:{name}" for name in (
            "is_feasible", "penalty", "sensitivity_loss", "violations",
            "power_values.hard", "power_values.soft", "power_values.detector",
        )],
        "integer-feasibility",
        "negative-penalty",
        "scalar-penalty",
    ]
    rejected = sum(_attack_rejected_before_transition(attack) for attack in attacks)
    nan_run = _run_optimizer(
        "candidate", "feasibility_debt", [(3.0, 1.0, False)],
        seed=SEEDS["partial_and_chunks"], patience=8, attack="nan-penalty",
    )
    nan_improved = bool(
        nan_run["capture"].raw_events[0]["observed_member_improved"][0]
    )
    case_outcomes["auxiliary_fail_closed_matrix"] = (
        rejected == len(attacks) and not nan_improved
    )
    case_roots["auxiliary_fail_closed_matrix"] = _sha256(
        _canonical_json({"attacks": attacks, "rejected": rejected,
                         "nan_improved": nan_improved})
    )

    source = verify_source_boundary()
    case_outcomes["source_delta"] = source["valid"]
    case_roots["source_delta"] = source["boundary_root_sha256"]

    return {
        "case_outcomes": case_outcomes,
        "case_roots": case_roots,
        "candidate_source_sha256": source["candidate_text_sha256"],
        "protected_source_sha256": source["protected_text_sha256"],
        "first_treatment_restart_batch": 3,
        "source_boundary_root_sha256": source["boundary_root_sha256"],
    }


def _child_projection() -> dict[str, Any]:
    invocation_revision = os.environ.get("FDC_V1_INVOCATION_REVISION", "")
    if len(invocation_revision) != 40 or any(
        character not in "0123456789abcdef" for character in invocation_revision
    ):
        raise RuntimeError("missing authenticated invocation revision")
    plan_sha256 = _sha256(PLAN_PATH.read_bytes())
    if plan_sha256 != PLAN_SHA256:
        raise RuntimeError("frozen plan hash mismatch")
    projection = _case_projection()
    all_passed = all(projection["case_outcomes"].values())
    return {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": PLAN_REVISION,
        "plan_sha256": plan_sha256,
        "candidate_source_sha256": projection["candidate_source_sha256"],
        "fixture_source_sha256": _sha256(Path(__file__).read_bytes()),
        "protected_source_sha256": projection["protected_source_sha256"],
        "all_cases_passed": all_passed,
        "case_count": len(projection["case_outcomes"]),
        "case_outcomes": projection["case_outcomes"],
        "case_roots": projection["case_roots"],
        "first_treatment_restart_batch": projection[
            "first_treatment_restart_batch"
        ],
        "source_boundary_root_sha256": projection[
            "source_boundary_root_sha256"
        ],
        "core_root_sha256": _sha256(_canonical_json(projection)),
    }


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
        }
    )
    return retained


def _run_child(invocation_revision: str) -> tuple[bytes, dict[str, Any]]:
    environment = _scrubbed_environment()
    environment["FDC_V1_INVOCATION_REVISION"] = invocation_revision
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.candidates.feasibility_debt_clock_v1_fixture",
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
    if completed.returncode != 0:
        raise RuntimeError(f"child exited {completed.returncode}")
    if completed.stderr:
        raise RuntimeError("child emitted stderr")
    if len(completed.stdout) > 524_288:
        raise RuntimeError("child stdout exceeded cap")
    payload = json.loads(completed.stdout)
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError("child study ID mismatch")
    return completed.stdout, payload


def run_terminal_projection() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if status.returncode != 0 or status.stderr or status.stdout:
        raise RuntimeError("terminal projection requires a clean Git worktree")
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if revision_result.returncode != 0 or revision_result.stderr:
        raise RuntimeError("could not resolve invocation revision")
    invocation_revision = revision_result.stdout.decode("ascii").strip()
    first_bytes, first = _run_child(invocation_revision)
    second_bytes, second = _run_child(invocation_revision)
    runs_equal = first_bytes == second_bytes and first == second
    all_passed = bool(first["all_cases_passed"]) and runs_equal
    action = (
        "approve_feasibility_debt_candidate_for_fresh_panel_planning"
        if all_passed
        else "park_feasibility_debt_candidate"
    )
    return {
        "study_id": STUDY_ID,
        "invocation_revision": invocation_revision,
        "plan_revision": first["plan_revision"],
        "plan_sha256": first["plan_sha256"],
        "candidate_source_sha256": first["candidate_source_sha256"],
        "fixture_source_sha256": first["fixture_source_sha256"],
        "protected_source_sha256": first["protected_source_sha256"],
        "case_count": int(first["case_count"]),
        "all_cases_passed": all_passed,
        "runs_equal": runs_equal,
        "first_treatment_restart_batch": first[
            "first_treatment_restart_batch"
        ],
        "source_boundary_root_sha256": first[
            "source_boundary_root_sha256"
        ],
        "process_replay_root_sha256": _sha256(first_bytes),
        "action": action,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--child", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    payload = _child_projection() if args.child else run_terminal_projection()
    sys.stdout.buffer.write(_canonical_json(payload) + b"\n")
    return 0 if payload.get("all_cases_passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
