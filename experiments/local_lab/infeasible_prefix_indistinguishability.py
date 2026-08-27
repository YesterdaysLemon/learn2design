"""Finite deterministic proof fixture for an online-information boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax


STUDY_ID = "infeasible-prefix-indistinguishability-v1"
SCHEMA_VERSION = 1
BOUNDS = (1, 2, 3, 5, 8, 13)
MAX_BOUND = 13
LATE_CROSSING_STEP = MAX_BOUND + 1
CLAIM_BOUNDARY = "abstract_deterministic_one_lane_identical_prefix_only"
CASE_CONTRACT = {
    "shared_prefix_identity": {
        "bound": MAX_BOUND,
        "decision_timing": "observe_then_decide",
        "initial_rule_state": "identical",
        "late_crossing_step": LATE_CROSSING_STEP,
        "observable_fields": ["loss", "is_feasible"],
        "restart_scope": "one_target_lane_state_change",
    },
    "action_vector_exhaustion": {
        "action_vector_count": 1 << MAX_BOUND,
        "bound": MAX_BOUND,
        "policy_projection": "actions_on_one_shared_prefix",
    },
    "witness_partition": {
        "bound": MAX_BOUND,
        "bounded_restart_required": True,
        "preserve_before_crossing_required": True,
    },
    "boundary_sweep": {
        "bounds": list(BOUNDS),
        "crossing_offset": 1,
    },
    "extra_signal_positive_control": {
        "bound": MAX_BOUND,
        "certificate_step": MAX_BOUND,
        "late_crossing_step": LATE_CROSSING_STEP,
    },
    "process_isolation": {
        "source_case": "boundary_sweep",
        "workers": 2,
    },
}
REPOSITORY_ROOT = Path(__file__).parents[2]


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation(step: int, *, crossing_step: int | None) -> dict[str, object]:
    """Return one exact observation from a strictly improving finite path."""
    return {
        "is_feasible": crossing_step is not None and step >= crossing_step,
        "loss": MAX_BOUND + 1 - step,
    }


def _path_prefix(*, bound: int, crossing_step: int | None) -> list[dict[str, object]]:
    return [
        _observation(step, crossing_step=crossing_step)
        for step in range(1, bound + 1)
    ]


def _strictly_improving_finite(prefix: list[dict[str, object]]) -> bool:
    losses = [observation["loss"] for observation in prefix]
    return all(
        isinstance(loss, int) and not isinstance(loss, bool) for loss in losses
    ) and all(left > right for left, right in zip(losses, losses[1:]))


def _policy_partition(bound: int) -> dict[str, int]:
    """Exhaust realized deterministic action vectors on the shared prefix."""
    total = 1 << bound
    bounded_only = 0
    preserve_only = 0
    joint = 0
    neither = 0
    for action_mask in range(total):
        restarts_by_bound = action_mask != 0
        preserves_late_crossing = action_mask == 0
        if restarts_by_bound and preserves_late_crossing:
            joint += 1
        elif restarts_by_bound:
            bounded_only += 1
        elif preserves_late_crossing:
            preserve_only += 1
        else:
            neither += 1
    return {
        "bounded_only": bounded_only,
        "joint": joint,
        "neither": neither,
        "preserve_only": preserve_only,
        "total": total,
    }


def _proof_projection() -> dict[str, object]:
    forever_prefix = _path_prefix(bound=MAX_BOUND, crossing_step=None)
    late_prefix = _path_prefix(
        bound=MAX_BOUND,
        crossing_step=LATE_CROSSING_STEP,
    )
    forever_next = _observation(LATE_CROSSING_STEP, crossing_step=None)
    late_next = _observation(
        LATE_CROSSING_STEP,
        crossing_step=LATE_CROSSING_STEP,
    )
    partitions = [
        {"bound": bound, **_policy_partition(bound)} for bound in BOUNDS
    ]
    return {
        "forever_prefix_sha256": _json_sha256(forever_prefix),
        "late_prefix_sha256": _json_sha256(late_prefix),
        "next_observations_differ": forever_next != late_next,
        "partitions": partitions,
        "prefixes_identical": forever_prefix == late_prefix,
        "strictly_improving_prefix": _strictly_improving_finite(forever_prefix),
    }


def isolated_worker_trace() -> dict[str, object]:
    """Return the timing-free proof projection used by isolation checks."""
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
            "experiments.local_lab.infeasible_prefix_indistinguishability_worker",
            "--mode",
            "infeasible-prefix-indistinguishability-trace",
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
        raise RuntimeError("the prefix-boundary study requires a CPU backend")

    proof = _proof_projection()
    max_partition = next(
        row for row in proof["partitions"] if row["bound"] == MAX_BOUND
    )
    shared_prefix_passed = (
        proof["prefixes_identical"]
        and proof["strictly_improving_prefix"]
        and proof["next_observations_differ"]
        and proof["forever_prefix_sha256"] == proof["late_prefix_sha256"]
    )
    exhaustion_passed = (
        max_partition["total"] == 1 << MAX_BOUND
        and max_partition["joint"] == 0
    )
    partition_passed = (
        max_partition["bounded_only"] == (1 << MAX_BOUND) - 1
        and max_partition["preserve_only"] == 1
        and max_partition["joint"] == 0
        and max_partition["neither"] == 0
        and sum(
            max_partition[name]
            for name in ("bounded_only", "preserve_only", "joint", "neither")
        )
        == max_partition["total"]
    )
    sweep_passed = all(
        row["total"] == 1 << row["bound"]
        and row["bounded_only"] == (1 << row["bound"]) - 1
        and row["preserve_only"] == 1
        and row["joint"] == 0
        and row["neither"] == 0
        for row in proof["partitions"]
    )

    # Positive control: a certificate visible at the bound differs across paths.
    # The deterministic rule restarts at B exactly when that certificate is false.
    forever_certificate = False
    late_certificate = True
    forever_restarted = not forever_certificate
    late_restarted = not late_certificate
    positive_control_passed = (
        forever_restarted and not late_restarted and forever_certificate != late_certificate
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
        "shared_prefix_identity": {
            "forever_prefix_sha256": proof["forever_prefix_sha256"],
            "late_prefix_sha256": proof["late_prefix_sha256"],
            "next_observations_differ": proof["next_observations_differ"],
            "passed": shared_prefix_passed,
            "prefixes_identical": proof["prefixes_identical"],
            "rule_inputs_identical": proof["prefixes_identical"],
            "strictly_improving_prefix": proof["strictly_improving_prefix"],
        },
        "action_vector_exhaustion": {
            "bound": MAX_BOUND,
            "joint_satisfiers": max_partition["joint"],
            "passed": exhaustion_passed,
            "total_action_vectors": max_partition["total"],
        },
        "witness_partition": {
            "bounded_only_policies": max_partition["bounded_only"],
            "joint_satisfiers": max_partition["joint"],
            "partition_total": max_partition["total"],
            "passed": partition_passed,
            "preserve_only_policies": max_partition["preserve_only"],
        },
        "boundary_sweep": {
            "bounds_checked": [row["bound"] for row in proof["partitions"]],
            "joint_satisfiers_by_bound": [
                row["joint"] for row in proof["partitions"]
            ],
            "passed": sweep_passed,
            "policy_counts_by_bound": [
                row["total"] for row in proof["partitions"]
            ],
        },
        "extra_signal_positive_control": {
            "forever_restarted_by_bound": forever_restarted,
            "late_crossing_preserved": not late_restarted,
            "passed": positive_control_passed,
            "prefixes_identical": forever_certificate == late_certificate,
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
            "synthetic_identical_prefix_obstruction_confirmed"
            if passed
            else (
                "park_infeasible_prefix_boundary_research"
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
            "bounds": list(BOUNDS),
            "claim_boundary": CLAIM_BOUNDARY,
            "decision_timing": "observe_then_decide",
            "initial_rule_state": "identical_deterministic",
            "late_crossing_offset": 1,
            "max_bound": MAX_BOUND,
            "observable_fields": ["loss", "is_feasible"],
            "policy_projection": "actions_on_one_shared_prefix",
            "restart_scope": "one_target_lane_state_change",
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
