"""Run isolated one-candidate UIFO diagnostics on an otherwise idle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.uifo_paired.runner import (
    ROOT,
    UPSTREAM_REFERENCE,
    _git,
    _host_list,
    _parameter_hashes,
    _preflight_environment,
    _study_lock,
    atomic_json,
    atomic_text,
    canonical_text_sha256,
    environment_fingerprint,
    sha256,
    strict_json,
)


CANDIDATE_ROLES = ("anchor", "random_member_1", "semantic_prior")
OBJECTIVE_SAVE_FIELDS = (
    "eval_type",
    "batched_loss",
    "batched_sensitivity_loss",
    "batched_penalty",
    "batched_is_feasible",
)
JAX_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def jax_environment_policy() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in JAX_ENVIRONMENT_KEYS}


def build_probe_plan(
    *,
    topology: dict[str, object],
    optimizer_seed: int,
    population_size: int,
    roles: list[str],
    repeats: int,
    n_frequencies: int,
    worker_timeout_seconds: float,
    gpu_index: int,
    idle_samples: int,
    idle_sample_interval_seconds: float,
    max_idle_memory_mib: int,
    max_idle_utilization_percent: int,
    monitor_interval_seconds: float,
    cooldown_timeout_seconds: float,
) -> dict[str, object]:
    """Build a deterministic forward/reverse candidate-isolation plan."""
    if topology.get("kind") != "explicit" or not topology.get("value"):
        raise ValueError("candidate isolation requires an explicit topology")
    if population_size != 2:
        raise ValueError("this diagnostic reproduces exactly population_size=2")
    if repeats != 2:
        raise ValueError("this diagnostic requires exactly two order-balanced repeats")
    if n_frequencies < 1:
        raise ValueError("n_frequencies must be positive")
    if (
        not math.isfinite(worker_timeout_seconds)
        or not 0 < worker_timeout_seconds <= 300
        or monitor_interval_seconds <= 0
        or cooldown_timeout_seconds <= 0
    ):
        raise ValueError("worker timeout and monitor interval must be positive")
    if idle_samples < 1 or idle_sample_interval_seconds < 0:
        raise ValueError("idle sampling settings are invalid")
    if max_idle_memory_mib < 0 or not 0 <= max_idle_utilization_percent <= 100:
        raise ValueError("idle GPU thresholds are invalid")
    if gpu_index < 0:
        raise ValueError("gpu_index must be nonnegative")
    if roles != list(CANDIDATE_ROLES):
        raise ValueError(
            "candidate roles must be anchor, random_member_1, semantic_prior"
        )

    runs = []
    for repeat_index in range(repeats):
        ordered_roles = roles if repeat_index % 2 == 0 else list(reversed(roles))
        for order_index, role in enumerate(ordered_roles):
            runs.append(
                {
                    "run_id": (
                        f"repeat{repeat_index + 1:02d}__"
                        f"order{order_index + 1:02d}__{role}"
                    ),
                    "repeat_index": repeat_index,
                    "order_index": order_index,
                    "role": role,
                }
            )

    return {
        "format_version": 1,
        "study_kind": "isolated_uifo_candidate_probe",
        "configuration": {
            "topology": topology,
            "topology_sha256": hashlib.sha256(
                str(topology["value"]).encode()
            ).hexdigest(),
            "optimizer_seed": int(optimizer_seed),
            "population_size": int(population_size),
            "roles": list(roles),
            "repeats": int(repeats),
            "n_frequencies": int(n_frequencies),
            "worker_timeout_seconds": float(worker_timeout_seconds),
            "gpu_index": int(gpu_index),
            "idle_samples": int(idle_samples),
            "idle_sample_interval_seconds": float(idle_sample_interval_seconds),
            "max_idle_memory_mib": int(max_idle_memory_mib),
            "max_idle_utilization_percent": int(
                max_idle_utilization_percent
            ),
            "monitor_interval_seconds": float(monitor_interval_seconds),
            "cooldown_timeout_seconds": float(cooldown_timeout_seconds),
        },
        "runs": runs,
    }


def parse_gpu_sample(output: str) -> dict[str, object]:
    """Parse one nounits CSV row from ``nvidia-smi``."""
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU sample row, received {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(
            "GPU sample must contain index, UUID, name, used, total, and utilization"
        )
    try:
        index = int(fields[0])
        used, total, utilization = (int(field) for field in fields[3:])
    except ValueError as error:
        raise RuntimeError(f"invalid GPU sample: {rows[0]!r}") from error
    if (
        index < 0
        or not fields[1]
        or not fields[2]
        or used < 0
        or total <= 0
        or used > total
        or not 0 <= utilization <= 100
    ):
        raise RuntimeError(f"out-of-range GPU sample: {rows[0]!r}")
    return {
        "gpu_index": index,
        "gpu_uuid": fields[1],
        "gpu_name": fields[2],
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
    }


def parse_compute_processes(output: str) -> list[dict[str, object]]:
    """Parse the active compute-process rows reported by ``nvidia-smi``."""
    processes = []
    for row in (line.strip() for line in output.splitlines() if line.strip()):
        fields = [field.strip() for field in row.split(",", maxsplit=2)]
        if len(fields) != 3:
            raise RuntimeError(f"invalid compute-process row: {row!r}")
        try:
            pid = int(fields[0])
        except ValueError as error:
            raise RuntimeError(f"invalid compute-process PID: {row!r}") from error
        memory = None
        if fields[2] not in {"[N/A]", "N/A", "Not Supported"}:
            try:
                memory = int(fields[2])
            except ValueError as error:
                raise RuntimeError(
                    f"invalid compute-process memory: {row!r}"
                ) from error
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_gpu_memory_mib": memory,
            }
        )
    return processes


def query_gpu_sample(gpu_index: int) -> dict[str, object]:
    """Capture total-device memory and utilization without importing JAX."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi GPU sampling failed: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if process_query.returncode != 0:
        raise RuntimeError(
            "nvidia-smi compute-process query failed: "
            + (process_query.stderr.strip() or f"exit {process_query.returncode}")
        )
    return {
        **parse_gpu_sample(completed.stdout),
        "compute_processes": parse_compute_processes(process_query.stdout),
        "captured_utc": datetime.now(UTC).isoformat(),
    }


def assess_idle_samples(
    samples: list[dict[str, object]],
    *,
    max_memory_mib: int,
    max_utilization_percent: int,
) -> dict[str, object]:
    """Return a fail-closed idle decision from pre-worker device samples."""
    if not samples:
        raise ValueError("at least one idle sample is required")
    peak_memory = max(int(sample["memory_used_mib"]) for sample in samples)
    peak_utilization = max(
        int(sample["utilization_percent"]) for sample in samples
    )
    uuids = {str(sample["gpu_uuid"]) for sample in samples}
    compute_processes = [
        process
        for sample in samples
        for process in sample.get("compute_processes", [])
    ]
    return {
        "passed": (
            peak_memory <= max_memory_mib
            and peak_utilization <= max_utilization_percent
            and len(uuids) == 1
            and not compute_processes
        ),
        "sample_count": len(samples),
        "peak_memory_used_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "gpu_uuid": next(iter(uuids)) if len(uuids) == 1 else None,
        "compute_processes": compute_processes,
        "thresholds": {
            "max_memory_used_mib": max_memory_mib,
            "max_utilization_percent": max_utilization_percent,
        },
        "samples": samples,
    }


def collect_idle_report(configuration: dict[str, object]) -> dict[str, object]:
    samples = []
    count = int(configuration["idle_samples"])
    interval = float(configuration["idle_sample_interval_seconds"])
    for index in range(count):
        samples.append(query_gpu_sample(int(configuration["gpu_index"])))
        if index + 1 < count and interval:
            time.sleep(interval)
    return assess_idle_samples(
        samples,
        max_memory_mib=int(configuration["max_idle_memory_mib"]),
        max_utilization_percent=int(
            configuration["max_idle_utilization_percent"]
        ),
    )


def wait_for_idle(configuration: dict[str, object]) -> dict[str, object]:
    """Wait a bounded interval for the worker context to release the device."""
    deadline = time.monotonic() + float(configuration["cooldown_timeout_seconds"])
    report = collect_idle_report(configuration)
    while not report["passed"] and time.monotonic() < deadline:
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        report = collect_idle_report(configuration)
    report["cooldown_timeout_seconds"] = float(
        configuration["cooldown_timeout_seconds"]
    )
    return report


def require_gpu_identity(
    report: dict[str, object], expected_gpu_uuid: str
) -> dict[str, object]:
    """Bind an idle report to the GPU selected by the initial preflight."""
    matches = report.get("gpu_uuid") == expected_gpu_uuid
    report["expected_gpu_uuid"] = expected_gpu_uuid
    report["gpu_identity_matches"] = matches
    report["passed"] = bool(report.get("passed")) and matches
    return report


def _candidate_stats(candidate) -> dict[str, object]:
    import numpy as np

    values = np.asarray(candidate, dtype=np.float64)
    clipped = np.clip(values, -50.0, 50.0)
    unit = 1.0 / (1.0 + np.exp(-clipped))
    return {
        "n_params": int(values.size),
        "finite_fraction": float(np.isfinite(values).mean()),
        "unbounded_min": float(np.nanmin(values)),
        "unbounded_max": float(np.nanmax(values)),
        "unit_min": float(np.nanmin(unit)),
        "unit_median": float(np.nanmedian(unit)),
        "unit_max": float(np.nanmax(unit)),
    }


def host_pytree(value):
    """Recursively transfer every auxiliary pytree leaf to JSON-safe host data."""
    import jax

    return jax.tree.map(_host_list, value)


def construct_candidates(objective, optimizer_seed: int, population_size: int):
    """Reproduce the current population-2 candidate construction exactly."""
    from submission.submission import BatchedRestartAdam

    algorithm = BatchedRestartAdam()
    algorithm.prepare(objective, unbounded=True, random_seed=optimizer_seed)
    random_population = objective.random_params_unbounded(population_size)
    semantic_prior = algorithm._semantic_prior(objective)
    if semantic_prior is None:
        raise RuntimeError("semantic prior could not be constructed")
    return (
        {
            "anchor": algorithm._feasibility_anchor(objective),
            "random_member_1": random_population[1],
            "semantic_prior": semantic_prior,
        },
        random_population,
    )


def execute_probe(
    config: dict[str, object],
    milestone_path: Path,
    runtime_environment: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate exactly one selected candidate through the public Objective API."""
    import jax
    import numpy as np

    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("candidate isolation requires a GPU accelerator")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["expected_gpu_uuid"]:
        raise RuntimeError("worker is not bound to the planned physical GPU")
    if str(jax.devices()[0].device_kind) != config["expected_gpu_name"]:
        raise RuntimeError("worker GPU kind differs from the planned device")
    worker_environment = runtime_environment or environment_fingerprint()
    worker_jax_environment = jax_environment_policy()

    from dfbench import Objective
    from dfbench.problems import UIFOProblem

    topology = config["topology"]
    problem_kwargs = {
        "size": 3,
        "n_frequencies": int(config["n_frequencies"]),
    }
    problem_kwargs["topology"] = str(topology["value"])

    problem = UIFOProblem(**problem_kwargs)
    objective = Objective(
        problem,
        max_evals=1,
        save_time_steps=True,
        save_params_history=False,
        save_batched_params_history=False,
        save=list(OBJECTIVE_SAVE_FIELDS),
        verbose=0,
    )
    optimizer_seed = int(config["optimizer_seed"])
    population_size = int(config["population_size"])
    candidates, random_population = construct_candidates(
        objective, optimizer_seed, population_size
    )
    role = str(config["role"])
    if role not in candidates:
        raise ValueError(f"unknown candidate role: {role!r}")
    candidate = candidates[role]

    host_candidate = np.asarray(jax.device_get(candidate))
    if host_candidate.shape != (objective.n_params,):
        raise RuntimeError("selected candidate has the wrong parameter shape")
    if not np.all(np.isfinite(host_candidate)):
        raise RuntimeError("selected candidate contains non-finite parameters")
    random_hashes = _parameter_hashes(
        np.asarray(jax.device_get(random_population))
    )
    candidate_hashes = {
        name: _parameter_hashes(np.asarray(jax.device_get(value)))[0]
        for name, value in candidates.items()
    }
    candidate_hash = candidate_hashes[role]

    topology_string = str(problem.topology_string)
    topology_sha256 = hashlib.sha256(topology_string.encode()).hexdigest()
    if topology_string != str(topology["value"]):
        raise RuntimeError("resolved topology differs from the explicit input")
    if topology_sha256 != config["topology_sha256"]:
        raise RuntimeError("resolved topology digest differs from the probe plan")
    milestone = {
        "format_version": 1,
        "status": "candidate_ready",
        "worker_pid": os.getpid(),
        "environment": worker_environment,
        "jax_environment": worker_jax_environment,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "problem": {
            "n_params": objective.n_params,
            "topology_string": topology_string,
            "topology_sha256": topology_sha256,
        },
        "candidate": {
            "role": role,
            "sha256": candidate_hash,
            "stats": _candidate_stats(host_candidate),
            "all_candidate_sha256": candidate_hashes,
            "random_population_sha256": random_hashes,
        },
    }
    atomic_json(milestone_path, milestone)

    objective.start_logging()
    milestone["status"] = "evaluation_started"
    milestone["evaluation_started_utc"] = datetime.now(UTC).isoformat()
    atomic_json(milestone_path, milestone)
    started = time.perf_counter()
    loss, gradient, auxiliary = objective.value_and_grad_aux(candidate)
    loss, gradient, auxiliary = jax.block_until_ready(
        (loss, gradient, auxiliary)
    )
    call_wall_seconds = time.perf_counter() - started

    host_loss = float(np.asarray(jax.device_get(loss)))
    host_gradient = np.asarray(jax.device_get(gradient), dtype=np.float64)
    host_auxiliary = host_pytree(auxiliary)
    if objective.eval_count != 1 or objective.log_call_count != 1:
        raise RuntimeError("one-candidate probe produced unexpected accounting")
    if len(objective.loss_history) != 1 or len(objective.is_feasible_history) != 1:
        raise RuntimeError("one-candidate probe did not admit exactly one history item")

    return {
        "format_version": 1,
        "status": "complete",
        "completed_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "environment": worker_environment,
        "jax_environment": worker_jax_environment,
        "problem": {
            "n_params": objective.n_params,
            "spec": objective.problem_spec,
            "topology_string": topology_string,
            "topology_sha256": topology_sha256,
        },
        "candidate": {
            "role": role,
            "sha256": candidate_hash,
            "stats": _candidate_stats(host_candidate),
            "all_candidate_sha256": candidate_hashes,
            "random_population_sha256": random_hashes,
        },
        "result": {
            "loss": host_loss if math.isfinite(host_loss) else None,
            "gradient_norm": (
                float(np.linalg.norm(host_gradient))
                if np.all(np.isfinite(host_gradient))
                else None
            ),
            "gradient_finite_fraction": float(
                np.isfinite(host_gradient).mean()
            ),
            "auxiliary": host_auxiliary,
            "call_wall_seconds": call_wall_seconds,
            "objective_time_elapsed_seconds": float(objective.time_elapsed),
        },
        "objective_accounting": {
            "eval_count": objective.eval_count,
            "log_call_count": objective.log_call_count,
            "eval_type_counts": objective.eval_type_counts,
        },
        "objective_configuration": {
            "max_evals": 1,
            "max_time_seconds": None,
            "save": list(OBJECTIVE_SAVE_FIELDS),
            "save_params_history": False,
            "save_batched_params_history": False,
            "save_time_steps": True,
        },
    }


def _terminate_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def run_monitored_worker(
    *,
    config_path: Path,
    output_path: Path,
    milestone_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    configuration: dict[str, object],
    expected_gpu_uuid: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Run one fresh worker while sampling total GPU usage."""
    command = [
        sys.executable,
        "-m",
        "experiments.uifo_paired.candidate_probe",
        "--worker-config",
        str(config_path),
        "--worker-output",
        str(output_path),
        "--worker-milestone",
        str(milestone_path),
    ]
    worker_environment = os.environ.copy()
    worker_environment["CUDA_VISIBLE_DEVICES"] = expected_gpu_uuid
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=worker_environment,
            start_new_session=os.name != "nt",
        )
        started = time.perf_counter()
        samples = []
        timed_out = False
        monitoring_error = None
        external_processes = []
        visible_worker_process = None
        try:
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > float(configuration["worker_timeout_seconds"]):
                    timed_out = True
                    _terminate_worker(process)
                    break
                try:
                    sample = query_gpu_sample(int(configuration["gpu_index"]))
                    sample["worker_elapsed_seconds"] = elapsed
                    samples.append(sample)
                    if sample["gpu_uuid"] != expected_gpu_uuid:
                        monitoring_error = "GPU UUID changed during the worker"
                        _terminate_worker(process)
                        break
                    processes = sample.get("compute_processes", [])
                    unexpected = [
                        item
                        for item in processes
                        if int(item["pid"]) != process.pid
                    ]
                    if unexpected:
                        external_processes.extend(unexpected)
                        _terminate_worker(process)
                        break
                    matched = [
                        item for item in processes if int(item["pid"]) == process.pid
                    ]
                    if matched:
                        visible_worker_process = {
                            "pid": process.pid,
                            "process_name": str(matched[0]["process_name"]),
                        }
                except Exception as error:  # fail closed if provenance is incomplete
                    monitoring_error = f"{type(error).__name__}: {error}"
                    _terminate_worker(process)
                    break
                time.sleep(float(configuration["monitor_interval_seconds"]))
        except BaseException:
            _terminate_worker(process)
            raise
        process.wait()

    wall_seconds = time.perf_counter() - started
    peak_memory = (
        max(int(sample["memory_used_mib"]) for sample in samples)
        if samples
        else None
    )
    peak_utilization = (
        max(int(sample["utilization_percent"]) for sample in samples)
        if samples
        else None
    )
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        "",
        "",
    )
    monitor = {
        "worker_pid": process.pid,
        "sample_count": len(samples),
        "peak_memory_used_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "timed_out": timed_out,
        "monitoring_error": monitoring_error,
        "external_compute_processes": external_processes,
        "visible_worker_process": (
            visible_worker_process
        ),
        "worker_compute_context_observed": visible_worker_process is not None,
        "contaminated": bool(external_processes),
        "full_wall_seconds": wall_seconds,
        "samples": samples,
    }
    return completed, monitor


def summarize_probe_records(
    plan: dict[str, object], records: list[dict[str, object]]
) -> dict[str, object]:
    """Summarize completed and cleanly censored diagnostic outcomes."""
    expected_ids = {run["run_id"] for run in plan["runs"]}
    observed_ids = {record["config"]["run_id"] for record in records}
    completed = [
        record
        for record in records
        if record.get("status") == "complete"
        and record.get("integrity_validated") is True
    ]
    clean_timeouts = [
        record
        for record in records
        if record.get("status") == "error"
        and record.get("error", {}).get("type") == "WorkerTimeout"
        and record.get("candidate_identity_validated") is True
        and record.get("provenance_validated") is True
        and record.get("worker_process", {}).get("timed_out") is True
        and record.get("gpu_monitor", {}).get("timed_out") is True
    ]
    admissible = completed + clean_timeouts
    construction_maps = {
        json.dumps(record["candidate"]["all_candidate_sha256"], sort_keys=True)
        for record in admissible
    }
    construction_consistent = len(construction_maps) == 1 and bool(admissible)
    role_summaries = {}
    hash_consistent = True
    for role in plan["configuration"]["roles"]:
        role_records = [
            record
            for record in admissible
            if record["config"].get("role") == role
        ]
        role_completed = [
            record
            for record in role_records
            if record.get("status") == "complete"
            and record.get("integrity_validated") is True
        ]
        role_timeouts = [
            record
            for record in role_records
            if record.get("error", {}).get("type") == "WorkerTimeout"
        ]
        hashes = sorted({record["candidate"]["sha256"] for record in role_records})
        role_summaries[role] = {
            "identified_runs": len(role_records),
            "completed_runs": len(role_completed),
            "timeout_runs": len(role_timeouts),
            "other_error_runs": (
                len(role_records) - len(role_completed) - len(role_timeouts)
            ),
            "candidate_sha256": hashes[0] if len(hashes) == 1 else None,
            "candidate_hash_consistent": len(hashes) == 1 and bool(role_records),
            "call_wall_seconds": [
                record["result"]["call_wall_seconds"]
                for record in role_completed
            ],
            "peak_memory_used_mib": [
                record["gpu_monitor"]["peak_memory_used_mib"]
                for record in role_records
            ],
        }
        hash_consistent &= role_summaries[role]["candidate_hash_consistent"]

    expected_per_role = int(plan["configuration"]["repeats"])
    protocol_complete = (
        observed_ids == expected_ids
        and len(admissible) == len(plan["runs"])
        and all(
            summary["identified_runs"] == expected_per_role
            for summary in role_summaries.values()
        )
        and hash_consistent
        and construction_consistent
    )
    return {
        "format_version": 1,
        "study_kind": plan["study_kind"],
        "expected_runs": len(expected_ids),
        "observed_runs": len(observed_ids),
        "candidate_identified_runs": sum(
            record.get("candidate_identity_validated") is True
            for record in records
        ),
        "admissible_diagnostic_outcomes": len(admissible),
        "completed_runs": len(completed),
        "error_runs": len(records) - len(completed),
        "missing_run_ids": sorted(expected_ids - observed_ids),
        "unexpected_run_ids": sorted(observed_ids - expected_ids),
        "candidate_hashes_consistent": hash_consistent,
        "candidate_construction_consistent": construction_consistent,
        "all_evaluations_completed": len(completed) == len(expected_ids),
        "diagnostic_complete": protocol_complete,
        "performance_inference_ready": False,
        "role_summary": role_summaries,
        "note": (
            "Diagnostic completeness permits clean right-censoring with a "
            "validated candidate identity. This is not an optimizer or "
            "competition-throughput comparison."
        ),
    }


def _error_record(
    config: dict[str, object],
    environment: dict[str, object],
    error_type: str,
    message: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "status": "error",
        "completed_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "environment": environment,
        "error": {"type": error_type, "message": message},
    }


def _safe_environment_fingerprint() -> dict[str, object]:
    try:
        return environment_fingerprint()
    except Exception as error:
        return {
            "unavailable": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def _validate_artifact_descriptor(
    output_dir: Path,
    descriptor: dict[str, object],
    expected_relative_path: str,
) -> Path:
    if descriptor.get("path") != expected_relative_path:
        raise RuntimeError(f"unexpected artifact path: {descriptor.get('path')!r}")
    path = (output_dir / expected_relative_path).resolve()
    if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
        raise RuntimeError(f"missing or escaped artifact path: {expected_relative_path}")
    if descriptor.get("sha256") != sha256(path):
        raise RuntimeError(f"artifact digest mismatch: {expected_relative_path}")
    return path


def _validate_worker_environment(
    environment: dict[str, object],
    worker_jax_environment: dict[str, object],
    expected_config: dict[str, object],
    expected_environment: dict[str, object],
) -> None:
    expected_uuid = str(expected_config["expected_gpu_uuid"])
    if (
        environment.get("backend") != "gpu"
        or environment.get("device_count") != 1
        or environment.get("versions") != expected_environment.get("versions")
        or environment.get("platform") != expected_environment.get("platform")
        or environment.get("python") != expected_environment.get("python")
    ):
        raise RuntimeError("worker runtime environment drifted")
    device_kinds = environment.get("device_kinds", [])
    if (
        len(device_kinds) != 1
        or str(device_kinds[0]) != expected_config["expected_gpu_name"]
    ):
        raise RuntimeError("worker did not expose exactly one planned GPU")
    if worker_jax_environment.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("worker was not bound to the planned GPU UUID")
    parent_policy = jax_environment_policy()
    for name in JAX_ENVIRONMENT_KEYS:
        if name == "CUDA_VISIBLE_DEVICES":
            continue
        if worker_jax_environment.get(name) != parent_policy.get(name):
            raise RuntimeError(f"worker environment drifted for {name}")


def _validate_candidate_payload(
    candidate: dict[str, object], expected_config: dict[str, object]
) -> None:
    role = expected_config["role"]
    candidate_hashes = candidate.get("all_candidate_sha256", {})
    if not isinstance(candidate_hashes, dict) or set(candidate_hashes) != set(
        CANDIDATE_ROLES
    ):
        raise RuntimeError("candidate construction hash set is incomplete")
    hashes = list(candidate_hashes.values())
    if (
        any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
        or len(set(hashes)) != len(CANDIDATE_ROLES)
    ):
        raise RuntimeError("candidate role hashes are invalid or not distinct")
    if candidate.get("role") != role or candidate.get("sha256") != candidate_hashes[role]:
        raise RuntimeError("selected candidate identity drifted")
    random_hashes = candidate.get("random_population_sha256", [])
    if (
        len(random_hashes) != 2
        or candidate_hashes["random_member_1"] != random_hashes[1]
    ):
        raise RuntimeError("preserved random member identity drifted")


def validate_candidate_identity_record(
    record: dict[str, object],
    expected_config: dict[str, object],
    expected_environment: dict[str, object],
    output_dir: Path,
) -> None:
    """Validate candidate identity even when its evaluation is right-censored."""
    if record.get("config") != expected_config:
        raise RuntimeError("probe record configuration drifted")
    run_id = str(expected_config["run_id"])
    descriptor = record.get("worker_milestone")
    if not isinstance(descriptor, dict):
        raise RuntimeError("worker is missing its candidate milestone")
    milestone_path = _validate_artifact_descriptor(
        output_dir, descriptor, f"milestones/{run_id}.json"
    )
    milestone = json.loads(milestone_path.read_text(encoding="utf-8"))
    if (
        descriptor.get("payload") != milestone
        or milestone.get("format_version") != 1
        or milestone.get("status") != "evaluation_started"
        or milestone.get("config") != expected_config
        or int(milestone.get("worker_pid", 0)) < 1
    ):
        raise RuntimeError("candidate milestone is incomplete or drifted")
    _validate_worker_environment(
        milestone.get("environment", {}),
        milestone.get("jax_environment", {}),
        expected_config,
        expected_environment,
    )

    problem = milestone.get("problem", {})
    if (
        problem.get("topology_string") != expected_config["topology"]["value"]
        or problem.get("topology_sha256") != expected_config["topology_sha256"]
        or int(problem.get("n_params", 0)) < 1
    ):
        raise RuntimeError("milestone problem identity drifted")
    candidate = milestone.get("candidate", {})
    _validate_candidate_payload(candidate, expected_config)
    record_problem = record.get("problem", {})
    if (
        record.get("candidate") != candidate
        or record_problem.get("n_params") != problem.get("n_params")
        or record_problem.get("topology_string") != problem.get("topology_string")
        or record_problem.get("topology_sha256") != problem.get("topology_sha256")
    ):
        raise RuntimeError("record does not preserve its milestone identity")
    process = record.get("worker_process", {})
    if process.get("pid") != milestone.get("worker_pid"):
        raise RuntimeError("parent and worker PID handshake failed")


def validate_probe_provenance(
    record: dict[str, object],
    expected_config: dict[str, object],
    expected_environment: dict[str, object],
    output_dir: Path,
) -> None:
    """Validate device, process, log, and telemetry provenance for any outcome."""
    validate_candidate_identity_record(
        record, expected_config, expected_environment, output_dir
    )
    expected_uuid = str(expected_config["expected_gpu_uuid"])
    idle = record.get("idle_preflight", {})
    postflight = record.get("postflight_idle", {})
    if (
        not idle.get("passed")
        or not postflight.get("passed")
        or idle.get("gpu_uuid") != expected_uuid
        or postflight.get("gpu_uuid") != expected_uuid
    ):
        raise RuntimeError("probe idle gates did not pass on the planned GPU")

    process = record.get("worker_process", {})
    if int(process.get("pid", 0)) < 1 or float(
        process.get("full_wall_seconds", 0)
    ) <= 0:
        raise RuntimeError("worker process provenance is incomplete")
    run_id = str(expected_config["run_id"])
    _validate_artifact_descriptor(
        output_dir, process.get("stdout", {}), f"logs/{run_id}.stdout.log"
    )
    _validate_artifact_descriptor(
        output_dir, process.get("stderr", {}), f"logs/{run_id}.stderr.log"
    )

    monitor = record.get("gpu_monitor", {})
    visible = monitor.get("visible_worker_process") or {}
    if (
        monitor.get("worker_pid") != process.get("pid")
        or monitor.get("monitoring_error") is not None
        or monitor.get("contaminated") is not False
        or monitor.get("external_compute_processes")
        or monitor.get("worker_compute_context_observed") is not True
        or visible.get("pid") != process.get("pid")
        or int(monitor.get("sample_count", 0)) < 1
    ):
        raise RuntimeError("GPU monitor did not authenticate a clean worker trace")
    telemetry_path = _validate_artifact_descriptor(
        output_dir,
        monitor.get("telemetry", {}),
        f"telemetry/{run_id}.jsonl",
    )
    samples = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(samples) != monitor["sample_count"]:
        raise RuntimeError("GPU telemetry row count drifted")
    if any(sample.get("gpu_uuid") != expected_uuid for sample in samples):
        raise RuntimeError("GPU telemetry identity drifted")
    worker_pid = int(process["pid"])
    telemetry_processes = [
        item
        for sample in samples
        for item in sample.get("compute_processes", [])
    ]
    foreign_processes = [
        item for item in telemetry_processes if int(item["pid"]) != worker_pid
    ]
    worker_processes = [
        item for item in telemetry_processes if int(item["pid"]) == worker_pid
    ]
    if foreign_processes or not worker_processes:
        raise RuntimeError("GPU telemetry does not prove an exclusive worker context")
    if visible.get("process_name") not in {
        str(item["process_name"]) for item in worker_processes
    }:
        raise RuntimeError("GPU monitor process identity drifted from telemetry")
    if max(int(sample["memory_used_mib"]) for sample in samples) != monitor.get(
        "peak_memory_used_mib"
    ):
        raise RuntimeError("GPU telemetry peak memory drifted")
    if max(int(sample["utilization_percent"]) for sample in samples) != monitor.get(
        "peak_utilization_percent"
    ):
        raise RuntimeError("GPU telemetry peak utilization drifted")


def validate_completed_probe_record(
    record: dict[str, object],
    expected_config: dict[str, object],
    expected_environment: dict[str, object],
    output_dir: Path,
) -> None:
    """Fail closed on any completed-record or referenced-artifact drift."""
    if record.get("format_version") != 1 or record.get("status") != "complete":
        raise RuntimeError("probe record is not a format-1 completion")
    validate_probe_provenance(
        record, expected_config, expected_environment, output_dir
    )
    milestone = record["worker_milestone"]["payload"]
    if (
        record.get("environment") != milestone.get("environment")
        or record.get("jax_environment") != milestone.get("jax_environment")
    ):
        raise RuntimeError("completed record runtime differs from its milestone")

    accounting = record.get("objective_accounting", {})
    if accounting.get("eval_count") != 1 or accounting.get("log_call_count") != 1:
        raise RuntimeError("probe Objective accounting is not exactly one call")
    if record.get("objective_configuration") != {
        "max_evals": 1,
        "max_time_seconds": None,
        "save": list(OBJECTIVE_SAVE_FIELDS),
        "save_params_history": False,
        "save_batched_params_history": False,
        "save_time_steps": True,
    }:
        raise RuntimeError("probe Objective configuration drifted")
    result = record.get("result", {})
    call_wall_seconds = float(result.get("call_wall_seconds", 0))
    gradient_finite_fraction = float(
        result.get("gradient_finite_fraction", -1)
    )
    if (
        not math.isfinite(call_wall_seconds)
        or call_wall_seconds <= 0
        or not 0.0 <= gradient_finite_fraction <= 1.0
    ):
        raise RuntimeError("probe result diagnostics are invalid")

    process = record.get("worker_process", {})
    if (
        process.get("returncode") != 0
        or process.get("timed_out") is not False
        or float(process.get("full_wall_seconds", 0)) <= 0
    ):
        raise RuntimeError("worker process did not exit cleanly")
    monitor = record.get("gpu_monitor", {})
    if monitor.get("timed_out") is not False:
        raise RuntimeError("completed worker was marked timed out")


def _write_indexes(
    output_dir: Path,
    plan: dict[str, object],
) -> list[dict[str, object]]:
    records = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_text(
        output_dir / "runs.jsonl",
        "".join(
            json.dumps(strict_json(record), sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
    )
    atomic_json(output_dir / "summary.json", summarize_probe_records(plan, records))
    return records


def orchestrate_probe(plan: dict[str, object], output_dir: Path) -> int:
    """Run a new fail-closed candidate-isolation study."""
    if output_dir.exists():
        raise FileExistsError(f"probe already exists at {output_dir}")
    manifest_path = output_dir / "manifest.json"
    with _study_lock(output_dir):
        unexpected = [
            path for path in output_dir.iterdir() if path.name != ".study.lock"
        ]
        if unexpected:
            raise RuntimeError("refusing a nonempty probe directory")
        revision = _git("rev-parse", "HEAD")
        if _git("status", "--porcelain"):
            raise RuntimeError("refusing to run a GPU probe from a dirty tree")
        for name in ("runs", "configs", "logs", "gpu", "milestones", "telemetry"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        configuration = plan["configuration"]
        initial_idle = collect_idle_report(configuration)
        atomic_json(output_dir / "gpu" / "initial-idle.json", initial_idle)
        if not initial_idle["passed"]:
            raise RuntimeError(
                "GPU failed the initial idle gate; inspect gpu/initial-idle.json"
            )
        environment = _preflight_environment(output_dir)
        if environment["backend"] == "cpu":
            raise RuntimeError("candidate isolation requires a GPU accelerator")
        post_environment_idle = wait_for_idle(configuration)
        require_gpu_identity(post_environment_idle, str(initial_idle["gpu_uuid"]))
        atomic_json(
            output_dir / "gpu" / "post-environment-idle.json",
            post_environment_idle,
        )
        if not post_environment_idle["passed"]:
            raise RuntimeError("GPU did not return to idle after environment preflight")

        manifest = {
            **plan,
            "project_revision": revision,
            "working_tree_dirty": False,
            "semantic_prior_canonical_sha256": canonical_text_sha256(
                ROOT / "submission" / "semantic_prior.json"
            ),
            "upstream_reference": UPSTREAM_REFERENCE,
            "environment": environment,
            "jax_environment": jax_environment_policy(),
            "initial_idle_preflight": initial_idle,
            "post_environment_idle": post_environment_idle,
        }
        atomic_json(manifest_path, manifest)

        failures = 0
        for run in plan["runs"]:
            config = {
                **configuration,
                **run,
                "expected_gpu_uuid": str(initial_idle["gpu_uuid"]),
                "expected_gpu_name": str(initial_idle["samples"][0]["gpu_name"]),
            }
            run_id = str(run["run_id"])
            output_path = output_dir / "runs" / f"{run_id}.json"
            config_path = output_dir / "configs" / f"{run_id}.json"
            milestone_path = output_dir / "milestones" / f"{run_id}.json"
            stdout_path = output_dir / "logs" / f"{run_id}.stdout.log"
            stderr_path = output_dir / "logs" / f"{run_id}.stderr.log"
            atomic_json(config_path, config)

            idle_report = collect_idle_report(configuration)
            require_gpu_identity(idle_report, str(initial_idle["gpu_uuid"]))
            atomic_json(output_dir / "gpu" / f"{run_id}.idle.json", idle_report)
            if not idle_report["passed"]:
                record = _error_record(
                    config,
                    environment,
                    "GPUNotIdle",
                    "device exceeded the predeclared idle threshold",
                )
                record["idle_preflight"] = idle_report
                atomic_json(output_path, record)
                failures += 1
                _write_indexes(output_dir, plan)
                break

            try:
                completed, monitor = run_monitored_worker(
                    config_path=config_path,
                    output_path=output_path,
                    milestone_path=milestone_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    configuration=configuration,
                    expected_gpu_uuid=str(initial_idle["gpu_uuid"]),
                )
            except Exception as error:
                record = _error_record(
                    config,
                    environment,
                    "WorkerLaunchError",
                    f"{type(error).__name__}: {error}",
                )
                record["idle_preflight"] = idle_report
                record["integrity_validated"] = False
                atomic_json(output_path, record)
                failures += 1
                _write_indexes(output_dir, plan)
                break
            telemetry_path = output_dir / "telemetry" / f"{run_id}.jsonl"
            atomic_text(
                telemetry_path,
                "".join(
                    json.dumps(strict_json(sample), sort_keys=True) + "\n"
                    for sample in monitor.pop("samples")
                ),
            )
            monitor["telemetry"] = {
                "path": f"telemetry/{telemetry_path.name}",
                "sha256": sha256(telemetry_path),
            }
            atomic_json(output_dir / "gpu" / f"{run_id}.monitor.json", monitor)
            postflight_idle = wait_for_idle(configuration)
            require_gpu_identity(postflight_idle, str(initial_idle["gpu_uuid"]))
            atomic_json(
                output_dir / "gpu" / f"{run_id}.postflight-idle.json",
                postflight_idle,
            )

            if output_path.is_file():
                record = json.loads(output_path.read_text(encoding="utf-8"))
            elif monitor["contaminated"]:
                record = _error_record(
                    config,
                    environment,
                    "GPUContamination",
                    "an external compute process appeared during the worker",
                )
                record["status"] = "contaminated"
            elif monitor["timed_out"]:
                record = _error_record(
                    config,
                    environment,
                    "WorkerTimeout",
                    f"worker exceeded {configuration['worker_timeout_seconds']} seconds",
                )
            elif monitor["monitoring_error"]:
                record = _error_record(
                    config,
                    environment,
                    "GPUMonitorError",
                    str(monitor["monitoring_error"]),
                )
            else:
                record = _error_record(
                    config,
                    environment,
                    "WorkerProcessError",
                    f"worker exited {completed.returncode} without a record",
                )
            if monitor["timed_out"]:
                record["status"] = "error"
                record["error"] = {
                    "type": "WorkerTimeout",
                    "message": (
                        f"worker exceeded "
                        f"{configuration['worker_timeout_seconds']} seconds"
                    ),
                }
            elif monitor["monitoring_error"]:
                record["status"] = "error"
                record["error"] = {
                    "type": "GPUMonitorError",
                    "message": str(monitor["monitoring_error"]),
                }
            if monitor["contaminated"] or not postflight_idle["passed"]:
                record["status"] = "contaminated"
                record["error"] = {
                    "type": "GPUContamination",
                    "message": (
                        "external compute activity was observed during or after "
                        "the worker"
                    ),
                }
            record["idle_preflight"] = idle_report
            record["postflight_idle"] = postflight_idle
            record["gpu_monitor"] = monitor
            record["worker_milestone"] = (
                {
                    "path": f"milestones/{milestone_path.name}",
                    "sha256": sha256(milestone_path),
                    "payload": json.loads(milestone_path.read_text(encoding="utf-8")),
                }
                if milestone_path.is_file()
                else None
            )
            if record["worker_milestone"] is not None:
                milestone_payload = record["worker_milestone"]["payload"]
                record.setdefault("candidate", milestone_payload.get("candidate"))
                record.setdefault("problem", milestone_payload.get("problem"))
            record["worker_process"] = {
                "pid": monitor["worker_pid"],
                "returncode": completed.returncode,
                "timed_out": monitor["timed_out"],
                "full_wall_seconds": monitor["full_wall_seconds"],
                "stdout": {
                    "path": f"logs/{stdout_path.name}",
                    "sha256": sha256(stdout_path),
                },
                "stderr": {
                    "path": f"logs/{stderr_path.name}",
                    "sha256": sha256(stderr_path),
                },
            }
            record["candidate_identity_validated"] = False
            record["provenance_validated"] = False
            record["integrity_validated"] = False
            try:
                validate_candidate_identity_record(
                    record, config, environment, output_dir
                )
                record["candidate_identity_validated"] = True
                validate_probe_provenance(record, config, environment, output_dir)
                record["provenance_validated"] = True
            except Exception as error:
                record["validation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            if record.get("status") == "complete":
                try:
                    validate_completed_probe_record(
                        record, config, environment, output_dir
                    )
                    record["integrity_validated"] = True
                except Exception as error:
                    record["status"] = "error"
                    record["error"] = {
                        "type": "ArtifactValidationError",
                        "message": str(error),
                    }
            atomic_json(output_path, record)
            _write_indexes(output_dir, plan)
            if record.get("status") != "complete":
                failures += 1
                if not (
                    record["candidate_identity_validated"]
                    and record["provenance_validated"]
                    and record.get("status") == "error"
                    and record.get("error", {}).get("type") == "WorkerTimeout"
                    and record["worker_process"].get("timed_out") is True
                    and record["gpu_monitor"].get("timed_out") is True
                ):
                    break

        _write_indexes(output_dir, plan)
        return 1 if failures else 0


def run_worker(config_path: Path, output_path: Path, milestone_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        atomic_json(output_path, execute_probe(config, milestone_path))
        return 0
    except Exception as error:
        atomic_json(
            output_path,
            _error_record(
                config,
                _safe_environment_fingerprint(),
                type(error).__name__,
                str(error),
            ),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology")
    parser.add_argument("--optimizer-seed", type=int, default=7)
    parser.add_argument("--n-frequencies", type=int, default=50)
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--idle-samples", type=int, default=5)
    parser.add_argument("--idle-sample-interval", type=float, default=0.5)
    parser.add_argument("--max-idle-memory-mib", type=int)
    parser.add_argument("--max-idle-utilization", type=int)
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--cooldown-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-milestone", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker_paths = (args.worker_config, args.worker_output, args.worker_milestone)
    if any(path is not None for path in worker_paths):
        if any(path is None for path in worker_paths):
            raise SystemExit("worker mode requires all three worker paths")
        return run_worker(
            args.worker_config, args.worker_output, args.worker_milestone
        )

    if args.topology is None:
        raise SystemExit("provide an explicit --topology")
    if args.max_idle_memory_mib is None or args.max_idle_utilization is None:
        raise SystemExit(
            "provide explicit --max-idle-memory-mib and --max-idle-utilization"
        )
    if args.output is None and not args.dry_run:
        raise SystemExit("provide --output unless --dry-run is used")
    topology = {"kind": "explicit", "value": args.topology}
    plan = build_probe_plan(
        topology=topology,
        optimizer_seed=args.optimizer_seed,
        population_size=2,
        roles=list(CANDIDATE_ROLES),
        repeats=2,
        n_frequencies=args.n_frequencies,
        worker_timeout_seconds=args.worker_timeout,
        gpu_index=args.gpu_index,
        idle_samples=args.idle_samples,
        idle_sample_interval_seconds=args.idle_sample_interval,
        max_idle_memory_mib=args.max_idle_memory_mib,
        max_idle_utilization_percent=args.max_idle_utilization,
        monitor_interval_seconds=args.monitor_interval,
        cooldown_timeout_seconds=args.cooldown_timeout,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return orchestrate_probe(plan, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
