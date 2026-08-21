"""Orchestrate isolated paired UIFO runs and persist resumable artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from experiments.uifo_paired.analysis import summarize_records
from experiments.uifo_paired.metrics import flatten_histories, summarize_rows
from experiments.uifo_paired.optimizer_telemetry import (
    OPTIMIZER_TELEMETRY_MODE,
    OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
    OPTIMIZER_TELEMETRY_SCHEMA,
    OptimizerTelemetryCapture,
    summarize_optimizer_telemetry,
    validate_optimizer_telemetry,
)
from experiments.uifo_paired.optimizer_settings import (
    BATCHED_SETTINGS,
    validate_batched_settings,
)
from experiments.uifo_paired.plan import VALID_ARMS, build_plan
from experiments.uifo_paired.restart_analysis import summarize_restart_records
from experiments.uifo_paired.study_profiles import profile_names

ROOT = Path(__file__).parents[2]
UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"
OFFICIAL_DATASET_SHA256 = (
    "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
)
HISTORY_SCHEMA = {
    "call_index": {"dtype": "int32", "unit": None},
    "candidate_index": {"dtype": "int16", "unit": None},
    "eval_count_after_call": {"dtype": "int64", "unit": "evaluations"},
    "time_seconds": {"dtype": "float64", "unit": "seconds"},
    "loss": {"dtype": "float64", "unit": "competition loss"},
    "sensitivity_loss": {"dtype": "float64", "unit": "loss"},
    "penalty": {"dtype": "float64", "unit": "loss"},
    "is_feasible": {"dtype": "bool", "unit": None},
    "initial_params_unbounded": {"dtype": "runtime", "unit": "active space"},
}
TIME_GRID = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400]
EVAL_GRID = [1, 8, 32, 128, 512, 2048, 8192, 32768, 131072]
JAX_RUNTIME_ENVIRONMENT_KEYS = (
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_CACHE_PATH",
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_EXPECT_PGLE",
    "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_COMPILATION_CACHE_MAX_SIZE",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
    "LD_LIBRARY_PATH",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def strict_json(value):
    """Convert non-finite floats and array-like scalars to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        return strict_json(value.item())
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_fingerprint() -> dict[str, object]:
    """Return the runtime/device identity that must remain fixed within a study."""
    nvidia_smi = _nvidia_smi_snapshot(include_dynamic=False)
    import jax

    devices = jax.devices()
    device_kinds = [
        str(getattr(device, "device_kind", "unknown")) for device in devices
    ]
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kinds": device_kinds,
        "device_platforms": [str(device.platform) for device in devices],
        "devices": [str(device) for device in devices],
        "competition_aligned_a100": any(
            "A100" in kind.upper() for kind in device_kinds
        ),
        "jax_runtime_environment": {
            name: os.environ.get(name) for name in JAX_RUNTIME_ENVIRONMENT_KEYS
        },
        "jax_runtime_configuration": {
            "compilation_cache_dir": jax.config.jax_compilation_cache_dir,
            "enable_compilation_cache": bool(jax.config.jax_enable_compilation_cache),
        },
        "nvidia_smi": nvidia_smi,
        "versions": {
            "dfbench": _package_version("dfbench"),
            "differometor": _package_version("differometor"),
            "jax": _package_version("jax"),
            "jax-cuda12-pjrt": _package_version("jax-cuda12-pjrt"),
            "jax-cuda12-plugin": _package_version("jax-cuda12-plugin"),
            "jaxlib": _package_version("jaxlib"),
            "optax": _package_version("optax"),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def cache_disabled_jax_environment() -> dict[str, str]:
    """Return a child environment that cannot reuse persistent JAX executables."""
    environment = os.environ.copy()
    xla_flags = environment.get("XLA_FLAGS", "")
    if "cache" in xla_flags.lower():
        raise RuntimeError(
            "scored studies reject cache-related XLA_FLAGS; unset XLA_FLAGS"
        )
    for name in tuple(environment):
        if "CACHE" in name.upper() and name.startswith(("CUDA_", "JAX_")):
            environment.pop(name, None)
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["CUDA_CACHE_DISABLE"] = "1"
    _validate_cache_disabled_environment(environment)
    return environment


def _jax_runtime_environment_policy(
    environment: dict[str, str] | None = None,
) -> dict[str, str | None]:
    source = os.environ if environment is None else environment
    return {name: source.get(name) for name in JAX_RUNTIME_ENVIRONMENT_KEYS}


def _validate_cache_disabled_environment(
    environment: dict[str, str | None],
) -> None:
    if str(environment.get("JAX_ENABLE_COMPILATION_CACHE") or "").lower() != "false":
        raise RuntimeError("scored studies require the persistent JAX cache disabled")
    if environment.get("JAX_COMPILATION_CACHE_DIR") is not None:
        raise RuntimeError("scored studies cannot set JAX_COMPILATION_CACHE_DIR")
    if environment.get("CUDA_CACHE_DISABLE") != "1":
        raise RuntimeError("scored studies require the CUDA driver cache disabled")
    allowed_cache_settings = {
        "CUDA_CACHE_DISABLE",
        "JAX_ENABLE_COMPILATION_CACHE",
    }
    unexpected_cache_settings = sorted(
        name
        for name, value in environment.items()
        if value is not None
        and "CACHE" in name.upper()
        and name.startswith(("CUDA_", "JAX_"))
        and name not in allowed_cache_settings
    )
    if unexpected_cache_settings:
        raise RuntimeError(
            "scored studies reject cache-related environment settings: "
            + ", ".join(unexpected_cache_settings)
        )
    if "cache" in str(environment.get("XLA_FLAGS") or "").lower():
        raise RuntimeError("scored studies reject cache-related XLA_FLAGS")


def _validate_cache_disabled_runtime(runtime_environment: dict[str, object]) -> None:
    configuration = runtime_environment.get("jax_runtime_configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("runtime is missing effective JAX cache configuration")
    if configuration.get("enable_compilation_cache") is not False:
        raise RuntimeError("effective JAX compilation cache is not disabled")
    if configuration.get("compilation_cache_dir") is not None:
        raise RuntimeError("effective JAX compilation cache directory is set")


def _nvidia_smi_snapshot(*, include_dynamic: bool) -> dict[str, object]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "mig.mode.current",
    ]
    if include_dynamic:
        fields.extend(["memory.used", "utilization.gpu"])
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    if completed.returncode != 0:
        return {
            "status": "error",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }

    rows = []
    for values in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(fields):
            return {
                "status": "error",
                "message": "unexpected nvidia-smi field count",
            }
        row = dict(zip(fields, (value.strip() for value in values), strict=True))
        try:
            parsed = {
                "index": int(row["index"]),
                "uuid": row["uuid"],
                "name": row["name"],
                "driver_version": row["driver_version"],
                "memory_total_mib": int(row["memory.total"]),
                "mig_mode_current": row["mig.mode.current"],
            }
            if include_dynamic:
                parsed.update(
                    {
                        "memory_used_mib": int(row["memory.used"]),
                        "utilization_percent": int(row["utilization.gpu"]),
                    }
                )
        except (KeyError, ValueError) as error:
            return {
                "status": "error",
                "message": f"could not parse nvidia-smi output: {error}",
            }
        rows.append(parsed)
    return {"status": "ok", "gpus": rows}


def _rental_preflight(
    output_dir: Path, configuration: dict[str, object]
) -> dict[str, object]:
    disk = shutil.disk_usage(output_dir)
    result: dict[str, object] = {
        "disk": {
            "path": str(output_dir.resolve()),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
        "gpu_idle": _nvidia_smi_snapshot(include_dynamic=True),
    }
    minimum_disk = configuration.get("minimum_free_disk_gib")
    if minimum_disk is not None and disk.free < float(minimum_disk) * 1024**3:
        raise RuntimeError(
            f"rental preflight requires at least {minimum_disk} GiB free at "
            f"{output_dir}; found {disk.free / 1024**3:.2f} GiB"
        )
    if not bool(configuration.get("require_a100")):
        return result

    snapshot = result["gpu_idle"]
    if not isinstance(snapshot, dict) or snapshot.get("status") != "ok":
        raise RuntimeError("rental preflight could not query nvidia-smi")
    gpus = snapshot.get("gpus", [])
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise RuntimeError("rental preflight requires exactly one visible GPU")
    gpu = gpus[0]
    if "A100" not in str(gpu.get("name", "")).upper():
        raise RuntimeError("rental preflight requires an NVIDIA A100")
    if str(gpu.get("mig_mode_current", "")).lower() != "disabled":
        raise RuntimeError("rental preflight requires MIG mode disabled")
    minimum_memory = configuration.get("minimum_gpu_memory_mib")
    if minimum_memory is not None and int(gpu["memory_total_mib"]) < int(
        minimum_memory
    ):
        raise RuntimeError(
            f"rental preflight requires at least {minimum_memory} MiB GPU memory"
        )
    max_memory = configuration.get("max_idle_gpu_memory_mib")
    if max_memory is not None and int(gpu["memory_used_mib"]) > int(max_memory):
        raise RuntimeError(
            f"idle GPU memory {gpu['memory_used_mib']} MiB exceeds {max_memory} MiB"
        )
    max_utilization = configuration.get("max_idle_gpu_utilization_percent")
    if max_utilization is not None and int(gpu["utilization_percent"]) > int(
        max_utilization
    ):
        raise RuntimeError(
            "idle GPU utilization "
            f"{gpu['utilization_percent']}% exceeds {max_utilization}%"
        )
    return result


def _validate_required_a100(runtime_environment: dict[str, object]) -> None:
    if (
        runtime_environment.get("backend") != "gpu"
        or runtime_environment.get("device_count") != 1
        or not runtime_environment.get("competition_aligned_a100")
    ):
        raise RuntimeError("this study requires exactly one JAX-visible NVIDIA A100")


def run_preflight(output_path: Path) -> int:
    atomic_json(output_path, environment_fingerprint())
    return 0


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _recover_stale_study_lock(output_dir: Path, lock_path: Path) -> None:
    try:
        record = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(record["pid"])
        hostname = str(record["hostname"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("study lock is malformed; refusing recovery") from error
    if hostname != platform.node():
        raise RuntimeError("study lock belongs to a different host; refusing recovery")
    if _pid_is_alive(pid):
        raise RuntimeError(f"study lock owner process {pid} is still alive")
    recovery_dir = output_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()[:12]
    recovered = recovery_dir / f"stale-study-lock-{digest}.json"
    if recovered.exists():
        raise RuntimeError(f"stale lock recovery artifact already exists: {recovered}")
    os.replace(lock_path, recovered)


@contextmanager
def _study_lock(output_dir: Path, *, recover_stale: bool = False):
    """Prevent concurrent writers from mutating the same study directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".study.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        if not recover_stale:
            raise RuntimeError(f"study is already locked: {lock_path}") from error
        _recover_stale_study_lock(output_dir, lock_path)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as retry_error:
            raise RuntimeError(
                f"study was relocked during recovery: {lock_path}"
            ) from retry_error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": platform.node(),
                        "created_utc": datetime.now(UTC).isoformat(),
                    }
                )
                + "\n"
            )
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class _FirstEvaluationCapture:
    """Retain the immutable first evaluated parameter array without syncing it."""

    def __init__(self) -> None:
        self.params = None

    def install(self, objective) -> None:
        original_single = objective.value_and_grad
        original_single_aux = objective.value_and_grad_aux
        original_batch = objective.vmap_value_and_grad_aux

        def single(params):
            if self.params is None:
                self.params = params
            return original_single(params)

        def batch(params):
            if self.params is None:
                self.params = params
            return original_batch(params)

        def single_aux(params):
            if self.params is None:
                self.params = params
            return original_single_aux(params)

        objective.value_and_grad = single
        objective.value_and_grad_aux = single_aux
        objective.vmap_value_and_grad_aux = batch

    def capture_population(self, params) -> None:
        if self.params is None:
            self.params = params


def execute_run(
    config: dict[str, object],
    history_path: Path,
    runtime_environment: dict[str, object] | None = None,
    optimizer_telemetry_path: Path | None = None,
) -> dict[str, object]:
    """Execute exactly one topology, seed, and arm in the worker process."""
    import jax
    import numpy as np

    backend = jax.default_backend()
    if backend == "cpu" and not bool(config["allow_cpu"]):
        raise RuntimeError(
            "UIFO evaluation requires an accelerator; pass --allow-cpu only "
            "for an explicitly non-representative mechanics run"
        )

    from dfbench import Objective
    from dfbench.problems import UIFOProblem

    from experiments.uifo_paired.baselines import SingleStartAdam
    from submission.submission import BatchedRestartAdam

    topology = config["topology"]
    problem_kwargs = {
        "size": 3,
        "n_frequencies": int(config["n_frequencies"]),
    }
    if topology["kind"] == "seed":
        problem_kwargs["topology_seed"] = int(topology["value"])
    else:
        problem_kwargs["topology"] = str(topology["value"])

    problem = UIFOProblem(**problem_kwargs)
    objective = Objective(
        problem,
        max_time=config["max_time_seconds"],
        max_evals=config["max_evals"],
        save_time_steps=True,
        save_params_history=False,
        save_batched_params_history=False,
        save=[
            "eval_type",
            "batched_loss",
            "batched_sensitivity_loss",
            "batched_penalty",
            "batched_is_feasible",
        ],
        verbose=0,
    )
    capture = _FirstEvaluationCapture()
    capture.install(objective)
    telemetry_mode = config.get("optimizer_telemetry")
    if telemetry_mode not in (None, OPTIMIZER_TELEMETRY_MODE):
        raise RuntimeError("unsupported optimizer telemetry mode")
    if (telemetry_mode is None) != (optimizer_telemetry_path is None):
        raise RuntimeError("optimizer telemetry configuration/path mismatch")
    telemetry_capture = (
        OptimizerTelemetryCapture() if telemetry_mode is not None else None
    )

    arm = str(config["arm"])
    optimizer_seed = int(config["optimizer_seed"])
    if arm == "adam" and telemetry_capture is not None:
        raise RuntimeError("optimizer telemetry is unsupported for the Adam arm")
    started = time.perf_counter()
    if arm == "adam":
        SingleStartAdam().optimize(objective, random_seed=optimizer_seed)
        algorithm_settings = {
            "module": "experiments.uifo_paired.baselines",
            "class": "SingleStartAdam",
            "algorithm_str": "paired_single_start_adam",
            "kwargs": {
                "learning_rate": 0.1,
                "patience": None,
                "random_seed": optimizer_seed,
            },
        }
    else:
        algorithm = BatchedRestartAdam()
        optimizer_settings = validate_batched_settings(
            config.get("optimizer_settings", BATCHED_SETTINGS)
        )
        if arm == "semantic_prior" and algorithm._semantic_prior(objective) is None:
            raise RuntimeError("semantic-prior arm could not load a valid prior")
        algorithm.optimize(
            objective,
            random_seed=optimizer_seed,
            population_size=int(config["population_size"]),
            use_semantic_prior=arm == "semantic_prior",
            evaluation_chunk_size=config.get("evaluation_chunk_size"),
            initial_population_callback=capture.capture_population,
            optimizer_telemetry_callback=telemetry_capture,
            **optimizer_settings,
        )
        algorithm_settings = {
            "module": "submission.submission",
            "class": "BatchedRestartAdam",
            "algorithm_str": "batched_restart_adam",
            "kwargs": {
                **optimizer_settings,
                "population_size": int(config["population_size"]),
                "random_seed": optimizer_seed,
                "use_semantic_prior": arm == "semantic_prior",
                "evaluation_chunk_size": config.get("evaluation_chunk_size"),
            },
        }
    host_duration = time.perf_counter() - started
    objective_elapsed_snapshot = float(objective.time_elapsed)
    objective_summary = objective.get_summary()
    last_admitted_time = (
        float(objective.time_steps[-1]) if objective.time_steps else None
    )

    loss_history = [_host_list(value) for value in objective.loss_history]
    feasible_history = [_host_list(value) for value in objective.is_feasible_history]
    sensitivity_history = [
        _host_list(value) for value in objective.sensitivity_loss_history
    ]
    penalty_history = [_host_list(value) for value in objective.penalty_history]
    rows = flatten_histories(
        loss_history,
        feasible_history,
        objective.time_steps,
        sensitivity_history,
        penalty_history,
    )
    if not rows:
        raise RuntimeError("run completed without an admitted candidate history")
    if len(rows) > objective.eval_count:
        raise RuntimeError("logged candidate count exceeds Objective eval_count")
    if capture.params is None:
        raise RuntimeError("run completed without capturing its first evaluation")
    initial_params = np.asarray(jax.device_get(capture.params))
    initial_hashes = _parameter_hashes(initial_params)

    time_grid, eval_grid = _metric_grids(config)
    metrics = summarize_rows(
        rows,
        list(config["target_losses"]),
        time_grid=time_grid,
        eval_grid=eval_grid,
    )
    if int(metrics["logged_calls"]) != objective.log_call_count:
        raise RuntimeError("logged call count disagrees with Objective accounting")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_history = history_path.with_suffix(".tmp.npz")
    with temporary_history.open("wb") as handle:
        np.savez_compressed(
            handle,
            call_index=np.asarray([row["call_index"] for row in rows], dtype=np.int32),
            candidate_index=np.asarray(
                [row["candidate_index"] for row in rows], dtype=np.int16
            ),
            eval_count_after_call=np.asarray(
                [row["eval_count_after_call"] for row in rows], dtype=np.int64
            ),
            time_seconds=np.asarray(
                [row["time_seconds"] for row in rows], dtype=np.float64
            ),
            loss=np.asarray(
                [float("nan") if row["loss"] is None else row["loss"] for row in rows],
                dtype=np.float64,
            ),
            sensitivity_loss=np.asarray(
                [
                    float("nan")
                    if row["sensitivity_loss"] is None
                    else row["sensitivity_loss"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            penalty=np.asarray(
                [
                    float("nan") if row["penalty"] is None else row["penalty"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            is_feasible=np.asarray(
                [row["is_feasible"] for row in rows], dtype=np.bool_
            ),
            initial_params_unbounded=initial_params,
        )
    os.replace(temporary_history, history_path)
    validate_history_artifact(history_path, expected_rows=len(rows))

    telemetry_record = None
    if telemetry_capture is not None:
        assert optimizer_telemetry_path is not None
        telemetry_summary = telemetry_capture.write(optimizer_telemetry_path)
        telemetry_record = {
            "format_version": 1,
            "mode": telemetry_mode,
            "path": (
                f"{optimizer_telemetry_path.parent.name}/"
                f"{optimizer_telemetry_path.name}"
            ),
            "rows": telemetry_capture.rows,
            "schema": OPTIMIZER_TELEMETRY_SCHEMA,
            "metadata_schema": OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
            "sha256": sha256(optimizer_telemetry_path),
            "summary": telemetry_summary,
        }

    problem_spec = objective.problem_spec
    topology_string = str(problem.topology_string)
    result = {
        "status": "complete",
        "completed_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "environment": runtime_environment or environment_fingerprint(),
        "problem": {
            "n_params": objective.n_params,
            "spec": problem_spec,
            "topology_sha256": hashlib.sha256(topology_string.encode()).hexdigest(),
            "topology_string": topology_string,
        },
        "objective_summary": objective_summary,
        "algorithm": algorithm_settings,
        "objective_configuration": {
            "max_evals": config["max_evals"],
            "max_time_seconds": config["max_time_seconds"],
            "save": [
                "eval_type",
                "batched_loss",
                "batched_sensitivity_loss",
                "batched_penalty",
                "batched_is_feasible",
            ],
            "save_batched_params_history": False,
            "save_params_history": False,
            "save_time_steps": True,
        },
        "objective_accounting": {
            "eval_count": objective.eval_count,
            "log_call_count": objective.log_call_count,
            "eval_type_counts": objective.eval_type_counts,
        },
        "objective_time_elapsed_snapshot": objective_elapsed_snapshot,
        "host_optimize_duration_seconds": host_duration,
        "last_admitted_time_seconds": last_admitted_time,
        "initial_population_roles": _initial_population_roles(
            arm, int(config["population_size"])
        ),
        "initial_parameter_hashes": initial_hashes,
        "metrics": metrics,
        "history": {
            "format_version": 1,
            "path": f"{history_path.parent.name}/{history_path.name}",
            "rows": len(rows),
            "schema": HISTORY_SCHEMA,
            "sha256": sha256(history_path),
        },
    }
    if telemetry_record is not None:
        result["optimizer_telemetry"] = telemetry_record
    return result


def _host_list(value):
    import jax
    import numpy as np

    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _initial_population_roles(arm: str, population_size: int) -> list[str]:
    if arm == "adam":
        return ["random"]
    roles = ["anchor"]
    if arm == "semantic_prior":
        roles.append("semantic_prior")
    return roles + ["random"] * (population_size - len(roles))


def _parameter_hashes(params) -> list[str]:
    import numpy as np

    array = np.asarray(params)
    members = array[None, :] if array.ndim == 1 else array
    result = []
    for member in members:
        contiguous = np.ascontiguousarray(member)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
        result.append(digest.hexdigest())
    return result


def validate_history_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Validate a pickle-free NPZ and return its arrays for recomputation."""
    import numpy as np

    if not path.is_file():
        raise RuntimeError(f"missing history artifact: {path}")
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RuntimeError(f"history digest mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    expected_names = {f"{name}.npy" for name in HISTORY_SCHEMA}
    if names != expected_names:
        raise RuntimeError(f"history schema mismatch: {sorted(names)}")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in HISTORY_SCHEMA}
    row_fields = [name for name in HISTORY_SCHEMA if name != "initial_params_unbounded"]
    expected_dtypes = {
        "call_index": np.dtype("int32"),
        "candidate_index": np.dtype("int16"),
        "eval_count_after_call": np.dtype("int64"),
        "time_seconds": np.dtype("float64"),
        "loss": np.dtype("float64"),
        "sensitivity_loss": np.dtype("float64"),
        "penalty": np.dtype("float64"),
        "is_feasible": np.dtype("bool"),
    }
    for name in row_fields:
        if arrays[name].ndim != 1 or arrays[name].dtype != expected_dtypes[name]:
            raise RuntimeError(f"history field has invalid shape/dtype: {name}")
    lengths = {int(arrays[name].shape[0]) for name in row_fields}
    if len(lengths) != 1:
        raise RuntimeError("history row arrays have inconsistent lengths")
    rows = next(iter(lengths))
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError("history row count mismatch")
    if rows < 1:
        raise RuntimeError("history artifact contains no candidate rows")
    if arrays["initial_params_unbounded"].ndim not in (1, 2):
        raise RuntimeError("initial parameter history must be a vector or matrix")
    return arrays


def _rows_from_history_arrays(arrays) -> list[dict[str, object]]:
    import numpy as np

    rows = []
    for index in range(len(arrays["loss"])):
        rows.append(
            {
                "call_index": int(arrays["call_index"][index]),
                "candidate_index": int(arrays["candidate_index"][index]),
                "eval_count_after_call": int(arrays["eval_count_after_call"][index]),
                "time_seconds": float(arrays["time_seconds"][index]),
                "loss": (
                    None
                    if not np.isfinite(arrays["loss"][index])
                    else float(arrays["loss"][index])
                ),
                "sensitivity_loss": (
                    None
                    if not np.isfinite(arrays["sensitivity_loss"][index])
                    else float(arrays["sensitivity_loss"][index])
                ),
                "penalty": (
                    None
                    if not np.isfinite(arrays["penalty"][index])
                    else float(arrays["penalty"][index])
                ),
                "is_feasible": bool(arrays["is_feasible"][index]),
            }
        )
    return rows


def _validate_optimizer_telemetry_against_history(
    telemetry_arrays: dict[str, object],
    history_arrays: dict[str, object],
    *,
    minimum_improvement: float,
) -> None:
    """Bind optimizer observations to the authenticated candidate history."""
    import numpy as np

    losses = np.asarray(history_arrays["loss"])
    feasible = np.asarray(history_arrays["is_feasible"])
    finite = np.isfinite(losses)
    loss_float_bits = np.unique(telemetry_arrays["loss_float_bits"])
    if len(loss_float_bits) != 1 or int(loss_float_bits[0]) not in (32, 64):
        raise RuntimeError("optimizer telemetry loss precision is invalid")
    runtime_float = np.float32 if int(loss_float_bits[0]) == 32 else np.float64
    runtime_losses = losses.astype(runtime_float)
    runtime_improvement = runtime_float(minimum_improvement)
    if len(telemetry_arrays["batch_index"]) != len(losses):
        raise RuntimeError("optimizer telemetry/history row count mismatch")
    if not np.array_equal(telemetry_arrays["finite_loss"], finite):
        raise RuntimeError("optimizer telemetry finite flags mismatch history")
    if not np.array_equal(telemetry_arrays["feasible"], feasible):
        raise RuntimeError("optimizer telemetry feasibility mismatch history")

    expected_improved = np.zeros(len(losses), dtype=bool)
    expected_best = np.full(len(losses), np.inf, dtype=np.float64)
    member_best: dict[tuple[int, int], object] = {}
    for row_index in range(len(losses)):
        key = (
            int(telemetry_arrays["member_index"][row_index]),
            int(telemetry_arrays["evaluated_generation"][row_index]),
        )
        previous_best = member_best.get(key, runtime_float(np.inf))
        improved = bool(
            finite[row_index]
            and runtime_losses[row_index]
            < runtime_float(previous_best - runtime_improvement)
        )
        observed_best = (
            runtime_losses[row_index] if improved else previous_best
        )
        member_best[key] = observed_best
        expected_improved[row_index] = improved
        expected_best[row_index] = float(observed_best)
    if not np.array_equal(
        telemetry_arrays["observed_member_improved"], expected_improved
    ):
        raise RuntimeError("optimizer telemetry member improvements mismatch history")
    if not np.array_equal(
        telemetry_arrays["observed_member_best_loss"], expected_best
    ):
        raise RuntimeError("optimizer telemetry member bests mismatch history")

    expected_global_source = np.zeros(len(losses), dtype=bool)
    global_best = math.inf
    batch = np.asarray(telemetry_arrays["batch_index"])
    for batch_index in np.unique(batch):
        row_indices = np.flatnonzero(batch == batch_index)
        feasible_losses = np.where(
            finite[row_indices] & feasible[row_indices],
            losses[row_indices],
            np.inf,
        )
        source_offset = int(np.argmin(feasible_losses))
        batch_best = float(feasible_losses[source_offset])
        if batch_best < global_best:
            expected_global_source[row_indices[source_offset]] = True
            global_best = batch_best
    if not np.array_equal(
        telemetry_arrays["global_feasible_improvement"],
        expected_global_source,
    ):
        raise RuntimeError(
            "optimizer telemetry global improvement source mismatch history"
        )


def _metric_grids(config: dict[str, object]) -> tuple[list[float], list[int]]:
    times = [
        value
        for value in TIME_GRID
        if config["max_time_seconds"] is None
        or value <= float(config["max_time_seconds"])
    ]
    evals = [
        value
        for value in EVAL_GRID
        if config["max_evals"] is None or value <= int(config["max_evals"])
    ]
    return times, evals


def validate_completed_record(
    record: dict[str, object],
    expected_config: dict[str, object],
    history_path: Path,
    expected_environment: dict[str, object] | None = None,
) -> None:
    if record.get("format_version") != 1 or record.get("status") != "complete":
        raise RuntimeError("resume record is not a complete format-version-1 run")
    if record.get("run_id") != expected_config["run_id"]:
        raise RuntimeError("resume run ID mismatch")
    if strict_json(record.get("config")) != strict_json(expected_config):
        raise RuntimeError("resume run configuration mismatch")
    if expected_environment is not None and strict_json(
        record.get("environment")
    ) != strict_json(expected_environment):
        raise RuntimeError("resume runtime environment mismatch")

    history = record.get("history", {})
    expected_relative = f"{history_path.parent.name}/{history_path.name}"
    if history.get("path") != expected_relative or history.get("format_version") != 1:
        raise RuntimeError("resume history reference mismatch")
    arrays = validate_history_artifact(
        history_path,
        expected_sha256=str(history.get("sha256")),
        expected_rows=int(history.get("rows")),
    )
    if strict_json(history.get("schema")) != strict_json(HISTORY_SCHEMA):
        raise RuntimeError("resume history schema metadata mismatch")

    study_dir = history_path.parent.parent
    telemetry_mode = expected_config.get("optimizer_telemetry")
    telemetry_path = (
        study_dir / "optimizer-telemetry" / f"{expected_config['run_id']}.npz"
    )
    telemetry = record.get("optimizer_telemetry")
    telemetry_arrays = None
    if telemetry_mode is None:
        if telemetry is not None or telemetry_path.exists():
            raise RuntimeError("resume record has unsolicited optimizer telemetry")
    else:
        if telemetry_mode != OPTIMIZER_TELEMETRY_MODE or not isinstance(
            telemetry, dict
        ):
            raise RuntimeError("resume optimizer telemetry metadata is missing")
        expected_telemetry_relative = (
            f"optimizer-telemetry/{expected_config['run_id']}.npz"
        )
        if (
            telemetry.get("format_version") != 1
            or telemetry.get("mode") != telemetry_mode
            or telemetry.get("path") != expected_telemetry_relative
            or strict_json(telemetry.get("schema"))
            != strict_json(OPTIMIZER_TELEMETRY_SCHEMA)
            or strict_json(telemetry.get("metadata_schema"))
            != strict_json(OPTIMIZER_TELEMETRY_METADATA_SCHEMA)
        ):
            raise RuntimeError("resume optimizer telemetry reference mismatch")
        telemetry_arrays = validate_optimizer_telemetry(
            telemetry_path,
            expected_sha256=str(telemetry.get("sha256")),
            expected_rows=int(telemetry.get("rows")),
            expected_population_size=int(expected_config["population_size"]),
            expected_patience=int(record["algorithm"]["kwargs"]["patience"]),
        )
        recorded_summary = telemetry.get("summary")
        if not isinstance(recorded_summary, dict):
            raise RuntimeError("resume optimizer telemetry summary is missing")
        recomputed_telemetry_summary = summarize_optimizer_telemetry(
            telemetry_arrays
        )
        if strict_json(recorded_summary) != strict_json(recomputed_telemetry_summary):
            raise RuntimeError("resume optimizer telemetry summary mismatch")

    time_grid, eval_grid = _metric_grids(expected_config)
    recomputed = summarize_rows(
        _rows_from_history_arrays(arrays),
        list(expected_config["target_losses"]),
        time_grid=time_grid,
        eval_grid=eval_grid,
    )
    if strict_json(record.get("metrics")) != strict_json(recomputed):
        raise RuntimeError("resume metrics do not match the history artifact")
    if record.get("initial_parameter_hashes") != _parameter_hashes(
        arrays["initial_params_unbounded"]
    ):
        raise RuntimeError("resume initial-parameter hashes do not match history")
    roles = record.get("initial_population_roles")
    hashes = record.get("initial_parameter_hashes")
    expected_members = (
        1
        if expected_config["arm"] == "adam"
        else int(expected_config["population_size"])
    )
    if not isinstance(roles, list) or not isinstance(hashes, list):
        raise RuntimeError("resume initial-population evidence is missing")
    if len(roles) != expected_members or len(hashes) != expected_members:
        raise RuntimeError("resume initial-population evidence has wrong size")
    if roles != _initial_population_roles(
        str(expected_config["arm"]), int(expected_config["population_size"])
    ):
        raise RuntimeError("resume initial-population roles are inconsistent")
    if any(not isinstance(value, str) or not value for value in hashes):
        raise RuntimeError("resume initial-parameter hash is invalid")

    problem = record.get("problem", {})
    topology_string = str(problem.get("topology_string", ""))
    if (
        not topology_string
        or problem.get("topology_sha256")
        != hashlib.sha256(topology_string.encode()).hexdigest()
    ):
        raise RuntimeError("resume topology identity evidence is invalid")
    configured_topology = expected_config.get("topology")
    if (
        isinstance(configured_topology, dict)
        and configured_topology.get("kind") == "string"
        and topology_string != configured_topology.get("value")
    ):
        raise RuntimeError("resume topology string differs from explicit plan")

    objective_configuration = record.get("objective_configuration", {})
    if objective_configuration.get("max_evals") != expected_config["max_evals"]:
        raise RuntimeError("resume Objective evaluation budget mismatch")
    if (
        objective_configuration.get("max_time_seconds")
        != expected_config["max_time_seconds"]
    ):
        raise RuntimeError("resume Objective time budget mismatch")

    algorithm = record.get("algorithm", {})
    algorithm_kwargs = algorithm.get("kwargs", {})
    if expected_config["arm"] == "adam":
        expected_algorithm = {
            "module": "experiments.uifo_paired.baselines",
            "class": "SingleStartAdam",
            "algorithm_str": "paired_single_start_adam",
            "kwargs": {
                "learning_rate": 0.1,
                "patience": None,
                "random_seed": expected_config["optimizer_seed"],
            },
        }
    else:
        expected_prior = expected_config["arm"] == "semantic_prior"
        expected_optimizer_settings = validate_batched_settings(
            expected_config.get("optimizer_settings", BATCHED_SETTINGS)
        )
        expected_algorithm = {
            "module": "submission.submission",
            "class": "BatchedRestartAdam",
            "algorithm_str": "batched_restart_adam",
            "kwargs": {
                **expected_optimizer_settings,
                "population_size": expected_config["population_size"],
                "random_seed": expected_config["optimizer_seed"],
                "use_semantic_prior": expected_prior,
                "evaluation_chunk_size": expected_config.get(
                    "evaluation_chunk_size"
                ),
            },
        }
    if strict_json(algorithm) != strict_json(expected_algorithm):
        raise RuntimeError("resume algorithm configuration mismatch")
    if telemetry_arrays is not None:
        _validate_optimizer_telemetry_against_history(
            telemetry_arrays,
            arrays,
            minimum_improvement=float(algorithm_kwargs["minimum_improvement"]),
        )

    accounting = record.get("objective_accounting", {})
    if int(accounting.get("log_call_count", -1)) != int(recomputed["logged_calls"]):
        raise RuntimeError("resume Objective log-call accounting mismatch")
    if int(accounting.get("eval_count", -1)) < int(recomputed["logged_candidates"]):
        raise RuntimeError("resume Objective evaluation accounting mismatch")

    process = record.get("worker_process")
    if not isinstance(process, dict):
        raise RuntimeError("resume worker-process evidence is missing")
    wall_seconds = process.get("full_wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0
    ):
        raise RuntimeError("resume worker wall time is invalid")
    if process.get("returncode") != 0 or process.get("timed_out") is not False:
        raise RuntimeError("resume complete record has invalid worker exit status")
    if process.get("within_official_4h30_container_limit") is not (
        wall_seconds <= 4.5 * 60 * 60
    ):
        raise RuntimeError("resume worker runtime-limit evidence is inconsistent")
    for stream in ("stdout", "stderr"):
        stream_record = process.get(stream, {})
        expected_stream_path = f"logs/{expected_config['run_id']}.{stream}.log"
        if stream_record.get("path") != expected_stream_path:
            raise RuntimeError(f"resume worker {stream} path mismatch")
        stream_path = study_dir / str(stream_record.get("path", ""))
        if not stream_path.is_file() or sha256(stream_path) != stream_record.get(
            "sha256"
        ):
            raise RuntimeError(f"resume worker {stream} evidence mismatch")


def run_worker(
    config_path: Path,
    output_path: Path,
    history_path: Path,
    optimizer_telemetry_path: Path | None = None,
) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = {
        "format_version": 1,
        "run_id": config["run_id"],
        "started_utc": datetime.now(UTC).isoformat(),
    }
    try:
        runtime_environment = environment_fingerprint()
        if config.get("jax_compilation_cache_policy") == "disabled":
            _validate_cache_disabled_runtime(runtime_environment)
        result = {
            **base,
            **execute_run(
                config,
                history_path,
                runtime_environment,
                optimizer_telemetry_path=optimizer_telemetry_path,
            ),
        }
        atomic_json(output_path, result)
        return 0
    except KeyboardInterrupt as error:
        atomic_json(
            output_path,
            {
                **base,
                "status": "interrupted",
                "completed_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 130
    except Exception as error:
        atomic_json(
            output_path,
            {
                **base,
                "status": "error",
                "completed_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        return 1


def orchestrate(
    plan: dict[str, object],
    output_dir: Path,
    resume: bool,
    subprocess_environment: dict[str, str] | None = None,
    recover_stale_lock: bool = False,
) -> int:
    study_profile = plan.get("configuration", {}).get("study_profile")
    if resume and study_profile in {"restart-mechanics-v1", "restart-screen-v1"}:
        raise RuntimeError(
            "restart study profiles are non-resumable; preserve and package the "
            "terminal partial attempt"
        )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"study already exists at {output_dir}; pass --resume to continue it"
        )
    with _study_lock(output_dir, recover_stale=recover_stale_lock):
        return _orchestrate_locked(
            plan,
            output_dir,
            resume,
            subprocess_environment=subprocess_environment,
        )


def _orchestrate_locked(
    plan: dict[str, object],
    output_dir: Path,
    resume: bool,
    subprocess_environment: dict[str, str] | None = None,
) -> int:
    session_started = time.perf_counter()
    session_started_utc = datetime.now(UTC).isoformat()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        unexpected = [
            path
            for path in output_dir.iterdir()
            if path.name not in {".study.lock", "recovery"}
        ]
        if unexpected:
            raise RuntimeError(
                "refusing to adopt a nonempty study directory without its manifest"
            )
    revision = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        raise RuntimeError("refusing to run an accelerator study from a dirty tree")

    common = plan["configuration"]
    _validate_mechanics_revision(common, revision)
    _require_provider_time(
        common,
        float(common.get("max_session_wall_seconds") or 0)
        + float(common.get("provider_evacuation_reserve_seconds") or 0),
        "start the study",
    )
    child_environment = (
        cache_disabled_jax_environment()
        if subprocess_environment is None
        else subprocess_environment
    )
    _validate_cache_disabled_environment(child_environment)
    inherited_jax_environment = _jax_runtime_environment_policy()
    effective_jax_environment = _jax_runtime_environment_policy(child_environment)
    rental_preflight = _rental_preflight(output_dir, common)
    preflight_stem = "preflight"
    if manifest_path.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        preflight_stem = f"preflight-resume-{timestamp}-{os.getpid()}"
    atomic_json(
        output_dir / f"{preflight_stem}.host-environment.json",
        {
            "captured_utc": datetime.now(UTC).isoformat(),
            "inherited_cache_environment": {
                name: value
                for name, value in sorted(os.environ.items())
                if "CACHE" in name.upper() and name.startswith(("CUDA_", "JAX_"))
            },
            "inherited_environment": inherited_jax_environment,
            "effective_environment": effective_jax_environment,
        },
    )
    runtime_environment = _preflight_environment(
        output_dir,
        subprocess_environment=child_environment,
        artifact_stem=preflight_stem,
    )
    _validate_cache_disabled_runtime(runtime_environment)
    if runtime_environment["backend"] == "cpu" and not common["allow_cpu"]:
        raise RuntimeError(
            "UIFO evaluation requires an accelerator; pass --allow-cpu only "
            "for an explicitly non-representative mechanics run"
        )
    if bool(common.get("require_a100")):
        _validate_required_a100(runtime_environment)

    prior_path = ROOT / "submission" / "semantic_prior.json"
    manifest = {
        **plan,
        "project_revision": revision,
        "working_tree_dirty": dirty,
        "semantic_prior_canonical_sha256": canonical_text_sha256(prior_path),
        "upstream_reference": UPSTREAM_REFERENCE,
        "environment": runtime_environment,
        "rental_preflight": rental_preflight,
        "runtime_policy": {
            "jax_compilation_cache": {
                "policy": "disabled",
                "effective_environment": effective_jax_environment,
            }
        },
    }
    if manifest_path.exists():
        _validate_resume_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), manifest
        )
    else:
        atomic_json(manifest_path, manifest)

    runs_dir = output_dir / "runs"
    histories_dir = output_dir / "histories"
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    for directory in (runs_dir, histories_dir, configs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    telemetry_dir = output_dir / "optimizer-telemetry"
    if common.get("optimizer_telemetry") is not None:
        telemetry_dir.mkdir(parents=True, exist_ok=True)

    expected_configs = {run["run_id"]: _run_config(run, common) for run in plan["runs"]}
    if resume:
        _recover_orphaned_provisional_results(
            output_dir, expected_configs, runtime_environment
        )
    _rebuild_indexes(output_dir, expected_configs, runtime_environment)
    atomic_json(
        output_dir / "session.json",
        {
            "status": "running",
            "started_utc": session_started_utc,
            "max_session_wall_seconds": common.get("max_session_wall_seconds"),
        },
    )

    failures = 0
    try:
        for run in plan["runs"]:
            run_id = run["run_id"]
            config = expected_configs[run_id]
            output_path = runs_dir / f"{run_id}.json"
            history_path = histories_dir / f"{run_id}.npz"
            optimizer_telemetry_path = (
                telemetry_dir / f"{run_id}.npz"
                if config.get("optimizer_telemetry") is not None
                else None
            )
            if resume and output_path.exists():
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete":
                    validate_completed_record(
                        existing, config, history_path, runtime_environment
                    )
                    continue

            config_path = configs_dir / f"{run_id}.json"
            atomic_json(config_path, config)
            configured_timeout = float(common["worker_timeout_seconds"])
            session_limit = common.get("max_session_wall_seconds")
            if session_limit is not None:
                remaining = float(session_limit) - (
                    time.perf_counter() - session_started
                )
                if remaining < configured_timeout:
                    atomic_json(
                        output_dir / "session.json",
                        {
                            "status": "wall_limit_reached",
                            "started_utc": session_started_utc,
                            "completed_utc": datetime.now(UTC).isoformat(),
                            "elapsed_seconds": time.perf_counter() - session_started,
                            "max_session_wall_seconds": session_limit,
                            "next_run_id": run_id,
                        },
                    )
                    _rebuild_indexes(output_dir, expected_configs, runtime_environment)
                    return 2
            provider_reserve = float(
                common.get("provider_evacuation_reserve_seconds") or 0
            )
            try:
                _require_provider_time(
                    common,
                    configured_timeout + provider_reserve,
                    f"start worker {run_id}",
                )
            except RuntimeError:
                atomic_json(
                    output_dir / "session.json",
                    {
                        "status": "provider_deadline_guard",
                        "started_utc": session_started_utc,
                        "completed_utc": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": time.perf_counter() - session_started,
                        "provider_stop_utc": common.get("provider_stop_utc"),
                        "next_run_id": run_id,
                    },
                )
                _rebuild_indexes(output_dir, expected_configs, runtime_environment)
                return 2
            started = time.perf_counter()
            completed = None
            timed_out = None
            try:
                worker_command = [
                    sys.executable,
                    "-m",
                    "experiments.uifo_paired.runner",
                    "--worker-config",
                    str(config_path),
                    "--worker-output",
                    str(output_path),
                    "--history-output",
                    str(history_path),
                ]
                if optimizer_telemetry_path is not None:
                    worker_command.extend(
                        [
                            "--optimizer-telemetry-output",
                            str(optimizer_telemetry_path),
                        ]
                    )
                completed = subprocess.run(
                    worker_command,
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=configured_timeout,
                    env=child_environment,
                )
            except subprocess.TimeoutExpired as error:
                timed_out = error
            full_wall_seconds = time.perf_counter() - started
            stdout = _process_text(
                timed_out.stdout if timed_out is not None else completed.stdout
            )
            stderr = _process_text(
                timed_out.stderr if timed_out is not None else completed.stderr
            )
            process_info = _persist_worker_process(
                output_dir,
                run_id,
                stdout,
                stderr,
                None if timed_out is not None else completed.returncode,
                full_wall_seconds,
                timed_out is not None,
            )

            if timed_out is not None:
                record = _synthetic_worker_error(
                    config,
                    runtime_environment,
                    process_info,
                    "WorkerTimeout",
                    f"worker exceeded {configured_timeout} seconds",
                )
                atomic_json(output_path, record)
                failures += 1
            elif not output_path.is_file():
                record = _synthetic_worker_error(
                    config,
                    runtime_environment,
                    process_info,
                    "WorkerProcessError",
                    f"worker exited {completed.returncode} without a run record",
                )
                atomic_json(output_path, record)
                failures += 1
            else:
                record = json.loads(output_path.read_text(encoding="utf-8"))
                record["worker_process"] = process_info
                atomic_json(output_path, record)
                if completed.returncode != 0 or record.get("status") != "complete":
                    failures += 1
                else:
                    validate_completed_record(
                        record, config, history_path, runtime_environment
                    )

            _rebuild_indexes(
                output_dir,
                expected_configs,
                runtime_environment,
                validate_complete_records=False,
            )
            if failures >= int(common.get("max_worker_failures", 1)):
                break
    except KeyboardInterrupt:
        atomic_json(
            output_dir / "session.json",
            {
                "status": "interrupted",
                "started_utc": session_started_utc,
                "completed_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - session_started,
                "max_session_wall_seconds": common.get("max_session_wall_seconds"),
            },
        )
        _rebuild_indexes(output_dir, expected_configs, runtime_environment)
        raise

    _rebuild_indexes(output_dir, expected_configs, runtime_environment)
    atomic_json(
        output_dir / "session.json",
        {
            "status": ("error" if failures else "complete"),
            "started_utc": session_started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - session_started,
            "max_session_wall_seconds": common.get("max_session_wall_seconds"),
        },
    )
    return 1 if failures else 0


def _run_config(run: dict[str, object], common: dict[str, object]) -> dict[str, object]:
    config = {
        **run,
        "allow_cpu": common["allow_cpu"],
        "evaluation_chunk_size": common.get("evaluation_chunk_size"),
        "max_evals": common["max_evals"],
        "max_time_seconds": common["max_time_seconds"],
        "n_frequencies": common["n_frequencies"],
        "population_size": common["population_size"],
        "require_a100": common.get("require_a100", False),
        "jax_compilation_cache_policy": common.get(
            "jax_compilation_cache_policy", "disabled"
        ),
        "target_losses": common["target_losses"],
    }
    for key in (
        "max_worker_failures",
        "study_profile",
        "decision_policy",
        "mechanics_evidence",
        "provider_stop_utc",
        "provider_deadline_maximum_horizon_seconds",
        "provider_evacuation_reserve_seconds",
    ):
        if key in common:
            config[key] = common[key]
    if "optimizer_telemetry" in common:
        config["optimizer_telemetry"] = common["optimizer_telemetry"]
    if "arm_optimizer_settings" in common:
        settings = common["arm_optimizer_settings"]
        if not isinstance(settings, dict) or run["arm"] not in settings:
            raise RuntimeError("plan is missing optimizer settings for an arm")
        config["optimizer_settings"] = validate_batched_settings(
            settings[run["arm"]]
        )
    return config


def _recover_orphaned_provisional_results(
    output_dir: Path,
    expected_configs: dict[str, dict[str, object]],
    runtime_environment: dict[str, object],
) -> None:
    """Preserve worker-complete records that lost their parent process evidence."""
    runs_dir = output_dir / "runs"
    for output_path in sorted(runs_dir.glob("*.json")):
        record = json.loads(output_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or "worker_process" in record:
            continue
        run_id = output_path.stem
        expected_config = expected_configs.get(run_id)
        if expected_config is None:
            raise RuntimeError(
                f"orphaned provisional record is outside the plan: {run_id}"
            )
        if record.get("run_id") != run_id:
            raise RuntimeError(
                f"orphaned provisional record filename/ID mismatch: {run_id}"
            )
        if strict_json(record.get("config")) != strict_json(expected_config):
            raise RuntimeError(
                f"orphaned provisional record configuration mismatch: {run_id}"
            )
        if strict_json(record.get("environment")) != strict_json(runtime_environment):
            raise RuntimeError(
                f"orphaned provisional record environment mismatch: {run_id}"
            )

        digest = sha256(output_path)[:12]
        recovery_dir = output_dir / "recovery" / "orphaned-workers" / run_id
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovered_record = recovery_dir / f"{digest}.json"
        if recovered_record.exists():
            raise RuntimeError(
                f"orphaned provisional recovery artifact exists: {recovered_record}"
            )
        history_path = output_dir / "histories" / f"{run_id}.npz"
        recovered_history = recovery_dir / f"{digest}.npz"
        if history_path.exists() and recovered_history.exists():
            raise RuntimeError(
                "orphaned provisional history recovery artifact exists: "
                f"{recovered_history}"
            )
        telemetry_path = output_dir / "optimizer-telemetry" / f"{run_id}.npz"
        recovered_telemetry = recovery_dir / f"{digest}.optimizer-telemetry.npz"
        if telemetry_path.exists() and recovered_telemetry.exists():
            raise RuntimeError(
                "orphaned provisional optimizer telemetry recovery artifact exists: "
                f"{recovered_telemetry}"
            )
        log_moves = []
        for stream in ("stdout", "stderr"):
            source = output_dir / "logs" / f"{run_id}.{stream}.log"
            destination = recovery_dir / f"{digest}.{stream}.log"
            if source.exists() and destination.exists():
                raise RuntimeError(
                    f"orphaned provisional log recovery artifact exists: {destination}"
                )
            if source.exists():
                log_moves.append((source, destination))
        os.replace(output_path, recovered_record)
        if history_path.exists():
            os.replace(history_path, recovered_history)
        if telemetry_path.exists():
            os.replace(telemetry_path, recovered_telemetry)
        for source, destination in log_moves:
            os.replace(source, destination)


def _preflight_environment(
    output_dir: Path,
    subprocess_environment: dict[str, str] | None = None,
    artifact_stem: str = "preflight",
) -> dict[str, object]:
    preflight_path = output_dir / f"{artifact_stem}.json"
    stdout_path = output_dir / f"{artifact_stem}.stdout.log"
    stderr_path = output_dir / f"{artifact_stem}.stderr.log"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.uifo_paired.runner",
                "--preflight-output",
                str(preflight_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=subprocess_environment,
        )
    except subprocess.TimeoutExpired as error:
        atomic_text(stdout_path, _process_text(error.stdout))
        atomic_text(stderr_path, _process_text(error.stderr))
        raise RuntimeError("runtime preflight exceeded 120 seconds") from error
    atomic_text(stdout_path, completed.stdout)
    atomic_text(stderr_path, completed.stderr)
    if completed.returncode != 0 or not preflight_path.is_file():
        raise RuntimeError(
            f"runtime preflight failed; inspect {stderr_path.name} "
            f"(exit {completed.returncode})"
        )
    return json.loads(preflight_path.read_text(encoding="utf-8"))


def _process_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _persist_worker_process(
    output_dir: Path,
    run_id: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    wall_seconds: float,
    timed_out: bool,
) -> dict[str, object]:
    stdout_path = output_dir / "logs" / f"{run_id}.stdout.log"
    stderr_path = output_dir / "logs" / f"{run_id}.stderr.log"
    atomic_text(stdout_path, stdout)
    atomic_text(stderr_path, stderr)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "full_wall_seconds": wall_seconds,
        "within_official_4h30_container_limit": wall_seconds <= 4.5 * 60 * 60,
        "stdout": {
            "path": f"logs/{stdout_path.name}",
            "sha256": sha256(stdout_path),
        },
        "stderr": {
            "path": f"logs/{stderr_path.name}",
            "sha256": sha256(stderr_path),
        },
    }


def _synthetic_worker_error(
    config: dict[str, object],
    runtime_environment: dict[str, object],
    process_info: dict[str, object],
    error_type: str,
    message: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "run_id": config["run_id"],
        "status": "error",
        "completed_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "environment": runtime_environment,
        "worker_process": process_info,
        "error": {"type": error_type, "message": message},
    }


def _validate_resume_manifest(existing: dict, current: dict) -> None:
    for key in (
        "format_version",
        "plan_id",
        "project_revision",
        "semantic_prior_canonical_sha256",
        "upstream_reference",
        "environment",
        "runtime_policy",
    ):
        if existing.get(key) != current.get(key):
            raise RuntimeError(f"resume manifest mismatch for {key}")


def _require_provider_time(
    configuration: dict[str, object], required_seconds: float, action: str
) -> None:
    deadline_text = configuration.get("provider_stop_utc")
    if deadline_text is None:
        return
    if not isinstance(deadline_text, str):
        raise RuntimeError("provider stop deadline is invalid")
    try:
        deadline = datetime.fromisoformat(deadline_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("provider stop deadline is invalid") from error
    remaining = (deadline.astimezone(UTC) - datetime.now(UTC)).total_seconds()
    maximum_horizon = configuration.get(
        "provider_deadline_maximum_horizon_seconds"
    )
    if maximum_horizon is not None and remaining > float(maximum_horizon) + 60.0:
        raise RuntimeError(
            "provider stop deadline exceeds the frozen eight-hour horizon"
        )
    if remaining < required_seconds:
        raise RuntimeError(
            f"insufficient provider time to {action}: {remaining:.1f}s remains; "
            f"{required_seconds:.1f}s is required"
        )


def _validate_mechanics_revision(
    configuration: dict[str, object], revision: str
) -> None:
    evidence = configuration.get("mechanics_evidence")
    if evidence is None:
        return
    if (
        not isinstance(evidence, dict)
        or evidence.get("project_revision") != revision
    ):
        raise RuntimeError(
            "restart screen revision differs from passed mechanics evidence"
        )


def _rebuild_indexes(
    output_dir: Path,
    expected_configs: dict[str, dict[str, object]] | None = None,
    expected_environment: dict[str, object] | None = None,
    *,
    validate_complete_records: bool = True,
) -> None:
    records = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if expected_configs is not None:
            run_id = path.stem
            if run_id not in expected_configs:
                raise RuntimeError(f"unexpected run record outside plan: {path.name}")
            if record.get("run_id") != run_id:
                raise RuntimeError(f"run record filename/ID mismatch: {path.name}")
            expected_config = expected_configs[run_id]
            if strict_json(record.get("config")) != strict_json(expected_config):
                raise RuntimeError(f"run record configuration mismatch: {run_id}")
            if record.get("status") == "complete":
                if validate_complete_records:
                    validate_completed_record(
                        record,
                        expected_config,
                        output_dir / "histories" / f"{run_id}.npz",
                        expected_environment,
                    )
            elif record.get("status") not in {"error", "interrupted"}:
                raise RuntimeError(f"invalid run status for {run_id}")
        records.append(record)

    if expected_configs is not None:
        _validate_resolved_topology_identities(records)

    jsonl = "".join(
        json.dumps(strict_json(record), sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    temporary = output_dir / "runs.jsonl.tmp"
    temporary.write_text(jsonl, encoding="utf-8")
    os.replace(temporary, output_dir / "runs.jsonl")

    restart_profiles = {"restart-mechanics-v1", "restart-screen-v1"}
    use_restart_summary = bool(expected_configs) and {
        str(config.get("study_profile")) for config in expected_configs.values()
    } <= restart_profiles
    summary = (
        summarize_restart_records(
            records,
            expected_configs,
            compute_bootstrap=validate_complete_records,
        )
        if use_restart_summary
        else summarize_records(
            records,
            expected_configs,
            compute_bootstrap=validate_complete_records,
        )
    )
    atomic_json(
        output_dir / "summary.json",
        summary,
    )


def _validate_resolved_topology_identities(
    records: list[dict[str, object]],
) -> None:
    by_hash: dict[str, str] = {}
    by_configuration: dict[str, str] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        topology_hash = str(record["problem"]["topology_sha256"])
        configured = json.dumps(
            record["config"]["topology"], sort_keys=True, separators=(",", ":")
        )
        previous = by_hash.setdefault(topology_hash, configured)
        if previous != configured:
            raise RuntimeError(
                "distinct planned topologies resolved to the same topology identity"
            )
        previous_hash = by_configuration.setdefault(configured, topology_hash)
        if previous_hash != topology_hash:
            raise RuntimeError(
                "one planned topology resolved to multiple topology identities"
            )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def parse_topology_panel(
    path: Path, *, require_archive_exclusion: bool = False
) -> tuple[list[str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "source_kind": "json_topology_panel",
        "source_name": path.name,
        "source_sha256": sha256(path),
        "archive_exclusion_verified": False,
        "official_dataset_sha256": None,
        "panel_id": None,
    }
    if isinstance(payload, dict):
        topologies = payload.get("topologies")
        if payload.get("archive_exclusion_verified") is True:
            raise ValueError(
                "self-attested archive exclusion is not accepted; pass the "
                "pinned dataset with --official-dataset"
            )
        panel_id = payload.get("panel_id")
        if panel_id is not None and not isinstance(panel_id, str):
            raise ValueError("panel_id must be a string")
        metadata["panel_id"] = panel_id
    else:
        topologies = payload
    if not isinstance(topologies, list) or not all(
        isinstance(value, str) for value in topologies
    ):
        raise ValueError(
            "topology file must be a JSON list or {topologies: [...]} object"
        )
    if require_archive_exclusion:
        raise ValueError(
            "archive exclusion must be computed from --official-dataset, not "
            "asserted inside the topology panel"
        )
    metadata["topology_count"] = len(topologies)
    return topologies, metadata


def parse_topologies(path: Path) -> list[str]:
    """Backward-compatible convenience wrapper for unaudited smoke panels."""
    return parse_topology_panel(path)[0]


def audit_topology_exclusion(
    topologies: list[str],
    dataset_path: Path,
    prior_panel_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Compute exact archive/prior-panel set exclusion with hashed provenance."""
    import h5py

    dataset_digest = sha256(dataset_path)
    if dataset_digest != OFFICIAL_DATASET_SHA256:
        raise ValueError(
            "official dataset SHA-256 mismatch; refusing an exclusion claim"
        )
    with h5py.File(dataset_path, "r") as archive:
        entries = archive["entries"]
        raw_topologies = entries["topology_string"][:]
        archive_topologies = {
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_topologies
        }
        archive_entries = int(len(entries))

    panel_set = set(topologies)
    archive_overlap = sorted(panel_set & archive_topologies)
    if archive_overlap:
        raise ValueError(
            f"topology panel overlaps the official archive in "
            f"{len(archive_overlap)} identities"
        )

    prior_audits = []
    for prior_path in prior_panel_paths or []:
        prior_topologies, prior_metadata = parse_topology_panel(prior_path)
        prior_overlap = sorted(panel_set & set(prior_topologies))
        if prior_overlap:
            raise ValueError(
                f"topology panel overlaps prior panel {prior_path.name!r} in "
                f"{len(prior_overlap)} identities"
            )
        prior_audits.append(
            {
                "source_name": prior_path.name,
                "source_sha256": prior_metadata["source_sha256"],
                "topology_count": len(prior_topologies),
                "overlap_count": 0,
            }
        )

    identity_digest = hashlib.sha256(
        json.dumps(sorted(panel_set), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "method": "exact topology-string set intersection",
        "panel_identity_sha256": identity_digest,
        "panel_topology_count": len(panel_set),
        "official_dataset": {
            "source_name": dataset_path.name,
            "sha256": dataset_digest,
            "size_bytes": dataset_path.stat().st_size,
            "entries": archive_entries,
            "unique_topologies": len(archive_topologies),
            "overlap_count": 0,
        },
        "prior_panels": prior_audits,
    }


def parse_arm_patience(values: list[str]) -> dict[str, int] | None:
    if not values:
        return None
    result = {}
    for value in values:
        arm, separator, raw_patience = value.partition("=")
        if not separator or not arm or not raw_patience:
            raise ValueError("--arm-patience must use ARM=INTEGER")
        if arm in result:
            raise ValueError(f"duplicate --arm-patience for {arm!r}")
        try:
            result[arm] = int(raw_patience)
        except ValueError as error:
            raise ValueError(
                f"invalid --arm-patience integer for {arm!r}"
            ) from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--topology-seeds", nargs="+", type=int)
    source.add_argument("--topologies-file", type=Path)
    parser.add_argument("--optimizer-seeds", nargs="+", type=int, default=[7])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=VALID_ARMS,
        default=["no_prior", "semantic_prior"],
    )
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--max-evals", type=int)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--evaluation-chunk-size", type=int)
    parser.add_argument("--arm-patience", action="append", default=[])
    parser.add_argument(
        "--pair-order-policy",
        choices=["rotate_pairs", "alternate_topology_and_seed"],
        default="rotate_pairs",
    )
    parser.add_argument(
        "--optimizer-telemetry", choices=[OPTIMIZER_TELEMETRY_MODE]
    )
    parser.add_argument("--n-frequencies", type=int, default=50)
    parser.add_argument("--target-loss", action="append", type=float, default=[])
    parser.add_argument("--worker-timeout", type=float)
    parser.add_argument("--max-session-wall", type=float)
    parser.add_argument("--max-worker-failures", type=int, default=1)
    parser.add_argument("--provider-stop-utc")
    parser.add_argument("--provider-evacuation-reserve", type=float)
    parser.add_argument("--mechanics-study-dir", type=Path)
    parser.add_argument("--mechanics-package", type=Path)
    parser.add_argument("--study-profile", choices=profile_names())
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--minimum-gpu-memory-mib", type=int)
    parser.add_argument("--max-idle-gpu-memory-mib", type=int)
    parser.add_argument("--max-idle-gpu-utilization", type=int)
    parser.add_argument("--minimum-free-disk-gib", type=float)
    parser.add_argument("--require-archive-exclusion", action="store_true")
    parser.add_argument("--official-dataset", type=Path)
    parser.add_argument("--exclude-prior-panel", action="append", type=Path, default=[])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/generated/uifo-paired")
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--history-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--optimizer-telemetry-output", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument("--preflight-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.preflight_output:
        raise SystemExit(run_preflight(args.preflight_output))

    if args.worker_config:
        if not args.worker_output or not args.history_output:
            parser.error("worker mode requires output and history paths")
        raise SystemExit(
            run_worker(
                args.worker_config,
                args.worker_output,
                args.history_output,
                args.optimizer_telemetry_output,
            )
        )

    if not args.topology_seeds and not args.topologies_file:
        parser.error("one topology source is required")
    try:
        if bool(args.mechanics_study_dir) != bool(args.mechanics_package):
            raise ValueError(
                "--mechanics-study-dir and --mechanics-package are required together"
            )
        mechanics_evidence = None
        if args.mechanics_study_dir:
            from experiments.uifo_paired.restart_evidence import (
                validate_mechanics_predecessor,
            )

            mechanics_evidence = validate_mechanics_predecessor(
                args.mechanics_study_dir, args.mechanics_package
            )
        if args.topologies_file:
            topologies, topology_panel = parse_topology_panel(args.topologies_file)
            if args.require_archive_exclusion and not args.official_dataset:
                raise ValueError(
                    "--require-archive-exclusion requires --official-dataset"
                )
            if args.exclude_prior_panel and not args.official_dataset:
                raise ValueError("--exclude-prior-panel requires --official-dataset")
            if args.official_dataset:
                topology_panel["archive_exclusion_audit"] = audit_topology_exclusion(
                    topologies,
                    args.official_dataset,
                    args.exclude_prior_panel,
                )
                topology_panel["archive_exclusion_verified"] = True
                topology_panel["official_dataset_sha256"] = OFFICIAL_DATASET_SHA256
        else:
            if (
                args.require_archive_exclusion
                or args.official_dataset
                or args.exclude_prior_panel
            ):
                raise ValueError("archive-exclusion audit requires --topologies-file")
            topologies = None
            topology_panel = {
                "source_kind": "topology_seeds",
                "archive_exclusion_verified": False,
                "official_dataset_sha256": None,
                "topology_count": len(args.topology_seeds),
            }
        plan = build_plan(
            topology_seeds=args.topology_seeds,
            topologies=topologies,
            optimizer_seeds=args.optimizer_seeds,
            arms=args.arms,
            max_time_seconds=args.max_time,
            max_evals=args.max_evals,
            population_size=args.population_size,
            n_frequencies=args.n_frequencies,
            target_losses=args.target_loss,
            allow_cpu=args.allow_cpu,
            worker_timeout_seconds=args.worker_timeout,
            topology_panel=topology_panel,
            evaluation_chunk_size=args.evaluation_chunk_size,
            require_a100=args.require_a100,
            minimum_gpu_memory_mib=args.minimum_gpu_memory_mib,
            max_idle_gpu_memory_mib=args.max_idle_gpu_memory_mib,
            max_idle_gpu_utilization_percent=args.max_idle_gpu_utilization,
            minimum_free_disk_gib=args.minimum_free_disk_gib,
            max_session_wall_seconds=args.max_session_wall,
            max_worker_failures=args.max_worker_failures,
            study_profile=args.study_profile,
            optimizer_telemetry=args.optimizer_telemetry,
            arm_patience=parse_arm_patience(args.arm_patience),
            pair_order_policy=args.pair_order_policy,
            mechanics_evidence=mechanics_evidence,
            provider_stop_utc=args.provider_stop_utc,
            provider_evacuation_reserve_seconds=(
                args.provider_evacuation_reserve
            ),
        )
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    output_dir = args.output / plan["plan_id"]
    if args.recover_stale_lock and not args.resume:
        parser.error("--recover-stale-lock requires --resume")
    raise SystemExit(
        orchestrate(
            plan,
            output_dir,
            resume=args.resume,
            recover_stale_lock=args.recover_stale_lock,
        )
    )


if __name__ == "__main__":
    main()
