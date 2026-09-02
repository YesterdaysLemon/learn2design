"""Dedicated heterogeneous-arm worker for the frozen candidate screen.

The module is import-safe on CPU. Scientific dependencies and optimizer source
are loaded only inside :func:`execute_run`, after the caller has authenticated
the source/runtime locks.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import textwrap
import zipfile
from typing import Any, Callable

from .analysis import HISTORY_FIELDS, project_history
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .contract import (
    MAX_TIME_SECONDS,
    N_FREQUENCIES,
    OFFICIAL_ARCHIVE_SHA256,
    PANEL_ID,
    POPULATION_SIZE,
    ROUND1_ARCHIVE_SHA256,
    ROUND1_MANIFEST_SHA256,
    ROUND1_MEMBER_SHA256,
    ROUND1_REVISION,
    STAGE1_OPTIMIZER_SEED,
    STAGE2_OPTIMIZER_SEED,
    STUDY_ID,
    arm_spec,
)
from .packet import BoundedTextCapture


class WorkerError(RuntimeError):
    pass


CONFIG_KEYS = {
    "schema_version",
    "study_id",
    "run_id",
    "stage",
    "member_index",
    "execution_position",
    "arm_id",
    "optimizer_seed",
    "topology",
    "topology_sha256",
    "panel_sha256",
    "split_receipt_sha256",
    "selection_receipt_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "revision",
    "arm_profile",
    "panel_commitment_sha256",
    "max_time_seconds",
    "max_evals",
    "population_size",
    "n_frequencies",
    "allow_cpu",
}


def validate_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise WorkerError("worker config schema mismatch")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["study_id"] != STUDY_ID:
        raise WorkerError("worker config identity mismatch")
    if type(config["stage"]) is not int or config["stage"] not in (1, 2):
        raise WorkerError("worker stage is invalid")
    expected_optimizer_seed = (
        STAGE1_OPTIMIZER_SEED if config["stage"] == 1 else STAGE2_OPTIMIZER_SEED
    )
    if (
        type(config["optimizer_seed"]) is not int
        or config["optimizer_seed"] != expected_optimizer_seed
    ):
        raise WorkerError("worker optimizer seed is invalid")
    if (
        isinstance(config["member_index"], bool)
        or not isinstance(config["member_index"], int)
        or config["member_index"] not in range(8)
        or isinstance(config["execution_position"], bool)
        or not isinstance(config["execution_position"], int)
        or config["execution_position"] not in range(16)
    ):
        raise WorkerError("worker run position is invalid")
    spec = arm_spec(config["arm_id"])
    if isinstance(config["max_time_seconds"], bool) or config["max_time_seconds"] != MAX_TIME_SECONDS:
        raise WorkerError("worker time budget mismatch")
    if config["max_evals"] is not None:
        raise WorkerError("worker evaluation cap must be null")
    if type(config["population_size"]) is not int or config["population_size"] != POPULATION_SIZE:
        raise WorkerError("worker population mismatch")
    if type(config["n_frequencies"]) is not int or config["n_frequencies"] != N_FREQUENCIES:
        raise WorkerError("worker frequency count mismatch")
    if config["allow_cpu"] is not False:
        raise WorkerError("scored worker cannot allow CPU")
    if not isinstance(config["run_id"], str) or not config["run_id"]:
        raise WorkerError("worker run ID is invalid")
    topology = config["topology"]
    if not isinstance(topology, str) or not topology:
        raise WorkerError("worker topology is invalid")
    if sha256_bytes(topology.encode("utf-8")) != config["topology_sha256"]:
        raise WorkerError("worker topology digest mismatch")
    for key in (
        "panel_sha256",
        "panel_commitment_sha256",
        "split_receipt_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
    ):
        value = config[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(token not in "0123456789abcdef" for token in value)
        ):
            raise WorkerError(f"worker {key} is invalid")
    revision = config["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(token not in "0123456789abcdef" for token in revision)
    ):
        raise WorkerError("worker revision is invalid")
    profile = config["arm_profile"]
    if not isinstance(profile, dict):
        raise WorkerError("worker arm profile is absent")
    package_closure = profile.get("package_closure_sha256")
    if (
        not isinstance(package_closure, str)
        or len(package_closure) != 64
        or any(token not in "0123456789abcdef" for token in package_closure)
        or profile != spec.lock_row(package_closure)
    ):
        raise WorkerError("worker arm profile mismatch")
    selection = config["selection_receipt_sha256"]
    if config["stage"] == 1 and selection is not None:
        raise WorkerError("Stage 1 cannot bind a selection receipt")
    if config["stage"] == 2 and (
        not isinstance(selection, str)
        or len(selection) != 64
        or any(token not in "0123456789abcdef" for token in selection)
    ):
        raise WorkerError("Stage 2 lacks a selection receipt")
    if spec.arm_id == "A_round1_control" and config["stage"] not in (1, 2):
        raise WorkerError("Round-1 control stage mismatch")
    return dict(config)


def _load_round1_manifest(
    manifest_path: Path, *, evaluated_revision: str
) -> dict[str, object]:
    if evaluated_revision != ROUND1_REVISION:
        raise WorkerError("Round-1 evaluated revision mismatch")
    if sha256_file(manifest_path) != ROUND1_MANIFEST_SHA256:
        raise WorkerError("Round-1 manifest digest mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerError("Round-1 manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "archive",
        "archive_sha256",
        "created_utc",
        "project_revision",
        "source_files",
        "upstream_reference",
        "working_tree_dirty",
    }:
        raise WorkerError("Round-1 manifest schema mismatch")
    expected_rows = [
        {
            "path": name,
            "sha256": digest,
            "size_bytes": size,
        }
        for (name, digest), size in zip(
            ROUND1_MEMBER_SHA256.items(), (74, 18300, 26863), strict=True
        )
    ]
    if (
        manifest["archive"] != "submission.zip"
        or manifest["archive_sha256"] != ROUND1_ARCHIVE_SHA256
        or manifest["source_files"] != expected_rows
        or manifest["working_tree_dirty"] is not False
        or not isinstance(manifest["project_revision"], str)
        or len(manifest["project_revision"]) != 40
    ):
        raise WorkerError("Round-1 manifest identity mismatch")
    return manifest


def _exec_isolated_source(
    *, name: str, source_bytes: bytes, logical_file: str
) -> types.ModuleType:
    if name in sys.modules:
        raise WorkerError(f"isolated module namespace is already occupied: {name}")
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WorkerError("arm source is not UTF-8") from error
    module = types.ModuleType(name)
    module.__file__ = logical_file
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[name] = module
    try:
        exec(compile(source, logical_file, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def _round1_module(
    archive_path: Path,
    manifest_path: Path,
    *,
    evaluated_revision: str,
) -> types.ModuleType:
    _load_round1_manifest(manifest_path, evaluated_revision=evaluated_revision)
    if sha256_file(archive_path) != ROUND1_ARCHIVE_SHA256:
        raise WorkerError("Round-1 archive digest mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != list(ROUND1_MEMBER_SHA256):
            raise WorkerError("Round-1 archive member/order mismatch")
        if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
            raise WorkerError("Round-1 archive contains an invalid member")
        member_bytes: dict[str, bytes] = {}
        for info in infos:
            content = archive.read(info)
            if sha256_bytes(content) != ROUND1_MEMBER_SHA256[info.filename]:
                raise WorkerError("Round-1 archive member digest mismatch")
            member_bytes[info.filename] = content
    return _exec_isolated_source(
        name="l2d_round1_control_submission",
        source_bytes=member_bytes["submission.py"],
        logical_file="round1_zip::submission.py",
    )


def load_arm_class(
    arm_id: str,
    *,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    evaluated_revision: str = ROUND1_REVISION,
) -> type:
    spec = arm_spec(arm_id)
    if arm_id == "A_round1_control":
        module = _round1_module(
            round1_archive,
            round1_manifest,
            evaluated_revision=evaluated_revision,
        )
    else:
        source_path = repository_root / spec.logical_module_id
        if not source_path.is_file():
            raise WorkerError(f"arm source is absent: {arm_id}")
        source_bytes = source_path.read_bytes()
        if sha256_bytes(source_bytes) != spec.source_sha256:
            raise WorkerError(f"arm source digest mismatch: {arm_id}")
        module = _exec_isolated_source(
            name=f"l2d_screen_{arm_id}",
            source_bytes=source_bytes,
            logical_file=spec.logical_module_id,
        )
    candidate = getattr(module, spec.class_name, None)
    if not isinstance(candidate, type):
        raise WorkerError("arm class is absent")
    if getattr(candidate, "algorithm_str", None) != spec.algorithm_str:
        raise WorkerError("arm algorithm string mismatch")
    return candidate


def verify_warmup_source(algorithm_class: type) -> dict[str, object]:
    """Prove the exact warmup method has no random or outcome-bearing call."""
    import inspect

    method = getattr(algorithm_class, "_warmup_population_evaluation", None)
    if method is None:
        raise WorkerError("arm warmup method is absent")
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            name = f"{target.value.id}.{target.attr}"
        else:
            raise WorkerError("warmup source contains an indirect call")
        calls.append(name)
    allowed = {
        "min",
        "range",
        "sorted",
        "objective.warmup_value_and_grad_aux",
        "objective.warmup_vmap_value_and_grad_aux",
    }
    if any(name not in allowed or "random" in name.lower() for name in calls):
        raise WorkerError("warmup source contains a forbidden call")
    if not any(name.startswith("objective.warmup_") for name in calls):
        raise WorkerError("warmup source does not traverse the public warmup API")
    return {
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "calls": calls,
        "random_sampling_calls": 0,
    }


def _array_hash(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _member_hashes(array: Any) -> list[str]:
    import numpy as np

    value = np.asarray(array)
    if value.ndim != 2 or value.shape[0] != POPULATION_SIZE:
        raise WorkerError("captured population shape mismatch")
    return [_array_hash(member) for member in value]


def _host(value: Any) -> Any:
    import jax
    import numpy as np

    array = np.asarray(jax.device_get(value))
    return array.item() if array.ndim == 0 else array.tolist()


def _flatten(objective: Any) -> list[dict[str, object]]:
    loss_history = [_host(value) for value in objective.loss_history]
    feasible_history = [_host(value) for value in objective.is_feasible_history]
    sensitivity_history = [
        _host(value) for value in objective.sensitivity_loss_history
    ]
    penalty_history = [_host(value) for value in objective.penalty_history]
    if not (
        len(loss_history)
        == len(feasible_history)
        == len(sensitivity_history)
        == len(penalty_history)
        == len(objective.time_steps)
    ):
        raise WorkerError("Objective history lists are misaligned")
    rows: list[dict[str, object]] = []
    evaluations = 0

    def numeric_or_null(value: Any, label: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkerError(f"Objective {label} is not numeric")
        converted = float(value)
        return converted if math.isfinite(converted) else None

    for call_index in range(len(loss_history)):
        def values(item: Any) -> list[Any]:
            return item if isinstance(item, list) else [item]

        losses = values(loss_history[call_index])
        feasible = values(feasible_history[call_index])
        sensitivity = values(sensitivity_history[call_index])
        penalty = values(penalty_history[call_index])
        if not (len(losses) == len(feasible) == len(sensitivity) == len(penalty)):
            raise WorkerError("Objective call batch shapes are misaligned")
        evaluations += len(losses)
        for candidate_index, loss in enumerate(losses):
            feasible_value = feasible[candidate_index]
            if type(feasible_value) is not bool:
                raise WorkerError("Objective feasibility is not a strict Boolean")
            elapsed = _host(objective.time_steps[call_index])
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                raise WorkerError("Objective time step is not numeric")
            elapsed_value = float(elapsed)
            if not math.isfinite(elapsed_value):
                raise WorkerError("Objective time step is non-finite")
            rows.append(
                {
                    "call_index": call_index,
                    "candidate_index": candidate_index,
                    "eval_count_after_call": evaluations,
                    "time_seconds": elapsed_value,
                    "loss": numeric_or_null(loss, "loss"),
                    "sensitivity_loss": numeric_or_null(
                        sensitivity[candidate_index], "sensitivity loss"
                    ),
                    "penalty": numeric_or_null(
                        penalty[candidate_index], "penalty"
                    ),
                    "is_feasible": feasible_value,
                }
            )
    return rows


def _snapshot_objective(
    objective: Any, draw_calls: int, initial_population: Any
) -> dict[str, object]:
    def length(name: str) -> int:
        value = getattr(objective, name, None)
        return len(value) if value is not None else 0

    rng_key = getattr(objective, "_rng_key", None)
    rng_key_sha256 = None if rng_key is None else _array_hash(_host(rng_key))
    parameter_sha256 = (
        None if initial_population is None else _array_hash(_host(initial_population))
    )

    def scalar(name: str, default: Any = None) -> Any:
        value = getattr(objective, name, default)
        return _host(value) if value is not None else None

    histories = {
        name: length(name)
        for name in (
            "time_steps",
            "loss_history",
            "sensitivity_loss_history",
            "penalty_history",
            "is_feasible_history",
            "eval_type_history",
            "params_history",
            "grad_history",
            "hessian_history",
            "power_values_history",
            "violations_history",
        )
    }
    return {
        "random_draw_calls": draw_calls,
        "rng_seed": scalar("_seed"),
        "rng_key_sha256": rng_key_sha256,
        "parameter_sha256": parameter_sha256,
        "eval_count": int(objective.eval_count),
        "log_call_count": int(objective.log_call_count),
        "time_elapsed": float(objective.time_elapsed),
        "time_left": scalar("time_left"),
        "evals_left": scalar("evals_left"),
        "max_time": scalar("_max_time"),
        "max_evals": scalar("_max_evals"),
        "time_offset": scalar("_time_offset", 0.0),
        "start_time_is_none": getattr(objective, "_start_time", None) is None,
        "histories": histories,
        "budget_exceeded": objective.budget_exceeded,
        "time_exceeded": scalar("time_exceeded", False),
        "evals_exceeded": scalar("evals_exceeded", False),
    }


class _Instrumentation:
    def __init__(
        self,
        objective: Any,
        *,
        expect_warmup: bool,
        warmup_source_proof: dict[str, object],
    ) -> None:
        self.objective = objective
        self.expect_warmup = expect_warmup
        self.draw_calls = 0
        self.raw_population: Any = None
        self.initial_population: Any = None
        self.before_warmup: dict[str, object] | None = None
        self.after_warmup: dict[str, object] | None = None
        self.warmup_source_proof = warmup_source_proof
        self._install()

    def _install(self) -> None:
        original_random = self.objective.random_params_unbounded

        def random_params_unbounded(*args: Any, **kwargs: Any) -> Any:
            value = original_random(*args, **kwargs)
            self.draw_calls += 1
            if self.raw_population is None:
                self.raw_population = value
            return value

        self.objective.random_params_unbounded = random_params_unbounded

        for name in (
            "warmup_value_and_grad_aux",
            "warmup_vmap_value_and_grad_aux",
        ):
            original = getattr(self.objective, name)

            def wrapper(*args: Any, __original: Callable[..., Any] = original, **kwargs: Any) -> Any:
                if self.expect_warmup and self.before_warmup is None:
                    self.before_warmup = _snapshot_objective(
                        self.objective, self.draw_calls, self.initial_population
                    )
                return __original(*args, **kwargs)

            setattr(self.objective, name, wrapper)

        original_start = self.objective.start_logging

        def start_logging(*args: Any, **kwargs: Any) -> Any:
            if self.expect_warmup:
                if self.before_warmup is None:
                    raise WorkerError("warmup did not traverse an Objective evaluator")
                self.after_warmup = _snapshot_objective(
                    self.objective, self.draw_calls, self.initial_population
                )
            else:
                snapshot = _snapshot_objective(
                    self.objective, self.draw_calls, self.initial_population
                )
                self.before_warmup = snapshot
                self.after_warmup = dict(snapshot)
            return original_start(*args, **kwargs)

        self.objective.start_logging = start_logging

    def capture_raw(self, value: Any) -> None:
        if self.raw_population is None:
            self.raw_population = value

    def capture_initial(self, value: Any) -> None:
        if self.initial_population is not None:
            raise WorkerError("optimizer emitted initial population twice")
        self.initial_population = value

    def receipt(self) -> dict[str, object]:
        import jax
        import numpy as np

        if self.raw_population is None or self.initial_population is None:
            raise WorkerError("initial population instrumentation is incomplete")
        if self.before_warmup is None or self.after_warmup is None:
            raise WorkerError("warmup boundary instrumentation is incomplete")
        if self.before_warmup != self.after_warmup:
            raise WorkerError("warmup mutated Objective/RNG/budget state")
        raw = np.asarray(jax.device_get(self.raw_population))
        initial = np.asarray(jax.device_get(self.initial_population))
        return {
            "raw_population_sha256": _array_hash(raw),
            "raw_member_sha256": _member_hashes(raw),
            "initial_population_sha256": _array_hash(initial),
            "initial_member_sha256": _member_hashes(initial),
            "warmup_enabled": self.expect_warmup,
            "warmup_source_proof": self.warmup_source_proof,
            "before_warmup": self.before_warmup,
            "after_warmup": self.after_warmup,
        }


def _write_history(
    path: Path, rows: list[dict[str, object]], initial: Any, raw: Any
) -> str:
    import jax
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise WorkerError("history target already exists")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
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
                [math.nan if row["loss"] is None else row["loss"] for row in rows],
                dtype=np.float64,
            ),
            sensitivity_loss=np.asarray(
                [
                    math.nan
                    if row["sensitivity_loss"] is None
                    else row["sensitivity_loss"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            penalty=np.asarray(
                [
                    math.nan if row["penalty"] is None else row["penalty"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            is_feasible=np.asarray(
                [row["is_feasible"] for row in rows], dtype=np.bool_
            ),
            initial_params_unbounded=np.asarray(jax.device_get(initial)),
            raw_params_unbounded=np.asarray(jax.device_get(raw)),
        )
    os.replace(temporary, path)
    return sha256_file(path)


def authenticate_execution_locks(
    config: dict[str, Any],
    *,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    expected_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate every committed byte before importing scientific code."""
    from .locks import read_runtime_lock, read_source_lock
    from .source_closure import logical_source_closure

    if expected_revision != config["revision"]:
        raise WorkerError("worker invocation revision mismatch")
    runtime, runtime_digest = read_runtime_lock(runtime_lock_path)
    if runtime_digest != config["runtime_lock_sha256"]:
        raise WorkerError("worker runtime-lock digest mismatch")
    logical_sources = logical_source_closure(
        repository_root, round1_archive, round1_manifest
    )
    source, source_digest = read_source_lock(
        source_lock_path,
        runtime_lock_sha256=runtime_digest,
        logical_sources=logical_sources,
        expected_revision=expected_revision,
    )
    if source_digest != config["source_lock_sha256"]:
        raise WorkerError("worker source-lock digest mismatch")
    expected_components = {
        "worker_sha256": sha256_file(Path(__file__)),
        "orchestrator_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/orchestrator.py"
        ),
        "production_analyzer_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/analysis.py"
        ),
        "reference_analyzer_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/reference_analysis.py"
        ),
    }
    if any(source[key] != value for key, value in expected_components.items()):
        raise WorkerError("worker locked component digest mismatch")
    if source["panel_commitment_sha256"] != config["panel_commitment_sha256"]:
        raise WorkerError("worker panel commitment binding mismatch")
    profile = next(
        (
            row
            for row in source["arm_profiles"]
            if row.get("arm_id") == config["arm_id"]
        ),
        None,
    )
    if profile != config["arm_profile"]:
        raise WorkerError("worker config/source-lock arm profile mismatch")
    if profile["package_closure_sha256"] != runtime["package_closure_sha256"]:
        raise WorkerError("worker arm/runtime package closure mismatch")
    return runtime, source


def execute_run(
    config: dict[str, Any],
    *,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    expected_revision: str,
    history_path: Path,
) -> dict[str, object]:
    config = validate_config(config)
    authenticate_execution_locks(
        config,
        repository_root=repository_root,
        round1_archive=round1_archive,
        round1_manifest=round1_manifest,
        runtime_lock_path=runtime_lock_path,
        source_lock_path=source_lock_path,
        expected_revision=expected_revision,
    )
    import jax
    from dfbench import Objective
    from dfbench.problems import UIFOProblem

    if jax.default_backend() == "cpu":
        raise WorkerError("scored candidate screen requires an accelerator")
    device_kinds = [str(getattr(device, "device_kind", "")) for device in jax.devices()]
    if len(device_kinds) != 1 or device_kinds[0] != "NVIDIA H100 80GB HBM3":
        raise WorkerError("worker device is not the exact frozen H100")
    spec = arm_spec(config["arm_id"])
    algorithm_class = load_arm_class(
        spec.arm_id,
        repository_root=repository_root,
        round1_archive=round1_archive,
        round1_manifest=round1_manifest,
        evaluated_revision=ROUND1_REVISION,
    )
    problem = UIFOProblem(
        size=3,
        n_frequencies=N_FREQUENCIES,
        topology=config["topology"],
    )
    if str(problem.topology_string) != config["topology"]:
        raise WorkerError("constructed problem topology mismatch")
    objective = Objective(
        problem,
        max_time=MAX_TIME_SECONDS,
        max_evals=None,
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
    instrumentation = _Instrumentation(
        objective,
        expect_warmup=spec.preclock_warmup,
        warmup_source_proof=verify_warmup_source(algorithm_class),
    )
    kwargs = {
        **spec.fixed_kwargs(),
        "random_seed": config["optimizer_seed"],
        "initial_population_callback": instrumentation.capture_initial,
    }
    if spec.arm_id != "A_round1_control":
        kwargs["raw_initial_population_callback"] = instrumentation.capture_raw
    started = time.perf_counter()
    algorithm_class().optimize(objective, **kwargs)
    wall_seconds = time.perf_counter() - started
    rows = _flatten(objective)
    if not rows:
        raise WorkerError("worker completed without candidate-call history")
    if int(objective.eval_count) != int(rows[-1]["eval_count_after_call"]):
        raise WorkerError("Objective evaluation count differs from full history")
    population_receipt = instrumentation.receipt()
    history_sha256 = _write_history(
        history_path,
        rows,
        instrumentation.initial_population,
        instrumentation.raw_population,
    )
    projection = project_history(rows)
    metrics = projection.as_dict()
    metrics.pop("rows")
    metrics.pop("logged_calls")
    return {
        "run_id": config["run_id"],
        "config": config,
        "algorithm": {
            "logical_module_id": spec.logical_module_id,
            "python_module_name": spec.python_module_name,
            "class_name": spec.class_name,
            "algorithm_str": spec.algorithm_str,
            "source_sha256": spec.source_sha256,
            "kwargs": {**spec.fixed_kwargs(), "random_seed": config["optimizer_seed"]},
        },
        "topology_sha256": config["topology_sha256"],
        "initial_population": population_receipt,
        "history": {
            "sha256": history_sha256,
            "rows": len(rows),
            "schema": list(HISTORY_FIELDS),
        },
        "metrics": metrics,
        "objective_accounting": {
            "eval_count": projection.evaluation_count,
            "log_call_count": projection.logged_calls,
        },
        "worker_measurement": {
            "optimize_wall_seconds": wall_seconds,
            "objective_time_elapsed_snapshot": float(objective.time_elapsed),
            "panel_id": PANEL_ID,
            "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        },
    }


def run_packet(
    config: dict[str, Any],
    *,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    expected_revision: str,
    history_path: Path,
) -> bytes:
    capture_stdout = BoundedTextCapture(max_bytes=1_048_576)
    capture_stderr = BoundedTextCapture(max_bytes=1_048_576)
    with redirect_stdout(capture_stdout), redirect_stderr(capture_stderr):
        record = execute_run(
            config,
            repository_root=repository_root,
            round1_archive=round1_archive,
            round1_manifest=round1_manifest,
            runtime_lock_path=runtime_lock_path,
            source_lock_path=source_lock_path,
            expected_revision=expected_revision,
            history_path=history_path,
        )
    if capture_stdout.getvalue() or capture_stderr.getvalue():
        raise WorkerError("scientific runtime emitted unsolicited output")
    return canonical_json_bytes(
        {
            "record": record,
            "run_id": config["run_id"],
            "schema_version": 1,
            "study_id": STUDY_ID,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--round1-archive", type=Path, required=True)
    parser.add_argument("--round1-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerError("worker config is not JSON") from error
    if canonical_json_bytes(config) != config_bytes:
        raise WorkerError("worker config is not canonical JSON")
    packet = run_packet(
        config,
        repository_root=args.repository_root,
        round1_archive=args.round1_archive,
        round1_manifest=args.round1_manifest,
        runtime_lock_path=args.runtime_lock,
        source_lock_path=args.source_lock,
        expected_revision=args.revision,
        history_path=args.history,
    )
    sys.stdout.buffer.write(packet)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
