"""Deterministic public-interface fixtures for current constraint signals."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dfbench.core.objective import Objective
from dfbench.core.problem import ContinuousProblem
from dfbench.problems.base_problem import OpticalSetupProblem
from submission.submission import BatchedRestartAdam


STUDY_ID = "public-signal-surface-v1"
SCHEMA_VERSION = 1
DFBENCH_VERSION = "0.3.3"
DFBENCH_WHEEL_SHA256 = (
    "1f96d75b813ea42f93992da5c1f50d6a4f59dd7a507bcf561676b0e416378c43"
)
N_PARAMS = 3
CLAIM_BOUNDARY = "public_interface_current_state_only_no_predictive_or_performance_claim"
EXPECTED_SOURCE_SHA256 = {
    "objective": "9e2c2bb54517f59efacf4c2a59908ffd55c7fb2e15089d53263ece796e71daa2",
    "optical": "e7768eb3afd061b2684dbdb761e4211c9a9709852a54b45fff17d21a851ee95d",
    "uifo": "a6e3e95275987799761831a64a2c7b0aa793656d741df4d5d5e78b64c13f7d08",
}
SUBMISSION_SOURCE_SHA256 = (
    "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
)
AUX_TOP_LEVEL_FIELDS = [
    "is_feasible",
    "penalty",
    "power_values",
    "sensitivity_loss",
    "violations",
]
AUX_POWER_FIELDS = ["detector", "hard", "soft"]
AUX_LEAF_PATHS = [
    "is_feasible",
    "penalty",
    "power_values.detector",
    "power_values.hard",
    "power_values.soft",
    "sensitivity_loss",
    "violations",
]
POINTS = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0],
        [-0.5, 0.1, -0.25],
    ],
    dtype=jnp.float32,
)
CASE_CONTRACT = {
    "dependency_source_identity": {
        "dfbench_version": DFBENCH_VERSION,
        "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
        "objective_source_sha256": EXPECTED_SOURCE_SHA256["objective"],
        "optical_source_sha256": EXPECTED_SOURCE_SHA256["optical"],
        "uifo_source_sha256": EXPECTED_SOURCE_SHA256["uifo"],
    },
    "uifo_aux_schema": {
        "power_fields": AUX_POWER_FIELDS,
        "top_level_fields": AUX_TOP_LEVEL_FIELDS,
        "total_loss_components": ["penalty", "sensitivity_loss"],
    },
    "scalar_batch_roundtrip": {
        "aux_leaf_paths": AUX_LEAF_PATHS,
        "points": 4,
        "required_max_abs_difference": 0.0,
    },
    "candidate_passthrough_modes": {
        "aux_leaf_paths": AUX_LEAF_PATHS,
        "chunk_modes": ["full", "1", "2", "4"],
        "points": 4,
    },
    "infeasible_magnitude_control": {
        "feasible_control_index": 0,
        "infeasible_indices": [1, 2],
        "required_distinct_non_boolean_groups": [
            "penalty",
            "power_values",
            "violations",
        ],
    },
    "no_aux_negative_control": {
        "expected_error_type": "RuntimeError",
        "rich_aux_universal": False,
    },
    "consumer_boundary": {
        "available_aux_fields": AUX_LEAF_PATHS,
        "consumed_aux_fields": ["is_feasible"],
        "helper_aux_methods": [
            "value_and_grad_aux",
            "vmap_value_and_grad_aux",
        ],
    },
    "process_isolation": {
        "source_projection": "complete_non_process_cases",
        "workers": 2,
    },
}
REPOSITORY_ROOT = Path(__file__).parents[2]
PRIVATE_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT.with_name(f"{REPOSITORY_ROOT.name}-local-lab")
    / "worker-tmp"
    / "public-signal-surface-checkpoints"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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


POINTS_SHA256 = _array_sha256(POINTS)


class _SyntheticAuxProblem(ContinuousProblem):
    """Pure bounded problem exposing the same public aux shape as UIFO."""

    name = "synthetic_public_aux"

    def __init__(self) -> None:
        self.optimization_pairs = [
            ("synthetic", "x0"),
            ("synthetic", "x1"),
            ("synthetic", "x2"),
        ]

        def objective_function_aux(params):
            params = jnp.asarray(params, dtype=jnp.float32)
            center = jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float32)
            sensitivity_loss = jnp.sum(jnp.square(params - center))
            violations = jnp.maximum(
                jnp.asarray(
                    [params[0] - 0.2, -params[1] - 0.2, params[2] - 0.4],
                    dtype=jnp.float32,
                ),
                0.0,
            )
            penalty = jnp.sum(violations)
            aux = {
                "sensitivity_loss": sensitivity_loss,
                "penalty": penalty,
                "is_feasible": jnp.all(violations == 0.0),
                "violations": violations,
                "power_values": {
                    "hard": jnp.square(params[:1]) + 0.5,
                    "soft": jnp.square(params[1:2]) + 0.25,
                    "detector": jnp.square(params[2:3]) + 0.75,
                },
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


class _SyntheticNoAuxProblem(ContinuousProblem):
    """Negative control for problems without objective_function_aux."""

    name = "synthetic_no_aux"

    def __init__(self) -> None:
        self.optimization_pairs = [
            ("synthetic", "x0"),
            ("synthetic", "x1"),
            ("synthetic", "x2"),
        ]
        self.objective_function = jax.jit(lambda params: jnp.sum(jnp.square(params)))

    @property
    def bounds(self):
        return jnp.asarray(
            [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=jnp.float32
        )

    def to_spec(self) -> dict[str, object]:
        return {"type": type(self).__name__}


def _new_objective(*, aux: bool = True, max_evals: int = 16) -> Objective:
    problem = _SyntheticAuxProblem() if aux else _SyntheticNoAuxProblem()
    objective = Objective(
        problem,
        unbounded=False,
        max_evals=max_evals,
        save_time_steps=False,
        save_params_history=False,
        save=[],
        verbose=0,
        checkpoint_dir=PRIVATE_CHECKPOINT_ROOT,
    )
    objective.start_logging()
    return objective


def _source_paths() -> dict[str, Path]:
    objective_path = inspect.getsourcefile(Objective)
    optical_path = inspect.getsourcefile(OpticalSetupProblem)
    uifo_spec = importlib.util.find_spec("dfbench.problems.uifo.uifo_problem")
    if objective_path is None or optical_path is None:
        raise RuntimeError("public dfbench source files are unavailable")
    if uifo_spec is None or uifo_spec.origin is None:
        raise RuntimeError("public UIFO source file is unavailable")
    return {
        "objective": Path(objective_path).resolve(),
        "optical": Path(optical_path).resolve(),
        "uifo": Path(uifo_spec.origin).resolve(),
    }


def _locked_dependency_projection() -> dict[str, str]:
    with (REPOSITORY_ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    packages = [
        package
        for package in lock.get("package", [])
        if package.get("name") == "dfbench"
    ]
    if len(packages) != 1:
        raise RuntimeError("dependency lock has no unique dfbench package")
    package = packages[0]
    wheel_hashes = [
        wheel.get("hash", "").removeprefix("sha256:")
        for wheel in package.get("wheels", [])
    ]
    if len(wheel_hashes) != 1:
        raise RuntimeError("dependency lock has no unique dfbench wheel")
    return {
        "dfbench_version": str(package.get("version")),
        "dfbench_wheel_sha256": wheel_hashes[0],
    }


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise RuntimeError(f"authenticated source is missing {class_name}.{method_name}")


def _nested_function(method: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in method.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"authenticated source is missing nested function {name}")


def _string_dict_keys(node: ast.Dict) -> list[str]:
    keys = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise RuntimeError("authenticated aux schema has a non-string key")
        keys.append(key.value)
    return sorted(keys)


def _returned_dict(function: ast.FunctionDef) -> ast.Dict:
    for node in function.body:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return node.value
    raise RuntimeError("authenticated aux builder has no direct dict return")


def _aux_schema_projection(paths: dict[str, Path]) -> dict[str, object]:
    optical_tree = ast.parse(paths["optical"].read_text(encoding="utf-8"))
    uifo_tree = ast.parse(paths["uifo"].read_text(encoding="utf-8"))
    build_aux = _class_method(optical_tree, "OpticalSetupProblem", "_build_aux")
    aux_dict = _returned_dict(build_aux)
    top_level = _string_dict_keys(aux_dict)
    power_index = next(
        index
        for index, key in enumerate(aux_dict.keys)
        if isinstance(key, ast.Constant) and key.value == "power_values"
    )
    power_node = aux_dict.values[power_index]
    if not isinstance(power_node, ast.Dict):
        raise RuntimeError("authenticated power_values schema is not a dict")
    power_fields = _string_dict_keys(power_node)

    build_objective = _class_method(
        uifo_tree, "UIFOProblem", "_build_objective_function"
    )
    scalar_function = _nested_function(build_objective, "objective_function")
    aux_function = _nested_function(build_objective, "objective_function_aux")
    scalar_return = next(
        node.value
        for node in scalar_function.body
        if isinstance(node, ast.Return)
    )
    total_components = sorted(
        node.id for node in ast.walk(scalar_return) if isinstance(node, ast.Name)
    )
    shared_aux_calls = [
        node
        for node in ast.walk(aux_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_build_aux"
    ]
    return {
        "power_fields": power_fields,
        "top_level_fields": top_level,
        "total_loss_components": total_components,
        "uses_shared_aux_builder": len(shared_aux_calls) == 1,
    }


def _leaf_paths(tree: object) -> list[str]:
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths = []
    for path, _leaf in paths_and_leaves:
        parts = []
        for entry in path:
            key = getattr(entry, "key", getattr(entry, "idx", None))
            parts.append(str(key))
        paths.append(".".join(parts))
    return sorted(paths)


def _tree_digest(*trees: object) -> str:
    digest = hashlib.sha256()
    for tree_index, tree in enumerate(trees):
        leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
        digest.update(f"tree:{tree_index}\0".encode("ascii"))
        for path, leaf in leaves_with_paths:
            rendered_path = ".".join(
                str(getattr(entry, "key", getattr(entry, "idx", None)))
                for entry in path
            )
            digest.update(rendered_path.encode("utf-8") + b"\0")
            digest.update(_array_bytes(leaf))
    return digest.hexdigest()


def _max_abs_tree_difference(left: object, right: object) -> float:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return float("inf")
    largest = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if left_array.shape != right_array.shape:
            return float("inf")
        if left_array.dtype == np.bool_ or right_array.dtype == np.bool_:
            if not np.array_equal(left_array, right_array):
                return 1.0
            continue
        if left_array.size:
            largest = max(
                largest,
                float(np.max(np.abs(left_array - right_array))),
            )
    return largest


def _roundtrip_projection() -> dict[str, object]:
    scalar_objective = _new_objective(max_evals=8)
    scalar_values = []
    scalar_grads = []
    scalar_aux = []
    for point in POINTS:
        value, grad, aux = scalar_objective.value_and_grad_aux(point)
        scalar_values.append(value)
        scalar_grads.append(grad)
        scalar_aux.append(aux)
    stacked_values = jnp.stack(scalar_values)
    stacked_grads = jnp.stack(scalar_grads)
    stacked_aux = jax.tree.map(lambda *values: jnp.stack(values), *scalar_aux)

    batch_objective = _new_objective(max_evals=8)
    batch_values, batch_grads, batch_aux = batch_objective.vmap_value_and_grad_aux(
        POINTS
    )
    return {
        "aux_leaf_paths": _leaf_paths(batch_aux),
        "max_abs_aux_difference": _max_abs_tree_difference(
            stacked_aux, batch_aux
        ),
        "max_abs_gradient_difference": _max_abs_tree_difference(
            stacked_grads, batch_grads
        ),
        "max_abs_loss_difference": _max_abs_tree_difference(
            stacked_values, batch_values
        ),
        "points": int(POINTS.shape[0]),
        "projection_sha256": _tree_digest(batch_values, batch_grads, batch_aux),
    }


def _candidate_projection() -> dict[str, object]:
    modes: list[tuple[str, int | None]] = [
        ("full", None),
        ("1", 1),
        ("2", 2),
        ("4", 4),
    ]
    digests = []
    observed_paths = []
    for label, chunk_size in modes:
        objective = _new_objective(max_evals=8)
        values, grads, aux = BatchedRestartAdam._evaluate_population(
            objective,
            POINTS,
            chunk_size,
        )
        digests.append({"mode": label, "sha256": _tree_digest(values, grads, aux)})
        observed_paths.append(_leaf_paths(aux))
    return {
        "aux_leaf_paths": observed_paths[0],
        "chunk_modes": [label for label, _ in modes],
        "mode_projection_sha256": _json_sha256(digests),
        "modes_identical": len({row["sha256"] for row in digests}) == 1,
        "paths_identical": all(paths == observed_paths[0] for paths in observed_paths),
        "points": int(POINTS.shape[0]),
    }


def _magnitude_projection() -> dict[str, object]:
    objective = _new_objective(max_evals=8)
    _values, _grads, aux = objective.vmap_value_and_grad_aux(POINTS)
    feasible = np.asarray(jax.device_get(aux["is_feasible"]), dtype=bool)
    penalties = np.asarray(jax.device_get(aux["penalty"]))
    violations = np.asarray(jax.device_get(aux["violations"]))
    powers = {
        name: np.asarray(jax.device_get(value))
        for name, value in aux["power_values"].items()
    }
    left, right = 1, 2
    return {
        "feasible_control_present": bool(feasible[0]),
        "infeasible_points": 2,
        "penalties_differ": bool(penalties[left] != penalties[right]),
        "power_values_differ": any(
            not np.array_equal(value[left], value[right])
            for value in powers.values()
        ),
        "same_infeasible_boolean": bool(
            not feasible[left] and not feasible[right]
        ),
        "violations_differ": not np.array_equal(
            violations[left], violations[right]
        ),
    }


def _negative_control_projection() -> dict[str, object]:
    objective = _new_objective(aux=False, max_evals=2)
    error_type = "none"
    message_identifies_aux_contract = False
    try:
        objective.value_and_grad_aux(POINTS[0])
    except RuntimeError as error:
        error_type = type(error).__name__
        message_identifies_aux_contract = "does not expose objective_function_aux" in str(
            error
        )
    return {
        "error_type": error_type,
        "message_identifies_aux_contract": message_identifies_aux_contract,
        "rich_aux_universal": False,
    }


def _aux_subscript_fields(function: ast.FunctionDef) -> list[str]:
    fields = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "aux":
            continue
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            fields.add(index.value)
    return sorted(fields)


def _consumer_projection() -> dict[str, object]:
    source_path = REPOSITORY_ROOT / "submission" / "submission.py"
    if _normalized_text_sha256(source_path) != SUBMISSION_SOURCE_SHA256:
        raise RuntimeError("protected optimizer source identity changed")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    optimize = _class_method(tree, "BatchedRestartAdam", "optimize")
    helper = _class_method(tree, "BatchedRestartAdam", "_evaluate_population")
    consumed = _aux_subscript_fields(optimize)
    helper_methods = sorted(
        {
            node.attr
            for node in ast.walk(helper)
            if isinstance(node, ast.Attribute)
            and node.attr in {"value_and_grad_aux", "vmap_value_and_grad_aux"}
        }
    )
    return {
        "available_aux_fields": AUX_LEAF_PATHS,
        "consumed_aux_fields": consumed,
        "helper_aux_methods": helper_methods,
        "unused_available_fields": [
            field for field in AUX_LEAF_PATHS if field not in consumed
        ],
    }


def isolated_worker_trace() -> dict[str, object]:
    """Return the complete deterministic projection used for isolation."""
    paths = _source_paths()
    return {
        "candidate_passthrough": _candidate_projection(),
        "consumer_boundary": _consumer_projection(),
        "dependency": {
            "dfbench_version": importlib.metadata.version("dfbench"),
            "locked": _locked_dependency_projection(),
            "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        },
        "infeasible_magnitude": _magnitude_projection(),
        "negative_control": _negative_control_projection(),
        "roundtrip": _roundtrip_projection(),
        "uifo_aux_schema": _aux_schema_projection(paths),
    }


def _isolated_trace() -> dict[str, object]:
    safe_names = {
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
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
            "experiments.local_lab.public_signal_surface_worker",
            "--mode",
            "public-signal-surface-trace",
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
        raise RuntimeError("the signal-surface study requires a CPU backend")

    projection = isolated_worker_trace()
    dependency = projection["dependency"]
    schema = projection["uifo_aux_schema"]
    roundtrip = projection["roundtrip"]
    candidate = projection["candidate_passthrough"]
    magnitude = projection["infeasible_magnitude"]
    negative = projection["negative_control"]
    consumer = projection["consumer_boundary"]

    source_hashes = dependency["source_sha256"]
    dependency_passed = (
        dependency["dfbench_version"] == DFBENCH_VERSION
        and dependency["locked"]
        == {
            "dfbench_version": DFBENCH_VERSION,
            "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
        }
        and source_hashes == EXPECTED_SOURCE_SHA256
    )
    schema_passed = schema == {
        "power_fields": AUX_POWER_FIELDS,
        "top_level_fields": AUX_TOP_LEVEL_FIELDS,
        "total_loss_components": ["penalty", "sensitivity_loss"],
        "uses_shared_aux_builder": True,
    }
    roundtrip_passed = (
        roundtrip["aux_leaf_paths"] == AUX_LEAF_PATHS
        and roundtrip["points"] == int(POINTS.shape[0])
        and roundtrip["max_abs_aux_difference"] == 0.0
        and roundtrip["max_abs_gradient_difference"] == 0.0
        and roundtrip["max_abs_loss_difference"] == 0.0
    )
    candidate_passed = (
        candidate["aux_leaf_paths"] == AUX_LEAF_PATHS
        and candidate["chunk_modes"] == ["full", "1", "2", "4"]
        and candidate["modes_identical"]
        and candidate["paths_identical"]
        and candidate["points"] == int(POINTS.shape[0])
    )
    magnitude_passed = all(
        bool(magnitude[name])
        for name in (
            "feasible_control_present",
            "penalties_differ",
            "power_values_differ",
            "same_infeasible_boolean",
            "violations_differ",
        )
    ) and magnitude["infeasible_points"] == 2
    negative_passed = (
        negative["error_type"] == "RuntimeError"
        and negative["message_identifies_aux_contract"]
        and not negative["rich_aux_universal"]
    )
    consumer_passed = consumer == {
        "available_aux_fields": AUX_LEAF_PATHS,
        "consumed_aux_fields": ["is_feasible"],
        "helper_aux_methods": [
            "value_and_grad_aux",
            "vmap_value_and_grad_aux",
        ],
        "unused_available_fields": [
            field for field in AUX_LEAF_PATHS if field != "is_feasible"
        ],
    }

    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = isolated_left == isolated_right == projection
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = "not-run-in-focused-test"

    cases = {
        "dependency_source_identity": {
            "dfbench_version": dependency["dfbench_version"],
            "dfbench_wheel_sha256": dependency["locked"][
                "dfbench_wheel_sha256"
            ],
            "objective_source_sha256": source_hashes["objective"],
            "optical_source_sha256": source_hashes["optical"],
            "passed": dependency_passed,
            "uifo_source_sha256": source_hashes["uifo"],
        },
        "uifo_aux_schema": {
            "passed": schema_passed,
            "power_fields": schema["power_fields"],
            "top_level_fields": schema["top_level_fields"],
            "total_loss_components": schema["total_loss_components"],
            "uses_shared_aux_builder": schema["uses_shared_aux_builder"],
        },
        "scalar_batch_roundtrip": {
            "aux_leaf_paths": roundtrip["aux_leaf_paths"],
            "max_abs_aux_difference": roundtrip["max_abs_aux_difference"],
            "max_abs_gradient_difference": roundtrip[
                "max_abs_gradient_difference"
            ],
            "max_abs_loss_difference": roundtrip["max_abs_loss_difference"],
            "passed": roundtrip_passed,
            "points": roundtrip["points"],
            "projection_sha256": roundtrip["projection_sha256"],
        },
        "candidate_passthrough_modes": {
            "aux_leaf_paths": candidate["aux_leaf_paths"],
            "chunk_modes": candidate["chunk_modes"],
            "mode_projection_sha256": candidate["mode_projection_sha256"],
            "modes_identical": candidate["modes_identical"],
            "passed": candidate_passed,
            "paths_identical": candidate["paths_identical"],
            "points": candidate["points"],
        },
        "infeasible_magnitude_control": {
            "feasible_control_present": magnitude["feasible_control_present"],
            "infeasible_points": magnitude["infeasible_points"],
            "passed": magnitude_passed,
            "penalties_differ": magnitude["penalties_differ"],
            "power_values_differ": magnitude["power_values_differ"],
            "same_infeasible_boolean": magnitude["same_infeasible_boolean"],
            "violations_differ": magnitude["violations_differ"],
        },
        "no_aux_negative_control": {
            "error_type": negative["error_type"],
            "message_identifies_aux_contract": negative[
                "message_identifies_aux_contract"
            ],
            "passed": negative_passed,
            "rich_aux_universal": negative["rich_aux_universal"],
        },
        "consumer_boundary": {
            "available_aux_fields": consumer["available_aux_fields"],
            "consumed_aux_fields": consumer["consumed_aux_fields"],
            "helper_aux_methods": consumer["helper_aux_methods"],
            "passed": consumer_passed,
            "unused_available_fields": consumer["unused_available_fields"],
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
            "public_current_constraint_signals_confirmed"
            if passed
            else (
                "park_public_signal_surface_research"
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
            "aux_power_fields": AUX_POWER_FIELDS,
            "aux_top_level_fields": AUX_TOP_LEVEL_FIELDS,
            "case_contract": CASE_CONTRACT,
            "claim_boundary": CLAIM_BOUNDARY,
            "dfbench_version": DFBENCH_VERSION,
            "dfbench_wheel_sha256": DFBENCH_WHEEL_SHA256,
            "n_params": N_PARAMS,
            "objective_source_sha256": EXPECTED_SOURCE_SHA256["objective"],
            "optical_source_sha256": EXPECTED_SOURCE_SHA256["optical"],
            "points_sha256": POINTS_SHA256,
            "submission_source_sha256": SUBMISSION_SOURCE_SHA256,
            "uifo_source_sha256": EXPECTED_SOURCE_SHA256["uifo"],
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
