"""Exact committed-source boundary for feasibility-debt-clock-v3."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PATH = ROOT / "submission" / "submission.py"
CANDIDATE_PATH = ROOT / "experiments" / "candidates" / "feasibility_debt_clock_v3.py"
FIXTURE_PATH = (
    ROOT / "experiments" / "candidates" / "feasibility_debt_clock_v3_fixture.py"
)
WORKER_PATH = (
    ROOT / "experiments" / "candidates" / "feasibility_debt_clock_v3_worker.py"
)
PLAN_PATH = ROOT / "research" / "2026-09-01-feasibility-debt-clock-v3-plan.md"
PLAN_REVISION = "a61ba6003ec7cc5de5f41fc0c4349e62364ebd89"
PLAN_SHA256 = "1bf96ddd42c95dd9aa4ea516b1813929b6835f3949c4feb516fd2d7db62f57b8"
EXPECTED_OPTIMIZE_AST_SHA256 = (
    "85adb3dec2cfa83e0ed3ce6a6b826fd45fbcfe3d2c806dfdc44dd2711f2e990b"
)
EXPECTED_VALIDATOR_AST_SHA256 = (
    "40b6552a65518c97cff5cbb4d87647c6ee7cc7c303a0f5699eeff3315616db21"
)
EXPECTED_TRANSITION_AST_SHA256 = (
    "c4d1173d3235bc7dbb45439b1ff09c39503cedf84144fe8bdf5361fa16fb7f55"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one class named {name}")
    return matches[0]


def _methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    result: dict[str, ast.FunctionDef] = {}
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not isinstance(item, ast.FunctionDef) or item.name in result:
                raise RuntimeError("candidate class method layout is invalid")
            result[item.name] = item
    return result


def _ast_sha256(node: ast.AST) -> str:
    return _sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
    )


def _default_digest(defaults: list[ast.expr]) -> str:
    return _sha256(
        _canonical_json(
            [
                ast.dump(
                    item,
                    annotate_fields=True,
                    include_attributes=False,
                )
                for item in defaults
            ]
        )
    )


def _algorithm_string(node: ast.ClassDef) -> str:
    values = []
    for item in node.body:
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "algorithm_str"
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            values.append(item.value.value)
    if len(values) != 1:
        raise RuntimeError("algorithm_str assignment is not unique")
    return values[0]


def _validate_signature(
    protected: ast.FunctionDef, candidate: ast.FunctionDef
) -> None:
    protected_positional = [
        argument.arg
        for argument in protected.args.posonlyargs + protected.args.args
    ]
    candidate_positional = [
        argument.arg
        for argument in candidate.args.posonlyargs + candidate.args.args
    ]
    if protected_positional != candidate_positional:
        raise RuntimeError("candidate positional optimize ABI changed")
    if _default_digest(protected.args.defaults) != _default_digest(
        candidate.args.defaults
    ):
        raise RuntimeError("candidate positional optimize defaults changed")
    if [item.arg for item in candidate.args.kwonlyargs] != ["progress_mode"]:
        raise RuntimeError("candidate must add only keyword-only progress_mode")
    if candidate.args.kw_defaults != [None]:
        raise RuntimeError("candidate progress_mode must be required")
    if protected.args.vararg is not None or candidate.args.vararg is not None:
        raise RuntimeError("unexpected optimize varargs")
    protected_kwarg = protected.args.kwarg.arg if protected.args.kwarg else None
    candidate_kwarg = candidate.args.kwarg.arg if candidate.args.kwarg else None
    if protected_kwarg != "kwargs" or candidate_kwarg != "kwargs":
        raise RuntimeError("optimize kwargs ABI changed")


def source_projection() -> dict[str, Any]:
    if _sha256(PLAN_PATH.read_bytes()) != PLAN_SHA256:
        raise RuntimeError("frozen V3 plan hash mismatch")
    protected_tree = ast.parse(PROTECTED_PATH.read_text(encoding="utf-8"))
    candidate_tree = ast.parse(CANDIDATE_PATH.read_text(encoding="utf-8"))
    protected_class = _class(protected_tree, "BatchedRestartAdam")
    candidate_class = _class(
        candidate_tree, "FeasibilityDebtBatchedRestartAdamV3"
    )
    if _algorithm_string(protected_class) != "batched_restart_adam":
        raise RuntimeError("protected algorithm identity drifted")
    if (
        _algorithm_string(candidate_class)
        != "feasibility_debt_batched_restart_adam_v3"
    ):
        raise RuntimeError("candidate algorithm identity drifted")

    protected_methods = _methods(protected_class)
    candidate_methods = _methods(candidate_class)
    allowed_extra = {
        "_validate_feasibility_debt_aux",
        "_progress_transition",
    }
    if set(candidate_methods) != set(protected_methods) | allowed_extra:
        raise RuntimeError("candidate method surface differs from allowlist")
    for name in sorted(set(protected_methods) - {"optimize"}):
        if _ast_sha256(protected_methods[name]) != _ast_sha256(candidate_methods[name]):
            raise RuntimeError(f"protected method changed outside allowlist: {name}")
    _validate_signature(protected_methods["optimize"], candidate_methods["optimize"])

    optimize_digest = _ast_sha256(candidate_methods["optimize"])
    if optimize_digest != EXPECTED_OPTIMIZE_AST_SHA256:
        raise RuntimeError("candidate optimize AST differs from frozen implementation")
    validator_digest = _ast_sha256(
        candidate_methods["_validate_feasibility_debt_aux"]
    )
    transition_digest = _ast_sha256(candidate_methods["_progress_transition"])
    if validator_digest != EXPECTED_VALIDATOR_AST_SHA256:
        raise RuntimeError("candidate auxiliary validator AST drifted")
    if transition_digest != EXPECTED_TRANSITION_AST_SHA256:
        raise RuntimeError("candidate progress transition AST drifted")
    delta = {
        "allowed_class_name": "FeasibilityDebtBatchedRestartAdamV3",
        "allowed_algorithm_str": "feasibility_debt_batched_restart_adam_v3",
        "candidate_optimize_ast_sha256": optimize_digest,
        "validator_ast_sha256": validator_digest,
        "transition_ast_sha256": transition_digest,
        "unchanged_method_names": sorted(set(protected_methods) - {"optimize"}),
    }
    source_hashes = {
        "protected_source_sha256": _sha256(PROTECTED_PATH.read_bytes()),
        "candidate_source_sha256": _sha256(CANDIDATE_PATH.read_bytes()),
        "fixture_source_sha256": _sha256(FIXTURE_PATH.read_bytes()),
        "worker_source_sha256": _sha256(WORKER_PATH.read_bytes()),
        "plan_sha256": PLAN_SHA256,
    }
    return {
        **source_hashes,
        "normalized_delta_sha256": _sha256(_canonical_json(delta)),
        "source_boundary_root_sha256": _sha256(
            _canonical_json({"delta": delta, "sources": source_hashes})
        ),
    }
