"""Exact source-delta verifier for the frozen feasibility-debt candidate."""

from __future__ import annotations

import ast
import difflib
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PATH = ROOT / "submission" / "submission.py"
CANDIDATE_PATH = ROOT / "experiments" / "candidates" / "feasibility_debt_clock_v1.py"
PROTECTED_TEXT_SHA256 = "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
CANDIDATE_TEXT_SHA256 = "4abb914475a670c456882346ed474dd2078bdc16d2f43d39e6d01ee4e9493997"

# The complete unified-diff hunk payload is pinned, not merely its location.
# Region labels are the frozen semantic allowlist from the plan.
EXPECTED_HUNKS = (
    ("module_identity", "81628c3f5efcad13a687cfd7cfbc8680fcfe3113d9301e343f938b74bceb047c"),
    ("candidate_class_identity", "fe8af31b2db008467aa2646b0963375c443b813e4b35ea602af812673a48ab7b"),
    ("aux_validation", "9a4f8431f1b3881401216387aede3d5bc4698fe0e1c30d62d4743a61977e6864"),
    ("mode_validation", "6fcc939a400b7f31d2417e32f689d7082fc1c71f7ffe0d19fca63a3701c07b28"),
    ("progress_state_declaration", "c39e261ecec6b0249fda9f0985782395391a890f6a9a139e30a12b2e9c6132b7"),
    ("progress_transition", "f24ae0ba844ac28b9add0aecf0e51f0e62da0f6a22688f85e04337eabdb03c80"),
    ("progress_event_projection", "067270b897c41a40052671e1a5ff4d67f27f16ee3168934d3cb7d10483218c9d"),
    ("restart_progress_reset", "06dcc0e9049620f2c83f5bfa12fd3b681e41526fa583058382cf70cc58b62c2e"),
    ("telemetry_event_binding", "80048900ea22ea0d64282e1bbad4df2c133205a9fd8048de4f36a63a7dfba0b0"),
    ("telemetry_progress_extension", "c6c80b9915eec87b547e4a0a478756b52b81662d6b7b84b2fc22f50efa508338"),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hunk_hashes(protected: str, candidate: str) -> tuple[str, ...]:
    diff = difflib.unified_diff(
        protected.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile="protected",
        tofile="candidate",
        n=3,
    )
    hunks: list[str] = []
    current: list[str] = []
    for line in diff:
        if line.startswith("@@"):
            if current:
                hunks.append(_sha256_text("".join(current)))
            current = [line]
        elif current:
            current.append(line)
    if current:
        hunks.append(_sha256_text("".join(current)))
    return tuple(hunks)


def _class_methods(source: str, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name: item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
    raise ValueError(f"missing class {class_name}")


def verify_source_boundary() -> dict[str, Any]:
    protected = PROTECTED_PATH.read_text(encoding="utf-8")
    candidate = CANDIDATE_PATH.read_text(encoding="utf-8")
    protected_hash = _sha256_text(protected)
    candidate_hash = _sha256_text(candidate)
    observed_hunks = _hunk_hashes(protected, candidate)
    expected_hunks = tuple(value for _, value in EXPECTED_HUNKS)

    protected_methods = _class_methods(protected, "BatchedRestartAdam")
    candidate_methods = _class_methods(
        candidate, "FeasibilityDebtBatchedRestartAdam"
    )
    unchanged_methods = sorted(
        set(protected_methods).difference({"optimize"})
    )
    methods_equal = all(
        ast.dump(protected_methods[name], include_attributes=False)
        == ast.dump(candidate_methods[name], include_attributes=False)
        for name in unchanged_methods
    )
    exact_method_set = set(candidate_methods) == (
        set(protected_methods) | {"_validate_feasibility_debt_aux"}
    )
    valid = (
        protected_hash == PROTECTED_TEXT_SHA256
        and candidate_hash == CANDIDATE_TEXT_SHA256
        and observed_hunks == expected_hunks
        and methods_equal
        and exact_method_set
    )
    boundary_projection = {
        "candidate_text_sha256": candidate_hash,
        "exact_method_set": exact_method_set,
        "hunk_sha256": list(observed_hunks),
        "methods_equal": methods_equal,
        "protected_text_sha256": protected_hash,
        "region_labels": [name for name, _ in EXPECTED_HUNKS],
        "valid": valid,
    }
    return {
        **boundary_projection,
        "boundary_root_sha256": _sha256_text(
            str(sorted(boundary_projection.items()))
        ),
    }
