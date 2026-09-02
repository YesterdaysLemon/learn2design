"""Exact source-delta verifier for the frozen feasibility-debt v2 candidate."""

from __future__ import annotations

import ast
import difflib
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PATH = ROOT / "submission" / "submission.py"
CANDIDATE_PATH = ROOT / "experiments" / "candidates" / "feasibility_debt_clock_v2.py"
PROTECTED_TEXT_SHA256 = "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
CANDIDATE_TEXT_SHA256 = "bb827efa045db38146af0cf2c9bfdd6b30db8b9668d088cd658d44a3ae030c15"

# The complete unified-diff hunk payload is pinned, not merely its location.
# Region labels are the frozen semantic allowlist from the plan.
EXPECTED_HUNKS = (
    ("module_identity", "8a7892661e995f682d49a991c45cd8792e873b3a68ab2dae1f0c8ac17ccd3860"),
    ("candidate_class_identity", "50f2a6a9e8fdeccd960595646d12c0db3706f71e1598465951f25c32d64d1c66"),
    ("aux_validation", "98b33252877c3916dfbdcf1fb5f54e576232fb580a722a348ee081a77e7b8417"),
    ("mode_validation", "bd769673d724f6bf220faf9c42860d2750deb8f9a0f5adeb3599dc973a372438"),
    ("progress_state_declaration", "1aa7ca6c83e37f0f9dd75f0af2f9789e25db27152220d01468a270982d5feb2b"),
    ("progress_transition", "984ffd3e7107ba681eb47b3c8742c0c49c62078960995f520a0c2f4d7523b52f"),
    ("progress_event_projection", "1bb5f9ea6b1ed4adb3cba1a91679c00f6fb55af37d8aab505b6ef0a2d56274bf"),
    ("restart_progress_reset", "33da15287fb2e83e3b296fc8dc1dbcf77be4c4e45248be4fa76ae3cca24a4b38"),
    ("telemetry_event_binding", "fb8c8f7b76061fc469621aa938e8239ae0e6c17e2abfecbba51de91a0374534a"),
    ("telemetry_progress_extension", "588b42091ca43c06555393040782353ea02ec1ddf485f38fdb1fccac2f4ae4f7"),
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
        candidate, "FeasibilityDebtBatchedRestartAdamV2"
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
