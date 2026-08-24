"""Fail-closed agreement checks for H100 coverage replays."""

from __future__ import annotations

import hashlib
import json
import math
from types import MappingProxyType
from typing import Protocol

from experiments.uifo_paired.coverage_profiles import (
    CoverageProfileSpec,
    coverage_profile_spec,
)
from experiments.uifo_paired.results_ingestion import StudyValidationError


RUNS = 48
TOPOLOGIES = 12
CRITERIA = 13


class CoverageReplayStudy(Protocol):
    """Study identity needed to bind a replay agreement to one archive."""

    source_hashes: dict[str, str]
    plan: dict[str, object]
    manifest: dict[str, object]
    configs: dict[str, dict[str, object]]


def coverage_study_identity_sha256(study: CoverageReplayStudy) -> str:
    """Hash the authenticated study identity without opening its summary."""
    configuration = study.plan.get("configuration")
    evidence = (
        configuration.get("candidate_package_evidence")
        if isinstance(configuration, dict)
        else None
    )
    project_revision = study.manifest.get("project_revision")
    if project_revision is None and isinstance(evidence, dict):
        project_revision = evidence.get("project_revision")
    identity = {
        "plan_id": study.plan.get("plan_id"),
        "project_revision": project_revision,
        "source_hashes": study.source_hashes,
        "run_ids": sorted(study.configs),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compare(left: object, right: object, path: str) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise StudyValidationError(f"coverage replay mismatch at {path}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(
                float(left), float(right), rel_tol=1e-11, abs_tol=1e-12
            )
        ):
            raise StudyValidationError(f"coverage numeric mismatch at {path}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise StudyValidationError(f"coverage schema mismatch at {path}")
        for key in sorted(left):
            _compare(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise StudyValidationError(f"coverage list length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]")
        return
    if left != right or type(left) is not type(right):
        raise StudyValidationError(f"coverage replay mismatch at {path}")


def _validate_shape(
    value: dict[str, object], label: str
) -> CoverageProfileSpec:
    try:
        specification = coverage_profile_spec(value.get("study_profile"))
    except ValueError as error:
        raise StudyValidationError(f"{label} profile is invalid") from error
    if value.get("completed_runs") != specification.runs:
        raise StudyValidationError(
            f"{label} does not contain {specification.runs} completed runs"
        )
    topologies = value.get("topology_differences")
    pairs = value.get("optimizer_seed_pair_rows")
    decision = value.get("predeclared_decision")
    if not isinstance(topologies, list) or len(topologies) != specification.topologies:
        raise StudyValidationError(f"{label} topology table is incomplete")
    if not isinstance(pairs, list) or len(pairs) != specification.pairs:
        raise StudyValidationError(f"{label} pair table is incomplete")
    if not isinstance(decision, dict) or not isinstance(
        decision.get("criteria"), dict
    ):
        raise StudyValidationError(f"{label} decision receipt is missing")
    if len(decision["criteria"]) != specification.criteria:
        raise StudyValidationError(f"{label} decision criteria count drifted")
    return specification


def _build_replay_comparator():
    """Keep the issuance token outside the module namespace.

    This is an accidental-misuse barrier, not a cryptographic trust boundary.
    The detached Stage-A summary commitment supplies the durable pre-result
    binding.
    """
    seal = object()

    class _CoverageReplayAgreement:
        __slots__ = ("_payload",)

        def __init__(self, payload: dict[str, object], *, _seal: object) -> None:
            if _seal is not seal:
                raise TypeError("coverage replay agreements are comparator-issued")
            self._payload = MappingProxyType(dict(payload))

        def as_dict(self) -> dict[str, object]:
            return dict(self._payload)

    def _compare_replays(
        production: dict[str, object],
        reference: dict[str, object],
        *,
        study: CoverageReplayStudy,
    ) -> _CoverageReplayAgreement:
        production_spec = _validate_shape(
            production, "production coverage replay"
        )
        reference_spec = _validate_shape(reference, "reference coverage replay")
        if production_spec != reference_spec:
            raise StudyValidationError("coverage replay profile mismatch")
        _compare(production, reference, "coverage")
        if production.get("run_ids") != sorted(study.configs):
            raise StudyValidationError(
                "coverage replay run IDs do not match the authenticated study"
            )
        return _CoverageReplayAgreement(
            {
                "status": "matched",
                "runs_compared": production_spec.runs,
                "topology_values_compared": production_spec.topologies,
                "optimizer_seed_pairs_compared": production_spec.pairs,
                "frozen_criteria_compared": production_spec.criteria,
                "study_identity_sha256": coverage_study_identity_sha256(study),
            },
            _seal=seal,
        )

    return _CoverageReplayAgreement, _compare_replays


CoverageReplayAgreement, compare_coverage_replays = _build_replay_comparator()
del _build_replay_comparator


def compare_coverage_archived_summary(
    production: dict[str, object],
    reference: dict[str, object],
    archived: dict[str, object],
) -> dict[str, object]:
    """Require archived summary agreement after the two raw replays match."""
    production_spec = _validate_shape(production, "production coverage replay")
    reference_spec = _validate_shape(reference, "reference coverage replay")
    archived_spec = _validate_shape(archived, "archived coverage summary")
    if not production_spec == reference_spec == archived_spec:
        raise StudyValidationError("coverage archived-summary profile mismatch")
    _compare(production, archived, "coverage.archived_vs_production")
    _compare(reference, archived, "coverage.archived_vs_reference")
    return {
        "status": "matched",
        "runs_compared": production_spec.runs,
        "topology_values_compared": production_spec.topologies,
        "optimizer_seed_pairs_compared": production_spec.pairs,
        "frozen_criteria_compared": production_spec.criteria,
        "archived_summary_compared": True,
    }
