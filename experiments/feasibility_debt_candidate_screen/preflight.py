"""Opaque clean-surface authorization after all pre-result locks authenticate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical import ReceiptError, read_receipt, sha256_file
from .contract import (
    OFFICIAL_ARCHIVE_SHA256,
    OPTIMIZER_SETTINGS_SHA256,
    PYPROJECT_SHA256,
    REFERENCE_GENERATOR_SHA256,
    STUDY_ID,
    UV_LOCK_SHA256,
)
from .locks import read_runtime_lock, read_source_lock
from .orchestrator import LockDigests, build_stage1_configs
from .source_closure import logical_source_closure


class PreflightError(RuntimeError):
    pass


_SENTINEL = object()


@dataclass(frozen=True)
class PreflightAuthorization:
    revision: str
    panel_sha256: str
    split_receipt_sha256: str
    ci_evidence_sha256: str
    locks: LockDigests
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _SENTINEL:
            raise PreflightError("preflight tokens are validator-issued only")


PANEL_COMMITMENT_KEYS = {
    "panel_id",
    "panel_sha256",
    "official_archive",
    "prior_panels",
    "candidate_seed_start",
    "candidate_seed_attempts",
    "eligible_unique_candidates",
    "archive_overlap_count",
    "prior_panel_overlap_count",
    "smoke_topology_seed",
    "smoke_topology_sha256",
    "smoke_overlap_count",
    "upstream_reference",
}
SPLIT_KEYS = {
    "panel_sha256",
    "candidate_rows",
    "legal_split_rows",
    "chosen_stage1_indices",
    "chosen_stage2_indices",
    "stratum_counts",
    "independent_verification",
}


def _verify_frozen_repository_sources(repository_root: Path) -> None:
    expected = {
        "pyproject.toml": PYPROJECT_SHA256,
        "uv.lock": UV_LOCK_SHA256,
        "experiments/uifo_paired/optimizer_settings.py": OPTIMIZER_SETTINGS_SHA256,
        "tools/build_topology_panels.py": REFERENCE_GENERATOR_SHA256,
    }
    for relative, digest in expected.items():
        if sha256_file(repository_root / relative) != digest:
            raise PreflightError(f"frozen repository source drift: {relative}")


def _verify_backup(
    primary: Path,
    backup: Path,
    *,
    repository_root: Path,
    panel_sha256: str,
) -> None:
    required = (
        "panel.json",
        "panel-commitment.json",
        "panel-commitment.json.sha256",
        "split-receipt.json",
        "split-receipt.json.sha256",
    )
    repository = repository_root.resolve()
    if (
        primary.resolve() == backup.resolve()
        or primary.resolve() == repository
        or primary.resolve().is_relative_to(repository)
        or backup.resolve() == repository
        or backup.resolve().is_relative_to(repository)
        or backup.name != panel_sha256
    ):
        raise PreflightError("panel backup is not a distinct location")
    for name in required:
        left = primary / name
        right = backup / name
        if not left.is_file() or not right.is_file() or left.read_bytes() != right.read_bytes():
            raise PreflightError(f"panel backup mismatch: {name}")


def authenticate_pre_result_surface(
    *,
    repository_root: Path,
    expected_revision: str,
    observed_revision: str,
    worktree_clean: bool,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    panel_bundle_dir: Path,
    panel_backup_dir: Path,
    ci_evidence_path: Path,
) -> PreflightAuthorization:
    if observed_revision != expected_revision or not worktree_clean:
        raise PreflightError("revision/worktree pre-result guard failed")
    _verify_frozen_repository_sources(repository_root)
    runtime, runtime_digest = read_runtime_lock(runtime_lock_path)
    logical_sources = logical_source_closure(
        repository_root, round1_archive, round1_manifest
    )
    source, source_digest = read_source_lock(
        source_lock_path,
        runtime_lock_sha256=runtime_digest,
        logical_sources=logical_sources,
        expected_revision=expected_revision,
    )
    if any(
        row.get("package_closure_sha256") != runtime["package_closure_sha256"]
        for row in source["arm_profiles"]
    ):
        raise PreflightError("arm/runtime package-closure binding mismatch")
    expected_component_hashes = {
        "worker_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/worker.py"
        ),
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
    if any(source[key] != value for key, value in expected_component_hashes.items()):
        raise PreflightError("source lock component digest mismatch")
    commitment, commitment_digest = read_receipt(
        panel_bundle_dir / "panel-commitment.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="panel_commitment",
        expected_payload_keys=PANEL_COMMITMENT_KEYS,
    )
    if source["panel_commitment_sha256"] != commitment_digest:
        raise PreflightError("source lock panel commitment mismatch")
    archive_row = commitment["official_archive"]
    if (
        not isinstance(archive_row, dict)
        or archive_row.get("sha256") != OFFICIAL_ARCHIVE_SHA256
        or commitment["archive_overlap_count"] != 0
        or commitment["prior_panel_overlap_count"] != 0
        or commitment["smoke_overlap_count"] != 0
    ):
        raise PreflightError("panel exclusion commitment mismatch")
    split, split_digest = read_receipt(
        panel_bundle_dir / "split-receipt.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="split_receipt",
        expected_payload_keys=SPLIT_KEYS,
    )
    panel_path = panel_bundle_dir / "panel.json"
    panel_digest = sha256_file(panel_path)
    if (
        commitment["panel_sha256"] != panel_digest
        or split["panel_sha256"] != panel_digest
        or split["independent_verification"].get("status") != "matched"
    ):
        raise PreflightError("panel/split digest or replay mismatch")
    _verify_backup(
        panel_bundle_dir,
        panel_backup_dir,
        repository_root=repository_root,
        panel_sha256=panel_digest,
    )
    ci, ci_digest = read_receipt(
        ci_evidence_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="ci_evidence",
        expected_payload_keys={
            "revision",
            "pull_request",
            "required_jobs",
            "all_required_green",
            "authenticated_response_sha256",
        },
    )
    jobs = ci["required_jobs"]
    response_digest = ci["authenticated_response_sha256"]
    if (
        ci["revision"] != expected_revision
        or type(ci["pull_request"]) is not int
        or ci["pull_request"] < 1
        or not isinstance(jobs, list)
        or not jobs
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "conclusion"}
            or not isinstance(row["name"], str)
            or not row["name"]
            or row["conclusion"] != "SUCCESS"
            for row in jobs
        )
        or [row["name"] for row in jobs] != sorted(row["name"] for row in jobs)
        or ci["all_required_green"] is not True
        or not isinstance(response_digest, str)
        or len(response_digest) != 64
        or any(token not in "0123456789abcdef" for token in response_digest)
    ):
        raise PreflightError("CI evidence is not green and revision-bound")
    return PreflightAuthorization(
        revision=expected_revision,
        panel_sha256=panel_digest,
        split_receipt_sha256=split_digest,
        ci_evidence_sha256=ci_digest,
        locks=LockDigests(
            source_lock_sha256=source_digest,
            runtime_lock_sha256=runtime_digest,
            revision=expected_revision,
            package_closure_sha256=str(runtime["package_closure_sha256"]),
            panel_commitment_sha256=commitment_digest,
        ),
        _sentinel=_SENTINEL,
    )


def authorized_stage1_configs(
    authorization: PreflightAuthorization,
    *,
    panel_path: Path,
    split_receipt_path: Path,
) -> list[dict[str, object]]:
    if authorization._sentinel is not _SENTINEL:
        raise PreflightError("Stage 1 lacks authentic pre-result authorization")
    if (
        sha256_file(panel_path) != authorization.panel_sha256
        or sha256_file(split_receipt_path) != authorization.split_receipt_sha256
    ):
        raise PreflightError("authorized panel/split bytes changed")
    return build_stage1_configs(
        panel_path=panel_path,
        panel_commitment_path=panel_path.parent / "panel-commitment.json",
        split_receipt_path=split_receipt_path,
        locks=authorization.locks,
    )
