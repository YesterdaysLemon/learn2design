"""Streaming, append-only evacuation of sealed attempt evidence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .canonical import canonical_json_bytes, sha256_file, write_receipt
from .cleanup import (
    EvidenceHandoffBinding,
    EvidenceEvacuationAuthorization,
    ProcessCleanupAuthorization,
    assert_process_cleanup,
    authenticate_evidence_evacuation,
)
from .contract import STUDY_ID
from .orchestrator import TerminalOutcomeAuthorization, assert_terminal_outcome


class EvacuationError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_files(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in {
            "post-cleanup",
            "terminal-evidence-index.json",
            "terminal-evidence-index.json.sha256",
        }:
            raise EvacuationError("pod evidence occupies a host-reserved path")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise EvacuationError(f"evidence tree contains a non-regular member: {relative}")
        rows.append((relative, path))
    if not rows:
        raise EvacuationError("evidence tree is empty")
    return rows


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    _fsync_directory(destination.parent)
    return digest.hexdigest(), size


def evacuate_and_authenticate(
    *,
    source_attempt_root: Path,
    destination_root: Path,
    source_terminal_path: Path,
    process_cleanup: ProcessCleanupAuthorization,
    process_cleanup_receipt_path: Path,
    terminal_authorization: TerminalOutcomeAuthorization,
    pod_error_code: str | None,
    binding: EvidenceHandoffBinding,
) -> EvidenceEvacuationAuthorization:
    process_cleanup = assert_process_cleanup(process_cleanup)
    terminal_authorization = assert_terminal_outcome(terminal_authorization)
    if pod_error_code is not None and (
        not isinstance(pod_error_code, str) or not pod_error_code
    ):
        raise EvacuationError("pod error code is invalid")
    source = source_attempt_root.resolve(strict=True)
    destination = destination_root.resolve()
    if (
        destination.exists()
        or destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise EvacuationError("evidence destination is not a fresh independent tree")
    if not source_terminal_path.resolve(strict=True).is_relative_to(source):
        raise EvacuationError("terminal index is outside the source attempt")
    if (
        not process_cleanup_receipt_path.resolve(strict=True).is_relative_to(source)
        or sha256_file(process_cleanup_receipt_path)
        != process_cleanup.receipt_sha256
    ):
        raise EvacuationError("process-cleanup receipt is outside or changed")
    terminal_sha256 = sha256_file(source_terminal_path)
    if terminal_sha256 != terminal_authorization.receipt_sha256:
        raise EvacuationError("terminal authorization differs from source bytes")
    members = _source_files(source)
    destination.mkdir(parents=True, exist_ok=False)
    _fsync_directory(destination.parent)
    rows: list[dict[str, object]] = []
    for logical_id, source_path in members:
        target = destination.joinpath(*logical_id.split("/"))
        digest, size = _copy_and_hash(source_path, target)
        if digest != sha256_file(source_path) or digest != sha256_file(target):
            raise EvacuationError("streaming evidence copy did not verify")
        rows.append(
            {
                "logical_id": logical_id,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if not any(row["sha256"] == terminal_sha256 for row in rows):
        raise EvacuationError("terminal index is absent from the copied evidence")
    manifest_path = destination / "evidence-destination-manifest.json"
    manifest_sha256 = write_receipt(
        manifest_path,
        study_id=STUDY_ID,
        receipt_type="evidence_destination_manifest",
        payload={
            "terminal_index_sha256": terminal_sha256,
            "process_cleanup_receipt_sha256": process_cleanup.receipt_sha256,
            "member_count": len(rows),
            "member_rows": rows,
            "source_tree_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
            "raw_paths_included": False,
        },
    )
    receipt_path = destination / "pod-handoff" / "evidence-evacuation.json"
    evacuation_sha256 = write_receipt(
        receipt_path,
        study_id=STUDY_ID,
        receipt_type="evidence_evacuation",
        payload={
            "status": "sealed_and_verified",
            "terminal_index_sha256": terminal_sha256,
            "destination_manifest_sha256": manifest_sha256,
            "process_cleanup_receipt_sha256": process_cleanup.receipt_sha256,
            "resource_id": binding.resource_id,
            "panel_sha256": binding.panel_sha256,
            "panel_commitment_sha256": binding.panel_commitment_sha256,
            "split_receipt_sha256": binding.split_receipt_sha256,
            "package_closure_sha256": binding.package_closure_sha256,
            "provider_launch_receipt_sha256": (
                binding.provider_launch_receipt_sha256
            ),
            "host_finalizer_receipt_sha256": (
                binding.host_finalizer_receipt_sha256
            ),
            "terminal_attempt_sha256": binding.terminal_attempt_sha256,
            "implementation_revision": binding.implementation_revision,
            "source_lock_sha256": binding.source_lock_sha256,
            "runtime_lock_sha256": binding.runtime_lock_sha256,
            "terminal_status": terminal_authorization.status,
            "terminal_action": terminal_authorization.action,
            "pod_error_code": pod_error_code,
            "all_members_verified": True,
            "raw_paths_included": False,
        },
    )
    return authenticate_evidence_evacuation(
        receipt_path,
        destination_root=destination,
        expected_receipt_sha256=evacuation_sha256,
        expected_binding=binding,
    )
