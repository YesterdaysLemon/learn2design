"""Authenticated process-tree, evidence-evacuation, and provider cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence

from .canonical import canonical_json_bytes, read_receipt, sha256_bytes, sha256_file, write_receipt
from .contract import STUDY_ID


class CleanupError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    start_timestamp: str
    executable_sha256: str
    command_line_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.pid) is not int
            or self.pid < 1
            or type(self.parent_pid) is not int
            or self.parent_pid < 0
            or not isinstance(self.start_timestamp, str)
            or not self.start_timestamp
        ):
            raise CleanupError("process identity fields are invalid")
        for digest in (self.executable_sha256, self.command_line_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise CleanupError("process identity digest is invalid")


class ProcessAdapter(Protocol):
    def snapshot_tree(self, root_pid: int) -> Sequence[ProcessIdentity]: ...

    def identity(self, pid: int) -> ProcessIdentity | None: ...

    def terminate(self, pid: int) -> None: ...

    def descendants(self, root_pid: int) -> Sequence[int]: ...


class ResourceAdapter(Protocol):
    """Host-only provider adapter that persists every control-plane response."""

    def delete(
        self,
        kind: str,
        resource_id: str,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None: ...

    def scoped_inventory(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None: ...

    def billing_receipt(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None: ...


@dataclass(frozen=True)
class BillingBinding:
    resource_id: str
    task_scope_sha256: str
    quote_sha256: str
    launch_response_sha256: str
    gpu_hourly_rate_usd: str
    combined_hourly_rate_usd: str
    fixed_charge_usd: str = "0"
    metering_quantum_seconds: int = 1
    round_up_each_quantum: bool = False
    max_provider_seconds: int = 25_200

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise CleanupError("billing binding resource ID is invalid")
        for digest in (
            self.task_scope_sha256,
            self.quote_sha256,
            self.launch_response_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise CleanupError("billing binding digest is invalid")
        try:
            gpu_rate = Decimal(self.gpu_hourly_rate_usd)
            combined_rate = Decimal(self.combined_hourly_rate_usd)
            fixed_charge = Decimal(self.fixed_charge_usd)
        except (InvalidOperation, ValueError) as error:
            raise CleanupError("billing binding rate is invalid") from error
        if (
            str(gpu_rate) != self.gpu_hourly_rate_usd
            or str(combined_rate) != self.combined_hourly_rate_usd
            or not gpu_rate.is_finite()
            or not combined_rate.is_finite()
            or gpu_rate <= 0
            or gpu_rate > Decimal("3.29")
            or combined_rate < gpu_rate
            or combined_rate > Decimal("3.5714285714")
            or str(fixed_charge) != self.fixed_charge_usd
            or fixed_charge != 0
            or type(self.metering_quantum_seconds) is not int
            or self.metering_quantum_seconds != 1
            or type(self.round_up_each_quantum) is not bool
            or self.round_up_each_quantum is not False
            or type(self.max_provider_seconds) is not int
            or self.max_provider_seconds != 25_200
        ):
            raise CleanupError("billing binding exceeds the frozen envelope")


_EVIDENCE_SENTINEL = object()
_PROCESS_CLEANUP_SENTINEL = object()


@dataclass(frozen=True)
class EvidenceHandoffBinding:
    resource_id: str
    panel_sha256: str
    panel_commitment_sha256: str
    split_receipt_sha256: str
    package_closure_sha256: str
    provider_launch_receipt_sha256: str
    host_finalizer_receipt_sha256: str
    terminal_attempt_sha256: str
    implementation_revision: str
    source_lock_sha256: str
    runtime_lock_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise CleanupError("evidence handoff resource ID is invalid")
        for value in (
            self.panel_sha256,
            self.panel_commitment_sha256,
            self.split_receipt_sha256,
            self.package_closure_sha256,
            self.provider_launch_receipt_sha256,
            self.host_finalizer_receipt_sha256,
            self.terminal_attempt_sha256,
            self.source_lock_sha256,
            self.runtime_lock_sha256,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(token not in "0123456789abcdef" for token in value)
            ):
                raise CleanupError("evidence handoff digest is invalid")
        if (
            not isinstance(self.implementation_revision, str)
            or len(self.implementation_revision) != 40
            or any(
                token not in "0123456789abcdef"
                for token in self.implementation_revision
            )
        ):
            raise CleanupError("evidence handoff revision is invalid")


@dataclass(frozen=True)
class ProcessCleanupAuthorization:
    receipt_sha256: str
    status: str
    root_pid: int
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _PROCESS_CLEANUP_SENTINEL:
            raise CleanupError("process-cleanup authorizations are verifier-issued only")


def assert_process_cleanup(value: object) -> ProcessCleanupAuthorization:
    if (
        not isinstance(value, ProcessCleanupAuthorization)
        or value._sentinel is not _PROCESS_CLEANUP_SENTINEL
    ):
        raise CleanupError("process-cleanup authorization is invalid")
    return value


@dataclass(frozen=True)
class EvidenceEvacuationAuthorization:
    receipt_sha256: str
    terminal_index_sha256: str
    destination_manifest_sha256: str
    process_cleanup_receipt_sha256: str
    terminal_status: str
    terminal_action: str
    pod_error_code: str | None
    binding: EvidenceHandoffBinding
    _destination_root: Path
    _destination_manifest_path: Path
    _terminal_logical_id: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _EVIDENCE_SENTINEL:
            raise CleanupError("evidence authorizations are verifier-issued only")


def assert_evidence_evacuation(
    value: object,
) -> EvidenceEvacuationAuthorization:
    if (
        not isinstance(value, EvidenceEvacuationAuthorization)
        or value._sentinel is not _EVIDENCE_SENTINEL
    ):
        raise CleanupError("evidence evacuation authorization is invalid")
    return value


def authenticate_evidence_evacuation(
    path: Path,
    *,
    destination_root: Path,
    expected_receipt_sha256: str,
    expected_binding: EvidenceHandoffBinding,
) -> EvidenceEvacuationAuthorization:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="evidence_evacuation",
        expected_payload_keys={
            "status",
            "terminal_index_sha256",
            "destination_manifest_sha256",
            "process_cleanup_receipt_sha256",
            "resource_id",
            "panel_sha256",
            "panel_commitment_sha256",
            "split_receipt_sha256",
            "package_closure_sha256",
            "provider_launch_receipt_sha256",
            "host_finalizer_receipt_sha256",
            "terminal_attempt_sha256",
            "implementation_revision",
            "source_lock_sha256",
            "runtime_lock_sha256",
            "terminal_status",
            "terminal_action",
            "pod_error_code",
            "all_members_verified",
            "raw_paths_included",
        },
    )
    if digest != expected_receipt_sha256:
        raise CleanupError("evidence evacuation receipt digest mismatch")
    binding_exact = {
        "resource_id": expected_binding.resource_id,
        "panel_sha256": expected_binding.panel_sha256,
        "panel_commitment_sha256": expected_binding.panel_commitment_sha256,
        "split_receipt_sha256": expected_binding.split_receipt_sha256,
        "package_closure_sha256": expected_binding.package_closure_sha256,
        "provider_launch_receipt_sha256": (
            expected_binding.provider_launch_receipt_sha256
        ),
        "host_finalizer_receipt_sha256": (
            expected_binding.host_finalizer_receipt_sha256
        ),
        "terminal_attempt_sha256": expected_binding.terminal_attempt_sha256,
        "implementation_revision": expected_binding.implementation_revision,
        "source_lock_sha256": expected_binding.source_lock_sha256,
        "runtime_lock_sha256": expected_binding.runtime_lock_sha256,
    }
    if any(payload.get(key) != value for key, value in binding_exact.items()):
        raise CleanupError("evidence evacuation paid-attempt binding mismatch")
    expected_terminal_index_sha256 = payload["terminal_index_sha256"]
    expected_destination_manifest_sha256 = payload[
        "destination_manifest_sha256"
    ]
    process_cleanup_receipt_sha256 = payload[
        "process_cleanup_receipt_sha256"
    ]
    for value in (
        expected_terminal_index_sha256,
        expected_destination_manifest_sha256,
        process_cleanup_receipt_sha256,
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(token not in "0123456789abcdef" for token in value)
        ):
            raise CleanupError("evidence evacuation digest is invalid")
    destination_root = destination_root.resolve(strict=True)
    destination_manifest_path = (
        destination_root / "evidence-destination-manifest.json"
    )
    if (
        payload["status"] != "sealed_and_verified"
        or payload["all_members_verified"] is not True
        or payload["raw_paths_included"] is not False
    ):
        raise CleanupError("evidence evacuation receipt does not authorize cleanup")
    if (
        sha256_file(destination_manifest_path)
        != expected_destination_manifest_sha256
    ):
        raise CleanupError("evacuated evidence source/manifest digest mismatch")
    manifest, manifest_digest = read_receipt(
        destination_manifest_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="evidence_destination_manifest",
        expected_payload_keys={
            "terminal_index_sha256",
            "process_cleanup_receipt_sha256",
            "member_count",
            "member_rows",
            "source_tree_sha256",
            "raw_paths_included",
        },
    )
    rows = manifest["member_rows"]
    if (
        manifest_digest != expected_destination_manifest_sha256
        or manifest["terminal_index_sha256"] != expected_terminal_index_sha256
        or manifest["process_cleanup_receipt_sha256"]
        != process_cleanup_receipt_sha256
        or type(manifest["member_count"]) is not int
        or not isinstance(rows, list)
        or manifest["member_count"] != len(rows)
        or manifest["source_tree_sha256"]
        != sha256_bytes(canonical_json_bytes(rows))
        or manifest["raw_paths_included"] is not False
    ):
        raise CleanupError("evacuated evidence manifest schema mismatch")
    terminal_logical_ids: list[str] = []
    process_cleanup_logical_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "logical_id",
            "sha256",
            "size_bytes",
        }:
            raise CleanupError("evacuated evidence member row mismatch")
        logical_id = row["logical_id"]
        pure = PurePosixPath(logical_id) if isinstance(logical_id, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise CleanupError("evacuated evidence logical path is unsafe")
        target = destination_root.joinpath(*pure.parts)
        if (
            not target.is_file()
            or target.stat().st_size != row["size_bytes"]
            or sha256_file(target) != row["sha256"]
        ):
            raise CleanupError("evacuated evidence member bytes mismatch")
        if row["sha256"] == expected_terminal_index_sha256:
            terminal_logical_ids.append(logical_id)
        if row["sha256"] == process_cleanup_receipt_sha256:
            process_cleanup_logical_ids.append(logical_id)
    if len(terminal_logical_ids) != 1:
        raise CleanupError("evacuated evidence omits the terminal index")
    if len(process_cleanup_logical_ids) != 1:
        raise CleanupError("evacuated evidence omits the process-cleanup receipt")
    process_cleanup_path = destination_root.joinpath(
        *process_cleanup_logical_ids[0].split("/")
    )
    authenticate_process_cleanup_receipt(
        process_cleanup_path,
        expected_receipt_sha256=process_cleanup_receipt_sha256,
    )
    terminal_path = destination_root.joinpath(*terminal_logical_ids[0].split("/"))
    from .orchestrator import authenticate_terminal_outcome

    terminal = authenticate_terminal_outcome(
        terminal_path,
        evidence_root=destination_root,
        expected_revision=expected_binding.implementation_revision,
        expected_panel_sha256=expected_binding.panel_sha256,
        expected_panel_commitment_sha256=(
            expected_binding.panel_commitment_sha256
        ),
        expected_split_receipt_sha256=expected_binding.split_receipt_sha256,
        expected_package_closure_sha256=(
            expected_binding.package_closure_sha256
        ),
        expected_source_lock_sha256=expected_binding.source_lock_sha256,
        expected_runtime_lock_sha256=expected_binding.runtime_lock_sha256,
        expected_terminal_attempt_sha256=expected_binding.terminal_attempt_sha256,
    )
    if terminal.receipt_sha256 != expected_terminal_index_sha256:
        raise CleanupError("evacuated terminal outcome digest mismatch")
    pod_error_code = payload["pod_error_code"]
    if (
        payload["terminal_status"] != terminal.status
        or payload["terminal_action"] != terminal.action
        or (
            pod_error_code is not None
            and (not isinstance(pod_error_code, str) or not pod_error_code)
        )
    ):
        raise CleanupError("evacuated terminal projection mismatch")
    return EvidenceEvacuationAuthorization(
        receipt_sha256=digest,
        terminal_index_sha256=expected_terminal_index_sha256,
        destination_manifest_sha256=expected_destination_manifest_sha256,
        process_cleanup_receipt_sha256=process_cleanup_receipt_sha256,
        terminal_status=terminal.status,
        terminal_action=terminal.action,
        pod_error_code=pod_error_code,
        binding=expected_binding,
        _destination_root=destination_root.resolve(strict=True),
        _destination_manifest_path=destination_manifest_path.resolve(strict=True),
        _terminal_logical_id=terminal_logical_ids[0],
        _sentinel=_EVIDENCE_SENTINEL,
    )


def _validate_tree(snapshot: Sequence[ProcessIdentity], *, root_pid: int) -> None:
    if not snapshot:
        raise CleanupError("process tree snapshot is empty")
    by_pid = {item.pid: item for item in snapshot}
    if len(by_pid) != len(snapshot) or root_pid not in by_pid:
        raise CleanupError("process tree root or PID uniqueness is invalid")
    for item in snapshot:
        if item.pid == root_pid:
            continue
        cursor = item
        seen: set[int] = set()
        while cursor.pid != root_pid:
            if cursor.pid in seen or cursor.parent_pid not in by_pid:
                raise CleanupError("process tree has a cycle or foreign parent")
            seen.add(cursor.pid)
            cursor = by_pid[cursor.parent_pid]


def authenticate_tree(
    adapter: ProcessAdapter, snapshot: Sequence[ProcessIdentity], *, root_pid: int
) -> None:
    _validate_tree(snapshot, root_pid=root_pid)
    for expected in snapshot:
        if adapter.identity(expected.pid) != expected:
            raise CleanupError("process identity changed before cleanup")


def terminate_tree_bottom_up(
    adapter: ProcessAdapter,
    snapshot: Sequence[ProcessIdentity],
    *,
    root_pid: int,
) -> list[int]:
    authenticate_tree(adapter, snapshot, root_pid=root_pid)
    parent = {item.pid: item.parent_pid for item in snapshot}
    expected = {item.pid: item for item in snapshot}

    def depth(pid: int) -> int:
        value = 0
        cursor = pid
        while cursor != root_pid:
            cursor = parent[cursor]
            value += 1
        return value

    order = sorted(expected, key=lambda pid: (depth(pid), pid), reverse=True)
    terminated: list[int] = []
    for pid in order:
        # Re-authenticate immediately before every destructive process action;
        # a disappeared or reused PID is a collision, never a new target.
        if adapter.identity(pid) != expected[pid]:
            raise CleanupError("process identity changed immediately before termination")
        adapter.terminate(pid)
        terminated.append(pid)
    survivors = [pid for pid in order if adapter.identity(pid) is not None]
    descendants = list(adapter.descendants(root_pid))
    if survivors or descendants:
        raise CleanupError("process tree cleanup left surviving processes")
    return terminated


def cleanup_process_tree_once(
    path: Path,
    *,
    adapter: ProcessAdapter,
    root_pid: int,
) -> ProcessCleanupAuthorization:
    """Terminate only the pod-owned worker tree and seal the attempted result.

    Provider deletion is intentionally absent.  This function may run in the
    pod coordinator, but it may never target that coordinator's own PID.
    """
    errors: list[str] = []
    snapshot: list[ProcessIdentity] = []
    termination_order: list[int] = []
    if type(root_pid) is not int or root_pid < 1 or root_pid == os.getpid():
        errors.append("process_cleanup:invalid_or_self_root")
    else:
        try:
            snapshot = list(adapter.snapshot_tree(root_pid))
            termination_order = terminate_tree_bottom_up(
                adapter,
                snapshot,
                root_pid=root_pid,
            )
        except BaseException as error:
            errors.append(f"process_cleanup:{type(error).__name__}:{error}")
    status = "complete" if not errors else "attempt_not_evaluable"
    digest = write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="process_cleanup",
        payload={
            "status": status,
            "root_pid": root_pid,
            "process_snapshot": [item.__dict__ for item in snapshot],
            "termination_order": termination_order,
            "cleanup_errors": errors,
            "provider_actions_included": False,
        },
    )
    return authenticate_process_cleanup_receipt(
        path,
        expected_receipt_sha256=digest,
    )


def authenticate_process_cleanup_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> ProcessCleanupAuthorization:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="process_cleanup",
        expected_payload_keys={
            "status",
            "root_pid",
            "process_snapshot",
            "termination_order",
            "cleanup_errors",
            "provider_actions_included",
        },
    )
    if digest != expected_receipt_sha256:
        raise CleanupError("process-cleanup receipt digest mismatch")
    root_pid = payload["root_pid"]
    snapshot_payload = payload["process_snapshot"]
    termination_order = payload["termination_order"]
    errors = payload["cleanup_errors"]
    if (
        payload["status"] not in {"complete", "attempt_not_evaluable"}
        or type(root_pid) is not int
        or root_pid < 1
        or not isinstance(snapshot_payload, list)
        or not isinstance(termination_order, list)
        or any(type(pid) is not int or pid < 1 for pid in termination_order)
        or len(termination_order) != len(set(termination_order))
        or not isinstance(errors, list)
        or any(not isinstance(error, str) or not error for error in errors)
        or payload["provider_actions_included"] is not False
    ):
        raise CleanupError("process-cleanup receipt schema mismatch")
    try:
        snapshot = [ProcessIdentity(**row) for row in snapshot_payload]
    except (TypeError, CleanupError) as error:
        raise CleanupError("process-cleanup identity is invalid") from error
    if payload["status"] == "complete":
        _validate_tree(snapshot, root_pid=root_pid)
        if errors or set(termination_order) != {row.pid for row in snapshot}:
            raise CleanupError("complete process cleanup is not exhaustive")
    elif not errors:
        raise CleanupError("failed process cleanup lacks an error")
    return ProcessCleanupAuthorization(
        receipt_sha256=digest,
        status=payload["status"],
        root_pid=root_pid,
        _sentinel=_PROCESS_CLEANUP_SENTINEL,
    )


def validate_resource_manifest(manifest: dict[str, object]) -> list[dict[str, str]]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "provider",
        "cloud_type",
        "pod",
        "ephemeral_disk_gib",
        "other_resources",
        "task_scope_sha256",
        "resource_request_sha256",
    }:
        raise CleanupError("resource manifest schema mismatch")
    if manifest["provider"] != "runpod" or manifest["cloud_type"] != "SECURE":
        raise CleanupError("resource manifest is not secure cloud")
    disk = manifest["ephemeral_disk_gib"]
    if type(disk) is not int or not 0 < disk <= 40:
        raise CleanupError("ephemeral disk exceeds the frozen bound")
    pod = manifest["pod"]
    if not isinstance(pod, dict) or set(pod) != {
        "id",
        "gpu_type_id",
        "gpu_count",
        "immutable_image_digest",
    }:
        raise CleanupError("pod manifest schema mismatch")
    if (
        not isinstance(pod["id"], str)
        or not pod["id"]
        or pod["gpu_type_id"] != "NVIDIA H100 80GB HBM3"
        or type(pod["gpu_count"]) is not int
        or pod["gpu_count"] != 1
        or not isinstance(pod["immutable_image_digest"], str)
        or not pod["immutable_image_digest"].startswith("sha256:")
        or len(pod["immutable_image_digest"]) != 71
        or any(
            token not in "0123456789abcdef"
            for token in pod["immutable_image_digest"].removeprefix("sha256:")
        )
    ):
        raise CleanupError("pod manifest identity mismatch")
    if manifest["other_resources"] != []:
        raise CleanupError("extra provider objects are forbidden")
    for field in ("task_scope_sha256", "resource_request_sha256"):
        digest = manifest[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(token not in "0123456789abcdef" for token in digest)
        ):
            raise CleanupError(f"resource manifest digest is invalid: {field}")
    return [{"kind": "pod", "id": str(pod["id"])}]


def _scoped_inventory_targets(
    rows: Sequence[dict[str, object]], *, task_scope_sha256: str
) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "id",
            "task_scope_sha256",
        }:
            raise CleanupError("provider inventory row schema mismatch")
        kind = row["kind"]
        resource_id = row["id"]
        if (
            kind not in {"pod", "volume", "endpoint", "template"}
            or not isinstance(resource_id, str)
            or not resource_id
            or row["task_scope_sha256"] != task_scope_sha256
        ):
            raise CleanupError("provider inventory row is outside the task scope")
        targets.append({"kind": str(kind), "id": resource_id})
    return targets


def validate_billing_receipt(
    value: object, *, binding: BillingBinding
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "resource_id",
        "task_scope_sha256",
        "quote_sha256",
        "launch_response_sha256",
        "observed_utc",
        "billing_query",
        "billing_query_sha256",
        "provider_record_count",
        "provider_unique_pod_count",
        "currency",
        "envelope_seconds",
        "gpu_hourly_rate_usd",
        "combined_hourly_rate_usd",
        "fixed_charge_usd",
        "metering_quantum_seconds",
        "round_up_each_quantum",
        "gpu_charge_bound_usd",
        "all_in_charge_bound_usd",
        "provider_gpu_charge_usd",
        "provider_disk_charge_usd",
        "provider_total_charge_usd",
        "provider_receipt_sha256",
    }:
        raise CleanupError("final billing receipt schema mismatch")
    digest = value["provider_receipt_sha256"]
    query = value["billing_query"]
    if (
        value["provider"] != "runpod"
        or value["resource_id"] != binding.resource_id
        or value["task_scope_sha256"] != binding.task_scope_sha256
        or value["quote_sha256"] != binding.quote_sha256
        or value["launch_response_sha256"] != binding.launch_response_sha256
        or not isinstance(query, dict)
        or set(query) != {"startTime", "endTime", "bucketSize", "podId"}
        or query.get("bucketSize") != "hour"
        or query.get("podId") != binding.resource_id
        or not isinstance(query.get("startTime"), str)
        or not isinstance(query.get("endTime"), str)
        or sha256_bytes(canonical_json_bytes(query))
        != value["billing_query_sha256"]
        or not isinstance(value["billing_query_sha256"], str)
        or len(value["billing_query_sha256"]) != 64
        or type(value["provider_record_count"]) is not int
        or value["provider_record_count"] < 0
        or type(value["provider_unique_pod_count"]) is not int
        or value["provider_unique_pod_count"] not in {0, 1}
        or value["currency"] != "USD"
        or type(value["envelope_seconds"]) is not int
        or not 0 <= value["envelope_seconds"] <= binding.max_provider_seconds
        or value["gpu_hourly_rate_usd"] != binding.gpu_hourly_rate_usd
        or value["combined_hourly_rate_usd"] != binding.combined_hourly_rate_usd
        or value["fixed_charge_usd"] != binding.fixed_charge_usd
        or value["metering_quantum_seconds"]
        != binding.metering_quantum_seconds
        or value["round_up_each_quantum"]
        is not binding.round_up_each_quantum
        or not isinstance(value["gpu_charge_bound_usd"], str)
        or not isinstance(value["all_in_charge_bound_usd"], str)
        or not isinstance(value["provider_gpu_charge_usd"], str)
        or not isinstance(value["provider_disk_charge_usd"], str)
        or not isinstance(value["provider_total_charge_usd"], str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(token not in "0123456789abcdef" for token in digest)
    ):
        raise CleanupError("final billing receipt identity mismatch")
    try:
        from .runtime import parse_utc

        query_start = parse_utc(query["startTime"])
        query_end = parse_utc(query["endTime"])
        gpu_amount = Decimal(value["gpu_charge_bound_usd"])
        amount = Decimal(value["all_in_charge_bound_usd"])
        provider_gpu = Decimal(value["provider_gpu_charge_usd"])
        provider_disk = Decimal(value["provider_disk_charge_usd"])
        provider_total = Decimal(value["provider_total_charge_usd"])
    except (InvalidOperation, ValueError) as error:
        raise CleanupError("final billing amount is invalid") from error
    if query_end <= query_start:
        raise CleanupError("final billing query window is invalid")
    from .runtime import BillingSemantics

    semantics = BillingSemantics(
        currency="USD",
        gpu_hourly_rate_usd=Decimal(binding.gpu_hourly_rate_usd),
        combined_hourly_rate_usd=Decimal(binding.combined_hourly_rate_usd),
        fixed_charge_usd=Decimal(binding.fixed_charge_usd),
        metering_quantum_seconds=binding.metering_quantum_seconds,
        round_up_each_quantum=binding.round_up_each_quantum,
    )
    gpu_semantics = BillingSemantics(
        currency="USD",
        gpu_hourly_rate_usd=Decimal(binding.gpu_hourly_rate_usd),
        combined_hourly_rate_usd=Decimal(binding.gpu_hourly_rate_usd),
        fixed_charge_usd=Decimal("0"),
        metering_quantum_seconds=binding.metering_quantum_seconds,
        round_up_each_quantum=binding.round_up_each_quantum,
    )
    expected_gpu = gpu_semantics.charge_at_seconds(value["envelope_seconds"])
    expected_all_in = semantics.charge_at_seconds(value["envelope_seconds"])
    if (
        str(gpu_amount) != value["gpu_charge_bound_usd"]
        or str(amount) != value["all_in_charge_bound_usd"]
        or str(provider_gpu) != value["provider_gpu_charge_usd"]
        or str(provider_disk) != value["provider_disk_charge_usd"]
        or str(provider_total) != value["provider_total_charge_usd"]
        or not gpu_amount.is_finite()
        or not amount.is_finite()
        or not provider_gpu.is_finite()
        or not provider_disk.is_finite()
        or not provider_total.is_finite()
        or gpu_amount < 0
        or gpu_amount != expected_gpu
        or gpu_amount > Decimal("23.03")
        or amount < gpu_amount
        or amount != expected_all_in
        or amount > Decimal("25.00")
        or provider_gpu < 0
        or provider_disk < 0
        or provider_total != provider_gpu + provider_disk
        or provider_gpu > gpu_amount
        or provider_gpu > Decimal("23.03")
        or provider_total > amount
        or provider_total > Decimal("25.00")
    ):
        raise CleanupError("final billing amount exceeds the frozen cap")
    return value
