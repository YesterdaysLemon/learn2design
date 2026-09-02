"""Host-resident provider cleanup and terminal evidence finalization.

This module is deliberately separate from the pod-side study runner.  The
process that owns a :class:`HostFinalizerAuthorization` must be outside the
provider resource it deletes, and every provider action must persist an
authenticated response before the final attempt can be called complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .canonical import (
    SHA256_RE,
    canonical_json_bytes,
    read_receipt,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from .cleanup import (
    BillingBinding,
    EvidenceHandoffBinding,
    EvidenceEvacuationAuthorization,
    ProcessIdentity,
    ResourceAdapter,
    assert_evidence_evacuation,
    authenticate_evidence_evacuation,
    authenticate_process_cleanup_receipt,
    validate_billing_receipt,
    validate_resource_manifest,
)
from .contract import STUDY_ID
from .hard_stop import HardStopAuthorization, assert_hard_stop
from .runtime import DeadlineClock, ProviderLaunchReceipt, RuntimeGuardError, parse_utc


class HostFinalizerError(RuntimeError):
    pass


_HOST_FINALIZER_SENTINEL = object()


HOST_FINALIZER_KEYS = {
    "status",
    "execution_domain",
    "provider",
    "resource_id",
    "panel_sha256",
    "panel_commitment_sha256",
    "split_receipt_sha256",
    "package_closure_sha256",
    "task_scope_sha256",
    "resource_request_sha256",
    "resource_manifest_sha256",
    "provider_launch_receipt_sha256",
    "launch_response_sha256",
    "quote_sha256",
    "hard_stop_receipt_sha256",
    "owner_paid_authorization_sha256",
    "terminal_attempt_sha256",
    "implementation_revision",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "host_source_closure_sha256",
    "host_environment_sha256",
    "host_process_identity",
    "host_started_utc",
    "armed_utc",
    "delete_by_utc",
    "evidence_destination_sha256",
    "authenticated_control_response_sha256",
    "provider_credentials_present",
    "inside_provider_resource",
}


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise HostFinalizerError(f"{label} is not a lowercase SHA-256")
    return value


def evidence_destination_sha256(path: Path) -> str:
    """Private path binding; only its digest enters portable receipts."""
    return sha256_bytes(str(path.resolve()).encode("utf-8"))


@dataclass(frozen=True)
class HostFinalizerAuthorization:
    receipt_sha256: str
    resource_id: str
    panel_sha256: str
    panel_commitment_sha256: str
    split_receipt_sha256: str
    package_closure_sha256: str
    task_scope_sha256: str
    resource_request_sha256: str
    resource_manifest_sha256: str
    provider_launch_receipt_sha256: str
    launch_response_sha256: str
    quote_sha256: str
    hard_stop_receipt_sha256: str
    owner_paid_authorization_sha256: str
    terminal_attempt_sha256: str
    implementation_revision: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    delete_by_utc: str
    armed_utc: str
    evidence_destination_sha256: str
    _destination_root: Path = field(repr=False, compare=False)
    _payload_bytes: bytes = field(repr=False, compare=False)
    _verify_live_host: Callable[[dict[str, object]], bool] = field(
        repr=False,
        compare=False,
    )
    _sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._sentinel is not _HOST_FINALIZER_SENTINEL:
            raise HostFinalizerError(
                "host-finalizer authorizations are verifier-issued only"
            )

    def assert_live(self) -> None:
        from .canonical import parse_canonical_json

        payload = parse_canonical_json(self._payload_bytes)
        if (
            not isinstance(payload, dict)
            or set(payload) != HOST_FINALIZER_KEYS
            or payload["resource_id"] != self.resource_id
            or payload["panel_sha256"] != self.panel_sha256
            or payload["panel_commitment_sha256"]
            != self.panel_commitment_sha256
            or payload["split_receipt_sha256"] != self.split_receipt_sha256
            or payload["package_closure_sha256"]
            != self.package_closure_sha256
            or payload["task_scope_sha256"] != self.task_scope_sha256
            or payload["resource_request_sha256"]
            != self.resource_request_sha256
            or payload["resource_manifest_sha256"]
            != self.resource_manifest_sha256
            or payload["provider_launch_receipt_sha256"]
            != self.provider_launch_receipt_sha256
            or payload["launch_response_sha256"]
            != self.launch_response_sha256
            or payload["quote_sha256"] != self.quote_sha256
            or payload["hard_stop_receipt_sha256"]
            != self.hard_stop_receipt_sha256
            or payload["owner_paid_authorization_sha256"]
            != self.owner_paid_authorization_sha256
            or payload["terminal_attempt_sha256"]
            != self.terminal_attempt_sha256
            or payload["implementation_revision"]
            != self.implementation_revision
            or payload["source_lock_sha256"] != self.source_lock_sha256
            or payload["runtime_lock_sha256"] != self.runtime_lock_sha256
            or payload["delete_by_utc"] != self.delete_by_utc
            or payload["armed_utc"] != self.armed_utc
            or payload["evidence_destination_sha256"]
            != self.evidence_destination_sha256
            or evidence_destination_sha256(self._destination_root)
            != self.evidence_destination_sha256
            or self._verify_live_host(payload) is not True
        ):
            raise HostFinalizerError("host-finalizer liveness binding changed")


def authenticate_host_finalizer(
    path: Path,
    *,
    authenticated_control_response_path: Path,
    destination_root: Path,
    launch: ProviderLaunchReceipt,
    provider_launch_receipt_sha256: str,
    hard_stop: HardStopAuthorization,
    owner_paid_authorization_sha256: str,
    panel_sha256: str,
    panel_commitment_sha256: str,
    split_receipt_sha256: str,
    package_closure_sha256: str,
    terminal_attempt_sha256: str,
    implementation_revision: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    expected_host_source_closure_sha256: str,
    expected_host_environment_sha256: str,
    verify_live_host: Callable[[dict[str, object]], bool],
) -> HostFinalizerAuthorization:
    """Authenticate the independent host process before any pod work begins."""
    if destination_root.exists():
        raise HostFinalizerError("host evidence destination is not fresh")
    hard_stop = assert_hard_stop(hard_stop)
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="host_finalizer",
        expected_payload_keys=HOST_FINALIZER_KEYS,
    )
    for field_name in (
        "task_scope_sha256",
        "panel_sha256",
        "panel_commitment_sha256",
        "split_receipt_sha256",
        "package_closure_sha256",
        "resource_request_sha256",
        "resource_manifest_sha256",
        "provider_launch_receipt_sha256",
        "launch_response_sha256",
        "quote_sha256",
        "hard_stop_receipt_sha256",
        "owner_paid_authorization_sha256",
        "terminal_attempt_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
        "host_source_closure_sha256",
        "host_environment_sha256",
        "evidence_destination_sha256",
        "authenticated_control_response_sha256",
    ):
        _digest(payload[field_name], label=field_name)
    identity_payload = payload["host_process_identity"]
    if not isinstance(identity_payload, dict) or set(identity_payload) != {
        "pid",
        "parent_pid",
        "start_timestamp",
        "executable_sha256",
        "command_line_sha256",
    }:
        raise HostFinalizerError("host process identity schema mismatch")
    try:
        ProcessIdentity(**identity_payload)
        host_started = parse_utc(payload["host_started_utc"])
        armed = parse_utc(payload["armed_utc"])
        delete_by = parse_utc(payload["delete_by_utc"])
        launch_created = parse_utc(launch.create_utc)
    except (TypeError, RuntimeGuardError) as error:
        raise HostFinalizerError("host finalizer chronology is invalid") from error
    exact = {
        "status": "ARMED",
        "execution_domain": "owner_host_outside_provider_resource",
        "provider": "runpod",
        "resource_id": launch.resource_id,
        "panel_sha256": panel_sha256,
        "panel_commitment_sha256": panel_commitment_sha256,
        "split_receipt_sha256": split_receipt_sha256,
        "package_closure_sha256": package_closure_sha256,
        "task_scope_sha256": launch.task_scope_sha256,
        "resource_request_sha256": launch.resource_request_sha256,
        "resource_manifest_sha256": launch.resource_manifest_sha256,
        "provider_launch_receipt_sha256": provider_launch_receipt_sha256,
        "launch_response_sha256": launch.authenticated_response_sha256,
        "quote_sha256": launch.quote_sha256,
        "hard_stop_receipt_sha256": hard_stop.receipt_sha256,
        "owner_paid_authorization_sha256": (
            owner_paid_authorization_sha256
        ),
        "terminal_attempt_sha256": terminal_attempt_sha256,
        "implementation_revision": implementation_revision,
        "source_lock_sha256": source_lock_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "host_source_closure_sha256": (
            expected_host_source_closure_sha256
        ),
        "host_environment_sha256": expected_host_environment_sha256,
        "delete_by_utc": hard_stop.delete_by_utc,
        "evidence_destination_sha256": evidence_destination_sha256(
            destination_root
        ),
        "provider_credentials_present": True,
        "inside_provider_resource": False,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise HostFinalizerError("host finalizer is not bound to the paid attempt")
    if (
        sha256_file(authenticated_control_response_path)
        != payload["authenticated_control_response_sha256"]
        or host_started > launch_created
        or host_started > armed
        or armed >= delete_by
        or delete_by
        > DeadlineClock.from_authenticated_receipt(launch).hard_horizon
        or verify_live_host(payload) is not True
    ):
        raise HostFinalizerError("host finalizer is stale, reordered, or not live")
    return HostFinalizerAuthorization(
        receipt_sha256=digest,
        resource_id=launch.resource_id,
        panel_sha256=panel_sha256,
        panel_commitment_sha256=panel_commitment_sha256,
        split_receipt_sha256=split_receipt_sha256,
        package_closure_sha256=package_closure_sha256,
        task_scope_sha256=launch.task_scope_sha256,
        resource_request_sha256=launch.resource_request_sha256,
        resource_manifest_sha256=launch.resource_manifest_sha256,
        provider_launch_receipt_sha256=provider_launch_receipt_sha256,
        launch_response_sha256=launch.authenticated_response_sha256,
        quote_sha256=launch.quote_sha256,
        hard_stop_receipt_sha256=hard_stop.receipt_sha256,
        owner_paid_authorization_sha256=owner_paid_authorization_sha256,
        terminal_attempt_sha256=terminal_attempt_sha256,
        implementation_revision=implementation_revision,
        source_lock_sha256=source_lock_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        delete_by_utc=hard_stop.delete_by_utc,
        armed_utc=payload["armed_utc"],
        evidence_destination_sha256=payload["evidence_destination_sha256"],
        _destination_root=destination_root.resolve(),
        _payload_bytes=canonical_json_bytes(payload),
        _verify_live_host=verify_live_host,
        _sentinel=_HOST_FINALIZER_SENTINEL,
    )


def assert_host_finalizer(value: object) -> HostFinalizerAuthorization:
    if (
        not isinstance(value, HostFinalizerAuthorization)
        or value._sentinel is not _HOST_FINALIZER_SENTINEL
    ):
        raise HostFinalizerError("provider cleanup lacks a host-finalizer capability")
    value.assert_live()
    return value


def _fresh_provider_targets(
    root: Path,
    stem: str,
) -> tuple[Path, Path]:
    receipt = root / f"{stem}.json"
    response = root / f"{stem}.response"
    for path in (
        receipt,
        receipt.with_name(receipt.name + ".sha256"),
        response,
    ):
        if path.exists():
            raise HostFinalizerError("provider evidence target is not fresh")
    return receipt, response


def _authenticate_delete(
    receipt_path: Path,
    response_path: Path,
    *,
    resource: dict[str, str],
    authorization: HostFinalizerAuthorization,
) -> tuple[str, datetime]:
    payload, digest = read_receipt(
        receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_delete",
        expected_payload_keys={
            "provider",
            "status",
            "kind",
            "id",
            "task_scope_sha256",
            "resource_request_sha256",
            "quote_sha256",
            "launch_response_sha256",
            "observed_utc",
            "authenticated_response_sha256",
        },
    )
    response_digest = _digest(
        payload["authenticated_response_sha256"],
        label="delete response digest",
    )
    exact = {
        "provider": "runpod",
        "status": "DELETED",
        "kind": resource["kind"],
        "id": resource["id"],
        "task_scope_sha256": authorization.task_scope_sha256,
        "resource_request_sha256": authorization.resource_request_sha256,
        "quote_sha256": authorization.quote_sha256,
    }
    try:
        observed = parse_utc(payload["observed_utc"])
        armed = parse_utc(authorization.armed_utc)
        delete_by = parse_utc(authorization.delete_by_utc)
    except RuntimeGuardError as error:
        raise HostFinalizerError("provider delete timestamp is invalid") from error
    if (
        any(payload.get(key) != value for key, value in exact.items())
        or payload["launch_response_sha256"]
        != authorization.launch_response_sha256
        or sha256_file(response_path) != response_digest
        or observed < armed
        or observed > delete_by
    ):
        raise HostFinalizerError("provider delete receipt is not authenticated")
    return digest, observed


def _authenticate_zero_inventory(
    receipt_path: Path,
    response_path: Path,
    *,
    authorization: HostFinalizerAuthorization,
) -> tuple[str, datetime]:
    payload, digest = read_receipt(
        receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_inventory",
        expected_payload_keys={
            "provider",
            "observed_utc",
            "authenticated_response_sha256",
            "task_scope_sha256",
            "pods",
            "clusters",
            "network_volumes",
            "endpoints",
            "templates",
            "registries",
        },
    )
    response_digest = _digest(
        payload["authenticated_response_sha256"],
        label="inventory response digest",
    )
    try:
        observed = parse_utc(payload["observed_utc"])
        armed = parse_utc(authorization.armed_utc)
        delete_by = parse_utc(authorization.delete_by_utc)
    except RuntimeGuardError as error:
        raise HostFinalizerError("provider inventory timestamp is invalid") from error
    if (
        payload["provider"] != "runpod"
        or payload["task_scope_sha256"] != authorization.task_scope_sha256
        or any(
            payload[name] != []
            for name in (
                "pods",
                "clusters",
                "network_volumes",
                "endpoints",
                "templates",
                "registries",
            )
        )
        or sha256_file(response_path) != response_digest
        or observed < armed
        or observed > delete_by
    ):
        raise HostFinalizerError("provider inventory is not authenticated zero state")
    return digest, observed


def _authenticate_billing(
    receipt_path: Path,
    response_path: Path,
    *,
    binding: BillingBinding,
    authorization: HostFinalizerAuthorization,
) -> tuple[dict[str, object], str, datetime]:
    payload, digest = read_receipt(
        receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_billing",
        expected_payload_keys={
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
        },
    )
    response_digest = _digest(
        payload["provider_receipt_sha256"],
        label="billing response digest",
    )
    if sha256_file(response_path) != response_digest:
        raise HostFinalizerError("provider billing response bytes changed")
    try:
        observed = parse_utc(payload["observed_utc"])
        armed = parse_utc(authorization.armed_utc)
        delete_by = parse_utc(authorization.delete_by_utc)
    except RuntimeGuardError as error:
        raise HostFinalizerError("provider billing timestamp is invalid") from error
    if observed < armed or observed > delete_by:
        raise HostFinalizerError("provider billing crossed the host horizon")
    return validate_billing_receipt(payload, binding=binding), digest, observed


def _member_path_by_digest(
    authorization: EvidenceEvacuationAuthorization,
    digest: str,
) -> Path:
    manifest, _ = read_receipt(
        authorization._destination_manifest_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="evidence_destination_manifest",
    )
    rows = manifest.get("member_rows")
    if not isinstance(rows, list):
        raise HostFinalizerError("evacuated manifest lacks member rows")
    matches = [
        authorization._destination_root.joinpath(*row["logical_id"].split("/"))
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("logical_id"), str)
        and row.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise HostFinalizerError("evacuated evidence digest is not unique")
    return matches[0]


def _process_cleanup_status(
    authorization: EvidenceEvacuationAuthorization,
) -> str:
    path = _member_path_by_digest(
        authorization,
        authorization.process_cleanup_receipt_sha256,
    )
    return authenticate_process_cleanup_receipt(
        path,
        expected_receipt_sha256=(
            authorization.process_cleanup_receipt_sha256
        ),
    ).status


def _study_outcome(
    authorization: EvidenceEvacuationAuthorization,
) -> tuple[str, str, str, str | None]:
    return (
        authorization.terminal_action,
        authorization.terminal_index_sha256,
        authorization.terminal_status,
        authorization.pod_error_code,
    )


def _terminal_evidence_index(root: Path) -> str:
    index_path = root / "terminal-evidence-index.json"
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == index_path:
            continue
        rows.append(
            {
                "logical_id": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return write_receipt(
        index_path,
        study_id=STUDY_ID,
        receipt_type="terminal_evidence_index",
        payload={
            "member_count": len(rows),
            "member_rows": rows,
            "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
            "raw_paths_included": False,
        },
    )


@dataclass(frozen=True)
class HostAttemptResult:
    finalization_sha256: str
    final_action: str
    study_outcome_sha256: str | None
    cleanup_result_sha256: str
    terminal_evidence_index_sha256: str
    terminal_evidence_seal_sha256: str


def _require_host_deadline(
    authorization: HostFinalizerAuthorization,
    now: datetime,
) -> datetime:
    try:
        hard_horizon = parse_utc(authorization.delete_by_utc)
    except RuntimeGuardError as error:
        raise HostFinalizerError("host hard horizon is invalid") from error
    if (
        now.tzinfo is None
        or now.utcoffset() != timedelta(0)
        or now >= hard_horizon
    ):
        raise HostFinalizerError("host provider action crossed the hard horizon")
    return now


def finalize_host_attempt(
    *,
    authorization: HostFinalizerAuthorization,
    evidence_evacuation_receipt_sha256: str | None,
    resource_adapter: ResourceAdapter,
    resource_manifest: dict[str, object],
    billing_binding: BillingBinding,
    now_utc: Callable[[], datetime],
) -> HostAttemptResult:
    """Delete from the owner host and seal all post-delete provider evidence."""
    authorization = assert_host_finalizer(authorization)
    if (
        sha256_bytes(canonical_json_bytes(resource_manifest))
        != authorization.resource_manifest_sha256
        or billing_binding.resource_id != authorization.resource_id
        or billing_binding.task_scope_sha256 != authorization.task_scope_sha256
        or billing_binding.quote_sha256 != authorization.quote_sha256
        or billing_binding.launch_response_sha256
        != authorization.launch_response_sha256
    ):
        raise HostFinalizerError("host cleanup inputs changed after authorization")
    destination = authorization._destination_root
    destination.mkdir(parents=True, exist_ok=True)
    post_cleanup = destination / "post-cleanup"
    post_cleanup.mkdir(exist_ok=False)
    errors: list[str] = []
    study_action = "retain_round1_control_attempt_not_evaluable"
    study_sha256: str | None = None
    study_status = "missing"
    pod_error_code: str | None = None
    process_cleanup_status = "missing"
    evacuation_sha256: str | None = None
    evidence_authorization: EvidenceEvacuationAuthorization | None = None
    if evidence_evacuation_receipt_sha256 is None:
        errors.append("evidence_evacuation:missing")
    else:
        try:
            evidence_authorization = authenticate_evidence_evacuation(
                destination
                / "pod-handoff"
                / "evidence-evacuation.json",
                destination_root=destination,
                expected_receipt_sha256=(
                    evidence_evacuation_receipt_sha256
                ),
                expected_binding=EvidenceHandoffBinding(
                    resource_id=authorization.resource_id,
                    panel_sha256=authorization.panel_sha256,
                    panel_commitment_sha256=(
                        authorization.panel_commitment_sha256
                    ),
                    split_receipt_sha256=authorization.split_receipt_sha256,
                    package_closure_sha256=(
                        authorization.package_closure_sha256
                    ),
                    provider_launch_receipt_sha256=(
                        authorization.provider_launch_receipt_sha256
                    ),
                    host_finalizer_receipt_sha256=(
                        authorization.receipt_sha256
                    ),
                    terminal_attempt_sha256=(
                        authorization.terminal_attempt_sha256
                    ),
                    implementation_revision=(
                        authorization.implementation_revision
                    ),
                    source_lock_sha256=authorization.source_lock_sha256,
                    runtime_lock_sha256=authorization.runtime_lock_sha256,
                ),
            )
            evidence_authorization = assert_evidence_evacuation(
                evidence_authorization
            )
            if (
                evidence_authorization._destination_root != destination
                or evidence_destination_sha256(destination)
                != authorization.evidence_destination_sha256
            ):
                raise HostFinalizerError("evacuation destination binding mismatch")
            evacuation_sha256 = evidence_authorization.receipt_sha256
            process_cleanup_status = _process_cleanup_status(
                evidence_authorization
            )
            (
                study_action,
                study_sha256,
                study_status,
                pod_error_code,
            ) = _study_outcome(evidence_authorization)
            if process_cleanup_status != "complete":
                errors.append("process_cleanup:not_complete")
        except BaseException as error:
            errors.append(f"evidence_authentication:{type(error).__name__}")

    try:
        resources = validate_resource_manifest(resource_manifest)
    except BaseException as error:
        errors.append(f"resource_manifest:{type(error).__name__}")
        resources = [{"kind": "pod", "id": billing_binding.resource_id}]
    delete_receipts: list[str] = []
    provider_observed = parse_utc(authorization.armed_utc)
    for index, resource in enumerate(resources):
        try:
            authorization.assert_live()
            _require_host_deadline(authorization, now_utc())
            receipt_path, response_path = _fresh_provider_targets(
                post_cleanup,
                f"delete-{index:02d}",
            )
            resource_adapter.delete(
                resource["kind"],
                resource["id"],
                receipt_path=receipt_path,
                authenticated_response_path=response_path,
                deadline_utc=authorization.delete_by_utc,
            )
            completed = _require_host_deadline(authorization, now_utc())
            delete_digest, observed = _authenticate_delete(
                receipt_path,
                response_path,
                resource=resource,
                authorization=authorization,
            )
            if observed < provider_observed or observed > completed:
                raise HostFinalizerError("provider delete receipt is reordered")
            provider_observed = observed
            delete_receipts.append(delete_digest)
        except BaseException as error:
            errors.append(f"resource_delete:{type(error).__name__}")

    inventory_sha256: str | None = None
    try:
        authorization.assert_live()
        _require_host_deadline(authorization, now_utc())
        receipt_path, response_path = _fresh_provider_targets(
            post_cleanup,
            "inventory-zero",
        )
        resource_adapter.scoped_inventory(
            receipt_path=receipt_path,
            authenticated_response_path=response_path,
            deadline_utc=authorization.delete_by_utc,
        )
        completed = _require_host_deadline(authorization, now_utc())
        inventory_sha256, observed = _authenticate_zero_inventory(
            receipt_path,
            response_path,
            authorization=authorization,
        )
        if observed < provider_observed or observed > completed:
            raise HostFinalizerError("provider inventory receipt is reordered")
        provider_observed = observed
    except BaseException as error:
        errors.append(f"resource_inventory:{type(error).__name__}")

    billing: dict[str, object] | None = None
    billing_sha256: str | None = None
    try:
        authorization.assert_live()
        _require_host_deadline(authorization, now_utc())
        receipt_path, response_path = _fresh_provider_targets(
            post_cleanup,
            "billing-final",
        )
        resource_adapter.billing_receipt(
            receipt_path=receipt_path,
            authenticated_response_path=response_path,
            deadline_utc=authorization.delete_by_utc,
        )
        completed = _require_host_deadline(authorization, now_utc())
        billing, billing_sha256, observed = _authenticate_billing(
            receipt_path,
            response_path,
            binding=billing_binding,
            authorization=authorization,
        )
        if observed < provider_observed or observed > completed:
            raise HostFinalizerError("provider billing receipt is reordered")
        provider_observed = observed
    except BaseException as error:
        errors.append(f"billing_receipt:{type(error).__name__}")

    cleanup_complete = not errors
    cleanup_sha256 = write_receipt(
        post_cleanup / "cleanup-result.json",
        study_id=STUDY_ID,
        receipt_type="cleanup_result",
        payload={
            "status": "complete" if cleanup_complete else "attempt_not_evaluable",
            "host_finalizer_receipt_sha256": authorization.receipt_sha256,
            "evidence_evacuation_receipt_sha256": evacuation_sha256,
            "process_cleanup_status": process_cleanup_status,
            "pod_terminal_status": study_status,
            "pod_error_code": pod_error_code,
            "provider_delete_receipt_sha256s": delete_receipts,
            "zero_inventory_receipt_sha256": inventory_sha256,
            "billing_receipt_sha256": billing_sha256,
            "billing_receipt": billing,
            "cleanup_errors": errors,
        },
    )
    study_evaluable = (
        study_sha256 is not None
        and study_status in {
            "evaluable_stage1_failed",
            "evaluable_stage2_complete",
        }
        and pod_error_code is None
    )
    final_action = (
        study_action
        if cleanup_complete and study_evaluable
        else "retain_round1_control_attempt_not_evaluable"
    )
    finalization_sha256 = write_receipt(
        post_cleanup / "final-attempt.json",
        study_id=STUDY_ID,
        receipt_type="attempt_finalization",
        payload={
            "status": (
                "evidence_ready"
                if cleanup_complete and study_evaluable
                else "not_evaluable"
            ),
            "action": final_action,
            "study_outcome_sha256": study_sha256,
            "cleanup_result_sha256": cleanup_sha256,
            "owner_paid_authorization_sha256": (
                authorization.owner_paid_authorization_sha256
            ),
            "resource_request_sha256": authorization.resource_request_sha256,
            "provider_quote_sha256": authorization.quote_sha256,
            "provider_launch_receipt_sha256": (
                authorization.provider_launch_receipt_sha256
            ),
            "hard_stop_receipt_sha256": authorization.hard_stop_receipt_sha256,
            "host_finalizer_receipt_sha256": authorization.receipt_sha256,
            "error_code": (
                None
                if cleanup_complete and study_evaluable
                else (
                    "study_not_evaluable"
                    if cleanup_complete
                    else "host_cleanup_incomplete"
                )
            ),
            "organizer_score_comparable": False,
            "completion_authority": "terminal_evidence_seal_only",
        },
    )
    terminal_index_sha256 = _terminal_evidence_index(destination)
    terminal_seal_sha256 = write_receipt(
        destination / "terminal-evidence-seal.json",
        study_id=STUDY_ID,
        receipt_type="terminal_evidence_seal",
        payload={
            "terminal_evidence_index_sha256": terminal_index_sha256,
            "finalization_sha256": finalization_sha256,
            "cleanup_result_sha256": cleanup_sha256,
            "host_finalizer_receipt_sha256": authorization.receipt_sha256,
            "status": (
                "complete"
                if cleanup_complete and study_evaluable
                else "not_evaluable"
            ),
            "action": final_action,
            "raw_paths_included": False,
            "completion_authority": "terminal_evidence_seal_only",
        },
    )
    return HostAttemptResult(
        finalization_sha256=finalization_sha256,
        final_action=final_action,
        study_outcome_sha256=study_sha256,
        cleanup_result_sha256=cleanup_sha256,
        terminal_evidence_index_sha256=terminal_index_sha256,
        terminal_evidence_seal_sha256=terminal_seal_sha256,
    )
