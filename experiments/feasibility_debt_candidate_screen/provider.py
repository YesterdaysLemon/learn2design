"""Pre-provision provider evidence and one-resource authorization.

This module never provisions a resource.  It converts authenticated, immutable
provider evidence plus the exact owner receipt into opaque capabilities that a
separate provider adapter must require before it may create the sole pod.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .authorization import PaidAttemptAuthorization, validate_paid_bindings
from .canonical import SHA256_RE, read_receipt, sha256_file
from .contract import STUDY_ID
from .cleanup import validate_resource_manifest
from .preflight import PreflightAuthorization
from .orchestrator import TerminalAttemptAuthorization, assert_terminal_attempt
from .runtime import (
    BillingSemantics,
    GPU_PRICE_CEILING_USD_PER_HOUR,
    MAX_HORIZON_SECONDS,
    PRICE_CEILING_USD_PER_HOUR,
    PROVIDER_CAP_USD,
    ProviderLaunchReceipt,
    RuntimeGuardError,
    parse_utc,
)


class ProviderGuardError(RuntimeError):
    pass


PROVIDER = "runpod"
CLOUD_TYPE = "SECURE"
GPU_MODEL = "NVIDIA H100 80GB HBM3"
MAX_EPHEMERAL_DISK_GIB = 40
MAX_GPU_CHARGE_USD = "23.03"


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProviderGuardError(f"{label} is not a lowercase SHA-256")
    return value


def _revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(token not in "0123456789abcdef" for token in value)
    ):
        raise ProviderGuardError("implementation revision is invalid")
    return value


def _image_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or SHA256_RE.fullmatch(value.removeprefix("sha256:")) is None
    ):
        raise ProviderGuardError("immutable image digest is invalid")
    return value


def _billing(value: object) -> BillingSemantics:
    if not isinstance(value, dict) or set(value) != {
        "currency",
        "gpu_hourly_rate_usd",
        "combined_hourly_rate_usd",
        "fixed_charge_usd",
        "metering_quantum_seconds",
        "round_up_each_quantum",
    }:
        raise ProviderGuardError("provider billing schema mismatch")
    from decimal import Decimal, InvalidOperation

    def decimal_field(name: str):
        raw = value[name]
        if not isinstance(raw, str):
            raise ProviderGuardError("provider billing decimals must be strings")
        try:
            parsed = Decimal(raw)
        except InvalidOperation as error:
            raise ProviderGuardError("provider billing decimal is invalid") from error
        if str(parsed) != raw:
            raise ProviderGuardError("provider billing decimal is not canonical")
        return parsed

    try:
        return BillingSemantics(
            currency=value["currency"],
            gpu_hourly_rate_usd=decimal_field("gpu_hourly_rate_usd"),
            combined_hourly_rate_usd=decimal_field("combined_hourly_rate_usd"),
            fixed_charge_usd=decimal_field("fixed_charge_usd"),
            metering_quantum_seconds=value["metering_quantum_seconds"],
            round_up_each_quantum=value["round_up_each_quantum"],
        )
    except RuntimeGuardError as error:
        raise ProviderGuardError(str(error)) from error


_QUOTE_SENTINEL = object()
_REQUEST_SENTINEL = object()
_INVENTORY_SENTINEL = object()
_PROVISION_SENTINEL = object()


@dataclass(frozen=True)
class ProviderQuoteAuthorization:
    receipt_sha256: str
    authenticated_response_sha256: str
    observed_utc: datetime
    billing: BillingSemantics
    _sentinel: object


@dataclass(frozen=True)
class ResourceRequestAuthorization:
    receipt_sha256: str
    implementation_revision: str
    panel_sha256: str
    panel_commitment_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    ci_evidence_sha256: str
    terminal_attempt_sha256: str
    quote_sha256: str
    task_scope_sha256: str
    immutable_image_digest: str
    ephemeral_disk_gib: int
    _sentinel: object


@dataclass(frozen=True)
class CleanInventoryAuthorization:
    receipt_sha256: str
    authenticated_response_sha256: str
    task_scope_sha256: str
    observed_utc: datetime
    _sentinel: object


@dataclass(frozen=True)
class ProvisionAuthorization:
    owner_authorization_sha256: str
    resource_request_sha256: str
    quote_sha256: str
    inventory_sha256: str
    task_scope_sha256: str
    immutable_image_digest: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _PROVISION_SENTINEL:
            raise ProviderGuardError("provision authorizations are verifier-issued")


QUOTE_KEYS = {
    "provider",
    "observed_utc",
    "authenticated_response_sha256",
    "cloud_type",
    "gpu_model",
    "gpu_count",
    "capacity_available",
    "max_ephemeral_disk_gib",
    "billing",
}


def authenticate_provider_quote(
    receipt_path: Path, *, authenticated_response_path: Path
) -> ProviderQuoteAuthorization:
    payload, digest = read_receipt(
        receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_quote",
        expected_payload_keys=QUOTE_KEYS,
    )
    response_digest = _digest(
        payload["authenticated_response_sha256"], label="provider response digest"
    )
    if sha256_file(authenticated_response_path) != response_digest:
        raise ProviderGuardError("provider quote response bytes changed")
    exact = {
        "provider": PROVIDER,
        "cloud_type": CLOUD_TYPE,
        "gpu_model": GPU_MODEL,
        "gpu_count": 1,
        "capacity_available": True,
        "max_ephemeral_disk_gib": MAX_EPHEMERAL_DISK_GIB,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ProviderGuardError("provider quote is outside the frozen envelope")
    try:
        observed = parse_utc(payload["observed_utc"])
    except RuntimeGuardError as error:
        raise ProviderGuardError(str(error)) from error
    return ProviderQuoteAuthorization(
        receipt_sha256=digest,
        authenticated_response_sha256=response_digest,
        observed_utc=observed,
        billing=_billing(payload["billing"]),
        _sentinel=_QUOTE_SENTINEL,
    )


RESOURCE_REQUEST_KEYS = {
    "provider",
    "implementation_revision",
    "panel_sha256",
    "panel_commitment_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "ci_evidence_sha256",
    "terminal_attempt_sha256",
    "quote_sha256",
    "task_scope_sha256",
    "cloud_type",
    "gpu_model",
    "gpu_count",
    "ephemeral_disk_gib",
    "network_volume_count",
    "endpoint_count",
    "template_count",
    "immutable_image_digest",
    "max_provider_seconds",
    "max_gpu_hourly_rate_usd",
    "max_combined_hourly_rate_usd",
    "max_gpu_charge_usd",
    "max_all_in_charge_usd",
    "one_pod_only",
}


def authenticate_resource_request(path: Path) -> ResourceRequestAuthorization:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="resource_request",
        expected_payload_keys=RESOURCE_REQUEST_KEYS,
    )
    exact = {
        "provider": PROVIDER,
        "cloud_type": CLOUD_TYPE,
        "gpu_model": GPU_MODEL,
        "gpu_count": 1,
        "network_volume_count": 0,
        "endpoint_count": 0,
        "template_count": 0,
        "max_provider_seconds": MAX_HORIZON_SECONDS,
        "max_gpu_hourly_rate_usd": str(GPU_PRICE_CEILING_USD_PER_HOUR),
        "max_combined_hourly_rate_usd": str(PRICE_CEILING_USD_PER_HOUR),
        "max_gpu_charge_usd": MAX_GPU_CHARGE_USD,
        "max_all_in_charge_usd": str(PROVIDER_CAP_USD),
        "one_pod_only": True,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ProviderGuardError("resource request is outside the frozen envelope")
    disk = payload["ephemeral_disk_gib"]
    if type(disk) is not int or not 0 < disk <= MAX_EPHEMERAL_DISK_GIB:
        raise ProviderGuardError("resource request disk is outside the frozen envelope")
    for field in (
        "panel_sha256",
        "panel_commitment_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
        "ci_evidence_sha256",
        "terminal_attempt_sha256",
        "quote_sha256",
        "task_scope_sha256",
    ):
        _digest(payload[field], label=field)
    return ResourceRequestAuthorization(
        receipt_sha256=digest,
        implementation_revision=_revision(payload["implementation_revision"]),
        panel_sha256=payload["panel_sha256"],
        panel_commitment_sha256=payload["panel_commitment_sha256"],
        source_lock_sha256=payload["source_lock_sha256"],
        runtime_lock_sha256=payload["runtime_lock_sha256"],
        ci_evidence_sha256=payload["ci_evidence_sha256"],
        terminal_attempt_sha256=payload["terminal_attempt_sha256"],
        quote_sha256=payload["quote_sha256"],
        task_scope_sha256=payload["task_scope_sha256"],
        immutable_image_digest=_image_digest(payload["immutable_image_digest"]),
        ephemeral_disk_gib=disk,
        _sentinel=_REQUEST_SENTINEL,
    )


INVENTORY_KEYS = {
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
}


def authenticate_clean_inventory(
    receipt_path: Path,
    *,
    authenticated_response_path: Path,
    expected_task_scope_sha256: str,
) -> CleanInventoryAuthorization:
    payload, digest = read_receipt(
        receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_inventory",
        expected_payload_keys=INVENTORY_KEYS,
    )
    response_digest = _digest(
        payload["authenticated_response_sha256"], label="inventory response digest"
    )
    if sha256_file(authenticated_response_path) != response_digest:
        raise ProviderGuardError("provider inventory response bytes changed")
    if (
        payload["provider"] != PROVIDER
        or payload["task_scope_sha256"] != expected_task_scope_sha256
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
    ):
        raise ProviderGuardError("provider inventory is not clean and scope-bound")
    try:
        observed = parse_utc(payload["observed_utc"])
    except RuntimeGuardError as error:
        raise ProviderGuardError(str(error)) from error
    return CleanInventoryAuthorization(
        receipt_sha256=digest,
        authenticated_response_sha256=response_digest,
        task_scope_sha256=expected_task_scope_sha256,
        observed_utc=observed,
        _sentinel=_INVENTORY_SENTINEL,
    )


def authorize_single_provision(
    *,
    preflight: PreflightAuthorization,
    paid_authorization: PaidAttemptAuthorization,
    request: ResourceRequestAuthorization,
    quote: ProviderQuoteAuthorization,
    inventory: CleanInventoryAuthorization,
    terminal_attempt: TerminalAttemptAuthorization,
    now_utc: datetime,
) -> ProvisionAuthorization:
    if (
        request._sentinel is not _REQUEST_SENTINEL
        or quote._sentinel is not _QUOTE_SENTINEL
        or inventory._sentinel is not _INVENTORY_SENTINEL
    ):
        raise ProviderGuardError("provider pre-provision evidence is not authentic")
    try:
        terminal_attempt = assert_terminal_attempt(terminal_attempt)
    except Exception as error:
        raise ProviderGuardError("provider pre-provision terminal claim is invalid") from error
    validate_paid_bindings(
        paid_authorization,
        implementation_revision=preflight.revision,
        panel_sha256=preflight.panel_sha256,
        panel_commitment_sha256=preflight.locks.panel_commitment_sha256,
        source_lock_sha256=preflight.locks.source_lock_sha256,
        runtime_lock_sha256=preflight.locks.runtime_lock_sha256,
        ci_evidence_sha256=preflight.ci_evidence_sha256,
        quote_sha256=quote.receipt_sha256,
        resource_request_sha256=request.receipt_sha256,
        terminal_attempt_sha256=terminal_attempt.receipt_sha256,
    )
    expected = {
        "implementation_revision": preflight.revision,
        "panel_sha256": preflight.panel_sha256,
        "panel_commitment_sha256": preflight.locks.panel_commitment_sha256,
        "source_lock_sha256": preflight.locks.source_lock_sha256,
        "runtime_lock_sha256": preflight.locks.runtime_lock_sha256,
        "ci_evidence_sha256": preflight.ci_evidence_sha256,
        "terminal_attempt_sha256": terminal_attempt.receipt_sha256,
        "quote_sha256": quote.receipt_sha256,
    }
    if any(getattr(request, field) != value for field, value in expected.items()):
        raise ProviderGuardError("resource request/preflight binding mismatch")
    if (
        terminal_attempt.revision != preflight.revision
        or terminal_attempt.panel_sha256 != preflight.panel_sha256
        or terminal_attempt.source_lock_sha256 != preflight.locks.source_lock_sha256
    ):
        raise ProviderGuardError("terminal attempt/preflight binding mismatch")
    if inventory.task_scope_sha256 != request.task_scope_sha256:
        raise ProviderGuardError("clean inventory/resource scope binding mismatch")
    if (
        now_utc.tzinfo is None
        or now_utc.utcoffset() != timedelta(0)
        or quote.observed_utc > inventory.observed_utc
        or inventory.observed_utc > now_utc
        or now_utc - quote.observed_utc > timedelta(seconds=300)
        or now_utc - inventory.observed_utc > timedelta(seconds=300)
    ):
        raise ProviderGuardError("provider quote/inventory preflight is stale or reordered")
    return ProvisionAuthorization(
        owner_authorization_sha256=paid_authorization.receipt_sha256,
        resource_request_sha256=request.receipt_sha256,
        quote_sha256=quote.receipt_sha256,
        inventory_sha256=inventory.receipt_sha256,
        task_scope_sha256=request.task_scope_sha256,
        immutable_image_digest=request.immutable_image_digest,
        _sentinel=_PROVISION_SENTINEL,
    )


def assert_provision_authorization(value: object) -> ProvisionAuthorization:
    if (
        not isinstance(value, ProvisionAuthorization)
        or value._sentinel is not _PROVISION_SENTINEL
    ):
        raise ProviderGuardError("resource creation lacks exact pre-provision authorization")
    return value


def validate_provisioned_resource(
    *,
    manifest: dict[str, object],
    launch: ProviderLaunchReceipt,
    request: ResourceRequestAuthorization,
    quote: ProviderQuoteAuthorization,
    provision: ProvisionAuthorization,
) -> None:
    assert_provision_authorization(provision)
    validate_resource_manifest(manifest)
    pod = manifest["pod"]
    assert isinstance(pod, dict)
    if (
        request._sentinel is not _REQUEST_SENTINEL
        or quote._sentinel is not _QUOTE_SENTINEL
        or provision.resource_request_sha256 != request.receipt_sha256
        or provision.quote_sha256 != quote.receipt_sha256
        or provision.task_scope_sha256 != request.task_scope_sha256
        or manifest["provider"] != PROVIDER
        or manifest["task_scope_sha256"] != request.task_scope_sha256
        or manifest["resource_request_sha256"] != request.receipt_sha256
        or manifest["ephemeral_disk_gib"] != request.ephemeral_disk_gib
        or pod["immutable_image_digest"] != request.immutable_image_digest
        or launch.resource_id != pod["id"]
        or launch.task_scope_sha256 != request.task_scope_sha256
        or launch.resource_request_sha256 != request.receipt_sha256
        or launch.quote_sha256 != quote.receipt_sha256
        or launch.immutable_image_digest != request.immutable_image_digest
        or launch.ephemeral_disk_gib != request.ephemeral_disk_gib
        or launch.billing != quote.billing
    ):
        raise ProviderGuardError("provisioned resource differs from its authorized request")
