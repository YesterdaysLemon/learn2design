"""Exact owner authorization receipts for the sole conditional paid attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical import read_receipt, sha256_bytes
from .contract import PLAN_REVISION, STUDY_ID


class AuthorizationError(RuntimeError):
    pass


PAID_ATTEMPT_APPROVAL_TEXT = (
    "I authorize one secure NVIDIA H100 80GB HBM3 attempt for "
    f"{STUDY_ID} under its authenticated panel, source, runtime, quote, CI, "
    "loss-blind-smoke-first, no-retry, seven-hour, and $25.00 all-in gates; "
    "task-owned cleanup is included, while portal action, candidate integration, "
    "private outcomes, and any later paid run remain unauthorized."
)
PAID_ATTEMPT_APPROVAL_TEXT_SHA256 = sha256_bytes(
    PAID_ATTEMPT_APPROVAL_TEXT.encode("utf-8")
)

_PAID_SENTINEL = object()


@dataclass(frozen=True)
class PaidAttemptAuthorization:
    receipt_sha256: str
    implementation_revision: str
    panel_sha256: str
    panel_commitment_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    ci_evidence_sha256: str
    quote_sha256: str
    resource_request_sha256: str
    terminal_attempt_sha256: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _PAID_SENTINEL:
            raise AuthorizationError("paid authorizations are verifier-issued only")


PAID_AUTHORIZATION_KEYS = {
    "approval_text",
    "approval_text_sha256",
    "study_id",
    "plan_revision",
    "implementation_revision",
    "panel_sha256",
    "panel_commitment_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "ci_evidence_sha256",
    "quote_sha256",
    "resource_request_sha256",
    "terminal_attempt_sha256",
    "cloud_type",
    "gpu_model",
    "gpu_count",
    "max_ephemeral_disk_gib",
    "max_gpu_hourly_rate_usd",
    "max_gpu_charge_usd",
    "max_all_in_charge_usd",
    "max_provider_seconds",
    "smoke_first",
    "one_attempt",
    "no_retry",
    "cleanup_included",
    "portal_authorized",
    "candidate_integration_authorized",
    "private_outcomes_authorized",
}


def authenticate_paid_attempt_authorization(path: Path) -> PaidAttemptAuthorization:
    payload, receipt_sha256 = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="owner_paid_attempt_authorization",
        expected_payload_keys=PAID_AUTHORIZATION_KEYS,
    )
    exact = {
        "approval_text": PAID_ATTEMPT_APPROVAL_TEXT,
        "approval_text_sha256": PAID_ATTEMPT_APPROVAL_TEXT_SHA256,
        "study_id": STUDY_ID,
        "plan_revision": PLAN_REVISION,
        "cloud_type": "SECURE",
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "max_ephemeral_disk_gib": 40,
        "max_gpu_hourly_rate_usd": "3.29",
        "max_gpu_charge_usd": "23.03",
        "max_all_in_charge_usd": "25.00",
        "max_provider_seconds": 25_200,
        "smoke_first": True,
        "one_attempt": True,
        "no_retry": True,
        "cleanup_included": True,
        "portal_authorized": False,
        "candidate_integration_authorized": False,
        "private_outcomes_authorized": False,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise AuthorizationError("owner paid-attempt authorization scope mismatch")
    revision = payload["implementation_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(token not in "0123456789abcdef" for token in revision)
    ):
        raise AuthorizationError("paid-attempt implementation revision is invalid")
    digest_fields = (
        "panel_sha256",
        "panel_commitment_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
        "ci_evidence_sha256",
        "quote_sha256",
        "resource_request_sha256",
        "terminal_attempt_sha256",
    )
    for field in digest_fields:
        digest = payload[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(token not in "0123456789abcdef" for token in digest)
        ):
            raise AuthorizationError(f"paid-attempt digest is invalid: {field}")
    return PaidAttemptAuthorization(
        receipt_sha256=receipt_sha256,
        implementation_revision=revision,
        panel_sha256=payload["panel_sha256"],
        panel_commitment_sha256=payload["panel_commitment_sha256"],
        source_lock_sha256=payload["source_lock_sha256"],
        runtime_lock_sha256=payload["runtime_lock_sha256"],
        ci_evidence_sha256=payload["ci_evidence_sha256"],
        quote_sha256=payload["quote_sha256"],
        resource_request_sha256=payload["resource_request_sha256"],
        terminal_attempt_sha256=payload["terminal_attempt_sha256"],
        _sentinel=_PAID_SENTINEL,
    )


def validate_paid_bindings(
    authorization: PaidAttemptAuthorization,
    *,
    implementation_revision: str,
    panel_sha256: str,
    panel_commitment_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    ci_evidence_sha256: str,
    quote_sha256: str,
    resource_request_sha256: str,
    terminal_attempt_sha256: str,
) -> None:
    if (
        not isinstance(authorization, PaidAttemptAuthorization)
        or authorization._sentinel is not _PAID_SENTINEL
        or authorization.implementation_revision != implementation_revision
        or authorization.panel_sha256 != panel_sha256
        or authorization.panel_commitment_sha256 != panel_commitment_sha256
        or authorization.source_lock_sha256 != source_lock_sha256
        or authorization.runtime_lock_sha256 != runtime_lock_sha256
        or authorization.ci_evidence_sha256 != ci_evidence_sha256
        or authorization.quote_sha256 != quote_sha256
        or authorization.resource_request_sha256 != resource_request_sha256
        or authorization.terminal_attempt_sha256 != terminal_attempt_sha256
    ):
        raise AuthorizationError("paid-attempt authorization bindings changed")
