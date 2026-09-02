"""Owner-host supervisor that always crosses into provider cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .attempt import PodAttemptResult
from .cleanup import BillingBinding, ResourceAdapter
from .host_finalizer import (
    HostAttemptResult,
    HostFinalizerAuthorization,
    finalize_host_attempt,
)


@dataclass(frozen=True)
class SupervisedAttemptResult:
    pod: PodAttemptResult
    host: HostAttemptResult


def supervise_paid_attempt(
    *,
    invoke_pod: Callable[[], PodAttemptResult],
    authorization: HostFinalizerAuthorization,
    resource_adapter: ResourceAdapter,
    resource_manifest: dict[str, object],
    billing_binding: BillingBinding,
    now_utc: Callable[[], datetime],
    ensure_delete_claim: Callable[[], object] | None = None,
) -> SupervisedAttemptResult:
    """Invoke the pod once and finalize from the owner host on every exit."""
    try:
        pod = invoke_pod()
        if not isinstance(pod, PodAttemptResult):
            raise TypeError("pod invocation returned an invalid result")
    except BaseException as error:
        pod = PodAttemptResult(
            study_outcome_sha256=None,
            evidence_evacuation_receipt_sha256=None,
            provider_launch_receipt_sha256=None,
            error_code=f"host_observed_pod_failure:{type(error).__name__}",
        )
    try:
        host = finalize_host_attempt(
            authorization=authorization,
            evidence_evacuation_receipt_sha256=(
                pod.evidence_evacuation_receipt_sha256
            ),
            resource_adapter=resource_adapter,
            resource_manifest=resource_manifest,
            billing_binding=billing_binding,
            now_utc=now_utc,
        )
    finally:
        # The detached watchdog remains the only provider DELETE owner.  This
        # callback can only publish/re-authenticate its single spool claim, so
        # even destination creation or finalizer setup failures cross cleanup.
        if ensure_delete_claim is not None:
            ensure_delete_claim()
    return SupervisedAttemptResult(pod=pod, host=host)
