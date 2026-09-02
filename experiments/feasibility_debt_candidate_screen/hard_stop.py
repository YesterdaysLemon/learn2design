"""Independent hard-stop receipt required before cold smoke."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import json
from pathlib import Path
from typing import Callable

from .canonical import SHA256_RE, canonical_json_bytes, read_receipt, sha256_file
from .contract import STUDY_ID
from .runtime import DeadlineClock, ProviderLaunchReceipt, RuntimeGuardError, parse_utc


class HardStopError(RuntimeError):
    pass


_SENTINEL = object()


@dataclass(frozen=True)
class HardStopAuthorization:
    receipt_sha256: str
    resource_id: str
    task_scope_sha256: str
    provider_launch_receipt_sha256: str
    resource_manifest_sha256: str
    watchdog_trigger_utc: str
    delete_by_utc: str
    _watchdog_payload_bytes: bytes = field(repr=False, compare=False)
    _verify_live_watchdog: Callable[[dict[str, object]], bool] = field(
        repr=False, compare=False
    )
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _SENTINEL:
            raise HardStopError("hard-stop authorizations are verifier-issued only")

    def assert_live(self) -> None:
        """Recheck the independent provider watchdog at every operation boundary."""
        try:
            payload = json.loads(self._watchdog_payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HardStopError("hard-stop watchdog payload is no longer canonical") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != HARD_STOP_KEYS
            or canonical_json_bytes(payload) != self._watchdog_payload_bytes
            or payload["resource_id"] != self.resource_id
            or payload["task_scope_sha256"] != self.task_scope_sha256
            or payload["provider_launch_receipt_sha256"]
            != self.provider_launch_receipt_sha256
            or payload["resource_manifest_sha256"] != self.resource_manifest_sha256
            or payload["watchdog_trigger_utc"] != self.watchdog_trigger_utc
            or payload["delete_by_utc"] != self.delete_by_utc
        ):
            raise HardStopError("hard-stop watchdog liveness binding changed")
        if self._verify_live_watchdog(payload) is not True:
            raise HardStopError("independent hard-stop watchdog is no longer live")


HARD_STOP_KEYS = {
    "status",
    "provider",
    "resource_id",
    "task_scope_sha256",
    "resource_request_sha256",
    "resource_manifest_sha256",
    "provider_launch_receipt_sha256",
    "quote_sha256",
    "control_kind",
    "watchdog_source_sha256",
    "watchdog_process_identity_sha256",
    "watchdog_command_sha256",
    "watchdog_started_utc",
    "watchdog_trigger_utc",
    "delete_by_utc",
    "credential_scope",
    "authenticated_control_response_sha256",
}


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise HardStopError(f"hard-stop digest is invalid: {label}")
    return value


def authenticate_hard_stop(
    path: Path,
    *,
    authenticated_control_response_path: Path,
    launch: ProviderLaunchReceipt,
    provider_launch_receipt_sha256: str,
    deadline: DeadlineClock,
    verify_live_watchdog: Callable[[dict[str, object]], bool],
) -> HardStopAuthorization:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_hard_stop",
        expected_payload_keys=HARD_STOP_KEYS,
    )
    for field in (
        "task_scope_sha256",
        "resource_request_sha256",
        "resource_manifest_sha256",
        "provider_launch_receipt_sha256",
        "quote_sha256",
        "watchdog_source_sha256",
        "watchdog_process_identity_sha256",
        "watchdog_command_sha256",
        "authenticated_control_response_sha256",
    ):
        _digest(payload[field], label=field)
    exact = {
        "status": "ARMED",
        "provider": "runpod",
        "resource_id": launch.resource_id,
        "task_scope_sha256": launch.task_scope_sha256,
        "resource_request_sha256": launch.resource_request_sha256,
        "resource_manifest_sha256": launch.resource_manifest_sha256,
        "provider_launch_receipt_sha256": provider_launch_receipt_sha256,
        "quote_sha256": launch.quote_sha256,
        "control_kind": "independent_watchdog_process",
        "credential_scope": "delete_task_pod_only",
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise HardStopError("hard-stop receipt is not launch-bound")
    if (
        sha256_file(authenticated_control_response_path)
        != payload["authenticated_control_response_sha256"]
    ):
        raise HardStopError("hard-stop control response bytes changed")
    try:
        started = parse_utc(payload["watchdog_started_utc"])
        trigger = parse_utc(payload["watchdog_trigger_utc"])
        delete_by = parse_utc(payload["delete_by_utc"])
    except RuntimeGuardError as error:
        raise HardStopError(str(error)) from error
    if (
        started < deadline.t0
        or started >= trigger
        or trigger != deadline.hard_horizon - timedelta(seconds=60)
        or trigger > delete_by - timedelta(seconds=30)
        or delete_by > deadline.hard_horizon
        or delete_by != deadline.hard_horizon
    ):
        raise HardStopError("hard-stop timing does not enforce the frozen horizon")
    if verify_live_watchdog(payload) is not True:
        raise HardStopError("independent hard-stop watchdog is not live and authentic")
    return HardStopAuthorization(
        receipt_sha256=digest,
        resource_id=launch.resource_id,
        task_scope_sha256=launch.task_scope_sha256,
        provider_launch_receipt_sha256=provider_launch_receipt_sha256,
        resource_manifest_sha256=launch.resource_manifest_sha256,
        watchdog_trigger_utc=payload["watchdog_trigger_utc"],
        delete_by_utc=payload["delete_by_utc"],
        _watchdog_payload_bytes=canonical_json_bytes(payload),
        _verify_live_watchdog=verify_live_watchdog,
        _sentinel=_SENTINEL,
    )


def assert_hard_stop(value: object) -> HardStopAuthorization:
    if not isinstance(value, HardStopAuthorization) or value._sentinel is not _SENTINEL:
        raise HardStopError("cold smoke lacks an authentic independent hard stop")
    value.assert_live()
    return value
