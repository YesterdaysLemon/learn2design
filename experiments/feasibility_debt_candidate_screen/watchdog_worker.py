"""Independent bound-pod watchdog process.

The provider credential is accepted only from a one-shot stdin bootstrap.  It
is retained in this process's memory until the fixed trigger, never placed in
argv or written to disk, and used for exactly one arm GET plus one terminal
DELETE.  The worker does not discover credentials, retry, create, update, or
operate on any resource other than the bootstrap-bound pod ID.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import sys
import time
from typing import Callable
from urllib.parse import quote

from .canonical import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from .contract import STUDY_ID
from .runpod_adapter import (
    BearerHttpsTransport,
    RunpodAdapterError,
    RunpodTransport,
    transcript_bytes,
)
from .runtime import parse_utc
from .canonical import exclusive_write_bytes


class WatchdogWorkerError(RuntimeError):
    pass


BOOTSTRAP_KEYS = {
    "arm_deadline_utc",
    "bearer_token",
    "command_sha256",
    "credential_fingerprint_sha256",
    "delete_by_utc",
    "resource_id",
    "source_closure_sha256",
    "task_scope_sha256",
    "trigger_utc",
}
MAX_BOOTSTRAP_BYTES = 16_384


def watchdog_source_closure_sha256() -> str:
    root = Path(__file__).parent
    rows = []
    for name in (
        "canonical.py",
        "contract.py",
        "runpod_adapter.py",
        "runtime.py",
        "watchdog_worker.py",
    ):
        path = root / name
        rows.append({"logical_id": name, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(rows))


@dataclass(frozen=True)
class WatchdogBootstrap:
    arm_deadline_utc: str
    bearer_token: str
    command_sha256: str
    credential_fingerprint_sha256: str
    delete_by_utc: str
    resource_id: str
    source_closure_sha256: str
    task_scope_sha256: str
    trigger_utc: str

    @classmethod
    def parse(cls, content: bytes) -> "WatchdogBootstrap":
        if len(content) > MAX_BOOTSTRAP_BYTES:
            raise WatchdogWorkerError("watchdog bootstrap exceeds the fixed cap")
        value = parse_canonical_json(content)
        if not isinstance(value, dict) or set(value) != BOOTSTRAP_KEYS:
            raise WatchdogWorkerError("watchdog bootstrap schema mismatch")
        try:
            result = cls(**value)
        except TypeError as error:
            raise WatchdogWorkerError("watchdog bootstrap is malformed") from error
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not self.bearer_token
            or any(token in self.bearer_token for token in ("\r", "\n", " "))
            or sha256_bytes(self.bearer_token.encode("utf-8"))
            != self.credential_fingerprint_sha256
            or not self.resource_id
        ):
            raise WatchdogWorkerError("watchdog credential or resource binding is invalid")
        for digest in (
            self.credential_fingerprint_sha256,
            self.command_sha256,
            self.source_closure_sha256,
            self.task_scope_sha256,
        ):
            if len(digest) != 64 or any(
                token not in "0123456789abcdef" for token in digest
            ):
                raise WatchdogWorkerError("watchdog digest binding is invalid")
        if self.source_closure_sha256 != watchdog_source_closure_sha256():
            raise WatchdogWorkerError("watchdog source closure changed")
        arm_deadline = parse_utc(self.arm_deadline_utc)
        trigger = parse_utc(self.trigger_utc)
        delete_by = parse_utc(self.delete_by_utc)
        if arm_deadline >= trigger or trigger > delete_by - timedelta(seconds=30):
            raise WatchdogWorkerError("watchdog chronology lacks deletion reserve")


def execute_watchdog(
    *,
    bootstrap: WatchdogBootstrap,
    transport: RunpodTransport,
    arm_response_path: Path,
    arm_receipt_path: Path,
    delete_response_path: Path,
    delete_receipt_path: Path,
    delete_claim_path: Path,
    now_utc: Callable[[], datetime],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Arm once, wait monotonically, then issue one bound DELETE."""
    bootstrap.validate()
    pod_path = f"/v2/pods/{quote(bootstrap.resource_id, safe='')}"
    arm = transport.request(
        "GET",
        pod_path,
        body=None,
        deadline_utc=bootstrap.arm_deadline_utc,
    )
    exclusive_write_bytes(arm_response_path, transcript_bytes([arm]))
    if arm.status != 200:
        raise WatchdogWorkerError(f"watchdog arm GET returned HTTP {arm.status}")
    write_receipt(
        arm_receipt_path,
        study_id=STUDY_ID,
        receipt_type="watchdog_arm_ack",
        payload={
            "status": "ARMED",
            "resource_id": bootstrap.resource_id,
            "task_scope_sha256": bootstrap.task_scope_sha256,
            "process_id": os.getpid(),
            "source_closure_sha256": bootstrap.source_closure_sha256,
            "command_sha256": bootstrap.command_sha256,
            "armed_utc": arm.observed_utc,
            "trigger_utc": bootstrap.trigger_utc,
            "delete_by_utc": bootstrap.delete_by_utc,
            "authenticated_control_response_sha256": sha256_file(
                arm_response_path
            ),
            "credential_persisted": False,
            "zero_retry": True,
        },
    )

    trigger = parse_utc(bootstrap.trigger_utc)
    initial_now = now_utc()
    if initial_now.tzinfo is None or initial_now.utcoffset() != timedelta(0):
        raise WatchdogWorkerError("watchdog clock is not UTC")
    remaining = max(0.0, (trigger - initial_now).total_seconds())
    monotonic_deadline = monotonic() + remaining
    claim_sha256: str | None = None
    trigger_kind = "hard_horizon"
    while True:
        if delete_claim_path.exists():
            from .canonical import read_receipt

            try:
                claim, claim_sha256 = read_receipt(
                    delete_claim_path,
                    expected_study_id=STUDY_ID,
                    expected_receipt_type="watchdog_delete_claim",
                    expected_payload_keys={
                        "resource_id",
                        "task_scope_sha256",
                        "claim_utc",
                        "delete_by_utc",
                        "reason",
                        "one_attempt",
                    },
                )
                if (
                    claim["resource_id"] != bootstrap.resource_id
                    or claim["task_scope_sha256"] != bootstrap.task_scope_sha256
                    or claim["delete_by_utc"] != bootstrap.delete_by_utc
                    or claim["reason"] != "host_cleanup"
                    or claim["one_attempt"] is not True
                    or parse_utc(claim["claim_utc"]) > parse_utc(
                        bootstrap.delete_by_utc
                    )
                ):
                    raise WatchdogWorkerError("watchdog delete claim changed binding")
                trigger_kind = "host_cleanup_claim"
            except BaseException:
                trigger_kind = "malformed_claim_fail_safe"
                claim_sha256 = None
            break
        pending = monotonic_deadline - monotonic()
        if pending <= 0:
            break
        sleep(min(1.0, pending))

    exchange = transport.request(
        "DELETE",
        pod_path,
        body=None,
        deadline_utc=bootstrap.delete_by_utc,
    )
    exclusive_write_bytes(delete_response_path, transcript_bytes([exchange]))
    status = (
        "DELETED"
        if exchange.status == 204
        else "ALREADY_ABSENT"
        if exchange.status == 404
        else "FAILED"
    )
    write_receipt(
        delete_receipt_path,
        study_id=STUDY_ID,
        receipt_type="watchdog_delete",
        payload={
            "status": status,
            "resource_id": bootstrap.resource_id,
            "task_scope_sha256": bootstrap.task_scope_sha256,
            "trigger_utc": bootstrap.trigger_utc,
            "delete_by_utc": bootstrap.delete_by_utc,
            "observed_utc": exchange.observed_utc,
            "trigger_kind": trigger_kind,
            "delete_claim_sha256": claim_sha256,
            "authenticated_response_sha256": sha256_file(
                delete_response_path
            ),
            "zero_retry": True,
        },
    )
    if exchange.status not in {204, 404}:
        raise WatchdogWorkerError(
            f"watchdog DELETE returned HTTP {exchange.status}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm-response", required=True, type=Path)
    parser.add_argument("--arm-receipt", required=True, type=Path)
    parser.add_argument("--delete-response", required=True, type=Path)
    parser.add_argument("--delete-receipt", required=True, type=Path)
    parser.add_argument("--delete-claim", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    content = sys.stdin.buffer.read(MAX_BOOTSTRAP_BYTES + 1)
    bootstrap = WatchdogBootstrap.parse(content)
    transport = BearerHttpsTransport(bearer_token=bootstrap.bearer_token)
    try:
        execute_watchdog(
            bootstrap=bootstrap,
            transport=transport,
            arm_response_path=args.arm_response,
            arm_receipt_path=args.arm_receipt,
            delete_response_path=args.delete_response,
            delete_receipt_path=args.delete_receipt,
            delete_claim_path=args.delete_claim,
            now_utc=lambda: datetime.now(UTC),
            monotonic=time.monotonic,
            sleep=time.sleep,
        )
    except (WatchdogWorkerError, RunpodAdapterError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
