"""Owner-host arming and one-owner provider deletion.

The watchdog spool is created before provisioning and is deliberately separate
from the evidence destination.  The detached worker is the sole process that
may issue the provider DELETE.  Host cleanup can only publish one authenticated
claim into that spool and project the worker's immutable response into the
terminal evidence tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Protocol

from .canonical import (
    canonical_json_bytes,
    exclusive_write_bytes,
    read_receipt,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from .cleanup import BillingBinding, ResourceAdapter
from .contract import STUDY_ID
from .hard_stop import HardStopAuthorization, authenticate_hard_stop
from .runpod_adapter import RunpodAdapterError
from .runtime import DeadlineClock, ProviderLaunchReceipt, parse_utc
from .watchdog_worker import WatchdogBootstrap, watchdog_source_closure_sha256


class WatchdogError(RuntimeError):
    pass


ARM_KEYS = {
    "status",
    "resource_id",
    "task_scope_sha256",
    "process_id",
    "source_closure_sha256",
    "command_sha256",
    "armed_utc",
    "trigger_utc",
    "delete_by_utc",
    "authenticated_control_response_sha256",
    "credential_persisted",
    "zero_retry",
}
DELETE_KEYS = {
    "status",
    "resource_id",
    "task_scope_sha256",
    "trigger_utc",
    "delete_by_utc",
    "observed_utc",
    "trigger_kind",
    "delete_claim_sha256",
    "authenticated_response_sha256",
    "zero_retry",
}


class _Process(Protocol):
    pid: int
    stdin: object

    def poll(self) -> int | None: ...


@dataclass(frozen=True)
class WatchdogSpool:
    root: Path
    manifest_path: Path
    arm_response_path: Path
    arm_receipt_path: Path
    delete_claim_path: Path
    delete_response_path: Path
    delete_receipt_path: Path
    hard_stop_receipt_path: Path
    manifest_sha256: str


def prepare_watchdog_spool(root: Path) -> WatchdogSpool:
    """Create the cleanup control plane before any provider mutation."""
    root = root.resolve()
    if root.exists():
        raise WatchdogError("watchdog spool must be a fresh path")
    root.mkdir(parents=True, exist_ok=False)
    try:
        os.chmod(root, 0o700)
        paths = {
            "manifest_path": root / "watchdog-spool.json",
            "arm_response_path": root / "arm.response",
            "arm_receipt_path": root / "arm.json",
            "delete_claim_path": root / "delete-claim.json",
            "delete_response_path": root / "delete.response",
            "delete_receipt_path": root / "delete.json",
            "hard_stop_receipt_path": root / "provider-hard-stop.json",
        }
        manifest_sha256 = write_receipt(
            paths["manifest_path"],
            study_id=STUDY_ID,
            receipt_type="watchdog_spool",
            payload={
                "status": "PRECREATED",
                "execution_domain": "owner_host_outside_provider_resource",
                "provider_credential_persisted": False,
                "provider_delete_owner": "detached_watchdog_worker",
                "host_delete_role": "single_append_only_claim",
                "zero_retry": True,
            },
        )
        return WatchdogSpool(
            root=root,
            manifest_sha256=manifest_sha256,
            **paths,
        )
    except BaseException:
        # The fresh directory contains no user data and no provider credential.
        # Leave any partial object in place so a later invocation fails closed.
        raise


def _watchdog_command(spool: WatchdogSpool) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-S",
        "-m",
        "experiments.feasibility_debt_candidate_screen.watchdog_worker",
        "--arm-response",
        str(spool.arm_response_path),
        "--arm-receipt",
        str(spool.arm_receipt_path),
        "--delete-response",
        str(spool.delete_response_path),
        "--delete-receipt",
        str(spool.delete_receipt_path),
        "--delete-claim",
        str(spool.delete_claim_path),
    ]


def _minimal_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "PATH",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    return result


def _spawn(command: list[str], *, cwd: Path) -> subprocess.Popen[bytes]:
    flags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=_minimal_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
        start_new_session=start_new_session,
    )


@dataclass
class WatchdogHandle:
    spool: WatchdogSpool
    process: _Process = field(repr=False)
    resource_id: str
    task_scope_sha256: str
    resource_request_sha256: str
    quote_sha256: str
    launch_response_sha256: str
    source_closure_sha256: str
    command_sha256: str
    process_identity_sha256: str
    trigger_utc: str
    delete_by_utc: str
    now_utc: Callable[[], datetime] = field(repr=False)
    monotonic: Callable[[], float] = field(repr=False)
    sleep: Callable[[float], None] = field(repr=False)

    def _authenticate_arm(self) -> str:
        payload, digest = read_receipt(
            self.spool.arm_receipt_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="watchdog_arm_ack",
            expected_payload_keys=ARM_KEYS,
        )
        exact = {
            "status": "ARMED",
            "resource_id": self.resource_id,
            "task_scope_sha256": self.task_scope_sha256,
            "process_id": self.process.pid,
            "source_closure_sha256": self.source_closure_sha256,
            "command_sha256": self.command_sha256,
            "trigger_utc": self.trigger_utc,
            "delete_by_utc": self.delete_by_utc,
            "credential_persisted": False,
            "zero_retry": True,
        }
        if (
            any(payload.get(key) != value for key, value in exact.items())
            or sha256_file(self.spool.arm_response_path)
            != payload["authenticated_control_response_sha256"]
            or parse_utc(payload["armed_utc"]) > parse_utc(self.trigger_utc)
        ):
            raise WatchdogError("watchdog arm receipt changed its binding")
        return digest

    def verify_live(self, payload: dict[str, object]) -> bool:
        try:
            self._authenticate_arm()
            return (
                self.process.poll() is None
                and payload.get("resource_id") == self.resource_id
                and payload.get("task_scope_sha256") == self.task_scope_sha256
                and payload.get("watchdog_source_sha256")
                == self.source_closure_sha256
                and payload.get("watchdog_process_identity_sha256")
                == self.process_identity_sha256
                and payload.get("watchdog_command_sha256") == self.command_sha256
                and payload.get("watchdog_trigger_utc") == self.trigger_utc
                and payload.get("delete_by_utc") == self.delete_by_utc
            )
        except BaseException:
            return False

    def claim_delete_once(self) -> str:
        if not self.spool.delete_claim_path.exists():
            now = self.now_utc()
            if (
                now.tzinfo is None
                or now.utcoffset() != timedelta(0)
                or now > parse_utc(self.delete_by_utc)
            ):
                raise WatchdogError("host delete claim crossed the hard horizon")
            write_receipt(
                self.spool.delete_claim_path,
                study_id=STUDY_ID,
                receipt_type="watchdog_delete_claim",
                payload={
                    "resource_id": self.resource_id,
                    "task_scope_sha256": self.task_scope_sha256,
                    "claim_utc": now.isoformat().replace("+00:00", "Z"),
                    "delete_by_utc": self.delete_by_utc,
                    "reason": "host_cleanup",
                    "one_attempt": True,
                },
            )
        payload, digest = read_receipt(
            self.spool.delete_claim_path,
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
            payload["resource_id"] != self.resource_id
            or payload["task_scope_sha256"] != self.task_scope_sha256
            or payload["delete_by_utc"] != self.delete_by_utc
            or payload["reason"] != "host_cleanup"
            or payload["one_attempt"] is not True
            or parse_utc(payload["claim_utc"]) > parse_utc(self.delete_by_utc)
        ):
            raise WatchdogError("existing delete claim changed its binding")
        return digest

    def wait_for_delete(self) -> tuple[dict[str, object], str]:
        deadline = parse_utc(self.delete_by_utc)
        now = self.now_utc()
        wait_seconds = max(0.0, (deadline - now).total_seconds())
        monotonic_deadline = self.monotonic() + wait_seconds
        while True:
            if (
                self.spool.delete_receipt_path.is_file()
                and self.spool.delete_response_path.is_file()
            ):
                payload, digest = read_receipt(
                    self.spool.delete_receipt_path,
                    expected_study_id=STUDY_ID,
                    expected_receipt_type="watchdog_delete",
                    expected_payload_keys=DELETE_KEYS,
                )
                if (
                    payload["status"] not in {"DELETED", "ALREADY_ABSENT"}
                    or payload["resource_id"] != self.resource_id
                    or payload["task_scope_sha256"] != self.task_scope_sha256
                    or payload["trigger_utc"] != self.trigger_utc
                    or payload["delete_by_utc"] != self.delete_by_utc
                    or payload["trigger_kind"]
                    not in {
                        "host_cleanup_claim",
                        "hard_horizon",
                        "malformed_claim_fail_safe",
                    }
                    or payload["zero_retry"] is not True
                    or sha256_file(self.spool.delete_response_path)
                    != payload["authenticated_response_sha256"]
                ):
                    raise WatchdogError("watchdog delete receipt changed binding")
                return payload, digest
            if self.process.poll() is not None:
                raise WatchdogError("watchdog exited without a terminal delete receipt")
            pending = monotonic_deadline - self.monotonic()
            if pending <= 0:
                raise WatchdogError("watchdog delete receipt missed the hard horizon")
            self.sleep(min(0.25, pending))

    def request_delete_once(self) -> tuple[dict[str, object], str]:
        self.claim_delete_once()
        return self.wait_for_delete()


@dataclass(frozen=True)
class ArmedWatchdog:
    handle: WatchdogHandle
    hard_stop: HardStopAuthorization


def arm_independent_watchdog(
    *,
    spool: WatchdogSpool,
    bearer_token: str,
    launch: ProviderLaunchReceipt,
    provider_launch_receipt_sha256: str,
    deadline: DeadlineClock,
    repository_root: Path,
    now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    spawn: Callable[[list[str], Path], _Process] | None = None,
) -> ArmedWatchdog:
    """Start the independent process and seal its authenticated hard stop."""
    if not spool.manifest_path.is_file() or sha256_file(spool.manifest_path) != spool.manifest_sha256:
        raise WatchdogError("watchdog spool changed after precreation")
    if (
        not isinstance(bearer_token, str)
        or not bearer_token
        or any(token in bearer_token for token in ("\r", "\n", " "))
    ):
        raise WatchdogError("injected provider credential is invalid")
    started = now_utc()
    if started.tzinfo is None or started.utcoffset() != timedelta(0):
        raise WatchdogError("watchdog host clock is not UTC")
    trigger = deadline.hard_horizon - timedelta(seconds=60)
    if started >= trigger - timedelta(seconds=30):
        raise WatchdogError("insufficient time remains to arm the watchdog")
    arm_deadline = min(started + timedelta(seconds=60), trigger - timedelta(seconds=30))
    command = _watchdog_command(spool)
    command_sha256 = sha256_bytes(canonical_json_bytes(command))
    source_sha256 = watchdog_source_closure_sha256()
    spawn_process = spawn or (lambda value, cwd: _spawn(value, cwd=cwd))
    process = spawn_process(command, repository_root.resolve())
    if process.stdin is None:
        raise WatchdogError("watchdog child lacks its one-shot stdin")
    bootstrap = WatchdogBootstrap(
        arm_deadline_utc=arm_deadline.isoformat().replace("+00:00", "Z"),
        bearer_token=bearer_token,
        command_sha256=command_sha256,
        credential_fingerprint_sha256=sha256_bytes(bearer_token.encode("utf-8")),
        delete_by_utc=deadline.hard_horizon.isoformat().replace("+00:00", "Z"),
        resource_id=launch.resource_id,
        source_closure_sha256=source_sha256,
        task_scope_sha256=launch.task_scope_sha256,
        trigger_utc=trigger.isoformat().replace("+00:00", "Z"),
    )
    bootstrap.validate()
    content = canonical_json_bytes(bootstrap.__dict__)
    try:
        process.stdin.write(content)  # type: ignore[attr-defined]
        process.stdin.flush()  # type: ignore[attr-defined]
    finally:
        process.stdin.close()  # type: ignore[attr-defined]
    process_identity_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "pid": process.pid,
                "parent_pid": os.getpid(),
                "started_utc": started.isoformat().replace("+00:00", "Z"),
                "executable_sha256": sha256_file(Path(sys.executable).resolve()),
                "command_sha256": command_sha256,
                "environment_sha256": sha256_bytes(
                    canonical_json_bytes(_minimal_environment())
                ),
            }
        )
    )
    handle = WatchdogHandle(
        spool=spool,
        process=process,
        resource_id=launch.resource_id,
        task_scope_sha256=launch.task_scope_sha256,
        resource_request_sha256=launch.resource_request_sha256,
        quote_sha256=launch.quote_sha256,
        launch_response_sha256=launch.authenticated_response_sha256,
        source_closure_sha256=source_sha256,
        command_sha256=command_sha256,
        process_identity_sha256=process_identity_sha256,
        trigger_utc=bootstrap.trigger_utc,
        delete_by_utc=bootstrap.delete_by_utc,
        now_utc=now_utc,
        monotonic=monotonic,
        sleep=sleep,
    )
    monotonic_deadline = monotonic() + max(
        0.0, (arm_deadline - now_utc()).total_seconds()
    )
    while not (
        spool.arm_receipt_path.is_file() and spool.arm_response_path.is_file()
    ):
        if process.poll() is not None:
            raise WatchdogError("watchdog exited before arming")
        pending = monotonic_deadline - monotonic()
        if pending <= 0:
            raise WatchdogError("watchdog missed its arm deadline")
        sleep(min(0.1, pending))
    handle._authenticate_arm()
    write_receipt(
        spool.hard_stop_receipt_path,
        study_id=STUDY_ID,
        receipt_type="provider_hard_stop",
        payload={
            "status": "ARMED",
            "provider": "runpod",
            "resource_id": launch.resource_id,
            "task_scope_sha256": launch.task_scope_sha256,
            "resource_request_sha256": launch.resource_request_sha256,
            "resource_manifest_sha256": launch.resource_manifest_sha256,
            "provider_launch_receipt_sha256": provider_launch_receipt_sha256,
            "quote_sha256": launch.quote_sha256,
            "control_kind": "independent_watchdog_process",
            "watchdog_source_sha256": source_sha256,
            "watchdog_process_identity_sha256": process_identity_sha256,
            "watchdog_command_sha256": command_sha256,
            "watchdog_started_utc": started.isoformat().replace("+00:00", "Z"),
            "watchdog_trigger_utc": bootstrap.trigger_utc,
            "delete_by_utc": bootstrap.delete_by_utc,
            "credential_scope": "delete_task_pod_only",
            "authenticated_control_response_sha256": sha256_file(
                spool.arm_response_path
            ),
        },
    )
    hard_stop = authenticate_hard_stop(
        spool.hard_stop_receipt_path,
        authenticated_control_response_path=spool.arm_response_path,
        launch=launch,
        provider_launch_receipt_sha256=provider_launch_receipt_sha256,
        deadline=deadline,
        verify_live_watchdog=handle.verify_live,
    )
    return ArmedWatchdog(handle=handle, hard_stop=hard_stop)


@dataclass
class WatchdogOwnedResourceAdapter(ResourceAdapter):
    """Read through the host adapter; route the sole DELETE to the watchdog."""

    watchdog: WatchdogHandle
    read_adapter: ResourceAdapter = field(repr=False)
    binding: BillingBinding

    def delete(
        self,
        kind: str,
        resource_id: str,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        if (
            kind != "pod"
            or resource_id != self.binding.resource_id
            or resource_id != self.watchdog.resource_id
            or deadline_utc != self.watchdog.delete_by_utc
        ):
            raise WatchdogError("watchdog delete request changed its sole binding")
        payload, _ = self.watchdog.request_delete_once()
        exclusive_write_bytes(
            authenticated_response_path,
            self.watchdog.spool.delete_response_path.read_bytes(),
        )
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_delete",
            payload={
                "provider": "runpod",
                "status": "DELETED",
                "kind": "pod",
                "id": resource_id,
                "task_scope_sha256": self.binding.task_scope_sha256,
                "resource_request_sha256": self.watchdog.resource_request_sha256,
                "quote_sha256": self.binding.quote_sha256,
                "launch_response_sha256": self.binding.launch_response_sha256,
                "observed_utc": payload["observed_utc"],
                "authenticated_response_sha256": sha256_file(
                    authenticated_response_path
                ),
            },
        )

    def scoped_inventory(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.read_adapter.scoped_inventory(
            receipt_path=receipt_path,
            authenticated_response_path=authenticated_response_path,
            deadline_utc=deadline_utc,
        )

    def billing_receipt(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.read_adapter.billing_receipt(
            receipt_path=receipt_path,
            authenticated_response_path=authenticated_response_path,
            deadline_utc=deadline_utc,
        )


__all__ = [
    "ArmedWatchdog",
    "WatchdogError",
    "WatchdogHandle",
    "WatchdogOwnedResourceAdapter",
    "WatchdogSpool",
    "arm_independent_watchdog",
    "prepare_watchdog_spool",
]
