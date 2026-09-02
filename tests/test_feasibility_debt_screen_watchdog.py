from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import io
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen.canonical import (
    canonical_json_bytes,
    exclusive_write_bytes,
    read_receipt,
    sha256_bytes,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import STUDY_ID
from experiments.feasibility_debt_candidate_screen.runpod_adapter import (
    ApiExchange,
    transcript_bytes,
)
from experiments.feasibility_debt_candidate_screen.runtime import (
    BillingSemantics,
    DeadlineClock,
    ProviderLaunchReceipt,
)
from experiments.feasibility_debt_candidate_screen.watchdog import (
    arm_independent_watchdog,
    prepare_watchdog_spool,
)
from experiments.feasibility_debt_candidate_screen.watchdog_worker import (
    WatchdogBootstrap,
    execute_watchdog,
    watchdog_source_closure_sha256,
)


TOKEN = "watchdog-test-token"
RESOURCE_ID = "pod-one"
TASK_SCOPE = "1" * 64
DELETE_BY = "2026-09-01T07:00:00Z"
TRIGGER = "2026-09-01T06:59:00Z"


class _QueueTransport:
    def __init__(self, statuses: list[tuple[str, int, str]]) -> None:
        self.statuses = list(statuses)
        self.calls: list[tuple[str, str, str]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        deadline_utc: str,
    ) -> ApiExchange:
        assert body is None
        expected_method, status, observed_utc = self.statuses.pop(0)
        assert method == expected_method
        self.calls.append((method, path, deadline_utc))
        return ApiExchange(
            method=method,
            path=path,
            status=status,
            observed_utc=observed_utc,
            content_type="application/json",
            request_id="watchdog-test-request",
            body=b'{"id":"pod-one"}' if method == "GET" else b"",
            credential_fingerprint_sha256=sha256_bytes(TOKEN.encode("utf-8")),
        )


def _bootstrap() -> WatchdogBootstrap:
    value = {
        "arm_deadline_utc": "2026-09-01T00:02:00Z",
        "bearer_token": TOKEN,
        "command_sha256": "2" * 64,
        "credential_fingerprint_sha256": sha256_bytes(TOKEN.encode("utf-8")),
        "delete_by_utc": DELETE_BY,
        "resource_id": RESOURCE_ID,
        "source_closure_sha256": watchdog_source_closure_sha256(),
        "task_scope_sha256": TASK_SCOPE,
        "trigger_utc": TRIGGER,
    }
    return WatchdogBootstrap.parse(canonical_json_bytes(value))


def _paths(root: Path) -> dict[str, Path]:
    return {
        "arm_response_path": root / "arm.response",
        "arm_receipt_path": root / "arm.json",
        "delete_response_path": root / "delete.response",
        "delete_receipt_path": root / "delete.json",
        "delete_claim_path": root / "delete-claim.json",
    }


def test_watchdog_consumes_valid_host_claim_and_deletes_exactly_once(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    write_receipt(
        paths["delete_claim_path"],
        study_id=STUDY_ID,
        receipt_type="watchdog_delete_claim",
        payload={
            "resource_id": RESOURCE_ID,
            "task_scope_sha256": TASK_SCOPE,
            "claim_utc": "2026-09-01T00:10:00Z",
            "delete_by_utc": DELETE_BY,
            "reason": "host_cleanup",
            "one_attempt": True,
        },
    )
    transport = _QueueTransport(
        [
            ("GET", 200, "2026-09-01T00:01:00Z"),
            ("DELETE", 204, "2026-09-01T00:10:01Z"),
        ]
    )
    execute_watchdog(
        bootstrap=_bootstrap(),
        transport=transport,
        now_utc=lambda: datetime(2026, 9, 1, 0, 1, tzinfo=UTC),
        monotonic=lambda: 10.0,
        sleep=lambda _seconds: pytest.fail("valid claim should delete immediately"),
        **paths,
    )
    assert transport.calls == [
        ("GET", "/v2/pods/pod-one", "2026-09-01T00:02:00Z"),
        ("DELETE", "/v2/pods/pod-one", DELETE_BY),
    ]
    delete, _ = read_receipt(
        paths["delete_receipt_path"],
        expected_study_id=STUDY_ID,
        expected_receipt_type="watchdog_delete",
    )
    assert delete["status"] == "DELETED"
    assert delete["trigger_kind"] == "host_cleanup_claim"
    assert delete["delete_claim_sha256"] is not None
    assert TOKEN.encode("utf-8") not in paths["arm_response_path"].read_bytes()
    assert TOKEN.encode("utf-8") not in paths["delete_response_path"].read_bytes()


def test_watchdog_deletes_once_at_hard_trigger_without_host_claim(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    transport = _QueueTransport(
        [
            ("GET", 200, "2026-09-01T00:01:00Z"),
            ("DELETE", 404, "2026-09-01T06:59:01Z"),
        ]
    )
    execute_watchdog(
        bootstrap=_bootstrap(),
        transport=transport,
        now_utc=lambda: datetime(2026, 9, 1, 6, 59, tzinfo=UTC),
        monotonic=lambda: 20.0,
        sleep=lambda _seconds: pytest.fail("trigger is already due"),
        **paths,
    )
    assert [row[0] for row in transport.calls] == ["GET", "DELETE"]
    delete, _ = read_receipt(
        paths["delete_receipt_path"],
        expected_study_id=STUDY_ID,
        expected_receipt_type="watchdog_delete",
    )
    assert delete["status"] == "ALREADY_ABSENT"
    assert delete["trigger_kind"] == "hard_horizon"
    assert delete["delete_claim_sha256"] is None


def test_malformed_claim_fails_safe_to_one_delete(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["delete_claim_path"].write_bytes(b"not-a-receipt")
    transport = _QueueTransport(
        [
            ("GET", 200, "2026-09-01T00:01:00Z"),
            ("DELETE", 204, "2026-09-01T00:01:01Z"),
        ]
    )
    execute_watchdog(
        bootstrap=_bootstrap(),
        transport=transport,
        now_utc=lambda: datetime(2026, 9, 1, 0, 1, tzinfo=UTC),
        monotonic=lambda: 30.0,
        sleep=lambda _seconds: pytest.fail("malformed claim should fail safe"),
        **paths,
    )
    assert [row[0] for row in transport.calls] == ["GET", "DELETE"]
    delete, _ = read_receipt(
        paths["delete_receipt_path"],
        expected_study_id=STUDY_ID,
        expected_receipt_type="watchdog_delete",
    )
    assert delete["trigger_kind"] == "malformed_claim_fail_safe"
    assert delete["delete_claim_sha256"] is None


class _ArmStdin(io.BytesIO):
    def __init__(self, spool, process) -> None:
        super().__init__()
        self._spool = spool
        self._process = process

    def close(self) -> None:
        content = self.getvalue()
        bootstrap = WatchdogBootstrap.parse(content)
        exchange = ApiExchange(
            method="GET",
            path=f"/v2/pods/{bootstrap.resource_id}",
            status=200,
            observed_utc="2026-09-01T00:01:01Z",
            content_type="application/json",
            request_id="watchdog-arm-test",
            body=b'{"id":"pod-one"}',
            credential_fingerprint_sha256=(
                bootstrap.credential_fingerprint_sha256
            ),
        )
        exclusive_write_bytes(
            self._spool.arm_response_path,
            transcript_bytes([exchange]),
        )
        write_receipt(
            self._spool.arm_receipt_path,
            study_id=STUDY_ID,
            receipt_type="watchdog_arm_ack",
            payload={
                "status": "ARMED",
                "resource_id": bootstrap.resource_id,
                "task_scope_sha256": bootstrap.task_scope_sha256,
                "process_id": self._process.pid,
                "source_closure_sha256": bootstrap.source_closure_sha256,
                "command_sha256": bootstrap.command_sha256,
                "armed_utc": "2026-09-01T00:01:01Z",
                "trigger_utc": bootstrap.trigger_utc,
                "delete_by_utc": bootstrap.delete_by_utc,
                "authenticated_control_response_sha256": sha256_bytes(
                    self._spool.arm_response_path.read_bytes()
                ),
                "credential_persisted": False,
                "zero_retry": True,
            },
        )
        super().close()


class _ArmProcess:
    pid = 4242

    def __init__(self, spool) -> None:
        self.stdin = _ArmStdin(spool, self)

    def poll(self) -> int | None:
        return None


def _launch() -> ProviderLaunchReceipt:
    return ProviderLaunchReceipt(
        provider="runpod",
        quote_sha256="a" * 64,
        authenticated_response_sha256="b" * 64,
        resource_request_sha256="c" * 64,
        resource_manifest_sha256="d" * 64,
        task_scope_sha256=TASK_SCOPE,
        resource_id=RESOURCE_ID,
        immutable_image_digest="sha256:" + "e" * 64,
        status="RUNNING",
        create_utc="2026-09-01T00:00:00Z",
        running_utc="2026-09-01T00:00:00Z",
        billable_utc="2026-09-01T00:00:00Z",
        cloud_type="SECURE",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        ephemeral_disk_gib=40,
        provider_running_hourly_cost_usd="3.29",
        billing=BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal("3.29"),
            combined_hourly_rate_usd=Decimal("3.30"),
            fixed_charge_usd=Decimal("0"),
            metering_quantum_seconds=1,
            round_up_each_quantum=False,
        ),
    )


def test_parent_arms_detached_watchdog_without_credential_in_command_or_disk(
    tmp_path: Path,
) -> None:
    spool = prepare_watchdog_spool(tmp_path / "watchdog-spool")
    launch = _launch()
    commands: list[list[str]] = []

    def spawn(command: list[str], _cwd: Path):
        commands.append(command)
        return _ArmProcess(spool)

    armed = arm_independent_watchdog(
        spool=spool,
        bearer_token=TOKEN,
        launch=launch,
        provider_launch_receipt_sha256="f" * 64,
        deadline=DeadlineClock.from_authenticated_receipt(launch),
        repository_root=tmp_path,
        now_utc=lambda: datetime(2026, 9, 1, 0, 1, tzinfo=UTC),
        monotonic=lambda: 10.0,
        sleep=lambda _seconds: pytest.fail("arm receipt is synchronously ready"),
        spawn=spawn,
    )
    assert armed.hard_stop.watchdog_trigger_utc == TRIGGER
    assert armed.handle.verify_live(
        read_receipt(
            spool.hard_stop_receipt_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="provider_hard_stop",
        )[0]
    )
    assert TOKEN not in " ".join(commands[0])
    for path in spool.root.rglob("*"):
        if path.is_file():
            assert TOKEN.encode("utf-8") not in path.read_bytes()
