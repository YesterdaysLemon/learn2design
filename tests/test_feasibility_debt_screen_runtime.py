from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import os
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen.cleanup import (
    BillingBinding,
    CleanupError,
    EvidenceHandoffBinding,
    ProcessIdentity,
    cleanup_process_tree_once,
    terminate_tree_bottom_up,
    validate_resource_manifest,
)
from experiments.feasibility_debt_candidate_screen.packet import (
    PacketError,
    parse_worker_packet,
    write_closed_failure_receipt,
)
from experiments.feasibility_debt_candidate_screen.canonical import (
    canonical_json_bytes,
    read_receipt,
    sha256_bytes,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import STUDY_ID
from experiments.feasibility_debt_candidate_screen.runtime import (
    BillingSemantics,
    DeadlineClock,
    DeadlineController,
    Phase,
    ProviderLaunchReceipt,
    RuntimeGuardError,
    validate_success_bucket_total,
)
from experiments.feasibility_debt_candidate_screen.evacuation import (
    EvacuationError,
    evacuate_and_authenticate,
)
from experiments.feasibility_debt_candidate_screen.orchestrator import (
    authenticate_terminal_outcome,
)


def _launch_receipt(
    *,
    running_utc: str = "2026-09-01T00:00:00Z",
    billable_utc: str | None = "2026-09-01T00:00:00Z",
    create_utc: str = "2026-09-01T00:00:00Z",
) -> ProviderLaunchReceipt:
    return ProviderLaunchReceipt(
        provider="runpod",
        quote_sha256="1" * 64,
        authenticated_response_sha256="2" * 64,
        resource_request_sha256="4" * 64,
        resource_manifest_sha256="3" * 64,
        task_scope_sha256="5" * 64,
        resource_id="pod-task-owned",
        immutable_image_digest="sha256:" + "6" * 64,
        status="RUNNING",
        create_utc=create_utc,
        running_utc=running_utc,
        billable_utc=billable_utc,
        cloud_type="SECURE",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        ephemeral_disk_gib=40,
        provider_running_hourly_cost_usd="3.29",
        billing=BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal("3.29"),
            combined_hourly_rate_usd=Decimal("3.29"),
        ),
    )


def test_decimal_deadline_uses_earliest_t0_b0_and_charge_cap() -> None:
    receipt = _launch_receipt(
        running_utc="2026-09-01T00:01:00Z",
        billable_utc=None,
        create_utc="2026-09-01T00:00:00Z",
    )
    billing = receipt.billing
    clock = DeadlineClock.from_authenticated_receipt(receipt)
    assert clock.t0 == datetime(2026, 9, 1, 0, 1, tzinfo=UTC)
    assert clock.b0 == datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    assert clock.hard_horizon == datetime(2026, 9, 1, 7, 0, tzinfo=UTC)
    assert clock.dispatch_deadline == datetime(2026, 9, 1, 6, 30, tzinfo=UTC)
    assert billing.charge_at_seconds(25_200) == Decimal("23.03")
    assert validate_success_bucket_total() == 20_100


def test_deadline_admission_preserves_cleanup_reserve_and_no_borrowing() -> None:
    clock = DeadlineClock.from_authenticated_receipt(_launch_receipt())
    clock.require_success_envelope(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    clock.admit("scored_worker", datetime(2026, 9, 1, 6, 18, tzinfo=UTC))
    with pytest.raises(RuntimeGuardError, match="does not fit"):
        clock.admit("scored_worker", datetime(2026, 9, 1, 6, 19, tzinfo=UTC))
    clock.admit("cleanup_deletion", datetime(2026, 9, 1, 6, 45, tzinfo=UTC))
    with pytest.raises(RuntimeGuardError):
        clock.admit("cleanup_deletion", datetime(2026, 9, 1, 6, 46, tzinfo=UTC))

    liveness_checks = {"count": 0}

    def assert_live() -> None:
        liveness_checks["count"] += 1

    controller = DeadlineController(
        clock,
        provider_launch_receipt_sha256="b" * 64,
        resource_manifest_sha256="c" * 64,
        hard_stop_receipt_sha256="d" * 64,
        hard_stop_liveness=assert_live,
    )
    controller.transition(Phase.SMOKE)
    with pytest.raises(RuntimeGuardError, match="verified cold-smoke"):
        controller.transition(Phase.STAGE1)
    controller.accept_verified_smoke("a" * 64)
    controller.admit_scored_worker(datetime(2026, 9, 1, 0, 10, tzinfo=UTC))
    assert controller.dispatched_scored_runs == 1
    assert liveness_checks["count"] == 1
    assert controller.enforce_watchdog(clock.hard_horizon)
    assert controller.phase is Phase.CLEANUP


def test_packet_parser_is_binary_strict_and_failure_receipt_is_hash_only(
    tmp_path: Path,
) -> None:
    packet = {"a": 1, "b": False}
    raw = canonical_json_bytes(packet)
    assert parse_worker_packet(
        raw,
        raw_stderr=b"",
        returncode=0,
        timed_out=False,
        expected_keys={"a", "b"},
    ) == packet
    for bad, code in (
        (raw + b"\n", "noncanonical_framing"),
        (b'{"a":1,"a":1,"b":false}', "duplicate_key"),
        (b'{"a":NaN,"b":false}', "nonfinite_number"),
    ):
        with pytest.raises(PacketError) as raised:
            parse_worker_packet(
                bad,
                raw_stderr=b"",
                returncode=0,
                timed_out=False,
                expected_keys={"a", "b"},
            )
        assert raised.value.code == code
    with pytest.raises(PacketError) as raised:
        parse_worker_packet(
            raw,
            raw_stderr=b"warning",
            returncode=0,
            timed_out=False,
            expected_keys={"a", "b"},
        )
    assert raised.value.code == "stderr_not_empty"

    path = tmp_path / "worker-failure.json"
    write_closed_failure_receipt(
        path,
        raw_stdout=b"secret outcome",
        raw_stderr=b"trace",
        returncode=1,
        timed_out=False,
        error_code="invalid_json",
    )
    payload, _ = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="worker_failure",
    )
    assert payload["raw_output_included"] is False
    assert "secret" not in path.read_text(encoding="utf-8")
    assert set(payload) == {
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
        "returncode",
        "timed_out",
        "error_code",
        "raw_output_included",
    }


class _ProcessAdapter:
    def __init__(self, rows: list[ProcessIdentity]) -> None:
        self.rows = {row.pid: row for row in rows}
        self.terminated: list[int] = []

    def snapshot_tree(self, root_pid: int):
        return list(self.rows.values())

    def identity(self, pid: int):
        return self.rows.get(pid)

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)
        self.rows.pop(pid, None)

    def descendants(self, root_pid: int):
        return list(self.rows)


class _Resources:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def delete(
        self,
        kind: str,
        resource_id: str,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.deleted.append((kind, resource_id))

    def scoped_inventory(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        return None

    def billing_receipt(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> dict[str, object]:
        amount = BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal("3.29"),
            combined_hourly_rate_usd=Decimal("3.29"),
        ).charge_at_seconds(19_235)
        return {
            "provider": "runpod",
            "resource_id": "pod-task-owned",
            "task_scope_sha256": "c" * 64,
            "quote_sha256": "1" * 64,
            "launch_response_sha256": "2" * 64,
            "observed_utc": "2026-09-01T06:42:00Z",
            "billing_query": {
                "startTime": "2026-09-01T00:00:00Z",
                "endTime": "2026-09-01T07:00:00Z",
                "bucketSize": "hour",
                "podId": "pod-task-owned",
            },
            "billing_query_sha256": sha256_bytes(
                canonical_json_bytes(
                    {
                        "startTime": "2026-09-01T00:00:00Z",
                        "endTime": "2026-09-01T07:00:00Z",
                        "bucketSize": "hour",
                        "podId": "pod-task-owned",
                    }
                )
            ),
            "provider_record_count": 7,
            "provider_unique_pod_count": 1,
            "currency": "USD",
            "envelope_seconds": 19_235,
            "gpu_hourly_rate_usd": "3.29",
            "combined_hourly_rate_usd": "3.29",
            "fixed_charge_usd": "0",
            "metering_quantum_seconds": 1,
            "round_up_each_quantum": False,
            "gpu_charge_bound_usd": str(amount),
            "all_in_charge_bound_usd": str(amount),
            "provider_gpu_charge_usd": str(amount),
            "provider_disk_charge_usd": "0",
            "provider_total_charge_usd": str(amount),
            "provider_receipt_sha256": "d" * 64,
        }


def _identity(pid: int, parent: int) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        parent_pid=parent,
        start_timestamp=f"t-{pid}",
        executable_sha256=f"{pid:064x}",
        command_line_sha256=f"{pid + 1:064x}",
    )


def _manifest() -> dict[str, object]:
    return {
        "provider": "runpod",
        "cloud_type": "SECURE",
        "pod": {
            "id": "pod-task-owned",
            "gpu_type_id": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "immutable_image_digest": "sha256:" + "6" * 64,
        },
        "ephemeral_disk_gib": 40,
        "other_resources": [],
        "task_scope_sha256": "c" * 64,
        "resource_request_sha256": "4" * 64,
    }


def _billing_binding() -> BillingBinding:
    return BillingBinding(
        resource_id="pod-task-owned",
        task_scope_sha256="c" * 64,
        quote_sha256="1" * 64,
        launch_response_sha256="2" * 64,
        gpu_hourly_rate_usd="3.29",
        combined_hourly_rate_usd="3.29",
    )


def _evidence_authorization(tmp_path: Path):
    source = tmp_path / "attempt-source"
    source.mkdir()
    terminal, terminal_authorization = _write_terminal(source)
    process_cleanup_path = source / "process-cleanup.json"
    process_cleanup = cleanup_process_tree_once(
        process_cleanup_path,
        adapter=_ProcessAdapter([_identity(20, 0), _identity(21, 20)]),
        root_pid=20,
    )
    return evacuate_and_authenticate(
        source_attempt_root=source,
        destination_root=tmp_path / "evacuated",
        source_terminal_path=terminal,
        process_cleanup=process_cleanup,
        process_cleanup_receipt_path=process_cleanup_path,
        terminal_authorization=terminal_authorization,
        pod_error_code="provider_preflight:RuntimeError",
        binding=_handoff_binding(),
    )


def _handoff_binding() -> EvidenceHandoffBinding:
    return EvidenceHandoffBinding(
        resource_id="pod-task-owned",
        panel_sha256="7" * 64,
        panel_commitment_sha256="8" * 64,
        split_receipt_sha256="9" * 64,
        package_closure_sha256="a" * 64,
        provider_launch_receipt_sha256="1" * 64,
        host_finalizer_receipt_sha256="2" * 64,
        terminal_attempt_sha256="3" * 64,
        implementation_revision="4" * 40,
        source_lock_sha256="5" * 64,
        runtime_lock_sha256="6" * 64,
    )


def _write_terminal(source: Path):
    terminal = source / "sealed" / "study-outcome.json"
    terminal.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(
        terminal,
        study_id=STUDY_ID,
        receipt_type="terminal_outcome",
        payload={
            "status": "not_evaluable",
            "action": "retain_round1_control_attempt_not_evaluable",
            "revision": "4" * 40,
            "panel_sha256": "7" * 64,
            "source_lock_sha256": "5" * 64,
            "runtime_lock_sha256": "6" * 64,
            "terminal_attempt_sha256": "3" * 64,
            "failed_phase": "provider_preflight",
            "error_code": "provider_preflight:RuntimeError",
            "selection_receipt_sha256": None,
            "stage2_verification_sha256": None,
            "stage2_outcome_opened": False,
            "organizer_score_comparable": False,
            "raw_output_included": False,
        },
    )
    return terminal, authenticate_terminal_outcome(
        terminal,
        evidence_root=source,
        expected_revision="4" * 40,
        expected_panel_sha256="7" * 64,
        expected_panel_commitment_sha256="8" * 64,
        expected_split_receipt_sha256="9" * 64,
        expected_package_closure_sha256="a" * 64,
        expected_source_lock_sha256="5" * 64,
        expected_runtime_lock_sha256="6" * 64,
        expected_terminal_attempt_sha256="3" * 64,
    )


def test_process_cleanup_precedes_authenticated_evacuation(
    tmp_path: Path,
) -> None:
    rows = [_identity(10, 0), _identity(11, 10), _identity(12, 11)]
    processes = _ProcessAdapter(rows)
    source = tmp_path / "source"
    source.mkdir()
    terminal, terminal_authorization = _write_terminal(source)
    process_path = source / "process-cleanup.json"
    authorization = cleanup_process_tree_once(
        process_path,
        adapter=processes,
        root_pid=10,
    )
    assert processes.terminated == [12, 11, 10]
    assert authorization.status == "complete"
    evacuated = evacuate_and_authenticate(
        source_attempt_root=source,
        destination_root=tmp_path / "destination",
        source_terminal_path=terminal,
        process_cleanup=authorization,
        process_cleanup_receipt_path=process_path,
        terminal_authorization=terminal_authorization,
        pod_error_code="provider_preflight:RuntimeError",
        binding=_handoff_binding(),
    )
    assert evacuated.process_cleanup_receipt_sha256 == authorization.receipt_sha256


def test_process_cleanup_failure_is_sealed_and_self_delete_is_forbidden(
    tmp_path: Path,
) -> None:
    rows = [_identity(10, 0), _identity(11, 10)]
    processes = _ProcessAdapter(rows)
    processes.rows[11] = _identity(11, 999)
    authorization = cleanup_process_tree_once(
        tmp_path / "failed-process-cleanup.json",
        adapter=processes,
        root_pid=10,
    )
    assert authorization.status == "attempt_not_evaluable"
    self_authorization = cleanup_process_tree_once(
        tmp_path / "self-process-cleanup.json",
        adapter=_ProcessAdapter([]),
        root_pid=os.getpid(),
    )
    assert self_authorization.status == "attempt_not_evaluable"
    bad = _manifest()
    bad["other_resources"] = [{"kind": "volume", "id": "x"}]
    with pytest.raises(CleanupError, match="extra"):
        validate_resource_manifest(bad)


def test_pod_evidence_cannot_prepopulate_host_finalizer_namespace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    reserved = source / "post-cleanup"
    reserved.mkdir(parents=True)
    (reserved / "collision").write_bytes(b"forbidden")
    terminal, terminal_authorization = _write_terminal(source)
    process_path = source / "process-cleanup.json"
    process_cleanup = cleanup_process_tree_once(
        process_path,
        adapter=_ProcessAdapter([_identity(30, 0)]),
        root_pid=30,
    )
    with pytest.raises(EvacuationError, match="host-reserved"):
        evacuate_and_authenticate(
            source_attempt_root=source,
            destination_root=tmp_path / "destination",
            source_terminal_path=terminal,
            process_cleanup=process_cleanup,
            process_cleanup_receipt_path=process_path,
            terminal_authorization=terminal_authorization,
            pod_error_code="provider_preflight:RuntimeError",
            binding=_handoff_binding(),
        )
