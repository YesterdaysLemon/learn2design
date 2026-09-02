from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen.canonical import (
    canonical_json_bytes,
    exclusive_write_bytes,
    read_receipt,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.cleanup import (
    BillingBinding,
    EvidenceHandoffBinding,
    ProcessIdentity,
    authenticate_evidence_evacuation,
    cleanup_process_tree_once,
)
from experiments.feasibility_debt_candidate_screen.contract import STUDY_ID
from experiments.feasibility_debt_candidate_screen.evacuation import (
    evacuate_and_authenticate,
)
from experiments.feasibility_debt_candidate_screen.hard_stop import (
    authenticate_hard_stop,
)
from experiments.feasibility_debt_candidate_screen.orchestrator import (
    authenticate_terminal_outcome,
)
from experiments.feasibility_debt_candidate_screen.host_finalizer import (
    HostFinalizerError,
    authenticate_host_finalizer,
    evidence_destination_sha256,
    finalize_host_attempt,
)
from experiments.feasibility_debt_candidate_screen.runtime import (
    BillingSemantics,
    DeadlineClock,
    ProviderLaunchReceipt,
)
from experiments.feasibility_debt_candidate_screen.supervisor import (
    supervise_paid_attempt,
)


SCOPE = "6" * 64
REQUEST = "4" * 64
QUOTE = "1" * 64
LAUNCH_RESPONSE = "2" * 64
LAUNCH_RECEIPT = "7" * 64
HARD_STOP_RECEIPT_SOURCE = "8" * 64
TERMINAL_ATTEMPT = "9" * 64
SOURCE = "a" * 64
RUNTIME = "b" * 64
REVISION = "c" * 40
OWNER = "5" * 64
PANEL = "0" * 64
PANEL_COMMITMENT = "e" * 64
SPLIT = "f" * 64
PACKAGE = "3" * 64

_MANIFEST = {
    "provider": "runpod",
    "cloud_type": "SECURE",
    "pod": {
        "id": "pod-task-owned",
        "gpu_type_id": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "immutable_image_digest": "sha256:" + "d" * 64,
    },
    "ephemeral_disk_gib": 40,
    "other_resources": [],
    "task_scope_sha256": SCOPE,
    "resource_request_sha256": REQUEST,
}
MANIFEST_DIGEST = sha256_bytes(canonical_json_bytes(_MANIFEST))


def _launch() -> ProviderLaunchReceipt:
    return ProviderLaunchReceipt(
        provider="runpod",
        quote_sha256=QUOTE,
        authenticated_response_sha256=LAUNCH_RESPONSE,
        resource_request_sha256=REQUEST,
        resource_manifest_sha256=MANIFEST_DIGEST,
        task_scope_sha256=SCOPE,
        resource_id="pod-task-owned",
        immutable_image_digest="sha256:" + "d" * 64,
        status="RUNNING",
        create_utc="2026-09-01T00:00:00Z",
        running_utc="2026-09-01T00:01:00Z",
        billable_utc="2026-09-01T00:00:00Z",
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


def _manifest() -> dict[str, object]:
    import copy

    return copy.deepcopy(_MANIFEST)


def _billing_binding() -> BillingBinding:
    return BillingBinding(
        resource_id="pod-task-owned",
        task_scope_sha256=SCOPE,
        quote_sha256=QUOTE,
        launch_response_sha256=LAUNCH_RESPONSE,
        gpu_hourly_rate_usd="3.29",
        combined_hourly_rate_usd="3.29",
    )


class _Processes:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.rows = {
            20: ProcessIdentity(20, 0, "t-20", "1" * 64, "2" * 64),
            21: ProcessIdentity(21, 20, "t-21", "3" * 64, "4" * 64),
        }

    def snapshot_tree(self, root_pid: int):
        return list(self.rows.values())

    def identity(self, pid: int):
        return self.rows.get(pid)

    def terminate(self, pid: int) -> None:
        self.events.append(f"terminate:{pid}")
        self.rows.pop(pid, None)

    def descendants(self, root_pid: int):
        return list(self.rows)


class _HostResources:
    def __init__(self, events: list[str], *, nonzero_inventory: bool = False) -> None:
        self.events = events
        self.nonzero_inventory = nonzero_inventory
        self.deadlines: list[str] = []

    @staticmethod
    def _response(path: Path, content: bytes) -> str:
        exclusive_write_bytes(path, content)
        return sha256_file(path)

    def delete(
        self,
        kind: str,
        resource_id: str,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.events.append("delete")
        self.deadlines.append(deadline_utc)
        response_sha256 = self._response(
            authenticated_response_path,
            b"authenticated provider delete response",
        )
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_delete",
            payload={
                "provider": "runpod",
                "status": "DELETED",
                "kind": kind,
                "id": resource_id,
                "task_scope_sha256": SCOPE,
                "resource_request_sha256": REQUEST,
                "quote_sha256": QUOTE,
                "launch_response_sha256": LAUNCH_RESPONSE,
                "observed_utc": "2026-09-01T06:40:00Z",
                "authenticated_response_sha256": response_sha256,
            },
        )

    def scoped_inventory(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.events.append("inventory")
        self.deadlines.append(deadline_utc)
        response_sha256 = self._response(
            authenticated_response_path,
            b"authenticated provider inventory response",
        )
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_inventory",
            payload={
                "provider": "runpod",
                "observed_utc": "2026-09-01T06:41:00Z",
                "authenticated_response_sha256": response_sha256,
                "task_scope_sha256": SCOPE,
                "pods": (
                    [{"id": "pod-task-owned"}]
                    if self.nonzero_inventory
                    else []
                ),
                "clusters": [],
                "network_volumes": [],
                "endpoints": [],
                "templates": [],
                "registries": [],
            },
        )

    def billing_receipt(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        self.events.append("billing")
        self.deadlines.append(deadline_utc)
        response_sha256 = self._response(
            authenticated_response_path,
            b"authenticated provider billing response",
        )
        gpu_amount = BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal("3.29"),
            combined_hourly_rate_usd=Decimal("3.29"),
        ).charge_at_seconds(19_235)
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_billing",
            payload={
                "provider": "runpod",
                "resource_id": "pod-task-owned",
                "task_scope_sha256": SCOPE,
                "quote_sha256": QUOTE,
                "launch_response_sha256": LAUNCH_RESPONSE,
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
                "gpu_charge_bound_usd": str(gpu_amount),
                "all_in_charge_bound_usd": str(gpu_amount),
                "provider_gpu_charge_usd": str(gpu_amount),
                "provider_disk_charge_usd": "0",
                "provider_total_charge_usd": str(gpu_amount),
                "provider_receipt_sha256": response_sha256,
            },
        )


def _hard_stop(tmp_path: Path):
    launch = _launch()
    response = tmp_path / "watchdog.response"
    response.write_bytes(b"watchdog control response")
    path = tmp_path / "watchdog.json"
    payload = {
        "status": "ARMED",
        "provider": "runpod",
        "resource_id": launch.resource_id,
        "task_scope_sha256": SCOPE,
        "resource_request_sha256": REQUEST,
        "resource_manifest_sha256": MANIFEST_DIGEST,
        "provider_launch_receipt_sha256": LAUNCH_RECEIPT,
        "quote_sha256": QUOTE,
        "control_kind": "independent_watchdog_process",
        "watchdog_source_sha256": HARD_STOP_RECEIPT_SOURCE,
        "watchdog_process_identity_sha256": "e" * 64,
        "watchdog_command_sha256": "f" * 64,
        "watchdog_started_utc": "2026-09-01T00:01:01Z",
        "watchdog_trigger_utc": "2026-09-01T06:59:00Z",
        "delete_by_utc": "2026-09-01T07:00:00Z",
        "credential_scope": "delete_task_pod_only",
        "authenticated_control_response_sha256": sha256_file(response),
    }
    write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="provider_hard_stop",
        payload=payload,
    )
    return authenticate_hard_stop(
        path,
        authenticated_control_response_path=response,
        launch=launch,
        provider_launch_receipt_sha256=LAUNCH_RECEIPT,
        deadline=DeadlineClock.from_authenticated_receipt(launch),
        verify_live_watchdog=lambda observed: observed == payload,
    )


def _host_authorization(
    tmp_path: Path,
    destination: Path,
    *,
    inside_provider_resource: bool = False,
):
    hard_stop = _hard_stop(tmp_path)
    response = tmp_path / "host-finalizer.response"
    response.write_bytes(b"authenticated owner-host process response")
    path = tmp_path / "host-finalizer.json"
    payload = {
        "status": "ARMED",
        "execution_domain": "owner_host_outside_provider_resource",
        "provider": "runpod",
        "resource_id": "pod-task-owned",
        "panel_sha256": PANEL,
        "panel_commitment_sha256": PANEL_COMMITMENT,
        "split_receipt_sha256": SPLIT,
        "package_closure_sha256": PACKAGE,
        "task_scope_sha256": SCOPE,
        "resource_request_sha256": REQUEST,
        "resource_manifest_sha256": MANIFEST_DIGEST,
        "provider_launch_receipt_sha256": LAUNCH_RECEIPT,
        "launch_response_sha256": LAUNCH_RESPONSE,
        "quote_sha256": QUOTE,
        "hard_stop_receipt_sha256": hard_stop.receipt_sha256,
        "owner_paid_authorization_sha256": OWNER,
        "terminal_attempt_sha256": TERMINAL_ATTEMPT,
        "implementation_revision": REVISION,
        "source_lock_sha256": SOURCE,
        "runtime_lock_sha256": RUNTIME,
        "host_source_closure_sha256": "1" * 64,
        "host_environment_sha256": "2" * 64,
        "host_process_identity": {
            "pid": 100,
            "parent_pid": 10,
            "start_timestamp": "host-process-100",
            "executable_sha256": "3" * 64,
            "command_line_sha256": "4" * 64,
        },
        "host_started_utc": "2026-08-31T23:59:00Z",
        "armed_utc": "2026-09-01T00:01:02Z",
        "delete_by_utc": "2026-09-01T07:00:00Z",
        "evidence_destination_sha256": evidence_destination_sha256(destination),
        "authenticated_control_response_sha256": sha256_file(response),
        "provider_credentials_present": True,
        "inside_provider_resource": inside_provider_resource,
    }
    write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="host_finalizer",
        payload=payload,
    )
    return authenticate_host_finalizer(
        path,
        authenticated_control_response_path=response,
        destination_root=destination,
        launch=_launch(),
        provider_launch_receipt_sha256=LAUNCH_RECEIPT,
        hard_stop=hard_stop,
        owner_paid_authorization_sha256=OWNER,
        panel_sha256=PANEL,
        panel_commitment_sha256=PANEL_COMMITMENT,
        split_receipt_sha256=SPLIT,
        package_closure_sha256=PACKAGE,
        terminal_attempt_sha256=TERMINAL_ATTEMPT,
        implementation_revision=REVISION,
        source_lock_sha256=SOURCE,
        runtime_lock_sha256=RUNTIME,
        expected_host_source_closure_sha256="1" * 64,
        expected_host_environment_sha256="2" * 64,
        verify_live_host=lambda observed: observed == payload,
    )


def _evacuated(
    tmp_path: Path,
    destination: Path,
    events: list[str],
    *,
    host_finalizer_receipt_sha256: str,
):
    source = tmp_path / "pod-attempt"
    source.mkdir()
    sealed = source / "sealed"
    sealed.mkdir()
    terminal = sealed / "study-outcome.json"
    write_receipt(
        terminal,
        study_id=STUDY_ID,
        receipt_type="terminal_outcome",
        payload={
            "status": "not_evaluable",
            "action": "retain_round1_control_attempt_not_evaluable",
            "revision": REVISION,
            "panel_sha256": PANEL,
            "source_lock_sha256": SOURCE,
            "runtime_lock_sha256": RUNTIME,
            "terminal_attempt_sha256": TERMINAL_ATTEMPT,
            "failed_phase": "provider_preflight",
            "error_code": "provider_preflight:RuntimeError",
            "selection_receipt_sha256": None,
            "stage2_verification_sha256": None,
            "stage2_outcome_opened": False,
            "organizer_score_comparable": False,
            "raw_output_included": False,
        },
    )
    terminal_authorization = authenticate_terminal_outcome(
        terminal,
        evidence_root=source,
        expected_revision=REVISION,
        expected_panel_sha256=PANEL,
        expected_panel_commitment_sha256=PANEL_COMMITMENT,
        expected_split_receipt_sha256=SPLIT,
        expected_package_closure_sha256=PACKAGE,
        expected_source_lock_sha256=SOURCE,
        expected_runtime_lock_sha256=RUNTIME,
        expected_terminal_attempt_sha256=TERMINAL_ATTEMPT,
    )
    process_path = sealed / "process-cleanup.json"
    process_cleanup = cleanup_process_tree_once(
        process_path,
        adapter=_Processes(events),
        root_pid=20,
    )
    events.append("evacuate")
    return evacuate_and_authenticate(
        source_attempt_root=source,
        destination_root=destination,
        source_terminal_path=terminal,
        process_cleanup=process_cleanup,
        process_cleanup_receipt_path=process_path,
        terminal_authorization=terminal_authorization,
        pod_error_code=None,
        binding=EvidenceHandoffBinding(
            resource_id="pod-task-owned",
            panel_sha256=PANEL,
            panel_commitment_sha256=PANEL_COMMITMENT,
            split_receipt_sha256=SPLIT,
            package_closure_sha256=PACKAGE,
            provider_launch_receipt_sha256=LAUNCH_RECEIPT,
            host_finalizer_receipt_sha256=(
                host_finalizer_receipt_sha256
            ),
            terminal_attempt_sha256=TERMINAL_ATTEMPT,
            implementation_revision=REVISION,
            source_lock_sha256=SOURCE,
            runtime_lock_sha256=RUNTIME,
        ),
    )


def test_host_finalizer_survives_provider_deletion_and_seals_terminal_chain(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    destination = tmp_path / "host-evidence"
    host = _host_authorization(tmp_path, destination)
    evidence = _evacuated(
        tmp_path,
        destination,
        events,
        host_finalizer_receipt_sha256=host.receipt_sha256,
    )
    result = finalize_host_attempt(
        authorization=host,
        evidence_evacuation_receipt_sha256=evidence.receipt_sha256,
        resource_adapter=_HostResources(events),
        resource_manifest=_manifest(),
        billing_binding=_billing_binding(),
        now_utc=lambda: datetime(2026, 9, 1, 6, 50, tzinfo=UTC),
    )
    assert events == [
        "terminate:21",
        "terminate:20",
        "evacuate",
        "delete",
        "inventory",
        "billing",
    ]
    assert result.final_action == "retain_round1_control_attempt_not_evaluable"
    final_payload, _ = read_receipt(
        destination / "post-cleanup/final-attempt.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="attempt_finalization",
    )
    assert final_payload["status"] == "not_evaluable"
    assert final_payload["completion_authority"] == "terminal_evidence_seal_only"
    assert (destination / "terminal-evidence-index.json").is_file()
    assert result.terminal_evidence_index_sha256 == sha256_file(
        destination / "terminal-evidence-index.json"
    )
    terminal_index, _ = read_receipt(
        destination / "terminal-evidence-index.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="terminal_evidence_index",
    )
    logical_ids = {row["logical_id"] for row in terminal_index["member_rows"]}
    assert "pod-handoff/evidence-evacuation.json" in logical_ids
    assert "pod-handoff/evidence-evacuation.json.sha256" in logical_ids
    assert result.terminal_evidence_seal_sha256 == sha256_file(
        destination / "terminal-evidence-seal.json"
    )
    terminal_seal, _ = read_receipt(
        destination / "terminal-evidence-seal.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="terminal_evidence_seal",
    )
    assert terminal_seal["completion_authority"] == (
        "terminal_evidence_seal_only"
    )


def test_nonzero_inventory_is_terminally_not_evaluable(tmp_path: Path) -> None:
    events: list[str] = []
    destination = tmp_path / "host-evidence"
    host = _host_authorization(tmp_path, destination)
    evidence = _evacuated(
        tmp_path,
        destination,
        events,
        host_finalizer_receipt_sha256=host.receipt_sha256,
    )
    result = finalize_host_attempt(
        authorization=host,
        evidence_evacuation_receipt_sha256=evidence.receipt_sha256,
        resource_adapter=_HostResources(events, nonzero_inventory=True),
        resource_manifest=_manifest(),
        billing_binding=_billing_binding(),
        now_utc=lambda: datetime(2026, 9, 1, 6, 50, tzinfo=UTC),
    )
    assert result.final_action == "retain_round1_control_attempt_not_evaluable"
    cleanup, _ = read_receipt(
        destination / "post-cleanup/cleanup-result.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="cleanup_result",
    )
    assert cleanup["status"] == "attempt_not_evaluable"
    assert "resource_inventory:HostFinalizerError" in cleanup["cleanup_errors"]


def test_host_can_reauthenticate_remote_evacuation_without_pod_token(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "host-evidence"
    token = _evacuated(
        tmp_path,
        destination,
        [],
        host_finalizer_receipt_sha256="8" * 64,
    )
    reconstructed = authenticate_evidence_evacuation(
        destination / "pod-handoff/evidence-evacuation.json",
        destination_root=destination,
        expected_receipt_sha256=token.receipt_sha256,
        expected_binding=token.binding,
    )
    assert reconstructed.receipt_sha256 == token.receipt_sha256
    assert reconstructed.process_cleanup_receipt_sha256 == (
        token.process_cleanup_receipt_sha256
    )


def test_missing_pod_evidence_still_deletes_but_cannot_complete(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    destination = tmp_path / "host-evidence"
    result = finalize_host_attempt(
        authorization=_host_authorization(tmp_path, destination),
        evidence_evacuation_receipt_sha256=None,
        resource_adapter=_HostResources(events),
        resource_manifest=_manifest(),
        billing_binding=_billing_binding(),
        now_utc=lambda: datetime(2026, 9, 1, 6, 50, tzinfo=UTC),
    )
    assert events == ["delete", "inventory", "billing"]
    assert result.final_action == "retain_round1_control_attempt_not_evaluable"
    assert result.study_outcome_sha256 is None
    assert (destination / "terminal-evidence-index.json").is_file()


def test_host_supervisor_finalizes_when_pod_setup_raises(tmp_path: Path) -> None:
    events: list[str] = []
    destination = tmp_path / "host-evidence"

    def fail_before_handoff():
        raise RuntimeError("sealed setup failed")

    supervised = supervise_paid_attempt(
        invoke_pod=fail_before_handoff,
        authorization=_host_authorization(tmp_path, destination),
        resource_adapter=_HostResources(events),
        resource_manifest=_manifest(),
        billing_binding=_billing_binding(),
        now_utc=lambda: datetime(2026, 9, 1, 6, 50, tzinfo=UTC),
    )
    assert supervised.pod.error_code == "host_observed_pod_failure:RuntimeError"
    assert events == ["delete", "inventory", "billing"]
    assert supervised.host.final_action == (
        "retain_round1_control_attempt_not_evaluable"
    )


def test_in_pod_finalizer_identity_is_rejected_before_deletion(
    tmp_path: Path,
) -> None:
    with pytest.raises(HostFinalizerError, match="not bound"):
        _host_authorization(
            tmp_path,
            tmp_path / "host-evidence",
            inside_provider_resource=True,
        )


def test_provider_call_crossing_hard_horizon_cannot_complete(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    destination = tmp_path / "host-evidence"
    host = _host_authorization(tmp_path, destination)
    evidence = _evacuated(
        tmp_path,
        destination,
        events,
        host_finalizer_receipt_sha256=host.receipt_sha256,
    )
    resources = _HostResources(events)
    times = iter(
        [
            datetime(2026, 9, 1, 6, 59, 58, tzinfo=UTC),
            datetime(2026, 9, 1, 7, 0, 0, tzinfo=UTC),
            datetime(2026, 9, 1, 7, 0, 1, tzinfo=UTC),
            datetime(2026, 9, 1, 7, 0, 2, tzinfo=UTC),
        ]
    )
    result = finalize_host_attempt(
        authorization=host,
        evidence_evacuation_receipt_sha256=evidence.receipt_sha256,
        resource_adapter=resources,
        resource_manifest=_manifest(),
        billing_binding=_billing_binding(),
        now_utc=lambda: next(times),
    )
    assert events[-1] == "delete"
    assert resources.deadlines == ["2026-09-01T07:00:00Z"]
    assert result.final_action == "retain_round1_control_attempt_not_evaluable"
    final_payload, _ = read_receipt(
        destination / "post-cleanup/final-attempt.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="attempt_finalization",
    )
    assert final_payload["status"] == "not_evaluable"
