from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen import preflight as preflight_module
from experiments.feasibility_debt_candidate_screen.authorization import (
    PAID_ATTEMPT_APPROVAL_TEXT,
    PAID_ATTEMPT_APPROVAL_TEXT_SHA256,
    authenticate_paid_attempt_authorization,
)
from experiments.feasibility_debt_candidate_screen.canonical import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    PLAN_REVISION,
    STUDY_ID,
)
from experiments.feasibility_debt_candidate_screen.hard_stop import (
    HardStopError,
    authenticate_hard_stop,
)
from experiments.feasibility_debt_candidate_screen.orchestrator import (
    LockDigests,
    claim_terminal_attempt_once,
)
from experiments.feasibility_debt_candidate_screen.preflight import (
    PreflightAuthorization,
)
from experiments.feasibility_debt_candidate_screen.provider import (
    ProviderGuardError,
    authenticate_clean_inventory,
    authenticate_provider_quote,
    authenticate_resource_request,
    authorize_single_provision,
    validate_provisioned_resource,
)
from experiments.feasibility_debt_candidate_screen.runtime import (
    DeadlineClock,
    load_provider_launch_receipt,
)
from experiments.feasibility_debt_candidate_screen.runpod_adapter import (
    ApiExchange,
    BearerHttpsTransport,
    RunpodAdapterError,
    RunpodControlPlane,
    RunpodResourceAdapter,
)
from experiments.feasibility_debt_candidate_screen.cleanup import (
    BillingBinding,
    validate_billing_receipt,
)
from experiments.feasibility_debt_candidate_screen.canonical import read_receipt


REVISION = "a" * 40
PANEL = "1" * 64
COMMITMENT = "2" * 64
SOURCE = "3" * 64
RUNTIME = "4" * 64
CI = "5" * 64
SCOPE = "6" * 64
IMAGE = "sha256:" + "7" * 64


def _billing() -> dict[str, object]:
    return {
        "currency": "USD",
        "gpu_hourly_rate_usd": "3.29",
        "combined_hourly_rate_usd": "3.29",
        "fixed_charge_usd": "0",
        "metering_quantum_seconds": 1,
        "round_up_each_quantum": False,
    }


def _preflight() -> PreflightAuthorization:
    return PreflightAuthorization(
        revision=REVISION,
        panel_sha256=PANEL,
        split_receipt_sha256="8" * 64,
        ci_evidence_sha256=CI,
        locks=LockDigests(
            source_lock_sha256=SOURCE,
            runtime_lock_sha256=RUNTIME,
            revision=REVISION,
            package_closure_sha256="9" * 64,
            panel_commitment_sha256=COMMITMENT,
        ),
        _sentinel=preflight_module._SENTINEL,
    )


def _provider_gate(tmp_path: Path):
    terminal_attempt = claim_terminal_attempt_once(
        tmp_path / "terminal-control",
        attempt_root=tmp_path / "attempt-root",
        revision=REVISION,
        panel_sha256=PANEL,
        source_lock_sha256=SOURCE,
    )
    quote_response = tmp_path / "quote-response.bin"
    quote_response.write_bytes(b"authenticated quote response")
    quote_path = tmp_path / "provider-quote.json"
    quote_digest = write_receipt(
        quote_path,
        study_id=STUDY_ID,
        receipt_type="provider_quote",
        payload={
            "provider": "runpod",
            "observed_utc": "2026-09-01T00:00:00Z",
            "authenticated_response_sha256": sha256_file(quote_response),
            "cloud_type": "SECURE",
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "capacity_available": True,
            "max_ephemeral_disk_gib": 40,
            "billing": _billing(),
        },
    )
    quote = authenticate_provider_quote(
        quote_path, authenticated_response_path=quote_response
    )
    assert quote.receipt_sha256 == quote_digest

    request_path = tmp_path / "resource-request.json"
    request_digest = write_receipt(
        request_path,
        study_id=STUDY_ID,
        receipt_type="resource_request",
        payload={
            "provider": "runpod",
            "implementation_revision": REVISION,
            "panel_sha256": PANEL,
            "panel_commitment_sha256": COMMITMENT,
            "source_lock_sha256": SOURCE,
            "runtime_lock_sha256": RUNTIME,
            "ci_evidence_sha256": CI,
            "terminal_attempt_sha256": terminal_attempt.receipt_sha256,
            "quote_sha256": quote_digest,
            "task_scope_sha256": SCOPE,
            "cloud_type": "SECURE",
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "ephemeral_disk_gib": 40,
            "network_volume_count": 0,
            "endpoint_count": 0,
            "template_count": 0,
            "immutable_image_digest": IMAGE,
            "max_provider_seconds": 25_200,
            "max_gpu_hourly_rate_usd": "3.29",
            "max_combined_hourly_rate_usd": "3.5714285714",
            "max_gpu_charge_usd": "23.03",
            "max_all_in_charge_usd": "25.00",
            "one_pod_only": True,
        },
    )
    request = authenticate_resource_request(request_path)
    assert request.receipt_sha256 == request_digest

    paid_path = tmp_path / "paid-authorization.json"
    write_receipt(
        paid_path,
        study_id=STUDY_ID,
        receipt_type="owner_paid_attempt_authorization",
        payload={
            "approval_text": PAID_ATTEMPT_APPROVAL_TEXT,
            "approval_text_sha256": PAID_ATTEMPT_APPROVAL_TEXT_SHA256,
            "study_id": STUDY_ID,
            "plan_revision": PLAN_REVISION,
            "implementation_revision": REVISION,
            "panel_sha256": PANEL,
            "panel_commitment_sha256": COMMITMENT,
            "source_lock_sha256": SOURCE,
            "runtime_lock_sha256": RUNTIME,
            "ci_evidence_sha256": CI,
            "quote_sha256": quote_digest,
            "resource_request_sha256": request_digest,
            "terminal_attempt_sha256": terminal_attempt.receipt_sha256,
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
        },
    )
    paid = authenticate_paid_attempt_authorization(paid_path)

    inventory_response = tmp_path / "inventory-response.bin"
    inventory_response.write_bytes(b"authenticated clean inventory")
    inventory_path = tmp_path / "inventory.json"
    write_receipt(
        inventory_path,
        study_id=STUDY_ID,
        receipt_type="provider_inventory",
        payload={
            "provider": "runpod",
            "observed_utc": "2026-09-01T00:02:00Z",
            "authenticated_response_sha256": sha256_file(inventory_response),
            "task_scope_sha256": SCOPE,
            "pods": [],
            "clusters": [],
            "network_volumes": [],
            "endpoints": [],
            "templates": [],
            "registries": [],
        },
    )
    inventory = authenticate_clean_inventory(
        inventory_path,
        authenticated_response_path=inventory_response,
        expected_task_scope_sha256=SCOPE,
    )
    provision = authorize_single_provision(
        preflight=_preflight(),
        paid_authorization=paid,
        request=request,
        quote=quote,
        inventory=inventory,
        terminal_attempt=terminal_attempt,
        now_utc=datetime(2026, 9, 1, 0, 4, tzinfo=UTC),
    )
    return request, quote, provision, terminal_attempt


def test_preprovision_authorization_binds_owner_quote_request_and_clean_inventory(
    tmp_path: Path,
) -> None:
    request, quote, provision, terminal_attempt = _provider_gate(tmp_path)
    assert provision.resource_request_sha256 == request.receipt_sha256
    assert provision.quote_sha256 == quote.receipt_sha256

    # A fresh immutable receipt is still rejected when the evidence has aged
    # beyond the exact pre-provision freshness boundary.
    with pytest.raises(ProviderGuardError, match="stale"):
        authorize_single_provision(
            preflight=_preflight(),
            paid_authorization=authenticate_paid_attempt_authorization(
                tmp_path / "paid-authorization.json"
            ),
            request=request,
            quote=quote,
            inventory=authenticate_clean_inventory(
                tmp_path / "inventory.json",
                authenticated_response_path=tmp_path / "inventory-response.bin",
                expected_task_scope_sha256=SCOPE,
            ),
            terminal_attempt=terminal_attempt,
            now_utc=datetime(2026, 9, 1, 0, 6, tzinfo=UTC),
        )


def test_launch_resource_and_independent_hard_stop_are_exactly_bound(
    tmp_path: Path,
) -> None:
    request, quote, provision, _terminal_attempt = _provider_gate(tmp_path)
    manifest = {
        "provider": "runpod",
        "cloud_type": "SECURE",
        "pod": {
            "id": "pod-one",
            "gpu_type_id": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "immutable_image_digest": IMAGE,
        },
        "ephemeral_disk_gib": 40,
        "other_resources": [],
        "task_scope_sha256": SCOPE,
        "resource_request_sha256": request.receipt_sha256,
    }
    manifest_digest = sha256_bytes(canonical_json_bytes(manifest))
    launch_response = tmp_path / "launch-response.bin"
    launch_response.write_bytes(b"authenticated launch response")
    launch_path = tmp_path / "provider-launch.json"
    launch_digest = write_receipt(
        launch_path,
        study_id=STUDY_ID,
        receipt_type="provider_launch",
        payload={
            "provider": "runpod",
            "quote_sha256": quote.receipt_sha256,
            "authenticated_response_sha256": sha256_file(launch_response),
            "resource_request_sha256": request.receipt_sha256,
            "resource_manifest_sha256": manifest_digest,
            "task_scope_sha256": SCOPE,
            "resource_id": "pod-one",
            "immutable_image_digest": IMAGE,
            "status": "RUNNING",
            "create_utc": "2026-09-01T00:00:00Z",
            "running_utc": "2026-09-01T00:01:00Z",
            "billable_utc": "2026-09-01T00:00:00Z",
            "cloud_type": "SECURE",
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "ephemeral_disk_gib": 40,
            "provider_running_hourly_cost_usd": "3.29",
            "billing": _billing(),
        },
    )
    launch, observed_launch_digest = load_provider_launch_receipt(
        launch_path,
        expected_resource_manifest_sha256=manifest_digest,
        expected_resource_request_sha256=request.receipt_sha256,
        expected_quote_sha256=quote.receipt_sha256,
        authenticated_response_path=launch_response,
    )
    assert observed_launch_digest == launch_digest
    validate_provisioned_resource(
        manifest=manifest,
        launch=launch,
        request=request,
        quote=quote,
        provision=provision,
    )

    clock = DeadlineClock.from_authenticated_receipt(launch)
    control_response = tmp_path / "hard-stop-response.bin"
    control_response.write_bytes(b"authenticated watchdog arm response")
    hard_stop_path = tmp_path / "hard-stop.json"
    payload = {
        "status": "ARMED",
        "provider": "runpod",
        "resource_id": "pod-one",
        "task_scope_sha256": SCOPE,
        "resource_request_sha256": request.receipt_sha256,
        "resource_manifest_sha256": manifest_digest,
        "provider_launch_receipt_sha256": launch_digest,
        "quote_sha256": quote.receipt_sha256,
        "control_kind": "independent_watchdog_process",
        "watchdog_source_sha256": "a" * 64,
        "watchdog_process_identity_sha256": "b" * 64,
        "watchdog_command_sha256": "c" * 64,
        "watchdog_started_utc": "2026-09-01T00:01:01Z",
        "watchdog_trigger_utc": "2026-09-01T06:59:00Z",
        "delete_by_utc": "2026-09-01T07:00:00Z",
        "credential_scope": "delete_task_pod_only",
        "authenticated_control_response_sha256": sha256_file(control_response),
    }
    write_receipt(
        hard_stop_path,
        study_id=STUDY_ID,
        receipt_type="provider_hard_stop",
        payload=payload,
    )
    live = {"value": True}
    authorization = authenticate_hard_stop(
        hard_stop_path,
        authenticated_control_response_path=control_response,
        launch=launch,
        provider_launch_receipt_sha256=launch_digest,
        deadline=clock,
        verify_live_watchdog=lambda observed: live["value"] and observed == payload,
    )
    assert authorization.delete_by_utc == "2026-09-01T07:00:00Z"
    authorization.assert_live()
    live["value"] = False
    with pytest.raises(HardStopError, match="no longer live"):
        authorization.assert_live()
    with pytest.raises(HardStopError, match="not live"):
        authenticate_hard_stop(
            hard_stop_path,
            authenticated_control_response_path=control_response,
            launch=launch,
            provider_launch_receipt_sha256=launch_digest,
            deadline=clock,
            verify_live_watchdog=lambda _observed: False,
        )


class _QueueTransport:
    def __init__(self, rows: list[tuple[str, str, int, bytes, str]]) -> None:
        self.rows = list(rows)
        self.calls: list[tuple[str, str, bytes | None, str]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        deadline_utc: str,
    ) -> ApiExchange:
        self.calls.append((method, path, body, deadline_utc))
        expected_method, expected_path, status, content, observed = self.rows.pop(0)
        assert (method, path) == (expected_method, expected_path)
        return ApiExchange(
            method=method,
            path=path,
            status=status,
            observed_utc=observed,
            content_type="application/json",
            request_id="request-test",
            body=content,
            credential_fingerprint_sha256="f" * 64,
            request_body_sha256=sha256_bytes(body or b""),
        )


def _billing_binding() -> BillingBinding:
    return BillingBinding(
        resource_id="pod-one",
        task_scope_sha256=SCOPE,
        quote_sha256="a" * 64,
        launch_response_sha256="b" * 64,
        gpu_hourly_rate_usd="3.29",
        combined_hourly_rate_usd="3.30",
    )


def test_concrete_runpod_adapter_persists_zero_retry_provider_evidence(
    tmp_path: Path,
) -> None:
    deadline = "2026-09-01T07:00:00Z"
    transport = _QueueTransport(
        [
            (
                "DELETE",
                "/v2/pods/pod-one",
                204,
                b"",
                "2026-09-01T00:09:00Z",
            ),
            (
                "GET",
                "/v2/pods?includeClusterPods=true",
                200,
                b'{"pods":[]}',
                "2026-09-01T00:09:01Z",
            ),
            (
                "GET",
                "/v2/clusters",
                200,
                b'{"clusters":[]}',
                "2026-09-01T00:09:02Z",
            ),
            (
                "GET",
                "/v2/network-volumes",
                200,
                b'{"networkVolumes":[]}',
                "2026-09-01T00:09:03Z",
            ),
            (
                "GET",
                "/v2/serverless",
                200,
                b'{"endpoints":[]}',
                "2026-09-01T00:09:04Z",
            ),
            (
                "GET",
                "/v2/templates",
                200,
                b'{"templates":[]}',
                "2026-09-01T00:09:05Z",
            ),
            (
                "GET",
                "/v2/registries",
                200,
                b'{"registries":[]}',
                "2026-09-01T00:09:06Z",
            ),
            (
                "GET",
                "/v2/billing/pods?startTime=2026-09-01T00%3A00%3A00Z&"
                "endTime=2026-09-01T07%3A00%3A00Z&bucketSize=hour&"
                "podId=pod-one",
                200,
                (
                    b'{"metadata":{"query":{"bucketSize":"hour",'
                    b'"endTime":"2026-09-01T07:00:00Z",'
                    b'"podId":"pod-one",'
                    b'"startTime":"2026-09-01T00:00:00Z"},'
                    b'"recordCount":1,"uniquePodCount":1,'
                    b'"totals":{"cpuAmount":0,"diskAmount":0.01,'
                    b'"gpuAmount":0.5,"totalAmount":0.51}},"records":['
                    b'{"cpuAmount":0,"diskAmount":0.01,"gpuAmount":0.5,'
                    b'"podId":"pod-one","startTime":'
                    b'"2026-09-01T00:00:00Z","endTime":'
                    b'"2026-09-01T01:00:00Z","totalAmount":0.51}]}'
                ),
                "2026-09-01T00:10:00Z",
            ),
        ]
    )
    adapter = RunpodResourceAdapter(
        transport=transport,
        binding=_billing_binding(),
        resource_request_sha256="c" * 64,
        billable_utc="2026-09-01T00:00:00Z",
    )
    adapter.delete(
        "pod",
        "pod-one",
        receipt_path=tmp_path / "delete.json",
        authenticated_response_path=tmp_path / "delete.response",
        deadline_utc=deadline,
    )
    adapter.scoped_inventory(
        receipt_path=tmp_path / "inventory.json",
        authenticated_response_path=tmp_path / "inventory.response",
        deadline_utc=deadline,
    )
    adapter.billing_receipt(
        receipt_path=tmp_path / "billing.json",
        authenticated_response_path=tmp_path / "billing.response",
        deadline_utc=deadline,
    )
    assert len(transport.calls) == 8
    inventory, _ = read_receipt(
        tmp_path / "inventory.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_inventory",
    )
    assert all(
        inventory[name] == []
        for name in (
            "pods",
            "clusters",
            "network_volumes",
            "endpoints",
            "templates",
            "registries",
        )
    )
    billing, _ = read_receipt(
        tmp_path / "billing.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_billing",
    )
    validated = validate_billing_receipt(billing, binding=_billing_binding())
    assert validated["provider_total_charge_usd"] == "0.51"
    assert validated["envelope_seconds"] == 600
    assert validated["provider_record_count"] == 1
    assert b"Bearer" not in (tmp_path / "billing.response").read_bytes()


def test_concrete_adapter_rejects_provider_charge_above_elapsed_envelope(
    tmp_path: Path,
) -> None:
    body = (
        b'{"metadata":{"query":{"bucketSize":"hour",'
        b'"endTime":"2026-09-01T07:00:00Z","podId":"pod-one",'
        b'"startTime":"2026-09-01T00:00:00Z"},'
        b'"recordCount":1,"uniquePodCount":1,"totals":{'
        b'"cpuAmount":0,"diskAmount":0,"gpuAmount":1,"totalAmount":1}},'
        b'"records":[{"cpuAmount":0,"diskAmount":0,"gpuAmount":1,'
        b'"podId":"pod-one","startTime":"2026-09-01T00:00:00Z",'
        b'"endTime":"2026-09-01T01:00:00Z","totalAmount":1}]}'
    )
    adapter = RunpodResourceAdapter(
        transport=_QueueTransport(
            [
                (
                    "GET",
                    "/v2/billing/pods?startTime=2026-09-01T00%3A00%3A00Z&"
                    "endTime=2026-09-01T07%3A00%3A00Z&bucketSize=hour&"
                    "podId=pod-one",
                    200,
                    body,
                    "2026-09-01T00:00:10Z",
                )
            ]
        ),
        binding=_billing_binding(),
        resource_request_sha256="c" * 64,
        billable_utc="2026-09-01T00:00:00Z",
    )
    with pytest.raises(RunpodAdapterError, match="exceeds"):
        adapter.billing_receipt(
            receipt_path=tmp_path / "billing.json",
            authenticated_response_path=tmp_path / "billing.response",
            deadline_utc="2026-09-01T07:00:00Z",
        )
    assert (tmp_path / "billing.response").is_file()
    assert not (tmp_path / "billing.json").exists()


def test_bearer_transport_never_repr_exposes_injected_token() -> None:
    transport = BearerHttpsTransport(bearer_token="secret-provider-token")
    assert "secret-provider-token" not in repr(transport)
    with pytest.raises(AttributeError):
        transport.bearer_token = "replacement-token"


def test_control_plane_captures_exact_h100_quote_without_live_access(
    tmp_path: Path,
) -> None:
    path = (
        "/v2/catalog/gpus/NVIDIA%20H100%2080GB%20HBM3?"
        "include=AVAILABILITY&product=POD&count=1&cloud=SECURE&"
        "cudaVersions=13.0"
    )
    response = (
        b'{"availability":"LOW","dataCenters":[{"id":"US-TEST"}],'
        b'"cudaVersions":[{"available":true,"version":"13.0"}],'
        b'"id":"NVIDIA H100 80GB HBM3","manufacturer":"NVIDIA",'
        b'"maxCount":{"community":0,"secure":1},"memory":80,'
        b'"price":{"community":0,"secure":3.29},"secure":true}'
    )
    quote = RunpodControlPlane(
        _QueueTransport(
            [("GET", path, 200, response, "2026-09-01T00:00:00Z")]
        )
    ).capture_quote(
        ephemeral_disk_gib=40,
        receipt_path=tmp_path / "quote.json",
        authenticated_response_path=tmp_path / "quote.response",
        deadline_utc="2026-09-01T00:01:00Z",
    )
    assert quote.billing.gpu_hourly_rate_usd == Decimal("3.29")
    assert quote.billing.combined_hourly_rate_usd > Decimal("3.29")
    assert quote.billing.combined_hourly_rate_usd < Decimal("3.30")


def test_inventory_failure_preserves_each_completed_provider_exchange(
    tmp_path: Path,
) -> None:
    transport = _QueueTransport(
        [
            (
                "GET",
                "/v2/pods?includeClusterPods=true",
                200,
                b'{"pods":[]}',
                "2026-09-01T00:00:00Z",
            ),
            (
                "GET",
                "/v2/clusters",
                200,
                b'{"clusters":[]}',
                "2026-09-01T00:00:01Z",
            ),
            (
                "GET",
                "/v2/network-volumes",
                500,
                b'{"detail":"test failure"}',
                "2026-09-01T00:00:02Z",
            ),
        ]
    )
    response = tmp_path / "inventory.response"
    with pytest.raises(RunpodAdapterError, match="HTTP 500"):
        RunpodControlPlane(transport).capture_clean_inventory(
            task_scope_sha256=SCOPE,
            receipt_path=tmp_path / "inventory.json",
            authenticated_response_path=response,
            deadline_utc="2026-09-01T00:01:00Z",
        )
    assert len(transport.calls) == 3
    assert not response.exists()
    assert not (tmp_path / "inventory.json").exists()
    assert all(
        response.with_name(f"{response.name}.part-{index:02d}").is_file()
        for index in range(3)
    )


def test_control_plane_creates_once_then_authenticates_exact_running_pod(
    tmp_path: Path,
) -> None:
    request, quote_authorization, provision, _terminal_attempt = _provider_gate(
        tmp_path
    )
    image = f"example.invalid/l2d@{IMAGE}"
    name = f"l2d-fd-v1-{SCOPE[:16]}"
    created_body = canonical_json_bytes(
        {
            "cloud": "SECURE",
            "createdAt": "2026-09-01T00:04:01Z",
            "disk": 40,
            "id": "pod-one",
            "image": image,
            "name": name,
            "status": "PROVISIONING",
        }
    )
    running_body = canonical_json_bytes(
        {
            "cloud": "SECURE",
            "cost": 3.20,
            "createdAt": "2026-09-01T00:04:01Z",
            "cudaVersion": "13.0",
            "disk": 40,
            "globalNetworking": {"enabled": False},
            "gpu": {"count": 1, "id": "NVIDIA H100 80GB HBM3"},
            "id": "pod-one",
            "image": image,
            "mounts": {},
            "name": name,
            "ports": ["22/tcp"],
            "registry": None,
            "startedAt": "2026-09-01T00:04:30Z",
            "status": "RUNNING",
            "template": None,
        }
    )
    transport = _QueueTransport(
        [
            (
                "POST",
                "/v2/pods",
                201,
                created_body,
                "2026-09-01T00:04:01Z",
            ),
            (
                "GET",
                "/v2/pods/pod-one",
                200,
                running_body,
                "2026-09-01T00:04:30Z",
            ),
        ]
    )
    control = RunpodControlPlane(transport)
    created = control.create_pod_once(
        request=request,
        quote_authorization=quote_authorization,
        provision_authorization=provision,
        immutable_image_reference=image,
        ssh_public_key="ssh-ed25519 AAAATEST owner@test",
        create_intent_path=tmp_path / "create-intent.json",
        create_receipt_path=tmp_path / "create.json",
        authenticated_response_path=tmp_path / "create.response",
        manifest_path=tmp_path / "resource-manifest.json",
        deadline_utc="2026-09-01T00:10:00Z",
    )
    launch, launch_digest = control.await_running_and_seal_launch(
        created=created,
        request=request,
        quote_authorization=quote_authorization,
        provision_authorization=provision,
        launch_receipt_path=tmp_path / "launch.json",
        authenticated_response_path=tmp_path / "launch.response",
        deadline_utc="2026-09-01T00:10:00Z",
        sleep=lambda _seconds: None,
    )
    assert launch.resource_id == "pod-one"
    assert launch.provider_running_hourly_cost_usd == "3.2"
    assert created.create_intent_sha256 == sha256_file(
        tmp_path / "create-intent.json"
    )
    assert created.create_receipt_sha256 == sha256_file(tmp_path / "create.json")
    assert launch_digest == sha256_file(tmp_path / "launch.json")
    transcript = (tmp_path / "launch.response").read_bytes()
    assert b"ssh-ed25519" not in transcript
    assert len(transport.calls) == 2


def test_create_intent_makes_ambiguous_post_nonrepeatable(tmp_path: Path) -> None:
    request, quote_authorization, provision, _terminal_attempt = _provider_gate(
        tmp_path
    )
    transport = _QueueTransport(
        [
            (
                "POST",
                "/v2/pods",
                201,
                b'{"createdAt":"2026-09-01T00:04:01Z"}',
                "2026-09-01T00:04:01Z",
            )
        ]
    )
    control = RunpodControlPlane(transport)
    arguments = {
        "request": request,
        "quote_authorization": quote_authorization,
        "provision_authorization": provision,
        "immutable_image_reference": f"example.invalid/l2d@{IMAGE}",
        "ssh_public_key": "ssh-ed25519 AAAATEST owner@test",
        "create_intent_path": tmp_path / "create-intent.json",
        "create_receipt_path": tmp_path / "create.json",
        "authenticated_response_path": tmp_path / "create.response",
        "manifest_path": tmp_path / "resource-manifest.json",
        "deadline_utc": "2026-09-01T00:10:00Z",
    }
    with pytest.raises(RunpodAdapterError, match="changed pod identity"):
        control.create_pod_once(**arguments)
    assert (tmp_path / "create-intent.json").is_file()
    assert (tmp_path / "create.response").is_file()
    assert not (tmp_path / "create.json").exists()
    assert not (tmp_path / "resource-manifest.json").exists()
    with pytest.raises(Exception, match="already exists"):
        control.create_pod_once(**arguments)
    assert len(transport.calls) == 1
