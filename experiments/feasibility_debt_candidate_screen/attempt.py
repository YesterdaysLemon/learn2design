"""Single-use end-to-end state machine for the frozen paid screen attempt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .canonical import read_receipt, sha256_bytes
from .authorization import PaidAttemptAuthorization, validate_paid_bindings
from .cleanup import (
    EvidenceHandoffBinding,
    EvidenceEvacuationAuthorization,
    ProcessAdapter,
    cleanup_process_tree_once,
)
from .contract import STUDY_ID
from .evacuation import evacuate_and_authenticate
from .orchestrator import (
    WorkerCapture,
    TerminalAttemptAuthorization,
    authenticate_terminal_outcome,
    assert_terminal_attempt,
    assert_prepared_attempt_root,
    build_stage2_configs,
    execute_stage_once,
    invoke_cold_smoke,
    invoke_worker,
    seal_and_evaluate_stage2,
    seal_and_select_stage1,
    seal_panel_bundle_once,
    write_not_evaluable_outcome_once,
    write_stage1_failed_outcome_once,
    write_stage2_outcome_once,
)
from .preflight import PreflightAuthorization, authorized_stage1_configs
from .provider import (
    ProviderQuoteAuthorization,
    ProvisionAuthorization,
    ResourceRequestAuthorization,
    assert_provision_authorization,
    validate_provisioned_resource,
)
from .hard_stop import HardStopAuthorization, assert_hard_stop
from .host_finalizer import HOST_FINALIZER_KEYS
from .runtime import (
    DeadlineClock,
    DeadlineController,
    Phase,
    load_provider_launch_receipt,
)


class AttemptError(RuntimeError):
    pass


@dataclass(frozen=True)
class PodAttemptInputs:
    attempt_root: Path
    panel_bundle_dir: Path
    provider_launch_receipt_path: Path
    provider_launch_authenticated_response_path: Path
    resource_manifest: dict[str, object]
    resource_request: ResourceRequestAuthorization
    provider_quote: ProviderQuoteAuthorization
    provision_authorization: ProvisionAuthorization
    hard_stop_authorization: HardStopAuthorization
    host_finalizer_receipt_path: Path
    preflight: PreflightAuthorization
    paid_authorization: PaidAttemptAuthorization
    terminal_attempt: TerminalAttemptAuthorization
    repository_root: Path
    round1_archive: Path
    round1_manifest: Path
    runtime_lock_path: Path
    source_lock_path: Path
    worker_environment: dict[str, str]
    process_root_pid: int
    evidence_destination_root: Path


@dataclass(frozen=True)
class PodAttemptResult:
    study_outcome_sha256: str | None
    evidence_evacuation_receipt_sha256: str | None
    provider_launch_receipt_sha256: str | None
    error_code: str | None


def _resource_manifest_sha256(value: dict[str, object]) -> str:
    from .canonical import canonical_json_bytes

    return sha256_bytes(canonical_json_bytes(value))


def _transition_to_cleanup(controller: DeadlineController) -> None:
    if controller.phase is Phase.CLEANUP:
        return
    if controller.phase is Phase.TERMINAL:
        raise AttemptError("attempt reached terminal state before cleanup")
    controller.transition(Phase.CLEANUP)


def _read_study_action(path: Path) -> tuple[str, str]:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="terminal_outcome",
    )
    action = payload.get("action")
    if not isinstance(action, str):
        raise AttemptError("study outcome action is absent")
    return action, digest


def _run_pod_attempt_impl(
    inputs: PodAttemptInputs,
    *,
    process_adapter: ProcessAdapter,
    now_utc: Callable[[], datetime],
) -> PodAttemptResult:
    """Run study work, terminate its worker tree, and evacuate evidence.

    Provider deletion is intentionally impossible here.  A separately
    authenticated owner-host finalizer must consume the evacuation capability.
    """
    root = inputs.attempt_root
    terminal_attempt = assert_terminal_attempt(inputs.terminal_attempt)
    root = assert_prepared_attempt_root(terminal_attempt, root)
    sealed = root / "sealed"
    sealed.mkdir(exist_ok=False)
    panel_path = inputs.panel_bundle_dir / "panel.json"
    commitment_path = inputs.panel_bundle_dir / "panel-commitment.json"
    split_path = inputs.panel_bundle_dir / "split-receipt.json"
    study_outcome_path = sealed / "study-outcome.json"
    if (
        terminal_attempt.revision != inputs.preflight.revision
        or terminal_attempt.panel_sha256 != inputs.preflight.panel_sha256
        or terminal_attempt.source_lock_sha256
        != inputs.preflight.locks.source_lock_sha256
    ):
        raise AttemptError("paid attempt lacks its pre-provision terminal claim")
    terminal_claim_sha256 = terminal_attempt.receipt_sha256
    controller: DeadlineController | None = None
    error_code: str | None = None
    evidence_authorization: EvidenceEvacuationAuthorization | None = None
    launch_receipt_sha256: str | None = None
    host_finalizer_receipt_sha256: str | None = None
    launch_resource_id: str | None = None
    try:
        seal_panel_bundle_once(
            panel_path=panel_path,
            panel_commitment_path=commitment_path,
            split_receipt_path=split_path,
            sealed_dir=sealed,
            locks=inputs.preflight.locks,
        )
        provision = assert_provision_authorization(inputs.provision_authorization)
        hard_stop = assert_hard_stop(inputs.hard_stop_authorization)
        resource_manifest_sha256 = _resource_manifest_sha256(
            inputs.resource_manifest
        )
        launch, launch_receipt_sha256 = load_provider_launch_receipt(
            inputs.provider_launch_receipt_path,
            expected_resource_manifest_sha256=resource_manifest_sha256,
            expected_resource_request_sha256=inputs.resource_request.receipt_sha256,
            expected_quote_sha256=inputs.provider_quote.receipt_sha256,
            authenticated_response_path=inputs.provider_launch_authenticated_response_path,
        )
        validate_provisioned_resource(
            manifest=inputs.resource_manifest,
            launch=launch,
            request=inputs.resource_request,
            quote=inputs.provider_quote,
            provision=provision,
        )
        host_finalizer_payload, host_finalizer_receipt_sha256 = read_receipt(
            inputs.host_finalizer_receipt_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="host_finalizer",
            expected_payload_keys=HOST_FINALIZER_KEYS,
        )
        host_finalizer_exact = {
            "status": "ARMED",
            "execution_domain": "owner_host_outside_provider_resource",
            "provider": "runpod",
            "resource_id": launch.resource_id,
            "panel_sha256": inputs.preflight.panel_sha256,
            "panel_commitment_sha256": (
                inputs.preflight.locks.panel_commitment_sha256
            ),
            "split_receipt_sha256": inputs.preflight.split_receipt_sha256,
            "package_closure_sha256": (
                inputs.preflight.locks.package_closure_sha256
            ),
            "task_scope_sha256": launch.task_scope_sha256,
            "resource_request_sha256": launch.resource_request_sha256,
            "resource_manifest_sha256": launch.resource_manifest_sha256,
            "provider_launch_receipt_sha256": launch_receipt_sha256,
            "launch_response_sha256": launch.authenticated_response_sha256,
            "quote_sha256": launch.quote_sha256,
            "hard_stop_receipt_sha256": hard_stop.receipt_sha256,
            "owner_paid_authorization_sha256": (
                inputs.paid_authorization.receipt_sha256
            ),
            "terminal_attempt_sha256": terminal_claim_sha256,
            "implementation_revision": inputs.preflight.revision,
            "source_lock_sha256": inputs.preflight.locks.source_lock_sha256,
            "runtime_lock_sha256": inputs.preflight.locks.runtime_lock_sha256,
            "provider_credentials_present": True,
            "inside_provider_resource": False,
        }
        if any(
            host_finalizer_payload.get(key) != value
            for key, value in host_finalizer_exact.items()
        ):
            raise AttemptError("pod dispatch lacks its external host finalizer")
        launch_resource_id = launch.resource_id
        validate_paid_bindings(
            inputs.paid_authorization,
            implementation_revision=inputs.preflight.revision,
            panel_sha256=inputs.preflight.panel_sha256,
            panel_commitment_sha256=inputs.preflight.locks.panel_commitment_sha256,
            source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
            runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
            ci_evidence_sha256=inputs.preflight.ci_evidence_sha256,
            quote_sha256=launch.quote_sha256,
            resource_request_sha256=inputs.resource_request.receipt_sha256,
            terminal_attempt_sha256=terminal_claim_sha256,
        )
        if (
            hard_stop.provider_launch_receipt_sha256 != launch_receipt_sha256
            or hard_stop.resource_manifest_sha256 != resource_manifest_sha256
            or hard_stop.resource_id != launch.resource_id
            or hard_stop.task_scope_sha256 != launch.task_scope_sha256
        ):
            raise AttemptError("independent hard stop is not bound to this launch")
        clock = DeadlineClock.from_authenticated_receipt(launch)
        controller = DeadlineController(
            clock,
            provider_launch_receipt_sha256=launch_receipt_sha256,
            resource_manifest_sha256=resource_manifest_sha256,
            hard_stop_receipt_sha256=hard_stop.receipt_sha256,
            hard_stop_liveness=hard_stop.assert_live,
        )
        now = now_utc()
        clock.require_success_envelope(now)  # type: ignore[arg-type]
        controller.admit_operation("post_running_preflight", now)  # type: ignore[arg-type]
        controller.transition(Phase.SMOKE)
        from .smoke import build_smoke_config

        smoke_config = build_smoke_config(
            revision=inputs.preflight.revision,
            source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
            runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
            package_closure_sha256=inputs.preflight.locks.package_closure_sha256,
            panel_commitment_path=commitment_path,
            provider_launch_receipt_sha256=launch_receipt_sha256,
            resource_manifest_sha256=controller.resource_manifest_sha256,
            hard_stop_receipt_sha256=controller.hard_stop_receipt_sha256,
            deadline_snapshot=clock.snapshot(),
        )
        smoke_authorization = invoke_cold_smoke(
            smoke_dir=root / "smoke",
            config=smoke_config,
            repository_root=inputs.repository_root,
            round1_archive=inputs.round1_archive,
            round1_manifest=inputs.round1_manifest,
            runtime_lock_path=inputs.runtime_lock_path,
            source_lock_path=inputs.source_lock_path,
            panel_commitment_path=commitment_path,
            revision=inputs.preflight.revision,
            environment=inputs.worker_environment,
            deadline=controller,
            now_utc=now_utc,
        )
        stage1_configs = authorized_stage1_configs(
            inputs.preflight,
            panel_path=panel_path,
            split_receipt_path=split_path,
        )

        def invoke_scored(config_path: Path, history_path: Path) -> WorkerCapture:
            return invoke_worker(
                config_path=config_path,
                history_path=history_path,
                repository_root=inputs.repository_root,
                round1_archive=inputs.round1_archive,
                round1_manifest=inputs.round1_manifest,
                runtime_lock_path=inputs.runtime_lock_path,
                source_lock_path=inputs.source_lock_path,
                revision=inputs.preflight.revision,
                environment=inputs.worker_environment,
            )

        execute_stage_once(
            stage_dir=root / "stage1",
            configs=stage1_configs,
            invoke=invoke_scored,
            smoke_authorization=smoke_authorization,
            deadline=controller,
            now_utc=now_utc,
        )
        controller.admit_operation("stage1_seal_and_replay", now_utc())  # type: ignore[arg-type]
        selection_path = sealed / "selection.json"
        stage1_verification_path = sealed / "stage1-verification.json"
        stage1_archive_path = sealed / "stage1.zip"
        split_payload, split_sha256 = read_receipt(
            split_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="split_receipt",
        )
        stage1_indices = split_payload.get("chosen_stage1_indices")
        if not isinstance(stage1_indices, list):
            raise AttemptError("Stage-1 split indices are absent")
        (
            selection,
            _selection_sha256,
            _verification_sha256,
            stage1_authorization,
        ) = seal_and_select_stage1(
            stage_dir=root / "stage1",
            archive_path=stage1_archive_path,
            expected_configs=stage1_configs,
            ordered_run_ids=[str(config["run_id"]) for config in stage1_configs],
            stage1_indices=stage1_indices,
            panel_sha256=inputs.preflight.panel_sha256,
            split_receipt_sha256=split_sha256,
            source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
            runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
            selection_receipt_path=selection_path,
            verification_receipt_path=stage1_verification_path,
        )
        controller.transition(Phase.STAGE1_SEALED)
        if selection["finalist"] is None:
            controller.admit_operation("terminal_seal", now_utc())  # type: ignore[arg-type]
            write_stage1_failed_outcome_once(
                study_outcome_path,
                revision=inputs.preflight.revision,
                panel_sha256=inputs.preflight.panel_sha256,
                source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
                runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
                terminal_attempt_sha256=terminal_claim_sha256,
                selection_receipt_path=selection_path,
                stage1_verification_path=stage1_verification_path,
                stage1_archive_path=stage1_archive_path,
                authorization=stage1_authorization,
            )
        else:
            controller.admit_operation("stage2_materialization", now_utc())  # type: ignore[arg-type]
            stage2_configs, stage2_authorization = build_stage2_configs(
                panel_path=panel_path,
                panel_commitment_path=commitment_path,
                split_receipt_path=split_path,
                selection_receipt_path=selection_path,
                stage1_verification_path=stage1_verification_path,
                stage1_archive_path=stage1_archive_path,
                locks=inputs.preflight.locks,
                stage1_authorization=stage1_authorization,
            )
            controller.transition(Phase.STAGE2)
            execute_stage_once(
                stage_dir=root / "stage2",
                configs=stage2_configs,
                invoke=invoke_scored,
                stage2_authorization=stage2_authorization,
                deadline=controller,
                now_utc=now_utc,
            )
            controller.transition(Phase.FINALIZE)
            controller.admit_operation("stage2_seal_and_replay", now_utc())  # type: ignore[arg-type]
            (
                _stage2_result,
                _stage2_verification_sha256,
                stage2_result_authorization,
            ) = seal_and_evaluate_stage2(
                stage_dir=root / "stage2",
                archive_path=sealed / "stage2.zip",
                expected_configs=stage2_configs,
                ordered_run_ids=[str(config["run_id"]) for config in stage2_configs],
                panel_path=panel_path,
                panel_commitment_path=commitment_path,
                split_receipt_path=split_path,
                selection_receipt_path=selection_path,
                stage1_verification_path=stage1_verification_path,
                stage1_archive_path=stage1_archive_path,
                locks=inputs.preflight.locks,
                stage2_verification_path=sealed / "stage2-verification.json",
                stage1_authorization=stage1_authorization,
                stage2_authorization=stage2_authorization,
            )
            controller.admit_operation("terminal_seal", now_utc())  # type: ignore[arg-type]
            write_stage2_outcome_once(
                study_outcome_path,
                revision=inputs.preflight.revision,
                panel_sha256=inputs.preflight.panel_sha256,
                source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
                runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
                terminal_attempt_sha256=terminal_claim_sha256,
                selection_receipt_path=selection_path,
                stage2_verification_path=sealed / "stage2-verification.json",
                stage2_archive_path=sealed / "stage2.zip",
                authorization=stage2_result_authorization,
            )
    except BaseException as error:
        failed_phase = (
            controller.phase.value if controller is not None else "provider_preflight"
        )
        error_code = f"{failed_phase}:{type(error).__name__}"
        if not study_outcome_path.exists():
            write_not_evaluable_outcome_once(
                study_outcome_path,
                revision=inputs.preflight.revision,
                panel_sha256=inputs.preflight.panel_sha256,
                source_lock_sha256=inputs.preflight.locks.source_lock_sha256,
                runtime_lock_sha256=inputs.preflight.locks.runtime_lock_sha256,
                terminal_attempt_sha256=terminal_claim_sha256,
                failed_phase=failed_phase,
                error_code=error_code,
            )
        if controller is not None and controller.phase not in {
            Phase.NOT_EVALUABLE,
            Phase.CLEANUP,
        }:
            try:
                controller.transition(Phase.NOT_EVALUABLE)
            except BaseException:
                pass
    finally:
        if controller is not None:
            try:
                _transition_to_cleanup(controller)
            except BaseException as error:
                error_code = error_code or f"cleanup_transition:{type(error).__name__}"
            try:
                controller.admit_operation("cleanup_evacuation", now_utc())  # type: ignore[arg-type]
            except BaseException as error:
                error_code = error_code or f"cleanup_evacuation_deadline:{type(error).__name__}"
        process_cleanup_path = sealed / "process-cleanup.json"
        process_cleanup = cleanup_process_tree_once(
            process_cleanup_path,
            adapter=process_adapter,
            root_pid=inputs.process_root_pid,
        )
        if process_cleanup.status != "complete":
            error_code = error_code or "process_cleanup:incomplete"
        try:
            if (
                launch_receipt_sha256 is None
                or host_finalizer_receipt_sha256 is None
                or launch_resource_id is None
            ):
                raise AttemptError("pod evidence lacks its paid host binding")
            evidence_authorization = evacuate_and_authenticate(
                source_attempt_root=root,
                destination_root=inputs.evidence_destination_root,
                source_terminal_path=study_outcome_path,
                process_cleanup=process_cleanup,
                process_cleanup_receipt_path=process_cleanup_path,
                terminal_authorization=authenticate_terminal_outcome(
                    study_outcome_path,
                    evidence_root=root,
                    expected_revision=inputs.preflight.revision,
                    expected_panel_sha256=inputs.preflight.panel_sha256,
                    expected_panel_commitment_sha256=(
                        inputs.preflight.locks.panel_commitment_sha256
                    ),
                    expected_split_receipt_sha256=(
                        inputs.preflight.split_receipt_sha256
                    ),
                    expected_package_closure_sha256=(
                        inputs.preflight.locks.package_closure_sha256
                    ),
                    expected_source_lock_sha256=(
                        inputs.preflight.locks.source_lock_sha256
                    ),
                    expected_runtime_lock_sha256=(
                        inputs.preflight.locks.runtime_lock_sha256
                    ),
                    expected_terminal_attempt_sha256=terminal_claim_sha256,
                ),
                pod_error_code=error_code,
                binding=EvidenceHandoffBinding(
                    resource_id=launch_resource_id,
                    panel_sha256=inputs.preflight.panel_sha256,
                    panel_commitment_sha256=(
                        inputs.preflight.locks.panel_commitment_sha256
                    ),
                    split_receipt_sha256=(
                        inputs.preflight.split_receipt_sha256
                    ),
                    package_closure_sha256=(
                        inputs.preflight.locks.package_closure_sha256
                    ),
                    provider_launch_receipt_sha256=launch_receipt_sha256,
                    host_finalizer_receipt_sha256=(
                        host_finalizer_receipt_sha256
                    ),
                    terminal_attempt_sha256=terminal_claim_sha256,
                    implementation_revision=inputs.preflight.revision,
                    source_lock_sha256=(
                        inputs.preflight.locks.source_lock_sha256
                    ),
                    runtime_lock_sha256=(
                        inputs.preflight.locks.runtime_lock_sha256
                    ),
                ),
            )
        except BaseException as error:
            error_code = error_code or f"evidence_evacuation:{type(error).__name__}"

    _, study_sha256 = _read_study_action(study_outcome_path)
    return PodAttemptResult(
        study_outcome_sha256=study_sha256,
        evidence_evacuation_receipt_sha256=(
            evidence_authorization.receipt_sha256
            if evidence_authorization is not None
            else None
        ),
        provider_launch_receipt_sha256=launch_receipt_sha256,
        error_code=error_code,
    )


def run_pod_attempt(
    inputs: PodAttemptInputs,
    *,
    process_adapter: ProcessAdapter,
    now_utc: Callable[[], datetime],
) -> PodAttemptResult:
    """Never let a pod-side exception suppress the external host finalizer."""
    try:
        return _run_pod_attempt_impl(
            inputs,
            process_adapter=process_adapter,
            now_utc=now_utc,
        )
    except BaseException as error:
        return PodAttemptResult(
            study_outcome_sha256=None,
            evidence_evacuation_receipt_sha256=None,
            provider_launch_receipt_sha256=None,
            error_code=f"pod_outer:{type(error).__name__}",
        )
