from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import experiments.feasibility_debt_candidate_screen.orchestrator as orchestrator_module

from experiments.feasibility_debt_candidate_screen.archive import (
    ArchiveError,
    inspect_stage_archive,
    seal_stage_archive,
)
from experiments.feasibility_debt_candidate_screen.canonical import (
    ReceiptError,
    canonical_json_bytes,
    sha256_bytes,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    ARM_ORDER,
    OFFICIAL_ARCHIVE_SHA256,
    PANEL_SEED_ATTEMPTS,
    PANEL_SEED_START,
    PRIOR_PANEL_SHA256,
    SMOKE_TOPOLOGY_SEED,
    STUDY_ID,
    UPSTREAM_REFERENCE,
    arm_spec,
    stage2_order,
)
from experiments.feasibility_debt_candidate_screen.orchestrator import (
    LockDigests,
    OrchestratorError,
    SmokeAuthorization,
    WorkerInvocationError,
    build_stage1_configs,
    build_stage2_configs,
    claim_terminal_attempt_once,
    execute_stage_once,
    seal_panel_bundle_once,
    validate_population_pairing,
    write_stage1_failed_outcome_once,
)
from experiments.feasibility_debt_candidate_screen.panel import (
    enumerate_legal_splits,
    topology_features,
)
from experiments.feasibility_debt_candidate_screen.smoke import build_smoke_config


def _topology(seed: int) -> str:
    offset = seed - PANEL_SEED_START
    stratum = offset % 18
    pair = stratum // 2
    readout = "D" if stratum % 2 == 0 else "H"
    pairs = (
        ("low", "low"),
        ("middle", "middle"),
        ("high", "high"),
        ("low", "middle"),
        ("middle", "high"),
        ("high", "low"),
        ("low", "high"),
        ("middle", "low"),
        ("high", "middle"),
    )
    squeezer_bin, directional_bin = pairs[pair]
    directional_count = {"low": 2, "middle": 4, "high": 6}[directional_bin]
    squeezer_count = {"low": 3, "middle": 5, "high": 7}[squeezer_bin]
    value = max(offset, 0)
    digits: list[int] = []
    for _ in range(9):
        digits.append(value % 4)
        value //= 4
    directional = "EFGH"
    nondirectional = "ABCD"
    interior = "".join(
        (directional if index < directional_count else nondirectional)[digit]
        for index, digit in enumerate(digits)
    )
    boundary = readout + "S" * squeezer_count + "L" * (11 - squeezer_count)
    return f"{interior}-{boundary}"


def _counts(rows: list[dict[str, object]], indices: list[int]):
    result: dict[str, dict[str, int]] = {}
    for field in ("readout", "squeezer_bin", "directional_bin"):
        values: dict[str, int] = {}
        for index in indices:
            value = str(rows[index][field])
            values[value] = values.get(value, 0) + 1
        result[field] = dict(sorted(values.items()))
    return result


def _bundle(tmp_path: Path):
    members = [
        {
            "topology_seed": PANEL_SEED_START + index,
            "topology": _topology(PANEL_SEED_START + index),
            **topology_features(_topology(PANEL_SEED_START + index)),
        }
        for index in range(8)
    ]
    topologies = [str(row["topology"]) for row in members]
    panel = {
        "format_version": 1,
        "panel_id": STUDY_ID,
        "generation": {
            "method": "first eight exact round-robin eligible candidates",
            "seed_start": PANEL_SEED_START,
            "seed_attempts": PANEL_SEED_ATTEMPTS,
            "upstream_reference": UPSTREAM_REFERENCE,
        },
        "members": members,
        "topologies": topologies,
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_bytes(canonical_json_bytes(panel))
    panel_sha = sha256_bytes(panel_path.read_bytes())
    commitment_path = tmp_path / "panel-commitment.json"
    commitment_sha = write_receipt(
        commitment_path,
        study_id=STUDY_ID,
        receipt_type="panel_commitment",
        payload={
            "panel_id": STUDY_ID,
            "panel_sha256": panel_sha,
            "official_archive": {
                "sha256": OFFICIAL_ARCHIVE_SHA256,
                "entries": 1,
                "unique_topologies": 1,
            },
            "prior_panels": [
                {
                    "logical_id": name,
                    "sha256": digest,
                    "topology_count": 1,
                    "previous_exclusion_overlap_count": 0,
                }
                for name, digest in PRIOR_PANEL_SHA256.items()
            ],
            "candidate_seed_start": PANEL_SEED_START,
            "candidate_seed_attempts": PANEL_SEED_ATTEMPTS,
            "eligible_unique_candidates": PANEL_SEED_ATTEMPTS,
            "archive_overlap_count": 0,
            "prior_panel_overlap_count": 0,
            "smoke_topology_seed": SMOKE_TOPOLOGY_SEED,
            "smoke_topology_sha256": sha256_bytes(
                _topology(SMOKE_TOPOLOGY_SEED).encode("utf-8")
            ),
            "smoke_overlap_count": 0,
            "upstream_reference": UPSTREAM_REFERENCE,
        },
    )
    legal, chosen = enumerate_legal_splits(members)
    candidate_rows = [
        {
            "selection_index": index,
            "topology_seed": row["topology_seed"],
            "topology_sha256": sha256_bytes(str(row["topology"]).encode("utf-8")),
            "readout": row["readout"],
            "squeezer_count": row["squeezer_count"],
            "squeezer_bin": row["squeezer_bin"],
            "directional_interior_count": row["directional_interior_count"],
            "directional_bin": row["directional_bin"],
        }
        for index, row in enumerate(members)
    ]
    split_path = tmp_path / "split-receipt.json"
    split_sha = write_receipt(
        split_path,
        study_id=STUDY_ID,
        receipt_type="split_receipt",
        payload={
            "panel_sha256": panel_sha,
            "candidate_rows": candidate_rows,
            "legal_split_rows": legal,
            "chosen_stage1_indices": chosen["stage1_indices"],
            "chosen_stage2_indices": chosen["stage2_indices"],
            "stratum_counts": {
                "stage1": _counts(members, chosen["stage1_indices"]),
                "stage2": _counts(members, chosen["stage2_indices"]),
            },
            "independent_verification": {
                "status": "matched",
                "candidate_count": PANEL_SEED_ATTEMPTS,
                "selected_count": 8,
                "legal_split_count": len(legal),
            },
        },
    )
    return (
        panel_path,
        commitment_path,
        split_path,
        panel_sha,
        commitment_sha,
        split_sha,
    )


def _locks(panel_commitment_sha256: str) -> LockDigests:
    return LockDigests(
        source_lock_sha256="a" * 64,
        runtime_lock_sha256="b" * 64,
        revision="1" * 40,
        package_closure_sha256="c" * 64,
        panel_commitment_sha256=panel_commitment_sha256,
    )


def _smoke_authorization(
    tmp_path: Path, *, commitment_path: Path, locks: LockDigests
):
    config = build_smoke_config(
        revision=locks.revision,
        source_lock_sha256=locks.source_lock_sha256,
        runtime_lock_sha256=locks.runtime_lock_sha256,
        package_closure_sha256=locks.package_closure_sha256,
        panel_commitment_path=commitment_path,
        provider_launch_receipt_sha256="7" * 64,
        resource_manifest_sha256="8" * 64,
        hard_stop_receipt_sha256="9" * 64,
        deadline_snapshot={
            "t0_utc": "2026-09-01T00:00:00Z",
            "b0_utc": "2026-09-01T00:00:00Z",
            "hard_horizon_utc": "2026-09-01T07:00:00Z",
            "dispatch_deadline_utc": "2026-09-01T06:30:00Z",
        },
    )
    return SmokeAuthorization(
        receipt_sha256="6" * 64,
        panel_commitment_sha256=str(config["panel_commitment_sha256"]),
        source_lock_sha256=str(config["source_lock_sha256"]),
        runtime_lock_sha256=str(config["runtime_lock_sha256"]),
        revision=str(config["revision"]),
        provider_launch_receipt_sha256=str(
            config["provider_launch_receipt_sha256"]
        ),
        resource_manifest_sha256=str(config["resource_manifest_sha256"]),
        hard_stop_receipt_sha256=str(config["hard_stop_receipt_sha256"]),
        hard_horizon_utc=str(config["hard_horizon_utc"]),
        _sentinel=orchestrator_module._SMOKE_SENTINEL,
    )


def test_sealed_panel_bundle_is_the_external_stage_membership_authority(
    tmp_path: Path,
) -> None:
    (tmp_path / "source").mkdir()
    panel_path, commitment_path, split_path, _panel, commitment, split_sha = (
        _bundle(tmp_path / "source")
    )
    locks = _locks(commitment)
    sealed = tmp_path / "attempt" / "sealed"
    sealed.mkdir(parents=True)
    panel, split, observed_split = seal_panel_bundle_once(
        panel_path=panel_path,
        panel_commitment_path=commitment_path,
        split_receipt_path=split_path,
        sealed_dir=sealed,
        locks=locks,
    )
    assert observed_split == split_sha
    configs = build_stage1_configs(
        panel_path=sealed / "panel.json",
        panel_commitment_path=sealed / "panel-commitment.json",
        split_receipt_path=sealed / "split-receipt.json",
        locks=locks,
    )
    indices, finalist = orchestrator_module._bind_stage_configs_to_panel(
        configs,
        stage=1,
        panel=panel,
        split=split,
        split_receipt_sha256=split_sha,
        locks=locks,
        selection_receipt_sha256=None,
    )
    assert indices == split["chosen_stage1_indices"]
    assert finalist is None
    changed = json.loads(json.dumps(configs))
    changed[0]["topology"] = panel["topologies"][
        split["chosen_stage2_indices"][0]
    ]
    changed[0]["topology_sha256"] = sha256_bytes(
        changed[0]["topology"].encode("utf-8")
    )
    with pytest.raises(OrchestratorError, match="external panel lock"):
        orchestrator_module._bind_stage_configs_to_panel(
            changed,
            stage=1,
            panel=panel,
            split=split,
            split_receipt_sha256=split_sha,
            locks=locks,
            selection_receipt_sha256=None,
        )
def _selection(
    path: Path,
    *,
    panel_sha: str,
    split_sha: str,
    finalist: str | None,
) -> str:
    return write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="selection_receipt",
        payload={
            "panel_sha256": panel_sha,
            "split_receipt_sha256": split_sha,
            "stage1_archive_sha256": "c" * 64,
            "ordered_run_ids": [f"run-{index}" for index in range(16)],
            "challenger_rows": [],
            "eligible_ids": [] if finalist is None else [finalist],
            "finalist": finalist,
            "action": (
                "retain_round1_control_stage1_failed"
                if finalist is None
                else "advance_selected_finalist_to_stage2"
            ),
            "stage2_outcome_opened": False,
        },
    )


def test_stage2_is_not_materialized_without_authenticated_selection(
    tmp_path: Path,
) -> None:
    panel, commitment, split, panel_sha, commitment_sha, split_sha = _bundle(
        tmp_path
    )
    locks = _locks(commitment_sha)
    stage1 = build_stage1_configs(
        panel_path=panel,
        panel_commitment_path=commitment,
        split_receipt_path=split,
        locks=locks,
    )
    assert len(stage1) == 16
    assert [config["arm_id"] for config in stage1[:4]] == [
        "A_round1_control",
        "B_round1_warmup",
        "C_v3_random",
        "D_v3_coverage",
    ]
    assert all(config["selection_receipt_sha256"] is None for config in stage1)

    failed_path = tmp_path / "failed-selection.json"
    _selection(
        failed_path,
        panel_sha=panel_sha,
        split_sha=split_sha,
        finalist=None,
    )
    verification_path = tmp_path / "forged-verification.json"
    write_receipt(
        verification_path,
        study_id=STUDY_ID,
        receipt_type="stage1_verification",
        payload={
            "status": "matched",
            "stage": 1,
            "values_compared": 1,
            "archive_sha256": "c" * 64,
            "ordered_run_ids": [str(config["run_id"]) for config in stage1],
            "production_sha256": "d" * 64,
            "reference_sha256": "d" * 64,
            "panel_sha256": panel_sha,
            "split_receipt_sha256": split_sha,
            "source_lock_sha256": locks.source_lock_sha256,
            "runtime_lock_sha256": locks.runtime_lock_sha256,
            "selection_receipt_sha256": sha256_bytes(failed_path.read_bytes()),
            "detached_sha256": "e" * 64,
            "detached_values_compared": 1,
        },
    )
    with pytest.raises((OrchestratorError, FileNotFoundError)):
        build_stage2_configs(
            panel_path=panel,
            panel_commitment_path=commitment,
            split_receipt_path=split,
            selection_receipt_path=failed_path,
            stage1_verification_path=verification_path,
            stage1_archive_path=tmp_path / "absent-stage1.zip",
            locks=locks,
            stage1_authorization=None,  # type: ignore[arg-type]
        )
    terminal_path = tmp_path / "forged-stage1-terminal.json"
    with pytest.raises(OrchestratorError, match="valid Stage-1 failure"):
        write_stage1_failed_outcome_once(
            terminal_path,
            revision=locks.revision,
            panel_sha256=panel_sha,
            source_lock_sha256=locks.source_lock_sha256,
            runtime_lock_sha256=locks.runtime_lock_sha256,
            terminal_attempt_sha256="9" * 64,
            selection_receipt_path=failed_path,
            stage1_verification_path=verification_path,
            stage1_archive_path=tmp_path / "absent-stage1.zip",
            authorization=None,  # type: ignore[arg-type]
        )
    assert not terminal_path.exists()


def test_stage2_dispatch_requires_exact_opaque_selection_gate(tmp_path: Path) -> None:
    panel, commitment, split, panel_sha, commitment_sha, split_sha = _bundle(
        tmp_path
    )
    locks = _locks(commitment_sha)
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    finalist = "C_v3_random"
    indices = [1, 3, 5, 7]
    configs = []
    within: dict[int, int] = {}
    for position, (member, arm) in enumerate(stage2_order(indices, finalist)):
        local = within.get(member, 0)
        within[member] = local + 1
        configs.append(
            orchestrator_module._base_config(
                stage=2,
                member_index=member,
                execution_position=position,
                within_member_position=local,
                arm_id=arm,
                optimizer_seed=20260902,
                topology=panel_value["topologies"][member],
                panel_sha256=panel_sha,
                split_receipt_sha256=split_sha,
                selection_receipt_sha256="4" * 64,
                locks=locks,
            )
        )

    calls: list[str] = []

    def should_not_run(config_path: Path, history_path: Path):
        calls.append(config_path.name)
        raise AssertionError(history_path)

    with pytest.raises(OrchestratorError, match="selection-issued dispatch gate"):
        execute_stage_once(
            stage_dir=tmp_path / "stage2-no-token",
            configs=configs,
            invoke=should_not_run,
        )
    wrong = orchestrator_module.Stage2DispatchAuthorization(
        configs_sha256="0" * 64,
        selection_receipt_sha256="4" * 64,
        stage1_verification_sha256="5" * 64,
        stage1_archive_sha256="6" * 64,
        finalist=finalist,
        _sentinel=orchestrator_module._STAGE2_DISPATCH_SENTINEL,
    )
    with pytest.raises(OrchestratorError, match="selection-issued dispatch gate"):
        execute_stage_once(
            stage_dir=tmp_path / "stage2-wrong-token",
            configs=configs,
            invoke=should_not_run,
            stage2_authorization=wrong,
        )
    assert calls == []


def test_population_pairing_binds_same_raw_draw_per_topology() -> None:
    records = []
    for member in (0, 2, 4, 6):
        for arm in (
            "A_round1_control",
            "B_round1_warmup",
            "C_v3_random",
            "D_v3_coverage",
        ):
            records.append(
                {
                    "config": {"member_index": member, "arm_id": arm},
                    "initial_population": {
                        "raw_population_sha256": f"{member + 1:064x}",
                        "raw_member_sha256": [f"{member + 2:064x}"] * 8,
                        "before_warmup": {"count": 1},
                        "after_warmup": {"count": 1},
                    },
                }
            )
    validate_population_pairing(records, stage=1)
    records[-1]["initial_population"]["raw_population_sha256"] = "f" * 64
    with pytest.raises(OrchestratorError, match="differs"):
        validate_population_pairing(records, stage=1)


def test_stage_archive_is_deterministic_safe_and_authenticated(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    (stage_dir / "configs").mkdir(parents=True)
    (stage_dir / "configs/a.json").write_bytes(b"{}")
    (stage_dir / "histories").mkdir()
    (stage_dir / "histories/a.bin").write_bytes(b"history")
    first = seal_stage_archive(
        stage_dir,
        tmp_path / "first.zip",
        stage=1,
        ordered_run_ids=["a"],
    )
    second = seal_stage_archive(
        stage_dir,
        tmp_path / "second.zip",
        stage=1,
        ordered_run_ids=["a"],
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    manifest = inspect_stage_archive(
        tmp_path / "first.zip",
        expected_sha256=first["archive_sha256"],
        expected_stage=1,
        expected_run_ids=["a"],
    )
    assert manifest["stage"] == 1
    with pytest.raises(ArchiveError, match="already exists"):
        seal_stage_archive(
            stage_dir,
            tmp_path / "first.zip",
            stage=1,
            ordered_run_ids=["a"],
        )


def test_first_worker_failure_is_terminal_and_stage_cannot_resume(tmp_path: Path) -> None:
    panel, commitment, split, _panel_sha, commitment_sha, _split_sha = _bundle(
        tmp_path
    )
    configs = build_stage1_configs(
        panel_path=panel,
        panel_commitment_path=commitment,
        split_receipt_path=split,
        locks=_locks(commitment_sha),
    )
    smoke_authorization = _smoke_authorization(
        tmp_path,
        commitment_path=commitment,
        locks=_locks(commitment_sha),
    )
    calls: list[str] = []

    def fail(config_path: Path, history_path: Path):
        calls.append(config_path.name)
        raise WorkerInvocationError(
            "invalid_json",
            stdout=b"private raw",
            stderr=b"",
            returncode=0,
            timed_out=False,
        )

    with pytest.raises(OrchestratorError, match="cold-smoke authorization"):
        execute_stage_once(
            stage_dir=tmp_path / "unauthorized-stage1",
            configs=configs,
            invoke=fail,
        )
    assert calls == []
    stage_dir = tmp_path / "stage1"
    with pytest.raises(OrchestratorError, match="first worker failure"):
        execute_stage_once(
            stage_dir=stage_dir,
            configs=configs,
            invoke=fail,
            smoke_authorization=smoke_authorization,
        )
    assert calls == [f"{configs[0]['run_id']}.json"]
    failure = (stage_dir / "worker-failure.json").read_text(encoding="utf-8")
    assert "private raw" not in failure
    with pytest.raises(OrchestratorError, match="retries are forbidden"):
        execute_stage_once(
            stage_dir=stage_dir,
            configs=configs,
            invoke=fail,
            smoke_authorization=smoke_authorization,
        )


def test_terminal_attempt_claim_is_global_write_once(tmp_path: Path) -> None:
    claim_terminal_attempt_once(
        tmp_path,
        attempt_root=tmp_path / "attempt-root",
        revision="1" * 40,
        panel_sha256="2" * 64,
        source_lock_sha256="3" * 64,
    )
    with pytest.raises(ReceiptError, match="already exists"):
        claim_terminal_attempt_once(
            tmp_path,
            attempt_root=tmp_path / "different-attempt-root",
            revision="1" * 40,
            panel_sha256="2" * 64,
            source_lock_sha256="3" * 64,
        )
