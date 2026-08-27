from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytestmark = pytest.mark.integration

import experiments.uifo_paired.coverage_evidence as coverage_evidence
from experiments.uifo_paired.coverage_results_ingestion import (
    _validate_coverage_history_chronology,
    _validate_coverage_run_chronology,
    _validate_plan,
)
from experiments.uifo_paired.coverage_analysis import summarize_coverage_records
from experiments.uifo_paired.coverage_evidence import (
    CoverageReplayAgreement,
    compare_coverage_replays,
)
from experiments.uifo_paired.coverage_reference_analysis import (
    reference_coverage_screen,
)
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.package import package_study
from experiments.uifo_paired.results_ingestion import (
    ExpectedSources,
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
)
from experiments.uifo_paired.runner import OFFICIAL_DATASET_SHA256, _run_config
from experiments.uifo_paired.runner import (
    _claim_terminal_attempt,
    _terminal_attempt_receipt_path,
    freeze_or_authenticate_plan_output,
    orchestrate,
)
from tests.test_coverage_robustness import _candidate_evidence, _records


ROOT = Path(__file__).parents[1]
PROFILE = "coverage-triage-screen-v1"
PANEL_PATH = (
    ROOT
    / "experiments"
    / "uifo_paired"
    / "panels"
    / "coverage-triage-v1.json"
)
PANEL_SHA256 = "f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972"
PRIOR_PANEL_NAMES = (
    "development-v1.json",
    "confirmation-v1.json",
    "submission-like-v1.json",
    "coverage-robustness-v1.json",
    "restart-mechanics-v1.json",
    "restart-screen-v1.json",
)


def _triage_panel_evidence(panel: dict[str, object]) -> dict[str, object]:
    panel_dir = PANEL_PATH.parent
    topologies = list(panel["topologies"])
    prior_panels = []
    for name in PRIOR_PANEL_NAMES:
        path = panel_dir / name
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_panels.append(
            {
                "source_name": name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "topology_count": len(prior["topologies"]),
                "overlap_count": 0,
            }
        )
    identity_digest = hashlib.sha256(
        json.dumps(sorted(set(topologies)), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_kind": "json_topology_panel",
        "source_name": PANEL_PATH.name,
        "source_sha256": PANEL_SHA256,
        "archive_exclusion_verified": True,
        "official_dataset_sha256": OFFICIAL_DATASET_SHA256,
        "panel_id": panel["panel_id"],
        "topology_count": len(topologies),
        "archive_exclusion_audit": {
            "method": "exact topology-string set intersection",
            "panel_identity_sha256": identity_digest,
            "panel_topology_count": len(topologies),
            "official_dataset": {
                "source_name": "dataset.h5",
                "sha256": OFFICIAL_DATASET_SHA256,
                "size_bytes": 74_920_439,
                "entries": 29_650,
                "unique_topologies": 12_437,
                "overlap_count": 0,
            },
            "prior_panels": prior_panels,
        },
    }


def _triage_plan() -> dict[str, object]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    return build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[37, 41],
        arms=["no_prior", "coverage_balanced"],
        max_time_seconds=600.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        allow_cpu=False,
        worker_timeout_seconds=1_200.0,
        topology_panel=_triage_panel_evidence(panel),
        evaluation_chunk_size=None,
        require_h100=True,
        required_gpu_name="NVIDIA H100 80GB HBM3",
        preclock_warmup=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20.0,
        max_session_wall_seconds=7 * 60 * 60,
        max_worker_failures=1,
        study_profile=PROFILE,
        optimizer_telemetry=None,
        pair_order_policy="alternate_topology_and_seed",
        seed_order_policy="mirrored_sweeps",
        candidate_package_evidence=_candidate_evidence(),
        provider_stop_utc="2026-08-25T12:00:00Z",
        provider_evacuation_reserve_seconds=1_800.0,
        provider_deadline_maximum_horizon_seconds=8 * 60 * 60.0,
    )


def _study(plan: dict[str, object]) -> ValidatedStudy:
    records, configs = _records(plan, winning_topologies=7)
    histories = {}
    for record in records:
        run_id = str(record["run_id"])
        histories[run_id] = [
            {
                "call_index": 0,
                "candidate_index": 0,
                "eval_count_after_call": int(
                    record["objective_accounting"]["eval_count"]
                ),
                "time_seconds": 599.0,
                "loss": float(record["metrics"]["best_feasible_loss"]),
                "sensitivity_loss": 0.0,
                "penalty": 0.0,
                "is_feasible": True,
            }
        ]
    return ValidatedStudy(
        sources=SourcePaths(
            archive=Path("triage.zip"),
            checksum=Path("triage.zip.sha256"),
            package_manifest=Path("triage.zip.manifest.json"),
            plan=Path("triage.plan.json"),
        ),
        source_hashes={},
        archive_members=(),
        plan=plan,
        manifest={},
        package_state={},
        session={},
        configs=configs,
        records=records,
        history_rows=histories,
        integrity={"summary_content_opened": False},
    )


def test_triage_plan_is_exactly_32_serial_runs_under_the_cost_cap() -> None:
    plan = _triage_plan()
    configuration = plan["configuration"]

    assert len(plan["runs"]) == 32
    assert plan["primary_pair_order"] == {
        "complete_primary_pairs": 16,
        "no_prior_first": 8,
        "coverage_balanced_first": 8,
        "absolute_imbalance": 0,
    }
    assert configuration["execution_mode"] == "serial"
    assert configuration["max_time_seconds"] == 600.0
    assert configuration["worker_timeout_seconds"] == 1_200.0
    assert configuration["max_session_wall_seconds"] == 25_200
    assert configuration["provider_deadline_maximum_horizon_seconds"] == 28_800.0
    assert configuration["resource_budget"] == {
        "cloud_type": "SECURE",
        "currency": "USD",
        "gpu_count": 1,
        "gpu_type_id": "NVIDIA H100 80GB HBM3",
        "maximum_gpu_hourly_price": 3.29,
        "maximum_provider_charge": 30.0,
        "maximum_provider_hours": 8.0,
        "planned_runs": 32,
        "scored_objective_seconds": 19_200,
    }
    policy = configuration["decision_policy"]
    assert policy["minimum_coverage_topology_wins"] == 7
    assert policy["action_if_passed"] == (
        "review_precommitted_stage_b_design_and_seek_owner_approval"
    )
    assert policy["changes_packaged_candidate_default"] is False
    stage_b = policy["stage_b_design_precommitment"]
    assert stage_b["status"] == "design_only_not_executable"
    assert stage_b["optimizer_seeds"] == [43, 47]
    assert stage_b["requires_profile_implementation_and_refreeze"] is True
    assert stage_b["requires_separate_owner_approval"] is True

    configs = {
        str(run["run_id"]): _run_config(run, configuration)
        for run in plan["runs"]
    }
    assert {config["initial_population_mode"] for config in configs.values()} == {
        "random",
        "coverage_balanced",
    }


def test_triage_ingestion_accepts_the_real_exclusion_audit_shape_only() -> None:
    plan = _triage_plan()
    expected = ExpectedSources(
        zip_sha256="a" * 64,
        package_manifest_sha256="b" * 64,
        checksum_file_sha256="d" * 64,
        plan_sha256="e" * 64,
        plan_id=str(plan["plan_id"]),
        project_revision="c" * 40,
        study_profile=PROFILE,
    )
    _validate_plan(plan, expected)

    mutated = json.loads(json.dumps(plan))
    del mutated["configuration"]["topology_panel"]["archive_exclusion_audit"]
    with pytest.raises(StudyValidationError, match="committed panel bytes"):
        _validate_plan(mutated, expected)


def test_triage_panel_is_fresh_disjoint_and_all_older_hashes_are_stable() -> None:
    assert hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest() == PANEL_SHA256
    triage = set(json.loads(PANEL_PATH.read_text(encoding="utf-8"))["topologies"])
    prior_hashes = {
        "development-v1.json": "d5f660261e413f59b179d4fadf1f157b30f117aa265fd230d1d130bd6d69246b",
        "confirmation-v1.json": "52fe189709b27e2abb7de659fae0c080faf25b89f3ce66a3b1a13025be221dba",
        "submission-like-v1.json": (
            "d85227f216528d635e56a93094e661721f62f379808707f310bf4da60d8fa57b"
        ),
        "coverage-robustness-v1.json": (
            "e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309"
        ),
    }
    for name, digest in prior_hashes.items():
        path = PANEL_PATH.with_name(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        prior = set(json.loads(path.read_text(encoding="utf-8"))["topologies"])
        assert triage.isdisjoint(prior), name
    for name in ("restart-mechanics-v1.json", "restart-screen-v1.json"):
        path = PANEL_PATH.with_name(name)
        prior = set(json.loads(path.read_text(encoding="utf-8"))["topologies"])
        assert triage.isdisjoint(prior), name
    audit = json.loads(PANEL_PATH.with_name("audit.json").read_text(encoding="utf-8"))
    record = next(item for item in audit["panels"] if item["panel_id"] == "coverage-triage-v1")
    assert record["source_sha256"] == PANEL_SHA256
    assert record["archive_overlap_count"] == 0
    assert set(record["previous_panel_overlap_counts"].values()) == {0}


def test_seven_of_eight_is_required_and_stage_a_cannot_promote() -> None:
    plan = _triage_plan()
    seven_records, configs = _records(plan, winning_topologies=7)
    passed = summarize_coverage_records(seven_records, configs)
    assert passed["wins_ties_losses"] == {
        "coverage_balanced_wins": 7,
        "ties": 0,
        "coverage_balanced_losses": 1,
    }
    assert passed["predeclared_decision"] == {
        "status": "passed",
        "passed": True,
        "action": "review_precommitted_stage_b_design_and_seek_owner_approval",
        "criteria": passed["predeclared_decision"]["criteria"],
    }
    assert len(passed["predeclared_decision"]["criteria"]) == 14

    six_records, configs = _records(plan, winning_topologies=6)
    failed = summarize_coverage_records(six_records, configs)
    assert failed["predeclared_decision"]["status"] == "failed"
    assert failed["predeclared_decision"]["action"] == "retain_random_start_candidate"
    assert failed["predeclared_decision"]["criteria"]["minimum_topology_wins_met"] is False


def test_triage_reference_replay_is_independent_and_shape_bound() -> None:
    study = _study(_triage_plan())
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)
    agreement = compare_coverage_replays(production, reference, study=study)

    receipt = agreement.as_dict()
    assert receipt == {
        "status": "matched",
        "runs_compared": 32,
        "topology_values_compared": 8,
        "optimizer_seed_pairs_compared": 16,
        "frozen_criteria_compared": 14,
        "study_identity_sha256": receipt["study_identity_sha256"],
    }


def _chronology_case(
    *,
    worker_seconds: float = 100.0,
    completed_utc: str = "2026-08-24T00:01:40+00:00",
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    expected_config = {
        "run_id": "synthetic-run",
        "population_size": 2,
        "max_time_seconds": 600.0,
    }
    rows = [
        {
            "call_index": 0,
            "candidate_index": candidate_index,
            "eval_count_after_call": 2,
            "time_seconds": 90.0,
        }
        for candidate_index in range(2)
    ]
    record = {
        "started_utc": "2026-08-24T00:00:05+00:00",
        "completed_utc": completed_utc,
        "worker_process": {
            "started_utc": "2026-08-24T00:00:00+00:00",
            "completed_utc": "2026-08-24T00:01:40+00:00",
            "full_wall_seconds": worker_seconds,
        },
    }
    return rows, expected_config, record


def test_triage_chronology_rejects_worker_wall_mutations() -> None:
    rows, expected_config, record = _chronology_case(worker_seconds=610.0)
    with pytest.raises(StudyValidationError, match="UTC process interval"):
        _validate_coverage_history_chronology(
            rows, expected_config, record, worker_timeout_seconds=1_200.0
        )

    rows, expected_config, record = _chronology_case(worker_seconds=80.0)
    with pytest.raises(StudyValidationError, match="UTC process interval"):
        _validate_coverage_history_chronology(
            rows, expected_config, record, worker_timeout_seconds=1_200.0
        )


def test_triage_chronology_rejects_final_objective_after_run_interval() -> None:
    rows, expected_config, record = _chronology_case(
        worker_seconds=100.0,
        completed_utc="2026-08-24T00:01:24+00:00",
    )
    with pytest.raises(
        StudyValidationError, match="final Objective time exceeds its UTC run interval"
    ):
        _validate_coverage_history_chronology(
            rows, expected_config, record, worker_timeout_seconds=1_200.0
        )


def test_triage_chronology_rejects_overlap_and_out_of_plan_order() -> None:
    session_started = datetime(2026, 8, 24, tzinfo=UTC)
    session_completed = datetime(2026, 8, 24, 1, tzinfo=UTC)
    with pytest.raises(
        StudyValidationError, match="overlap or violate plan order"
    ):
        _validate_coverage_run_chronology(
            [
                (
                    "run-1",
                    datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
                    datetime(2026, 8, 24, 0, 10, tzinfo=UTC),
                ),
                (
                    "run-2",
                    datetime(2026, 8, 24, 0, 5, tzinfo=UTC),
                    datetime(2026, 8, 24, 0, 15, tzinfo=UTC),
                ),
            ],
            session_started,
            session_completed,
        )

    with pytest.raises(
        StudyValidationError, match="overlap or violate plan order"
    ):
        _validate_coverage_run_chronology(
            [
                (
                    "run-1",
                    datetime(2026, 8, 24, 0, 10, tzinfo=UTC),
                    datetime(2026, 8, 24, 0, 20, tzinfo=UTC),
                ),
                (
                    "run-2",
                    datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
                    datetime(2026, 8, 24, 0, 5, tzinfo=UTC),
                ),
            ],
            session_started,
            session_completed,
        )


def test_triage_chronology_rejects_run_outside_session() -> None:
    session_started = datetime(2026, 8, 24, tzinfo=UTC)
    session_completed = datetime(2026, 8, 24, 1, tzinfo=UTC)
    with pytest.raises(StudyValidationError, match="outside the session"):
        _validate_coverage_run_chronology(
            [
                (
                    "run-1",
                    datetime(2026, 8, 23, 23, 59, tzinfo=UTC),
                    datetime(2026, 8, 24, 0, 10, tzinfo=UTC),
                )
            ],
            session_started,
            session_completed,
        )


def test_replay_agreement_has_no_importable_module_seal() -> None:
    assert not hasattr(coverage_evidence, "_REPLAY_AGREEMENT_SEAL")
    with pytest.raises(TypeError):
        CoverageReplayAgreement({})


def test_triage_requires_plan_approval_and_one_shared_attempt_ledger(
    tmp_path: Path,
) -> None:
    plan = _triage_plan()
    plan_path = tmp_path / "coverage-triage.plan.json"
    alternate_plan_path = tmp_path / "renamed-approved-plan.json"
    assert _terminal_attempt_receipt_path(
        plan_path, PROFILE
    ) == _terminal_attempt_receipt_path(alternate_plan_path, PROFILE)
    with pytest.raises(ValueError, match="approved-plan-sha256"):
        freeze_or_authenticate_plan_output(
            plan,
            plan_path,
            dry_run=False,
            approved_sha256=None,
        )
    with pytest.raises(RuntimeError, match="non-resumable"):
        orchestrate(plan, tmp_path / "attempt-a", resume=True)
    with pytest.raises(RuntimeError, match="authenticated approved plan"):
        orchestrate(plan, tmp_path / "attempt-a", resume=False)

    ledger = tmp_path / "coverage-triage.terminal-attempt.json"
    first = _claim_terminal_attempt(
        plan,
        tmp_path / "attempt-a",
        revision="c" * 40,
        receipt_path=ledger,
    )
    assert first is not None
    with pytest.raises(RuntimeError, match="already claimed"):
        _claim_terminal_attempt(
            plan,
            tmp_path / "different-output-root" / "attempt-b",
            revision="c" * 40,
            receipt_path=ledger,
        )


def test_triage_source_lock_schema_is_separate_and_exact() -> None:
    path = (
        ROOT
        / "experiments"
        / "uifo_paired"
        / "schemas"
        / "coverage-triage-source-lock.schema.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["properties"]["study_profile"]["const"] == PROFILE
    assert payload["properties"]["files"]["minProperties"] == 6


def test_triage_packager_detaches_and_commits_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    for name in ("configs", "histories", "logs", "runs"):
        (study / name).mkdir(parents=True, exist_ok=True)
    run_id = "pair__no_prior"
    run = {
        "planned_run_index": 0,
        "run_id": run_id,
        "pair_id": "pair",
        "run_order_within_pair": 0,
        "topology": {"kind": "string", "value": "AAAAAAAAA-DLLLLLLLLLLL"},
        "optimizer_seed": 37,
        "arm": "no_prior",
    }
    effective = {
        "CUDA_CACHE_DISABLE": "1",
        "JAX_COMPILATION_CACHE_DIR": None,
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "XLA_FLAGS": None,
    }
    configuration = {
        "allow_cpu": False,
        "evaluation_chunk_size": None,
        "max_evals": None,
        "max_time_seconds": 600.0,
        "n_frequencies": 50,
        "population_size": 8,
        "require_a100": False,
        "jax_compilation_cache_policy": "disabled",
        "target_losses": [4.0, 1.0, 0.5, 0.0],
        "study_profile": PROFILE,
    }
    environment = {
        "backend": "gpu",
        "jax_runtime_configuration": {
            "compilation_cache_dir": None,
            "enable_compilation_cache": False,
        },
    }
    manifest = {
        "format_version": 1,
        "plan_id": "synthetic-plan",
        "project_revision": "c" * 40,
        "configuration": configuration,
        "runs": [run],
        "environment": environment,
        "runtime_policy": {
            "jax_compilation_cache": {
                "policy": "disabled",
                "effective_environment": effective,
            }
        },
    }
    payloads = {
        "manifest.json": json.dumps(manifest),
        "preflight.host-environment.json": "{}",
        "preflight.json": "{}",
        "preflight.stderr.log": "",
        "preflight.stdout.log": "",
        "runs.jsonl": "{}\n",
        "session.json": '{"status":"complete"}',
        "summary.json": '{"private_result":"committed"}\n',
        f"configs/{run_id}.json": "{}",
        f"histories/{run_id}.npz": "history",
        f"logs/{run_id}.stdout.log": "",
        f"logs/{run_id}.stderr.log": "",
        f"runs/{run_id}.json": '{"status":"complete"}',
    }
    for relative, content in payloads.items():
        (study / relative).write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "experiments.uifo_paired.package._rebuild_indexes",
        lambda *args, **kwargs: None,
    )

    output = tmp_path / "triage.zip"
    result = package_study(study, output)
    release = tmp_path / "triage.zip.summary.json"
    assert release.read_text(encoding="utf-8") == payloads["summary.json"]
    assert result["summary_release"]["sha256"] == hashlib.sha256(
        release.read_bytes()
    ).hexdigest()
    with zipfile.ZipFile(output) as archive:
        assert "summary.json" not in archive.namelist()
        commitment = json.loads(archive.read("summary.commitment.json"))
    assert commitment["summary_sha256"] == result["summary_release"]["sha256"]
