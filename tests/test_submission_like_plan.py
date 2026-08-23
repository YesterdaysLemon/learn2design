from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments.uifo_paired import runner as runner_module
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.runner import (
    _claim_terminal_attempt,
    _preserve_record_integrity_failure,
    _rebuild_indexes,
    _run_config,
    freeze_or_authenticate_plan_output,
    orchestrate,
    validate_candidate_package,
)
from experiments.uifo_paired.results_ingestion import (
    ExpectedSources,
    StudyValidationError,
    _expected_archive_members,
)
from experiments.uifo_paired.submission_like_results_ingestion import _validate_plan
from experiments.uifo_paired.submission_like_analysis import (
    summarize_submission_like_records,
)


ROOT = Path(__file__).parents[1]
PANEL_PATH = (
    ROOT / "experiments" / "uifo_paired" / "panels" / "submission-like-v1.json"
)
PROVIDER_STOP = "2099-01-01T00:00:00Z"


def _plan() -> dict[str, object]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    metadata = {
        "source_kind": "json_topology_panel",
        "source_name": PANEL_PATH.name,
        "source_sha256": hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest(),
        "archive_exclusion_verified": True,
        "official_dataset_sha256": "test-only",
        "panel_id": panel["panel_id"],
        "topology_count": len(panel["topologies"]),
    }
    return build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[29, 31],
        arms=["no_prior"],
        max_time_seconds=1_200,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=2_100,
        topology_panel=metadata,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20,
        max_session_wall_seconds=9 * 60 * 60,
        max_worker_failures=1,
        study_profile="submission-like-screen-v1",
        seed_order_policy="mirrored_sweeps",
        candidate_package_evidence={
            "format_version": 1,
            "archive_name": "submission.zip",
            "archive_sha256": "1" * 64,
            "builder_manifest_name": "submission.manifest.json",
            "builder_manifest_sha256": "2" * 64,
            "project_revision": "3" * 40,
            "source_files": [
                {"path": "submission.py", "sha256": "4" * 64, "size_bytes": 1},
                {"path": "requirements.txt", "sha256": "5" * 64, "size_bytes": 1},
            ],
            "upstream_reference": "6" * 40,
        },
        provider_stop_utc=PROVIDER_STOP,
        provider_evacuation_reserve_seconds=1_800,
        provider_deadline_maximum_horizon_seconds=10 * 60 * 60,
    )


def _expected(plan: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(run["run_id"]): _run_config(run, plan["configuration"])
        for run in plan["runs"]
    }


def _records(
    expected: dict[str, dict[str, object]],
    *,
    gaps: list[float] | None = None,
) -> list[dict[str, object]]:
    topology_order: list[str] = []
    for config in expected.values():
        topology = str(config["topology"]["value"])
        if topology not in topology_order:
            topology_order.append(topology)
    gap_by_topology = dict(zip(topology_order, gaps or [0.1] * 10))
    records = []
    for config in expected.values():
        topology = str(config["topology"]["value"])
        seed_offset = 0.0 if config["optimizer_seed"] == 29 else gap_by_topology[topology]
        records.append(
            {
                "run_id": config["run_id"],
                "status": "complete",
                "config": config,
                "metrics": {
                    "has_feasible": True,
                    "has_finite_feasible": True,
                    "best_feasible_loss": 1.0 + topology_order.index(topology) + seed_offset,
                },
                "problem": {
                    "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                    "topology_string": topology,
                },
            }
        )
    return records


def test_submission_like_plan_is_exact_cost_bounded_and_order_balanced() -> None:
    plan = _plan()
    assert len(plan["runs"]) == 20
    assert plan["configuration"]["arms"] == ["no_prior"]
    assert plan["configuration"]["execution_mode"] == "serial"
    assert plan["configuration"]["optimizer_seeds"] == [29, 31]
    assert plan["configuration"]["max_time_seconds"] == 1_200
    assert plan["configuration"]["max_session_wall_seconds"] == 9 * 60 * 60
    assert (
        plan["configuration"]["provider_deadline_maximum_horizon_seconds"]
        == 10 * 60 * 60
    )
    assert plan["optimizer_seed_order_policy"] == "mirrored_sweeps"
    assert [run["optimizer_seed"] for run in plan["runs"]] == [29] * 10 + [31] * 10
    first_sweep = [run["topology"] for run in plan["runs"][:10]]
    second_sweep = [run["topology"] for run in plan["runs"][10:]]
    assert second_sweep == list(reversed(first_sweep))
    assert plan["configuration"]["decision_policy"]["changes_packaged_candidate"] is False
    assert plan["configuration"]["decision_policy"]["official_budget_claim_allowed"] is False
    assert len(_expected_archive_members(_expected(plan))) == 109
    assert plan["configuration"]["resource_budget"] == {
        "currency": "USD",
        "gpu_count": 1,
        "maximum_gpu_hourly_price": 1.6,
        "maximum_provider_charge": 16.0,
        "maximum_provider_hours": 10.0,
        "planned_runs": 20,
        "scored_objective_seconds": 24_000,
    }


def test_submission_like_profile_rejects_budget_seed_and_panel_drift() -> None:
    plan = _plan()
    configuration = plan["configuration"]
    panel = configuration["topology_panel"]
    topologies = [item["value"] for item in configuration["topologies"]]
    common = {
        "topology_seeds": None,
        "topologies": topologies,
        "optimizer_seeds": [29, 31],
        "arms": ["no_prior"],
        "max_time_seconds": 1_200,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
        "target_losses": [4.0, 1.0, 0.5, 0.0],
        "worker_timeout_seconds": 2_100,
        "topology_panel": panel,
        "require_a100": True,
        "minimum_gpu_memory_mib": 75_000,
        "max_idle_gpu_memory_mib": 1_000,
        "max_idle_gpu_utilization_percent": 5,
        "minimum_free_disk_gib": 20,
        "max_session_wall_seconds": 9 * 60 * 60,
        "max_worker_failures": 1,
        "study_profile": "submission-like-screen-v1",
        "seed_order_policy": "mirrored_sweeps",
        "candidate_package_evidence": configuration["candidate_package_evidence"],
        "provider_stop_utc": PROVIDER_STOP,
        "provider_evacuation_reserve_seconds": 1_800,
        "provider_deadline_maximum_horizon_seconds": 10 * 60 * 60,
    }
    with pytest.raises(ValueError, match="requires max_time_seconds"):
        build_plan(**{**common, "max_time_seconds": 1_199})
    with pytest.raises(ValueError, match="requires optimizer_seeds"):
        build_plan(**{**common, "optimizer_seeds": [29, 41]})
    with pytest.raises(ValueError, match="requires seed_order_policy"):
        build_plan(**{**common, "seed_order_policy": "listed"})
    with pytest.raises(ValueError, match="requires panel source_sha256"):
        build_plan(
            **{
                **common,
                "topology_panel": {**panel, "source_sha256": "0" * 64},
            }
        )


def test_submission_like_attempt_is_non_resumable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="non-resumable"):
        orchestrate(_plan(), tmp_path / "attempt", resume=True)


def test_record_integrity_failure_is_preserved_as_packageable_error(
    tmp_path: Path,
) -> None:
    plan = _plan()
    expected = _expected(plan)
    config = next(iter(expected.values()))
    path = tmp_path / "runs" / f"{config['run_id']}.json"
    path.parent.mkdir(parents=True)
    original = {
        "format_version": 1,
        "run_id": config["run_id"],
        "status": "complete",
        "config": config,
        "metrics": {"best_feasible_loss": 1.0},
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    preserved = _preserve_record_integrity_failure(
        path, original, RuntimeError("synthetic checksum mismatch")
    )
    assert preserved["status"] == "error"
    assert preserved["metrics"] == original["metrics"]
    assert preserved["error"]["type"] == "RecordIntegrityError"
    assert preserved["error"]["original_worker_status"] == "complete"
    assert preserved["error"]["original_record_sha256"] == original_digest
    assert json.loads(path.read_text(encoding="utf-8")) == preserved
    _rebuild_indexes(tmp_path, expected, {}, validate_complete_records=True)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["error_runs"] == 1
    assert summary["predeclared_decision"]["status"] == "not_evaluable"


def test_terminal_attempt_receipt_is_cross_plan_and_fail_closed(tmp_path: Path) -> None:
    plan = _plan()
    output = tmp_path / str(plan["plan_id"])
    first = _claim_terminal_attempt(plan, output, revision="a" * 40)
    assert first is not None
    receipt = tmp_path / "submission-like-screen-v1.terminal-attempt.json"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == first["receipt_sha256"]

    changed_plan = {**plan, "plan_id": "b" * 16}
    with pytest.raises(RuntimeError, match="already claimed"):
        _claim_terminal_attempt(changed_plan, tmp_path / ("b" * 16), revision="a" * 40)


def test_submission_like_executes_only_the_exact_reviewed_plan(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "submission-like-plan.json"
    digest, reviewed = freeze_or_authenticate_plan_output(
        plan, plan_path, dry_run=True, approved_sha256=None
    )
    assert len(digest) == 64
    rebuilt = _plan()
    rebuilt["created_utc"] = "2099-01-01T00:00:00+00:00"
    observed, executed = freeze_or_authenticate_plan_output(
        rebuilt, plan_path, dry_run=False, approved_sha256=digest
    )
    assert observed == digest
    assert executed == reviewed

    changed = json.loads(json.dumps(plan))
    changed["configuration"]["provider_stop_utc"] = "2099-01-01T00:00:01Z"
    with pytest.raises(ValueError, match="differs from the rebuilt frozen plan"):
        freeze_or_authenticate_plan_output(
            changed, plan_path, dry_run=False, approved_sha256=digest
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze_or_authenticate_plan_output(
            plan, plan_path, dry_run=False, approved_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="requires --approved-plan-sha256"):
        freeze_or_authenticate_plan_output(
            plan, plan_path, dry_run=False, approved_sha256=None
        )


def test_submission_like_summary_uses_topology_blocks_and_is_deterministic() -> None:
    expected = _expected(_plan())
    first = summarize_submission_like_records(_records(expected), expected)
    second = summarize_submission_like_records(_records(expected), expected)
    assert first == second
    assert first["completed_runs"] == 20
    assert first["complete_topologies"] == 10
    assert first["finite_feasible_topologies"] == 10
    assert len(first["topology_rows"]) == 10
    assert first["predeclared_decision"]["status"] == "passed"
    assert first["predeclared_decision"]["action"] == (
        "candidate_evidence_complete_for_submission_review"
    )


def test_submission_like_seed_gap_is_descriptive_and_censoring_fails_closed() -> None:
    expected = _expected(_plan())
    at_boundary = summarize_submission_like_records(
        _records(expected, gaps=[0.1] * 8 + [0.5, 0.5]), expected
    )
    assert at_boundary["topology_p90_absolute_seed_gap"] == pytest.approx(0.5)
    assert at_boundary["predeclared_decision"]["status"] == "passed"

    above = summarize_submission_like_records(
        _records(expected, gaps=[0.1] * 8 + [0.51, 0.51]), expected
    )
    assert above["topology_p90_absolute_seed_gap"] == pytest.approx(0.51)
    assert above["predeclared_decision"]["status"] == "passed"

    censored_records = _records(expected)
    censored_records[0]["metrics"] = {
        "has_feasible": True,
        "has_finite_feasible": False,
        "best_feasible_loss": None,
    }
    censored = summarize_submission_like_records(censored_records, expected)
    assert censored["predeclared_decision"]["status"] == "failed"
    assert censored["predeclared_decision"]["criteria"][
        "all_runs_finite_feasible"
    ] is False

    errored_records = _records(expected)
    errored_records[0] = {**errored_records[0], "status": "error"}
    errored = summarize_submission_like_records(errored_records, expected)
    assert errored["predeclared_decision"]["status"] == "not_evaluable"
    assert errored["predeclared_decision"]["action"] == (
        "retain_candidate_attempt_not_evaluable"
    )


def test_seed_order_policy_rejects_nonpaired_seed_count() -> None:
    with pytest.raises(ValueError, match="exactly two optimizer seeds"):
        build_plan(
            topology_seeds=[1],
            topologies=None,
            optimizer_seeds=[1, 2, 3],
            arms=["no_prior"],
            max_time_seconds=1,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
            seed_order_policy="mirrored_sweeps",
        )


def test_submission_like_ingestion_rejects_broken_hierarchy() -> None:
    plan = _plan()
    expected = ExpectedSources(
        zip_sha256="0" * 64,
        package_manifest_sha256="1" * 64,
        checksum_file_sha256="2" * 64,
        plan_sha256="3" * 64,
        plan_id=str(plan["plan_id"]),
        project_revision="4" * 40,
    )
    assert len(_validate_plan(plan, expected)) == 20
    broken = json.loads(json.dumps(plan))
    broken["runs"][10]["optimizer_seed"] = 29
    with pytest.raises(
        StudyValidationError,
        match="seed or arm mismatch|blocks are broken|run identity mismatch",
    ):
        _validate_plan(broken, expected)


def test_candidate_package_is_hash_and_revision_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "submission.zip"
    source_root = ROOT / "submission"
    source_payloads = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        text = path.read_text(encoding="utf-8")
        source_payloads[path.relative_to(source_root).as_posix()] = (
            text.replace("\r\n", "\n").replace("\r", "\n").encode()
        )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in source_payloads.items():
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload)
    revision = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        runner_module,
        "_git",
        lambda *args: revision if args == ("rev-parse", "HEAD") else "",
    )
    manifest = tmp_path / "submission.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "created_utc": "2026-08-22T00:00:00+00:00",
                "project_revision": revision,
                "source_files": [
                    {
                        "path": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for name, payload in source_payloads.items()
                ],
                "upstream_reference": "b" * 40,
                "working_tree_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    evidence = validate_candidate_package(archive, manifest)
    assert evidence["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(
        runner_module,
        "_git",
        lambda *args: revision if args == ("rev-parse", "HEAD") else " M file",
    )
    with pytest.raises(ValueError, match="clean checkout"):
        validate_candidate_package(archive, manifest)
    monkeypatch.setattr(
        runner_module,
        "_git",
        lambda *args: revision if args == ("rev-parse", "HEAD") else "",
    )
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_candidate_package(archive, manifest)


def test_candidate_package_rejects_zip_bytes_not_executed_by_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "submission.zip"
    manifest = tmp_path / "submission.manifest.json"
    revision = "a" * 40
    monkeypatch.setattr(
        runner_module,
        "_git",
        lambda *args: revision if args == ("rev-parse", "HEAD") else "",
    )
    source_root = ROOT / "submission"
    entries = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source_root.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            name = path.relative_to(source_root).as_posix()
            text = path.read_text(encoding="utf-8")
            payload = text.replace("\r\n", "\n").replace("\r", "\n").encode()
            if name == "submission.py":
                payload += b"\n# tampered candidate\n"
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload)
            entries.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
    manifest.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "created_utc": "2026-08-22T00:00:00+00:00",
                "project_revision": revision,
                "source_files": entries,
                "upstream_reference": "b" * 40,
                "working_tree_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from manifest or checkout"):
        validate_candidate_package(archive, manifest)
