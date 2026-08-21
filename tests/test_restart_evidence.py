from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.restart_evidence import (
    validate_mechanics_predecessor,
)


ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    study = tmp_path / "study"
    (study / "runs").mkdir(parents=True)
    (study / "histories").mkdir()
    (study / "optimizer-telemetry").mkdir()
    revision = "2" * 40
    panel_path = (
        ROOT
        / "experiments"
        / "uifo_paired"
        / "panels"
        / "restart-mechanics-v1.json"
    )
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    topology_panel = {
        "source_kind": "json_topology_panel",
        "source_name": panel_path.name,
        "source_sha256": _sha256(panel_path),
        "archive_exclusion_verified": True,
        "official_dataset_sha256": "test-only",
        "panel_id": panel["panel_id"],
        "topology_count": 1,
    }
    plan = build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[11],
        arms=["no_prior_p200"],
        max_time_seconds=600,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=1_200,
        topology_panel=topology_panel,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20,
        max_session_wall_seconds=1_800,
        max_worker_failures=1,
        study_profile="restart-mechanics-v1",
        optimizer_telemetry="member-v1",
        arm_patience={"no_prior_p200": 200},
        provider_stop_utc="2099-01-01T00:00:00Z",
        provider_evacuation_reserve_seconds=1_800,
    )
    plan_id = plan["plan_id"]
    run_id = plan["runs"][0]["run_id"]
    manifest = {
        **plan,
        "project_revision": revision,
    }
    (study / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (study / "summary.json").write_text(
        json.dumps(
            {
                "predeclared_decision": {
                    "status": "passed",
                    "passed": True,
                    "action": "run_restart_screen_v1",
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    history_path = study / "histories" / f"{run_id}.npz"
    telemetry_path = study / "optimizer-telemetry" / f"{run_id}.npz"
    history_path.write_bytes(b"history")
    telemetry_path.write_bytes(b"telemetry")
    record = {
        "run_id": run_id,
        "status": "complete",
        "history": {"sha256": _sha256(history_path)},
        "optimizer_telemetry": {"sha256": _sha256(telemetry_path)},
    }
    (study / "runs" / f"{run_id}.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )
    validation = {
        "study_complete": True,
        "planned_runs": 1,
        "completed_runs": 1,
        "manifest": manifest,
    }
    monkeypatch.setattr(
        "experiments.uifo_paired.package.validate_complete_study",
        lambda _study: validation,
    )

    package = tmp_path / "mechanics.zip"
    local_files = sorted(path for path in study.rglob("*") if path.is_file())
    package_state = {
        "format_version": 1,
        "study_complete": True,
        "planned_runs": 1,
        "completed_runs": 1,
        "incomplete_runs": [],
    }
    with zipfile.ZipFile(package, "w") as archive:
        for path in local_files:
            archive.write(path, path.relative_to(study).as_posix())
        archive.writestr("package-state.json", json.dumps(package_state))
    digest = _sha256(package)
    package.with_suffix(".zip.sha256").write_text(
        f"{digest}  {package.name}\n", encoding="utf-8"
    )
    package_manifest = {
        "format_version": 1,
        "study_plan_id": plan_id,
        "study_project_revision": revision,
        "study_complete": True,
        "planned_runs": 1,
        "completed_runs": 1,
        "incomplete_runs": [],
        "archive": {
            "sha256": digest,
            "size_bytes": package.stat().st_size,
            "files": len(local_files) + 1,
        },
    }
    package.with_suffix(".zip.manifest.json").write_text(
        json.dumps(package_manifest, sort_keys=True), encoding="utf-8"
    )
    return study, package, validation


def test_mechanics_predecessor_authenticates_package_and_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, package, _ = _fixture(tmp_path, monkeypatch)
    evidence = validate_mechanics_predecessor(study, package)
    assert evidence["study_profile"] == "restart-mechanics-v1"
    assert evidence["package_sha256"] == _sha256(package)
    assert evidence["decision_status"] == "passed"
    assert evidence["decision_action"] == "run_restart_screen_v1"
    assert set(evidence) == {
        "format_version",
        "study_profile",
        "plan_id",
        "project_revision",
        "package_sha256",
        "package_manifest_sha256",
        "record_sha256",
        "history_sha256",
        "optimizer_telemetry_sha256",
        "decision_status",
        "decision_action",
    }


def test_mechanics_predecessor_rejects_failed_incomplete_and_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, package, validation = _fixture(tmp_path, monkeypatch)
    (study / "summary.json").write_text(
        json.dumps(
            {
                "predeclared_decision": {
                    "status": "failed",
                    "passed": False,
                    "action": "retain_patience_600",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        validate_mechanics_predecessor(study, package)

    study, package, validation = _fixture(tmp_path / "incomplete", monkeypatch)
    validation["study_complete"] = False
    validation["completed_runs"] = 0
    with pytest.raises(RuntimeError, match="not one complete run"):
        validate_mechanics_predecessor(study, package)

    study, package, _ = _fixture(tmp_path / "tampered", monkeypatch)
    with package.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="sidecar does not match"):
        validate_mechanics_predecessor(study, package)
