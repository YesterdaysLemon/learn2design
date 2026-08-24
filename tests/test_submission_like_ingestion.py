from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="submission-like fixtures require NumPy")
pytestmark = pytest.mark.integration

from experiments.uifo_paired.metrics import summarize_rows
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.results_ingestion import (
    ExpectedSources,
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
    sha256_bytes,
    sha256_path,
)
from experiments.uifo_paired.submission_like_results_ingestion import (
    authenticate_submission_like_source_lock,
    load_submission_like_summary_after_reproduction,
    submission_like_package_is_complete,
    validate_submission_like_archive,
    validate_submission_like_terminal_partial,
)
from experiments.uifo_paired.runner import (
    HISTORY_SCHEMA,
    JAX_RUNTIME_ENVIRONMENT_KEYS,
    _expected_algorithm_record,
    _initial_population_roles,
    _metric_grids,
    _parameter_hashes,
    _rows_from_history_arrays,
    _run_config,
    strict_json,
)
from experiments.uifo_paired.submission_like_analysis import (
    summarize_submission_like_records,
)
from tools.create_submission_like_source_lock import build_source_lock
from tools.analyze_submission_like import run_analysis


ROOT = Path(__file__).parents[1]
PANEL_PATH = (
    ROOT / "experiments" / "uifo_paired" / "panels" / "submission-like-v1.json"
)
REVISION = "7" * 40
UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"


@dataclass(frozen=True)
class _CompleteBundle:
    sources: SourcePaths
    receipt: Path
    source_lock: Path


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = (
                zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, payload, compresslevel=6)
    return buffer.getvalue()


def _environment() -> dict[str, object]:
    runtime_environment = {name: None for name in JAX_RUNTIME_ENVIRONMENT_KEYS}
    runtime_environment.update(
        {
            "CUDA_CACHE_DISABLE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    return {
        "backend": "gpu",
        "device_count": 1,
        "device_kinds": ["NVIDIA A100 80GB PCIe"],
        "device_platforms": ["gpu"],
        "devices": ["TFRT_CUDA_0"],
        "competition_aligned_a100": True,
        "jax_runtime_environment": runtime_environment,
        "jax_runtime_configuration": {
            "compilation_cache_dir": None,
            "enable_compilation_cache": False,
        },
        "nvidia_smi": {"status": "ok", "gpus": []},
        "versions": {},
        "platform": "Linux-synthetic-test",
        "python": "3.12.11",
    }


def _submission_like_plan() -> dict[str, object]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    return build_plan(
        topology_seeds=None,
        topologies=list(panel["topologies"]),
        optimizer_seeds=[29, 31],
        arms=["no_prior"],
        max_time_seconds=1_200,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=2_100,
        topology_panel={
            "source_kind": "json_topology_panel",
            "source_name": PANEL_PATH.name,
            "source_sha256": hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest(),
            "archive_exclusion_verified": True,
            "official_dataset_sha256": "synthetic-test-only",
            "panel_id": panel["panel_id"],
            "topology_count": len(panel["topologies"]),
        },
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
            "project_revision": REVISION,
            "source_files": [
                {"path": "submission.py", "sha256": "3" * 64, "size_bytes": 1},
                {
                    "path": "requirements.txt",
                    "sha256": "4" * 64,
                    "size_bytes": 1,
                },
            ],
            "upstream_reference": UPSTREAM_REFERENCE,
        },
        provider_stop_utc="2099-01-01T00:00:00Z",
        provider_evacuation_reserve_seconds=1_800,
        provider_deadline_maximum_horizon_seconds=10 * 60 * 60,
    )


def _history_payload(loss: float) -> tuple[bytes, list[dict[str, object]]]:
    buffer = io.BytesIO()
    initial = np.arange(16, dtype=np.float64).reshape(8, 2)
    np.savez(
        buffer,
        call_index=np.zeros(8, dtype=np.int32),
        candidate_index=np.arange(8, dtype=np.int16),
        eval_count_after_call=np.full(8, 8, dtype=np.int64),
        time_seconds=np.full(8, 1.0, dtype=np.float64),
        loss=np.asarray(
            [loss + candidate / 100 for candidate in range(8)], dtype=np.float64
        ),
        sensitivity_loss=np.zeros(8, dtype=np.float64),
        penalty=np.zeros(8, dtype=np.float64),
        is_feasible=np.ones(8, dtype=np.bool_),
        initial_params_unbounded=initial,
    )
    payload = buffer.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in HISTORY_SCHEMA}
    return payload, _rows_from_history_arrays(arrays)


def _build_complete_bundle(tmp_path: Path) -> _CompleteBundle:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = _submission_like_plan()
    configuration = plan["configuration"]
    environment = _environment()
    receipt = tmp_path / "submission-like-screen-v1.terminal-attempt.json"
    receipt.write_bytes(
        _json_bytes(
            {
                "format_version": 1,
                "study_profile": "submission-like-screen-v1",
                "plan_id": plan["plan_id"],
                "project_revision": REVISION,
                "claimed_utc": "2026-08-22T00:00:00+00:00",
                "rule": (
                    "first result-bearing attempt is terminal; resume and rerun "
                    "forbidden"
                ),
            }
        )
    )
    manifest = {
        **plan,
        "project_revision": REVISION,
        "working_tree_dirty": False,
        "semantic_prior_canonical_sha256": "synthetic-no-prior-study",
        "upstream_reference": UPSTREAM_REFERENCE,
        "environment": environment,
        "rental_preflight": {
            "disk": {"path": "/workspace/results", "free_bytes": 30 * 1024**3},
            "gpu_idle": {
                "status": "ok",
                "gpus": [
                    {
                        "name": "NVIDIA A100 80GB PCIe",
                        "mig_mode_current": "Disabled",
                        "memory_total_mib": 81_920,
                    }
                ],
            },
        },
        "runtime_policy": {
            "jax_compilation_cache": {
                "policy": "disabled",
                "effective_environment": environment["jax_runtime_environment"],
            }
        },
        "terminal_attempt": {
            "receipt_name": receipt.name,
            "receipt_sha256": sha256_path(receipt),
        },
    }
    members = {
        "manifest.json": _json_bytes(manifest),
        "package-state.json": _json_bytes(
            {
                "format_version": 1,
                "study_complete": True,
                "planned_runs": 20,
                "completed_runs": 20,
                "incomplete_runs": [],
            }
        ),
        "session.json": _json_bytes(
            {
                "status": "complete",
                "started_utc": "2026-08-22T00:00:00+00:00",
                "completed_utc": "2026-08-22T06:00:00+00:00",
                "elapsed_seconds": 21_600.0,
                "max_session_wall_seconds": 32_400.0,
            }
        ),
        "summary.json": b"synthetic summary is deliberately sealed and malformed\n",
        "preflight.host-environment.json": _json_bytes({"format_version": 1}),
        "preflight.json": _json_bytes(environment),
        "preflight.stderr.log": b"",
        "preflight.stdout.log": b"synthetic preflight passed\n",
    }
    records = []
    for index, run in enumerate(plan["runs"]):
        config = _run_config(run, configuration)
        run_id = str(config["run_id"])
        history_payload, rows = _history_payload(1.0 + index / 100)
        time_grid, eval_grid = _metric_grids(config)
        metrics = summarize_rows(
            rows,
            list(config["target_losses"]),
            time_grid=time_grid,
            eval_grid=eval_grid,
        )
        with np.load(io.BytesIO(history_payload), allow_pickle=False) as history:
            initial = np.asarray(history["initial_params_unbounded"])
        stdout = f"synthetic completion {run_id}\n".encode()
        stderr = b""
        topology = str(config["topology"]["value"])
        record = {
            "format_version": 1,
            "run_id": run_id,
            "status": "complete",
            "started_utc": f"2026-08-22T00:{index:02d}:00+00:00",
            "completed_utc": f"2026-08-22T00:{index:02d}:30+00:00",
            "config": config,
            "environment": environment,
            "metrics": metrics,
            "problem": {
                "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                "topology_string": topology,
                "spec": config["topology"],
                "n_params": 2,
            },
            "objective_configuration": {
                "max_evals": config["max_evals"],
                "max_time_seconds": config["max_time_seconds"],
                "save": ["batched_loss", "batched_is_feasible"],
            },
            "algorithm": _expected_algorithm_record(config),
            "objective_accounting": {"log_call_count": 1, "eval_count": 8},
            "initial_population_roles": _initial_population_roles("no_prior", 8),
            "initial_parameter_hashes": _parameter_hashes(initial),
            "history": {
                "format_version": 1,
                "path": f"histories/{run_id}.npz",
                "rows": len(rows),
                "schema": HISTORY_SCHEMA,
                "sha256": sha256_bytes(history_payload),
            },
            "worker_process": {
                "returncode": 0,
                "timed_out": False,
                "full_wall_seconds": 1_201.0,
                "within_official_4h30_container_limit": True,
                "stdout": {
                    "path": f"logs/{run_id}.stdout.log",
                    "sha256": sha256_bytes(stdout),
                },
                "stderr": {
                    "path": f"logs/{run_id}.stderr.log",
                    "sha256": sha256_bytes(stderr),
                },
            },
        }
        records.append(record)
        members[f"configs/{run_id}.json"] = _json_bytes(config)
        members[f"histories/{run_id}.npz"] = history_payload
        members[f"logs/{run_id}.stdout.log"] = stdout
        members[f"logs/{run_id}.stderr.log"] = stderr
        members[f"runs/{run_id}.json"] = _json_bytes(record)
    records.sort(key=lambda record: str(record["run_id"]))
    members["runs.jsonl"] = b"".join(
        json.dumps(strict_json(record), sort_keys=True, allow_nan=False).encode()
        + b"\n"
        for record in records
    )
    assert len(members) == 109

    archive = tmp_path / "submission-like-screen-v1.zip"
    archive.write_bytes(_zip_bytes(members))
    plan_path = tmp_path / "submission-like-screen-v1-plan.json"
    plan_path.write_bytes(_json_bytes(plan))
    checksum = tmp_path / "submission-like-screen-v1.zip.sha256"
    checksum.write_text(
        f"{sha256_path(archive)}  {archive.name}\n", encoding="utf-8"
    )
    package_manifest = tmp_path / "submission-like-screen-v1.zip.manifest.json"
    package_manifest.write_bytes(
        _json_bytes(
            {
                "format_version": 1,
                "study_plan_id": plan["plan_id"],
                "study_project_revision": REVISION,
                "study_complete": True,
                "planned_runs": 20,
                "completed_runs": 20,
                "incomplete_runs": [],
                "archive": {
                    "path": "/workspace/results/submission-like-screen-v1.zip",
                    "sha256": sha256_path(archive),
                    "size_bytes": archive.stat().st_size,
                    "files": len(members),
                },
            }
        )
    )
    sources = SourcePaths(archive, checksum, package_manifest, plan_path)
    source_lock, _ = _write_lock(tmp_path, sources, receipt)
    return _CompleteBundle(sources, receipt, source_lock)


def _authenticate(bundle: _CompleteBundle) -> ExpectedSources:
    return authenticate_submission_like_source_lock(
        bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        sources=bundle.sources,
        terminal_attempt_receipt=bundle.receipt,
    )


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _refresh_complete_bundle(
    bundle: _CompleteBundle, members: dict[str, bytes]
) -> ExpectedSources:
    bundle.sources.archive.write_bytes(_zip_bytes(members))
    archive_digest = sha256_path(bundle.sources.archive)
    bundle.sources.checksum.write_text(
        f"{archive_digest}  {bundle.sources.archive.name}\n", encoding="utf-8"
    )
    package = json.loads(
        bundle.sources.package_manifest.read_text(encoding="utf-8")
    )
    package["archive"].update(
        {
            "sha256": archive_digest,
            "size_bytes": bundle.sources.archive.stat().st_size,
            "files": len(members),
        }
    )
    bundle.sources.package_manifest.write_bytes(_json_bytes(package))
    payload = build_source_lock(
        archive=bundle.sources.archive,
        checksum=bundle.sources.checksum,
        package_manifest=bundle.sources.package_manifest,
        plan=bundle.sources.plan,
        terminal_attempt_receipt=bundle.receipt,
    )
    bundle.source_lock.write_bytes(_json_bytes(payload))
    return _authenticate(bundle)


def _make_terminal_partial(
    bundle: _CompleteBundle,
    *,
    status: str = "error",
) -> ExpectedSources:
    plan = json.loads(bundle.sources.plan.read_text(encoding="utf-8"))
    run_id = str(plan["runs"][-1]["run_id"])
    members = _read_members(bundle.sources.archive)
    for name in (
        f"configs/{run_id}.json",
        f"histories/{run_id}.npz",
        f"logs/{run_id}.stdout.log",
        f"logs/{run_id}.stderr.log",
        f"runs/{run_id}.json",
    ):
        members.pop(name)
    incomplete = [{"run_id": run_id, "status": "missing"}]
    members["package-state.json"] = _json_bytes(
        {
            "format_version": 1,
            "study_complete": False,
            "planned_runs": 20,
            "completed_runs": 19,
            "incomplete_runs": incomplete,
        }
    )
    session: dict[str, object] = {
        "status": status,
        "started_utc": "2026-08-22T00:00:00+00:00",
    }
    if status == "running":
        session["max_session_wall_seconds"] = 32_400.0
        recovery = _json_bytes(
            {
                "pid": 1234,
                "hostname": "synthetic-dead-writer",
                "created_utc": "2026-08-22T05:40:00+00:00",
            }
        )
        receipt_name = f"recovery/stale-study-lock-{sha256_bytes(recovery)[:12]}.json"
        members[receipt_name] = recovery
    else:
        session.update(
            {
                "completed_utc": "2026-08-22T05:40:00+00:00",
                "elapsed_seconds": 20_400.0,
            }
        )
        if status in {"error", "interrupted", "wall_limit_reached"}:
            session["max_session_wall_seconds"] = 32_400.0
        if status in {"wall_limit_reached", "provider_deadline_guard"}:
            session["next_run_id"] = run_id
        if status == "provider_deadline_guard":
            session["provider_stop_utc"] = plan["configuration"]["provider_stop_utc"]
    members["session.json"] = _json_bytes(session)
    package = json.loads(bundle.sources.package_manifest.read_text(encoding="utf-8"))
    package.update(
        {
            "study_complete": False,
            "completed_runs": 19,
            "incomplete_runs": incomplete,
        }
    )
    bundle.sources.package_manifest.write_bytes(_json_bytes(package))
    return _refresh_complete_bundle(bundle, members)


def _source_files(tmp_path: Path) -> tuple[SourcePaths, Path, Path]:
    archive = tmp_path / "submission-like-screen-v1.zip"
    archive.write_bytes(b"archive")
    checksum = tmp_path / "submission-like-screen-v1.zip.sha256"
    checksum.write_text(f"{'a' * 64}  {archive.name}\n", encoding="utf-8")
    package_manifest = tmp_path / "submission-like-screen-v1.zip.manifest.json"
    package_manifest.write_text(
        json.dumps(
            {
                "study_plan_id": "b" * 16,
                "study_project_revision": "c" * 40,
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "submission-like-screen-v1-plan.json"
    plan.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "submission-like-screen-v1.terminal-attempt.json"
    receipt.write_text("{}", encoding="utf-8")
    return SourcePaths(archive, checksum, package_manifest, plan), receipt, package_manifest


def _write_lock(
    tmp_path: Path, sources: SourcePaths, receipt: Path
) -> tuple[Path, str]:
    payload = build_source_lock(
        archive=sources.archive,
        checksum=sources.checksum,
        package_manifest=sources.package_manifest,
        plan=sources.plan,
        terminal_attempt_receipt=receipt,
    )
    path = tmp_path / "submission-like-screen-v1-source-lock.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, sha256_path(path)


def test_source_lock_authenticates_hashes_sizes_and_basenames(tmp_path: Path) -> None:
    sources, receipt, _ = _source_files(tmp_path)
    lock, lock_hash = _write_lock(tmp_path, sources, receipt)
    expected = authenticate_submission_like_source_lock(
        lock,
        expected_source_lock_sha256=lock_hash,
        sources=sources,
        terminal_attempt_receipt=receipt,
    )
    assert expected.plan_id == "b" * 16
    assert expected.project_revision == "c" * 40
    assert expected.zip_sha256 == sha256_path(sources.archive)

    sources.archive.write_bytes(b"changed-size-and-hash")
    with pytest.raises(StudyValidationError, match="file mismatch"):
        authenticate_submission_like_source_lock(
            lock,
            expected_source_lock_sha256=lock_hash,
            sources=sources,
            terminal_attempt_receipt=receipt,
        )


def test_source_lock_digest_is_checked_before_json_is_parsed(tmp_path: Path) -> None:
    sources, receipt, _ = _source_files(tmp_path)
    lock = tmp_path / "malformed-source-lock.json"
    lock.write_text("not json and must remain unread", encoding="utf-8")
    with pytest.raises(StudyValidationError, match="SHA-256 mismatch"):
        authenticate_submission_like_source_lock(
            lock,
            expected_source_lock_sha256="d" * 64,
            sources=sources,
            terminal_attempt_receipt=receipt,
        )


def test_source_lock_rejects_filename_and_size_schema_corruption(tmp_path: Path) -> None:
    sources, receipt, _ = _source_files(tmp_path)
    lock, _ = _write_lock(tmp_path, sources, receipt)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    archive_entry = payload["files"].pop(sources.archive.name)
    payload["files"]["foreign.zip"] = archive_entry
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudyValidationError, match="basenames"):
        authenticate_submission_like_source_lock(
            lock,
            expected_source_lock_sha256=sha256_path(lock),
            sources=sources,
            terminal_attempt_receipt=receipt,
        )

    payload["files"].pop("foreign.zip")
    payload["files"][sources.archive.name] = {
        "sha256": sha256_path(sources.archive),
        "size_bytes": -1,
    }
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudyValidationError, match="size is invalid"):
        authenticate_submission_like_source_lock(
            lock,
            expected_source_lock_sha256=sha256_path(lock),
            sources=sources,
            terminal_attempt_receipt=receipt,
        )


def test_archived_summary_stays_sealed_until_full_replay_agreement(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("summary.json", '{"status":"archived"}')
    study = ValidatedStudy(
        sources=SourcePaths(archive, Path("a"), Path("b"), Path("c")),
        source_hashes={},
        archive_members=("summary.json",),
        plan={},
        manifest={},
        package_state={},
        session={},
        configs={},
        records=[],
        history_rows={},
        integrity={"summary_content_opened": False},
    )
    with pytest.raises(StudyValidationError, match="remains locked"):
        load_submission_like_summary_after_reproduction(
            study, {"status": "matched", "topology_values_compared": 10}
        )
    summary = load_submission_like_summary_after_reproduction(
        study,
        {
            "status": "matched",
            "topology_values_compared": 10,
            "runs_compared": 20,
            "frozen_criteria_compared": 5,
        },
    )
    assert summary == {"status": "archived"}


def test_committed_source_lock_schema_is_valid_json() -> None:
    root = Path(__file__).parents[1]
    schema = json.loads(
        (
            root
            / "experiments/uifo_paired/schemas/submission-like-source-lock.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["files"]["minProperties"] == 5


def test_complete_109_member_archive_validates_end_to_end_and_keeps_summary_sealed(
    tmp_path: Path,
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    expected = _authenticate(bundle)
    study = validate_submission_like_archive(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    assert len(study.archive_members) == 109
    assert len(study.configs) == 20
    assert len(study.records) == 20
    assert len(study.history_rows) == 20
    assert study.integrity["records"] == 20
    assert study.integrity["histories"] == 20
    assert study.integrity["topologies"] == 10
    assert study.integrity["optimizer_seed_repetitions"] == 20
    assert study.integrity["external_hashes"] == "passed"
    assert study.integrity["source_lock"] == "passed"
    assert study.integrity["terminal_attempt_receipt"] == "passed"
    assert study.integrity["summary_content_opened"] is False


def test_complete_analysis_writes_safe_aggregates_and_private_diagnostics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("markdown", reason="submission-like analysis dependency")
    pytest.importorskip("matplotlib", reason="submission-like analysis dependency")
    pytest.importorskip("scipy", reason="submission-like analysis dependency")
    bundle = _build_complete_bundle(tmp_path / "bundle")
    expected = _authenticate(bundle)
    study = validate_submission_like_archive(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    members = _read_members(bundle.sources.archive)
    members["summary.json"] = _json_bytes(
        summarize_submission_like_records(study.records, study.configs)
    )
    _refresh_complete_bundle(bundle, members)

    output = tmp_path / "complete-analysis"
    result = run_analysis(
        sources=bundle.sources,
        source_lock=bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        terminal_attempt_receipt=bundle.receipt,
        output=output,
    )

    assert result["raw_replay"] == "matched"
    assert result["archived_summary"] == "matched"
    assert result["figures"] == 5
    assert (output / "private_posthoc_diagnostics.json").is_file()
    assert (output / "posthoc_analysis.json").is_file()
    assert (output / "analysis_report.md").is_file()
    assert (output / "analysis_report.html").is_file()
    assert (output / "handoff.json").is_file()
    assert sorted(path.name for path in (output / "figures").iterdir()) == [
        "leave_one_topology_out.png",
        "run_order_and_throughput.png",
        "target_hitting_outcomes.png",
        "topology_seed_outcomes.png",
        "trajectory_alignment.png",
    ]
    safe_serialized = (output / "posthoc_analysis.json").read_text(
        encoding="utf-8"
    )
    assert "topology_sha256" not in safe_serialized
    assert "run_id" not in safe_serialized
    assert "recommended_next_evidence_gate" not in safe_serialized


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_submission_like_archive_rejects_missing_or_unexpected_member(
    tmp_path: Path, corruption: str
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    members = _read_members(bundle.sources.archive)
    if corruption == "missing":
        members.pop(next(name for name in members if name.startswith("histories/")))
    else:
        members["optimizer-telemetry/unsolicited.npz"] = b"synthetic unsolicited data"
    expected = _refresh_complete_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="archive member count mismatch"):
        validate_submission_like_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_submission_like_archive_rejects_broken_history_after_outer_reauthentication(
    tmp_path: Path,
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    members = _read_members(bundle.sources.archive)
    history_name = next(name for name in members if name.startswith("histories/"))
    members[history_name] += b"synthetic corruption"
    expected = _refresh_complete_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="history SHA-256 mismatch"):
        validate_submission_like_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_submission_like_archive_rejects_topology_hierarchy_corruption(
    tmp_path: Path,
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    members = _read_members(bundle.sources.archive)
    run_name = next(name for name in members if name.startswith("runs/"))
    record = json.loads(members[run_name])
    replacement = "AAAAAAAAA-LSSSSSSSSSSS"
    record["problem"]["topology_string"] = replacement
    record["problem"]["topology_sha256"] = hashlib.sha256(
        replacement.encode()
    ).hexdigest()
    members[run_name] = _json_bytes(record)
    expected = _refresh_complete_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="differs from explicit plan"):
        validate_submission_like_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_terminal_partial_is_structurally_authenticated_without_opening_outcomes(
    tmp_path: Path,
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    expected = _make_terminal_partial(bundle)
    assert submission_like_package_is_complete(bundle.sources, expected) is False
    result = validate_submission_like_terminal_partial(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    assert result["status"] == "not_evaluable"
    assert result["action"] == "retain_candidate_attempt_not_evaluable"
    assert result["completed_runs"] == 19
    assert result["summary_content_opened"] is False
    assert result["run_records_opened"] is False
    assert result["histories_opened"] is False
    command_result = run_analysis(
        sources=bundle.sources,
        source_lock=bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        terminal_attempt_receipt=bundle.receipt,
        output=tmp_path / "partial-analysis",
    )
    assert command_result["status"] == "not_evaluable"
    assert command_result["summary_content_opened"] is False
    assert (tmp_path / "partial-analysis/terminal_partial_integrity.json").is_file()


@pytest.mark.parametrize(
    "status", ["interrupted", "wall_limit_reached", "provider_deadline_guard", "running"]
)
def test_terminal_partial_accepts_declared_terminal_session_modes(
    tmp_path: Path, status: str
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    expected = _make_terminal_partial(bundle, status=status)
    result = validate_submission_like_terminal_partial(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    assert result["status"] == "not_evaluable"
    assert result["summary_content_opened"] is False


def test_terminal_partial_requires_artifacts_for_every_claimed_complete_run(
    tmp_path: Path,
) -> None:
    bundle = _build_complete_bundle(tmp_path)
    _make_terminal_partial(bundle)
    members = _read_members(bundle.sources.archive)
    plan = json.loads(bundle.sources.plan.read_text(encoding="utf-8"))
    claimed_complete = str(plan["runs"][0]["run_id"])
    members.pop(f"histories/{claimed_complete}.npz")
    expected = _refresh_complete_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="claimed complete run lacks artifacts"):
        validate_submission_like_terminal_partial(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_terminal_partial_rejects_package_state_disagreement(tmp_path: Path) -> None:
    bundle = _build_complete_bundle(tmp_path)
    plan = json.loads(bundle.sources.plan.read_text(encoding="utf-8"))
    run_id = str(plan["runs"][-1]["run_id"])
    members = _read_members(bundle.sources.archive)
    incomplete = [{"run_id": run_id, "status": "missing"}]
    members["package-state.json"] = _json_bytes(
        {
            "format_version": 1,
            "study_complete": False,
            "planned_runs": 20,
            "completed_runs": 18,
            "incomplete_runs": incomplete,
        }
    )
    members["session.json"] = _json_bytes(
        {
            "status": "error",
            "elapsed_seconds": 1.0,
            "max_session_wall_seconds": 32_400.0,
        }
    )
    package = json.loads(bundle.sources.package_manifest.read_text(encoding="utf-8"))
    package.update(
        {
            "study_complete": False,
            "completed_runs": 19,
            "incomplete_runs": incomplete,
        }
    )
    bundle.sources.package_manifest.write_bytes(_json_bytes(package))
    expected = _refresh_complete_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="package-state mismatch"):
        validate_submission_like_terminal_partial(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )
