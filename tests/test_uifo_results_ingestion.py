from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.uifo_paired.metrics import summarize_rows
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.results_ingestion import (
    ArchiveLimits,
    ExpectedSources,
    SourcePaths,
    StudyValidationError,
    _load_history_arrays,
    _validate_member_name,
    _validate_plan_contract,
    inspect_zip_integrity,
    load_summary_after_reproduction,
    sha256_bytes,
    sha256_path,
    validate_study_archive,
)
from experiments.uifo_paired.reference_analysis import reference_replay
from experiments.uifo_paired.posthoc_analysis import (
    exact_mean_sign_flip_test,
    exact_two_sided_sign_test,
)
from experiments.uifo_paired.results_workflow import (
    compare_production_and_reference,
)
from experiments.uifo_paired.analysis import summarize_records
from experiments.uifo_paired.runner import (
    HISTORY_SCHEMA,
    JAX_RUNTIME_ENVIRONMENT_KEYS,
    _initial_population_roles,
    _metric_grids,
    _parameter_hashes,
    _rows_from_history_arrays,
    _run_config,
    strict_json,
)


ROOT = Path(__file__).parents[1]


def _development_plan() -> dict[str, object]:
    panel = json.loads(
        (ROOT / "experiments/uifo_paired/panels/development-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return build_plan(
        topology_seeds=None,
        topologies=list(panel["topologies"]),
        optimizer_seeds=[7, 11],
        arms=["no_prior", "semantic_prior"],
        max_time_seconds=600.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        allow_cpu=False,
        worker_timeout_seconds=1200.0,
        topology_panel={
            "panel_id": "development-v1",
            "topology_count": 16,
            "archive_exclusion_verified": True,
        },
        evaluation_chunk_size=None,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20.0,
        max_session_wall_seconds=57_600.0,
        max_worker_failures=2,
        study_profile="development-v2",
    )


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
        "platform": "Linux-test",
        "python": "3.12.11",
    }


def _history_payload(
    loss: float, arm: str = "no_prior"
) -> tuple[bytes, list[dict[str, object]]]:
    buffer = io.BytesIO()
    initial = np.arange(16, dtype=np.float64).reshape(8, 2)
    if arm == "semantic_prior":
        initial[1] += 1000.0
    np.savez(
        buffer,
        call_index=np.zeros(8, dtype=np.int32),
        candidate_index=np.arange(8, dtype=np.int16),
        eval_count_after_call=np.full(8, 8, dtype=np.int64),
        time_seconds=np.full(8, 1.0, dtype=np.float64),
        loss=np.asarray([loss + index / 100 for index in range(8)], dtype=np.float64),
        sensitivity_loss=np.zeros(8, dtype=np.float64),
        penalty=np.zeros(8, dtype=np.float64),
        is_feasible=np.ones(8, dtype=np.bool_),
        initial_params_unbounded=initial,
    )
    payload = buffer.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in HISTORY_SCHEMA}
    return payload, _rows_from_history_arrays(arrays)


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


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _fixture_files(tmp_path: Path) -> tuple[SourcePaths, ExpectedSources]:
    plan = _development_plan()
    environment = _environment()
    configuration = plan["configuration"]
    manifest = {
        **plan,
        "project_revision": "dbd557b713ab657ac971957369d89eb67649d09f",
        "working_tree_dirty": False,
        "semantic_prior_canonical_sha256": "prior",
        "upstream_reference": "upstream",
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
    }
    members = {
        "manifest.json": _json_bytes(manifest),
        "package-state.json": _json_bytes(
            {
                "format_version": 1,
                "study_complete": True,
                "planned_runs": 64,
                "completed_runs": 64,
                "incomplete_runs": [],
            }
        ),
        "session.json": _json_bytes(
            {
                "status": "complete",
                "started_utc": "2026-08-20T00:00:00+00:00",
                "completed_utc": "2026-08-20T12:00:00+00:00",
                "elapsed_seconds": 43_200.0,
                "max_session_wall_seconds": 57_600.0,
            }
        ),
        "summary.json": b'{"outcome":"deliberately unopened"}\n',
        "preflight.host-environment.json": _json_bytes({"format_version": 1}),
        "preflight.json": _json_bytes(environment),
        "preflight.stderr.log": b"",
        "preflight.stdout.log": b"",
    }
    records = []
    for index, run in enumerate(plan["runs"]):
        config = _run_config(run, configuration)
        run_id = str(run["run_id"])
        loss = 1.0 + (index % 7) / 10
        history_payload, rows = _history_payload(loss, str(config["arm"]))
        time_grid, eval_grid = _metric_grids(config)
        metrics = summarize_rows(
            rows,
            list(config["target_losses"]),
            time_grid=time_grid,
            eval_grid=eval_grid,
        )
        with np.load(io.BytesIO(history_payload), allow_pickle=False) as archive:
            initial = np.asarray(archive["initial_params_unbounded"])
        stdout = f"completed {run_id}\n".encode()
        stderr = b""
        topology = str(config["topology"]["value"])
        record = {
            "format_version": 1,
            "run_id": run_id,
            "status": "complete",
            "started_utc": "2026-08-20T00:00:00+00:00",
            "completed_utc": "2026-08-20T00:10:00+00:00",
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
                "max_evals": None,
                "max_time_seconds": 600.0,
                "save": ["batched_loss", "batched_is_feasible"],
            },
            "algorithm": {
                "module": "submission.submission",
                "class": "BatchedRestartAdam",
                "algorithm_str": "batched_restart_adam",
                "kwargs": {
                    "random_seed": config["optimizer_seed"],
                    "population_size": 8,
                    "evaluation_chunk_size": None,
                    "use_semantic_prior": config["arm"] == "semantic_prior",
                },
            },
            "objective_accounting": {"log_call_count": 1, "eval_count": 8},
            "initial_population_roles": _initial_population_roles(
                str(config["arm"]), 8
            ),
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
                "full_wall_seconds": 601.0,
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
        json.dumps(strict_json(record), sort_keys=True, allow_nan=False).encode() + b"\n"
        for record in records
    )

    archive_path = tmp_path / "development-v2.zip"
    archive_path.write_bytes(_zip_bytes(members))
    plan_path = tmp_path / "development-v2-plan.json"
    plan_path.write_bytes(_json_bytes(plan))
    checksum_path = tmp_path / "development-v2.zip.sha256"
    checksum_path.write_text(
        f"{sha256_path(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    package_manifest = {
        "format_version": 1,
        "study_plan_id": plan["plan_id"],
        "study_project_revision": manifest["project_revision"],
        "study_complete": True,
        "planned_runs": 64,
        "completed_runs": 64,
        "incomplete_runs": [],
        "archive": {
            "path": "/workspace/results/development-v2.zip",
            "sha256": sha256_path(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "files": len(members),
        },
    }
    package_manifest_path = tmp_path / "development-v2.zip.manifest.json"
    package_manifest_path.write_bytes(_json_bytes(package_manifest))
    sources = SourcePaths(
        archive=archive_path,
        checksum=checksum_path,
        package_manifest=package_manifest_path,
        plan=plan_path,
    )
    expected = ExpectedSources(
        zip_sha256=sha256_path(archive_path),
        package_manifest_sha256=sha256_path(package_manifest_path),
        checksum_file_sha256=sha256_path(checksum_path),
        plan_sha256=sha256_path(plan_path),
        plan_id=str(plan["plan_id"]),
        project_revision=str(manifest["project_revision"]),
    )
    return sources, expected


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _refresh_sources(
    sources: SourcePaths,
    expected: ExpectedSources,
    members: dict[str, bytes],
) -> ExpectedSources:
    sources.archive.write_bytes(_zip_bytes(members))
    archive_digest = sha256_path(sources.archive)
    sources.checksum.write_text(
        f"{archive_digest}  {sources.archive.name}\n", encoding="utf-8"
    )
    package_manifest = json.loads(sources.package_manifest.read_text(encoding="utf-8"))
    package_manifest["archive"].update(
        {
            "sha256": archive_digest,
            "size_bytes": sources.archive.stat().st_size,
            "files": len(members),
        }
    )
    sources.package_manifest.write_bytes(_json_bytes(package_manifest))
    return replace(
        expected,
        zip_sha256=archive_digest,
        package_manifest_sha256=sha256_path(sources.package_manifest),
        checksum_file_sha256=sha256_path(sources.checksum),
    )


def test_synthetic_archive_validates_without_opening_summary(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    study = validate_study_archive(sources, expected)
    assert study.integrity["summary_content_opened"] is False
    assert study.integrity["records"] == 64
    assert study.integrity["topologies"] == 16
    assert len(study.history_rows) == 64


def test_independent_reference_replay_collapses_to_topology_blocks(
    tmp_path: Path,
) -> None:
    sources, expected = _fixture_files(tmp_path)
    study = validate_study_archive(sources, expected)
    result = reference_replay(study)
    assert result["completed_runs"] == 64
    assert result["complete_optimizer_seed_pairs"] == 32
    assert result["complete_topologies"] == 16
    assert len(result["topology_rows"]) == 16
    assert result["topology_bootstrap_mean_difference_ci_95"]["seed"] == 20260819
    assert result["topology_bootstrap_mean_difference_ci_95"]["resamples"] == 10_000
    source = (
        ROOT / "experiments/uifo_paired/reference_analysis.py"
    ).read_text(encoding="utf-8")
    assert "from experiments.uifo_paired.analysis import" not in source


def test_production_reference_comparison_fails_closed_on_discrepancy(
    tmp_path: Path,
) -> None:
    sources, expected = _fixture_files(tmp_path)
    study = validate_study_archive(sources, expected)
    production = summarize_records(study.records, study.configs)
    reference = reference_replay(study)
    agreement = compare_production_and_reference(production, reference)
    assert agreement["status"] == "matched"
    with pytest.raises(StudyValidationError, match="summary remains locked"):
        load_summary_after_reproduction(study, {"status": "not-matched"})
    assert load_summary_after_reproduction(study, agreement) == {
        "outcome": "deliberately unopened"
    }
    broken = copy.deepcopy(reference)
    broken["topology_mean_difference"] += 1e-3
    with pytest.raises(StudyValidationError, match="three-way numerical mismatch"):
        compare_production_and_reference(production, broken)


def test_exact_exploratory_tests_are_separate_and_fully_enumerated() -> None:
    sign_flip = exact_mean_sign_flip_test([1.0, -1.0, 2.0, -2.0])
    assert sign_flip["assignments_enumerated"] == 16
    assert sign_flip["two_sided_p_value"] == 1.0
    sign = exact_two_sided_sign_test(7, 9, 0)
    assert sign["two_sided_p_value"] == pytest.approx(0.803619384765625)
    assert sign["estimand"].startswith("direction only")


def test_external_checksum_failure_fails_before_zip_parsing(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    bad = replace(expected, zip_sha256="0" * 64)
    with pytest.raises(StudyValidationError, match="external archive SHA-256 mismatch"):
        validate_study_archive(sources, bad)


def test_missing_history_is_rejected(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    members = _read_members(sources.archive)
    missing = next(name for name in members if name.startswith("histories/"))
    del members[missing]
    expected = _refresh_sources(sources, expected, members)
    with pytest.raises(StudyValidationError, match="archive member set mismatch"):
        validate_study_archive(sources, expected)


def test_log_checksum_failure_is_rejected(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    members = _read_members(sources.archive)
    log = next(name for name in members if name.endswith(".stdout.log") and name.startswith("logs/"))
    members[log] += b"corruption"
    expected = _refresh_sources(sources, expected, members)
    with pytest.raises(StudyValidationError, match="worker stdout evidence mismatch"):
        validate_study_archive(sources, expected)


def test_duplicate_record_id_is_rejected(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    members = _read_members(sources.archive)
    record_names = sorted(name for name in members if name.startswith("runs/") and name.endswith(".json"))
    first = json.loads(members[record_names[0]])
    second = json.loads(members[record_names[1]])
    second["run_id"] = first["run_id"]
    members[record_names[1]] = _json_bytes(second)
    expected = _refresh_sources(sources, expected, members)
    with pytest.raises(StudyValidationError, match="duplicate run record ID"):
        validate_study_archive(sources, expected)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    sources, expected = _fixture_files(tmp_path)
    members = _read_members(sources.archive)
    members["session.json"] = b'{"status":'
    expected = _refresh_sources(sources, expected, members)
    with pytest.raises(StudyValidationError, match="malformed JSON in session.json"):
        validate_study_archive(sources, expected)


def test_broken_pairing_in_plan_is_rejected() -> None:
    plan = _development_plan()
    changed = copy.deepcopy(plan)
    changed["runs"][0]["pair_id"] = "broken-pair"
    core = {
        "configuration": changed["configuration"],
        "run_order_policy": changed["run_order_policy"],
        "primary_pair_order": changed["primary_pair_order"],
        "runs": changed["runs"],
    }
    changed["plan_id"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    expected = ExpectedSources("0" * 64, "0" * 64, "0" * 64, "0" * 64, changed["plan_id"], "revision")
    with pytest.raises(StudyValidationError, match="pair order|32 topology-seed pairs"):
        _validate_plan_contract(changed, expected)


def test_incorrect_topology_hierarchy_is_rejected() -> None:
    plan = _development_plan()
    changed = copy.deepcopy(plan)
    changed["configuration"]["topologies"][1] = changed["configuration"]["topologies"][0]
    core = {
        "configuration": changed["configuration"],
        "run_order_policy": changed["run_order_policy"],
        "primary_pair_order": changed["primary_pair_order"],
        "runs": changed["runs"],
    }
    changed["plan_id"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    expected = ExpectedSources("0" * 64, "0" * 64, "0" * 64, "0" * 64, changed["plan_id"], "revision")
    with pytest.raises(StudyValidationError, match="planned topology identities"):
        _validate_plan_contract(changed, expected)


def test_zip_duplicate_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("same.txt", b"one")
            archive.writestr("same.txt", b"two")
    with pytest.raises(StudyValidationError, match="duplicate member names"):
        inspect_zip_integrity(path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute"])
def test_zip_unsafe_paths_are_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"payload")
    with pytest.raises(StudyValidationError, match="unsafe ZIP member"):
        inspect_zip_integrity(path)


def test_backslash_member_name_is_rejected_before_path_normalization() -> None:
    with pytest.raises(StudyValidationError, match="unsafe ZIP member"):
        _validate_member_name("dir\\file")


def test_zip_symlink_like_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"target")
    with pytest.raises(StudyValidationError, match="symlink"):
        inspect_zip_integrity(path)


def test_zip_size_and_ratio_limits_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "large.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.txt", b"0" * 10_000)
    with pytest.raises(StudyValidationError, match="uncompressed-size limit"):
        inspect_zip_integrity(path, ArchiveLimits(max_entry_uncompressed_bytes=100))
    with pytest.raises(StudyValidationError, match="compression ratio"):
        inspect_zip_integrity(path, ArchiveLimits(max_compression_ratio=2.0))


def test_zip_crc_corruption_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "crc.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.bin", b"unique-crc-payload")
    raw = path.read_bytes()
    offset = raw.index(b"unique-crc-payload")
    path.write_bytes(raw[:offset] + b"X" + raw[offset + 1 :])
    with pytest.raises(StudyValidationError, match="invalid ZIP archive|CRC"):
        inspect_zip_integrity(path)


def test_npz_object_arrays_are_never_unpickled() -> None:
    buffer = io.BytesIO()
    arrays = {
        "call_index": np.zeros(1, dtype=np.int32),
        "candidate_index": np.zeros(1, dtype=np.int16),
        "eval_count_after_call": np.ones(1, dtype=np.int64),
        "time_seconds": np.ones(1, dtype=np.float64),
        "loss": np.ones(1, dtype=np.float64),
        "sensitivity_loss": np.ones(1, dtype=np.float64),
        "penalty": np.ones(1, dtype=np.float64),
        "is_feasible": np.ones(1, dtype=np.bool_),
        "initial_params_unbounded": np.asarray([object()], dtype=object),
    }
    np.savez(buffer, **arrays)
    with pytest.raises(StudyValidationError, match="pickle-free NPZ"):
        _load_history_arrays(buffer.getvalue(), "object-history.npz")
