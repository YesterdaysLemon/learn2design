from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration

import experiments.uifo_paired.coverage_results_ingestion as coverage_ingestion
from experiments.uifo_paired.coverage_analysis import summarize_coverage_records
from experiments.uifo_paired.coverage_evidence import (
    compare_coverage_archived_summary,
    compare_coverage_replays,
)
from experiments.uifo_paired.coverage_profiles import coverage_profile_spec
from experiments.uifo_paired.coverage_reference_analysis import (
    reference_coverage_screen,
)
from experiments.uifo_paired.coverage_results_ingestion import (
    authenticate_coverage_source_lock,
    load_coverage_summary_after_reproduction,
    validate_coverage_archive,
    validate_coverage_terminal_partial,
)
from experiments.uifo_paired.metrics import summarize_rows
from experiments.uifo_paired.results_ingestion import (
    ExpectedSources,
    SourcePaths,
    StudyValidationError,
    sha256_bytes,
    sha256_path,
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
from tests.test_coverage_robustness import _plan
from tests.test_coverage_triage import _triage_plan
from tools.create_submission_like_source_lock import build_source_lock
from tools.analyze_coverage_robustness import run_analysis


ROOT = Path(__file__).parents[1]
REVISION = "c" * 40
PROFILE = "coverage-robustness-screen-v1"


@dataclass(frozen=True)
class _Bundle:
    sources: SourcePaths
    receipt: Path
    source_lock: Path
    summary_release: Path | None = None
    provider_billing_receipt: Path | None = None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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
        "device_kinds": ["NVIDIA H100 80GB HBM3"],
        "device_platforms": ["gpu"],
        "devices": ["CudaDevice(id=0)"],
        "jax_platform_versions": ["cuda 13000"],
        "competition_aligned_a100": False,
        "competition_aligned_h100": True,
        "jax_runtime_environment": runtime_environment,
        "jax_runtime_configuration": {
            "compilation_cache_dir": None,
            "enable_compilation_cache": False,
        },
        "nvidia_smi": {"status": "ok", "gpus": []},
        "versions": {
            "dfbench": "0.3.3",
            "differometor": "0.3.5",
            "jax": "0.9.0.1",
            "jax-cuda12-pjrt": "not-installed",
            "jax-cuda12-plugin": "not-installed",
            "jax-cuda13-pjrt": "0.9.0.1",
            "jax-cuda13-plugin": "0.9.0.1",
            "nvidia-cuda-runtime": "13.0.96",
            "nvidia-cuda-runtime-cu12": "not-installed",
            "jaxlib": "0.9.0.1",
            "optax": "0.2.6",
        },
        "platform": "Linux-synthetic-test",
        "python": "3.12.11",
    }


def _raw_population(pair_index: int) -> np.ndarray:
    values = np.arange(16, dtype=np.float64).reshape(8, 2)
    return values / 10.0 + pair_index


def _initial_population(
    arm: str, raw: np.ndarray
) -> np.ndarray:
    anchor = np.zeros((1, raw.shape[1]), dtype=np.float64)
    if arm == "no_prior":
        return np.vstack([anchor, raw[1:]])
    suffix_raw = raw[1:]
    order = np.argsort(suffix_raw, axis=0, kind="stable")
    ranks = np.argsort(order, axis=0, kind="stable")
    unit = (ranks.astype(np.float64) + 0.5) / suffix_raw.shape[0]
    suffix = np.log(unit) - np.log1p(-unit)
    return np.vstack([anchor, suffix])


def _history_payload(
    loss: float, initial: np.ndarray
) -> tuple[bytes, list[dict[str, object]]]:
    buffer = io.BytesIO()
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


def _build_bundle(
    tmp_path: Path,
    *,
    plan_factory=_plan,
    profile: str = PROFILE,
    winning_topologies: int = 9,
) -> _Bundle:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = plan_factory()
    specification = coverage_profile_spec(profile)
    session_started = datetime(2026, 8, 24, tzinfo=UTC)
    synthetic_run_stride = timedelta(seconds=60)
    session_elapsed = float((specification.runs + 1) * 60)
    session_completed = session_started + timedelta(seconds=session_elapsed)
    assert plan["configuration"]["candidate_package_evidence"][
        "project_revision"
    ] == REVISION
    configuration = plan["configuration"]
    environment = _environment()
    receipt = tmp_path / "coverage.terminal-attempt.json"
    receipt.write_bytes(
        _json_bytes(
            {
                "format_version": 1,
                "study_profile": profile,
                "plan_id": plan["plan_id"],
                "project_revision": REVISION,
                "claimed_utc": "2026-08-24T00:00:00+00:00",
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
        "semantic_prior_canonical_sha256": "synthetic-coverage-study",
        "upstream_reference": (
            "1bb7f54737dec6a08b59879a8831d125f08f8a0b"
        ),
        "environment": environment,
        "rental_preflight": {
            "disk": {"path": "/workspace/results", "free_bytes": 30 * 1024**3},
            "gpu_idle": {
                "status": "ok",
                "gpus": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "mig_mode_current": "Disabled",
                        "memory_total_mib": 81_559,
                        "memory_used_mib": 12,
                        "utilization_percent": 0,
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
    members: dict[str, bytes] = {
        "manifest.json": _json_bytes(manifest),
        "package-state.json": _json_bytes(
            {
                "format_version": 1,
                "study_complete": True,
                "planned_runs": specification.runs,
                "completed_runs": specification.runs,
                "incomplete_runs": [],
            }
        ),
        "session.json": _json_bytes(
            {
                "status": "complete",
                "started_utc": session_started.isoformat(),
                "completed_utc": session_completed.isoformat(),
                "elapsed_seconds": session_elapsed,
                "max_session_wall_seconds": specification.session_wall_seconds,
            }
        ),
        "preflight.host-environment.json": _json_bytes(
            {
                "captured_utc": "2026-08-24T00:00:00+00:00",
                "inherited_cache_environment": {},
                "inherited_environment": {},
                "effective_environment": environment[
                    "jax_runtime_environment"
                ],
            }
        ),
        "preflight.json": _json_bytes(environment),
        "preflight.stderr.log": b"",
        "preflight.stdout.log": b"synthetic H100 preflight passed\n",
    }
    records = []
    configs = {}
    topology_order = {
        json.dumps(topology, sort_keys=True): index
        for index, topology in enumerate(configuration["topologies"])
    }
    pair_indices: dict[str, int] = {}
    for run_index, run in enumerate(plan["runs"]):
        config = _run_config(run, configuration)
        run_id = str(config["run_id"])
        pair_id = str(config["pair_id"])
        pair_index = pair_indices.setdefault(pair_id, len(pair_indices))
        raw = _raw_population(pair_index)
        initial = _initial_population(str(config["arm"]), raw)
        topology_index = topology_order[json.dumps(config["topology"], sort_keys=True)]
        difference = -0.10 if topology_index < winning_topologies else 0.02
        control_loss = 1.0 + topology_index / 100
        loss = control_loss if config["arm"] == "no_prior" else control_loss + difference
        history_payload, rows = _history_payload(loss, initial)
        time_grid, eval_grid = _metric_grids(config)
        metrics = summarize_rows(
            rows,
            list(config["target_losses"]),
            time_grid=time_grid,
            eval_grid=eval_grid,
        )
        stdout = f"synthetic completion {run_id}\n".encode()
        stderr = b""
        topology = str(config["topology"]["value"])
        run_started = session_started + run_index * synthetic_run_stride
        run_completed = run_started + timedelta(seconds=50)
        record = {
            "format_version": 1,
            "run_id": run_id,
            "status": "complete",
            "started_utc": run_started.isoformat(),
            "completed_utc": run_completed.isoformat(),
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
            "initial_population_roles": _initial_population_roles(
                str(config["arm"]), 8
            ),
            "initial_parameter_hashes": _parameter_hashes(initial),
            "raw_suffix_parameter_hashes": _parameter_hashes(raw[1:]),
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
                "started_utc": run_started.isoformat(),
                "completed_utc": run_completed.isoformat(),
                "full_wall_seconds": 45.0,
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
        configs[run_id] = config
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
    summary = summarize_coverage_records(records, configs)
    summary_bytes = _json_bytes(summary)
    summary_release = None
    if profile == "coverage-triage-screen-v1":
        members["summary.commitment.json"] = _json_bytes(
            {
                "format_version": 1,
                "study_profile": profile,
                "summary_sha256": sha256_bytes(summary_bytes),
                "size_bytes": len(summary_bytes),
            }
        )
        summary_release = tmp_path / "coverage-screen.zip.summary.json"
        summary_release.write_bytes(summary_bytes)
    else:
        members["summary.json"] = summary_bytes
    assert len(members) == specification.archive_members

    archive = tmp_path / "coverage-screen.zip"
    archive.write_bytes(_zip_bytes(members))
    plan_path = tmp_path / "coverage-plan.json"
    plan_path.write_bytes(_json_bytes(plan))
    checksum = tmp_path / "coverage-screen.zip.sha256"
    checksum.write_text(f"{sha256_path(archive)}  {archive.name}\n", encoding="utf-8")
    package_manifest = tmp_path / "coverage-screen.zip.manifest.json"
    package_manifest.write_bytes(
        _json_bytes(
            {
                "format_version": 1,
                "study_plan_id": plan["plan_id"],
                "study_project_revision": REVISION,
                "study_complete": True,
                "planned_runs": specification.runs,
                "completed_runs": specification.runs,
                "incomplete_runs": [],
                "archive": {
                    "path": "/workspace/results/coverage-screen.zip",
                    "sha256": sha256_path(archive),
                    "size_bytes": archive.stat().st_size,
                    "files": len(members),
                },
                **(
                    {
                        "summary_release": {
                            "path": summary_release.name,
                            "sha256": sha256_path(summary_release),
                            "size_bytes": summary_release.stat().st_size,
                            "archive_member": "summary.commitment.json",
                        }
                    }
                    if summary_release is not None
                    else {}
                ),
            }
        )
    )
    sources = SourcePaths(archive, checksum, package_manifest, plan_path)
    provider_billing_receipt = None
    if profile == "coverage-triage-screen-v1":
        provider_billing_receipt = tmp_path / "runpod-billing-receipt.json"
        provider_billing_receipt.write_bytes(
            _json_bytes(
                {
                    "format_version": 1,
                    "study_profile": profile,
                    "plan_id": plan["plan_id"],
                    "provider": "Runpod",
                    "gpu_type_id": "NVIDIA H100 80GB HBM3",
                    "cloud_type": "SECURE",
                    "gpu_count": 1,
                    "maximum_gpu_hourly_price": 3.29,
                    "provider_hours": 6.5,
                    "gpu_charge": 21.385,
                    "total_provider_charge": 21.385,
                    "resources_deleted": True,
                    "captured_utc": "2026-08-24T07:00:00Z",
                }
            )
        )
    source_lock = tmp_path / "coverage-source-lock.json"
    source_lock_payload = build_source_lock(
        archive=archive,
        checksum=checksum,
        package_manifest=package_manifest,
        plan=plan_path,
        terminal_attempt_receipt=receipt,
        study_profile=profile,
    )
    if provider_billing_receipt is not None:
        source_lock_payload["files"][provider_billing_receipt.name] = {
            "sha256": sha256_path(provider_billing_receipt),
            "size_bytes": provider_billing_receipt.stat().st_size,
        }
    source_lock.write_bytes(_json_bytes(source_lock_payload))
    return _Bundle(
        sources,
        receipt,
        source_lock,
        summary_release,
        provider_billing_receipt,
    )


def _authenticate(bundle: _Bundle) -> ExpectedSources:
    return authenticate_coverage_source_lock(
        bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        sources=bundle.sources,
        terminal_attempt_receipt=bundle.receipt,
        provider_billing_receipt=bundle.provider_billing_receipt,
    )


def _refresh_bundle(bundle: _Bundle, members: dict[str, bytes]) -> ExpectedSources:
    bundle.sources.archive.write_bytes(_zip_bytes(members))
    archive_digest = sha256_path(bundle.sources.archive)
    bundle.sources.checksum.write_text(
        f"{archive_digest}  {bundle.sources.archive.name}\n", encoding="utf-8"
    )
    package = json.loads(bundle.sources.package_manifest.read_text(encoding="utf-8"))
    package["archive"].update(
        {
            "sha256": archive_digest,
            "size_bytes": bundle.sources.archive.stat().st_size,
            "files": len(members),
        }
    )
    bundle.sources.package_manifest.write_bytes(_json_bytes(package))
    bundle.source_lock.write_bytes(
        _json_bytes(
            build_source_lock(
                archive=bundle.sources.archive,
                checksum=bundle.sources.checksum,
                package_manifest=bundle.sources.package_manifest,
                plan=bundle.sources.plan,
                terminal_attempt_receipt=bundle.receipt,
                study_profile=PROFILE,
            )
        )
    )
    return _authenticate(bundle)


def _rewrite_runs_jsonl(members: dict[str, bytes]) -> None:
    run_names = sorted(name for name in members if name.startswith("runs/"))
    records = [json.loads(members[name]) for name in run_names]
    records.sort(key=lambda record: str(record["run_id"]))
    members["runs.jsonl"] = b"".join(
        json.dumps(strict_json(record), sort_keys=True, allow_nan=False).encode()
        + b"\n"
        for record in records
    )


def _rewrite_history(
    members: dict[str, bytes],
    run_id: str,
    mutate,
) -> None:
    history_name = f"histories/{run_id}.npz"
    with np.load(io.BytesIO(members[history_name]), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in HISTORY_SCHEMA}
    mutate(arrays)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    history_payload = buffer.getvalue()
    members[history_name] = history_payload

    rows = _rows_from_history_arrays(arrays)
    record_name = f"runs/{run_id}.json"
    record = json.loads(members[record_name])
    config = record["config"]
    time_grid, eval_grid = _metric_grids(config)
    record["metrics"] = summarize_rows(
        rows,
        list(config["target_losses"]),
        time_grid=time_grid,
        eval_grid=eval_grid,
    )
    record["objective_accounting"] = {
        "log_call_count": record["metrics"]["logged_calls"],
        "eval_count": max(int(row["eval_count_after_call"]) for row in rows),
    }
    record["initial_parameter_hashes"] = _parameter_hashes(
        arrays["initial_params_unbounded"]
    )
    record["history"].update(
        {
            "rows": len(rows),
            "sha256": sha256_bytes(history_payload),
        }
    )
    members[record_name] = _json_bytes(record)
    _rewrite_runs_jsonl(members)


def test_complete_169_member_triage_archive_uses_detached_summary_commitment(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(
        tmp_path,
        plan_factory=_triage_plan,
        profile="coverage-triage-screen-v1",
        winning_topologies=7,
    )
    assert bundle.summary_release is not None
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        assert len(archive.namelist()) == 169
        assert "summary.json" not in archive.namelist()
        assert "summary.commitment.json" in archive.namelist()

    expected = _authenticate(bundle)
    assert expected.study_profile == "coverage-triage-screen-v1"
    study = validate_coverage_archive(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    assert study.integrity["records"] == 32
    assert study.integrity["topologies"] == 8
    assert study.integrity["optimizer_seed_pairs"] == 16
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)
    replay = compare_coverage_replays(production, reference, study=study)
    with pytest.raises(StudyValidationError, match="release is missing"):
        load_coverage_summary_after_reproduction(study, replay)
    archived = load_coverage_summary_after_reproduction(
        study,
        replay,
        summary_release_path=bundle.summary_release,
    )
    assert compare_coverage_archived_summary(
        production, reference, archived
    )["status"] == "matched"


def test_triage_source_lock_cannot_be_relabelled_as_v1(tmp_path: Path) -> None:
    bundle = _build_bundle(
        tmp_path,
        plan_factory=_triage_plan,
        profile="coverage-triage-screen-v1",
        winning_topologies=7,
    )
    lock = json.loads(bundle.source_lock.read_text(encoding="utf-8"))
    lock["study_profile"] = PROFILE
    assert bundle.provider_billing_receipt is not None
    lock["files"].pop(bundle.provider_billing_receipt.name)
    bundle.source_lock.write_bytes(_json_bytes(lock))
    expected = authenticate_coverage_source_lock(
        bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        sources=bundle.sources,
        terminal_attempt_receipt=bundle.receipt,
    )
    with pytest.raises(
        StudyValidationError, match="schema mismatch|study profile mismatch"
    ):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_triage_billing_receipt_enforces_the_provider_cap(tmp_path: Path) -> None:
    bundle = _build_bundle(
        tmp_path,
        plan_factory=_triage_plan,
        profile="coverage-triage-screen-v1",
        winning_topologies=7,
    )
    assert bundle.provider_billing_receipt is not None
    billing = json.loads(
        bundle.provider_billing_receipt.read_text(encoding="utf-8")
    )
    billing["total_provider_charge"] = 30.01
    bundle.provider_billing_receipt.write_bytes(_json_bytes(billing))
    lock = json.loads(bundle.source_lock.read_text(encoding="utf-8"))
    lock["files"][bundle.provider_billing_receipt.name] = {
        "sha256": sha256_path(bundle.provider_billing_receipt),
        "size_bytes": bundle.provider_billing_receipt.stat().st_size,
    }
    bundle.source_lock.write_bytes(_json_bytes(lock))
    with pytest.raises(StudyValidationError, match="billing receipt is invalid"):
        authenticate_coverage_source_lock(
            bundle.source_lock,
            expected_source_lock_sha256=sha256_path(bundle.source_lock),
            sources=bundle.sources,
            terminal_attempt_receipt=bundle.receipt,
            provider_billing_receipt=bundle.provider_billing_receipt,
        )


def test_triage_analysis_cli_path_replays_then_releases_summary(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(
        tmp_path / "bundle",
        plan_factory=_triage_plan,
        profile="coverage-triage-screen-v1",
        winning_topologies=7,
    )
    assert bundle.summary_release is not None
    assert bundle.provider_billing_receipt is not None
    output = tmp_path / "analysis"
    receipt = run_analysis(
        archive=bundle.sources.archive,
        checksum=bundle.sources.checksum,
        package_manifest=bundle.sources.package_manifest,
        plan=bundle.sources.plan,
        terminal_attempt_receipt=bundle.receipt,
        source_lock=bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        output=output,
        summary_release=bundle.summary_release,
        provider_billing_receipt=bundle.provider_billing_receipt,
    )
    assert receipt["status"] == "validated"
    assert receipt["study_profile"] == "coverage-triage-screen-v1"
    assert receipt["replay_agreement"]["runs_compared"] == 32
    assert receipt["predeclared_decision"]["action"] == (
        "review_precommitted_stage_b_design_and_seek_owner_approval"
    )
    assert (output / "validation_receipt.json").is_file()


def test_complete_249_member_coverage_archive_replays_with_summary_sealed(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    expected = _authenticate(bundle)

    study = validate_coverage_archive(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)
    agreement = compare_coverage_replays(production, reference, study=study)

    assert len(study.archive_members) == 249
    assert study.integrity["coverage_initial_arrays"] == 24
    assert study.integrity["pretransform_draw_pairs"] == 24
    assert study.integrity["summary_content_opened"] is False
    assert agreement.as_dict()["status"] == "matched"
    archived = load_coverage_summary_after_reproduction(study, agreement)
    assert archived == production

    with pytest.raises(StudyValidationError, match="comparator-issued"):
        load_coverage_summary_after_reproduction(
            study,
            {
                "status": "matched",
                "runs_compared": 48,
                "topology_values_compared": 12,
                "optimizer_seed_pairs_compared": 24,
                "frozen_criteria_compared": 13,
                "study_identity_sha256": "f" * 64,
            },
        )


def test_sealed_coverage_cli_runs_the_full_replay_before_opening_summary(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    output = tmp_path / "private-analysis"

    receipt = run_analysis(
        archive=bundle.sources.archive,
        checksum=bundle.sources.checksum,
        package_manifest=bundle.sources.package_manifest,
        plan=bundle.sources.plan,
        terminal_attempt_receipt=bundle.receipt,
        source_lock=bundle.source_lock,
        expected_source_lock_sha256=sha256_path(bundle.source_lock),
        output=output,
    )

    assert receipt["status"] == "validated"
    assert receipt["replay_agreement"]["status"] == "matched"
    assert receipt["archived_summary_agreement"]["status"] == "matched"
    assert receipt["summary_content_opened_after_replay"] is True
    assert sorted(path.name for path in output.iterdir()) == [
        "archived_summary.json",
        "independent_reference.json",
        "production_replay.json",
        "validation_receipt.json",
    ]


def test_source_lock_hash_is_checked_before_malformed_json_is_parsed(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    bundle.source_lock.write_text("not json", encoding="utf-8")

    with pytest.raises(StudyValidationError, match="SHA-256 mismatch"):
        authenticate_coverage_source_lock(
            bundle.source_lock,
            expected_source_lock_sha256="d" * 64,
            sources=bundle.sources,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_terminal_partial_is_structural_and_not_evaluable(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    plan = json.loads(bundle.sources.plan.read_text(encoding="utf-8"))
    run_id = str(plan["runs"][-1]["run_id"])
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    for name in (
        f"configs/{run_id}.json",
        f"histories/{run_id}.npz",
        f"logs/{run_id}.stdout.log",
        f"logs/{run_id}.stderr.log",
        f"runs/{run_id}.json",
    ):
        members.pop(name)
    incomplete = [{"run_id": run_id, "status": "error"}]
    members["package-state.json"] = _json_bytes(
        {
            "format_version": 1,
            "study_complete": False,
            "planned_runs": 48,
            "completed_runs": 47,
            "incomplete_runs": incomplete,
        }
    )
    members["session.json"] = _json_bytes(
        {
            "status": "error",
            "started_utc": "2026-08-24T00:00:00+00:00",
            "completed_utc": "2026-08-24T19:00:00+00:00",
            "elapsed_seconds": 68_400.0,
            "max_session_wall_seconds": 79_200.0,
        }
    )
    package = json.loads(bundle.sources.package_manifest.read_text(encoding="utf-8"))
    package.update(
        {
            "study_complete": False,
            "completed_runs": 47,
            "incomplete_runs": incomplete,
        }
    )
    bundle.sources.package_manifest.write_bytes(_json_bytes(package))
    expected = _refresh_bundle(bundle, members)

    result = validate_coverage_terminal_partial(
        bundle.sources,
        expected=expected,
        terminal_attempt_receipt=bundle.receipt,
    )

    assert result["status"] == "not_evaluable"
    assert result["completed_runs"] == 47
    assert result["summary_content_opened"] is False
    assert result["run_records_opened"] is False
    assert result["histories_opened"] is False


def test_coverage_archive_rejects_wrong_cuda_stack_and_raw_draw_drift(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(members["manifest.json"])
    manifest["environment"]["jax_platform_versions"] = ["cuda 12090"]
    members["manifest.json"] = _json_bytes(manifest)
    expected = _refresh_bundle(bundle, members)

    with pytest.raises(StudyValidationError, match="JAX backend is not CUDA 13"):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )

    raw_bundle = _build_bundle(tmp_path / "raw-drift")
    with zipfile.ZipFile(raw_bundle.sources.archive, "r") as archive:
        raw_members = {
            info.filename: archive.read(info) for info in archive.infolist()
        }
    run_names = sorted(name for name in raw_members if name.startswith("runs/"))
    treatment_name = next(
        name for name in run_names if name.endswith("__coverage_balanced.json")
    )
    treatment = json.loads(raw_members[treatment_name])
    treatment["raw_suffix_parameter_hashes"][0] = "f" * 64
    raw_members[treatment_name] = _json_bytes(treatment)
    records = [json.loads(raw_members[name]) for name in run_names]
    records.sort(key=lambda record: str(record["run_id"]))
    raw_members["runs.jsonl"] = b"".join(
        json.dumps(strict_json(record), sort_keys=True, allow_nan=False).encode()
        + b"\n"
        for record in records
    )
    raw_expected = _refresh_bundle(raw_bundle, raw_members)
    with pytest.raises(StudyValidationError, match="paired pre-transform random draw"):
        validate_coverage_archive(
            raw_bundle.sources,
            expected=raw_expected,
            terminal_attempt_receipt=raw_bundle.receipt,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eval_count_after_call", 100_000, "evaluation accounting is not row-exact"),
        ("time_seconds", 99_999.0, "chronology exceeds its budget"),
    ],
)
def test_coverage_archive_rejects_forged_objective_chronology(
    tmp_path: Path,
    field: str,
    value: int | float,
    message: str,
) -> None:
    bundle = _build_bundle(tmp_path)
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    run_id = str(json.loads(bundle.sources.plan.read_text())["runs"][0]["run_id"])

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays[field].fill(value)

    _rewrite_history(members, run_id, mutate)
    expected = _refresh_bundle(bundle, members)

    with pytest.raises(StudyValidationError, match=message):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_coverage_archive_rejects_unrelated_valid_lhs_permutation(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    plan = json.loads(bundle.sources.plan.read_text())
    run_id = str(
        next(
            run["run_id"]
            for run in plan["runs"]
            if run["arm"] == "coverage_balanced"
        )
    )

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["initial_params_unbounded"][1:, 1] = np.roll(
            arrays["initial_params_unbounded"][1:, 1], 2
        )

    _rewrite_history(members, run_id, mutate)
    expected = _refresh_bundle(bundle, members)

    with pytest.raises(StudyValidationError, match="not the ranked raw draw"):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_coverage_archive_rejects_worker_and_session_budget_overruns(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path / "worker")
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    run_name = next(name for name in sorted(members) if name.startswith("runs/"))
    record = json.loads(members[run_name])
    record["worker_process"]["full_wall_seconds"] = 2_101.0
    members[run_name] = _json_bytes(record)
    _rewrite_runs_jsonl(members)
    expected = _refresh_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="exceeded its timeout"):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )

    session_bundle = _build_bundle(tmp_path / "session")
    with zipfile.ZipFile(session_bundle.sources.archive, "r") as archive:
        session_members = {
            info.filename: archive.read(info) for info in archive.infolist()
        }
    session = json.loads(session_members["session.json"])
    session["elapsed_seconds"] = 79_201.0
    session_members["session.json"] = _json_bytes(session)
    session_expected = _refresh_bundle(session_bundle, session_members)
    with pytest.raises(StudyValidationError, match="session completion evidence"):
        validate_coverage_archive(
            session_bundle.sources,
            expected=session_expected,
            terminal_attempt_receipt=session_bundle.receipt,
        )


def test_coverage_archive_cross_checks_preflight_environment(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    with zipfile.ZipFile(bundle.sources.archive, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    host = json.loads(members["preflight.host-environment.json"])
    host["effective_environment"]["CUDA_CACHE_DISABLE"] = "0"
    members["preflight.host-environment.json"] = _json_bytes(host)
    expected = _refresh_bundle(bundle, members)
    with pytest.raises(StudyValidationError, match="cache environments disagree"):
        validate_coverage_archive(
            bundle.sources,
            expected=expected,
            terminal_attempt_receipt=bundle.receipt,
        )


def test_coverage_archive_rejects_noncanonical_logits_and_interleaved_rows(
    tmp_path: Path,
) -> None:
    logit_bundle = _build_bundle(tmp_path / "logit")
    with zipfile.ZipFile(logit_bundle.sources.archive, "r") as archive:
        logit_members = {
            info.filename: archive.read(info) for info in archive.infolist()
        }
    plan = json.loads(logit_bundle.sources.plan.read_text())
    treatment_id = str(
        next(
            run["run_id"]
            for run in plan["runs"]
            if run["arm"] == "coverage_balanced"
        )
    )

    def perturb_logit(arrays: dict[str, np.ndarray]) -> None:
        arrays["initial_params_unbounded"][1, 0] += 2e-6

    _rewrite_history(logit_members, treatment_id, perturb_logit)
    logit_expected = _refresh_bundle(logit_bundle, logit_members)
    with pytest.raises(StudyValidationError, match="midpoint logits drifted"):
        validate_coverage_archive(
            logit_bundle.sources,
            expected=logit_expected,
            terminal_attempt_receipt=logit_bundle.receipt,
        )

    order_bundle = _build_bundle(tmp_path / "row-order")
    with zipfile.ZipFile(order_bundle.sources.archive, "r") as archive:
        order_members = {
            info.filename: archive.read(info) for info in archive.infolist()
        }
    run_id = str(
        json.loads(order_bundle.sources.plan.read_text())["runs"][0]["run_id"]
    )

    def interleave_calls(arrays: dict[str, np.ndarray]) -> None:
        row_fields = [
            name for name in HISTORY_SCHEMA if name != "initial_params_unbounded"
        ]
        expanded = {
            name: np.concatenate([arrays[name], arrays[name]])
            for name in row_fields
        }
        expanded["call_index"][8:] = 1
        expanded["eval_count_after_call"][8:] = 16
        expanded["time_seconds"][8:] = 2.0
        order = np.asarray(
            [
                index
                for pair in zip(range(8), range(8, 16), strict=True)
                for index in pair
            ]
        )
        for name in row_fields:
            arrays[name] = expanded[name][order]

    _rewrite_history(order_members, run_id, interleave_calls)
    order_expected = _refresh_bundle(order_bundle, order_members)
    with pytest.raises(StudyValidationError, match="row chronology is interleaved"):
        validate_coverage_archive(
            order_bundle.sources,
            expected=order_expected,
            terminal_attempt_receipt=order_bundle.receipt,
        )


def test_coverage_archive_requires_ordered_utc_timestamps_and_exact_panel_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_bundle = _build_bundle(tmp_path / "timestamps")
    with zipfile.ZipFile(time_bundle.sources.archive, "r") as archive:
        time_members = {
            info.filename: archive.read(info) for info in archive.infolist()
        }
    run_name = next(
        name for name in sorted(time_members) if name.startswith("runs/")
    )
    record = json.loads(time_members[run_name])
    record["completed_utc"] = "2026-08-23T23:59:59+00:00"
    time_members[run_name] = _json_bytes(record)
    _rewrite_runs_jsonl(time_members)
    time_expected = _refresh_bundle(time_bundle, time_members)
    with pytest.raises(StudyValidationError, match="timestamps are reversed"):
        validate_coverage_archive(
            time_bundle.sources,
            expected=time_expected,
            terminal_attempt_receipt=time_bundle.receipt,
        )

    panel_bundle = _build_bundle(tmp_path / "panel")
    panel = json.loads(coverage_ingestion.PANEL_PATH.read_text(encoding="utf-8"))
    panel["hostile_metadata_drift"] = True
    drifted_panel = tmp_path / "drifted-coverage-panel.json"
    drifted_panel.write_bytes(_json_bytes(panel))
    monkeypatch.setattr(coverage_ingestion, "PANEL_PATH", drifted_panel)
    with pytest.raises(StudyValidationError, match="exact committed panel bytes"):
        validate_coverage_archive(
            panel_bundle.sources,
            expected=_authenticate(panel_bundle),
            terminal_attempt_receipt=panel_bundle.receipt,
        )


def test_committed_coverage_source_lock_schema_is_valid_json() -> None:
    path = (
        ROOT
        / "experiments"
        / "uifo_paired"
        / "schemas"
        / "coverage-source-lock.schema.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["properties"]["study_profile"]["const"] == PROFILE
