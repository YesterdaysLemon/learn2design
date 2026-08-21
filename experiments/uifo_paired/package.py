"""Validate and package a completed UIFO study for off-machine recovery."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from experiments.uifo_paired.runner import (
    _rebuild_indexes,
    _recover_stale_study_lock,
    _run_config,
    _validate_cache_disabled_environment,
    _validate_cache_disabled_runtime,
    atomic_json,
    atomic_text,
    sha256,
)


def validate_complete_study(
    study_dir: Path,
    *,
    allow_incomplete: bool = False,
    recover_stale_lock: bool = False,
) -> dict[str, object]:
    study_dir = study_dir.resolve()
    lock_path = study_dir / ".study.lock"
    if lock_path.exists():
        if not recover_stale_lock:
            raise RuntimeError("cannot package a study while its writer lock exists")
        _recover_stale_study_lock(study_dir, lock_path)
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing study manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise RuntimeError("unsupported study manifest format")
    configuration = manifest.get("configuration")
    runs = manifest.get("runs")
    environment = manifest.get("environment")
    if not isinstance(configuration, dict) or not isinstance(runs, list):
        raise TypeError("study manifest is missing its plan")
    if not isinstance(environment, dict):
        raise TypeError("study manifest is missing its runtime environment")
    runtime_policy = manifest.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        raise TypeError("study manifest is missing its runtime policy")
    cache_policy = runtime_policy.get("jax_compilation_cache")
    if not isinstance(cache_policy, dict) or cache_policy.get("policy") != "disabled":
        raise RuntimeError("study manifest does not prove a cache-disabled run")
    effective_environment = cache_policy.get("effective_environment")
    if not isinstance(effective_environment, dict):
        raise TypeError("study manifest is missing its effective cache environment")
    _validate_cache_disabled_environment(effective_environment)
    _validate_cache_disabled_runtime(environment)
    expected_configs = {
        str(run["run_id"]): _run_config(run, configuration) for run in runs
    }
    _rebuild_indexes(study_dir, expected_configs, environment)

    completed = []
    incomplete = []
    for run_id in expected_configs:
        record_path = study_dir / "runs" / f"{run_id}.json"
        if not record_path.is_file():
            incomplete.append({"run_id": run_id, "status": "missing"})
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            incomplete.append(
                {"run_id": run_id, "status": str(record.get("status"))}
            )
            continue
        completed.append(run_id)
    if incomplete and not allow_incomplete:
        first = incomplete[0]
        if first["status"] == "missing":
            raise RuntimeError(f"study is incomplete; missing run {first['run_id']}")
        raise RuntimeError(
            f"study is incomplete; run {first['run_id']} has status "
            f"{first['status']!r}"
        )
    if not incomplete and len(completed) != len(expected_configs):
        raise RuntimeError("study completion count disagrees with its plan")
    return {
        "manifest": manifest,
        "planned_runs": len(expected_configs),
        "completed_runs": len(completed),
        "incomplete_runs": incomplete,
        "study_complete": not incomplete,
    }


def package_study(
    study_dir: Path,
    output_path: Path,
    *,
    allow_incomplete: bool = False,
    recover_stale_lock: bool = False,
) -> dict[str, object]:
    study_dir = study_dir.resolve()
    output_path = output_path.resolve()
    sidecars = (
        output_path,
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
        output_path.with_suffix(output_path.suffix + ".sha256"),
    )
    existing = [path for path in sidecars if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite package output: {existing[0]}")
    if study_dir == output_path or study_dir in output_path.parents:
        raise ValueError("study package must be written outside the study directory")
    validation = validate_complete_study(
        study_dir,
        allow_incomplete=allow_incomplete,
        recover_stale_lock=recover_stale_lock,
    )
    manifest = validation["manifest"]
    assert isinstance(manifest, dict)
    files = sorted(
        path
        for path in study_dir.rglob("*")
        if path.is_file() and path.name != ".study.lock"
    )
    if not files:
        raise RuntimeError("study contains no files to package")
    package_state = {
        "format_version": 1,
        "study_complete": validation["study_complete"],
        "planned_runs": validation["planned_runs"],
        "completed_runs": validation["completed_runs"],
        "incomplete_runs": validation["incomplete_runs"],
    }
    package_state_bytes = (
        json.dumps(package_state, indent=2, sort_keys=True) + "\n"
    ).encode()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary package path already exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for path in files:
                relative = path.relative_to(study_dir).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                if path.suffix == ".npz":
                    info.compress_type = zipfile.ZIP_STORED
                    archive.writestr(info, path.read_bytes())
                else:
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes(), compresslevel=6)
            state_info = zipfile.ZipInfo(
                "package-state.json", date_time=(1980, 1, 1, 0, 0, 0)
            )
            state_info.external_attr = 0o100644 << 16
            state_info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(state_info, package_state_bytes, compresslevel=6)
        with zipfile.ZipFile(temporary, "r") as archive:
            corrupted = archive.testzip()
        if corrupted is not None:
            raise RuntimeError(f"study package failed CRC validation: {corrupted}")
        os.replace(temporary, output_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    result = {
        "format_version": 1,
        "study_plan_id": manifest.get("plan_id"),
        "study_project_revision": manifest.get("project_revision"),
        "study_complete": validation["study_complete"],
        "planned_runs": validation["planned_runs"],
        "completed_runs": validation["completed_runs"],
        "incomplete_runs": validation["incomplete_runs"],
        "source_directory": str(study_dir),
        "archive": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "files": len(files) + 1,
        },
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), result)
    atomic_text(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{result['archive']['sha256']}  {output_path.name}\n",
    )
    return result
