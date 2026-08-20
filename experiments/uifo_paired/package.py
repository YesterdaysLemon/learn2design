"""Validate and package a completed UIFO study for off-machine recovery."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from experiments.uifo_paired.runner import (
    _rebuild_indexes,
    _run_config,
    _validate_cache_disabled_environment,
    _validate_cache_disabled_runtime,
    atomic_json,
    atomic_text,
    sha256,
)


def validate_complete_study(study_dir: Path) -> dict[str, object]:
    study_dir = study_dir.resolve()
    if (study_dir / ".study.lock").exists():
        raise RuntimeError("cannot package a study while its writer lock exists")
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
    for run_id in expected_configs:
        record_path = study_dir / "runs" / f"{run_id}.json"
        if not record_path.is_file():
            raise RuntimeError(f"study is incomplete; missing run {run_id}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            raise RuntimeError(
                f"study is incomplete; run {run_id} has status {record.get('status')!r}"
            )
        completed.append(run_id)
    if len(completed) != len(expected_configs):
        raise RuntimeError("study completion count disagrees with its plan")
    return manifest


def package_study(study_dir: Path, output_path: Path) -> dict[str, object]:
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
    manifest = validate_complete_study(study_dir)
    files = sorted(
        path
        for path in study_dir.rglob("*")
        if path.is_file() and path.name != ".study.lock"
    )
    if not files:
        raise RuntimeError("study contains no files to package")

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
        "source_directory": str(study_dir),
        "archive": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "files": len(files),
        },
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), result)
    atomic_text(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{result['archive']['sha256']}  {output_path.name}\n",
    )
    return result
