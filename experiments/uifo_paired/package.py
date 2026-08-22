"""Validate and package a completed UIFO study for off-machine recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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


_BASE_STUDY_MEMBERS = {
    "manifest.json",
    "preflight.host-environment.json",
    "preflight.json",
    "preflight.stderr.log",
    "preflight.stdout.log",
    "runs.jsonl",
    "session.json",
    "summary.json",
}
_RECOVERY_RECEIPT = re.compile(r"^recovery/stale-study-lock-([0-9a-f]{12})\.json$")


def _inside_git_checkout(path: Path) -> bool:
    current = path.resolve()
    if not current.is_dir():
        current = current.parent
    return any((parent / ".git").exists() for parent in (current, *current.parents))


def _collect_regular_files(study_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for root, directories, names in os.walk(study_dir, followlinks=False):
        root_path = Path(root)
        for name in directories:
            candidate = root_path / name
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise RuntimeError(f"study contains a symlink directory: {candidate}")
        for name in names:
            candidate = root_path / name
            if name == ".study.lock":
                continue
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"study contains a symlink or special file: {candidate}")
            relative = candidate.relative_to(study_dir).as_posix()
            files[relative] = candidate
    return files


def _expected_complete_members(
    expected_configs: dict[str, dict[str, object]],
) -> set[str]:
    members = set(_BASE_STUDY_MEMBERS)
    for run_id, config in expected_configs.items():
        members.update(
            {
                f"configs/{run_id}.json",
                f"histories/{run_id}.npz",
                f"logs/{run_id}.stdout.log",
                f"logs/{run_id}.stderr.log",
                f"runs/{run_id}.json",
            }
        )
        if config.get("optimizer_telemetry") == "member-v1":
            members.add(f"optimizer-telemetry/{run_id}.npz")
    return members


def _validated_recovery_receipts(files_by_name: dict[str, Path]) -> set[str]:
    receipts: set[str] = set()
    for name, path in files_by_name.items():
        if not name.startswith("recovery/"):
            continue
        match = _RECOVERY_RECEIPT.fullmatch(name)
        if match is None:
            raise RuntimeError(f"study contains unexpected recovery artifact: {name}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest()[:12] != match.group(1):
            raise RuntimeError(f"stale-lock recovery receipt digest mismatch: {name}")
        try:
            record = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"stale-lock recovery receipt is malformed: {name}") from error
        if (
            not isinstance(record, dict)
            or set(record) != {"pid", "hostname", "created_utc"}
            or isinstance(record["pid"], bool)
            or not isinstance(record["pid"], int)
            or record["pid"] <= 0
            or not isinstance(record["hostname"], str)
            or not record["hostname"]
            or not isinstance(record["created_utc"], str)
            or not record["created_utc"]
        ):
            raise RuntimeError(f"stale-lock recovery receipt schema mismatch: {name}")
        receipts.add(name)
    return receipts


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
    if not incomplete and configuration.get("study_profile") is not None:
        session_path = study_dir / "session.json"
        if not session_path.is_file():
            raise RuntimeError("complete profiled study is missing session.json")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if session.get("status") != "complete":
            raise RuntimeError("complete profiled study session is not complete")
    return {
        "manifest": manifest,
        "expected_configs": expected_configs,
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
    if _inside_git_checkout(output_path):
        raise ValueError("study package must be written outside every Git checkout")
    validation = validate_complete_study(
        study_dir,
        allow_incomplete=allow_incomplete,
        recover_stale_lock=recover_stale_lock,
    )
    manifest = validation["manifest"]
    assert isinstance(manifest, dict)
    files_by_name = _collect_regular_files(study_dir)
    if not files_by_name:
        raise RuntimeError("study contains no files to package")
    expected_configs = validation["expected_configs"]
    assert isinstance(expected_configs, dict)
    expected_members = _expected_complete_members(expected_configs)
    observed_members = set(files_by_name)
    if validation["study_complete"] and manifest["configuration"].get(
        "study_profile"
    ) is not None:
        missing = sorted(expected_members - observed_members)
        unexpected = sorted(observed_members - expected_members)
        if missing or unexpected:
            raise RuntimeError(
                f"profiled study member set mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
    else:
        recovery_receipts = _validated_recovery_receipts(files_by_name)
        unexpected = sorted(observed_members - expected_members - recovery_receipts)
        if unexpected:
            raise RuntimeError(f"study contains unexpected package input: {unexpected[0]}")
    files = [files_by_name[name] for name in sorted(files_by_name)]
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
        "archive": {
            "path": output_path.name,
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
