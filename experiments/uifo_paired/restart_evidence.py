"""Authenticate the mechanics predecessor before a paid restart screen."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from experiments.uifo_paired.results_ingestion import (
    SIDECAR_PATTERN,
    inspect_zip_integrity,
    sha256_path,
    strict_json_loads,
)
from experiments.uifo_paired.study_profiles import bind_study_profile


def validate_mechanics_predecessor(
    study_dir: Path, package_path: Path
) -> dict[str, object]:
    """Validate a passed mechanics study/package and return path-free evidence."""
    from experiments.uifo_paired.package import validate_complete_study

    study_dir = study_dir.resolve()
    package_path = package_path.resolve()
    if not package_path.is_file():
        raise RuntimeError(f"missing mechanics package: {package_path}")
    sidecar_path = package_path.with_suffix(package_path.suffix + ".sha256")
    package_manifest_path = package_path.with_suffix(
        package_path.suffix + ".manifest.json"
    )
    for label, path in (
        ("checksum sidecar", sidecar_path),
        ("package manifest", package_manifest_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"missing mechanics {label}: {path}")

    validation = validate_complete_study(study_dir)
    if (
        validation.get("study_complete") is not True
        or validation.get("planned_runs") != 1
        or validation.get("completed_runs") != 1
    ):
        raise RuntimeError("mechanics predecessor is not one complete run")
    manifest = validation.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("mechanics predecessor manifest is invalid")
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("study_profile") != "restart-mechanics-v1"
    ):
        raise RuntimeError("mechanics predecessor has the wrong study profile")
    expected_policy = bind_study_profile("restart-mechanics-v1", configuration)
    if configuration.get("decision_policy") != expected_policy:
        raise RuntimeError("mechanics predecessor decision policy is invalid")
    plan_id = str(manifest.get("plan_id", ""))
    revision = str(manifest.get("project_revision", ""))
    if not re.fullmatch(r"[0-9a-f]{16}", plan_id):
        raise RuntimeError("mechanics predecessor plan ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("mechanics predecessor revision is invalid")
    core = {
        "configuration": configuration,
        "run_order_policy": manifest.get("run_order_policy"),
        "primary_pair_order": manifest.get("primary_pair_order"),
        "runs": manifest.get("runs"),
    }
    recomputed_plan_id = hashlib.sha256(
        _json_bytes(core)
    ).hexdigest()[:16]
    if recomputed_plan_id != plan_id:
        raise RuntimeError("mechanics predecessor plan ID does not match its plan")

    summary_path = study_dir / "summary.json"
    summary = strict_json_loads(summary_path.read_bytes(), "mechanics summary")
    if not isinstance(summary, dict):
        raise RuntimeError("mechanics predecessor summary is invalid")
    decision = summary.get("predeclared_decision")
    if not isinstance(decision, dict):
        raise RuntimeError("mechanics predecessor decision is missing")
    if (
        decision.get("status") != "passed"
        or decision.get("passed") is not True
        or decision.get("action") != "run_restart_screen_v1"
    ):
        raise RuntimeError("mechanics predecessor did not pass its frozen gate")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise RuntimeError("mechanics predecessor plan membership is invalid")
    run_id = str(runs[0].get("run_id", ""))
    record_path = study_dir / "runs" / f"{run_id}.json"
    record = strict_json_loads(record_path.read_bytes(), "mechanics run record")
    if not isinstance(record, dict) or record.get("status") != "complete":
        raise RuntimeError("mechanics predecessor run record is not complete")
    history = record.get("history")
    telemetry = record.get("optimizer_telemetry")
    if not isinstance(history, dict) or not isinstance(telemetry, dict):
        raise RuntimeError("mechanics predecessor artifacts are not bound")

    package_digest = sha256_path(package_path)
    sidecar_match = SIDECAR_PATTERN.fullmatch(
        sidecar_path.read_text(encoding="utf-8")
    )
    if sidecar_match is None:
        raise RuntimeError("mechanics package checksum sidecar is malformed")
    sidecar_digest, sidecar_name = sidecar_match.groups()
    if sidecar_name != package_path.name or sidecar_digest != package_digest:
        raise RuntimeError("mechanics package checksum sidecar does not match")

    package_manifest = strict_json_loads(
        package_manifest_path.read_bytes(), "mechanics package manifest"
    )
    if not isinstance(package_manifest, dict):
        raise RuntimeError("mechanics package manifest is invalid")
    archive_meta = package_manifest.get("archive")
    expected_package_fields = {
        "format_version": 1,
        "study_plan_id": plan_id,
        "study_project_revision": revision,
        "study_complete": True,
        "planned_runs": 1,
        "completed_runs": 1,
        "incomplete_runs": [],
    }
    for key, expected in expected_package_fields.items():
        if package_manifest.get(key) != expected:
            raise RuntimeError(f"mechanics package manifest has invalid {key}")
    if (
        not isinstance(archive_meta, dict)
        or archive_meta.get("sha256") != package_digest
        or archive_meta.get("size_bytes") != package_path.stat().st_size
    ):
        raise RuntimeError("mechanics package manifest archive metadata is invalid")

    integrity = inspect_zip_integrity(package_path)
    local_files = sorted(
        path
        for path in study_dir.rglob("*")
        if path.is_file() and path.name != ".study.lock"
    )
    expected_names = {path.relative_to(study_dir).as_posix() for path in local_files}
    expected_names.add("package-state.json")
    if set(integrity["member_names"]) != expected_names:
        raise RuntimeError("mechanics package membership disagrees with study")
    if archive_meta.get("files") != len(expected_names):
        raise RuntimeError("mechanics package file count is invalid")
    with zipfile.ZipFile(package_path, "r") as archive:
        for path in local_files:
            name = path.relative_to(study_dir).as_posix()
            with archive.open(name, "r") as handle:
                archived_digest = _sha256_stream(handle)
            if archived_digest != sha256_path(path):
                raise RuntimeError(
                    f"mechanics package member disagrees with study: {name}"
                )
        package_state = strict_json_loads(
            archive.read("package-state.json"), "mechanics package state"
        )
    if package_state != {
        "format_version": 1,
        "study_complete": True,
        "planned_runs": 1,
        "completed_runs": 1,
        "incomplete_runs": [],
    }:
        raise RuntimeError("mechanics package state is invalid")

    return {
        "format_version": 1,
        "study_profile": "restart-mechanics-v1",
        "plan_id": plan_id,
        "project_revision": revision,
        "package_sha256": package_digest,
        "package_manifest_sha256": sha256_path(package_manifest_path),
        "record_sha256": sha256_path(record_path),
        "history_sha256": str(history.get("sha256")),
        "optimizer_telemetry_sha256": str(telemetry.get("sha256")),
        "decision_status": "passed",
        "decision_action": "run_restart_screen_v1",
    }


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
