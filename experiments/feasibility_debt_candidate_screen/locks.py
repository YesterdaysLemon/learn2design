"""Strict source/runtime lock validation for the pre-result surface."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .canonical import ReceiptError, assert_digest, read_receipt, sha256_file
from .contract import ARM_ORDER, STUDY_ID, arm_specs


SOURCE_LOCK_KEYS = {
    "revision",
    "arm_profiles",
    "sources",
    "runtime_lock_sha256",
    "worker_sha256",
    "orchestrator_sha256",
    "production_analyzer_sha256",
    "reference_analyzer_sha256",
    "panel_commitment_sha256",
}

RUNTIME_LOCK_KEYS = {
    "python",
    "base_image_digest",
    "kernel",
    "nvidia_driver",
    "cuda_packages",
    "python_distributions",
    "environment_allowlist",
    "device",
    "package_closure_sha256",
}


def verify_source_rows(
    rows: object, *, logical_sources: dict[str, Path | bytes]
) -> list[dict[str, object]]:
    if not isinstance(rows, list) or not rows:
        raise ReceiptError("source lock rows are missing")
    if [row.get("logical_id") for row in rows if isinstance(row, dict)] != sorted(
        logical_sources
    ):
        raise ReceiptError("source lock rows are not in exact logical-ID order")
    verified: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "logical_id",
            "sha256",
            "size_bytes",
        }:
            raise ReceiptError("source lock row schema mismatch")
        logical_id = row["logical_id"]
        if logical_id not in logical_sources:
            raise ReceiptError("source lock contains an unexpected logical ID")
        source = logical_sources[str(logical_id)]
        if isinstance(source, Path):
            if not source.is_file():
                raise ReceiptError("source lock path is absent")
            content_digest = sha256_file(source)
            content_size = source.stat().st_size
        else:
            import hashlib

            content_digest = hashlib.sha256(source).hexdigest()
            content_size = len(source)
        digest = assert_digest(row["sha256"], label="source digest")
        if content_digest != digest or content_size != row["size_bytes"]:
            raise ReceiptError("source lock bytes mismatch")
        verified.append(row)
    return verified


def read_runtime_lock(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="runtime_lock",
        expected_payload_keys=RUNTIME_LOCK_KEYS,
    )
    python = payload["python"]
    if not isinstance(python, dict) or set(python) != {
        "version",
        "interpreter_sha256",
    }:
        raise ReceiptError("runtime Python lock schema mismatch")
    assert_digest(python["interpreter_sha256"], label="interpreter digest")
    image = payload["base_image_digest"]
    if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ReceiptError("runtime base image is not immutable")
    assert_digest(payload["package_closure_sha256"], label="package closure")
    for key in ("kernel", "nvidia_driver"):
        row = payload[key]
        expected_keys = (
            {"release", "version", "machine"}
            if key == "kernel"
            else {"version", "library_tree_sha256"}
        )
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ReceiptError(f"runtime {key} schema mismatch")
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise ReceiptError(f"runtime {key} value is invalid")
        if key == "nvidia_driver":
            assert_digest(row["library_tree_sha256"], label="driver tree")
    cuda_packages = payload["cuda_packages"]
    if not isinstance(cuda_packages, list) or not cuda_packages:
        raise ReceiptError("runtime CUDA package closure is empty")
    _validate_package_rows(cuda_packages, digest_key="artifact_sha256")
    distributions = payload["python_distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ReceiptError("runtime Python distribution closure is empty")
    _validate_package_rows(distributions, digest_key="record_tree_sha256")
    observed_versions = {
        str(row["name"]).lower().replace("_", "-"): str(row["version"])
        for row in distributions
    }
    required_versions = {
        "dfbench": "0.3.3",
        "jax": "0.9.0.1",
        "jaxlib": "0.9.0.1",
        "jax-cuda13-pjrt": "0.9.0.1",
        "jax-cuda13-plugin": "0.9.0.1",
    }
    if any(observed_versions.get(name) != version for name, version in required_versions.items()):
        raise ReceiptError("runtime required distribution version mismatch")
    environment = payload["environment_allowlist"]
    if not isinstance(environment, list) or [
        row.get("name") for row in environment if isinstance(row, dict)
    ] != sorted(row.get("name") for row in environment if isinstance(row, dict)):
        raise ReceiptError("runtime environment allowlist order mismatch")
    for row in environment:
        if not isinstance(row, dict) or set(row) != {
            "name",
            "is_set",
            "value_sha256",
        }:
            raise ReceiptError("runtime environment allowlist row mismatch")
        if not isinstance(row["name"], str) or not row["name"].isupper():
            raise ReceiptError("runtime environment name is invalid")
        if row["is_set"] is True:
            assert_digest(row["value_sha256"], label="environment value")
        elif row["is_set"] is False and row["value_sha256"] is not None:
            raise ReceiptError("unset environment row carries a digest")
        elif row["is_set"] is not False:
            raise ReceiptError("environment is_set must be Boolean")
    device = payload["device"]
    if not isinstance(device, dict) or set(device) != {
        "model",
        "count",
        "mig_enabled",
        "compute_capability",
        "device_receipt_sha256",
    } or (
        device["model"] != "NVIDIA H100 80GB HBM3"
        or device["count"] != 1
        or device["mig_enabled"] is not False
        or not isinstance(device["compute_capability"], str)
        or not device["compute_capability"]
    ):
        raise ReceiptError("runtime device identity mismatch")
    assert_digest(device["device_receipt_sha256"], label="device receipt")
    return payload, digest


def _validate_package_rows(rows: list[object], *, digest_key: str) -> None:
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "version", digest_key}:
            raise ReceiptError("runtime package row schema mismatch")
        name = row["name"]
        version = row["version"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            raise ReceiptError("runtime package identity is invalid")
        assert_digest(row[digest_key], label=f"runtime package {digest_key}")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ReceiptError("runtime package rows are not unique and sorted")


def read_source_lock(
    path: Path,
    *,
    runtime_lock_sha256: str,
    logical_sources: dict[str, Path | bytes],
    expected_revision: str | None = None,
) -> tuple[dict[str, Any], str]:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="source_lock",
        expected_payload_keys=SOURCE_LOCK_KEYS,
    )
    if payload["runtime_lock_sha256"] != runtime_lock_sha256:
        raise ReceiptError("source/runtime lock binding mismatch")
    revision = payload["revision"]
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or (expected_revision is not None and revision != expected_revision)
    ):
        raise ReceiptError("source lock revision mismatch")
    for key in (
        "worker_sha256",
        "orchestrator_sha256",
        "production_analyzer_sha256",
        "reference_analyzer_sha256",
        "panel_commitment_sha256",
    ):
        assert_digest(payload[key], label=key)
    profiles = payload["arm_profiles"]
    if not isinstance(profiles, list) or [
        row.get("arm_id") for row in profiles if isinstance(row, dict)
    ] != list(ARM_ORDER):
        raise ReceiptError("source lock arm order mismatch")
    frozen = arm_specs()
    for row in profiles:
        if not isinstance(row, dict) or set(row) != {
            "arm_id",
            "logical_module_id",
            "python_module_name",
            "class_name",
            "algorithm_str",
            "source_sha256",
            "patience",
            "population_mode",
            "preclock_warmup",
            "progress_mode",
            "kwargs",
            "package_closure_sha256",
        }:
            raise ReceiptError("source lock arm row is invalid")
        arm_id = row.get("arm_id")
        if arm_id not in frozen:
            raise ReceiptError("source lock arm is unknown")
        expected = frozen[str(arm_id)]
        for key, value in {
            "logical_module_id": expected.logical_module_id,
            "python_module_name": expected.python_module_name,
            "class_name": expected.class_name,
            "algorithm_str": expected.algorithm_str,
            "source_sha256": expected.source_sha256,
            "patience": expected.patience,
            "population_mode": expected.population_mode,
            "preclock_warmup": expected.preclock_warmup,
            "progress_mode": expected.progress_mode,
            "kwargs": expected.fixed_kwargs(),
        }.items():
            if row.get(key) != value:
                raise ReceiptError(f"source lock arm field mismatch: {key}")
        assert_digest(row.get("package_closure_sha256"), label="arm package closure")
    verify_source_rows(payload["sources"], logical_sources=logical_sources)
    return payload, digest
