"""Outcome-blind validation and normalization for packaged UIFO studies.

The validator never extracts the source archive.  It authenticates the external
files, validates ZIP structure and CRCs, and recomputes every run metric from a
pickle-free NPZ history before callers are allowed to parse ``summary.json``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterable

from experiments.uifo_paired.metrics import summarize_rows
from experiments.uifo_paired.plan import primary_pair_order_counts
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
from experiments.uifo_paired.study_profiles import bind_study_profile


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SIDECAR_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^/\\\r\n]+)\r?\n?$")
RUN_MEMBER_PATTERN = re.compile(r"^(configs|histories|runs)/([^/]+)\.(json|npz)$")
LOG_MEMBER_PATTERN = re.compile(r"^logs/([^/]+)\.(stdout|stderr)\.log$")
PREFLIGHT_MEMBERS = {
    "preflight.host-environment.json",
    "preflight.json",
    "preflight.stderr.log",
    "preflight.stdout.log",
}
FIXED_MEMBERS = {
    "manifest.json",
    "package-state.json",
    "runs.jsonl",
    "session.json",
    "summary.json",
}
HISTORY_MAX_NPY_HEADER_BYTES = 10_000
HISTORY_MAX_ARRAY_ELEMENTS = 32 * 1024 * 1024


class StudyValidationError(RuntimeError):
    """Raised when private study evidence cannot be trusted."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_entries: int = 512
    max_entry_uncompressed_bytes: int = 64 * 1024 * 1024
    max_total_uncompressed_bytes: int = 512 * 1024 * 1024
    max_compression_ratio: float = 250.0


@dataclass(frozen=True)
class ExpectedSources:
    zip_sha256: str
    package_manifest_sha256: str
    checksum_file_sha256: str
    plan_sha256: str
    plan_id: str
    project_revision: str


DEVELOPMENT_V2_SOURCES = ExpectedSources(
    zip_sha256="7b28509299e81e1f5151c4854bb5591d022de44e17b12c81e71fa2a08eabce24",
    package_manifest_sha256=(
        "533f28f44e69ab9efe817520daca1962ecc6aa2d16a2ef709d4c88a120a7fafc"
    ),
    checksum_file_sha256=(
        "74c6e781fcf3c74144ee764cf3a7cb8a4ea14c6e05da2da5dcc8382b80c4c4cf"
    ),
    plan_sha256="a6cd004891809f2ecc07370c31cfae876836d924b8928ffbacda59fb5c7c6108",
    plan_id="49ff0e783f4f6a10",
    project_revision="dbd557b713ab657ac971957369d89eb67649d09f",
)


@dataclass(frozen=True)
class SourcePaths:
    archive: Path
    checksum: Path
    package_manifest: Path
    plan: Path


@dataclass
class ValidatedStudy:
    sources: SourcePaths
    source_hashes: dict[str, str]
    archive_members: tuple[str, ...]
    plan: dict[str, object]
    manifest: dict[str, object]
    package_state: dict[str, object]
    session: dict[str, object]
    configs: dict[str, dict[str, object]]
    records: list[dict[str, object]]
    history_rows: dict[str, list[dict[str, object]]]
    integrity: dict[str, object]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StudyValidationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes | str, label: str) -> object:
    try:
        text = payload.decode("utf-8", errors="strict") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                StudyValidationError(f"{label} contains non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StudyValidationError(f"malformed JSON in {label}: {error}") from error


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StudyValidationError(f"{label} must be a JSON object")
    return value


def verify_external_sources(
    sources: SourcePaths,
    expected: ExpectedSources = DEVELOPMENT_V2_SOURCES,
    *,
    expected_run_count: int = 64,
) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    paths = {
        "archive": sources.archive,
        "checksum": sources.checksum,
        "package_manifest": sources.package_manifest,
        "plan": sources.plan,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise StudyValidationError(f"missing external {label} file: {path}")
    hashes = {label: sha256_path(path) for label, path in paths.items()}
    expected_hashes = {
        "archive": expected.zip_sha256,
        "checksum": expected.checksum_file_sha256,
        "package_manifest": expected.package_manifest_sha256,
        "plan": expected.plan_sha256,
    }
    for label, digest in hashes.items():
        if digest != expected_hashes[label]:
            raise StudyValidationError(
                f"external {label} SHA-256 mismatch: expected "
                f"{expected_hashes[label]}, observed {digest}"
            )

    sidecar_text = sources.checksum.read_text(encoding="utf-8")
    match = SIDECAR_PATTERN.fullmatch(sidecar_text)
    if match is None:
        raise StudyValidationError("malformed SHA-256 sidecar")
    sidecar_digest, sidecar_filename = match.groups()
    if sidecar_filename != sources.archive.name:
        raise StudyValidationError(
            "SHA-256 sidecar filename does not name the supplied ZIP"
        )
    if sidecar_digest != hashes["archive"]:
        raise StudyValidationError(
            "SHA-256 sidecar digest does not match the supplied ZIP"
        )

    package_manifest = _require_mapping(
        strict_json_loads(sources.package_manifest.read_bytes(), "package manifest"),
        "package manifest",
    )
    plan = _require_mapping(
        strict_json_loads(sources.plan.read_bytes(), "external plan"),
        "external plan",
    )
    archive_meta = _require_mapping(
        package_manifest.get("archive"), "package manifest archive"
    )
    if package_manifest.get("format_version") != 1:
        raise StudyValidationError("unsupported package manifest format")
    if package_manifest.get("study_plan_id") != expected.plan_id:
        raise StudyValidationError("package manifest plan ID mismatch")
    if package_manifest.get("study_project_revision") != expected.project_revision:
        raise StudyValidationError("package manifest project revision mismatch")
    if package_manifest.get("study_complete") is not True:
        raise StudyValidationError("package manifest does not describe a complete study")
    if package_manifest.get("incomplete_runs") != []:
        raise StudyValidationError("package manifest lists incomplete runs")
    if package_manifest.get("planned_runs") != expected_run_count or package_manifest.get(
        "completed_runs"
    ) != expected_run_count:
        raise StudyValidationError(
            "package manifest run counts are not "
            f"{expected_run_count}/{expected_run_count}"
        )
    if archive_meta.get("sha256") != hashes["archive"]:
        raise StudyValidationError("package manifest archive digest mismatch")
    if archive_meta.get("size_bytes") != sources.archive.stat().st_size:
        raise StudyValidationError("package manifest archive size mismatch")
    if plan.get("format_version") != 1 or plan.get("plan_id") != expected.plan_id:
        raise StudyValidationError("external plan identity mismatch")
    return hashes, package_manifest, plan


def _validate_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise StudyValidationError(f"unsafe ZIP member name: {name!r}")
    pure = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        name.startswith("/")
        or pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise StudyValidationError(f"unsafe ZIP member path: {name!r}")
    if pure.as_posix() != name:
        raise StudyValidationError(f"non-canonical ZIP member path: {name!r}")


def _validate_member_type(info: zipfile.ZipInfo) -> None:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    allowed = {0, stat.S_IFREG}
    if info.is_dir() or file_type not in allowed:
        raise StudyValidationError(
            f"ZIP contains a directory, symlink, or special entry: {info.filename!r}"
        )


def inspect_zip_integrity(
    archive_path: Path,
    limits: ArchiveLimits = ArchiveLimits(),
) -> dict[str, object]:
    """Validate ZIP names, sizes, compression metadata, and every member CRC."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) > limits.max_entries:
                raise StudyValidationError("ZIP contains too many entries")
            if len(names) != len(set(names)):
                raise StudyValidationError("ZIP contains duplicate member names")
            casefolded = [name.casefold() for name in names]
            if len(casefolded) != len(set(casefolded)):
                raise StudyValidationError(
                    "ZIP contains case-colliding duplicate member names"
                )
            total = 0
            maximum_ratio = 1.0
            for info in infos:
                _validate_member_name(info.filename)
                _validate_member_type(info)
                if info.flag_bits & 0x1:
                    raise StudyValidationError(
                        f"encrypted ZIP member is not supported: {info.filename!r}"
                    )
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise StudyValidationError(
                        f"unexpected ZIP compression method: {info.filename!r}"
                    )
                if info.file_size > limits.max_entry_uncompressed_bytes:
                    raise StudyValidationError(
                        f"ZIP member exceeds uncompressed-size limit: {info.filename!r}"
                    )
                total += info.file_size
                if total > limits.max_total_uncompressed_bytes:
                    raise StudyValidationError("ZIP exceeds total uncompressed-size limit")
                if info.file_size:
                    if info.compress_size == 0:
                        raise StudyValidationError(
                            f"ZIP member has suspicious infinite ratio: {info.filename!r}"
                        )
                    ratio = info.file_size / info.compress_size
                    maximum_ratio = max(maximum_ratio, ratio)
                    if ratio > limits.max_compression_ratio:
                        raise StudyValidationError(
                            f"ZIP member has suspicious compression ratio: {info.filename!r}"
                        )
                with archive.open(info, "r") as handle:
                    observed = 0
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        observed += len(chunk)
                if observed != info.file_size:
                    raise StudyValidationError(
                        f"ZIP member length disagrees with metadata: {info.filename!r}"
                    )
            corrupted = archive.testzip()
            if corrupted is not None:
                raise StudyValidationError(f"ZIP CRC failure: {corrupted!r}")
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        raise StudyValidationError(f"invalid ZIP archive: {error}") from error
    return {
        "entries": len(infos),
        "total_uncompressed_bytes": total,
        "maximum_compression_ratio": maximum_ratio,
        "crc_check": "passed",
        "member_names": tuple(names),
    }


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        with archive.open(name, "r") as handle:
            return handle.read()
    except KeyError as error:
        raise StudyValidationError(f"missing ZIP member: {name}") from error


def _load_json_member(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    return _require_mapping(strict_json_loads(_read_member(archive, name), name), name)


def _validate_plan_contract(
    plan: dict[str, object], expected: ExpectedSources
) -> dict[str, dict[str, object]]:
    if plan.get("format_version") != 1 or plan.get("plan_id") != expected.plan_id:
        raise StudyValidationError("study plan identity mismatch")
    configuration = _require_mapping(plan.get("configuration"), "plan configuration")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != 64:
        raise StudyValidationError("study plan must contain exactly 64 runs")
    if len({str(run.get("run_id")) for run in runs if isinstance(run, dict)}) != 64:
        raise StudyValidationError("study plan contains duplicate run IDs")
    if [run.get("planned_run_index") for run in runs if isinstance(run, dict)] != list(
        range(64)
    ):
        raise StudyValidationError("study plan run indexes are not exact serial order")
    expected_configuration = {
        "arms": ["no_prior", "semantic_prior"],
        "evaluation_chunk_size": None,
        "jax_compilation_cache_policy": "disabled",
        "max_evals": None,
        "max_time_seconds": 600.0,
        "n_frequencies": 50,
        "optimizer_seeds": [7, 11],
        "population_size": 8,
        "require_a100": True,
        "study_profile": "development-v2",
        "target_losses": [4.0, 1.0, 0.5, 0.0],
    }
    for key, value in expected_configuration.items():
        if configuration.get(key) != value:
            raise StudyValidationError(
                f"frozen configuration mismatch for {key}: {configuration.get(key)!r}"
            )
    policy = bind_study_profile("development-v2", configuration)
    if strict_json(configuration.get("decision_policy")) != strict_json(policy):
        raise StudyValidationError("frozen decision policy mismatch")
    policy_mapping = _require_mapping(policy, "development decision policy")
    if policy_mapping.get("policy_id") != "semantic-prior-development-v2":
        raise StudyValidationError("frozen decision policy ID mismatch")
    if plan.get("run_order_policy") != "rotate arms once per topology-seed pair":
        raise StudyValidationError("study was not planned for serial arm rotation")
    order_counts = primary_pair_order_counts(runs)
    required_order = {
        "complete_primary_pairs": 32,
        "no_prior_first": 16,
        "semantic_prior_first": 16,
        "absolute_imbalance": 0,
    }
    if order_counts != required_order or plan.get("primary_pair_order") != required_order:
        raise StudyValidationError("primary pair order is not exactly balanced")
    core = {
        "configuration": configuration,
        "run_order_policy": plan["run_order_policy"],
        "primary_pair_order": plan["primary_pair_order"],
        "runs": runs,
    }
    recomputed_plan_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if recomputed_plan_id != expected.plan_id:
        raise StudyValidationError("plan ID does not match its canonical contents")

    topologies = configuration.get("topologies")
    if not isinstance(topologies, list) or len(topologies) != 16:
        raise StudyValidationError("study must contain exactly 16 planned topologies")
    topology_keys = {
        json.dumps(topology, sort_keys=True, separators=(",", ":"))
        for topology in topologies
    }
    if len(topology_keys) != 16:
        raise StudyValidationError("planned topology identities are not unique")
    pair_ids = {str(run["pair_id"]) for run in runs}
    if len(pair_ids) != 32:
        raise StudyValidationError("study must contain exactly 32 topology-seed pairs")
    expected_configs = {
        str(run["run_id"]): _run_config(run, configuration) for run in runs
    }
    return expected_configs


def _validate_environment(
    manifest: dict[str, object],
    expected_project_revision: str = DEVELOPMENT_V2_SOURCES.project_revision,
) -> None:
    if manifest.get("project_revision") != expected_project_revision:
        raise StudyValidationError("study project revision mismatch")
    if manifest.get("working_tree_dirty") is not False:
        raise StudyValidationError("study manifest does not prove a clean worktree")
    environment = _require_mapping(manifest.get("environment"), "runtime environment")
    if environment.get("python") is None or not str(environment["python"]).startswith(
        "3.12."
    ):
        raise StudyValidationError("study did not use Python 3.12")
    if (
        environment.get("backend") != "gpu"
        or environment.get("device_count") != 1
        or environment.get("competition_aligned_a100") is not True
    ):
        raise StudyValidationError("runtime was not exactly one JAX-visible A100")
    kinds = environment.get("device_kinds")
    if not isinstance(kinds, list) or len(kinds) != 1 or "A100" not in str(kinds[0]).upper():
        raise StudyValidationError("runtime device identity is not one A100")
    runtime_configuration = _require_mapping(
        environment.get("jax_runtime_configuration"), "JAX runtime configuration"
    )
    if runtime_configuration.get("enable_compilation_cache") is not False:
        raise StudyValidationError("persistent JAX compilation caching was enabled")
    if runtime_configuration.get("compilation_cache_dir") is not None:
        raise StudyValidationError("persistent JAX compilation cache directory was set")
    runtime_environment = _require_mapping(
        environment.get("jax_runtime_environment"), "JAX runtime environment"
    )
    allowed_overrides = {
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    for name in JAX_RUNTIME_ENVIRONMENT_KEYS:
        value = runtime_environment.get(name)
        if name in allowed_overrides:
            if value != allowed_overrides[name]:
                raise StudyValidationError(f"unexpected required runtime value for {name}")
        elif value is not None:
            raise StudyValidationError(f"forbidden JAX/XLA/CUDA override: {name}")

    rental = _require_mapping(manifest.get("rental_preflight"), "rental preflight")
    snapshot = _require_mapping(rental.get("gpu_idle"), "rental GPU snapshot")
    gpus = snapshot.get("gpus")
    if snapshot.get("status") != "ok" or not isinstance(gpus, list) or len(gpus) != 1:
        raise StudyValidationError("preflight does not prove exactly one physical GPU")
    gpu = _require_mapping(gpus[0], "physical GPU")
    gpu_name = str(gpu.get("name", "")).upper()
    if "A100" not in gpu_name or "PCIE" not in gpu_name or "80GB" not in gpu_name:
        raise StudyValidationError("physical GPU was not an A100 80GB PCIe")
    if str(gpu.get("mig_mode_current", "")).lower() != "disabled":
        raise StudyValidationError("A100 MIG mode was not disabled")
    if int(gpu.get("memory_total_mib", 0)) < 75_000:
        raise StudyValidationError("A100 physical memory evidence is insufficient")


def _validate_history_npy_header(
    handle: BinaryIO,
    *,
    info: zipfile.ZipInfo,
    label: str,
) -> None:
    """Reject unsafe NPY declarations before NumPy may allocate their arrays."""
    import numpy as np

    field = info.filename.removesuffix(".npy")
    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle, max_header_size=HISTORY_MAX_NPY_HEADER_BYTES
            )
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                handle, max_header_size=HISTORY_MAX_NPY_HEADER_BYTES
            )
        else:
            raise StudyValidationError(
                f"invalid pickle-free NPZ history {label}: unsupported NPY "
                f"version {version!r} in {info.filename}"
            )
    except StudyValidationError:
        raise
    except (EOFError, OSError, ValueError) as error:
        raise StudyValidationError(
            f"invalid pickle-free NPZ history {label}: malformed NPY header in "
            f"{info.filename}: {error}"
        ) from error

    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise StudyValidationError(
            f"invalid pickle-free NPZ history {label}: object dtype in "
            f"{info.filename}"
        )
    if not isinstance(shape, tuple) or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        raise StudyValidationError(
            f"invalid pickle-free NPZ history {label}: unsafe declared shape in "
            f"{info.filename}"
        )

    expected_dtype = HISTORY_SCHEMA[field]["dtype"]
    if field == "initial_params_unbounded":
        if dtype not in {np.dtype("float32"), np.dtype("float64")} or len(shape) not in {
            1,
            2,
        }:
            raise StudyValidationError(
                f"history field has invalid declared shape/dtype: "
                f"{label}:{field}"
            )
    elif dtype != np.dtype(str(expected_dtype)) or len(shape) != 1:
        raise StudyValidationError(
            f"history field has invalid declared shape/dtype: {label}:{field}"
        )

    element_count = 1
    for dimension in shape:
        if dimension > HISTORY_MAX_ARRAY_ELEMENTS or (
            dimension and element_count > HISTORY_MAX_ARRAY_ELEMENTS // dimension
        ):
            raise StudyValidationError(
                f"invalid pickle-free NPZ history {label}: declared shape exceeds "
                f"safety limits in {info.filename}"
            )
        element_count *= dimension
    expected_payload_bytes = element_count * dtype.itemsize
    header_bytes = handle.tell()
    if (
        expected_payload_bytes > 32 * 1024 * 1024
        or header_bytes > info.file_size
        or info.file_size - header_bytes != expected_payload_bytes
    ):
        raise StudyValidationError(
            f"invalid pickle-free NPZ history {label}: NPY payload-size mismatch "
            f"in {info.filename}"
        )


def _load_history_arrays(payload: bytes, label: str) -> dict[str, object]:
    import numpy as np

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as nested:
            infos = nested.infolist()
            names = [info.filename for info in infos]
            expected_names = {f"{name}.npy" for name in HISTORY_SCHEMA}
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise StudyValidationError(f"history schema mismatch in {label}")
            total_uncompressed = 0
            for info in infos:
                _validate_member_name(info.filename)
                _validate_member_type(info)
                if info.flag_bits & 0x1:
                    raise StudyValidationError(f"encrypted history member in {label}")
                if info.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise StudyValidationError(
                        f"unexpected history compression method in {label}"
                    )
                if info.file_size > 32 * 1024 * 1024:
                    raise StudyValidationError(f"history member too large in {label}")
                total_uncompressed += info.file_size
                if total_uncompressed > 64 * 1024 * 1024:
                    raise StudyValidationError(f"history archive too large in {label}")
                if info.file_size:
                    if info.compress_size == 0 or (
                        info.file_size / info.compress_size > 250.0
                    ):
                        raise StudyValidationError(
                            f"suspicious history compression ratio in {label}"
                        )
                with nested.open(info, "r") as handle:
                    _validate_history_npy_header(handle, info=info, label=label)
            if nested.testzip() is not None:
                raise StudyValidationError(f"history CRC failure in {label}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in HISTORY_SCHEMA}
    except (zipfile.BadZipFile, ValueError, KeyError, OSError) as error:
        raise StudyValidationError(f"invalid pickle-free NPZ history {label}: {error}") from error

    row_fields = [name for name in HISTORY_SCHEMA if name != "initial_params_unbounded"]
    expected_dtypes = {
        "call_index": np.dtype("int32"),
        "candidate_index": np.dtype("int16"),
        "eval_count_after_call": np.dtype("int64"),
        "time_seconds": np.dtype("float64"),
        "loss": np.dtype("float64"),
        "sensitivity_loss": np.dtype("float64"),
        "penalty": np.dtype("float64"),
        "is_feasible": np.dtype("bool"),
    }
    for name in row_fields:
        if arrays[name].ndim != 1 or arrays[name].dtype != expected_dtypes[name]:
            raise StudyValidationError(f"history field has invalid shape/dtype: {label}:{name}")
    lengths = {int(arrays[name].shape[0]) for name in row_fields}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise StudyValidationError(f"history row arrays are empty or inconsistent: {label}")
    if arrays["initial_params_unbounded"].ndim not in (1, 2):
        raise StudyValidationError(f"invalid initial parameter array: {label}")
    return arrays


def _validate_record(
    record: dict[str, object],
    expected_config: dict[str, object],
    archive: zipfile.ZipFile,
    expected_environment: dict[str, object],
) -> list[dict[str, object]]:
    run_id = str(expected_config["run_id"])
    if record.get("format_version") != 1 or record.get("status") != "complete":
        raise StudyValidationError(f"run is not complete format-version-1: {run_id}")
    if record.get("run_id") != run_id:
        raise StudyValidationError(f"run record filename/ID mismatch: {run_id}")
    if strict_json(record.get("config")) != strict_json(expected_config):
        raise StudyValidationError(f"run configuration mismatch: {run_id}")
    if strict_json(record.get("environment")) != strict_json(expected_environment):
        raise StudyValidationError(f"run environment mismatch: {run_id}")

    history = _require_mapping(record.get("history"), f"history metadata {run_id}")
    history_name = f"histories/{run_id}.npz"
    if history.get("path") != history_name or history.get("format_version") != 1:
        raise StudyValidationError(f"history reference mismatch: {run_id}")
    history_payload = _read_member(archive, history_name)
    if history.get("sha256") != sha256_bytes(history_payload):
        raise StudyValidationError(f"history SHA-256 mismatch: {run_id}")
    arrays = _load_history_arrays(history_payload, history_name)
    rows = _rows_from_history_arrays(arrays)
    if history.get("rows") != len(rows):
        raise StudyValidationError(f"history row count mismatch: {run_id}")
    if strict_json(history.get("schema")) != strict_json(HISTORY_SCHEMA):
        raise StudyValidationError(f"history schema metadata mismatch: {run_id}")
    time_grid, eval_grid = _metric_grids(expected_config)
    recomputed_metrics = summarize_rows(
        rows,
        list(expected_config["target_losses"]),
        time_grid=time_grid,
        eval_grid=eval_grid,
    )
    if strict_json(record.get("metrics")) != strict_json(recomputed_metrics):
        raise StudyValidationError(f"run metrics do not match history: {run_id}")
    expected_roles = _initial_population_roles(
        str(expected_config["arm"]), int(expected_config["population_size"])
    )
    recorded_roles = record.get("initial_population_roles")
    recorded_hashes = record.get("initial_parameter_hashes")
    if not isinstance(recorded_roles, list) or not isinstance(recorded_hashes, list):
        raise StudyValidationError(f"initial population evidence is missing: {run_id}")
    if len(recorded_roles) != len(expected_roles) or len(recorded_hashes) != len(
        expected_roles
    ):
        raise StudyValidationError(f"initial population evidence has wrong size: {run_id}")
    if recorded_roles != expected_roles:
        raise StudyValidationError(f"initial population hierarchy mismatch: {run_id}")
    if any(
        not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
        for digest in recorded_hashes
    ):
        raise StudyValidationError(f"initial parameter hash is invalid: {run_id}")
    initial_params = arrays["initial_params_unbounded"]
    observed_members = 1 if initial_params.ndim == 1 else int(initial_params.shape[0])
    if observed_members != len(expected_roles):
        raise StudyValidationError(f"initial parameter population has wrong size: {run_id}")
    if recorded_hashes != _parameter_hashes(initial_params):
        raise StudyValidationError(f"initial parameter hashes mismatch: {run_id}")

    problem = _require_mapping(record.get("problem"), f"problem {run_id}")
    topology = str(problem.get("topology_string", ""))
    if not topology or problem.get("topology_sha256") != hashlib.sha256(
        topology.encode()
    ).hexdigest():
        raise StudyValidationError(f"topology identity/hash mismatch: {run_id}")
    configured_topology = expected_config.get("topology")
    if (
        isinstance(configured_topology, dict)
        and configured_topology.get("kind") == "string"
        and topology != configured_topology.get("value")
    ):
        raise StudyValidationError(
            f"resolved topology differs from explicit plan: {run_id}"
        )
    objective = _require_mapping(
        record.get("objective_configuration"), f"Objective configuration {run_id}"
    )
    if (
        objective.get("max_time_seconds") != expected_config["max_time_seconds"]
        or objective.get("max_evals") != expected_config["max_evals"]
    ):
        raise StudyValidationError(f"Objective budget mismatch: {run_id}")
    algorithm = _require_mapping(record.get("algorithm"), f"algorithm {run_id}")
    if strict_json(algorithm) != strict_json(_expected_algorithm_record(expected_config)):
        raise StudyValidationError(f"algorithm configuration mismatch: {run_id}")
    telemetry_mode = expected_config.get("optimizer_telemetry")
    if telemetry_mode is None and record.get("optimizer_telemetry") is not None:
        raise StudyValidationError(f"unsolicited optimizer telemetry: {run_id}")
    if telemetry_mode is not None:
        raise StudyValidationError(
            f"external telemetry validation is not supported by this archive path: {run_id}"
        )

    accounting = _require_mapping(
        record.get("objective_accounting"), f"Objective accounting {run_id}"
    )
    expected_eval_count = max(int(row["eval_count_after_call"]) for row in rows)
    if (
        accounting.get("log_call_count") != recomputed_metrics["logged_calls"]
        or accounting.get("eval_count") != expected_eval_count
    ):
        raise StudyValidationError(f"Objective accounting mismatch: {run_id}")

    process = _require_mapping(record.get("worker_process"), f"worker process {run_id}")
    if process.get("returncode") != 0 or process.get("timed_out") is not False:
        raise StudyValidationError(f"worker did not exit cleanly: {run_id}")
    if process.get("within_official_4h30_container_limit") is not True:
        raise StudyValidationError(f"worker exceeded container limit: {run_id}")
    wall = process.get("full_wall_seconds")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
        raise StudyValidationError(f"worker wall time is invalid: {run_id}")
    for stream in ("stdout", "stderr"):
        metadata = _require_mapping(process.get(stream), f"{stream} metadata {run_id}")
        member = f"logs/{run_id}.{stream}.log"
        payload = _read_member(archive, member)
        if metadata.get("path") != member or metadata.get("sha256") != sha256_bytes(payload):
            raise StudyValidationError(f"worker {stream} evidence mismatch: {run_id}")
    return rows


def _expected_archive_members(run_ids: Iterable[str]) -> set[str]:
    members = set(FIXED_MEMBERS) | set(PREFLIGHT_MEMBERS)
    for run_id in run_ids:
        members.update(
            {
                f"configs/{run_id}.json",
                f"histories/{run_id}.npz",
                f"logs/{run_id}.stdout.log",
                f"logs/{run_id}.stderr.log",
                f"runs/{run_id}.json",
            }
        )
    return members


def validate_study_archive(
    sources: SourcePaths,
    expected: ExpectedSources = DEVELOPMENT_V2_SOURCES,
    limits: ArchiveLimits = ArchiveLimits(),
) -> ValidatedStudy:
    """Authenticate and validate all raw evidence except ``summary.json`` content."""
    source_hashes, package_manifest, external_plan = verify_external_sources(
        sources, expected
    )
    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    if package_manifest["archive"].get("files") != len(member_names):
        raise StudyValidationError("package manifest member count mismatch")

    with zipfile.ZipFile(sources.archive, "r") as archive:
        manifest = _load_json_member(archive, "manifest.json")
        package_state = _load_json_member(archive, "package-state.json")
        session = _load_json_member(archive, "session.json")
        for key in (
            "format_version",
            "plan_id",
            "configuration",
            "run_order_policy",
            "primary_pair_order",
            "runs",
        ):
            if strict_json(manifest.get(key)) != strict_json(external_plan.get(key)):
                raise StudyValidationError(f"internal manifest/external plan mismatch: {key}")
        expected_configs = _validate_plan_contract(manifest, expected)
        _validate_environment(manifest)

        if package_state != {
            "format_version": 1,
            "study_complete": True,
            "planned_runs": 64,
            "completed_runs": 64,
            "incomplete_runs": [],
        }:
            raise StudyValidationError("package-state.json does not prove 64/64 completion")
        if session.get("status") != "complete":
            raise StudyValidationError("session did not complete cleanly")
        elapsed = session.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise StudyValidationError("session elapsed time is invalid")

        expected_members = _expected_archive_members(expected_configs)
        observed_members = set(member_names)
        missing = sorted(expected_members - observed_members)
        unexpected = sorted(observed_members - expected_members)
        if missing or unexpected:
            raise StudyValidationError(
                f"archive member set mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )

        configs: dict[str, dict[str, object]] = {}
        records_by_id: dict[str, dict[str, object]] = {}
        history_rows: dict[str, list[dict[str, object]]] = {}
        expected_environment = _require_mapping(
            manifest.get("environment"), "runtime environment"
        )
        resolved_topology_by_config: dict[str, str] = {}
        for run_id, expected_config in expected_configs.items():
            config = _load_json_member(archive, f"configs/{run_id}.json")
            if strict_json(config) != strict_json(expected_config):
                raise StudyValidationError(f"config artifact mismatch: {run_id}")
            record = _load_json_member(archive, f"runs/{run_id}.json")
            if str(record.get("run_id")) in records_by_id:
                raise StudyValidationError(f"duplicate run record ID: {record.get('run_id')}")
            configs[run_id] = config
            records_by_id[run_id] = record
            history_rows[run_id] = _validate_record(
                record, expected_config, archive, expected_environment
            )
            configured = json.dumps(
                expected_config["topology"], sort_keys=True, separators=(",", ":")
            )
            topology_hash = str(record["problem"]["topology_sha256"])
            previous = resolved_topology_by_config.setdefault(configured, topology_hash)
            if previous != topology_hash:
                raise StudyValidationError(
                    "one planned topology resolved to multiple topology hashes"
                )

        records = [records_by_id[run_id] for run_id in sorted(records_by_id)]
        jsonl_payload = _read_member(archive, "runs.jsonl")
        try:
            lines = jsonl_payload.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as error:
            raise StudyValidationError("runs.jsonl is not UTF-8") from error
        if len(lines) != 64:
            raise StudyValidationError("runs.jsonl must contain exactly 64 records")
        jsonl_records = [
            _require_mapping(strict_json_loads(line, f"runs.jsonl line {index}"), "run")
            for index, line in enumerate(lines, start=1)
        ]
        if strict_json(jsonl_records) != strict_json(records):
            raise StudyValidationError("runs.jsonl does not match per-run records")

        topology_hashes = {
            str(record["problem"]["topology_sha256"]) for record in records
        }
        if len(resolved_topology_by_config) != 16 or len(topology_hashes) != 16:
            raise StudyValidationError("incorrect topology hierarchy; expected n=16")
        hierarchy: dict[tuple[str, int], set[str]] = {}
        for record in records:
            config = record["config"]
            key = (
                str(record["problem"]["topology_sha256"]),
                int(config["optimizer_seed"]),
            )
            hierarchy.setdefault(key, set()).add(str(config["arm"]))
        if len(hierarchy) != 32 or any(
            arms != {"no_prior", "semantic_prior"} for arms in hierarchy.values()
        ):
            raise StudyValidationError("broken topology/seed/arm pairing hierarchy")

    integrity = {
        **{key: value for key, value in integrity.items() if key != "member_names"},
        "external_hashes": "passed",
        "sidecar_filename_and_digest": "passed",
        "records": 64,
        "histories": 64,
        "configs": 64,
        "stdout_logs": 64,
        "stderr_logs": 64,
        "topologies": 16,
        "optimizer_seed_pairs": 32,
        "summary_content_opened": False,
    }
    return ValidatedStudy(
        sources=sources,
        source_hashes=source_hashes,
        archive_members=member_names,
        plan=external_plan,
        manifest=manifest,
        package_state=package_state,
        session=session,
        configs=configs,
        records=records,
        history_rows=history_rows,
        integrity=integrity,
    )


def load_summary_after_reproduction(
    study: ValidatedStudy,
    reproduction_agreement: dict[str, object],
) -> dict[str, object]:
    """Open the summary only with evidence that both raw replays already agreed."""
    if (
        reproduction_agreement.get("status") != "matched"
        or reproduction_agreement.get("topology_values_compared") != 16
        or reproduction_agreement.get("target_thresholds_compared") != 4
    ):
        raise StudyValidationError(
            "summary remains locked until production/reference replay agreement"
        )
    with zipfile.ZipFile(study.sources.archive, "r") as archive:
        return _load_json_member(archive, "summary.json")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_normalized_tables(
    study: ValidatedStudy,
    output_dir: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Write hierarchical machine-readable tables without candidate arrays."""
    resolved_output = output_dir.resolve()
    if repository_root is not None:
        root = repository_root.resolve()
        if resolved_output == root or root in resolved_output.parents:
            raise StudyValidationError("generated analysis output must stay outside Git")
    resolved_output.mkdir(parents=True, exist_ok=False)

    run_rows = []
    history_output_rows = []
    target_rows = []
    pair_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    topology_groups: dict[str, list[dict[str, object]]] = {}
    for record in study.records:
        config = record["config"]
        metrics = record["metrics"]
        problem = record["problem"]
        run_id = str(record["run_id"])
        topology_hash = str(problem["topology_sha256"])
        seed = int(config["optimizer_seed"])
        run_rows.append(
            {
                "run_id": run_id,
                "pair_id": config["pair_id"],
                "topology_sha256": topology_hash,
                "topology_string": problem["topology_string"],
                "optimizer_seed": seed,
                "arm": config["arm"],
                "planned_run_index": config["planned_run_index"],
                "run_order_within_pair": config["run_order_within_pair"],
                "has_feasible": metrics["has_feasible"],
                "best_feasible_loss": metrics["best_feasible_loss"],
                "logged_calls": metrics["logged_calls"],
                "logged_candidates": metrics["logged_candidates"],
                "worker_wall_seconds": record["worker_process"]["full_wall_seconds"],
            }
        )
        pair_groups.setdefault((str(config["pair_id"]), seed), []).append(record)
        topology_groups.setdefault(topology_hash, []).append(record)
        for row in study.history_rows[run_id]:
            history_output_rows.append(
                {
                    "run_id": run_id,
                    "pair_id": config["pair_id"],
                    "topology_sha256": topology_hash,
                    "optimizer_seed": seed,
                    "arm": config["arm"],
                    **row,
                }
            )
        for target, hit in metrics["targets"].items():
            target_rows.append(
                {
                    "run_id": run_id,
                    "pair_id": config["pair_id"],
                    "topology_sha256": topology_hash,
                    "optimizer_seed": seed,
                    "arm": config["arm"],
                    "target_loss": target,
                    "reached": hit["time_seconds"] is not None,
                    "time_seconds": hit["time_seconds"],
                    "eval_count": hit["eval_count"],
                    "censor_time_seconds": metrics["last_logged_time_seconds"],
                    "censor_eval_count": metrics["last_logged_eval_count"],
                }
            )

    pair_rows = []
    for (_, _), records in sorted(pair_groups.items()):
        by_arm = {str(record["config"]["arm"]): record for record in records}
        control = by_arm["no_prior"]
        treatment = by_arm["semantic_prior"]
        c_loss = control["metrics"]["best_feasible_loss"]
        t_loss = treatment["metrics"]["best_feasible_loss"]
        pair_rows.append(
            {
                "pair_id": control["config"]["pair_id"],
                "topology_sha256": control["problem"]["topology_sha256"],
                "optimizer_seed": control["config"]["optimizer_seed"],
                "first_arm": (
                    "no_prior"
                    if control["config"]["run_order_within_pair"]
                    < treatment["config"]["run_order_within_pair"]
                    else "semantic_prior"
                ),
                "no_prior_has_feasible": control["metrics"]["has_feasible"],
                "semantic_prior_has_feasible": treatment["metrics"]["has_feasible"],
                "no_prior_best_feasible_loss": c_loss,
                "semantic_prior_best_feasible_loss": t_loss,
                "difference_semantic_minus_no_prior": (
                    None if c_loss is None or t_loss is None else float(t_loss) - float(c_loss)
                ),
            }
        )

    topology_rows = []
    for topology_hash, records in sorted(topology_groups.items()):
        by_seed_arm = {
            (int(record["config"]["optimizer_seed"]), str(record["config"]["arm"])): record
            for record in records
        }
        differences = []
        for seed in (7, 11):
            control = by_seed_arm[(seed, "no_prior")]["metrics"]["best_feasible_loss"]
            treatment = by_seed_arm[(seed, "semantic_prior")]["metrics"]["best_feasible_loss"]
            if control is not None and treatment is not None:
                differences.append(float(treatment) - float(control))
        topology_rows.append(
            {
                "topology_sha256": topology_hash,
                "topology_string": records[0]["problem"]["topology_string"],
                "optimizer_seeds": "7;11",
                "optimizer_seed_pairs": 2,
                "mean_seed_difference_semantic_minus_no_prior": (
                    sum(differences) / len(differences) if len(differences) == 2 else None
                ),
                "inference_unit": "topology",
            }
        )

    table_specs = {
        "runs.csv": (
            list(run_rows[0]),
            run_rows,
            "One row per run; 64 repeated measurements, not 64 inference units.",
        ),
        "topology_seed_pairs.csv": (
            list(pair_rows[0]),
            pair_rows,
            "One row per topology/optimizer-seed pair; seeds remain repeated measures.",
        ),
        "topologies.csv": (
            list(topology_rows[0]),
            topology_rows,
            "One row per topology; this is the primary inference table (n=16).",
        ),
        "history_rows.csv": (
            list(history_output_rows[0]),
            history_output_rows,
            "Evaluation/timing history rows nested under runs; never independent units.",
        ),
        "target_hits.csv": (
            list(target_rows[0]),
            target_rows,
            "Target events and explicit censoring horizons nested under runs.",
        ),
    }
    dictionary = {
        "format_version": 1,
        "primary_inference_unit": "topology",
        "topology_count": 16,
        "privacy": (
            "Generated outside Git; excludes initial candidate arrays, raw logs, GPU UUIDs, "
            "secrets, and provider-local absolute paths."
        ),
        "tables": {},
    }
    row_counts = {}
    for filename, (fields, rows, description) in table_specs.items():
        row_counts[filename] = _write_csv(resolved_output / filename, fields, rows)
        dictionary["tables"][filename] = {
            "description": description,
            "columns": fields,
            "rows": row_counts[filename],
        }
    (resolved_output / "data_dictionary.json").write_text(
        json.dumps(dictionary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    normalized_manifest = {
        "format_version": 1,
        "source_sha256": study.source_hashes,
        "plan_id": study.manifest["plan_id"],
        "project_revision": study.manifest["project_revision"],
        "row_counts": row_counts,
        "primary_inference_unit": "topology",
    }
    (resolved_output / "normalized_manifest.json").write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return normalized_manifest
