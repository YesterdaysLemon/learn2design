"""Deterministic, append-only stage archive packaging and authentication."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

from .canonical import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
    sha256_file,
    exclusive_write_bytes,
)
from .contract import POPULATION_SIZE, STUDY_ID, arm_spec


class ArchiveError(RuntimeError):
    pass


FIXED_ZIP_TIME = (2026, 9, 1, 0, 0, 0)
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
HISTORY_SCHEMA = [
    "call_index",
    "candidate_index",
    "eval_count_after_call",
    "time_seconds",
    "loss",
    "sensitivity_loss",
    "penalty",
    "is_feasible",
]


def _array_hash(array: object) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_population_and_algorithm(
    record: dict[str, object], history_bytes: bytes
) -> None:
    import numpy as np

    if set(record) != {
        "run_id",
        "config",
        "algorithm",
        "topology_sha256",
        "initial_population",
        "history",
        "metrics",
        "objective_accounting",
        "worker_measurement",
    }:
        raise ArchiveError("worker record schema mismatch")
    config = record["config"]
    if not isinstance(config, dict):
        raise ArchiveError("worker record config is absent")
    spec = arm_spec(config.get("arm_id"))
    expected_algorithm = {
        "logical_module_id": spec.logical_module_id,
        "python_module_name": spec.python_module_name,
        "class_name": spec.class_name,
        "algorithm_str": spec.algorithm_str,
        "source_sha256": spec.source_sha256,
        "kwargs": {
            **spec.fixed_kwargs(),
            "random_seed": config.get("optimizer_seed"),
        },
    }
    if record["algorithm"] != expected_algorithm:
        raise ArchiveError("worker algorithm binding mismatch")
    history = record["history"]
    if not isinstance(history, dict) or set(history) != {
        "sha256",
        "rows",
        "schema",
    }:
        raise ArchiveError("worker history receipt schema mismatch")
    if history["schema"] != HISTORY_SCHEMA:
        raise ArchiveError("worker history schema declaration mismatch")
    with np.load(io.BytesIO(history_bytes), allow_pickle=False) as loaded:
        initial = np.asarray(loaded["initial_params_unbounded"])
        raw = np.asarray(loaded["raw_params_unbounded"])
    if (
        initial.ndim != 2
        or initial.shape[0] != POPULATION_SIZE
        or not np.all(np.isfinite(initial))
        or raw.shape != initial.shape
        or raw.dtype != initial.dtype
        or not np.all(np.isfinite(raw))
    ):
        raise ArchiveError("worker initial population artifact is invalid")
    population = record["initial_population"]
    if not isinstance(population, dict) or set(population) != {
        "raw_population_sha256",
        "raw_member_sha256",
        "initial_population_sha256",
        "initial_member_sha256",
        "warmup_enabled",
        "warmup_source_proof",
        "before_warmup",
        "after_warmup",
    }:
        raise ArchiveError("initial-population receipt schema mismatch")
    initial_members = [_array_hash(member) for member in initial]
    raw_members_from_artifact = [_array_hash(member) for member in raw]
    if (
        population["initial_population_sha256"] != _array_hash(initial)
        or population["initial_member_sha256"] != initial_members
        or population["raw_population_sha256"] != _array_hash(raw)
        or population["raw_member_sha256"] != raw_members_from_artifact
    ):
        raise ArchiveError("initial-population receipt differs from NPZ")
    raw_members = population["raw_member_sha256"]
    raw_population_sha256 = population["raw_population_sha256"]
    if (
        not isinstance(raw_population_sha256, str)
        or len(raw_population_sha256) != 64
        or any(token not in "0123456789abcdef" for token in raw_population_sha256)
        or not isinstance(raw_members, list)
        or len(raw_members) != POPULATION_SIZE
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(token not in "0123456789abcdef" for token in value)
            for value in raw_members
        )
    ):
        raise ArchiveError("raw-population receipt digest schema mismatch")
    if (
        population["warmup_enabled"] is not spec.preclock_warmup
        or population["before_warmup"] != population["after_warmup"]
    ):
        raise ArchiveError("warmup receipt mismatch")
    snapshot = population["before_warmup"]
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "random_draw_calls",
        "rng_seed",
        "rng_key_sha256",
        "parameter_sha256",
        "eval_count",
        "log_call_count",
        "time_elapsed",
        "time_left",
        "evals_left",
        "max_time",
        "max_evals",
        "time_offset",
        "start_time_is_none",
        "histories",
        "budget_exceeded",
        "time_exceeded",
        "evals_exceeded",
    }:
        raise ArchiveError("warmup snapshot schema mismatch")
    proof = population["warmup_source_proof"]
    if (
        not isinstance(proof, dict)
        or set(proof) != {"source_sha256", "calls", "random_sampling_calls"}
        or not isinstance(proof["source_sha256"], str)
        or len(proof["source_sha256"]) != 64
        or not isinstance(proof["calls"], list)
        or not proof["calls"]
        or proof["random_sampling_calls"] != 0
    ):
        raise ArchiveError("warmup source proof mismatch")
    histories = snapshot["histories"]
    expected_histories = {
        name: 0
        for name in (
            "time_steps",
            "loss_history",
            "sensitivity_loss_history",
            "penalty_history",
            "is_feasible_history",
            "eval_type_history",
            "params_history",
            "grad_history",
            "hessian_history",
            "power_values_history",
            "violations_history",
        )
    }
    if (
        snapshot["random_draw_calls"] != 1
        or snapshot["rng_seed"] != config.get("optimizer_seed")
        or not isinstance(snapshot["rng_key_sha256"], str)
        or len(snapshot["rng_key_sha256"]) != 64
        or snapshot["parameter_sha256"] != population["initial_population_sha256"]
        or snapshot["eval_count"] != 0
        or snapshot["log_call_count"] != 0
        or snapshot["time_elapsed"] != 0.0
        or snapshot["time_left"] != 600.0
        or snapshot["evals_left"] is not None
        or snapshot["max_time"] != 600.0
        or snapshot["max_evals"] is not None
        or snapshot["time_offset"] != 0.0
        or snapshot["start_time_is_none"] is not True
        or histories != expected_histories
        or snapshot["budget_exceeded"] is not False
        or snapshot["time_exceeded"] is not False
        or snapshot["evals_exceeded"] is not False
    ):
        raise ArchiveError("warmup snapshot is not state-neutral")
    if spec.population_mode == "coverage_balanced":
        if not np.array_equal(initial[0], raw[0]):
            raise ArchiveError("coverage-balanced transform changed the anchor row")
        suffix = initial[1:]
        unit = (
            np.arange(len(suffix), dtype=suffix.dtype)
            + np.asarray(0.5, dtype=suffix.dtype)
        ) / np.asarray(len(suffix), dtype=suffix.dtype)
        expected = np.broadcast_to(
            (np.log(unit) - np.log1p(-unit))[:, None], suffix.shape
        )
        try:
            np.testing.assert_array_max_ulp(
                np.sort(suffix, axis=0), expected, maxulp=4
            )
        except AssertionError as error:
            raise ArchiveError("coverage-balanced suffix levels mismatch") from error
    elif not np.array_equal(initial, raw):
        raise ArchiveError("random population arm changed the authenticated raw draw")


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ArchiveError("archive member path is invalid")
    return relative


def _source_rows(stage_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(stage_dir.rglob("*")):
        if path.is_symlink():
            raise ArchiveError("stage archive rejects symlinks")
        if not path.is_file():
            continue
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ArchiveError("stage archive rejects special files")
        size = path.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise ArchiveError("stage archive member exceeds size cap")
        rows.append(
            {
                "name": _safe_relative(path, stage_dir),
                "sha256": sha256_file(path),
                "size_bytes": size,
            }
        )
    if not rows:
        raise ArchiveError("stage archive has no evidence members")
    if len({row["name"] for row in rows}) != len(rows):
        raise ArchiveError("stage archive member names are not unique")
    return rows


def seal_stage_archive(
    stage_dir: Path,
    archive_path: Path,
    *,
    stage: int,
    ordered_run_ids: list[str],
) -> dict[str, object]:
    if stage not in (1, 2):
        raise ArchiveError("archive stage is invalid")
    if archive_path.exists() or archive_path.with_name(
        archive_path.name + ".sha256"
    ).exists():
        raise ArchiveError("append-only archive target already exists")
    rows = _source_rows(stage_dir)
    manifest = {
        "format_version": 1,
        "stage": stage,
        "ordered_run_ids": ordered_run_ids,
        "members": rows,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        manifest_info = zipfile.ZipInfo("manifest.json", FIXED_ZIP_TIME)
        manifest_info.create_system = 3
        manifest_info.external_attr = 0o100600 << 16
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(manifest_info, manifest_bytes)
        for row in rows:
            info = zipfile.ZipInfo(str(row["name"]), FIXED_ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (stage_dir / str(row["name"])).read_bytes())
    if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
        temporary.unlink(missing_ok=True)
        raise ArchiveError("stage archive exceeds total size cap")
    # Windows does not permit fsync on a read-only CRT file descriptor.
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(temporary, archive_path)
    except FileExistsError as error:
        raise ArchiveError("append-only archive target already exists") from error
    temporary.unlink()
    if os.name == "posix":
        descriptor = os.open(
            archive_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    digest = sha256_file(archive_path)
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    exclusive_write_bytes(
        sidecar, f"{digest}  {archive_path.name}\n".encode("ascii")
    )
    return {
        "archive_sha256": digest,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "member_count": len(rows) + 1,
    }


def inspect_stage_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_stage: int,
    expected_run_ids: list[str],
) -> dict[str, object]:
    if sha256_file(archive_path) != expected_sha256:
        raise ArchiveError("stage archive digest mismatch")
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    expected_sidecar = f"{expected_sha256}  {archive_path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        raise ArchiveError("stage archive sidecar mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or not names or names[0] != "manifest.json":
            raise ArchiveError("stage archive member order/uniqueness mismatch")
        if any(
            info.is_dir()
            or info.flag_bits & 0x1
            or info.file_size > MAX_MEMBER_BYTES
            or PurePosixPath(info.filename).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(info.filename).parts)
            for info in infos
        ):
            raise ArchiveError("stage archive contains an unsafe member")
        manifest_bytes = archive.read("manifest.json")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArchiveError("stage manifest is invalid JSON") from error
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise ArchiveError("stage manifest is not canonical")
        if not isinstance(manifest, dict) or set(manifest) != {
            "format_version",
            "stage",
            "ordered_run_ids",
            "members",
        }:
            raise ArchiveError("stage manifest schema mismatch")
        if (
            manifest["format_version"] != 1
            or manifest["stage"] != expected_stage
            or manifest["ordered_run_ids"] != expected_run_ids
        ):
            raise ArchiveError("stage manifest identity mismatch")
        rows = manifest["members"]
        if not isinstance(rows, list) or [row.get("name") for row in rows] != names[1:]:
            raise ArchiveError("stage manifest member list mismatch")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "name",
                "sha256",
                "size_bytes",
            }:
                raise ArchiveError("stage manifest row schema mismatch")
            content = archive.read(str(row["name"]))
            if len(content) != row["size_bytes"] or sha256_bytes(content) != row["sha256"]:
                raise ArchiveError("stage archive member digest mismatch")
    return manifest


def load_stage_archive_evidence(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_stage: int,
    expected_run_ids: list[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Reopen only sealed bytes and bind configs, records, histories, and logs."""
    inspect_stage_archive(
        archive_path,
        expected_sha256=expected_sha256,
        expected_stage=expected_stage,
        expected_run_ids=expected_run_ids,
    )
    expected_names = {
        f"{directory}/{run_id}.{suffix}"
        for run_id in expected_run_ids
        for directory, suffix in (
            ("configs", "json"),
            ("records", "json"),
            ("documents", "json"),
            ("histories", "npz"),
            ("logs", "stdout.bin"),
            ("logs", "stderr.bin"),
        )
    }
    documents: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    reference_packets: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        from .analysis import load_history_npz_bytes

        observed = set(archive.namelist()) - {"manifest.json"}
        if observed != expected_names:
            raise ArchiveError("stage archive evidence layout mismatch")
        for run_id in expected_run_ids:
            config = parse_canonical_json(archive.read(f"configs/{run_id}.json"))
            record = parse_canonical_json(archive.read(f"records/{run_id}.json"))
            document = parse_canonical_json(archive.read(f"documents/{run_id}.json"))
            history = archive.read(f"histories/{run_id}.npz")
            stdout = archive.read(f"logs/{run_id}.stdout.bin")
            stderr = archive.read(f"logs/{run_id}.stderr.bin")
            if not all(isinstance(value, dict) for value in (config, record, document)):
                raise ArchiveError("stage archive JSON object schema mismatch")
            if (
                config.get("run_id") != run_id
                or record.get("run_id") != run_id
                or document.get("run_id") != run_id
                or record.get("config") != config
                or document.get("config") != config
            ):
                raise ArchiveError("stage archive run/config binding mismatch")
            history_row = record.get("history")
            if (
                not isinstance(history_row, dict)
                or history_row.get("sha256") != sha256_bytes(history)
                or history_row.get("rows")
                != len(document.get("history_rows", []))
            ):
                raise ArchiveError("stage archive history binding mismatch")
            if document.get("history_rows") != load_history_npz_bytes(history):
                raise ArchiveError("stage document rows differ from NPZ bytes")
            _validate_population_and_algorithm(record, history)
            if (
                record.get("metrics") != document.get("metrics")
                or record.get("objective_accounting")
                != document.get("objective_accounting")
            ):
                raise ArchiveError("stage archive record/document projection mismatch")
            runtime = document.get("runtime")
            if (
                not isinstance(runtime, dict)
                or runtime.get("stdout_bytes") != len(stdout)
                or runtime.get("stderr_bytes") != len(stderr)
                or stderr != b""
            ):
                raise ArchiveError("stage archive process-log binding mismatch")
            packet = parse_canonical_json(stdout)
            if not isinstance(packet, dict) or set(packet) != {
                "record",
                "run_id",
                "schema_version",
                "study_id",
            }:
                raise ArchiveError("stage archive stdout packet schema mismatch")
            if (
                packet["record"] != record
                or packet["run_id"] != run_id
                or packet["schema_version"] != 1
                or packet["study_id"] != STUDY_ID
            ):
                raise ArchiveError("stage archive stdout packet binding mismatch")
            documents.append(document)
            records.append(record)
            reference_packets.append(
                {"config": config, "history_bytes": history}
            )
    return documents, records, reference_packets
