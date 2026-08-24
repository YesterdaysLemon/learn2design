"""Outcome-blind ingestion for ``coverage-robustness-screen-v1``.

The source-lock digest is authenticated before its JSON is parsed.  Complete
archives are replayed from pickle-free histories while ``summary.json`` stays
sealed.  Terminal partials are classified structurally without opening run
outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from experiments.uifo_paired.coverage_evidence import (
    CoverageReplayAgreement,
    coverage_study_identity_sha256,
)
from experiments.uifo_paired.plan import TOPOLOGY_PATTERN
from experiments.uifo_paired.results_ingestion import (
    ArchiveLimits,
    ExpectedSources,
    FIXED_MEMBERS,
    PREFLIGHT_MEMBERS,
    SIDECAR_PATTERN,
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
    _expected_archive_members,
    _load_history_arrays,
    _load_json_member,
    _read_member,
    _require_mapping,
    _validate_record,
    inspect_zip_integrity,
    sha256_path,
    strict_json_loads,
)
from experiments.uifo_paired.runner import (
    H100_CUDA13_PACKAGE_VERSIONS,
    JAX_RUNTIME_ENVIRONMENT_KEYS,
    _parameter_hashes,
    _run_config,
    strict_json,
)
from experiments.uifo_paired.study_profiles import bind_study_profile


PROFILE = "coverage-robustness-screen-v1"
PANEL_ID = "coverage-robustness-v1"
POLICY_ID = "coverage-robustness-development-screen-v1"
RUNS = 48
PAIRS = 24
TOPOLOGIES = 12
SEEDS = {37, 41}
ARMS = {"no_prior", "coverage_balanced"}
ARCHIVE_MEMBERS = 249
SESSION_WALL_SECONDS = 22 * 60 * 60.0
WORKER_TIMEOUT_SECONDS = 2_100.0
EXACT_GPU_NAME = "NVIDIA H100 80GB HBM3"
UPSTREAM_REFERENCE = "1bb7f54737dec6a08b59879a8831d125f08f8a0b"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PANEL_PATH = Path(__file__).with_name("panels") / "coverage-robustness-v1.json"
RECOVERY_RECEIPT = re.compile(r"^recovery/stale-study-lock-([0-9a-f]{12})\.json$")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise StudyValidationError(f"{label} is missing or is not a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise StudyValidationError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StudyValidationError(f"{label} must be expressed in UTC")
    return parsed.astimezone(UTC)


def _validate_utc_interval(value: dict[str, object], label: str) -> float:
    started = _parse_utc_timestamp(value.get("started_utc"), f"{label} start")
    completed = _parse_utc_timestamp(
        value.get("completed_utc"), f"{label} completion"
    )
    if completed <= started:
        raise StudyValidationError(f"{label} timestamps are reversed or empty")
    return (completed - started).total_seconds()


def _coverage_logits_from_ranks(ranks, member_count: int, dtype):
    import numpy as np

    floating = np.dtype(dtype)
    unit = (
        np.asarray(ranks, dtype=floating)
        + np.asarray(0.5, dtype=floating)
    ) / np.asarray(member_count, dtype=floating)
    return np.log(unit) - np.log1p(-unit)


def _require_coverage_ulp_match(actual, expected, label: str) -> None:
    import numpy as np

    observed = np.asarray(actual)
    target = np.asarray(expected, dtype=observed.dtype)
    if observed.shape != target.shape:
        raise StudyValidationError(label)
    try:
        np.testing.assert_array_max_ulp(observed, target, maxulp=4)
    except AssertionError as error:
        raise StudyValidationError(label) from error


def _pair_id(topology: str, seed: int) -> str:
    digest = hashlib.sha256(topology.encode()).hexdigest()[:12]
    return f"topo{digest}__oseed{seed:010d}"


def authenticate_coverage_source_lock(
    source_lock_path: Path,
    *,
    expected_source_lock_sha256: str,
    sources: SourcePaths,
    terminal_attempt_receipt: Path,
) -> ExpectedSources:
    """Authenticate the out-of-band lock digest before parsing its JSON."""
    if SHA256.fullmatch(expected_source_lock_sha256) is None:
        raise StudyValidationError("expected source-lock SHA-256 is malformed")
    if not source_lock_path.is_file():
        raise StudyValidationError("source-lock file is missing")
    if sha256_path(source_lock_path) != expected_source_lock_sha256:
        raise StudyValidationError("source-lock SHA-256 mismatch")

    lock = _require_mapping(
        strict_json_loads(source_lock_path.read_bytes(), "coverage source lock"),
        "coverage source lock",
    )
    if set(lock) != {
        "format_version",
        "study_profile",
        "plan_id",
        "project_revision",
        "files",
    }:
        raise StudyValidationError("coverage source-lock schema mismatch")
    if lock.get("format_version") != 1 or lock.get("study_profile") != PROFILE:
        raise StudyValidationError("coverage source-lock profile or version mismatch")
    plan_id = lock.get("plan_id")
    revision = lock.get("project_revision")
    if not isinstance(plan_id, str) or re.fullmatch(r"[0-9a-f]{16}", plan_id) is None:
        raise StudyValidationError("coverage source-lock plan ID is invalid")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise StudyValidationError("coverage source-lock revision is invalid")

    files = _require_mapping(lock.get("files"), "coverage source-lock files")
    paths = {
        sources.archive.name: sources.archive,
        sources.checksum.name: sources.checksum,
        sources.package_manifest.name: sources.package_manifest,
        sources.plan.name: sources.plan,
        terminal_attempt_receipt.name: terminal_attempt_receipt,
    }
    if set(files) != set(paths):
        raise StudyValidationError("coverage source-lock basenames do not match inputs")
    for basename, path in paths.items():
        entry = files.get(basename)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise StudyValidationError(f"invalid source-lock file entry: {basename}")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise StudyValidationError(f"invalid source-lock digest: {basename}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StudyValidationError(f"invalid source-lock size: {basename}")
        if (
            not path.is_file()
            or path.stat().st_size != size
            or sha256_path(path) != digest
        ):
            raise StudyValidationError(f"coverage source-lock file mismatch: {basename}")

    return ExpectedSources(
        zip_sha256=str(files[sources.archive.name]["sha256"]),
        package_manifest_sha256=str(
            files[sources.package_manifest.name]["sha256"]
        ),
        checksum_file_sha256=str(files[sources.checksum.name]["sha256"]),
        plan_sha256=str(files[sources.plan.name]["sha256"]),
        plan_id=plan_id,
        project_revision=revision,
    )


def _panel_topologies() -> list[str]:
    payload = _require_mapping(
        strict_json_loads(PANEL_PATH.read_bytes(), "committed coverage panel"),
        "committed coverage panel",
    )
    topologies = payload.get("topologies")
    if payload.get("panel_id") != PANEL_ID or not isinstance(topologies, list):
        raise StudyValidationError("committed coverage panel is invalid")
    if len(topologies) != TOPOLOGIES or not all(
        isinstance(value, str) and TOPOLOGY_PATTERN.fullmatch(value) is not None
        for value in topologies
    ):
        raise StudyValidationError("committed coverage panel topology mismatch")
    return [str(value) for value in topologies]


def _validate_plan(
    plan: dict[str, object], expected: ExpectedSources
) -> dict[str, dict[str, object]]:
    if plan.get("format_version") != 1 or plan.get("plan_id") != expected.plan_id:
        raise StudyValidationError("coverage plan identity mismatch")
    configuration = _require_mapping(plan.get("configuration"), "plan configuration")
    if configuration.get("study_profile") != PROFILE:
        raise StudyValidationError("coverage study profile mismatch")
    try:
        policy = bind_study_profile(PROFILE, configuration)
    except ValueError as error:
        raise StudyValidationError(f"coverage frozen profile mismatch: {error}") from error
    if strict_json(configuration.get("decision_policy")) != strict_json(policy):
        raise StudyValidationError("coverage frozen decision policy mismatch")
    if _require_mapping(policy, "decision policy").get("policy_id") != POLICY_ID:
        raise StudyValidationError("coverage decision policy ID mismatch")

    committed = _panel_topologies()
    expected_specs = [{"kind": "string", "value": value} for value in committed]
    if configuration.get("topologies") != expected_specs:
        raise StudyValidationError("coverage panel contents or order drifted")
    panel_evidence = _require_mapping(
        configuration.get("topology_panel"), "coverage topology panel evidence"
    )
    if panel_evidence != {
        "panel_id": PANEL_ID,
        "topology_count": TOPOLOGIES,
        "source_sha256": sha256_path(PANEL_PATH),
        "archive_exclusion_verified": True,
    }:
        raise StudyValidationError(
            "coverage plan is not bound to the exact committed panel bytes"
        )
    evidence = _require_mapping(
        configuration.get("candidate_package_evidence"),
        "candidate package evidence",
    )
    if evidence.get("project_revision") != expected.project_revision:
        raise StudyValidationError("candidate package revision is not study-bound")
    if evidence.get("upstream_reference") != UPSTREAM_REFERENCE:
        raise StudyValidationError("candidate package upstream reference drifted")

    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != RUNS or not all(
        isinstance(run, dict) for run in runs
    ):
        raise StudyValidationError("coverage plan must contain exactly 48 runs")
    run_rows = [run for run in runs if isinstance(run, dict)]
    if [run.get("planned_run_index") for run in run_rows] != list(range(RUNS)):
        raise StudyValidationError("coverage planned-run indexes drifted")
    if len({str(run.get("run_id")) for run in run_rows}) != RUNS:
        raise StudyValidationError("coverage plan contains duplicate run IDs")

    expected_run_keys = {
        "planned_run_index",
        "run_id",
        "pair_id",
        "run_order_within_pair",
        "topology",
        "optimizer_seed",
        "arm",
    }
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_topology: dict[str, set[int]] = defaultdict(set)
    for run in run_rows:
        if set(run) != expected_run_keys:
            raise StudyValidationError("coverage plan run schema drift")
        topology = _require_mapping(run.get("topology"), "run topology")
        value = topology.get("value")
        seed = run.get("optimizer_seed")
        arm = run.get("arm")
        if topology not in expected_specs or not isinstance(value, str):
            raise StudyValidationError("coverage run references unknown topology")
        if seed not in SEEDS or arm not in ARMS:
            raise StudyValidationError("coverage run seed or arm mismatch")
        pair_id = _pair_id(value, int(seed))
        if run.get("pair_id") != pair_id or run.get("run_id") != f"{pair_id}__{arm}":
            raise StudyValidationError("coverage run identity mismatch")
        if run.get("run_order_within_pair") not in {0, 1}:
            raise StudyValidationError("coverage within-pair order is invalid")
        by_pair[pair_id].append(run)
        by_topology[value].add(int(seed))

    if len(by_pair) != PAIRS or any(
        len(rows) != 2
        or {str(row["arm"]) for row in rows} != ARMS
        or {int(row["run_order_within_pair"]) for row in rows} != {0, 1}
        for rows in by_pair.values()
    ):
        raise StudyValidationError("coverage arm-pair hierarchy is broken")
    if set(by_topology) != set(committed) or any(
        seeds != SEEDS for seeds in by_topology.values()
    ):
        raise StudyValidationError("coverage topology/seed hierarchy is broken")

    pair_blocks = [run_rows[index : index + 2] for index in range(0, RUNS, 2)]
    first_sweep = pair_blocks[:TOPOLOGIES]
    second_sweep = pair_blocks[TOPOLOGIES:]
    if (
        [str(block[0]["topology"]["value"]) for block in first_sweep] != committed
        or [int(block[0]["optimizer_seed"]) for block in first_sweep]
        != [37] * TOPOLOGIES
        or [str(block[0]["topology"]["value"]) for block in second_sweep]
        != list(reversed(committed))
        or [int(block[0]["optimizer_seed"]) for block in second_sweep]
        != [41] * TOPOLOGIES
    ):
        raise StudyValidationError("coverage mirrored sweep order drifted")
    if any(block[0]["pair_id"] != block[1]["pair_id"] for block in pair_blocks):
        raise StudyValidationError("coverage pair runs are not contiguous")
    if plan.get("optimizer_seed_order_policy") != "mirrored_sweeps":
        raise StudyValidationError("coverage seed-order policy mismatch")
    if plan.get("run_order_policy") != (
        "alternate arm order by topology and optimizer-seed index"
    ):
        raise StudyValidationError("coverage run-order policy mismatch")
    required_order = {
        "complete_primary_pairs": 24,
        "no_prior_first": 12,
        "coverage_balanced_first": 12,
        "absolute_imbalance": 0,
    }
    recomputed_order = {
        "complete_primary_pairs": len(pair_blocks),
        "no_prior_first": sum(
            block[0]["arm"] == "no_prior" for block in pair_blocks
        ),
        "coverage_balanced_first": sum(
            block[0]["arm"] == "coverage_balanced" for block in pair_blocks
        ),
        "absolute_imbalance": abs(
            sum(block[0]["arm"] == "no_prior" for block in pair_blocks)
            - sum(
                block[0]["arm"] == "coverage_balanced"
                for block in pair_blocks
            )
        ),
    }
    if (
        recomputed_order != required_order
        or plan.get("primary_pair_order") != required_order
    ):
        raise StudyValidationError("coverage pair order is not exactly balanced")

    core = {
        "configuration": configuration,
        "run_order_policy": plan["run_order_policy"],
        "primary_pair_order": plan["primary_pair_order"],
        "runs": runs,
        "optimizer_seed_order_policy": plan["optimizer_seed_order_policy"],
    }
    recomputed = hashlib.sha256(_canonical(core).encode()).hexdigest()[:16]
    if recomputed != expected.plan_id:
        raise StudyValidationError("coverage plan ID does not match contents")
    return {str(run["run_id"]): _run_config(run, configuration) for run in run_rows}


def _validate_terminal_receipt(
    path: Path,
    *,
    expected: ExpectedSources,
    manifest: dict[str, object],
) -> None:
    receipt = _require_mapping(
        strict_json_loads(path.read_bytes(), "terminal attempt receipt"),
        "terminal attempt receipt",
    )
    if set(receipt) != {
        "format_version",
        "study_profile",
        "plan_id",
        "project_revision",
        "claimed_utc",
        "rule",
    }:
        raise StudyValidationError("terminal attempt receipt schema mismatch")
    if (
        receipt.get("format_version") != 1
        or receipt.get("study_profile") != PROFILE
        or receipt.get("plan_id") != expected.plan_id
        or receipt.get("project_revision") != expected.project_revision
        or not isinstance(receipt.get("claimed_utc"), str)
        or receipt.get("rule")
        != "first result-bearing attempt is terminal; resume and rerun forbidden"
    ):
        raise StudyValidationError("terminal attempt receipt contents mismatch")
    if _require_mapping(manifest.get("terminal_attempt"), "terminal attempt") != {
        "receipt_name": path.name,
        "receipt_sha256": sha256_path(path),
    }:
        raise StudyValidationError("manifest terminal-attempt evidence mismatch")


def _authenticated_external_metadata(
    sources: SourcePaths,
    expected: ExpectedSources,
) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    paths = {
        "archive": sources.archive,
        "checksum": sources.checksum,
        "package_manifest": sources.package_manifest,
        "plan": sources.plan,
    }
    expected_hashes = {
        "archive": expected.zip_sha256,
        "checksum": expected.checksum_file_sha256,
        "package_manifest": expected.package_manifest_sha256,
        "plan": expected.plan_sha256,
    }
    if any(not path.is_file() for path in paths.values()):
        raise StudyValidationError("coverage external evidence file is missing")
    hashes = {label: sha256_path(path) for label, path in paths.items()}
    for label, digest in hashes.items():
        if digest != expected_hashes[label]:
            raise StudyValidationError(f"external {label} SHA-256 mismatch")
    try:
        sidecar_text = sources.checksum.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise StudyValidationError("coverage SHA-256 sidecar is malformed") from error
    match = SIDECAR_PATTERN.fullmatch(sidecar_text)
    if (
        match is None
        or match.group(2) != sources.archive.name
        or match.group(1) != hashes["archive"]
    ):
        raise StudyValidationError("coverage checksum sidecar does not bind the ZIP")

    package_manifest = _require_mapping(
        strict_json_loads(sources.package_manifest.read_bytes(), "package manifest"),
        "package manifest",
    )
    external_plan = _require_mapping(
        strict_json_loads(sources.plan.read_bytes(), "external plan"),
        "external plan",
    )
    required_manifest = {
        "format_version",
        "study_plan_id",
        "study_project_revision",
        "study_complete",
        "planned_runs",
        "completed_runs",
        "incomplete_runs",
        "archive",
    }
    if set(package_manifest) != required_manifest:
        raise StudyValidationError("coverage package manifest schema mismatch")
    archive_meta = _require_mapping(package_manifest.get("archive"), "archive metadata")
    if set(archive_meta) != {"path", "sha256", "size_bytes", "files"}:
        raise StudyValidationError("coverage archive metadata schema mismatch")
    if (
        package_manifest.get("format_version") != 1
        or package_manifest.get("study_plan_id") != expected.plan_id
        or package_manifest.get("study_project_revision") != expected.project_revision
        or package_manifest.get("planned_runs") != RUNS
        or not isinstance(archive_meta.get("path"), str)
        or not archive_meta.get("path")
        or archive_meta.get("sha256") != hashes["archive"]
        or archive_meta.get("size_bytes") != sources.archive.stat().st_size
        or external_plan.get("format_version") != 1
        or external_plan.get("plan_id") != expected.plan_id
    ):
        raise StudyValidationError("coverage external metadata mismatch")
    return hashes, package_manifest, external_plan


def _validate_h100_environment(
    manifest: dict[str, object],
    *,
    expected_project_revision: str,
) -> None:
    if manifest.get("project_revision") != expected_project_revision:
        raise StudyValidationError("coverage project revision mismatch")
    if manifest.get("working_tree_dirty") is not False:
        raise StudyValidationError("coverage manifest does not prove a clean worktree")
    if manifest.get("upstream_reference") != UPSTREAM_REFERENCE:
        raise StudyValidationError("coverage upstream reference mismatch")
    environment = _require_mapping(manifest.get("environment"), "runtime environment")
    if not str(environment.get("python", "")).startswith("3.12."):
        raise StudyValidationError("coverage study did not use Python 3.12")
    if (
        environment.get("backend") != "gpu"
        or environment.get("device_count") != 1
        or environment.get("competition_aligned_h100") is not True
    ):
        raise StudyValidationError("runtime was not exactly one JAX-visible H100")
    kinds = environment.get("device_kinds")
    if kinds != [EXACT_GPU_NAME]:
        raise StudyValidationError("JAX device identity is not the exact frozen H100")
    platform_versions = environment.get("jax_platform_versions")
    if (
        not isinstance(platform_versions, list)
        or len(platform_versions) != 1
        or re.search(r"\bcuda(?:\s+|_)?13", str(platform_versions[0]).lower())
        is None
    ):
        raise StudyValidationError("JAX backend is not CUDA 13")
    versions = _require_mapping(environment.get("versions"), "runtime versions")
    for package, version in H100_CUDA13_PACKAGE_VERSIONS.items():
        if versions.get(package) != version:
            raise StudyValidationError(f"coverage runtime requires {package}=={version}")
    if versions.get("nvidia-cuda-runtime") in {None, "not-installed"}:
        raise StudyValidationError("coverage runtime lacks the CUDA 13 runtime wheel")
    if versions.get("nvidia-cuda-runtime-cu12") != "not-installed":
        raise StudyValidationError("coverage runtime contains the CUDA 12 runtime wheel")
    for package in ("jax-cuda12-pjrt", "jax-cuda12-plugin"):
        if versions.get(package) != "not-installed":
            raise StudyValidationError(
                f"coverage runtime contains the CUDA 12 package {package}"
            )

    runtime_configuration = _require_mapping(
        environment.get("jax_runtime_configuration"), "JAX runtime configuration"
    )
    if (
        runtime_configuration.get("enable_compilation_cache") is not False
        or runtime_configuration.get("compilation_cache_dir") is not None
    ):
        raise StudyValidationError("coverage JAX compilation cache was enabled")
    runtime_environment = _require_mapping(
        environment.get("jax_runtime_environment"), "JAX runtime environment"
    )
    allowed = {
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    for name in JAX_RUNTIME_ENVIRONMENT_KEYS:
        value = runtime_environment.get(name)
        if name in allowed:
            if value != allowed[name]:
                raise StudyValidationError(f"unexpected runtime value for {name}")
        elif value is not None:
            raise StudyValidationError(f"forbidden JAX/XLA/CUDA override: {name}")

    rental = _require_mapping(manifest.get("rental_preflight"), "rental preflight")
    disk = _require_mapping(rental.get("disk"), "rental disk evidence")
    if int(disk.get("free_bytes", 0)) < 20 * 1024**3:
        raise StudyValidationError("rental disk evidence is below 20 GiB")
    snapshot = _require_mapping(rental.get("gpu_idle"), "rental GPU snapshot")
    gpus = snapshot.get("gpus")
    if snapshot.get("status") != "ok" or not isinstance(gpus, list) or len(gpus) != 1:
        raise StudyValidationError("preflight does not prove one physical H100")
    gpu = _require_mapping(gpus[0], "physical H100")
    if str(gpu.get("name", "")).strip().upper() != EXACT_GPU_NAME.upper():
        raise StudyValidationError("physical GPU is not the exact frozen H100")
    if str(gpu.get("mig_mode_current", "")).lower() != "disabled":
        raise StudyValidationError("H100 MIG mode was not disabled")
    if int(gpu.get("memory_total_mib", 0)) < 75_000:
        raise StudyValidationError("H100 memory evidence is insufficient")
    if int(gpu.get("memory_used_mib", 1_001)) > 1_000:
        raise StudyValidationError("H100 was not idle at preflight")
    if int(gpu.get("utilization_percent", 6)) > 5:
        raise StudyValidationError("H100 utilization was not idle at preflight")


def _validate_preflight_members(
    archive: zipfile.ZipFile,
    manifest: dict[str, object],
) -> None:
    environment = _require_mapping(manifest.get("environment"), "runtime environment")
    if strict_json(_load_json_member(archive, "preflight.json")) != strict_json(
        environment
    ):
        raise StudyValidationError("runtime preflight artifact disagrees with manifest")
    host = _load_json_member(archive, "preflight.host-environment.json")
    effective = _require_mapping(
        host.get("effective_environment"), "effective host runtime environment"
    )
    runtime_environment = _require_mapping(
        environment.get("jax_runtime_environment"), "JAX runtime environment"
    )
    if strict_json(effective) != strict_json(runtime_environment):
        raise StudyValidationError("host and child cache environments disagree")
    runtime_policy = _require_mapping(
        manifest.get("runtime_policy"), "runtime policy"
    )
    cache_policy = _require_mapping(
        runtime_policy.get("jax_compilation_cache"), "JAX cache policy"
    )
    if cache_policy.get("policy") != "disabled" or strict_json(
        cache_policy.get("effective_environment")
    ) != strict_json(runtime_environment):
        raise StudyValidationError("manifest runtime cache policy drifted")


def _validate_coverage_record_evidence(
    record: dict[str, object],
    expected_config: dict[str, object],
    initial_params,
) -> None:
    import numpy as np

    run_id = str(expected_config["run_id"])
    raw_hashes = record.get("raw_suffix_parameter_hashes")
    population_size = int(expected_config["population_size"])
    if (
        not isinstance(raw_hashes, list)
        or len(raw_hashes) != population_size - 1
        or any(
            not isinstance(value, str) or SHA256.fullmatch(value) is None
            for value in raw_hashes
        )
    ):
        raise StudyValidationError(f"raw random-draw evidence is invalid: {run_id}")
    recorded_hashes = record.get("initial_parameter_hashes")
    if expected_config["arm"] == "no_prior":
        if recorded_hashes[1:] != raw_hashes:
            raise StudyValidationError(
                f"control suffix differs from the authenticated random draw: {run_id}"
            )
        return

    members = np.asarray(initial_params)
    suffix = members[1:]
    if suffix.shape[0] != population_size - 1 or not np.all(np.isfinite(suffix)):
        raise StudyValidationError(f"coverage population is invalid: {run_id}")
    canonical = _coverage_logits_from_ranks(
        np.arange(suffix.shape[0])[:, None], suffix.shape[0], suffix.dtype
    )
    expected = np.broadcast_to(canonical, suffix.shape)
    _require_coverage_ulp_match(
        np.sort(suffix, axis=0),
        expected,
        f"coverage midpoint logits drifted: {run_id}",
    )


def _validate_coverage_history_chronology(
    rows: list[dict[str, object]],
    expected_config: dict[str, object],
    record: dict[str, object],
) -> None:
    """Reconstruct exact full-vmap calls and enforce all frozen time ceilings."""
    run_id = str(expected_config["run_id"])
    _validate_utc_interval(record, f"coverage run {run_id}")
    population_size = int(expected_config["population_size"])
    by_call: dict[int, list[dict[str, object]]] = defaultdict(list)
    observed_sequence: list[tuple[int, int]] = []
    for row in rows:
        call_index = row.get("call_index")
        candidate_index = row.get("candidate_index")
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
        ):
            raise StudyValidationError(f"coverage history indexes are invalid: {run_id}")
        by_call[call_index].append(row)
        observed_sequence.append((call_index, candidate_index))
    if sorted(by_call) != list(range(len(by_call))):
        raise StudyValidationError(f"coverage call chronology is not contiguous: {run_id}")
    expected_sequence = [
        (call_index, candidate_index)
        for call_index in range(len(by_call))
        for candidate_index in range(population_size)
    ]
    if observed_sequence != expected_sequence:
        raise StudyValidationError(
            f"coverage stored row chronology is interleaved or reordered: {run_id}"
        )

    previous_time = -math.inf
    expected_eval_count = 0
    for call_index in range(len(by_call)):
        call_rows = by_call[call_index]
        if len(call_rows) != population_size:
            raise StudyValidationError(f"coverage call is not full-vmap: {run_id}")
        if [int(row["candidate_index"]) for row in call_rows] != list(
            range(population_size)
        ):
            raise StudyValidationError(
                f"coverage candidate chronology is invalid: {run_id}"
            )
        expected_eval_count += population_size
        eval_counts = {row.get("eval_count_after_call") for row in call_rows}
        if eval_counts != {expected_eval_count}:
            raise StudyValidationError(
                f"coverage evaluation accounting is not row-exact: {run_id}"
            )
        times = {row.get("time_seconds") for row in call_rows}
        if len(times) != 1:
            raise StudyValidationError(f"coverage call times disagree: {run_id}")
        call_time = next(iter(times))
        if (
            isinstance(call_time, bool)
            or not isinstance(call_time, (int, float))
            or not math.isfinite(float(call_time))
            or float(call_time) < 0
            or float(call_time) < previous_time
            or float(call_time) > float(expected_config["max_time_seconds"])
        ):
            raise StudyValidationError(
                f"coverage Objective chronology exceeds its budget: {run_id}"
            )
        previous_time = float(call_time)

    process = _require_mapping(record.get("worker_process"), f"worker process {run_id}")
    wall = process.get("full_wall_seconds")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0
        or float(wall) > WORKER_TIMEOUT_SECONDS
    ):
        raise StudyValidationError(f"coverage worker exceeded its timeout: {run_id}")


def coverage_package_is_complete(
    sources: SourcePaths, expected: ExpectedSources
) -> bool:
    """Authenticate structural metadata and classify without opening outcomes."""
    _, package_manifest, _ = _authenticated_external_metadata(sources, expected)
    value = package_manifest.get("study_complete")
    if type(value) is not bool:
        raise StudyValidationError("coverage completion flag is invalid")
    return value


def validate_coverage_terminal_partial(
    sources: SourcePaths,
    *,
    expected: ExpectedSources,
    terminal_attempt_receipt: Path,
    limits: ArchiveLimits = ArchiveLimits(),
) -> dict[str, object]:
    """Authenticate a terminal partial structurally without reading outcomes."""
    source_hashes, package_manifest, external_plan = _authenticated_external_metadata(
        sources, expected
    )
    if package_manifest.get("study_complete") is not False:
        raise StudyValidationError("terminal-partial validator requires an incomplete package")
    completed = package_manifest.get("completed_runs")
    incomplete = package_manifest.get("incomplete_runs")
    if (
        type(completed) is not int
        or not 0 <= completed < RUNS
        or not isinstance(incomplete, list)
        or len(incomplete) != RUNS - completed
    ):
        raise StudyValidationError("coverage terminal-partial run counts are invalid")
    expected_configs = _validate_plan(external_plan, expected)
    planned_ids = set(expected_configs)
    incomplete_ids: set[str] = set()
    for item in incomplete:
        if not isinstance(item, dict) or set(item) != {"run_id", "status"}:
            raise StudyValidationError("terminal-partial incomplete-run schema mismatch")
        run_id = item.get("run_id")
        if (
            not isinstance(run_id, str)
            or run_id not in planned_ids
            or run_id in incomplete_ids
            or item.get("status") not in {"missing", "error", "interrupted"}
        ):
            raise StudyValidationError("terminal-partial run evidence mismatch")
        incomplete_ids.add(run_id)

    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    observed = set(member_names)
    archive_meta = _require_mapping(package_manifest.get("archive"), "archive metadata")
    if archive_meta.get("files") != len(member_names):
        raise StudyValidationError("terminal-partial archive count mismatch")
    expected_members = _expected_archive_members(expected_configs)
    recovery_names = {name for name in observed if name.startswith("recovery/")}
    if not (FIXED_MEMBERS | PREFLIGHT_MEMBERS) <= observed:
        raise StudyValidationError("terminal partial lacks structural evidence")
    if observed - expected_members - recovery_names:
        raise StudyValidationError("terminal partial contains an unexpected member")

    with zipfile.ZipFile(sources.archive, "r") as archive:
        for name in recovery_names:
            match = RECOVERY_RECEIPT.fullmatch(name)
            payload = _read_member(archive, name)
            receipt = _require_mapping(strict_json_loads(payload, name), name)
            if (
                match is None
                or hashlib.sha256(payload).hexdigest()[:12] != match.group(1)
                or set(receipt) != {"pid", "hostname", "created_utc"}
            ):
                raise StudyValidationError("terminal partial recovery receipt is invalid")
        manifest = _load_json_member(archive, "manifest.json")
        package_state = _load_json_member(archive, "package-state.json")
        session = _load_json_member(archive, "session.json")
        for key in (
            "format_version",
            "plan_id",
            "configuration",
            "run_order_policy",
            "primary_pair_order",
            "optimizer_seed_order_policy",
            "runs",
        ):
            if strict_json(manifest.get(key)) != strict_json(external_plan.get(key)):
                raise StudyValidationError(f"internal/external plan mismatch: {key}")
        _validate_h100_environment(
            manifest, expected_project_revision=expected.project_revision
        )
        _validate_preflight_members(archive, manifest)
        _validate_terminal_receipt(
            terminal_attempt_receipt, expected=expected, manifest=manifest
        )
        if package_state != {
            "format_version": 1,
            "study_complete": False,
            "planned_runs": RUNS,
            "completed_runs": completed,
            "incomplete_runs": incomplete,
        }:
            raise StudyValidationError("terminal-partial package state mismatch")
        complete_ids = planned_ids - incomplete_ids
        for run_id in complete_ids:
            required = {
                f"configs/{run_id}.json",
                f"histories/{run_id}.npz",
                f"logs/{run_id}.stdout.log",
                f"logs/{run_id}.stderr.log",
                f"runs/{run_id}.json",
            }
            if not required <= observed:
                raise StudyValidationError(
                    f"terminal partial complete run lacks artifacts: {run_id}"
                )
        status = session.get("status")
        if status not in {
            "error",
            "interrupted",
            "wall_limit_reached",
            "provider_deadline_guard",
            "running",
        }:
            raise StudyValidationError("terminal-partial session status is invalid")
        if status == "running" and (
            not recovery_names
            or not isinstance(session.get("started_utc"), str)
            or session.get("max_session_wall_seconds") != SESSION_WALL_SECONDS
        ):
            raise StudyValidationError("running partial lacks stale-writer recovery")
        if status == "running":
            _parse_utc_timestamp(
                session.get("started_utc"), "terminal-partial session start"
            )
        if status != "running":
            timestamp_elapsed = _validate_utc_interval(
                session, "terminal-partial session"
            )
            elapsed = session.get("elapsed_seconds")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) <= 0
                or float(elapsed) > SESSION_WALL_SECONDS
                or abs(float(elapsed) - timestamp_elapsed) > 5.0
            ):
                raise StudyValidationError("terminal-partial elapsed time is invalid")
            if session.get("max_session_wall_seconds") != SESSION_WALL_SECONDS:
                raise StudyValidationError("terminal-partial wall limit drifted")
            if status in {"wall_limit_reached", "provider_deadline_guard"}:
                next_run = session.get("next_run_id")
                if not isinstance(next_run, str) or next_run not in incomplete_ids:
                    raise StudyValidationError("terminal-partial next-run evidence drifted")
            if status == "provider_deadline_guard":
                configuration = _require_mapping(
                    external_plan.get("configuration"), "plan configuration"
                )
                if session.get("provider_stop_utc") != configuration.get(
                    "provider_stop_utc"
                ):
                    raise StudyValidationError("terminal-partial provider stop drifted")

    return {
        "status": "not_evaluable",
        "action": "retain_candidate_attempt_not_evaluable",
        "study_profile": PROFILE,
        "plan_id": expected.plan_id,
        "project_revision": expected.project_revision,
        "completed_runs": completed,
        "incomplete_runs": len(incomplete),
        "archive_entries": len(member_names),
        "archive_integrity": {
            key: value for key, value in integrity.items() if key != "member_names"
        },
        "external_hashes": source_hashes,
        "terminal_attempt_receipt": "passed",
        "summary_content_opened": False,
        "run_records_opened": False,
        "histories_opened": False,
    }


def validate_coverage_archive(
    sources: SourcePaths,
    *,
    expected: ExpectedSources,
    terminal_attempt_receipt: Path,
    limits: ArchiveLimits = ArchiveLimits(),
) -> ValidatedStudy:
    """Authenticate complete raw evidence while leaving the summary sealed."""
    source_hashes, package_manifest, external_plan = _authenticated_external_metadata(
        sources, expected
    )
    if (
        package_manifest.get("study_complete") is not True
        or package_manifest.get("completed_runs") != RUNS
        or package_manifest.get("incomplete_runs") != []
    ):
        raise StudyValidationError("complete coverage package state mismatch")
    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    archive_meta = _require_mapping(package_manifest.get("archive"), "archive metadata")
    if archive_meta.get("files") != len(member_names) or len(member_names) != ARCHIVE_MEMBERS:
        raise StudyValidationError("coverage archive member count mismatch")

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
            "optimizer_seed_order_policy",
            "runs",
        ):
            if strict_json(manifest.get(key)) != strict_json(external_plan.get(key)):
                raise StudyValidationError(f"internal/external plan mismatch: {key}")
        expected_configs = _validate_plan(manifest, expected)
        _validate_h100_environment(
            manifest, expected_project_revision=expected.project_revision
        )
        _validate_preflight_members(archive, manifest)
        _validate_terminal_receipt(
            terminal_attempt_receipt, expected=expected, manifest=manifest
        )
        if package_state != {
            "format_version": 1,
            "study_complete": True,
            "planned_runs": RUNS,
            "completed_runs": RUNS,
            "incomplete_runs": [],
        }:
            raise StudyValidationError("package state does not prove 48/48 completion")
        elapsed = session.get("elapsed_seconds")
        timestamp_elapsed = _validate_utc_interval(session, "coverage session")
        if (
            session.get("status") != "complete"
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
            or float(elapsed) > SESSION_WALL_SECONDS
            or abs(float(elapsed) - timestamp_elapsed) > 5.0
            or session.get("max_session_wall_seconds") != SESSION_WALL_SECONDS
        ):
            raise StudyValidationError("coverage session completion evidence mismatch")

        expected_members = _expected_archive_members(expected_configs)
        missing = sorted(expected_members - set(member_names))
        unexpected = sorted(set(member_names) - expected_members)
        if missing or unexpected:
            raise StudyValidationError(
                f"coverage member set mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )

        expected_environment = _require_mapping(
            manifest.get("environment"), "runtime environment"
        )
        configs: dict[str, dict[str, object]] = {}
        records_by_id: dict[str, dict[str, object]] = {}
        histories: dict[str, list[dict[str, object]]] = {}
        initial_arrays: dict[str, object] = {}
        topology_to_hash: dict[str, str] = {}
        hierarchy: dict[str, set[tuple[int, str]]] = defaultdict(set)
        for run_id, expected_config in expected_configs.items():
            config = _load_json_member(archive, f"configs/{run_id}.json")
            if strict_json(config) != strict_json(expected_config):
                raise StudyValidationError(f"coverage config artifact mismatch: {run_id}")
            record = _load_json_member(archive, f"runs/{run_id}.json")
            record_id = str(record.get("run_id"))
            if record_id in records_by_id:
                raise StudyValidationError(f"duplicate coverage run ID: {record_id}")
            if "optimizer_telemetry" in record:
                raise StudyValidationError(f"unsolicited optimizer telemetry: {run_id}")
            history_payload = _read_member(archive, f"histories/{run_id}.npz")
            arrays = _load_history_arrays(history_payload, f"histories/{run_id}.npz")
            initial_arrays[run_id] = arrays["initial_params_unbounded"]
            histories[run_id] = _validate_record(
                record, expected_config, archive, expected_environment
            )
            _validate_coverage_history_chronology(
                histories[run_id], expected_config, record
            )
            _validate_coverage_record_evidence(
                record, expected_config, arrays["initial_params_unbounded"]
            )
            configs[run_id] = config
            records_by_id[record_id] = record
            topology = str(record["problem"]["topology_string"])
            topology_hash = str(record["problem"]["topology_sha256"])
            if topology_to_hash.setdefault(topology, topology_hash) != topology_hash:
                raise StudyValidationError("coverage topology hash identity drifted")
            hierarchy[topology_hash].add(
                (int(expected_config["optimizer_seed"]), str(expected_config["arm"]))
            )

        if set(records_by_id) != set(expected_configs):
            raise StudyValidationError("coverage record IDs do not match the plan")
        expected_cells = {(seed, arm) for seed in SEEDS for arm in ARMS}
        if len(topology_to_hash) != TOPOLOGIES or any(
            cells != expected_cells for cells in hierarchy.values()
        ):
            raise StudyValidationError("coverage topology hierarchy is broken")
        import numpy as np

        for pair_id in {str(config["pair_id"]) for config in expected_configs.values()}:
            pair = {
                str(record["config"]["arm"]): record
                for record in records_by_id.values()
                if record["config"]["pair_id"] == pair_id
            }
            control = pair["no_prior"]
            treatment = pair["coverage_balanced"]
            if control["raw_suffix_parameter_hashes"] != treatment[
                "raw_suffix_parameter_hashes"
            ]:
                raise StudyValidationError(
                    f"paired pre-transform random draw differs: {pair_id}"
                )
            if control["initial_parameter_hashes"][0] != treatment[
                "initial_parameter_hashes"
            ][0]:
                raise StudyValidationError(f"paired anchor differs: {pair_id}")
            control_initial = initial_arrays[str(control["run_id"])]
            treatment_initial = initial_arrays[str(treatment["run_id"])]
            raw_suffix = np.asarray(control_initial)[1:]
            order = np.argsort(raw_suffix, axis=0, kind="stable")
            ranks = np.argsort(order, axis=0, kind="stable")
            expected_suffix = _coverage_logits_from_ranks(
                ranks, raw_suffix.shape[0], np.asarray(treatment_initial).dtype
            )
            _require_coverage_ulp_match(
                np.asarray(treatment_initial)[1:],
                expected_suffix,
                "paired coverage transform is not the ranked raw draw: "
                f"{pair_id}",
            )

        records = [records_by_id[run_id] for run_id in sorted(records_by_id)]
        try:
            lines = _read_member(archive, "runs.jsonl").decode(
                "utf-8", errors="strict"
            ).splitlines()
        except UnicodeDecodeError as error:
            raise StudyValidationError("coverage runs.jsonl is not UTF-8") from error
        if len(lines) != RUNS:
            raise StudyValidationError("coverage runs.jsonl must contain 48 records")
        jsonl_records = [
            _require_mapping(strict_json_loads(line, "runs.jsonl line"), "run")
            for line in lines
        ]
        if strict_json(jsonl_records) != strict_json(records):
            raise StudyValidationError("coverage runs.jsonl disagrees with records")

    receipt = {
        **{key: value for key, value in integrity.items() if key != "member_names"},
        "external_hashes": "passed",
        "source_lock": "passed",
        "sidecar_filename_and_digest": "passed",
        "terminal_attempt_receipt": "passed",
        "records": RUNS,
        "histories": RUNS,
        "configs": RUNS,
        "stdout_logs": RUNS,
        "stderr_logs": RUNS,
        "topologies": TOPOLOGIES,
        "optimizer_seed_pairs": PAIRS,
        "pretransform_draw_pairs": PAIRS,
        "coverage_initial_arrays": RUNS // 2,
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
        history_rows=histories,
        integrity=receipt,
    )


def load_coverage_summary_after_reproduction(
    study: ValidatedStudy,
    reproduction_agreement: CoverageReplayAgreement,
) -> dict[str, object]:
    """Open the archived summary only after two raw-data replays agree."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("coverage summary receipt is invalid")
    if not isinstance(reproduction_agreement, CoverageReplayAgreement):
        raise StudyValidationError(
            "coverage summary requires a comparator-issued replay agreement"
        )
    agreement = reproduction_agreement.as_dict()
    if agreement != {
        "status": "matched",
        "runs_compared": RUNS,
        "topology_values_compared": TOPOLOGIES,
        "optimizer_seed_pairs_compared": PAIRS,
        "frozen_criteria_compared": 13,
        "study_identity_sha256": coverage_study_identity_sha256(study),
    }:
        raise StudyValidationError(
            "coverage summary remains locked until two raw replays agree"
        )
    with zipfile.ZipFile(study.sources.archive, "r") as archive:
        return _load_json_member(archive, "summary.json")
