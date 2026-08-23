"""Outcome-blind ingestion for ``submission-like-screen-v1``.

The source-lock digest must be recorded out of band before this module parses
the lock or any study JSON. The archived ``summary.json`` remains sealed until
the production and independent history-first replays agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

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
    _load_json_member,
    _read_member,
    _require_mapping,
    _validate_environment,
    _validate_record,
    inspect_zip_integrity,
    sha256_path,
    strict_json_loads,
)
from experiments.uifo_paired.runner import _run_config, strict_json
from experiments.uifo_paired.study_profiles import bind_study_profile


PROFILE = "submission-like-screen-v1"
PANEL_ID = "submission-like-v1"
POLICY_ID = "no-prior-submission-like-screen-v1"
RUNS = 20
TOPOLOGIES = 10
SEEDS = {29, 31}
ARM = "no_prior"
ARCHIVE_MEMBERS = 109
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PANEL_PATH = Path(__file__).with_name("panels") / "submission-like-v1.json"
RECOVERY_RECEIPT = re.compile(r"^recovery/stale-study-lock-([0-9a-f]{12})\.json$")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pair_id(topology: str, seed: int) -> str:
    digest = hashlib.sha256(topology.encode()).hexdigest()[:12]
    return f"topo{digest}__oseed{seed:010d}"


def authenticate_submission_like_source_lock(
    source_lock_path: Path,
    *,
    expected_source_lock_sha256: str,
    sources: SourcePaths,
    terminal_attempt_receipt: Path,
) -> ExpectedSources:
    """Authenticate the out-of-band source lock before parsing its contents."""
    if SHA256.fullmatch(expected_source_lock_sha256) is None:
        raise StudyValidationError("expected source-lock SHA-256 is malformed")
    if not source_lock_path.is_file():
        raise StudyValidationError("source-lock file is missing")
    observed_lock_hash = sha256_path(source_lock_path)
    if observed_lock_hash != expected_source_lock_sha256:
        raise StudyValidationError("source-lock SHA-256 mismatch")

    lock = _require_mapping(
        strict_json_loads(source_lock_path.read_bytes(), "submission source lock"),
        "submission source lock",
    )
    if set(lock) != {
        "format_version",
        "study_profile",
        "plan_id",
        "project_revision",
        "files",
    }:
        raise StudyValidationError("source-lock schema mismatch")
    if lock.get("format_version") != 1 or lock.get("study_profile") != PROFILE:
        raise StudyValidationError("source-lock profile or version mismatch")
    plan_id = lock.get("plan_id")
    revision = lock.get("project_revision")
    if not isinstance(plan_id, str) or re.fullmatch(r"[0-9a-f]{16}", plan_id) is None:
        raise StudyValidationError("source-lock plan ID is invalid")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise StudyValidationError("source-lock revision is invalid")

    files = _require_mapping(lock.get("files"), "source-lock files")
    paths = {
        sources.archive.name: sources.archive,
        sources.checksum.name: sources.checksum,
        sources.package_manifest.name: sources.package_manifest,
        sources.plan.name: sources.plan,
        terminal_attempt_receipt.name: terminal_attempt_receipt,
    }
    if set(files) != set(paths):
        raise StudyValidationError("source-lock basenames do not match supplied files")
    for basename, path in paths.items():
        entry = files.get(basename)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise StudyValidationError(f"source-lock file schema is invalid: {basename}")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_hash, str) or SHA256.fullmatch(expected_hash) is None:
            raise StudyValidationError(f"source-lock digest is invalid: {basename}")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise StudyValidationError(f"source-lock size is invalid: {basename}")
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_path(path) != expected_hash
        ):
            raise StudyValidationError(f"source-lock file mismatch: {basename}")

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
        strict_json_loads(PANEL_PATH.read_bytes(), "committed submission panel"),
        "committed submission panel",
    )
    values = payload.get("topologies")
    if payload.get("panel_id") != PANEL_ID or not isinstance(values, list):
        raise StudyValidationError("committed submission panel is invalid")
    if len(values) != TOPOLOGIES or not all(isinstance(value, str) for value in values):
        raise StudyValidationError("committed submission panel topology count mismatch")
    return [str(value) for value in values]


def _validate_plan(
    plan: dict[str, object], expected: ExpectedSources
) -> dict[str, dict[str, object]]:
    if plan.get("format_version") != 1 or plan.get("plan_id") != expected.plan_id:
        raise StudyValidationError("submission-like plan identity mismatch")
    configuration = _require_mapping(plan.get("configuration"), "plan configuration")
    if configuration.get("study_profile") != PROFILE:
        raise StudyValidationError("submission-like study profile mismatch")
    try:
        policy = bind_study_profile(PROFILE, configuration)
    except ValueError as error:
        raise StudyValidationError(f"submission-like profile mismatch: {error}") from error
    if strict_json(configuration.get("decision_policy")) != strict_json(policy):
        raise StudyValidationError("submission-like frozen decision policy mismatch")
    if _require_mapping(policy, "decision policy").get("policy_id") != POLICY_ID:
        raise StudyValidationError("submission-like decision policy ID mismatch")

    committed = _panel_topologies()
    topology_specs = configuration.get("topologies")
    expected_specs = [{"kind": "string", "value": value} for value in committed]
    if topology_specs != expected_specs:
        raise StudyValidationError("submission-like panel contents or order drifted")
    if any(TOPOLOGY_PATTERN.fullmatch(value) is None for value in committed):
        raise StudyValidationError("submission-like panel contains invalid topology")

    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != RUNS or not all(
        isinstance(run, dict) for run in runs
    ):
        raise StudyValidationError("submission-like plan must contain 20 runs")
    run_rows = [run for run in runs if isinstance(run, dict)]
    if [run.get("planned_run_index") for run in run_rows] != list(range(RUNS)):
        raise StudyValidationError("submission-like indexes are not exact serial order")
    if len({str(run.get("run_id")) for run in run_rows}) != RUNS:
        raise StudyValidationError("submission-like plan contains duplicate run IDs")

    by_topology: dict[str, list[dict[str, object]]] = defaultdict(list)
    expected_run_keys = {
        "planned_run_index",
        "run_id",
        "pair_id",
        "run_order_within_pair",
        "topology",
        "optimizer_seed",
        "arm",
    }
    for run in run_rows:
        if set(run) != expected_run_keys:
            raise StudyValidationError("submission-like plan run schema drift")
        topology = _require_mapping(run.get("topology"), "run topology")
        value = topology.get("value")
        seed = run.get("optimizer_seed")
        if topology not in expected_specs or not isinstance(value, str):
            raise StudyValidationError("submission-like run references unknown topology")
        if seed not in SEEDS or run.get("arm") != ARM:
            raise StudyValidationError("submission-like seed or arm mismatch")
        pair_id = _pair_id(value, int(seed))
        if run.get("pair_id") != pair_id or run.get("run_id") != f"{pair_id}__{ARM}":
            raise StudyValidationError("submission-like run identity mismatch")
        if run.get("run_order_within_pair") != 0:
            raise StudyValidationError("single-arm within-pair order must be zero")
        by_topology[value].append(run)

    if set(by_topology) != set(committed) or any(
        {int(run["optimizer_seed"]) for run in rows} != SEEDS or len(rows) != 2
        for rows in by_topology.values()
    ):
        raise StudyValidationError("submission-like topology blocks are broken")
    first_sweep = run_rows[:TOPOLOGIES]
    second_sweep = run_rows[TOPOLOGIES:]
    if (
        [int(run["optimizer_seed"]) for run in first_sweep] != [29] * TOPOLOGIES
        or [int(run["optimizer_seed"]) for run in second_sweep]
        != [31] * TOPOLOGIES
        or [str(run["topology"]["value"]) for run in first_sweep] != committed
        or [str(run["topology"]["value"]) for run in second_sweep]
        != list(reversed(committed))
    ):
        raise StudyValidationError("submission-like mirrored seed sweeps drifted")
    if plan.get("optimizer_seed_order_policy") != "mirrored_sweeps":
        raise StudyValidationError("submission-like seed-order policy mismatch")
    if plan.get("run_order_policy") != "rotate arms once per topology-seed pair":
        raise StudyValidationError("submission-like serial run-order policy mismatch")
    required_primary = {
        "complete_primary_pairs": 0,
        "no_prior_first": 0,
        "semantic_prior_first": 0,
        "absolute_imbalance": 0,
    }
    if plan.get("primary_pair_order") != required_primary:
        raise StudyValidationError("submission-like primary pair evidence drifted")

    core = {
        "configuration": configuration,
        "run_order_policy": plan["run_order_policy"],
        "primary_pair_order": plan["primary_pair_order"],
        "runs": runs,
        "optimizer_seed_order_policy": plan["optimizer_seed_order_policy"],
    }
    recomputed = hashlib.sha256(_canonical(core).encode()).hexdigest()[:16]
    if recomputed != expected.plan_id:
        raise StudyValidationError("submission-like plan ID does not match contents")
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
    manifest_receipt = _require_mapping(
        manifest.get("terminal_attempt"), "manifest terminal attempt"
    )
    if manifest_receipt != {
        "receipt_name": path.name,
        "receipt_sha256": sha256_path(path),
    }:
        raise StudyValidationError("manifest terminal attempt evidence mismatch")


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
    for label, path in paths.items():
        if not path.is_file():
            raise StudyValidationError(f"missing external {label} file")
    hashes = {label: sha256_path(path) for label, path in paths.items()}
    for label, digest in hashes.items():
        if digest != expected_hashes[label]:
            raise StudyValidationError(f"external {label} SHA-256 mismatch")
    try:
        sidecar_text = sources.checksum.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise StudyValidationError("malformed SHA-256 sidecar") from error
    match = SIDECAR_PATTERN.fullmatch(sidecar_text)
    if match is None:
        raise StudyValidationError("malformed SHA-256 sidecar")
    if match.group(2) != sources.archive.name or match.group(1) != hashes["archive"]:
        raise StudyValidationError("SHA-256 sidecar does not bind the supplied ZIP")

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
        raise StudyValidationError("package manifest schema mismatch")
    archive_meta = _require_mapping(
        package_manifest.get("archive"), "package manifest archive"
    )
    if set(archive_meta) != {"path", "sha256", "size_bytes", "files"}:
        raise StudyValidationError("package manifest archive schema mismatch")
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
        raise StudyValidationError("external submission-like metadata mismatch")
    return hashes, package_manifest, external_plan


def submission_like_package_is_complete(
    sources: SourcePaths,
    expected: ExpectedSources,
) -> bool:
    """Authenticate structural metadata and classify without opening outcomes."""
    _, package_manifest, _ = _authenticated_external_metadata(sources, expected)
    value = package_manifest.get("study_complete")
    if type(value) is not bool:
        raise StudyValidationError("package manifest completion flag is invalid")
    return value


def validate_submission_like_terminal_partial(
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
        raise StudyValidationError("terminal partial run counts are invalid")

    expected_configs = _validate_plan(external_plan, expected)
    planned_ids = set(expected_configs)
    incomplete_ids: set[str] = set()
    for item in incomplete:
        if not isinstance(item, dict) or set(item) != {"run_id", "status"}:
            raise StudyValidationError("terminal partial incomplete-run schema mismatch")
        run_id = item.get("run_id")
        if (
            not isinstance(run_id, str)
            or run_id not in planned_ids
            or run_id in incomplete_ids
            or item.get("status") not in {"missing", "error", "interrupted"}
        ):
            raise StudyValidationError("terminal partial incomplete-run evidence mismatch")
        incomplete_ids.add(run_id)

    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    archive_meta = _require_mapping(package_manifest.get("archive"), "archive metadata")
    if archive_meta.get("files") != len(member_names):
        raise StudyValidationError("terminal partial archive member count mismatch")
    expected_members = _expected_archive_members(expected_configs)
    required_structural = FIXED_MEMBERS | PREFLIGHT_MEMBERS
    observed = set(member_names)
    if not required_structural <= observed:
        raise StudyValidationError("terminal partial lacks structural evidence")
    recovery_names = {name for name in observed if name.startswith("recovery/")}
    if observed - expected_members - recovery_names:
        raise StudyValidationError("terminal partial contains an unexpected member")

    with zipfile.ZipFile(sources.archive, "r") as archive:
        for name in recovery_names:
            match = RECOVERY_RECEIPT.fullmatch(name)
            if match is None:
                raise StudyValidationError("terminal partial recovery receipt name is invalid")
            payload = _read_member(archive, name)
            receipt = _require_mapping(
                strict_json_loads(payload, name), "recovery receipt"
            )
            if (
                hashlib.sha256(payload).hexdigest()[:12] != match.group(1)
                or set(receipt) != {"pid", "hostname", "created_utc"}
                or isinstance(receipt.get("pid"), bool)
                or not isinstance(receipt.get("pid"), int)
                or int(receipt["pid"]) <= 0
                or not isinstance(receipt.get("hostname"), str)
                or not receipt["hostname"]
                or not isinstance(receipt.get("created_utc"), str)
                or not receipt["created_utc"]
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
                raise StudyValidationError(
                    f"internal manifest/external plan mismatch: {key}"
                )
        _validate_environment(manifest, expected.project_revision)
        _validate_terminal_receipt(
            terminal_attempt_receipt, expected=expected, manifest=manifest
        )
        expected_state = {
            "format_version": 1,
            "study_complete": False,
            "planned_runs": RUNS,
            "completed_runs": completed,
            "incomplete_runs": incomplete,
        }
        if package_state != expected_state:
            raise StudyValidationError("terminal partial package-state mismatch")

        complete_ids = planned_ids - incomplete_ids
        for run_id in complete_ids:
            required_run_members = {
                f"configs/{run_id}.json",
                f"histories/{run_id}.npz",
                f"logs/{run_id}.stdout.log",
                f"logs/{run_id}.stderr.log",
                f"runs/{run_id}.json",
            }
            if not required_run_members <= observed:
                raise StudyValidationError(
                    f"terminal partial claimed complete run lacks artifacts: {run_id}"
                )

        status = session.get("status")
        elapsed = session.get("elapsed_seconds")
        if status not in {
            "error",
            "interrupted",
            "wall_limit_reached",
            "provider_deadline_guard",
            "running",
        }:
            raise StudyValidationError("terminal partial session evidence mismatch")
        if status == "running":
            if (
                not recovery_names
                or not isinstance(session.get("started_utc"), str)
                or session.get("max_session_wall_seconds") != 32_400.0
            ):
                raise StudyValidationError("running partial lacks stale-writer recovery")
        else:
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) <= 0
            ):
                raise StudyValidationError("terminal partial session timing mismatch")
            if status in {"error", "interrupted", "wall_limit_reached"} and (
                session.get("max_session_wall_seconds") != 32_400.0
            ):
                raise StudyValidationError("terminal partial session limit mismatch")
            if status in {"wall_limit_reached", "provider_deadline_guard"}:
                next_run = session.get("next_run_id")
                if not isinstance(next_run, str) or next_run not in incomplete_ids:
                    raise StudyValidationError("terminal partial next-run evidence mismatch")
            if status == "provider_deadline_guard":
                configuration = _require_mapping(
                    external_plan.get("configuration"), "plan configuration"
                )
                if session.get("provider_stop_utc") != configuration.get(
                    "provider_stop_utc"
                ):
                    raise StudyValidationError("terminal partial provider stop mismatch")

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


def validate_submission_like_archive(
    sources: SourcePaths,
    *,
    expected: ExpectedSources,
    terminal_attempt_receipt: Path,
    limits: ArchiveLimits = ArchiveLimits(),
) -> ValidatedStudy:
    """Authenticate all raw evidence while leaving ``summary.json`` sealed."""
    source_hashes, package_manifest, external_plan = _authenticated_external_metadata(
        sources, expected
    )
    if (
        package_manifest.get("study_complete") is not True
        or package_manifest.get("completed_runs") != RUNS
        or package_manifest.get("incomplete_runs") != []
    ):
        raise StudyValidationError("complete submission-like package state mismatch")
    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    archive_meta = _require_mapping(package_manifest.get("archive"), "archive metadata")
    if archive_meta.get("files") != len(member_names) or len(member_names) != ARCHIVE_MEMBERS:
        raise StudyValidationError("submission-like archive member count mismatch")

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
                raise StudyValidationError(
                    f"internal manifest/external plan mismatch: {key}"
                )
        expected_configs = _validate_plan(manifest, expected)
        _validate_environment(manifest, expected.project_revision)
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
            raise StudyValidationError("package-state does not prove 20/20 completion")
        if session.get("status") != "complete":
            raise StudyValidationError("submission-like session did not complete")
        elapsed = session.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
            or session.get("max_session_wall_seconds") != 32_400.0
        ):
            raise StudyValidationError("submission-like session timing mismatch")

        expected_members = _expected_archive_members(expected_configs)
        missing = sorted(expected_members - set(member_names))
        unexpected = sorted(set(member_names) - expected_members)
        if missing or unexpected:
            raise StudyValidationError(
                f"archive member set mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}"
            )

        expected_environment = _require_mapping(
            manifest.get("environment"), "runtime environment"
        )
        configs: dict[str, dict[str, object]] = {}
        records_by_id: dict[str, dict[str, object]] = {}
        histories: dict[str, list[dict[str, object]]] = {}
        topology_to_hash: dict[str, str] = {}
        hash_to_topology: dict[str, str] = {}
        hierarchy: dict[str, set[int]] = defaultdict(set)
        for run_id, expected_config in expected_configs.items():
            config = _load_json_member(archive, f"configs/{run_id}.json")
            if strict_json(config) != strict_json(expected_config):
                raise StudyValidationError(f"config artifact mismatch: {run_id}")
            record = _load_json_member(archive, f"runs/{run_id}.json")
            record_id = str(record.get("run_id"))
            if record_id in records_by_id:
                raise StudyValidationError(f"duplicate run record ID: {record_id}")
            if "optimizer_telemetry" in record:
                raise StudyValidationError(f"unsolicited optimizer telemetry: {run_id}")
            configs[run_id] = config
            records_by_id[record_id] = record
            histories[run_id] = _validate_record(
                record, expected_config, archive, expected_environment
            )
            topology = str(record["problem"]["topology_string"])
            topology_hash = str(record["problem"]["topology_sha256"])
            if topology_to_hash.setdefault(topology, topology_hash) != topology_hash:
                raise StudyValidationError("one topology resolved to multiple hashes")
            if hash_to_topology.setdefault(topology_hash, topology) != topology:
                raise StudyValidationError("distinct topologies resolved to one hash")
            hierarchy[topology_hash].add(int(expected_config["optimizer_seed"]))

        if set(records_by_id) != set(expected_configs):
            raise StudyValidationError("submission-like record IDs do not match plan")
        if len(topology_to_hash) != TOPOLOGIES or any(
            seeds != SEEDS for seeds in hierarchy.values()
        ):
            raise StudyValidationError("submission-like topology hierarchy is broken")
        records = [records_by_id[run_id] for run_id in sorted(records_by_id)]
        try:
            lines = _read_member(archive, "runs.jsonl").decode(
                "utf-8", errors="strict"
            ).splitlines()
        except UnicodeDecodeError as error:
            raise StudyValidationError("runs.jsonl is not UTF-8") from error
        if len(lines) != RUNS:
            raise StudyValidationError("runs.jsonl must contain exactly 20 records")
        jsonl_records = [
            _require_mapping(strict_json_loads(line, "runs.jsonl line"), "run")
            for line in lines
        ]
        if strict_json(jsonl_records) != strict_json(records):
            raise StudyValidationError("runs.jsonl does not match per-run records")

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
        "optimizer_seed_repetitions": RUNS,
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


def load_submission_like_summary_after_reproduction(
    study: ValidatedStudy,
    reproduction_agreement: dict[str, object],
) -> dict[str, object]:
    """Open archived summary only after both raw-data replays fully agree."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("submission-like summary receipt is invalid")
    if (
        reproduction_agreement.get("status") != "matched"
        or reproduction_agreement.get("topology_values_compared") != TOPOLOGIES
        or reproduction_agreement.get("runs_compared") != RUNS
        or reproduction_agreement.get("frozen_criteria_compared") != 5
    ):
        raise StudyValidationError(
            "submission-like summary remains locked until two raw replays agree"
        )
    with zipfile.ZipFile(study.sources.archive, "r") as archive:
        return _load_json_member(archive, "summary.json")
