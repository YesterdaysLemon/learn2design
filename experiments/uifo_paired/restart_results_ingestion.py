"""Outcome-blind ingestion for the sealed patience-200 restart screen.

This module deliberately reuses the hardened byte/ZIP/record primitives from
``results_ingestion`` while keeping the restart-screen statistical replay in a
separate module.  It never parses ``summary.json`` during validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import defaultdict

from experiments.uifo_paired.plan import (
    RESTART_SCREEN_ARMS,
    TOPOLOGY_PATTERN,
    primary_pair_order_counts,
)
from experiments.uifo_paired.results_ingestion import (
    ArchiveLimits,
    ExpectedSources,
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
    strict_json_loads,
    verify_external_sources,
)
from experiments.uifo_paired.runner import _run_config, strict_json
from experiments.uifo_paired.study_profiles import bind_study_profile


RESTART_SCREEN_V1_SOURCES = ExpectedSources(
    zip_sha256="7dcf9baa6f4ce08f68248207b001984275e08851557e0394b22ebf798661251e",
    package_manifest_sha256=(
        "1ad2bee26d896928f41af8c79ce694e2e5080e474ce475842cb5542ffc0d148a"
    ),
    checksum_file_sha256=(
        "ae74633e6227d7bd0692cb02cf9aa470457a15c13e46bb414668c2bf4bd18fc0"
    ),
    plan_sha256="f093055089c580c49ee540f93f80d5eedbaf0e6be2b01fc8a619d12437698450",
    plan_id="4af939af65a5314f",
    project_revision="811ade10288562481fcacbf99306cd44ff0d4886",
)

RESTART_SCREEN_RUNS = 32
RESTART_SCREEN_PAIRS = 16
RESTART_SCREEN_TOPOLOGIES = 8
RESTART_SCREEN_SEEDS = {19, 23}
RESTART_SCREEN_POLICY_ID = "patience-200-development-screen-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _restart_pair_id(topology: str, seed: int) -> str:
    topology_id = hashlib.sha256(topology.encode()).hexdigest()[:12]
    return f"topo{topology_id}__oseed{seed:010d}"


def _validate_restart_plan(
    plan: dict[str, object], expected: ExpectedSources
) -> dict[str, dict[str, object]]:
    if plan.get("format_version") != 1 or plan.get("plan_id") != expected.plan_id:
        raise StudyValidationError("restart plan identity mismatch")
    configuration = _require_mapping(plan.get("configuration"), "plan configuration")
    if configuration.get("study_profile") != "restart-screen-v1":
        raise StudyValidationError("restart study profile mismatch")
    if "optimizer_telemetry" in configuration:
        raise StudyValidationError("restart screen must not configure telemetry")
    try:
        policy = bind_study_profile("restart-screen-v1", configuration)
    except ValueError as error:
        raise StudyValidationError(f"restart profile mismatch: {error}") from error
    if strict_json(configuration.get("decision_policy")) != strict_json(policy):
        raise StudyValidationError("restart frozen decision policy mismatch")
    policy_mapping = _require_mapping(policy, "restart decision policy")
    if policy_mapping.get("policy_id") != RESTART_SCREEN_POLICY_ID:
        raise StudyValidationError("restart decision policy ID mismatch")

    mechanics = _require_mapping(
        configuration.get("mechanics_evidence"), "mechanics evidence"
    )
    if mechanics.get("project_revision") != expected.project_revision:
        raise StudyValidationError("mechanics evidence revision mismatch")
    if configuration.get("provider_deadline_maximum_horizon_seconds") != 28_800.0:
        raise StudyValidationError("provider deadline horizon mismatch")
    if configuration.get("provider_evacuation_reserve_seconds") != 1_800.0:
        raise StudyValidationError("provider evacuation reserve mismatch")

    topologies = configuration.get("topologies")
    if not isinstance(topologies, list) or len(topologies) != RESTART_SCREEN_TOPOLOGIES:
        raise StudyValidationError("restart plan must contain exactly 8 topologies")
    topology_strings: list[str] = []
    for topology in topologies:
        if not isinstance(topology, dict) or set(topology) != {"kind", "value"}:
            raise StudyValidationError("restart topology specification is malformed")
        value = topology.get("value")
        if (
            topology.get("kind") != "string"
            or not isinstance(value, str)
            or TOPOLOGY_PATTERN.fullmatch(value) is None
        ):
            raise StudyValidationError("restart topology is not an explicit size-3 string")
        topology_strings.append(value)
    if len(set(topology_strings)) != RESTART_SCREEN_TOPOLOGIES:
        raise StudyValidationError("restart topology identities are not unique")

    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != RESTART_SCREEN_RUNS:
        raise StudyValidationError("restart plan must contain exactly 32 runs")
    if not all(isinstance(run, dict) for run in runs):
        raise StudyValidationError("restart plan run is not an object")
    run_mappings = [run for run in runs if isinstance(run, dict)]
    if [run.get("planned_run_index") for run in run_mappings] != list(
        range(RESTART_SCREEN_RUNS)
    ):
        raise StudyValidationError("restart plan indexes are not exact serial order")
    if len({str(run.get("run_id")) for run in run_mappings}) != RESTART_SCREEN_RUNS:
        raise StudyValidationError("restart plan contains duplicate run IDs")

    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in run_mappings:
        if set(run) != {
            "planned_run_index",
            "run_id",
            "pair_id",
            "run_order_within_pair",
            "topology",
            "optimizer_seed",
            "arm",
        }:
            raise StudyValidationError("restart plan run schema drift")
        topology = _require_mapping(run.get("topology"), "run topology")
        topology_value = topology.get("value")
        seed = run.get("optimizer_seed")
        arm = run.get("arm")
        if topology not in topologies or not isinstance(topology_value, str):
            raise StudyValidationError("restart run references an unknown topology")
        if seed not in RESTART_SCREEN_SEEDS or arm not in RESTART_SCREEN_ARMS:
            raise StudyValidationError("restart run seed/arm mismatch")
        pair_id = _restart_pair_id(topology_value, int(seed))
        if run.get("pair_id") != pair_id or run.get("run_id") != f"{pair_id}__{arm}":
            raise StudyValidationError("restart run or pair identity mismatch")
        if run.get("run_order_within_pair") not in (0, 1):
            raise StudyValidationError("restart within-pair order mismatch")
        by_pair[pair_id].append(run)

    if len(by_pair) != RESTART_SCREEN_PAIRS:
        raise StudyValidationError("restart plan must contain exactly 16 pairs")
    for pair_runs in by_pair.values():
        if (
            {str(run["arm"]) for run in pair_runs} != set(RESTART_SCREEN_ARMS)
            or {int(run["run_order_within_pair"]) for run in pair_runs} != {0, 1}
            or len({int(run["optimizer_seed"]) for run in pair_runs}) != 1
            or len({_canonical(run["topology"]) for run in pair_runs}) != 1
        ):
            raise StudyValidationError("restart pair hierarchy is broken")

    required_order = {
        "complete_primary_pairs": 16,
        "no_prior_p600_first": 8,
        "no_prior_p200_first": 8,
        "absolute_imbalance": 0,
    }
    if plan.get("run_order_policy") != (
        "alternate arm order by topology and optimizer-seed index"
    ):
        raise StudyValidationError("restart serial arm-order policy mismatch")
    if (
        primary_pair_order_counts(run_mappings) != required_order
        or plan.get("primary_pair_order") != required_order
    ):
        raise StudyValidationError("restart arm order is not exactly balanced")
    for seed in sorted(RESTART_SCREEN_SEEDS):
        seed_first = [
            str(run["arm"])
            for run in run_mappings
            if run["optimizer_seed"] == seed and run["run_order_within_pair"] == 0
        ]
        if seed_first.count("no_prior_p600") != 4 or seed_first.count(
            "no_prior_p200"
        ) != 4:
            raise StudyValidationError("restart per-seed arm order is not balanced")

    core = {
        "configuration": configuration,
        "run_order_policy": plan["run_order_policy"],
        "primary_pair_order": plan["primary_pair_order"],
        "runs": runs,
    }
    recomputed = hashlib.sha256(_canonical(core).encode()).hexdigest()[:16]
    if recomputed != expected.plan_id:
        raise StudyValidationError("restart plan ID does not match canonical contents")
    return {
        str(run["run_id"]): _run_config(run, configuration) for run in run_mappings
    }


def validate_restart_screen_archive(
    sources: SourcePaths,
    expected: ExpectedSources = RESTART_SCREEN_V1_SOURCES,
    limits: ArchiveLimits = ArchiveLimits(),
) -> ValidatedStudy:
    """Authenticate and validate all restart evidence except summary content."""
    source_hashes, package_manifest, external_plan = verify_external_sources(
        sources, expected, expected_run_count=RESTART_SCREEN_RUNS
    )
    integrity = inspect_zip_integrity(sources.archive, limits)
    member_names = tuple(integrity["member_names"])
    if package_manifest["archive"].get("files") != len(member_names):
        raise StudyValidationError("package manifest member count mismatch")
    if len(member_names) != 169:
        raise StudyValidationError("restart archive must contain exactly 169 members")

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
                raise StudyValidationError(
                    f"internal manifest/external plan mismatch: {key}"
                )
        expected_configs = _validate_restart_plan(manifest, expected)
        _validate_environment(manifest, expected.project_revision)

        expected_state = {
            "format_version": 1,
            "study_complete": True,
            "planned_runs": RESTART_SCREEN_RUNS,
            "completed_runs": RESTART_SCREEN_RUNS,
            "incomplete_runs": [],
        }
        if package_state != expected_state:
            raise StudyValidationError(
                "package-state.json does not prove 32/32 completion"
            )
        if session.get("status") != "complete":
            raise StudyValidationError("restart session did not complete cleanly")
        elapsed = session.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
        ):
            raise StudyValidationError("restart session elapsed time is invalid")
        if session.get("max_session_wall_seconds") != 23_400.0:
            raise StudyValidationError("restart session wall limit mismatch")

        expected_members = _expected_archive_members(expected_configs)
        observed_members = set(member_names)
        missing = sorted(expected_members - observed_members)
        unexpected = sorted(observed_members - expected_members)
        if missing or unexpected:
            raise StudyValidationError(
                f"archive member set mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )

        expected_environment = _require_mapping(
            manifest.get("environment"), "runtime environment"
        )
        configs: dict[str, dict[str, object]] = {}
        records_by_id: dict[str, dict[str, object]] = {}
        history_rows: dict[str, list[dict[str, object]]] = {}
        topology_to_hash: dict[str, str] = {}
        hash_to_topology: dict[str, str] = {}
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
            history_rows[run_id] = _validate_record(
                record, expected_config, archive, expected_environment
            )
            topology = str(record["problem"]["topology_string"])
            topology_hash = str(record["problem"]["topology_sha256"])
            previous_hash = topology_to_hash.setdefault(topology, topology_hash)
            previous_topology = hash_to_topology.setdefault(topology_hash, topology)
            if previous_hash != topology_hash or previous_topology != topology:
                raise StudyValidationError(
                    "planned and resolved topology identities are not one-to-one"
                )

        if set(records_by_id) != set(expected_configs):
            raise StudyValidationError("restart record IDs do not match the plan")
        records = [records_by_id[run_id] for run_id in sorted(records_by_id)]
        try:
            lines = _read_member(archive, "runs.jsonl").decode(
                "utf-8", errors="strict"
            ).splitlines()
        except UnicodeDecodeError as error:
            raise StudyValidationError("runs.jsonl is not UTF-8") from error
        if len(lines) != RESTART_SCREEN_RUNS:
            raise StudyValidationError("runs.jsonl must contain exactly 32 records")
        jsonl_records = [
            _require_mapping(
                strict_json_loads(line, f"runs.jsonl line {index}"), "run"
            )
            for index, line in enumerate(lines, start=1)
        ]
        if strict_json(jsonl_records) != strict_json(records):
            raise StudyValidationError("runs.jsonl does not match per-run records")

        hierarchy: dict[tuple[str, int], set[str]] = defaultdict(set)
        for record in records:
            config = _require_mapping(record.get("config"), "run config")
            hierarchy[
                (
                    str(record["problem"]["topology_sha256"]),
                    int(config["optimizer_seed"]),
                )
            ].add(str(config["arm"]))
        if (
            len(topology_to_hash) != RESTART_SCREEN_TOPOLOGIES
            or len(hash_to_topology) != RESTART_SCREEN_TOPOLOGIES
            or len(hierarchy) != RESTART_SCREEN_PAIRS
            or any(arms != set(RESTART_SCREEN_ARMS) for arms in hierarchy.values())
        ):
            raise StudyValidationError("broken restart topology/seed/arm hierarchy")

    receipt = {
        **{key: value for key, value in integrity.items() if key != "member_names"},
        "external_hashes": "passed",
        "sidecar_filename_and_digest": "passed",
        "records": RESTART_SCREEN_RUNS,
        "histories": RESTART_SCREEN_RUNS,
        "configs": RESTART_SCREEN_RUNS,
        "stdout_logs": RESTART_SCREEN_RUNS,
        "stderr_logs": RESTART_SCREEN_RUNS,
        "topologies": RESTART_SCREEN_TOPOLOGIES,
        "optimizer_seed_pairs": RESTART_SCREEN_PAIRS,
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
        integrity=receipt,
    )


def load_restart_summary_after_reproduction(
    study: ValidatedStudy,
    reproduction_agreement: dict[str, object],
) -> dict[str, object]:
    """Parse archived summary only after both raw restart replays agree."""
    if study.integrity.get("summary_content_opened") is not False:
        raise StudyValidationError("restart summary integrity receipt is invalid")
    if (
        reproduction_agreement.get("status") != "matched"
        or reproduction_agreement.get("topology_values_compared")
        != RESTART_SCREEN_TOPOLOGIES
        or reproduction_agreement.get("seed_pairs_compared") != RESTART_SCREEN_PAIRS
        or reproduction_agreement.get("frozen_criteria_compared") != 11
    ):
        raise StudyValidationError(
            "restart summary remains locked until production/reference agreement"
        )
    with zipfile.ZipFile(study.sources.archive, "r") as archive:
        return _load_json_member(archive, "summary.json")
