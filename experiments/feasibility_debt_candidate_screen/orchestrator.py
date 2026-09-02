"""Receipt-gated two-stage orchestrator for the frozen candidate screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from .analysis import (
    AnalysisError,
    authenticate_run_document,
    evaluate_stage2,
    load_history_npz,
    select_stage1_finalist,
)
from .archive import (
    inspect_stage_archive,
    load_stage_archive_evidence,
    seal_stage_archive,
)
from .canonical import (
    ReceiptError,
    canonical_json_bytes,
    exclusive_write_bytes,
    parse_canonical_json,
    read_receipt,
    sha256_bytes,
    sha256_file,
    write_receipt,
)
from .contract import (
    ARM_ORDER,
    MAX_TIME_SECONDS,
    N_FREQUENCIES,
    OFFICIAL_ARCHIVE_SHA256,
    PANEL_SEED_ATTEMPTS,
    PANEL_SEED_START,
    POPULATION_SIZE,
    PRIOR_PANEL_SHA256,
    SMOKE_TOPOLOGY_SEED,
    STAGE1_OPTIMIZER_SEED,
    STAGE2_OPTIMIZER_SEED,
    STUDY_ID,
    UPSTREAM_REFERENCE,
    arm_spec,
    run_id,
    stage1_order,
    stage2_order,
)
from .evidence import ReplayAgreement, compare_detached_summary, compare_replays
from .detached_analysis import detached_stage1, detached_stage2
from .packet import (
    MAX_PACKET_BYTES,
    PacketError,
    parse_worker_packet,
    write_closed_failure_receipt,
)
from .panel import enumerate_legal_splits, topology_features
from .reference_analysis import reference_stage1, reference_stage2
from .runtime import DeadlineController, Phase


class OrchestratorError(RuntimeError):
    pass


class WorkerInvocationError(OrchestratorError):
    def __init__(
        self,
        code: str,
        *,
        stdout: bytes,
        stderr: bytes,
        returncode: int | None,
        timed_out: bool,
        stdout_sha256: str | None = None,
        stdout_size_bytes: int | None = None,
        stderr_sha256: str | None = None,
        stderr_size_bytes: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out
        self.stdout_sha256 = stdout_sha256 or sha256_bytes(stdout)
        self.stdout_size_bytes = (
            len(stdout) if stdout_size_bytes is None else stdout_size_bytes
        )
        self.stderr_sha256 = stderr_sha256 or sha256_bytes(stderr)
        self.stderr_size_bytes = (
            len(stderr) if stderr_size_bytes is None else stderr_size_bytes
        )


SPLIT_KEYS = {
    "panel_sha256",
    "candidate_rows",
    "legal_split_rows",
    "chosen_stage1_indices",
    "chosen_stage2_indices",
    "stratum_counts",
    "independent_verification",
}
SELECTION_KEYS = {
    "panel_sha256",
    "split_receipt_sha256",
    "stage1_archive_sha256",
    "ordered_run_ids",
    "challenger_rows",
    "eligible_ids",
    "finalist",
    "action",
    "stage2_outcome_opened",
}
STAGE1_VERIFICATION_KEYS = {
    "status",
    "stage",
    "values_compared",
    "archive_sha256",
    "ordered_run_ids",
    "production_sha256",
    "reference_sha256",
    "panel_sha256",
    "split_receipt_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "selection_receipt_sha256",
    "detached_sha256",
    "detached_values_compared",
}
STAGE2_VERIFICATION_KEYS = {
    "status",
    "stage",
    "values_compared",
    "archive_sha256",
    "ordered_run_ids",
    "production_sha256",
    "reference_sha256",
    "panel_sha256",
    "split_receipt_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "selection_receipt_sha256",
    "detached_sha256",
    "detached_values_compared",
    "finalist",
    "action",
}
STAGE2_RESULT_KEYS = {
    "stage2_archive_sha256",
    "stage2_verification_sha256",
    "finalist",
    "topology_rows",
    "differences",
    "wins",
    "ties",
    "losses",
    "mean_difference",
    "median_difference",
    "p90_harm",
    "maximum_harm",
    "bootstrap_mean_95",
    "passed",
    "action",
    "stage2_outcome_opened",
}
TERMINAL_OUTCOME_KEYS = {
    "status",
    "action",
    "revision",
    "panel_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "terminal_attempt_sha256",
    "failed_phase",
    "error_code",
    "selection_receipt_sha256",
    "stage2_verification_sha256",
    "stage2_outcome_opened",
    "organizer_score_comparable",
    "raw_output_included",
}


@dataclass(frozen=True)
class LockDigests:
    source_lock_sha256: str
    runtime_lock_sha256: str
    revision: str
    package_closure_sha256: str
    panel_commitment_sha256: str


_SMOKE_SENTINEL = object()


@dataclass(frozen=True)
class SmokeAuthorization:
    receipt_sha256: str
    panel_commitment_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    revision: str
    provider_launch_receipt_sha256: str
    resource_manifest_sha256: str
    hard_stop_receipt_sha256: str
    hard_horizon_utc: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _SMOKE_SENTINEL:
            raise OrchestratorError("smoke authorizations are verifier-issued only")


_STAGE1_SENTINEL = object()
_STAGE2_DISPATCH_SENTINEL = object()
_STAGE2_RESULT_SENTINEL = object()
_TERMINAL_ATTEMPT_SENTINEL = object()
_TERMINAL_OUTCOME_SENTINEL = object()


@dataclass(frozen=True)
class TerminalAttemptAuthorization:
    receipt_sha256: str
    revision: str
    panel_sha256: str
    source_lock_sha256: str
    attempt_root_anchor_sha256: str
    _attempt_root: Path
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _TERMINAL_ATTEMPT_SENTINEL:
            raise OrchestratorError("terminal-attempt authorizations are claimer-issued")
        for digest in (
            self.receipt_sha256,
            self.panel_sha256,
            self.source_lock_sha256,
            self.attempt_root_anchor_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise OrchestratorError("terminal-attempt authorization digest is invalid")
        if (
            not isinstance(self.revision, str)
            or len(self.revision) != 40
            or any(token not in "0123456789abcdef" for token in self.revision)
        ):
            raise OrchestratorError("terminal-attempt authorization revision is invalid")


def assert_terminal_attempt(value: object) -> TerminalAttemptAuthorization:
    if (
        not isinstance(value, TerminalAttemptAuthorization)
        or value._sentinel is not _TERMINAL_ATTEMPT_SENTINEL
    ):
        raise OrchestratorError("operation lacks the global terminal-attempt claim")
    return value


def assert_prepared_attempt_root(
    authorization: TerminalAttemptAuthorization, attempt_root: Path
) -> Path:
    authorization = assert_terminal_attempt(authorization)
    expected_root = authorization._attempt_root
    try:
        observed_root = attempt_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise OrchestratorError("pre-provision attempt root is absent") from error
    anchor = observed_root / "control" / "attempt-root-anchor.json"
    sidecar = anchor.with_name(anchor.name + ".sha256")
    expected_members = {
        "control/attempt-root-anchor.json",
        "control/attempt-root-anchor.json.sha256",
    }
    observed_members = {
        path.relative_to(observed_root).as_posix()
        for path in observed_root.rglob("*")
        if path.is_file()
    }
    if (
        observed_root != expected_root
        or observed_members != expected_members
        or not sidecar.is_file()
        or sha256_file(anchor) != authorization.attempt_root_anchor_sha256
    ):
        raise OrchestratorError("pre-provision attempt root binding changed")
    return observed_root


@dataclass(frozen=True)
class Stage1Authorization:
    selection_receipt_sha256: str
    verification_receipt_sha256: str
    archive_sha256: str
    panel_sha256: str
    split_receipt_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    revision: str
    finalist: str | None
    action: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _STAGE1_SENTINEL:
            raise OrchestratorError("Stage-1 authorizations are sealer-issued only")
        for digest in (
            self.selection_receipt_sha256,
            self.verification_receipt_sha256,
            self.archive_sha256,
            self.panel_sha256,
            self.split_receipt_sha256,
            self.source_lock_sha256,
            self.runtime_lock_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise OrchestratorError("Stage-1 authorization digest is invalid")
        if (
            not isinstance(self.revision, str)
            or len(self.revision) != 40
            or any(token not in "0123456789abcdef" for token in self.revision)
            or self.finalist not in {None, *ARM_ORDER[1:]}
            or self.action
            != (
                "retain_round1_control_stage1_failed"
                if self.finalist is None
                else "advance_selected_finalist_to_stage2"
            )
        ):
            raise OrchestratorError("Stage-1 authorization identity is invalid")


@dataclass(frozen=True)
class Stage2DispatchAuthorization:
    configs_sha256: str
    selection_receipt_sha256: str
    stage1_verification_sha256: str
    stage1_archive_sha256: str
    finalist: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _STAGE2_DISPATCH_SENTINEL:
            raise OrchestratorError("Stage-2 dispatch authorizations are builder-issued")
        for digest in (
            self.configs_sha256,
            self.selection_receipt_sha256,
            self.stage1_verification_sha256,
            self.stage1_archive_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise OrchestratorError("Stage-2 dispatch digest is invalid")
        if self.finalist not in ARM_ORDER[1:]:
            raise OrchestratorError("Stage-2 dispatch finalist is invalid")


@dataclass(frozen=True)
class Stage2ResultAuthorization:
    result_bytes: bytes
    result_sha256: str
    selection_receipt_sha256: str
    stage2_verification_sha256: str
    stage2_archive_sha256: str
    revision: str
    panel_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    finalist: str
    action: str
    passed: bool
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _STAGE2_RESULT_SENTINEL:
            raise OrchestratorError("Stage-2 result authorizations are sealer-issued only")
        if not isinstance(self.result_bytes, bytes):
            raise OrchestratorError("Stage-2 result authorization bytes are invalid")
        for digest in (
            self.result_sha256,
            self.selection_receipt_sha256,
            self.stage2_verification_sha256,
            self.stage2_archive_sha256,
            self.panel_sha256,
            self.source_lock_sha256,
            self.runtime_lock_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise OrchestratorError("Stage-2 result authorization digest is invalid")
        if (
            sha256_bytes(self.result_bytes) != self.result_sha256
            or not isinstance(self.revision, str)
            or len(self.revision) != 40
            or any(token not in "0123456789abcdef" for token in self.revision)
            or self.finalist not in ARM_ORDER[1:]
            or type(self.passed) is not bool
            or self.action
            not in {
                "review_selected_bundle_for_round2_candidate_integration",
                "retain_round1_control",
            }
        ):
            raise OrchestratorError("Stage-2 result authorization identity is invalid")


@dataclass(frozen=True)
class TerminalOutcomeAuthorization:
    receipt_sha256: str
    status: str
    action: str
    error_code: str | None
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _TERMINAL_OUTCOME_SENTINEL:
            raise OrchestratorError("terminal outcomes are verifier-issued only")
        if (
            not isinstance(self.receipt_sha256, str)
            or len(self.receipt_sha256) != 64
            or any(token not in "0123456789abcdef" for token in self.receipt_sha256)
        ):
            raise OrchestratorError("terminal outcome digest is invalid")


def assert_terminal_outcome(value: object) -> TerminalOutcomeAuthorization:
    if (
        not isinstance(value, TerminalOutcomeAuthorization)
        or value._sentinel is not _TERMINAL_OUTCOME_SENTINEL
    ):
        raise OrchestratorError("terminal outcome lacks verifier authorization")
    return value


def _load_panel(panel_path: Path, expected_sha256: str) -> dict[str, object]:
    content = panel_path.read_bytes()
    if sha256_bytes(content) != expected_sha256:
        raise OrchestratorError("panel digest mismatch")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrchestratorError("panel is invalid JSON") from error
    if canonical_json_bytes(value) != content:
        raise OrchestratorError("panel bytes are not canonical")
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "panel_id",
        "generation",
        "members",
        "topologies",
    }:
        raise OrchestratorError("panel schema mismatch")
    if value["format_version"] != 1 or value["panel_id"] != STUDY_ID:
        raise OrchestratorError("panel identity mismatch")
    members = value["members"]
    topologies = value["topologies"]
    if (
        not isinstance(members, list)
        or len(members) != 8
        or not isinstance(topologies, list)
        or topologies != [row.get("topology") for row in members if isinstance(row, dict)]
        or len(set(topologies)) != 8
    ):
        raise OrchestratorError("panel member schema mismatch")
    return value


def _panel_candidate_rows(panel: dict[str, object]) -> list[dict[str, object]]:
    members = panel["members"]
    if not isinstance(members, list):
        raise OrchestratorError("panel members are absent")
    result: list[dict[str, object]] = []
    for index, row in enumerate(members):
        if not isinstance(row, dict) or set(row) != {
            "topology_seed",
            "topology",
            "readout",
            "directional_interior_count",
            "squeezer_count",
            "directional_bin",
            "squeezer_bin",
        }:
            raise OrchestratorError("panel member row schema mismatch")
        topology = row["topology"]
        if not isinstance(topology, str) or topology_features(topology) != {
            key: row[key]
            for key in (
                "readout",
                "directional_interior_count",
                "squeezer_count",
                "directional_bin",
                "squeezer_bin",
            )
        }:
            raise OrchestratorError("panel member feature replay mismatch")
        seed = row["topology_seed"]
        if type(seed) is not int or seed not in range(
            PANEL_SEED_START, PANEL_SEED_START + PANEL_SEED_ATTEMPTS
        ):
            raise OrchestratorError("panel member seed is outside the frozen range")
        result.append(
            {
                "selection_index": index,
                "topology_seed": seed,
                "topology_sha256": sha256_bytes(topology.encode("utf-8")),
                "readout": row["readout"],
                "squeezer_count": row["squeezer_count"],
                "squeezer_bin": row["squeezer_bin"],
                "directional_interior_count": row["directional_interior_count"],
                "directional_bin": row["directional_bin"],
            }
        )
    return result


def _panel_stratum_counts(
    rows: Sequence[dict[str, object]], indices: Sequence[int]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in ("readout", "squeezer_bin", "directional_bin"):
        counts: dict[str, int] = {}
        for index in indices:
            value = str(rows[index][field])
            counts[value] = counts.get(value, 0) + 1
        result[field] = dict(sorted(counts.items()))
    return result


def _authenticate_panel_bundle(
    *,
    panel_path: Path,
    panel_commitment_path: Path,
    split_receipt_path: Path,
    locks: LockDigests,
) -> tuple[dict[str, object], dict[str, object], str]:
    commitment, commitment_sha256 = read_receipt(
        panel_commitment_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="panel_commitment",
        expected_payload_keys={
            "panel_id",
            "panel_sha256",
            "official_archive",
            "prior_panels",
            "candidate_seed_start",
            "candidate_seed_attempts",
            "eligible_unique_candidates",
            "archive_overlap_count",
            "prior_panel_overlap_count",
            "smoke_topology_seed",
            "smoke_topology_sha256",
            "smoke_overlap_count",
            "upstream_reference",
        },
    )
    if commitment_sha256 != locks.panel_commitment_sha256:
        raise OrchestratorError("panel commitment/source-lock binding mismatch")
    archive = commitment["official_archive"]
    prior = commitment["prior_panels"]
    if (
        commitment["panel_id"] != STUDY_ID
        or not isinstance(archive, dict)
        or set(archive) != {"sha256", "entries", "unique_topologies"}
        or archive["sha256"] != OFFICIAL_ARCHIVE_SHA256
        or type(archive["entries"]) is not int
        or archive["entries"] < 1
        or type(archive["unique_topologies"]) is not int
        or archive["unique_topologies"] < 1
        or commitment["candidate_seed_start"] != PANEL_SEED_START
        or commitment["candidate_seed_attempts"] != PANEL_SEED_ATTEMPTS
        or type(commitment["eligible_unique_candidates"]) is not int
        or commitment["eligible_unique_candidates"] < 8
        or commitment["archive_overlap_count"] != 0
        or commitment["prior_panel_overlap_count"] != 0
        or commitment["smoke_topology_seed"] != SMOKE_TOPOLOGY_SEED
        or commitment["smoke_overlap_count"] != 0
        or commitment["upstream_reference"] != UPSTREAM_REFERENCE
    ):
        raise OrchestratorError("panel commitment frozen metadata mismatch")
    if not isinstance(prior, list) or [
        row.get("logical_id") for row in prior if isinstance(row, dict)
    ] != list(PRIOR_PANEL_SHA256):
        raise OrchestratorError("prior-panel commitment order mismatch")
    for row, (name, digest) in zip(prior, PRIOR_PANEL_SHA256.items(), strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "logical_id",
                "sha256",
                "topology_count",
                "previous_exclusion_overlap_count",
            }
            or row["logical_id"] != name
            or row["sha256"] != digest
            or type(row["topology_count"]) is not int
            or row["topology_count"] < 1
            or type(row["previous_exclusion_overlap_count"]) is not int
            or row["previous_exclusion_overlap_count"] < 0
        ):
            raise OrchestratorError("prior-panel commitment row mismatch")
    panel_sha256 = commitment["panel_sha256"]
    if not isinstance(panel_sha256, str):
        raise OrchestratorError("panel commitment digest is invalid")
    panel = _load_panel(panel_path, panel_sha256)
    generation = panel["generation"]
    if generation != {
        "method": "first eight exact round-robin eligible candidates",
        "seed_start": PANEL_SEED_START,
        "seed_attempts": PANEL_SEED_ATTEMPTS,
        "upstream_reference": UPSTREAM_REFERENCE,
    }:
        raise OrchestratorError("panel generation metadata mismatch")
    split, split_sha256 = read_receipt(
        split_receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="split_receipt",
        expected_payload_keys=SPLIT_KEYS,
    )
    candidate_rows = _panel_candidate_rows(panel)
    members = panel["members"]
    assert isinstance(members, list)
    legal, chosen = enumerate_legal_splits(members)
    stage1 = split["chosen_stage1_indices"]
    stage2 = split["chosen_stage2_indices"]
    if (
        not isinstance(stage1, list)
        or not isinstance(stage2, list)
        or any(type(index) is not int or index not in range(8) for index in stage1)
        or any(type(index) is not int or index not in range(8) for index in stage2)
    ):
        raise OrchestratorError("panel split indices are invalid")
    if (
        split["panel_sha256"] != panel_sha256
        or split["candidate_rows"] != candidate_rows
        or split["legal_split_rows"] != legal
        or stage1 != chosen["stage1_indices"]
        or stage2 != chosen["stage2_indices"]
        or set(stage1) | set(stage2) != set(range(8))
        or set(stage1) & set(stage2)
        or split["stratum_counts"]
        != {
            "stage1": _panel_stratum_counts(members, stage1),
            "stage2": _panel_stratum_counts(members, stage2),
        }
        or not isinstance(split["independent_verification"], dict)
        or split["independent_verification"]
        != {
            "status": "matched",
            "candidate_count": commitment["eligible_unique_candidates"],
            "selected_count": 8,
            "legal_split_count": len(legal),
        }
    ):
        raise OrchestratorError("panel/split independent replay mismatch")
    smoke_digest = commitment["smoke_topology_sha256"]
    if (
        not isinstance(smoke_digest, str)
        or len(smoke_digest) != 64
        or any(token not in "0123456789abcdef" for token in smoke_digest)
        or smoke_digest in {row["topology_sha256"] for row in candidate_rows}
    ):
        raise OrchestratorError("smoke topology commitment mismatch")
    return panel, split, split_sha256


def seal_panel_bundle_once(
    *,
    panel_path: Path,
    panel_commitment_path: Path,
    split_receipt_path: Path,
    sealed_dir: Path,
    locks: LockDigests,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Copy the authenticated panel contract into the evacuated evidence tree."""
    _authenticate_panel_bundle(
        panel_path=panel_path,
        panel_commitment_path=panel_commitment_path,
        split_receipt_path=split_receipt_path,
        locks=locks,
    )
    sources = (
        (panel_path, sealed_dir / "panel.json"),
        (panel_commitment_path, sealed_dir / "panel-commitment.json"),
        (
            panel_commitment_path.with_name(panel_commitment_path.name + ".sha256"),
            sealed_dir / "panel-commitment.json.sha256",
        ),
        (split_receipt_path, sealed_dir / "split-receipt.json"),
        (
            split_receipt_path.with_name(split_receipt_path.name + ".sha256"),
            sealed_dir / "split-receipt.json.sha256",
        ),
    )
    for source, destination in sources:
        if not source.is_file():
            raise OrchestratorError("panel evidence source is absent")
        exclusive_write_bytes(destination, source.read_bytes())
    return _authenticate_panel_bundle(
        panel_path=sealed_dir / "panel.json",
        panel_commitment_path=sealed_dir / "panel-commitment.json",
        split_receipt_path=sealed_dir / "split-receipt.json",
        locks=locks,
    )


def _bind_stage_configs_to_panel(
    configs: Sequence[dict[str, object]],
    *,
    stage: int,
    panel: dict[str, object],
    split: dict[str, object],
    split_receipt_sha256: str,
    locks: LockDigests,
    selection_receipt_sha256: str | None,
) -> tuple[list[int], str | None]:
    observed_stage, indices, finalist = _validate_exact_stage_configs(configs)
    expected_indices = split[
        "chosen_stage1_indices" if stage == 1 else "chosen_stage2_indices"
    ]
    topologies = panel.get("topologies")
    if (
        observed_stage != stage
        or not isinstance(expected_indices, list)
        or indices != expected_indices
        or not isinstance(topologies, list)
    ):
        raise OrchestratorError("stage archive does not use the frozen panel split")
    for config in configs:
        member_index = config["member_index"]
        arm_id = config["arm_id"]
        if type(member_index) is not int or not isinstance(arm_id, str):
            raise OrchestratorError("stage config identity is malformed")
        exact = {
            "topology": topologies[member_index],
            "topology_sha256": sha256_bytes(
                str(topologies[member_index]).encode("utf-8")
            ),
            "panel_sha256": split["panel_sha256"],
            "panel_commitment_sha256": locks.panel_commitment_sha256,
            "split_receipt_sha256": split_receipt_sha256,
            "selection_receipt_sha256": selection_receipt_sha256,
            "source_lock_sha256": locks.source_lock_sha256,
            "runtime_lock_sha256": locks.runtime_lock_sha256,
            "revision": locks.revision,
            "arm_profile": arm_spec(arm_id).lock_row(
                locks.package_closure_sha256
            ),
        }
        if any(config.get(key) != value for key, value in exact.items()):
            raise OrchestratorError("stage config escaped the external panel lock")
    return indices, finalist


def _base_config(
    *,
    stage: int,
    member_index: int,
    execution_position: int,
    within_member_position: int,
    arm_id: str,
    optimizer_seed: int,
    topology: str,
    panel_sha256: str,
    split_receipt_sha256: str,
    selection_receipt_sha256: str | None,
    locks: LockDigests,
) -> dict[str, object]:
    spec = arm_spec(arm_id)
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "run_id": run_id(stage, member_index, within_member_position, arm_id),
        "stage": stage,
        "member_index": member_index,
        "execution_position": execution_position,
        "arm_id": arm_id,
        "optimizer_seed": optimizer_seed,
        "topology": topology,
        "topology_sha256": sha256_bytes(topology.encode("utf-8")),
        "panel_sha256": panel_sha256,
        "panel_commitment_sha256": locks.panel_commitment_sha256,
        "split_receipt_sha256": split_receipt_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
        "source_lock_sha256": locks.source_lock_sha256,
        "runtime_lock_sha256": locks.runtime_lock_sha256,
        "revision": locks.revision,
        "arm_profile": spec.lock_row(locks.package_closure_sha256),
        "max_time_seconds": MAX_TIME_SECONDS,
        "max_evals": None,
        "population_size": POPULATION_SIZE,
        "n_frequencies": N_FREQUENCIES,
        "allow_cpu": False,
    }


def build_stage1_configs(
    *,
    panel_path: Path,
    panel_commitment_path: Path,
    split_receipt_path: Path,
    locks: LockDigests,
) -> list[dict[str, object]]:
    panel, split, split_sha256 = _authenticate_panel_bundle(
        panel_path=panel_path,
        panel_commitment_path=panel_commitment_path,
        split_receipt_path=split_receipt_path,
        locks=locks,
    )
    indices = split["chosen_stage1_indices"]
    if not isinstance(indices, list):
        raise OrchestratorError("Stage-1 split indices are invalid")
    order = stage1_order(indices)
    configs: list[dict[str, object]] = []
    for execution_position, (member_index, arm_id) in enumerate(order):
        within = sum(1 for item in configs if item["member_index"] == member_index)
        configs.append(
            _base_config(
                stage=1,
                member_index=member_index,
                execution_position=execution_position,
                within_member_position=within,
                arm_id=arm_id,
                optimizer_seed=STAGE1_OPTIMIZER_SEED,
                topology=str(panel["topologies"][member_index]),
                panel_sha256=str(split["panel_sha256"]),
                split_receipt_sha256=split_sha256,
                selection_receipt_sha256=None,
                locks=locks,
            )
        )
    return configs


def build_stage2_configs(
    *,
    panel_path: Path,
    panel_commitment_path: Path,
    split_receipt_path: Path,
    selection_receipt_path: Path,
    stage1_verification_path: Path,
    stage1_archive_path: Path,
    locks: LockDigests,
    stage1_authorization: Stage1Authorization,
) -> tuple[list[dict[str, object]], Stage2DispatchAuthorization]:
    if (
        not isinstance(stage1_authorization, Stage1Authorization)
        or stage1_authorization._sentinel is not _STAGE1_SENTINEL
    ):
        raise OrchestratorError("Stage 2 lacks a sealer-issued Stage-1 authorization")
    panel, split, split_sha256 = _authenticate_panel_bundle(
        panel_path=panel_path,
        panel_commitment_path=panel_commitment_path,
        split_receipt_path=split_receipt_path,
        locks=locks,
    )
    selection, selection_sha256 = read_receipt(
        selection_receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="selection_receipt",
        expected_payload_keys=SELECTION_KEYS,
    )
    verification, verification_sha256 = read_receipt(
        stage1_verification_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="stage1_verification",
        expected_payload_keys=STAGE1_VERIFICATION_KEYS,
    )
    if (
        selection["panel_sha256"] != split["panel_sha256"]
        or selection["split_receipt_sha256"] != split_sha256
        or selection["stage2_outcome_opened"] is not False
        or verification["selection_receipt_sha256"] != selection_sha256
        or verification["status"] != "matched"
        or verification["stage"] != 1
        or verification["panel_sha256"] != split["panel_sha256"]
        or verification["split_receipt_sha256"] != split_sha256
        or verification["source_lock_sha256"] != locks.source_lock_sha256
        or verification["runtime_lock_sha256"] != locks.runtime_lock_sha256
        or selection_sha256 != stage1_authorization.selection_receipt_sha256
        or verification_sha256
        != stage1_authorization.verification_receipt_sha256
    ):
        raise OrchestratorError("Stage-1 verification/selection binding mismatch")
    ordered_run_ids = verification["ordered_run_ids"]
    stage1_indices = split["chosen_stage1_indices"]
    if not isinstance(ordered_run_ids, list) or not isinstance(stage1_indices, list):
        raise OrchestratorError("Stage-1 verification run set is invalid")
    expected_stage1_ids = [
        run_id(1, member, within, arm)
        for member in stage1_indices
        for within, arm in enumerate(
            [arm for candidate_member, arm in stage1_order(stage1_indices) if candidate_member == member]
        )
    ]
    if ordered_run_ids != expected_stage1_ids:
        raise OrchestratorError("Stage-1 verification run order mismatch")
    archive_sha256 = verification["archive_sha256"]
    if not isinstance(archive_sha256, str):
        raise OrchestratorError("Stage-1 verification archive digest is invalid")
    documents, records, reference_packets = load_stage_archive_evidence(
        stage1_archive_path,
        expected_sha256=archive_sha256,
        expected_stage=1,
        expected_run_ids=ordered_run_ids,
    )
    validate_population_pairing(records, stage=1)
    production = select_stage1_finalist(documents, stage1_indices)
    reference = reference_stage1(reference_packets, stage1_indices)
    detached = detached_stage1(reference_packets, stage1_indices)
    agreement = compare_replays(
        production,
        reference,
        stage=1,
        archive_sha256=archive_sha256,
        ordered_run_ids=ordered_run_ids,
        panel_sha256=str(split["panel_sha256"]),
        split_receipt_sha256=split_sha256,
        source_lock_sha256=locks.source_lock_sha256,
        runtime_lock_sha256=locks.runtime_lock_sha256,
    )
    detached_agreement = compare_detached_summary(
        production, reference, detached, agreement
    )
    expected_verification = {
        **agreement.receipt_payload(),
        "selection_receipt_sha256": selection_sha256,
        "detached_sha256": sha256_bytes(canonical_json_bytes(detached)),
        "detached_values_compared": detached_agreement["detached_values"],
    }
    expected_selection = {
        "panel_sha256": split["panel_sha256"],
        "split_receipt_sha256": split_sha256,
        "stage1_archive_sha256": archive_sha256,
        "ordered_run_ids": ordered_run_ids,
        **production,
    }
    if verification != expected_verification or selection != expected_selection:
        raise OrchestratorError("Stage-1 sealed replay does not authorize Stage 2")
    finalist = selection["finalist"]
    if finalist is None:
        raise OrchestratorError("Stage-1-failed branch cannot materialize Stage 2")
    if selection["action"] != "advance_selected_finalist_to_stage2":
        raise OrchestratorError("selection receipt action does not authorize Stage 2")
    if (
        stage1_authorization.archive_sha256 != archive_sha256
        or stage1_authorization.panel_sha256 != split["panel_sha256"]
        or stage1_authorization.split_receipt_sha256 != split_sha256
        or stage1_authorization.source_lock_sha256 != locks.source_lock_sha256
        or stage1_authorization.runtime_lock_sha256 != locks.runtime_lock_sha256
        or stage1_authorization.revision != locks.revision
        or stage1_authorization.finalist != finalist
        or stage1_authorization.action != selection["action"]
    ):
        raise OrchestratorError("Stage-1 authorization bindings changed")
    indices = split["chosen_stage2_indices"]
    if not isinstance(indices, list):
        raise OrchestratorError("Stage-2 split indices are invalid")
    order = stage2_order(indices, str(finalist))
    configs: list[dict[str, object]] = []
    for execution_position, (member_index, arm_id) in enumerate(order):
        within = sum(1 for item in configs if item["member_index"] == member_index)
        configs.append(
            _base_config(
                stage=2,
                member_index=member_index,
                execution_position=execution_position,
                within_member_position=within,
                arm_id=arm_id,
                optimizer_seed=STAGE2_OPTIMIZER_SEED,
                topology=str(panel["topologies"][member_index]),
                panel_sha256=str(split["panel_sha256"]),
                split_receipt_sha256=split_sha256,
                selection_receipt_sha256=selection_sha256,
                locks=locks,
            )
        )
    dispatch = Stage2DispatchAuthorization(
        configs_sha256=sha256_bytes(canonical_json_bytes(configs)),
        selection_receipt_sha256=selection_sha256,
        stage1_verification_sha256=verification_sha256,
        stage1_archive_sha256=archive_sha256,
        finalist=str(finalist),
        _sentinel=_STAGE2_DISPATCH_SENTINEL,
    )
    return configs, dispatch


def write_configs_once(directory: Path, configs: Sequence[dict[str, object]]) -> None:
    if directory.exists():
        raise OrchestratorError("config stage directory already exists")
    directory.mkdir(parents=True, exist_ok=False)
    for config in configs:
        path = directory / f"{config['run_id']}.json"
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(config))


@dataclass(frozen=True)
class WorkerCapture:
    document: dict[str, object]
    record: dict[str, object]
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class _BoundedProcessOutput:
    stdout: bytes
    stderr: bytes
    stdout_sha256: str
    stderr_sha256: str
    stdout_size_bytes: int
    stderr_size_bytes: int
    returncode: int | None
    timed_out: bool
    overflow_code: str | None
    tree_killed: bool


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _bounded_process_output(
    process: subprocess.Popen[bytes], *, timeout_seconds: float
) -> _BoundedProcessOutput:
    """Stream, hash, and cap worker pipes without an unbounded communicate()."""
    if process.stdout is None or process.stderr is None:
        raise OrchestratorError("worker pipes are unavailable")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 900:
        raise OrchestratorError("worker timeout is outside the frozen bound")
    selector = selectors.DefaultSelector()
    streams = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    for name, stream in streams.items():
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    digests = {name: hashlib.sha256() for name in streams}
    sizes = {name: 0 for name in streams}
    buffers = {name: bytearray() for name in streams}
    started = time.monotonic()
    hard_deadline = started + float(timeout_seconds)
    reap_reserve = min(1.0, float(timeout_seconds) / 10.0)
    kill_deadline = hard_deadline - reap_reserve
    termination_started = False
    timed_out = False
    overflow_code: str | None = None
    tree_killed = False
    try:
        while selector.get_map():
            now = time.monotonic()
            if not termination_started and now >= kill_deadline:
                timed_out = True
                tree_killed = True
                _kill_process_group(process.pid)
                termination_started = True
            if now >= hard_deadline:
                _kill_process_group(process.pid)
                raise OrchestratorError(
                    "worker process pipes did not close within the hard timeout"
                )
            active_deadline = hard_deadline if termination_started else kill_deadline
            wait = max(0.0, min(0.25, active_deadline - now))
            for key, _events in selector.select(wait):
                name = str(key.data)
                try:
                    chunk = os.read(key.fd, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                digests[name].update(chunk)
                sizes[name] += len(chunk)
                remaining = MAX_PACKET_BYTES - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if sizes[name] > MAX_PACKET_BYTES and overflow_code is None:
                    overflow_code = f"{name}_too_large"
                    tree_killed = True
                    _kill_process_group(process.pid)
                    termination_started = True
        try:
            remaining = max(0.0, hard_deadline - time.monotonic())
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            tree_killed = True
            _kill_process_group(process.pid)
            raise OrchestratorError(
                "worker process did not reap within the hard timeout"
            ) from error
    except BaseException as error:
        tree_killed = True
        _kill_process_group(process.pid)
        try:
            process.wait(timeout=max(0.0, hard_deadline - time.monotonic()))
        except BaseException:
            pass
        raise WorkerInvocationError(
            "pipe_capture_failure",
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
            returncode=process.poll(),
            timed_out=timed_out,
            stdout_sha256=digests["stdout"].hexdigest(),
            stdout_size_bytes=sizes["stdout"],
            stderr_sha256=digests["stderr"].hexdigest(),
            stderr_size_bytes=sizes["stderr"],
        ) from error
    finally:
        selector.close()
        for stream in streams.values():
            if not stream.closed:
                stream.close()
    return _BoundedProcessOutput(
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        stdout_sha256=digests["stdout"].hexdigest(),
        stderr_sha256=digests["stderr"].hexdigest(),
        stdout_size_bytes=sizes["stdout"],
        stderr_size_bytes=sizes["stderr"],
        returncode=returncode,
        timed_out=timed_out,
        overflow_code=overflow_code,
        tree_killed=tree_killed,
    )


def invoke_cold_smoke(
    *,
    smoke_dir: Path,
    config: dict[str, object],
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    panel_commitment_path: Path,
    revision: str,
    environment: dict[str, str],
    deadline: DeadlineController,
    now_utc: Callable[[], Any],
) -> SmokeAuthorization:
    """Run the sole loss-blind smoke once and issue an opaque dispatch gate."""
    from .smoke import validate_smoke_config, validate_smoke_projection

    if os.name != "posix":
        raise OrchestratorError("cold-smoke invocation requires the frozen Linux runtime")
    if deadline.phase is not Phase.SMOKE:
        raise OrchestratorError("cold smoke is outside the smoke phase")
    deadline.admit_operation("cold_smoke", now_utc())
    validated = validate_smoke_config(config)
    clock_snapshot = deadline.clock.snapshot()
    if (
        validated["provider_launch_receipt_sha256"]
        != deadline.provider_launch_receipt_sha256
        or validated["resource_manifest_sha256"]
        != deadline.resource_manifest_sha256
        or validated["hard_stop_receipt_sha256"]
        != deadline.hard_stop_receipt_sha256
        or validated["t0_utc"] != clock_snapshot["t0_utc"]
        or validated["b0_utc"] != clock_snapshot["b0_utc"]
        or validated["hard_horizon_utc"] != clock_snapshot["hard_horizon_utc"]
        or validated["dispatch_deadline_utc"]
        != clock_snapshot["dispatch_deadline_utc"]
    ):
        raise OrchestratorError("cold smoke is not bound to this provider launch")
    if smoke_dir.exists():
        raise OrchestratorError("cold-smoke output exists; retries are forbidden")
    smoke_dir.mkdir(parents=True, exist_ok=False)
    config_path = smoke_dir / "config.json"
    with config_path.open("xb") as handle:
        handle.write(canonical_json_bytes(validated))
    executable, sanitized_environment = _locked_process_environment(
        runtime_lock_path, environment
    )
    command = [
        executable,
        "-m",
        "experiments.feasibility_debt_candidate_screen.smoke",
        "--config",
        str(config_path),
        "--repository-root",
        str(repository_root),
        "--round1-archive",
        str(round1_archive),
        "--round1-manifest",
        str(round1_manifest),
        "--runtime-lock",
        str(runtime_lock_path),
        "--source-lock",
        str(source_lock_path),
        "--panel-commitment",
        str(panel_commitment_path),
        "--revision",
        revision,
    ]
    started = time.perf_counter()
    process_baseline = _linux_process_instances()
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=sanitized_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        start_new_session=True,
    )
    try:
        identity = _linux_process_identity(process.pid, command)
        output = _bounded_process_output(process, timeout_seconds=300.0)
        wall = time.perf_counter() - started
        if output.overflow_code is not None:
            raise WorkerInvocationError(
                output.overflow_code,
                stdout=output.stdout,
                stderr=output.stderr,
                returncode=output.returncode,
                timed_out=output.timed_out,
                stdout_sha256=output.stdout_sha256,
                stdout_size_bytes=output.stdout_size_bytes,
                stderr_sha256=output.stderr_sha256,
                stderr_size_bytes=output.stderr_size_bytes,
            )
        group_members = _linux_process_group_members(process.pid)
        namespace_survivors = sorted(_linux_process_instances() - process_baseline)
        if group_members or namespace_survivors:
            _kill_process_group(process.pid)
            raise WorkerInvocationError(
                (
                    "surviving_process_group"
                    if group_members
                    else "surviving_process_namespace"
                ),
                stdout=output.stdout,
                stderr=output.stderr,
                returncode=output.returncode,
                timed_out=output.timed_out,
            )
        packet = parse_worker_packet(
            output.stdout,
            raw_stderr=output.stderr,
            returncode=-1 if output.returncode is None else output.returncode,
            timed_out=output.timed_out,
            expected_keys={"schema_version", "study_id", "smoke"},
        )
        if packet["schema_version"] != 1 or packet["study_id"] != STUDY_ID:
            raise OrchestratorError("cold-smoke packet identity mismatch")
        smoke = validate_smoke_projection(packet["smoke"], config=validated)
        runtime = {
            "returncode": output.returncode,
            "timed_out": output.timed_out,
            "wall_seconds": wall,
            "stdout_bytes": output.stdout_size_bytes,
            "stderr_bytes": output.stderr_size_bytes,
            "stdout_sha256": output.stdout_sha256,
            "stderr_sha256": output.stderr_sha256,
            **identity,
            "timeout_tree_killed": output.tree_killed,
            "zero_descendants_after_exit": True,
        }
        (smoke_dir / "stdout.bin").write_bytes(output.stdout)
        (smoke_dir / "stderr.bin").write_bytes(output.stderr)
        receipt_path = smoke_dir / "receipt.json"
        receipt_payload = {
            "config_sha256": sha256_bytes(canonical_json_bytes(validated)),
            "smoke": smoke,
            "runtime": runtime,
        }
        receipt_digest = write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="cold_smoke",
            payload=receipt_payload,
        )
        persisted_payload, persisted_digest = read_receipt(
            receipt_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="cold_smoke",
            expected_payload_keys={"config_sha256", "smoke", "runtime"},
        )
        if (
            persisted_payload != receipt_payload
            or persisted_digest != receipt_digest
            or runtime["stdout_sha256"] != sha256_bytes(output.stdout)
            or runtime["stderr_sha256"] != sha256_bytes(output.stderr)
        ):
            raise OrchestratorError("cold-smoke receipt differs from invoked evidence")
        authorization = SmokeAuthorization(
            receipt_sha256=receipt_digest,
            panel_commitment_sha256=str(validated["panel_commitment_sha256"]),
            source_lock_sha256=str(validated["source_lock_sha256"]),
            runtime_lock_sha256=str(validated["runtime_lock_sha256"]),
            revision=str(validated["revision"]),
            provider_launch_receipt_sha256=str(
                validated["provider_launch_receipt_sha256"]
            ),
            resource_manifest_sha256=str(validated["resource_manifest_sha256"]),
            hard_stop_receipt_sha256=str(validated["hard_stop_receipt_sha256"]),
            hard_horizon_utc=str(validated["hard_horizon_utc"]),
            _sentinel=_SMOKE_SENTINEL,
        )
        deadline.accept_verified_smoke(authorization.receipt_sha256)
        return authorization
    except BaseException as error:
        _kill_process_group(process.pid)
        if isinstance(error, WorkerInvocationError):
            write_closed_failure_receipt(
                smoke_dir / "failure.json",
                raw_stdout=error.stdout,
                raw_stderr=error.stderr,
                returncode=error.returncode,
                timed_out=error.timed_out,
                error_code=error.code,
                stdout_sha256=error.stdout_sha256,
                stdout_size_bytes=error.stdout_size_bytes,
                stderr_sha256=error.stderr_sha256,
                stderr_size_bytes=error.stderr_size_bytes,
            )
        elif not (smoke_dir / "failure.json").exists():
            write_receipt(
                smoke_dir / "failure.json",
                study_id=STUDY_ID,
                receipt_type="smoke_failure",
                payload={
                    "error_type": type(error).__name__,
                    "raw_output_included": False,
                },
            )
        raise OrchestratorError("cold smoke failed; scored dispatch is forbidden") from error


def _linux_stat_fields(pid: int) -> tuple[int, int, int]:
    content = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = content.rfind(")")
    if close < 0:
        raise OrchestratorError("Linux process stat record is malformed")
    fields = content[close + 2 :].split()
    if len(fields) < 20:
        raise OrchestratorError("Linux process stat record is truncated")
    return int(fields[1]), int(fields[2]), int(fields[19])


def _linux_process_identity(pid: int, command: list[str]) -> dict[str, object]:
    parent_pid, process_group_id, start_ticks = _linux_stat_fields(pid)
    executable = Path(f"/proc/{pid}/exe").resolve(strict=True)
    command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
    expected_executable = Path(command[0]).resolve(strict=True)
    expected_command_line = b"\0".join(os.fsencode(value) for value in command) + b"\0"
    if (
        parent_pid != os.getpid()
        or process_group_id != pid
        or start_ticks < 1
        or executable != expected_executable
        or command_line != expected_command_line
    ):
        raise OrchestratorError("worker process identity boundary mismatch")
    return {
        "root_pid": pid,
        "parent_pid": parent_pid,
        "process_group_id": process_group_id,
        "start_ticks": start_ticks,
        "executable_sha256": sha256_file(executable),
        "command_line_sha256": sha256_bytes(command_line),
    }


def _linux_process_instances() -> set[tuple[int, int]]:
    instances: set[tuple[int, int]] = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            _parent, _group, start_ticks = _linux_stat_fields(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, ValueError) as error:
            raise OrchestratorError(
                f"cannot authenticate Linux process instance {path.name}"
            ) from error
        instances.add((int(path.name), start_ticks))
    return instances


def _linux_process_group_members(process_group_id: int) -> list[int]:
    members: list[int] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            _parent, observed_group, _start = _linux_stat_fields(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, ValueError) as error:
            raise OrchestratorError(
                f"cannot authenticate Linux process-group member {path.name}"
            ) from error
        if observed_group == process_group_id:
            members.append(int(path.name))
    return sorted(members)


def _locked_process_environment(
    runtime_lock_path: Path, supplied: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Derive the complete child ABI from the authenticated runtime lock."""
    from .locks import read_runtime_lock
    import platform

    runtime, _digest = read_runtime_lock(runtime_lock_path)
    python_row = runtime["python"]
    executable = str(Path(sys.executable).resolve(strict=True))
    if (
        sha256_file(Path(executable)) != python_row["interpreter_sha256"]
        or platform.python_version() != python_row["version"]
    ):
        raise OrchestratorError("current interpreter differs from the runtime lock")
    if not isinstance(supplied, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in supplied.items()
    ):
        raise OrchestratorError("runtime environment source is invalid")
    sanitized: dict[str, str] = {}
    for row in runtime["environment_allowlist"]:
        name = row["name"]
        if row["is_set"] is True:
            if (
                name not in supplied
                or sha256_bytes(supplied[name].encode("utf-8"))
                != row["value_sha256"]
            ):
                raise OrchestratorError(f"runtime environment lock mismatch: {name}")
            sanitized[name] = supplied[name]
        elif name in supplied:
            raise OrchestratorError(f"runtime environment must be unset: {name}")
    return executable, sanitized


def invoke_worker(
    *,
    config_path: Path,
    history_path: Path,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    revision: str,
    environment: dict[str, str],
    timeout_seconds: float = 900.0,
) -> WorkerCapture:
    if os.name != "posix":
        raise OrchestratorError(
            "result-bearing worker invocation is restricted to the frozen Linux runtime"
        )
    executable, sanitized_environment = _locked_process_environment(
        runtime_lock_path, environment
    )
    command = [
        executable,
        "-m",
        "experiments.feasibility_debt_candidate_screen.worker",
        "--config",
        str(config_path),
        "--history",
        str(history_path),
        "--repository-root",
        str(repository_root),
        "--round1-archive",
        str(round1_archive),
        "--round1-manifest",
        str(round1_manifest),
        "--runtime-lock",
        str(runtime_lock_path),
        "--source-lock",
        str(source_lock_path),
        "--revision",
        revision,
    ]
    started = time.perf_counter()
    process_baseline = _linux_process_instances()
    process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=sanitized_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
        )
    try:
        identity = _linux_process_identity(process.pid, command)
    except BaseException:
        _kill_process_group(process.pid)
        try:
            _bounded_process_output(process, timeout_seconds=10.0)
        except BaseException:
            pass
        raise
    output = _bounded_process_output(process, timeout_seconds=timeout_seconds)
    stdout = output.stdout
    stderr = output.stderr
    returncode = output.returncode
    timed_out = output.timed_out
    timeout_tree_killed = output.tree_killed
    wall = time.perf_counter() - started
    if output.overflow_code is not None:
        raise WorkerInvocationError(
            output.overflow_code,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            stdout_sha256=output.stdout_sha256,
            stdout_size_bytes=output.stdout_size_bytes,
            stderr_sha256=output.stderr_sha256,
            stderr_size_bytes=output.stderr_size_bytes,
        )
    group_members = _linux_process_group_members(process.pid)
    namespace_survivors = sorted(_linux_process_instances() - process_baseline)
    if group_members or namespace_survivors:
        _kill_process_group(process.pid)
        raise WorkerInvocationError(
            (
                "surviving_process_group"
                if group_members
                else "surviving_process_namespace"
            ),
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            stdout_sha256=output.stdout_sha256,
            stdout_size_bytes=output.stdout_size_bytes,
            stderr_sha256=output.stderr_sha256,
            stderr_size_bytes=output.stderr_size_bytes,
        )
    try:
        packet = parse_worker_packet(
            stdout,
            raw_stderr=stderr,
            returncode=-1 if returncode is None else returncode,
            timed_out=timed_out,
            expected_keys={"record", "run_id", "schema_version", "study_id"},
        )
    except PacketError as error:
        raise WorkerInvocationError(
            error.code,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            stdout_sha256=output.stdout_sha256,
            stdout_size_bytes=output.stdout_size_bytes,
            stderr_sha256=output.stderr_sha256,
            stderr_size_bytes=output.stderr_size_bytes,
        ) from error
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            packet["schema_version"] != 1
            or packet["study_id"] != STUDY_ID
            or packet["run_id"] != config["run_id"]
            or not isinstance(packet["record"], dict)
        ):
            raise OrchestratorError("worker packet identity mismatch")
        record = packet["record"]
        if record.get("config") != config:
            raise OrchestratorError("worker packet config mismatch")
        if (
            not history_path.is_file()
            or record.get("history", {}).get("sha256")
            != sha256_file(history_path)
        ):
            raise OrchestratorError("worker history binding mismatch")
        history_rows = load_history_npz(history_path)
        document = {
            "run_id": config["run_id"],
            "config": config,
            "history_rows": history_rows,
            "metrics": record["metrics"],
            "objective_accounting": record["objective_accounting"],
            "runtime": {
                "returncode": returncode,
                "timed_out": timed_out,
                "wall_seconds": wall,
                "stdout_bytes": len(stdout),
                "stderr_bytes": len(stderr),
                **identity,
                "timeout_tree_killed": timeout_tree_killed,
                "zero_descendants_after_exit": True,
            },
        }
        authenticate_run_document(document)
    except BaseException as error:
        raise WorkerInvocationError(
            "integrity_mismatch",
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            stdout_sha256=output.stdout_sha256,
            stdout_size_bytes=output.stdout_size_bytes,
            stderr_sha256=output.stderr_sha256,
            stderr_size_bytes=output.stderr_size_bytes,
        ) from error
    return WorkerCapture(document=document, record=record, stdout=stdout, stderr=stderr)


def _write_capture(stage_dir: Path, capture: WorkerCapture, history_source: Path) -> None:
    run_id_value = str(capture.document["run_id"])
    paths = {
        stage_dir / "records" / f"{run_id_value}.json": canonical_json_bytes(capture.record),
        stage_dir / "documents" / f"{run_id_value}.json": canonical_json_bytes(capture.document),
        stage_dir / "logs" / f"{run_id_value}.stdout.bin": capture.stdout,
        stage_dir / "logs" / f"{run_id_value}.stderr.bin": capture.stderr,
    }
    for path, content in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    destination = stage_dir / "histories" / f"{run_id_value}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise OrchestratorError("history destination already exists")
    os.replace(history_source, destination)


def _validate_exact_stage_configs(
    configs: Sequence[dict[str, object]],
) -> tuple[int, list[int], str | None]:
    from .worker import WorkerError, validate_config

    try:
        validated = [validate_config(config) for config in configs]
    except WorkerError as error:
        raise OrchestratorError("stage config schema is invalid") from error
    stages = {config["stage"] for config in validated}
    if stages not in ({1}, {2}):
        raise OrchestratorError("stage config identities are mixed")
    stage = int(validated[0]["stage"])
    indices = sorted({int(config["member_index"]) for config in validated})
    if len(indices) != 4:
        raise OrchestratorError("stage does not contain exactly four members")
    finalist: str | None = None
    if stage == 1:
        expected = stage1_order(indices)
        expected_seed = STAGE1_OPTIMIZER_SEED
    else:
        arms = {
            str(config["arm_id"])
            for config in validated
            if config["arm_id"] != ARM_ORDER[0]
        }
        if len(arms) != 1:
            raise OrchestratorError("Stage 2 does not contain one finalist")
        finalist = next(iter(arms))
        expected = stage2_order(indices, finalist)
        expected_seed = STAGE2_OPTIMIZER_SEED
    if len(validated) != len(expected):
        raise OrchestratorError("stage run set is incomplete")
    counts: dict[int, int] = {}
    shared_keys = (
        "panel_sha256",
        "panel_commitment_sha256",
        "split_receipt_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
        "revision",
    )
    shared = {key: validated[0][key] for key in shared_keys}
    for position, (config, (member, arm)) in enumerate(
        zip(validated, expected, strict=True)
    ):
        within = counts.get(member, 0)
        counts[member] = within + 1
        exact = {
            "stage": stage,
            "member_index": member,
            "arm_id": arm,
            "execution_position": position,
            "optimizer_seed": expected_seed,
            "run_id": run_id(stage, member, within, arm),
        }
        if any(config[key] != value for key, value in exact.items()) or any(
            config[key] != value for key, value in shared.items()
        ):
            raise OrchestratorError("stage config set differs from the frozen order")
    return stage, indices, finalist


def execute_stage_once(
    *,
    stage_dir: Path,
    configs: Sequence[dict[str, object]],
    invoke: Callable[[Path, Path], WorkerCapture],
    smoke_authorization: SmokeAuthorization | None = None,
    stage2_authorization: Stage2DispatchAuthorization | None = None,
    deadline: DeadlineController | None = None,
    now_utc: Callable[[], Any] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Execute a fresh serial stage once; the first failure is terminal."""
    if not configs:
        raise OrchestratorError("stage has no configs")
    stage, _indices, _finalist = _validate_exact_stage_configs(configs)
    if stage == 1:
        first = configs[0]
        if (
            not isinstance(smoke_authorization, SmokeAuthorization)
            or smoke_authorization._sentinel is not _SMOKE_SENTINEL
            or smoke_authorization.panel_commitment_sha256
            != first["panel_commitment_sha256"]
            or smoke_authorization.source_lock_sha256
            != first["source_lock_sha256"]
            or smoke_authorization.runtime_lock_sha256
            != first["runtime_lock_sha256"]
            or smoke_authorization.revision != first["revision"]
        ):
            raise OrchestratorError("Stage 1 lacks a matching cold-smoke authorization")
        if deadline is not None and (
            deadline.phase is not Phase.STAGE1
            or deadline.smoke_receipt_sha256
            != smoke_authorization.receipt_sha256
            or deadline.provider_launch_receipt_sha256
            != smoke_authorization.provider_launch_receipt_sha256
            or deadline.resource_manifest_sha256
            != smoke_authorization.resource_manifest_sha256
            or deadline.hard_stop_receipt_sha256
            != smoke_authorization.hard_stop_receipt_sha256
            or deadline.clock.snapshot()["hard_horizon_utc"]
            != smoke_authorization.hard_horizon_utc
        ):
            raise OrchestratorError("Stage-1 deadline state lacks the cold-smoke receipt")
    else:
        if (
            not isinstance(stage2_authorization, Stage2DispatchAuthorization)
            or stage2_authorization._sentinel is not _STAGE2_DISPATCH_SENTINEL
            or stage2_authorization.configs_sha256
            != sha256_bytes(canonical_json_bytes(list(configs)))
            or stage2_authorization.finalist != _finalist
            or any(
                config["selection_receipt_sha256"]
                != stage2_authorization.selection_receipt_sha256
                for config in configs
            )
        ):
            raise OrchestratorError("Stage 2 lacks its exact selection-issued dispatch gate")
        if deadline is not None and deadline.phase is not Phase.STAGE2:
            raise OrchestratorError("Stage 2 is outside the Stage-2 runtime phase")
    if stage_dir.exists():
        raise OrchestratorError("stage output already exists; retries are forbidden")
    config_dir = stage_dir / "configs"
    write_configs_once(config_dir, configs)
    temporary_dir = stage_dir / "worker-temporary"
    temporary_dir.mkdir()
    documents: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for config in configs:
        run_id_value = str(config["run_id"])
        config_path = config_dir / f"{run_id_value}.json"
        history_path = temporary_dir / f"{run_id_value}.npz"
        if deadline is not None:
            if now_utc is None:
                raise OrchestratorError("deadline execution requires a UTC clock")
            deadline.admit_scored_worker(now_utc())
        try:
            capture = invoke(config_path, history_path)
        except WorkerInvocationError as error:
            write_closed_failure_receipt(
                stage_dir / "worker-failure.json",
                raw_stdout=error.stdout,
                raw_stderr=error.stderr,
                returncode=error.returncode,
                timed_out=error.timed_out,
                error_code=error.code,
                stdout_sha256=error.stdout_sha256,
                stdout_size_bytes=error.stdout_size_bytes,
                stderr_sha256=error.stderr_sha256,
                stderr_size_bytes=error.stderr_size_bytes,
            )
            raise OrchestratorError(
                f"first worker failure halted the attempt: {error.code}"
            ) from error
        except BaseException as error:
            write_receipt(
                stage_dir / "worker-failure.json",
                study_id=STUDY_ID,
                receipt_type="worker_failure",
                payload={
                    "run_id": run_id_value,
                    "error_code": "integrity_or_controller_failure",
                    "error_type": type(error).__name__,
                    "raw_output_included": False,
                },
            )
            raise OrchestratorError(
                "first integrity/controller failure halted the attempt"
            ) from error
        _write_capture(stage_dir, capture, history_path)
        documents.append(capture.document)
        records.append(capture.record)
    try:
        temporary_dir.rmdir()
    except OSError as error:
        raise OrchestratorError("worker temporary directory is not empty") from error
    return documents, records


def claim_terminal_attempt_once(
    private_control_dir: Path,
    *,
    attempt_root: Path,
    revision: str,
    panel_sha256: str,
    source_lock_sha256: str,
) -> TerminalAttemptAuthorization:
    """Claim the sole result-bearing attempt before any smoke or dispatch."""
    private_control_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = private_control_dir / f"{STUDY_ID}.terminal-attempt.json"
    if terminal_path.exists():
        raise ReceiptError("receipt already exists")
    if attempt_root.exists():
        raise OrchestratorError("attempt root exists before the terminal claim")
    attempt_root.mkdir(parents=True, exist_ok=False)
    root_control = attempt_root / "control"
    root_control.mkdir()
    root_anchor_sha256 = write_receipt(
        root_control / "attempt-root-anchor.json",
        study_id=STUDY_ID,
        receipt_type="attempt_root_anchor",
        payload={
            "revision": revision,
            "panel_sha256": panel_sha256,
            "source_lock_sha256": source_lock_sha256,
            "fresh_before_claim": True,
        },
    )
    digest = write_receipt(
        terminal_path,
        study_id=STUDY_ID,
        receipt_type="terminal_attempt",
        payload={
            "revision": revision,
            "panel_sha256": panel_sha256,
            "source_lock_sha256": source_lock_sha256,
            "attempt_root_anchor_sha256": root_anchor_sha256,
            "no_resume": True,
            "no_retry": True,
        },
    )
    return TerminalAttemptAuthorization(
        receipt_sha256=digest,
        revision=revision,
        panel_sha256=panel_sha256,
        source_lock_sha256=source_lock_sha256,
        attempt_root_anchor_sha256=root_anchor_sha256,
        _attempt_root=attempt_root.resolve(strict=True),
        _sentinel=_TERMINAL_ATTEMPT_SENTINEL,
    )


def validate_population_pairing(records: Sequence[dict[str, object]], *, stage: int) -> None:
    expected_per_member = 4 if stage == 1 else 2
    by_member: dict[int, list[dict[str, object]]] = {}
    for record in records:
        config = record.get("config") if isinstance(record, dict) else None
        population = record.get("initial_population") if isinstance(record, dict) else None
        if not isinstance(config, dict) or not isinstance(population, dict):
            raise OrchestratorError("initial population receipt is absent")
        by_member.setdefault(int(config["member_index"]), []).append(record)
        if population.get("before_warmup") != population.get("after_warmup"):
            raise OrchestratorError("warmup state receipt differs")
    for member, group in by_member.items():
        if len(group) != expected_per_member:
            raise OrchestratorError("population pairing block is incomplete")
        raw_hashes = {
            str(record["initial_population"]["raw_population_sha256"])
            for record in group
        }
        raw_member_hashes = {
            tuple(record["initial_population"]["raw_member_sha256"])
            for record in group
        }
        if len(raw_hashes) != 1 or len(raw_member_hashes) != 1:
            raise OrchestratorError(
                f"raw initial population differs within member {member}"
            )


def seal_and_select_stage1(
    *,
    stage_dir: Path,
    archive_path: Path,
    expected_configs: Sequence[dict[str, object]],
    ordered_run_ids: list[str],
    stage1_indices: Sequence[int],
    panel_sha256: str,
    split_receipt_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    selection_receipt_path: Path,
    verification_receipt_path: Path,
) -> tuple[dict[str, object], str, str, Stage1Authorization]:
    stage, validated_indices, finalist = _validate_exact_stage_configs(
        expected_configs
    )
    if (
        stage != 1
        or finalist is not None
        or list(stage1_indices) != validated_indices
        or [config["run_id"] for config in expected_configs] != ordered_run_ids
    ):
        raise OrchestratorError("Stage-1 seal inputs differ from the frozen run set")
    archive = seal_stage_archive(
        stage_dir, archive_path, stage=1, ordered_run_ids=ordered_run_ids
    )
    inspect_stage_archive(
        archive_path,
        expected_sha256=str(archive["archive_sha256"]),
        expected_stage=1,
        expected_run_ids=ordered_run_ids,
    )
    documents, records, reference_packets = load_stage_archive_evidence(
        archive_path,
        expected_sha256=str(archive["archive_sha256"]),
        expected_stage=1,
        expected_run_ids=ordered_run_ids,
    )
    if [document["config"] for document in documents] != list(expected_configs):
        raise OrchestratorError("sealed Stage-1 configs differ from authorized configs")
    validate_population_pairing(records, stage=1)
    production = select_stage1_finalist(documents, stage1_indices)
    reference = reference_stage1(reference_packets, stage1_indices)
    detached = detached_stage1(reference_packets, stage1_indices)
    agreement = compare_replays(
        production,
        reference,
        stage=1,
        archive_sha256=str(archive["archive_sha256"]),
        ordered_run_ids=ordered_run_ids,
        panel_sha256=panel_sha256,
        split_receipt_sha256=split_receipt_sha256,
        source_lock_sha256=source_lock_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
    )
    detached_agreement = compare_detached_summary(
        production, reference, detached, agreement
    )
    payload = {
        "panel_sha256": panel_sha256,
        "split_receipt_sha256": split_receipt_sha256,
        "stage1_archive_sha256": archive["archive_sha256"],
        "ordered_run_ids": ordered_run_ids,
        **production,
    }
    digest = write_receipt(
        selection_receipt_path,
        study_id=STUDY_ID,
        receipt_type="selection_receipt",
        payload=payload,
    )
    verification_digest = write_receipt(
        verification_receipt_path,
        study_id=STUDY_ID,
        receipt_type="stage1_verification",
        payload={
            **agreement.receipt_payload(),
            "selection_receipt_sha256": digest,
            "detached_sha256": sha256_bytes(canonical_json_bytes(detached)),
            "detached_values_compared": detached_agreement["detached_values"],
        },
    )
    authorization = Stage1Authorization(
        selection_receipt_sha256=digest,
        verification_receipt_sha256=verification_digest,
        archive_sha256=str(archive["archive_sha256"]),
        panel_sha256=panel_sha256,
        split_receipt_sha256=split_receipt_sha256,
        source_lock_sha256=source_lock_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        finalist=production["finalist"] if isinstance(production["finalist"], str) else None,
        action=str(production["action"]),
        _sentinel=_STAGE1_SENTINEL,
    )
    return payload, digest, verification_digest, authorization


def seal_and_evaluate_stage2(
    *,
    stage_dir: Path,
    archive_path: Path,
    expected_configs: Sequence[dict[str, object]],
    ordered_run_ids: list[str],
    panel_path: Path,
    panel_commitment_path: Path,
    split_receipt_path: Path,
    selection_receipt_path: Path,
    stage1_verification_path: Path,
    stage1_archive_path: Path,
    locks: LockDigests,
    stage2_verification_path: Path,
    stage1_authorization: Stage1Authorization,
    stage2_authorization: Stage2DispatchAuthorization,
) -> tuple[dict[str, object], str, Stage2ResultAuthorization]:
    authorized_configs, rebuilt_dispatch = build_stage2_configs(
        panel_path=panel_path,
        panel_commitment_path=panel_commitment_path,
        split_receipt_path=split_receipt_path,
        selection_receipt_path=selection_receipt_path,
        stage1_verification_path=stage1_verification_path,
        stage1_archive_path=stage1_archive_path,
        locks=locks,
        stage1_authorization=stage1_authorization,
    )
    stage, stage2_indices, finalist = _validate_exact_stage_configs(
        expected_configs
    )
    if (
        stage != 2
        or finalist is None
        or list(expected_configs) != authorized_configs
        or stage2_authorization != rebuilt_dispatch
        or [config["run_id"] for config in expected_configs] != ordered_run_ids
    ):
        raise OrchestratorError("Stage-2 seal inputs are not Stage-1 authorized")
    archive = seal_stage_archive(
        stage_dir, archive_path, stage=2, ordered_run_ids=ordered_run_ids
    )
    inspect_stage_archive(
        archive_path,
        expected_sha256=str(archive["archive_sha256"]),
        expected_stage=2,
        expected_run_ids=ordered_run_ids,
    )
    documents, records, reference_packets = load_stage_archive_evidence(
        archive_path,
        expected_sha256=str(archive["archive_sha256"]),
        expected_stage=2,
        expected_run_ids=ordered_run_ids,
    )
    if [document["config"] for document in documents] != list(expected_configs):
        raise OrchestratorError("sealed Stage-2 configs differ from authorized configs")
    validate_population_pairing(records, stage=2)
    selection_sha256 = sha256_file(selection_receipt_path)
    if selection_sha256 != stage2_authorization.selection_receipt_sha256:
        raise OrchestratorError("Stage-2 selection receipt changed before analysis")
    production = evaluate_stage2(
        documents, stage2_indices, finalist, selection_sha256
    )
    reference = reference_stage2(
        reference_packets, stage2_indices, finalist, selection_sha256
    )
    first_config = expected_configs[0]
    agreement = compare_replays(
        production,
        reference,
        stage=2,
        archive_sha256=str(archive["archive_sha256"]),
        ordered_run_ids=ordered_run_ids,
        panel_sha256=str(first_config["panel_sha256"]),
        split_receipt_sha256=str(first_config["split_receipt_sha256"]),
        source_lock_sha256=str(first_config["source_lock_sha256"]),
        runtime_lock_sha256=str(first_config["runtime_lock_sha256"]),
    )
    detached_summary = detached_stage2(
        reference_packets, stage2_indices, finalist, selection_sha256
    )
    detached_agreement = compare_detached_summary(
        production, reference, detached_summary, agreement
    )
    verification_digest = write_receipt(
        stage2_verification_path,
        study_id=STUDY_ID,
        receipt_type="stage2_verification",
        payload={
            **agreement.receipt_payload(),
            "selection_receipt_sha256": selection_sha256,
            "detached_sha256": sha256_bytes(canonical_json_bytes(detached_summary)),
            "detached_values_compared": detached_agreement["detached_values"],
            "finalist": finalist,
            "action": production["action"],
        },
    )
    result = {
        "stage2_archive_sha256": archive["archive_sha256"],
        "stage2_verification_sha256": verification_digest,
        **production,
    }
    result_bytes = canonical_json_bytes(result)
    result_authorization = Stage2ResultAuthorization(
        result_bytes=result_bytes,
        result_sha256=sha256_bytes(result_bytes),
        selection_receipt_sha256=selection_sha256,
        stage2_verification_sha256=verification_digest,
        stage2_archive_sha256=str(archive["archive_sha256"]),
        revision=str(first_config["revision"]),
        panel_sha256=str(first_config["panel_sha256"]),
        source_lock_sha256=str(first_config["source_lock_sha256"]),
        runtime_lock_sha256=str(first_config["runtime_lock_sha256"]),
        finalist=finalist,
        action=str(production["action"]),
        passed=bool(production["passed"]),
        _sentinel=_STAGE2_RESULT_SENTINEL,
    )
    return result, verification_digest, result_authorization


def fail_closed_worker_receipt(
    path: Path,
    *,
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    timed_out: bool,
    error_code: str,
) -> str:
    return write_closed_failure_receipt(
        path,
        raw_stdout=stdout,
        raw_stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
        error_code=error_code,
    )


def write_not_evaluable_outcome_once(
    path: Path,
    *,
    revision: str,
    panel_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    terminal_attempt_sha256: str,
    failed_phase: str,
    error_code: str,
) -> str:
    return write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="terminal_outcome",
        payload={
            "status": "not_evaluable",
            "action": "retain_round1_control_attempt_not_evaluable",
            "revision": revision,
            "panel_sha256": panel_sha256,
            "source_lock_sha256": source_lock_sha256,
            "runtime_lock_sha256": runtime_lock_sha256,
            "terminal_attempt_sha256": terminal_attempt_sha256,
            "failed_phase": failed_phase,
            "error_code": error_code,
            "selection_receipt_sha256": None,
            "stage2_verification_sha256": None,
            "stage2_outcome_opened": False,
            "organizer_score_comparable": False,
            "raw_output_included": False,
        },
    )


def write_stage1_failed_outcome_once(
    path: Path,
    *,
    revision: str,
    panel_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    terminal_attempt_sha256: str,
    selection_receipt_path: Path,
    stage1_verification_path: Path,
    stage1_archive_path: Path,
    authorization: Stage1Authorization,
) -> str:
    selection, selection_sha256 = read_receipt(
        selection_receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="selection_receipt",
        expected_payload_keys=SELECTION_KEYS,
    )
    verification, verification_sha256 = read_receipt(
        stage1_verification_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="stage1_verification",
        expected_payload_keys=STAGE1_VERIFICATION_KEYS,
    )
    if (
        not isinstance(authorization, Stage1Authorization)
        or authorization._sentinel is not _STAGE1_SENTINEL
        or selection["finalist"] is not None
        or selection["action"] != "retain_round1_control_stage1_failed"
        or selection["stage2_outcome_opened"] is not False
        or selection_sha256 != authorization.selection_receipt_sha256
        or verification_sha256 != authorization.verification_receipt_sha256
        or sha256_file(stage1_archive_path) != authorization.archive_sha256
        or verification["archive_sha256"] != authorization.archive_sha256
        or verification["selection_receipt_sha256"] != selection_sha256
        or verification["status"] != "matched"
        or verification["stage"] != 1
        or verification["panel_sha256"] != authorization.panel_sha256
        or verification["split_receipt_sha256"]
        != authorization.split_receipt_sha256
        or verification["source_lock_sha256"] != authorization.source_lock_sha256
        or verification["runtime_lock_sha256"] != authorization.runtime_lock_sha256
        or selection["panel_sha256"] != authorization.panel_sha256
        or selection["split_receipt_sha256"]
        != authorization.split_receipt_sha256
        or selection["stage1_archive_sha256"] != authorization.archive_sha256
        or authorization.finalist is not None
        or authorization.action != "retain_round1_control_stage1_failed"
        or authorization.panel_sha256 != panel_sha256
        or authorization.source_lock_sha256 != source_lock_sha256
        or authorization.runtime_lock_sha256 != runtime_lock_sha256
        or authorization.revision != revision
    ):
        raise OrchestratorError("selection receipt is not a valid Stage-1 failure")
    return write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="terminal_outcome",
        payload={
            "status": "evaluable_stage1_failed",
            "action": "retain_round1_control_stage1_failed",
            "revision": revision,
            "panel_sha256": panel_sha256,
            "source_lock_sha256": source_lock_sha256,
            "runtime_lock_sha256": runtime_lock_sha256,
            "terminal_attempt_sha256": terminal_attempt_sha256,
            "failed_phase": None,
            "error_code": None,
            "selection_receipt_sha256": selection_sha256,
            "stage2_verification_sha256": None,
            "stage2_outcome_opened": False,
            "organizer_score_comparable": False,
            "raw_output_included": False,
        },
    )


def write_stage2_outcome_once(
    path: Path,
    *,
    revision: str,
    panel_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    terminal_attempt_sha256: str,
    selection_receipt_path: Path,
    stage2_verification_path: Path,
    stage2_archive_path: Path,
    authorization: Stage2ResultAuthorization,
) -> str:
    selection, selection_sha256 = read_receipt(
        selection_receipt_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="selection_receipt",
        expected_payload_keys=SELECTION_KEYS,
    )
    verification_payload, verification_sha256 = read_receipt(
        stage2_verification_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="stage2_verification",
        expected_payload_keys=STAGE2_VERIFICATION_KEYS,
    )
    if (
        not isinstance(authorization, Stage2ResultAuthorization)
        or authorization._sentinel is not _STAGE2_RESULT_SENTINEL
        or sha256_bytes(authorization.result_bytes) != authorization.result_sha256
    ):
        raise OrchestratorError("Stage-2 result lacks a sealer-issued authorization")
    result = parse_canonical_json(authorization.result_bytes)
    if not isinstance(result, dict) or set(result) != STAGE2_RESULT_KEYS:
        raise OrchestratorError("Stage-2 authorized result is not an object")
    action = result.get("action")
    passed = result.get("passed")
    wins = result.get("wins")
    mean = result.get("mean_difference")
    topology_rows = result.get("topology_rows")
    differences = result.get("differences")
    ties = result.get("ties")
    losses = result.get("losses")
    gate = bool(
        type(wins) is int
        and wins == 4
        and not isinstance(mean, bool)
        and isinstance(mean, (int, float))
        and math.isfinite(float(mean))
        and float(mean) <= -0.05
    )
    expected_action = (
        "review_selected_bundle_for_round2_candidate_integration"
        if gate
        else "retain_round1_control"
    )
    if (
        selection_sha256 != authorization.selection_receipt_sha256
        or verification_sha256 != authorization.stage2_verification_sha256
        or sha256_file(stage2_archive_path) != authorization.stage2_archive_sha256
        or result.get("stage2_archive_sha256") != authorization.stage2_archive_sha256
        or result.get("stage2_verification_sha256") != verification_sha256
        or action != expected_action
        or action != authorization.action
        or type(passed) is not bool
        or passed is not gate
        or passed is not authorization.passed
        or result.get("stage2_outcome_opened") is not True
        or not isinstance(topology_rows, list)
        or len(topology_rows) != 4
        or not isinstance(differences, list)
        or len(differences) != 4
        or type(wins) is not int
        or type(ties) is not int
        or type(losses) is not int
        or wins + ties + losses != 4
        or any(
            not isinstance(row, dict)
            or set(row) != {"member_index", "difference"}
            or type(row["member_index"]) is not int
            or isinstance(row["difference"], bool)
            or not isinstance(row["difference"], (int, float))
            or not math.isfinite(float(row["difference"]))
            for row in topology_rows
        )
        or [row["difference"] for row in topology_rows] != differences
        or len({row["member_index"] for row in topology_rows}) != 4
        or selection["finalist"] != authorization.finalist
        or selection["action"] != "advance_selected_finalist_to_stage2"
        or selection["stage2_outcome_opened"] is not False
        or authorization.revision != revision
        or authorization.panel_sha256 != panel_sha256
        or authorization.source_lock_sha256 != source_lock_sha256
        or authorization.runtime_lock_sha256 != runtime_lock_sha256
        or verification_payload["status"] != "matched"
        or verification_payload["stage"] != 2
        or verification_payload["archive_sha256"]
        != authorization.stage2_archive_sha256
        or verification_payload["selection_receipt_sha256"] != selection_sha256
        or verification_payload["panel_sha256"] != panel_sha256
        or verification_payload["source_lock_sha256"] != source_lock_sha256
        or verification_payload["runtime_lock_sha256"] != runtime_lock_sha256
        or verification_payload["finalist"] != selection["finalist"]
        or verification_payload["finalist"] != result.get("finalist")
        or verification_payload["action"] != action
    ):
        raise OrchestratorError("Stage-2 result cannot form a terminal outcome")
    result_sha256 = authorization.result_sha256
    return write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="terminal_outcome",
        payload={
            "status": "evaluable_stage2_complete",
            "action": action,
            "revision": revision,
            "panel_sha256": panel_sha256,
            "source_lock_sha256": source_lock_sha256,
            "runtime_lock_sha256": runtime_lock_sha256,
            "terminal_attempt_sha256": terminal_attempt_sha256,
            "failed_phase": None,
            "error_code": None,
            "selection_receipt_sha256": selection_sha256,
            "stage2_verification_sha256": verification_sha256,
            "stage2_result_sha256": result_sha256,
            "stage2_outcome_opened": True,
            "organizer_score_comparable": False,
            "raw_output_included": False,
        },
    )


def _reauthenticate_stage1_terminal_evidence(
    evidence_root: Path,
    *,
    panel: dict[str, object],
    split: dict[str, object],
    split_receipt_sha256: str,
    locks: LockDigests,
) -> tuple[dict[str, object], str]:
    selection_path = evidence_root / "sealed" / "selection.json"
    verification_path = evidence_root / "sealed" / "stage1-verification.json"
    archive_path = evidence_root / "sealed" / "stage1.zip"
    selection, selection_sha256 = read_receipt(
        selection_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="selection_receipt",
        expected_payload_keys=SELECTION_KEYS,
    )
    verification, _ = read_receipt(
        verification_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="stage1_verification",
        expected_payload_keys=STAGE1_VERIFICATION_KEYS,
    )
    ordered_run_ids = selection["ordered_run_ids"]
    archive_sha256 = selection["stage1_archive_sha256"]
    split_sha256 = selection["split_receipt_sha256"]
    if (
        selection["panel_sha256"] != split["panel_sha256"]
        or split_sha256 != split_receipt_sha256
        or not isinstance(ordered_run_ids, list)
        or len(ordered_run_ids) != 16
        or any(not isinstance(run_id_value, str) for run_id_value in ordered_run_ids)
        or not isinstance(archive_sha256, str)
        or not isinstance(split_sha256, str)
    ):
        raise OrchestratorError("terminal Stage-1 identity is invalid")
    documents, records, reference_packets = load_stage_archive_evidence(
        archive_path,
        expected_sha256=archive_sha256,
        expected_stage=1,
        expected_run_ids=ordered_run_ids,
    )
    configs = [document["config"] for document in documents]
    if any(not isinstance(config, dict) for config in configs):
        raise OrchestratorError("terminal Stage-1 config is invalid")
    stage1_indices, finalist = _bind_stage_configs_to_panel(
        configs,
        stage=1,
        panel=panel,
        split=split,
        split_receipt_sha256=split_receipt_sha256,
        locks=locks,
        selection_receipt_sha256=None,
    )
    if finalist is not None:
        raise OrchestratorError("terminal Stage-1 archive is not the frozen stage")
    validate_population_pairing(records, stage=1)
    production = select_stage1_finalist(documents, stage1_indices)
    reference = reference_stage1(reference_packets, stage1_indices)
    agreement = compare_replays(
        production,
        reference,
        stage=1,
        archive_sha256=archive_sha256,
        ordered_run_ids=ordered_run_ids,
        panel_sha256=str(split["panel_sha256"]),
        split_receipt_sha256=split_sha256,
        source_lock_sha256=locks.source_lock_sha256,
        runtime_lock_sha256=locks.runtime_lock_sha256,
    )
    detached = detached_stage1(reference_packets, stage1_indices)
    detached_agreement = compare_detached_summary(
        production, reference, detached, agreement
    )
    expected_selection = {
        "panel_sha256": split["panel_sha256"],
        "split_receipt_sha256": split_sha256,
        "stage1_archive_sha256": archive_sha256,
        "ordered_run_ids": ordered_run_ids,
        **production,
    }
    expected_verification = {
        **agreement.receipt_payload(),
        "selection_receipt_sha256": selection_sha256,
        "detached_sha256": sha256_bytes(canonical_json_bytes(detached)),
        "detached_values_compared": detached_agreement["detached_values"],
    }
    if selection != expected_selection or verification != expected_verification:
        raise OrchestratorError("terminal Stage-1 evidence does not replay")
    return selection, selection_sha256


def authenticate_terminal_outcome(
    path: Path,
    *,
    evidence_root: Path,
    expected_revision: str,
    expected_panel_sha256: str,
    expected_panel_commitment_sha256: str,
    expected_split_receipt_sha256: str,
    expected_package_closure_sha256: str,
    expected_source_lock_sha256: str,
    expected_runtime_lock_sha256: str,
    expected_terminal_attempt_sha256: str,
) -> TerminalOutcomeAuthorization:
    """Reopen a terminal branch from sealed bytes and independently replay it."""
    root = evidence_root.resolve(strict=True)
    expected_path = root / "sealed" / "study-outcome.json"
    if path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise OrchestratorError("terminal outcome is not at its fixed logical path")
    payload, digest = read_receipt(
        expected_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="terminal_outcome",
    )
    expected_keys = set(TERMINAL_OUTCOME_KEYS)
    if payload.get("status") == "evaluable_stage2_complete":
        expected_keys.add("stage2_result_sha256")
    if set(payload) != expected_keys:
        raise OrchestratorError("terminal outcome branch schema mismatch")
    exact = {
        "revision": expected_revision,
        "panel_sha256": expected_panel_sha256,
        "source_lock_sha256": expected_source_lock_sha256,
        "runtime_lock_sha256": expected_runtime_lock_sha256,
        "terminal_attempt_sha256": expected_terminal_attempt_sha256,
        "organizer_score_comparable": False,
        "raw_output_included": False,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise OrchestratorError("terminal outcome attempt binding mismatch")
    status = payload["status"]
    action = payload["action"]
    error_code = payload["error_code"]
    locks = LockDigests(
        source_lock_sha256=expected_source_lock_sha256,
        runtime_lock_sha256=expected_runtime_lock_sha256,
        revision=expected_revision,
        package_closure_sha256=expected_package_closure_sha256,
        panel_commitment_sha256=expected_panel_commitment_sha256,
    )
    panel: dict[str, object] | None = None
    split: dict[str, object] | None = None
    split_receipt_sha256: str | None = None
    if status != "not_evaluable":
        panel, split, split_receipt_sha256 = _authenticate_panel_bundle(
            panel_path=root / "sealed" / "panel.json",
            panel_commitment_path=root / "sealed" / "panel-commitment.json",
            split_receipt_path=root / "sealed" / "split-receipt.json",
            locks=locks,
        )
        if (
            split["panel_sha256"] != expected_panel_sha256
            or split_receipt_sha256 != expected_split_receipt_sha256
        ):
            raise OrchestratorError("terminal panel bundle changed its external lock")
    if status == "not_evaluable":
        if (
            action != "retain_round1_control_attempt_not_evaluable"
            or not isinstance(payload["failed_phase"], str)
            or not payload["failed_phase"]
            or not isinstance(error_code, str)
            or not error_code
            or payload["selection_receipt_sha256"] is not None
            or payload["stage2_verification_sha256"] is not None
            or payload["stage2_outcome_opened"] is not False
        ):
            raise OrchestratorError("not-evaluable terminal branch is malformed")
    elif status == "evaluable_stage1_failed":
        assert panel is not None and split is not None
        assert split_receipt_sha256 is not None
        selection, selection_sha256 = _reauthenticate_stage1_terminal_evidence(
            root,
            panel=panel,
            split=split,
            split_receipt_sha256=split_receipt_sha256,
            locks=locks,
        )
        if (
            action != "retain_round1_control_stage1_failed"
            or selection["finalist"] is not None
            or selection["action"] != action
            or payload["failed_phase"] is not None
            or error_code is not None
            or payload["selection_receipt_sha256"] != selection_sha256
            or payload["stage2_verification_sha256"] is not None
            or payload["stage2_outcome_opened"] is not False
        ):
            raise OrchestratorError("Stage-1 terminal branch is malformed")
    elif status == "evaluable_stage2_complete":
        assert panel is not None and split is not None
        assert split_receipt_sha256 is not None
        selection, selection_sha256 = _reauthenticate_stage1_terminal_evidence(
            root,
            panel=panel,
            split=split,
            split_receipt_sha256=split_receipt_sha256,
            locks=locks,
        )
        verification_path = root / "sealed" / "stage2-verification.json"
        archive_path = root / "sealed" / "stage2.zip"
        verification, verification_sha256 = read_receipt(
            verification_path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="stage2_verification",
            expected_payload_keys=STAGE2_VERIFICATION_KEYS,
        )
        archive_sha256 = verification["archive_sha256"]
        ordered_run_ids = verification["ordered_run_ids"]
        if (
            not isinstance(archive_sha256, str)
            or not isinstance(ordered_run_ids, list)
            or len(ordered_run_ids) != 8
            or any(not isinstance(run_id_value, str) for run_id_value in ordered_run_ids)
        ):
            raise OrchestratorError("terminal Stage-2 identity is invalid")
        documents, records, reference_packets = load_stage_archive_evidence(
            archive_path,
            expected_sha256=archive_sha256,
            expected_stage=2,
            expected_run_ids=ordered_run_ids,
        )
        configs = [document["config"] for document in documents]
        if any(not isinstance(config, dict) for config in configs):
            raise OrchestratorError("terminal Stage-2 config is invalid")
        stage2_indices, finalist = _bind_stage_configs_to_panel(
            configs,
            stage=2,
            panel=panel,
            split=split,
            split_receipt_sha256=split_receipt_sha256,
            locks=locks,
            selection_receipt_sha256=selection_sha256,
        )
        if finalist is None:
            raise OrchestratorError("terminal Stage-2 archive is not the frozen stage")
        validate_population_pairing(records, stage=2)
        production = evaluate_stage2(
            documents, stage2_indices, finalist, selection_sha256
        )
        reference = reference_stage2(
            reference_packets, stage2_indices, finalist, selection_sha256
        )
        agreement = compare_replays(
            production,
            reference,
            stage=2,
            archive_sha256=archive_sha256,
            ordered_run_ids=ordered_run_ids,
            panel_sha256=expected_panel_sha256,
            split_receipt_sha256=str(selection["split_receipt_sha256"]),
            source_lock_sha256=expected_source_lock_sha256,
            runtime_lock_sha256=expected_runtime_lock_sha256,
        )
        detached = detached_stage2(
            reference_packets, stage2_indices, finalist, selection_sha256
        )
        detached_agreement = compare_detached_summary(
            production, reference, detached, agreement
        )
        expected_verification = {
            **agreement.receipt_payload(),
            "selection_receipt_sha256": selection_sha256,
            "detached_sha256": sha256_bytes(canonical_json_bytes(detached)),
            "detached_values_compared": detached_agreement["detached_values"],
            "finalist": finalist,
            "action": production["action"],
        }
        result = {
            "stage2_archive_sha256": archive_sha256,
            "stage2_verification_sha256": verification_sha256,
            **production,
        }
        if (
            selection["finalist"] != finalist
            or selection["action"] != "advance_selected_finalist_to_stage2"
            or selection["stage2_outcome_opened"] is not False
            or verification != expected_verification
            or action != production["action"]
            or payload["failed_phase"] is not None
            or error_code is not None
            or payload["selection_receipt_sha256"] != selection_sha256
            or payload["stage2_verification_sha256"] != verification_sha256
            or payload["stage2_result_sha256"]
            != sha256_bytes(canonical_json_bytes(result))
            or payload["stage2_outcome_opened"] is not True
        ):
            raise OrchestratorError("Stage-2 terminal branch does not replay")
    else:
        raise OrchestratorError("terminal outcome status is invalid")
    if not isinstance(action, str):
        raise OrchestratorError("terminal outcome action is invalid")
    return TerminalOutcomeAuthorization(
        receipt_sha256=digest,
        status=status,
        action=action,
        error_code=error_code,
        _sentinel=_TERMINAL_OUTCOME_SENTINEL,
    )
