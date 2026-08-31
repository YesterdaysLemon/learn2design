"""Run one frozen, unpaid local mechanics study and retain private evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping


ROOT = Path(__file__).parents[1].resolve()
PRIVATE_ROOT = ROOT.with_name(f"{ROOT.name}-local-lab").resolve()
STUDY_REGISTRY_PATH = ROOT / "experiments" / "local_lab" / "studies.json"
EXPECTED_STUDY_REGISTRY_SHA256 = (
    "4dd756ee655ab6bc816eb1639ba69ecc85d678005fc224708a418aee5c637e65"
)
EXPECTED_SUBMISSION_SOURCE_SHA256 = (
    "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
)
EXPECTED_SUBMISSION_TREE_OID = "e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588"
PROTECTED_LOCAL_ARTIFACTS = {
    "artifacts/generated/submission.manifest.json": (
        "99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a"
    ),
    "artifacts/generated/submission.zip": (
        "4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b"
    ),
}
REQUIRED_SOURCE_KEYS = {
    "dependency_lock",
    "fixture_source",
    "lab_protocol",
    "study_plan",
    "worker_source",
}
V3_BOOLEAN_CASE_FIELDS = {
    "action_balance",
    "cross_episode_rejected",
    "duplicate_append_rejected",
    "exact_identity_rejected",
    "exact_positive_sets",
    "identity_fields_excluded",
    "immutable_observations",
    "invalid_sweeps_rejected",
    "only_signal_changed",
    "pending_cleared_after_rejection",
    "positive_gate_rejected",
    "reward_origin_bijection",
    "table_unchanged_after_attacks",
    "target_balance",
    "transition_involution",
    "zero_early_origin_materializations",
}
V3_STRING_CASE_FIELDS = {
    "action_dtype",
    "numpy_version",
    "observation_dtype",
    "reward_dtype",
    "structure_kind",
}
V3_EXTRA_CONTAINER_EXPECTATIONS = {
    ("all_boundary_terminal_dependency", "aggregate_lookups_by_sweep"): [
        32,
        32,
        32,
        32,
    ],
    ("reward_origin_control", "positive_cells_by_sweep"): [32, 48, 56, 60],
    ("synchronous_td_order", "aggregate_lookups_by_sweep"): [32, 32, 32, 32],
}
V3_CASE_CONTRACT_CONTAINER_FIELDS = {
    ("all_boundary_terminal_dependency", "changed_cells_by_sweep"),
    ("generator_partition", "episode_counts"),
    ("generator_partition", "regime_counts"),
    ("lazy_information_boundary", "policy_input_fields"),
    ("sanitized_result_contract", "top_level_fields"),
    ("synchronous_td_order", "positive_cells_by_sweep"),
    ("synchronous_td_order", "writes_by_sweep"),
    ("typed_episodic_contract", "action_values"),
    ("typed_episodic_contract", "event_order"),
    ("typed_episodic_contract", "observation_fields"),
    ("typed_episodic_contract", "observation_shape"),
    ("typed_episodic_contract", "policy_input_fields"),
    ("typed_episodic_contract", "reward_values"),
}
WORKER_MODULE_PATHS = {
    "experiments.local_lab.constraint_aware_progress_toy_worker": (
        "experiments/local_lab/constraint_aware_progress_toy_worker.py"
    ),
    "experiments.local_lab.multistep_td_action_prefix_v3_worker": (
        "experiments/local_lab/multistep_td_action_prefix_v3_worker.py"
    ),
    "experiments.local_lab.two_step_delayed_credit_worker": (
        "experiments/local_lab/two_step_delayed_credit_worker.py"
    ),
    "experiments.local_lab.contextual_bandit_toy_signal_worker": (
        "experiments/local_lab/contextual_bandit_toy_signal_worker.py"
    ),
    "experiments.local_lab.supervised_toy_signal_worker": (
        "experiments/local_lab/supervised_toy_signal_worker.py"
    ),
    "experiments.local_lab.full_surface_prefix_worker": (
        "experiments/local_lab/full_surface_prefix_worker.py"
    ),
    "experiments.local_lab.normal_path_jax_boundary_worker": (
        "experiments/local_lab/normal_path_jax_boundary_worker.py"
    ),
    "experiments.local_lab.feasible_progress_clock_worker": (
        "experiments/local_lab/feasible_progress_clock_worker.py"
    ),
    "experiments.local_lab.infeasible_prefix_indistinguishability_worker": (
        "experiments/local_lab/infeasible_prefix_indistinguishability_worker.py"
    ),
    "experiments.local_lab.public_signal_surface_worker": (
        "experiments/local_lab/public_signal_surface_worker.py"
    ),
    "experiments.local_lab.worker": "experiments/local_lab/worker.py",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ALLOWED_BRANCH = "codex/autonomous-local-lab"
ALLOWED_BRANCH_PREFIX = "codex/lab-"
CYCLE_TIMEOUT_SECONDS = 60 * 60
HEARTBEAT_SECONDS = 30
MAX_WORKER_OUTPUT_BYTES = 5 * 1024 * 1024
CONSTRAINT_PROGRESS_OUTPUT_BYTES = 1_048_576
OUTPUT_POLL_SECONDS = 1
STATE_SCHEMA_VERSION = 1


class DuplicateStudyError(RuntimeError):
    """A terminal study was requested again; refuse without parking state."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate JSON object key")
        value[key] = item
    return value


def _loads_json(value: bytes | str):
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("malformed JSON document") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _load_study_registry() -> dict[str, object]:
    encoded = STUDY_REGISTRY_PATH.read_bytes().replace(b"\r\n", b"\n")
    if _sha256_bytes(encoded) != EXPECTED_STUDY_REGISTRY_SHA256:
        raise RuntimeError("the approved local-lab study registry changed")
    registry = _loads_json(encoded)
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported local-lab study registry")
    studies = registry.get("studies")
    if not isinstance(studies, dict) or not studies:
        raise RuntimeError("the local-lab study registry is empty or malformed")
    return registry


def _protected_artifact_snapshot() -> dict[str, dict[str, object]]:
    snapshot = {}
    for relative_path, expected_digest in sorted(PROTECTED_LOCAL_ARTIFACTS.items()):
        path = ROOT / relative_path
        present = path.is_file()
        digest = _sha256(path) if present else None
        if present and digest != expected_digest:
            raise RuntimeError(f"protected local artifact changed: {relative_path}")
        snapshot[relative_path] = {
            "present": present,
            "sha256": digest,
        }
    return snapshot


def _source_paths(entry: dict[str, object]) -> dict[str, str]:
    source_paths = entry.get("source_paths")
    if not isinstance(source_paths, dict) or set(source_paths) != REQUIRED_SOURCE_KEYS:
        raise RuntimeError("study source paths do not match the frozen source set")
    validated = {}
    for name, relative_path in source_paths.items():
        if not isinstance(name, str) or not isinstance(relative_path, str):
            raise RuntimeError("malformed approved study source path")
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or relative_path != pure_path.as_posix()
        ):
            raise RuntimeError(f"unsafe approved study source path: {name}")
        validated[name] = relative_path
    worker_module = entry.get("worker_module")
    if not isinstance(worker_module, str) or worker_module not in WORKER_MODULE_PATHS:
        raise RuntimeError("study worker module is not allowlisted")
    if validated["worker_source"] != WORKER_MODULE_PATHS[worker_module]:
        raise RuntimeError("study worker module disagrees with its frozen source")
    return validated


def _repository_snapshot(entry: dict[str, object]) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("the local lab refuses a dirty worktree")
    branch = _git("branch", "--show-current")
    if branch != ALLOWED_BRANCH and not branch.startswith(ALLOWED_BRANCH_PREFIX):
        raise RuntimeError(f"the local lab refuses unapproved branch: {branch}")

    revision = _git("rev-parse", "HEAD")
    submission_tree_oid = _git("rev-parse", "HEAD:submission")
    if submission_tree_oid != EXPECTED_SUBMISSION_TREE_OID:
        raise RuntimeError("the protected submission tree changed")
    committed_source = _git_bytes("show", "HEAD:submission/submission.py")
    if _sha256_bytes(committed_source) != EXPECTED_SUBMISSION_SOURCE_SHA256:
        raise RuntimeError("the protected submission source digest changed")

    source_paths = _source_paths(entry)
    committed_hashes = {
        name: _sha256_bytes(_git_bytes("show", f"HEAD:{relative_path}"))
        for name, relative_path in sorted(source_paths.items())
    }
    return {
        "branch": branch,
        "committed_file_sha256": committed_hashes,
        "committed_source_paths": source_paths,
        "protected_local_artifacts": _protected_artifact_snapshot(),
        "revision": revision,
        "submission_source_sha256": EXPECTED_SUBMISSION_SOURCE_SHA256,
        "submission_tree_oid": submission_tree_oid,
        "working_tree_dirty": False,
    }


def _study_entry(
    registry: dict[str, object], study: str
) -> dict[str, object]:
    studies = registry["studies"]
    if not isinstance(studies, dict) or study not in studies:
        raise RuntimeError(f"study is not present in the approved registry: {study}")
    entry = studies[study]
    if not isinstance(entry, dict):
        raise RuntimeError(f"malformed approved study entry: {study}")
    return entry


def _validate_study_approval(
    entry: dict[str, object], snapshot: dict[str, object]
) -> None:
    approved = entry.get("approved_file_sha256")
    committed = snapshot.get("committed_file_sha256")
    committed_paths = snapshot.get("committed_source_paths")
    if (
        not isinstance(approved, dict)
        or not isinstance(committed, dict)
        or not isinstance(committed_paths, dict)
    ):
        raise RuntimeError("malformed approved study source manifest")
    if set(approved) != set(committed) or set(approved) != set(committed_paths):
        raise RuntimeError("approved study source manifest is incomplete")
    for name, expected_digest in approved.items():
        if (
            not isinstance(expected_digest, str)
            or SHA256_PATTERN.fullmatch(expected_digest) is None
        ):
            raise RuntimeError(f"malformed approved study source digest: {name}")
        if committed.get(name) != expected_digest:
            raise RuntimeError(f"approved study source changed: {name}")


def _constraint_progress_contract_sha256(
    registry: dict[str, object], entry: dict[str, object]
) -> str:
    plan_path = entry.get("plan_path")
    if not isinstance(plan_path, str):
        raise RuntimeError("constraint-progress study has no frozen plan path")
    plan_blob = _git_bytes("show", f"HEAD:{plan_path}")
    normalized_registry = json.dumps(
        registry,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(
        b"L2D-constraint-progress-v1/contract\0"
        + plan_blob
        + b"\0"
        + normalized_registry
    )


def _constraint_progress_runtime_identity() -> dict[str, str]:
    worker_path = ROOT / WORKER_MODULE_PATHS[
        "experiments.local_lab.constraint_aware_progress_toy_worker"
    ]
    command = [
        sys.executable,
        "-S",
        "-P",
        str(worker_path),
        "--mode",
        "constraint-aware-progress-toy-v1-runtime-probe",
    ]
    process = None
    with tempfile.TemporaryDirectory(prefix="l2d-runtime-probe-") as directory:
        root = Path(directory)
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        try:
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=_worker_environment(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                    start_new_session=os.name != "nt",
                )
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started > 60:
                        _terminate_process_tree(process)
                        raise RuntimeError("constraint-progress runtime probe timed out")
                    if stdout_path.stat().st_size + stderr_path.stat().st_size > 16_384:
                        _terminate_process_tree(process)
                        raise RuntimeError("constraint-progress runtime probe exceeded its cap")
                    time.sleep(0.05)
            stdout = stdout_path.read_bytes()
            stderr = stderr_path.read_bytes()
        finally:
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
    if (
        process is None
        or process.returncode != 0
        or stderr
        or not (0 < len(stdout) <= 16_384)
    ):
        raise RuntimeError("constraint-progress isolated runtime probe failed")
    value = _loads_json(stdout)
    if not isinstance(value, dict) or any(
        type(item) is not str for item in value.values()
    ):
        raise RuntimeError("constraint-progress runtime probe is malformed")
    return value


def _validate_constraint_progress_runtime(entry: dict[str, object]) -> None:
    expected = entry.get("runtime_identity")
    if not isinstance(expected, dict) or _constraint_progress_runtime_identity() != expected:
        raise RuntimeError("constraint-progress runtime identity changed")


def _constraint_metric_type(value: object, type_code: str) -> bool:
    if type_code == "B":
        return type(value) is bool
    if type_code == "I":
        return type(value) is int
    if type_code == "F":
        return type(value) is float and math.isfinite(value)
    if type_code == "H":
        return type(value) is str and SHA256_PATTERN.fullmatch(value) is not None
    if type_code == "A[B;2]":
        return (
            type(value) is list
            and len(value) == 2
            and all(type(item) is bool for item in value)
        )
    return False


def _constraint_rows(
    world_aggregates: list[object], family: str, arm: str, split: str
) -> list[dict[str, object]]:
    rows = []
    for item in world_aggregates:
        assert isinstance(item, dict)
        world = item["world"]
        assert type(world) is int
        parity = sum((world >> bit) & 1 for bit in range(4)) % 2
        row_split = "development" if parity == 0 else "heldout"
        if item["family"] == family and item["arm"] == arm and row_split == split:
            rows.append(item)
    if len(rows) != 8:
        raise RuntimeError("constraint-progress aggregate subset is incomplete")
    return rows


def _constraint_comparison(
    world_aggregates: list[object], family: str, treatment: str, baseline: str
) -> tuple[float, float, list[float], int, int, int]:
    treatment_rows = _constraint_rows(
        world_aggregates, family, treatment, "heldout"
    )
    baseline_rows = _constraint_rows(world_aggregates, family, baseline, "heldout")
    treatment_by_world = {
        int(item["world"]): float(item["mean_gap"]) for item in treatment_rows
    }
    baseline_by_world = {
        int(item["world"]): float(item["mean_gap"]) for item in baseline_rows
    }
    differences = [
        treatment_by_world[world] - baseline_by_world[world]
        for world in sorted(treatment_by_world)
    ]
    wins = sum(value < -1.0e-12 for value in differences)
    ties = sum(abs(value) <= 1.0e-12 for value in differences)
    losses = 8 - wins - ties
    return (
        float(sum(treatment_by_world.values()) / 8),
        float(sum(baseline_by_world.values()) / 8),
        differences,
        wins,
        ties,
        losses,
    )


def _constraint_normalize(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("constraint-progress nonfinite root value")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_constraint_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_constraint_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _constraint_normalize(item) for key, item in value.items()}
    return value


def _constraint_line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            _constraint_normalize(dict(value)),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _constraint_root(name: str, rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"L2D-constraint-progress-v1/")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    for row in rows:
        digest.update(_constraint_line(row))
    return digest.hexdigest()


def _constraint_world_rows() -> list[dict[str, object]]:
    rows = []
    for family in ("canonical", "aligned", "impossible"):
        for world in range(16):
            bits = [
                (world >> 3) & 1,
                (world >> 2) & 1,
                (world >> 1) & 1,
                world & 1,
            ]
            a = float(0.80 + 0.20 * bits[0])
            b = float(-0.50 + 1.00 * bits[1])
            k = float(0.50 + 0.50 * bits[2])
            t = float(0.10 + 0.06 * bits[3])
            c = float(-0.25 if (bits[0] ^ bits[1]) == 0 else 0.25)
            threshold = float(2.25 if family == "impossible" else 0.0)
            split = "development" if sum(bits) % 2 == 0 else "heldout"
            if family == "impossible":
                reference_x0 = reference_sensitivity = denominator = None
            else:
                q = -t if family == "aligned" else t
                lower = a / 2.0
                upper = a if q > 0.0 else min(1.5 * a, 1.99)
                for _ in range(80):
                    middle = (lower + upper) / 2.0
                    derivative = 4.0 * middle * (middle * middle - a * a) + q
                    if derivative <= 0.0:
                        lower = middle
                    else:
                        upper = middle
                reference_x0 = float((lower + upper) / 2.0)
                reference_sensitivity = float(
                    (reference_x0 * reference_x0 - a * a) ** 2
                    + q * reference_x0
                )
                denominator = float(
                    a**4 + k * b * b + 0.5 * c * c - reference_sensitivity
                )
            rows.append(
                {
                    "family": family,
                    "world": world,
                    "bits": bits,
                    "split": split,
                    "a": a,
                    "b": b,
                    "k": k,
                    "t": t,
                    "c": c,
                    "threshold": threshold,
                    "reference_x0": reference_x0,
                    "reference_sensitivity": reference_sensitivity,
                    "denominator": denominator,
                }
            )
    return rows


def _constraint_intervention_root() -> str:
    stored = {
        "mode": "lex",
        "observed": True,
        "feasible": False,
        "first": 1.0,
        "second": 5.0,
    }
    canonical = ((0.9, 6.0, True), (0.8, 7.0, True))
    donor = ((1.1, 0.0, False), (1.2, 0.0, False))
    ablated = ((0.0, 0.0, True), (0.0, 0.0, False))
    rows: list[dict[str, object]] = []
    for ordinal in range(2):
        rows.append(
            {
                "sentinel_id": "progress-consumer-v1",
                "ordinal": ordinal,
                "member": 0,
                "stored": stored,
                "canonical_tuple": {
                    "is_feasible": False,
                    "penalty": canonical[ordinal][0],
                    "sensitivity": canonical[ordinal][1],
                },
                "donor_member": 1,
                "donor_tuple": {
                    "is_feasible": False,
                    "penalty": donor[ordinal][0],
                    "sensitivity": donor[ordinal][1],
                },
                "ablated_tuple": {
                    "is_feasible": False,
                    "penalty": 0.0,
                    "sensitivity": 0.0,
                },
                "canonical_decision": canonical[ordinal][2],
                "donor_decision": donor[ordinal][2],
                "ablated_decision": ablated[ordinal][2],
            }
        )
    attacks = (
        "family", "split", "world", "bits", "seed", "a", "b", "k", "t", "c",
        "threshold", "reference-x0", "reference-sensitivity", "denominator", "gap",
        "environment-call", "evaluator-call", "oracle-call", "source-call",
        "canonical-aux-bypass", "future-transcript-read", "transcript-fingerprint",
    )
    for attack in attacks:
        path = (
            "restart-draw-provider.future-read"
            if attack == "future-transcript-read"
            else "restart-draw-provider.fingerprint"
            if attack == "transcript-fingerprint"
            else f"optimizer-adapter.{attack}"
        )
        rows.append(
            {
                "attack_id": attack,
                "injection_path": path,
                "rejection_code": "capability-denied",
                "consumer_reached": True,
                "state_mutations": 0,
            }
        )
    return _constraint_root("InterventionRecord", rows)


def _constraint_attack_root() -> str:
    matrix = (
        ("nan-loss", "observation.loss", "nonfinite-loss"),
        ("gradient-dtype", "observation.gradient", "gradient-dtype"),
        ("gradient-shape", "observation.gradient", "gradient-shape"),
        ("feasible-type", "observation.canonical_is_feasible", "feasible-type"),
        ("negative-penalty", "observation.decision_penalty", "negative-penalty"),
        ("duplicate-observation", "observation.identity", "duplicate-key"),
        ("missing-member", "observation.identity", "missing-member"),
        ("wrong-arm", "observation.arm", "arm-identity"),
        ("cross-seed", "transition.seed", "cross-seed-join"),
        ("cross-order", "transition.order", "cross-order-join"),
        ("extra-result-field", "result", "result-schema"),
        ("transcript-hash", "transcript.sha256", "transcript-hash"),
    )
    return _constraint_root(
        "AttackReceipt",
        [
            {
                "attack_id": attack,
                "injection_path": path,
                "rejection_code": code,
                "consumer_reached": True,
                "state_mutations": 0,
            }
            for attack, path, code in matrix
        ],
    )


def _constraint_source_root(world_aggregates: list[object]) -> str:
    world_rows = _constraint_world_rows()
    receipts = []
    for family in ("canonical", "aligned", "impossible"):
        for split in ("development", "heldout"):
            phase_worlds = [
                item
                for item in world_rows
                if item["family"] == family and item["split"] == split
            ]
            phase_aggregates = [
                item
                for item in world_aggregates
                if isinstance(item, dict)
                and item["family"] == family
                and (
                    "development"
                    if sum((int(item["world"]) >> bit) & 1 for bit in range(4)) % 2 == 0
                    else "heldout"
                )
                == split
            ]
            receipts.append(
                {
                    "family": family,
                    "split": split,
                    "world_keys": [int(item["world"]) for item in phase_worlds],
                    "attempted_reads": len(phase_worlds),
                    "forbidden_reads": 0,
                    "forbidden_payload_rows": 0,
                    "sentinel_connected": True,
                    "input_root_sha256": _constraint_root("PhaseInput", phase_worlds),
                    "output_root_sha256": _constraint_root(
                        "WorldAggregateRecord", phase_aggregates
                    ),
                }
            )
    return _constraint_root("PhaseReceipt", receipts)


def _validate_constraint_progress_result(
    entry: dict[str, object],
    result: dict[str, object],
    *,
    study_revision: str | None,
    contract_sha256: str | None,
    worker_receipt: Mapping[str, object] | None,
) -> None:
    top_level = entry.get("result_top_level_fields")
    if not isinstance(top_level, list) or tuple(result) != tuple(top_level):
        raise RuntimeError("constraint-progress result has the wrong top-level schema")
    if result.get("study_id") != "constraint-aware-progress-toy-v1":
        raise RuntimeError("constraint-progress result has the wrong study identity")
    if result.get("plan_revision") != entry.get("plan_revision"):
        raise RuntimeError("constraint-progress result has the wrong plan revision")
    if study_revision is None or result.get("study_revision") != study_revision:
        raise RuntimeError("constraint-progress result has the wrong study revision")
    if contract_sha256 is None or result.get("contract_sha256") != contract_sha256:
        raise RuntimeError("constraint-progress result has the wrong contract digest")
    transcript = entry.get("transcript_commitment")
    if (
        not isinstance(transcript, dict)
        or result.get("transcript_root_sha256") != transcript.get("root_sha256")
    ):
        raise RuntimeError("constraint-progress result has the wrong transcript root")

    families = ("canonical", "aligned", "impossible")
    arms = (
        "protected_raw_progress",
        "constraint_lexicographic_progress",
        "shuffled_progress_control",
        "ablated_progress_control",
        "no_restart_comparator",
    )
    world_aggregates = result.get("world_aggregates")
    if type(world_aggregates) is not list or len(world_aggregates) != 240:
        raise RuntimeError("constraint-progress result has the wrong aggregate count")
    expected_keys = [
        (family, world, arm)
        for family in families
        for world in range(16)
        for arm in arms
    ]
    actual_keys = []
    for row in world_aggregates:
        if not isinstance(row, dict) or tuple(row) != (
            "family",
            "world",
            "arm",
            "seed_gaps",
            "mean_gap",
        ):
            raise RuntimeError("constraint-progress aggregate schema changed")
        family = row["family"]
        world = row["world"]
        arm = row["arm"]
        seed_gaps = row["seed_gaps"]
        mean_gap = row["mean_gap"]
        if (
            type(family) is not str
            or family not in families
            or type(world) is not int
            or world not in range(16)
            or type(arm) is not str
            or arm not in arms
            or type(seed_gaps) is not list
            or len(seed_gaps) != 4
            or any(type(item) is not float or not math.isfinite(item) for item in seed_gaps)
            or type(mean_gap) is not float
            or not math.isfinite(mean_gap)
            or mean_gap != float(sum(seed_gaps) / 4)
        ):
            raise RuntimeError("constraint-progress aggregate value is malformed")
        actual_keys.append((family, world, arm))
    if actual_keys != expected_keys:
        raise RuntimeError("constraint-progress aggregate identity order changed")

    case_ids = entry.get("case_ids")
    schema = entry.get("case_metric_schema")
    cases = result.get("cases")
    if (
        not isinstance(case_ids, list)
        or not isinstance(schema, dict)
        or type(cases) is not list
        or len(cases) != 12
    ):
        raise RuntimeError("constraint-progress case contract is malformed")
    by_id: dict[str, dict[str, object]] = {}
    for expected_id, case in zip(case_ids, cases, strict=True):
        if (
            not isinstance(case, dict)
            or tuple(case) != ("case_id", "passed", "metrics")
            or case.get("case_id") != expected_id
            or type(case.get("passed")) is not bool
            or not isinstance(case.get("metrics"), dict)
        ):
            raise RuntimeError("constraint-progress case schema changed")
        fields = schema.get(expected_id)
        metrics = case["metrics"]
        assert isinstance(metrics, dict)
        if not isinstance(fields, list):
            raise RuntimeError("constraint-progress registry case schema is malformed")
        expected_names = []
        for field in fields:
            if (
                not isinstance(field, list)
                or len(field) != 2
                or not all(isinstance(item, str) for item in field)
            ):
                raise RuntimeError("constraint-progress registry metric is malformed")
            name, type_code = field
            expected_names.append(name)
            if name not in metrics or not _constraint_metric_type(metrics[name], type_code):
                raise RuntimeError(
                    f"constraint-progress metric type changed: {expected_id}.{name}"
                )
        if tuple(metrics) != tuple(expected_names):
            raise RuntimeError("constraint-progress metric order changed")
        by_id[expected_id] = case

    def metrics(case_id: str) -> dict[str, object]:
        value = by_id[case_id]["metrics"]
        assert isinstance(value, dict)
        return value

    def require(case_id: str, name: str, expected: object) -> None:
        if metrics(case_id).get(name) != expected:
            raise RuntimeError(
                f"constraint-progress derived metric disagrees: {case_id}.{name}"
            )

    structural = {
        "family_replay": {
            "world_records": 48,
            "constrained_references": 32,
            "reference_exclusions": 16,
            "development_worlds_per_family": 8,
            "heldout_worlds_per_family": 8,
            "formula_mismatches": 0,
            "reference_mismatches": 0,
            "nonpositive_denominators": 0,
            "duplicate_world_keys": 0,
        },
        "transcript_commitment": {
            "transcripts": 4,
            "values_per_transcript": 3093,
            "transcript_values": 12372,
            "trajectories": 1920,
            "evaluations": 983040,
            "unequal_arm_counts": 0,
            "order_twin_mismatches": 0,
        },
        "typed_aux_and_intervention": {
            "observations": 983040,
            "schema_valid_observations": 983040,
            "join_failures": 0,
            "capability_attacks": 22,
            "capability_rejected": 22,
            "capability_state_mutations": 0,
            "canonical_decisions": [True, True],
            "donor_decisions": [False, False],
            "ablated_decisions": [True, False],
        },
        "chronology_replay": {
            "batches": 122880,
            "batch_receipts": 122880,
            "transitions": 983040,
            "replay_mismatches": 0,
            "order_mismatches": 0,
            "reset_mismatches": 0,
            "incumbent_tie_mismatches": 0,
            "incumbent_state_mismatches": 0,
        },
        "development_and_source_isolation": {
            "development_aggregates": 120,
            "development_receipts": 3,
            "heldout_receipts": 3,
            "forbidden_reads": 0,
            "heldout_source_in_development": 0,
            "development_outputs_in_heldout": 0,
        },
        "impossible_control": {
            "trajectories": 640,
            "observations": 327680,
            "feasible_observations": 0,
            "nonunit_gaps": 0,
            "references_used": 0,
            "false_feasible_joins": 0,
        },
        "process_and_sanitizer": {
            "launches": 2,
            "projections_equal": True,
            "stderr_bytes": 0,
            "surviving_children": 0,
            "attacks": 12,
            "attacks_rejected": 12,
            "attack_state_mutations": 0,
        },
    }
    for case_id, expected_fields in structural.items():
        for name, expected in expected_fields.items():
            require(case_id, name, expected)

    for case_id, left, right in (
        ("family_replay", "implementation_root_sha256", "oracle_root_sha256"),
        ("transcript_commitment", "committed_root_sha256", "observed_root_sha256"),
        ("chronology_replay", "implementation_state_root_sha256", "oracle_state_root_sha256"),
        ("development_and_source_isolation", "implementation_source_root_sha256", "oracle_source_root_sha256"),
        ("process_and_sanitizer", "implementation_attack_root_sha256", "oracle_attack_root_sha256"),
    ):
        equal = metrics(case_id)[left] == metrics(case_id)[right]
        require(case_id, "roots_equal", equal)
    typed_metrics = metrics("typed_aux_and_intervention")
    typed_roots_equal = (
        typed_metrics["implementation_schema_root_sha256"]
        == typed_metrics["oracle_schema_root_sha256"]
        and typed_metrics["implementation_intervention_root_sha256"]
        == typed_metrics["oracle_intervention_root_sha256"]
    )
    require("typed_aux_and_intervention", "roots_equal", typed_roots_equal)
    expected_family_root = _constraint_root("WorldRecord", _constraint_world_rows())
    for name in ("implementation_root_sha256", "oracle_root_sha256"):
        require("family_replay", name, expected_family_root)
    expected_intervention_root = _constraint_intervention_root()
    for name in (
        "implementation_intervention_root_sha256",
        "oracle_intervention_root_sha256",
    ):
        require("typed_aux_and_intervention", name, expected_intervention_root)
    expected_source_root = _constraint_source_root(world_aggregates)
    for name in (
        "implementation_source_root_sha256",
        "oracle_source_root_sha256",
    ):
        require("development_and_source_isolation", name, expected_source_root)
    expected_attack_root = _constraint_attack_root()
    for name in (
        "implementation_attack_root_sha256",
        "oracle_attack_root_sha256",
    ):
        require("process_and_sanitizer", name, expected_attack_root)
    require(
        "transcript_commitment",
        "committed_root_sha256",
        transcript["root_sha256"],
    )
    if metrics("chronology_replay")["restart_events"] < 0:
        raise RuntimeError("constraint-progress restart count is negative")
    inner_stdout_bytes = metrics("process_and_sanitizer")["maximum_stdout_bytes"]
    if not (0 <= inner_stdout_bytes <= CONSTRAINT_PROGRESS_OUTPUT_BYTES):
        raise RuntimeError("constraint-progress inner projection exceeded its cap")
    if (
        not isinstance(worker_receipt, dict)
        or tuple(worker_receipt)
        != ("stderr_bytes", "stderr_sha256", "stdout_bytes")
        or type(worker_receipt.get("stdout_bytes")) is not int
        or not (0 < worker_receipt["stdout_bytes"] <= CONSTRAINT_PROGRESS_OUTPUT_BYTES)
        or worker_receipt.get("stderr_bytes") != 0
        or worker_receipt.get("stderr_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError("constraint-progress outer worker receipt is invalid")

    treatment, baseline, differences, wins, ties, losses = _constraint_comparison(
        world_aggregates,
        "canonical",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    improvement = baseline - treatment
    harm = max(differences)
    primary_expected = {
        "treatment_mean_gap": treatment,
        "baseline_mean_gap": baseline,
        "mean_improvement": improvement,
        "heldout_wins": wins,
        "heldout_ties": ties,
        "heldout_losses": losses,
        "maximum_signed_world_harm": harm,
        "mean_gate": improvement >= 0.05,
        "win_gate": wins >= 6,
        "harm_gate": harm <= 0.15,
    }
    for name, expected in primary_expected.items():
        require("heldout_primary", name, expected)

    no_restart, treatment_again, _differences, *_ = _constraint_comparison(
        world_aggregates,
        "canonical",
        "no_restart_comparator",
        "constraint_lexicographic_progress",
    )
    if treatment_again != treatment:
        raise RuntimeError("constraint-progress comparator treatment join changed")
    comparator_improvement = no_restart - treatment
    comparator_expected = {
        "treatment_mean_gap": treatment,
        "no_restart_mean_gap": no_restart,
        "mean_improvement": comparator_improvement,
        "minimum_arm_evaluations": 512,
        "maximum_arm_evaluations": 512,
        "evaluation_parity": True,
        "transcript_parity": True,
        "comparator_gate": comparator_improvement >= 0.05,
    }
    for name, expected in comparator_expected.items():
        require("restart_comparators", name, expected)

    control_passes = {}
    for case_id, arm in (
        ("shuffled_signal_control", "shuffled_progress_control"),
        ("ablated_signal_control", "ablated_progress_control"),
    ):
        control, control_baseline, control_diff, control_wins, *_ = _constraint_comparison(
            world_aggregates, "canonical", arm, "protected_raw_progress"
        )
        control_improvement = control_baseline - control
        control_harm = max(control_diff)
        mean_gate = control_improvement >= 0.05
        win_gate = control_wins >= 6
        harm_gate = control_harm <= 0.15
        recovered = mean_gate and win_gate and harm_gate
        expected_fields = {
            "control_mean_gap": control,
            "baseline_mean_gap": control_baseline,
            "mean_improvement": control_improvement,
            "heldout_wins": control_wins,
            "maximum_signed_world_harm": control_harm,
            "substituted_mean_gate": mean_gate,
            "substituted_win_gate": win_gate,
            "substituted_harm_gate": harm_gate,
            "positive_gate_recovered": recovered,
        }
        for name, expected in expected_fields.items():
            require(case_id, name, expected)
        control_passes[case_id] = (not recovered) and control_improvement < 0.05

    aligned, aligned_baseline, aligned_diff, *_ = _constraint_comparison(
        world_aggregates,
        "aligned",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    aligned_abs = abs(aligned - aligned_baseline)
    aligned_harm = max(aligned_diff)
    aligned_expected = {
        "treatment_mean_gap": aligned,
        "baseline_mean_gap": aligned_baseline,
        "absolute_mean_difference": aligned_abs,
        "maximum_signed_world_harm": aligned_harm,
        "trajectories": 640,
        "mean_gate": aligned_abs <= 0.03,
        "harm_gate": aligned_harm <= 0.10,
    }
    for name, expected in aligned_expected.items():
        require("aligned_control", name, expected)
    impossible_rows = [
        row for row in world_aggregates if isinstance(row, dict) and row["family"] == "impossible"
    ]
    if len(impossible_rows) != 80 or any(
        row["mean_gap"] != 1.0
        or any(seed_gap != 1.0 for seed_gap in row["seed_gaps"])
        for row in impossible_rows
    ):
        raise RuntimeError("constraint-progress impossible aggregate changed")

    recomputed_passes = {
        "family_replay": metrics("family_replay")["roots_equal"] is True,
        "transcript_commitment": metrics("transcript_commitment")["roots_equal"] is True,
        "typed_aux_and_intervention": typed_roots_equal,
        "chronology_replay": metrics("chronology_replay")["roots_equal"] is True,
        "development_and_source_isolation": metrics("development_and_source_isolation")["roots_equal"] is True,
        "heldout_primary": primary_expected["mean_gate"] and primary_expected["win_gate"] and primary_expected["harm_gate"],
        "restart_comparators": comparator_expected["comparator_gate"],
        **control_passes,
        "aligned_control": aligned_expected["mean_gate"] and aligned_expected["harm_gate"],
        "impossible_control": (
            metrics("impossible_control")["trajectories"] == 640
            and metrics("impossible_control")["observations"] == 327680
            and metrics("impossible_control")["feasible_observations"] == 0
            and metrics("impossible_control")["nonunit_gaps"] == 0
            and metrics("impossible_control")["references_used"] == 0
            and metrics("impossible_control")["false_feasible_joins"] == 0
        ),
        "process_and_sanitizer": (
            metrics("process_and_sanitizer")["roots_equal"] is True
            and metrics("process_and_sanitizer")["maximum_stdout_bytes"] <= CONSTRAINT_PROGRESS_OUTPUT_BYTES
        ),
    }
    for case_id in case_ids:
        if by_id[case_id]["passed"] is not recomputed_passes[case_id]:
            raise RuntimeError(f"constraint-progress case pass disagrees: {case_id}")
    passed = all(recomputed_passes.values())
    expected_status = "passed" if passed else "failed"
    expected_action = entry.get("success_action") if passed else entry.get("failure_action")
    if result.get("status") != expected_status or result.get("action") != expected_action:
        raise RuntimeError("constraint-progress terminal decision disagrees")


def _validate_study_result(
    study: str,
    entry: dict[str, object],
    result: dict[str, object],
    *,
    study_revision: str | None = None,
    contract_sha256: str | None = None,
    worker_receipt: Mapping[str, object] | None = None,
) -> None:
    if study == "constraint-aware-progress-toy-v1":
        _validate_constraint_progress_result(
            entry,
            result,
            study_revision=study_revision,
            contract_sha256=contract_sha256,
            worker_receipt=worker_receipt,
        )
        return
    expected_top_level = {
        "action",
        "cases",
        "environment",
        "fixture",
        "schema_version",
        "status",
        "study_id",
    }
    if set(result) != expected_top_level:
        raise RuntimeError("worker returned an unexpected top-level result field")
    _validate_sanitized_value(
        result, strict_v3=study == "multistep-td-action-prefix-v3"
    )
    if result.get("study_id") != study:
        raise RuntimeError("worker returned the wrong study identity")
    if result.get("schema_version") != entry.get("result_schema_version"):
        raise RuntimeError("worker returned the wrong result schema")

    required_case_fields = entry.get("case_required_fields")
    expected_case_contract = entry.get("case_contract")
    cases = result.get("cases")
    if (
        not isinstance(required_case_fields, dict)
        or not isinstance(expected_case_contract, dict)
        or not isinstance(cases, dict)
    ):
        raise RuntimeError("worker returned a malformed case collection")
    if set(cases) != set(required_case_fields):
        raise RuntimeError("worker returned the wrong frozen case set")

    case_passes = []
    for case_name, required_fields in required_case_fields.items():
        case = cases.get(case_name)
        if not isinstance(case, dict) or not isinstance(required_fields, list):
            raise RuntimeError(f"worker returned malformed case: {case_name}")
        if set(case) != set(required_fields):
            raise RuntimeError(
                f"worker returned the wrong frozen fields for case: {case_name}"
            )
        frozen_case = expected_case_contract.get(case_name)
        if not isinstance(frozen_case, dict):
            raise RuntimeError(f"registry has a malformed case contract: {case_name}")
        if not isinstance(case.get("passed"), bool):
            raise RuntimeError(f"worker returned non-terminal case: {case_name}")
        strict_boolean_suffixes = (
            "_disjoint",
            "_empty",
            "_exact",
            "_preserved",
            "_unchanged",
            "_unreachable",
        )
        for field_name, field_value in case.items():
            if study == "multistep-td-action-prefix-v3":
                container_key = (case_name, field_name)
                if container_key in V3_EXTRA_CONTAINER_EXPECTATIONS:
                    expected_container = V3_EXTRA_CONTAINER_EXPECTATIONS[
                        container_key
                    ]
                elif container_key in V3_CASE_CONTRACT_CONTAINER_FIELDS:
                    expected_container = frozen_case.get(field_name)
                    if not isinstance(expected_container, (list, dict)):
                        raise RuntimeError(
                            f"registry has a malformed V3 container contract: "
                            f"{case_name}.{field_name}"
                        )
                else:
                    expected_container = None
                if expected_container is not None:
                    if (
                        type(expected_container) is not type(field_value)
                        or field_value != expected_container
                    ):
                        raise RuntimeError(
                            f"worker returned a non-frozen V3 container: "
                            f"{case_name}.{field_name}"
                        )
                elif isinstance(field_value, (list, dict)):
                    raise RuntimeError(
                        f"worker returned a non-frozen V3 container: "
                        f"{case_name}.{field_name}"
                    )
                elif field_name == "passed":
                    if type(field_value) is not bool:
                        raise RuntimeError(
                            f"worker returned a malformed V3 field: "
                            f"{case_name}.{field_name}"
                        )
                elif (
                    field_name in V3_BOOLEAN_CASE_FIELDS
                    or field_name.endswith(strict_boolean_suffixes)
                ):
                    if type(field_value) is not bool:
                        raise RuntimeError(
                            f"worker returned a malformed V3 field: "
                            f"{case_name}.{field_name}"
                        )
                elif (
                    field_name.endswith("sha256")
                    or field_name in V3_STRING_CASE_FIELDS
                ):
                    if type(field_value) is not str or not field_value:
                        raise RuntimeError(
                            f"worker returned a malformed V3 field: "
                            f"{case_name}.{field_name}"
                        )
                elif "return" in field_name or field_name.endswith("_gap"):
                    if type(field_value) is not float:
                        raise RuntimeError(
                            f"worker returned a malformed V3 field: "
                            f"{case_name}.{field_name}"
                        )
                elif type(field_value) is not int:
                    raise RuntimeError(
                        f"worker returned a malformed V3 field: "
                        f"{case_name}.{field_name}"
                    )
            if field_name.endswith(strict_boolean_suffixes) and not isinstance(
                field_value, bool
            ):
                raise RuntimeError(
                    f"worker returned a non-Boolean invariant: {case_name}.{field_name}"
                )
            if (
                study == "multistep-td-action-prefix-v3"
                and field_name in V3_BOOLEAN_CASE_FIELDS
                and type(field_value) is not bool
            ):
                raise RuntimeError(
                    f"worker returned a non-Boolean V3 invariant: "
                    f"{case_name}.{field_name}"
                )
        case_passes.append(case["passed"])

    passed = all(case_passes)
    expected_status = "passed" if passed else "failed"
    expected_action = (
        entry.get("success_action") if passed else entry.get("failure_action")
    )
    if result.get("status") != expected_status:
        raise RuntimeError("worker status disagrees with the frozen case results")
    if result.get("action") != expected_action:
        raise RuntimeError("worker action disagrees with the frozen decision rule")

    fixture = result.get("fixture")
    expected_fixture = entry.get("fixture_identity")
    if not isinstance(fixture, dict) or not isinstance(expected_fixture, dict):
        raise RuntimeError("worker returned malformed fixture identity")
    if set(fixture) != set(expected_fixture) | {"case_contract"}:
        raise RuntimeError("worker returned unexpected fixture fields")
    if any(fixture.get(name) != value for name, value in expected_fixture.items()):
        raise RuntimeError("worker returned the wrong frozen fixture identity")
    case_contract = fixture.get("case_contract")
    if (
        not isinstance(case_contract, dict)
        or not isinstance(expected_case_contract, dict)
        or case_contract != expected_case_contract
        or set(case_contract) != set(cases)
    ):
        raise RuntimeError("worker returned the wrong frozen case contract")

    environment = result.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"device_kind", "jax_version", "platform", "python"}
        or environment.get("platform") != "cpu"
        or any(
            type(environment.get(name)) is not str or not environment.get(name)
            for name in ("device_kind", "jax_version", "platform", "python")
        )
    ):
        raise RuntimeError("worker did not authenticate the CPU backend")


def _validate_sanitized_value(
    value: object, *, depth: int = 0, strict_v3: bool = True
) -> None:
    if depth > 8:
        raise RuntimeError("worker result exceeded the sanitized nesting limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("worker result contained a non-finite number")
        return
    if isinstance(value, str):
        if len(value) > 512:
            raise RuntimeError("worker result contained an oversized string")
        if str(ROOT).lower() in value.lower():
            raise RuntimeError("worker result exposed a repository path")
        if strict_v3 and (
            re.match(r"^[a-zA-Z]:", value) is not None
            or "/" in value
            or "\\" in value
        ):
            raise RuntimeError("worker result exposed an absolute path")
        return
    if isinstance(value, list):
        if len(value) > 64:
            raise RuntimeError("worker result contained an oversized list")
        for item in value:
            _validate_sanitized_value(
                item, depth=depth + 1, strict_v3=strict_v3
            )
        return
    if isinstance(value, dict):
        if len(value) > 64:
            raise RuntimeError("worker result contained an oversized object")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise RuntimeError("worker result contained a malformed key")
            if strict_v3 and (
                re.match(r"^[a-zA-Z]:", key) is not None
                or "/" in key
                or "\\" in key
            ):
                raise RuntimeError("worker result contained a malformed key")
            lowered = key.lower()
            raw_exact = {
                "action",
                "actions",
                "context",
                "contexts",
                "donor",
                "donor_array",
                "gradient",
                "gradients",
                "log",
                "logs",
                "observation",
                "observations",
                "origin",
                "origin_array",
                "path",
                "paths",
                "policy_state",
                "q_table",
                "q_value",
                "q_values",
                "return",
                "returns",
                "reward",
                "rewards",
                "state",
                "states",
                "successor",
                "successors",
                "target",
                "targets",
                "trajectory",
                "trajectories",
                "transition",
                "transitions",
                "value_table",
                "weight",
                "weights",
            }
            raw_container_exact = {
                "action",
                "array",
                "arrays",
                "context",
                "data",
                "donor",
                "gradient",
                "log",
                "observation",
                "origin",
                "path",
                "q_value",
                "record",
                "records",
                "return",
                "reward",
                "row",
                "rows",
                "state",
                "successor",
                "successors",
                "target",
                "trajectory",
                "transition",
                "weight",
            }
            base_forbidden_fragments = (
                "credential",
                "history",
                "parameter_values",
                "raw_gradient",
                "secret",
                "topology",
            )
            v3_forbidden_fragments = (
                "policy_parameters",
                "private_evidence",
                "raw_action",
                "raw_observation",
                "raw_reward",
                "raw_state",
                "raw_target",
                "raw_transition",
                "raw_trajectory",
            )
            tokens = tuple(
                token for token in re.split(r"[^a-z0-9]+", lowered) if token
            )
            raw_container_tokens = {
                "action",
                "context",
                "donor",
                "gradient",
                "log",
                "observation",
                "origin",
                "path",
                "reward",
                "state",
                "successor",
                "target",
                "trajectory",
                "transition",
                "weight",
            }
            raw_shape_tokens = {
                "array",
                "copy",
                "data",
                "list",
                "payload",
                "records",
                "rows",
                "stream",
                "table",
                "vector",
            }
            if any(
                fragment in lowered for fragment in base_forbidden_fragments
            ):
                raise RuntimeError("worker result contained a forbidden field")
            if strict_v3 and (
                (lowered in raw_exact and not (
                    lowered == "action" and depth == 0 and isinstance(item, str)
                ))
                or any(
                    fragment in lowered for fragment in v3_forbidden_fragments
                )
                or (
                    isinstance(item, (list, dict))
                    and lowered in raw_container_exact
                )
                or (
                    isinstance(item, (list, dict))
                    and any(token in raw_container_tokens for token in tokens)
                    and any(token in raw_shape_tokens for token in tokens)
                )
            ):
                raise RuntimeError("worker result contained a forbidden field")
            if lowered.endswith("sha256") and (
                not isinstance(item, str)
                or SHA256_PATTERN.fullmatch(item) is None
            ):
                raise RuntimeError("worker result contained a malformed SHA-256")
            _validate_sanitized_value(
                item, depth=depth + 1, strict_v3=strict_v3
            )
        return
    raise RuntimeError("worker result contained a non-JSON value")


def _validate_output(output: Path) -> tuple[Path, Path]:
    resolved = output.resolve()
    if resolved == PRIVATE_ROOT or not resolved.is_relative_to(PRIVATE_ROOT):
        raise RuntimeError(
            f"local-lab output must be a file beneath the private root: {PRIVATE_ROOT}"
        )
    sidecar = resolved.with_name(f"{resolved.name}.sha256")
    for path in (resolved, sidecar):
        if path.exists():
            raise RuntimeError(f"local-lab evidence already exists: {path.name}")
    return resolved, sidecar


def _write_atomic_bytes(path: Path, encoded: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite local-lab output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-volume hard link is an atomic create-if-absent operation. Unlike
        # os.replace(), it cannot overwrite an output created by another process.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(encoded)


def _write_atomic(path: Path, payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _write_atomic_bytes(path, encoded)


def _write_mutable_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(private_root: Path, event: dict[str, object]) -> None:
    private_root.mkdir(parents=True, exist_ok=True)
    payload = {"event_schema_version": 1, **event}
    encoded = (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with (private_root / "lab-events.jsonl").open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lease(
    private_root: Path,
    *,
    cycle_id: str,
    revision: str,
    study: str,
) -> tuple[Path, str]:
    private_root.mkdir(parents=True, exist_ok=True)
    lock_directory = private_root / "lab.lock"
    try:
        lock_directory.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "a local-lab lease already exists; inspect it manually and do not "
            "auto-recover it"
        ) from error

    lease_id = uuid.uuid4().hex
    now = _utc_now()
    lease = {
        "cycle_id": cycle_id,
        "heartbeat_utc": now,
        "hostname": socket.gethostname(),
        "lease_id": lease_id,
        "phase": "preflight",
        "pid": os.getpid(),
        "process_started_utc": now,
        "project_revision": revision,
        "schema_version": 1,
        "study": study,
    }
    try:
        _write_mutable_json(lock_directory / "lease.json", lease)
    except BaseException:
        if not any(lock_directory.iterdir()):
            lock_directory.rmdir()
        raise
    return lock_directory, lease_id


def _read_lease(lock_directory: Path, lease_id: str) -> dict[str, object]:
    lease_path = lock_directory / "lease.json"
    if not lease_path.is_file():
        raise RuntimeError("local-lab lease metadata disappeared")
    lease = _loads_json(lease_path.read_text(encoding="utf-8"))
    if lease.get("lease_id") != lease_id:
        raise RuntimeError("local-lab lease identity changed during execution")
    return lease


def _heartbeat_lease(lock_directory: Path, lease_id: str, *, phase: str) -> None:
    lease = _read_lease(lock_directory, lease_id)
    lease["heartbeat_utc"] = _utc_now()
    lease["phase"] = phase
    _write_mutable_json(lock_directory / "lease.json", lease)


def _release_lease(lock_directory: Path, lease_id: str) -> None:
    _read_lease(lock_directory, lease_id)
    (lock_directory / "lease.json").unlink()
    lock_directory.rmdir()


def _default_state() -> dict[str, object]:
    return {
        "active_cycle": None,
        "completed_studies": {},
        "failure_streak": 0,
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "idle",
        "stop_reason": None,
        "updated_utc": _utc_now(),
    }


def _load_state(private_root: Path) -> dict[str, object]:
    state_path = private_root / "lab-state.json"
    if not state_path.exists():
        return _default_state()
    state = _loads_json(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported or malformed local-lab state")
    if not isinstance(state.get("completed_studies"), dict):
        raise RuntimeError("malformed local-lab completed-study ledger")
    return state


def _save_state(private_root: Path, state: dict[str, object]) -> None:
    state["updated_utc"] = _utc_now()
    _write_mutable_json(private_root / "lab-state.json", state)


def _worker_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    safe_names = {
        "COMSPEC",
        "LD_LIBRARY_PATH",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "VIRTUAL_ENV",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name.upper() in safe_names
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    if overrides:
        allowed = {
            "L2D_CONTRACT_SHA256",
            "L2D_PLAN_REVISION",
            "L2D_STUDY_REVISION",
        }
        if set(overrides) - allowed or any(
            type(name) is not str or type(value) is not str
            for name, value in overrides.items()
        ):
            raise RuntimeError("unsafe local-lab worker environment override")
        environment.update(overrides)
    return environment


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _run_worker(
    worker_mode: str,
    *,
    cycle_id: str,
    heartbeat,
    worker_module: str = "experiments.local_lab.worker",
    environment_overrides: Mapping[str, str] | None = None,
    max_output_bytes: int = MAX_WORKER_OUTPUT_BYTES,
    require_empty_stderr: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    if worker_module not in WORKER_MODULE_PATHS:
        raise RuntimeError("local-lab worker module is not allowlisted")
    if worker_module == "experiments.local_lab.constraint_aware_progress_toy_worker":
        command = [
            sys.executable,
            "-S",
            "-P",
            str(ROOT / WORKER_MODULE_PATHS[worker_module]),
            "--mode",
            worker_mode,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            worker_module,
            "--mode",
            worker_mode,
        ]
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    temporary_root = PRIVATE_ROOT / "worker-tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    stdout_path = temporary_root / f"{cycle_id}.stdout"
    stderr_path = temporary_root / f"{cycle_id}.stderr"
    process = None
    try:
        with stdout_path.open("xb") as stdout_handle, stderr_path.open(
            "xb"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_worker_environment(environment_overrides),
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            started = time.monotonic()
            next_heartbeat = started + HEARTBEAT_SECONDS
            heartbeat(process.pid, 0.0)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                remaining = CYCLE_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    _terminate_process_tree(process)
                    raise TimeoutError(
                        f"local-lab worker exceeded {CYCLE_TIMEOUT_SECONDS} seconds"
                    )
                if stdout_path.stat().st_size + stderr_path.stat().st_size > (
                    max_output_bytes
                ):
                    _terminate_process_tree(process)
                    raise RuntimeError(
                        "local-lab worker output exceeded the retention cap"
                    )
                if time.monotonic() >= next_heartbeat:
                    heartbeat(process.pid, elapsed)
                    next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
                try:
                    process.wait(timeout=min(float(OUTPUT_POLL_SECONDS), remaining))
                except subprocess.TimeoutExpired:
                    pass

        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        if len(stdout_bytes) + len(stderr_bytes) > max_output_bytes:
            raise RuntimeError("local-lab worker output exceeded the retention cap")
        if require_empty_stderr and stderr_bytes:
            raise RuntimeError("local-lab worker emitted forbidden stderr")
        if process.returncode != 0:
            tail = stderr_bytes[-2000:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"local-lab worker exited {process.returncode}: "
                f"{tail or 'no stderr'}"
            )
        result = _loads_json(stdout_bytes)
        if not isinstance(result, dict):
            raise RuntimeError("local-lab worker returned a non-object result")
        return result, {
            "stderr_bytes": len(stderr_bytes),
            "stderr_sha256": _sha256_bytes(stderr_bytes),
            "stdout_bytes": len(stdout_bytes),
        }
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        for temporary in (stdout_path, stderr_path):
            if temporary.exists():
                temporary.unlink()


def _begin_cycle(
    private_root: Path,
    *,
    cycle_id: str,
    output: Path,
    snapshot: dict[str, object],
    study: str,
) -> dict[str, object]:
    if (private_root / "stop.request.json").exists():
        raise RuntimeError(
            "owner stop request is present; the local lab remains stopped"
        )
    state = _load_state(private_root)
    if state.get("status") not in {"idle", "awaiting_study"}:
        raise RuntimeError(f"local-lab state is not idle: {state.get('status')}")
    completed = state["completed_studies"]
    assert isinstance(completed, dict)
    if study in completed:
        raise DuplicateStudyError(
            f"the frozen study already has a terminal record: {study}"
        )
    state["active_cycle"] = {
        "cycle_id": cycle_id,
        "output": output.relative_to(private_root).as_posix(),
        "revision": snapshot["revision"],
        "started_utc": _utc_now(),
        "study": study,
    }
    state["status"] = "active"
    state["stop_reason"] = None
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "event": "cycle_started",
            "revision": snapshot["revision"],
            "study": study,
            "utc": _utc_now(),
        },
    )
    return state


def _park_cycle(
    private_root: Path,
    state: dict[str, object],
    *,
    cycle_id: str,
    error: BaseException,
    study: str,
) -> None:
    state["active_cycle"] = None
    state["failure_streak"] = int(state.get("failure_streak", 0)) + 1
    state["status"] = "parked"
    state["stop_reason"] = f"{type(error).__name__}: {str(error)[:500]}"
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "error_type": type(error).__name__,
            "event": "cycle_parked",
            "study": study,
            "utc": _utc_now(),
        },
    )


def _park_preflight(
    private_root: Path,
    *,
    cycle_id: str,
    error: BaseException,
    study: str,
) -> None:
    state = _load_state(private_root)
    if state.get("status") not in {"idle", "awaiting_study"}:
        return
    owner_stopped = (private_root / "stop.request.json").exists()
    state["active_cycle"] = None
    state["failure_streak"] = (
        int(state.get("failure_streak", 0))
        if owner_stopped
        else int(state.get("failure_streak", 0)) + 1
    )
    state["status"] = "stopped" if owner_stopped else "parked"
    state["stop_reason"] = f"{type(error).__name__}: {str(error)[:500]}"
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "error_type": type(error).__name__,
            "event": "preflight_stopped" if owner_stopped else "preflight_parked",
            "study": study,
            "utc": _utc_now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output, sidecar = _validate_output(args.output)
    cycle_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    prelease_revision = _git("rev-parse", "HEAD")
    lock_directory, lease_id = _acquire_lease(
        PRIVATE_ROOT,
        cycle_id=cycle_id,
        revision=prelease_revision,
        study=args.study,
    )
    state = None
    exit_code = 0
    try:
        registry = _load_study_registry()
        entry = _study_entry(registry, args.study)
        snapshot = _repository_snapshot(entry)
        if snapshot["revision"] != prelease_revision:
            raise RuntimeError("repository revision changed while acquiring the lease")
        _validate_study_approval(entry, snapshot)
        constraint_contract_sha256 = None
        worker_environment_overrides = None
        worker_output_cap_bytes = MAX_WORKER_OUTPUT_BYTES
        require_empty_worker_stderr = False
        if args.study == "constraint-aware-progress-toy-v1":
            _validate_constraint_progress_runtime(entry)
            constraint_contract_sha256 = _constraint_progress_contract_sha256(
                registry, entry
            )
            plan_revision = entry.get("plan_revision")
            study_revision = snapshot.get("revision")
            if not isinstance(plan_revision, str) or not isinstance(
                study_revision, str
            ):
                raise RuntimeError("constraint-progress revision contract is malformed")
            worker_environment_overrides = {
                "L2D_CONTRACT_SHA256": constraint_contract_sha256,
                "L2D_PLAN_REVISION": plan_revision,
                "L2D_STUDY_REVISION": study_revision,
            }
            worker_output_cap_bytes = CONSTRAINT_PROGRESS_OUTPUT_BYTES
            require_empty_worker_stderr = True
        _heartbeat_lease(lock_directory, lease_id, phase="preflight-complete")
        state = _begin_cycle(
            PRIVATE_ROOT,
            cycle_id=cycle_id,
            output=output,
            snapshot=snapshot,
            study=args.study,
        )

        def heartbeat(worker_pid: int, elapsed_seconds: float) -> None:
            _heartbeat_lease(lock_directory, lease_id, phase="worker-running")
            _append_event(
                PRIVATE_ROOT,
                {
                    "cycle_id": cycle_id,
                    "elapsed_seconds": round(float(elapsed_seconds), 3),
                    "event": "heartbeat",
                    "study": args.study,
                    "utc": _utc_now(),
                    "worker_pid": worker_pid,
                },
            )

        worker_mode = entry.get("worker_mode")
        if not isinstance(worker_mode, str):
            raise RuntimeError("approved study has no worker mode")
        worker_module = entry.get("worker_module")
        if not isinstance(worker_module, str):
            raise RuntimeError("approved study has no worker module")
        result, worker_receipt = _run_worker(
            worker_mode,
            cycle_id=cycle_id,
            heartbeat=heartbeat,
            worker_module=worker_module,
            environment_overrides=worker_environment_overrides,
            max_output_bytes=worker_output_cap_bytes,
            require_empty_stderr=require_empty_worker_stderr,
        )
        _validate_study_result(
            args.study,
            entry,
            result,
            study_revision=snapshot["revision"],
            contract_sha256=constraint_contract_sha256,
            worker_receipt=worker_receipt,
        )
        post_snapshot = _repository_snapshot(entry)
        if post_snapshot != snapshot:
            raise RuntimeError(
                "repository or protected-artifact drift during the cycle"
            )

        payload = {
            "artifact_format_version": 2,
            "cycle_id": cycle_id,
            "execution_contract": {
                "cycle_timeout_seconds": CYCLE_TIMEOUT_SECONDS,
                "device": "cpu",
                "network": "disabled-in-worker",
                "worker_output_cap_bytes": worker_output_cap_bytes,
            },
            "provenance": snapshot,
            "result": result,
            "worker_receipt": worker_receipt,
        }
        digest = _write_atomic(output, payload)
        _write_atomic_bytes(
            sidecar,
            f"{digest}  {output.name}\n".encode("ascii"),
        )

        completed = state["completed_studies"]
        assert isinstance(completed, dict)
        completed[args.study] = {
            "cycle_id": cycle_id,
            "result_sha256": digest,
            "revision": snapshot["revision"],
            "status": result["status"],
        }
        state["active_cycle"] = None
        state["failure_streak"] = 0 if result["status"] == "passed" else 1
        if result["status"] == "passed":
            studies = registry["studies"]
            assert isinstance(studies, dict)
            pending = set(studies) - set(completed)
            state["status"] = "idle" if pending else "awaiting_study"
            state["stop_reason"] = (
                None if pending else "no_approved_study_pending"
            )
        else:
            state["status"] = "parked"
            state["stop_reason"] = result.get("action")
        _save_state(PRIVATE_ROOT, state)
        _append_event(
            PRIVATE_ROOT,
            {
                "cycle_id": cycle_id,
                "event": "cycle_completed",
                "result_sha256": digest,
                "status": result["status"],
                "study": args.study,
                "utc": _utc_now(),
            },
        )
        _heartbeat_lease(lock_directory, lease_id, phase="terminal-recorded")
        print(f"wrote {output} ({digest})")
        if result["status"] != "passed":
            exit_code = 1
    except BaseException as error:
        if state is not None and state.get("status") == "active":
            _park_cycle(
                PRIVATE_ROOT,
                state,
                cycle_id=cycle_id,
                error=error,
                study=args.study,
            )
        elif state is None and not isinstance(error, DuplicateStudyError):
            _park_preflight(
                PRIVATE_ROOT,
                cycle_id=cycle_id,
                error=error,
                study=args.study,
            )
        raise
    finally:
        _release_lease(lock_directory, lease_id)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
