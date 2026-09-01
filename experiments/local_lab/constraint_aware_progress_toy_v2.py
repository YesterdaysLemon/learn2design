"""Frozen synthetic constrained-progress fixture for Round-2 mechanics research.

This module contains the primary deterministic environment, optimizer, independent
replay oracle, phase execution, and aggregate projection. Process supervision lives
only in the dedicated V2 bootstrap. Importing this module never executes a study.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np


fx = sys.modules[__name__]


STUDY_ID = "constraint-aware-progress-toy-v2"
PLAN_REVISION = "c5314afaa50490e39c53669d971114d280e43c07"
PLAN_PATH = "research/2026-08-31-constraint-aware-progress-toy-v2-plan.md"
SCHEMA_DOMAIN = b"L2D-constraint-progress-v1/"
TRANSCRIPT_DOMAIN = b"L2D-constraint-progress-transcript-v1"

FAMILIES = ("canonical", "aligned", "impossible")
SPLITS = ("development", "heldout")
ORDERS = ("forward", "reverse")
ARMS = (
    "protected_raw_progress",
    "constraint_lexicographic_progress",
    "shuffled_progress_control",
    "ablated_progress_control",
    "no_restart_comparator",
)
SEEDS = (2026083001, 2026083003, 2026083007, 2026083011)

POPULATION = 8
BATCHES = 64
PATIENCE = 8
EVALUATIONS_PER_TRAJECTORY = POPULATION * BATCHES
TRANSITIONS_PER_TRAJECTORY = EVALUATIONS_PER_TRAJECTORY
VALUES_PER_TRANSCRIPT = 3093
TOTAL_TRANSCRIPT_VALUES = VALUES_PER_TRANSCRIPT * len(SEEDS)
IMPROVEMENT_TOLERANCE = 1.0e-7
MAX_PACKET_BYTES = 1_048_576

TRANSCRIPT_HASHES = {
    2026083001: "fc826363fe2234af0a73c0730639c483ad1e29c053915d7c5197b05bffef01e9",
    2026083003: "a6ce093cb6a53ded68786ca4cb044f7dd974c45a23a82fe1072e572eea492a15",
    2026083007: "6f90f43dd4673a388470502bba1959686fc9b35f2e3bf4de0feef2087ae01c41",
    2026083011: "74e07f6003668543bd197188633d11011e99c6e78803b9d8eed5a792cb0f83bd",
}
TRANSCRIPT_ROOT_SHA256 = (
    "9c250412d296b7e60a5ab0e02f4cf69925d165bb6c3f61e3a29e00b475d99edd"
)

RUNTIME_IDENTITY = {
    "machine": "AMD64",
    "numpy_init_sha256": (
        "a6958cb364663b7acce81ccfd58eeb65a2b34d5376157f924777b97211a73be4"
    ),
    "numpy_metadata_sha256": (
        "6ae45122ee97050e48849438320430d05f01814f72e66e69cbeed027d2c6a1e8"
    ),
    "numpy_version": "2.5.1",
    "pcg64_identity": "numpy.random._pcg64.PCG64",
    "pcg64_module_sha256": (
        "210bd962e911039f1639d0137f6e41444e37db23aba1622635d9dba8abc6a1c9"
    ),
    "python_architecture": "64bit",
    "python_executable_sha256": (
        "ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec"
    ),
    "python_implementation": "CPython",
    "python_version": "3.13.14",
    "seed_sequence_identity": "numpy.random.bit_generator.SeedSequence",
    "seed_sequence_module_sha256": (
        "08355a330efec79a840b5767bb5356ad21e3b0f14acce9a3c969208626daad7f"
    ),
}

CAPABILITY_ATTACKS = (
    "family",
    "split",
    "world",
    "bits",
    "seed",
    "a",
    "b",
    "k",
    "t",
    "c",
    "threshold",
    "reference-x0",
    "reference-sensitivity",
    "denominator",
    "gap",
    "environment-call",
    "evaluator-call",
    "oracle-call",
    "source-call",
    "canonical-aux-bypass",
    "future-transcript-read",
    "transcript-fingerprint",
)
CAPABILITY_PATHS = {
    attack: (
        "restart-draw-provider.future-read"
        if attack == "future-transcript-read"
        else "restart-draw-provider.fingerprint"
        if attack == "transcript-fingerprint"
        else f"optimizer-adapter.{attack}"
    )
    for attack in CAPABILITY_ATTACKS
}

MALFORMED_ATTACKS = (
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


class ContractError(RuntimeError):
    """Fail-closed violation of the frozen synthetic contract."""


def _positive_zero(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def normalize_json(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("nonfinite-canonical-value")
        return _positive_zero(value)
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json(item) for key, item in value.items()}
    return value


def canonical_line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            normalize_json(dict(value)),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_domain(domain: str, payload: bytes = b"") -> str:
    return hashlib.sha256(SCHEMA_DOMAIN + domain.encode("ascii") + b"\0" + payload).hexdigest()


class RecordRoot:
    def __init__(self, record_type: str):
        self._hash = hashlib.sha256()
        self._hash.update(SCHEMA_DOMAIN)
        self._hash.update(record_type.encode("ascii"))
        self._hash.update(b"\0")

    def update_mapping(self, value: Mapping[str, object]) -> None:
        self._hash.update(canonical_line(value))

    def update_bytes(self, value: bytes) -> None:
        self._hash.update(value)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


@dataclass(frozen=True, slots=True)
class ProgressState:
    mode: str
    observed: bool
    feasible: bool
    first: float
    second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "observed": self.observed,
            "feasible": self.feasible,
            "first": self.first,
            "second": self.second,
        }


@dataclass(frozen=True, slots=True)
class DecisionTuple:
    is_feasible: bool
    penalty: float
    sensitivity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "is_feasible": self.is_feasible,
            "penalty": self.penalty,
            "sensitivity": self.sensitivity,
        }


@dataclass(frozen=True, slots=True)
class WorldRecord:
    family: str
    world: int
    bits: tuple[int, int, int, int]
    split: str
    a: float
    b: float
    k: float
    t: float
    c: float
    threshold: float
    reference_x0: float | None
    reference_sensitivity: float | None
    denominator: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "world": self.world,
            "bits": list(self.bits),
            "split": self.split,
            "a": self.a,
            "b": self.b,
            "k": self.k,
            "t": self.t,
            "c": self.c,
            "threshold": self.threshold,
            "reference_x0": self.reference_x0,
            "reference_sensitivity": self.reference_sensitivity,
            "denominator": self.denominator,
        }


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    family: str
    world: int
    seed: int
    order: str
    arm: str
    batch: int
    member: int
    u: tuple[float, float, float]
    x: tuple[float, float, float]
    loss: float
    gradient: tuple[float, float, float]
    gradient_nonfinite: tuple[bool, bool, bool]
    canonical_is_feasible: bool
    sensitivity: float
    penalty: float
    violation: float
    decision_source: str
    decision_donor_member: int
    decision_is_feasible: bool
    decision_penalty: float
    decision_sensitivity: float

    def to_dict(self, *, omit_order: bool = False) -> dict[str, object]:
        result = {
            "family": self.family,
            "world": self.world,
            "seed": self.seed,
            "order": self.order,
            "arm": self.arm,
            "batch": self.batch,
            "member": self.member,
            "u": list(self.u),
            "x": list(self.x),
            "loss": self.loss,
            "gradient": list(self.gradient),
            "gradient_nonfinite": list(self.gradient_nonfinite),
            "canonical_is_feasible": self.canonical_is_feasible,
            "sensitivity": self.sensitivity,
            "penalty": self.penalty,
            "violation": self.violation,
            "decision_source": self.decision_source,
            "decision_donor_member": self.decision_donor_member,
            "decision_is_feasible": self.decision_is_feasible,
            "decision_penalty": self.decision_penalty,
            "decision_sensitivity": self.decision_sensitivity,
        }
        if omit_order:
            del result["order"]
        return result


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    family: str
    world: int
    seed: int
    order: str
    arm: str
    batch: int
    member: int
    progress_before: ProgressState
    progress_after: ProgressState
    stall_before: int
    stall_after: int
    adam_age_before: int
    adam_age_after: int
    update_applied: bool
    restart_triggered: bool
    restart_kind: str
    restart_round: int
    center_source: str
    state_before_sha256: str
    state_after_sha256: str

    def to_dict(self, *, omit_order: bool = False) -> dict[str, object]:
        result = {
            "family": self.family,
            "world": self.world,
            "seed": self.seed,
            "order": self.order,
            "arm": self.arm,
            "batch": self.batch,
            "member": self.member,
            "progress_before": self.progress_before.to_dict(),
            "progress_after": self.progress_after.to_dict(),
            "stall_before": self.stall_before,
            "stall_after": self.stall_after,
            "adam_age_before": self.adam_age_before,
            "adam_age_after": self.adam_age_after,
            "update_applied": self.update_applied,
            "restart_triggered": self.restart_triggered,
            "restart_kind": self.restart_kind,
            "restart_round": self.restart_round,
            "center_source": self.center_source,
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
        }
        if omit_order:
            del result["order"]
        return result


@dataclass(frozen=True, slots=True)
class BatchReceipt:
    family: str
    world: int
    seed: int
    order: str
    arm: str
    batch: int
    incumbent_present: bool
    incumbent_sensitivity: float | None
    incumbent_source_batch: int
    incumbent_source_member: int
    incumbent_center_sha256: str
    restart_round_before: int
    restart_round_after: int
    restart_mask: tuple[bool, ...]
    fresh_draw_sha256: str
    perturb_draw_sha256: str

    def to_dict(self, *, omit_order: bool = False) -> dict[str, object]:
        result = {
            "family": self.family,
            "world": self.world,
            "seed": self.seed,
            "order": self.order,
            "arm": self.arm,
            "batch": self.batch,
            "incumbent_present": self.incumbent_present,
            "incumbent_sensitivity": self.incumbent_sensitivity,
            "incumbent_source_batch": self.incumbent_source_batch,
            "incumbent_source_member": self.incumbent_source_member,
            "incumbent_center_sha256": self.incumbent_center_sha256,
            "restart_round_before": self.restart_round_before,
            "restart_round_after": self.restart_round_after,
            "restart_mask": list(self.restart_mask),
            "fresh_draw_sha256": self.fresh_draw_sha256,
            "perturb_draw_sha256": self.perturb_draw_sha256,
        }
        if omit_order:
            del result["order"]
        return result


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    family: str
    world: int
    seed: int
    order: str
    arm: str
    evaluations: int
    transitions: int
    best_feasible_sensitivity: float | None
    gap: float
    normalized_twin_sha256: str
    event_root_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "world": self.world,
            "seed": self.seed,
            "order": self.order,
            "arm": self.arm,
            "evaluations": self.evaluations,
            "transitions": self.transitions,
            "best_feasible_sensitivity": self.best_feasible_sensitivity,
            "gap": self.gap,
            "normalized_twin_sha256": self.normalized_twin_sha256,
            "event_root_sha256": self.event_root_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorldAggregateRecord:
    family: str
    world: int
    arm: str
    seed_gaps: tuple[float, float, float, float]
    mean_gap: float

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "world": self.world,
            "arm": self.arm,
            "seed_gaps": list(self.seed_gaps),
            "mean_gap": self.mean_gap,
        }


@dataclass(frozen=True, slots=True)
class PhaseReceipt:
    family: str
    split: str
    world_keys: tuple[int, ...]
    attempted_reads: int
    forbidden_reads: int
    forbidden_payload_rows: int
    sentinel_connected: bool
    input_root_sha256: str
    output_root_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "split": self.split,
            "world_keys": list(self.world_keys),
            "attempted_reads": self.attempted_reads,
            "forbidden_reads": self.forbidden_reads,
            "forbidden_payload_rows": self.forbidden_payload_rows,
            "sentinel_connected": self.sentinel_connected,
            "input_root_sha256": self.input_root_sha256,
            "output_root_sha256": self.output_root_sha256,
        }


@dataclass(frozen=True, slots=True)
class AttackReceipt:
    attack_id: str
    injection_path: str
    rejection_code: str
    consumer_reached: bool
    state_mutations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "attack_id": self.attack_id,
            "injection_path": self.injection_path,
            "rejection_code": self.rejection_code,
            "consumer_reached": self.consumer_reached,
            "state_mutations": self.state_mutations,
        }


@dataclass(frozen=True, slots=True)
class SentinelRecord:
    sentinel_id: str
    ordinal: int
    member: int
    stored: ProgressState
    canonical_tuple: DecisionTuple
    donor_member: int
    donor_tuple: DecisionTuple
    ablated_tuple: DecisionTuple
    canonical_decision: bool
    donor_decision: bool
    ablated_decision: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sentinel_id": self.sentinel_id,
            "ordinal": self.ordinal,
            "member": self.member,
            "stored": self.stored.to_dict(),
            "canonical_tuple": self.canonical_tuple.to_dict(),
            "donor_member": self.donor_member,
            "donor_tuple": self.donor_tuple.to_dict(),
            "ablated_tuple": self.ablated_tuple.to_dict(),
            "canonical_decision": self.canonical_decision,
            "donor_decision": self.donor_decision,
            "ablated_decision": self.ablated_decision,
        }


@dataclass(frozen=True, slots=True)
class Transcript:
    seed: int
    suffix: np.ndarray
    fresh: np.ndarray
    perturb: np.ndarray
    fresh_hashes: tuple[str, ...]
    perturb_hashes: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    u: tuple[float, float, float]
    x: tuple[float, float, float]
    loss: float
    gradient: tuple[float, float, float]
    nonfinite: tuple[bool, bool, bool]
    feasible: bool
    sensitivity: float
    penalty: float
    violation: float


@dataclass(frozen=True, slots=True)
class OptimizerPacket:
    u: np.ndarray
    learning_rates: np.ndarray
    m: np.ndarray
    v: np.ndarray
    ages: np.ndarray
    stalls: np.ndarray
    progress: tuple[ProgressState, ...]
    losses: np.ndarray
    gradients: np.ndarray
    decisions: tuple[DecisionTuple, ...]
    incumbent_present: bool
    incumbent_center: np.ndarray
    restart_round: int
    budget_fraction: float


@dataclass(frozen=True, slots=True)
class InterventionConfig:
    arm_id: str
    progress_comparator: str
    tuple_adapter: str
    restart_enabled: bool


class OptimizerCapabilityBoundary:
    """A sealed capability surface used by live denial probes.

    Production packets never carry this object.  The hostile probe enters the
    same adapter method as a real optimizer call and records the attempted
    read before failing closed.
    """

    def __init__(self) -> None:
        self._attempted_paths: list[str] = []

    @property
    def attempted_paths(self) -> tuple[str, ...]:
        return tuple(self._attempted_paths)

    def read_forbidden(self, path: str) -> None:
        self._attempted_paths.append(path)
        raise ContractError("capability-denied")


@dataclass(frozen=True, slots=True)
class CollectorEnvelope:
    observations: tuple[ObservationRecord, ...]
    transitions: tuple[TransitionRecord, ...]
    receipts: tuple[BatchReceipt, ...]


@dataclass(frozen=True, slots=True)
class EvaluatorEnvelope:
    world: WorldRecord
    trajectory: TrajectoryRecord


@dataclass(frozen=True, slots=True)
class StepOutcome:
    u: np.ndarray
    m: np.ndarray
    v: np.ndarray
    ages: np.ndarray
    stalls: np.ndarray
    progress: tuple[ProgressState, ...]
    restart_mask: tuple[bool, ...]
    restart_kinds: tuple[str, ...]
    center_sources: tuple[str, ...]
    restart_round_after: int


@dataclass(frozen=True, slots=True)
class PreparedStep:
    u: np.ndarray
    m: np.ndarray
    v: np.ndarray
    ages: np.ndarray
    stalls: np.ndarray
    progress: tuple[ProgressState, ...]
    restart_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class DrawAuthorization:
    batch: int
    restart_round: int
    restart_mask: tuple[bool, ...]
    fresh_draw_sha256: str
    perturb_draw_sha256: str


@dataclass(frozen=True, slots=True)
class SealedDrawAuthorization:
    authorization: DrawAuthorization
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryEvidence:
    collector: CollectorEnvelope
    evaluator: EvaluatorEnvelope
    restart_events: int


class OracleConsumer(Protocol):
    def consume(
        self,
        world: WorldRecord,
        transcript: Transcript,
        evidence: TrajectoryEvidence,
    ) -> None: ...


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float64)
    value.flags.writeable = False
    return value


def _readonly_int(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.int64)
    value.flags.writeable = False
    return value


def _bits(world: int) -> tuple[int, int, int, int]:
    return (
        (world >> 3) & 1,
        (world >> 2) & 1,
        (world >> 1) & 1,
        world & 1,
    )


def _family_q(family: str, t: float) -> float:
    return -t if family == "aligned" else t


def _sensitivity(world: WorldRecord, x: Sequence[float]) -> float:
    q = _family_q(world.family, world.t)
    x0, x1, x2 = (float(value) for value in x)
    return float(
        (x0 * x0 - world.a * world.a) ** 2
        + q * x0
        + world.k * (x1 - world.b) ** 2
        + 0.5 * (x2 - world.c) ** 2
    )


def _reference_x0(a: float, q: float) -> float:
    lower = a / 2.0
    upper = a if q > 0.0 else min(1.5 * a, 1.99)
    lower_derivative = 4.0 * lower * (lower * lower - a * a) + q
    upper_derivative = 4.0 * upper * (upper * upper - a * a) + q
    if lower_derivative > 0.0 or upper_derivative < 0.0:
        raise ContractError("reference-bracket-sign")
    for _ in range(80):
        middle = (lower + upper) / 2.0
        derivative = 4.0 * middle * (middle * middle - a * a) + q
        if derivative <= 0.0:
            lower = middle
        else:
            upper = middle
    return float((lower + upper) / 2.0)


def build_world_records() -> tuple[WorldRecord, ...]:
    records: list[WorldRecord] = []
    for family in FAMILIES:
        for world in range(16):
            bits = _bits(world)
            a = 0.80 + 0.20 * bits[0]
            b = -0.50 + 1.00 * bits[1]
            k = 0.50 + 0.50 * bits[2]
            t = 0.10 + 0.06 * bits[3]
            c = -0.25 if (bits[0] ^ bits[1]) == 0 else 0.25
            threshold = 2.25 if family == "impossible" else 0.0
            split = "development" if sum(bits) % 2 == 0 else "heldout"
            if family == "impossible":
                reference_x0 = None
                reference_sensitivity = None
                denominator = None
            else:
                q = _family_q(family, t)
                reference_x0 = _reference_x0(a, q)
                reference_lower = a / 2.0
                reference_upper = a if q > 0.0 else min(1.5 * a, 1.99)
                if not reference_lower <= reference_x0 <= reference_upper:
                    raise ContractError("reference-bracket")
                reference_sensitivity = float(
                    (reference_x0 * reference_x0 - a * a) ** 2
                    + q * reference_x0
                )
                anchor_sensitivity = float(a**4 + k * b * b + 0.5 * c * c)
                denominator = float(anchor_sensitivity - reference_sensitivity)
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise ContractError("nonpositive-reference-denominator")
            records.append(
                WorldRecord(
                    family=family,
                    world=world,
                    bits=bits,
                    split=split,
                    a=float(a),
                    b=float(b),
                    k=float(k),
                    t=float(t),
                    c=float(c),
                    threshold=float(threshold),
                    reference_x0=reference_x0,
                    reference_sensitivity=reference_sensitivity,
                    denominator=denominator,
                )
            )
    if len(records) != 48 or len({(r.family, r.world) for r in records}) != 48:
        raise ContractError("family-record-cardinality")
    for family in FAMILIES:
        family_records = [item for item in records if item.family == family]
        if (
            sum(item.split == "development" for item in family_records) != 8
            or sum(item.split == "heldout" for item in family_records) != 8
        ):
            raise ContractError("family-split-balance")
    return tuple(records)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def evaluate(world: WorldRecord, u: Sequence[float]) -> Evaluation:
    u_array = np.asarray(u, dtype=np.float64)
    if u_array.shape != (3,):
        raise ContractError("optimizer-state-shape")
    sigmoid = _sigmoid(u_array)
    x = 4.0 * sigmoid - 2.0
    violation = max(0.0, world.threshold - float(x[0]))
    penalty = float(0.02 * violation * violation)
    sensitivity = _sensitivity(world, x)
    loss = float(sensitivity + penalty)
    q = _family_q(world.family, world.t)
    penalty_dx0 = -0.04 * violation if float(x[0]) < world.threshold else 0.0
    active_gradient = np.asarray(
        [
            4.0 * float(x[0]) * (float(x[0]) ** 2 - world.a**2)
            + q
            + penalty_dx0,
            2.0 * world.k * (float(x[1]) - world.b),
            float(x[2]) - world.c,
        ],
        dtype=np.float64,
    )
    raw_gradient = active_gradient * (4.0 * sigmoid * (1.0 - sigmoid))
    nonfinite = tuple(bool(value) for value in ~np.isfinite(raw_gradient))
    gradient = np.nan_to_num(raw_gradient, nan=0.0, posinf=0.0, neginf=0.0)
    feasible = bool(float(x[0]) >= world.threshold)
    return Evaluation(
        u=tuple(float(value) for value in u_array),
        x=tuple(float(value) for value in x),
        loss=loss,
        gradient=tuple(float(value) for value in gradient),
        nonfinite=nonfinite,
        feasible=feasible,
        sensitivity=sensitivity,
        penalty=penalty,
        violation=float(violation),
    )


def build_transcript(seed: int) -> Transcript:
    if seed not in SEEDS:
        raise ContractError("transcript-seed")
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20260830, seed]))
    )
    suffix = generator.standard_normal((7, 3), dtype=np.float64)
    fresh = np.empty((BATCHES, POPULATION, 3), dtype=np.float64)
    perturb = np.empty_like(fresh)
    flattened: list[np.ndarray] = [suffix.reshape(-1)]
    for batch in range(BATCHES):
        fresh[batch] = generator.standard_normal((POPULATION, 3), dtype=np.float64)
        perturb[batch] = generator.standard_normal((POPULATION, 3), dtype=np.float64)
        flattened.extend((fresh[batch].reshape(-1), perturb[batch].reshape(-1)))
    values = np.concatenate(flattened).astype("<f8", copy=False)
    if values.size != VALUES_PER_TRANSCRIPT:
        raise ContractError("transcript-draw-count")
    digest = hashlib.sha256(
        TRANSCRIPT_DOMAIN
        + b"\0"
        + struct.pack("<Q", seed)
        + values.tobytes(order="C")
    ).hexdigest()
    if digest != TRANSCRIPT_HASHES[seed]:
        raise ContractError("transcript-hash")
    return Transcript(
        seed=seed,
        suffix=_readonly(suffix),
        fresh=_readonly(fresh),
        perturb=_readonly(perturb),
        fresh_hashes=tuple(
            draw_hash("fresh", seed, batch, fresh[batch])
            for batch in range(BATCHES)
        ),
        perturb_hashes=tuple(
            draw_hash("perturb", seed, batch, perturb[batch])
            for batch in range(BATCHES)
        ),
        sha256=digest,
    )


def transcript_root(transcripts: Sequence[Transcript]) -> str:
    if tuple(item.seed for item in transcripts) != SEEDS:
        raise ContractError("transcript-order")
    preimage = TRANSCRIPT_DOMAIN + b"/root\0" + b"".join(
        item.sha256.encode("ascii") + b"\n" for item in transcripts
    )
    digest = hashlib.sha256(preimage).hexdigest()
    if digest != TRANSCRIPT_ROOT_SHA256:
        raise ContractError("transcript-root")
    return digest


def draw_hash(kind: str, seed: int, batch: int, values: np.ndarray) -> str:
    if kind not in {"fresh", "perturb"}:
        raise ContractError("draw-kind")
    array = np.asarray(values)
    if array.shape != (POPULATION, 3) or array.dtype != np.float64:
        raise ContractError("draw-shape")
    payload = np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(
        SCHEMA_DOMAIN
        + f"{kind}-draw".encode("ascii")
        + b"\0"
        + struct.pack("<Q", seed)
        + struct.pack("<i", batch)
        + payload
    ).hexdigest()


def incumbent_center_hash(present: bool, center: Sequence[float]) -> str:
    if not present:
        return hashlib.sha256(
            b"L2D-constraint-progress-v1/absent-incumbent\0"
        ).hexdigest()
    values = np.asarray([_positive_zero(float(item)) for item in center], dtype="<f8")
    if values.shape != (3,):
        raise ContractError("incumbent-center-shape")
    return hashlib.sha256(
        b"L2D-constraint-progress-v1/incumbent-center\0"
        + values.tobytes(order="C")
    ).hexdigest()


def progress_decision(
    state: ProgressState,
    *,
    loss: float,
    decision: DecisionTuple,
) -> tuple[bool, ProgressState]:
    if state.mode == "raw":
        improved = (not state.observed) or (
            math.isfinite(loss) and loss < state.first - IMPROVEMENT_TOLERANCE
        )
        return (
            improved,
            ProgressState("raw", True, False, float(loss), 0.0)
            if improved
            else state,
        )
    if state.mode != "lex":
        raise ContractError("progress-mode")
    candidate = (
        ProgressState("lex", True, True, decision.sensitivity, 0.0)
        if decision.is_feasible
        else ProgressState(
            "lex", True, False, decision.penalty, decision.sensitivity
        )
    )
    if not state.observed:
        return True, candidate
    if decision.is_feasible and not state.feasible:
        return True, candidate
    if not decision.is_feasible and state.feasible:
        return False, state
    if decision.is_feasible:
        improved = decision.sensitivity < state.first - IMPROVEMENT_TOLERANCE
        return improved, candidate if improved else state
    penalty_improved = decision.penalty < state.first - IMPROVEMENT_TOLERANCE
    tied_penalty = abs(decision.penalty - state.first) <= IMPROVEMENT_TOLERANCE
    sensitivity_improved = (
        decision.sensitivity < state.second - IMPROVEMENT_TOLERANCE
    )
    improved = penalty_improved or (tied_penalty and sensitivity_improved)
    return improved, candidate if improved else state


def intervention_config(arm: str) -> InterventionConfig:
    configs = {
        "protected_raw_progress": InterventionConfig(
            "protected_raw_progress", "raw", "unused", True
        ),
        "constraint_lexicographic_progress": InterventionConfig(
            "constraint_lexicographic_progress", "lex", "canonical", True
        ),
        "shuffled_progress_control": InterventionConfig(
            "shuffled_progress_control", "lex", "cyclic-donor", True
        ),
        "ablated_progress_control": InterventionConfig(
            "ablated_progress_control", "lex", "ablated", True
        ),
        "no_restart_comparator": InterventionConfig(
            "no_restart_comparator", "raw", "unused", False
        ),
    }
    try:
        return configs[arm]
    except KeyError as error:
        raise ContractError("arm-identity") from error


def decision_tuple(
    arm: str,
    member: int,
    evaluations: Sequence[Evaluation],
) -> tuple[str, int, DecisionTuple]:
    adapter = intervention_config(arm).tuple_adapter
    if adapter == "unused":
        return "unused", -1, DecisionTuple(False, 0.0, 0.0)
    if adapter == "canonical":
        item = evaluations[member]
        return (
            "canonical",
            -1,
            DecisionTuple(item.feasible, item.penalty, item.sensitivity),
        )
    if adapter == "cyclic-donor":
        donor = (member + 1) % POPULATION
        item = evaluations[donor]
        return (
            "cyclic-donor",
            donor,
            DecisionTuple(item.feasible, item.penalty, item.sensitivity),
        )
    if adapter == "ablated":
        return "ablated", -1, DecisionTuple(False, 0.0, 0.0)
    raise ContractError("tuple-adapter")


def state_hash(
    *,
    u: Sequence[float],
    m: Sequence[float],
    v: Sequence[float],
    age: int,
    stall: int,
    progress: ProgressState,
    incumbent_present: bool,
    incumbent_sensitivity: float | None,
    incumbent_source_batch: int,
    incumbent_source_member: int,
    incumbent_center: Sequence[float],
    restart_round: int,
) -> str:
    record = {
        "u": list(float(value) for value in u),
        "m": list(float(value) for value in m),
        "v": list(float(value) for value in v),
        "age": int(age),
        "stall": int(stall),
        "progress": progress.to_dict(),
        "incumbent_present": bool(incumbent_present),
        "incumbent_sensitivity": incumbent_sensitivity,
        "incumbent_source_batch": int(incumbent_source_batch),
        "incumbent_source_member": int(incumbent_source_member),
        "incumbent_center": list(float(value) for value in incumbent_center),
        "restart_round": int(restart_round),
    }
    return sha256_domain("OptimizerState", canonical_line(record))


class RestartDrawProvider:
    def __init__(self, transcript: Transcript):
        self._transcript = transcript
        self._released: set[int] = set()
        self._next_batch = 0
        self._pending: DrawAuthorization | None = None
        self._denied_attempts: list[str] = []

    @property
    def denied_attempts(self) -> tuple[str, ...]:
        return tuple(self._denied_attempts)

    def authorize(
        self,
        restart_mask: Sequence[bool],
        restart_round: int,
    ) -> DrawAuthorization:
        batch = self._next_batch
        if batch >= BATCHES:
            raise ContractError("draw-budget-exhausted")
        if batch in self._released or self._pending is not None:
            raise ContractError("duplicate-draw-release")
        mask = tuple(restart_mask)
        if len(mask) != POPULATION or any(type(item) is not bool for item in mask):
            raise ContractError("restart-mask-shape")
        authorization = DrawAuthorization(
            batch=batch,
            restart_round=int(restart_round),
            restart_mask=mask,
            fresh_draw_sha256=self._transcript.fresh_hashes[batch],
            perturb_draw_sha256=self._transcript.perturb_hashes[batch],
        )
        self._pending = authorization
        return authorization

    def seal(
        self,
        authorization: DrawAuthorization,
        receipt: BatchReceipt,
    ) -> SealedDrawAuthorization:
        if self._pending is not authorization:
            raise ContractError("draw-authorization-identity")
        expected_after = authorization.restart_round + int(
            any(authorization.restart_mask)
        )
        if (
            receipt.batch != authorization.batch
            or receipt.restart_round_before != authorization.restart_round
            or receipt.restart_round_after != expected_after
            or receipt.restart_mask != authorization.restart_mask
            or receipt.fresh_draw_sha256 != authorization.fresh_draw_sha256
            or receipt.perturb_draw_sha256 != authorization.perturb_draw_sha256
        ):
            raise ContractError("draw-receipt-authorization")
        return SealedDrawAuthorization(
            authorization=authorization,
            receipt_sha256=sha256_domain(
                "BatchReceiptCommitment", canonical_line(receipt.to_dict())
            ),
        )

    def apply(
        self,
        sealed: SealedDrawAuthorization,
        prepared: PreparedStep,
        *,
        receipt: BatchReceipt,
        incumbent_present: bool,
        incumbent_center: Sequence[float],
        budget_fraction: float,
    ) -> StepOutcome:
        authorization = sealed.authorization
        if self._pending is not authorization:
            raise ContractError("draw-authorization-identity")
        if sealed.receipt_sha256 != sha256_domain(
            "BatchReceiptCommitment", canonical_line(receipt.to_dict())
        ):
            raise ContractError("draw-receipt-commitment")
        if prepared.restart_mask != authorization.restart_mask:
            raise ContractError("draw-mask-commitment")
        if (
            receipt.incumbent_present is not incumbent_present
            or receipt.incumbent_center_sha256
            != incumbent_center_hash(incumbent_present, incumbent_center)
        ):
            raise ContractError("draw-center-commitment")
        if budget_fraction != float((authorization.batch + 1) / BATCHES):
            raise ContractError("draw-budget-commitment")
        u = np.array(prepared.u, copy=True)
        m = np.array(prepared.m, copy=True)
        v = np.array(prepared.v, copy=True)
        ages = np.array(prepared.ages, copy=True)
        stalls = np.array(prepared.stalls, copy=True)
        progress = list(prepared.progress)
        restart_mask = authorization.restart_mask
        restart_kinds = ["none"] * POPULATION
        center_sources = ["none"] * POPULATION
        round_after = authorization.restart_round
        if any(restart_mask):
            fresh = self._transcript.fresh[authorization.batch]
            perturb = self._transcript.perturb[authorization.batch]
            scale = 0.35 * max(0.10, 1.0 - budget_fraction)
            centered = perturb - np.mean(perturb, axis=0, dtype=np.float64)
            normalized = centered / (
                np.std(perturb, axis=0, ddof=0, dtype=np.float64) + 1.0e-6
            )
            center = np.asarray(incumbent_center, dtype=np.float64)
            for member, selected in enumerate(restart_mask):
                if not selected:
                    continue
                use_incumbent = incumbent_present and (
                    (member + authorization.restart_round) % 2 == 0
                )
                if use_incumbent:
                    u[member] = center + scale * normalized[member]
                    restart_kinds[member] = "incumbent"
                    center_sources[member] = "global-feasible"
                else:
                    u[member] = fresh[member]
                    restart_kinds[member] = "fresh"
                    center_sources[member] = "fresh"
                m[member] = 0.0
                v[member] = 0.0
                ages[member] = 0
                stalls[member] = 0
                progress[member] = ProgressState(
                    prepared.progress[member].mode, False, False, 0.0, 0.0
                )
            round_after += 1
        self._released.add(authorization.batch)
        self._next_batch += 1
        self._pending = None
        return StepOutcome(
            u=_readonly(u),
            m=_readonly(m),
            v=_readonly(v),
            ages=_readonly_int(ages),
            stalls=_readonly_int(stalls),
            progress=tuple(progress),
            restart_mask=restart_mask,
            restart_kinds=tuple(restart_kinds),
            center_sources=tuple(center_sources),
            restart_round_after=round_after,
        )

    def deny_optimizer_access(self, capability: str) -> None:
        if capability not in {"future-transcript-read", "transcript-fingerprint"}:
            raise ContractError("unknown-capability-probe")
        self._denied_attempts.append(capability)
        raise ContractError("capability-denied")

    def finalize(self) -> None:
        if self._pending is not None or self._released != set(range(BATCHES)):
            raise ContractError("unused-draw-not-authenticated")


class OptimizerAdapter:
    def __init__(self, arm: str):
        self.config = intervention_config(arm)
        self.arm = self.config.arm_id

    def probe_forbidden(
        self,
        capability: str,
        boundary: OptimizerCapabilityBoundary,
        provider: RestartDrawProvider | None = None,
    ) -> None:
        if capability not in CAPABILITY_ATTACKS:
            raise ContractError("unknown-capability-probe")
        if capability in {"future-transcript-read", "transcript-fingerprint"}:
            if provider is None:
                raise ContractError("capability-probe-boundary")
            provider.deny_optimizer_access(capability)
        boundary.read_forbidden(CAPABILITY_PATHS[capability])

    def consume_progress(
        self,
        state: ProgressState,
        *,
        loss: float,
        decision: DecisionTuple,
    ) -> tuple[bool, ProgressState]:
        """The single production progress-consumer boundary."""
        return progress_decision(state, loss=loss, decision=decision)

    def prepare(
        self,
        packet: OptimizerPacket,
    ) -> PreparedStep:
        # Progress is consumed before the Adam update, matching the frozen
        # evaluate -> incumbent -> progress -> update -> restart chronology.
        next_progress: list[ProgressState] = []
        next_stalls = np.empty(POPULATION, dtype=np.int64)
        for member in range(POPULATION):
            improved, progress = self.consume_progress(
                packet.progress[member],
                loss=float(packet.losses[member]),
                decision=packet.decisions[member],
            )
            next_progress.append(progress)
            next_stalls[member] = 0 if improved else int(packet.stalls[member]) + 1

        gradients = np.asarray(packet.gradients, dtype=np.float64)
        norms = np.sqrt(np.sum(gradients * gradients, axis=1))
        scales = np.minimum(1.0, 1.0 / (norms + 1.0e-12))
        clipped = gradients * scales[:, None]
        ages = np.asarray(packet.ages, dtype=np.int64) + 1
        m = 0.9 * packet.m + 0.1 * clipped
        v = 0.999 * packet.v + 0.001 * clipped * clipped
        mhat = m / (1.0 - np.power(0.9, ages))[:, None]
        vhat = v / (1.0 - np.power(0.999, ages))[:, None]
        learning_rates = np.asarray(packet.learning_rates, dtype=np.float64)
        if learning_rates.shape != (POPULATION,):
            raise ContractError("learning-rate-shape")
        u = packet.u - learning_rates[:, None] * mhat / (
            np.sqrt(vhat) + 1.0e-8
        )

        planned_mask = next_stalls >= PATIENCE
        restart_mask_array = (
            np.zeros(POPULATION, dtype=bool)
            if not self.config.restart_enabled
            else planned_mask
        )
        restart_mask = tuple(bool(value) for value in restart_mask_array)
        return PreparedStep(
            u=_readonly(u),
            m=_readonly(m),
            v=_readonly(v),
            ages=_readonly_int(ages),
            stalls=_readonly_int(next_stalls),
            progress=tuple(next_progress),
            restart_mask=restart_mask,
        )


OBSERVATION_FIELDS = tuple(ObservationRecord.__dataclass_fields__)
TRANSITION_FIELDS = tuple(TransitionRecord.__dataclass_fields__)


def _valid_hash(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(item) is float and math.isfinite(item) for item in value)
    )


def _validate_progress_mapping(value: object) -> None:
    if not isinstance(value, Mapping) or tuple(value) != tuple(
        ProgressState.__dataclass_fields__
    ):
        raise ContractError("progress-schema")
    if value["mode"] not in {"raw", "lex"}:
        raise ContractError("progress-mode")
    if type(value["observed"]) is not bool or type(value["feasible"]) is not bool:
        raise ContractError("progress-type")
    if any(
        type(value[name]) is not float or not math.isfinite(value[name])
        for name in ("first", "second")
    ):
        raise ContractError("progress-float")


def validate_observation_mapping(value: Mapping[str, object]) -> None:
    if tuple(value) != OBSERVATION_FIELDS:
        raise ContractError("observation-schema")
    if value["family"] not in FAMILIES:
        raise ContractError("family-identity")
    if type(value["world"]) is not int or value["world"] not in range(16):
        raise ContractError("world-identity")
    if type(value["seed"]) is not int or value["seed"] not in SEEDS:
        raise ContractError("seed-identity")
    if value["order"] not in ORDERS:
        raise ContractError("order-identity")
    if value["arm"] not in ARMS:
        raise ContractError("arm-identity")
    if type(value["batch"]) is not int or value["batch"] not in range(BATCHES):
        raise ContractError("batch-identity")
    if type(value["member"]) is not int or value["member"] not in range(POPULATION):
        raise ContractError("member-identity")
    if not _finite_vector(value["u"], 3) or not _finite_vector(value["x"], 3):
        raise ContractError("parameter-vector")
    if type(value["loss"]) is not float or not math.isfinite(value["loss"]):
        raise ContractError("nonfinite-loss")
    gradient = value["gradient"]
    if not isinstance(gradient, list):
        raise ContractError("gradient-dtype")
    if len(gradient) != 3:
        raise ContractError("gradient-shape")
    if any(type(item) is not float for item in gradient):
        raise ContractError("gradient-dtype")
    if any(not math.isfinite(item) for item in gradient):
        raise ContractError("gradient-nonfinite")
    nonfinite = value["gradient_nonfinite"]
    if not isinstance(nonfinite, list) or len(nonfinite) != 3 or any(
        type(item) is not bool for item in nonfinite
    ):
        raise ContractError("gradient-mask")
    if type(value["canonical_is_feasible"]) is not bool:
        raise ContractError("feasible-type")
    for name in ("sensitivity", "penalty", "violation", "decision_sensitivity"):
        if type(value[name]) is not float or not math.isfinite(value[name]):
            raise ContractError("observation-float")
    if type(value["decision_penalty"]) is not float or value["decision_penalty"] < 0.0:
        raise ContractError("negative-penalty")
    if not math.isfinite(value["decision_penalty"]):
        raise ContractError("observation-float")
    if value["decision_source"] not in {
        "unused",
        "canonical",
        "cyclic-donor",
        "ablated",
    }:
        raise ContractError("decision-source")
    if type(value["decision_donor_member"]) is not int or value[
        "decision_donor_member"
    ] not in range(-1, POPULATION):
        raise ContractError("decision-donor")
    if type(value["decision_is_feasible"]) is not bool:
        raise ContractError("decision-feasible-type")


def validate_transition_mapping(
    value: Mapping[str, object],
    *,
    expected_seed: int,
    expected_order: str,
) -> None:
    if tuple(value) != TRANSITION_FIELDS:
        raise ContractError("transition-schema")
    if value["seed"] != expected_seed:
        raise ContractError("cross-seed-join")
    if value["order"] != expected_order:
        raise ContractError("cross-order-join")
    if value["family"] not in FAMILIES:
        raise ContractError("family-identity")
    if type(value["world"]) is not int or value["world"] not in range(16):
        raise ContractError("world-identity")
    if value["arm"] not in ARMS:
        raise ContractError("arm-identity")
    if type(value["batch"]) is not int or value["batch"] not in range(BATCHES):
        raise ContractError("batch-identity")
    if type(value["member"]) is not int or value["member"] not in range(POPULATION):
        raise ContractError("member-identity")
    _validate_progress_mapping(value["progress_before"])
    _validate_progress_mapping(value["progress_after"])
    for name in (
        "stall_before",
        "stall_after",
        "adam_age_before",
        "adam_age_after",
        "restart_round",
    ):
        if type(value[name]) is not int or value[name] < (-1 if name == "restart_round" else 0):
            raise ContractError("transition-integer")
    if type(value["update_applied"]) is not bool or value["update_applied"] is not True:
        raise ContractError("update-applied")
    if type(value["restart_triggered"]) is not bool:
        raise ContractError("restart-type")
    if value["restart_kind"] not in {"none", "fresh", "incumbent"}:
        raise ContractError("restart-kind")
    if value["center_source"] not in {"none", "fresh", "global-feasible"}:
        raise ContractError("center-source")
    triggered = value["restart_triggered"]
    if triggered:
        if value["restart_kind"] == "none" or value["restart_round"] < 0:
            raise ContractError("restart-identity")
        expected_center = (
            "global-feasible"
            if value["restart_kind"] == "incumbent"
            else "fresh"
        )
        if value["center_source"] != expected_center:
            raise ContractError("restart-center")
    elif (
        value["restart_kind"] != "none"
        or value["restart_round"] != -1
        or value["center_source"] != "none"
    ):
        raise ContractError("restart-identity")
    if not _valid_hash(value["state_before_sha256"]) or not _valid_hash(
        value["state_after_sha256"]
    ):
        raise ContractError("state-hash")


def validate_batch_identity(
    observations: Sequence[Mapping[str, object]],
    *,
    family: str,
    world: int,
    seed: int,
    order: str,
    arm: str,
    batch: int,
) -> None:
    keys = [
        (
            item.get("family"),
            item.get("world"),
            item.get("seed"),
            item.get("order"),
            item.get("arm"),
            item.get("batch"),
            item.get("member"),
        )
        for item in observations
    ]
    if len(keys) != len(set(keys)):
        raise ContractError("duplicate-key")
    expected = [
        (family, world, seed, order, arm, batch, member)
        for member in range(POPULATION)
    ]
    if keys != expected:
        if len(keys) != POPULATION or {key[-1] for key in keys} != set(
            range(POPULATION)
        ):
            raise ContractError("missing-member")
        raise ContractError("observation-identity")


def _update_incumbent(
    evaluations: Sequence[Evaluation],
    *,
    batch: int,
    present: bool,
    sensitivity: float | None,
    source_batch: int,
    source_member: int,
    center: np.ndarray,
) -> tuple[bool, float | None, int, int, np.ndarray]:
    candidates = [
        (item.sensitivity, member)
        for member, item in enumerate(evaluations)
        if item.feasible and math.isfinite(item.sensitivity)
    ]
    if not candidates:
        return present, sensitivity, source_batch, source_member, center
    candidate_sensitivity, member = min(candidates)
    if not present or (
        sensitivity is not None and candidate_sensitivity < sensitivity
    ):
        return (
            True,
            float(candidate_sensitivity),
            batch,
            member,
            _readonly(np.asarray(evaluations[member].u, dtype=np.float64)),
        )
    return present, sensitivity, source_batch, source_member, center


def _trajectory_roots(
    observations: Sequence[ObservationRecord],
    transitions: Sequence[TransitionRecord],
    receipts: Sequence[BatchReceipt],
) -> tuple[str, str]:
    event = RecordRoot("TrajectoryEvent")
    twin = RecordRoot("TrajectoryTwin")
    for batch in range(BATCHES):
        for member in range(POPULATION):
            observation = observations[batch * POPULATION + member]
            event.update_mapping(observation.to_dict())
            twin.update_mapping(observation.to_dict(omit_order=True))
        for member in range(POPULATION):
            transition = transitions[batch * POPULATION + member]
            event.update_mapping(transition.to_dict())
            twin.update_mapping(transition.to_dict(omit_order=True))
        event.update_mapping(receipts[batch].to_dict())
        twin.update_mapping(receipts[batch].to_dict(omit_order=True))
    return event.hexdigest(), twin.hexdigest()


def execute_trajectory(
    world: WorldRecord,
    transcript: Transcript,
    *,
    seed: int,
    order: str,
    arm: str,
    oracle: OracleConsumer | None = None,
) -> TrajectoryEvidence:
    if seed != transcript.seed or seed not in SEEDS:
        raise ContractError("trajectory-seed")
    if order not in ORDERS or arm not in ARMS:
        raise ContractError("trajectory-identity")
    u = np.zeros((POPULATION, 3), dtype=np.float64)
    u[1:] = transcript.suffix
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    ages = np.zeros(POPULATION, dtype=np.int64)
    stalls = np.zeros(POPULATION, dtype=np.int64)
    mode = intervention_config(arm).progress_comparator
    progress = tuple(
        ProgressState(mode, False, False, 0.0, 0.0) for _ in range(POPULATION)
    )
    incumbent_present = False
    incumbent_sensitivity: float | None = None
    incumbent_source_batch = -1
    incumbent_source_member = -1
    incumbent_center = _readonly(np.zeros(3, dtype=np.float64))
    restart_round = 0
    best_feasible: float | None = None
    observations: list[ObservationRecord] = []
    transitions: list[TransitionRecord] = []
    receipts: list[BatchReceipt] = []
    provider = RestartDrawProvider(transcript)
    adapter = OptimizerAdapter(arm)

    for batch in range(BATCHES):
        evaluations = tuple(evaluate(world, u[member]) for member in range(POPULATION))
        for item in evaluations:
            if item.feasible and (
                best_feasible is None or item.sensitivity < best_feasible
            ):
                best_feasible = item.sensitivity
        (
            incumbent_present,
            incumbent_sensitivity,
            incumbent_source_batch,
            incumbent_source_member,
            incumbent_center,
        ) = _update_incumbent(
            evaluations,
            batch=batch,
            present=incumbent_present,
            sensitivity=incumbent_sensitivity,
            source_batch=incumbent_source_batch,
            source_member=incumbent_source_member,
            center=incumbent_center,
        )
        center_hash = incumbent_center_hash(incumbent_present, incumbent_center)
        decisions: list[DecisionTuple] = []
        batch_observations: list[ObservationRecord] = []
        for member, item in enumerate(evaluations):
            source, donor, decision = decision_tuple(arm, member, evaluations)
            decisions.append(decision)
            record = ObservationRecord(
                family=world.family,
                world=world.world,
                seed=seed,
                order=order,
                arm=arm,
                batch=batch,
                member=member,
                u=item.u,
                x=item.x,
                loss=item.loss,
                gradient=item.gradient,
                gradient_nonfinite=item.nonfinite,
                canonical_is_feasible=item.feasible,
                sensitivity=item.sensitivity,
                penalty=item.penalty,
                violation=item.violation,
                decision_source=source,
                decision_donor_member=donor,
                decision_is_feasible=decision.is_feasible,
                decision_penalty=decision.penalty,
                decision_sensitivity=decision.sensitivity,
            )
            validate_observation_mapping(record.to_dict())
            batch_observations.append(record)
        validate_batch_identity(
            [item.to_dict() for item in batch_observations],
            family=world.family,
            world=world.world,
            seed=seed,
            order=order,
            arm=arm,
            batch=batch,
        )
        observations.extend(batch_observations)
        before_hashes = tuple(
            state_hash(
                u=u[member],
                m=m[member],
                v=v[member],
                age=int(ages[member]),
                stall=int(stalls[member]),
                progress=progress[member],
                incumbent_present=incumbent_present,
                incumbent_sensitivity=incumbent_sensitivity,
                incumbent_source_batch=incumbent_source_batch,
                incumbent_source_member=incumbent_source_member,
                incumbent_center=incumbent_center,
                restart_round=restart_round,
            )
            for member in range(POPULATION)
        )
        packet = OptimizerPacket(
            u=_readonly(u),
            learning_rates=_readonly(
                np.geomspace(0.03, 0.15, POPULATION, dtype=np.float64)
            ),
            m=_readonly(m),
            v=_readonly(v),
            ages=_readonly_int(ages),
            stalls=_readonly_int(stalls),
            progress=progress,
            losses=_readonly(np.asarray([item.loss for item in evaluations])),
            gradients=_readonly(
                np.asarray([item.gradient for item in evaluations], dtype=np.float64)
            ),
            decisions=tuple(decisions),
            incumbent_present=incumbent_present,
            incumbent_center=incumbent_center,
            restart_round=restart_round,
            budget_fraction=float((batch + 1) / BATCHES),
        )
        prepared = adapter.prepare(packet)
        authorization = provider.authorize(prepared.restart_mask, restart_round)
        receipt = BatchReceipt(
            family=world.family,
            world=world.world,
            seed=seed,
            order=order,
            arm=arm,
            batch=batch,
            incumbent_present=incumbent_present,
            incumbent_sensitivity=incumbent_sensitivity,
            incumbent_source_batch=incumbent_source_batch,
            incumbent_source_member=incumbent_source_member,
            incumbent_center_sha256=center_hash,
            restart_round_before=restart_round,
            restart_round_after=restart_round + int(any(prepared.restart_mask)),
            restart_mask=prepared.restart_mask,
            fresh_draw_sha256=authorization.fresh_draw_sha256,
            perturb_draw_sha256=authorization.perturb_draw_sha256,
        )
        receipts.append(receipt)
        sealed = provider.seal(authorization, receipt)
        outcome = provider.apply(
            sealed,
            prepared,
            receipt=receipt,
            incumbent_present=incumbent_present,
            incumbent_center=incumbent_center,
            budget_fraction=packet.budget_fraction,
        )
        batch_transitions: list[TransitionRecord] = []
        for member in range(POPULATION):
            after_hash = state_hash(
                u=outcome.u[member],
                m=outcome.m[member],
                v=outcome.v[member],
                age=int(outcome.ages[member]),
                stall=int(outcome.stalls[member]),
                progress=outcome.progress[member],
                incumbent_present=incumbent_present,
                incumbent_sensitivity=incumbent_sensitivity,
                incumbent_source_batch=incumbent_source_batch,
                incumbent_source_member=incumbent_source_member,
                incumbent_center=incumbent_center,
                restart_round=outcome.restart_round_after,
            )
            transition = TransitionRecord(
                family=world.family,
                world=world.world,
                seed=seed,
                order=order,
                arm=arm,
                batch=batch,
                member=member,
                progress_before=progress[member],
                progress_after=outcome.progress[member],
                stall_before=int(stalls[member]),
                stall_after=int(outcome.stalls[member]),
                adam_age_before=int(ages[member]),
                adam_age_after=int(outcome.ages[member]),
                update_applied=True,
                restart_triggered=outcome.restart_mask[member],
                restart_kind=outcome.restart_kinds[member],
                restart_round=(
                    restart_round if outcome.restart_mask[member] else -1
                ),
                center_source=outcome.center_sources[member],
                state_before_sha256=before_hashes[member],
                state_after_sha256=after_hash,
            )
            validate_transition_mapping(
                transition.to_dict(), expected_seed=seed, expected_order=order
            )
            batch_transitions.append(transition)
        transitions.extend(batch_transitions)
        if outcome.restart_round_after != receipt.restart_round_after:
            raise ContractError("draw-round-commitment")
        u = np.array(outcome.u, copy=True)
        m = np.array(outcome.m, copy=True)
        v = np.array(outcome.v, copy=True)
        ages = np.array(outcome.ages, copy=True)
        stalls = np.array(outcome.stalls, copy=True)
        progress = outcome.progress
        restart_round = outcome.restart_round_after
    provider.finalize()
    if world.family == "impossible":
        if best_feasible is not None:
            raise ContractError("false-feasible-join")
        gap = 1.0
    elif best_feasible is None:
        gap = 1.0
    else:
        assert world.reference_sensitivity is not None and world.denominator is not None
        gap = float(
            (best_feasible - world.reference_sensitivity) / world.denominator
        )
        if gap < -1.0e-10:
            raise ContractError("reference-contract")
    event_root, twin_root = _trajectory_roots(observations, transitions, receipts)
    trajectory = TrajectoryRecord(
        family=world.family,
        world=world.world,
        seed=seed,
        order=order,
        arm=arm,
        evaluations=EVALUATIONS_PER_TRAJECTORY,
        transitions=TRANSITIONS_PER_TRAJECTORY,
        best_feasible_sensitivity=best_feasible,
        gap=gap,
        normalized_twin_sha256=twin_root,
        event_root_sha256=event_root,
    )
    evidence = TrajectoryEvidence(
        collector=CollectorEnvelope(
            observations=tuple(observations),
            transitions=tuple(transitions),
            receipts=tuple(receipts),
        ),
        evaluator=EvaluatorEnvelope(world=world, trajectory=trajectory),
        restart_events=sum(
            int(item.restart_triggered) for item in transitions
        ),
    )
    if oracle is not None:
        oracle.consume(world, transcript, evidence)
    return evidence


def progress_sentinel_records() -> tuple[SentinelRecord, SentinelRecord]:
    stored = ProgressState("lex", True, False, 1.0, 5.0)
    canonical = (
        DecisionTuple(False, 0.9, 6.0),
        DecisionTuple(False, 0.8, 7.0),
    )
    donor = (
        DecisionTuple(False, 1.1, 0.0),
        DecisionTuple(False, 1.2, 0.0),
    )
    ablated = (DecisionTuple(False, 0.0, 0.0),) * 2

    adapter = OptimizerAdapter("constraint_lexicographic_progress")

    def consume(values: Sequence[DecisionTuple]) -> tuple[bool, ...]:
        state = stored
        result: list[bool] = []
        for value in values:
            improved, state = adapter.consume_progress(
                state, loss=0.0, decision=value
            )
            result.append(improved)
        return tuple(result)

    outcomes = consume(canonical), consume(donor), consume(ablated)
    if outcomes != ((True, True), (False, False), (True, False)):
        raise ContractError("progress-sentinel")
    return tuple(
        SentinelRecord(
            sentinel_id="progress-consumer-v1",
            ordinal=ordinal,
            member=0,
            stored=stored,
            canonical_tuple=canonical[ordinal],
            donor_member=1,
            donor_tuple=donor[ordinal],
            ablated_tuple=ablated[ordinal],
            canonical_decision=outcomes[0][ordinal],
            donor_decision=outcomes[1][ordinal],
            ablated_decision=outcomes[2][ordinal],
        )
        for ordinal in range(2)
    )  # type: ignore[return-value]


def progress_sentinel() -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]]:
    records = progress_sentinel_records()
    return (
        tuple(record.canonical_decision for record in records),
        tuple(record.donor_decision for record in records),
        tuple(record.ablated_decision for record in records),
    )


def capability_attack_receipts() -> tuple[AttackReceipt, ...]:
    transcript = build_transcript(SEEDS[0])
    receipts: list[AttackReceipt] = []
    for attack in CAPABILITY_ATTACKS:
        adapter = OptimizerAdapter("constraint_lexicographic_progress")
        boundary = OptimizerCapabilityBoundary()
        draw_provider = RestartDrawProvider(transcript)
        before = (
            adapter.config,
            boundary.attempted_paths,
            draw_provider._next_batch,
            tuple(sorted(draw_provider._released)),
            draw_provider._pending,
        )
        try:
            adapter.probe_forbidden(attack, boundary, draw_provider)
        except ContractError as error:
            if str(error) != "capability-denied":
                raise
        else:
            raise ContractError("capability-probe-not-rejected")
        after = (
            adapter.config,
            tuple() if attack in {
                "future-transcript-read",
                "transcript-fingerprint",
            } else boundary.attempted_paths,
            draw_provider._next_batch,
            tuple(sorted(draw_provider._released)),
            draw_provider._pending,
        )
        expected_paths = (
            tuple()
            if attack in {"future-transcript-read", "transcript-fingerprint"}
            else (CAPABILITY_PATHS[attack],)
        )
        if boundary.attempted_paths != expected_paths:
            raise ContractError("capability-consumer-not-reached")
        expected_provider_attempts = (
            (attack,)
            if attack in {"future-transcript-read", "transcript-fingerprint"}
            else tuple()
        )
        if draw_provider.denied_attempts != expected_provider_attempts:
            raise ContractError("capability-consumer-not-reached")
        receipts.append(
            AttackReceipt(
                attack_id=attack,
                injection_path=CAPABILITY_PATHS[attack],
                rejection_code="capability-denied",
                consumer_reached=(
                    draw_provider.denied_attempts == (attack,)
                    or boundary.attempted_paths == (CAPABILITY_PATHS[attack],)
                ),
                state_mutations=int(
                    before[0] != after[0]
                    or before[2:] != after[2:]
                ),
            )
        )
    return tuple(receipts)


def world_from_mapping(value: Mapping[str, object]) -> WorldRecord:
    expected = tuple(WorldRecord.__dataclass_fields__)
    if tuple(value) != expected:
        raise ContractError("world-schema")
    bits = value["bits"]
    if not isinstance(bits, list) or len(bits) != 4 or any(
        type(item) is not int or item not in {0, 1} for item in bits
    ):
        raise ContractError("world-bits")
    if type(value["family"]) is not str or value["family"] not in FAMILIES:
        raise ContractError("world-family")
    if type(value["world"]) is not int or value["world"] not in range(16):
        raise ContractError("world-identity")
    if type(value["split"]) is not str or value["split"] not in SPLITS:
        raise ContractError("world-split")
    for name in ("a", "b", "k", "t", "c", "threshold"):
        if type(value[name]) is not float or not math.isfinite(value[name]):
            raise ContractError("world-float")
    for name in ("reference_x0", "reference_sensitivity", "denominator"):
        item = value[name]
        if item is not None and (type(item) is not float or not math.isfinite(item)):
            raise ContractError("world-reference")
    return WorldRecord(
        family=value["family"],
        world=value["world"],
        bits=tuple(bits),  # type: ignore[arg-type]
        split=value["split"],
        a=value["a"],
        b=value["b"],
        k=value["k"],
        t=value["t"],
        c=value["c"],
        threshold=value["threshold"],
        reference_x0=(
            None if value["reference_x0"] is None else value["reference_x0"]
        ),
        reference_sensitivity=(
            None
            if value["reference_sensitivity"] is None
            else value["reference_sensitivity"]
        ),
        denominator=(
            None if value["denominator"] is None else value["denominator"]
        ),
    )


def phase_input_root(records: Sequence[WorldRecord]) -> str:
    root = RecordRoot("PhaseInput")
    for record in records:
        root.update_mapping(record.to_dict())
    return root.hexdigest()


def root_of_roots(domain: str, children: Sequence[str]) -> str:
    if any(len(item) != 64 for item in children):
        raise ContractError("child-root")
    return sha256_domain(
        domain,
        b"".join(item.encode("ascii") + b"\n" for item in children),
    )


__all__ = [
    "ARMS",
    "AttackReceipt",
    "BATCHES",
    "BatchReceipt",
    "CAPABILITY_ATTACKS",
    "CAPABILITY_PATHS",
    "ContractError",
    "DecisionTuple",
    "EVALUATIONS_PER_TRAJECTORY",
    "FAMILIES",
    "InterventionConfig",
    "MALFORMED_ATTACKS",
    "ObservationRecord",
    "OptimizerAdapter",
    "OptimizerCapabilityBoundary",
    "OptimizerPacket",
    "ORDERS",
    "PLAN_PATH",
    "PLAN_REVISION",
    "POPULATION",
    "ProgressState",
    "RUNTIME_IDENTITY",
    "RecordRoot",
    "SEEDS",
    "SentinelRecord",
    "SPLITS",
    "STUDY_ID",
    "TRANSCRIPT_HASHES",
    "TRANSCRIPT_ROOT_SHA256",
    "TRANSITIONS_PER_TRAJECTORY",
    "TrajectoryEvidence",
    "TrajectoryRecord",
    "WorldAggregateRecord",
    "WorldRecord",
    "build_transcript",
    "build_world_records",
    "canonical_line",
    "capability_attack_receipts",
    "decision_tuple",
    "draw_hash",
    "evaluate",
    "execute_trajectory",
    "incumbent_center_hash",
    "intervention_config",
    "normalize_json",
    "phase_input_root",
    "progress_decision",
    "progress_sentinel",
    "progress_sentinel_records",
    "root_of_roots",
    "sha256_domain",
    "state_hash",
    "transcript_root",
    "validate_batch_identity",
    "validate_observation_mapping",
    "validate_transition_mapping",
    "world_from_mapping",
]

# Independent replay/oracle and result projection copied under the frozen V1
# scientific protocol. Process supervision is injected by the V2 bootstrap.
def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise fx.ContractError("duplicate-json-key")
        value[key] = item
    return value


def _loads_json(value: bytes | str):
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise fx.ContractError("json-packet") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_executable_path() -> Path:
    if os.name != "nt":
        return Path(sys.executable).resolve()
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    if length <= 0 or length >= len(buffer):
        raise fx.ContractError("runtime-executable-identity")
    return Path(buffer.value)


def _runtime_identity() -> dict[str, str]:
    import importlib.metadata
    import numpy.random._pcg64 as pcg64_module
    import numpy.random.bit_generator as bit_generator_module

    numpy_init = Path(np.__file__).resolve()
    metadata_path = Path(importlib.metadata.distribution("numpy")._path) / "METADATA"
    identity = {
        "machine": platform.machine(),
        "numpy_init_sha256": _sha256_file(numpy_init),
        "numpy_metadata_sha256": _sha256_file(metadata_path),
        "numpy_version": np.__version__,
        "pcg64_identity": f"{np.random.PCG64.__module__}.{np.random.PCG64.__name__}",
        "pcg64_module_sha256": _sha256_file(Path(pcg64_module.__file__).resolve()),
        "python_architecture": platform.architecture()[0],
        "python_executable_sha256": _sha256_file(_process_executable_path()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "seed_sequence_identity": (
            f"{np.random.SeedSequence.__module__}."
            f"{np.random.SeedSequence.__name__}"
        ),
        "seed_sequence_module_sha256": _sha256_file(
            Path(bit_generator_module.__file__).resolve()
        ),
    }
    if identity != fx.RUNTIME_IDENTITY:
        raise fx.ContractError("runtime-identity")
    return identity


def _oracle_positive_zero(value: float) -> float:
    value = float(value)
    return 0.0 if value == 0.0 else value


def _normalize(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise fx.ContractError("nonfinite-canonical-value")
        return _oracle_positive_zero(value)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            _normalize(dict(value)),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class _Root:
    def __init__(self, name: str):
        self._hash = hashlib.sha256()
        self._hash.update(SCHEMA_DOMAIN)
        self._hash.update(name.encode("ascii"))
        self._hash.update(b"\0")

    def update(self, value: Mapping[str, object]) -> None:
        self._hash.update(_line(value))

    def update_bytes(self, value: bytes) -> None:
        self._hash.update(value)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _root_of_roots(name: str, roots: Sequence[str]) -> str:
    if any(type(item) is not str or len(item) != 64 for item in roots):
        raise fx.ContractError("child-root")
    return hashlib.sha256(
        SCHEMA_DOMAIN
        + name.encode("ascii")
        + b"\0"
        + b"".join(item.encode("ascii") + b"\n" for item in roots)
    ).hexdigest()


def _state_hash(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        SCHEMA_DOMAIN + b"OptimizerState\0" + _line(record)
    ).hexdigest()


def _draw_hash(kind: str, seed: int, batch: int, values) -> str:
    if kind not in {"fresh", "perturb"}:
        raise fx.ContractError("draw-kind")
    array = np.asarray(values)
    if array.shape != (8, 3) or array.dtype != np.float64:
        raise fx.ContractError("draw-shape")
    return hashlib.sha256(
        SCHEMA_DOMAIN
        + f"{kind}-draw".encode("ascii")
        + b"\0"
        + struct.pack("<Q", seed)
        + struct.pack("<i", batch)
        + np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _center_hash(present: bool, center) -> str:
    if not present:
        return hashlib.sha256(
            b"L2D-constraint-progress-v1/absent-incumbent\0"
        ).hexdigest()
    values = np.asarray([_oracle_positive_zero(value) for value in center], dtype="<f8")
    return hashlib.sha256(
        b"L2D-constraint-progress-v1/incumbent-center\0"
        + values.tobytes(order="C")
    ).hexdigest()


def _oracle_reference(a: float, q: float) -> float:
    low = a / 2.0
    high = a if q > 0.0 else min(1.5 * a, 1.99)
    low_derivative = 4.0 * low * (low * low - a * a) + q
    high_derivative = 4.0 * high * (high * high - a * a) + q
    if low_derivative > 0.0 or high_derivative < 0.0:
        raise fx.ContractError("oracle-reference-bracket-sign")
    for _ in range(80):
        middle = (low + high) / 2.0
        derivative = 4.0 * middle * (middle * middle - a * a) + q
        if derivative <= 0.0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def _oracle_worlds() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
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
                reference_x0 = _oracle_reference(a, q)
                reference_lower = a / 2.0
                reference_upper = a if q > 0.0 else min(1.5 * a, 1.99)
                if not reference_lower <= reference_x0 <= reference_upper:
                    raise fx.ContractError("oracle-reference-bracket")
                reference_sensitivity = float(
                    (reference_x0 * reference_x0 - a * a) ** 2
                    + q * reference_x0
                )
                denominator = float(
                    a**4 + k * b * b + 0.5 * c * c - reference_sensitivity
                )
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise fx.ContractError("oracle-reference-denominator")
            records.append(
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
    return tuple(records)


def _world_replay():
    implementation = tuple(item.to_dict() for item in fx.build_world_records())
    oracle = _oracle_worlds()
    if len(implementation) != 48 or implementation != oracle:
        raise fx.ContractError("family-replay")
    implementation_root = _Root("WorldRecord")
    oracle_root = _Root("WorldRecord")
    for left, right in zip(implementation, oracle, strict=True):
        implementation_root.update(left)
        oracle_root.update(right)
    return implementation, implementation_root.hexdigest(), oracle_root.hexdigest()


def _oracle_transcript(seed: int) -> dict[str, object]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20260830, seed]))
    )
    suffix = generator.standard_normal((7, 3), dtype=np.float64)
    fresh = np.empty((64, 8, 3), dtype=np.float64)
    perturb = np.empty_like(fresh)
    pieces = [suffix.reshape(-1)]
    for batch in range(64):
        fresh[batch] = generator.standard_normal((8, 3), dtype=np.float64)
        perturb[batch] = generator.standard_normal((8, 3), dtype=np.float64)
        pieces.extend((fresh[batch].reshape(-1), perturb[batch].reshape(-1)))
    values = np.concatenate(pieces).astype("<f8", copy=False)
    digest = hashlib.sha256(
        TRANSCRIPT_DOMAIN
        + b"\0"
        + struct.pack("<Q", seed)
        + values.tobytes(order="C")
    ).hexdigest()
    return {
        "seed": seed,
        "suffix": suffix,
        "fresh": fresh,
        "perturb": perturb,
        "sha256": digest,
    }


def _transcript_replay():
    primary = tuple(fx.build_transcript(seed) for seed in fx.SEEDS)
    oracle = tuple(_oracle_transcript(seed) for seed in fx.SEEDS)
    for left, right in zip(primary, oracle, strict=True):
        _validate_transcript_hash(left.seed, left.sha256)
        if (
            left.sha256 != right["sha256"]
            or left.suffix.tobytes() != right["suffix"].tobytes()
            or left.fresh.tobytes() != right["fresh"].tobytes()
            or left.perturb.tobytes() != right["perturb"].tobytes()
        ):
            raise fx.ContractError("transcript-replay")
    observed = fx.transcript_root(primary)
    root = hashlib.sha256(
        TRANSCRIPT_DOMAIN
        + b"/root\0"
        + b"".join(item["sha256"].encode("ascii") + b"\n" for item in oracle)
    ).hexdigest()
    if observed != root or root != fx.TRANSCRIPT_ROOT_SHA256:
        raise fx.ContractError("transcript-root")
    return primary, oracle, root


def _oracle_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    mask = values >= 0.0
    result[mask] = 1.0 / (1.0 + np.exp(-values[mask]))
    exponential = np.exp(values[~mask])
    result[~mask] = exponential / (1.0 + exponential)
    return result


def _evaluate(world: Mapping[str, object], u) -> dict[str, object]:
    values = np.asarray(u, dtype=np.float64)
    sig = _oracle_sigmoid(values)
    x = 4.0 * sig - 2.0
    a = float(world["a"])
    b = float(world["b"])
    k = float(world["k"])
    t = float(world["t"])
    c = float(world["c"])
    threshold = float(world["threshold"])
    q = -t if world["family"] == "aligned" else t
    violation = max(0.0, threshold - float(x[0]))
    penalty = float(0.02 * violation * violation)
    sensitivity = float(
        (float(x[0]) ** 2 - a * a) ** 2
        + q * float(x[0])
        + k * (float(x[1]) - b) ** 2
        + 0.5 * (float(x[2]) - c) ** 2
    )
    loss = float(sensitivity + penalty)
    penalty_dx0 = -0.04 * violation if float(x[0]) < threshold else 0.0
    active = np.asarray(
        [
            4.0 * float(x[0]) * (float(x[0]) ** 2 - a * a)
            + q
            + penalty_dx0,
            2.0 * k * (float(x[1]) - b),
            float(x[2]) - c,
        ],
        dtype=np.float64,
    )
    raw_gradient = active * (4.0 * sig * (1.0 - sig))
    nonfinite = [bool(value) for value in ~np.isfinite(raw_gradient)]
    gradient = np.nan_to_num(raw_gradient, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "u": [float(value) for value in values],
        "x": [float(value) for value in x],
        "loss": loss,
        "gradient": [float(value) for value in gradient],
        "gradient_nonfinite": nonfinite,
        "canonical_is_feasible": bool(float(x[0]) >= threshold),
        "sensitivity": sensitivity,
        "penalty": penalty,
        "violation": float(violation),
    }


def _progress(mode: str, state: Mapping[str, object], loss: float, decision):
    if mode == "raw":
        improved = (not state["observed"]) or (
            math.isfinite(loss) and loss < float(state["first"]) - 1.0e-7
        )
        candidate = {
            "mode": "raw",
            "observed": True,
            "feasible": False,
            "first": float(loss),
            "second": 0.0,
        }
        return improved, candidate if improved else dict(state)
    candidate = (
        {
            "mode": "lex",
            "observed": True,
            "feasible": True,
            "first": float(decision["sensitivity"]),
            "second": 0.0,
        }
        if decision["is_feasible"]
        else {
            "mode": "lex",
            "observed": True,
            "feasible": False,
            "first": float(decision["penalty"]),
            "second": float(decision["sensitivity"]),
        }
    )
    if not state["observed"]:
        return True, candidate
    if decision["is_feasible"] and not state["feasible"]:
        return True, candidate
    if not decision["is_feasible"] and state["feasible"]:
        return False, dict(state)
    if decision["is_feasible"]:
        improved = float(decision["sensitivity"]) < float(state["first"]) - 1.0e-7
        return improved, candidate if improved else dict(state)
    penalty_improved = float(decision["penalty"]) < float(state["first"]) - 1.0e-7
    tied = abs(float(decision["penalty"]) - float(state["first"])) <= 1.0e-7
    sensitivity_improved = (
        float(decision["sensitivity"]) < float(state["second"]) - 1.0e-7
    )
    improved = penalty_improved or (tied and sensitivity_improved)
    return improved, candidate if improved else dict(state)


def _decision(arm: str, member: int, evaluations):
    if arm in {"protected_raw_progress", "no_restart_comparator"}:
        return "unused", -1, {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0}
    if arm == "constraint_lexicographic_progress":
        item = evaluations[member]
        return "canonical", -1, {
            "is_feasible": item["canonical_is_feasible"],
            "penalty": item["penalty"],
            "sensitivity": item["sensitivity"],
        }
    if arm == "shuffled_progress_control":
        donor = (member + 1) % 8
        item = evaluations[donor]
        return "cyclic-donor", donor, {
            "is_feasible": item["canonical_is_feasible"],
            "penalty": item["penalty"],
            "sensitivity": item["sensitivity"],
        }
    return "ablated", -1, {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0}


def _unobserved(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "observed": False,
        "feasible": False,
        "first": 0.0,
        "second": 0.0,
    }


class _ReplayOracle:
    def __init__(self, world, transcript, roots):
        self.world = world
        self.transcript = transcript
        self.roots = roots
        self.summary = None

    def consume(self, world_record, primary_transcript, evidence) -> None:
        del primary_transcript
        world = self.world
        if world_record.to_dict() != world:
            raise fx.ContractError("oracle-world-join")
        trajectory = evidence.evaluator.trajectory
        seed = trajectory.seed
        order = trajectory.order
        arm = trajectory.arm
        observations = evidence.collector.observations
        transitions = evidence.collector.transitions
        receipts = evidence.collector.receipts
        if len(observations) != 512 or len(transitions) != 512 or len(receipts) != 64:
            raise fx.ContractError("trajectory-cardinality")

        u = np.zeros((8, 3), dtype=np.float64)
        u[1:] = self.transcript["suffix"]
        m = np.zeros_like(u)
        v = np.zeros_like(u)
        ages = np.zeros(8, dtype=np.int64)
        stalls = np.zeros(8, dtype=np.int64)
        mode = "lex" if arm in fx.ARMS[1:4] else "raw"
        progress = [_unobserved(mode) for _ in range(8)]
        incumbent_present = False
        incumbent_sensitivity = None
        incumbent_batch = -1
        incumbent_member = -1
        incumbent_center = np.zeros(3, dtype=np.float64)
        restart_round = 0
        best_feasible = None
        event = _Root("TrajectoryEvent")
        twin = _Root("TrajectoryTwin")
        feasible_observations = 0
        restart_events = 0
        schema_valid_observations = 0
        replayed_transitions = 0
        replayed_receipts = 0
        reset_checks = 0
        incumbent_tie_checks = 0
        incumbent_state_checks = 0

        for batch in range(64):
            evaluations = [_evaluate(world, u[member]) for member in range(8)]
            candidates = [
                (item["sensitivity"], member)
                for member, item in enumerate(evaluations)
                if item["canonical_is_feasible"] and math.isfinite(item["sensitivity"])
            ]
            feasible_observations += len(candidates)
            for sensitivity, _member in candidates:
                if best_feasible is None or sensitivity < best_feasible:
                    best_feasible = sensitivity
            if candidates:
                candidate_sensitivity, candidate_member = min(candidates)
                if not incumbent_present or candidate_sensitivity < incumbent_sensitivity:
                    incumbent_present = True
                    incumbent_sensitivity = float(candidate_sensitivity)
                    incumbent_batch = batch
                    incumbent_member = candidate_member
                    incumbent_center = np.asarray(
                        evaluations[candidate_member]["u"], dtype=np.float64
                    )

            decisions = []
            expected_observations = []
            for member, evaluation in enumerate(evaluations):
                source, donor, decision = _decision(arm, member, evaluations)
                decisions.append(decision)
                expected = {
                    "family": world["family"],
                    "world": world["world"],
                    "seed": seed,
                    "order": order,
                    "arm": arm,
                    "batch": batch,
                    "member": member,
                    **evaluation,
                    "decision_source": source,
                    "decision_donor_member": donor,
                    "decision_is_feasible": decision["is_feasible"],
                    "decision_penalty": decision["penalty"],
                    "decision_sensitivity": decision["sensitivity"],
                }
                actual = observations[batch * 8 + member].to_dict()
                if actual != expected:
                    raise fx.ContractError("observation-replay")
                schema_valid_observations += 1
                expected_observations.append(expected)
                self.roots["implementation_schema"].update(actual)
                self.roots["oracle_schema"].update(expected)
                event.update(expected)
                normalized = dict(expected)
                del normalized["order"]
                twin.update(normalized)

            before_hashes = []
            for member in range(8):
                before_hashes.append(
                    _state_hash(
                        {
                            "u": [float(value) for value in u[member]],
                            "m": [float(value) for value in m[member]],
                            "v": [float(value) for value in v[member]],
                            "age": int(ages[member]),
                            "stall": int(stalls[member]),
                            "progress": progress[member],
                            "incumbent_present": incumbent_present,
                            "incumbent_sensitivity": incumbent_sensitivity,
                            "incumbent_source_batch": incumbent_batch,
                            "incumbent_source_member": incumbent_member,
                            "incumbent_center": [float(value) for value in incumbent_center],
                            "restart_round": restart_round,
                        }
                    )
                )

            next_progress = []
            next_stalls = np.empty(8, dtype=np.int64)
            for member in range(8):
                improved, next_state = _progress(
                    mode, progress[member], evaluations[member]["loss"], decisions[member]
                )
                next_progress.append(next_state)
                next_stalls[member] = 0 if improved else int(stalls[member]) + 1

            gradients = np.asarray(
                [item["gradient"] for item in evaluations], dtype=np.float64
            )
            norms = np.sqrt(np.sum(gradients * gradients, axis=1))
            clipped = gradients * np.minimum(1.0, 1.0 / (norms + 1.0e-12))[:, None]
            next_ages = ages + 1
            next_m = 0.9 * m + 0.1 * clipped
            next_v = 0.999 * v + 0.001 * clipped * clipped
            mhat = next_m / (1.0 - np.power(0.9, next_ages))[:, None]
            vhat = next_v / (1.0 - np.power(0.999, next_ages))[:, None]
            rates = np.geomspace(0.03, 0.15, 8, dtype=np.float64)
            next_u = u - rates[:, None] * mhat / (np.sqrt(vhat) + 1.0e-8)
            mask_array = next_stalls >= 8
            if arm == "no_restart_comparator":
                mask_array = np.zeros(8, dtype=bool)
            mask = [bool(value) for value in mask_array]
            fresh = self.transcript["fresh"][batch]
            perturb = self.transcript["perturb"][batch]
            kinds = ["none"] * 8
            centers = ["none"] * 8
            round_after = restart_round
            if any(mask):
                fraction = (batch + 1) / 64
                scale = 0.35 * max(0.10, 1.0 - fraction)
                centered = perturb - np.mean(perturb, axis=0, dtype=np.float64)
                normalized = centered / (
                    np.std(perturb, axis=0, ddof=0, dtype=np.float64) + 1.0e-6
                )
                for member, selected in enumerate(mask):
                    if not selected:
                        continue
                    if incumbent_present and (member + restart_round) % 2 == 0:
                        next_u[member] = incumbent_center + scale * normalized[member]
                        kinds[member] = "incumbent"
                        centers[member] = "global-feasible"
                    else:
                        next_u[member] = fresh[member]
                        kinds[member] = "fresh"
                        centers[member] = "fresh"
                    next_m[member] = 0.0
                    next_v[member] = 0.0
                    next_ages[member] = 0
                    next_stalls[member] = 0
                    next_progress[member] = _unobserved(mode)
                round_after += 1

            for member in range(8):
                after_hash = _state_hash(
                    {
                        "u": [float(value) for value in next_u[member]],
                        "m": [float(value) for value in next_m[member]],
                        "v": [float(value) for value in next_v[member]],
                        "age": int(next_ages[member]),
                        "stall": int(next_stalls[member]),
                        "progress": next_progress[member],
                        "incumbent_present": incumbent_present,
                        "incumbent_sensitivity": incumbent_sensitivity,
                        "incumbent_source_batch": incumbent_batch,
                        "incumbent_source_member": incumbent_member,
                        "incumbent_center": [float(value) for value in incumbent_center],
                        "restart_round": round_after,
                    }
                )
                expected_transition = {
                    "family": world["family"],
                    "world": world["world"],
                    "seed": seed,
                    "order": order,
                    "arm": arm,
                    "batch": batch,
                    "member": member,
                    "progress_before": progress[member],
                    "progress_after": next_progress[member],
                    "stall_before": int(stalls[member]),
                    "stall_after": int(next_stalls[member]),
                    "adam_age_before": int(ages[member]),
                    "adam_age_after": int(next_ages[member]),
                    "update_applied": True,
                    "restart_triggered": mask[member],
                    "restart_kind": kinds[member],
                    "restart_round": restart_round if mask[member] else -1,
                    "center_source": centers[member],
                    "state_before_sha256": before_hashes[member],
                    "state_after_sha256": after_hash,
                }
                actual_transition = transitions[batch * 8 + member].to_dict()
                if actual_transition != expected_transition:
                    raise fx.ContractError("transition-replay")
                replayed_transitions += 1
                if mask[member]:
                    reset_checks += int(
                        next_ages[member] == 0
                        and next_stalls[member] == 0
                        and next_progress[member] == _unobserved(mode)
                        and np.all(next_m[member] == 0.0)
                        and np.all(next_v[member] == 0.0)
                    )
                self.roots["implementation_state"].update(actual_transition)
                self.roots["oracle_state"].update(expected_transition)
                event.update(expected_transition)
                normalized_transition = dict(expected_transition)
                del normalized_transition["order"]
                twin.update(normalized_transition)
                restart_events += int(mask[member])

            expected_receipt = {
                "family": world["family"],
                "world": world["world"],
                "seed": seed,
                "order": order,
                "arm": arm,
                "batch": batch,
                "incumbent_present": incumbent_present,
                "incumbent_sensitivity": incumbent_sensitivity,
                "incumbent_source_batch": incumbent_batch,
                "incumbent_source_member": incumbent_member,
                "incumbent_center_sha256": _center_hash(
                    incumbent_present, incumbent_center
                ),
                "restart_round_before": restart_round,
                "restart_round_after": round_after,
                "restart_mask": mask,
                "fresh_draw_sha256": _draw_hash("fresh", seed, batch, fresh),
                "perturb_draw_sha256": _draw_hash("perturb", seed, batch, perturb),
            }
            actual_receipt = receipts[batch].to_dict()
            if actual_receipt != expected_receipt:
                raise fx.ContractError("receipt-replay")
            replayed_receipts += 1
            incumbent_tie_checks += 1
            incumbent_state_checks += 1
            self.roots["implementation_state"].update(actual_receipt)
            self.roots["oracle_state"].update(expected_receipt)
            event.update(expected_receipt)
            normalized_receipt = dict(expected_receipt)
            del normalized_receipt["order"]
            twin.update(normalized_receipt)

            u = np.array(next_u, copy=True)
            m = np.array(next_m, copy=True)
            v = np.array(next_v, copy=True)
            ages = np.array(next_ages, copy=True)
            stalls = np.array(next_stalls, copy=True)
            progress = next_progress
            restart_round = round_after

        if world["family"] == "impossible":
            gap = 1.0
            if best_feasible is not None:
                raise fx.ContractError("false-feasible-join")
        elif best_feasible is None:
            gap = 1.0
        else:
            gap = float(
                (best_feasible - world["reference_sensitivity"])
                / world["denominator"]
            )
        expected_trajectory = {
            "family": world["family"],
            "world": world["world"],
            "seed": seed,
            "order": order,
            "arm": arm,
            "evaluations": 512,
            "transitions": 512,
            "best_feasible_sensitivity": best_feasible,
            "gap": gap,
            "normalized_twin_sha256": twin.hexdigest(),
            "event_root_sha256": event.hexdigest(),
        }
        actual_trajectory = trajectory.to_dict()
        if actual_trajectory != expected_trajectory:
            raise fx.ContractError("trajectory-replay")
        self.roots["implementation_trajectory"].update(actual_trajectory)
        self.roots["oracle_trajectory"].update(expected_trajectory)
        self.summary = {
            "gap": gap,
            "order": order,
            "normalized_twin_sha256": twin.hexdigest(),
            "event_root_sha256": event.hexdigest(),
            "observations": len(observations),
            "transitions": len(transitions),
            "batches": len(receipts),
            "trajectories": 1,
            "feasible_observations": feasible_observations,
            "restart_events": restart_events,
            "schema_valid_observations": schema_valid_observations,
            "replayed_transitions": replayed_transitions,
            "replayed_receipts": replayed_receipts,
            "reset_checks": reset_checks,
            "incumbent_tie_checks": incumbent_tie_checks,
            "incumbent_state_checks": incumbent_state_checks,
            "reference_used": int(
                world["reference_sensitivity"] is not None and best_feasible is not None
            ),
            "false_feasible_joins": int(
                world["family"] == "impossible" and best_feasible is not None
            ),
            "nonunit_gap": int(world["family"] == "impossible" and gap != 1.0),
            "transcript_sha256": self.transcript["sha256"],
        }

def _validate_phase_payload(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict) or tuple(payload) != (
        "family",
        "split",
        "contract_sha256",
        "transcript_root_sha256",
        "worlds",
    ):
        raise ContractError("phase-packet-schema")
    return dict(payload)


class _ExplodingSourceSentinel:
    def __init__(self, forbidden_split: str) -> None:
        if forbidden_split not in {"development", "heldout"}:
            raise fx.ContractError("source-sentinel-split")
        self.forbidden_split = forbidden_split
        self.attached_source = None

    def attach(self, source) -> None:
        if self.attached_source is not None:
            raise fx.ContractError("duplicate-source-sentinel")
        self.attached_source = source

    def reject(self, attempted_split: str) -> None:
        if attempted_split != self.forbidden_split:
            raise fx.ContractError("source-sentinel-identity")
        raise fx.ContractError("forbidden-source-read")


class _WorldSource:
    """The exercised phase source boundary with an attached exploding sentinel."""

    def __init__(self, family: str, split: str, records, sentinel):
        self.family = family
        self.split = split
        self._records = {(item.family, item.world): item for item in records}
        self.attempted_reads = 0
        self.forbidden_reads = 0
        self._sentinel = sentinel
        sentinel.attach(self)

    @property
    def sentinel_connected(self) -> bool:
        return self._sentinel.attached_source is self

    def get(self, world: int):
        self.attempted_reads += 1
        key = (self.family, world)
        item = self._records.get(key)
        if item is None or item.split != self.split:
            self.forbidden_reads += 1
            attempted_split = (
                "heldout" if self.split == "development" else "development"
            )
            self._sentinel.reject(attempted_split)
        return item


def run_phase(payload: Mapping[str, object]) -> dict[str, object]:
    _runtime_identity()
    payload = _validate_phase_payload(payload)
    family = payload["family"]
    split = payload["split"]
    contract_sha256 = payload["contract_sha256"]
    transcript_root_sha256 = payload["transcript_root_sha256"]
    raw_worlds = payload["worlds"]
    if family not in fx.FAMILIES or split not in fx.SPLITS:
        raise fx.ContractError("phase-identity")
    if not isinstance(raw_worlds, list) or len(raw_worlds) != 8:
        raise fx.ContractError("phase-world-count")
    if (
        contract_sha256 != _required_environment("L2D_CONTRACT_SHA256", 64)
        or transcript_root_sha256 != fx.TRANSCRIPT_ROOT_SHA256
    ):
        raise fx.ContractError("phase-commitment-root")
    parsed_worlds = tuple(fx.world_from_mapping(item) for item in raw_worlds)
    forbidden_payload_rows = sum(
        item.family != family or item.split != split for item in parsed_worlds
    )
    if forbidden_payload_rows:
        raise fx.ContractError("phase-source-join")
    expected_keys = tuple(
        world for world in range(16) if sum(((world >> bit) & 1) for bit in range(4)) % 2 == (0 if split == "development" else 1)
    )
    if tuple(item.world for item in parsed_worlds) != expected_keys:
        raise fx.ContractError("phase-source-order")
    forbidden_split = "heldout" if split == "development" else "development"
    probe_sentinel = _ExplodingSourceSentinel(forbidden_split)
    probe_source = _WorldSource(family, split, parsed_worlds, probe_sentinel)
    forbidden_key = next(world for world in range(16) if world not in expected_keys)
    try:
        probe_source.get(forbidden_key)
    except fx.ContractError as error:
        if str(error) != "forbidden-source-read":
            raise
    else:
        raise fx.ContractError("source-sentinel-not-reached")
    sentinel = _ExplodingSourceSentinel(forbidden_split)
    source = _WorldSource(family, split, parsed_worlds, sentinel)
    worlds = tuple(source.get(world) for world in expected_keys)

    primary_transcripts, oracle_transcripts, _ = _transcript_replay()
    primary_by_seed = {item.seed: item for item in primary_transcripts}
    oracle_by_seed = {item["seed"]: item for item in oracle_transcripts}
    roots = {
        "implementation_schema": _Root("ObservationRecord"),
        "oracle_schema": _Root("ObservationRecord"),
        "implementation_state": _Root("ChronologyRecord"),
        "oracle_state": _Root("ChronologyRecord"),
        "implementation_trajectory": _Root("TrajectoryRecord"),
        "oracle_trajectory": _Root("TrajectoryRecord"),
    }
    keyed: dict[tuple[int, int, str], dict[str, object]] = {}
    order_mismatches = 0
    observations = 0
    transitions = 0
    batches = 0
    trajectories = 0
    restart_events = 0
    feasible_observations = 0
    references_used = 0
    false_feasible_joins = 0
    nonunit_gaps = 0
    schema_valid_observations = 0
    replayed_transitions = 0
    replayed_receipts = 0
    reset_checks = 0
    incumbent_tie_checks = 0
    incumbent_state_checks = 0
    arm_evaluations = {arm: 0 for arm in fx.ARMS}
    arm_transcript_receipts = {arm: 0 for arm in fx.ARMS}
    for world_record, raw_world in zip(worlds, raw_worlds, strict=True):
        for seed in fx.SEEDS:
            for order in fx.ORDERS:
                arm_order = fx.ARMS if order == "forward" else tuple(reversed(fx.ARMS))
                for arm in arm_order:
                    oracle = _ReplayOracle(raw_world, oracle_by_seed[seed], roots)
                    evidence = fx.execute_trajectory(
                        world_record,
                        primary_by_seed[seed],
                        seed=seed,
                        order=order,
                        arm=arm,
                        oracle=oracle,
                    )
                    summary = oracle.summary
                    if not isinstance(summary, dict):
                        raise fx.ContractError("oracle-summary")
                    restart_events += int(summary["restart_events"])
                    feasible_observations += int(summary["feasible_observations"])
                    observations += int(summary["observations"])
                    transitions += int(summary["transitions"])
                    batches += int(summary["batches"])
                    trajectories += int(summary["trajectories"])
                    references_used += int(summary["reference_used"])
                    false_feasible_joins += int(summary["false_feasible_joins"])
                    nonunit_gaps += int(summary["nonunit_gap"])
                    schema_valid_observations += int(
                        summary["schema_valid_observations"]
                    )
                    replayed_transitions += int(summary["replayed_transitions"])
                    replayed_receipts += int(summary["replayed_receipts"])
                    reset_checks += int(summary["reset_checks"])
                    incumbent_tie_checks += int(summary["incumbent_tie_checks"])
                    incumbent_state_checks += int(summary["incumbent_state_checks"])
                    arm_evaluations[arm] += int(summary["observations"])
                    if summary["transcript_sha256"] != fx.TRANSCRIPT_HASHES[seed]:
                        raise fx.ContractError("trajectory-transcript-join")
                    arm_transcript_receipts[arm] += 1
                    key = (world_record.world, seed, arm)
                    if key in keyed:
                        earlier = keyed[key]
                        if (
                            earlier["order"] != "forward"
                            or summary["order"] != "reverse"
                            or earlier["gap"] != summary["gap"]
                            or earlier["normalized_twin_sha256"]
                            != summary["normalized_twin_sha256"]
                            or earlier["event_root_sha256"]
                            == summary["event_root_sha256"]
                        ):
                            order_mismatches += 1
                    else:
                        keyed[key] = summary

    if order_mismatches:
        raise fx.ContractError("order-twin-mismatch")
    aggregate_rows = []
    aggregate_root = _Root("WorldAggregateRecord")
    for world_record in worlds:
        for arm in fx.ARMS:
            gaps = [float(keyed[(world_record.world, seed, arm)]["gap"]) for seed in fx.SEEDS]
            row = {
                "family": family,
                "world": world_record.world,
                "arm": arm,
                "seed_gaps": gaps,
                "mean_gap": float(sum(gaps) / len(gaps)),
            }
            aggregate_rows.append(row)
            aggregate_root.update(row)
    input_root = _Root("PhaseInput")
    for raw_world in raw_worlds:
        input_root.update(raw_world)
    output_root = aggregate_root.hexdigest()
    receipt = {
        "family": family,
        "split": split,
        "world_keys": list(expected_keys),
        "attempted_reads": source.attempted_reads,
        "forbidden_reads": source.forbidden_reads,
        "forbidden_payload_rows": int(forbidden_payload_rows),
        "sentinel_connected": source.sentinel_connected,
        "input_root_sha256": input_root.hexdigest(),
        "output_root_sha256": output_root,
    }
    return {
        "family": family,
        "split": split,
        "world_aggregates": aggregate_rows,
        "phase_receipt": receipt,
        "observations": observations,
        "transitions": transitions,
        "batches": batches,
        "trajectories": trajectories,
        "feasible_observations": feasible_observations,
        "restart_events": restart_events,
        "references_used": references_used,
        "false_feasible_joins": false_feasible_joins,
        "nonunit_gaps": nonunit_gaps,
        "schema_valid_observations": schema_valid_observations,
        "replayed_transitions": replayed_transitions,
        "replayed_receipts": replayed_receipts,
        "reset_checks": reset_checks,
        "incumbent_tie_checks": incumbent_tie_checks,
        "incumbent_state_checks": incumbent_state_checks,
        "arm_evaluations": arm_evaluations,
        "arm_transcript_receipts": arm_transcript_receipts,
        "order_mismatches": order_mismatches,
        "roots": {name: root.hexdigest() for name, root in roots.items()},
        "source_sentinel_probe_rejected": probe_source.forbidden_reads == 1,
    }


def _parse_phase(value: object) -> dict[str, object]:
    expected = (
        "family",
        "split",
        "world_aggregates",
        "phase_receipt",
        "observations",
        "transitions",
        "batches",
        "trajectories",
        "feasible_observations",
        "restart_events",
        "references_used",
        "false_feasible_joins",
        "nonunit_gaps",
        "schema_valid_observations",
        "replayed_transitions",
        "replayed_receipts",
        "reset_checks",
        "incumbent_tie_checks",
        "incumbent_state_checks",
        "arm_evaluations",
        "arm_transcript_receipts",
        "order_mismatches",
        "roots",
        "source_sentinel_probe_rejected",
    )
    if not isinstance(value, dict) or tuple(value) != expected:
        raise fx.ContractError("phase-result-schema")
    if value["family"] not in fx.FAMILIES or value["split"] not in fx.SPLITS:
        raise fx.ContractError("phase-result-identity")
    for name in (
        "observations",
        "transitions",
        "batches",
        "trajectories",
        "feasible_observations",
        "restart_events",
        "references_used",
        "false_feasible_joins",
        "nonunit_gaps",
        "schema_valid_observations",
        "replayed_transitions",
        "replayed_receipts",
        "reset_checks",
        "incumbent_tie_checks",
        "incumbent_state_checks",
        "order_mismatches",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise fx.ContractError("phase-result-count")
    if value["source_sentinel_probe_rejected"] is not True:
        raise fx.ContractError("source-sentinel-proof")
    if (
        not isinstance(value["world_aggregates"], list)
        or len(value["world_aggregates"]) != 40
        or not isinstance(value["phase_receipt"], dict)
        or tuple(value["phase_receipt"])
        != (
            "family",
            "split",
            "world_keys",
            "attempted_reads",
            "forbidden_reads",
            "forbidden_payload_rows",
            "sentinel_connected",
            "input_root_sha256",
            "output_root_sha256",
        )
    ):
        raise fx.ContractError("phase-result-evidence")
    for name in ("arm_evaluations", "arm_transcript_receipts"):
        arm_counts = value[name]
        if (
            not isinstance(arm_counts, dict)
            or tuple(arm_counts) != fx.ARMS
            or any(type(item) is not int or item < 0 for item in arm_counts.values())
        ):
            raise fx.ContractError("phase-result-arm-counts")
    roots = value["roots"]
    if (
        not isinstance(roots, dict)
        or tuple(roots)
        != (
            "implementation_schema",
            "oracle_schema",
            "implementation_state",
            "oracle_state",
            "implementation_trajectory",
            "oracle_trajectory",
        )
        or any(
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in roots.values()
        )
    ):
        raise fx.ContractError("phase-result-roots")
    return value


def _intervention_whitelist_rows():
    expected = {
        "protected_raw_progress": ("raw", "unused", True),
        "constraint_lexicographic_progress": ("lex", "canonical", True),
        "shuffled_progress_control": ("lex", "cyclic-donor", True),
        "ablated_progress_control": ("lex", "ablated", True),
        "no_restart_comparator": ("raw", "unused", False),
    }
    allowed_packet_fields = (
        "u",
        "learning_rates",
        "m",
        "v",
        "ages",
        "stalls",
        "progress",
        "losses",
        "gradients",
        "decisions",
        "incumbent_present",
        "incumbent_center",
        "restart_round",
        "budget_fraction",
    )
    if tuple(fx.OptimizerPacket.__dataclass_fields__) != allowed_packet_fields:
        raise fx.ContractError("optimizer-packet-whitelist")

    # The production adapter may branch on an arm only through the four-field
    # InterventionConfig.  Direct `self.arm` comparisons would create an
    # undeclared static difference and fail this AST audit.
    adapter_source = inspect.getsource(fx.OptimizerAdapter)
    tree = ast.parse(adapter_source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "arm"
            and not isinstance(getattr(node, "ctx", None), ast.Store)
        ):
            raise fx.ContractError("optimizer-static-difference")
    prepare = next(
        node
        for node in tree.body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )
    packet_reads = {
        node.attr
        for node in ast.walk(prepare)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "packet"
        and isinstance(node.ctx, ast.Load)
    }
    if packet_reads & {"incumbent_present", "incumbent_center", "restart_round"}:
        raise fx.ContractError("pre-receipt-center-consumption")

    implementation_rows = []
    oracle_rows = []
    for arm in fx.ARMS:
        config = fx.intervention_config(arm)
        implementation = {
            "arm_id": config.arm_id,
            "progress_comparator": config.progress_comparator,
            "tuple_adapter": config.tuple_adapter,
            "restart_enabled": config.restart_enabled,
        }
        comparator, tuple_adapter, restart_enabled = expected[arm]
        oracle = {
            "arm_id": arm,
            "progress_comparator": comparator,
            "tuple_adapter": tuple_adapter,
            "restart_enabled": restart_enabled,
        }
        if implementation != oracle:
            raise fx.ContractError("optimizer-difference-whitelist")
        implementation_rows.append(implementation)
        oracle_rows.append(oracle)
    return implementation_rows, oracle_rows


def _sentinel_and_capabilities():
    consumed_records = fx.progress_sentinel_records()
    outcomes = (
        tuple(item.canonical_decision for item in consumed_records),
        tuple(item.donor_decision for item in consumed_records),
        tuple(item.ablated_decision for item in consumed_records),
    )
    expected_outcomes = ((True, True), (False, False), (True, False))
    if outcomes != expected_outcomes:
        raise fx.ContractError("sentinel-result")
    stored = {
        "mode": "lex",
        "observed": True,
        "feasible": False,
        "first": 1.0,
        "second": 5.0,
    }
    canonical = (
        {"is_feasible": False, "penalty": 0.9, "sensitivity": 6.0},
        {"is_feasible": False, "penalty": 0.8, "sensitivity": 7.0},
    )
    donor = (
        {"is_feasible": False, "penalty": 1.1, "sensitivity": 0.0},
        {"is_feasible": False, "penalty": 1.2, "sensitivity": 0.0},
    )
    ablated = (
        {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0},
        {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0},
    )
    implementation_root = _Root("InterventionRecord")
    oracle_root = _Root("InterventionRecord")
    for ordinal in range(2):
        implementation_row = consumed_records[ordinal].to_dict()
        oracle_row = {
            "sentinel_id": "progress-consumer-v1",
            "ordinal": ordinal,
            "member": 0,
            "stored": stored,
            "canonical_tuple": canonical[ordinal],
            "donor_member": 1,
            "donor_tuple": donor[ordinal],
            "ablated_tuple": ablated[ordinal],
            "canonical_decision": expected_outcomes[0][ordinal],
            "donor_decision": expected_outcomes[1][ordinal],
            "ablated_decision": expected_outcomes[2][ordinal],
        }
        if implementation_row != oracle_row:
            raise fx.ContractError("sentinel-record-replay")
        implementation_root.update(implementation_row)
        oracle_root.update(oracle_row)
    receipts = fx.capability_attack_receipts()
    if len(receipts) != 22:
        raise fx.ContractError("capability-count")
    for receipt, attack_id in zip(receipts, fx.CAPABILITY_ATTACKS, strict=True):
        row = receipt.to_dict()
        if (
            row["attack_id"] != attack_id
            or row["injection_path"] != fx.CAPABILITY_PATHS[attack_id]
            or row["rejection_code"] != "capability-denied"
            or row["consumer_reached"] is not True
            or row["state_mutations"] != 0
        ):
            raise fx.ContractError("capability-receipt")
        implementation_root.update(row)
        oracle_root.update(
            {
                "attack_id": attack_id,
                "injection_path": fx.CAPABILITY_PATHS[attack_id],
                "rejection_code": "capability-denied",
                "consumer_reached": True,
                "state_mutations": 0,
            }
        )
    implementation_rows, oracle_rows = _intervention_whitelist_rows()
    if implementation_rows != oracle_rows:
        raise fx.ContractError("optimizer-difference-whitelist")
    return outcomes, receipts, implementation_root.hexdigest(), oracle_root.hexdigest()


def _minimal_envelope():
    # Reconstruct the valid attack envelope without any production fixture
    # constructor, evaluator, progress, optimizer-update, or state-hash helper.
    # The worker-oracle paths used here are independent; fixture functions
    # below are only the parser/consumer boundary under attack.
    world = _oracle_worlds()[0]
    transcript = _oracle_transcript(2026083001)
    u = np.zeros((8, 3), dtype=np.float64)
    u[1:] = transcript["suffix"]
    evaluations = [_evaluate(world, u[member]) for member in range(8)]
    candidates = [
        (float(item["sensitivity"]), member)
        for member, item in enumerate(evaluations)
        if item["canonical_is_feasible"]
    ]
    if not candidates:
        raise fx.ContractError("attack-envelope-incumbent")
    incumbent_sensitivity, incumbent_member = min(candidates)
    incumbent_center = [float(value) for value in u[incumbent_member]]
    observations = []
    for member, evaluation in enumerate(evaluations):
        observations.append(
            {
                "family": "canonical",
                "world": 0,
                "seed": 2026083001,
                "order": "forward",
                "arm": "constraint_lexicographic_progress",
                "batch": 0,
                "member": member,
                **evaluation,
                "decision_source": "canonical",
                "decision_donor_member": -1,
                "decision_is_feasible": evaluation["canonical_is_feasible"],
                "decision_penalty": evaluation["penalty"],
                "decision_sensitivity": evaluation["sensitivity"],
            }
        )
    progress_before = _unobserved("lex")
    decision = {
        "is_feasible": evaluations[0]["canonical_is_feasible"],
        "penalty": evaluations[0]["penalty"],
        "sensitivity": evaluations[0]["sensitivity"],
    }
    improved, progress_after = _progress(
        "lex", progress_before, float(evaluations[0]["loss"]), decision
    )
    if not improved:
        raise fx.ContractError("attack-envelope-progress")
    gradients = np.asarray(
        [item["gradient"] for item in evaluations], dtype=np.float64
    )
    norms = np.sqrt(np.sum(gradients * gradients, axis=1))
    clipped = gradients * np.minimum(1.0, 1.0 / (norms + 1.0e-12))[:, None]
    next_m = 0.1 * clipped
    next_v = 0.001 * clipped * clipped
    next_ages = np.ones(8, dtype=np.int64)
    mhat = next_m / (1.0 - np.power(0.9, next_ages))[:, None]
    vhat = next_v / (1.0 - np.power(0.999, next_ages))[:, None]
    rates = np.geomspace(0.03, 0.15, 8, dtype=np.float64)
    next_u = u - rates[:, None] * mhat / (np.sqrt(vhat) + 1.0e-8)
    state_before = _state_hash(
        {
            "u": [float(value) for value in u[0]],
            "m": [0.0, 0.0, 0.0],
            "v": [0.0, 0.0, 0.0],
            "age": 0,
            "stall": 0,
            "progress": progress_before,
            "incumbent_present": True,
            "incumbent_sensitivity": incumbent_sensitivity,
            "incumbent_source_batch": 0,
            "incumbent_source_member": incumbent_member,
            "incumbent_center": incumbent_center,
            "restart_round": 0,
        }
    )
    state_after = _state_hash(
        {
            "u": [float(value) for value in next_u[0]],
            "m": [float(value) for value in next_m[0]],
            "v": [float(value) for value in next_v[0]],
            "age": 1,
            "stall": 0,
            "progress": progress_after,
            "incumbent_present": True,
            "incumbent_sensitivity": incumbent_sensitivity,
            "incumbent_source_batch": 0,
            "incumbent_source_member": incumbent_member,
            "incumbent_center": incumbent_center,
            "restart_round": 0,
        }
    )
    transition = {
        "family": "canonical",
        "world": 0,
        "seed": 2026083001,
        "order": "forward",
        "arm": "constraint_lexicographic_progress",
        "batch": 0,
        "member": 0,
        "progress_before": progress_before,
        "progress_after": progress_after,
        "stall_before": 0,
        "stall_after": 0,
        "adam_age_before": 0,
        "adam_age_after": 1,
        "update_applied": True,
        "restart_triggered": False,
        "restart_kind": "none",
        "restart_round": -1,
        "center_source": "none",
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
    }
    for observation in observations:
        fx.validate_observation_mapping(observation)
    fx.validate_batch_identity(
        observations,
        family="canonical",
        world=0,
        seed=fx.SEEDS[0],
        order="forward",
        arm="constraint_lexicographic_progress",
        batch=0,
    )
    fx.validate_transition_mapping(
        transition, expected_seed=fx.SEEDS[0], expected_order="forward"
    )
    return observations, transition


def _validate_result_top(value: Mapping[str, object]) -> None:
    if tuple(value) != (
        "study_id",
        "plan_revision",
        "study_revision",
        "contract_sha256",
        "transcript_root_sha256",
        "status",
        "action",
        "world_aggregates",
        "cases",
    ):
        raise fx.ContractError("result-schema")


def _validate_transcript_hash(seed: int, value: str) -> None:
    if seed not in fx.SEEDS or value != fx.TRANSCRIPT_HASHES[seed]:
        raise fx.ContractError("transcript-hash")


ATTACK_MATRIX = (
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


class _AttackHarness:
    """Instrumented real-consumer boundary for one reconstructed attack."""

    def __init__(self) -> None:
        observations, transition = _minimal_envelope()
        self._observations = copy.deepcopy(observations)
        self._transition = copy.deepcopy(transition)
        self._consumer_entries = 0
        self._candidate_kind: str | None = None
        self._candidate = None

    @property
    def consumer_entries(self) -> int:
        return self._consumer_entries

    def state_digest(self) -> str:
        digest = hashlib.sha256(b"L2D-constraint-progress-v1/attack-state\0")

        def update(value) -> None:
            if value is None:
                digest.update(b"N")
            elif type(value) is bool:
                digest.update(b"B1" if value else b"B0")
            elif type(value) is int:
                encoded = str(value).encode("ascii")
                digest.update(b"I" + len(encoded).to_bytes(4, "little") + encoded)
            elif type(value) is float:
                digest.update(b"F" + struct.pack("<d", value))
            elif type(value) is str:
                encoded = value.encode("utf-8")
                digest.update(b"S" + len(encoded).to_bytes(4, "little") + encoded)
            elif isinstance(value, list):
                digest.update(b"L" + len(value).to_bytes(4, "little"))
                for item in value:
                    update(item)
            elif isinstance(value, dict):
                digest.update(b"D" + len(value).to_bytes(4, "little"))
                for key, item in value.items():
                    update(key)
                    update(item)
            else:
                raise fx.ContractError("attack-state-type")

        update(
            {
                "observations": self._observations,
                "transition": self._transition,
                "candidate_kind": self._candidate_kind,
                "candidate": self._candidate,
            }
        )
        return digest.hexdigest()

    def observation(self) -> dict[str, object]:
        self._candidate_kind = "observation"
        self._candidate = copy.deepcopy(self._observations[0])
        return self._candidate

    def observations(self) -> list[dict[str, object]]:
        self._candidate_kind = "batch"
        self._candidate = copy.deepcopy(self._observations)
        return self._candidate

    def transition(self) -> dict[str, object]:
        self._candidate_kind = "transition"
        self._candidate = copy.deepcopy(self._transition)
        return self._candidate

    def result(self, value: dict[str, object]) -> None:
        self._candidate_kind = "result"
        self._candidate = copy.deepcopy(value)

    def transcript_hash(self, seed: int, value: str) -> None:
        self._candidate_kind = "transcript"
        self._candidate = {"seed": seed, "value": value}

    def _enter(self) -> None:
        self._consumer_entries += 1

    def _validate_envelope(self) -> None:
        for observation in self._observations:
            fx.validate_observation_mapping(observation)
        fx.validate_batch_identity(
            self._observations,
            family="canonical",
            world=0,
            seed=fx.SEEDS[0],
            order="forward",
            arm="constraint_lexicographic_progress",
            batch=0,
        )
        fx.validate_transition_mapping(
            self._transition,
            expected_seed=fx.SEEDS[0],
            expected_order="forward",
        )

    def _selected(self, kind: str):
        if self._candidate_kind != kind:
            raise fx.ContractError("attack-candidate-kind")
        return self._candidate

    def consume_observation(self) -> None:
        self._enter()
        fx.validate_observation_mapping(self._selected("observation"))

    def consume_batch(self) -> None:
        self._enter()
        fx.validate_batch_identity(
            self._selected("batch"),
            family="canonical",
            world=0,
            seed=fx.SEEDS[0],
            order="forward",
            arm="constraint_lexicographic_progress",
            batch=0,
        )

    def consume_transition(self) -> None:
        self._enter()
        fx.validate_transition_mapping(
            self._selected("transition"),
            expected_seed=fx.SEEDS[0],
            expected_order="forward",
        )

    def consume_result(self) -> None:
        self._enter()
        self._validate_envelope()
        _validate_result_top(self._selected("result"))

    def consume_transcript_hash(self) -> None:
        self._enter()
        self._validate_envelope()
        candidate = self._selected("transcript")
        if not isinstance(candidate, dict):
            raise fx.ContractError("attack-candidate-type")
        _validate_transcript_hash(candidate["seed"], candidate["value"])


def _malformed_attacks():
    implementation_root = _Root("AttackReceipt")
    oracle_root = _Root("AttackReceipt")
    receipts = []
    if tuple(fx.MALFORMED_ATTACKS) != ATTACK_MATRIX:
        raise fx.ContractError("attack-contract-drift")
    for attack_id, injection_path, expected_code in ATTACK_MATRIX:
        harness = _AttackHarness()
        if attack_id in {
            "nan-loss",
            "gradient-dtype",
            "gradient-shape",
            "feasible-type",
            "negative-penalty",
            "wrong-arm",
        }:
            observation = harness.observation()
            if attack_id == "nan-loss":
                observation["loss"] = float("nan")
            elif attack_id == "gradient-dtype":
                observation["gradient"] = ["0", "0", "0"]
            elif attack_id == "gradient-shape":
                observation["gradient"] = [0.0, 0.0]
            elif attack_id == "feasible-type":
                observation["canonical_is_feasible"] = 1
            elif attack_id == "negative-penalty":
                observation["decision_penalty"] = -1.0e-9
            else:
                observation["arm"] = "unknown-arm"
            consumer = harness.consume_observation
        elif attack_id in {"duplicate-observation", "missing-member"}:
            observations = harness.observations()
            if attack_id == "duplicate-observation":
                observations.insert(1, dict(observations[0]))
            else:
                observations.pop()
            consumer = harness.consume_batch
        elif attack_id in {"cross-seed", "cross-order"}:
            transition = harness.transition()
            if attack_id == "cross-seed":
                transition["seed"] = fx.SEEDS[1]
            else:
                transition["order"] = "reverse"
            consumer = harness.consume_transition
        elif attack_id == "extra-result-field":
            harness.result(
                {
                    "study_id": fx.STUDY_ID,
                    "plan_revision": fx.PLAN_REVISION,
                    "study_revision": "0" * 40,
                    "contract_sha256": "0" * 64,
                    "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                    "status": "passed",
                    "action": "x",
                    "world_aggregates": [],
                    "cases": [],
                    "unexpected": True,
                }
            )
            consumer = harness.consume_result
        else:
            original = fx.TRANSCRIPT_HASHES[fx.SEEDS[0]]
            changed = ("0" if original[0] != "0" else "1") + original[1:]
            harness.transcript_hash(fx.SEEDS[0], changed)
            consumer = harness.consume_transcript_hash
        state_before = harness.state_digest()
        try:
            consumer()
        except fx.ContractError as error:
            observed_code = str(error)
            if observed_code != expected_code:
                raise
        else:
            raise fx.ContractError("malformed-attack-not-rejected")
        state_after = harness.state_digest()
        implementation_receipt = {
            "attack_id": attack_id,
            "injection_path": injection_path,
            "rejection_code": observed_code,
            "consumer_reached": harness.consumer_entries == 1,
            "state_mutations": int(state_before != state_after),
        }
        oracle_receipt = {
            "attack_id": attack_id,
            "injection_path": injection_path,
            "rejection_code": expected_code,
            "consumer_reached": True,
            "state_mutations": 0,
        }
        if implementation_receipt != oracle_receipt:
            raise fx.ContractError("attack-receipt-replay")
        receipts.append(implementation_receipt)
        implementation_root.update(implementation_receipt)
        oracle_root.update(oracle_receipt)
    return receipts, implementation_root.hexdigest(), oracle_root.hexdigest()


def _case(case_id: str, passed: bool, metrics: dict[str, object]) -> dict[str, object]:
    return {"case_id": case_id, "passed": bool(passed), "metrics": metrics}


def _rows_for(rows, family: str, arm: str, split: str):
    result = []
    for row in rows:
        world = int(row["world"])
        parity = sum(((world >> bit) & 1) for bit in range(4)) % 2
        expected = "development" if parity == 0 else "heldout"
        if row["family"] == family and row["arm"] == arm and expected == split:
            result.append(row)
    return result


def _comparison(rows, family: str, treatment: str, baseline: str):
    treatment_rows = _rows_for(rows, family, treatment, "heldout")
    baseline_rows = _rows_for(rows, family, baseline, "heldout")
    if len(treatment_rows) != 8 or len(baseline_rows) != 8:
        raise fx.ContractError("aggregate-subset")
    treatment_by_world = {int(row["world"]): float(row["mean_gap"]) for row in treatment_rows}
    baseline_by_world = {int(row["world"]): float(row["mean_gap"]) for row in baseline_rows}
    differences = [treatment_by_world[key] - baseline_by_world[key] for key in sorted(treatment_by_world)]
    wins = sum(value < -1.0e-12 for value in differences)
    ties = sum(abs(value) <= 1.0e-12 for value in differences)
    losses = 8 - wins - ties
    treatment_mean = float(sum(treatment_by_world.values()) / 8)
    baseline_mean = float(sum(baseline_by_world.values()) / 8)
    return treatment_mean, baseline_mean, differences, wins, ties, losses


def run_projection(run_child) -> dict[str, object]:
    _runtime_identity()
    implementation_worlds, implementation_family_root, oracle_family_root = _world_replay()
    primary_transcripts, _oracle_transcripts, transcript_root = _transcript_replay()
    outcomes, capabilities, implementation_intervention_root, oracle_intervention_root = _sentinel_and_capabilities()
    attacks, implementation_attack_root, oracle_attack_root = _malformed_attacks()

    phase_results = []
    phase_child_receipts = []
    surviving_children = 0
    for family in fx.FAMILIES:
        for split in fx.SPLITS:
            worlds = [
                item
                for item in implementation_worlds
                if item["family"] == family and item["split"] == split
            ]
            stdout, survivor, child_receipt = run_child(
                "constraint-aware-progress-toy-v2-phase",
                {
                    "family": family,
                    "split": split,
                    "contract_sha256": _required_environment(
                        "L2D_CONTRACT_SHA256", 64
                    ),
                    "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                    "worlds": worlds,
                },
            )
            surviving_children += survivor
            phase_child_receipts.append(child_receipt)
            result = _parse_phase(_loads_json(stdout))
            phase_results.append(result)

    world_aggregates = [
        row for phase in phase_results for row in phase["world_aggregates"]
    ]
    world_aggregates.sort(
        key=lambda row: (
            fx.FAMILIES.index(row["family"]),
            int(row["world"]),
            fx.ARMS.index(row["arm"]),
        )
    )
    if len(world_aggregates) != 240:
        raise fx.ContractError("aggregate-cardinality")
    schema_impl = _root_of_roots(
        "SchemaRoots",
        [phase["roots"]["implementation_schema"] for phase in phase_results],
    )
    schema_oracle = _root_of_roots(
        "SchemaRoots",
        [phase["roots"]["oracle_schema"] for phase in phase_results],
    )
    state_impl = _root_of_roots(
        "StateRoots",
        [
            root
            for phase in phase_results
            for root in (
                phase["roots"]["implementation_state"],
                phase["roots"]["implementation_trajectory"],
            )
        ],
    )
    state_oracle = _root_of_roots(
        "StateRoots",
        [
            root
            for phase in phase_results
            for root in (
                phase["roots"]["oracle_state"],
                phase["roots"]["oracle_trajectory"],
            )
        ],
    )
    source_impl_root = _Root("PhaseReceipt")
    source_oracle_root = _Root("PhaseReceipt")
    oracle_worlds = _oracle_worlds()
    for phase in phase_results:
        implementation_receipt = phase["phase_receipt"]
        expected_worlds = [
            item
            for item in oracle_worlds
            if item["family"] == phase["family"] and item["split"] == phase["split"]
        ]
        expected_input = _Root("PhaseInput")
        for item in expected_worlds:
            expected_input.update(item)
        expected_output = _Root("WorldAggregateRecord")
        for item in phase["world_aggregates"]:
            expected_output.update(item)
        oracle_receipt = {
            "family": phase["family"],
            "split": phase["split"],
            "world_keys": [item["world"] for item in expected_worlds],
            "attempted_reads": len(expected_worlds),
            "forbidden_reads": 0,
            "forbidden_payload_rows": 0,
            "sentinel_connected": True,
            "input_root_sha256": expected_input.hexdigest(),
            "output_root_sha256": expected_output.hexdigest(),
        }
        if implementation_receipt != oracle_receipt:
            raise fx.ContractError("phase-receipt-replay")
        source_impl_root.update(implementation_receipt)
        source_oracle_root.update(oracle_receipt)

    primary_mean, baseline_mean, primary_diff, wins, ties, losses = _comparison(
        world_aggregates,
        "canonical",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    primary_improvement = baseline_mean - primary_mean
    primary_mean_gate = primary_improvement >= 0.05
    primary_win_gate = wins >= 6
    primary_harm = max(primary_diff)
    primary_harm_gate = primary_harm <= 0.15

    no_restart_mean, _treatment_mean_again, no_restart_diff, *_ = _comparison(
        world_aggregates,
        "canonical",
        "no_restart_comparator",
        "constraint_lexicographic_progress",
    )
    comparator_improvement = no_restart_mean - primary_mean

    shuffled_mean, shuffled_baseline, shuffled_diff, shuffled_wins, *_ = _comparison(
        world_aggregates,
        "canonical",
        "shuffled_progress_control",
        "protected_raw_progress",
    )
    shuffled_improvement = shuffled_baseline - shuffled_mean
    shuffled_mean_gate = shuffled_improvement >= 0.05
    shuffled_win_gate = shuffled_wins >= 6
    shuffled_harm = max(shuffled_diff)
    shuffled_harm_gate = shuffled_harm <= 0.15
    shuffled_recovered = shuffled_mean_gate and shuffled_win_gate and shuffled_harm_gate

    ablated_mean, ablated_baseline, ablated_diff, ablated_wins, *_ = _comparison(
        world_aggregates,
        "canonical",
        "ablated_progress_control",
        "protected_raw_progress",
    )
    ablated_improvement = ablated_baseline - ablated_mean
    ablated_mean_gate = ablated_improvement >= 0.05
    ablated_win_gate = ablated_wins >= 6
    ablated_harm = max(ablated_diff)
    ablated_harm_gate = ablated_harm <= 0.15
    ablated_recovered = ablated_mean_gate and ablated_win_gate and ablated_harm_gate

    aligned_treatment, aligned_baseline, aligned_diff, *_ = _comparison(
        world_aggregates,
        "aligned",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    aligned_abs = abs(aligned_treatment - aligned_baseline)
    aligned_harm = max(aligned_diff)

    observations = sum(int(item["observations"]) for item in phase_results)
    transitions = sum(int(item["transitions"]) for item in phase_results)
    batches = sum(int(item["batches"]) for item in phase_results)
    trajectories = sum(int(item["trajectories"]) for item in phase_results)
    schema_valid_observations = sum(
        int(item["schema_valid_observations"]) for item in phase_results
    )
    replayed_transitions = sum(
        int(item["replayed_transitions"]) for item in phase_results
    )
    replayed_receipts = sum(
        int(item["replayed_receipts"]) for item in phase_results
    )
    reset_checks = sum(int(item["reset_checks"]) for item in phase_results)
    incumbent_tie_checks = sum(
        int(item["incumbent_tie_checks"]) for item in phase_results
    )
    incumbent_state_checks = sum(
        int(item["incumbent_state_checks"]) for item in phase_results
    )
    restart_events = sum(int(item["restart_events"]) for item in phase_results)
    impossible_phases = [item for item in phase_results if item["family"] == "impossible"]
    impossible_feasible = sum(int(item["feasible_observations"]) for item in impossible_phases)
    development_receipts = [
        item["phase_receipt"]
        for item in phase_results
        if item["split"] == "development"
    ]
    heldout_receipts = [
        item["phase_receipt"] for item in phase_results if item["split"] == "heldout"
    ]
    forbidden_reads = sum(
        int(item["phase_receipt"]["forbidden_reads"]) for item in phase_results
    )
    heldout_source_in_development = sum(
        int(item["phase_receipt"]["forbidden_payload_rows"])
        + sum(
            sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 1
            for row in item["world_aggregates"]
        )
        for item in phase_results
        if item["split"] == "development"
    )
    development_outputs_in_heldout = sum(
        sum(
            sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 0
            for row in item["world_aggregates"]
        )
        for item in phase_results
        if item["split"] == "heldout"
    )

    oracle_worlds_by_key = {
        (item["family"], item["world"]): item for item in oracle_worlds
    }
    formula_fields = (
        "family",
        "world",
        "bits",
        "split",
        "a",
        "b",
        "k",
        "t",
        "c",
        "threshold",
    )
    reference_fields = ("reference_x0", "reference_sensitivity", "denominator")
    formula_mismatches = sum(
        any(item[name] != oracle_worlds_by_key[(item["family"], item["world"])][name] for name in formula_fields)
        for item in implementation_worlds
    )
    reference_mismatches = sum(
        any(item[name] != oracle_worlds_by_key[(item["family"], item["world"])][name] for name in reference_fields)
        for item in implementation_worlds
    )
    duplicate_world_keys = len(implementation_worlds) - len(
        {(item["family"], item["world"]) for item in implementation_worlds}
    )
    constrained_references = sum(
        item["reference_x0"] is not None for item in implementation_worlds
    )
    reference_exclusions = sum(
        item["family"] == "impossible"
        and all(item[name] is None for name in reference_fields)
        for item in implementation_worlds
    )
    nonpositive_denominators = sum(
        item["denominator"] is not None and item["denominator"] <= 0.0
        for item in implementation_worlds
    )
    development_counts = [
        sum(item["family"] == family and item["split"] == "development" for item in implementation_worlds)
        for family in fx.FAMILIES
    ]
    heldout_counts = [
        sum(item["family"] == family and item["split"] == "heldout" for item in implementation_worlds)
        for family in fx.FAMILIES
    ]
    arm_evaluations = {
        arm: sum(int(item["arm_evaluations"][arm]) for item in phase_results)
        for arm in fx.ARMS
    }
    arm_trajectory_counts = {
        arm: sum(int(item["arm_transcript_receipts"][arm]) for item in phase_results)
        for arm in fx.ARMS
    }
    evaluations_per_arm_trajectory = {
        arm: arm_evaluations[arm] // arm_trajectory_counts[arm]
        for arm in fx.ARMS
        if arm_trajectory_counts[arm] > 0
        and arm_evaluations[arm] % arm_trajectory_counts[arm] == 0
    }
    evaluation_parity = (
        len(evaluations_per_arm_trajectory) == len(fx.ARMS)
        and len(set(evaluations_per_arm_trajectory.values())) == 1
    )
    transcript_parity = len(set(arm_trajectory_counts.values())) == 1

    cases = [
        _case(
            "family_replay",
            implementation_family_root == oracle_family_root
            and len(implementation_worlds) == 48
            and constrained_references == 32
            and reference_exclusions == 16
            and development_counts == [8, 8, 8]
            and heldout_counts == [8, 8, 8]
            and formula_mismatches == 0
            and reference_mismatches == 0
            and nonpositive_denominators == 0
            and duplicate_world_keys == 0,
            {
                "world_records": len(implementation_worlds),
                "constrained_references": constrained_references,
                "reference_exclusions": reference_exclusions,
                "development_worlds_per_family": min(development_counts),
                "heldout_worlds_per_family": min(heldout_counts),
                "formula_mismatches": formula_mismatches,
                "reference_mismatches": reference_mismatches,
                "nonpositive_denominators": nonpositive_denominators,
                "duplicate_world_keys": duplicate_world_keys,
                "implementation_root_sha256": implementation_family_root,
                "oracle_root_sha256": oracle_family_root,
                "roots_equal": implementation_family_root == oracle_family_root,
            },
        ),
        _case(
            "transcript_commitment",
            transcript_root == fx.TRANSCRIPT_ROOT_SHA256
            and sum(int(item["order_mismatches"]) for item in phase_results) == 0
            and trajectories == 1920
            and observations == 983040
            and len(set(arm_trajectory_counts.values())) == 1,
            {
                "transcripts": len(primary_transcripts),
                "values_per_transcript": int(
                    primary_transcripts[0].suffix.size
                    + primary_transcripts[0].fresh.size
                    + primary_transcripts[0].perturb.size
                ),
                "transcript_values": sum(
                    int(item.suffix.size + item.fresh.size + item.perturb.size)
                    for item in primary_transcripts
                ),
                "trajectories": trajectories,
                "evaluations": observations,
                "unequal_arm_counts": len(set(arm_trajectory_counts.values())) - 1,
                "order_twin_mismatches": sum(int(item["order_mismatches"]) for item in phase_results),
                "committed_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                "observed_root_sha256": transcript_root,
                "roots_equal": transcript_root == fx.TRANSCRIPT_ROOT_SHA256,
            },
        ),
        _case(
            "typed_aux_and_intervention",
            schema_impl == schema_oracle
            and implementation_intervention_root == oracle_intervention_root
            and schema_valid_observations == observations
            and all(item.state_mutations == 0 for item in capabilities)
            and all(item.consumer_reached for item in capabilities),
            {
                "observations": observations,
                "schema_valid_observations": schema_valid_observations,
                "join_failures": observations - schema_valid_observations,
                "capability_attacks": len(capabilities),
                "capability_rejected": sum(
                    item.rejection_code == "capability-denied"
                    and item.consumer_reached
                    for item in capabilities
                ),
                "capability_state_mutations": sum(
                    item.state_mutations for item in capabilities
                ),
                "canonical_decisions": list(outcomes[0]),
                "donor_decisions": list(outcomes[1]),
                "ablated_decisions": list(outcomes[2]),
                "implementation_schema_root_sha256": schema_impl,
                "oracle_schema_root_sha256": schema_oracle,
                "implementation_intervention_root_sha256": implementation_intervention_root,
                "oracle_intervention_root_sha256": oracle_intervention_root,
                "roots_equal": schema_impl == schema_oracle
                and implementation_intervention_root == oracle_intervention_root,
            },
        ),
        _case(
            "chronology_replay",
            state_impl == state_oracle
            and replayed_transitions == transitions
            and replayed_receipts == batches
            and reset_checks == restart_events
            and incumbent_tie_checks == batches
            and incumbent_state_checks == batches
            and sum(int(item["order_mismatches"]) for item in phase_results) == 0,
            {
                "batches": batches,
                "batch_receipts": replayed_receipts,
                "transitions": transitions,
                "replay_mismatches": transitions - replayed_transitions,
                "order_mismatches": sum(
                    int(item["order_mismatches"]) for item in phase_results
                ),
                "reset_mismatches": restart_events - reset_checks,
                "incumbent_tie_mismatches": batches - incumbent_tie_checks,
                "incumbent_state_mismatches": batches - incumbent_state_checks,
                "restart_events": restart_events,
                "implementation_state_root_sha256": state_impl,
                "oracle_state_root_sha256": state_oracle,
                "roots_equal": state_impl == state_oracle,
            },
        ),
        _case(
            "development_and_source_isolation",
            source_impl_root.hexdigest() == source_oracle_root.hexdigest()
            and len(development_receipts) == 3
            and len(heldout_receipts) == 3
            and forbidden_reads == 0
            and heldout_source_in_development == 0
            and development_outputs_in_heldout == 0,
            {
                "development_aggregates": sum(
                    sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 0
                    for row in world_aggregates
                ),
                "development_receipts": len(development_receipts),
                "heldout_receipts": len(heldout_receipts),
                "forbidden_reads": forbidden_reads,
                "heldout_source_in_development": heldout_source_in_development,
                "development_outputs_in_heldout": development_outputs_in_heldout,
                "implementation_source_root_sha256": source_impl_root.hexdigest(),
                "oracle_source_root_sha256": source_oracle_root.hexdigest(),
                "roots_equal": source_impl_root.hexdigest() == source_oracle_root.hexdigest(),
            },
        ),
        _case(
            "heldout_primary",
            primary_mean_gate and primary_win_gate and primary_harm_gate,
            {
                "treatment_mean_gap": primary_mean,
                "baseline_mean_gap": baseline_mean,
                "mean_improvement": primary_improvement,
                "heldout_wins": wins,
                "heldout_ties": ties,
                "heldout_losses": losses,
                "maximum_signed_world_harm": primary_harm,
                "mean_gate": primary_mean_gate,
                "win_gate": primary_win_gate,
                "harm_gate": primary_harm_gate,
            },
        ),
        _case(
            "restart_comparators",
            comparator_improvement >= 0.05
            and evaluation_parity
            and transcript_parity,
            {
                "treatment_mean_gap": primary_mean,
                "no_restart_mean_gap": no_restart_mean,
                "mean_improvement": comparator_improvement,
                "minimum_arm_evaluations": min(evaluations_per_arm_trajectory.values()),
                "maximum_arm_evaluations": max(evaluations_per_arm_trajectory.values()),
                "evaluation_parity": evaluation_parity,
                "transcript_parity": transcript_parity,
                "comparator_gate": comparator_improvement >= 0.05
                and evaluation_parity
                and transcript_parity,
            },
        ),
        _case(
            "shuffled_signal_control",
            (not shuffled_recovered) and shuffled_improvement < 0.05,
            {
                "control_mean_gap": shuffled_mean,
                "baseline_mean_gap": shuffled_baseline,
                "mean_improvement": shuffled_improvement,
                "heldout_wins": shuffled_wins,
                "maximum_signed_world_harm": shuffled_harm,
                "substituted_mean_gate": shuffled_mean_gate,
                "substituted_win_gate": shuffled_win_gate,
                "substituted_harm_gate": shuffled_harm_gate,
                "positive_gate_recovered": shuffled_recovered,
            },
        ),
        _case(
            "ablated_signal_control",
            (not ablated_recovered) and ablated_improvement < 0.05,
            {
                "control_mean_gap": ablated_mean,
                "baseline_mean_gap": ablated_baseline,
                "mean_improvement": ablated_improvement,
                "heldout_wins": ablated_wins,
                "maximum_signed_world_harm": ablated_harm,
                "substituted_mean_gate": ablated_mean_gate,
                "substituted_win_gate": ablated_win_gate,
                "substituted_harm_gate": ablated_harm_gate,
                "positive_gate_recovered": ablated_recovered,
            },
        ),
        _case(
            "aligned_control",
            aligned_abs <= 0.03 and aligned_harm <= 0.10,
            {
                "treatment_mean_gap": aligned_treatment,
                "baseline_mean_gap": aligned_baseline,
                "absolute_mean_difference": aligned_abs,
                "maximum_signed_world_harm": aligned_harm,
                "trajectories": sum(
                    int(item["trajectories"])
                    for item in phase_results
                    if item["family"] == "aligned"
                ),
                "mean_gate": aligned_abs <= 0.03,
                "harm_gate": aligned_harm <= 0.10,
            },
        ),
        _case(
            "impossible_control",
            impossible_feasible == 0
            and sum(int(item["nonunit_gaps"]) for item in impossible_phases) == 0
            and sum(int(item["references_used"]) for item in impossible_phases) == 0
            and sum(int(item["false_feasible_joins"]) for item in impossible_phases) == 0,
            {
                "trajectories": sum(int(item["trajectories"]) for item in impossible_phases),
                "observations": sum(int(item["observations"]) for item in impossible_phases),
                "feasible_observations": impossible_feasible,
                "nonunit_gaps": sum(
                    int(item["nonunit_gaps"]) for item in impossible_phases
                ),
                "references_used": sum(
                    int(item["references_used"]) for item in impossible_phases
                ),
                "false_feasible_joins": sum(
                    int(item["false_feasible_joins"]) for item in impossible_phases
                ),
            },
        ),
    ]
    return {
        "projection_version": 1,
        "world_aggregates": world_aggregates,
        "cases": cases,
        "attacks": len(attacks),
        "attacks_rejected": sum(
            item["consumer_reached"] is True
            and item["rejection_code"] == expected[2]
            for item, expected in zip(attacks, ATTACK_MATRIX, strict=True)
        ),
        "attack_state_mutations": sum(
            int(item["state_mutations"]) for item in attacks
        ),
        "implementation_attack_root_sha256": implementation_attack_root,
        "oracle_attack_root_sha256": oracle_attack_root,
        "maximum_child_stdout_bytes": max(
            int(item["stdout_bytes"]) for item in phase_child_receipts
        ),
        "stderr_bytes": sum(
            int(item["stderr_bytes"]) for item in phase_child_receipts
        ),
        "surviving_children": surviving_children,
    }


def _parse_projection(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != (
        "projection_version",
        "world_aggregates",
        "cases",
        "attacks",
        "attacks_rejected",
        "attack_state_mutations",
        "implementation_attack_root_sha256",
        "oracle_attack_root_sha256",
        "maximum_child_stdout_bytes",
        "stderr_bytes",
        "surviving_children",
    ):
        raise fx.ContractError("projection-schema")
    if value["projection_version"] != 1:
        raise fx.ContractError("projection-version")
    if (
        not isinstance(value["world_aggregates"], list)
        or len(value["world_aggregates"]) != 240
        or not isinstance(value["cases"], list)
        or len(value["cases"]) != 11
    ):
        raise fx.ContractError("projection-cardinality")
    for name in (
        "attacks",
        "attacks_rejected",
        "attack_state_mutations",
        "maximum_child_stdout_bytes",
        "stderr_bytes",
        "surviving_children",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise fx.ContractError("projection-count")
    if value["maximum_child_stdout_bytes"] > MAX_PACKET_BYTES:
        raise fx.ContractError("projection-child-output-cap")
    for name in (
        "implementation_attack_root_sha256",
        "oracle_attack_root_sha256",
    ):
        item = value[name]
        if (
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise fx.ContractError("projection-root")
    return value


def _required_environment(name: str, length: int) -> str:
    value = os.environ.get(name, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise fx.ContractError("controller-contract-environment")
    return value


def run_full(run_child) -> dict[str, object]:
    _runtime_identity()
    plan_revision = _required_environment("L2D_PLAN_REVISION", 40)
    study_revision = _required_environment("L2D_STUDY_REVISION", 40)
    contract_sha256 = _required_environment("L2D_CONTRACT_SHA256", 64)
    if plan_revision != fx.PLAN_REVISION:
        raise fx.ContractError("plan-revision")
    raw_projections = []
    parsed = []
    projection_receipts = []
    surviving_children = 0
    for _ in range(2):
        stdout, survivor, child_receipt = run_child(
            "constraint-aware-progress-toy-v2-projection"
        )
        raw_projections.append(stdout)
        surviving_children += survivor
        projection_receipts.append(child_receipt)
        parsed.append(_parse_projection(_loads_json(stdout)))
    projections_equal = raw_projections[0] == raw_projections[1]
    if not projections_equal:
        raise fx.ContractError("projection-nondeterminism")
    projection = parsed[0]
    surviving_children += sum(
        int(item["surviving_children"]) for item in parsed
    )
    case12_metrics = {
        "launches": 2,
        "projections_equal": projections_equal,
        "maximum_stdout_bytes": max(
            [len(item) for item in raw_projections]
            + [int(item["maximum_child_stdout_bytes"]) for item in parsed]
        ),
        "stderr_bytes": sum(
            int(item["stderr_bytes"]) for item in projection_receipts
        )
        + sum(int(item["stderr_bytes"]) for item in parsed),
        "surviving_children": surviving_children,
        "attacks": projection["attacks"],
        "attacks_rejected": projection["attacks_rejected"],
        "attack_state_mutations": projection["attack_state_mutations"],
        "implementation_attack_root_sha256": projection[
            "implementation_attack_root_sha256"
        ],
        "oracle_attack_root_sha256": projection["oracle_attack_root_sha256"],
        "roots_equal": projection["implementation_attack_root_sha256"]
        == projection["oracle_attack_root_sha256"],
    }
    case12_passed = (
        projections_equal
        and case12_metrics["maximum_stdout_bytes"] <= MAX_PACKET_BYTES
        and case12_metrics["stderr_bytes"] == 0
        and surviving_children == 0
        and case12_metrics["attacks"] == 12
        and case12_metrics["attacks_rejected"] == 12
        and case12_metrics["attack_state_mutations"] == 0
        and case12_metrics["roots_equal"] is True
    )
    cases = [*projection["cases"], _case("process_and_sanitizer", case12_passed, case12_metrics)]
    passed = all(case["passed"] for case in cases)
    result = {
        "study_id": fx.STUDY_ID,
        "plan_revision": plan_revision,
        "study_revision": study_revision,
        "contract_sha256": contract_sha256,
        "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
        "status": "passed" if passed else "failed",
        "action": (
            "constraint_progress_mechanics_ready_for_candidate_audit"
            if passed
            else "park_round2_constraint_progress_research"
        ),
        "world_aggregates": projection["world_aggregates"],
        "cases": cases,
    }
    _validate_result_top(result)
    return result


def validate_runtime_identity() -> dict[str, str]:
    return _runtime_identity()


__all__ += [
    "run_full",
    "run_phase",
    "run_projection",
    "validate_runtime_identity",
]
