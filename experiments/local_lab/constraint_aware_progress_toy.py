"""Frozen synthetic constrained-progress fixture for Round-2 mechanics research.

This module contains only the primary deterministic environment and optimizer
path.  The independent replay oracle and all process orchestration live in the
dedicated worker module.  Importing this module never executes a study.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np


STUDY_ID = "constraint-aware-progress-toy-v1"
PLAN_REVISION = "02c3e2329b4906aa49d80ea0256a7db9774d491c"
PLAN_PATH = "research/2026-08-30-round1-feedback-and-round2-program.md"
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
