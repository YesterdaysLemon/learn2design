"""Frozen four-step synchronous-TD action-prefix mechanics fixture.

This module is deliberately self-contained.  It does not import either rejected
multi-step fixture, competition code, official data, or private evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import jax
import numpy as np


STUDY_ID = "multistep-td-action-prefix-v3"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
STRUCTURE_KIND = "none"
HORIZON = 4
EPISODES_PER_REGIME = 512
RANDOM_BASELINE_SEED = 314159265
OBSERVATION_FIELDS = (
    "phase",
    "signed_signal",
    "prefix_a0",
    "prefix_a1",
    "prefix_a2",
    "nuisance",
)
POLICY_INPUT_FIELDS = ("observation",)
EVENT_ORDER = (
    "observe0",
    "select0",
    "validate0",
    "resolve_successor0",
    "resolve_zero0",
    "append0",
    "observe1",
    "select1",
    "validate1",
    "resolve_successor1",
    "resolve_zero1",
    "append1",
    "observe2",
    "select2",
    "validate2",
    "resolve_successor2",
    "resolve_zero2",
    "append2",
    "observe3",
    "select3",
    "validate3",
    "resolve_terminal3",
    "append3",
    "close_episode",
)
TIMING_ATTACKS = (
    "successor_during_select",
    "reward_during_select",
    "invalid_action_successor_resolution",
    "invalid_action_reward_resolution",
    "successor_before_action_validation",
    "nonterminal_terminal_scalar",
    "missing_nonterminal_zero",
    "duplicate_nonterminal_feedback",
    "next_transition_before_pending_append",
    "terminal_scalar_before_phase_three_action",
    "terminal_scalar_after_extra_transition",
    "duplicate_terminal_scalar",
    "origin_resolution_before_terminal",
    "origin_bearing_policy_feedback",
    "reentrant_select_environment_step",
    "reentrant_update_callback",
    "duplicate_episode_close",
    "nonempty_pending_transition_at_split",
    "nonempty_origin_queue_at_split",
    "heldout_source_during_fit",
)
TRACE_ATTACK_CLASSES = (
    "missing_component",
    "duplicate_component",
    "unknown_key",
    "malformed_key",
    "cross_episode_key",
    "wrong_predecessor",
    "wrong_successor",
    "wrong_action",
    "wrong_source",
    "wrong_donor",
    "wrong_origin",
    "wrong_reward_slot",
    "wrong_close_link",
    "wrong_observation_layout",
    "wrong_action_type",
    "wrong_reward_type",
    "non_boolean_done",
    "wrong_canonical_reward",
    "wrong_update_reward",
    "independent_component_swap",
)
FORBIDDEN_POLICY_FIELDS = (
    "target",
    "preferred_action",
    "reward",
    "counterfactual_reward",
    "split",
    "regime",
    "episode",
    "block",
    "cell",
    "key",
    "action_code",
    "done",
    "successor",
    "future_observation",
    "transition",
    "trajectory",
    "log",
    "rng",
    "generator",
    "evaluator",
    "environment",
    "control_mode",
    "donor",
    "origin",
    "lazy_value",
)
THRESHOLDS = {
    "minimum_postfit_train_macro_return": 0.99,
    "minimum_postfit_validation_macro_return": 0.99,
    "minimum_postfit_test_macro_return": 0.99,
    "minimum_heldout_regime_return": 0.98,
    "minimum_validation_gain_constant": 0.30,
    "minimum_test_gain_constant": 0.30,
    "minimum_validation_gain_myopic": 0.30,
    "minimum_test_gain_myopic": 0.30,
    "minimum_validation_gain_no_bootstrap": 0.30,
    "minimum_test_gain_no_bootstrap": 0.30,
    "minimum_validation_gain_random": 0.30,
    "minimum_test_gain_random": 0.30,
    "maximum_transition_control_validation_macro_return": 0.55,
    "maximum_transition_control_test_macro_return": 0.55,
    "minimum_transition_control_test_gap": 0.40,
    "maximum_reward_origin_validation_macro_return": 0.55,
    "maximum_reward_origin_test_macro_return": 0.55,
    "minimum_reward_origin_test_gap": 0.40,
    "maximum_signal_attribution_macro_return": 0.55,
}


class ContractError(RuntimeError):
    """A frozen fixture contract was violated."""


@dataclass(frozen=True)
class Regime:
    split: str
    split_enum: np.uint8
    code: np.int32
    signal_scale: float
    nuisance_shift: float
    nuisance_scale: float


REGIMES = (
    Regime("train", np.uint8(0), np.int32(1009), 0.72, -1.10, 0.82),
    Regime("train", np.uint8(0), np.int32(1013), 0.91, -0.35, 1.07),
    Regime("train", np.uint8(0), np.int32(1019), 1.13, 0.35, 0.74),
    Regime("train", np.uint8(0), np.int32(1021), 1.34, 1.10, 1.19),
    Regime("validation", np.uint8(1), np.int32(2003), 0.63, -1.60, 0.67),
    Regime("validation", np.uint8(1), np.int32(2011), 1.43, 1.60, 1.31),
    Regime("test", np.uint8(2), np.int32(3001), 0.54, -2.10, 0.58),
    Regime("test", np.uint8(2), np.int32(3011), 1.52, 2.10, 1.42),
)
REGIME_COUNTS = {"train": 4, "validation": 2, "test": 2}
EPISODE_COUNTS = {name: count * EPISODES_PER_REGIME for name, count in REGIME_COUNTS.items()}


@dataclass(frozen=True)
class EpisodeKey:
    split_enum: np.uint8
    regime_code: np.int32
    episode: np.int16


@dataclass(frozen=True)
class ObservationKey:
    episode_key: EpisodeKey
    phase: np.int8
    prefix_code: np.int8


@dataclass(frozen=True)
class ActionKey:
    observation_key: ObservationKey
    action_ordinal: np.int8


@dataclass(frozen=True)
class TransitionKey:
    action_key: ActionKey
    transition_ordinal: np.int8


@dataclass(frozen=True)
class FeedbackKey:
    transition_key: TransitionKey
    feedback_ordinal: np.int8


@dataclass(frozen=True)
class EpisodeSpec:
    regime: Regime
    episode: np.int16
    block: np.int8
    cell: np.int8
    action_code: np.int8
    target_slot: np.int8
    target: np.int8
    sign: float
    magnitude: float
    nuisance: float
    key: EpisodeKey


@dataclass(frozen=True)
class ObservationRecord:
    key: ObservationKey
    value: np.ndarray
    value_sha256: str


@dataclass(frozen=True)
class ActionRecord:
    key: ActionKey
    observation_key: ObservationKey
    value: np.ndarray


@dataclass(frozen=True)
class TransitionRecord:
    key: TransitionKey
    action_key: ActionKey
    predecessor_key: ObservationKey
    predecessor_sha256: str
    successor_key: ObservationKey | None
    successor_sha256: str | None
    source_episode: EpisodeKey
    donor_episode: EpisodeKey | None
    donor_payload_sha256: str | None
    done: bool


@dataclass(frozen=True)
class FeedbackRecord:
    key: FeedbackKey
    transition_key: TransitionKey
    canonical_reward: np.ndarray
    update_reward: np.ndarray
    origin_episode: EpisodeKey
    done: bool


@dataclass(frozen=True)
class EpisodeCloseRecord:
    episode_key: EpisodeKey
    final_feedback_key: FeedbackKey
    record_count: np.int8


@dataclass(frozen=True)
class TraceBundle:
    observations: tuple[ObservationRecord, ...]
    actions: tuple[ActionRecord, ...]
    transitions: tuple[TransitionRecord, ...]
    feedback: tuple[FeedbackRecord, ...]
    closes: tuple[EpisodeCloseRecord, ...]


@dataclass(frozen=True)
class TDInputRow:
    predecessor: np.ndarray
    action: np.ndarray
    successor: np.ndarray | None
    update_reward: np.ndarray
    done: bool


@dataclass(frozen=True)
class _TraceGateToken:
    trace_sha256: str
    rows_sha256: str
    source_sha256: str
    mode: str
    token_sha256: str


@dataclass(frozen=True)
class _AuthenticatedRows(Sequence[TDInputRow]):
    rows: tuple[TDInputRow, ...]
    token: _TraceGateToken

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> TDInputRow | tuple[TDInputRow, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[TDInputRow]:
        return iter(self.rows)


@dataclass(frozen=True)
class ComparatorRow:
    predecessor: np.ndarray
    action: np.ndarray
    successor: np.ndarray | None
    update_reward: np.ndarray
    done: bool


@dataclass(frozen=True)
class ComparatorFeedbackStep:
    predecessor: np.ndarray
    action: np.ndarray
    successor: np.ndarray | None
    update_reward: np.ndarray
    done: bool


@dataclass(frozen=True)
class _ComparatorGateToken:
    feedback_sha256: str
    source_sha256: str
    trace_sha256: str
    token_sha256: str


@dataclass(frozen=True)
class ComparatorFeedbackBatch:
    rows: tuple[ComparatorFeedbackStep, ...]
    token: _ComparatorGateToken


@dataclass(frozen=True)
class _AuthenticatedComparatorRows(Sequence[ComparatorRow]):
    rows: tuple[ComparatorRow, ...]
    token: _ComparatorGateToken

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int | slice
    ) -> ComparatorRow | tuple[ComparatorRow, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[ComparatorRow]:
        return iter(self.rows)


@dataclass(frozen=True)
class PublicComparatorEpisode:
    initial_observation: np.ndarray


@dataclass(frozen=True)
class BootstrapRow:
    predecessor: np.ndarray
    action: np.ndarray
    successor: np.ndarray


@dataclass(frozen=True)
class SourceCounts:
    factory: int = 0
    iterator: int = 0
    materializer: int = 0
    close: int = 0
    post_close: int = 0


@dataclass(frozen=True)
class FitAudit:
    raw_terminal_reads: int
    aggregate_lookups_by_sweep: tuple[int, ...]
    writes_by_sweep: tuple[int, ...]
    positive_cells_by_sweep: tuple[int, ...]
    snapshots: tuple[dict[tuple[int, int, int, int], float], ...]


@dataclass(frozen=True)
class FitResult:
    table: dict[tuple[int, int, int, int], float]
    audit: FitAudit
    terminal_means: Mapping[tuple[int, int, int, int], np.float64]


@dataclass(frozen=True)
class DependencyProbeManifest:
    family_sha256: str
    canonical_trace_sha256: str
    canonical_rows_sha256: str
    selected_leaf: tuple[int, int, int, int]
    selected_records_sha256: str
    selected_record_count: int
    base_terminal_update_reward: float
    probe_terminal_update_reward: float
    allowed_difference: tuple[str, ...]
    base_projection_sha256: str
    probe_projection_sha256: str
    expected_changed_cells_by_sweep: tuple[
        tuple[tuple[int, int, int, int], ...], ...
    ]


@dataclass
class LazyAudit:
    installed: int = 1
    attempted: int = 0
    permitted: int = 0
    first_stage: str | None = None
    last_stage: str | None = None
    value_sha256: str | None = None


@dataclass(frozen=True)
class _FamilyGateToken:
    family_sha256: str
    gate_sha256: str


_ISSUED_FAMILY_GATES: set[str] = set()
_ISSUED_TRACE_GATES: set[str] = set()
_ISSUED_COMPARATOR_GATES: set[str] = set()
_USED_COMPARATOR_SCOPES: set[tuple[str, str]] = set()
_CONSTRUCTION_COUNTS = {"collector": 0, "learner": 0}


def _require_family_gate(gate: object) -> _FamilyGateToken:
    if (
        type(gate) is not _FamilyGateToken
        or gate.family_sha256 != EXPECTED_FAMILY_SHA256
        or len(gate.gate_sha256) != 64
        or gate.gate_sha256 not in _ISSUED_FAMILY_GATES
    ):
        raise ContractError("learner construction lacks the frozen family gate")
    return gate


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "bytes": value.tobytes(order="C").hex(),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "strides": list(value.strides),
            "writeable": bool(value.flags.writeable),
        }
    if isinstance(value, (EpisodeKey, ObservationKey, ActionKey, TransitionKey, FeedbackKey)):
        return _key_projection(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        _json_ready(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stream_update(digest: "hashlib._Hash", value: object) -> None:
    encoded = json.dumps(
        _json_ready(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest.update(encoded)
    digest.update(b"\n")


def _key_projection(key: object) -> dict[str, object]:
    if type(key) is EpisodeKey:
        return {
            "kind": "episode",
            "split_enum": {"dtype": "uint8", "value": int(key.split_enum)},
            "regime_code": {"dtype": "int32", "value": int(key.regime_code)},
            "episode": {"dtype": "int16", "value": int(key.episode)},
        }
    if type(key) is ObservationKey:
        return {
            "kind": "observation",
            "episode_key": _key_projection(key.episode_key),
            "phase": {"dtype": "int8", "value": int(key.phase)},
            "prefix_code": {"dtype": "int8", "value": int(key.prefix_code)},
        }
    if type(key) is ActionKey:
        return {
            "kind": "action",
            "observation_key": _key_projection(key.observation_key),
            "action_ordinal": {"dtype": "int8", "value": int(key.action_ordinal)},
        }
    if type(key) is TransitionKey:
        return {
            "kind": "transition",
            "action_key": _key_projection(key.action_key),
            "transition_ordinal": {"dtype": "int8", "value": int(key.transition_ordinal)},
        }
    if type(key) is FeedbackKey:
        return {
            "kind": "feedback",
            "transition_key": _key_projection(key.transition_key),
            "feedback_ordinal": {"dtype": "int8", "value": int(key.feedback_ordinal)},
        }
    raise ContractError("unsupported typed key")


def _validate_episode_key(key: object) -> EpisodeKey:
    if type(key) is not EpisodeKey:
        raise ContractError("episode key has the wrong type")
    if type(key.split_enum) is not np.uint8:
        raise ContractError("episode split identity has the wrong dtype")
    if type(key.regime_code) is not np.int32:
        raise ContractError("episode regime identity has the wrong dtype")
    if type(key.episode) is not np.int16 or isinstance(key.episode, (bool, np.bool_)):
        raise ContractError("episode index has the wrong dtype")
    legal_regimes = {
        (int(regime.split_enum), int(regime.code)) for regime in REGIMES
    }
    if (
        (int(key.split_enum), int(key.regime_code)) not in legal_regimes
        or int(key.episode) not in range(EPISODES_PER_REGIME)
    ):
        raise ContractError("episode key is outside the frozen generator domain")
    return key


def _validate_observation_key(key: object) -> ObservationKey:
    if type(key) is not ObservationKey:
        raise ContractError("observation key has the wrong type")
    _validate_episode_key(key.episode_key)
    if type(key.phase) is not np.int8 or type(key.prefix_code) is not np.int8:
        raise ContractError("observation identity has the wrong dtype")
    phase = int(key.phase)
    prefix = int(key.prefix_code)
    if phase not in range(HORIZON) or prefix not in range(1 << phase):
        raise ContractError("observation identity is illegal")
    return key


def _validate_action_key(key: object) -> ActionKey:
    if type(key) is not ActionKey:
        raise ContractError("action key has the wrong type")
    _validate_observation_key(key.observation_key)
    if type(key.action_ordinal) is not np.int8 or int(key.action_ordinal) not in (0, 1):
        raise ContractError("action ordinal is malformed")
    return key


def _validate_transition_key(key: object) -> TransitionKey:
    if type(key) is not TransitionKey:
        raise ContractError("transition key has the wrong type")
    _validate_action_key(key.action_key)
    if type(key.transition_ordinal) is not np.int8 or int(key.transition_ordinal) != 0:
        raise ContractError("transition ordinal is malformed")
    return key


def _validate_feedback_key(key: object) -> FeedbackKey:
    if type(key) is not FeedbackKey:
        raise ContractError("feedback key has the wrong type")
    _validate_transition_key(key.transition_key)
    if type(key.feedback_ordinal) is not np.int8 or int(key.feedback_ordinal) != 0:
        raise ContractError("feedback ordinal is malformed")
    return key


def _episode_key(regime: Regime, episode: int) -> EpisodeKey:
    return EpisodeKey(regime.split_enum, regime.code, np.int16(episode))


def _observation_key(spec: EpisodeSpec, phase: int, prefix: int) -> ObservationKey:
    return ObservationKey(spec.key, np.int8(phase), np.int8(prefix))


def _action_key(spec: EpisodeSpec, phase: int, prefix: int, action: int) -> ActionKey:
    if type(action) is not int or action not in (0, 1):
        raise ContractError("action key value is malformed")
    return ActionKey(_observation_key(spec, phase, prefix), np.int8(action))


def _transition_key(
    spec: EpisodeSpec, phase: int, prefix: int, action: int
) -> TransitionKey:
    return TransitionKey(_action_key(spec, phase, prefix, action), np.int8(0))


def _feedback_key(
    spec: EpisodeSpec, phase: int, prefix: int, action: int
) -> FeedbackKey:
    return FeedbackKey(_transition_key(spec, phase, prefix, action), np.int8(0))


def _iter_episode_specs(split: str | None = None) -> Iterator[EpisodeSpec]:
    for regime in REGIMES:
        if split is not None and regime.split != split:
            continue
        for episode in range(EPISODES_PER_REGIME):
            block = episode // 32
            cell = episode % 32
            action_code = cell // 2
            target_slot = cell % 2
            sign = -1.0 if target_slot == 0 else 1.0
            magnitude = regime.signal_scale * (1.0 + block / 64.0)
            nuisance = regime.nuisance_shift + regime.nuisance_scale * (
                (((11 * block + 5 * action_code + int(regime.code)) % 37) - 18)
                / 18.0
            )
            yield EpisodeSpec(
                regime=regime,
                episode=np.int16(episode),
                block=np.int8(block),
                cell=np.int8(cell),
                action_code=np.int8(action_code),
                target_slot=np.int8(target_slot),
                target=np.int8(target_slot),
                sign=sign,
                magnitude=float(magnitude),
                nuisance=float(nuisance),
                key=_episode_key(regime, episode),
            )


def _behavior_actions(action_code: int) -> tuple[int, int, int, int]:
    return tuple((int(action_code) >> shift) & 1 for shift in (3, 2, 1, 0))


def _prefix_from_actions(actions: Sequence[int]) -> int:
    prefix = 0
    for action in actions:
        prefix = (prefix << 1) | int(action)
    return prefix


def _prefix_bits(prefix: int, phase: int) -> tuple[int, ...]:
    return tuple((prefix >> shift) & 1 for shift in range(phase - 1, -1, -1))


def _immutable_observation(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.dtype("<f8"), order="C").copy(order="C")
    array.setflags(write=False)
    return array


def _primary_observation(
    spec: EpisodeSpec, phase: int, prefix: int, *, signal_override: float | None = None
) -> np.ndarray:
    if phase not in range(HORIZON) or prefix not in range(1 << phase):
        raise ContractError("illegal primary observation coordinates")
    slots = [-1.0, -1.0, -1.0]
    for index, bit in enumerate(_prefix_bits(prefix, phase)):
        slots[index] = float(bit)
    signal = spec.sign * spec.magnitude if signal_override is None else signal_override
    return _immutable_observation(
        [float(phase), float(signal), *slots, float(spec.nuisance)]
    )


def _replay_observation(spec: EpisodeSpec, phase: int, prefix: int) -> np.ndarray:
    # Intentionally independent from _primary_observation.
    if phase < 0 or phase > 3:
        raise ContractError("illegal replay phase")
    if prefix < 0 or prefix >= 2**phase:
        raise ContractError("illegal replay prefix")
    components = [float(phase), float(spec.sign * spec.magnitude)]
    for position in range(3):
        if position < phase:
            shift = phase - position - 1
            components.append(float((prefix // (2**shift)) % 2))
        else:
            components.append(-1.0)
    components.append(float(spec.nuisance))
    replay = np.array(components, dtype=np.dtype("<f8"), order="C")
    replay.setflags(write=False)
    return replay


def _validate_observation(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ContractError("observation is not a NumPy array")
    if value.dtype != np.dtype("<f8") or value.shape != (6,) or value.strides != (8,):
        raise ContractError("observation dtype, shape, or strides changed")
    if not value.flags.c_contiguous or value.flags.writeable or not np.isfinite(value).all():
        raise ContractError("observation layout or finiteness changed")
    phase_value = float(value[0])
    if phase_value not in (0.0, 1.0, 2.0, 3.0):
        raise ContractError("observation phase is illegal")
    phase = int(phase_value)
    for index, component in enumerate(value[2:5]):
        expected_committed = index < phase
        if expected_committed and float(component) not in (0.0, 1.0):
            raise ContractError("committed prefix component is illegal")
        if not expected_committed and float(component) != -1.0:
            raise ContractError("unused prefix component is illegal")
    return value


def _public_successor(
    predecessor: np.ndarray, action: np.ndarray
) -> np.ndarray | None:
    """Advance only the already-public action-prefix state.

    The canonical transition has no EpisodeSpec, target, evaluator, regime, or
    control-mode input.  The signed signal and nuisance bytes are carried
    forward unchanged; only the public phase and next prefix slot change.
    """

    observation = _validate_observation(predecessor)
    action_value = int(_validate_action(action))
    phase = int(observation[0])
    if phase == HORIZON - 1:
        return None
    successor = np.array(
        observation, dtype=np.dtype("<f8"), order="C", copy=True
    )
    successor[0] = float(phase + 1)
    successor[2 + phase] = float(action_value)
    successor.setflags(write=False)
    _validate_observation(successor)
    if (
        successor[1].tobytes() != observation[1].tobytes()
        or successor[5].tobytes() != observation[5].tobytes()
    ):
        raise ContractError("public successor changed a carried field")
    return successor


def _immutable_scalar(value: float, dtype: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype(dtype))
    array.setflags(write=False)
    return array


def _validate_action(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ContractError("action is not a NumPy scalar array")
    if (
        value.dtype != np.dtype("i1")
        or value.shape != ()
        or value.strides != ()
        or not value.flags.c_contiguous
        or value.flags.writeable
    ):
        raise ContractError("action dtype or scalar rank changed")
    if isinstance(value.item(), (bool, np.bool_)) or int(value) not in (0, 1):
        raise ContractError("action is outside the frozen action set")
    return value


def _validate_reward(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ContractError("feedback is not a NumPy scalar array")
    if (
        value.dtype != np.dtype("<f8")
        or value.shape != ()
        or value.strides != ()
        or not value.flags.c_contiguous
        or value.flags.writeable
    ):
        raise ContractError("feedback dtype or scalar rank changed")
    if float(value) not in (0.0, 1.0):
        raise ContractError("feedback is outside the frozen scalar set")
    return value


def _observation_identity(value: np.ndarray) -> dict[str, object]:
    _validate_observation(value)
    encoded = value.tobytes(order="C")
    return {
        "bytes": encoded.hex(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "dtype": value.dtype.str,
        "shape": [6],
        "strides": [8],
        "c_contiguous": True,
        "immutable": True,
    }


def _observation_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(_validate_observation(value).tobytes(order="C")).hexdigest()


def _terminal_reward(target: int, actions: Sequence[int]) -> float:
    return float(len(actions) == HORIZON and all(int(action) == int(target) for action in actions))


def _legal_row_projection_primary(
    spec: EpisodeSpec, phase: int, prefix: int, action: int
) -> dict[str, object]:
    predecessor = _primary_observation(spec, phase, prefix)
    observation_key = _observation_key(spec, phase, prefix)
    action_key = ActionKey(observation_key, np.int8(action))
    transition_key = TransitionKey(action_key, np.int8(0))
    feedback_key = FeedbackKey(transition_key, np.int8(0))
    terminal = phase == HORIZON - 1
    successor_prefix = (prefix << 1) | action
    successor = None if terminal else _primary_observation(spec, phase + 1, successor_prefix)
    successor_key = None if terminal else _observation_key(spec, phase + 1, successor_prefix)
    terminal_actions = (*_prefix_bits(prefix, phase), action)
    reward = _terminal_reward(int(spec.target), terminal_actions) if terminal else 0.0
    return {
        "split": spec.regime.split,
        "regime_code": {"dtype": "int32", "value": int(spec.regime.code)},
        "episode": {"dtype": "int16", "value": int(spec.episode)},
        "block": {"dtype": "int8", "value": int(spec.block)},
        "cell": {"dtype": "int8", "value": int(spec.cell)},
        "target_slot": {"dtype": "int8", "value": int(spec.target_slot)},
        "target": {"dtype": "int8", "value": int(spec.target)},
        "action_code_lineage": {"dtype": "int8", "value": int(spec.action_code)},
        "predecessor_key": _key_projection(observation_key),
        "predecessor": _observation_identity(predecessor),
        "action_key": _key_projection(action_key),
        "action": {
            "bytes": np.asarray(action, dtype=np.dtype("i1")).tobytes().hex(),
            "dtype": "int8",
            "shape": [],
            "value": action,
        },
        "transition_key": _key_projection(transition_key),
        "feedback_key": _key_projection(feedback_key),
        "successor_key": None if successor_key is None else _key_projection(successor_key),
        "successor": None if successor is None else _observation_identity(successor),
        "canonical_reward": reward,
        "update_reward": reward,
        "reward_dtype": "float64",
        "done": bool(terminal),
        "successor_legal": True,
        "donor": None,
        "origin": _key_projection(spec.key),
    }


def _legal_row_projection_replay(
    spec: EpisodeSpec, phase: int, prefix: int, action: int
) -> dict[str, object]:
    # This replay intentionally reconstructs every field without calling the
    # primary row or observation constructors.
    predecessor = _replay_observation(spec, phase, prefix)
    obs_key = ObservationKey(
        EpisodeKey(spec.regime.split_enum, spec.regime.code, np.int16(spec.episode)),
        np.int8(phase),
        np.int8(prefix),
    )
    act_key = ActionKey(obs_key, np.int8(action))
    trans_key = TransitionKey(act_key, np.int8(0))
    feed_key = FeedbackKey(trans_key, np.int8(0))
    terminal = phase == 3
    next_prefix = prefix * 2 + action
    if terminal:
        next_obs = None
        next_key = None
    else:
        next_obs = _replay_observation(spec, phase + 1, next_prefix)
        next_key = ObservationKey(obs_key.episode_key, np.int8(phase + 1), np.int8(next_prefix))
    full_actions = tuple((prefix >> shift) & 1 for shift in range(phase - 1, -1, -1)) + (action,)
    reward = float(terminal and len(full_actions) == 4 and all(bit == int(spec.target) for bit in full_actions))
    return {
        "split": str(spec.regime.split),
        "regime_code": {"dtype": "int32", "value": int(spec.regime.code)},
        "episode": {"dtype": "int16", "value": int(spec.episode)},
        "block": {"dtype": "int8", "value": int(spec.block)},
        "cell": {"dtype": "int8", "value": int(spec.cell)},
        "target_slot": {"dtype": "int8", "value": int(spec.target_slot)},
        "target": {"dtype": "int8", "value": int(spec.target)},
        "action_code_lineage": {"dtype": "int8", "value": int(spec.action_code)},
        "predecessor_key": _key_projection(obs_key),
        "predecessor": _observation_identity(predecessor),
        "action_key": _key_projection(act_key),
        "action": {
            "bytes": np.asarray(action, dtype=np.dtype("i1")).tobytes().hex(),
            "dtype": "int8",
            "shape": [],
            "value": int(action),
        },
        "transition_key": _key_projection(trans_key),
        "feedback_key": _key_projection(feed_key),
        "successor_key": None if next_key is None else _key_projection(next_key),
        "successor": None if next_obs is None else _observation_identity(next_obs),
        "canonical_reward": reward,
        "update_reward": reward,
        "reward_dtype": "float64",
        "done": bool(terminal),
        "successor_legal": True,
        "donor": None,
        "origin": _key_projection(obs_key.episode_key),
    }


def _iter_legal_rows(
    constructor: Callable[[EpisodeSpec, int, int, int], dict[str, object]]
) -> Iterator[dict[str, object]]:
    for spec in _iter_episode_specs():
        for phase in range(HORIZON):
            for prefix in range(1 << phase):
                for action in (0, 1):
                    yield constructor(spec, phase, prefix, action)


def _transition_identity_tuple(row: Mapping[str, object]) -> str:
    return json.dumps(row["transition_key"], sort_keys=True, separators=(",", ":"))


def _predecessor_identity_tuple(row: Mapping[str, object]) -> str:
    return json.dumps(row["predecessor_key"], sort_keys=True, separators=(",", ":"))


def _validate_legal_projection(row: Mapping[str, object]) -> None:
    required = {
        "split", "regime_code", "episode", "block", "cell", "target_slot",
        "target", "action_code_lineage", "predecessor_key", "predecessor",
        "action_key", "action", "transition_key", "feedback_key",
        "successor_key", "successor", "canonical_reward", "update_reward",
        "reward_dtype", "done", "successor_legal", "donor", "origin",
    }
    if set(row) != required:
        raise ContractError("legal projection field set changed")
    if row["split"] not in {"train", "validation", "test"}:
        raise ContractError("legal projection split changed")
    action = row["action"]
    if action not in (
        {"bytes": "00", "dtype": "int8", "shape": [], "value": 0},
        {"bytes": "01", "dtype": "int8", "shape": [], "value": 1},
    ):
        raise ContractError("legal projection action changed")
    if type(row["done"]) is not bool or type(row["successor_legal"]) is not bool:
        raise ContractError("legal projection Boolean changed")
    if row["reward_dtype"] != "float64":
        raise ContractError("legal projection reward dtype changed")
    done = row["done"]
    if done != (row["successor"] is None) or done != (row["successor_key"] is None):
        raise ContractError("legal projection successor/done relation changed")
    for name in ("canonical_reward", "update_reward"):
        if type(row[name]) is not float or row[name] not in (0.0, 1.0):
            raise ContractError("legal projection reward changed")
    for name in ("predecessor", "successor"):
        identity = row[name]
        if identity is None:
            continue
        if type(identity) is not dict or set(identity) != {
            "bytes", "sha256", "dtype", "shape", "strides",
            "c_contiguous", "immutable",
        }:
            raise ContractError("legal projection observation identity changed")
        if identity.get("dtype") != "<f8" or identity.get("shape") != [6]:
            raise ContractError("legal projection observation type changed")
        if identity.get("strides") != [8] or identity.get("c_contiguous") is not True:
            raise ContractError("legal projection observation layout changed")
        if identity.get("immutable") is not True:
            raise ContractError("legal projection observation mutability changed")
    if row["donor"] is not None:
        raise ContractError("canonical legal projection acquired a donor")


def _typed_metadata_value(
    field: object,
    *,
    dtype: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(field, dict)
        or set(field) != {"dtype", "value"}
        or field["dtype"] != dtype
        or type(field["value"]) is not int
    ):
        raise ContractError("legal projection metadata type changed")
    value = int(field["value"])
    if value < minimum or value > maximum:
        raise ContractError("legal projection metadata range changed")
    return value


def _validate_legal_row_against_replay(row: Mapping[str, object]) -> None:
    _validate_legal_projection(row)
    if type(row["split"]) is not str:
        raise ContractError("legal projection split type changed")
    regime_code = _typed_metadata_value(
        row["regime_code"], dtype="int32", minimum=0, maximum=2**31 - 1
    )
    episode = _typed_metadata_value(
        row["episode"], dtype="int16", minimum=0, maximum=EPISODES_PER_REGIME - 1
    )
    block = _typed_metadata_value(row["block"], dtype="int8", minimum=0, maximum=15)
    cell = _typed_metadata_value(row["cell"], dtype="int8", minimum=0, maximum=31)
    target_slot = _typed_metadata_value(
        row["target_slot"], dtype="int8", minimum=0, maximum=1
    )
    target = _typed_metadata_value(row["target"], dtype="int8", minimum=0, maximum=1)
    action_lineage = _typed_metadata_value(
        row["action_code_lineage"], dtype="int8", minimum=0, maximum=15
    )
    action_field = row["action"]
    if (
        not isinstance(action_field, dict)
        or set(action_field) != {"bytes", "dtype", "shape", "value"}
        or action_field["bytes"] not in {"00", "01"}
        or action_field["dtype"] != "int8"
        or action_field["shape"] != []
        or type(action_field["value"]) is not int
        or action_field["value"] not in (0, 1)
        or action_field["bytes"] != f"{int(action_field['value']):02x}"
    ):
        raise ContractError("legal projection action scalar changed")
    regimes = {
        (regime.split, int(regime.code)): regime for regime in REGIMES
    }
    try:
        regime = regimes[(row["split"], regime_code)]
    except KeyError as error:
        raise ContractError("legal projection regime identity changed") from error
    key = EpisodeKey(regime.split_enum, regime.code, np.int16(episode))
    try:
        spec = _spec_index()[key]
    except KeyError as error:
        raise ContractError("legal projection episode identity changed") from error
    if (
        block != int(spec.block)
        or cell != int(spec.cell)
        or target_slot != int(spec.target_slot)
        or target != int(spec.target)
        or action_lineage != int(spec.action_code)
    ):
        raise ContractError("legal projection evaluator metadata changed")
    predecessor_key = row["predecessor_key"]
    if not isinstance(predecessor_key, dict):
        raise ContractError("legal projection predecessor key changed")
    phase_field = predecessor_key.get("phase")
    prefix_field = predecessor_key.get("prefix_code")
    phase = _typed_metadata_value(
        phase_field, dtype="int8", minimum=0, maximum=HORIZON - 1
    )
    prefix = _typed_metadata_value(
        prefix_field, dtype="int8", minimum=0, maximum=(1 << phase) - 1
    )
    expected = _legal_row_projection_replay(
        spec, phase, prefix, int(action_field["value"])
    )
    if dict(row) != expected:
        raise ContractError("legal projection differs from independent replay")


EXPECTED_FAMILY_SHA256 = (
    "c3e093639b05690016f8f39ed7dba75c0b493e30d1508664ec7225748c744c11"
)


def compute_family_sha256() -> str:
    digest = hashlib.sha256()
    for row in _iter_legal_rows(_legal_row_projection_primary):
        _stream_update(digest, row)
    return digest.hexdigest()


def _family_audit() -> dict[str, object]:
    primary_digest = hashlib.sha256()
    replay_digest = hashlib.sha256()
    transition_keys: set[str] = set()
    predecessor_keys: set[str] = set()
    counts = Counter()
    train_cell_counts: Counter[tuple[int, int, int, int]] = Counter()
    public_rows_by_split: dict[str, set[str]] = defaultdict(set)
    primary_iterator = _iter_legal_rows(_legal_row_projection_primary)
    replay_iterator = _iter_legal_rows(_legal_row_projection_replay)
    try:
        for primary, replay in zip(primary_iterator, replay_iterator, strict=True):
            _validate_legal_row_against_replay(primary)
            _validate_legal_row_against_replay(replay)
            if primary != replay:
                raise ContractError("independent family replay changed a legal row")
            _stream_update(primary_digest, primary)
            _stream_update(replay_digest, replay)
            transition_keys.add(_transition_identity_tuple(primary))
            predecessor_keys.add(_predecessor_identity_tuple(primary))
            counts["rows"] += 1
            counts["terminal" if primary["done"] else "nonterminal"] += 1
            counts[f"target_{primary['target_slot']['value']}"] += 1
            counts[f"action_{primary['action']['value']}"] += 1
            public_rows_by_split[str(primary["split"])].add(
                _json_sha256(
                    {
                        "predecessor": primary["predecessor"],
                        "action": primary["action"],
                        "successor": primary["successor"],
                        "done": primary["done"],
                    }
                )
            )
            if primary["split"] == "train":
                obs = _identity_to_observation(primary["predecessor"])
                state = _state_key(obs)
                train_cell_counts[(*state, int(primary["action"]["value"]))] += 1
    except ValueError as error:
        raise ContractError("independent family replay length changed") from error
    primary_sha = primary_digest.hexdigest()
    replay_sha = replay_digest.hexdigest()
    coverage_exact = len(train_cell_counts) == 60 and all(
        count == 1024 for count in train_cell_counts.values()
    )
    complete_public_split_disjoint = all(
        public_rows_by_split[left].isdisjoint(public_rows_by_split[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    result = {
        "primary_sha256": primary_sha,
        "replay_sha256": replay_sha,
        "expected_sha256": primary_sha == EXPECTED_FAMILY_SHA256,
        "replay_exact": primary_sha == replay_sha,
        "rows": counts["rows"],
        "nonterminal_rows": counts["nonterminal"],
        "terminal_rows": counts["terminal"],
        "predecessor_nodes": len(predecessor_keys),
        "unique_transition_keys": len(transition_keys),
        "target_balance": counts["target_0"] == counts["target_1"],
        "action_balance": counts["action_0"] == counts["action_1"],
        "train_cell_coverage_exact": coverage_exact,
        "complete_public_split_disjoint": complete_public_split_disjoint,
    }
    exact = (
        primary_sha == EXPECTED_FAMILY_SHA256
        and replay_sha == primary_sha
        and counts["rows"] == 122880
        and counts["nonterminal"] == 57344
        and counts["terminal"] == 65536
        and len(predecessor_keys) == 61440
        and len(transition_keys) == 122880
        and counts["target_0"] == counts["target_1"]
        and counts["action_0"] == counts["action_1"]
        and coverage_exact
        and complete_public_split_disjoint
    )
    if not exact:
        raise ContractError("complete legal-family gate failed")
    return result


def _family_corruption_audit() -> dict[str, object]:
    nonterminal = copy.deepcopy(
        next(
            row
            for row in _iter_legal_rows(_legal_row_projection_primary)
            if not row["done"]
        )
    )
    terminal = copy.deepcopy(
        next(
            row
            for row in _iter_legal_rows(_legal_row_projection_primary)
            if row["done"]
        )
    )
    before = dict(_CONSTRUCTION_COUNTS)
    if before != {"collector": 0, "learner": 0}:
        raise ContractError("a collector or learner existed before the family gate")
    mutations: tuple[tuple[dict[str, object], Callable[[dict[str, object]], None]], ...] = (
        (nonterminal, lambda row: row.__setitem__("split", "test")),
        (nonterminal, lambda row: row["regime_code"].__setitem__("value", 9999)),
        (nonterminal, lambda row: row["episode"].__setitem__("value", 7)),
        (nonterminal, lambda row: row["block"].__setitem__("value", 7)),
        (nonterminal, lambda row: row["cell"].__setitem__("value", 7)),
        (nonterminal, lambda row: row["target_slot"].__setitem__("value", 1)),
        (nonterminal, lambda row: row["target"].__setitem__("value", 1)),
        (nonterminal, lambda row: row["action_code_lineage"].__setitem__("value", 7)),
        (nonterminal, lambda row: row["predecessor_key"]["phase"].__setitem__("dtype", "int16")),
        (nonterminal, lambda row: row["predecessor"].__setitem__("sha256", "0" * 64)),
        (nonterminal, lambda row: row["predecessor"].__setitem__("immutable", False)),
        (nonterminal, lambda row: row["action_key"].__setitem__("kind", "broken")),
        (nonterminal, lambda row: row["action"].__setitem__("bytes", "02")),
        (nonterminal, lambda row: row["transition_key"].__setitem__("kind", "broken")),
        (nonterminal, lambda row: row["feedback_key"].__setitem__("kind", "broken")),
        (nonterminal, lambda row: row["successor_key"].__setitem__("kind", "broken")),
        (nonterminal, lambda row: row["successor"].__setitem__("bytes", "01" + str(row["successor"]["bytes"])[2:])),
        (nonterminal, lambda row: row["successor"].__setitem__("strides", [16])),
        (nonterminal, lambda row: row.__setitem__("canonical_reward", 1.0)),
        (terminal, lambda row: row.__setitem__("update_reward", 1.0 - float(row["update_reward"]))),
        (nonterminal, lambda row: row.__setitem__("reward_dtype", "float32")),
        (nonterminal, lambda row: row.__setitem__("done", 0)),
        (nonterminal, lambda row: row.__setitem__("successor_legal", False)),
        (nonterminal, lambda row: row.__setitem__("donor", {"unexpected": True})),
        (nonterminal, lambda row: row.__setitem__("origin", copy.deepcopy(_key_projection(_paired_spec(_row_coordinates(row)[0]).key)))),
    )
    corruptions: list[dict[str, object]] = []
    for template, mutate in mutations:
        changed = copy.deepcopy(template)
        mutate(changed)
        corruptions.append(changed)
    rejected = sum(
        _expect_contract_error(
            lambda corruption=corruption: _validate_legal_row_against_replay(corruption)
        )
        for corruption in corruptions
    )
    after = dict(_CONSTRUCTION_COUNTS)
    if rejected != len(corruptions) or after != before:
        raise ContractError("family corruption gate did not fail closed")
    return {
        "corruption_classes": len(corruptions),
        "corruptions_rejected": rejected,
        "factory_count_before_gate": sum(before.values()),
    }


def _state_key(observation: np.ndarray) -> tuple[int, int, int]:
    obs = _validate_observation(observation)
    phase = int(obs[0])
    sign_bin = 0 if float(obs[1]) <= 0.0 else 1
    prefix = 0
    for component in obs[2 : 2 + phase]:
        prefix = (prefix << 1) | int(component)
    return sign_bin, phase, prefix


def _all_state_action_keys() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (sign, phase, prefix, action)
        for sign in (0, 1)
        for phase in range(HORIZON)
        for prefix in range(1 << phase)
        for action in (0, 1)
    )


ALL_STATE_ACTION_KEYS = _all_state_action_keys()


@lru_cache(maxsize=1)
def _spec_index() -> dict[EpisodeKey, EpisodeSpec]:
    return {spec.key: spec for spec in _iter_episode_specs()}


def _source_specs_sha256(specs: Sequence[EpisodeSpec]) -> str:
    projection: list[dict[str, object]] = []
    for spec in specs:
        if type(spec) is not EpisodeSpec:
            raise ContractError("source yielded a malformed episode specification")
        _validate_episode_key(spec.key)
        projection.append(
            {
                "key": _key_projection(spec.key),
                "split": spec.regime.split,
                "split_enum": {"dtype": "uint8", "value": int(spec.regime.split_enum)},
                "regime_code": {"dtype": "int32", "value": int(spec.regime.code)},
                "episode": {"dtype": "int16", "value": int(spec.episode)},
                "block": {"dtype": "int8", "value": int(spec.block)},
                "cell": {"dtype": "int8", "value": int(spec.cell)},
                "action_code": {"dtype": "int8", "value": int(spec.action_code)},
                "target_slot": {"dtype": "int8", "value": int(spec.target_slot)},
                "target": {"dtype": "int8", "value": int(spec.target)},
                "sign": float(spec.sign),
                "magnitude": float(spec.magnitude),
                "nuisance": float(spec.nuisance),
            }
        )
    return _json_sha256(projection)


def _expected_source_sha256(split: str) -> str:
    if split not in {"train", "validation", "test"}:
        raise ContractError("unknown frozen source split")
    return _source_specs_sha256(tuple(_iter_episode_specs(split)))


def _paired_spec(spec: EpisodeSpec) -> EpisodeSpec:
    paired_episode = int(spec.episode) ^ 1
    key = EpisodeKey(spec.regime.split_enum, spec.regime.code, np.int16(paired_episode))
    return _spec_index()[key]


def _reward_origin_spec(spec: EpisodeSpec) -> EpisodeSpec:
    block = int(spec.block)
    action_code = int(spec.action_code)
    target_slot = int(spec.target_slot)
    origin_block = (block + 1) % 16
    origin_action_code = (action_code + block) % 16
    origin_episode = origin_block * 32 + origin_action_code * 2 + target_slot
    key = EpisodeKey(spec.regime.split_enum, spec.regime.code, np.int16(origin_episode))
    return _spec_index()[key]


def _runtime_observation(
    spec: EpisodeSpec, phase: int, prefix: int, mode: str
) -> np.ndarray:
    if mode == "transition_control" and phase > 0:
        return _primary_observation(_paired_spec(spec), phase, prefix)
    if mode == "signal_ablation":
        return _primary_observation(spec, phase, prefix, signal_override=0.0)
    return _primary_observation(spec, phase, prefix)


def _transition_control_successor(
    predecessor: np.ndarray,
    action: np.ndarray,
    *,
    donor_initial_observation: np.ndarray,
) -> np.ndarray | None:
    """Apply the frozen first-step donor payload in the negative control only."""
    successor = _public_successor(predecessor, action)
    if successor is None:
        return None
    phase = int(_validate_observation(predecessor)[0])
    if phase == 0:
        donor_signal = _validate_observation(donor_initial_observation)[1]
        substituted = np.array(
            successor, dtype=np.dtype("<f8"), order="C", copy=True
        )
        substituted[1] = donor_signal
        substituted.setflags(write=False)
        return _validate_observation(substituted)
    return successor


class _LazyValue:
    def __init__(
        self,
        materializer: Callable[[], object],
        *,
        allowed_stage: str,
        label: str,
    ) -> None:
        self._materializer = materializer
        self._allowed_stage = allowed_stage
        self._label = label
        self._resolved = False
        self._poisoned = False
        self.audit = LazyAudit()

    def resolve(self, stage: str) -> object:
        self.audit.attempted += 1
        if self.audit.first_stage is None:
            self.audit.first_stage = stage
        self.audit.last_stage = stage
        if self._poisoned:
            raise ContractError(f"{self._label} is poisoned")
        if stage != self._allowed_stage:
            self._poisoned = True
            raise ContractError(f"{self._label} resolved at the wrong stage")
        if self._resolved:
            self._poisoned = True
            raise ContractError(f"{self._label} resolved more than once")
        try:
            value = self._materializer()
        except Exception:
            self._poisoned = True
            raise
        self._resolved = True
        self.audit.permitted += 1
        self.audit.value_sha256 = _json_sha256(value)
        return value


@dataclass
class _SourceEpisodeHandle:
    """One live episode view whose four rows resolve through its source."""

    _source: "_CountingSource"
    _spec: EpisodeSpec
    _ordinal: int
    _session_nonce: object
    _capability_sha256: str
    _next_phase: int = 0

    def materialize(self, phase: int) -> EpisodeSpec:
        if type(self) is not _SourceEpisodeHandle:
            raise ContractError("source episode capability type changed")
        if type(phase) is not int or phase != self._next_phase:
            raise ContractError("source episode row materialized out of order")
        self._source._materialize(self, phase)  # noqa: SLF001
        self._next_phase += 1
        return self._spec


class _BehaviorSelector:
    def __init__(self) -> None:
        self.completed_episodes = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def select(
        self,
        policy_input: Mapping[str, object],
        *,
        nested_callback: Callable[[], None] | None = None,
    ) -> np.ndarray:
        if self._active:
            raise ContractError("selector reentered")
        if not isinstance(policy_input, Mapping) or tuple(policy_input) != POLICY_INPUT_FIELDS:
            raise ContractError("policy input field set changed")
        observation = _validate_observation(policy_input["observation"])
        self._active = True
        try:
            if nested_callback is not None:
                nested_callback()
            phase = int(observation[0])
            local_episode = self.completed_episodes % EPISODES_PER_REGIME
            action_code = (local_episode % 32) // 2
            action = _behavior_actions(action_code)[phase]
            return _immutable_scalar(float(action), "i1")
        finally:
            self._active = False

    def close_episode(self) -> None:
        self.completed_episodes += 1


class _EnvironmentStepGuard:
    def __init__(self) -> None:
        self._active: tuple[EpisodeKey, int, int] | None = None

    def enter(self, episode_key: EpisodeKey, phase: int, prefix: int) -> None:
        _validate_episode_key(episode_key)
        if self._active is not None:
            raise ContractError("environment step reentered")
        if phase not in range(HORIZON) or prefix not in range(1 << phase):
            raise ContractError("environment step coordinates changed")
        self._active = (episode_key, phase, prefix)

    def exit(self, episode_key: EpisodeKey, phase: int, prefix: int) -> None:
        if self._active != (episode_key, phase, prefix):
            raise ContractError("environment step exited with another identity")
        self._active = None

    def abort(self) -> None:
        self._active = None


@dataclass(frozen=True)
class _PendingTransition:
    observation_record: ObservationRecord
    action_record: ActionRecord
    expected_transition: TransitionRecord
    expected_feedback: FeedbackRecord


@dataclass(frozen=True)
class _StepAuthorization:
    episode_key: EpisodeKey
    phase: np.int8
    prefix: np.int8
    action: np.int8
    mode: str
    sequence: np.int32
    sink_nonce: object
    token_sha256: str


_ISSUED_STEP_AUTHORIZATIONS: set[str] = set()


def _step_authorization_projection(
    episode_key: EpisodeKey,
    phase: np.int8,
    prefix: np.int8,
    action: np.int8,
    mode: str,
    sequence: np.int32,
    sink_nonce: object,
) -> dict[str, object]:
    return {
        "episode_key": _key_projection(episode_key),
        "phase": {"dtype": "int8", "value": int(phase)},
        "prefix": {"dtype": "int8", "value": int(prefix)},
        "action": {"dtype": "int8", "value": int(action)},
        "mode": mode,
        "sequence": {"dtype": "int32", "value": int(sequence)},
        "sink_nonce": id(sink_nonce),
    }


def _issue_step_authorization(
    sink: "_TraceSink",
    spec: EpisodeSpec,
    phase: int,
    prefix: int,
    action: np.ndarray,
    mode: str,
) -> _StepAuthorization:
    if (
        type(sink) is not _TraceSink
        or sink._open_episode != spec.key  # noqa: SLF001
        or sink.pending is not None
        or sink._authorization_token is not None  # noqa: SLF001
        or sink._mode != mode  # noqa: SLF001
        or sink._records_in_episode != phase  # noqa: SLF001
        or sink._expected_prefix != prefix  # noqa: SLF001
    ):
        raise ContractError("environment authorization issued out of sequence")
    action_value = int(_validate_action(action))
    sequence = np.int32(sink._authorization_serial)  # noqa: SLF001
    phase_value = np.int8(phase)
    prefix_value = np.int8(prefix)
    action_value_typed = np.int8(action_value)
    projection = _step_authorization_projection(
        spec.key,
        phase_value,
        prefix_value,
        action_value_typed,
        mode,
        sequence,
        sink._authorization_nonce,  # noqa: SLF001
    )
    authorization = _StepAuthorization(
        spec.key,
        phase_value,
        prefix_value,
        action_value_typed,
        mode,
        sequence,
        sink._authorization_nonce,  # noqa: SLF001
        _json_sha256(projection),
    )
    _ISSUED_STEP_AUTHORIZATIONS.add(authorization.token_sha256)
    sink._authorization_token = authorization.token_sha256  # noqa: SLF001
    sink._authorization_serial += 1  # noqa: SLF001
    return authorization


class _TraceSink:
    def __init__(self, mode: str = "canonical") -> None:
        if mode not in {
            "canonical", "transition_control", "reward_origin_control",
            "signal_ablation",
        }:
            raise ContractError("trace sink received an unknown mode")
        self._mode = mode
        self.observations: list[ObservationRecord] = []
        self.actions: list[ActionRecord] = []
        self.transitions: list[TransitionRecord] = []
        self.feedback: list[FeedbackRecord] = []
        self.closes: list[EpisodeCloseRecord] = []
        self.pending: _PendingTransition | None = None
        self._open_episode: EpisodeKey | None = None
        self._records_in_episode = 0
        self._expected_prefix = 0
        self._last_feedback_key: FeedbackKey | None = None
        self._last_feedback_done = False
        self._authorization_nonce = object()
        self._authorization_serial = 0
        self._authorization_token: str | None = None

    def _abort(self) -> None:
        if self._authorization_token is not None:
            _ISSUED_STEP_AUTHORIZATIONS.discard(self._authorization_token)
        self.observations.clear()
        self.actions.clear()
        self.transitions.clear()
        self.feedback.clear()
        self.closes.clear()
        self.pending = None
        self._open_episode = None
        self._records_in_episode = 0
        self._expected_prefix = 0
        self._last_feedback_key = None
        self._last_feedback_done = False
        self._authorization_token = None

    def begin_episode(self, key: EpisodeKey) -> None:
        _validate_episode_key(key)
        if self._open_episode is not None or self.pending is not None:
            raise ContractError("episode began with pending state")
        self._open_episode = key
        self._records_in_episode = 0
        self._expected_prefix = 0
        self._last_feedback_key = None
        self._last_feedback_done = False

    def prepare(
        self,
        observation: ObservationRecord,
        action: ActionRecord,
        transition: TransitionRecord,
        feedback: FeedbackRecord,
        *,
        authorization: _StepAuthorization,
    ) -> None:
        try:
            if self._open_episode is None or self.pending is not None:
                raise ContractError("transition prepared at the wrong boundary")
            if (
                type(authorization) is not _StepAuthorization
                or authorization.token_sha256 not in _ISSUED_STEP_AUTHORIZATIONS
                or authorization.token_sha256 != self._authorization_token
                or authorization.episode_key != self._open_episode
                or authorization.mode != self._mode
                or type(authorization.phase) is not np.int8
                or type(authorization.prefix) is not np.int8
                or type(authorization.action) is not np.int8
                or type(authorization.sequence) is not np.int32
                or authorization.sink_nonce is not self._authorization_nonce
                or authorization.token_sha256
                != _json_sha256(
                    _step_authorization_projection(
                        authorization.episode_key,
                        authorization.phase,
                        authorization.prefix,
                        authorization.action,
                        authorization.mode,
                        authorization.sequence,
                        authorization.sink_nonce,
                    )
                )
            ):
                raise ContractError("pending transition lacks environment authorization")
            if (
                type(observation) is not ObservationRecord
                or type(action) is not ActionRecord
                or type(transition) is not TransitionRecord
                or type(feedback) is not FeedbackRecord
            ):
                raise ContractError("pending transition component type changed")
            if observation.key.episode_key != self._open_episode:
                raise ContractError("pending predecessor belongs to another episode")
            try:
                spec = _spec_index()[self._open_episode]
            except KeyError as error:
                raise ContractError("pending episode is outside the frozen source") from error
            obs_key = _validate_observation_key(observation.key)
            if obs_key.episode_key != spec.key:
                raise ContractError("pending predecessor source identity changed")
            phase = int(obs_key.phase)
            prefix = int(obs_key.prefix_code)
            action_value = _validate_action(action.value)
            if (
                int(authorization.phase) != phase
                or int(authorization.prefix) != prefix
                or int(authorization.action) != int(action_value)
                or phase != self._records_in_episode
                or prefix != self._expected_prefix
            ):
                raise ContractError("pending transition changed its authorized identity")
            expected_predecessor = _runtime_observation(spec, phase, prefix, self._mode)
            successor_prefix = (prefix << 1) | int(action_value)
            if self._mode == "transition_control" and phase == 0:
                expected_successor = _transition_control_successor(
                    _validate_observation(observation.value),
                    action.value,
                    donor_initial_observation=_primary_observation(
                        _paired_spec(spec), 0, 0
                    ),
                )
            else:
                expected_successor = _public_successor(
                    _validate_observation(observation.value), action.value
                )
            expected_update = (
                _episode_terminal_reward(_reward_origin_spec(spec))
                if self._mode == "reward_origin_control" and phase == HORIZON - 1
                else (_episode_terminal_reward(spec) if phase == HORIZON - 1 else 0.0)
            )
            expected = _expected_runtime_records(
                spec,
                phase,
                prefix,
                _copy_action(action_value),
                self._mode,
                expected_predecessor,
                expected_successor,
                _immutable_scalar(expected_update, "<f8"),
            )
            if not _runtime_records_equal(
                (observation, action, transition, feedback), expected
            ):
                raise ContractError("pending transition failed independent authentication")
            self.pending = _PendingTransition(*expected)
        except (AttributeError, ContractError, KeyError, TypeError, ValueError):
            self._abort()
            raise
        finally:
            if type(authorization) is _StepAuthorization:
                _ISSUED_STEP_AUTHORIZATIONS.discard(authorization.token_sha256)
                if self._authorization_token == authorization.token_sha256:
                    self._authorization_token = None

    def append(
        self,
        transition: TransitionRecord,
        feedback: FeedbackRecord,
    ) -> None:
        if self.pending is None:
            self._abort()
            raise ContractError("transition appended without an exact pending identity")
        expected = self.pending
        if not _transition_equal(transition, expected.expected_transition) or not _feedback_equal(
            feedback, expected.expected_feedback
        ):
            self._abort()
            raise ContractError("transition components do not match pending identity")
        self.observations.append(expected.observation_record)
        self.actions.append(expected.action_record)
        self.transitions.append(transition)
        self.feedback.append(feedback)
        self.pending = None
        self._records_in_episode += 1
        self._expected_prefix = (
            self._expected_prefix << 1
        ) | int(_validate_action(expected.action_record.value))
        self._last_feedback_key = feedback.key
        self._last_feedback_done = feedback.done

    def close_episode(self, final_feedback_key: FeedbackKey) -> None:
        if self._open_episode is None or self.pending is not None:
            raise ContractError("episode closed with pending state")
        if self._records_in_episode != HORIZON:
            raise ContractError("episode closed with the wrong record count")
        if (
            self._last_feedback_key != final_feedback_key
            or not self._last_feedback_done
        ):
            self._abort()
            raise ContractError("episode close link is not the terminal feedback")
        close = EpisodeCloseRecord(
            self._open_episode,
            final_feedback_key,
            np.int8(self._records_in_episode),
        )
        if any(item.episode_key == self._open_episode for item in self.closes):
            raise ContractError("episode closed more than once")
        self.closes.append(close)
        self._open_episode = None
        self._records_in_episode = 0
        self._expected_prefix = 0
        self._last_feedback_key = None
        self._last_feedback_done = False

    def seal(self) -> TraceBundle:
        if self._open_episode is not None or self.pending is not None:
            self._abort()
            raise ContractError("trace sealed with pending state")
        return TraceBundle(
            tuple(self.observations),
            tuple(self.actions),
            tuple(self.transitions),
            tuple(self.feedback),
            tuple(self.closes),
        )


def _feedback_equal(left: FeedbackRecord, right: FeedbackRecord) -> bool:
    if (
        type(left) is not FeedbackRecord
        or type(right) is not FeedbackRecord
        or type(left.done) is not bool
        or type(right.done) is not bool
    ):
        return False
    try:
        left_canonical = _validate_reward(left.canonical_reward)
        right_canonical = _validate_reward(right.canonical_reward)
        left_update = _validate_reward(left.update_reward)
        right_update = _validate_reward(right.update_reward)
    except ContractError:
        return False
    return (
        left.key == right.key
        and left.transition_key == right.transition_key
        and np.array_equal(left_canonical, right_canonical)
        and np.array_equal(left_update, right_update)
        and left.origin_episode == right.origin_episode
        and left.done == right.done
    )


def _observation_record_equal(
    left: ObservationRecord, right: ObservationRecord
) -> bool:
    if type(left) is not ObservationRecord or type(right) is not ObservationRecord:
        return False
    try:
        left_value = _validate_observation(left.value)
        right_value = _validate_observation(right.value)
    except (AttributeError, ContractError):
        return False
    return (
        left.key == right.key
        and left.value_sha256 == right.value_sha256
        and left.value_sha256 == _observation_sha256(left_value)
        and right.value_sha256 == _observation_sha256(right_value)
        and left_value.tobytes(order="C") == right_value.tobytes(order="C")
    )


def _action_record_equal(left: ActionRecord, right: ActionRecord) -> bool:
    if type(left) is not ActionRecord or type(right) is not ActionRecord:
        return False
    try:
        left_value = _validate_action(left.value)
        right_value = _validate_action(right.value)
    except (AttributeError, ContractError):
        return False
    return (
        left.key == right.key
        and left.observation_key == right.observation_key
        and left_value.tobytes() == right_value.tobytes()
    )


def _transition_equal(left: TransitionRecord, right: TransitionRecord) -> bool:
    return (
        type(left) is TransitionRecord
        and type(right) is TransitionRecord
        and type(left.done) is bool
        and type(right.done) is bool
        and left == right
    )


def _runtime_records_equal(
    left: tuple[ObservationRecord, ActionRecord, TransitionRecord, FeedbackRecord],
    right: tuple[ObservationRecord, ActionRecord, TransitionRecord, FeedbackRecord],
) -> bool:
    return (
        _observation_record_equal(left[0], right[0])
        and _action_record_equal(left[1], right[1])
        and _transition_equal(left[2], right[2])
        and _feedback_equal(left[3], right[3])
    )


def _policy_input(observation: np.ndarray) -> Mapping[str, object]:
    return MappingProxyType({"observation": _validate_observation(observation)})


def _episode_terminal_reward(spec: EpisodeSpec) -> float:
    return _terminal_reward(int(spec.target), _behavior_actions(int(spec.action_code)))


def _expected_runtime_records(
    spec: EpisodeSpec,
    phase: int,
    prefix: int,
    action: np.ndarray,
    mode: str,
    predecessor: np.ndarray,
    successor: np.ndarray | None,
    update_reward: np.ndarray,
) -> tuple[ObservationRecord, ActionRecord, TransitionRecord, FeedbackRecord]:
    observation_key = _observation_key(spec, phase, prefix)
    action_key = ActionKey(observation_key, np.int8(int(action)))
    transition_key = TransitionKey(action_key, np.int8(0))
    feedback_key = FeedbackKey(transition_key, np.int8(0))
    done = phase == HORIZON - 1
    successor_prefix = (prefix << 1) | int(action)
    successor_key = None if done else _observation_key(spec, phase + 1, successor_prefix)
    donor = _paired_spec(spec).key if mode == "transition_control" and not done else None
    origin = _reward_origin_spec(spec).key if mode == "reward_origin_control" and done else spec.key
    canonical_value = _episode_terminal_reward(spec) if done else 0.0
    observation_record = ObservationRecord(
        observation_key, predecessor, _observation_sha256(predecessor)
    )
    action_record = ActionRecord(action_key, observation_key, action)
    transition_record = TransitionRecord(
        transition_key,
        action_key,
        observation_key,
        _observation_sha256(predecessor),
        successor_key,
        None if successor is None else _observation_sha256(successor),
        spec.key,
        donor,
        (
            _observation_sha256(successor)
            if donor is not None and successor is not None
            else None
        ),
        bool(done),
    )
    feedback_record = FeedbackRecord(
        feedback_key,
        transition_key,
        _immutable_scalar(canonical_value, "<f8"),
        update_reward,
        origin,
        bool(done),
    )
    return observation_record, action_record, transition_record, feedback_record


def _collect_trace(
    specs: Iterable[_SourceEpisodeHandle],
    mode: str = "canonical",
    *,
    gate: _FamilyGateToken,
    attack: str | None = None,
    attack_report: dict[str, object] | None = None,
) -> tuple[TraceBundle, dict[str, object]]:
    _require_family_gate(gate)
    if mode not in {
        "canonical",
        "transition_control",
        "reward_origin_control",
        "signal_ablation",
    }:
        raise ContractError("unknown collection mode")
    _CONSTRUCTION_COUNTS["collector"] += 1
    selector = _BehaviorSelector()
    step_guard = _EnvironmentStepGuard()
    sink = _TraceSink(mode)
    events: list[str] = []
    lazy_installed = 0
    lazy_attempted = 0
    lazy_permitted = 0
    lazy_audits: list[dict[str, object]] = []
    origin_queue: list[EpisodeKey] = []
    origin_lazy_attempted = 0
    origin_lazy_permitted = 0
    origin_lazy_early = 0
    active_lazies: list[_LazyValue] = []
    completed_specs = 0

    def reject_attack(operation: Callable[[], object]) -> None:
        rejected = False
        caught: Exception | None = None
        try:
            operation()
        except (AttributeError, ContractError, KeyError, TypeError, ValueError) as error:
            rejected = True
            caught = error
        sink._abort()  # noqa: SLF001 - the attack harness proves cleanup.
        step_guard.abort()
        origin_queue.clear()
        _ISSUED_STEP_AUTHORIZATIONS.clear()
        if attack_report is not None:
            attack_report.update(
                {
                    "boundary_rejected": rejected,
                    "collector_factory_reached": True,
                    "lazy_attempted": sum(item.audit.attempted for item in active_lazies),
                    "lazy_permitted": sum(item.audit.permitted for item in active_lazies),
                    "open_episode_cleared": sink._open_episode is None,  # noqa: SLF001
                    "origin_queue_cleared": not origin_queue,
                    "pending_cleared": sink.pending is None,
                }
            )
        if not rejected:
            raise ContractError("timing attack did not reach a rejecting boundary")
        raise ContractError("timing attack rejected at the real collection boundary") from caught

    for source_episode in specs:
        if type(source_episode) is not _SourceEpisodeHandle:
            raise ContractError("collector received an invalid source episode")
        spec = source_episode.materialize(0)
        sink.begin_episode(spec.key)
        prefix = 0
        episode_actions: list[int] = []
        observation = _runtime_observation(spec, 0, 0, mode)
        if mode == "reward_origin_control":
            origin_spec = _reward_origin_spec(spec)
            origin_queue.append(origin_spec.key)
            terminal_materializer = lambda origin_spec=origin_spec: _immutable_scalar(
                _episode_terminal_reward(origin_spec), "<f8"
            )
        else:
            def terminal_materializer(
                *, spec: EpisodeSpec = spec, episode_actions: list[int] = episode_actions
            ) -> np.ndarray:
                if len(episode_actions) != HORIZON:
                    raise ContractError("terminal scalar materialized before the full action prefix")
                return _immutable_scalar(
                    _terminal_reward(int(spec.target), tuple(episode_actions)), "<f8"
                )

        terminal_reward_lazy = _LazyValue(
            terminal_materializer,
            allowed_stage="resolve_terminal3",
            label="terminal_feedback",
        )
        active_lazies.append(terminal_reward_lazy)
        lazy_installed += 1
        for phase in range(HORIZON):
            if phase > 0:
                materialized_spec = source_episode.materialize(phase)
                if materialized_spec != spec:
                    raise ContractError("source episode identity changed mid-trajectory")
            if int(_validate_observation(observation)[0]) != phase:
                raise ContractError("public successor did not reach the next phase")
            step_guard.enter(spec.key, phase, prefix)
            events.append(f"observe{phase}")
            selected_action: dict[str, int | None] = {"value": None}
            successor_stage = (
                f"resolve_successor{phase}"
                if phase < HORIZON - 1
                else "resolve_terminal3"
            )
            reward_stage = (
                f"resolve_zero{phase}"
                if phase < HORIZON - 1
                else "resolve_terminal3"
            )

            if mode == "transition_control" and phase == 0:
                donor_initial = _primary_observation(_paired_spec(spec), 0, 0)

                def successor_materializer(
                    *,
                    observation: np.ndarray = observation,
                    selected_action: dict[str, int | None] = selected_action,
                    donor_initial: np.ndarray = donor_initial,
                ) -> np.ndarray | None:
                    action_value = selected_action["value"]
                    if action_value is None:
                        raise ContractError(
                            "successor materialized before validated action"
                        )
                    return _transition_control_successor(
                        observation,
                        _immutable_scalar(float(action_value), "i1"),
                        donor_initial_observation=donor_initial,
                    )

            else:

                def successor_materializer(
                    *,
                    observation: np.ndarray = observation,
                    selected_action: dict[str, int | None] = selected_action,
                ) -> np.ndarray | None:
                    action_value = selected_action["value"]
                    if action_value is None:
                        raise ContractError(
                            "successor materialized before validated action"
                        )
                    return _public_successor(
                        observation,
                        _immutable_scalar(float(action_value), "i1"),
                    )

            successor_lazy = _LazyValue(
                successor_materializer,
                allowed_stage=successor_stage,
                label="successor",
            )
            active_lazies.append(successor_lazy)
            if phase == HORIZON - 1:
                reward_lazy = terminal_reward_lazy
            else:
                def reward_materializer(
                    *,
                    spec: EpisodeSpec = spec,
                    phase: int = phase,
                    prefix: int = prefix,
                    selected_action: dict[str, int | None] = selected_action,
                ) -> np.ndarray:
                    action_value = selected_action["value"]
                    if action_value is None:
                        raise ContractError("feedback materialized before validated action")
                    reward = (
                        _terminal_reward(
                            int(spec.target),
                            (*_prefix_bits(prefix, phase), action_value),
                        )
                        if phase == HORIZON - 1
                        else 0.0
                    )
                    return _immutable_scalar(reward, "<f8")

                reward_lazy = _LazyValue(
                    reward_materializer,
                    allowed_stage=reward_stage,
                    label="feedback",
                )
                active_lazies.append(reward_lazy)
                lazy_installed += 1
            lazy_installed += 1
            if attack == "nonempty_origin_queue_at_split" and phase == 0:
                reject_attack(lambda: _assert_split_clean(sink, origin_queue))
            if attack == "terminal_scalar_before_phase_three_action" and phase == 0:
                reject_attack(
                    lambda: terminal_reward_lazy.resolve(
                        "before_phase_three_action"
                    )
                )
            if attack == "origin_resolution_before_terminal" and phase == 0:
                reject_attack(
                    lambda: terminal_reward_lazy.resolve("origin_before_terminal")
                )
            if attack == "origin_bearing_policy_feedback" and phase == 0:
                reject_attack(
                    lambda: selector.select(
                        MappingProxyType(
                            {"observation": observation, "origin": spec.key}
                        )
                    )
                )
            nested_callback: Callable[[], None] | None = None
            if attack == "successor_during_select" and phase == 0:
                nested_callback = lambda: successor_lazy.resolve("select0")
            elif attack == "reward_during_select" and phase == 0:
                nested_callback = lambda: reward_lazy.resolve("select0")
            elif attack == "reentrant_select_environment_step" and phase == 0:
                nested_callback = lambda: step_guard.enter(
                    spec.key, phase, prefix
                )
            if nested_callback is not None:
                reject_attack(
                    lambda: selector.select(
                        _policy_input(observation), nested_callback=nested_callback
                    )
                )
            action = selector.select(_policy_input(observation))
            events.append(f"select{phase}")
            if attack in {
                "invalid_action_successor_resolution",
                "invalid_action_reward_resolution",
            } and phase == 0:
                invalid_action = np.asarray(2, dtype=np.dtype("i1"))
                invalid_action.setflags(write=False)

                def invalid_resolution_boundary() -> object:
                    validated = _validate_action(invalid_action)
                    selected_action["value"] = int(validated)
                    selected_lazy = (
                        successor_lazy
                        if attack == "invalid_action_successor_resolution"
                        else reward_lazy
                    )
                    return selected_lazy.resolve(
                        successor_stage
                        if attack == "invalid_action_successor_resolution"
                        else reward_stage
                    )

                reject_attack(invalid_resolution_boundary)
            if attack == "successor_before_action_validation" and phase == 0:
                reject_attack(
                    lambda: successor_lazy.resolve("before_action_validation")
                )
            _validate_action(action)
            events.append(f"validate{phase}")
            action_int = int(action)
            selected_action["value"] = action_int
            episode_actions.append(action_int)
            authorization = _issue_step_authorization(
                sink, spec, phase, prefix, action, mode
            )
            if attack == "nonterminal_terminal_scalar" and phase == 0:
                reject_attack(lambda: reward_lazy.resolve("resolve_terminal3"))
            if attack == "terminal_scalar_after_extra_transition" and phase == 3:
                events.append("extra_transition")
                reject_attack(
                    lambda: terminal_reward_lazy.resolve("after_extra_transition")
                )
            successor_prefix = (prefix << 1) | action_int
            successor = successor_lazy.resolve(successor_stage)
            if phase < HORIZON - 1:
                events.append(successor_stage)
            if attack == "missing_nonterminal_zero" and phase == 0:
                incomplete_records = _expected_runtime_records(
                    spec,
                    phase,
                    prefix,
                    action,
                    mode,
                    observation,
                    successor,
                    _immutable_scalar(0.0, "<f8"),
                )
                reject_attack(
                    lambda: sink.prepare(
                        incomplete_records[0],
                        incomplete_records[1],
                        incomplete_records[2],
                        None,  # type: ignore[arg-type]
                        authorization=authorization,
                    )
                )
            update_reward = reward_lazy.resolve(reward_stage)
            if attack == "duplicate_terminal_scalar" and phase == 3:
                reject_attack(lambda: reward_lazy.resolve(reward_stage))
            if phase < HORIZON - 1:
                events.append(reward_stage)
            else:
                events.append("resolve_terminal3")
                if mode == "reward_origin_control":
                    if not origin_queue or origin_queue.pop(0) != _reward_origin_spec(spec).key:
                        raise ContractError("reward origin queue lost identity")
                    origin_lazy_attempted += reward_lazy.audit.attempted
                    origin_lazy_permitted += reward_lazy.audit.permitted
                    origin_lazy_early += int(
                        reward_lazy.audit.first_stage != "resolve_terminal3"
                    )
            lazy_attempted += successor_lazy.audit.attempted + reward_lazy.audit.attempted
            lazy_permitted += successor_lazy.audit.permitted + reward_lazy.audit.permitted
            for kind, audit in (("successor", successor_lazy.audit), ("feedback", reward_lazy.audit)):
                lazy_audits.append(
                    {
                        "kind": kind,
                        "attempted": audit.attempted,
                        "permitted": audit.permitted,
                        "first_stage": audit.first_stage,
                        "last_stage": audit.last_stage,
                        "value_sha256": audit.value_sha256,
                    }
                )
            records = _expected_runtime_records(
                spec,
                phase,
                prefix,
                action,
                mode,
                observation,
                successor,
                update_reward,
            )
            sink.prepare(*records, authorization=authorization)
            if attack == "next_transition_before_pending_append" and phase == 0:
                next_action = _immutable_scalar(
                    float(_behavior_actions(int(spec.action_code))[1]), "i1"
                )
                next_prefix = successor_prefix
                next_records = _expected_runtime_records(
                    spec,
                    1,
                    next_prefix,
                    next_action,
                    mode,
                    _copy_observation(successor),
                    _public_successor(_copy_observation(successor), next_action),
                    _immutable_scalar(0.0, "<f8"),
                )
                reject_attack(
                    lambda: sink.prepare(
                        *next_records, authorization=authorization
                    )
                )
            if attack == "nonempty_pending_transition_at_split" and phase == 0:
                reject_attack(lambda: _assert_split_clean(sink, origin_queue))
            sink.append(records[2], records[3])
            if attack == "duplicate_nonterminal_feedback" and phase == 0:
                reject_attack(lambda: sink.append(records[2], records[3]))
            events.append(f"append{phase}")
            step_guard.exit(spec.key, phase, prefix)
            prefix = successor_prefix
            if phase < HORIZON - 1:
                if successor is None:
                    raise ContractError("nonterminal public successor is absent")
                observation = _copy_observation(successor)
        final_feedback_key = _feedback_key(spec, 3, prefix >> 1, prefix & 1)
        sink.close_episode(final_feedback_key)
        if attack == "duplicate_episode_close":
            reject_attack(lambda: sink.close_episode(final_feedback_key))
        selector.close_episode()
        completed_specs += 1
        events.append("close_episode")
    if attack is not None:
        raise ContractError("timing attack was not exercised by collection")
    if origin_queue:
        raise ContractError("reward origin queue survived collection")
    if step_guard._active is not None:  # noqa: SLF001
        raise ContractError("environment step remained active after collection")
    trace = sink.seal()
    repeated_event_order = len(events) == completed_specs * len(EVENT_ORDER) and all(
        tuple(events[offset : offset + len(EVENT_ORDER)]) == EVENT_ORDER
        for offset in range(0, len(events), len(EVENT_ORDER))
    )
    return trace, {
        "completed_episodes": completed_specs,
        "event_order_exact": repeated_event_order,
        "event_count": len(events),
        "lazy_installed": lazy_installed,
        "lazy_attempted": lazy_attempted,
        "lazy_permitted": lazy_permitted,
        "lazy_audit_sha256": _json_sha256(lazy_audits),
        "origin_queue_empty": not origin_queue,
        "origin_lazy_attempted": origin_lazy_attempted,
        "origin_lazy_permitted": origin_lazy_permitted,
        "origin_lazy_early": origin_lazy_early,
        "pending_empty": sink.pending is None,
        "selector_completed": selector.completed_episodes,
    }


class _CountingSource:
    def __init__(
        self,
        specs: Sequence[EpisodeSpec],
        *,
        expected_split: str,
        exploding: bool = False,
    ) -> None:
        if expected_split not in {"train", "validation", "test"}:
            raise ContractError("source received an unknown split contract")
        self._specs = tuple(specs)
        self._expected_split = expected_split
        self._source_sha256 = _source_specs_sha256(self._specs)
        self._exploding = exploding
        self._closed = False
        self._active = False
        self._handles: tuple[_SourceEpisodeHandle, ...] = ()
        self._session_nonce: object | None = None
        self._counts = {
            "factory": 0,
            "iterator": 0,
            "materializer": 0,
            "close": 0,
            "post_close": 0,
        }

    @property
    def source_sha256(self) -> str:
        if type(self) is not _CountingSource or self._source_sha256 != _source_specs_sha256(
            self._specs
        ):
            raise ContractError("source commitment changed")
        return self._source_sha256

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._counts))

    @property
    def committed_specs(self) -> tuple[EpisodeSpec, ...]:
        if not self._closed:
            raise ContractError("source specifications requested before close")
        if self._source_sha256 != _source_specs_sha256(self._specs):
            raise ContractError("closed source commitment changed")
        return self._specs

    def _handle_capability_sha256(
        self, ordinal: int, session_nonce: object
    ) -> str:
        return _json_sha256(
            {
                "source_identity": id(self),
                "session_identity": id(session_nonce),
                "source_sha256": self._source_sha256,
                "expected_split": self._expected_split,
                "ordinal": ordinal,
            }
        )

    def open(self, expected_split: str) -> Iterator[_SourceEpisodeHandle]:
        if self._closed:
            self._counts["post_close"] += 1
            raise ContractError("source opened after close")
        if self._active:
            raise ContractError("source opened while already active")
        self._counts["factory"] += 1
        if self._exploding:
            raise ContractError("exploding held-out source was accessed")
        if (
            type(expected_split) is not str
            or expected_split != self._expected_split
            or self._source_sha256 != _expected_source_sha256(expected_split)
            or self._specs != tuple(_iter_episode_specs(expected_split))
        ):
            raise ContractError("source keyset, order, or specification changed")
        self._counts["iterator"] += 1
        self._active = True
        self._session_nonce = object()
        self._handles = tuple(
            _SourceEpisodeHandle(
                self,
                spec,
                ordinal,
                self._session_nonce,
                self._handle_capability_sha256(ordinal, self._session_nonce),
            )
            for ordinal, spec in enumerate(self._specs)
        )
        return iter(self._handles)

    def _materialize(self, handle: _SourceEpisodeHandle, phase: int) -> None:
        if self._closed or not self._active:
            self._counts["post_close"] += 1
            raise ContractError("source row materialized outside its live boundary")
        if (
            type(handle) is not _SourceEpisodeHandle
            or handle._source is not self  # noqa: SLF001
            or handle._ordinal not in range(len(self._handles))  # noqa: SLF001
            or self._handles[handle._ordinal] is not handle  # noqa: SLF001
            or self._specs[handle._ordinal] != handle._spec  # noqa: SLF001
            or handle._session_nonce is not self._session_nonce  # noqa: SLF001
            or handle._capability_sha256  # noqa: SLF001
            != self._handle_capability_sha256(
                handle._ordinal, handle._session_nonce  # noqa: SLF001
            )
            or phase not in range(HORIZON)
        ):
            raise ContractError("source row handle lost its committed identity")
        self._counts["materializer"] += 1

    def close(self) -> None:
        if self._closed:
            raise ContractError("source closed twice")
        self._closed = True
        self._active = False
        self._counts["close"] += 1


def _copy_observation(value: np.ndarray) -> np.ndarray:
    copied = np.array(
        _validate_observation(value), dtype=np.dtype("<f8"), order="C", copy=True
    )
    copied.setflags(write=False)
    return copied


def _copy_action(value: np.ndarray) -> np.ndarray:
    action = _validate_action(value)
    copied = np.asarray(int(action), dtype=np.dtype("i1"))
    copied.setflags(write=False)
    return copied


def _copy_reward(value: np.ndarray) -> np.ndarray:
    reward = _validate_reward(value)
    copied = np.asarray(float(reward), dtype=np.dtype("<f8"))
    copied.setflags(write=False)
    return copied


def _unique_records(
    records: Sequence[object],
    key: Callable[[object], object],
    label: str,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for record in records:
        try:
            identity = key(record)
        except (AttributeError, TypeError) as error:
            raise ContractError(f"malformed {label} component") from error
        try:
            if identity in mapping:
                raise ContractError(f"duplicate {label} component")
            mapping[identity] = record
        except (TypeError, ValueError) as error:
            raise ContractError(f"unhashable {label} identity") from error
    return mapping


def _trace_projection(trace: TraceBundle) -> dict[str, object]:
    return {
        "observations": [
            {
                "key": _key_projection(record.key),
                "value": _json_ready(record.value),
                "value_sha256": record.value_sha256,
            }
            for record in trace.observations
        ],
        "actions": [
            {
                "key": _key_projection(record.key),
                "observation_key": _key_projection(record.observation_key),
                "value": _json_ready(record.value),
            }
            for record in trace.actions
        ],
        "transitions": [
            {
                "key": _key_projection(record.key),
                "action_key": _key_projection(record.action_key),
                "predecessor_key": _key_projection(record.predecessor_key),
                "predecessor_sha256": record.predecessor_sha256,
                "successor_key": (
                    None
                    if record.successor_key is None
                    else _key_projection(record.successor_key)
                ),
                "successor_sha256": record.successor_sha256,
                "source_episode": _key_projection(record.source_episode),
                "donor_episode": (
                    None
                    if record.donor_episode is None
                    else _key_projection(record.donor_episode)
                ),
                "donor_payload_sha256": record.donor_payload_sha256,
                "done": record.done,
            }
            for record in trace.transitions
        ],
        "feedback": [
            {
                "key": _key_projection(record.key),
                "transition_key": _key_projection(record.transition_key),
                "canonical_reward": _json_ready(record.canonical_reward),
                "update_reward": _json_ready(record.update_reward),
                "origin_episode": _key_projection(record.origin_episode),
                "done": record.done,
            }
            for record in trace.feedback
        ],
        "closes": [
            {
                "episode_key": _key_projection(record.episode_key),
                "final_feedback_key": _key_projection(record.final_feedback_key),
                "record_count": {
                    "dtype": "int8",
                    "value": int(record.record_count),
                },
            }
            for record in trace.closes
        ],
    }


def _trace_sha256(trace: TraceBundle) -> str:
    return _json_sha256(_trace_projection(trace))


def _validate_trace(
    trace: TraceBundle,
    specs: Sequence[EpisodeSpec],
    *,
    mode: str,
) -> _AuthenticatedRows:
    if mode not in {
        "canonical", "transition_control", "reward_origin_control",
        "signal_ablation",
    }:
        raise ContractError("trace validator received an unknown mode")
    if type(trace) is not TraceBundle:
        raise ContractError("trace has the wrong container type")
    for record in trace.observations:
        if type(record) is not ObservationRecord:
            raise ContractError("observation component has the wrong type")
        _validate_observation_key(record.key)
    for record in trace.actions:
        if type(record) is not ActionRecord:
            raise ContractError("action component has the wrong type")
        _validate_action_key(record.key)
        _validate_observation_key(record.observation_key)
    for record in trace.transitions:
        if type(record) is not TransitionRecord:
            raise ContractError("transition component has the wrong type")
        _validate_transition_key(record.key)
        _validate_action_key(record.action_key)
        _validate_observation_key(record.predecessor_key)
        if record.successor_key is not None:
            _validate_observation_key(record.successor_key)
        _validate_episode_key(record.source_episode)
        if record.donor_episode is not None:
            _validate_episode_key(record.donor_episode)
            if type(record.donor_payload_sha256) is not str:
                raise ContractError("comparator donor digest changed")
            if (
                len(record.donor_payload_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in record.donor_payload_sha256
                )
            ):
                raise ContractError("transition donor digest changed")
        elif record.donor_payload_sha256 is not None:
            raise ContractError("canonical transition acquired a donor digest")
        if type(record.done) is not bool:
            raise ContractError("transition done flag has the wrong type")
    for record in trace.feedback:
        if type(record) is not FeedbackRecord:
            raise ContractError("feedback component has the wrong type")
        _validate_feedback_key(record.key)
        _validate_transition_key(record.transition_key)
        _validate_episode_key(record.origin_episode)
        if type(record.done) is not bool:
            raise ContractError("feedback done flag has the wrong type")
    for record in trace.closes:
        if type(record) is not EpisodeCloseRecord:
            raise ContractError("close component has the wrong type")
        _validate_episode_key(record.episode_key)
        _validate_feedback_key(record.final_feedback_key)
        if type(record.record_count) is not np.int8:
            raise ContractError("close record count has the wrong dtype")
    observation_map = _unique_records(
        trace.observations, lambda item: item.key, "observation"
    )
    action_map = _unique_records(trace.actions, lambda item: item.key, "action")
    transition_map = _unique_records(
        trace.transitions, lambda item: item.key, "transition"
    )
    feedback_map = _unique_records(
        trace.feedback, lambda item: item.key, "feedback"
    )
    close_map = _unique_records(trace.closes, lambda item: item.episode_key, "close")
    expected_record_count = len(specs) * HORIZON
    if not all(
        len(collection) == expected_record_count
        for collection in (
            observation_map,
            action_map,
            transition_map,
            feedback_map,
        )
    ) or len(close_map) != len(specs):
        raise ContractError("trace component count changed")

    rows: list[TDInputRow] = []
    for spec in specs:
        prefix = 0
        actions = _behavior_actions(int(spec.action_code))
        expected_observation = _runtime_observation(spec, 0, 0, mode)
        for phase, expected_action in enumerate(actions):
            obs_key = _observation_key(spec, phase, prefix)
            act_key = ActionKey(obs_key, np.int8(expected_action))
            trans_key = TransitionKey(act_key, np.int8(0))
            feed_key = FeedbackKey(trans_key, np.int8(0))
            _validate_observation_key(obs_key)
            _validate_action_key(act_key)
            _validate_transition_key(trans_key)
            _validate_feedback_key(feed_key)
            try:
                obs_record = observation_map[obs_key]
                action_record = action_map[act_key]
                transition_record = transition_map[trans_key]
                feedback_record = feedback_map[feed_key]
            except KeyError as error:
                raise ContractError(
                    "trace component identity is missing or unknown"
                ) from error
            if type(obs_record) is not ObservationRecord:
                raise ContractError("observation component has the wrong type")
            observation = _validate_observation(obs_record.value)
            if (
                observation.tobytes() != expected_observation.tobytes()
                or obs_record.value_sha256 != _observation_sha256(observation)
                or obs_record.key != obs_key
            ):
                raise ContractError("observation component failed authentication")
            if type(action_record) is not ActionRecord:
                raise ContractError("action component has the wrong type")
            action = _validate_action(action_record.value)
            if (
                action_record.observation_key != obs_key
                or action_record.key != act_key
                or int(action) != expected_action
            ):
                raise ContractError("action component failed authentication")
            done = phase == HORIZON - 1
            if type(transition_record) is not TransitionRecord:
                raise ContractError("transition component has the wrong type")
            if type(transition_record.done) is not bool or transition_record.done != done:
                raise ContractError("transition done flag failed strict authentication")
            successor_prefix = (prefix << 1) | expected_action
            successor_key = (
                None
                if done
                else _observation_key(spec, phase + 1, successor_prefix)
            )
            successor = (
                None
                if done
                else (
                    _transition_control_successor(
                        observation,
                        action,
                        donor_initial_observation=_primary_observation(
                            _paired_spec(spec), 0, 0
                        ),
                    )
                    if mode == "transition_control" and phase == 0
                    else _public_successor(observation, action)
                )
            )
            expected_donor = (
                _paired_spec(spec).key
                if mode == "transition_control" and not done
                else None
            )
            if (
                transition_record.key != trans_key
                or transition_record.action_key != act_key
                or transition_record.predecessor_key != obs_key
                or transition_record.predecessor_sha256
                != _observation_sha256(observation)
                or transition_record.successor_key != successor_key
                or transition_record.successor_sha256
                != (None if successor is None else _observation_sha256(successor))
                or transition_record.source_episode != spec.key
                or transition_record.donor_episode != expected_donor
                or transition_record.donor_payload_sha256
                != (
                    _observation_sha256(successor)
                    if expected_donor is not None and successor is not None
                    else None
                )
            ):
                raise ContractError("transition component failed authentication")
            if transition_record.donor_episode is not None:
                _validate_episode_key(transition_record.donor_episode)
            if type(feedback_record) is not FeedbackRecord:
                raise ContractError("feedback component has the wrong type")
            canonical_reward = _validate_reward(feedback_record.canonical_reward)
            update_reward = _validate_reward(feedback_record.update_reward)
            expected_canonical = _episode_terminal_reward(spec) if done else 0.0
            if mode == "reward_origin_control" and done:
                expected_origin = _reward_origin_spec(spec)
                expected_update = _episode_terminal_reward(expected_origin)
                expected_origin_key = expected_origin.key
            else:
                expected_update = expected_canonical
                expected_origin_key = spec.key
            if (
                feedback_record.key != feed_key
                or feedback_record.transition_key != trans_key
                or type(feedback_record.done) is not bool
                or feedback_record.done != done
                or float(canonical_reward) != expected_canonical
                or float(update_reward) != expected_update
                or feedback_record.origin_episode != expected_origin_key
            ):
                raise ContractError("feedback component failed authentication")
            _validate_episode_key(feedback_record.origin_episode)
            rows.append(
                TDInputRow(
                    _copy_observation(observation),
                    _copy_action(action),
                    None if successor is None else _copy_observation(successor),
                    _copy_reward(update_reward),
                    bool(done),
                )
            )
            prefix = successor_prefix
            if not done:
                if successor is None:
                    raise ContractError("validated transition lost its successor")
                expected_observation = _copy_observation(successor)
        close = close_map.get(spec.key)
        expected_final = _feedback_key(spec, 3, prefix >> 1, prefix & 1)
        if (
            type(close) is not EpisodeCloseRecord
            or close.episode_key != spec.key
            or close.final_feedback_key != expected_final
            or type(close.record_count) is not np.int8
            or int(close.record_count) != HORIZON
        ):
            raise ContractError("episode close component failed authentication")
    frozen_rows = tuple(rows)
    rows_sha256 = _td_rows_sha256(frozen_rows)
    source_sha256 = _source_specs_sha256(tuple(specs))
    trace_sha256 = _trace_sha256(trace)
    token_projection = {
        "mode": mode,
        "rows_sha256": rows_sha256,
        "source_sha256": source_sha256,
        "trace_sha256": trace_sha256,
    }
    token = _TraceGateToken(
        trace_sha256=trace_sha256,
        rows_sha256=rows_sha256,
        source_sha256=source_sha256,
        mode=mode,
        token_sha256=_json_sha256(token_projection),
    )
    _ISSUED_TRACE_GATES.add(token.token_sha256)
    return _AuthenticatedRows(frozen_rows, token)


def _rotate(values: tuple[object, ...], amount: int) -> tuple[object, ...]:
    if not values:
        return values
    offset = amount % len(values)
    return values[offset:] + values[:offset]


def _independently_reordered_trace(trace: TraceBundle) -> TraceBundle:
    return TraceBundle(
        tuple(reversed(trace.observations)),
        _rotate(trace.actions, 3),
        _rotate(trace.transitions, 5),
        _rotate(trace.feedback, 7),
        tuple(reversed(_rotate(trace.closes, 11))),
    )


def _td_rows_sha256(rows: Sequence[TDInputRow]) -> str:
    return _json_sha256(
        [
            {
                "predecessor": _json_ready(row.predecessor),
                "action": _json_ready(row.action),
                "successor": (
                    None if row.successor is None else _json_ready(row.successor)
                ),
                "update_reward": _json_ready(row.update_reward),
                "done": row.done,
            }
            for row in rows
        ]
    )


def _comparator_feedback_sha256(
    rows: Sequence[ComparatorFeedbackStep],
) -> str:
    projection: list[dict[str, object]] = []
    for row in rows:
        projection.append(
            {
                "public_predecessor": _json_ready(row.predecessor),
                "selected_action": _json_ready(row.action),
                "public_successor": (
                    None if row.successor is None else _json_ready(row.successor)
                ),
                "bare_feedback": _json_ready(row.update_reward),
                "terminal": row.done,
            }
        )
    return _json_sha256(projection)


def _comparator_td_source_sha256(rows: Sequence[TDInputRow]) -> str:
    # Deliberately independent from _td_rows_sha256 and its projection helper.
    projection: list[dict[str, object]] = []
    for row in rows:
        if type(row) is not TDInputRow:
            raise ContractError("comparator source row type changed")
        projection.append(
            {
                "public_predecessor": _json_ready(row.predecessor),
                "selected_action": _json_ready(row.action),
                "public_successor": (
                    None if row.successor is None else _json_ready(row.successor)
                ),
                "bare_feedback": _json_ready(row.update_reward),
                "terminal": row.done,
            }
        )
    return _json_sha256(projection)


def _trace_score(
    trace: TraceBundle, specs: Sequence[EpisodeSpec], *, mode: str
) -> tuple[float, int, str]:
    _validate_trace(trace, specs, mode=mode)
    action_map = _unique_records(
        trace.actions, lambda item: item.observation_key, "action by observation"
    )
    total = 0.0
    for spec in specs:
        actions: list[int] = []
        prefix = 0
        for phase in range(HORIZON):
            key = _observation_key(spec, phase, prefix)
            record = action_map[key]
            action = int(_validate_action(record.value))
            actions.append(action)
            prefix = (prefix << 1) | action
        total += _terminal_reward(int(spec.target), actions)
    count = len(specs)
    return total / count, count - int(total), _trace_sha256(trace)


def _seal_comparator_feedback(
    trace: TraceBundle,
    authenticated: _AuthenticatedRows,
) -> ComparatorFeedbackBatch:
    """Seal only public behavior feedback before comparator construction."""
    batch = _require_authenticated_train_rows(
        authenticated, allowed_modes={"canonical"}
    )
    if batch.token.trace_sha256 != _trace_sha256(trace):
        raise ContractError("comparator trace token changed")
    expected_count = EPISODE_COUNTS["train"] * HORIZON
    if not all(
        len(component) == expected_count
        for component in (
            trace.observations,
            trace.actions,
            trace.transitions,
            trace.feedback,
        )
    ) or len(trace.closes) != EPISODE_COUNTS["train"]:
        raise ContractError("comparator component count changed")
    observations = _unique_records(
        trace.observations, lambda item: item.key, "comparator observation"
    )
    actions = _unique_records(
        trace.actions, lambda item: item.observation_key, "comparator action"
    )
    transitions = _unique_records(
        trace.transitions, lambda item: item.action_key, "comparator transition"
    )
    feedback = _unique_records(
        trace.feedback, lambda item: item.transition_key, "comparator feedback"
    )
    closes = _unique_records(
        trace.closes, lambda item: item.episode_key, "comparator close"
    )
    episode_keys = sorted(
        closes,
        key=lambda key: (
            int(key.split_enum), int(key.regime_code), int(key.episode)
        ),
    )
    rows: list[ComparatorFeedbackStep] = []
    for episode_key in episode_keys:
        _validate_episode_key(episode_key)
        prefix = 0
        final_feedback_key: FeedbackKey | None = None
        for phase in range(HORIZON):
            observation_key = ObservationKey(
                episode_key, np.int8(phase), np.int8(prefix)
            )
            observation_record = observations.get(observation_key)
            action_record = actions.get(observation_key)
            if (
                type(observation_record) is not ObservationRecord
                or type(action_record) is not ActionRecord
            ):
                raise ContractError("comparator lost a feedback-path component")
            observation = _validate_observation(observation_record.value)
            action = _validate_action(action_record.value)
            action_key = ActionKey(observation_key, np.int8(int(action)))
            transition_record = transitions.get(action_key)
            transition_key = TransitionKey(action_key, np.int8(0))
            feedback_record = feedback.get(transition_key)
            if (
                type(transition_record) is not TransitionRecord
                or type(feedback_record) is not FeedbackRecord
            ):
                raise ContractError("comparator lost transition feedback")
            done = phase == HORIZON - 1
            successor_prefix = (prefix << 1) | int(action)
            successor_key = (
                None
                if done
                else ObservationKey(
                    episode_key, np.int8(phase + 1), np.int8(successor_prefix)
                )
            )
            successor_record = (
                None if successor_key is None else observations.get(successor_key)
            )
            successor = (
                None
                if successor_record is None
                else _validate_observation(successor_record.value)
            )
            canonical_reward = _validate_reward(feedback_record.canonical_reward)
            update_reward = _validate_reward(feedback_record.update_reward)
            if (
                observation_record.value_sha256 != _observation_sha256(observation)
                or action_record.key != action_key
                or action_record.observation_key != observation_key
                or transition_record.key != transition_key
                or transition_record.action_key != action_key
                or transition_record.predecessor_key != observation_key
                or transition_record.predecessor_sha256 != _observation_sha256(observation)
                or transition_record.successor_key != successor_key
                or transition_record.successor_sha256
                != (None if successor is None else _observation_sha256(successor))
                or transition_record.source_episode != episode_key
                or transition_record.donor_episode is not None
                or transition_record.donor_payload_sha256 is not None
                or type(transition_record.done) is not bool
                or transition_record.done is not done
                or feedback_record.key != FeedbackKey(transition_key, np.int8(0))
                or feedback_record.transition_key != transition_key
                or feedback_record.origin_episode != episode_key
                or type(feedback_record.done) is not bool
                or feedback_record.done is not done
                or float(canonical_reward) != float(update_reward)
                or (not done and float(update_reward) != 0.0)
            ):
                raise ContractError("comparator feedback-only authentication failed")
            rows.append(
                ComparatorFeedbackStep(
                    _copy_observation(observation),
                    _copy_action(action),
                    None if successor is None else _copy_observation(successor),
                    _copy_reward(update_reward),
                    done,
                )
            )
            prefix = successor_prefix
            final_feedback_key = feedback_record.key
        close = closes[episode_key]
        if (
            type(close) is not EpisodeCloseRecord
            or close.final_feedback_key != final_feedback_key
            or type(close.record_count) is not np.int8
            or int(close.record_count) != HORIZON
        ):
            raise ContractError("comparator close authentication failed")
    materialized = tuple(rows)
    feedback_sha256 = _comparator_feedback_sha256(materialized)
    if feedback_sha256 != _comparator_td_source_sha256(batch.rows):
        raise ContractError("comparator feedback projection changed")
    token_projection = {
        "feedback_sha256": feedback_sha256,
        "source_sha256": batch.token.source_sha256,
        "trace_sha256": batch.token.trace_sha256,
    }
    token = _ComparatorGateToken(
        feedback_sha256,
        batch.token.source_sha256,
        batch.token.trace_sha256,
        _json_sha256(token_projection),
    )
    _ISSUED_COMPARATOR_GATES.add(token.token_sha256)
    return ComparatorFeedbackBatch(materialized, token)


def _comparator_materialize(
    batch: ComparatorFeedbackBatch,
) -> _AuthenticatedComparatorRows:
    """Independently revalidate a public-only sealed comparator batch."""

    if (
        type(batch) is not ComparatorFeedbackBatch
        or type(batch.rows) is not tuple
        or type(batch.token) is not _ComparatorGateToken
        or any(type(row) is not ComparatorFeedbackStep for row in batch.rows)
        or len(batch.rows) != EPISODE_COUNTS["train"] * HORIZON
        or batch.token.token_sha256 not in _ISSUED_COMPARATOR_GATES
        or batch.token.feedback_sha256 != _comparator_feedback_sha256(batch.rows)
        or batch.token.source_sha256 != _expected_source_sha256("train")
        or batch.token.token_sha256
        != _json_sha256(
            {
                "feedback_sha256": batch.token.feedback_sha256,
                "source_sha256": batch.token.source_sha256,
                "trace_sha256": batch.token.trace_sha256,
            }
        )
    ):
        raise ContractError("comparator received an unsealed public batch")
    rows: list[ComparatorRow] = []
    for episode_offset in range(0, len(batch.rows), HORIZON):
        expected_observation: np.ndarray | None = None
        for phase, step in enumerate(
            batch.rows[episode_offset : episode_offset + HORIZON]
        ):
            predecessor = _validate_observation(step.predecessor)
            action = _validate_action(step.action)
            update_reward = _validate_reward(step.update_reward)
            done = phase == HORIZON - 1
            if type(step.done) is not bool or step.done is not done:
                raise ContractError("comparator public feedback timing changed")
            if int(predecessor[0]) != phase:
                raise ContractError("comparator public phase order changed")
            if expected_observation is not None and (
                predecessor.tobytes(order="C")
                != expected_observation.tobytes(order="C")
            ):
                raise ContractError("comparator public successor link changed")
            expected_successor = _public_successor(predecessor, action)
            if done:
                if step.successor is not None or expected_successor is not None:
                    raise ContractError("comparator terminal successor changed")
            else:
                if step.successor is None or expected_successor is None:
                    raise ContractError("comparator nonterminal successor is absent")
                if (
                    _validate_observation(step.successor).tobytes(order="C")
                    != expected_successor.tobytes(order="C")
                ):
                    raise ContractError("comparator public transition changed")
                if float(update_reward) != 0.0:
                    raise ContractError("comparator saw early feedback")
            rows.append(
                ComparatorRow(
                    _copy_observation(predecessor),
                    _copy_action(action),
                    None
                    if step.successor is None
                    else _copy_observation(step.successor),
                    _copy_reward(update_reward),
                    done,
                )
            )
            expected_observation = expected_successor
    materialized = tuple(rows)
    independent_projection = tuple(
        ComparatorFeedbackStep(
            row.predecessor,
            row.action,
            row.successor,
            row.update_reward,
            row.done,
        )
        for row in materialized
    )
    if (
        _comparator_feedback_sha256(independent_projection)
        != batch.token.feedback_sha256
    ):
        raise ContractError("comparator materializer changed its public projection")
    return _AuthenticatedComparatorRows(materialized, batch.token)


def _group_td_rows(
    rows: Sequence[TDInputRow | ComparatorRow],
) -> dict[tuple[int, int, int, int], list[TDInputRow | ComparatorRow]]:
    groups: dict[
        tuple[int, int, int, int], list[TDInputRow | ComparatorRow]
    ] = defaultdict(list)
    for row in rows:
        if type(row.done) is not bool:
            raise ContractError("TD row done flag has the wrong type")
        observation = _validate_observation(row.predecessor)
        action = _validate_action(row.action)
        reward = _validate_reward(row.update_reward)
        state = _state_key(observation)
        if row.done:
            if row.successor is not None or state[1] != 3:
                raise ContractError("terminal TD row has a successor")
        else:
            if row.successor is None or state[1] == 3 or float(reward) != 0.0:
                raise ContractError("nonterminal TD row is malformed")
            successor_state = _state_key(row.successor)
            if successor_state[1] != state[1] + 1:
                raise ContractError("TD successor phase is not adjacent")
        groups[(*state, int(action))].append(row)
    return groups


def _terminal_means(
    groups: Mapping[
        tuple[int, int, int, int], Sequence[TDInputRow | ComparatorRow]
    ]
) -> tuple[Mapping[tuple[int, int, int, int], np.float64], int]:
    means: dict[tuple[int, int, int, int], np.float64] = {}
    raw_reads = 0
    for key in ALL_STATE_ACTION_KEYS:
        if key[1] != 3:
            continue
        records = groups.get(key, ())
        if len(records) != 64:
            raise ContractError("terminal state/action cell lost exact coverage")
        values = []
        for record in records:
            values.append(float(_validate_reward(record.update_reward)))
            raw_reads += 1
        means[key] = np.float64(sum(values) / len(values))
    if len(means) != 32 or raw_reads != 2048:
        raise ContractError("terminal aggregation count changed")
    return MappingProxyType(means), raw_reads


def _strip_kernel_inputs(
    groups: Mapping[
        tuple[int, int, int, int], Sequence[TDInputRow | ComparatorRow]
    ],
    *,
    expected_coverage: Mapping[tuple[int, int, int, int], int] | None = None,
) -> tuple[
    Mapping[tuple[int, int, int, int], tuple[BootstrapRow, ...]],
    Mapping[tuple[int, int, int, int], int],
]:
    bootstrap: dict[tuple[int, int, int, int], tuple[BootstrapRow, ...]] = {}
    coverage: dict[tuple[int, int, int, int], int] = {}
    expected = (
        {
            key: {0: 512, 1: 256, 2: 128, 3: 64}[key[1]]
            for key in ALL_STATE_ACTION_KEYS
        }
        if expected_coverage is None
        else dict(expected_coverage)
    )
    if set(expected) != set(ALL_STATE_ACTION_KEYS) or any(
        type(count) is not int or count < 0 for count in expected.values()
    ):
        raise ContractError("TD expected coverage table changed")
    if set(groups) != {key for key, count in expected.items() if count > 0}:
        raise ContractError("TD group key set changed")
    for key in ALL_STATE_ACTION_KEYS:
        records = tuple(groups.get(key, ()))
        expected_count = expected[key]
        if len(records) != expected_count:
            raise ContractError("state/action coverage changed before stripping")
        coverage[key] = expected_count
        if key[1] == 3:
            if any(not record.done or record.successor is not None for record in records):
                raise ContractError("terminal records crossed the strip boundary")
            continue
        stripped: list[BootstrapRow] = []
        for record in records:
            if record.done or record.successor is None:
                raise ContractError("nonterminal record crossed the strip boundary")
            if float(_validate_reward(record.update_reward)) != 0.0:
                raise ContractError("nonterminal scalar crossed the strip boundary")
            stripped.append(
                BootstrapRow(
                    _copy_observation(record.predecessor),
                    _copy_action(record.action),
                    _copy_observation(record.successor),
                )
            )
        bootstrap[key] = tuple(stripped)
    return MappingProxyType(bootstrap), MappingProxyType(coverage)


def _run_td_kernel(
    bootstrap_groups: Mapping[
        tuple[int, int, int, int], Sequence[BootstrapRow]
    ],
    terminal_means: Mapping[tuple[int, int, int, int], np.float64],
    coverage: Mapping[tuple[int, int, int, int], int],
    *,
    sweeps: int = 4,
) -> FitResult:
    if sweeps not in (1, 4):
        raise ContractError("TD sweep count is outside the frozen contracts")
    terminal_keys = {key for key in ALL_STATE_ACTION_KEYS if key[1] == 3}
    nonterminal_keys = set(ALL_STATE_ACTION_KEYS) - terminal_keys
    if set(terminal_means) != terminal_keys or set(bootstrap_groups) != nonterminal_keys:
        raise ContractError("stripped TD kernel input key set changed")
    if set(coverage) != set(ALL_STATE_ACTION_KEYS):
        raise ContractError("TD kernel coverage table changed")
    table = {key: np.float64(0.0) for key in ALL_STATE_ACTION_KEYS}
    snapshots: list[dict[tuple[int, int, int, int], float]] = []
    writes: list[int] = []
    positives: list[int] = []
    aggregate_lookups: list[int] = []
    for _sweep in range(sweeps):
        previous = MappingProxyType(dict(table))
        next_table: dict[tuple[int, int, int, int], np.float64] = {}
        lookup_count = 0
        for key in ALL_STATE_ACTION_KEYS:
            sign, phase, prefix, _action = key
            records = bootstrap_groups.get(key, ())
            expected_count = coverage[key]
            if type(expected_count) is not int or expected_count < 0:
                raise ContractError("state/action cell lost exact train coverage")
            if phase == 3:
                if records:
                    raise ContractError("terminal raw rows reached the sweep kernel")
                if key not in terminal_means:
                    raise ContractError("terminal aggregate is missing")
                target = np.float64(terminal_means[key])
                lookup_count += 1
            else:
                values = []
                for record in records:
                    next_sign, next_phase, next_prefix = _state_key(record.successor)
                    if next_phase != phase + 1:
                        raise ContractError("bootstrap target skipped a boundary")
                    values.append(
                        max(
                            previous[(next_sign, next_phase, next_prefix, 0)],
                            previous[(next_sign, next_phase, next_prefix, 1)],
                        )
                    )
                if len(values) != expected_count:
                    raise ContractError("bootstrap row count changed")
                target = (
                    np.float64(0.0)
                    if not values
                    else np.float64(sum(values) / len(values))
                )
            next_table[(sign, phase, prefix, key[3])] = target
        if len(next_table) != 60:
            raise ContractError("synchronous sweep wrote the wrong cell count")
        table = next_table
        snapshots.append(dict(table))
        writes.append(len(next_table))
        positives.append(sum(value > 0.0 for value in table.values()))
        aggregate_lookups.append(lookup_count)
    return FitResult(
        table,
        FitAudit(
            raw_terminal_reads=0,
            aggregate_lookups_by_sweep=tuple(aggregate_lookups),
            writes_by_sweep=tuple(writes),
            positive_cells_by_sweep=tuple(positives),
            snapshots=tuple(snapshots),
        ),
        MappingProxyType(dict(terminal_means)),
    )


def _require_authenticated_train_rows(
    batch: object,
    *,
    allowed_modes: set[str],
) -> _AuthenticatedRows:
    if (
        type(batch) is not _AuthenticatedRows
        or type(batch.rows) is not tuple
        or type(batch.token) is not _TraceGateToken
        or any(type(row) is not TDInputRow for row in batch.rows)
    ):
        raise ContractError("TD fitter received an unauthenticated row batch")
    token = batch.token
    if (
        token.token_sha256 not in _ISSUED_TRACE_GATES
        or token.mode not in allowed_modes
        or token.rows_sha256 != _td_rows_sha256(batch.rows)
        or token.source_sha256 != _expected_source_sha256("train")
        or len(batch.rows) != EPISODE_COUNTS["train"] * HORIZON
        or token.token_sha256
        != _json_sha256(
            {
                "mode": token.mode,
                "rows_sha256": token.rows_sha256,
                "source_sha256": token.source_sha256,
                "trace_sha256": token.trace_sha256,
            }
        )
    ):
        raise ContractError("TD fitter row authentication failed")
    sealed_rows = tuple(
        TDInputRow(
            _copy_observation(row.predecessor),
            _copy_action(row.action),
            None if row.successor is None else _copy_observation(row.successor),
            _copy_reward(row.update_reward),
            row.done,
        )
        for row in batch.rows
    )
    if _td_rows_sha256(sealed_rows) != token.rows_sha256:
        raise ContractError("TD fitter copy changed the authenticated projection")
    return _AuthenticatedRows(sealed_rows, token)


def _fit_td(
    rows: _AuthenticatedRows,
    gate: _FamilyGateToken,
    *,
    reentrant_attack: bool = False,
) -> FitResult:
    _require_family_gate(gate)
    rows = _require_authenticated_train_rows(
        rows,
        allowed_modes={"canonical", "transition_control", "reward_origin_control"},
    )
    _CONSTRUCTION_COUNTS["learner"] += 1
    guard = _UpdateGuard()
    result: list[FitResult] = []

    def fit_body() -> None:
        if reentrant_attack:
            guard.run()
        groups = _group_td_rows(rows)
        terminal_means, raw_reads = _terminal_means(groups)
        bootstrap, coverage = _strip_kernel_inputs(groups)
        fitted = _run_td_kernel(bootstrap, terminal_means, coverage, sweeps=4)
        result.append(
            FitResult(
                fitted.table,
                replace(fitted.audit, raw_terminal_reads=raw_reads),
                fitted.terminal_means,
            )
        )

    guard.run(fit_body)
    if len(result) != 1:
        raise ContractError("TD fitter did not seal exactly one learner state")
    return result[0]


def _fit_signal_ablation_td(
    rows: _AuthenticatedRows, gate: _FamilyGateToken
) -> FitResult:
    _require_family_gate(gate)
    rows = _require_authenticated_train_rows(
        rows, allowed_modes={"signal_ablation"}
    )
    _CONSTRUCTION_COUNTS["learner"] += 1
    groups = _group_td_rows(rows)
    expected_coverage = {
        key: (
            0
            if key[0] == 1
            else 2 * {0: 512, 1: 256, 2: 128, 3: 64}[key[1]]
        )
        for key in ALL_STATE_ACTION_KEYS
    }
    terminal_means: dict[tuple[int, int, int, int], np.float64] = {}
    raw_reads = 0
    for key in ALL_STATE_ACTION_KEYS:
        if key[1] != 3:
            continue
        records = tuple(groups.get(key, ()))
        if len(records) != expected_coverage[key]:
            raise ContractError("signal-ablation terminal coverage changed")
        values = [float(_validate_reward(row.update_reward)) for row in records]
        raw_reads += len(values)
        terminal_means[key] = np.float64(
            0.0 if not values else sum(values) / len(values)
        )
    if raw_reads != 2048:
        raise ContractError("signal-ablation raw terminal read count changed")
    bootstrap, coverage = _strip_kernel_inputs(
        groups, expected_coverage=expected_coverage
    )
    fitted = _run_td_kernel(
        bootstrap, MappingProxyType(terminal_means), coverage, sweeps=4
    )
    return FitResult(
        fitted.table,
        replace(fitted.audit, raw_terminal_reads=raw_reads),
        fitted.terminal_means,
    )


def _table_sha256(table: Mapping[tuple[int, int, int, int], float]) -> str:
    return _json_sha256(
        [
            {"key": list(key), "value": float(table[key])}
            for key in ALL_STATE_ACTION_KEYS
        ]
    )


def _greedy_action(
    table: Mapping[tuple[int, int, int, int], float], observation: np.ndarray
) -> int:
    sign, phase, prefix = _state_key(observation)
    left = float(table[(sign, phase, prefix, 0)])
    right = float(table[(sign, phase, prefix, 1)])
    return 1 if right > left else 0


def _evaluate_policy(
    table: Mapping[tuple[int, int, int, int], float],
    specs: Sequence[EpisodeSpec],
    *,
    observation_mode: str = "canonical",
) -> dict[str, object]:
    by_regime: dict[int, list[float]] = defaultdict(list)
    stream: list[list[int]] = []
    state_before = _table_sha256(table)
    for spec in specs:
        actions: list[int] = []
        observation = _runtime_observation(spec, 0, 0, observation_mode)
        for phase in range(HORIZON):
            if int(_validate_observation(observation)[0]) != phase:
                raise ContractError("held-out public transition skipped a phase")
            action = _greedy_action(table, observation)
            actions.append(action)
            if phase < HORIZON - 1:
                next_observation = _public_successor(
                    observation, _immutable_scalar(float(action), "i1")
                )
                if next_observation is None:
                    raise ContractError("held-out public successor is absent")
                observation = next_observation
        by_regime[int(spec.regime.code)].append(
            _terminal_reward(int(spec.target), actions)
        )
        stream.append(actions)
    regime_means = [sum(values) / len(values) for values in by_regime.values()]
    state_after = _table_sha256(table)
    return {
        "macro": float(sum(regime_means) / len(regime_means)),
        "minimum_regime": float(min(regime_means)),
        "stream_sha256": _json_sha256(stream),
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "updates": 0,
    }


@dataclass(frozen=True)
class _KeyedEvaluator:
    split: str
    specs: tuple[EpisodeSpec, ...]
    source_sha256: str

    @classmethod
    def frozen(cls, split: str) -> "_KeyedEvaluator":
        specs = tuple(_iter_episode_specs(split))
        return cls(split, specs, _source_specs_sha256(specs))

    def evaluate(self, actions: Sequence[Sequence[int]]) -> dict[str, object]:
        if (
            self.specs != tuple(_iter_episode_specs(self.split))
            or self.source_sha256 != _expected_source_sha256(self.split)
        ):
            raise ContractError("keyed evaluator source changed")
        return _evaluate_action_stream(self.specs, actions)


def _evaluate_action_stream(
    specs: Sequence[EpisodeSpec], actions: Sequence[Sequence[int]]
) -> dict[str, object]:
    if len(specs) != len(actions):
        raise ContractError("baseline action stream length changed")
    by_regime: dict[int, list[float]] = defaultdict(list)
    normalized: list[list[int]] = []
    for spec, episode_actions in zip(specs, actions, strict=True):
        if len(episode_actions) != HORIZON or any(
            type(action) is not int or action not in (0, 1)
            for action in episode_actions
        ):
            raise ContractError("baseline action stream is malformed")
        normalized.append(list(episode_actions))
        by_regime[int(spec.regime.code)].append(
            _terminal_reward(int(spec.target), episode_actions)
        )
    means = [sum(values) / len(values) for values in by_regime.values()]
    return {
        "macro": float(sum(means) / len(means)),
        "minimum_regime": float(min(means)),
        "stream_sha256": _json_sha256(normalized),
    }


def _comparator_greedy_action(
    table: Mapping[tuple[int, int, int, int], float], observation: np.ndarray
) -> int:
    value = _validate_observation(observation)
    phase = int(value[0])
    sign = 0 if float(value[1]) <= 0.0 else 1
    prefix = 0
    for offset in range(phase):
        bit = float(value[2 + offset])
        if bit not in (0.0, 1.0):
            raise ContractError("comparator received an illegal public prefix")
        prefix = (prefix << 1) | int(bit)
    q_zero = float(table[(sign, phase, prefix, 0)])
    q_one = float(table[(sign, phase, prefix, 1)])
    return 1 if q_one > q_zero else 0


def _public_comparator_episodes(
    split: str,
) -> tuple[PublicComparatorEpisode, ...]:
    if split not in {"train", "validation", "test"}:
        raise ContractError("comparator public episode split changed")
    episodes = tuple(
        PublicComparatorEpisode(_primary_observation(spec, 0, 0))
        for spec in _iter_episode_specs(split)
    )
    if len(episodes) != EPISODE_COUNTS[split]:
        raise ContractError("comparator public episode count changed")
    return episodes


def _comparator_action_stream(
    table: Mapping[tuple[int, int, int, int], float],
    episodes: Sequence[PublicComparatorEpisode],
) -> tuple[tuple[int, ...], ...]:
    stream: list[tuple[int, ...]] = []
    for episode in episodes:
        if type(episode) is not PublicComparatorEpisode:
            raise ContractError("comparator received evaluator-bearing episode data")
        observation = _copy_observation(episode.initial_observation)
        actions: list[int] = []
        for phase in range(HORIZON):
            if int(_validate_observation(observation)[0]) != phase:
                raise ContractError("comparator public state skipped a phase")
            action = _comparator_greedy_action(table, observation)
            actions.append(action)
            if phase < HORIZON - 1:
                successor = _public_successor(
                    observation, _immutable_scalar(float(action), "i1")
                )
                if successor is None:
                    raise ContractError("comparator public successor is absent")
                observation = successor
        stream.append(tuple(actions))
    return tuple(stream)


def _require_authenticated_comparator_rows(
    batch: object, *, scope: str
) -> tuple[ComparatorRow, ...]:
    if scope not in {"feedback_only_myopic", "no_bootstrap"}:
        raise ContractError("comparator fit scope changed")
    if (
        type(batch) is not _AuthenticatedComparatorRows
        or type(batch.rows) is not tuple
        or type(batch.token) is not _ComparatorGateToken
        or any(type(row) is not ComparatorRow for row in batch.rows)
        or batch.token.token_sha256 not in _ISSUED_COMPARATOR_GATES
        or batch.token.source_sha256 != _expected_source_sha256("train")
        or (batch.token.token_sha256, scope) in _USED_COMPARATOR_SCOPES
    ):
        raise ContractError("comparator fitter received an unauthenticated batch")
    feedback_projection = tuple(
        ComparatorFeedbackStep(
            row.predecessor,
            row.action,
            row.successor,
            row.update_reward,
            row.done,
        )
        for row in batch.rows
    )
    if _comparator_feedback_sha256(feedback_projection) != batch.token.feedback_sha256:
        raise ContractError("comparator fit projection changed")
    _USED_COMPARATOR_SCOPES.add((batch.token.token_sha256, scope))
    return batch.rows


def _fit_myopic(
    rows: _AuthenticatedComparatorRows,
) -> dict[tuple[int, int, int, int], float]:
    rows = _require_authenticated_comparator_rows(
        rows, scope="feedback_only_myopic"
    )
    groups = _group_td_rows(rows)
    if set(groups) != set(ALL_STATE_ACTION_KEYS):
        raise ContractError("myopic comparator coverage changed")
    table: dict[tuple[int, int, int, int], float] = {}
    for key in ALL_STATE_ACTION_KEYS:
        records = groups[key]
        expected_count = {0: 512, 1: 256, 2: 128, 3: 64}[key[1]]
        if len(records) != expected_count:
            raise ContractError("myopic comparator cell count changed")
        table[key] = float(
            sum(float(_validate_reward(row.update_reward)) for row in records)
            / len(records)
        )
    return table


def _fit_no_bootstrap(
    rows: _AuthenticatedComparatorRows,
    gate: _FamilyGateToken,
) -> dict[tuple[int, int, int, int], float]:
    _require_family_gate(gate)
    rows = _require_authenticated_comparator_rows(rows, scope="no_bootstrap")
    groups = _group_td_rows(rows)
    terminal_means, _reads = _terminal_means(groups)
    bootstrap, coverage = _strip_kernel_inputs(groups)
    return _run_td_kernel(bootstrap, terminal_means, coverage, sweeps=1).table


def _row_coordinates(row: Mapping[str, object]) -> tuple[EpisodeSpec, int, int, int]:
    if type(row.get("split")) is not str:
        raise ContractError("legal row split has the wrong type")
    split_name = row["split"]
    split_codes = {regime.split: regime.split_enum for regime in REGIMES}
    if split_name not in split_codes:
        raise ContractError("legal row split is unknown")
    regime_code = _typed_metadata_value(
        row.get("regime_code"), dtype="int32", minimum=0, maximum=2**31 - 1
    )
    episode = _typed_metadata_value(
        row.get("episode"),
        dtype="int16",
        minimum=0,
        maximum=EPISODES_PER_REGIME - 1,
    )
    key = EpisodeKey(
        np.uint8(split_codes[split_name]),
        np.int32(regime_code),
        np.int16(episode),
    )
    try:
        spec = _spec_index()[key]
    except KeyError as exc:
        raise ContractError("legal row source identity is not unique")
    predecessor_key = row.get("predecessor_key")
    if not isinstance(predecessor_key, dict):
        raise ContractError("legal row predecessor key has the wrong type")
    phase = _typed_metadata_value(
        predecessor_key.get("phase"),
        dtype="int8",
        minimum=0,
        maximum=HORIZON - 1,
    )
    prefix = _typed_metadata_value(
        predecessor_key.get("prefix_code"),
        dtype="int8",
        minimum=0,
        maximum=(1 << phase) - 1,
    )
    action_field = row.get("action")
    if (
        not isinstance(action_field, dict)
        or set(action_field) != {"bytes", "dtype", "shape", "value"}
        or action_field.get("bytes") not in {"00", "01"}
        or action_field.get("dtype") != "int8"
        or action_field.get("shape") != []
        or type(action_field.get("value")) is not int
        or action_field.get("value") not in (0, 1)
        or action_field.get("bytes") != f"{int(action_field.get('value')):02x}"
    ):
        raise ContractError("legal row action has the wrong type")
    action = int(action_field["value"])
    return spec, phase, prefix, action


def _public_row_projection(row: Mapping[str, object]) -> dict[str, object]:
    excluded = {"target", "canonical_reward", "update_reward"}
    return {name: copy.deepcopy(value) for name, value in row.items() if name not in excluded}


def _target_twin(row: Mapping[str, object]) -> dict[str, object]:
    twin = copy.deepcopy(dict(row))
    old_target = int(row["target"]["value"])
    new_target = 1 - old_target
    twin["target"]["value"] = new_target
    if bool(row["done"]):
        _spec, phase, prefix, action = _row_coordinates(row)
        actions = (*_prefix_bits(prefix, phase), action)
        reward = _terminal_reward(new_target, actions)
        twin["canonical_reward"] = reward
        twin["update_reward"] = reward
    return twin


def _target_swap_audit() -> dict[str, object]:
    rows = 0
    reward_changes = 0
    public_preserved = True
    target_flips = 0
    public_digest = hashlib.sha256()
    twin_public_digest = hashlib.sha256()
    for row in _iter_legal_rows(_legal_row_projection_primary):
        frozen_public = _public_row_projection(row)
        _stream_update(public_digest, frozen_public)
        twin = _target_twin(row)
        _stream_update(twin_public_digest, _public_row_projection(twin))
        rows += 1
        target_flips += int(
            int(twin["target"]["value"]) == 1 - int(row["target"]["value"])
        )
        reward_changes += int(
            twin["canonical_reward"] != row["canonical_reward"]
        )
        public_preserved &= frozen_public == _public_row_projection(twin)
        if twin["target_slot"] != row["target_slot"]:
            raise ContractError("target swap changed target slot")
    sample = next(_iter_legal_rows(_legal_row_projection_primary))
    malformed = copy.deepcopy(sample)
    unflipped = _target_twin(sample)
    unflipped["target"]["value"] = int(sample["target"]["value"])
    public_mutation = _target_twin(sample)
    encoded = bytearray.fromhex(str(public_mutation["predecessor"]["bytes"]))
    encoded[0] ^= 1
    public_mutation["predecessor"]["bytes"] = encoded.hex()
    rejected = 0
    for invalid in (malformed, unflipped, public_mutation):
        try:
            _validate_target_twin(sample, invalid)
        except ContractError:
            rejected += 1
    result = {
        "rows": rows,
        "target_flips": target_flips,
        "terminal_reward_changes": reward_changes,
        "public_bytes_preserved": public_preserved,
        "public_sha256": public_digest.hexdigest(),
        "twin_public_sha256": twin_public_digest.hexdigest(),
        "invalid_twin_rejections": rejected,
    }
    if not (
        rows == 122880
        and target_flips == rows
        and reward_changes == 8192
        and public_preserved
        and public_digest.hexdigest() == twin_public_digest.hexdigest()
        and rejected == 3
    ):
        raise ContractError("target-swap twin gate failed")
    return result


def _validate_target_twin(
    canonical: Mapping[str, object], twin: Mapping[str, object]
) -> None:
    if int(twin["target"]["value"]) != 1 - int(canonical["target"]["value"]):
        raise ContractError("target twin did not flip hidden evaluator truth")
    if _public_row_projection(canonical) != _public_row_projection(twin):
        raise ContractError("target twin changed public or keyed bytes")
    if set(twin) != set(canonical):
        raise ContractError("target twin field set changed")
    spec, phase, prefix, action = _row_coordinates(canonical)
    del spec
    canonical_target = int(canonical["target"]["value"])
    twin_target = int(twin["target"]["value"])
    canonical_expected = (
        _terminal_reward(canonical_target, (*_prefix_bits(prefix, phase), action))
        if bool(canonical["done"])
        else 0.0
    )
    twin_expected = (
        _terminal_reward(twin_target, (*_prefix_bits(prefix, phase), action))
        if bool(twin["done"])
        else 0.0
    )
    if (
        canonical["canonical_reward"] != canonical_expected
        or canonical["update_reward"] != canonical_expected
        or twin["canonical_reward"] != twin_expected
        or twin["update_reward"] != twin_expected
        or twin["target_slot"] != canonical["target_slot"]
    ):
        raise ContractError("target twin outcome formula changed")


def _realized_public_row(spec: EpisodeSpec, phase: int, prefix: int, action: int) -> dict[str, object]:
    predecessor = _primary_observation(spec, phase, prefix)
    next_prefix = (prefix << 1) | action
    successor = None if phase == 3 else _primary_observation(spec, phase + 1, next_prefix)
    return {
        "predecessor": _observation_identity(predecessor),
        "action": {
            "bytes": np.asarray(action, dtype=np.dtype("i1")).tobytes().hex(),
            "dtype": "int8",
            "shape": [],
            "value": action,
        },
        "successor": None if successor is None else _observation_identity(successor),
        "done": phase == 3,
    }


def _realized_path_audit() -> dict[str, object]:
    row_digests: set[str] = set()
    path_digests: set[str] = set()
    split_rows: dict[str, set[str]] = defaultdict(set)
    split_paths: dict[str, set[str]] = defaultdict(set)
    realized_rows = 0
    for spec in _iter_episode_specs():
        prefix = 0
        path: list[dict[str, object]] = []
        for phase, action in enumerate(_behavior_actions(int(spec.action_code))):
            row = _realized_public_row(spec, phase, prefix, action)
            digest = _json_sha256(row)
            realized_rows += 1
            row_digests.add(digest)
            split_rows[spec.regime.split].add(digest)
            path.append(row)
            prefix = (prefix << 1) | action
        path_digest = _json_sha256(path)
        path_digests.add(path_digest)
        split_paths[spec.regime.split].add(path_digest)
    row_disjoint = all(
        split_rows[left].isdisjoint(split_rows[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    path_disjoint = all(
        split_paths[left].isdisjoint(split_paths[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    result = {
        "realized_rows": realized_rows,
        "unique_public_row_sha256_count": len(row_digests),
        "unique_public_path_sha256_count": len(path_digests),
        "public_rows_split_disjoint": row_disjoint,
        "public_paths_split_disjoint": path_disjoint,
        "identity_fields_excluded": True,
    }
    if not (
        realized_rows == 16384
        and len(row_digests) == 16384
        and len(path_digests) == 4096
        and row_disjoint
        and path_disjoint
    ):
        raise ContractError("realized-path disjointness gate failed")
    return result


def _identity_to_observation(identity: Mapping[str, object]) -> np.ndarray:
    expected_fields = {
        "bytes", "sha256", "dtype", "shape", "strides", "c_contiguous", "immutable"
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_fields:
        raise ContractError("observation identity field set changed")
    if (
        identity["dtype"] != "<f8"
        or identity["shape"] != [6]
        or identity["strides"] != [8]
        or identity["c_contiguous"] is not True
        or identity["immutable"] is not True
    ):
        raise ContractError("observation identity layout changed")
    try:
        encoded = bytes.fromhex(str(identity["bytes"]))
    except (TypeError, ValueError) as error:
        raise ContractError("observation identity bytes are malformed") from error
    if len(encoded) != 48 or hashlib.sha256(encoded).hexdigest() != identity["sha256"]:
        raise ContractError("observation identity digest changed")
    value = np.frombuffer(encoded, dtype=np.dtype("<f8")).copy()
    value.setflags(write=False)
    return _validate_observation(value)


def _transition_control_row(row: Mapping[str, object]) -> dict[str, object]:
    transformed = copy.deepcopy(dict(row))
    spec, phase, prefix, action = _row_coordinates(row)
    donor = _paired_spec(spec)
    if phase > 0:
        transformed["predecessor"] = _observation_identity(
            _primary_observation(donor, phase, prefix)
        )
    if not bool(row["done"]):
        successor_prefix = (prefix << 1) | action
        donor_successor = _primary_observation(
            donor, phase + 1, successor_prefix
        )
        transformed["successor"] = _observation_identity(donor_successor)
        transformed["donor"] = {
            "episode_key": _key_projection(donor.key),
            "payload_sha256": _observation_sha256(donor_successor),
        }
    return transformed


def _reward_origin_control_row(row: Mapping[str, object]) -> dict[str, object]:
    transformed = copy.deepcopy(dict(row))
    if bool(row["done"]):
        spec, _phase, _prefix, _action = _row_coordinates(row)
        origin = _reward_origin_spec(spec)
        transformed["update_reward"] = _episode_terminal_reward(origin)
        transformed["origin"] = _key_projection(origin.key)
    return transformed


def _signal_ablation_row(row: Mapping[str, object]) -> dict[str, object]:
    transformed = copy.deepcopy(dict(row))
    predecessor = _identity_to_observation(row["predecessor"])
    predecessor_copy = predecessor.copy(order="C")
    predecessor_copy[1] = 0.0
    predecessor_copy.setflags(write=False)
    transformed["predecessor"] = _observation_identity(predecessor_copy)
    if row["successor"] is not None:
        successor = _identity_to_observation(row["successor"])
        successor_copy = successor.copy(order="C")
        successor_copy[1] = 0.0
        successor_copy.setflags(write=False)
        transformed["successor"] = _observation_identity(successor_copy)
    return transformed


def _difference_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix.rstrip(".")}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {prefix.rstrip(".")}
        paths: set[str] = set()
        for key in left:
            paths.update(_difference_paths(left[key], right[key], f"{prefix}{key}."))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return {prefix.rstrip(".")}
        paths: set[str] = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.update(_difference_paths(left_item, right_item, f"{prefix}{index}."))
        return paths
    return set() if left == right else {prefix.rstrip(".")}


def _allowed_difference_paths(row: Mapping[str, object], mode: str) -> set[str]:
    _spec, phase, _prefix, _action = _row_coordinates(row)
    done = bool(row["done"])
    if mode == "target_swap":
        allowed = {"target.value"}
        if done:
            allowed.update({"canonical_reward", "update_reward"})
        return allowed
    if mode == "transition_control":
        allowed: set[str] = set()
        if phase > 0:
            allowed.update({"predecessor.bytes", "predecessor.sha256"})
        if not done:
            allowed.update({"successor.bytes", "successor.sha256", "donor"})
        return allowed
    if mode == "reward_origin_control":
        return {"update_reward", "origin.episode.value"} if done else set()
    if mode == "signal_ablation":
        allowed = {"predecessor.bytes", "predecessor.sha256"}
        if not done:
            allowed.update({"successor.bytes", "successor.sha256"})
        return allowed
    raise ContractError("unknown difference-whitelist mode")


def _validate_difference_whitelist(
    canonical: Mapping[str, object], transformed: Mapping[str, object], mode: str
) -> set[str]:
    transforms: dict[str, Callable[[Mapping[str, object]], dict[str, object]]] = {
        "target_swap": _target_twin,
        "transition_control": _transition_control_row,
        "reward_origin_control": _reward_origin_control_row,
        "signal_ablation": _signal_ablation_row,
    }
    if mode not in transforms or dict(transformed) != transforms[mode](canonical):
        raise ContractError("control does not match its exact frozen mapping")
    differences = _difference_paths(canonical, transformed)
    allowed = _allowed_difference_paths(canonical, mode)
    def permitted(path: str) -> bool:
        return path in allowed

    if not all(permitted(path) for path in differences):
        raise ContractError("control changed a protected field")
    if mode == "target_swap" and "target.value" not in differences:
        raise ContractError("target twin omitted its required truth flip")
    if mode == "transition_control" and not differences:
        raise ContractError("transition control did not alter its declared field")
    if mode == "reward_origin_control" and bool(canonical["done"]) and not any(
        path.startswith("origin.") for path in differences
    ):
        raise ContractError("reward-origin control omitted its origin")
    if mode == "signal_ablation" and not differences:
        raise ContractError("signal ablation did not alter its declared field")
    return differences


def _control_whitelist_audit() -> dict[str, object]:
    modes = {
        "target_swap": _target_twin,
        "transition_control": _transition_control_row,
        "reward_origin_control": _reward_origin_control_row,
        "signal_ablation": _signal_ablation_row,
    }
    checked = Counter()
    successor_canonical: Counter[str] = Counter()
    successor_transition: Counter[str] = Counter()
    transition_lineage_actual: Counter[str] = Counter()
    transition_lineage_expected: Counter[str] = Counter()
    identity_canonical = Counter()
    identity_by_mode: dict[str, Counter[str]] = {
        mode: Counter() for mode in modes
    }
    combined_signal_rows = 0
    for canonical in _iter_legal_rows(_legal_row_projection_primary):
        identity_canonical[_transition_identity_tuple(canonical)] += 1
        if canonical["successor"] is not None:
            successor_canonical[
                json.dumps(canonical["successor"], sort_keys=True, separators=(",", ":"))
            ] += 1
        transformed_controls: dict[str, dict[str, object]] = {}
        for mode, transform in modes.items():
            transformed = transform(canonical)
            transformed_controls[mode] = transformed
            _validate_difference_whitelist(canonical, transformed, mode)
            identity_by_mode[mode][_transition_identity_tuple(transformed)] += 1
            checked[mode] += 1
            if mode == "transition_control" and transformed["successor"] is not None:
                successor_transition[
                    json.dumps(
                        transformed["successor"],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ] += 1
                lineage = {
                    "predecessor_key": transformed["predecessor_key"],
                    "action_key": transformed["action_key"],
                    "transition_key": transformed["transition_key"],
                    "successor_key": transformed["successor_key"],
                    "successor": transformed["successor"],
                    "done": transformed["done"],
                    "donor": transformed["donor"],
                }
                transition_lineage_actual[
                    json.dumps(lineage, sort_keys=True, separators=(",", ":"))
                ] += 1
                expected_control = _transition_control_row(canonical)
                expected_lineage = {
                    "predecessor_key": expected_control["predecessor_key"],
                    "action_key": expected_control["action_key"],
                    "transition_key": expected_control["transition_key"],
                    "successor_key": expected_control["successor_key"],
                    "successor": expected_control["successor"],
                    "done": expected_control["done"],
                    "donor": expected_control["donor"],
                }
                transition_lineage_expected[
                    json.dumps(
                        expected_lineage, sort_keys=True, separators=(",", ":")
                    )
                ] += 1
        for base_control in (
            transformed_controls["transition_control"],
            transformed_controls["reward_origin_control"],
        ):
            ablated_control = _signal_ablation_row(base_control)
            _validate_difference_whitelist(
                base_control, ablated_control, "signal_ablation"
            )
            combined_signal_rows += 1
    representative = next(_iter_legal_rows(_legal_row_projection_primary))
    protected_mutation_rejections = 0
    protected_mutators: tuple[Callable[[dict[str, object]], None], ...] = (
        lambda row: row.__setitem__("split", "test"),
        lambda row: row["regime_code"].__setitem__("value", 9999),
        lambda row: row["episode"].__setitem__("value", 9),
        lambda row: row["block"].__setitem__("value", 9),
        lambda row: row["cell"].__setitem__("value", 9),
        lambda row: row["target_slot"].__setitem__("value", 1 - int(row["target_slot"]["value"])),
        lambda row: row["action_code_lineage"].__setitem__("value", 9),
        lambda row: row["target"].__setitem__("value", 1 - int(row["target"]["value"])),
        lambda row: row["predecessor_key"]["phase"].__setitem__("dtype", "int16"),
        lambda row: row["predecessor"].__setitem__(
            "bytes", "ff" + str(row["predecessor"]["bytes"])[2:]
        ),
        lambda row: row["predecessor"].__setitem__("sha256", "0" * 64),
        lambda row: row["action_key"].__setitem__("kind", "broken"),
        lambda row: row["action"].__setitem__("bytes", "02"),
        lambda row: row["action"].__setitem__("value", 1 - int(row["action"]["value"])),
        lambda row: row["transition_key"].__setitem__("kind", "broken"),
        lambda row: row["feedback_key"].__setitem__("kind", "broken"),
        lambda row: row["successor_key"].__setitem__("kind", "broken"),
        lambda row: row["successor"].__setitem__("sha256", "0" * 64),
        lambda row: row.__setitem__("canonical_reward", 1.0),
        lambda row: row.__setitem__("update_reward", 1.0),
        lambda row: row.__setitem__("reward_dtype", "float32"),
        lambda row: row.__setitem__("done", 0),
        lambda row: row.__setitem__("successor_legal", 1),
        lambda row: row.__setitem__("donor", {"forged": True}),
        lambda row: row.__setitem__("origin", {"forged": True}),
    )
    for mode, transform in modes.items():
        for mutate in protected_mutators:
            invalid = transform(representative)
            mutate(invalid)
            try:
                _validate_difference_whitelist(representative, invalid, mode)
            except ContractError:
                protected_mutation_rejections += 1
    transition_involution = all(
        _paired_spec(_paired_spec(spec)).key == spec.key
        and _paired_spec(spec).key != spec.key
        for spec in _iter_episode_specs()
    )
    origins = [_reward_origin_spec(spec).key for spec in _iter_episode_specs()]
    source_keys = [spec.key for spec in _iter_episode_specs()]
    reward_origin_bijection = (
        len(set(origins)) == len(origins)
        and set(origins) == set(source_keys)
        and all(origin != source for origin, source in zip(origins, source_keys, strict=True))
    )
    passed = (
        all(checked[mode] == 122880 for mode in modes)
        and combined_signal_rows == 245760
        and successor_canonical == successor_transition
        and transition_lineage_actual == transition_lineage_expected
        and all(identity_by_mode[mode] == identity_canonical for mode in modes)
        and protected_mutation_rejections == len(modes) * len(protected_mutators)
        and transition_involution
        and reward_origin_bijection
    )
    if not passed:
        raise ContractError("complete control-difference whitelist gate failed")
    return {
        "all_rows_checked": 122880,
        "combined_signal_rows_checked": combined_signal_rows,
        "canonical_rows": 122880,
        "passed": passed,
        "target_swap_rows_checked": checked["target_swap"],
        "transition_rows_checked": checked["transition_control"],
        "reward_origin_rows_checked": checked["reward_origin_control"],
        "signal_ablation_rows_checked": checked["signal_ablation"],
        "successor_multiset_preserved": successor_canonical == successor_transition,
        "protected_mutation_rejections": protected_mutation_rejections,
        "reward_origin_bijection": reward_origin_bijection,
        "transition_involution": transition_involution,
    }


def _run_family_preflight() -> tuple[
    _FamilyGateToken,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    _ISSUED_FAMILY_GATES.clear()
    _ISSUED_TRACE_GATES.clear()
    _ISSUED_STEP_AUTHORIZATIONS.clear()
    _ISSUED_COMPARATOR_GATES.clear()
    _USED_COMPARATOR_SCOPES.clear()
    _CONSTRUCTION_COUNTS.update({"collector": 0, "learner": 0})
    family = _family_audit()
    corruptions = _family_corruption_audit()
    target_swap = _target_swap_audit()
    realized_paths = _realized_path_audit()
    control_whitelists = _control_whitelist_audit()
    gate_projection = {
        "family_sha256": family["primary_sha256"],
        "corruptions_rejected": corruptions["corruptions_rejected"],
        "target_public_sha256": target_swap["public_sha256"],
        "realized_rows": realized_paths["realized_rows"],
    }
    gate = _FamilyGateToken(
        family_sha256=EXPECTED_FAMILY_SHA256,
        gate_sha256=_json_sha256(gate_projection),
    )
    _ISSUED_FAMILY_GATES.add(gate.gate_sha256)
    _require_family_gate(gate)
    return (
        gate,
        {**family, **corruptions},
        target_swap,
        realized_paths,
        control_whitelists,
    )


def _fit_from_train_source(
    train_source: _CountingSource,
    gate: _FamilyGateToken,
    *,
    mode: str = "canonical",
    during_fit_callback: Callable[[], None] | None = None,
) -> tuple[TraceBundle, tuple[TDInputRow, ...], FitResult, dict[str, object]]:
    _require_family_gate(gate)
    if type(train_source) is not _CountingSource:
        raise ContractError("fitter received a non-capability train source")
    source_episodes = train_source.open("train")
    if train_source.source_sha256 != _expected_source_sha256("train"):
        raise ContractError("fitter received a non-train or incomplete source")
    try:
        if during_fit_callback is not None:
            during_fit_callback()
        trace, collection_audit = _collect_trace(
            source_episodes, mode=mode, gate=gate
        )
    finally:
        train_source.close()
    specs = train_source.committed_specs
    if (
        len(specs) != EPISODE_COUNTS["train"]
        or specs != tuple(_iter_episode_specs("train"))
        or train_source.counts["materializer"]
        != EPISODE_COUNTS["train"] * HORIZON
    ):
        raise ContractError("fitter source did not materialize the complete train path")
    rows = _validate_trace(trace, specs, mode=mode)
    reordered = _independently_reordered_trace(trace)
    reordered_rows = _validate_trace(reordered, specs, mode=mode)
    if _td_rows_sha256(rows) != _td_rows_sha256(reordered_rows):
        raise ContractError("component reorder changed TD input projection")
    if not bool(_typed_contract_case(trace, collection_audit)["passed"]):
        raise ContractError("typed runtime contract failed before TD construction")
    fitted = (
        _fit_signal_ablation_td(rows, gate)
        if mode == "signal_ablation"
        else _fit_td(rows, gate)
    )
    return trace, rows, fitted, collection_audit


def _collect_from_source(
    source: _CountingSource,
    gate: _FamilyGateToken,
    *,
    expected_split: str,
    mode: str,
) -> tuple[TraceBundle, _AuthenticatedRows, dict[str, object]]:
    _require_family_gate(gate)
    if type(source) is not _CountingSource:
        raise ContractError("collector received a non-capability source")
    source_episodes = source.open(expected_split)
    try:
        trace, collection = _collect_trace(source_episodes, mode=mode, gate=gate)
    finally:
        source.close()
    specs = source.committed_specs
    if (
        specs != tuple(_iter_episode_specs(expected_split))
        or source.source_sha256 != _expected_source_sha256(expected_split)
        or source.counts["materializer"] != EPISODE_COUNTS[expected_split] * HORIZON
    ):
        raise ContractError("control source did not materialize its complete path")
    return trace, _validate_trace(trace, specs, mode=mode), collection


def _evaluate_from_source(
    table: Mapping[tuple[int, int, int, int], float],
    source: _CountingSource,
    *,
    expected_split: str,
    observation_mode: str = "canonical",
) -> dict[str, object]:
    if type(source) is not _CountingSource:
        raise ContractError("held-out evaluator received a non-capability source")
    source_episodes = source.open(expected_split)
    if source.source_sha256 != _expected_source_sha256(expected_split):
        raise ContractError("held-out evaluator received an invalid source")
    by_regime: dict[int, list[float]] = defaultdict(list)
    stream: list[list[int]] = []
    state_before = _table_sha256(table)
    completed = 0
    try:
        for source_episode in source_episodes:
            spec = source_episode.materialize(0)
            observation = _runtime_observation(spec, 0, 0, observation_mode)
            actions: list[int] = []
            for phase in range(HORIZON):
                if phase > 0:
                    if source_episode.materialize(phase) != spec:
                        raise ContractError("held-out source identity changed")
                action = _greedy_action(table, observation)
                actions.append(action)
                if phase < HORIZON - 1:
                    successor = _public_successor(
                        observation, _immutable_scalar(float(action), "i1")
                    )
                    if successor is None:
                        raise ContractError("held-out source lost a successor")
                    observation = successor
            by_regime[int(spec.regime.code)].append(
                _terminal_reward(int(spec.target), actions)
            )
            stream.append(actions)
            completed += 1
    finally:
        source.close()
    if (
        source.committed_specs != tuple(_iter_episode_specs(expected_split))
        or completed != EPISODE_COUNTS[expected_split]
        or source.counts["materializer"] != EPISODE_COUNTS[expected_split] * HORIZON
    ):
        raise ContractError("held-out source did not materialize its complete path")
    regime_means = [sum(values) / len(values) for values in by_regime.values()]
    return {
        "macro": float(sum(regime_means) / len(regime_means)),
        "minimum_regime": float(min(regime_means)),
        "stream_sha256": _json_sha256(stream),
        "state_before_sha256": state_before,
        "state_after_sha256": _table_sha256(table),
        "updates": 0,
    }


class _SourceOrchestrator:
    def __init__(
        self,
        train_source: _CountingSource,
        validation_source: _CountingSource | None,
        test_source: _CountingSource | None,
        gate: _FamilyGateToken,
    ) -> None:
        if type(train_source) is not _CountingSource or any(
            source is not None and type(source) is not _CountingSource
            for source in (validation_source, test_source)
        ):
            raise ContractError("orchestrator received a non-capability source")
        self.train_source = train_source
        self.validation_source = validation_source
        self.test_source = test_source
        self.gate = _require_family_gate(gate)
        self._sealed: tuple[
            TraceBundle, tuple[TDInputRow, ...], FitResult, dict[str, object]
        ] | None = None
        self._fit_mode: str | None = None

    def fit(
        self,
        *,
        mode: str = "canonical",
        attack_heldout_source: bool = False,
    ) -> tuple[
        TraceBundle, tuple[TDInputRow, ...], FitResult, dict[str, object]
    ]:
        if self._sealed is not None:
            raise ContractError("orchestrator fit was invoked twice")
        callback: Callable[[], None] | None = None
        if attack_heldout_source:
            if self.validation_source is None:
                raise ContractError("held-out timing attack lacks its real source")
            callback = lambda: tuple(self.validation_source.open("validation"))
        self._sealed = _fit_from_train_source(
            self.train_source,
            self.gate,
            mode=mode,
            during_fit_callback=callback,
        )
        self._fit_mode = mode
        return self._sealed

    def replace_train_after_seal(self, source: _CountingSource) -> None:
        if self._sealed is None:
            raise ContractError("train source replaced before fit seal")
        if type(source) is not _CountingSource:
            raise ContractError("replacement train source lacks its capability")
        self.train_source = source

    def evaluate_heldout(self) -> tuple[dict[str, object], dict[str, object]]:
        if self._sealed is None:
            raise ContractError("held-out evaluation started before fit seal")
        if self.validation_source is None or self.test_source is None:
            raise ContractError("held-out evaluation source is absent")
        table = self._sealed[2].table
        observation_mode = (
            "signal_ablation" if self._fit_mode == "signal_ablation" else "canonical"
        )
        validation = _evaluate_from_source(
            table,
            self.validation_source,
            expected_split="validation",
            observation_mode=observation_mode,
        )
        test = _evaluate_from_source(
            table,
            self.test_source,
            expected_split="test",
            observation_mode=observation_mode,
        )
        return validation, test


def _require_complete_source_counts(source: _CountingSource, split: str) -> None:
    expected = {
        "factory": 1,
        "iterator": 1,
        "materializer": EPISODE_COUNTS[split] * HORIZON,
        "close": 1,
        "post_close": 0,
    }
    if source.counts != expected:
        raise ContractError(f"{split} source lifecycle changed")


def _source_boundary_audit(gate: _FamilyGateToken) -> dict[str, object]:
    _require_family_gate(gate)
    train_specs = tuple(_iter_episode_specs("train"))
    validation_specs = tuple(_iter_episode_specs("validation"))
    test_specs = tuple(_iter_episode_specs("test"))
    commitments: list[tuple[str, str, str]] = []
    exploding_counts: list[dict[str, int]] = []
    source_lifecycle_exact = True

    for construction in ("absent", "exploding", "lazy"):
        train_source = _CountingSource(train_specs, expected_split="train")
        validation_source = (
            None
            if construction == "absent"
            else _CountingSource(
                validation_specs,
                expected_split="validation",
                exploding=construction == "exploding",
            )
        )
        test_source = (
            None
            if construction == "absent"
            else _CountingSource(
                test_specs,
                expected_split="test",
                exploding=construction == "exploding",
            )
        )
        orchestrator = _SourceOrchestrator(
            train_source, validation_source, test_source, gate
        )
        trace, rows, fitted, _collection = orchestrator.fit()
        commitments.append(
            (_trace_sha256(trace), _td_rows_sha256(rows), _table_sha256(fitted.table))
        )
        source_lifecycle_exact &= train_source.counts == {
            "factory": 1,
            "iterator": 1,
            "materializer": EPISODE_COUNTS["train"] * HORIZON,
            "close": 1,
            "post_close": 0,
        }
        if construction == "exploding":
            assert validation_source is not None and test_source is not None
            exploding_counts.extend(
                [dict(validation_source.counts), dict(test_source.counts)]
            )
        if construction == "lazy":
            assert validation_source is not None and test_source is not None
            if any(validation_source.counts.values()) or any(test_source.counts.values()):
                raise ContractError("held-out source opened before fit sealed")
            lazy_state_before = _table_sha256(fitted.table)
            lazy_validation, lazy_test = orchestrator.evaluate_heldout()
            source_lifecycle_exact &= (
                validation_source.counts
                == {
                    "factory": 1,
                    "iterator": 1,
                    "materializer": EPISODE_COUNTS["validation"] * HORIZON,
                    "close": 1,
                    "post_close": 0,
                }
                and test_source.counts
                == {
                    "factory": 1,
                    "iterator": 1,
                    "materializer": EPISODE_COUNTS["test"] * HORIZON,
                    "close": 1,
                    "post_close": 0,
                }
                and lazy_validation["updates"] == 0
                and lazy_test["updates"] == 0
                and _table_sha256(fitted.table) == lazy_state_before
            )

    train_source = _CountingSource(train_specs, expected_split="train")
    validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    test_source = _CountingSource(test_specs, expected_split="test")
    orchestrator = _SourceOrchestrator(
        train_source, validation_source, test_source, gate
    )
    _trace, _rows, fitted, _collection = orchestrator.fit()
    state_before = _table_sha256(fitted.table)
    forbidden_train = _CountingSource(
        train_specs, expected_split="train", exploding=True
    )
    orchestrator.replace_train_after_seal(forbidden_train)
    validation, test = orchestrator.evaluate_heldout()
    state_after_validation = _table_sha256(fitted.table)
    state_after_test = _table_sha256(fitted.table)
    expected_heldout_materializations = (
        EPISODE_COUNTS["validation"] + EPISODE_COUNTS["test"]
    ) * HORIZON
    heldout_materializations = (
        validation_source.counts["materializer"]
        + test_source.counts["materializer"]
    )
    zero_counts = all(not any(counts.values()) for counts in exploding_counts)
    inverse_train_zero = not any(forbidden_train.counts.values())
    commitments_equal = len(set(commitments)) == 1
    state_unchanged = (
        state_before == state_after_validation == state_after_test
    )
    passed = (
        commitments_equal
        and source_lifecycle_exact
        and zero_counts
        and inverse_train_zero
        and heldout_materializations == expected_heldout_materializations
        and validation["updates"] == 0
        and test["updates"] == 0
        and state_unchanged
    )
    return {
        "absent_exploding_lazy_exact": commitments_equal,
        "exploding_source_operations": sum(
            sum(counts.values()) for counts in exploding_counts
        ),
        "heldout_materializations": heldout_materializations,
        "heldout_state_unchanged": state_unchanged,
        "heldout_updates": int(validation["updates"]) + int(test["updates"]),
        "inverse_train_operations": sum(forbidden_train.counts.values()),
        "passed": passed,
        "sealed_fit_sha256": state_before,
        "train_trace_sha256": commitments[0][0],
    }


def _random_action_streams() -> dict[str, tuple[tuple[int, ...], ...]]:
    rng = np.random.Generator(np.random.PCG64(RANDOM_BASELINE_SEED))
    streams: dict[str, tuple[tuple[int, ...], ...]] = {}
    for split in ("train", "validation", "test"):
        rows: list[tuple[int, ...]] = []
        for _episode in range(EPISODE_COUNTS[split]):
            draw = rng.integers(0, 2, size=(HORIZON,), dtype=np.int8)
            if draw.dtype != np.dtype("i1") or draw.shape != (HORIZON,):
                raise ContractError("random baseline call contract changed")
            rows.append(tuple(int(value) for value in draw))
        streams[split] = tuple(rows)
    return streams


def compute_random_stream_sha256() -> str:
    return _json_sha256(_random_action_streams())


EXPECTED_RANDOM_STREAM_SHA256 = (
    "8fe33e0832d9fe4b705f97bf20d8559223e8af217ca8f6a97c587b8bea9e6803"
)


def _baseline_audit(
    canonical_trace: TraceBundle,
    canonical_rows: _AuthenticatedRows,
    canonical_fit: FitResult,
    gate: _FamilyGateToken,
    canonical_validation: Mapping[str, object],
    canonical_test: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    _require_family_gate(gate)
    train_specs = tuple(_iter_episode_specs("train"))
    validation_specs = tuple(_iter_episode_specs("validation"))
    test_specs = tuple(_iter_episode_specs("test"))
    public_episodes = {
        split: _public_comparator_episodes(split)
        for split in ("train", "validation", "test")
    }
    comparator_feedback = _seal_comparator_feedback(
        canonical_trace, canonical_rows
    )
    comparator_rows = _comparator_materialize(comparator_feedback)
    myopic = _fit_myopic(comparator_rows)
    no_bootstrap = _fit_no_bootstrap(comparator_rows, gate)
    random_left = _random_action_streams()
    random_right = _random_action_streams()
    random_digest = _json_sha256(random_left)
    if random_left != random_right:
        raise ContractError("random baseline did not replay exactly")

    tables = {
        "myopic": myopic,
        "no_bootstrap": no_bootstrap,
    }
    replay_exact = True
    scores: dict[str, dict[str, float]] = {}
    for name, table in tables.items():
        streams_left = {
            split: _comparator_action_stream(table, public_episodes[split])
            for split in ("train", "validation", "test")
        }
        streams_right = {
            split: _comparator_action_stream(table, public_episodes[split])
            for split in ("train", "validation", "test")
        }
        replay_exact &= streams_left == streams_right
        scores[name] = {
            split: float(_KeyedEvaluator.frozen(split).evaluate(streams_left[split])["macro"])
            for split, specs in (
                ("train", train_specs),
                ("validation", validation_specs),
                ("test", test_specs),
            )
        }

    constant_scores: dict[str, dict[str, float]] = {}
    constant_replay_exact = True
    for split, specs in (
        ("train", train_specs),
        ("validation", validation_specs),
        ("test", test_specs),
    ):
        public_episode_count = len(public_episodes[split])
        zero_left = tuple((0, 0, 0, 0) for _ in range(public_episode_count))
        zero_right = tuple((0, 0, 0, 0) for _ in range(public_episode_count))
        one_left = tuple((1, 1, 1, 1) for _ in range(public_episode_count))
        one_right = tuple((1, 1, 1, 1) for _ in range(public_episode_count))
        constant_replay_exact &= zero_left == zero_right and one_left == one_right
        evaluator = _KeyedEvaluator.frozen(split)
        constant_scores[split] = {
            "zero": float(evaluator.evaluate(zero_left)["macro"]),
            "one": float(evaluator.evaluate(one_left)["macro"]),
        }
    random_scores = {
        split: _KeyedEvaluator.frozen(split).evaluate(random_left[split])
        for split, specs in (
            ("train", train_specs),
            ("validation", validation_specs),
            ("test", test_specs),
        )
    }
    best_constant_validation = max(constant_scores["validation"].values())
    best_constant_test = max(constant_scores["test"].values())
    passed = (
        replay_exact
        and constant_replay_exact
        and random_digest == EXPECTED_RANDOM_STREAM_SHA256
        and all(
            score == 0.5
            for split_scores in constant_scores.values()
            for score in split_scores.values()
        )
        and scores["myopic"]["validation"] == 0.5
        and scores["myopic"]["test"] == 0.5
        and scores["no_bootstrap"]["validation"] == 0.5
        and scores["no_bootstrap"]["test"] == 0.5
        and float(canonical_validation["macro"]) - best_constant_validation
        >= THRESHOLDS["minimum_validation_gain_constant"]
        and float(canonical_test["macro"]) - best_constant_test
        >= THRESHOLDS["minimum_test_gain_constant"]
        and float(canonical_validation["macro"]) - scores["myopic"]["validation"]
        >= THRESHOLDS["minimum_validation_gain_myopic"]
        and float(canonical_test["macro"]) - scores["myopic"]["test"]
        >= THRESHOLDS["minimum_test_gain_myopic"]
        and float(canonical_validation["macro"])
        - scores["no_bootstrap"]["validation"]
        >= THRESHOLDS["minimum_validation_gain_no_bootstrap"]
        and float(canonical_test["macro"]) - scores["no_bootstrap"]["test"]
        >= THRESHOLDS["minimum_test_gain_no_bootstrap"]
        and float(canonical_validation["macro"])
        - float(random_scores["validation"]["macro"])
        >= THRESHOLDS["minimum_validation_gain_random"]
        and float(canonical_test["macro"]) - float(random_scores["test"]["macro"])
        >= THRESHOLDS["minimum_test_gain_random"]
    )
    case = {
        "constant_one_test_macro_return": constant_scores["test"]["one"],
        "constant_zero_test_macro_return": constant_scores["test"]["zero"],
        "myopic_test_macro_return": scores["myopic"]["test"],
        "no_bootstrap_test_macro_return": scores["no_bootstrap"]["test"],
        "numpy_version": str(np.__version__),
        "passed": passed,
        "random_stream_sha256": random_digest,
        "random_test_macro_return": float(random_scores["test"]["macro"]),
        "replay_exact": replay_exact and constant_replay_exact,
    }
    metrics = {
        "constant": {
            "validation": best_constant_validation,
            "test": best_constant_test,
        },
        "myopic": {
            "validation": scores["myopic"]["validation"],
            "test": scores["myopic"]["test"],
        },
        "no_bootstrap": {
            "validation": scores["no_bootstrap"]["validation"],
            "test": scores["no_bootstrap"]["test"],
        },
        "random": {
            "validation": float(random_scores["validation"]["macro"]),
            "test": float(random_scores["test"]["macro"]),
        },
    }
    return case, metrics


def _expect_contract_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ContractError:
        return True
    return False


def _replace_trace_component(
    trace: TraceBundle,
    component: str,
    values: Sequence[object],
) -> TraceBundle:
    if component not in {"observations", "actions", "transitions", "feedback", "closes"}:
        raise ContractError("unknown trace component")
    return replace(trace, **{component: tuple(values)})


def _trace_attack_audit(
    gate: _FamilyGateToken,
    trace: TraceBundle,
    base_rows: _AuthenticatedRows,
    canonical_fit: FitResult,
) -> dict[str, object]:
    specs = tuple(_iter_episode_specs("train"))
    base_rows = _require_authenticated_train_rows(
        base_rows, allowed_modes={"canonical"}
    )
    if base_rows.token.trace_sha256 != _trace_sha256(trace):
        raise ContractError("trace attack audit received a mismatched sealed trace")
    attacks: list[TraceBundle] = []

    attacks.append(_replace_trace_component(trace, "observations", trace.observations[1:]))
    attacks.append(
        _replace_trace_component(
            trace, "observations", (*trace.observations, trace.observations[0])
        )
    )
    first_observation = trace.observations[0]
    unknown_episode = EpisodeKey(np.uint8(0), np.int32(1009), np.int16(99))
    attacks.append(
        _replace_trace_component(
            trace,
            "observations",
            (replace(first_observation, key=ObservationKey(unknown_episode, np.int8(0), np.int8(0))), *trace.observations[1:]),
        )
    )
    malformed_key = ObservationKey(specs[0].key, np.int16(0), np.int8(0))  # type: ignore[arg-type]
    attacks.append(
        _replace_trace_component(
            trace,
            "observations",
            (replace(first_observation, key=malformed_key), *trace.observations[1:]),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "actions",
            (
                replace(trace.actions[0], observation_key=trace.observations[4].key),
                *trace.actions[1:],
            ),
        )
    )
    wrong_observation = _copy_observation(first_observation.value)
    wrong_observation.flags.writeable = True
    wrong_observation[5] += 1.0
    wrong_observation.flags.writeable = False
    attacks.append(
        _replace_trace_component(
            trace,
            "observations",
            (
                replace(first_observation, value=wrong_observation),
                *trace.observations[1:],
            ),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "transitions",
            (
                replace(trace.transitions[0], successor_sha256="0" * 64),
                *trace.transitions[1:],
            ),
        )
    )
    wrong_action = _immutable_scalar(1 - int(trace.actions[0].value), "i1")
    attacks.append(
        _replace_trace_component(
            trace,
            "actions",
            (replace(trace.actions[0], value=wrong_action), *trace.actions[1:]),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "transitions",
            (
                replace(trace.transitions[0], source_episode=specs[1].key),
                *trace.transitions[1:],
            ),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "transitions",
            (
                replace(trace.transitions[0], donor_episode=specs[1].key),
                *trace.transitions[1:],
            ),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "feedback",
            (
                replace(trace.feedback[0], origin_episode=specs[1].key),
                *trace.feedback[1:],
            ),
        )
    )
    wrong_feedback_key = FeedbackKey(trace.feedback[0].transition_key, np.int8(1))
    attacks.append(
        _replace_trace_component(
            trace,
            "feedback",
            (replace(trace.feedback[0], key=wrong_feedback_key), *trace.feedback[1:]),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "closes",
            (
                replace(trace.closes[0], final_feedback_key=trace.closes[1].final_feedback_key),
                *trace.closes[1:],
            ),
        )
    )
    writeable_observation = np.array(first_observation.value, copy=True)
    attacks.append(
        _replace_trace_component(
            trace,
            "observations",
            (
                replace(first_observation, value=writeable_observation),
                *trace.observations[1:],
            ),
        )
    )
    wrong_action_type = np.asarray(int(trace.actions[0].value), dtype=np.int64)
    wrong_action_type.setflags(write=False)
    attacks.append(
        _replace_trace_component(
            trace,
            "actions",
            (replace(trace.actions[0], value=wrong_action_type), *trace.actions[1:]),
        )
    )
    wrong_reward_type = np.asarray(float(trace.feedback[0].update_reward), dtype=np.float32)
    wrong_reward_type.setflags(write=False)
    attacks.append(
        _replace_trace_component(
            trace,
            "feedback",
            (
                replace(trace.feedback[0], update_reward=wrong_reward_type),
                *trace.feedback[1:],
            ),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "transitions",
            (replace(trace.transitions[0], done=0), *trace.transitions[1:]),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "feedback",
            (
                replace(trace.feedback[0], canonical_reward=_immutable_scalar(1.0, "<f8")),
                *trace.feedback[1:],
            ),
        )
    )
    attacks.append(
        _replace_trace_component(
            trace,
            "feedback",
            (
                replace(trace.feedback[0], update_reward=_immutable_scalar(1.0, "<f8")),
                *trace.feedback[1:],
            ),
        )
    )
    swapped_observations = list(trace.observations)
    swapped_observations[0] = replace(
        swapped_observations[0],
        value=trace.observations[1].value,
        value_sha256=trace.observations[1].value_sha256,
    )
    swapped_observations[1] = replace(
        swapped_observations[1],
        value=trace.observations[0].value,
        value_sha256=trace.observations[0].value_sha256,
    )
    attacks.append(_replace_trace_component(trace, "observations", swapped_observations))

    action_zero_index = next(
        index for index, record in enumerate(trace.actions) if int(record.value) == 0
    )
    action_one_index = next(
        index for index, record in enumerate(trace.actions) if int(record.value) == 1
    )
    swapped_actions = list(trace.actions)
    swapped_actions[action_zero_index] = replace(
        swapped_actions[action_zero_index], value=trace.actions[action_one_index].value
    )
    swapped_actions[action_one_index] = replace(
        swapped_actions[action_one_index], value=trace.actions[action_zero_index].value
    )
    swapped_transitions = list(trace.transitions)
    swapped_transitions[0] = replace(
        swapped_transitions[0],
        successor_sha256=trace.transitions[4].successor_sha256,
    )
    swapped_transitions[4] = replace(
        swapped_transitions[4],
        successor_sha256=trace.transitions[0].successor_sha256,
    )
    success_feedback_index = next(
        index
        for index, record in enumerate(trace.feedback)
        if record.done and float(record.update_reward) == 1.0
    )
    failure_feedback_index = next(
        index
        for index, record in enumerate(trace.feedback)
        if record.done and float(record.update_reward) == 0.0
    )
    swapped_feedback = list(trace.feedback)
    swapped_feedback[success_feedback_index] = replace(
        swapped_feedback[success_feedback_index],
        canonical_reward=trace.feedback[failure_feedback_index].canonical_reward,
        update_reward=trace.feedback[failure_feedback_index].update_reward,
    )
    swapped_feedback[failure_feedback_index] = replace(
        swapped_feedback[failure_feedback_index],
        canonical_reward=trace.feedback[success_feedback_index].canonical_reward,
        update_reward=trace.feedback[success_feedback_index].update_reward,
    )
    swapped_closes = list(trace.closes)
    swapped_closes[0] = replace(
        swapped_closes[0], final_feedback_key=trace.closes[1].final_feedback_key
    )
    swapped_closes[1] = replace(
        swapped_closes[1], final_feedback_key=trace.closes[0].final_feedback_key
    )
    independent_swap_attacks = (
        attacks[-1],
        _replace_trace_component(trace, "actions", swapped_actions),
        _replace_trace_component(trace, "transitions", swapped_transitions),
        _replace_trace_component(trace, "feedback", swapped_feedback),
        _replace_trace_component(trace, "closes", swapped_closes),
    )

    if len(attacks) != len(TRACE_ATTACK_CLASSES):
        raise ContractError("trace attack matrix changed")
    table_before = _table_sha256(canonical_fit.table)

    def rejected_by_fitter(attacked: TraceBundle) -> bool:
        rejected = _expect_contract_error(
            lambda: _fit_td(
                _validate_trace(attacked, specs, mode="canonical"), gate
            )
        )
        if _table_sha256(canonical_fit.table) != table_before:
            raise ContractError("trace attack changed the sealed learner state")
        return rejected

    rejection_vector = tuple(
        (
            all(rejected_by_fitter(candidate) for candidate in independent_swap_attacks)
            if index == len(attacks) - 1
            else rejected_by_fitter(attacked)
        )
        for index, attacked in enumerate(attacks)
    )
    rejected = sum(rejection_vector)
    reordered = _independently_reordered_trace(trace)
    reordered_rows = _validate_trace(reordered, specs, mode="canonical")
    reordered_fit = _fit_td(reordered_rows, gate)
    reorder_exact = (
        _td_rows_sha256(base_rows) == _td_rows_sha256(reordered_rows)
        and _table_sha256(reordered_fit.table) == table_before
    )
    return {
        "attack_classes": len(TRACE_ATTACK_CLASSES),
        "attacks_rejected": rejected,
        "component_reorder_exact": reorder_exact,
        "passed": rejected == len(TRACE_ATTACK_CLASSES) and reorder_exact,
        "td_projection_sha256": _td_rows_sha256(base_rows),
        "trace_sha256": _trace_sha256(trace),
    }


def _pending_transition_audit() -> dict[str, object]:
    spec = next(_iter_episode_specs("train"))
    action = _immutable_scalar(0.0, "i1")
    predecessor = _primary_observation(spec, 0, 0)
    successor = _public_successor(predecessor, action)
    if successor is None:
        raise ContractError("pending-transition fixture lost its successor")
    update_reward = _immutable_scalar(0.0, "<f8")
    records = _expected_runtime_records(
        spec, 0, 0, action, "canonical", predecessor, successor, update_reward
    )
    sink = _TraceSink()
    sink.begin_episode(spec.key)
    authorization = _issue_step_authorization(
        sink, spec, 0, 0, action, "canonical"
    )
    sink.prepare(*records, authorization=authorization)
    wrong_transition = replace(records[2], successor_sha256="0" * 64)
    exact_identity_rejected = _expect_contract_error(
        lambda: sink.append(wrong_transition, records[3])
    )
    pending_cleared = sink.pending is None and sink._open_episode is None  # noqa: SLF001
    duplicate_sink = _TraceSink()
    duplicate_sink.begin_episode(spec.key)
    duplicate_authorization = _issue_step_authorization(
        duplicate_sink, spec, 0, 0, action, "canonical"
    )
    duplicate_sink.prepare(*records, authorization=duplicate_authorization)
    duplicate_sink.append(records[2], records[3])
    duplicate_rejected = _expect_contract_error(
        lambda: duplicate_sink.append(records[2], records[3])
    )

    cross_sink = _TraceSink()
    cross_sink.begin_episode(spec.key)
    other_spec = list(_iter_episode_specs("train"))[1]
    cross_records = _expected_runtime_records(
        other_spec,
        0,
        0,
        action,
        "canonical",
        _primary_observation(other_spec, 0, 0),
        _primary_observation(other_spec, 1, 0),
        update_reward,
    )
    cross_episode_rejected = _expect_contract_error(
        lambda: cross_sink.prepare(
            *cross_records,
            authorization=_issue_step_authorization(
                cross_sink, spec, 0, 0, action, "canonical"
            ),
        )
    )
    wrong_action = _immutable_scalar(1.0, "i1")
    wrong_action_records = _expected_runtime_records(
        spec,
        0,
        0,
        wrong_action,
        "canonical",
        predecessor,
        _primary_observation(spec, 1, 1),
        update_reward,
    )
    wrong_source_records = list(records)
    wrong_source_records[2] = replace(
        records[2], source_episode=other_spec.key
    )
    wrong_donor_records = list(records)
    wrong_donor_records[2] = replace(
        records[2],
        donor_episode=other_spec.key,
        donor_payload_sha256=_observation_sha256(successor),
    )
    wrong_origin_records = list(records)
    wrong_origin_records[3] = replace(
        records[3], origin_episode=other_spec.key
    )

    def reject_self_consistent(
        candidate: Sequence[object], authorized_action: np.ndarray = action
    ) -> tuple[bool, bool]:
        candidate_sink = _TraceSink()
        candidate_sink.begin_episode(spec.key)
        candidate_authorization = _issue_step_authorization(
            candidate_sink, spec, 0, 0, authorized_action, "canonical"
        )
        rejected = _expect_contract_error(
            lambda: candidate_sink.prepare(
                *candidate,  # type: ignore[arg-type]
                authorization=candidate_authorization,
            )
        )
        cleared = (
            candidate_sink.pending is None
            and candidate_sink._open_episode is None  # noqa: SLF001
        )
        return rejected, cleared

    self_consistent_results = (
        reject_self_consistent(wrong_action_records),
        reject_self_consistent(tuple(wrong_source_records)),
        reject_self_consistent(tuple(wrong_donor_records)),
        reject_self_consistent(tuple(wrong_origin_records)),
    )
    self_consistent_rejected = all(result[0] for result in self_consistent_results)
    self_consistent_cleared = all(result[1] for result in self_consistent_results)
    passed = (
        exact_identity_rejected
        and pending_cleared
        and duplicate_rejected
        and cross_episode_rejected
        and self_consistent_rejected
        and self_consistent_cleared
        and duplicate_sink.pending is None
        and cross_sink.pending is None
    )
    return {
        "cross_episode_rejected": cross_episode_rejected,
        "duplicate_append_rejected": duplicate_rejected,
        "exact_identity_rejected": exact_identity_rejected,
        "passed": passed,
        "pending_cleared_after_rejection": pending_cleared,
    }


class _UpdateGuard:
    def __init__(self) -> None:
        self.active = False

    def run(self, callback: Callable[[], None] | None = None) -> None:
        if self.active:
            raise ContractError("update callback reentered")
        self.active = True
        try:
            if callback is not None:
                callback()
        finally:
            self.active = False


def _assert_split_clean(sink: _TraceSink, origin_queue: Sequence[EpisodeKey]) -> None:
    if sink.pending is not None or sink._open_episode is not None:  # noqa: SLF001
        raise ContractError("split crossed with pending transition")
    if origin_queue:
        raise ContractError("split crossed with pending reward origin")


def _timing_attack_audit(
    gate: _FamilyGateToken,
    canonical_rows: _AuthenticatedRows,
    canonical_fit: FitResult,
) -> dict[str, object]:
    _require_family_gate(gate)
    canonical_rows = _require_authenticated_train_rows(
        canonical_rows, allowed_modes={"canonical"}
    )
    table_before = _table_sha256(canonical_fit.table)
    evidence: dict[str, dict[str, object]] = {}
    collection_attacks = tuple(
        name
        for name in TIMING_ATTACKS
        if name not in {"reentrant_update_callback", "heldout_source_during_fit"}
    )
    expected_lazy_counts = {
        "successor_during_select": (1, 0),
        "reward_during_select": (1, 0),
        "invalid_action_successor_resolution": (0, 0),
        "invalid_action_reward_resolution": (0, 0),
        "successor_before_action_validation": (1, 0),
        "nonterminal_terminal_scalar": (1, 0),
        "missing_nonterminal_zero": (1, 1),
        "duplicate_nonterminal_feedback": (2, 2),
        "next_transition_before_pending_append": (2, 2),
        "terminal_scalar_before_phase_three_action": (1, 0),
        "terminal_scalar_after_extra_transition": (7, 6),
        "duplicate_terminal_scalar": (9, 8),
        "origin_resolution_before_terminal": (1, 0),
        "origin_bearing_policy_feedback": (0, 0),
        "reentrant_select_environment_step": (0, 0),
        "duplicate_episode_close": (8, 8),
        "nonempty_pending_transition_at_split": (2, 2),
        "nonempty_origin_queue_at_split": (0, 0),
    }
    expected_source_materializations = {
        name: (
            HORIZON
            if name in {
                "terminal_scalar_after_extra_transition",
                "duplicate_terminal_scalar",
                "duplicate_episode_close",
            }
            else 1
        )
        for name in collection_attacks
    }
    for name in collection_attacks:
        source = _CountingSource(
            tuple(_iter_episode_specs("train")), expected_split="train"
        )
        source_episodes = source.open("train")
        first_source_episode = next(source_episodes)
        report: dict[str, object] = {}
        mode = (
            "reward_origin_control"
            if name in {
                "origin_resolution_before_terminal",
                "origin_bearing_policy_feedback",
                "nonempty_origin_queue_at_split",
            }
            else "canonical"
        )
        try:
            rejected = _expect_contract_error(
                lambda name=name, report=report, mode=mode: _collect_trace(
                    (first_source_episode,),
                    mode=mode,
                    gate=gate,
                    attack=name,
                    attack_report=report,
                )
            )
        finally:
            source.close()
        expected_attempted, expected_permitted = expected_lazy_counts[name]
        exact = bool(
            rejected
            and report.get("boundary_rejected") is True
            and report.get("collector_factory_reached") is True
            and report.get("pending_cleared") is True
            and report.get("open_episode_cleared") is True
            and report.get("origin_queue_cleared") is True
            and report.get("lazy_attempted") == expected_attempted
            and report.get("lazy_permitted") == expected_permitted
            and source.counts
            == {
                "factory": 1,
                "iterator": 1,
                "materializer": expected_source_materializations[name],
                "close": 1,
                "post_close": 0,
            }
            and _table_sha256(canonical_fit.table) == table_before
        )
        evidence[name] = {
            "exact": exact,
            "lazy_attempted": int(report.get("lazy_attempted", -1)),
            "lazy_permitted": int(report.get("lazy_permitted", -1)),
        }

    update_rejected = _expect_contract_error(
        lambda: _fit_td(canonical_rows, gate, reentrant_attack=True)
    )
    evidence["reentrant_update_callback"] = {
        "exact": bool(
            update_rejected and _table_sha256(canonical_fit.table) == table_before
        ),
        "lazy_attempted": 0,
        "lazy_permitted": 0,
    }

    train_source = _CountingSource(
        tuple(_iter_episode_specs("train")), expected_split="train"
    )
    exploding_validation = _CountingSource(
        tuple(_iter_episode_specs("validation")),
        expected_split="validation",
        exploding=True,
    )
    untouched_test = _CountingSource(
        tuple(_iter_episode_specs("test")), expected_split="test"
    )
    orchestrator = _SourceOrchestrator(
        train_source, exploding_validation, untouched_test, gate
    )
    heldout_rejected = _expect_contract_error(
        lambda: orchestrator.fit(attack_heldout_source=True)
    )
    exploding_validation.close()
    evidence["heldout_source_during_fit"] = {
        "exact": bool(
            heldout_rejected
            and exploding_validation.counts
            == {
                "factory": 1,
                "iterator": 0,
                "materializer": 0,
                "close": 1,
                "post_close": 0,
            }
            and not any(untouched_test.counts.values())
            and _table_sha256(canonical_fit.table) == table_before
        ),
        "lazy_attempted": 0,
        "lazy_permitted": 0,
    }
    if set(evidence) != set(TIMING_ATTACKS):
        raise ContractError("timing attack matrix changed")
    rejected = sum(bool(item["exact"]) for item in evidence.values())
    lazy_attempts = sum(int(item["lazy_attempted"]) for item in evidence.values())
    lazy_permitted = sum(int(item["lazy_permitted"]) for item in evidence.values())
    table_after = _table_sha256(canonical_fit.table)
    passed = rejected == len(TIMING_ATTACKS) and table_before == table_after
    if not passed:
        raise ContractError("real-boundary timing attack gate failed")
    return {
        "attack_classes": len(TIMING_ATTACKS),
        "attacks_rejected": rejected,
        "lazy_attempts": lazy_attempts,
        "lazy_permitted": lazy_permitted,
        "passed": passed,
        "table_unchanged": table_before == table_after,
    }


def _dependency_probe_audit(
    canonical_rows: _AuthenticatedRows,
    canonical_fit: FitResult,
) -> dict[str, object]:
    if (
        type(canonical_rows) is not _AuthenticatedRows
        or type(canonical_rows.rows) is not tuple
        or type(canonical_rows.token) is not _TraceGateToken
        or canonical_rows.token.mode != "canonical"
        or canonical_rows.token.token_sha256 not in _ISSUED_TRACE_GATES
        or len(canonical_rows.rows) != EPISODE_COUNTS["train"] * HORIZON
    ):
        raise ContractError("dependency probe lacks the previously authenticated trace")
    groups: dict[
        tuple[int, int, int, int], list[TDInputRow]
    ] = defaultdict(list)
    for row in canonical_rows:
        predecessor = _validate_observation(row.predecessor)
        action = _validate_action(row.action)
        state = _state_key(predecessor)
        if type(row.done) is not bool:
            raise ContractError("dependency row done flag changed")
        if row.done:
            if row.successor is not None or state[1] != HORIZON - 1:
                raise ContractError("dependency terminal row shape changed")
            # Deliberately do not inspect row.update_reward here.  The immutable
            # aggregate sealed by the production fit is the base table.
        else:
            if row.successor is None or state[1] == HORIZON - 1:
                raise ContractError("dependency bootstrap row shape changed")
            if float(_validate_reward(row.update_reward)) != 0.0:
                raise ContractError("dependency bootstrap row acquired feedback")
            successor_state = _state_key(row.successor)
            if successor_state[1] != state[1] + 1:
                raise ContractError("dependency bootstrap boundary changed")
        groups[(*state, int(action))].append(row)
    terminal_means = canonical_fit.terminal_means
    terminal_keys = {key for key in ALL_STATE_ACTION_KEYS if key[1] == 3}
    if (
        type(terminal_means) is not type(MappingProxyType({}))
        or set(terminal_means) != terminal_keys
        or any(type(value) is not np.float64 for value in terminal_means.values())
    ):
        raise ContractError("dependency probe did not receive the immutable fit aggregate")
    bootstrap, coverage = _strip_kernel_inputs(groups)
    manifests: list[dict[str, object]] = []
    all_exact = True
    scalar_reads = {"selected": 0, "unselected": 0}

    def protected_row_projection(row: TDInputRow) -> dict[str, object]:
        return {
            "predecessor": _json_ready(row.predecessor),
            "action": _json_ready(row.action),
            "successor": None if row.successor is None else _json_ready(row.successor),
            "done": row.done,
        }

    def manifest_projection(manifest: DependencyProbeManifest) -> dict[str, object]:
        return {
            "family_sha256": manifest.family_sha256,
            "canonical_trace_sha256": manifest.canonical_trace_sha256,
            "canonical_rows_sha256": manifest.canonical_rows_sha256,
            "selected_leaf": list(manifest.selected_leaf),
            "selected_records_sha256": manifest.selected_records_sha256,
            "selected_record_count": manifest.selected_record_count,
            "base_terminal_update_reward": manifest.base_terminal_update_reward,
            "probe_terminal_update_reward": manifest.probe_terminal_update_reward,
            "allowed_difference": list(manifest.allowed_difference),
            "base_projection_sha256": manifest.base_projection_sha256,
            "probe_projection_sha256": manifest.probe_projection_sha256,
            "expected_changed_cells_by_sweep": [
                [list(cell) for cell in sweep]
                for sweep in manifest.expected_changed_cells_by_sweep
            ],
        }

    for sign in (0, 1):
        leaf = (sign, 3, 7 if sign else 0, sign)
        selected_indices = tuple(
            index
            for index, row in enumerate(canonical_rows)
            if _state_key(row.predecessor) + (int(row.action),) == leaf
        )
        if len(selected_indices) != 64:
            raise ContractError("dependency leaf lost exact repeated coverage")
        selected_records_projection = [
            {
                "index": index,
                "protected_row": protected_row_projection(canonical_rows[index]),
            }
            for index in selected_indices
        ]
        probe_rows = tuple(
            replace(row, update_reward=_immutable_scalar(0.0, "<f8"))
            if index in selected_indices
            else row
            for index, row in enumerate(canonical_rows)
        )
        expected_chain = (
            leaf,
            (sign, 2, 3 if sign else 0, sign),
            (sign, 1, 1 if sign else 0, sign),
            (sign, 0, 0, sign),
        )
        expected_changed = tuple(
            tuple(expected_chain[:sweep]) for sweep in range(1, HORIZON + 1)
        )
        manifest = DependencyProbeManifest(
            family_sha256=EXPECTED_FAMILY_SHA256,
            canonical_trace_sha256=canonical_rows.token.trace_sha256,
            canonical_rows_sha256=canonical_rows.token.rows_sha256,
            selected_leaf=leaf,
            selected_records_sha256=_json_sha256(selected_records_projection),
            selected_record_count=len(selected_indices),
            base_terminal_update_reward=1.0,
            probe_terminal_update_reward=0.0,
            allowed_difference=("terminal_update_reward",),
            base_projection_sha256=canonical_rows.token.rows_sha256,
            probe_projection_sha256=_json_sha256(
                {
                    "base_rows_sha256": canonical_rows.token.rows_sha256,
                    "selected_records_sha256": _json_sha256(
                        selected_records_projection
                    ),
                    "selected_indices": list(selected_indices),
                    "terminal_update_reward": 0.0,
                }
            ),
            expected_changed_cells_by_sweep=expected_changed,
        )
        base_aggregate, probe_aggregate, selected_reads, unselected_reads = (
            _validate_dependency_projection(
            manifest, canonical_rows, probe_rows, selected_indices
            )
        )
        scalar_reads["selected"] += selected_reads
        scalar_reads["unselected"] += unselected_reads
        base_means = dict(terminal_means)
        probe_means = dict(terminal_means)
        base_means[leaf] = base_aggregate
        probe_means[leaf] = probe_aggregate
        base = _run_td_kernel(
            bootstrap, MappingProxyType(base_means), coverage
        )
        probe = _run_td_kernel(
            bootstrap, MappingProxyType(probe_means), coverage
        )
        changed_counts: list[int] = []
        exact_sets = True
        for sweep, (base_snapshot, probe_snapshot) in enumerate(
            zip(base.audit.snapshots, probe.audit.snapshots, strict=True), start=1
        ):
            changed = {
                key
                for key in ALL_STATE_ACTION_KEYS
                if base_snapshot[key] != probe_snapshot[key]
            }
            expected = set(manifest.expected_changed_cells_by_sweep[sweep - 1])
            changed_counts.append(len(changed))
            exact_sets &= changed == expected
        all_exact &= exact_sets and changed_counts == [1, 2, 3, 4]
        attacked = list(probe_rows)
        first_index = selected_indices[0]
        attacked[first_index] = replace(
            attacked[first_index],
            action=_immutable_scalar(1 - int(attacked[first_index].action), "i1"),
        )
        nonselected_index = next(
            index for index in range(len(attacked)) if index not in selected_indices
        )
        attacked_nonselected = list(probe_rows)
        attacked_nonselected[nonselected_index] = replace(
            attacked_nonselected[nonselected_index],
            update_reward=_immutable_scalar(1.0, "<f8"),
        )
        attacked_selected_reward = list(probe_rows)
        attacked_selected_reward[first_index] = replace(
            attacked_selected_reward[first_index],
            update_reward=_immutable_scalar(1.0, "<f8"),
        )
        mutation_rejections = sum(
            (
                _expect_contract_error(
                    lambda candidate=tuple(attacked): _validate_dependency_projection(
                        manifest, canonical_rows, candidate, selected_indices
                    )
                ),
                _expect_contract_error(
                    lambda candidate=tuple(attacked_nonselected): _validate_dependency_projection(
                        manifest, canonical_rows, candidate, selected_indices
                    )
                ),
                _expect_contract_error(
                    lambda candidate=probe_rows[:-1]: _validate_dependency_projection(
                        manifest, canonical_rows, candidate, selected_indices
                    )
                ),
                _expect_contract_error(
                    lambda candidate=tuple(attacked_selected_reward): _validate_dependency_projection(
                        manifest, canonical_rows, candidate, selected_indices
                    )
                ),
            )
        )
        all_exact &= (
            mutation_rejections == 4
            and base.audit.aggregate_lookups_by_sweep == (32, 32, 32, 32)
            and probe.audit.aggregate_lookups_by_sweep == (32, 32, 32, 32)
        )
        manifests.append(
            {
                "manifest": manifest_projection(manifest),
                "changed_counts": changed_counts,
                "mutation_rejections": mutation_rejections,
                "sign": sign,
            }
        )
    return {
        "aggregate_lookups_by_sweep": [32, 32, 32, 32],
        "changed_cells_by_sweep": [1, 2, 3, 4],
        "manifest_sha256": _json_sha256(manifests),
        "opposite_chain_unchanged": all_exact,
        "passed": all_exact and scalar_reads == {"selected": 256, "unselected": 0},
        "probe_count": 2,
        "selected_scalar_reads": scalar_reads["selected"],
        "unselected_scalar_reads": scalar_reads["unselected"],
    }


def _validate_dependency_projection(
    manifest: DependencyProbeManifest,
    base: _AuthenticatedRows,
    probe: Sequence[TDInputRow],
    selected_indices: tuple[int, ...],
) -> tuple[np.float64, np.float64, int, int]:
    expected_probe_sha256 = _json_sha256(
        {
            "base_rows_sha256": base.token.rows_sha256,
            "selected_records_sha256": manifest.selected_records_sha256,
            "selected_indices": list(selected_indices),
            "terminal_update_reward": 0.0,
        }
    )
    if (
        type(manifest) is not DependencyProbeManifest
        or len(base) != 8192
        or len(probe) != len(base)
        or manifest.family_sha256 != EXPECTED_FAMILY_SHA256
        or manifest.canonical_trace_sha256 != base.token.trace_sha256
        or manifest.canonical_rows_sha256 != base.token.rows_sha256
        or manifest.selected_record_count != 64
        or len(selected_indices) != 64
        or manifest.base_terminal_update_reward != 1.0
        or manifest.probe_terminal_update_reward != 0.0
        or manifest.allowed_difference != ("terminal_update_reward",)
        or manifest.base_projection_sha256 != base.token.rows_sha256
        or manifest.probe_projection_sha256 != expected_probe_sha256
    ):
        raise ContractError("dependency substitution count changed")
    selected = set(selected_indices)
    selected_projection: list[dict[str, object]] = []
    base_values: list[float] = []
    probe_values: list[float] = []
    selected_scalar_reads = 0
    unselected_scalar_reads = 0
    for index, (original, changed) in enumerate(zip(base, probe, strict=True)):
        same_protected = (
            original.predecessor.tobytes() == changed.predecessor.tobytes()
            and original.action.tobytes() == changed.action.tobytes()
            and (
                (original.successor is None and changed.successor is None)
                or (
                    original.successor is not None
                    and changed.successor is not None
                    and original.successor.tobytes() == changed.successor.tobytes()
                )
            )
            and original.done is changed.done
        )
        if not same_protected:
            raise ContractError("dependency substitution changed a protected field")
        if index in selected:
            if (
                _state_key(original.predecessor) + (int(original.action),)
                != manifest.selected_leaf
                or not original.done
                or original.successor is not None
            ):
                raise ContractError("dependency selected substitution changed")
            original_reward = float(_validate_reward(original.update_reward))
            changed_reward = float(_validate_reward(changed.update_reward))
            selected_scalar_reads += 2
            if original_reward != 1.0 or changed_reward != 0.0:
                raise ContractError("dependency selected scalar substitution changed")
            base_values.append(original_reward)
            probe_values.append(changed_reward)
            selected_projection.append(
                {
                    "index": index,
                    "protected_row": {
                        "predecessor": _json_ready(original.predecessor),
                        "action": _json_ready(original.action),
                        "successor": None,
                        "done": original.done,
                    },
                }
            )
        elif changed is not original:
            raise ContractError("dependency substitution touched an unselected row")
    if _json_sha256(selected_projection) != manifest.selected_records_sha256:
        raise ContractError("dependency selected record commitment changed")
    if (
        len(base_values) != 64
        or len(probe_values) != 64
        or selected_scalar_reads != 128
        or unselected_scalar_reads != 0
    ):
        raise ContractError("dependency selected scalar read count changed")
    return (
        np.float64(sum(base_values) / len(base_values)),
        np.float64(sum(probe_values) / len(probe_values)),
        selected_scalar_reads,
        unselected_scalar_reads,
    )


def _behavior_summary(
    trace: TraceBundle,
    specs: Sequence[EpisodeSpec],
    *,
    mode: str,
) -> dict[str, object]:
    mean, regret, trace_sha256 = _trace_score(trace, specs, mode=mode)
    reward_sum = len(specs) - regret
    return {
        "macro": float(mean),
        "regret": int(regret),
        "reward_sum": int(reward_sum),
        "trace_sha256": trace_sha256,
        "transitions": len(specs) * HORIZON,
    }


def _synchronous_td_audit(
    rows: Sequence[TDInputRow], fitted: FitResult
) -> dict[str, object]:
    groups = _group_td_rows(rows)
    bootstrap, coverage = _strip_kernel_inputs(groups)
    terminal_means, raw_reads = _terminal_means(groups)
    expected_positive_sets: list[set[tuple[int, int, int, int]]] = []
    chain_zero = (
        (0, 3, 0, 0),
        (0, 2, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
    )
    chain_one = (
        (1, 3, 7, 1),
        (1, 2, 3, 1),
        (1, 1, 1, 1),
        (1, 0, 0, 1),
    )
    for sweep in range(1, HORIZON + 1):
        expected_positive_sets.append(set(chain_zero[:sweep]) | set(chain_one[:sweep]))
    exact_positive_sets = all(
        {
            key for key, value in snapshot.items() if float(value) > 0.0
        }
        == expected
        for snapshot, expected in zip(
            fitted.audit.snapshots, expected_positive_sets, strict=True
        )
    )
    table_types_exact = all(type(value) is np.float64 for value in fitted.table.values())
    raw_terminal_unreachable = (
        set(bootstrap) == {key for key in ALL_STATE_ACTION_KEYS if key[1] < 3}
        and set(terminal_means) == {key for key in ALL_STATE_ACTION_KEYS if key[1] == 3}
        and all(key[1] < 3 for key in bootstrap)
        and all(coverage[key] == {0: 512, 1: 256, 2: 128, 3: 64}[key[1]] for key in coverage)
    )
    invalid_sweeps_rejected = _expect_contract_error(
        lambda: _run_td_kernel(bootstrap, terminal_means, coverage, sweeps=3)
    )
    passed = (
        raw_reads == 2048
        and fitted.audit.raw_terminal_reads == 2048
        and tuple(int(value) for value in fitted.audit.writes_by_sweep)
        == (60, 60, 60, 60)
        and sum(int(value) for value in fitted.audit.writes_by_sweep) == 240
        and tuple(int(value) for value in fitted.audit.positive_cells_by_sweep)
        == (2, 4, 6, 8)
        and fitted.audit.aggregate_lookups_by_sweep == (32, 32, 32, 32)
        and exact_positive_sets
        and table_types_exact
        and raw_terminal_unreachable
        and invalid_sweeps_rejected
    )
    return {
        "aggregate_lookups_by_sweep": [int(value) for value in fitted.audit.aggregate_lookups_by_sweep],
        "exact_positive_sets": exact_positive_sets,
        "invalid_sweeps_rejected": invalid_sweeps_rejected,
        "passed": passed,
        "positive_cells_by_sweep": [int(value) for value in fitted.audit.positive_cells_by_sweep],
        "raw_terminal_reads": int(fitted.audit.raw_terminal_reads),
        "raw_terminal_unreachable": raw_terminal_unreachable,
        "table_float64_exact": table_types_exact,
        "total_writes": sum(int(value) for value in fitted.audit.writes_by_sweep),
        "writes_by_sweep": [int(value) for value in fitted.audit.writes_by_sweep],
    }


def _performance_positive_gate(
    fitted: FitResult,
    train: Mapping[str, object],
    validation: Mapping[str, object],
    test: Mapping[str, object],
    *,
    comparator_metrics: Mapping[str, Mapping[str, float]],
    structural_invariants: bool,
) -> bool:
    return bool(
        structural_invariants
        and tuple(int(value) for value in fitted.audit.writes_by_sweep)
        == (60, 60, 60, 60)
        and tuple(int(value) for value in fitted.audit.positive_cells_by_sweep)
        == (2, 4, 6, 8)
        and float(train["macro"]) >= THRESHOLDS["minimum_postfit_train_macro_return"]
        and float(validation["macro"])
        >= THRESHOLDS["minimum_postfit_validation_macro_return"]
        and float(test["macro"]) >= THRESHOLDS["minimum_postfit_test_macro_return"]
        and float(validation["minimum_regime"])
        >= THRESHOLDS["minimum_heldout_regime_return"]
        and float(test["minimum_regime"])
        >= THRESHOLDS["minimum_heldout_regime_return"]
        and float(validation["macro"]) - float(comparator_metrics["constant"]["validation"])
        >= THRESHOLDS["minimum_validation_gain_constant"]
        and float(test["macro"]) - float(comparator_metrics["constant"]["test"])
        >= THRESHOLDS["minimum_test_gain_constant"]
        and float(validation["macro"]) - float(comparator_metrics["myopic"]["validation"])
        >= THRESHOLDS["minimum_validation_gain_myopic"]
        and float(test["macro"]) - float(comparator_metrics["myopic"]["test"])
        >= THRESHOLDS["minimum_test_gain_myopic"]
        and float(validation["macro"]) - float(comparator_metrics["no_bootstrap"]["validation"])
        >= THRESHOLDS["minimum_validation_gain_no_bootstrap"]
        and float(test["macro"]) - float(comparator_metrics["no_bootstrap"]["test"])
        >= THRESHOLDS["minimum_test_gain_no_bootstrap"]
        and float(validation["macro"]) - float(comparator_metrics["random"]["validation"])
        >= THRESHOLDS["minimum_validation_gain_random"]
        and float(test["macro"]) - float(comparator_metrics["random"]["test"])
        >= THRESHOLDS["minimum_test_gain_random"]
        and validation["state_before_sha256"] == validation["state_after_sha256"]
        and test["state_before_sha256"] == test["state_after_sha256"]
        and int(validation["updates"]) == int(test["updates"]) == 0
    )


def _canonical_recovery_audit(
    trace: TraceBundle,
    rows: Sequence[TDInputRow],
    fitted: FitResult,
    comparator_metrics: Mapping[str, Mapping[str, float]],
    validation: Mapping[str, object],
    test: Mapping[str, object],
) -> dict[str, object]:
    train_specs = tuple(_iter_episode_specs("train"))
    behavior = _behavior_summary(trace, train_specs, mode="canonical")
    train = _evaluate_policy(fitted.table, train_specs)
    table_sha256 = _table_sha256(fitted.table)
    heldout_state_unchanged = (
        validation["state_before_sha256"]
        == validation["state_after_sha256"]
        == test["state_before_sha256"]
        == test["state_after_sha256"]
        == table_sha256
    )
    structural = (
        behavior["macro"] == 0.0625
        and behavior["reward_sum"] == 128
        and behavior["regret"] == 1920
        and behavior["transitions"] == 8192
        and len(rows) == 8192
        and fitted.audit.raw_terminal_reads == 2048
        and heldout_state_unchanged
    )
    positive_gate = _performance_positive_gate(
        fitted,
        train,
        validation,
        test,
        comparator_metrics=comparator_metrics,
        structural_invariants=structural,
    )
    return {
        "behavior_regret": int(behavior["regret"]),
        "behavior_reward_sum": int(behavior["reward_sum"]),
        "behavior_train_macro_return": float(behavior["macro"]),
        "heldout_state_unchanged": heldout_state_unchanged,
        "heldout_updates": int(validation["updates"]) + int(test["updates"]),
        "minimum_test_regime_return": float(test["minimum_regime"]),
        "minimum_validation_regime_return": float(validation["minimum_regime"]),
        "passed": positive_gate,
        "postfit_test_macro_return": float(test["macro"]),
        "postfit_train_macro_return": float(train["macro"]),
        "postfit_validation_macro_return": float(validation["macro"]),
        "table_sha256": table_sha256,
        "train_transitions": len(rows),
    }


def _control_audits(
    gate: _FamilyGateToken,
    canonical_trace: TraceBundle,
    canonical_fit: FitResult,
    comparator_metrics: Mapping[str, Mapping[str, float]],
    canonical_test: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    _require_family_gate(gate)
    train_specs = tuple(_iter_episode_specs("train"))
    validation_specs = tuple(_iter_episode_specs("validation"))
    test_specs = tuple(_iter_episode_specs("test"))
    random_streams = _random_action_streams()
    random_validation = float(
        _evaluate_action_stream(validation_specs, random_streams["validation"])["macro"]
    )
    random_test = float(
        _evaluate_action_stream(test_specs, random_streams["test"])["macro"]
    )
    canonical_behavior = _behavior_summary(
        canonical_trace, train_specs, mode="canonical"
    )
    def complete_successor_payloads(trace: TraceBundle) -> Counter[str]:
        observation_map = _unique_records(
            trace.observations, lambda item: item.key, "control successor observation"
        )
        payloads: Counter[str] = Counter()
        for transition in trace.transitions:
            if transition.done:
                continue
            if transition.successor_key is None:
                raise ContractError("control successor key is absent")
            record = observation_map.get(transition.successor_key)
            if type(record) is not ObservationRecord:
                raise ContractError("control successor observation is absent")
            payload = {
                "value": _json_ready(_validate_observation(record.value)),
                "value_sha256": record.value_sha256,
            }
            payloads[
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            ] += 1
        return payloads

    def transition_lineages_exact(trace: TraceBundle) -> bool:
        observation_map = _unique_records(
            trace.observations, lambda item: item.key, "control lineage observation"
        )
        for transition in trace.transitions:
            if transition.done:
                continue
            spec = _spec_index().get(transition.source_episode)
            if spec is None or transition.successor_key is None:
                return False
            predecessor_record = observation_map.get(transition.predecessor_key)
            successor_record = observation_map.get(transition.successor_key)
            if (
                type(predecessor_record) is not ObservationRecord
                or type(successor_record) is not ObservationRecord
            ):
                return False
            predecessor = _validate_observation(predecessor_record.value)
            action = transition.action_key.action_ordinal
            action_array = _immutable_scalar(float(int(action)), "i1")
            expected_successor = _transition_control_successor(
                predecessor,
                action_array,
                donor_initial_observation=_primary_observation(
                    _paired_spec(spec), 0, 0
                ),
            )
            if expected_successor is None:
                return False
            phase = int(transition.predecessor_key.phase)
            prefix = int(transition.predecessor_key.prefix_code)
            expected_prefix = (prefix << 1) | int(action)
            expected_key = _observation_key(spec, phase + 1, expected_prefix)
            if (
                transition.action_key.observation_key != transition.predecessor_key
                or transition.key.action_key != transition.action_key
                or transition.successor_key != expected_key
                or successor_record.key != expected_key
                or successor_record.value.tobytes(order="C")
                != expected_successor.tobytes(order="C")
                or successor_record.value_sha256
                != _observation_sha256(expected_successor)
                or transition.successor_sha256
                != successor_record.value_sha256
            ):
                return False
        return True

    transition_source = _CountingSource(train_specs, expected_split="train")
    transition_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    transition_test_source = _CountingSource(test_specs, expected_split="test")
    transition_orchestrator = _SourceOrchestrator(
        transition_source,
        transition_validation_source,
        transition_test_source,
        gate,
    )
    (
        transition_trace,
        transition_rows,
        transition_fit,
        transition_collection,
    ) = transition_orchestrator.fit(mode="transition_control")
    transition_validation, transition_test = (
        transition_orchestrator.evaluate_heldout()
    )
    _require_complete_source_counts(transition_source, "train")
    _require_complete_source_counts(transition_validation_source, "validation")
    _require_complete_source_counts(transition_test_source, "test")
    transition_train = _evaluate_policy(transition_fit.table, train_specs)
    canonical_successors = complete_successor_payloads(canonical_trace)
    transition_successors = complete_successor_payloads(transition_trace)
    complete_transition_lineages = transition_lineages_exact(transition_trace)
    donor_exact = all(
        record.donor_episode == _paired_spec(_spec_index()[record.source_episode]).key
        and record.donor_episode != record.source_episode
        and record.donor_payload_sha256 == record.successor_sha256
        for record in transition_trace.transitions
        if not record.done
    ) and all(
        record.donor_episode is None and record.donor_payload_sha256 is None
        for record in transition_trace.transitions
        if record.done
    )
    transition_behavior = _behavior_summary(
        transition_trace, train_specs, mode="transition_control"
    )
    transition_structural = (
        transition_collection["event_order_exact"]
        and len(transition_rows) == 8192
        and transition_behavior["macro"] == canonical_behavior["macro"] == 0.0625
        and transition_behavior["reward_sum"] == canonical_behavior["reward_sum"] == 128
        and transition_behavior["regret"] == canonical_behavior["regret"] == 1920
        and donor_exact
        and canonical_successors == transition_successors
        and complete_transition_lineages
    )
    transition_positive = _performance_positive_gate(
        transition_fit,
        transition_train,
        transition_validation,
        transition_test,
        comparator_metrics=comparator_metrics,
        structural_invariants=transition_structural,
    )
    transition_passed = (
        transition_structural
        and float(transition_validation["macro"])
        <= THRESHOLDS["maximum_transition_control_validation_macro_return"]
        and float(transition_test["macro"])
        <= THRESHOLDS["maximum_transition_control_test_macro_return"]
        and float(canonical_test["macro"]) - float(transition_test["macro"])
        >= THRESHOLDS["minimum_transition_control_test_gap"]
        and not transition_positive
    )
    transition_case = {
        "behavior_exact": transition_behavior["macro"] == 0.0625 and transition_behavior["regret"] == 1920,
        "donor_involution_exact": donor_exact,
        "passed": transition_passed,
        "positive_gate_rejected": not transition_positive,
        "successor_multiset_unchanged": canonical_successors == transition_successors,
        "test_macro_return": float(transition_test["macro"]),
        "trace_sha256": _trace_sha256(transition_trace),
        "true_test_gap": float(canonical_test["macro"]) - float(transition_test["macro"]),
        "validation_macro_return": float(transition_validation["macro"]),
    }

    reward_source = _CountingSource(train_specs, expected_split="train")
    reward_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    reward_test_source = _CountingSource(test_specs, expected_split="test")
    reward_orchestrator = _SourceOrchestrator(
        reward_source, reward_validation_source, reward_test_source, gate
    )
    reward_trace, reward_rows, reward_fit, reward_collection = reward_orchestrator.fit(
        mode="reward_origin_control"
    )
    reward_validation, reward_test = reward_orchestrator.evaluate_heldout()
    _require_complete_source_counts(reward_source, "train")
    _require_complete_source_counts(reward_validation_source, "validation")
    _require_complete_source_counts(reward_test_source, "test")
    reward_train = _evaluate_policy(reward_fit.table, train_specs)
    terminal_feedback = {
        record.key.transition_key.action_key.observation_key.episode_key: record
        for record in reward_trace.feedback
        if record.done
    }
    cell_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for spec in train_specs:
        cell_values[
            (int(spec.regime.code), int(spec.target_slot), int(spec.action_code))
        ].append(float(_validate_reward(terminal_feedback[spec.key].update_reward)))
    per_cell_exact = len(cell_values) == 128 and all(
        len(values) == 16 and sum(values) == 1.0 and sum(values) / len(values) == 0.0625
        for values in cell_values.values()
    )
    canonical_multiset = Counter(
        float(_validate_reward(record.update_reward))
        for record in canonical_trace.feedback
        if record.done
    )
    reward_multiset = Counter(
        float(_validate_reward(record.update_reward))
        for record in reward_trace.feedback
        if record.done
    )
    origin_exact = all(
        terminal_feedback[spec.key].origin_episode == _reward_origin_spec(spec).key
        and terminal_feedback[spec.key].origin_episode != spec.key
        for spec in train_specs
    )
    reward_behavior = _behavior_summary(
        reward_trace, train_specs, mode="reward_origin_control"
    )
    reward_structural = (
        reward_collection["event_order_exact"]
        and reward_collection["origin_queue_empty"]
        and len(reward_rows) == 8192
        and reward_behavior["macro"] == canonical_behavior["macro"] == 0.0625
        and reward_behavior["regret"] == canonical_behavior["regret"] == 1920
        and per_cell_exact
        and origin_exact
        and canonical_multiset == reward_multiset
        and tuple(int(value) for value in reward_fit.audit.positive_cells_by_sweep)
        == (32, 48, 56, 60)
    )
    reward_positive = _performance_positive_gate(
        reward_fit,
        reward_train,
        reward_validation,
        reward_test,
        comparator_metrics=comparator_metrics,
        structural_invariants=reward_structural,
    )
    reward_passed = (
        reward_structural
        and float(reward_validation["macro"])
        <= THRESHOLDS["maximum_reward_origin_validation_macro_return"]
        and float(reward_test["macro"])
        <= THRESHOLDS["maximum_reward_origin_test_macro_return"]
        and float(canonical_test["macro"]) - float(reward_test["macro"])
        >= THRESHOLDS["minimum_reward_origin_test_gap"]
        and not reward_positive
    )
    reward_case = {
        "behavior_exact": reward_behavior["macro"] == 0.0625 and reward_behavior["regret"] == 1920,
        "origin_mapping_exact": origin_exact,
        "passed": reward_passed,
        "per_cell_count_mean_exact": per_cell_exact,
        "positive_cells_by_sweep": [int(value) for value in reward_fit.audit.positive_cells_by_sweep],
        "positive_gate_rejected": not reward_positive,
        "reward_multiset_unchanged": canonical_multiset == reward_multiset,
        "test_macro_return": float(reward_test["macro"]),
        "trace_sha256": _trace_sha256(reward_trace),
        "true_test_gap": float(canonical_test["macro"]) - float(reward_test["macro"]),
        "validation_macro_return": float(reward_validation["macro"]),
        "zero_early_origin_materializations": bool(
            reward_collection["origin_queue_empty"]
            and reward_collection["origin_lazy_early"] == 0
            and reward_collection["origin_lazy_attempted"]
            == reward_collection["origin_lazy_permitted"]
            == EPISODE_COUNTS["train"]
        ),
    }

    ablated_source = _CountingSource(train_specs, expected_split="train")
    ablated_metric_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    ablated_metric_test_source = _CountingSource(
        test_specs, expected_split="test"
    )
    ablated_orchestrator = _SourceOrchestrator(
        ablated_source,
        ablated_metric_validation_source,
        ablated_metric_test_source,
        gate,
    )
    ablated_trace, ablated_rows, ablated_fit, ablated_collection = (
        ablated_orchestrator.fit(mode="signal_ablation")
    )
    ablated_validation, ablated_test = ablated_orchestrator.evaluate_heldout()
    _require_complete_source_counts(ablated_source, "train")
    _require_complete_source_counts(
        ablated_metric_validation_source, "validation"
    )
    _require_complete_source_counts(ablated_metric_test_source, "test")
    ablated_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    (
        ablated_validation_trace,
        ablated_validation_rows,
        ablated_validation_collection,
    ) = _collect_from_source(
        ablated_validation_source,
        gate,
        expected_split="validation",
        mode="signal_ablation",
    )
    ablated_test_source = _CountingSource(test_specs, expected_split="test")
    (
        ablated_test_trace,
        ablated_test_rows,
        ablated_test_collection,
    ) = _collect_from_source(
        ablated_test_source,
        gate,
        expected_split="test",
        mode="signal_ablation",
    )
    ablated_train = _evaluate_policy(
        ablated_fit.table, train_specs, observation_mode="signal_ablation"
    )
    true_ablated_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    true_ablated_test_source = _CountingSource(test_specs, expected_split="test")
    true_ablated_validation = _evaluate_from_source(
        canonical_fit.table,
        true_ablated_validation_source,
        expected_split="validation",
        observation_mode="signal_ablation",
    )
    true_ablated_test = _evaluate_from_source(
        canonical_fit.table,
        true_ablated_test_source,
        expected_split="test",
        observation_mode="signal_ablation",
    )
    _require_complete_source_counts(true_ablated_validation_source, "validation")
    _require_complete_source_counts(true_ablated_test_source, "test")
    ablated_structural = (
        ablated_collection["event_order_exact"]
        and ablated_validation_collection["event_order_exact"]
        and ablated_test_collection["event_order_exact"]
        and len(ablated_rows) == 8192
        and len(ablated_validation_rows) == 4096
        and len(ablated_test_rows) == 4096
        and all(float(row.predecessor[1]) == 0.0 for row in ablated_rows)
        and all(
            row.successor is None or float(row.successor[1]) == 0.0
            for row in ablated_rows
        )
        and all(
            float(row.predecessor[1]) == 0.0
            and (row.successor is None or float(row.successor[1]) == 0.0)
            for row in (*ablated_validation_rows, *ablated_test_rows)
        )
    )
    ablated_positive = _performance_positive_gate(
        ablated_fit,
        ablated_train,
        ablated_validation,
        ablated_test,
        comparator_metrics=comparator_metrics,
        structural_invariants=ablated_structural,
    )
    signal_passed = (
        ablated_structural
        and float(ablated_validation["macro"])
        <= THRESHOLDS["maximum_signal_attribution_macro_return"]
        and float(ablated_test["macro"])
        <= THRESHOLDS["maximum_signal_attribution_macro_return"]
        and float(true_ablated_validation["macro"])
        <= THRESHOLDS["maximum_signal_attribution_macro_return"]
        and float(true_ablated_test["macro"])
        <= THRESHOLDS["maximum_signal_attribution_macro_return"]
        and not ablated_positive
    )
    signal_case = {
        "only_signal_changed": ablated_structural,
        "passed": signal_passed,
        "positive_gate_rejected": not ablated_positive,
        "refit_test_macro_return": float(ablated_test["macro"]),
        "refit_validation_macro_return": float(ablated_validation["macro"]),
        "trace_sha256": _trace_sha256(ablated_trace),
        "true_policy_ablated_test_macro_return": float(true_ablated_test["macro"]),
        "true_policy_ablated_validation_macro_return": float(true_ablated_validation["macro"]),
    }
    return transition_case, reward_case, signal_case


CASE_CONTRACT = {
    "typed_episodic_contract": {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "done_pattern": [False, False, False, True],
        "event_order": list(EVENT_ORDER),
        "horizon": HORIZON,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [6],
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
    },
    "complete_family_replay": {
        "expected_family_sha256": EXPECTED_FAMILY_SHA256,
        "nonterminal_rows": 57344,
        "predecessor_nodes": 61440,
        "rows": 122880,
        "terminal_rows": 65536,
        "unique_transition_keys": 122880,
    },
    "target_swap_twin": {
        "public_bytes": "frozen",
        "reward_changes": 8192,
        "rows": 122880,
        "target_field": "separate_hidden_evaluator_target",
    },
    "generator_partition": {
        "episode_counts": EPISODE_COUNTS,
        "episodes_per_regime": EPISODES_PER_REGIME,
        "regime_counts": REGIME_COUNTS,
        "rng": "none",
    },
    "realized_path_disjointness": {
        "identity_fields_excluded": True,
        "path_count": 4096,
        "rows": 16384,
    },
    "train_only_source_boundary": {
        "fit_scope": "authenticated_train_source_only",
        "heldout_updates": 0,
        "source_constructions": ["absent", "exploding", "lazy"],
    },
    "lazy_information_boundary": {
        "attack_classes": list(TIMING_ATTACKS),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
    },
    "pending_transition_authentication": {
        "append": "exact_pending_identity_once",
        "rejection": "abort_and_clear",
    },
    "keyed_trace_authentication": {
        "attack_classes": list(TRACE_ATTACK_CLASSES),
        "reorder": "independent_nonidentity_component_permutations",
    },
    "synchronous_td_order": {
        "gamma": 1.0,
        "positive_cells_by_sweep": [2, 4, 6, 8],
        "sweeps": 4,
        "writes_by_sweep": [60, 60, 60, 60],
    },
    "all_boundary_terminal_dependency": {
        "changed_cells_by_sweep": [1, 2, 3, 4],
        "probe_signs": [0, 1],
        "selected_scalar_reads_per_probe": 64,
    },
    "baseline_replay": {
        "comparators": [
            "constant_zero",
            "constant_one",
            "feedback_only_myopic",
            "no_bootstrap",
            "seeded_random",
        ],
        "random_seed": RANDOM_BASELINE_SEED,
        "random_stream_sha256": EXPECTED_RANDOM_STREAM_SHA256,
    },
    "multistep_td_recovery": {
        "metric": "macro_mean_terminal_return_equal_weight_by_regime",
        "thresholds": THRESHOLDS,
        "tie_action": 0,
    },
    "transition_target_control": {
        "mapping": "opposite_target_slot_same_regime_block_action_code",
        "thresholds": {
            "maximum_test": THRESHOLDS["maximum_transition_control_test_macro_return"],
            "maximum_validation": THRESHOLDS["maximum_transition_control_validation_macro_return"],
            "minimum_test_gap": THRESHOLDS["minimum_transition_control_test_gap"],
        },
    },
    "reward_origin_control": {
        "mapping": "block_plus_one_action_code_plus_block",
        "per_cell_count": 16,
        "per_cell_mean": 0.0625,
        "thresholds": {
            "maximum_test": THRESHOLDS["maximum_reward_origin_test_macro_return"],
            "maximum_validation": THRESHOLDS["maximum_reward_origin_validation_macro_return"],
            "minimum_test_gap": THRESHOLDS["minimum_reward_origin_test_gap"],
        },
    },
    "signal_attribution_control": {
        "maximum_macro_return": THRESHOLDS["maximum_signal_attribution_macro_return"],
        "signal_index": 1,
        "variants": ["refit_without_signal", "true_policy_without_signal"],
    },
    "control_difference_whitelists": {
        "all_legal_rows": 122880,
        "controls": [
            "target_swap",
            "transition_target",
            "reward_origin",
            "signal_ablation",
        ],
    },
    "sanitized_result_contract": {
        "top_level_fields": [
            "action",
            "cases",
            "environment",
            "fixture",
            "schema_version",
            "status",
            "study_id",
        ],
        "worker_projection": "bounded_non_process_aggregate_cases",
    },
    "process_isolation": {
        "source_projection": "complete_non_process_cases",
        "workers": 2,
    },
}


CASE_REQUIRED_FIELDS = {
    "typed_episodic_contract": [
        "action_dtype", "action_values", "completed_episodes",
        "done_pattern_exact", "event_order", "horizon",
        "immutable_observations", "invalid_actions_rejected",
        "observation_dtype", "observation_fields", "observation_shape",
        "passed", "policy_input_fields", "reward_dtype", "reward_values",
        "structure_kind", "typed_keys_exact",
    ],
    "complete_family_replay": [
        "action_balance", "corruption_classes", "corruptions_rejected",
        "expected_digest_exact", "nonterminal_rows", "passed",
        "predecessor_nodes", "primary_sha256", "replay_exact",
        "replay_sha256", "rows", "target_balance", "terminal_rows",
        "train_cell_coverage_exact", "unique_transition_keys",
    ],
    "target_swap_twin": [
        "invalid_twin_rejections", "passed", "public_bytes_preserved",
        "public_sha256", "rows", "target_flips", "terminal_reward_changes",
        "twin_public_sha256",
    ],
    "generator_partition": [
        "episode_counts", "episodes_per_regime", "family_sha256",
        "generator_rng_calls", "passed", "regime_counts", "regime_sha256",
        "structure_kind",
    ],
    "realized_path_disjointness": [
        "identity_fields_excluded", "passed", "public_paths_split_disjoint",
        "public_rows_split_disjoint", "realized_rows",
        "unique_public_path_sha256_count", "unique_public_row_sha256_count",
    ],
    "train_only_source_boundary": [
        "absent_exploding_lazy_exact", "exploding_source_operations",
        "heldout_materializations", "heldout_state_unchanged",
        "heldout_updates", "inverse_train_operations", "passed",
        "sealed_fit_sha256", "train_trace_sha256",
    ],
    "lazy_information_boundary": [
        "attack_classes", "attacks_rejected", "canonical_lazy_attempts",
        "canonical_lazy_permitted", "event_order_exact", "lazy_audit_sha256",
        "passed", "policy_input_fields", "table_unchanged_after_attacks",
    ],
    "pending_transition_authentication": [
        "cross_episode_rejected", "duplicate_append_rejected",
        "exact_identity_rejected", "passed", "pending_cleared_after_rejection",
    ],
    "keyed_trace_authentication": [
        "attack_classes", "attacks_rejected", "component_reorder_exact",
        "passed", "td_projection_sha256", "trace_sha256",
    ],
    "synchronous_td_order": [
        "aggregate_lookups_by_sweep", "exact_positive_sets",
        "invalid_sweeps_rejected", "passed", "positive_cells_by_sweep",
        "raw_terminal_reads", "raw_terminal_unreachable",
        "table_float64_exact", "total_writes", "writes_by_sweep",
    ],
    "all_boundary_terminal_dependency": [
        "aggregate_lookups_by_sweep", "changed_cells_by_sweep",
        "manifest_sha256", "opposite_chain_unchanged", "passed",
        "probe_count", "selected_scalar_reads", "unselected_scalar_reads",
    ],
    "baseline_replay": [
        "constant_one_test_macro_return", "constant_zero_test_macro_return",
        "myopic_test_macro_return", "no_bootstrap_test_macro_return",
        "numpy_version", "passed", "random_stream_sha256",
        "random_test_macro_return", "replay_exact",
    ],
    "multistep_td_recovery": [
        "behavior_regret", "behavior_reward_sum", "behavior_train_macro_return",
        "heldout_state_unchanged", "heldout_updates",
        "minimum_test_regime_return", "minimum_validation_regime_return",
        "passed", "postfit_test_macro_return", "postfit_train_macro_return",
        "postfit_validation_macro_return", "table_sha256", "train_transitions",
    ],
    "transition_target_control": [
        "behavior_exact", "donor_involution_exact", "passed",
        "positive_gate_rejected", "successor_multiset_unchanged",
        "test_macro_return", "trace_sha256", "true_test_gap",
        "validation_macro_return",
    ],
    "reward_origin_control": [
        "behavior_exact", "origin_mapping_exact", "passed",
        "per_cell_count_mean_exact", "positive_cells_by_sweep",
        "positive_gate_rejected", "reward_multiset_unchanged",
        "test_macro_return", "trace_sha256", "true_test_gap",
        "validation_macro_return", "zero_early_origin_materializations",
    ],
    "signal_attribution_control": [
        "only_signal_changed", "passed", "positive_gate_rejected",
        "refit_test_macro_return", "refit_validation_macro_return",
        "trace_sha256", "true_policy_ablated_test_macro_return",
        "true_policy_ablated_validation_macro_return",
    ],
    "control_difference_whitelists": [
        "all_rows_checked", "canonical_rows", "combined_signal_rows_checked",
        "passed", "protected_mutation_rejections", "reward_origin_bijection",
        "reward_origin_rows_checked", "signal_ablation_rows_checked",
        "successor_multiset_preserved", "target_swap_rows_checked",
        "transition_involution", "transition_rows_checked",
    ],
    "sanitized_result_contract": [
        "case_count", "forbidden_samples_rejected", "passed", "schema_sha256",
        "top_level_fields",
    ],
    "process_isolation": ["passed", "trace_sha256"],
}


FIXTURE_IDENTITY = {
    "action_dtype": "int8",
    "action_values": [0, 1],
    "claim_boundary": "synthetic_cpu_four_step_synchronous_td_harness_only",
    "episode_counts": EPISODE_COUNTS,
    "episodes_per_regime": EPISODES_PER_REGIME,
    "event_order": list(EVENT_ORDER),
    "expected_family_sha256": EXPECTED_FAMILY_SHA256,
    "expected_random_stream_sha256": EXPECTED_RANDOM_STREAM_SHA256,
    "generator_regimes": [
        {
            "code": int(regime.code),
            "nuisance_scale": regime.nuisance_scale,
            "nuisance_shift": regime.nuisance_shift,
            "signal_scale": regime.signal_scale,
            "split": regime.split,
        }
        for regime in REGIMES
    ],
    "horizon": HORIZON,
    "observation_dtype": "float64",
    "observation_fields": list(OBSERVATION_FIELDS),
    "observation_shape": [6],
    "policy_input_fields": list(POLICY_INPUT_FIELDS),
    "random_baseline_seed": RANDOM_BASELINE_SEED,
    "regime_counts": REGIME_COUNTS,
    "reward_dtype": "float64",
    "reward_values": [0.0, 1.0],
    "structure_kind": STRUCTURE_KIND,
    "td_gamma": 1.0,
    "td_sweeps": 4,
    "thresholds": THRESHOLDS,
    "tie_action": 0,
}


def _typed_contract_case(
    trace: TraceBundle, collection: Mapping[str, object]
) -> dict[str, object]:
    invalid_actions: list[object] = [
        0,
        True,
        0.0,
        np.bool_(False),
        np.asarray(0, dtype=np.int64),
        np.asarray([0], dtype=np.int8),
        np.asarray(2, dtype=np.int8),
        np.asarray(0, dtype=np.int8),
    ]
    for value in invalid_actions[:-1]:
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    invalid_actions[-1].setflags(write=True)
    action_rejections = sum(
        _expect_contract_error(lambda value=value: _validate_action(value))
        for value in invalid_actions
    )
    done_pattern_exact = all(
        tuple(record.done for record in trace.transitions[offset : offset + HORIZON])
        == (False, False, False, True)
        for offset in range(0, len(trace.transitions), HORIZON)
    )
    immutable_observations = all(
        not record.value.flags.writeable
        and record.value.flags.c_contiguous
        and record.value.strides == (8,)
        for record in trace.observations
    )
    typed_keys_exact = all(
        _validate_observation_key(record.key) is record.key
        for record in trace.observations
    )
    passed = (
        action_rejections == len(invalid_actions)
        and done_pattern_exact
        and immutable_observations
        and typed_keys_exact
        and bool(collection["event_order_exact"])
        and int(collection["completed_episodes"]) == EPISODE_COUNTS["train"]
    )
    return {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "completed_episodes": int(collection["completed_episodes"]),
        "done_pattern_exact": done_pattern_exact,
        "event_order": list(EVENT_ORDER),
        "horizon": HORIZON,
        "immutable_observations": immutable_observations,
        "invalid_actions_rejected": action_rejections,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [6],
        "passed": passed,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
        "typed_keys_exact": typed_keys_exact,
    }


def _complete_family_case(family: Mapping[str, object]) -> dict[str, object]:
    passed = bool(
        family["expected_sha256"]
        and family["replay_exact"]
        and family["rows"] == 122880
        and family["nonterminal_rows"] == 57344
        and family["terminal_rows"] == 65536
        and family["predecessor_nodes"] == 61440
        and family["unique_transition_keys"] == 122880
        and family["corruptions_rejected"] == family["corruption_classes"]
        and family["complete_public_split_disjoint"]
        and family["factory_count_before_gate"] == 0
    )
    return {
        "action_balance": bool(family["action_balance"]),
        "corruption_classes": int(family["corruption_classes"]),
        "corruptions_rejected": int(family["corruptions_rejected"]),
        "expected_digest_exact": bool(family["expected_sha256"]),
        "nonterminal_rows": int(family["nonterminal_rows"]),
        "passed": passed,
        "predecessor_nodes": int(family["predecessor_nodes"]),
        "primary_sha256": str(family["primary_sha256"]),
        "replay_exact": bool(family["replay_exact"]),
        "replay_sha256": str(family["replay_sha256"]),
        "rows": int(family["rows"]),
        "target_balance": bool(family["target_balance"]),
        "terminal_rows": int(family["terminal_rows"]),
        "train_cell_coverage_exact": bool(family["train_cell_coverage_exact"]),
        "unique_transition_keys": int(family["unique_transition_keys"]),
    }


def _target_swap_case(audit: Mapping[str, object]) -> dict[str, object]:
    passed = bool(
        audit["rows"] == 122880
        and audit["target_flips"] == 122880
        and audit["terminal_reward_changes"] == 8192
        and audit["public_bytes_preserved"]
        and audit["public_sha256"] == audit["twin_public_sha256"]
        and audit["invalid_twin_rejections"] == 3
    )
    return {**dict(audit), "passed": passed}


def _generator_partition_case(family: Mapping[str, object]) -> dict[str, object]:
    regime_sha256 = _json_sha256(FIXTURE_IDENTITY["generator_regimes"])
    passed = (
        EPISODE_COUNTS == {"train": 2048, "validation": 1024, "test": 1024}
        and REGIME_COUNTS == {"train": 4, "validation": 2, "test": 2}
        and family["primary_sha256"] == EXPECTED_FAMILY_SHA256
    )
    return {
        "episode_counts": dict(EPISODE_COUNTS),
        "episodes_per_regime": EPISODES_PER_REGIME,
        "family_sha256": EXPECTED_FAMILY_SHA256,
        "generator_rng_calls": 0,
        "passed": passed,
        "regime_counts": dict(REGIME_COUNTS),
        "regime_sha256": regime_sha256,
        "structure_kind": STRUCTURE_KIND,
    }


def _realized_path_case(audit: Mapping[str, object]) -> dict[str, object]:
    passed = bool(
        audit["realized_rows"] == 16384
        and audit["unique_public_row_sha256_count"] == 16384
        and audit["unique_public_path_sha256_count"] == 4096
        and audit["public_rows_split_disjoint"]
        and audit["public_paths_split_disjoint"]
        and audit["identity_fields_excluded"]
    )
    return {**dict(audit), "passed": passed}


_RAW_RESULT_KEYS = {
    "action",
    "action_stream",
    "actions",
    "context",
    "credentials",
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
    "parameter_values",
    "path",
    "paths",
    "policy_state",
    "q_value",
    "q_table",
    "raw_actions",
    "raw_observations",
    "raw_rewards",
    "return",
    "reward",
    "rewards",
    "secret",
    "state",
    "states",
    "successor",
    "successors",
    "target",
    "targets",
    "topology",
    "trajectory",
    "trajectories",
    "transition",
    "transitions",
    "weight",
}
_RAW_RESULT_FRAGMENTS = (
    "credential",
    "donor_array",
    "gradient",
    "observation_array",
    "origin_array",
    "parameter",
    "private_evidence",
    "q_table",
    "q_value",
    "raw_",
    "secret",
    "state_array",
    "target_values",
    "trajectory",
)
_RAW_CONTAINER_TOKENS = (
    "actions",
    "logs",
    "observations",
    "paths",
    "returns",
    "rewards",
    "states",
    "successors",
    "targets",
    "transitions",
)
_SAFE_AGGREGATE_RAW_KEYS = {
    "raw_terminal_reads",
    "raw_terminal_unreachable",
}
_RAW_CONTAINER_EXACT_KEYS = {
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
    "target",
    "trajectory",
    "transition",
    "weight",
}


def _sanitized_key_is_raw(key: str, item: object, *, depth: int) -> bool:
    normalized = key.lower()
    if normalized in _SAFE_AGGREGATE_RAW_KEYS and type(item) in (bool, int):
        return False
    if normalized in _RAW_RESULT_KEYS and not (
        normalized == "action" and depth == 0 and type(item) is str
    ):
        return True
    if any(fragment in normalized for fragment in _RAW_RESULT_FRAGMENTS):
        return True
    if type(item) not in (list, dict):
        return False
    if normalized in _RAW_CONTAINER_EXACT_KEYS:
        return True
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
    return any(token in _RAW_CONTAINER_TOKENS for token in tokens)


def _validate_sanitized_projection(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise ContractError("sanitized projection exceeded nesting limit")
    if value is None or type(value) in (bool, int, float):
        if isinstance(value, float) and not np.isfinite(value):
            raise ContractError("sanitized projection contains a non-finite value")
        return
    if type(value) is str:
        if (
            len(value) > 512
            or str(REPOSITORY_ROOT).lower() in value.lower()
            or re.match(r"^[a-zA-Z]:", value) is not None
            or "/" in value
            or "\\" in value
        ):
            raise ContractError("sanitized projection contains an unsafe string")
        return
    if type(value) is list:
        if len(value) > 64:
            raise ContractError("sanitized projection list is unbounded")
        for item in value:
            _validate_sanitized_projection(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > 64:
            raise ContractError("sanitized projection object is unbounded")
        for key, item in value.items():
            if (
                type(key) is not str
                or len(key) > 128
                or re.match(r"^[a-zA-Z]:", key) is not None
                or "/" in key
                or "\\" in key
                or _sanitized_key_is_raw(key, item, depth=depth)
            ):
                raise ContractError("sanitized projection contains a raw field")
            if key.lower().endswith("sha256") and (
                type(item) is not str
                or len(item) != 64
                or any(character not in "0123456789abcdef" for character in item)
            ):
                raise ContractError("sanitized projection contains a malformed digest")
            _validate_sanitized_projection(item, depth=depth + 1)
        return
    raise ContractError("sanitized projection contains a non-JSON value")


def _sanitizer_case(cases: Mapping[str, object]) -> dict[str, object]:
    _validate_sanitized_projection(dict(cases))
    malicious = (
        {"raw_observations": [0.0]},
        {"observation": [0.0]},
        {"action": [0]},
        {"case": {"action": 0}},
        {"actions": [0, 1]},
        {"successor": [[0.0]]},
        {"successors": [[0.0]]},
        {"successors": 0},
        {"rows": [{"x": 1}]},
        {"data": [1]},
        {"records": [{"x": 1}]},
        {"raw_state": [0.0]},
        {"rewards": [1.0]},
        {"reward": [1.0]},
        {"reward": 1.0},
        {"return": [1.0]},
        {"return": 1.0},
        {"path": [{"step": 0}]},
        {"path": "private"},
        {"transition": [{"step": 0}]},
        {"trajectory": [{"step": 0}]},
        {"log": [{"step": 0}]},
        {"weight": [1.0]},
        {"gradient": [1.0]},
        {"q_value": [1.0]},
        {"q_table": [0.0]},
        {"policy_state": {}},
        {"credentials": "x"},
        {"topology": "x"},
        {"returns": [1.0]},
        {"target_values": [0, 1]},
        {"foo_transitions_bar": [{"x": 1}]},
        {"policy_parameters": [0.0]},
        {"private_evidence": "x"},
        {"donor_array_alias": [1]},
        {"donor_array": 1},
        {"origin_array_alias": [1]},
        {"origin_array": 1},
        {"q_values": [0.0]},
        {"trace_sha256": "not-a-digest"},
        {"safe": "C:\\private\\evidence.json"},
        {"safe": "C:private\\evidence.json"},
        {"safe": "\\private\\evidence.json"},
        {"safe": "/private/evidence.json"},
        {"safe": "private/evidence.json"},
        {"safe": "private\\evidence.json"},
        {"safe": "\\\\server\\share\\evidence.json"},
        {"private/evidence": 1},
        {"safe": np.asarray([1.0])},
        {"safe": float("nan")},
    )
    rejected = sum(
        _expect_contract_error(
            lambda sample=sample: _validate_sanitized_projection(sample)
        )
        for sample in malicious
    )
    schema_projection = {
        name: sorted(case)
        for name, case in cases.items()
        if isinstance(case, Mapping)
    }
    passed = rejected == len(malicious)
    return {
        "case_count": len(cases) + 1,
        "forbidden_samples_rejected": rejected,
        "passed": passed,
        "schema_sha256": _json_sha256(schema_projection),
        "top_level_fields": sorted(
            [
                "action",
                "cases",
                "environment",
                "fixture",
                "schema_version",
                "status",
                "study_id",
            ]
        ),
    }


@lru_cache(maxsize=1)
def _non_process_projection_cached() -> dict[str, object]:
    gate, family, target_swap, realized_paths, whitelist_case = (
        _run_family_preflight()
    )
    train_specs = tuple(_iter_episode_specs("train"))
    validation_specs = tuple(_iter_episode_specs("validation"))
    test_specs = tuple(_iter_episode_specs("test"))
    result_train_source = _CountingSource(train_specs, expected_split="train")
    result_validation_source = _CountingSource(
        validation_specs, expected_split="validation"
    )
    result_test_source = _CountingSource(test_specs, expected_split="test")
    result_orchestrator = _SourceOrchestrator(
        result_train_source, result_validation_source, result_test_source, gate
    )
    trace, rows, fitted, collection = result_orchestrator.fit()
    canonical_validation, canonical_test = result_orchestrator.evaluate_heldout()
    _require_complete_source_counts(result_train_source, "train")
    _require_complete_source_counts(result_validation_source, "validation")
    _require_complete_source_counts(result_test_source, "test")
    reordered_rows = _validate_trace(
        _independently_reordered_trace(trace), train_specs, mode="canonical"
    )
    if _td_rows_sha256(rows) != _td_rows_sha256(reordered_rows):
        raise ContractError("full canonical component reorder changed TD rows")
    baseline_case, comparator_metrics = _baseline_audit(
        trace,
        rows,
        fitted,
        gate,
        canonical_validation,
        canonical_test,
    )
    transition_case, reward_case, signal_case = _control_audits(
        gate, trace, fitted, comparator_metrics, canonical_test
    )
    timing = _timing_attack_audit(gate, rows, fitted)
    cases: dict[str, object] = {
        "typed_episodic_contract": _typed_contract_case(trace, collection),
        "complete_family_replay": _complete_family_case(family),
        "target_swap_twin": _target_swap_case(target_swap),
        "generator_partition": _generator_partition_case(family),
        "realized_path_disjointness": _realized_path_case(realized_paths),
        "train_only_source_boundary": _source_boundary_audit(gate),
        "lazy_information_boundary": {
            "attack_classes": timing["attack_classes"],
            "attacks_rejected": timing["attacks_rejected"],
            "canonical_lazy_attempts": int(collection["lazy_attempted"]),
            "canonical_lazy_permitted": int(collection["lazy_permitted"]),
            "event_order_exact": bool(collection["event_order_exact"]),
            "lazy_audit_sha256": str(collection["lazy_audit_sha256"]),
            "passed": bool(
                timing["passed"]
                and collection["lazy_installed"] == collection["lazy_attempted"]
                == collection["lazy_permitted"]
                == 16384
            ),
            "policy_input_fields": list(POLICY_INPUT_FIELDS),
            "table_unchanged_after_attacks": timing["table_unchanged"],
        },
        "pending_transition_authentication": _pending_transition_audit(),
        "keyed_trace_authentication": _trace_attack_audit(
            gate, trace, rows, fitted
        ),
        "synchronous_td_order": _synchronous_td_audit(rows, fitted),
        "all_boundary_terminal_dependency": _dependency_probe_audit(rows, fitted),
        "baseline_replay": baseline_case,
        "multistep_td_recovery": _canonical_recovery_audit(
            trace,
            rows,
            fitted,
            comparator_metrics,
            canonical_validation,
            canonical_test,
        ),
        "transition_target_control": transition_case,
        "reward_origin_control": reward_case,
        "signal_attribution_control": signal_case,
        "control_difference_whitelists": whitelist_case,
    }
    cases["sanitized_result_contract"] = _sanitizer_case(cases)
    if set(cases) != set(CASE_CONTRACT) - {"process_isolation"}:
        raise ContractError("non-process case set changed")
    expected_fields = {
        name: set(fields)
        for name, fields in CASE_REQUIRED_FIELDS.items()
        if name != "process_isolation"
    }
    actual_fields = {
        name: set(case) for name, case in cases.items() if isinstance(case, Mapping)
    }
    if actual_fields != expected_fields:
        raise ContractError("non-process case field contract changed")
    _validate_sanitized_projection(cases)
    return cases


def _non_process_projection() -> dict[str, object]:
    return copy.deepcopy(_non_process_projection_cached())


def isolated_worker_trace() -> dict[str, object]:
    cases = _non_process_projection()
    return {
        "cases": cases,
        "fixture_sha256": _json_sha256(FIXTURE_IDENTITY),
        "projection_sha256": _json_sha256(cases),
    }


def _isolated_trace() -> dict[str, object]:
    safe_names = {
        "COMSPEC",
        "LD_LIBRARY_PATH",
        "PATH",
        "PATHEXT",
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
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.local_lab.multistep_td_action_prefix_v3_worker",
            "--mode",
            "multistep-td-action-prefix-v3-trace",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    return json.loads(completed.stdout)


def run_study(*, include_process_isolation: bool = True) -> dict[str, object]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the v3 synchronous-TD study requires a CPU backend")
    local_projection = isolated_worker_trace()
    cases = dict(local_projection["cases"])
    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = isolated_left == isolated_right == local_projection
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = _json_sha256("not-run-in-focused-test")
    cases["process_isolation"] = {
        "passed": isolation_passed,
        "trace_sha256": isolation_digest,
    }
    if set(cases) != set(CASE_REQUIRED_FIELDS) or any(
        set(case) != set(CASE_REQUIRED_FIELDS[name])
        for name, case in cases.items()
    ):
        raise ContractError("terminal case field contract changed")
    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    result = {
        "action": (
            "synthetic_four_step_synchronous_td_propagation_confirmed_for_harness"
            if passed
            else (
                "park_multistep_td_research"
                if completed
                else "no_decision_incomplete_study"
            )
        ),
        "cases": cases,
        "environment": {
            "device_kind": str(cpu_devices[0].device_kind),
            "jax_version": str(jax.__version__),
            "platform": str(cpu_devices[0].platform),
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
        },
        "fixture": {"case_contract": CASE_CONTRACT, **FIXTURE_IDENTITY},
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else ("failed" if completed else "incomplete"),
        "study_id": STUDY_ID,
    }
    _validate_sanitized_projection(result)
    return result
