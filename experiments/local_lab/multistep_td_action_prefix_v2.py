"""Pre-result-rejected synthetic multi-step TD action-prefix v2 fixture.

This source is retained only as an auditable preflight record. It is not an
approved local-lab study and must not be executed as terminal evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import jax
import numpy as np


STUDY_ID = "multistep-td-action-prefix-v2"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
EPISODES_PER_REGIME = 32
HORIZON = 4
RANDOM_BASELINE_SEED = 2026082817
OBSERVATION_FIELDS = ("phase", "signal", "action_prefix", "nuisance")
POLICY_INPUT_FIELDS = ("observation",)
STRUCTURE_KIND = "none"
EXPECTED_DATASET_SHA256 = (
    "dc20cd244cc2656a850f01dd1ce4bc1918a70488661317087ddde36913b9d8a1"
)
BEHAVIOR_ACTIONS = (
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
)
TRANSITION_DONOR_PERMUTATION = tuple(range(8, 16)) + tuple(range(8))
REWARD_ORIGIN_PERMUTATION = (
    12,
    13,
    4,
    5,
    14,
    15,
    8,
    9,
    0,
    1,
    6,
    7,
    2,
    3,
    10,
    11,
)
EVENT_ORDER = tuple(
    event
    for phase in range(HORIZON)
    for event in (
        f"observe{phase}",
        f"select{phase}",
        f"validate{phase}",
        f"transition{phase}",
        f"feedback{phase}",
        f"target{phase}",
        f"update{phase}",
        f"log{phase}",
    )
) + ("close_episode",)
FORBIDDEN_POLICY_FIELDS = (
    "target",
    "preferred_action",
    "reward",
    "done",
    "successor",
    "next_observation",
    "terminal_scalar",
    "step_key",
    "episode_key",
    "phase_counter",
    "split",
    "regime_code",
    "episode_identity",
    "rng",
    "evaluator",
    "generator",
    "heldout_source",
    "donor",
    "origin",
    "control_mode",
)
THRESHOLDS = {
    "maximum_attribution_test_macro_return": 0.55,
    "maximum_reward_origin_test_macro_return": 0.55,
    "maximum_reward_origin_validation_macro_return": 0.55,
    "maximum_transition_target_test_macro_return": 0.55,
    "maximum_transition_target_validation_macro_return": 0.55,
    "minimum_heldout_regime_return": 0.98,
    "minimum_postfit_test_macro_return": 0.99,
    "minimum_postfit_train_macro_return": 0.99,
    "minimum_postfit_validation_macro_return": 0.99,
    "minimum_test_gain_baseline": 0.30,
    "minimum_transition_target_gap": 0.40,
    "minimum_reward_origin_gap": 0.40,
    "minimum_validation_gain_baseline": 0.30,
}


@dataclass(frozen=True)
class Regime:
    split: str
    code: int
    signal_scale: float
    nuisance_shift: float
    nuisance_scale: float


REGIMES = (
    Regime("train", 1301, 0.70, -0.90, 0.80),
    Regime("train", 1303, 0.90, -0.30, 1.05),
    Regime("train", 1307, 1.10, 0.30, 0.75),
    Regime("train", 1319, 1.30, 0.90, 1.20),
    Regime("validation", 1409, 0.60, -1.40, 0.65),
    Regime("validation", 1423, 1.40, 1.40, 1.30),
    Regime("test", 1511, 0.50, -1.90, 0.55),
    Regime("test", 1523, 1.50, 1.90, 1.45),
)
REGIME_COUNTS = {"train": 4, "validation": 2, "test": 2}
EPISODE_COUNTS = {
    split: count * EPISODES_PER_REGIME
    for split, count in REGIME_COUNTS.items()
}

CASE_CONTRACT = {
    "typed_action_prefix_contract": [
        "actions_checked",
        "immutable_observations_checked",
        "legal_rows_checked",
        "passed",
        "trace_sha256",
    ],
    "target_independent_public_successors": [
        "passed",
        "successors_checked",
        "target_swap_outcomes_changed",
        "target_swap_twins_checked",
        "trace_sha256",
    ],
    "complete_legal_family_commitment": [
        "dataset_sha256",
        "episodes_checked",
        "legal_rows_checked",
        "nodes_checked",
        "nonterminal_rows_checked",
        "passed",
        "public_rows_disjoint",
        "split_keys_disjoint",
        "target_balance_exact",
        "terminal_rows_checked",
    ],
    "independent_family_replay": [
        "mutation_sentinels_checked",
        "mutation_sentinels_rejected",
        "passed",
        "replay_sha256",
        "trace_sha256",
    ],
    "physical_pre_action_boundary": [
        "exploding_sentinels_checked",
        "exploding_sentinels_rejected",
        "forbidden_fields_checked",
        "passed",
        "selector_inputs_checked",
        "trace_sha256",
    ],
    "heldout_absent_source": ["passed", "state_sha256", "train_trace_sha256"],
    "heldout_exploding_source": [
        "operations_checked",
        "operations_unreached",
        "passed",
        "state_sha256",
        "train_trace_sha256",
    ],
    "td_target_dependency": [
        "dependency_checks",
        "earlier_cells_unchanged",
        "passed",
        "trace_sha256",
    ],
    "td_update_order_and_terminal_dependency": [
        "bootstrap_updates",
        "event_orders_checked",
        "passed",
        "propagation_offsets_exact",
        "terminal_updates",
        "total_updates",
        "trace_sha256",
    ],
    "authenticated_component_recombination": [
        "components_authenticated",
        "passed",
        "reorder_variants",
        "scores_equal",
        "trace_sha256",
    ],
    "malformed_and_cross_episode_rejection": [
        "attacks_checked",
        "attacks_rejected",
        "passed",
        "state_unchanged_checks",
        "trace_sha256",
    ],
    "baseline_replay": [
        "best_baseline_test_macro_return",
        "best_baseline_validation_macro_return",
        "constant_returns_exact",
        "myopic_returns_exact",
        "no_bootstrap_returns_exact",
        "passed",
        "random_replay_exact",
        "trace_sha256",
    ],
    "multistep_value_recovery": [
        "behavior_regret",
        "behavior_return",
        "minimum_heldout_regime_return",
        "passed",
        "postfit_test_macro_return",
        "postfit_train_macro_return",
        "postfit_validation_macro_return",
        "test_gain_baseline",
        "trace_sha256",
        "validation_gain_baseline",
    ],
    "transition_target_control": [
        "donor_mapping_exact",
        "passed",
        "positive_gate_rejected",
        "test_macro_return",
        "trace_sha256",
        "true_test_gap",
        "validation_macro_return",
    ],
    "outcome_blind_reward_origin_control": [
        "cell_balance_exact",
        "mapping_outcome_blind",
        "origin_mapping_exact",
        "passed",
        "positive_gate_rejected",
        "reward_multiset_unchanged",
        "test_macro_return",
        "trace_sha256",
        "true_test_gap",
        "validation_macro_return",
    ],
    "full_timing_attack_matrix": [
        "attacks_checked",
        "attacks_rejected",
        "passed",
        "state_unchanged_checks",
        "trace_sha256",
    ],
    "all_trajectory_signal_ablation": [
        "legal_rows_checked",
        "only_signal_changed",
        "passed",
        "positive_gate_rejected",
        "refit_test_macro_return",
        "trace_sha256",
        "true_policy_test_macro_return",
    ],
    "process_isolation": ["passed", "trace_sha256"],
}

FIXTURE_IDENTITY = {
    "action_dtype": "int8",
    "action_values": [0, 1],
    "behavior_action_pattern": list(BEHAVIOR_ACTIONS),
    "canonical_donor_identity": "null",
    "claim_boundary": "synthetic_cpu_multistep_td_action_prefix_harness_only",
    "component_reorder_variants": [
        "canonical",
        "observations_reversed",
        "actions_rotate_left_one",
        "transitions_even_then_odd",
        "feedback_odd_then_even",
        "all_independent",
    ],
    "complete_family_counts": {
        "legal_rows": 7680,
        "nodes": 3840,
        "nonterminal_rows": 3584,
        "terminal_rows": 4096,
    },
    "discount": 1.0,
    "done_pattern": [False, False, False, True],
    "episode_counts": EPISODE_COUNTS,
    "episodes_per_regime": EPISODES_PER_REGIME,
    "event_order": list(EVENT_ORDER),
    "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
    "generator_regimes": [asdict(regime) for regime in REGIMES],
    "heldout_target_multiplier": 5,
    "horizon": HORIZON,
    "learning_rate": 1.0,
    "observation_dtype": "float64",
    "observation_fields": list(OBSERVATION_FIELDS),
    "observation_shape": [4],
    "policy_input_fields": list(POLICY_INPUT_FIELDS),
    "propagation_offsets_terminal_to_root": {
        "negative_signal": [0, 1, 2, 3],
        "positive_signal": [12, 13, 14, 15],
    },
    "random_baseline_seed": RANDOM_BASELINE_SEED,
    "random_split_tags": {"test": 2, "train": 0, "validation": 1},
    "regime_counts": REGIME_COUNTS,
    "reward_dtype": "float64",
    "reward_origin_permutation": list(REWARD_ORIGIN_PERMUTATION),
    "reward_values": [0.0, 1.0],
    "structure_kind": STRUCTURE_KIND,
    "table_shape": [4, 2, 8, 2],
    "thresholds": THRESHOLDS,
    "tie_rule": "lower_action",
    "timing_attacks": 43,
    "transition_donor_permutation": list(TRANSITION_DONOR_PERMUTATION),
}


@dataclass(frozen=True)
class Episode:
    regime: Regime
    index: int
    target: int
    episode_key: str


@dataclass(frozen=True)
class PendingSelection:
    observation_sha256: str
    coords: tuple[int, int, int]
    phase: int
    prefix: int
    action: int


@dataclass
class LearnerState:
    q: np.ndarray
    pending: PendingSelection | None
    expected_phase: int
    completed_episodes: int
    bootstrap_updates: int
    terminal_updates: int
    total_updates: int
    assigned_cell_checks: int


@dataclass(frozen=True)
class UpdateAudit:
    phase: int
    target_value: float
    successor_value: float | None
    changed_cells: int
    unchosen_cells_unchanged: bool


@dataclass(frozen=True)
class ObservationRecord:
    step_key: str
    episode_key: str
    phase: int
    predecessor_sha256: str
    observation_sha256: str
    action: int
    action_dtype: str
    done: bool


@dataclass(frozen=True)
class ActionRecord:
    step_key: str
    episode_key: str
    phase: int
    predecessor_sha256: str
    action: int
    action_dtype: str
    done: bool


@dataclass(frozen=True)
class TransitionRecord:
    step_key: str
    episode_key: str
    phase: int
    predecessor_sha256: str
    action: int
    action_dtype: str
    done: bool
    successor_key: str | None
    successor_sha256: str | None
    bootstrap_successor_sha256: str | None
    donor_episode: int | None


@dataclass(frozen=True)
class FeedbackRecord:
    step_key: str
    episode_key: str
    phase: int
    predecessor_sha256: str
    action: int
    action_dtype: str
    done: bool
    reward: float
    update_reward: float
    origin_episode: int | None


@dataclass(frozen=True)
class TraceBundle:
    observations: tuple[ObservationRecord, ...]
    actions: tuple[ActionRecord, ...]
    transitions: tuple[TransitionRecord, ...]
    feedback: tuple[FeedbackRecord, ...]


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _array_identity(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _observation_sha256(observation: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(observation).tobytes(order="C")
    ).hexdigest()


def _immutable_observation(values: Iterable[float]) -> np.ndarray:
    observation = np.ascontiguousarray(tuple(values), dtype=np.float64)
    if observation.shape != (4,) or not np.isfinite(observation).all():
        raise RuntimeError("invalid generated observation")
    observation.setflags(write=False)
    return observation


def _immutable_action(value: int) -> np.ndarray:
    action = np.asarray(value, dtype=np.int8)
    action.setflags(write=False)
    return action


def _immutable_reward(value: float) -> np.ndarray:
    reward = np.asarray(value, dtype=np.float64)
    reward.setflags(write=False)
    return reward


def _public_sign_bit(split: str, code: int, episode: int) -> int:
    if split == "train":
        return int(episode % 16 >= 8)
    return int(((5 * episode + code) % 32) >= 16)


def _evaluator_target(split: str, code: int, episode: int) -> int:
    if split == "train":
        local = episode - 16 * (episode // 16)
        return 0 if local < 8 else 1
    residue = (code + episode * 5) - 32 * ((code + episode * 5) // 32)
    return 1 if residue >= 16 else 0


def _public_observation(
    regime: Regime,
    episode: int,
    phase: int,
    prefix: int,
    *,
    ablate_signal: bool = False,
) -> np.ndarray:
    if phase not in range(HORIZON) or prefix not in range(2**phase):
        raise ValueError("illegal public action-prefix state")
    sign_bit = _public_sign_bit(regime.split, regime.code, episode)
    sign = -1.0 if sign_bit == 0 else 1.0
    signal = sign * regime.signal_scale * (
        1.0 + 0.02 * ((3 * episode + regime.code) % 9)
    )
    if ablate_signal:
        signal = 0.0
    base = regime.nuisance_shift + regime.nuisance_scale * (
        (((13 * episode + regime.code) % 31) - 15) / 15.0
    )
    nuisance = base + phase / 16.0 + prefix / 32.0
    return _immutable_observation((float(phase), signal, float(prefix), nuisance))


def _terminal_reward(target: int, prefix: int, action: int) -> float:
    word = 2 * prefix + action
    return float((target == 0 and word == 0) or (target == 1 and word == 15))


def _episode_key(regime: Regime, episode: int) -> str:
    return f"{regime.split}:{regime.code}:{episode}"


def _step_key(episode_key: str, phase: int, prefix: int) -> str:
    return f"{episode_key}:p{phase}:x{prefix}"


def _row_key(episode_key: str, phase: int, prefix: int, action: int) -> str:
    return f"{_step_key(episode_key, phase, prefix)}:a{action}"


def _episodes(split: str) -> tuple[Episode, ...]:
    rows = []
    for regime in REGIMES:
        if regime.split != split:
            continue
        for episode in range(EPISODES_PER_REGIME):
            target = _evaluator_target(split, regime.code, episode)
            rows.append(
                Episode(
                    regime=regime,
                    index=episode,
                    target=target,
                    episode_key=_episode_key(regime, episode),
                )
            )
    return tuple(rows)


def _family_projection() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        for episode in range(EPISODES_PER_REGIME):
            target = _evaluator_target(regime.split, regime.code, episode)
            episode_key = _episode_key(regime, episode)
            target_digest = _json_sha256(
                {"episode_key": episode_key, "target": target}
            )
            for phase in range(HORIZON):
                for prefix in range(2**phase):
                    predecessor = _public_observation(
                        regime, episode, phase, prefix
                    )
                    predecessor_digest = _observation_sha256(predecessor)
                    for action in (0, 1):
                        done = phase == HORIZON - 1
                        if done:
                            successor_key = None
                            successor_digest = None
                            reward = _terminal_reward(target, prefix, action)
                        else:
                            next_prefix = 2 * prefix + action
                            successor = _public_observation(
                                regime, episode, phase + 1, next_prefix
                            )
                            successor_key = _step_key(
                                episode_key, phase + 1, next_prefix
                            )
                            successor_digest = _observation_sha256(successor)
                            reward = 0.0
                        rows.append(
                            {
                                "action": action,
                                "done": done,
                                "episode": episode,
                                "episode_key": episode_key,
                                "observation_contract": {
                                    "c_contiguous": bool(
                                        predecessor.flags.c_contiguous
                                    ),
                                    "dtype": str(predecessor.dtype),
                                    "immutable": not predecessor.flags.writeable,
                                    "shape": list(predecessor.shape),
                                },
                                "phase": phase,
                                "predecessor_sha256": predecessor_digest,
                                "prefix": prefix,
                                "regime_code": regime.code,
                                "reward": reward,
                                "row_key": _row_key(
                                    episode_key, phase, prefix, action
                                ),
                                "split": regime.split,
                                "successor_key": successor_key,
                                "successor_sha256": successor_digest,
                                "target_sha256": target_digest,
                            }
                        )
    return tuple(sorted(rows, key=lambda row: str(row["row_key"])))


def _independent_family_projection() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        for episode in range(32):
            if regime.split == "train":
                target = 0 if episode - (episode // 16) * 16 < 8 else 1
                sign_bit = 0 if episode % 16 < 8 else 1
            else:
                residue = regime.code + 5 * episode
                residue -= 32 * (residue // 32)
                target = 1 if residue >= 16 else 0
                sign_bit = target
            sign = -1.0 if sign_bit == 0 else 1.0
            signal = sign * regime.signal_scale * (
                1.0 + 0.02 * ((regime.code + episode * 3) % 9)
            )
            base = regime.nuisance_shift + regime.nuisance_scale * (
                (((regime.code + episode * 13) % 31) - 15) / 15.0
            )
            episode_key = f"{regime.split}:{regime.code}:{episode}"
            target_digest = _json_sha256(
                {"episode_key": episode_key, "target": target}
            )
            for phase in (0, 1, 2, 3):
                for prefix in range(1 << phase):
                    predecessor = _immutable_observation(
                        (
                            float(phase),
                            signal,
                            float(prefix),
                            base + float(phase) / 16.0 + float(prefix) / 32.0,
                        )
                    )
                    predecessor_digest = hashlib.sha256(
                        predecessor.tobytes(order="C")
                    ).hexdigest()
                    for action in (0, 1):
                        terminal = phase == 3
                        if terminal:
                            successor_key = None
                            successor_digest = None
                            word = prefix * 2 + action
                            reward = float(
                                (target == 0 and word == 0)
                                or (target == 1 and word == 15)
                            )
                        else:
                            next_prefix = prefix * 2 + action
                            successor = _immutable_observation(
                                (
                                    float(phase + 1),
                                    signal,
                                    float(next_prefix),
                                    base
                                    + float(phase + 1) / 16.0
                                    + float(next_prefix) / 32.0,
                                )
                            )
                            successor_key = (
                                f"{episode_key}:p{phase + 1}:x{next_prefix}"
                            )
                            successor_digest = hashlib.sha256(
                                successor.tobytes(order="C")
                            ).hexdigest()
                            reward = 0.0
                        rows.append(
                            {
                                "action": action,
                                "done": terminal,
                                "episode": episode,
                                "episode_key": episode_key,
                                "observation_contract": {
                                    "c_contiguous": True,
                                    "dtype": "float64",
                                    "immutable": True,
                                    "shape": [4],
                                },
                                "phase": phase,
                                "predecessor_sha256": predecessor_digest,
                                "prefix": prefix,
                                "regime_code": regime.code,
                                "reward": reward,
                                "row_key": (
                                    f"{episode_key}:p{phase}:x{prefix}:a{action}"
                                ),
                                "split": regime.split,
                                "successor_key": successor_key,
                                "successor_sha256": successor_digest,
                                "target_sha256": target_digest,
                            }
                        )
    return tuple(sorted(rows, key=lambda row: str(row["row_key"])))


def generated_family_sha256() -> str:
    """Return the complete family commitment without executing a learner."""
    return _json_sha256(list(_family_projection()))


def _validate_observation(
    observation: object, *, allow_zero_signal: bool
) -> np.ndarray:
    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a NumPy array")
    if (
        observation.dtype != np.float64
        or observation.shape != (4,)
        or not observation.flags.c_contiguous
        or observation.flags.writeable
        or not np.isfinite(observation).all()
    ):
        raise ValueError("observation violates immutable float64[4]")
    phase_float, signal, prefix_float, _nuisance = map(float, observation)
    phase = int(phase_float)
    prefix = int(prefix_float)
    if (
        float(phase) != phase_float
        or phase not in range(HORIZON)
        or float(prefix) != prefix_float
        or prefix not in range(2**phase)
    ):
        raise ValueError("observation has an illegal action-prefix state")
    if signal == 0.0 and not allow_zero_signal:
        raise ValueError("canonical signal must be nonzero")
    return observation


def _validate_action(action: object) -> np.ndarray:
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, np.ndarray):
        raise TypeError("action must be a NumPy scalar int8 array")
    if action.dtype != np.int8 or action.shape != () or action.flags.writeable:
        raise ValueError("action violates immutable scalar int8")
    if int(action) not in (0, 1):
        raise ValueError("action must be zero or one")
    return action


def _validate_reward(reward: object) -> np.ndarray:
    if not isinstance(reward, np.ndarray):
        raise TypeError("reward must be a NumPy scalar array")
    if reward.dtype != np.float64 or reward.shape != () or reward.flags.writeable:
        raise ValueError("reward violates immutable scalar float64")
    if float(reward) not in (0.0, 1.0):
        raise ValueError("reward must be zero or one")
    return reward


def _policy_observation(record: object) -> np.ndarray:
    if not isinstance(record, dict) or set(record) != {"observation"}:
        raise RuntimeError("policy input must contain only observation")
    return _validate_observation(record["observation"], allow_zero_signal=True)


def _state_coords(observation: np.ndarray) -> tuple[int, int, int]:
    return (
        int(observation[0]),
        0 if float(observation[1]) <= 0.0 else 1,
        int(observation[2]),
    )


def _new_state() -> LearnerState:
    return LearnerState(
        q=np.zeros((4, 2, 8, 2), dtype=np.float64),
        pending=None,
        expected_phase=0,
        completed_episodes=0,
        bootstrap_updates=0,
        terminal_updates=0,
        total_updates=0,
        assigned_cell_checks=0,
    )


def _pending_projection(pending: PendingSelection | None) -> object:
    return None if pending is None else asdict(pending)


def _state_commitment(state: LearnerState) -> str:
    return _json_sha256(
        {
            "assigned_cell_checks": state.assigned_cell_checks,
            "bootstrap_updates": state.bootstrap_updates,
            "completed_episodes": state.completed_episodes,
            "expected_phase": state.expected_phase,
            "pending": _pending_projection(state.pending),
            "q": _array_identity(state.q),
            "terminal_updates": state.terminal_updates,
            "total_updates": state.total_updates,
        }
    )


def _behavior_action(completed_episodes: int) -> int:
    return BEHAVIOR_ACTIONS[completed_episodes % 16]


def _select(
    state: LearnerState,
    record: object,
    *,
    forced_action: int | None = None,
) -> np.ndarray:
    observation = _policy_observation(record)
    phase = int(observation[0])
    if state.pending is not None or phase != state.expected_phase:
        raise RuntimeError("selection is out of phase")
    if forced_action is None:
        values = state.q[_state_coords(observation)]
        action_value = int(np.argmax(values))
    else:
        if type(forced_action) is not int or forced_action not in (0, 1):
            raise TypeError("forced action is invalid")
        action_value = forced_action
    action = _immutable_action(action_value)
    _validate_action(action)
    pending = PendingSelection(
        observation_sha256=_observation_sha256(observation),
        coords=_state_coords(observation),
        phase=phase,
        prefix=int(observation[2]),
        action=action_value,
    )
    state.pending = pending
    return action


def _update(
    state: LearnerState,
    feedback: object,
    *,
    discount: float = 1.0,
) -> UpdateAudit:
    if state.pending is None:
        raise RuntimeError("update requires a pending selection")
    if not isinstance(feedback, dict) or set(feedback) != {
        "done",
        "next_observation",
        "reward",
    }:
        raise RuntimeError("update feedback fields are invalid")
    reward = _validate_reward(feedback["reward"])
    done = feedback["done"]
    if type(done) is not bool:
        raise TypeError("done must be a literal Python Boolean")
    pending = state.pending
    next_observation = feedback["next_observation"]
    if pending.phase == HORIZON - 1:
        if not done or next_observation is not None:
            raise RuntimeError("terminal update contract is invalid")
        successor_value = None
        target_value = float(reward)
    else:
        if done or float(reward) != 0.0 or next_observation is None:
            raise RuntimeError("nonterminal update contract is invalid")
        successor = _validate_observation(
            next_observation, allow_zero_signal=True
        )
        expected_prefix = 2 * pending.prefix + pending.action
        if (
            int(successor[0]) != pending.phase + 1
            or int(successor[2]) != expected_prefix
        ):
            raise RuntimeError("bootstrap successor is not action-consistent")
        successor_value = float(np.max(state.q[_state_coords(successor)]))
        target_value = float(reward) + discount * successor_value

    before = state.q.copy()
    index = (*pending.coords, pending.action)
    state.q[index] = target_value
    changed_cells = int(np.count_nonzero(before != state.q))
    mask = np.ones(state.q.shape, dtype=bool)
    mask[index] = False
    unchosen_unchanged = bool(np.array_equal(before[mask], state.q[mask]))
    if changed_cells > 1 or not unchosen_unchanged:
        raise RuntimeError("TD update changed an unchosen cell")

    state.pending = None
    state.total_updates += 1
    state.assigned_cell_checks += 1
    if done:
        state.terminal_updates += 1
        state.completed_episodes += 1
        state.expected_phase = 0
    else:
        state.bootstrap_updates += 1
        state.expected_phase = pending.phase + 1
    return UpdateAudit(
        phase=pending.phase,
        target_value=target_value,
        successor_value=successor_value,
        changed_cells=changed_cells,
        unchosen_cells_unchanged=unchosen_unchanged,
    )


class EpisodeProtocol:
    """Small fail-closed event-order state machine used by every episode."""

    def __init__(self) -> None:
        self.phase = 0
        self.stage = "observe"
        self.action: int | None = None
        self.events: list[str] = []

    def commitment(self) -> str:
        return _json_sha256(
            {
                "action": self.action,
                "events": self.events,
                "phase": self.phase,
                "stage": self.stage,
            }
        )

    def _advance(self, expected: str, next_stage: str, event: str) -> None:
        if self.stage != expected:
            raise RuntimeError(f"protocol expected {expected}, not {self.stage}")
        self.events.append(event)
        self.stage = next_stage

    def observe(self, phase: int) -> None:
        if phase != self.phase:
            raise RuntimeError("protocol phase is noncontiguous")
        self._advance("observe", "select", f"observe{phase}")

    def select(self) -> None:
        self._advance("select", "validate", f"select{self.phase}")

    def validate(self, action: np.ndarray) -> None:
        validated = _validate_action(action)
        if self.stage != "validate":
            raise RuntimeError("action validation is out of order")
        self.action = int(validated)
        self.events.append(f"validate{self.phase}")
        self.stage = "transition"

    def transition(
        self, *, action: int, successor_present: bool
    ) -> None:
        if self.stage != "transition" or action != self.action:
            raise RuntimeError("transition is out of order or action-inconsistent")
        expected_successor = self.phase < HORIZON - 1
        if type(successor_present) is not bool or (
            successor_present != expected_successor
        ):
            raise RuntimeError("transition successor presence is invalid")
        self.events.append(f"transition{self.phase}")
        self.stage = "feedback"

    def feedback(self, *, reward: float, done: bool) -> None:
        if self.stage != "feedback" or type(done) is not bool:
            raise RuntimeError("feedback is out of order or malformed")
        terminal = self.phase == HORIZON - 1
        if done != terminal or (not terminal and reward != 0.0):
            raise RuntimeError("feedback timing or value is invalid")
        if reward not in (0.0, 1.0):
            raise RuntimeError("feedback reward is invalid")
        self.events.append(f"feedback{self.phase}")
        self.stage = "target"

    def target(self) -> None:
        self._advance("target", "update", f"target{self.phase}")

    def update(self) -> None:
        self._advance("update", "log", f"update{self.phase}")

    def log(self) -> None:
        if self.stage != "log":
            raise RuntimeError("logging is out of order")
        self.events.append(f"log{self.phase}")
        self.action = None
        if self.phase == HORIZON - 1:
            self.stage = "close"
        else:
            self.phase += 1
            self.stage = "observe"

    def close(self) -> None:
        self._advance("close", "closed", "close_episode")


def _trace_sha256(trace: TraceBundle) -> str:
    return _json_sha256(
        {
            "actions": [asdict(row) for row in trace.actions],
            "feedback": [asdict(row) for row in trace.feedback],
            "observations": [asdict(row) for row in trace.observations],
            "transitions": [asdict(row) for row in trace.transitions],
        }
    )


def _action_commitment(trace: TraceBundle) -> str:
    return _json_sha256([asdict(row) for row in trace.actions])


def _actual_transition_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            {
                "action": row.action,
                "done": row.done,
                "episode_key": row.episode_key,
                "predecessor_sha256": row.predecessor_sha256,
                "step_key": row.step_key,
                "successor_key": row.successor_key,
                "successor_sha256": row.successor_sha256,
            }
            for row in trace.transitions
        ]
    )


def _canonical_feedback_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            {
                "done": row.done,
                "episode_key": row.episode_key,
                "reward": row.reward,
                "step_key": row.step_key,
            }
            for row in trace.feedback
        ]
    )


def _transition_donor_episode(episode: int) -> int:
    block = 16 * (episode // 16)
    return block + TRANSITION_DONOR_PERMUTATION[episode % 16]


def _reward_origin_episode(
    episode: int, *, poisoned_reward_oracle: Callable[[], object] | None = None
) -> int:
    del poisoned_reward_oracle
    block = 16 * (episode // 16)
    return block + REWARD_ORIGIN_PERMUTATION[episode % 16]


def _assert_boundary_clean(
    state: LearnerState, *, delayed_scalars: tuple[float, ...] = ()
) -> None:
    if (
        state.pending is not None
        or state.expected_phase != 0
        or delayed_scalars
    ):
        raise RuntimeError("learner has pending state at a split boundary")


@dataclass(frozen=True)
class TrainRun:
    state: LearnerState
    trace: TraceBundle
    event_orders: tuple[tuple[str, ...], ...]
    first_positive_offsets: tuple[tuple[int, int, int], ...]


class ExplodingSource:
    """Operation-counting source that must remain outside the train API."""

    OPERATIONS = (
        "attribute",
        "iteration",
        "next",
        "length",
        "indexing",
        "truth",
        "array",
    )

    def __init__(self) -> None:
        object.__setattr__(self, "counts", {name: 0 for name in self.OPERATIONS})

    def _explode(self, name: str) -> None:
        self.counts[name] += 1
        raise RuntimeError(f"held-out source operation reached: {name}")

    def __getattr__(self, _name: str) -> object:
        self._explode("attribute")

    def __iter__(self) -> object:
        self._explode("iteration")

    def __next__(self) -> object:
        self._explode("next")

    def __len__(self) -> int:
        self._explode("length")

    def __getitem__(self, _key: object) -> object:
        self._explode("indexing")

    def __bool__(self) -> bool:
        self._explode("truth")

    def __array__(self, _dtype: object = None) -> np.ndarray:
        self._explode("array")


def _copy_state(state: LearnerState) -> LearnerState:
    return LearnerState(
        q=state.q.copy(),
        pending=state.pending,
        expected_phase=state.expected_phase,
        completed_episodes=state.completed_episodes,
        bootstrap_updates=state.bootstrap_updates,
        terminal_updates=state.terminal_updates,
        total_updates=state.total_updates,
        assigned_cell_checks=state.assigned_cell_checks,
    )


def _terminal_origin_reward(episode: Episode, origin_index: int) -> float:
    """Materialize one origin outcome at the terminal evaluator boundary."""
    origin_target = _evaluator_target(
        episode.regime.split, episode.regime.code, origin_index
    )
    origin_action = BEHAVIOR_ACTIONS[origin_index % 16]
    prefix = 0 if origin_action == 0 else 7
    return _terminal_reward(origin_target, prefix, origin_action)


def _run_episode(
    state: LearnerState,
    episode: Episode,
    *,
    control_mode: str,
    forced_actions: tuple[int, ...] | None,
    update: bool,
    ablate_signal: bool,
    discount: float,
) -> tuple[
    tuple[ObservationRecord, ...],
    tuple[ActionRecord, ...],
    tuple[TransitionRecord, ...],
    tuple[FeedbackRecord, ...],
    tuple[str, ...],
    tuple[tuple[int, int, int], ...],
]:
    if control_mode not in {"canonical", "transition_target", "reward_origin"}:
        raise ValueError("unknown control mode")
    if forced_actions is not None and len(forced_actions) != HORIZON:
        raise ValueError("forced action sequence must have four actions")
    protocol = EpisodeProtocol()
    observations: list[ObservationRecord] = []
    actions: list[ActionRecord] = []
    transitions: list[TransitionRecord] = []
    feedback_rows: list[FeedbackRecord] = []
    positives: list[tuple[int, int, int]] = []
    prefix = 0

    for phase in range(HORIZON):
        observation = _public_observation(
            episode.regime,
            episode.index,
            phase,
            prefix,
            ablate_signal=ablate_signal,
        )
        predecessor_sha256 = _observation_sha256(observation)
        step_key = _step_key(episode.episode_key, phase, prefix)
        protocol.observe(phase)
        before_select = _state_commitment(state)
        forced_action = None if forced_actions is None else forced_actions[phase]
        if update:
            action_array = _select(
                state,
                {"observation": observation},
                forced_action=forced_action,
            )
        else:
            if state.pending is not None:
                raise RuntimeError("frozen evaluation inherited a pending update")
            values = state.q[_state_coords(observation)]
            selected = int(np.argmax(values)) if forced_action is None else forced_action
            action_array = _immutable_action(selected)
            _validate_action(action_array)
        protocol.select()
        protocol.validate(action_array)
        action = int(action_array)
        done = phase == HORIZON - 1

        if done:
            actual_successor = None
            actual_successor_sha256 = None
            successor_key = None
            bootstrap_successor = None
            bootstrap_successor_sha256 = None
            donor_episode = None
        else:
            next_prefix = 2 * prefix + action
            actual_successor = _public_observation(
                episode.regime,
                episode.index,
                phase + 1,
                next_prefix,
                ablate_signal=ablate_signal,
            )
            actual_successor_sha256 = _observation_sha256(actual_successor)
            successor_key = _step_key(
                episode.episode_key, phase + 1, next_prefix
            )
            if control_mode == "transition_target":
                donor_episode = _transition_donor_episode(episode.index)
                bootstrap_successor = _public_observation(
                    episode.regime,
                    donor_episode,
                    phase + 1,
                    next_prefix,
                    ablate_signal=ablate_signal,
                )
            else:
                donor_episode = None
                bootstrap_successor = actual_successor
            bootstrap_successor_sha256 = _observation_sha256(
                bootstrap_successor
            )
        protocol.transition(
            action=action, successor_present=actual_successor is not None
        )

        canonical_reward = (
            _terminal_reward(episode.target, prefix, action) if done else 0.0
        )
        if done and control_mode == "reward_origin":
            origin_episode = _reward_origin_episode(episode.index)
            update_reward = _terminal_origin_reward(episode, origin_episode)
        else:
            origin_episode = None
            update_reward = canonical_reward
        reward_array = _immutable_reward(update_reward)
        protocol.feedback(reward=update_reward, done=done)
        protocol.target()

        if update:
            audit = _update(
                state,
                {
                    "done": done,
                    "next_observation": bootstrap_successor,
                    "reward": reward_array,
                },
                discount=discount,
            )
            if audit.changed_cells == 1 and audit.target_value > 0.0:
                positives.append((int(observation[1] > 0.0), phase, state.completed_episodes))
        else:
            if _state_commitment(state) != before_select:
                raise RuntimeError("frozen evaluation changed learner state")
        protocol.update()

        observations.append(
            ObservationRecord(
                step_key=step_key,
                episode_key=episode.episode_key,
                phase=phase,
                predecessor_sha256=predecessor_sha256,
                observation_sha256=predecessor_sha256,
                action=action,
                action_dtype=str(action_array.dtype),
                done=done,
            )
        )
        actions.append(
            ActionRecord(
                step_key=step_key,
                episode_key=episode.episode_key,
                phase=phase,
                predecessor_sha256=predecessor_sha256,
                action=action,
                action_dtype=str(action_array.dtype),
                done=done,
            )
        )
        transitions.append(
            TransitionRecord(
                step_key=step_key,
                episode_key=episode.episode_key,
                phase=phase,
                predecessor_sha256=predecessor_sha256,
                action=action,
                action_dtype=str(action_array.dtype),
                done=done,
                successor_key=successor_key,
                successor_sha256=actual_successor_sha256,
                bootstrap_successor_sha256=bootstrap_successor_sha256,
                donor_episode=donor_episode,
            )
        )
        feedback_rows.append(
            FeedbackRecord(
                step_key=step_key,
                episode_key=episode.episode_key,
                phase=phase,
                predecessor_sha256=predecessor_sha256,
                action=action,
                action_dtype=str(action_array.dtype),
                done=done,
                reward=canonical_reward,
                update_reward=update_reward,
                origin_episode=origin_episode,
            )
        )
        protocol.log()
        if not done:
            prefix = 2 * prefix + action

    protocol.close()
    if tuple(protocol.events) != EVENT_ORDER:
        raise RuntimeError("episode event order drifted")
    if update and state.pending is not None:
        raise RuntimeError("episode closed with a pending update")
    return (
        tuple(observations),
        tuple(actions),
        tuple(transitions),
        tuple(feedback_rows),
        tuple(protocol.events),
        tuple(positives),
    )


def _train_policy(
    train_source: tuple[Episode, ...],
    *,
    control_mode: str = "canonical",
    ablate_signal: bool = False,
    discount: float = 1.0,
) -> TrainRun:
    if any(row.regime.split != "train" for row in train_source):
        raise RuntimeError("training source contains held-out data")
    state = _new_state()
    observations: list[ObservationRecord] = []
    actions: list[ActionRecord] = []
    transitions: list[TransitionRecord] = []
    feedback: list[FeedbackRecord] = []
    event_orders: list[tuple[str, ...]] = []
    first_positive: dict[tuple[int, int], int] = {}
    for ordinal, episode in enumerate(train_source):
        if ordinal != state.completed_episodes:
            raise RuntimeError("training episode order drifted")
        behavior = _behavior_action(state.completed_episodes)
        rows = _run_episode(
            state,
            episode,
            control_mode=control_mode,
            forced_actions=(behavior,) * HORIZON,
            update=True,
            ablate_signal=ablate_signal,
            discount=discount,
        )
        observations.extend(rows[0])
        actions.extend(rows[1])
        transitions.extend(rows[2])
        feedback.extend(rows[3])
        event_orders.append(rows[4])
        regime_offset = episode.index
        for sign_bin, phase, _completed in rows[5]:
            first_positive.setdefault((sign_bin, phase), regime_offset)
        if episode.index == EPISODES_PER_REGIME - 1:
            _assert_boundary_clean(state)
    expected_offsets = tuple(
        (sign_bin, phase, first_positive.get((sign_bin, phase), -1))
        for sign_bin in (0, 1)
        for phase in (3, 2, 1, 0)
    )
    return TrainRun(
        state=state,
        trace=TraceBundle(
            observations=tuple(observations),
            actions=tuple(actions),
            transitions=tuple(transitions),
            feedback=tuple(feedback),
        ),
        event_orders=tuple(event_orders),
        first_positive_offsets=expected_offsets,
    )


def _evaluate_policy(
    state: LearnerState,
    split: str,
    *,
    ablate_signal: bool = False,
    forced_policy: Callable[[Episode, int, np.ndarray], int] | None = None,
) -> tuple[float, float, dict[int, float], str]:
    before = _state_commitment(state)
    totals: dict[int, list[float]] = {}
    trace_rows: list[dict[str, object]] = []
    for episode in _episodes(split):
        prefix = 0
        for phase in range(HORIZON):
            observation = _public_observation(
                episode.regime,
                episode.index,
                phase,
                prefix,
                ablate_signal=ablate_signal,
            )
            if forced_policy is None:
                action = int(np.argmax(state.q[_state_coords(observation)]))
            else:
                action = forced_policy(episode, phase, observation)
            if type(action) is not int or action not in (0, 1):
                raise RuntimeError("evaluation policy returned an invalid action")
            if phase == HORIZON - 1:
                reward = _terminal_reward(episode.target, prefix, action)
                totals.setdefault(episode.regime.code, []).append(reward)
            else:
                reward = 0.0
            trace_rows.append(
                {
                    "action": action,
                    "done": phase == HORIZON - 1,
                    "observation_sha256": _observation_sha256(observation),
                    "phase": phase,
                    "reward": reward,
                    "step_key": _step_key(episode.episode_key, phase, prefix),
                }
            )
            prefix = 2 * prefix + action
    if _state_commitment(state) != before or state.pending is not None:
        raise RuntimeError("held-out evaluation changed learner state")
    per_regime = {
        code: float(np.mean(values)) for code, values in sorted(totals.items())
    }
    macro = float(np.mean(tuple(per_regime.values())))
    minimum = float(min(per_regime.values()))
    return macro, minimum, per_regime, _json_sha256(trace_rows)


def _behavior_metrics(split: str) -> tuple[float, float, dict[int, float]]:
    totals: dict[int, list[float]] = {}
    for episode in _episodes(split):
        action = BEHAVIOR_ACTIONS[episode.index % 16]
        prefix = 0 if action == 0 else 7
        reward = _terminal_reward(episode.target, prefix, action)
        totals.setdefault(episode.regime.code, []).append(reward)
    per_regime = {
        code: float(np.mean(values)) for code, values in sorted(totals.items())
    }
    macro = float(np.mean(tuple(per_regime.values())))
    regret = float(sum(len(values) - sum(values) for values in totals.values()))
    return macro, regret, per_regime


def _component_map(rows: tuple[object, ...], expected_type: type) -> dict[str, object]:
    result: dict[str, object] = {}
    for row in rows:
        if type(row) is not expected_type:
            raise RuntimeError("trace component has the wrong record type")
        key = row.step_key
        if type(key) is not str or key in result:
            raise RuntimeError("trace component has a duplicate or malformed key")
        result[key] = row
    return result


def _authenticate_common(
    row: object,
    *,
    expected_type: type,
    step_key: str,
    episode_key: str,
    phase: int,
    predecessor_sha256: str,
    action: int,
    done: bool,
) -> None:
    if type(row) is not expected_type:
        raise RuntimeError("component type drifted")
    expected = {
        "step_key": step_key,
        "episode_key": episode_key,
        "phase": phase,
        "predecessor_sha256": predecessor_sha256,
        "action": action,
        "action_dtype": "int8",
        "done": done,
    }
    for field, value in expected.items():
        actual = getattr(row, field)
        if field in {"phase", "action"} and (
            type(actual) is not int or type(actual) is bool
        ):
            raise RuntimeError("integer component field has the wrong type")
        if field == "done" and type(actual) is not bool:
            raise RuntimeError("done component field is not a literal Boolean")
        if actual != value:
            raise RuntimeError(f"component authentication failed: {field}")


def _score_trace(
    trace: TraceBundle,
    split: str,
    *,
    control_mode: str,
    ablate_signal: bool = False,
) -> tuple[float, str]:
    if type(trace) is not TraceBundle:
        raise RuntimeError("trace bundle type is invalid")
    if control_mode not in {"canonical", "transition_target", "reward_origin"}:
        raise RuntimeError("trace control mode is invalid")
    observation_map = _component_map(trace.observations, ObservationRecord)
    action_map = _component_map(trace.actions, ActionRecord)
    transition_map = _component_map(trace.transitions, TransitionRecord)
    feedback_map = _component_map(trace.feedback, FeedbackRecord)
    expected_count = len(_episodes(split)) * HORIZON
    if any(
        len(component) != expected_count
        for component in (
            observation_map,
            action_map,
            transition_map,
            feedback_map,
        )
    ):
        raise RuntimeError("trace component cardinality is invalid")

    visited: set[str] = set()
    terminal_rewards: dict[int, list[float]] = {}
    authenticated: list[dict[str, object]] = []
    for episode in _episodes(split):
        prefix = 0
        for phase in range(HORIZON):
            step_key = _step_key(episode.episode_key, phase, prefix)
            try:
                observation_row = observation_map[step_key]
                action_row = action_map[step_key]
                transition_row = transition_map[step_key]
                feedback_row = feedback_map[step_key]
            except KeyError as exc:
                raise RuntimeError("trace is missing an expected key") from exc
            visited.add(step_key)
            observation = _public_observation(
                episode.regime,
                episode.index,
                phase,
                prefix,
                ablate_signal=ablate_signal,
            )
            predecessor_sha256 = _observation_sha256(observation)
            if observation_row.observation_sha256 != predecessor_sha256:
                raise RuntimeError("observation record bytes are unauthenticated")
            action = action_row.action
            if type(action) is not int or type(action) is bool or action not in (0, 1):
                raise RuntimeError("action record value is malformed")
            done = phase == HORIZON - 1
            for row, row_type in (
                (observation_row, ObservationRecord),
                (action_row, ActionRecord),
                (transition_row, TransitionRecord),
                (feedback_row, FeedbackRecord),
            ):
                _authenticate_common(
                    row,
                    expected_type=row_type,
                    step_key=step_key,
                    episode_key=episode.episode_key,
                    phase=phase,
                    predecessor_sha256=predecessor_sha256,
                    action=action,
                    done=done,
                )

            if done:
                expected_successor_key = None
                expected_successor_sha256 = None
                expected_bootstrap_sha256 = None
                expected_donor = None
            else:
                next_prefix = 2 * prefix + action
                successor = _public_observation(
                    episode.regime,
                    episode.index,
                    phase + 1,
                    next_prefix,
                    ablate_signal=ablate_signal,
                )
                expected_successor_key = _step_key(
                    episode.episode_key, phase + 1, next_prefix
                )
                expected_successor_sha256 = _observation_sha256(successor)
                if control_mode == "transition_target":
                    expected_donor = _transition_donor_episode(episode.index)
                    donor_successor = _public_observation(
                        episode.regime,
                        expected_donor,
                        phase + 1,
                        next_prefix,
                        ablate_signal=ablate_signal,
                    )
                    expected_bootstrap_sha256 = _observation_sha256(
                        donor_successor
                    )
                else:
                    expected_donor = None
                    expected_bootstrap_sha256 = expected_successor_sha256
            transition_expected = (
                expected_successor_key,
                expected_successor_sha256,
                expected_bootstrap_sha256,
                expected_donor,
            )
            transition_actual = (
                transition_row.successor_key,
                transition_row.successor_sha256,
                transition_row.bootstrap_successor_sha256,
                transition_row.donor_episode,
            )
            if transition_actual != transition_expected:
                raise RuntimeError("transition component is unauthenticated")

            canonical_reward = (
                _terminal_reward(episode.target, prefix, action) if done else 0.0
            )
            if control_mode == "reward_origin" and done:
                expected_origin = _reward_origin_episode(episode.index)
                expected_update_reward = _terminal_origin_reward(
                    episode, expected_origin
                )
            else:
                expected_origin = None
                expected_update_reward = canonical_reward
            for scalar in (feedback_row.reward, feedback_row.update_reward):
                if type(scalar) is not float or not np.isfinite(scalar):
                    raise RuntimeError("feedback scalar is malformed")
            feedback_expected = (
                canonical_reward,
                expected_update_reward,
                expected_origin,
            )
            feedback_actual = (
                feedback_row.reward,
                feedback_row.update_reward,
                feedback_row.origin_episode,
            )
            if feedback_actual != feedback_expected:
                raise RuntimeError("feedback component is unauthenticated")
            if done:
                terminal_rewards.setdefault(episode.regime.code, []).append(
                    canonical_reward
                )
            authenticated.append(
                {
                    "action": action,
                    "done": done,
                    "feedback": feedback_actual,
                    "step_key": step_key,
                    "transition": transition_actual,
                }
            )
            prefix = 2 * prefix + action
    expected_keys = set(observation_map)
    if any(set(component) != expected_keys for component in (
        action_map,
        transition_map,
        feedback_map,
    )) or visited != expected_keys:
        raise RuntimeError("trace contains unknown or cross-episode keys")
    macro = float(
        np.mean(
            [float(np.mean(values)) for _, values in sorted(terminal_rewards.items())]
        )
    )
    return macro, _json_sha256(authenticated)


def _reordered_trace(trace: TraceBundle, variant: str) -> TraceBundle:
    if variant not in {
        "canonical",
        "observations_reversed",
        "actions_rotate_left_one",
        "transitions_even_then_odd",
        "feedback_odd_then_even",
        "all_independent",
    }:
        raise ValueError("unknown trace reorder variant")
    observations = trace.observations
    actions = trace.actions
    transitions = trace.transitions
    feedback = trace.feedback
    if variant in {"observations_reversed", "all_independent"}:
        observations = tuple(reversed(observations))
    if variant in {"actions_rotate_left_one", "all_independent"}:
        actions = actions[1:] + actions[:1]
    if variant in {"transitions_even_then_odd", "all_independent"}:
        transitions = transitions[::2] + transitions[1::2]
    if variant in {"feedback_odd_then_even", "all_independent"}:
        feedback = feedback[1::2] + feedback[::2]
    return TraceBundle(observations, actions, transitions, feedback)


def _fit_myopic_state() -> LearnerState:
    sums = np.zeros((2, 8, 2), dtype=np.float64)
    counts = np.zeros((2, 8, 2), dtype=np.int64)
    for episode in _episodes("train"):
        action = BEHAVIOR_ACTIONS[episode.index % 16]
        prefix = 0 if action == 0 else 7
        sign_bin = _public_sign_bit(
            episode.regime.split, episode.regime.code, episode.index
        )
        reward = _terminal_reward(episode.target, prefix, action)
        sums[sign_bin, prefix, action] += reward
        counts[sign_bin, prefix, action] += 1
    state = _new_state()
    visited = counts > 0
    state.q[3][visited] = sums[visited] / counts[visited]
    return state


def _random_policy(split_tag: int) -> Callable[[Episode, int, np.ndarray], int]:
    generator = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([RANDOM_BASELINE_SEED, split_tag])
        )
    )

    def choose(_episode: Episode, _phase: int, _observation: np.ndarray) -> int:
        return int(generator.integers(0, 2, dtype=np.int8))

    return choose


def _baseline_metrics() -> dict[str, object]:
    blank = _new_state()
    results: dict[str, dict[str, float]] = {}
    trace_parts: list[object] = []
    for name, policy in (
        ("constant_zero", lambda _e, _p, _o: 0),
        ("constant_one", lambda _e, _p, _o: 1),
    ):
        results[name] = {}
        for split in ("validation", "test"):
            macro, _minimum, _regimes, trace = _evaluate_policy(
                blank, split, forced_policy=policy
            )
            results[name][split] = macro
            trace_parts.append((name, split, macro, trace))

    myopic = _fit_myopic_state()
    no_bootstrap = _train_policy(_episodes("train"), discount=0.0).state
    for name, state in (("myopic", myopic), ("no_bootstrap", no_bootstrap)):
        results[name] = {}
        for split in ("validation", "test"):
            macro, _minimum, _regimes, trace = _evaluate_policy(state, split)
            results[name][split] = macro
            trace_parts.append((name, split, macro, trace))

    random_first: dict[str, float] = {}
    random_second: dict[str, float] = {}
    for split, tag in (("validation", 1), ("test", 2)):
        first = _evaluate_policy(
            blank, split, forced_policy=_random_policy(tag)
        )
        second = _evaluate_policy(
            blank, split, forced_policy=_random_policy(tag)
        )
        random_first[split] = first[0]
        random_second[split] = second[0]
        trace_parts.append(("random", split, first[0], first[3], second[3]))
    results["random"] = random_first
    best_validation = max(values["validation"] for values in results.values())
    best_test = max(values["test"] for values in results.values())
    return {
        "best_test": float(best_test),
        "best_validation": float(best_validation),
        "constant_exact": bool(
            results["constant_zero"] == {"validation": 0.5, "test": 0.5}
            and results["constant_one"] == {"validation": 0.5, "test": 0.5}
        ),
        "myopic_exact": bool(
            results["myopic"] == {"validation": 0.5, "test": 0.5}
        ),
        "no_bootstrap_exact": bool(
            results["no_bootstrap"] == {"validation": 0.5, "test": 0.5}
        ),
        "random_replay_exact": random_first == random_second,
        "trace_sha256": _json_sha256(trace_parts),
    }


def _family_and_contract_cases() -> dict[str, dict[str, object]]:
    family = _family_projection()
    independent = _independent_family_projection()
    dataset_sha256 = _json_sha256(list(family))
    nodes = {
        (row["episode_key"], row["phase"], row["prefix"])
        for row in family
    }
    nonterminal = [row for row in family if not row["done"]]
    terminal = [row for row in family if row["done"]]

    typed_ok = all(
        type(row["action"]) is int
        and row["action"] in (0, 1)
        and type(row["done"]) is bool
        and row["observation_contract"]
        == {
            "c_contiguous": True,
            "dtype": "float64",
            "immutable": True,
            "shape": [4],
        }
        for row in family
    )
    typed_case = {
        "actions_checked": len(family),
        "immutable_observations_checked": len(nodes),
        "legal_rows_checked": len(family),
        "passed": bool(typed_ok and len(family) == 7680 and len(nodes) == 3840),
        "trace_sha256": _json_sha256(
            {
                "actions": len(family),
                "nodes": len(nodes),
                "typed": typed_ok,
            }
        ),
    }

    target_swap_outcomes_changed = 0
    target_swap_twins_checked = 0
    successor_bytes_preserved = True
    for episode in _episodes("train") + _episodes("validation") + _episodes("test"):
        target_swap_twins_checked += 1
        for phase in range(HORIZON):
            for prefix in range(2**phase):
                canonical_observation = _public_observation(
                    episode.regime, episode.index, phase, prefix
                )
                frozen_bytes = canonical_observation.tobytes(order="C")
                for action in (0, 1):
                    if phase < HORIZON - 1:
                        successor = _public_observation(
                            episode.regime,
                            episode.index,
                            phase + 1,
                            2 * prefix + action,
                        )
                        successor_bytes_preserved = bool(
                            successor_bytes_preserved
                            and frozen_bytes == canonical_observation.tobytes(order="C")
                            and _observation_sha256(successor)
                            == _observation_sha256(successor.copy())
                        )
                    else:
                        canonical = _terminal_reward(
                            episode.target, prefix, action
                        )
                        twin = _terminal_reward(
                            1 - episode.target, prefix, action
                        )
                        target_swap_outcomes_changed += int(canonical != twin)
    successor_case = {
        "passed": bool(
            successor_bytes_preserved
            and len(nonterminal) == 3584
            and target_swap_twins_checked == 256
            and target_swap_outcomes_changed == 512
        ),
        "successors_checked": len(nonterminal),
        "target_swap_outcomes_changed": target_swap_outcomes_changed,
        "target_swap_twins_checked": target_swap_twins_checked,
        "trace_sha256": _json_sha256(
            {
                "outcomes_changed": target_swap_outcomes_changed,
                "successors": [row["successor_sha256"] for row in nonterminal],
                "twins": target_swap_twins_checked,
            }
        ),
    }

    split_key_sets = {
        split: {
            row["row_key"] for row in family if row["split"] == split
        }
        for split in ("train", "validation", "test")
    }
    split_keys_disjoint = bool(
        split_key_sets["train"].isdisjoint(split_key_sets["validation"])
        and split_key_sets["train"].isdisjoint(split_key_sets["test"])
        and split_key_sets["validation"].isdisjoint(split_key_sets["test"])
    )
    public_sets: dict[int, set[str]] = {}
    for row in family:
        public_sets.setdefault(int(row["regime_code"]), set()).add(
            str(row["predecessor_sha256"])
        )
    public_rows_disjoint = bool(
        all(len(values) == 480 for values in public_sets.values())
        and all(
            public_sets[left].isdisjoint(public_sets[right])
            for index, left in enumerate(sorted(public_sets))
            for right in sorted(public_sets)[index + 1 :]
        )
    )
    target_balance_exact = all(
        sum(episode.target for episode in _episodes(regime.split) if episode.regime.code == regime.code)
        == 16
        for regime in REGIMES
    )
    family_case = {
        "dataset_sha256": dataset_sha256,
        "episodes_checked": 256,
        "legal_rows_checked": len(family),
        "nodes_checked": len(nodes),
        "nonterminal_rows_checked": len(nonterminal),
        "passed": bool(
            dataset_sha256 == EXPECTED_DATASET_SHA256
            and family == independent
            and len(family) == 7680
            and len(nodes) == 3840
            and len(nonterminal) == 3584
            and len(terminal) == 4096
            and public_rows_disjoint
            and split_keys_disjoint
            and target_balance_exact
        ),
        "public_rows_disjoint": public_rows_disjoint,
        "split_keys_disjoint": split_keys_disjoint,
        "target_balance_exact": target_balance_exact,
        "terminal_rows_checked": len(terminal),
    }

    mutations: list[str] = []
    base = [dict(row) for row in family]
    predecessor_mutation = [dict(row) for row in base]
    predecessor_mutation[0]["predecessor_sha256"] = "f" * 64
    mutations.append(_json_sha256(predecessor_mutation))
    successor_index = next(
        index for index, row in enumerate(base) if not row["done"]
    )
    successor_mutation = [dict(row) for row in base]
    successor_mutation[successor_index]["successor_sha256"] = "e" * 64
    mutations.append(_json_sha256(successor_mutation))
    terminal_index = next(index for index, row in enumerate(base) if row["done"])
    reward_mutation = [dict(row) for row in base]
    reward_mutation[terminal_index]["reward"] = 1.0 - float(
        reward_mutation[terminal_index]["reward"]
    )
    mutations.append(_json_sha256(reward_mutation))
    done_mutation = [dict(row) for row in base]
    done_mutation[terminal_index]["done"] = False
    mutations.append(_json_sha256(done_mutation))
    mutation_rejections = sum(value != dataset_sha256 for value in mutations)
    replay_sha256 = _json_sha256(list(independent))
    replay_case = {
        "mutation_sentinels_checked": 4,
        "mutation_sentinels_rejected": mutation_rejections,
        "passed": bool(
            family == independent
            and replay_sha256 == EXPECTED_DATASET_SHA256
            and mutation_rejections == 4
        ),
        "replay_sha256": replay_sha256,
        "trace_sha256": _json_sha256(
            {"mutations": mutations, "replay": replay_sha256}
        ),
    }
    return {
        "typed_action_prefix_contract": typed_case,
        "target_independent_public_successors": successor_case,
        "complete_legal_family_commitment": family_case,
        "independent_family_replay": replay_case,
    }


def _physical_boundary_case() -> dict[str, object]:
    selector_inputs_checked = 0
    for regime in REGIMES:
        for episode in range(EPISODES_PER_REGIME):
            for phase in range(HORIZON):
                for prefix in range(2**phase):
                    observation = _public_observation(
                        regime, episode, phase, prefix
                    )
                    if _policy_observation({"observation": observation}) is not observation:
                        raise RuntimeError("policy observation was copied or replaced")
                    selector_inputs_checked += 1

    forbidden_rejected = 0
    sample = _public_observation(REGIMES[0], 0, 0, 0)
    for field in FORBIDDEN_POLICY_FIELDS:
        state = _new_state()
        before = _state_commitment(state)
        try:
            _select(state, {"observation": sample, field: object()})
        except (TypeError, ValueError, RuntimeError):
            if _state_commitment(state) == before:
                forbidden_rejected += 1

    successor_sentinel = ExplodingSource()
    reward_sentinel = ExplodingSource()
    invalid_rejected = 0
    for sentinel in (successor_sentinel, reward_sentinel):
        try:
            invalid = True
            _validate_action(invalid)
            bool(sentinel)
        except (TypeError, ValueError):
            invalid_rejected += 1
    operations = sum(successor_sentinel.counts.values()) + sum(
        reward_sentinel.counts.values()
    )
    passed = bool(
        selector_inputs_checked == 3840
        and forbidden_rejected == len(FORBIDDEN_POLICY_FIELDS)
        and invalid_rejected == 2
        and operations == 0
    )
    return {
        "exploding_sentinels_checked": 2,
        "exploding_sentinels_rejected": invalid_rejected,
        "forbidden_fields_checked": len(FORBIDDEN_POLICY_FIELDS),
        "passed": passed,
        "selector_inputs_checked": selector_inputs_checked,
        "trace_sha256": _json_sha256(
            {
                "forbidden_rejected": forbidden_rejected,
                "operations": operations,
                "selector_inputs": selector_inputs_checked,
            }
        ),
    }


def _td_dependency_case() -> dict[str, object]:
    predecessor = _public_observation(REGIMES[0], 0, 0, 0)
    successor = _public_observation(REGIMES[0], 0, 1, 0)
    targets: list[float] = []
    for successor_value in (0.25, 0.75):
        state = _new_state()
        state.q[(*_state_coords(successor), 0)] = successor_value
        _select(state, {"observation": predecessor}, forced_action=0)
        audit = _update(
            state,
            {
                "done": False,
                "next_observation": successor,
                "reward": _immutable_reward(0.0),
            },
        )
        targets.append(audit.target_value)
    successor_dependency = targets == [0.25, 0.75]

    terminal = _public_observation(REGIMES[0], 0, 3, 0)
    terminal_tables: list[np.ndarray] = []
    for scalar in (0.0, 1.0):
        state = _new_state()
        state.expected_phase = 3
        _select(state, {"observation": terminal}, forced_action=0)
        _update(
            state,
            {
                "done": True,
                "next_observation": None,
                "reward": _immutable_reward(scalar),
            },
        )
        terminal_tables.append(state.q.copy())
    differing = np.argwhere(terminal_tables[0] != terminal_tables[1])
    terminal_dependency = bool(
        differing.shape == (1, 4)
        and tuple(map(int, differing[0])) == (*_state_coords(terminal), 0)
    )

    state = _new_state()
    state.q[(*_state_coords(terminal), 0)] = 1.0
    phase_two = _public_observation(REGIMES[0], 0, 2, 0)
    phase_one = _public_observation(REGIMES[0], 0, 1, 0)
    earlier_before = float(state.q[(*_state_coords(phase_two), 0)]) == 0.0
    state.expected_phase = 2
    _select(state, {"observation": phase_two}, forced_action=0)
    audit = _update(
        state,
        {
            "done": False,
            "next_observation": terminal,
            "reward": _immutable_reward(0.0),
        },
    )
    phase_two_after = float(state.q[(*_state_coords(phase_two), 0)]) == 1.0
    phase_one_unchanged = float(state.q[(*_state_coords(phase_one), 0)]) == 0.0
    earlier_cells_unchanged = bool(
        earlier_before and phase_two_after and phase_one_unchanged
    )
    return {
        "dependency_checks": 3,
        "earlier_cells_unchanged": earlier_cells_unchanged,
        "passed": bool(
            successor_dependency
            and terminal_dependency
            and earlier_cells_unchanged
            and audit.target_value == 1.0
        ),
        "trace_sha256": _json_sha256(
            {
                "earlier": earlier_cells_unchanged,
                "successor_targets": targets,
                "terminal_cells": differing.tolist(),
            }
        ),
    }


def _with_component(
    trace: TraceBundle, component: str, rows: tuple[object, ...]
) -> TraceBundle:
    values = {
        "observations": trace.observations,
        "actions": trace.actions,
        "transitions": trace.transitions,
        "feedback": trace.feedback,
    }
    if component not in values:
        raise ValueError("unknown trace component")
    values[component] = rows
    return TraceBundle(
        observations=values["observations"],
        actions=values["actions"],
        transitions=values["transitions"],
        feedback=values["feedback"],
    )


def _malformed_component_case(trace: TraceBundle) -> dict[str, object]:
    attacks: list[tuple[str, TraceBundle]] = []
    component_rows = {
        "observations": trace.observations,
        "actions": trace.actions,
        "transitions": trace.transitions,
        "feedback": trace.feedback,
    }
    for name, rows in component_rows.items():
        attacks.extend(
            [
                (f"{name}_missing", _with_component(trace, name, rows[1:])),
                (
                    f"{name}_duplicate",
                    _with_component(trace, name, rows + rows[:1]),
                ),
                (
                    f"{name}_extra",
                    _with_component(
                        trace,
                        name,
                        rows
                        + (
                            replace(
                                rows[0],
                                step_key=f"unknown:{name}:extra",
                            ),
                        ),
                    ),
                ),
                (
                    f"{name}_wrong_type",
                    _with_component(trace, name, (object(),) + rows[1:]),
                ),
            ]
        )

    observations = trace.observations
    actions = trace.actions
    transitions = trace.transitions
    feedback = trace.feedback
    attacks.extend(
        [
            (
                "observation_wrong_phase",
                _with_component(
                    trace,
                    "observations",
                    (replace(observations[0], phase=1),) + observations[1:],
                ),
            ),
            (
                "observation_wrong_predecessor",
                _with_component(
                    trace,
                    "observations",
                    (
                        replace(
                            observations[0],
                            predecessor_sha256="a" * 64,
                            observation_sha256="a" * 64,
                        ),
                    )
                    + observations[1:],
                ),
            ),
            (
                "observation_cross_episode",
                _with_component(
                    trace,
                    "observations",
                    (
                        replace(
                            observations[0],
                            episode_key=observations[HORIZON].episode_key,
                        ),
                    )
                    + observations[1:],
                ),
            ),
            (
                "action_wrong_value",
                _with_component(
                    trace,
                    "actions",
                    (replace(actions[0], action=1 - actions[0].action),)
                    + actions[1:],
                ),
            ),
            (
                "action_boolean_value",
                _with_component(
                    trace,
                    "actions",
                    (replace(actions[0], action=True),) + actions[1:],
                ),
            ),
            (
                "action_wrong_dtype",
                _with_component(
                    trace,
                    "actions",
                    (replace(actions[0], action_dtype="int64"),) + actions[1:],
                ),
            ),
            (
                "action_cross_episode",
                _with_component(
                    trace,
                    "actions",
                    (
                        replace(
                            actions[0], episode_key=actions[HORIZON].episode_key
                        ),
                    )
                    + actions[1:],
                ),
            ),
            (
                "transition_wrong_actual_successor",
                _with_component(
                    trace,
                    "transitions",
                    (replace(transitions[0], successor_sha256="b" * 64),)
                    + transitions[1:],
                ),
            ),
            (
                "transition_wrong_bootstrap_successor",
                _with_component(
                    trace,
                    "transitions",
                    (
                        replace(
                            transitions[0], bootstrap_successor_sha256="c" * 64
                        ),
                    )
                    + transitions[1:],
                ),
            ),
            (
                "transition_wrong_donor",
                _with_component(
                    trace,
                    "transitions",
                    (replace(transitions[0], donor_episode=1),)
                    + transitions[1:],
                ),
            ),
            (
                "transition_wrong_done",
                _with_component(
                    trace,
                    "transitions",
                    (replace(transitions[0], done=True),) + transitions[1:],
                ),
            ),
            (
                "transition_nonboolean_done",
                _with_component(
                    trace,
                    "transitions",
                    (replace(transitions[0], done=np.bool_(False)),)
                    + transitions[1:],
                ),
            ),
            (
                "transition_cross_episode",
                _with_component(
                    trace,
                    "transitions",
                    (
                        replace(
                            transitions[0],
                            episode_key=transitions[HORIZON].episode_key,
                        ),
                    )
                    + transitions[1:],
                ),
            ),
            (
                "feedback_wrong_reward",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], reward=1.0),) + feedback[1:],
                ),
            ),
            (
                "feedback_wrong_update_reward",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], update_reward=1.0),) + feedback[1:],
                ),
            ),
            (
                "feedback_wrong_origin",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], origin_episode=1),) + feedback[1:],
                ),
            ),
            (
                "feedback_nonfinite_reward",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], reward=float("nan")),) + feedback[1:],
                ),
            ),
            (
                "feedback_nonfinite_update_reward",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], update_reward=float("inf")),)
                    + feedback[1:],
                ),
            ),
            (
                "feedback_nonboolean_done",
                _with_component(
                    trace,
                    "feedback",
                    (replace(feedback[0], done=0),) + feedback[1:],
                ),
            ),
            (
                "feedback_cross_episode",
                _with_component(
                    trace,
                    "feedback",
                    (
                        replace(
                            feedback[0], episode_key=feedback[HORIZON].episode_key
                        ),
                    )
                    + feedback[1:],
                ),
            ),
            (
                "transition_payload_in_canonical",
                _with_component(
                    trace,
                    "transitions",
                    (replace(transitions[0], donor_episode=8),)
                    + transitions[1:],
                ),
            ),
            (
                "reward_origin_payload_in_canonical",
                _with_component(
                    trace,
                    "feedback",
                    feedback[:3]
                    + (replace(feedback[3], origin_episode=12),)
                    + feedback[4:],
                ),
            ),
        ]
    )

    sentinel_state = _new_state()
    state_before = _state_commitment(sentinel_state)
    rejected = 0
    unchanged = 0
    labels: list[str] = []
    for label, attacked_trace in attacks:
        try:
            _score_trace(attacked_trace, "train", control_mode="canonical")
        except (TypeError, ValueError, RuntimeError, AttributeError):
            rejected += 1
            labels.append(label)
            unchanged += int(_state_commitment(sentinel_state) == state_before)
            canonical_score, _ = _score_trace(
                trace, "train", control_mode="canonical"
            )
            if canonical_score != 0.5:
                raise RuntimeError("canonical scorer was damaged by rejection")
    return {
        "attacks_checked": len(attacks),
        "attacks_rejected": rejected,
        "passed": bool(rejected == len(attacks) and unchanged == len(attacks)),
        "state_unchanged_checks": unchanged,
        "trace_sha256": _json_sha256(labels),
    }


def _timing_attack_case() -> dict[str, object]:
    sample0 = _public_observation(REGIMES[0], 0, 0, 0)
    sample1 = _public_observation(REGIMES[0], 0, 1, 0)
    sample1_wrong = _public_observation(REGIMES[0], 0, 1, 1)
    sample2 = _public_observation(REGIMES[0], 0, 2, 0)
    sample3 = _public_observation(REGIMES[0], 0, 3, 0)
    attacks: list[
        tuple[str, LearnerState, EpisodeProtocol | None, Callable[[], object]]
    ] = []

    def add(
        label: str,
        state: LearnerState,
        call: Callable[[], object],
        protocol: EpisodeProtocol | None = None,
    ) -> None:
        attacks.append((label, state, protocol, call))

    # 1. update before selection
    state = _new_state()
    add(
        "update_before_selection",
        state,
        lambda state=state: _update(
            state,
            {
                "done": False,
                "next_observation": sample1,
                "reward": _immutable_reward(0.0),
            },
        ),
    )
    # 2. phase-one selection before phase zero
    state = _new_state()
    add(
        "phase_one_before_zero",
        state,
        lambda state=state: _select(state, {"observation": sample1}),
    )
    # 3. repeated, skipped, rewound, and post-terminal selection
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "repeated_selection",
        state,
        lambda state=state: _select(state, {"observation": sample0}),
    )
    state = _new_state()
    add(
        "skipped_selection",
        state,
        lambda state=state: _select(state, {"observation": sample2}),
    )
    state = _new_state()
    state.expected_phase = 2
    add(
        "rewound_selection",
        state,
        lambda state=state: _select(state, {"observation": sample1}),
    )
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.stage = "closed"
    add(
        "post_terminal_selection",
        state,
        lambda protocol=protocol: protocol.observe(0),
        protocol,
    )
    # 4. transition before action validation
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.observe(0)
    protocol.select()
    add(
        "transition_before_validation",
        state,
        lambda protocol=protocol: protocol.transition(
            action=0, successor_present=True
        ),
        protocol,
    )
    # 5. successor before transition and successor for wrong action
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.observe(0)
    protocol.select()
    add(
        "successor_before_transition",
        state,
        lambda protocol=protocol: protocol.feedback(reward=0.0, done=False),
        protocol,
    )
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.observe(0)
    protocol.select()
    protocol.validate(_immutable_action(0))
    add(
        "successor_wrong_action",
        state,
        lambda protocol=protocol: protocol.transition(
            action=1, successor_present=True
        ),
        protocol,
    )
    # 6. nonterminal and terminal scalar materialized early
    for label, phase, done in (
        ("nonterminal_scalar_early", 0, False),
        ("terminal_scalar_early", 3, True),
    ):
        state = _new_state()
        protocol = EpisodeProtocol()
        protocol.phase = phase
        protocol.observe(phase)
        protocol.select()
        protocol.validate(_immutable_action(0))
        add(
            label,
            state,
            lambda protocol=protocol, done=done: protocol.feedback(
                reward=0.0, done=done
            ),
            protocol,
        )
    # 7. terminal scalar delivered late or twice
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.phase = 3
    protocol.observe(3)
    protocol.select()
    protocol.validate(_immutable_action(0))
    protocol.transition(action=0, successor_present=False)
    protocol.feedback(reward=1.0, done=True)
    protocol.target()
    protocol.update()
    add(
        "terminal_scalar_late",
        state,
        lambda protocol=protocol: protocol.feedback(reward=1.0, done=True),
        protocol,
    )
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.phase = 3
    protocol.observe(3)
    protocol.select()
    protocol.validate(_immutable_action(0))
    protocol.transition(action=0, successor_present=False)
    protocol.feedback(reward=1.0, done=True)
    add(
        "terminal_scalar_twice",
        state,
        lambda protocol=protocol: protocol.feedback(reward=1.0, done=True),
        protocol,
    )
    # 8. missing or nonzero nonterminal feedback
    for label, payload in (
        (
            "missing_nonterminal_feedback",
            {"done": False, "next_observation": sample1},
        ),
        (
            "nonzero_nonterminal_feedback",
            {
                "done": False,
                "next_observation": sample1,
                "reward": _immutable_reward(1.0),
            },
        ),
    ):
        state = _new_state()
        _select(state, {"observation": sample0}, forced_action=0)
        add(label, state, lambda state=state, payload=payload: _update(state, payload))
    # 9. done true nonterminal or done false terminal
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "done_true_nonterminal",
        state,
        lambda state=state: _update(
            state,
            {
                "done": True,
                "next_observation": sample1,
                "reward": _immutable_reward(0.0),
            },
        ),
    )
    state = _new_state()
    state.expected_phase = 3
    _select(state, {"observation": sample3}, forced_action=0)
    add(
        "done_false_terminal",
        state,
        lambda state=state: _update(
            state,
            {
                "done": False,
                "next_observation": None,
                "reward": _immutable_reward(1.0),
            },
        ),
    )
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "done_numpy_boolean_nonterminal",
        state,
        lambda state=state: _update(
            state,
            {
                "done": np.bool_(False),
                "next_observation": sample1,
                "reward": _immutable_reward(0.0),
            },
        ),
    )
    state = _new_state()
    state.expected_phase = 3
    _select(state, {"observation": sample3}, forced_action=0)
    add(
        "done_integer_terminal",
        state,
        lambda state=state: _update(
            state,
            {
                "done": 1,
                "next_observation": None,
                "reward": _immutable_reward(1.0),
            },
        ),
    )
    # 10. absent nonterminal successor or present terminal successor
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "absent_nonterminal_successor",
        state,
        lambda state=state: _update(
            state,
            {
                "done": False,
                "next_observation": None,
                "reward": _immutable_reward(0.0),
            },
        ),
    )
    state = _new_state()
    state.expected_phase = 3
    _select(state, {"observation": sample3}, forced_action=0)
    add(
        "present_terminal_successor",
        state,
        lambda state=state: _update(
            state,
            {
                "done": True,
                "next_observation": sample1,
                "reward": _immutable_reward(1.0),
            },
        ),
    )
    # 11. wrong observation, action, predecessor, or phase payload
    for field in ("observation", "action", "predecessor", "phase"):
        state = _new_state()
        _select(state, {"observation": sample0}, forced_action=0)
        payload = {
            "done": False,
            "next_observation": sample1,
            "reward": _immutable_reward(0.0),
            field: object(),
        }
        add(
            f"wrong_{field}_update_payload",
            state,
            lambda state=state, payload=payload: _update(state, payload),
        )
    # 12. duplicate update, log before update, close before update, update after close
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    _update(
        state,
        {
            "done": False,
            "next_observation": sample1,
            "reward": _immutable_reward(0.0),
        },
    )
    add(
        "duplicate_update",
        state,
        lambda state=state: _update(
            state,
            {
                "done": False,
                "next_observation": sample1,
                "reward": _immutable_reward(0.0),
            },
        ),
    )
    for label, method in (
        ("log_before_update", "log"),
        ("close_before_update", "close"),
    ):
        state = _new_state()
        protocol = EpisodeProtocol()
        protocol.observe(0)
        protocol.select()
        protocol.validate(_immutable_action(0))
        protocol.transition(action=0, successor_present=True)
        protocol.feedback(reward=0.0, done=False)
        protocol.target()
        add(
            label,
            state,
            lambda protocol=protocol, method=method: getattr(protocol, method)(),
            protocol,
        )
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.stage = "closed"
    add(
        "update_after_close",
        state,
        lambda protocol=protocol: protocol.update(),
        protocol,
    )
    # 13. next episode or split while pending
    for label in ("next_episode_while_pending", "next_split_while_pending"):
        state = _new_state()
        _select(state, {"observation": sample0}, forced_action=0)
        add(label, state, lambda state=state: _assert_boundary_clean(state))
    # 14. nonempty queue or pending state at boundary
    state = _new_state()
    add(
        "nonempty_scalar_queue_at_boundary",
        state,
        lambda state=state: _assert_boundary_clean(
            state, delayed_scalars=(1.0,)
        ),
    )
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "pending_state_at_boundary",
        state,
        lambda state=state: _assert_boundary_clean(state),
    )
    # 15. donor/origin identity or early origin scalar crossing learner boundary
    for field in ("donor", "origin"):
        state = _new_state()
        add(
            f"{field}_identity_exposed",
            state,
            lambda state=state, field=field: _select(
                state, {"observation": sample0, field: 1}
            ),
        )
    state = _new_state()
    protocol = EpisodeProtocol()
    protocol.observe(0)
    protocol.select()
    protocol.validate(_immutable_action(0))
    add(
        "origin_scalar_before_terminal",
        state,
        lambda protocol=protocol: protocol.feedback(reward=1.0, done=False),
        protocol,
    )
    # 16. mutable observation and pending observation mutation
    mutable = sample0.copy()
    state = _new_state()
    add(
        "mutable_observation",
        state,
        lambda state=state: _select(state, {"observation": mutable}),
    )
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "pending_observation_mutation",
        state,
        lambda: sample0.__setitem__(0, 1.0),
    )
    # 17. held-out updater call and held-out source access during training
    state = _new_state()
    add(
        "heldout_updater_call",
        state,
        lambda: _train_policy((_episodes("validation")[0],)),
    )
    source = ExplodingSource()
    state = _new_state()
    add("heldout_source_access", state, lambda source=source: bool(source))
    # 18. malformed, reentrant, and failed partial mutation calls
    state = _new_state()
    add("malformed_selector_call", state, lambda state=state: _select(state, object()))
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "reentrant_selector_call",
        state,
        lambda state=state: _select(state, {"observation": sample0}),
    )
    state = _new_state()
    _select(state, {"observation": sample0}, forced_action=0)
    add(
        "failed_call_partial_mutation",
        state,
        lambda state=state: _update(
            state,
            {
                "done": False,
                "next_observation": sample1_wrong,
                "reward": _immutable_reward(0.0),
            },
        ),
    )

    if len(attacks) != 43:
        raise RuntimeError("frozen timing attack cardinality drifted")
    rejected = 0
    unchanged = 0
    labels: list[str] = []
    for label, state, protocol, call in attacks:
        before = _json_sha256(
            {
                "protocol": None if protocol is None else protocol.commitment(),
                "state": _state_commitment(state),
            }
        )
        try:
            call()
        except (TypeError, ValueError, RuntimeError, AttributeError):
            rejected += 1
            after = _json_sha256(
                {
                    "protocol": None if protocol is None else protocol.commitment(),
                    "state": _state_commitment(state),
                }
            )
            unchanged += int(before == after)
            labels.append(label)
    return {
        "attacks_checked": len(attacks),
        "attacks_rejected": rejected,
        "passed": bool(rejected == len(attacks) and unchanged == len(attacks)),
        "state_unchanged_checks": unchanged,
        "trace_sha256": _json_sha256(labels),
    }


def _transition_control_case(
    true_state: LearnerState,
) -> tuple[dict[str, object], TrainRun]:
    control = _train_policy(
        _episodes("train"), control_mode="transition_target"
    )
    validation = _evaluate_policy(control.state, "validation")[0]
    test = _evaluate_policy(control.state, "test")[0]
    true_test = _evaluate_policy(true_state, "test")[0]
    mapping_exact = True
    for regime in (row for row in REGIMES if row.split == "train"):
        for block in (0, 16):
            destinations = tuple(range(block, block + 16))
            donors = tuple(_transition_donor_episode(row) for row in destinations)
            mapping_exact = bool(
                mapping_exact
                and donors
                == tuple(block + value for value in TRANSITION_DONOR_PERMUTATION)
                and sorted(donors) == list(destinations)
                and all(left != right for left, right in zip(destinations, donors))
                and all(
                    _behavior_action(left) == _behavior_action(right)
                    and _evaluator_target("train", regime.code, left)
                    != _evaluator_target("train", regime.code, right)
                    for left, right in zip(destinations, donors)
                )
            )
    score, score_trace = _score_trace(
        control.trace, "train", control_mode="transition_target"
    )
    positive_gate_rejected = bool(
        validation
        <= THRESHOLDS["maximum_transition_target_validation_macro_return"]
        and test <= THRESHOLDS["maximum_transition_target_test_macro_return"]
        and true_test - test >= THRESHOLDS["minimum_transition_target_gap"]
    )
    case = {
        "donor_mapping_exact": mapping_exact,
        "passed": bool(
            mapping_exact
            and score == 0.5
            and positive_gate_rejected
            and control.state.total_updates == 512
        ),
        "positive_gate_rejected": positive_gate_rejected,
        "test_macro_return": test,
        "trace_sha256": _json_sha256(
            {
                "actual_transition": _actual_transition_commitment(control.trace),
                "control_trace": _trace_sha256(control.trace),
                "score_trace": score_trace,
                "state": _state_commitment(control.state),
            }
        ),
        "true_test_gap": true_test - test,
        "validation_macro_return": validation,
    }
    return case, control


def _reward_origin_control_case(
    true_state: LearnerState,
) -> tuple[dict[str, object], TrainRun]:
    poison_calls = 0

    def poisoned_oracle() -> object:
        nonlocal poison_calls
        poison_calls += 1
        raise RuntimeError("outcome oracle must not be used by mapping")

    constructed = tuple(
        _reward_origin_episode(index, poisoned_reward_oracle=poisoned_oracle)
        for index in range(16)
    )
    origin_mapping_exact = bool(
        constructed == REWARD_ORIGIN_PERMUTATION
        and sorted(constructed) == list(range(16))
        and all(destination != origin for destination, origin in enumerate(constructed))
    )
    cell_balance_exact = True
    reward_multiset_unchanged = True
    cell_rows: list[object] = []
    for regime in (row for row in REGIMES if row.split == "train"):
        for block in (0, 16):
            canonical_values: list[int] = []
            assigned_values: list[int] = []
            cells: dict[tuple[int, int], list[int]] = {}
            for offset in range(16):
                destination = block + offset
                target = _evaluator_target("train", regime.code, destination)
                action = _behavior_action(destination)
                prefix = 0 if action == 0 else 7
                canonical = int(_terminal_reward(target, prefix, action))
                origin = block + REWARD_ORIGIN_PERMUTATION[offset]
                assigned = int(
                    _terminal_origin_reward(
                        Episode(
                            regime=regime,
                            index=destination,
                            target=target,
                            episode_key=_episode_key(regime, destination),
                        ),
                        origin,
                    )
                )
                canonical_values.append(canonical)
                assigned_values.append(assigned)
                cells.setdefault((target, action), []).append(assigned)
            reward_multiset_unchanged = bool(
                reward_multiset_unchanged
                and sorted(canonical_values) == sorted(assigned_values)
                and sum(canonical_values) == 8
            )
            for cell, values in sorted(cells.items()):
                exact = len(values) == 4 and sum(values) == 2
                cell_balance_exact = bool(cell_balance_exact and exact)
                cell_rows.append((regime.code, block, cell, len(values), sum(values)))

    control = _train_policy(_episodes("train"), control_mode="reward_origin")
    validation = _evaluate_policy(control.state, "validation")[0]
    test = _evaluate_policy(control.state, "test")[0]
    true_test = _evaluate_policy(true_state, "test")[0]
    score, score_trace = _score_trace(
        control.trace, "train", control_mode="reward_origin"
    )
    positive_gate_rejected = bool(
        validation
        <= THRESHOLDS["maximum_reward_origin_validation_macro_return"]
        and test <= THRESHOLDS["maximum_reward_origin_test_macro_return"]
        and true_test - test >= THRESHOLDS["minimum_reward_origin_gap"]
    )
    mapping_outcome_blind = poison_calls == 0
    case = {
        "cell_balance_exact": cell_balance_exact,
        "mapping_outcome_blind": mapping_outcome_blind,
        "origin_mapping_exact": origin_mapping_exact,
        "passed": bool(
            origin_mapping_exact
            and mapping_outcome_blind
            and cell_balance_exact
            and reward_multiset_unchanged
            and score == 0.5
            and positive_gate_rejected
        ),
        "positive_gate_rejected": positive_gate_rejected,
        "reward_multiset_unchanged": reward_multiset_unchanged,
        "test_macro_return": test,
        "trace_sha256": _json_sha256(
            {
                "cells": cell_rows,
                "control_trace": _trace_sha256(control.trace),
                "score_trace": score_trace,
                "state": _state_commitment(control.state),
            }
        ),
        "true_test_gap": true_test - test,
        "validation_macro_return": validation,
    }
    return case, control


def _signal_ablation_case(
    true_state: LearnerState,
) -> dict[str, object]:
    legal_rows_checked = 0
    only_signal_changed = True
    digest_rows: list[object] = []
    for regime in REGIMES:
        for episode in range(EPISODES_PER_REGIME):
            target = _evaluator_target(regime.split, regime.code, episode)
            for phase in range(HORIZON):
                for prefix in range(2**phase):
                    canonical = _public_observation(regime, episode, phase, prefix)
                    ablated = _public_observation(
                        regime,
                        episode,
                        phase,
                        prefix,
                        ablate_signal=True,
                    )
                    node_ok = bool(
                        canonical[1] != 0.0
                        and ablated[1] == 0.0
                        and not np.array_equal(canonical, ablated)
                        and canonical.dtype == ablated.dtype == np.float64
                        and canonical.shape == ablated.shape == (4,)
                        and not canonical.flags.writeable
                        and not ablated.flags.writeable
                        and canonical.flags.c_contiguous
                        and ablated.flags.c_contiguous
                        and canonical[[0, 2, 3]].tobytes(order="C")
                        == ablated[[0, 2, 3]].tobytes(order="C")
                    )
                    for action in (0, 1):
                        done = phase == HORIZON - 1
                        if done:
                            successor_equal = True
                            reward = _terminal_reward(target, prefix, action)
                        else:
                            next_prefix = 2 * prefix + action
                            canonical_successor = _public_observation(
                                regime, episode, phase + 1, next_prefix
                            )
                            ablated_successor = _public_observation(
                                regime,
                                episode,
                                phase + 1,
                                next_prefix,
                                ablate_signal=True,
                            )
                            successor_equal = bool(
                                canonical_successor[1] != 0.0
                                and ablated_successor[1] == 0.0
                                and canonical_successor[[0, 2, 3]].tobytes(order="C")
                                == ablated_successor[[0, 2, 3]].tobytes(order="C")
                            )
                            reward = 0.0
                        only_signal_changed = bool(
                            only_signal_changed and node_ok and successor_equal
                        )
                        legal_rows_checked += 1
                        digest_rows.append(
                            {
                                "action": action,
                                "ablated_sha256": _observation_sha256(ablated),
                                "canonical_sha256": _observation_sha256(canonical),
                                "done": done,
                                "donor": (
                                    _transition_donor_episode(episode)
                                    if regime.split == "train" and not done
                                    else None
                                ),
                                "origin": (
                                    _reward_origin_episode(episode)
                                    if regime.split == "train" and done
                                    else None
                                ),
                                "reward": reward,
                                "row_key": _row_key(
                                    _episode_key(regime, episode),
                                    phase,
                                    prefix,
                                    action,
                                ),
                            }
                        )
    ablated = _train_policy(_episodes("train"), ablate_signal=True)
    refit_test = _evaluate_policy(
        ablated.state, "test", ablate_signal=True
    )[0]
    true_policy_test = _evaluate_policy(
        true_state, "test", ablate_signal=True
    )[0]
    positive_gate_rejected = bool(
        refit_test <= THRESHOLDS["maximum_attribution_test_macro_return"]
        and true_policy_test
        <= THRESHOLDS["maximum_attribution_test_macro_return"]
    )
    return {
        "legal_rows_checked": legal_rows_checked,
        "only_signal_changed": only_signal_changed,
        "passed": bool(
            legal_rows_checked == 7680
            and only_signal_changed
            and positive_gate_rejected
        ),
        "positive_gate_rejected": positive_gate_rejected,
        "refit_test_macro_return": refit_test,
        "trace_sha256": _json_sha256(
            {
                "family": digest_rows,
                "refit_state": _state_commitment(ablated.state),
                "refit_trace": _trace_sha256(ablated.trace),
            }
        ),
        "true_policy_test_macro_return": true_policy_test,
    }


def _non_process_projection() -> dict[str, dict[str, object]]:
    cases = _family_and_contract_cases()
    cases["physical_pre_action_boundary"] = _physical_boundary_case()

    absent = _train_policy(_episodes("train"))
    absent_state_sha256 = _state_commitment(absent.state)
    absent_trace_sha256 = _trace_sha256(absent.trace)
    cases["heldout_absent_source"] = {
        "passed": True,
        "state_sha256": absent_state_sha256,
        "train_trace_sha256": absent_trace_sha256,
    }

    exploding_source = ExplodingSource()
    exploding = _train_policy(_episodes("train"))
    operations_unreached = sum(
        value == 0 for value in exploding_source.counts.values()
    )
    exploding_state_sha256 = _state_commitment(exploding.state)
    exploding_trace_sha256 = _trace_sha256(exploding.trace)
    cases["heldout_exploding_source"] = {
        "operations_checked": len(ExplodingSource.OPERATIONS),
        "operations_unreached": operations_unreached,
        "passed": bool(
            operations_unreached == len(ExplodingSource.OPERATIONS)
            and exploding_state_sha256 == absent_state_sha256
            and exploding_trace_sha256 == absent_trace_sha256
        ),
        "state_sha256": exploding_state_sha256,
        "train_trace_sha256": exploding_trace_sha256,
    }
    cases["td_target_dependency"] = _td_dependency_case()

    expected_offsets = (
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
        (1, 3, 12),
        (1, 2, 13),
        (1, 1, 14),
        (1, 0, 15),
    )
    propagation_offsets_exact = absent.first_positive_offsets == expected_offsets
    event_orders_exact = sum(order == EVENT_ORDER for order in absent.event_orders)
    cases["td_update_order_and_terminal_dependency"] = {
        "bootstrap_updates": absent.state.bootstrap_updates,
        "event_orders_checked": len(absent.event_orders),
        "passed": bool(
            absent.state.total_updates == 512
            and absent.state.bootstrap_updates == 384
            and absent.state.terminal_updates == 128
            and event_orders_exact == 128
            and propagation_offsets_exact
            and absent.state.pending is None
            and absent.state.expected_phase == 0
        ),
        "propagation_offsets_exact": propagation_offsets_exact,
        "terminal_updates": absent.state.terminal_updates,
        "total_updates": absent.state.total_updates,
        "trace_sha256": _json_sha256(
            {
                "event_orders": absent.event_orders,
                "offsets": absent.first_positive_offsets,
                "state": absent_state_sha256,
            }
        ),
    }

    reorder_variants = tuple(FIXTURE_IDENTITY["component_reorder_variants"])
    scores: list[float] = []
    score_traces: list[str] = []
    for variant in reorder_variants:
        score, trace_sha256 = _score_trace(
            _reordered_trace(absent.trace, variant),
            "train",
            control_mode="canonical",
        )
        scores.append(score)
        score_traces.append(trace_sha256)
    scores_equal = len(set(scores)) == 1 and scores[0] == 0.5
    cases["authenticated_component_recombination"] = {
        "components_authenticated": 4,
        "passed": bool(scores_equal and len(reorder_variants) == 6),
        "reorder_variants": len(reorder_variants),
        "scores_equal": scores_equal,
        "trace_sha256": _json_sha256(
            {"scores": scores, "traces": score_traces}
        ),
    }
    cases["malformed_and_cross_episode_rejection"] = _malformed_component_case(
        absent.trace
    )

    baselines = _baseline_metrics()
    cases["baseline_replay"] = {
        "best_baseline_test_macro_return": baselines["best_test"],
        "best_baseline_validation_macro_return": baselines["best_validation"],
        "constant_returns_exact": baselines["constant_exact"],
        "myopic_returns_exact": baselines["myopic_exact"],
        "no_bootstrap_returns_exact": baselines["no_bootstrap_exact"],
        "passed": bool(
            baselines["constant_exact"]
            and baselines["myopic_exact"]
            and baselines["no_bootstrap_exact"]
            and baselines["random_replay_exact"]
        ),
        "random_replay_exact": baselines["random_replay_exact"],
        "trace_sha256": baselines["trace_sha256"],
    }

    behavior_return, behavior_regret, behavior_regimes = _behavior_metrics("train")
    train_return, _train_minimum, _train_regimes, train_trace = _evaluate_policy(
        absent.state, "train"
    )
    validation_return, validation_minimum, _validation_regimes, validation_trace = (
        _evaluate_policy(absent.state, "validation")
    )
    test_return, test_minimum, _test_regimes, test_trace = _evaluate_policy(
        absent.state, "test"
    )
    minimum_heldout = min(validation_minimum, test_minimum)
    validation_gain = validation_return - float(baselines["best_validation"])
    test_gain = test_return - float(baselines["best_test"])
    recovery_passed = bool(
        behavior_return == 0.5
        and behavior_regret == 64.0
        and all(value == 0.5 for value in behavior_regimes.values())
        and train_return >= THRESHOLDS["minimum_postfit_train_macro_return"]
        and validation_return
        >= THRESHOLDS["minimum_postfit_validation_macro_return"]
        and test_return >= THRESHOLDS["minimum_postfit_test_macro_return"]
        and minimum_heldout >= THRESHOLDS["minimum_heldout_regime_return"]
        and validation_gain >= THRESHOLDS["minimum_validation_gain_baseline"]
        and test_gain >= THRESHOLDS["minimum_test_gain_baseline"]
    )
    cases["multistep_value_recovery"] = {
        "behavior_regret": behavior_regret,
        "behavior_return": behavior_return,
        "minimum_heldout_regime_return": minimum_heldout,
        "passed": recovery_passed,
        "postfit_test_macro_return": test_return,
        "postfit_train_macro_return": train_return,
        "postfit_validation_macro_return": validation_return,
        "test_gain_baseline": test_gain,
        "trace_sha256": _json_sha256(
            {
                "state": absent_state_sha256,
                "test": test_trace,
                "train": train_trace,
                "validation": validation_trace,
            }
        ),
        "validation_gain_baseline": validation_gain,
    }

    transition_case, transition_run = _transition_control_case(absent.state)
    cases["transition_target_control"] = transition_case
    origin_case, origin_run = _reward_origin_control_case(absent.state)
    cases["outcome_blind_reward_origin_control"] = origin_case
    cases["full_timing_attack_matrix"] = _timing_attack_case()
    cases["all_trajectory_signal_ablation"] = _signal_ablation_case(absent.state)

    if _action_commitment(absent.trace) != _action_commitment(transition_run.trace):
        raise RuntimeError("transition control changed the behavior action commitment")
    if _actual_transition_commitment(absent.trace) != _actual_transition_commitment(
        transition_run.trace
    ):
        raise RuntimeError("transition control changed canonical successors")
    if _canonical_feedback_commitment(absent.trace) != _canonical_feedback_commitment(
        transition_run.trace
    ):
        raise RuntimeError("transition control changed canonical feedback")
    if _action_commitment(absent.trace) != _action_commitment(origin_run.trace):
        raise RuntimeError("reward-origin control changed behavior actions")
    if _actual_transition_commitment(absent.trace) != _actual_transition_commitment(
        origin_run.trace
    ):
        raise RuntimeError("reward-origin control changed canonical successors")
    if _canonical_feedback_commitment(absent.trace) != _canonical_feedback_commitment(
        origin_run.trace
    ):
        raise RuntimeError("reward-origin control changed canonical feedback")

    if set(cases) != set(CASE_CONTRACT) - {"process_isolation"}:
        raise RuntimeError("non-process case contract drifted")
    for name, case in cases.items():
        if set(case) != set(CASE_CONTRACT[name]):
            raise RuntimeError(f"case field contract drifted: {name}")
    return cases


def isolated_worker_trace() -> dict[str, object]:
    """Return the complete sanitized non-process projection."""
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
        name: value
        for name, value in os.environ.items()
        if name.upper() in safe_names
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
            "experiments.local_lab.multistep_td_action_prefix_v2_worker",
            "--mode",
            "multistep-td-action-prefix-v2-trace",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def run_study(*, include_process_isolation: bool = True) -> dict[str, object]:
    """Execute the frozen study and return only sanitized aggregates."""
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the action-prefix study requires a CPU backend")
    local_projection = isolated_worker_trace()
    cases = dict(local_projection["cases"])
    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed: bool | None = bool(
            isolated_left == isolated_right == local_projection
        )
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = "not-run-in-focused-test"
    cases["process_isolation"] = {
        "passed": isolation_passed,
        "trace_sha256": isolation_digest,
    }
    for name, case in cases.items():
        if set(case) != set(CASE_CONTRACT[name]):
            raise RuntimeError(f"terminal case field contract drifted: {name}")
    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    return {
        "action": (
            "synthetic_multistep_td_action_prefix_recovered_for_harness"
            if passed
            else (
                "park_multistep_td_action_prefix_research"
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
