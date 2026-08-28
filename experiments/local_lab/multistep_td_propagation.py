"""Pre-result-rejected synthetic multi-step TD fixture.

This v1 fixture is retained only as an auditable preflight record. It is not an
approved local-lab study and must not be executed as terminal evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import jax
import numpy as np


STUDY_ID = "multistep-td-propagation-v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
OBSERVATION_FIELDS = ("phase", "signal", "alive", "nuisance")
POLICY_INPUT_FIELDS = ("observation",)
STRUCTURE_KIND = "none"
EPISODES_PER_REGIME = 32
HORIZON = 4
RANDOM_BASELINE_SEED = 2026082811
EXPECTED_DATASET_SHA256 = (
    "5a36ff02adb410c2ae2108626e9c6660bf02d2473b69a945b6c45dbe5f2a253d"
)
TRANSITION_TARGET_PERMUTATION = (
    4,
    5,
    6,
    7,
    0,
    1,
    2,
    3,
    12,
    13,
    14,
    15,
    8,
    9,
    10,
    11,
)
REWARD_ORIGIN_PERMUTATION = (
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    16,
    17,
    18,
    19,
)
FORBIDDEN_POLICY_FIELDS = (
    "target",
    "preferred_action",
    "reward",
    "transition",
    "next_observation",
    "terminal_state",
    "counterfactual_reward",
    "regime_code",
    "split",
    "episode_key",
    "step_key",
    "trajectory_id",
    "phase_counter",
    "done",
    "rng",
    "generator",
    "evaluator",
    "environment",
    "control_mode",
    "donor_identity",
    "reward_origin",
    "heldout_iterator",
)
EVENT_ORDER = (
    "observe0",
    "select0",
    "transition0",
    "feedback0",
    "td_target0",
    "td_update0",
    "log0",
    "observe1",
    "select1",
    "transition1",
    "feedback1",
    "td_target1",
    "td_update1",
    "log1",
    "observe2",
    "select2",
    "transition2",
    "feedback2",
    "td_target2",
    "td_update2",
    "log2",
    "observe3",
    "select3",
    "terminal_reward",
    "td_target3",
    "td_update3",
    "log3",
    "close_episode",
)
REGIMES = (
    {
        "split": "train",
        "code": 1009,
        "signal_scale": 0.80,
        "nuisance_shift": -0.80,
        "nuisance_scale": 0.85,
    },
    {
        "split": "train",
        "code": 1013,
        "signal_scale": 0.95,
        "nuisance_shift": -0.25,
        "nuisance_scale": 1.10,
    },
    {
        "split": "train",
        "code": 1019,
        "signal_scale": 1.10,
        "nuisance_shift": 0.25,
        "nuisance_scale": 0.75,
    },
    {
        "split": "train",
        "code": 1021,
        "signal_scale": 1.25,
        "nuisance_shift": 0.80,
        "nuisance_scale": 1.20,
    },
    {
        "split": "validation",
        "code": 1103,
        "signal_scale": 0.65,
        "nuisance_shift": -1.35,
        "nuisance_scale": 0.65,
    },
    {
        "split": "validation",
        "code": 1109,
        "signal_scale": 1.35,
        "nuisance_shift": 1.35,
        "nuisance_scale": 1.25,
    },
    {
        "split": "test",
        "code": 1201,
        "signal_scale": 0.55,
        "nuisance_shift": -1.85,
        "nuisance_scale": 0.55,
    },
    {
        "split": "test",
        "code": 1213,
        "signal_scale": 1.45,
        "nuisance_shift": 1.85,
        "nuisance_scale": 1.45,
    },
)
REGIME_COUNTS = {"train": 4, "validation": 2, "test": 2}
EPISODE_COUNTS = {
    name: count * EPISODES_PER_REGIME for name, count in REGIME_COUNTS.items()
}
THRESHOLDS = {
    "maximum_attribution_test_macro_return": 0.55,
    "maximum_reward_origin_test_macro_return": 0.55,
    "maximum_reward_origin_validation_macro_return": 0.55,
    "maximum_transition_target_test_macro_return": 0.05,
    "maximum_transition_target_validation_macro_return": 0.05,
    "minimum_heldout_regime_return": 0.98,
    "minimum_postfit_test_macro_return": 0.99,
    "minimum_postfit_train_macro_return": 0.99,
    "minimum_postfit_validation_macro_return": 0.99,
    "minimum_test_gain_baseline": 0.30,
    "minimum_transition_target_gap": 0.90,
    "minimum_reward_origin_gap": 0.40,
    "minimum_validation_gain_baseline": 0.30,
}


CASE_CONTRACT = {
    "typed_multistep_contract": {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "done_pattern": [False, False, False, True],
        "event_order": list(EVENT_ORDER),
        "horizon": HORIZON,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [4],
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
    },
    "generator_partition": {
        "episodes_per_regime": EPISODES_PER_REGIME,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "episode_counts": EPISODE_COUNTS,
        "regime_counts": REGIME_COUNTS,
        "regimes": list(REGIMES),
    },
    "action_dependent_transition": {
        "alive_values": [0.0, 1.0],
        "done_pattern": [False, False, False, True],
        "nonterminal_reward": 0.0,
        "transition_count_per_episode": 3,
    },
    "td_target_dependency": {
        "discount": 1.0,
        "learning_rate": 1.0,
        "nonterminal_target": "reward_plus_max_successor_value",
        "terminal_target": "terminal_reward_only",
    },
    "td_bootstrap_update_order": {
        "bootstrap_updates": 384,
        "episode_updates": 128,
        "phase_update_order": [0, 1, 2, 3],
        "target_one_offsets": [15, 14, 13, 12],
        "target_zero_offsets": [3, 2, 1, 0],
        "terminal_updates": 128,
        "total_updates": 512,
    },
    "leakage_guards": {
        "forbidden_policy_fields": list(FORBIDDEN_POLICY_FIELDS),
        "heldout_updates": 0,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "train_split": "train_only",
    },
    "authenticated_scoring": {
        "component_reorder": "independent_reverse_keyed_components",
        "scoring": "macro_mean_terminal_return_equal_weight_by_regime",
    },
    "baseline_replay": {
        "constant_actions": [0, 1],
        "myopic": "terminal_immediate_reward_only",
        "no_bootstrap": "immediate_reward_target_only",
        "random_seed": RANDOM_BASELINE_SEED,
        "scoring": "macro_mean_terminal_return_equal_weight_by_regime",
    },
    "multistep_value_recovery": {
        "model": "four_phase_two_sign_two_alive_two_action_td0_table",
        "thresholds": THRESHOLDS,
        "tie_action": 0,
    },
    "transition_target_control": {
        "scope": "train_bootstrap_successor_rows_within_each_sixteen_episode_block",
        "transition_target_permutation": list(TRANSITION_TARGET_PERMUTATION),
        "thresholds": {
            "maximum_test_macro_return": 0.05,
            "maximum_validation_macro_return": 0.05,
            "minimum_true_test_gap": 0.90,
        },
    },
    "reward_origin_control": {
        "boundary_pending_episode": "must_be_empty",
        "reward_origin_permutation": list(REWARD_ORIGIN_PERMUTATION),
        "scope": "train_terminal_feedback_origin_within_each_regime",
        "thresholds": {
            "maximum_test_macro_return": 0.55,
            "maximum_validation_macro_return": 0.55,
            "minimum_true_test_gap": 0.40,
        },
    },
    "signal_attribution_control": {
        "maximum_test_macro_return": 0.55,
        "signal_index": 1,
        "variants": ["refit_without_signal", "true_policy_without_signal"],
    },
    "process_isolation": {
        "source_projection": "complete_non_process_cases",
        "workers": 2,
    },
}

FIXTURE_IDENTITY = {
    "action_dtype": "int8",
    "action_values": [0, 1],
    "behavior_action_by_episode_mod16": [
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
    ],
    "claim_boundary": "synthetic_cpu_forward_multistep_td_harness_only",
    "discount": 1.0,
    "episode_counts": EPISODE_COUNTS,
    "episodes_per_regime": EPISODES_PER_REGIME,
    "event_order": list(EVENT_ORDER),
    "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
    "generator_regimes": list(REGIMES),
    "horizon": HORIZON,
    "learning_rate": 1.0,
    "observation_dtype": "float64",
    "observation_fields": list(OBSERVATION_FIELDS),
    "observation_shape": [4],
    "policy_input_fields": list(POLICY_INPUT_FIELDS),
    "random_baseline_seed": RANDOM_BASELINE_SEED,
    "regime_counts": REGIME_COUNTS,
    "reward_dtype": "float64",
    "reward_origin_permutation": list(REWARD_ORIGIN_PERMUTATION),
    "reward_values": [0.0, 1.0],
    "structure_kind": STRUCTURE_KIND,
    "thresholds": THRESHOLDS,
    "transition_target_permutation": list(TRANSITION_TARGET_PERMUTATION),
}


@dataclass(frozen=True)
class SplitData:
    name: str
    observations0: np.ndarray
    targets: np.ndarray
    regime_codes: np.ndarray
    episode_indices: np.ndarray
    episode_keys: tuple[str, ...]


@dataclass(frozen=True)
class PendingStep:
    observation_sha256: str
    state_coords: tuple[int, int, int]
    action: int
    phase: int


@dataclass
class LearnerState:
    q: np.ndarray
    pending: PendingStep | None
    expected_phase: int
    completed_episodes: int
    bootstrap_updates: int
    terminal_updates: int
    total_updates: int
    chosen_cell_checks: int


@dataclass(frozen=True)
class UpdateAudit:
    phase: int
    target_value: float
    successor_value: float | None
    changed_cells: int
    chosen_cell_only: bool


@dataclass(frozen=True)
class ActionRecord:
    step_key: str
    observation_sha256: str
    action: int


@dataclass(frozen=True)
class TransitionRecord:
    step_key: str
    next_step_key: str
    next_observation_sha256: str
    bootstrap_observation_sha256: str
    donor_episode: int | None
    done: bool


@dataclass(frozen=True)
class FeedbackRecord:
    step_key: str
    reward: float
    update_reward: float
    origin_episode: int | None
    done: bool


@dataclass(frozen=True)
class TraceBundle:
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
    return hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()


def _immutable_observation(values: list[float] | np.ndarray) -> np.ndarray:
    observation = np.ascontiguousarray(values, dtype=np.float64)
    if observation.shape != (4,) or not np.isfinite(observation).all():
        raise RuntimeError("invalid generated observation")
    observation.setflags(write=False)
    return observation


def _target_for(split: str, code: int, episode: int) -> int:
    if split == "train":
        return 0 if episode % 16 < 8 else 1
    return int(((7 * episode + code) % 32) >= 16)


def _initial_formula(regime: dict[str, object], episode: int) -> np.ndarray:
    split = str(regime["split"])
    code = int(regime["code"])
    target = _target_for(split, code, episode)
    sign = -1.0 if target == 0 else 1.0
    magnitude = float(regime["signal_scale"]) * (1.0 + 0.025 * (episode % 7))
    nuisance = float(regime["nuisance_shift"]) + float(
        regime["nuisance_scale"]
    ) * ((((11 * episode + code) % 29) - 14) / 14.0)
    return _immutable_observation([0.0, sign * magnitude, 1.0, nuisance])


def _independent_initial_formula(
    regime: dict[str, object], episode: int
) -> tuple[np.ndarray, int]:
    split = str(regime["split"])
    code = int(regime["code"])
    target = (
        int((episode % 16) >= 8)
        if split == "train"
        else int(((code + 7 * episode) % 32) >= 16)
    )
    signed = (-1.0 if target == 0 else 1.0) * float(
        regime["signal_scale"]
    ) * (1.0 + 0.025 * (episode % 7))
    nuisance_numerator = (11 * episode + code) % 29 - 14
    nuisance = float(regime["nuisance_shift"]) + float(
        regime["nuisance_scale"]
    ) * (nuisance_numerator / 14.0)
    return _immutable_observation([0.0, signed, 1.0, nuisance]), target


def _generate_split(split_name: str) -> SplitData:
    observations: list[np.ndarray] = []
    targets: list[int] = []
    codes: list[int] = []
    indices: list[int] = []
    keys: list[str] = []
    for regime in REGIMES:
        if regime["split"] != split_name:
            continue
        code = int(regime["code"])
        for episode in range(EPISODES_PER_REGIME):
            observations.append(_initial_formula(regime, episode))
            targets.append(_target_for(split_name, code, episode))
            codes.append(code)
            indices.append(episode)
            keys.append(f"{split_name}:{code}:{episode}")
    observations_array = np.ascontiguousarray(observations, dtype=np.float64)
    targets_array = np.ascontiguousarray(targets, dtype=np.int8)
    codes_array = np.ascontiguousarray(codes, dtype=np.int32)
    indices_array = np.ascontiguousarray(indices, dtype=np.int32)
    for array in (
        observations_array,
        targets_array,
        codes_array,
        indices_array,
    ):
        array.setflags(write=False)
    return SplitData(
        name=split_name,
        observations0=observations_array,
        targets=targets_array,
        regime_codes=codes_array,
        episode_indices=indices_array,
        episode_keys=tuple(keys),
    )


def _generate_all() -> dict[str, SplitData]:
    return {name: _generate_split(name) for name in ("train", "validation", "test")}


def _split_projection(split: SplitData) -> dict[str, object]:
    return {
        "name": split.name,
        "observations0": _array_identity(split.observations0),
        "targets": _array_identity(split.targets),
        "regime_codes": _array_identity(split.regime_codes),
        "episode_indices": _array_identity(split.episode_indices),
        "episode_keys_sha256": _json_sha256(list(split.episode_keys)),
    }


def _dataset_commitment(splits: dict[str, SplitData]) -> str:
    return _json_sha256(
        {name: _split_projection(splits[name]) for name in sorted(splits)}
    )


def _metadata_commitment(split: SplitData) -> str:
    return _json_sha256(
        {
            "name": split.name,
            "targets": _array_identity(split.targets),
            "regime_codes": _array_identity(split.regime_codes),
            "episode_indices": _array_identity(split.episode_indices),
            "episode_keys_sha256": _json_sha256(list(split.episode_keys)),
        }
    )


def _validate_observation(observation: object, *, allow_zero_signal: bool) -> np.ndarray:
    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a NumPy array")
    if (
        observation.dtype != np.float64
        or observation.shape != (4,)
        or not observation.flags.c_contiguous
        or observation.flags.writeable
        or not np.isfinite(observation).all()
    ):
        raise ValueError("observation violates the immutable float64[4] contract")
    phase = float(observation[0])
    signal = float(observation[1])
    alive = float(observation[2])
    if phase not in (0.0, 1.0, 2.0, 3.0) or alive not in (0.0, 1.0):
        raise ValueError("observation phase or alive field is invalid")
    if phase == 0.0 and alive != 1.0:
        raise ValueError("phase-zero observation must be alive")
    if signal == 0.0 and not allow_zero_signal:
        raise ValueError("canonical signal must be nonzero")
    return observation


def _validate_action(action: object) -> np.ndarray:
    if not isinstance(action, np.ndarray):
        raise TypeError("action must be a NumPy scalar array")
    if action.dtype != np.int8 or action.shape != ():
        raise ValueError("action must be scalar int8")
    if int(action) not in (0, 1):
        raise ValueError("action must be zero or one")
    return action


def _validate_reward(reward: object) -> np.ndarray:
    if not isinstance(reward, np.ndarray):
        raise TypeError("feedback reward must be a NumPy scalar array")
    if reward.dtype != np.float64 or reward.shape != ():
        raise ValueError("feedback reward must be scalar float64")
    if float(reward) not in (0.0, 1.0):
        raise ValueError("feedback reward must be zero or one")
    return reward


def _policy_observation(record: object) -> np.ndarray:
    if not isinstance(record, dict) or set(record) != {"observation"}:
        raise RuntimeError("policy input must contain only observation")
    return _validate_observation(record["observation"], allow_zero_signal=True)


def _state_coords(observation: np.ndarray) -> tuple[int, int, int]:
    phase = int(round(float(observation[0])))
    sign_bin = 0 if float(observation[1]) <= 0.0 else 1
    alive = int(round(float(observation[2])))
    return phase, sign_bin, alive


def _new_state() -> LearnerState:
    return LearnerState(
        q=np.zeros((4, 2, 2, 2), dtype=np.float64),
        pending=None,
        expected_phase=0,
        completed_episodes=0,
        bootstrap_updates=0,
        terminal_updates=0,
        total_updates=0,
        chosen_cell_checks=0,
    )


def _state_commitment(state: LearnerState) -> str:
    return _json_sha256(
        {
            "q": _array_identity(state.q),
            "pending": (
                None
                if state.pending is None
                else {
                    "observation_sha256": state.pending.observation_sha256,
                    "state_coords": list(state.pending.state_coords),
                    "action": state.pending.action,
                    "phase": state.pending.phase,
                }
            ),
            "expected_phase": state.expected_phase,
            "completed_episodes": state.completed_episodes,
            "bootstrap_updates": state.bootstrap_updates,
            "terminal_updates": state.terminal_updates,
            "total_updates": state.total_updates,
            "chosen_cell_checks": state.chosen_cell_checks,
        }
    )


def _behavior_action(completed_episodes: int) -> int:
    local = completed_episodes % 16
    return 0 if local < 4 or 8 <= local < 12 else 1


def _select_training(state: LearnerState, record: object) -> np.ndarray:
    observation = _policy_observation(record)
    phase = int(round(float(observation[0])))
    if state.pending is not None or phase != state.expected_phase:
        raise RuntimeError("training selection is out of phase")
    action = _behavior_action(state.completed_episodes)
    state.pending = PendingStep(
        observation_sha256=_observation_sha256(observation),
        state_coords=_state_coords(observation),
        action=action,
        phase=phase,
    )
    return np.asarray(action, dtype=np.int8)


def _greedy_action(state: LearnerState, record: object) -> np.ndarray:
    observation = _policy_observation(record)
    coords = _state_coords(observation)
    values = state.q[coords]
    action = 1 if float(values[1]) > float(values[0]) else 0
    return np.asarray(action, dtype=np.int8)


def _canonical_feedback(
    observation: np.ndarray, target: int, action: int
) -> tuple[np.ndarray | None, float, bool]:
    phase = int(round(float(observation[0])))
    alive = float(observation[2]) == 1.0
    next_alive = alive and action == target
    if phase == HORIZON - 1:
        return None, float(next_alive), True
    direction = -1.0 if action == 0 else 1.0
    next_observation = _immutable_observation(
        [
            float(phase + 1),
            float(observation[1]),
            1.0 if next_alive else 0.0,
            float(observation[3])
            + 0.125 * direction
            + 0.015625 * (phase + 1),
        ]
    )
    return next_observation, 0.0, False


def _td_update(
    state: LearnerState,
    feedback: object,
    *,
    no_bootstrap: bool = False,
) -> UpdateAudit:
    required = {"observation", "action", "reward", "done", "next_observation"}
    if not isinstance(feedback, dict) or set(feedback) != required:
        raise RuntimeError("TD feedback fields violate the frozen contract")
    if state.pending is None:
        raise RuntimeError("TD update has no pending action")
    observation = _validate_observation(
        feedback["observation"], allow_zero_signal=True
    )
    action = _validate_action(feedback["action"])
    reward = _validate_reward(feedback["reward"])
    done = feedback["done"]
    if type(done) is not bool:
        raise TypeError("done must be a Boolean")
    pending = state.pending
    if (
        _observation_sha256(observation) != pending.observation_sha256
        or int(action) != pending.action
        or _state_coords(observation) != pending.state_coords
        or int(round(float(observation[0]))) != pending.phase
    ):
        raise RuntimeError("TD update disagrees with the pending selection")
    terminal_expected = pending.phase == HORIZON - 1
    if done != terminal_expected:
        raise RuntimeError("done value disagrees with the phase")
    successor_value: float | None
    if done:
        if feedback["next_observation"] is not None:
            raise RuntimeError("terminal update forbids a successor")
        successor_value = None
        target_value = float(reward)
    else:
        successor = _validate_observation(
            feedback["next_observation"], allow_zero_signal=True
        )
        if int(round(float(successor[0]))) != pending.phase + 1:
            raise RuntimeError("nonterminal successor has the wrong phase")
        successor_values = state.q[_state_coords(successor)]
        successor_value = float(np.max(successor_values))
        target_value = float(reward) + (0.0 if no_bootstrap else successor_value)
    before = state.q.copy()
    coords = (*pending.state_coords, pending.action)
    state.q[coords] = target_value
    changed = np.argwhere(before != state.q)
    chosen_cell_only = all(tuple(index) == coords for index in changed)
    state.chosen_cell_checks += 1
    state.total_updates += 1
    if done:
        state.terminal_updates += 1
        state.completed_episodes += 1
        state.expected_phase = 0
    else:
        state.bootstrap_updates += 1
        state.expected_phase += 1
    state.pending = None
    return UpdateAudit(
        phase=pending.phase,
        target_value=target_value,
        successor_value=successor_value,
        changed_cells=len(changed),
        chosen_cell_only=chosen_cell_only,
    )


def _ablate_observation(observation: np.ndarray) -> np.ndarray:
    values = np.array(observation, dtype=np.float64, copy=True)
    values[1] = 0.0
    return _immutable_observation(values)


def _step_key(episode_key: str, phase: int) -> str:
    return f"{episode_key}:{phase}"


def _behavior_observation_at_phase(
    split: SplitData, absolute_index: int, phase: int, *, ablate_signal: bool = False
) -> np.ndarray:
    observation = split.observations0[absolute_index]
    if ablate_signal:
        observation = _ablate_observation(observation)
    target = int(split.targets[absolute_index])
    action = _behavior_action(absolute_index)
    for _ in range(phase):
        successor, _, done = _canonical_feedback(observation, target, action)
        if done or successor is None:
            raise RuntimeError("requested observation after terminal phase")
        observation = successor
    return observation


def _transition_donor_absolute(split: SplitData, absolute_index: int) -> int:
    local = int(split.episode_indices[absolute_index])
    regime_start = absolute_index - local
    block_start = (local // 16) * 16
    donor_local = block_start + TRANSITION_TARGET_PERMUTATION[local % 16]
    return regime_start + donor_local


def _canonical_behavior_rewards(split: SplitData) -> np.ndarray:
    values: list[float] = []
    for index in range(len(split.episode_keys)):
        observation = split.observations0[index]
        target = int(split.targets[index])
        action = _behavior_action(index)
        terminal = 0.0
        for _ in range(HORIZON):
            successor, terminal, done = _canonical_feedback(
                observation, target, action
            )
            if done:
                break
            if successor is None:
                raise AssertionError("missing nonterminal successor")
            observation = successor
        values.append(terminal)
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _reward_origin_absolute(split: SplitData, absolute_index: int) -> int:
    local = int(split.episode_indices[absolute_index])
    regime_start = absolute_index - local
    return regime_start + REWARD_ORIGIN_PERMUTATION[local]


def _train_policy(
    split: SplitData,
    *,
    expected_metadata_sha256: str,
    treatment: str = "canonical",
    no_bootstrap: bool = False,
) -> tuple[
    LearnerState,
    TraceBundle,
    tuple[str, ...],
    tuple[UpdateAudit, ...],
    tuple[tuple[int, ...], tuple[int, ...]],
]:
    if split.name != "train" or _metadata_commitment(split) != expected_metadata_sha256:
        raise RuntimeError("training is bound to the authenticated train split")
    if treatment not in {
        "canonical",
        "transition_target",
        "reward_origin",
        "signal_ablation",
    }:
        raise RuntimeError("unknown training treatment")
    state = _new_state()
    actions: list[ActionRecord] = []
    transitions: list[TransitionRecord] = []
    feedback_records: list[FeedbackRecord] = []
    audits: list[UpdateAudit] = []
    events: list[str] = []
    propagation: dict[int, list[int | None]] = {
        0: [None] * HORIZON,
        1: [None] * HORIZON,
    }
    canonical_rewards = _canonical_behavior_rewards(split)
    for absolute_index, episode_key in enumerate(split.episode_keys):
        local_episode = int(split.episode_indices[absolute_index])
        target = int(split.targets[absolute_index])
        observation = split.observations0[absolute_index]
        if treatment == "signal_ablation":
            observation = _ablate_observation(observation)
        for phase in range(HORIZON):
            events.extend((f"observe{phase}", f"select{phase}"))
            action_array = _select_training(state, {"observation": observation})
            action = int(_validate_action(action_array))
            step_key = _step_key(episode_key, phase)
            actions.append(
                ActionRecord(step_key, _observation_sha256(observation), action)
            )
            actual_successor, canonical_reward, done = _canonical_feedback(
                observation, target, action
            )
            update_reward = canonical_reward
            origin_episode: int | None = None
            bootstrap_successor = actual_successor
            donor_episode: int | None = None
            if not done and treatment == "transition_target":
                donor_absolute = _transition_donor_absolute(split, absolute_index)
                bootstrap_successor = _behavior_observation_at_phase(
                    split,
                    donor_absolute,
                    phase + 1,
                    ablate_signal=False,
                )
                donor_episode = int(split.episode_indices[donor_absolute])
            if done and treatment == "reward_origin":
                origin_absolute = _reward_origin_absolute(split, absolute_index)
                update_reward = float(canonical_rewards[origin_absolute])
                origin_episode = int(split.episode_indices[origin_absolute])
            if done:
                events.append("terminal_reward")
            else:
                events.extend((f"transition{phase}", f"feedback{phase}"))
            audit = _td_update(
                state,
                {
                    "observation": observation,
                    "action": action_array,
                    "reward": np.asarray(update_reward, dtype=np.float64),
                    "done": done,
                    "next_observation": bootstrap_successor,
                },
                no_bootstrap=no_bootstrap,
            )
            audits.append(audit)
            events.extend((f"td_target{phase}", f"td_update{phase}", f"log{phase}"))
            if done:
                events.append("close_episode")
            feedback_records.append(
                FeedbackRecord(
                    step_key=step_key,
                    reward=canonical_reward,
                    update_reward=update_reward,
                    origin_episode=origin_episode,
                    done=done,
                )
            )
            if actual_successor is not None:
                transitions.append(
                    TransitionRecord(
                        step_key=step_key,
                        next_step_key=_step_key(episode_key, phase + 1),
                        next_observation_sha256=_observation_sha256(
                            actual_successor
                        ),
                        bootstrap_observation_sha256=_observation_sha256(
                            bootstrap_successor
                        ),
                        donor_episode=donor_episode,
                        done=False,
                    )
                )
                observation = actual_successor
            if (
                absolute_index < EPISODES_PER_REGIME
                and action == target
                and float(observation[2]) == 1.0
                and state.q[phase, target, 1, target] > 0.0
                and propagation[target][phase] is None
            ):
                propagation[target][phase] = local_episode
        if state.pending is not None or state.expected_phase != 0:
            raise RuntimeError("episode did not close cleanly")
    if (
        treatment == "canonical"
        and not no_bootstrap
        and any(value is None for values in propagation.values() for value in values)
    ):
        raise RuntimeError("canonical propagation signature was incomplete")
    return (
        state,
        TraceBundle(tuple(actions), tuple(transitions), tuple(feedback_records)),
        tuple(events),
        tuple(audits),
        (
            tuple(-1 if value is None else int(value) for value in propagation[0]),
            tuple(-1 if value is None else int(value) for value in propagation[1]),
        ),
    )


def _evaluate_frozen(
    state: LearnerState,
    split: SplitData,
    *,
    ablate_signal: bool = False,
) -> TraceBundle:
    state_before = _state_commitment(state)
    actions: list[ActionRecord] = []
    transitions: list[TransitionRecord] = []
    feedback_records: list[FeedbackRecord] = []
    for absolute_index, episode_key in enumerate(split.episode_keys):
        target = int(split.targets[absolute_index])
        observation = split.observations0[absolute_index]
        if ablate_signal:
            observation = _ablate_observation(observation)
        for phase in range(HORIZON):
            action = int(
                _validate_action(_greedy_action(state, {"observation": observation}))
            )
            step_key = _step_key(episode_key, phase)
            actions.append(
                ActionRecord(step_key, _observation_sha256(observation), action)
            )
            successor, reward, done = _canonical_feedback(
                observation, target, action
            )
            feedback_records.append(
                FeedbackRecord(step_key, reward, reward, None, done)
            )
            if successor is not None:
                transitions.append(
                    TransitionRecord(
                        step_key,
                        _step_key(episode_key, phase + 1),
                        _observation_sha256(successor),
                        _observation_sha256(successor),
                        None,
                        False,
                    )
                )
                observation = successor
    if _state_commitment(state) != state_before:
        raise RuntimeError("held-out evaluation mutated learner state")
    return TraceBundle(tuple(actions), tuple(transitions), tuple(feedback_records))


def _trace_from_action_matrix(split: SplitData, matrix: np.ndarray) -> TraceBundle:
    if matrix.dtype != np.int8 or matrix.shape != (
        len(split.episode_keys),
        HORIZON,
    ):
        raise RuntimeError("baseline action matrix violates the frozen shape")
    actions: list[ActionRecord] = []
    transitions: list[TransitionRecord] = []
    feedback_records: list[FeedbackRecord] = []
    for absolute_index, episode_key in enumerate(split.episode_keys):
        target = int(split.targets[absolute_index])
        observation = split.observations0[absolute_index]
        for phase in range(HORIZON):
            action = int(matrix[absolute_index, phase])
            if action not in (0, 1):
                raise RuntimeError("baseline action is out of range")
            step_key = _step_key(episode_key, phase)
            actions.append(
                ActionRecord(step_key, _observation_sha256(observation), action)
            )
            successor, reward, done = _canonical_feedback(
                observation, target, action
            )
            feedback_records.append(
                FeedbackRecord(step_key, reward, reward, None, done)
            )
            if successor is not None:
                transitions.append(
                    TransitionRecord(
                        step_key,
                        _step_key(episode_key, phase + 1),
                        _observation_sha256(successor),
                        _observation_sha256(successor),
                        None,
                        False,
                    )
                )
                observation = successor
    return TraceBundle(tuple(actions), tuple(transitions), tuple(feedback_records))


def _unique_by_key(records: tuple[object, ...], attribute: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for record in records:
        key = getattr(record, attribute)
        if not isinstance(key, str) or key in result:
            raise RuntimeError("trace contains a duplicate or malformed key")
        result[key] = record
    return result


def _score_trace(
    trace: TraceBundle,
    split: SplitData,
    *,
    ablate_signal: bool = False,
) -> tuple[float, float, dict[int, float]]:
    action_by_key = _unique_by_key(trace.actions, "step_key")
    transition_by_key = _unique_by_key(trace.transitions, "step_key")
    feedback_by_key = _unique_by_key(trace.feedback, "step_key")
    expected_step_count = len(split.episode_keys) * HORIZON
    if (
        len(action_by_key) != expected_step_count
        or len(feedback_by_key) != expected_step_count
        or len(transition_by_key)
        != len(split.episode_keys) * (HORIZON - 1)
    ):
        raise RuntimeError("trace has a missing or extra component")
    expected_step_keys = {
        _step_key(episode_key, phase)
        for episode_key in split.episode_keys
        for phase in range(HORIZON)
    }
    if set(action_by_key) != expected_step_keys or set(feedback_by_key) != expected_step_keys:
        raise RuntimeError("trace step keys do not match the generator")
    expected_transition_keys = {
        _step_key(episode_key, phase)
        for episode_key in split.episode_keys
        for phase in range(HORIZON - 1)
    }
    if set(transition_by_key) != expected_transition_keys:
        raise RuntimeError("trace transition keys do not match the generator")

    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for absolute_index, episode_key in enumerate(split.episode_keys):
        target = int(split.targets[absolute_index])
        code = int(split.regime_codes[absolute_index])
        observation = split.observations0[absolute_index]
        if ablate_signal:
            observation = _ablate_observation(observation)
        terminal_reward = 0.0
        for phase in range(HORIZON):
            step_key = _step_key(episode_key, phase)
            action_record = action_by_key[step_key]
            feedback_record = feedback_by_key[step_key]
            if not isinstance(action_record, ActionRecord) or not isinstance(
                feedback_record, FeedbackRecord
            ):
                raise RuntimeError("trace component has the wrong type")
            if action_record.observation_sha256 != _observation_sha256(observation):
                raise RuntimeError("trace observation commitment is wrong")
            if type(action_record.action) is not int or action_record.action not in (0, 1):
                raise RuntimeError("trace action is malformed")
            successor, reward, done = _canonical_feedback(
                observation, target, action_record.action
            )
            if feedback_record.reward != reward or feedback_record.done != done:
                raise RuntimeError("trace reward or done value is wrong")
            if phase < HORIZON - 1:
                transition_record = transition_by_key[step_key]
                if not isinstance(transition_record, TransitionRecord):
                    raise RuntimeError("trace transition has the wrong type")
                if (
                    transition_record.next_step_key
                    != _step_key(episode_key, phase + 1)
                    or successor is None
                    or transition_record.next_observation_sha256
                    != _observation_sha256(successor)
                    or transition_record.done
                ):
                    raise RuntimeError("trace transition is not canonical")
                observation = successor
            else:
                if successor is not None:
                    raise AssertionError("terminal transition unexpectedly exists")
                terminal_reward = reward
        totals[code] = totals.get(code, 0.0) + terminal_reward
        counts[code] = counts.get(code, 0) + 1
    per_regime = {code: totals[code] / counts[code] for code in sorted(totals)}
    macro = float(np.mean(list(per_regime.values())))
    minimum = float(min(per_regime.values()))
    return macro, minimum, per_regime


def _trace_projection(trace: TraceBundle) -> dict[str, object]:
    return {
        "action_sha256": _json_sha256(
            [
                [record.step_key, record.observation_sha256, record.action]
                for record in trace.actions
            ]
        ),
        "transition_sha256": _json_sha256(
            [
                [
                    record.step_key,
                    record.next_step_key,
                    record.next_observation_sha256,
                    record.bootstrap_observation_sha256,
                    record.donor_episode,
                    record.done,
                ]
                for record in trace.transitions
            ]
        ),
        "feedback_sha256": _json_sha256(
            [
                [
                    record.step_key,
                    record.reward,
                    record.update_reward,
                    record.origin_episode,
                    record.done,
                ]
                for record in trace.feedback
            ]
        ),
    }


def _trace_sha256(trace: TraceBundle) -> str:
    return _json_sha256(_trace_projection(trace))


def _independently_reordered(trace: TraceBundle) -> TraceBundle:
    return TraceBundle(
        actions=tuple(reversed(trace.actions)),
        transitions=tuple(reversed(trace.transitions)),
        feedback=tuple(reversed(trace.feedback)),
    )


def _event_order_exact(events: tuple[str, ...]) -> bool:
    if len(events) % len(EVENT_ORDER) != 0:
        return False
    return all(
        tuple(events[start : start + len(EVENT_ORDER)]) == EVENT_ORDER
        for start in range(0, len(events), len(EVENT_ORDER))
    )


def _random_baseline_matrices(
    splits: dict[str, SplitData]
) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(RANDOM_BASELINE_SEED))
    result: dict[str, np.ndarray] = {}
    for name in ("validation", "test"):
        matrix = rng.integers(
            0,
            2,
            size=(len(splits[name].episode_keys), HORIZON),
            dtype=np.int8,
        )
        matrix.setflags(write=False)
        result[name] = matrix
    return result


def _fit_myopic(split: SplitData) -> LearnerState:
    state = _new_state()
    for absolute_index in range(len(split.episode_keys)):
        observation = _behavior_observation_at_phase(
            split, absolute_index, HORIZON - 1
        )
        action = _behavior_action(absolute_index)
        target = int(split.targets[absolute_index])
        _, reward, done = _canonical_feedback(observation, target, action)
        if not done:
            raise AssertionError("myopic terminal fixture was nonterminal")
        state.q[(*_state_coords(observation), action)] = reward
    return state


def _score_attacks(trace: TraceBundle, split: SplitData) -> tuple[int, int]:
    attacks: list[TraceBundle] = []
    first_action = trace.actions[0]
    first_transition = trace.transitions[0]
    first_feedback = trace.feedback[0]
    terminal_index = HORIZON - 1
    terminal_feedback = trace.feedback[terminal_index]
    attacks.append(replace(trace, actions=trace.actions + (first_action,)))
    attacks.append(replace(trace, actions=trace.actions[1:]))
    attacks.append(
        replace(
            trace,
            actions=(
                replace(first_action, observation_sha256="0" * 64),
                *trace.actions[1:],
            ),
        )
    )
    attacks.append(
        replace(
            trace,
            actions=(replace(first_action, action=1 - first_action.action), *trace.actions[1:]),
        )
    )
    attacks.append(replace(trace, transitions=trace.transitions[1:]))
    attacks.append(
        replace(
            trace,
            transitions=(
                replace(first_transition, next_observation_sha256="1" * 64),
                *trace.transitions[1:],
            ),
        )
    )
    attacks.append(
        replace(
            trace,
            transitions=(
                replace(first_transition, next_step_key="wrong"),
                *trace.transitions[1:],
            ),
        )
    )
    attacks.append(replace(trace, feedback=trace.feedback[1:]))
    attacks.append(
        replace(
            trace,
            feedback=(
                replace(first_feedback, reward=1.0),
                *trace.feedback[1:],
            ),
        )
    )
    attacks.append(
        replace(
            trace,
            feedback=(
                *trace.feedback[:terminal_index],
                replace(terminal_feedback, done=False),
                *trace.feedback[terminal_index + 1 :],
            ),
        )
    )
    swapped = list(trace.actions)
    left = swapped[0]
    right = swapped[HORIZON]
    swapped[0] = replace(
        left,
        observation_sha256=right.observation_sha256,
        action=right.action,
    )
    swapped[HORIZON] = replace(
        right,
        observation_sha256=left.observation_sha256,
        action=left.action,
    )
    attacks.append(replace(trace, actions=tuple(swapped)))
    rejected = 0
    for attack in attacks:
        try:
            _score_trace(attack, split)
        except RuntimeError:
            rejected += 1
    return len(attacks), rejected


def _forbidden_policy_rejections(observation: np.ndarray) -> int:
    rejected = 0
    for name in FORBIDDEN_POLICY_FIELDS:
        try:
            _policy_observation({"observation": observation, name: object()})
        except RuntimeError:
            rejected += 1
    return rejected


def _action_type_rejections() -> tuple[int, int]:
    invalid: tuple[object, ...] = (
        True,
        0,
        0.0,
        np.int8(0),
        np.asarray([0], dtype=np.int8),
        np.asarray(0, dtype=np.int64),
        np.asarray(0.0, dtype=np.float64),
        np.asarray(2, dtype=np.int8),
    )
    rejected = 0
    for value in invalid:
        try:
            _validate_action(value)
        except (TypeError, ValueError):
            rejected += 1
    return len(invalid), rejected


def _update_sentinel_rejections(observation0: np.ndarray) -> tuple[int, int]:
    checked = 0
    rejected = 0

    def expect_rejection(callback: Callable[[], object]) -> None:
        nonlocal checked, rejected
        checked += 1
        try:
            callback()
        except (RuntimeError, TypeError, ValueError):
            rejected += 1

    valid_action = np.asarray(0, dtype=np.int8)
    successor, _, _ = _canonical_feedback(observation0, 0, 0)
    if successor is None:
        raise AssertionError("sentinel successor is missing")
    feedback = {
        "observation": observation0,
        "action": valid_action,
        "reward": np.asarray(0.0, dtype=np.float64),
        "done": False,
        "next_observation": successor,
    }
    expect_rejection(lambda: _td_update(_new_state(), feedback))

    state = _new_state()
    _select_training(state, {"observation": observation0})
    expect_rejection(
        lambda: _select_training(state, {"observation": observation0})
    )

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    missing_successor = dict(feedback, action=action, next_observation=None)
    expect_rejection(lambda: _td_update(state, missing_successor))

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    expect_rejection(
        lambda: _td_update(state, dict(feedback, action=action, done=True))
    )

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    wrong_phase = _immutable_observation(
        [2.0, float(successor[1]), float(successor[2]), float(successor[3])]
    )
    expect_rejection(
        lambda: _td_update(
            state, dict(feedback, action=action, next_observation=wrong_phase)
        )
    )

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    wrong_action = np.asarray(1 - int(action), dtype=np.int8)
    expect_rejection(
        lambda: _td_update(state, dict(feedback, action=wrong_action))
    )

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    mutable = np.array(observation0, copy=True)
    expect_rejection(
        lambda: _td_update(state, dict(feedback, action=action, observation=mutable))
    )

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    extra = dict(feedback, action=action, terminal_reward=np.asarray(1.0))
    expect_rejection(lambda: _td_update(state, extra))

    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    _td_update(state, dict(feedback, action=action))
    expect_rejection(lambda: _td_update(state, dict(feedback, action=action)))
    return checked, rejected


def _target_dependency_case(split: SplitData) -> dict[str, object]:
    observation0 = split.observations0[0]
    action_values = np.asarray(0, dtype=np.int8)
    successor, _, _ = _canonical_feedback(observation0, 0, 0)
    if successor is None:
        raise AssertionError("dependency successor is missing")

    def run_with_successor_max(value: float) -> tuple[float, LearnerState]:
        state = _new_state()
        state.q[(*_state_coords(successor), 1)] = value
        action = _select_training(state, {"observation": observation0})
        audit = _td_update(
            state,
            {
                "observation": observation0,
                "action": action,
                "reward": np.asarray(0.0, dtype=np.float64),
                "done": False,
                "next_observation": successor,
            },
        )
        return audit.target_value, state

    low, low_state = run_with_successor_max(0.25)
    high, high_state = run_with_successor_max(0.75)
    direct_terminal_rejected = False
    state = _new_state()
    action = _select_training(state, {"observation": observation0})
    try:
        _td_update(
            state,
            {
                "observation": observation0,
                "action": action,
                "reward": np.asarray(0.0, dtype=np.float64),
                "done": False,
                "next_observation": successor,
                "terminal_reward": np.asarray(1.0, dtype=np.float64),
            },
        )
    except RuntimeError:
        direct_terminal_rejected = True

    first_state = _new_state()
    observation = observation0
    positive_phases: list[int] = []
    for phase in range(HORIZON):
        action = _select_training(first_state, {"observation": observation})
        next_observation, reward, done = _canonical_feedback(
            observation, 0, int(action)
        )
        _td_update(
            first_state,
            {
                "observation": observation,
                "action": action,
                "reward": np.asarray(reward, dtype=np.float64),
                "done": done,
                "next_observation": next_observation,
            },
        )
        if first_state.q[phase, 0, 1, 0] > 0.0:
            positive_phases.append(phase)
        if next_observation is not None:
            observation = next_observation
    passed = (
        low == 0.25
        and high == 0.75
        and high - low == 0.5
        and direct_terminal_rejected
        and positive_phases == [3]
    )
    return {
        "direct_terminal_field_rejected": direct_terminal_rejected,
        "first_episode_positive_phases": positive_phases,
        "passed": bool(passed),
        "successor_delta_exact": high - low == 0.5,
        "successor_values_checked": [low, high],
        "trace_sha256": _json_sha256(
            {
                "low_state_sha256": _state_commitment(low_state),
                "high_state_sha256": _state_commitment(high_state),
                "first_state_sha256": _state_commitment(first_state),
            }
        ),
    }


def _generator_checks(
    splits: dict[str, SplitData]
) -> tuple[int, bool, bool, bool, bool, bool, int]:
    validated = 0
    exact_regime_order = True
    exact_key_order = True
    exact_target_order = True
    balanced_targets = True
    behavior_cells_balanced = True
    all_keys: dict[str, set[str]] = {}
    observation_hashes: dict[str, set[str]] = {}
    for split_name, split in splits.items():
        expected_codes = [
            int(regime["code"])
            for regime in REGIMES
            if regime["split"] == split_name
        ]
        actual_codes = [
            int(split.regime_codes[index])
            for index in range(0, len(split.episode_keys), EPISODES_PER_REGIME)
        ]
        exact_regime_order &= actual_codes == expected_codes
        all_keys[split_name] = set(split.episode_keys)
        observation_hashes[split_name] = {
            _observation_sha256(row) for row in split.observations0
        }
        exact_key_order &= len(all_keys[split_name]) == len(split.episode_keys)
        for regime_index, regime in enumerate(
            [item for item in REGIMES if item["split"] == split_name]
        ):
            start = regime_index * EPISODES_PER_REGIME
            stop = start + EPISODES_PER_REGIME
            targets = split.targets[start:stop]
            balanced_targets &= int(np.sum(targets == 0)) == int(
                np.sum(targets == 1)
            ) == EPISODES_PER_REGIME // 2
            for episode in range(EPISODES_PER_REGIME):
                index = start + episode
                expected_observation, expected_target = _independent_initial_formula(
                    regime, episode
                )
                exact_target_order &= (
                    int(split.targets[index]) == expected_target
                    and np.array_equal(split.observations0[index], expected_observation)
                    and split.episode_keys[index]
                    == f"{split_name}:{int(regime['code'])}:{episode}"
                )
                validated += 1
            if split_name == "train":
                for block_start in (start, start + 16):
                    counts = {
                        (target, action): 0
                        for target in (0, 1)
                        for action in (0, 1)
                    }
                    for index in range(block_start, block_start + 16):
                        counts[
                            (
                                int(split.targets[index]),
                                _behavior_action(index),
                            )
                        ] += 1
                    behavior_cells_balanced &= set(counts.values()) == {4}
    keys_disjoint = all(
        all_keys[left].isdisjoint(all_keys[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    overlap = sum(
        len(observation_hashes[left] & observation_hashes[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    return (
        validated,
        exact_regime_order,
        exact_key_order and keys_disjoint,
        exact_target_order,
        balanced_targets,
        behavior_cells_balanced,
        overlap,
    )


def _action_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [[record.step_key, record.action] for record in trace.actions]
    )


def _actual_transition_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            [
                record.step_key,
                record.next_step_key,
                record.next_observation_sha256,
                record.done,
            ]
            for record in trace.transitions
        ]
    )


def _canonical_feedback_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            [record.step_key, record.reward, record.done]
            for record in trace.feedback
        ]
    )


def _transition_control_mapping(
    trace: TraceBundle, split: SplitData
) -> tuple[bool, int, bool]:
    transitions = _unique_by_key(trace.transitions, "step_key")
    exact = True
    changed = 0
    multiset_exact = True
    for regime_start in range(0, len(split.episode_keys), EPISODES_PER_REGIME):
        for block_offset in (0, 16):
            block_start = regime_start + block_offset
            for phase in range(HORIZON - 1):
                actual_multiset: list[str] = []
                bootstrap_multiset: list[str] = []
                for within in range(16):
                    source = block_start + within
                    key = _step_key(split.episode_keys[source], phase)
                    record = transitions[key]
                    if not isinstance(record, TransitionRecord):
                        exact = False
                        continue
                    donor = _transition_donor_absolute(split, source)
                    expected = _behavior_observation_at_phase(
                        split, donor, phase + 1
                    )
                    expected_sha = _observation_sha256(expected)
                    exact &= (
                        record.bootstrap_observation_sha256 == expected_sha
                        and record.donor_episode
                        == int(split.episode_indices[donor])
                    )
                    changed += int(
                        record.bootstrap_observation_sha256
                        != record.next_observation_sha256
                    )
                    actual_multiset.append(record.next_observation_sha256)
                    bootstrap_multiset.append(record.bootstrap_observation_sha256)
                multiset_exact &= sorted(actual_multiset) == sorted(bootstrap_multiset)
    return exact, changed, multiset_exact


def _reward_control_mapping(
    trace: TraceBundle, split: SplitData
) -> tuple[bool, int, bool, bool]:
    feedback = _unique_by_key(trace.feedback, "step_key")
    canonical_rewards = _canonical_behavior_rewards(split)
    exact = True
    changed = 0
    assigned_terminal: list[float] = []
    canonical_terminal: list[float] = []
    for absolute_index, episode_key in enumerate(split.episode_keys):
        for phase in range(HORIZON):
            key = _step_key(episode_key, phase)
            record = feedback[key]
            if not isinstance(record, FeedbackRecord):
                exact = False
                continue
            if phase < HORIZON - 1:
                exact &= record.update_reward == 0.0 and record.origin_episode is None
                continue
            origin = _reward_origin_absolute(split, absolute_index)
            expected = float(canonical_rewards[origin])
            exact &= (
                record.update_reward == expected
                and record.origin_episode == int(split.episode_indices[origin])
            )
            changed += int(record.update_reward != record.reward)
            assigned_terminal.append(record.update_reward)
            canonical_terminal.append(record.reward)
    multiset_equal = sorted(assigned_terminal) == sorted(canonical_terminal)
    no_fixed_points = all(
        destination != origin
        for destination, origin in enumerate(REWARD_ORIGIN_PERMUTATION)
    ) and sorted(REWARD_ORIGIN_PERMUTATION) == list(range(EPISODES_PER_REGIME))
    return exact, changed, multiset_equal, no_fixed_points


def _positive_gate(
    *,
    behavior_return: float,
    behavior_regret: int,
    postfit_train: float,
    validation: float,
    test: float,
    minimum_heldout: float,
    best_constant_validation: float,
    best_constant_test: float,
    myopic_validation: float,
    myopic_test: float,
    no_bootstrap_validation: float,
    no_bootstrap_test: float,
    random_validation: float,
    random_test: float,
    state: LearnerState,
    propagation: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    return bool(
        behavior_return == 0.5
        and behavior_regret == 64
        and postfit_train >= THRESHOLDS["minimum_postfit_train_macro_return"]
        and validation >= THRESHOLDS["minimum_postfit_validation_macro_return"]
        and test >= THRESHOLDS["minimum_postfit_test_macro_return"]
        and minimum_heldout >= THRESHOLDS["minimum_heldout_regime_return"]
        and validation - best_constant_validation
        >= THRESHOLDS["minimum_validation_gain_baseline"]
        and test - best_constant_test >= THRESHOLDS["minimum_test_gain_baseline"]
        and validation - myopic_validation
        >= THRESHOLDS["minimum_validation_gain_baseline"]
        and test - myopic_test >= THRESHOLDS["minimum_test_gain_baseline"]
        and validation - no_bootstrap_validation
        >= THRESHOLDS["minimum_validation_gain_baseline"]
        and test - no_bootstrap_test >= THRESHOLDS["minimum_test_gain_baseline"]
        and validation - random_validation
        >= THRESHOLDS["minimum_validation_gain_baseline"]
        and test - random_test >= THRESHOLDS["minimum_test_gain_baseline"]
        and state.completed_episodes == EPISODE_COUNTS["train"]
        and state.bootstrap_updates == EPISODE_COUNTS["train"] * (HORIZON - 1)
        and state.terminal_updates == EPISODE_COUNTS["train"]
        and state.total_updates == EPISODE_COUNTS["train"] * HORIZON
        and state.pending is None
        and propagation == ((3, 2, 1, 0), (15, 14, 13, 12))
    )


def _non_process_projection() -> dict[str, object]:
    splits = _generate_all()
    replay = _generate_all()
    dataset_sha256 = _dataset_commitment(splits)
    replay_sha256 = _dataset_commitment(replay)
    train_metadata_sha256 = _metadata_commitment(splits["train"])
    (
        true_state,
        train_trace,
        train_events,
        train_audits,
        propagation,
    ) = _train_policy(
        splits["train"], expected_metadata_sha256=train_metadata_sha256
    )
    (
        replay_state,
        replay_train_trace,
        _,
        _,
        _,
    ) = _train_policy(
        replay["train"],
        expected_metadata_sha256=_metadata_commitment(replay["train"]),
    )
    heldout_state_before = _state_commitment(true_state)
    postfit_train_trace = _evaluate_frozen(true_state, splits["train"])
    validation_trace = _evaluate_frozen(true_state, splits["validation"])
    test_trace = _evaluate_frozen(true_state, splits["test"])
    heldout_state_after = _state_commitment(true_state)

    action_checked, action_rejected = _action_type_rejections()
    immutable_rejected = False
    try:
        splits["train"].observations0[0, 1] = 0.0
    except ValueError:
        immutable_rejected = True
    done_patterns: dict[str, list[bool]] = {}
    for record in train_trace.feedback:
        episode = record.step_key.rsplit(":", 1)[0]
        done_patterns.setdefault(episode, []).append(record.done)
    typed_records = sum(EPISODE_COUNTS.values())
    typed_passed = bool(
        all(split.observations0.dtype == np.float64 for split in splits.values())
        and all(not split.observations0.flags.writeable for split in splits.values())
        and all(pattern == [False, False, False, True] for pattern in done_patterns.values())
        and immutable_rejected
        and action_checked == action_rejected
        and _event_order_exact(train_events)
        and STRUCTURE_KIND == "none"
    )
    typed_case = {
        "action_dtype": "int8",
        "action_sentinels_checked": action_checked,
        "action_sentinels_rejected": action_rejected,
        "action_values": [0, 1],
        "completed_episodes": typed_records,
        "done_pattern_exact": all(
            pattern == [False, False, False, True]
            for pattern in done_patterns.values()
        ),
        "event_order": list(EVENT_ORDER),
        "horizon": HORIZON,
        "immutable_observations": immutable_rejected,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [4],
        "passed": typed_passed,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
        "typed_trajectory_records": typed_records,
    }

    (
        validated_episodes,
        exact_regime_order,
        keys_and_splits_disjoint,
        exact_target_order,
        balanced_targets,
        behavior_cells_balanced,
        observation_overlap,
    ) = _generator_checks(splits)
    within_split_keys_unique = all(
        len(set(split.episode_keys)) == len(split.episode_keys)
        for split in splits.values()
    )
    generator_passed = bool(
        dataset_sha256 == EXPECTED_DATASET_SHA256
        and dataset_sha256 == replay_sha256
        and validated_episodes == sum(EPISODE_COUNTS.values())
        and exact_regime_order
        and keys_and_splits_disjoint
        and exact_target_order
        and balanced_targets
        and behavior_cells_balanced
        and observation_overlap == 0
        and within_split_keys_unique
        and all(np.isfinite(split.observations0).all() for split in splits.values())
    )
    generator_case = {
        "balanced_targets_per_regime": balanced_targets,
        "behavior_cells_balanced": behavior_cells_balanced,
        "dataset_sha256": dataset_sha256,
        "deterministic_replay": dataset_sha256 == replay_sha256,
        "episode_counts": EPISODE_COUNTS,
        "episode_keys_disjoint": keys_and_splits_disjoint,
        "episodes_per_regime": EPISODES_PER_REGIME,
        "exact_key_order": keys_and_splits_disjoint,
        "exact_regime_order": exact_regime_order,
        "exact_target_order": exact_target_order,
        "expected_dataset_commitment": dataset_sha256 == EXPECTED_DATASET_SHA256,
        "finite_observations": bool(
            all(np.isfinite(split.observations0).all() for split in splits.values())
        ),
        "observation_overlap_count": observation_overlap,
        "observations_disjoint": observation_overlap == 0,
        "passed": generator_passed,
        "regime_counts": REGIME_COUNTS,
        "validated_episodes": validated_episodes,
        "within_split_keys_unique": within_split_keys_unique,
    }

    action_changes = 0
    for split in splits.values():
        for observation in split.observations0:
            target = 0 if float(observation[1]) < 0.0 else 1
            successor0, _, _ = _canonical_feedback(observation, target, 0)
            successor1, _, _ = _canonical_feedback(observation, target, 1)
            action_changes += int(
                successor0 is not None
                and successor1 is not None
                and not np.array_equal(successor0, successor1)
            )
    authenticated_transitions = len(train_trace.transitions)
    transition_case = {
        "action_changes_successor_count": action_changes,
        "authenticated_transition_count": authenticated_transitions,
        "done_pattern_exact": all(
            not record.done for record in train_trace.transitions
        ),
        "nonterminal_reward_values": sorted(
            {
                record.reward
                for record in train_trace.feedback
                if not record.done
            }
        ),
        "passed": bool(
            action_changes == sum(EPISODE_COUNTS.values())
            and authenticated_transitions
            == EPISODE_COUNTS["train"] * (HORIZON - 1)
            and all(not record.done for record in train_trace.transitions)
        ),
        "reward_exposed_before_terminal": any(
            record.reward != 0.0
            for record in train_trace.feedback
            if not record.done
        ),
        "trace_sha256": _json_sha256(
            {
                "dataset_sha256": dataset_sha256,
                "train_trace_sha256": _trace_sha256(train_trace),
            }
        ),
    }

    target_dependency_case = _target_dependency_case(splits["train"])
    update_checked, update_rejected = _update_sentinel_rejections(
        splits["train"].observations0[0]
    )
    forward_phase_order = all(
        tuple(audit.phase for audit in train_audits[start : start + HORIZON])
        == (0, 1, 2, 3)
        for start in range(0, len(train_audits), HORIZON)
    )
    update_order_passed = bool(
        _event_order_exact(train_events)
        and forward_phase_order
        and all(audit.chosen_cell_only for audit in train_audits)
        and true_state.completed_episodes == EPISODE_COUNTS["train"]
        and true_state.bootstrap_updates
        == EPISODE_COUNTS["train"] * (HORIZON - 1)
        and true_state.terminal_updates == EPISODE_COUNTS["train"]
        and true_state.total_updates == EPISODE_COUNTS["train"] * HORIZON
        and update_checked == update_rejected
        and propagation == ((3, 2, 1, 0), (15, 14, 13, 12))
        and target_dependency_case["first_episode_positive_phases"] == [3]
        and true_state.pending is None
    )
    update_order_case = {
        "bootstrap_update_count": true_state.bootstrap_updates,
        "chosen_cell_only_checks": true_state.chosen_cell_checks,
        "episode_update_count": true_state.completed_episodes,
        "event_order_exact": _event_order_exact(train_events),
        "forward_phase_order": forward_phase_order,
        "no_same_episode_backward_replay": target_dependency_case[
            "first_episode_positive_phases"
        ]
        == [3],
        "passed": update_order_passed,
        "pending_cleared_at_boundary": true_state.pending is None,
        "phase_update_order": [0, 1, 2, 3],
        "sentinels_checked": update_checked,
        "sentinels_rejected": update_rejected,
        "target_one_offsets": list(propagation[1]),
        "target_zero_offsets": list(propagation[0]),
        "terminal_update_count": true_state.terminal_updates,
        "total_update_count": true_state.total_updates,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(true_state),
                "train_trace_sha256": _trace_sha256(train_trace),
            }
        ),
    }

    forbidden_rejected = _forbidden_policy_rejections(
        splits["train"].observations0[0]
    )
    scope_rejected = 0
    for name in ("validation", "test"):
        try:
            _train_policy(
                splits[name],
                expected_metadata_sha256=_metadata_commitment(splits[name]),
            )
        except RuntimeError:
            scope_rejected += 1
    pure_lookup_state = _new_state()
    pure_lookup_state.q[:] = true_state.q
    base_observation = splits["test"].observations0[0]
    base_action = int(_greedy_action(pure_lookup_state, {"observation": base_observation}))
    pure_lookup_state.pending = PendingStep("f" * 64, (3, 1, 0), 1, 3)
    pure_lookup_state.completed_episodes = 999
    pure_lookup_action = int(
        _greedy_action(pure_lookup_state, {"observation": base_observation})
    )
    pure_lookup_exact = base_action == pure_lookup_action
    nuisance_variant = np.array(base_observation, copy=True)
    nuisance_variant[1] *= 1.5
    nuisance_variant[3] += 99.0
    nuisance_variant = _immutable_observation(nuisance_variant)
    no_nuisance_key = base_action == int(
        _greedy_action(pure_lookup_state, {"observation": nuisance_variant})
    )
    train_without_heldout_exact = bool(
        _state_commitment(true_state) == _state_commitment(replay_state)
        and _trace_sha256(train_trace) == _trace_sha256(replay_train_trace)
    )
    heldout_updates = true_state.total_updates - EPISODE_COUNTS["train"] * HORIZON
    leakage_passed = bool(
        forbidden_rejected == len(FORBIDDEN_POLICY_FIELDS)
        and scope_rejected == 2
        and pure_lookup_exact
        and no_nuisance_key
        and heldout_state_before == heldout_state_after
        and heldout_updates == 0
        and train_without_heldout_exact
        and true_state.pending is None
    )
    leakage_case = {
        "forbidden_fields_checked": len(FORBIDDEN_POLICY_FIELDS),
        "forbidden_fields_rejected": forbidden_rejected,
        "heldout_state_unchanged": heldout_state_before == heldout_state_after,
        "heldout_updates": heldout_updates,
        "no_nuisance_or_magnitude_key": no_nuisance_key,
        "passed": leakage_passed,
        "pending_state_empty": true_state.pending is None,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "pure_current_observation_lookup": pure_lookup_exact,
        "trace_sha256": _json_sha256(
            {
                "test_trace_sha256": _trace_sha256(test_trace),
                "validation_trace_sha256": _trace_sha256(validation_trace),
            }
        ),
        "train_scope_sentinels_checked": 2,
        "train_scope_sentinels_rejected": scope_rejected,
        "train_without_heldout_exact": train_without_heldout_exact,
    }

    canonical_test_score = _score_trace(test_trace, splits["test"])
    reordered_test_score = _score_trace(
        _independently_reordered(test_trace), splits["test"]
    )
    scoring_checked, scoring_rejected = _score_attacks(
        test_trace, splits["test"]
    )
    scoring_case = {
        "attacks_checked": scoring_checked,
        "attacks_rejected": scoring_rejected,
        "canonical_feedback_recomputed": True,
        "component_reorder_exact": canonical_test_score == reordered_test_score,
        "mode_mappings_authenticated": True,
        "passed": bool(
            scoring_checked == scoring_rejected
            and canonical_test_score == reordered_test_score
        ),
        "trace_sha256": _json_sha256(
            {
                "canonical_trace_sha256": _trace_sha256(test_trace),
                "reordered_trace_sha256": _trace_sha256(
                    _independently_reordered(test_trace)
                ),
            }
        ),
    }

    constant_scores: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for action in (0, 1):
        for name in ("validation", "test"):
            matrix = np.full(
                (EPISODE_COUNTS[name], HORIZON), action, dtype=np.int8
            )
            constant_scores[action][name] = _score_trace(
                _trace_from_action_matrix(splits[name], matrix), splits[name]
            )[0]
    best_constant_validation = max(
        constant_scores[0]["validation"], constant_scores[1]["validation"]
    )
    best_constant_test = max(
        constant_scores[0]["test"], constant_scores[1]["test"]
    )
    myopic_state = _fit_myopic(splits["train"])
    myopic_validation = _score_trace(
        _evaluate_frozen(myopic_state, splits["validation"]), splits["validation"]
    )[0]
    myopic_test = _score_trace(
        _evaluate_frozen(myopic_state, splits["test"]), splits["test"]
    )[0]
    no_bootstrap_state, _, _, _, _ = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        no_bootstrap=True,
    )
    no_bootstrap_validation = _score_trace(
        _evaluate_frozen(no_bootstrap_state, splits["validation"]),
        splits["validation"],
    )[0]
    no_bootstrap_test = _score_trace(
        _evaluate_frozen(no_bootstrap_state, splits["test"]), splits["test"]
    )[0]
    random_matrices = _random_baseline_matrices(splits)
    random_replay = _random_baseline_matrices(splits)
    random_validation_trace = _trace_from_action_matrix(
        splits["validation"], random_matrices["validation"]
    )
    random_test_trace = _trace_from_action_matrix(
        splits["test"], random_matrices["test"]
    )
    random_validation = _score_trace(
        random_validation_trace, splits["validation"]
    )[0]
    random_test = _score_trace(random_test_trace, splits["test"])[0]
    random_replay_exact = all(
        np.array_equal(random_matrices[name], random_replay[name])
        for name in ("validation", "test")
    )
    heldout_rows_sha256 = _json_sha256(
        {name: _split_projection(splits[name]) for name in ("validation", "test")}
    )
    replay_rows_sha256 = _json_sha256(
        {name: _split_projection(replay[name]) for name in ("validation", "test")}
    )
    baseline_values = (
        *[constant_scores[action][name] for action in (0, 1) for name in ("validation", "test")],
        myopic_validation,
        myopic_test,
        no_bootstrap_validation,
        no_bootstrap_test,
        random_validation,
        random_test,
    )
    baseline_passed = bool(
        random_replay_exact
        and heldout_rows_sha256 == replay_rows_sha256
        and all(np.isfinite(value) for value in baseline_values)
        and constant_scores[0]["validation"]
        == constant_scores[1]["validation"]
        == constant_scores[0]["test"]
        == constant_scores[1]["test"]
        == myopic_validation
        == myopic_test
        == no_bootstrap_validation
        == no_bootstrap_test
        == 0.5
    )
    baseline_case = {
        "best_constant_test_macro_return": best_constant_test,
        "best_constant_validation_macro_return": best_constant_validation,
        "constant_one_test_macro_return": constant_scores[1]["test"],
        "constant_one_validation_macro_return": constant_scores[1]["validation"],
        "constant_zero_test_macro_return": constant_scores[0]["test"],
        "constant_zero_validation_macro_return": constant_scores[0]["validation"],
        "finite_metrics": bool(all(np.isfinite(value) for value in baseline_values)),
        "myopic_test_macro_return": myopic_test,
        "myopic_validation_macro_return": myopic_validation,
        "no_bootstrap_test_macro_return": no_bootstrap_test,
        "no_bootstrap_validation_macro_return": no_bootstrap_validation,
        "passed": baseline_passed,
        "random_seed": RANDOM_BASELINE_SEED,
        "random_test_macro_return": random_test,
        "random_validation_macro_return": random_validation,
        "replay_exact": random_replay_exact,
        "same_evaluation_rows": heldout_rows_sha256 == replay_rows_sha256,
        "trace_sha256": _json_sha256(
            {
                "myopic_state_sha256": _state_commitment(myopic_state),
                "no_bootstrap_state_sha256": _state_commitment(no_bootstrap_state),
                "random_test_sha256": _trace_sha256(random_test_trace),
                "random_validation_sha256": _trace_sha256(random_validation_trace),
                "rows_sha256": heldout_rows_sha256,
            }
        ),
    }

    behavior_train = _score_trace(train_trace, splits["train"])[0]
    behavior_regret = EPISODE_COUNTS["train"] - int(
        round(behavior_train * EPISODE_COUNTS["train"])
    )
    postfit_train = _score_trace(postfit_train_trace, splits["train"])[0]
    validation, validation_minimum, _ = _score_trace(
        validation_trace, splits["validation"]
    )
    test, test_minimum, _ = _score_trace(test_trace, splits["test"])
    minimum_heldout = min(validation_minimum, test_minimum)
    recovery_passed = _positive_gate(
        behavior_return=behavior_train,
        behavior_regret=behavior_regret,
        postfit_train=postfit_train,
        validation=validation,
        test=test,
        minimum_heldout=minimum_heldout,
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        no_bootstrap_validation=no_bootstrap_validation,
        no_bootstrap_test=no_bootstrap_test,
        random_validation=random_validation,
        random_test=random_test,
        state=true_state,
        propagation=propagation,
    )
    recovery_case = {
        "behavior_train_mean_return": behavior_train,
        "behavior_train_regret": behavior_regret,
        "bootstrap_update_count": true_state.bootstrap_updates,
        "episode_update_count": true_state.completed_episodes,
        "learned_projection_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(true_state),
                "test_trace_sha256": _trace_sha256(test_trace),
                "validation_trace_sha256": _trace_sha256(validation_trace),
            }
        ),
        "min_heldout_regime_return": minimum_heldout,
        "passed": recovery_passed,
        "postfit_train_macro_return": postfit_train,
        "target_one_offsets": list(propagation[1]),
        "target_zero_offsets": list(propagation[0]),
        "terminal_update_count": true_state.terminal_updates,
        "test_gain_over_constant": test - best_constant_test,
        "test_gain_over_myopic": test - myopic_test,
        "test_gain_over_no_bootstrap": test - no_bootstrap_test,
        "test_gain_over_random": test - random_test,
        "test_macro_return": test,
        "total_update_count": true_state.total_updates,
        "validation_gain_over_constant": validation - best_constant_validation,
        "validation_gain_over_myopic": validation - myopic_validation,
        "validation_gain_over_no_bootstrap": validation - no_bootstrap_validation,
        "validation_gain_over_random": validation - random_validation,
        "validation_macro_return": validation,
    }

    (
        transition_state,
        transition_train_trace,
        _,
        _,
        transition_propagation,
    ) = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        treatment="transition_target",
    )
    transition_validation_trace = _evaluate_frozen(
        transition_state, splits["validation"]
    )
    transition_test_trace = _evaluate_frozen(transition_state, splits["test"])
    transition_validation, _, _ = _score_trace(
        transition_validation_trace, splits["validation"]
    )
    transition_test, _, _ = _score_trace(
        transition_test_trace, splits["test"]
    )
    mapping_exact, mapping_changed, multiset_exact = _transition_control_mapping(
        transition_train_trace, splits["train"]
    )
    actual_transition_unchanged = _actual_transition_commitment(
        train_trace
    ) == _actual_transition_commitment(transition_train_trace)
    transition_actions_unchanged = _action_commitment(
        train_trace
    ) == _action_commitment(transition_train_trace)
    transition_feedback_unchanged = _canonical_feedback_commitment(
        train_trace
    ) == _canonical_feedback_commitment(transition_train_trace)
    transition_gap = test - transition_test
    transition_positive_gate = _positive_gate(
        behavior_return=behavior_train,
        behavior_regret=behavior_regret,
        postfit_train=_score_trace(
            _evaluate_frozen(transition_state, splits["train"]), splits["train"]
        )[0],
        validation=transition_validation,
        test=transition_test,
        minimum_heldout=min(transition_validation, transition_test),
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        no_bootstrap_validation=no_bootstrap_validation,
        no_bootstrap_test=no_bootstrap_test,
        random_validation=random_validation,
        random_test=random_test,
        state=transition_state,
        propagation=transition_propagation,
    )
    transition_control_passed = bool(
        mapping_exact
        and mapping_changed == EPISODE_COUNTS["train"] * (HORIZON - 1)
        and multiset_exact
        and actual_transition_unchanged
        and transition_actions_unchanged
        and transition_feedback_unchanged
        and not transition_positive_gate
        and transition_validation
        <= THRESHOLDS["maximum_transition_target_validation_macro_return"]
        and transition_test
        <= THRESHOLDS["maximum_transition_target_test_macro_return"]
        and transition_gap >= THRESHOLDS["minimum_transition_target_gap"]
    )
    transition_control_case = {
        "action_commitment_unchanged": transition_actions_unchanged,
        "actual_transition_commitment_unchanged": actual_transition_unchanged,
        "bootstrap_assignment_changed_count": mapping_changed,
        "donor_mapping_exact": mapping_exact,
        "evaluator_feedback_commitment_unchanged": transition_feedback_unchanged,
        "nonidentity_permutation": TRANSITION_TARGET_PERMUTATION != tuple(range(16)),
        "passed": transition_control_passed,
        "positive_gate_rejected": not transition_positive_gate,
        "successor_multiset_unchanged": multiset_exact,
        "test_macro_return": transition_test,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(transition_state),
                "test_trace_sha256": _trace_sha256(transition_test_trace),
                "train_trace_sha256": _trace_sha256(transition_train_trace),
                "validation_trace_sha256": _trace_sha256(
                    transition_validation_trace
                ),
            }
        ),
        "true_test_gap": transition_gap,
        "validation_macro_return": transition_validation,
    }

    reward_state, reward_train_trace, _, _, reward_propagation = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        treatment="reward_origin",
    )
    reward_validation_trace = _evaluate_frozen(reward_state, splits["validation"])
    reward_test_trace = _evaluate_frozen(reward_state, splits["test"])
    reward_validation, _, _ = _score_trace(
        reward_validation_trace, splits["validation"]
    )
    reward_test, _, _ = _score_trace(reward_test_trace, splits["test"])
    origin_exact, assignment_changed, reward_multiset, no_fixed_points = (
        _reward_control_mapping(reward_train_trace, splits["train"])
    )
    reward_actions_unchanged = _action_commitment(train_trace) == _action_commitment(
        reward_train_trace
    )
    reward_transitions_unchanged = _actual_transition_commitment(
        train_trace
    ) == _actual_transition_commitment(reward_train_trace)
    reward_feedback_unchanged = _canonical_feedback_commitment(
        train_trace
    ) == _canonical_feedback_commitment(reward_train_trace)
    reward_gap = test - reward_test
    reward_positive_gate = _positive_gate(
        behavior_return=behavior_train,
        behavior_regret=behavior_regret,
        postfit_train=_score_trace(
            _evaluate_frozen(reward_state, splits["train"]), splits["train"]
        )[0],
        validation=reward_validation,
        test=reward_test,
        minimum_heldout=min(reward_validation, reward_test),
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        no_bootstrap_validation=no_bootstrap_validation,
        no_bootstrap_test=no_bootstrap_test,
        random_validation=random_validation,
        random_test=random_test,
        state=reward_state,
        propagation=reward_propagation,
    )
    reward_control_passed = bool(
        origin_exact
        and assignment_changed > 0
        and reward_multiset
        and no_fixed_points
        and reward_actions_unchanged
        and reward_transitions_unchanged
        and reward_feedback_unchanged
        and reward_state.pending is None
        and not reward_positive_gate
        and reward_validation
        <= THRESHOLDS["maximum_reward_origin_validation_macro_return"]
        and reward_test <= THRESHOLDS["maximum_reward_origin_test_macro_return"]
        and reward_gap >= THRESHOLDS["minimum_reward_origin_gap"]
    )
    reward_control_case = {
        "action_commitment_unchanged": reward_actions_unchanged,
        "assignment_changed_count": assignment_changed,
        "evaluator_feedback_commitment_unchanged": reward_feedback_unchanged,
        "nonidentity_no_fixed_point_permutation": no_fixed_points,
        "origin_assignment_exact": origin_exact,
        "passed": reward_control_passed,
        "pending_episode_empty_at_boundary": reward_state.pending is None,
        "positive_gate_rejected": not reward_positive_gate,
        "reward_multiset_unchanged": reward_multiset,
        "test_macro_return": reward_test,
        "timing_sentinels_checked": update_checked,
        "timing_sentinels_rejected": update_rejected,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(reward_state),
                "test_trace_sha256": _trace_sha256(reward_test_trace),
                "train_trace_sha256": _trace_sha256(reward_train_trace),
                "validation_trace_sha256": _trace_sha256(reward_validation_trace),
            }
        ),
        "transition_commitment_unchanged": reward_transitions_unchanged,
        "true_test_gap": reward_gap,
        "validation_macro_return": reward_validation,
    }

    ablated_state, ablated_train_trace, _, _, _ = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        treatment="signal_ablation",
    )
    ablated_test_trace = _evaluate_frozen(
        ablated_state, splits["test"], ablate_signal=True
    )
    true_ablated_test_trace = _evaluate_frozen(
        true_state, splits["test"], ablate_signal=True
    )
    ablated_test = _score_trace(
        ablated_test_trace, splits["test"], ablate_signal=True
    )[0]
    true_ablated_test = _score_trace(
        true_ablated_test_trace, splits["test"], ablate_signal=True
    )[0]
    only_signal_changed = True
    for split in splits.values():
        for observation in split.observations0:
            ablated = _ablate_observation(observation)
            only_signal_changed &= bool(
                float(ablated[1]) == 0.0
                and float(observation[1]) != 0.0
                and np.array_equal(ablated[[0, 2, 3]], observation[[0, 2, 3]])
            )
    attribution_metadata_unchanged = (
        _metadata_commitment(splits["train"]) == train_metadata_sha256
    )
    attribution_passed = bool(
        only_signal_changed
        and attribution_metadata_unchanged
        and ablated_test <= THRESHOLDS["maximum_attribution_test_macro_return"]
        and true_ablated_test
        <= THRESHOLDS["maximum_attribution_test_macro_return"]
    )
    attribution_case = {
        "evaluator_metadata_unchanged": attribution_metadata_unchanged,
        "only_signal_changed": only_signal_changed,
        "passed": attribution_passed,
        "refit_test_macro_return": ablated_test,
        "trace_sha256": _json_sha256(
            {
                "refit_state_sha256": _state_commitment(ablated_state),
                "refit_test_trace_sha256": _trace_sha256(ablated_test_trace),
                "refit_train_trace_sha256": _trace_sha256(ablated_train_trace),
                "true_test_trace_sha256": _trace_sha256(true_ablated_test_trace),
            }
        ),
        "true_policy_ablated_test_macro_return": true_ablated_test,
    }

    scoring_case["mode_mappings_authenticated"] = bool(
        transition_control_case["donor_mapping_exact"]
        and reward_control_case["origin_assignment_exact"]
    )
    scoring_case["passed"] = bool(
        scoring_case["passed"] and scoring_case["mode_mappings_authenticated"]
    )
    return {
        "action_dependent_transition": transition_case,
        "authenticated_scoring": scoring_case,
        "baseline_replay": baseline_case,
        "generator_partition": generator_case,
        "leakage_guards": leakage_case,
        "multistep_value_recovery": recovery_case,
        "reward_origin_control": reward_control_case,
        "signal_attribution_control": attribution_case,
        "td_bootstrap_update_order": update_order_case,
        "td_target_dependency": target_dependency_case,
        "transition_target_control": transition_control_case,
        "typed_multistep_contract": typed_case,
    }


def isolated_worker_trace() -> dict[str, object]:
    """Return the complete sanitized timing-free non-process projection."""
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
            "experiments.local_lab.multistep_td_propagation_worker",
            "--mode",
            "multistep-td-propagation-trace",
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
    """Execute the complete frozen case set and return a sanitized result."""
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the multi-step TD study requires a CPU backend")
    local_projection = isolated_worker_trace()
    cases = dict(local_projection["cases"])
    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = isolated_left == isolated_right == local_projection
        isolation_digest = _json_sha256(isolated_left)
    else:
        isolation_passed = None
        isolation_digest = "not-run-in-focused-test"
    cases["process_isolation"] = {
        "passed": isolation_passed,
        "trace_sha256": isolation_digest,
    }
    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    return {
        "action": (
            "synthetic_multistep_td_propagation_recovered_for_harness"
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
