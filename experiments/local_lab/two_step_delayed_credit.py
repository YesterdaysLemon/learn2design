"""Deterministic two-step delayed-credit mechanics fixture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np


STUDY_ID = "two-step-delayed-credit-v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
OBSERVATION_FIELDS = ("phase", "signal", "branch", "nuisance")
POLICY_INPUT_FIELDS = ("observation",)
STRUCTURE_KIND = "none"
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
    "action_key",
    "trajectory_id",
    "step",
    "done",
    "delay",
    "rng",
    "generator",
    "evaluator",
    "environment",
    "control_mode",
    "donor_identity",
    "reward_origin",
)
EPISODES_PER_REGIME = 32
HORIZON = 2
ACTION_PAIR_PATTERN = (
    (0, 0),
    (0, 0),
    (0, 1),
    (0, 1),
    (1, 0),
    (1, 0),
    (1, 1),
    (1, 1),
)
TRANSITION_DONOR_PERMUTATION = (4, 5, 6, 7, 0, 1, 2, 3)
REWARD_ORIGIN_PERMUTATION = (
    3,
    7,
    4,
    5,
    6,
    23,
    9,
    10,
    11,
    12,
    8,
    13,
    14,
    17,
    24,
    18,
    19,
    20,
    21,
    15,
    22,
    25,
    26,
    31,
    0,
    27,
    28,
    29,
    16,
    30,
    1,
    2,
)
RANDOM_BASELINE_SEED = 2026082803
EXPECTED_DATASET_SHA256 = (
    "fdeed53ae38fed818dba8ec5d3aa203d982ee0f3d92e6bd179cfa87d47970b89"
)
REGIMES = (
    {
        "code": 701,
        "nuisance_scale": 0.85,
        "nuisance_shift": -0.80,
        "signal_scale": 0.80,
        "split": "train",
    },
    {
        "code": 709,
        "nuisance_scale": 1.10,
        "nuisance_shift": -0.25,
        "signal_scale": 0.95,
        "split": "train",
    },
    {
        "code": 719,
        "nuisance_scale": 0.75,
        "nuisance_shift": 0.25,
        "signal_scale": 1.10,
        "split": "train",
    },
    {
        "code": 727,
        "nuisance_scale": 1.20,
        "nuisance_shift": 0.80,
        "signal_scale": 1.25,
        "split": "train",
    },
    {
        "code": 803,
        "nuisance_scale": 0.65,
        "nuisance_shift": -1.35,
        "signal_scale": 0.65,
        "split": "validation",
    },
    {
        "code": 811,
        "nuisance_scale": 1.25,
        "nuisance_shift": 1.35,
        "signal_scale": 1.35,
        "split": "validation",
    },
    {
        "code": 907,
        "nuisance_scale": 0.55,
        "nuisance_shift": -1.85,
        "signal_scale": 0.55,
        "split": "test",
    },
    {
        "code": 919,
        "nuisance_scale": 1.45,
        "nuisance_shift": 1.85,
        "signal_scale": 1.45,
        "split": "test",
    },
)
REGIME_COUNTS = {"test": 2, "train": 4, "validation": 2}
EPISODE_COUNTS = {
    name: count * EPISODES_PER_REGIME for name, count in REGIME_COUNTS.items()
}
ACTION_COUNTS = {name: count * HORIZON for name, count in EPISODE_COUNTS.items()}
THRESHOLDS = {
    "maximum_attribution_test_macro_return": 0.55,
    "maximum_reward_delay_test_macro_return": 0.55,
    "maximum_reward_delay_validation_macro_return": 0.55,
    "maximum_transition_shuffle_test_macro_return": 0.05,
    "maximum_transition_shuffle_validation_macro_return": 0.05,
    "minimum_heldout_regime_return": 0.98,
    "minimum_postfit_test_macro_return": 0.99,
    "minimum_postfit_train_macro_return": 0.99,
    "minimum_postfit_validation_macro_return": 0.99,
    "minimum_test_gain_constant": 0.30,
    "minimum_test_gain_myopic": 0.30,
    "minimum_test_gain_random": 0.30,
    "minimum_transition_shuffle_gap": 0.90,
    "minimum_reward_delay_gap": 0.40,
    "minimum_validation_gain_constant": 0.30,
    "minimum_validation_gain_myopic": 0.30,
    "minimum_validation_gain_random": 0.30,
}
EVENT_ORDER = (
    "observe0",
    "select0",
    "transition",
    "observe1",
    "select1",
    "terminal_reward",
    "update_episode",
    "log_episode",
)
CELL_UPDATE_ORDER = ("phase1", "phase0")


CASE_CONTRACT = {
    "typed_episodic_contract": {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "done_pattern": [False, True],
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
        "regime_counts": REGIME_COUNTS,
        "regimes": list(REGIMES),
        "episode_counts": EPISODE_COUNTS,
    },
    "action_dependent_transition": {
        "branch_values": [-1.0, 1.0],
        "done_after_transition": False,
        "reward_after_transition": "none",
        "transition_donor_permutation": list(TRANSITION_DONOR_PERMUTATION),
    },
    "delayed_update_order": {
        "behavior_action_pairs": [list(pair) for pair in ACTION_PAIR_PATTERN],
        "cell_update_order": list(CELL_UPDATE_ORDER),
        "episode_updates": EPISODE_COUNTS["train"],
        "event_order": list(EVENT_ORDER),
        "table_cell_updates": ACTION_COUNTS["train"],
    },
    "leakage_guards": {
        "forbidden_policy_fields": list(FORBIDDEN_POLICY_FIELDS),
        "heldout_updates": 0,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reorder": "independent_reverse_keyed_components",
        "train_split": "train_only",
    },
    "baseline_replay": {
        "constant_pairs": [[0, 0], [1, 1]],
        "myopic": "transition_local_reward_only",
        "random_seed": RANDOM_BASELINE_SEED,
        "scoring": "macro_mean_terminal_return_equal_weight_by_regime",
    },
    "delayed_credit_recovery": {
        "model": "four_state_two_action_empirical_terminal_return_table",
        "thresholds": THRESHOLDS,
        "tie_action": 0,
    },
    "transition_shuffle_control": {
        "scope": "train_successor_rows_within_each_eight_episode_block",
        "transition_donor_permutation": list(TRANSITION_DONOR_PERMUTATION),
        "thresholds": {
            "maximum_test_macro_return": THRESHOLDS[
                "maximum_transition_shuffle_test_macro_return"
            ],
            "maximum_validation_macro_return": THRESHOLDS[
                "maximum_transition_shuffle_validation_macro_return"
            ],
            "minimum_true_test_gap": THRESHOLDS[
                "minimum_transition_shuffle_gap"
            ],
        },
    },
    "reward_delay_control": {
        "reward_origin_permutation": list(REWARD_ORIGIN_PERMUTATION),
        "scope": "train_terminal_reward_origin_within_each_regime",
        "thresholds": {
            "maximum_test_macro_return": THRESHOLDS[
                "maximum_reward_delay_test_macro_return"
            ],
            "maximum_validation_macro_return": THRESHOLDS[
                "maximum_reward_delay_validation_macro_return"
            ],
            "minimum_true_test_gap": THRESHOLDS["minimum_reward_delay_gap"],
        },
    },
    "signal_attribution_control": {
        "maximum_test_macro_return": THRESHOLDS[
            "maximum_attribution_test_macro_return"
        ],
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
    "action_pair_pattern": [list(pair) for pair in ACTION_PAIR_PATTERN],
    "action_values": [0, 1],
    "claim_boundary": "synthetic_cpu_two_step_delayed_credit_harness_only",
    "episode_counts": EPISODE_COUNTS,
    "episodes_per_regime": EPISODES_PER_REGIME,
    "event_order": list(EVENT_ORDER),
    "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
    "generator_regimes": list(REGIMES),
    "horizon": HORIZON,
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
    "transition_donor_permutation": list(TRANSITION_DONOR_PERMUTATION),
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
class PendingEpisode:
    observation0_sha256: str
    state0_bin: int
    action0: int
    observation1_sha256: str | None = None
    state1_bin: int | None = None
    action1: int | None = None


@dataclass
class LearnerState:
    counts: np.ndarray
    return_sums: np.ndarray
    episode_updates: int = 0
    cell_updates: int = 0
    chosen_cell_checks: int = 0
    pending: PendingEpisode | None = None


@dataclass(frozen=True)
class UpdateAudit:
    state0_bin: int
    action0: int
    state1_bin: int
    action1: int
    update_order: tuple[str, str]


@dataclass(frozen=True)
class ActionRecord:
    action_key: str
    episode_key: str
    phase: int
    observation_sha256: str
    action: int


@dataclass(frozen=True)
class TransitionRecord:
    episode_key: str
    source_action: int
    next_observation_sha256: str
    done: bool


@dataclass(frozen=True)
class RewardRecord:
    episode_key: str
    action0: int
    action1: int
    terminal_reward: float
    update_reward: float
    done: bool
    episode_updates_before: int
    episode_updates_after: int


@dataclass(frozen=True)
class TraceBundle:
    action_records: tuple[ActionRecord, ...]
    transition_records: tuple[TransitionRecord, ...]
    reward_records: tuple[RewardRecord, ...]


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_identity(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _observation_sha256(observation: np.ndarray) -> str:
    return str(_array_identity(observation)["sha256"])


def _immutable_observation(values: list[float] | np.ndarray) -> np.ndarray:
    observation = np.asarray(values, dtype=np.float64).copy()
    observation.setflags(write=False)
    return observation


def _generate_split(split_name: str) -> SplitData:
    observations0: list[np.ndarray] = []
    targets: list[int] = []
    regime_codes: list[int] = []
    episode_indices: list[int] = []
    episode_keys: list[str] = []
    for regime in REGIMES:
        if regime["split"] != split_name:
            continue
        code = int(regime["code"])
        signal_scale = float(regime["signal_scale"])
        nuisance_shift = float(regime["nuisance_shift"])
        nuisance_scale = float(regime["nuisance_scale"])
        for episode in range(EPISODES_PER_REGIME):
            sign = -1.0 if episode % 2 == 0 else 1.0
            target = 0 if sign < 0.0 else 1
            magnitude = signal_scale * (1.0 + 0.03 * (episode % 5))
            nuisance = nuisance_shift + nuisance_scale * (
                (((7 * episode + code) % 23) - 11) / 11.0
            )
            observations0.append(
                _immutable_observation([0.0, sign * magnitude, 0.0, nuisance])
            )
            targets.append(target)
            regime_codes.append(code)
            episode_indices.append(episode)
            episode_keys.append(f"{split_name}:{code}:{episode:02d}")
    matrix = np.stack(observations0).astype(np.float64, copy=False)
    matrix.setflags(write=False)
    target_array = np.asarray(targets, dtype=np.int8)
    target_array.setflags(write=False)
    code_array = np.asarray(regime_codes, dtype=np.int32)
    code_array.setflags(write=False)
    index_array = np.asarray(episode_indices, dtype=np.int32)
    index_array.setflags(write=False)
    return SplitData(
        name=split_name,
        observations0=matrix,
        targets=target_array,
        regime_codes=code_array,
        episode_indices=index_array,
        episode_keys=tuple(episode_keys),
    )


def _generate_all() -> dict[str, SplitData]:
    return {
        name: _generate_split(name) for name in ("train", "validation", "test")
    }


def _canonical_transition(observation0: np.ndarray, action0: int) -> np.ndarray:
    branch = -1.0 if action0 == 0 else 1.0
    nuisance = float(observation0[3])
    return _immutable_observation(
        [1.0, 0.0, branch, nuisance + 0.125 * branch]
    )


def _terminal_reward(target: int, action0: int, action1: int) -> float:
    return float(action0 == target and action1 == action0)


def _split_projection(split: SplitData) -> dict[str, object]:
    successor0 = np.stack(
        [_canonical_transition(row, 0) for row in split.observations0]
    )
    successor1 = np.stack(
        [_canonical_transition(row, 1) for row in split.observations0]
    )
    return {
        "episode_indices": _array_identity(split.episode_indices),
        "episode_keys_sha256": _json_sha256(list(split.episode_keys)),
        "observations0": _array_identity(split.observations0),
        "regime_codes": _array_identity(split.regime_codes),
        "successor0": _array_identity(successor0),
        "successor1": _array_identity(successor1),
        "targets": _array_identity(split.targets),
    }


def _dataset_commitment(splits: dict[str, SplitData]) -> str:
    return _json_sha256(
        {name: _split_projection(split) for name, split in sorted(splits.items())}
    )


def _metadata_commitment(split: SplitData) -> str:
    return _json_sha256(
        {
            "episode_indices": _array_identity(split.episode_indices),
            "episode_keys_sha256": _json_sha256(list(split.episode_keys)),
            "regime_codes": _array_identity(split.regime_codes),
            "targets": _array_identity(split.targets),
        }
    )


def _row_hashes(split: SplitData) -> set[str]:
    return {_observation_sha256(row) for row in split.observations0}


def _partition_flags(splits: dict[str, SplitData]) -> dict[str, bool]:
    names = ("train", "validation", "test")
    regimes = {
        name: {int(value) for value in splits[name].regime_codes} for name in names
    }
    keys = {name: set(splits[name].episode_keys) for name in names}
    rows = {name: _row_hashes(splits[name]) for name in names}

    def disjoint(values: dict[str, set[object]]) -> bool:
        return all(
            values[left].isdisjoint(values[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )

    return {
        "episode_keys": disjoint(keys),
        "observations": disjoint(rows),
        "regimes": disjoint(regimes),
    }


def _generator_checks(
    splits: dict[str, SplitData],
) -> tuple[int, bool, bool, bool, bool, int]:
    validated = 0
    exact_regime_order = True
    exact_key_order = True
    keys_unique = True
    balanced_targets = True
    for split_name in ("train", "validation", "test"):
        split = splits[split_name]
        cursor = 0
        expected_codes: list[int] = []
        expected_keys: list[str] = []
        for regime in REGIMES:
            if regime["split"] != split_name:
                continue
            code = int(regime["code"])
            signal_scale = float(regime["signal_scale"])
            nuisance_shift = float(regime["nuisance_shift"])
            nuisance_scale = float(regime["nuisance_scale"])
            regime_targets: list[int] = []
            for episode in range(EPISODES_PER_REGIME):
                sign = -1.0 if episode % 2 == 0 else 1.0
                target = 0 if sign < 0.0 else 1
                magnitude = signal_scale * (1.0 + 0.03 * (episode % 5))
                nuisance = nuisance_shift + nuisance_scale * (
                    (((7 * episode + code) % 23) - 11) / 11.0
                )
                expected = np.asarray(
                    [0.0, sign * magnitude, 0.0, nuisance], dtype=np.float64
                )
                if (
                    np.array_equal(split.observations0[cursor], expected)
                    and int(split.targets[cursor]) == target
                    and int(split.regime_codes[cursor]) == code
                    and int(split.episode_indices[cursor]) == episode
                ):
                    validated += 1
                regime_targets.append(target)
                expected_codes.append(code)
                expected_keys.append(f"{split_name}:{code}:{episode:02d}")
                cursor += 1
            balanced_targets = balanced_targets and (
                regime_targets.count(0) == regime_targets.count(1) == 16
            )
        exact_regime_order = exact_regime_order and np.array_equal(
            split.regime_codes, np.asarray(expected_codes, dtype=np.int32)
        )
        exact_key_order = exact_key_order and split.episode_keys == tuple(
            expected_keys
        )
        keys_unique = keys_unique and len(set(split.episode_keys)) == len(
            split.episode_keys
        )
    row_sets = {name: _row_hashes(split) for name, split in splits.items()}
    overlap = sum(
        len(row_sets[left] & row_sets[right])
        for index, left in enumerate(("train", "validation", "test"))
        for right in ("train", "validation", "test")[index + 1 :]
    )
    return (
        validated,
        exact_regime_order,
        exact_key_order,
        keys_unique,
        balanced_targets,
        overlap,
    )


def _policy_observation(record: dict[str, object]) -> np.ndarray:
    if tuple(record) != POLICY_INPUT_FIELDS:
        raise ValueError("the policy record contains a forbidden input field")
    observation = record["observation"]
    if (
        not isinstance(observation, np.ndarray)
        or observation.dtype != np.float64
        or observation.shape != (len(OBSERVATION_FIELDS),)
        or not observation.flags.c_contiguous
        or observation.flags.writeable
        or not np.isfinite(observation).all()
    ):
        raise TypeError("the policy record contains a malformed observation")
    return observation


def _validate_action(action: object) -> np.ndarray:
    if (
        not isinstance(action, np.ndarray)
        or action.dtype != np.int8
        or action.shape != ()
        or int(action) not in (0, 1)
    ):
        raise TypeError("the policy returned a malformed action")
    return action


def _state_bin(observation: np.ndarray) -> int:
    phase = float(observation[0])
    if phase == 0.0:
        if float(observation[2]) != 0.0:
            raise ValueError("a start observation contains a branch value")
        return 1 if float(observation[1]) > 0.0 else 0
    if phase == 1.0:
        if float(observation[1]) != 0.0 or float(observation[2]) not in (-1.0, 1.0):
            raise ValueError("a successor observation has an invalid phase schema")
        return 3 if float(observation[2]) > 0.0 else 2
    raise ValueError("an observation contains an invalid phase")


def _new_state() -> LearnerState:
    return LearnerState(
        counts=np.zeros((4, 2), dtype=np.int32),
        return_sums=np.zeros((4, 2), dtype=np.float64),
    )


def _pending_projection(pending: PendingEpisode | None) -> object:
    if pending is None:
        return None
    return {
        "action0": pending.action0,
        "action1": pending.action1,
        "observation0_sha256": pending.observation0_sha256,
        "observation1_sha256": pending.observation1_sha256,
        "state0_bin": pending.state0_bin,
        "state1_bin": pending.state1_bin,
    }


def _state_commitment(state: LearnerState) -> str:
    return _json_sha256(
        {
            "cell_updates": state.cell_updates,
            "chosen_cell_checks": state.chosen_cell_checks,
            "counts": _array_identity(state.counts),
            "episode_updates": state.episode_updates,
            "pending": _pending_projection(state.pending),
            "return_sums": _array_identity(state.return_sums),
        }
    )


def _cell_mean(state: LearnerState, state_bin: int, action: int) -> float:
    count = int(state.counts[state_bin, action])
    return float(state.return_sums[state_bin, action]) / count if count else 0.0


def _greedy_action(state: LearnerState, record: dict[str, object]) -> np.ndarray:
    observation = _policy_observation(record)
    state_bin = _state_bin(observation)
    means = tuple(_cell_mean(state, state_bin, action) for action in (0, 1))
    return _validate_action(
        np.asarray(1 if means[1] > means[0] else 0, dtype=np.int8)
    )


def _behavior_pair(episode_index: int) -> tuple[int, int]:
    return ACTION_PAIR_PATTERN[episode_index % len(ACTION_PAIR_PATTERN)]


def _select_training(
    state: LearnerState, record: dict[str, object]
) -> np.ndarray:
    observation = _policy_observation(record)
    state_bin = _state_bin(observation)
    pair = _behavior_pair(state.episode_updates)
    if float(observation[0]) == 0.0:
        if state.pending is not None:
            raise RuntimeError("a new episode cannot replace a pending episode")
        action = _validate_action(np.asarray(pair[0], dtype=np.int8))
        state.pending = PendingEpisode(
            observation0_sha256=_observation_sha256(observation),
            state0_bin=state_bin,
            action0=int(action),
        )
        return action
    if state.pending is None:
        raise RuntimeError("a successor selection requires a start selection")
    if state.pending.observation1_sha256 is not None:
        raise RuntimeError("a second successor selection is forbidden")
    action = _validate_action(np.asarray(pair[1], dtype=np.int8))
    state.pending = replace(
        state.pending,
        observation1_sha256=_observation_sha256(observation),
        state1_bin=state_bin,
        action1=int(action),
    )
    return action


def _validate_reward(reward: object) -> np.ndarray:
    if (
        not isinstance(reward, np.ndarray)
        or reward.dtype != np.float64
        or reward.shape != ()
        or not np.isfinite(reward)
        or float(reward) not in (0.0, 1.0)
    ):
        raise TypeError("the learner received a malformed terminal reward")
    return reward


def _update_episode(state: LearnerState, reward: object) -> UpdateAudit:
    value = _validate_reward(reward)
    pending = state.pending
    if (
        pending is None
        or pending.observation1_sha256 is None
        or pending.state1_bin is None
        or pending.action1 is None
    ):
        raise RuntimeError("terminal feedback requires two pending selections")
    counts_before = state.counts.copy()
    sums_before = state.return_sums.copy()
    state.counts[pending.state1_bin, pending.action1] += 1
    state.return_sums[pending.state1_bin, pending.action1] += float(value)
    state.counts[pending.state0_bin, pending.action0] += 1
    state.return_sums[pending.state0_bin, pending.action0] += float(value)
    expected_counts = counts_before.copy()
    expected_sums = sums_before.copy()
    expected_counts[pending.state1_bin, pending.action1] += 1
    expected_sums[pending.state1_bin, pending.action1] += float(value)
    expected_counts[pending.state0_bin, pending.action0] += 1
    expected_sums[pending.state0_bin, pending.action0] += float(value)
    if not (
        np.array_equal(state.counts, expected_counts)
        and np.array_equal(state.return_sums, expected_sums)
    ):
        raise RuntimeError("an episode update changed an unvisited table cell")
    audit = UpdateAudit(
        state0_bin=pending.state0_bin,
        action0=pending.action0,
        state1_bin=pending.state1_bin,
        action1=pending.action1,
        update_order=CELL_UPDATE_ORDER,
    )
    state.episode_updates += 1
    state.cell_updates += 2
    state.chosen_cell_checks += 2
    state.pending = None
    return audit


def _action_key(episode_key: str, phase: int) -> str:
    return f"{episode_key}:{phase}"


def _ablate_observation0(observation: np.ndarray) -> np.ndarray:
    result = observation.copy()
    result[1] = 0.0
    result.setflags(write=False)
    return result


def _transition_shuffle_observation(
    split: SplitData, index: int
) -> np.ndarray:
    local_episode = int(split.episode_indices[index])
    block_start = local_episode - (local_episode % 8)
    donor_local = block_start + TRANSITION_DONOR_PERMUTATION[local_episode % 8]
    donor_candidates = np.flatnonzero(
        (split.regime_codes == split.regime_codes[index])
        & (split.episode_indices == donor_local)
    )
    if donor_candidates.shape != (1,):
        raise RuntimeError("the transition donor is not unique")
    donor_index = int(donor_candidates[0])
    donor_action0 = _behavior_pair(donor_local)[0]
    return _canonical_transition(split.observations0[donor_index], donor_action0)


def _canonical_behavior_rewards(split: SplitData) -> np.ndarray:
    values = [
        _terminal_reward(
            int(split.targets[index]),
            *_behavior_pair(int(split.episode_indices[index])),
        )
        for index in range(len(split.episode_keys))
    ]
    return np.asarray(values, dtype=np.float64)


def _misaligned_reward(
    split: SplitData, index: int, canonical_rewards: np.ndarray
) -> float:
    local_episode = int(split.episode_indices[index])
    origin_local = REWARD_ORIGIN_PERMUTATION[local_episode]
    origin_candidates = np.flatnonzero(
        (split.regime_codes == split.regime_codes[index])
        & (split.episode_indices == origin_local)
    )
    if origin_candidates.shape != (1,):
        raise RuntimeError("the delayed reward origin is not unique")
    return float(canonical_rewards[int(origin_candidates[0])])


def _train_policy(
    split: SplitData,
    *,
    expected_metadata_sha256: str,
    treatment: str = "canonical",
) -> tuple[LearnerState, TraceBundle, tuple[str, ...], tuple[UpdateAudit, ...]]:
    if treatment not in {
        "canonical",
        "transition_shuffle",
        "reward_misalignment",
        "signal_ablation",
    }:
        raise ValueError("unknown delayed-credit treatment")
    if split.name != "train" or {
        key.split(":", 1)[0] for key in split.episode_keys
    } != {"train"}:
        raise ValueError("the updater may consume only the train split")
    if _metadata_commitment(split) != expected_metadata_sha256:
        raise ValueError("the updater received unauthenticated evaluator metadata")

    state = _new_state()
    action_records: list[ActionRecord] = []
    transition_records: list[TransitionRecord] = []
    reward_records: list[RewardRecord] = []
    events: list[str] = []
    audits: list[UpdateAudit] = []
    canonical_rewards = _canonical_behavior_rewards(split)

    for index, episode_key in enumerate(split.episode_keys):
        observation0 = split.observations0[index]
        if treatment == "signal_ablation":
            observation0 = _ablate_observation0(observation0)
        events.append("observe0")
        action0_array = _select_training(
            state, {"observation": observation0}
        )
        action0 = int(action0_array)
        events.append("select0")

        canonical_observation1 = _canonical_transition(
            split.observations0[index], action0
        )
        observation1 = (
            _transition_shuffle_observation(split, index)
            if treatment == "transition_shuffle"
            else canonical_observation1
        )
        events.append("transition")
        events.append("observe1")
        action1_array = _select_training(
            state, {"observation": observation1}
        )
        action1 = int(action1_array)
        events.append("select1")

        canonical_reward = _terminal_reward(
            int(split.targets[index]), action0, action1
        )
        update_reward = (
            _misaligned_reward(split, index, canonical_rewards)
            if treatment == "reward_misalignment"
            else canonical_reward
        )
        events.append("terminal_reward")
        update_before = state.episode_updates
        audit = _update_episode(
            state, np.asarray(update_reward, dtype=np.float64)
        )
        audits.append(audit)
        events.append("update_episode")

        action_records.extend(
            (
                ActionRecord(
                    action_key=_action_key(episode_key, 0),
                    episode_key=episode_key,
                    phase=0,
                    observation_sha256=_observation_sha256(observation0),
                    action=action0,
                ),
                ActionRecord(
                    action_key=_action_key(episode_key, 1),
                    episode_key=episode_key,
                    phase=1,
                    observation_sha256=_observation_sha256(observation1),
                    action=action1,
                ),
            )
        )
        transition_records.append(
            TransitionRecord(
                episode_key=episode_key,
                source_action=action0,
                next_observation_sha256=_observation_sha256(observation1),
                done=False,
            )
        )
        reward_records.append(
            RewardRecord(
                episode_key=episode_key,
                action0=action0,
                action1=action1,
                terminal_reward=canonical_reward,
                update_reward=update_reward,
                done=True,
                episode_updates_before=update_before,
                episode_updates_after=state.episode_updates,
            )
        )
        events.append("log_episode")

    if state.pending is not None:
        raise RuntimeError("training ended with a pending episode")
    return (
        state,
        TraceBundle(
            action_records=tuple(action_records),
            transition_records=tuple(transition_records),
            reward_records=tuple(reward_records),
        ),
        tuple(events),
        tuple(audits),
    )


def _evaluate_frozen(
    state: LearnerState,
    split: SplitData,
    *,
    ablate_signal: bool = False,
) -> TraceBundle:
    before = _state_commitment(state)
    action_records: list[ActionRecord] = []
    transition_records: list[TransitionRecord] = []
    reward_records: list[RewardRecord] = []
    for index, episode_key in enumerate(split.episode_keys):
        observation0 = split.observations0[index]
        if ablate_signal:
            observation0 = _ablate_observation0(observation0)
        action0 = int(
            _greedy_action(state, {"observation": observation0})
        )
        observation1 = _canonical_transition(split.observations0[index], action0)
        action1 = int(
            _greedy_action(state, {"observation": observation1})
        )
        reward = _terminal_reward(int(split.targets[index]), action0, action1)
        action_records.extend(
            (
                ActionRecord(
                    action_key=_action_key(episode_key, 0),
                    episode_key=episode_key,
                    phase=0,
                    observation_sha256=_observation_sha256(observation0),
                    action=action0,
                ),
                ActionRecord(
                    action_key=_action_key(episode_key, 1),
                    episode_key=episode_key,
                    phase=1,
                    observation_sha256=_observation_sha256(observation1),
                    action=action1,
                ),
            )
        )
        transition_records.append(
            TransitionRecord(
                episode_key=episode_key,
                source_action=action0,
                next_observation_sha256=_observation_sha256(observation1),
                done=False,
            )
        )
        reward_records.append(
            RewardRecord(
                episode_key=episode_key,
                action0=action0,
                action1=action1,
                terminal_reward=reward,
                update_reward=reward,
                done=True,
                episode_updates_before=state.episode_updates,
                episode_updates_after=state.episode_updates,
            )
        )
    if _state_commitment(state) != before or state.pending is not None:
        raise RuntimeError("heldout evaluation mutated the train state")
    return TraceBundle(
        action_records=tuple(action_records),
        transition_records=tuple(transition_records),
        reward_records=tuple(reward_records),
    )


def _trace_projection(trace: TraceBundle) -> dict[str, object]:
    return {
        "action_keys_sha256": _json_sha256(
            [record.action_key for record in trace.action_records]
        ),
        "action_values": _array_identity(
            np.asarray(
                [record.action for record in trace.action_records], dtype=np.int8
            )
        ),
        "episode_keys_sha256": _json_sha256(
            [record.episode_key for record in trace.reward_records]
        ),
        "observation_commitments_sha256": _json_sha256(
            [record.observation_sha256 for record in trace.action_records]
        ),
        "terminal_values": _array_identity(
            np.asarray(
                [record.terminal_reward for record in trace.reward_records],
                dtype=np.float64,
            )
        ),
        "transition_commitments_sha256": _json_sha256(
            [
                record.next_observation_sha256
                for record in trace.transition_records
            ]
        ),
        "update_values": _array_identity(
            np.asarray(
                [record.update_reward for record in trace.reward_records],
                dtype=np.float64,
            )
        ),
    }


def _trace_sha256(trace: TraceBundle) -> str:
    return _json_sha256(_trace_projection(trace))


def _validate_component_keys(
    records: tuple[object, ...], keys: list[str], expected: set[str]
) -> dict[str, object]:
    if len(keys) != len(set(keys)):
        raise ValueError("a completed trace contains duplicate keys")
    if set(keys) != expected:
        raise ValueError("a completed trace contains missing or unexpected keys")
    return dict(zip(keys, records, strict=True))


def _score_trace(
    trace: TraceBundle,
    split: SplitData,
    *,
    ablate_signal: bool = False,
) -> tuple[float, float, str]:
    expected_action_keys = {
        _action_key(episode_key, phase)
        for episode_key in split.episode_keys
        for phase in (0, 1)
    }
    action_by_key = _validate_component_keys(
        trace.action_records,
        [record.action_key for record in trace.action_records],
        expected_action_keys,
    )
    expected_episode_keys = set(split.episode_keys)
    transition_by_key = _validate_component_keys(
        trace.transition_records,
        [record.episode_key for record in trace.transition_records],
        expected_episode_keys,
    )
    reward_by_key = _validate_component_keys(
        trace.reward_records,
        [record.episode_key for record in trace.reward_records],
        expected_episode_keys,
    )
    rewards_by_regime: dict[int, list[float]] = {}
    keyed_projection: dict[str, object] = {}
    for index, episode_key in enumerate(split.episode_keys):
        action0_record = action_by_key[_action_key(episode_key, 0)]
        action1_record = action_by_key[_action_key(episode_key, 1)]
        transition_record = transition_by_key[episode_key]
        reward_record = reward_by_key[episode_key]
        if not isinstance(action0_record, ActionRecord) or not isinstance(
            action1_record, ActionRecord
        ):
            raise TypeError("the action trace has a malformed record")
        if not isinstance(transition_record, TransitionRecord) or not isinstance(
            reward_record, RewardRecord
        ):
            raise TypeError("the episode trace has a malformed record")
        if (
            action0_record.phase != 0
            or action1_record.phase != 1
            or action0_record.episode_key != episode_key
            or action1_record.episode_key != episode_key
            or action0_record.action not in (0, 1)
            or action1_record.action not in (0, 1)
        ):
            raise ValueError("an action record violates the phase contract")
        observation0 = split.observations0[index]
        if ablate_signal:
            observation0 = _ablate_observation0(observation0)
        expected_observation1 = _canonical_transition(
            split.observations0[index], action0_record.action
        )
        expected_reward = _terminal_reward(
            int(split.targets[index]),
            action0_record.action,
            action1_record.action,
        )
        if action0_record.observation_sha256 != _observation_sha256(observation0):
            raise ValueError("the phase-zero observation commitment is wrong")
        if (
            action1_record.observation_sha256
            != _observation_sha256(expected_observation1)
        ):
            raise ValueError("the phase-one observation commitment is wrong")
        if (
            transition_record.source_action != action0_record.action
            or transition_record.next_observation_sha256
            != _observation_sha256(expected_observation1)
            or transition_record.done
        ):
            raise ValueError("the action-dependent transition is unauthenticated")
        if (
            reward_record.action0 != action0_record.action
            or reward_record.action1 != action1_record.action
            or reward_record.terminal_reward != expected_reward
            or reward_record.update_reward != expected_reward
            or not reward_record.done
        ):
            raise ValueError("the terminal reward record is unauthenticated")
        code = int(split.regime_codes[index])
        rewards_by_regime.setdefault(code, []).append(expected_reward)
        keyed_projection[episode_key] = {
            "action0": action0_record.action,
            "action1": action1_record.action,
            "reward": expected_reward,
        }
    regime_means = [
        float(np.mean(rewards_by_regime[code])) for code in sorted(rewards_by_regime)
    ]
    return (
        float(np.mean(regime_means)),
        float(min(regime_means)),
        _json_sha256(keyed_projection),
    )


def _reverse_trace_components(trace: TraceBundle) -> TraceBundle:
    return TraceBundle(
        action_records=tuple(reversed(trace.action_records)),
        transition_records=tuple(reversed(trace.transition_records)),
        reward_records=tuple(reversed(trace.reward_records)),
    )


def _event_order_exact(events: tuple[str, ...]) -> bool:
    return events == EVENT_ORDER * EPISODE_COUNTS["train"]


def _trace_from_action_pairs(
    split: SplitData, pairs: np.ndarray
) -> TraceBundle:
    if pairs.dtype != np.int8 or pairs.shape != (len(split.episode_keys), 2):
        raise TypeError("baseline action pairs have the wrong typed shape")
    action_records: list[ActionRecord] = []
    transition_records: list[TransitionRecord] = []
    reward_records: list[RewardRecord] = []
    for index, episode_key in enumerate(split.episode_keys):
        action0 = int(pairs[index, 0])
        action1 = int(pairs[index, 1])
        if action0 not in (0, 1) or action1 not in (0, 1):
            raise ValueError("baseline action pairs contain an invalid action")
        observation0 = split.observations0[index]
        observation1 = _canonical_transition(observation0, action0)
        reward = _terminal_reward(int(split.targets[index]), action0, action1)
        action_records.extend(
            (
                ActionRecord(
                    action_key=_action_key(episode_key, 0),
                    episode_key=episode_key,
                    phase=0,
                    observation_sha256=_observation_sha256(observation0),
                    action=action0,
                ),
                ActionRecord(
                    action_key=_action_key(episode_key, 1),
                    episode_key=episode_key,
                    phase=1,
                    observation_sha256=_observation_sha256(observation1),
                    action=action1,
                ),
            )
        )
        transition_records.append(
            TransitionRecord(
                episode_key=episode_key,
                source_action=action0,
                next_observation_sha256=_observation_sha256(observation1),
                done=False,
            )
        )
        reward_records.append(
            RewardRecord(
                episode_key=episode_key,
                action0=action0,
                action1=action1,
                terminal_reward=reward,
                update_reward=reward,
                done=True,
                episode_updates_before=0,
                episode_updates_after=0,
            )
        )
    return TraceBundle(
        action_records=tuple(action_records),
        transition_records=tuple(transition_records),
        reward_records=tuple(reward_records),
    )


def _random_baseline_pairs(
    splits: dict[str, SplitData]
) -> dict[str, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(RANDOM_BASELINE_SEED))
    return {
        name: generator.integers(
            0,
            2,
            size=(EPISODE_COUNTS[name], HORIZON),
            dtype=np.int8,
        )
        for name in ("validation", "test")
    }


def _fit_myopic(split: SplitData) -> LearnerState:
    state = _new_state()
    for index in range(len(split.episode_keys)):
        action0, action1 = _behavior_pair(int(split.episode_indices[index]))
        observation1 = _canonical_transition(split.observations0[index], action0)
        branch_bin = _state_bin(observation1)
        reward = _terminal_reward(int(split.targets[index]), action0, action1)
        state.counts[branch_bin, action1] += 1
        state.return_sums[branch_bin, action1] += reward
    return state


def _positive_gate(
    *,
    behavior_train_return: float,
    behavior_train_regret: int,
    postfit_train_return: float,
    validation_return: float,
    test_return: float,
    minimum_heldout_regime_return: float,
    best_constant_validation: float,
    best_constant_test: float,
    myopic_validation: float,
    myopic_test: float,
    random_validation: float,
    random_test: float,
    episode_updates: int,
    cell_updates: int,
    heldout_updates: int,
) -> bool:
    return (
        behavior_train_return == 0.25
        and behavior_train_regret == 96
        and postfit_train_return
        >= THRESHOLDS["minimum_postfit_train_macro_return"]
        and validation_return
        >= THRESHOLDS["minimum_postfit_validation_macro_return"]
        and test_return >= THRESHOLDS["minimum_postfit_test_macro_return"]
        and minimum_heldout_regime_return
        >= THRESHOLDS["minimum_heldout_regime_return"]
        and validation_return - best_constant_validation
        >= THRESHOLDS["minimum_validation_gain_constant"]
        and test_return - best_constant_test
        >= THRESHOLDS["minimum_test_gain_constant"]
        and validation_return - myopic_validation
        >= THRESHOLDS["minimum_validation_gain_myopic"]
        and test_return - myopic_test
        >= THRESHOLDS["minimum_test_gain_myopic"]
        and validation_return - random_validation
        >= THRESHOLDS["minimum_validation_gain_random"]
        and test_return - random_test
        >= THRESHOLDS["minimum_test_gain_random"]
        and episode_updates == EPISODE_COUNTS["train"]
        and cell_updates == ACTION_COUNTS["train"]
        and heldout_updates == 0
    )


def _forbidden_policy_rejections(observation: np.ndarray) -> int:
    rejected = 0
    for field in FORBIDDEN_POLICY_FIELDS:
        try:
            _policy_observation(
                {"observation": observation, field: "forbidden-sentinel"}
            )
        except ValueError:
            rejected += 1
    return rejected


def _scope_sentinel_rejections(
    splits: dict[str, SplitData], train_metadata_sha256: str
) -> int:
    rejected = 0
    for name in ("validation", "test"):
        try:
            _train_policy(
                splits[name],
                expected_metadata_sha256=train_metadata_sha256,
            )
        except ValueError:
            rejected += 1
    return rejected


def _timing_sentinel_rejections(train: SplitData) -> tuple[int, int]:
    checked = 12
    rejected = 0
    observation0 = train.observations0[0]
    observation1 = _canonical_transition(observation0, 0)
    reward = np.asarray(1.0, dtype=np.float64)

    state = _new_state()
    try:
        _update_episode(state, reward)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    try:
        _update_episode(state, reward)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    try:
        _select_training(state, {"observation": observation1})
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    try:
        _select_training(state, {"observation": observation0})
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    _select_training(state, {"observation": observation1})
    try:
        _select_training(state, {"observation": observation1})
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    _select_training(state, {"observation": observation1})
    _update_episode(state, reward)
    try:
        _update_episode(state, reward)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    _select_training(state, {"observation": observation1})
    try:
        _update_episode(state, np.asarray(0.5, dtype=np.float64))
    except TypeError:
        rejected += 1

    state = _new_state()
    _select_training(state, {"observation": observation0})
    _select_training(state, {"observation": observation1})
    try:
        _update_episode(state, {"value": 1.0, "origin": "hidden"})
    except TypeError:
        rejected += 1

    mutable = observation0.copy()
    state = _new_state()
    try:
        _select_training(state, {"observation": mutable})
    except TypeError:
        rejected += 1

    for malformed in (
        np.asarray(True, dtype=np.bool_),
        np.asarray(0.0, dtype=np.float64),
        np.asarray(2, dtype=np.int8),
    ):
        try:
            _validate_action(malformed)
        except TypeError:
            rejected += 1
    return checked, rejected


def _scoring_sentinel_rejections(
    trace: TraceBundle, split: SplitData
) -> tuple[int, int, bool]:
    checked = 7
    rejected = 0

    duplicated_actions = list(trace.action_records)
    duplicated_actions[1] = replace(
        duplicated_actions[1], action_key=duplicated_actions[0].action_key
    )
    candidates = [
        TraceBundle(
            tuple(duplicated_actions),
            trace.transition_records,
            trace.reward_records,
        ),
        TraceBundle(
            trace.action_records[:-1],
            trace.transition_records,
            trace.reward_records,
        ),
        TraceBundle(
            trace.action_records,
            (
                replace(
                    trace.transition_records[0],
                    next_observation_sha256="f" * 64,
                ),
                *trace.transition_records[1:],
            ),
            trace.reward_records,
        ),
        TraceBundle(
            trace.action_records,
            (
                replace(
                    trace.transition_records[0],
                    source_action=1 - trace.transition_records[0].source_action,
                ),
                *trace.transition_records[1:],
            ),
            trace.reward_records,
        ),
        TraceBundle(
            trace.action_records,
            trace.transition_records,
            (
                replace(
                    trace.reward_records[0],
                    terminal_reward=1.0
                    - trace.reward_records[0].terminal_reward,
                ),
                *trace.reward_records[1:],
            ),
        ),
        TraceBundle(
            trace.action_records,
            trace.transition_records,
            (
                replace(trace.reward_records[0], done=False),
                *trace.reward_records[1:],
            ),
        ),
        TraceBundle(
            (
                replace(trace.action_records[0], phase=1),
                *trace.action_records[1:],
            ),
            trace.transition_records,
            trace.reward_records,
        ),
    ]
    for candidate in candidates:
        try:
            _score_trace(candidate, split)
        except (TypeError, ValueError):
            rejected += 1

    canonical = _score_trace(trace, split)
    reversed_score = _score_trace(_reverse_trace_components(trace), split)
    return checked, rejected, canonical == reversed_score


def _values_by_regime(
    records: tuple[TransitionRecord, ...], split: SplitData
) -> dict[int, list[str]]:
    code_by_key = {
        key: int(split.regime_codes[index])
        for index, key in enumerate(split.episode_keys)
    }
    grouped: dict[int, list[str]] = {}
    for record in records:
        grouped.setdefault(code_by_key[record.episode_key], []).append(
            record.next_observation_sha256
        )
    return grouped


def _transition_multisets_equal(
    canonical: TraceBundle, control: TraceBundle, split: SplitData
) -> bool:
    left = _values_by_regime(canonical.transition_records, split)
    right = _values_by_regime(control.transition_records, split)
    return {
        code: sorted(values) for code, values in left.items()
    } == {code: sorted(values) for code, values in right.items()}


def _action_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            (record.action_key, record.action)
            for record in trace.action_records
        ]
    )


def _terminal_commitment(trace: TraceBundle) -> str:
    return _json_sha256(
        [
            (record.episode_key, record.terminal_reward)
            for record in trace.reward_records
        ]
    )


def _update_reward_multiset(trace: TraceBundle) -> list[float]:
    return sorted(record.update_reward for record in trace.reward_records)


def _all_table_cell_means(state: LearnerState) -> list[float]:
    return [
        _cell_mean(state, state_bin, action)
        for state_bin in range(4)
        for action in (0, 1)
    ]


def _table_commitment(state: LearnerState) -> str:
    return _json_sha256(
        {
            "counts": _array_identity(state.counts),
            "return_sums": _array_identity(state.return_sums),
        }
    )


def _non_process_projection() -> dict[str, object]:
    splits = _generate_all()
    replay = _generate_all()
    dataset_sha256 = _dataset_commitment(splits)
    replay_sha256 = _dataset_commitment(replay)
    train_metadata_sha256 = _metadata_commitment(splits["train"])
    partition_flags = _partition_flags(splits)

    true_state, train_trace, train_events, train_audits = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
    )
    train_replay_state, train_replay_trace, _, _ = _train_policy(
        replay["train"],
        expected_metadata_sha256=_metadata_commitment(replay["train"]),
    )
    heldout_state_before = _state_commitment(true_state)
    postfit_train_trace = _evaluate_frozen(true_state, splits["train"])
    validation_trace = _evaluate_frozen(true_state, splits["validation"])
    validation_state_after = _state_commitment(true_state)
    test_trace = _evaluate_frozen(true_state, splits["test"])
    heldout_state_after = _state_commitment(true_state)

    immutable_observation = splits["train"].observations0[0]
    immutable_rejected = False
    try:
        immutable_observation[1] = 0.0
    except ValueError:
        immutable_rejected = True
    action_values = {record.action for record in train_trace.action_records}
    terminal_values = {
        record.terminal_reward for record in train_trace.reward_records
    }
    typed_trajectory_records = sum(
        len(split.episode_keys) for split in splits.values()
    )
    typed_passed = (
        all(split.observations0.dtype == np.float64 for split in splits.values())
        and all(
            split.observations0.shape
            == (EPISODE_COUNTS[name], len(OBSERVATION_FIELDS))
            for name, split in splits.items()
        )
        and all(split.targets.dtype == np.int8 for split in splits.values())
        and all(split.regime_codes.dtype == np.int32 for split in splits.values())
        and all(
            not split.observations0.flags.writeable for split in splits.values()
        )
        and action_values == {0, 1}
        and terminal_values == {0.0, 1.0}
        and immutable_rejected
        and _event_order_exact(train_events)
        and all(not record.done for record in train_trace.transition_records)
        and all(record.done for record in train_trace.reward_records)
        and STRUCTURE_KIND == "none"
    )
    typed_case = {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "completed_episodes": typed_trajectory_records,
        "done_pattern_exact": bool(
            all(not record.done for record in train_trace.transition_records)
            and all(record.done for record in train_trace.reward_records)
        ),
        "event_order": list(EVENT_ORDER),
        "horizon": HORIZON,
        "immutable_observations": immutable_rejected,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [4],
        "passed": bool(typed_passed),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
        "typed_trajectory_records": typed_trajectory_records,
    }

    (
        validated_episodes,
        exact_regime_order,
        exact_key_order,
        within_split_keys_unique,
        balanced_targets,
        observation_overlap_count,
    ) = _generator_checks(splits)
    generator_passed = (
        dataset_sha256 == EXPECTED_DATASET_SHA256
        and dataset_sha256 == replay_sha256
        and all(partition_flags.values())
        and validated_episodes == sum(EPISODE_COUNTS.values())
        and exact_regime_order
        and exact_key_order
        and within_split_keys_unique
        and balanced_targets
        and observation_overlap_count == 0
        and all(np.isfinite(split.observations0).all() for split in splits.values())
    )
    generator_case = {
        "balanced_targets_per_regime": balanced_targets,
        "dataset_sha256": dataset_sha256,
        "deterministic_replay": dataset_sha256 == replay_sha256,
        "episode_counts": EPISODE_COUNTS,
        "episodes_per_regime": EPISODES_PER_REGIME,
        "exact_key_order": exact_key_order,
        "exact_regime_order": exact_regime_order,
        "expected_dataset_commitment": dataset_sha256
        == EXPECTED_DATASET_SHA256,
        "finite_observations": bool(
            all(np.isfinite(split.observations0).all() for split in splits.values())
        ),
        "phase_zero_observation_overlap_count": observation_overlap_count,
        "phase_zero_observations_disjoint": partition_flags["observations"],
        "passed": bool(generator_passed),
        "regime_counts": REGIME_COUNTS,
        "regimes_disjoint": partition_flags["regimes"],
        "episode_keys_disjoint": partition_flags["episode_keys"],
        "validated_episodes": validated_episodes,
        "within_split_keys_unique": within_split_keys_unique,
    }

    action_changes_successor = 0
    authenticated_transitions = 0
    for split in splits.values():
        for observation0 in split.observations0:
            successor0 = _canonical_transition(observation0, 0)
            successor1 = _canonical_transition(observation0, 1)
            action_changes_successor += int(not np.array_equal(successor0, successor1))
            authenticated_transitions += int(
                float(successor0[0]) == float(successor1[0]) == 1.0
                and float(successor0[2]) == -1.0
                and float(successor1[2]) == 1.0
                and float(successor0[1]) == float(successor1[1]) == 0.0
            ) * 2
    transition_passed = (
        action_changes_successor == sum(EPISODE_COUNTS.values())
        and authenticated_transitions == 2 * sum(EPISODE_COUNTS.values())
        and all(not record.done for record in train_trace.transition_records)
    )
    transition_case = {
        "action_changes_successor_count": action_changes_successor,
        "authenticated_transition_count": authenticated_transitions,
        "branch_values": [-1.0, 1.0],
        "done_after_transition": False,
        "passed": bool(transition_passed),
        "reward_exposed_after_first_action": False,
        "trace_sha256": _json_sha256(
            {
                "dataset_sha256": dataset_sha256,
                "train_trace_sha256": _trace_sha256(train_trace),
            }
        ),
    }

    sentinels_checked, sentinels_rejected = _timing_sentinel_rejections(
        splits["train"]
    )
    no_preterminal_table_mutation = True
    sentinel_state = _new_state()
    table_before = _table_commitment(sentinel_state)
    sentinel_observation0 = splits["train"].observations0[0]
    sentinel_action0 = int(
        _select_training(sentinel_state, {"observation": sentinel_observation0})
    )
    sentinel_observation1 = _canonical_transition(
        sentinel_observation0, sentinel_action0
    )
    _select_training(sentinel_state, {"observation": sentinel_observation1})
    no_preterminal_table_mutation = (
        _table_commitment(sentinel_state) == table_before
    )
    update_order_exact = all(
        audit.update_order == CELL_UPDATE_ORDER for audit in train_audits
    )
    delayed_order_passed = (
        _event_order_exact(train_events)
        and no_preterminal_table_mutation
        and update_order_exact
        and true_state.episode_updates == EPISODE_COUNTS["train"]
        and true_state.cell_updates == ACTION_COUNTS["train"]
        and true_state.chosen_cell_checks == ACTION_COUNTS["train"]
        and sentinels_checked == sentinels_rejected == 12
        and true_state.pending is None
    )
    delayed_order_case = {
        "cell_update_count": true_state.cell_updates,
        "cell_update_order": list(CELL_UPDATE_ORDER),
        "chosen_cell_only_checks": true_state.chosen_cell_checks,
        "episode_update_count": true_state.episode_updates,
        "event_order_exact": _event_order_exact(train_events),
        "no_preterminal_table_mutation": no_preterminal_table_mutation,
        "passed": bool(delayed_order_passed),
        "pending_cleared_at_boundary": true_state.pending is None,
        "sentinels_checked": sentinels_checked,
        "sentinels_rejected": sentinels_rejected,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(true_state),
                "train_trace_sha256": _trace_sha256(train_trace),
            }
        ),
        "update_order_exact": update_order_exact,
    }

    forbidden_rejected = _forbidden_policy_rejections(
        splits["train"].observations0[0]
    )
    scope_rejected = _scope_sentinel_rejections(
        splits, train_metadata_sha256
    )
    scoring_checked, scoring_rejected, reorder_exact = (
        _scoring_sentinel_rejections(test_trace, splits["test"])
    )
    heldout_state_unchanged = (
        heldout_state_before
        == validation_state_after
        == heldout_state_after
    )
    train_without_heldout_exact = (
        _state_commitment(true_state) == _state_commitment(train_replay_state)
        and _trace_sha256(train_trace) == _trace_sha256(train_replay_trace)
    )
    heldout_updates = true_state.episode_updates - EPISODE_COUNTS["train"]
    leakage_passed = (
        forbidden_rejected == len(FORBIDDEN_POLICY_FIELDS)
        and scope_rejected == 2
        and scoring_checked == scoring_rejected == 7
        and reorder_exact
        and heldout_state_unchanged
        and heldout_updates == 0
        and train_without_heldout_exact
        and true_state.pending is None
    )
    leakage_case = {
        "component_reorder_exact": reorder_exact,
        "forbidden_fields_checked": len(FORBIDDEN_POLICY_FIELDS),
        "forbidden_fields_rejected": forbidden_rejected,
        "heldout_state_unchanged": heldout_state_unchanged,
        "heldout_updates": heldout_updates,
        "passed": bool(leakage_passed),
        "pending_state_empty": true_state.pending is None,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "scoring_sentinels_checked": scoring_checked,
        "scoring_sentinels_rejected": scoring_rejected,
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

    constant_scores: dict[int, dict[str, float]] = {}
    for action in (0, 1):
        constant_scores[action] = {}
        for name in ("validation", "test"):
            pairs = np.full(
                (EPISODE_COUNTS[name], HORIZON), action, dtype=np.int8
            )
            score, _, _ = _score_trace(
                _trace_from_action_pairs(splits[name], pairs), splits[name]
            )
            constant_scores[action][name] = score
    best_constant_validation = max(
        constant_scores[action]["validation"] for action in (0, 1)
    )
    best_constant_test = max(
        constant_scores[action]["test"] for action in (0, 1)
    )
    myopic_state = _fit_myopic(splits["train"])
    myopic_validation_trace = _evaluate_frozen(
        myopic_state, splits["validation"]
    )
    myopic_test_trace = _evaluate_frozen(myopic_state, splits["test"])
    myopic_validation, _, _ = _score_trace(
        myopic_validation_trace, splits["validation"]
    )
    myopic_test, _, _ = _score_trace(myopic_test_trace, splits["test"])
    random_pairs = _random_baseline_pairs(splits)
    random_replay = _random_baseline_pairs(splits)
    random_validation_trace = _trace_from_action_pairs(
        splits["validation"], random_pairs["validation"]
    )
    random_test_trace = _trace_from_action_pairs(
        splits["test"], random_pairs["test"]
    )
    random_validation, _, _ = _score_trace(
        random_validation_trace, splits["validation"]
    )
    random_test, _, _ = _score_trace(random_test_trace, splits["test"])
    random_replay_exact = all(
        np.array_equal(random_pairs[name], random_replay[name])
        for name in ("validation", "test")
    )
    heldout_rows_sha256 = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    replay_rows_sha256 = _json_sha256(
        {
            name: _split_projection(replay[name])
            for name in ("validation", "test")
        }
    )
    same_evaluation_rows = heldout_rows_sha256 == replay_rows_sha256
    baseline_values = (
        constant_scores[0]["validation"],
        constant_scores[1]["validation"],
        constant_scores[0]["test"],
        constant_scores[1]["test"],
        myopic_validation,
        myopic_test,
        random_validation,
        random_test,
    )
    baseline_passed = (
        random_replay_exact
        and same_evaluation_rows
        and all(np.isfinite(value) for value in baseline_values)
        and constant_scores[0]["validation"]
        == constant_scores[1]["validation"]
        == constant_scores[0]["test"]
        == constant_scores[1]["test"]
        == myopic_validation
        == myopic_test
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
        "passed": bool(baseline_passed),
        "random_seed": RANDOM_BASELINE_SEED,
        "random_test_macro_return": random_test,
        "random_validation_macro_return": random_validation,
        "replay_exact": random_replay_exact,
        "same_evaluation_rows": same_evaluation_rows,
        "trace_sha256": _json_sha256(
            {
                "myopic_state_sha256": _state_commitment(myopic_state),
                "random_test_sha256": _trace_sha256(random_test_trace),
                "random_validation_sha256": _trace_sha256(
                    random_validation_trace
                ),
                "rows_sha256": heldout_rows_sha256,
            }
        ),
    }

    behavior_train_total = int(
        sum(record.terminal_reward for record in train_trace.reward_records)
    )
    behavior_train_return = behavior_train_total / EPISODE_COUNTS["train"]
    behavior_train_regret = EPISODE_COUNTS["train"] - behavior_train_total
    postfit_train_return, _, _ = _score_trace(
        postfit_train_trace, splits["train"]
    )
    validation_return, validation_minimum, _ = _score_trace(
        validation_trace, splits["validation"]
    )
    test_return, test_minimum, _ = _score_trace(test_trace, splits["test"])
    minimum_heldout = min(validation_minimum, test_minimum)
    recovery_passed = _positive_gate(
        behavior_train_return=behavior_train_return,
        behavior_train_regret=behavior_train_regret,
        postfit_train_return=postfit_train_return,
        validation_return=validation_return,
        test_return=test_return,
        minimum_heldout_regime_return=minimum_heldout,
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        random_validation=random_validation,
        random_test=random_test,
        episode_updates=true_state.episode_updates,
        cell_updates=true_state.cell_updates,
        heldout_updates=heldout_updates,
    )
    recovery_case = {
        "behavior_train_mean_return": behavior_train_return,
        "behavior_train_regret": behavior_train_regret,
        "cell_update_count": true_state.cell_updates,
        "episode_update_count": true_state.episode_updates,
        "learned_projection_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(true_state),
                "test_trace_sha256": _trace_sha256(test_trace),
                "train_trace_sha256": _trace_sha256(postfit_train_trace),
                "validation_trace_sha256": _trace_sha256(validation_trace),
            }
        ),
        "min_heldout_regime_return": minimum_heldout,
        "passed": bool(recovery_passed),
        "postfit_train_macro_return": postfit_train_return,
        "test_gain_over_constant": test_return - best_constant_test,
        "test_gain_over_myopic": test_return - myopic_test,
        "test_gain_over_random": test_return - random_test,
        "test_macro_return": test_return,
        "validation_gain_over_constant": validation_return
        - best_constant_validation,
        "validation_gain_over_myopic": validation_return - myopic_validation,
        "validation_gain_over_random": validation_return - random_validation,
        "validation_macro_return": validation_return,
    }

    (
        transition_state,
        transition_train_trace,
        transition_events,
        transition_audits,
    ) = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        treatment="transition_shuffle",
    )
    transition_postfit_train_trace = _evaluate_frozen(
        transition_state, splits["train"]
    )
    transition_validation_trace = _evaluate_frozen(
        transition_state, splits["validation"]
    )
    transition_test_trace = _evaluate_frozen(
        transition_state, splits["test"]
    )
    transition_postfit_train, _, _ = _score_trace(
        transition_postfit_train_trace, splits["train"]
    )
    transition_validation, transition_validation_minimum, _ = _score_trace(
        transition_validation_trace, splits["validation"]
    )
    transition_test, transition_test_minimum, _ = _score_trace(
        transition_test_trace, splits["test"]
    )
    transition_behavior_total = int(
        sum(
            record.terminal_reward
            for record in transition_train_trace.reward_records
        )
    )
    transition_behavior_return = (
        transition_behavior_total / EPISODE_COUNTS["train"]
    )
    transition_positive_gate = _positive_gate(
        behavior_train_return=transition_behavior_return,
        behavior_train_regret=EPISODE_COUNTS["train"]
        - transition_behavior_total,
        postfit_train_return=transition_postfit_train,
        validation_return=transition_validation,
        test_return=transition_test,
        minimum_heldout_regime_return=min(
            transition_validation_minimum, transition_test_minimum
        ),
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        random_validation=random_validation,
        random_test=random_test,
        episode_updates=transition_state.episode_updates,
        cell_updates=transition_state.cell_updates,
        heldout_updates=0,
    )
    rowwise_transition_changed_count = sum(
        left.next_observation_sha256 != right.next_observation_sha256
        for left, right in zip(
            train_trace.transition_records,
            transition_train_trace.transition_records,
            strict=True,
        )
    )
    transition_multiset_unchanged = _transition_multisets_equal(
        train_trace, transition_train_trace, splits["train"]
    )
    transition_actions_unchanged = _action_commitment(
        train_trace
    ) == _action_commitment(transition_train_trace)
    transition_terminal_unchanged = _terminal_commitment(
        train_trace
    ) == _terminal_commitment(transition_train_trace)
    transition_heldout_unchanged = heldout_rows_sha256 == replay_rows_sha256
    transition_gap = test_return - transition_test
    transition_shuffle_passed = (
        transition_multiset_unchanged
        and transition_actions_unchanged
        and transition_terminal_unchanged
        and transition_heldout_unchanged
        and rowwise_transition_changed_count == EPISODE_COUNTS["train"]
        and _event_order_exact(transition_events)
        and all(
            audit.update_order == CELL_UPDATE_ORDER
            for audit in transition_audits
        )
        and not transition_positive_gate
        and transition_validation
        <= THRESHOLDS["maximum_transition_shuffle_validation_macro_return"]
        and transition_test
        <= THRESHOLDS["maximum_transition_shuffle_test_macro_return"]
        and transition_gap >= THRESHOLDS["minimum_transition_shuffle_gap"]
    )
    transition_shuffle_case = {
        "action_commitment_unchanged": transition_actions_unchanged,
        "evaluator_terminal_commitment_unchanged": transition_terminal_unchanged,
        "heldout_commitment_unchanged": transition_heldout_unchanged,
        "nonidentity_permutation": TRANSITION_DONOR_PERMUTATION
        != tuple(range(8)),
        "passed": bool(transition_shuffle_passed),
        "positive_gate_rejected": not transition_positive_gate,
        "rowwise_transition_changed_count": rowwise_transition_changed_count,
        "successor_multiset_unchanged": transition_multiset_unchanged,
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

    (
        reward_delay_state,
        reward_delay_train_trace,
        reward_delay_events,
        reward_delay_audits,
    ) = _train_policy(
        splits["train"],
        expected_metadata_sha256=train_metadata_sha256,
        treatment="reward_misalignment",
    )
    reward_delay_postfit_train_trace = _evaluate_frozen(
        reward_delay_state, splits["train"]
    )
    reward_delay_validation_trace = _evaluate_frozen(
        reward_delay_state, splits["validation"]
    )
    reward_delay_test_trace = _evaluate_frozen(
        reward_delay_state, splits["test"]
    )
    reward_delay_postfit_train, _, _ = _score_trace(
        reward_delay_postfit_train_trace, splits["train"]
    )
    reward_delay_validation, reward_delay_validation_minimum, _ = _score_trace(
        reward_delay_validation_trace, splits["validation"]
    )
    reward_delay_test, reward_delay_test_minimum, _ = _score_trace(
        reward_delay_test_trace, splits["test"]
    )
    reward_delay_behavior_total = int(
        sum(
            record.terminal_reward
            for record in reward_delay_train_trace.reward_records
        )
    )
    reward_delay_positive_gate = _positive_gate(
        behavior_train_return=reward_delay_behavior_total
        / EPISODE_COUNTS["train"],
        behavior_train_regret=EPISODE_COUNTS["train"]
        - reward_delay_behavior_total,
        postfit_train_return=reward_delay_postfit_train,
        validation_return=reward_delay_validation,
        test_return=reward_delay_test,
        minimum_heldout_regime_return=min(
            reward_delay_validation_minimum, reward_delay_test_minimum
        ),
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        myopic_validation=myopic_validation,
        myopic_test=myopic_test,
        random_validation=random_validation,
        random_test=random_test,
        episode_updates=reward_delay_state.episode_updates,
        cell_updates=reward_delay_state.cell_updates,
        heldout_updates=0,
    )
    reward_actions_unchanged = _action_commitment(
        train_trace
    ) == _action_commitment(reward_delay_train_trace)
    reward_transitions_unchanged = [
        record.next_observation_sha256
        for record in train_trace.transition_records
    ] == [
        record.next_observation_sha256
        for record in reward_delay_train_trace.transition_records
    ]
    reward_terminal_unchanged = _terminal_commitment(
        train_trace
    ) == _terminal_commitment(reward_delay_train_trace)
    reward_multiset_unchanged = _update_reward_multiset(
        train_trace
    ) == _update_reward_multiset(reward_delay_train_trace)
    reward_assignment_changed_count = sum(
        left.update_reward != right.update_reward
        for left, right in zip(
            train_trace.reward_records,
            reward_delay_train_trace.reward_records,
            strict=True,
        )
    )
    reward_origin_is_permutation = sorted(REWARD_ORIGIN_PERMUTATION) == list(
        range(EPISODES_PER_REGIME)
    )
    reward_origin_no_fixed_points = all(
        destination != origin
        for destination, origin in enumerate(REWARD_ORIGIN_PERMUTATION)
    )
    table_cell_means = _all_table_cell_means(reward_delay_state)
    per_cell_assigned_return_exact = all(
        value == 0.25 for value in table_cell_means
    )
    reward_delay_gap = test_return - reward_delay_test
    reward_delay_passed = (
        reward_origin_is_permutation
        and reward_origin_no_fixed_points
        and reward_actions_unchanged
        and reward_transitions_unchanged
        and reward_terminal_unchanged
        and reward_multiset_unchanged
        and reward_assignment_changed_count > 0
        and per_cell_assigned_return_exact
        and _event_order_exact(reward_delay_events)
        and all(
            audit.update_order == CELL_UPDATE_ORDER
            for audit in reward_delay_audits
        )
        and sentinels_checked == sentinels_rejected == 12
        and not reward_delay_positive_gate
        and reward_delay_validation
        <= THRESHOLDS["maximum_reward_delay_validation_macro_return"]
        and reward_delay_test
        <= THRESHOLDS["maximum_reward_delay_test_macro_return"]
        and reward_delay_gap >= THRESHOLDS["minimum_reward_delay_gap"]
    )
    reward_delay_case = {
        "action_commitment_unchanged": reward_actions_unchanged,
        "assignment_changed_count": reward_assignment_changed_count,
        "evaluator_terminal_commitment_unchanged": reward_terminal_unchanged,
        "nonidentity_no_fixed_point_permutation": bool(
            reward_origin_is_permutation and reward_origin_no_fixed_points
        ),
        "passed": bool(reward_delay_passed),
        "per_cell_assigned_return_exact": per_cell_assigned_return_exact,
        "positive_gate_rejected": not reward_delay_positive_gate,
        "reward_multiset_unchanged": reward_multiset_unchanged,
        "test_macro_return": reward_delay_test,
        "timing_sentinels_checked": sentinels_checked,
        "timing_sentinels_rejected": sentinels_rejected,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(reward_delay_state),
                "test_trace_sha256": _trace_sha256(reward_delay_test_trace),
                "train_trace_sha256": _trace_sha256(reward_delay_train_trace),
                "validation_trace_sha256": _trace_sha256(
                    reward_delay_validation_trace
                ),
            }
        ),
        "transition_commitment_unchanged": reward_transitions_unchanged,
        "true_test_gap": reward_delay_gap,
        "validation_macro_return": reward_delay_validation,
    }

    (
        ablated_state,
        ablated_train_trace,
        _,
        _,
    ) = _train_policy(
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
    ablated_test_return, _, _ = _score_trace(
        ablated_test_trace, splits["test"], ablate_signal=True
    )
    true_ablated_test_return, _, _ = _score_trace(
        true_ablated_test_trace, splits["test"], ablate_signal=True
    )
    only_signal_changed = True
    for split in splits.values():
        for observation in split.observations0:
            ablated = _ablate_observation0(observation)
            only_signal_changed = only_signal_changed and (
                float(ablated[1]) == 0.0
                and np.array_equal(ablated[[0, 2, 3]], observation[[0, 2, 3]])
                and float(observation[1]) != 0.0
            )
    attribution_metadata_unchanged = _metadata_commitment(
        splits["train"]
    ) == train_metadata_sha256
    attribution_passed = (
        only_signal_changed
        and attribution_metadata_unchanged
        and ablated_test_return
        <= THRESHOLDS["maximum_attribution_test_macro_return"]
        and true_ablated_test_return
        <= THRESHOLDS["maximum_attribution_test_macro_return"]
    )
    attribution_case = {
        "evaluator_metadata_unchanged": attribution_metadata_unchanged,
        "only_signal_changed": only_signal_changed,
        "passed": bool(attribution_passed),
        "refit_test_macro_return": ablated_test_return,
        "trace_sha256": _json_sha256(
            {
                "refit_state_sha256": _state_commitment(ablated_state),
                "refit_test_trace_sha256": _trace_sha256(ablated_test_trace),
                "refit_train_trace_sha256": _trace_sha256(
                    ablated_train_trace
                ),
                "true_ablated_test_trace_sha256": _trace_sha256(
                    true_ablated_test_trace
                ),
            }
        ),
        "true_policy_ablated_test_macro_return": true_ablated_test_return,
    }

    return {
        "action_dependent_transition": transition_case,
        "baseline_replay": baseline_case,
        "delayed_credit_recovery": recovery_case,
        "delayed_update_order": delayed_order_case,
        "generator_partition": generator_case,
        "leakage_guards": leakage_case,
        "reward_delay_control": reward_delay_case,
        "signal_attribution_control": attribution_case,
        "transition_shuffle_control": transition_shuffle_case,
        "typed_episodic_contract": typed_case,
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
            "experiments.local_lab.two_step_delayed_credit_worker",
            "--mode",
            "two-step-delayed-credit-trace",
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
        raise RuntimeError("the delayed-credit study requires a CPU backend")

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
            "synthetic_two_step_delayed_credit_recovered_for_harness"
            if passed
            else (
                "park_delayed_credit_research"
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
