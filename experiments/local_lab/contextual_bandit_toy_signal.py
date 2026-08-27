"""Deterministic contextual-bandit toy-signal learning fixture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np


STUDY_ID = "contextual-bandit-toy-signal-v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
CONTEXT_FIELDS = ("signal", "nuisance_pair", "nuisance_cycle")
POLICY_INPUT_FIELDS = ("context",)
STRUCTURE_KIND = "none"
FORBIDDEN_POLICY_FIELDS = (
    "preferred_action",
    "reward",
    "regime_code",
    "split",
    "sample_key",
    "trajectory_id",
    "step",
    "done",
    "next_context",
    "counterfactual_reward",
)
TRAJECTORIES_PER_REGIME = 32
STEPS_PER_TRAJECTORY = 8
PREFERRED_ACTION_PATTERN = (0, 1, 0, 1, 0, 1, 0, 1)
SIGN_PATTERN = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
FORCED_ACTION_PATTERN = (0, 0, 1, 1, 0, 0, 1, 1)
SHUFFLE_PERMUTATION = (0, 2, 4, 6, 1, 3, 5, 7)
RANDOM_BASELINE_SEED = 2026082713
EXPECTED_DATASET_SHA256 = (
    "ca1258a1f297b6bca16f6e42ab18e53743aa518a3e9d22f3ba361ac688de6628"
)
REGIMES = (
    {
        "code": 401,
        "nuisance_scale": 0.85,
        "nuisance_shift": -0.90,
        "signal_scale": 0.75,
        "split": "train",
    },
    {
        "code": 409,
        "nuisance_scale": 1.15,
        "nuisance_shift": -0.30,
        "signal_scale": 0.90,
        "split": "train",
    },
    {
        "code": 419,
        "nuisance_scale": 0.80,
        "nuisance_shift": 0.30,
        "signal_scale": 1.10,
        "split": "train",
    },
    {
        "code": 421,
        "nuisance_scale": 1.20,
        "nuisance_shift": 0.90,
        "signal_scale": 1.25,
        "split": "train",
    },
    {
        "code": 503,
        "nuisance_scale": 0.70,
        "nuisance_shift": -1.40,
        "signal_scale": 0.65,
        "split": "validation",
    },
    {
        "code": 509,
        "nuisance_scale": 1.30,
        "nuisance_shift": 1.40,
        "signal_scale": 1.35,
        "split": "validation",
    },
    {
        "code": 601,
        "nuisance_scale": 0.60,
        "nuisance_shift": -1.90,
        "signal_scale": 0.55,
        "split": "test",
    },
    {
        "code": 607,
        "nuisance_scale": 1.40,
        "nuisance_shift": 1.90,
        "signal_scale": 1.45,
        "split": "test",
    },
)
REGIME_COUNTS = {"test": 2, "train": 4, "validation": 2}
TRAJECTORY_COUNTS = {
    name: count * TRAJECTORIES_PER_REGIME
    for name, count in REGIME_COUNTS.items()
}
SPLIT_COUNTS = {
    name: count * STEPS_PER_TRAJECTORY
    for name, count in TRAJECTORY_COUNTS.items()
}
THRESHOLDS = {
    "maximum_attribution_test_macro_reward": 0.55,
    "maximum_shuffle_test_macro_reward": 0.55,
    "maximum_shuffle_validation_macro_reward": 0.55,
    "maximum_train_regret": 264,
    "minimum_heldout_regime_reward": 0.98,
    "minimum_test_gain_constant": 0.30,
    "minimum_test_gain_random": 0.25,
    "minimum_test_macro_reward": 0.99,
    "minimum_train_mean_reward": 0.74,
    "minimum_true_shuffle_gap": 0.40,
    "minimum_validation_gain_constant": 0.30,
    "minimum_validation_macro_reward": 0.99,
}
LOGGING_EVENT_ORDER = ("context", "select", "reward", "update", "log")


CASE_CONTRACT = {
    "typed_bandit_contract": {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "context_dtype": "float64",
        "context_fields": list(CONTEXT_FIELDS),
        "context_shape": [3],
        "done_dtype": "bool",
        "done_pattern": [False, False, False, False, False, False, False, True],
        "horizon": STEPS_PER_TRAJECTORY,
        "logging_event_order": list(LOGGING_EVENT_ORDER),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "preferred_action_dtype": "int8",
        "preferred_action_values": [0, 1],
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
    },
    "generator_partition": {
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "regime_counts": REGIME_COUNTS,
        "regimes": list(REGIMES),
        "split_counts": SPLIT_COUNTS,
        "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
        "trajectory_counts": TRAJECTORY_COUNTS,
    },
    "online_update_order": {
        "event_order": list(LOGGING_EVENT_ORDER),
        "exploration_schedule": list(FORCED_ACTION_PATTERN),
        "exploration_trajectories": "even",
        "exploitation_trajectories": "odd",
        "tie_action": 0,
        "train_updates": SPLIT_COUNTS["train"],
    },
    "leakage_guards": {
        "forbidden_policy_fields": list(FORBIDDEN_POLICY_FIELDS),
        "heldout_updates": 0,
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reorder": "reverse_completed_heldout_log",
        "train_split": "train_only",
    },
    "baseline_replay": {
        "constant_actions": [0, 1],
        "random_seed": RANDOM_BASELINE_SEED,
        "scoring": "macro_mean_reward_equal_weight_by_regime",
    },
    "contextual_recovery": {
        "model": "two_context_bin_two_action_empirical_reward_table",
        "thresholds": THRESHOLDS,
        "tie_action": 0,
    },
    "shuffled_context_control": {
        "permutation": list(SHUFFLE_PERMUTATION),
        "scope": "train_contexts_within_each_eight_step_trajectory",
        "thresholds": {
            "maximum_test_macro_reward": THRESHOLDS[
                "maximum_shuffle_test_macro_reward"
            ],
            "maximum_validation_macro_reward": THRESHOLDS[
                "maximum_shuffle_validation_macro_reward"
            ],
            "minimum_true_shuffle_gap": THRESHOLDS[
                "minimum_true_shuffle_gap"
            ],
        },
    },
    "signal_attribution_control": {
        "maximum_test_macro_reward": THRESHOLDS[
            "maximum_attribution_test_macro_reward"
        ],
        "signal_index": 0,
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
    "baseline_seed": RANDOM_BASELINE_SEED,
    "claim_boundary": "synthetic_cpu_contextual_bandit_harness_only",
    "context_dtype": "float64",
    "context_fields": list(CONTEXT_FIELDS),
    "context_shape": [3],
    "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
    "forced_action_pattern": list(FORCED_ACTION_PATTERN),
    "generator_regimes": list(REGIMES),
    "horizon": STEPS_PER_TRAJECTORY,
    "logging_event_order": list(LOGGING_EVENT_ORDER),
    "policy_input_fields": list(POLICY_INPUT_FIELDS),
    "preferred_action_pattern": list(PREFERRED_ACTION_PATTERN),
    "regime_counts": REGIME_COUNTS,
    "reward_dtype": "float64",
    "reward_values": [0.0, 1.0],
    "shuffle_permutation": list(SHUFFLE_PERMUTATION),
    "split_counts": SPLIT_COUNTS,
    "steps_per_trajectory": STEPS_PER_TRAJECTORY,
    "structure_kind": STRUCTURE_KIND,
    "thresholds": THRESHOLDS,
    "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
    "trajectory_counts": TRAJECTORY_COUNTS,
}


@dataclass(frozen=True)
class SplitData:
    contexts: np.ndarray
    preferred_actions: np.ndarray
    regime_codes: np.ndarray
    sample_keys: tuple[str, ...]
    done: np.ndarray


@dataclass(frozen=True)
class PendingSelection:
    context_sha256: str
    context_bin: int
    action: int
    update_index: int
    exploration: bool


@dataclass
class BanditState:
    counts: np.ndarray
    reward_sums: np.ndarray
    update_count: int = 0
    pending: PendingSelection | None = None
    chosen_cell_checks: int = 0


@dataclass(frozen=True)
class StepLog:
    sample_key: str
    context_sha256: str
    action: int
    reward: float
    update_before: int
    update_after: int
    exploration: bool
    done: bool


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


def _context_sha256(context: np.ndarray) -> str:
    return str(_array_identity(context)["sha256"])


def _split_projection(split: SplitData) -> dict[str, object]:
    return {
        "contexts": _array_identity(split.contexts),
        "done": _array_identity(split.done),
        "keys_sha256": _json_sha256(list(split.sample_keys)),
        "preferred_actions": _array_identity(split.preferred_actions),
        "regime_codes": _array_identity(split.regime_codes),
    }


def _metadata_commitment(split: SplitData) -> str:
    return _json_sha256(
        {
            "done": _array_identity(split.done),
            "keys_sha256": _json_sha256(list(split.sample_keys)),
            "preferred_actions": _array_identity(split.preferred_actions),
            "regime_codes": _array_identity(split.regime_codes),
        }
    )


def _generate_split(split_name: str) -> SplitData:
    contexts: list[list[float]] = []
    preferred_actions: list[int] = []
    regime_codes: list[int] = []
    sample_keys: list[str] = []
    done: list[bool] = []
    for regime in REGIMES:
        if regime["split"] != split_name:
            continue
        code = int(regime["code"])
        signal_scale = float(regime["signal_scale"])
        nuisance_shift = float(regime["nuisance_shift"])
        nuisance_scale = float(regime["nuisance_scale"])
        for trajectory in range(TRAJECTORIES_PER_REGIME):
            magnitude = signal_scale * (1.0 + 0.04 * (trajectory % 7))
            nuisance_pair = nuisance_shift + nuisance_scale * (
                ((trajectory % 11) - 5) / 5.0
            )
            nuisance_cycle = 0.4 * nuisance_shift + nuisance_scale * (
                (((5 * trajectory + code) % 19) - 9) / 9.0
            )
            for step, (sign, preferred) in enumerate(
                zip(SIGN_PATTERN, PREFERRED_ACTION_PATTERN, strict=True)
            ):
                contexts.append([sign * magnitude, nuisance_pair, nuisance_cycle])
                preferred_actions.append(preferred)
                regime_codes.append(code)
                sample_keys.append(
                    f"{split_name}:{code}:{trajectory:02d}:{step}"
                )
                done.append(step == STEPS_PER_TRAJECTORY - 1)
    return SplitData(
        contexts=np.asarray(contexts, dtype=np.float64),
        preferred_actions=np.asarray(preferred_actions, dtype=np.int8),
        regime_codes=np.asarray(regime_codes, dtype=np.int32),
        sample_keys=tuple(sample_keys),
        done=np.asarray(done, dtype=np.bool_),
    )


def _generate_all() -> dict[str, SplitData]:
    return {
        name: _generate_split(name) for name in ("train", "validation", "test")
    }


def _dataset_commitment(splits: dict[str, SplitData]) -> str:
    return _json_sha256(
        {name: _split_projection(split) for name, split in sorted(splits.items())}
    )


def _row_hashes(contexts: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in contexts
    }


def _partition_flags(splits: dict[str, SplitData]) -> dict[str, bool]:
    names = ("train", "validation", "test")
    regimes = {
        name: {int(value) for value in splits[name].regime_codes} for name in names
    }
    keys = {name: set(splits[name].sample_keys) for name in names}
    contexts = {name: _row_hashes(splits[name].contexts) for name in names}

    def disjoint(values: dict[str, set[object]]) -> bool:
        return all(
            values[left].isdisjoint(values[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )

    return {
        "contexts": disjoint(contexts),
        "regimes": disjoint(regimes),
        "sample_keys": disjoint(keys),
    }


def _cross_split_context_overlap_count(splits: dict[str, SplitData]) -> int:
    names = ("train", "validation", "test")
    rows = {name: _row_hashes(splits[name].contexts) for name in names}
    return sum(
        len(rows[left] & rows[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _generator_contract_checks(
    splits: dict[str, SplitData],
) -> tuple[int, bool, bool, bool, bool]:
    validated = 0
    exact_regime_order = True
    exact_key_order = True
    unique_keys = True
    done_pattern_exact = True
    expected_preferred = np.asarray(PREFERRED_ACTION_PATTERN, dtype=np.int8)
    expected_done = np.asarray(
        [False] * (STEPS_PER_TRAJECTORY - 1) + [True], dtype=np.bool_
    )
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
            for trajectory in range(TRAJECTORIES_PER_REGIME):
                magnitude = signal_scale * (1.0 + 0.04 * (trajectory % 7))
                nuisance_pair = nuisance_shift + nuisance_scale * (
                    ((trajectory % 11) - 5) / 5.0
                )
                nuisance_cycle = 0.4 * nuisance_shift + nuisance_scale * (
                    (((5 * trajectory + code) % 19) - 9) / 9.0
                )
                expected_contexts = np.asarray(
                    [
                        [sign * magnitude, nuisance_pair, nuisance_cycle]
                        for sign in SIGN_PATTERN
                    ],
                    dtype=np.float64,
                )
                stop = cursor + STEPS_PER_TRAJECTORY
                if (
                    np.array_equal(split.contexts[cursor:stop], expected_contexts)
                    and np.array_equal(
                        split.preferred_actions[cursor:stop], expected_preferred
                    )
                    and np.array_equal(
                        split.regime_codes[cursor:stop],
                        np.full(STEPS_PER_TRAJECTORY, code, dtype=np.int32),
                    )
                    and np.array_equal(split.done[cursor:stop], expected_done)
                ):
                    validated += 1
                expected_codes.extend([code] * STEPS_PER_TRAJECTORY)
                expected_keys.extend(
                    f"{split_name}:{code}:{trajectory:02d}:{step}"
                    for step in range(STEPS_PER_TRAJECTORY)
                )
                done_pattern_exact = done_pattern_exact and np.array_equal(
                    split.done[cursor:stop], expected_done
                )
                cursor = stop
        exact_regime_order = exact_regime_order and np.array_equal(
            split.regime_codes, np.asarray(expected_codes, dtype=np.int32)
        )
        exact_key_order = exact_key_order and split.sample_keys == tuple(expected_keys)
        unique_keys = unique_keys and len(set(split.sample_keys)) == len(
            split.sample_keys
        )
    return validated, exact_regime_order, exact_key_order, unique_keys, done_pattern_exact


def _new_state() -> BanditState:
    return BanditState(
        counts=np.zeros((2, 2), dtype=np.int32),
        reward_sums=np.zeros((2, 2), dtype=np.float64),
    )


def _state_commitment(state: BanditState) -> str:
    return _json_sha256(
        {
            "counts": _array_identity(state.counts),
            "pending": state.pending is not None,
            "reward_sums": _array_identity(state.reward_sums),
            "update_count": state.update_count,
        }
    )


def _policy_context(record: dict[str, object]) -> np.ndarray:
    if tuple(record) != POLICY_INPUT_FIELDS:
        raise ValueError("the policy record contains a forbidden input field")
    context = record["context"]
    if (
        not isinstance(context, np.ndarray)
        or context.dtype != np.float64
        or context.shape != (len(CONTEXT_FIELDS),)
        or not np.isfinite(context).all()
    ):
        raise TypeError("the policy record contains a malformed context")
    immutable = context.copy()
    immutable.setflags(write=False)
    return immutable


def _validate_action(action: object) -> np.ndarray:
    if (
        not isinstance(action, np.ndarray)
        or action.dtype != np.int8
        or action.shape != ()
        or int(action) not in (0, 1)
    ):
        raise TypeError("the policy returned a malformed action")
    return action


def _cell_mean(state: BanditState, context_bin: int, action: int) -> float:
    count = int(state.counts[context_bin, action])
    return (
        float(state.reward_sums[context_bin, action]) / count
        if count
        else 0.0
    )


def _greedy_action(state: BanditState, context: np.ndarray) -> np.ndarray:
    context_bin = 1 if float(context[0]) > 0.0 else 0
    means = tuple(_cell_mean(state, context_bin, action) for action in (0, 1))
    return np.asarray(1 if means[1] > means[0] else 0, dtype=np.int8)


def _select(state: BanditState, record: dict[str, object]) -> np.ndarray:
    if state.pending is not None:
        raise RuntimeError("a second selection cannot replace a pending selection")
    context = _policy_context(record)
    context_bin = 1 if float(context[0]) > 0.0 else 0
    exploration = ((state.update_count // STEPS_PER_TRAJECTORY) % 2) == 0
    if exploration:
        action = np.asarray(
            FORCED_ACTION_PATTERN[state.update_count % STEPS_PER_TRAJECTORY],
            dtype=np.int8,
        )
    else:
        action = _greedy_action(state, context)
    action = _validate_action(action)
    state.pending = PendingSelection(
        context_sha256=_context_sha256(context),
        context_bin=context_bin,
        action=int(action),
        update_index=state.update_count,
        exploration=exploration,
    )
    return action


def _update(
    state: BanditState,
    record: dict[str, object],
    action: object,
    reward: object,
) -> PendingSelection:
    if state.pending is None:
        raise RuntimeError("an update requires one pending selection")
    context = _policy_context(record)
    action_value = _validate_action(action)
    if (
        not isinstance(reward, np.ndarray)
        or reward.dtype != np.float64
        or reward.shape != ()
        or not np.isfinite(reward)
        or float(reward) not in (0.0, 1.0)
    ):
        raise TypeError("the policy received a malformed reward")
    pending = state.pending
    if pending.context_sha256 != _context_sha256(context):
        raise ValueError("the update context does not match its selection")
    if pending.action != int(action_value):
        raise ValueError("the update action does not match its selection")
    if pending.update_index != state.update_count:
        raise RuntimeError("the update index changed before reward incorporation")
    counts_before = state.counts.copy()
    sums_before = state.reward_sums.copy()
    state.counts[pending.context_bin, pending.action] += 1
    state.reward_sums[pending.context_bin, pending.action] += float(reward)
    expected_counts = counts_before.copy()
    expected_sums = sums_before.copy()
    expected_counts[pending.context_bin, pending.action] += 1
    expected_sums[pending.context_bin, pending.action] += float(reward)
    if not (
        np.array_equal(state.counts, expected_counts)
        and np.array_equal(state.reward_sums, expected_sums)
    ):
        raise RuntimeError("an update changed more than the selected table cell")
    state.chosen_cell_checks += 1
    state.update_count += 1
    state.pending = None
    return pending


def _train_policy(
    split: SplitData,
    *,
    expected_metadata_sha256: str,
) -> tuple[BanditState, tuple[StepLog, ...], tuple[str, ...]]:
    if _metadata_commitment(split) != expected_metadata_sha256:
        raise ValueError("the updater received unauthenticated evaluator metadata")
    if {key.split(":", 1)[0] for key in split.sample_keys} != {"train"}:
        raise ValueError("the updater may consume only the authenticated train split")
    state = _new_state()
    logs: list[StepLog] = []
    events: list[str] = []
    for index, sample_key in enumerate(split.sample_keys):
        context = split.contexts[index].copy()
        context.setflags(write=False)
        record = {"context": context}
        events.append("context")
        action = _select(state, record)
        pending = state.pending
        if pending is None:
            raise RuntimeError("selection did not create a pending commitment")
        events.append("select")
        reward = np.asarray(
            int(action) == int(split.preferred_actions[index]), dtype=np.float64
        )
        events.append("reward")
        update_before = state.update_count
        incorporated = _update(state, record, action, reward)
        events.append("update")
        logs.append(
            StepLog(
                sample_key=sample_key,
                context_sha256=incorporated.context_sha256,
                action=int(action),
                reward=float(reward),
                update_before=update_before,
                update_after=state.update_count,
                exploration=incorporated.exploration,
                done=bool(split.done[index]),
            )
        )
        events.append("log")
    if state.pending is not None:
        raise RuntimeError("training ended with a pending selection")
    return state, tuple(logs), tuple(events)


def _frozen_logs(state: BanditState, split: SplitData) -> tuple[StepLog, ...]:
    before = _state_commitment(state)
    logs: list[StepLog] = []
    for index, sample_key in enumerate(split.sample_keys):
        context = _policy_context({"context": split.contexts[index].copy()})
        action = _validate_action(_greedy_action(state, context))
        reward = float(int(action) == int(split.preferred_actions[index]))
        logs.append(
            StepLog(
                sample_key=sample_key,
                context_sha256=_context_sha256(context),
                action=int(action),
                reward=reward,
                update_before=state.update_count,
                update_after=state.update_count,
                exploration=False,
                done=bool(split.done[index]),
            )
        )
    if _state_commitment(state) != before or state.pending is not None:
        raise RuntimeError("heldout evaluation mutated the train state")
    return tuple(logs)


def _log_projection(logs: tuple[StepLog, ...]) -> dict[str, object]:
    return {
        "actions": _array_identity(
            np.asarray([entry.action for entry in logs], dtype=np.int8)
        ),
        "contexts_sha256": _json_sha256(
            [entry.context_sha256 for entry in logs]
        ),
        "done": _array_identity(
            np.asarray([entry.done for entry in logs], dtype=np.bool_)
        ),
        "exploration": _array_identity(
            np.asarray([entry.exploration for entry in logs], dtype=np.bool_)
        ),
        "keys_sha256": _json_sha256([entry.sample_key for entry in logs]),
        "rewards": _array_identity(
            np.asarray([entry.reward for entry in logs], dtype=np.float64)
        ),
        "update_after": _array_identity(
            np.asarray([entry.update_after for entry in logs], dtype=np.int32)
        ),
        "update_before": _array_identity(
            np.asarray([entry.update_before for entry in logs], dtype=np.int32)
        ),
    }


def _keyed_action_map(
    actions: np.ndarray,
    action_keys: tuple[str, ...],
    canonical_keys: tuple[str, ...],
) -> dict[str, int]:
    if actions.dtype != np.int8 or actions.shape != (len(action_keys),):
        raise ValueError("the action vector does not match its evaluator keys")
    if len(set(action_keys)) != len(action_keys):
        raise ValueError("evaluator keys must be unique")
    if set(action_keys) != set(canonical_keys):
        raise ValueError("evaluator keys are missing or unexpected")
    return {
        key: int(action) for key, action in zip(action_keys, actions, strict=True)
    }


def _score_actions(
    actions: np.ndarray,
    split: SplitData,
    *,
    action_keys: tuple[str, ...] | None = None,
) -> tuple[float, float, str]:
    keys = split.sample_keys if action_keys is None else action_keys
    keyed = _keyed_action_map(actions, keys, split.sample_keys)
    aligned = np.asarray([keyed[key] for key in split.sample_keys], dtype=np.int8)
    regime_rewards = [
        float(
            np.mean(
                aligned[split.regime_codes == code]
                == split.preferred_actions[split.regime_codes == code]
            )
        )
        for code in sorted({int(value) for value in split.regime_codes})
    ]
    return (
        float(np.mean(regime_rewards)),
        float(min(regime_rewards)),
        _json_sha256(keyed),
    )


def _validate_logs(logs: tuple[StepLog, ...], split: SplitData) -> bool:
    actions = np.asarray([entry.action for entry in logs], dtype=np.int8)
    keys = tuple(entry.sample_key for entry in logs)
    keyed = _keyed_action_map(actions, keys, split.sample_keys)
    by_key = {entry.sample_key: entry for entry in logs}
    if len(by_key) != len(logs):
        raise ValueError("the completed evaluator log contains duplicate keys")
    for index, key in enumerate(split.sample_keys):
        entry = by_key[key]
        expected = float(keyed[key] == int(split.preferred_actions[index]))
        if entry.reward != expected:
            raise ValueError("a completed evaluator log contains a wrong reward")
        if entry.done != bool(split.done[index]):
            raise ValueError("a completed evaluator log contains a wrong boundary")
    return True


def _score_logs(
    logs: tuple[StepLog, ...], split: SplitData
) -> tuple[float, float, str]:
    _validate_logs(logs, split)
    return _score_actions(
        np.asarray([entry.action for entry in logs], dtype=np.int8),
        split,
        action_keys=tuple(entry.sample_key for entry in logs),
    )


def _random_baseline(splits: dict[str, SplitData]) -> dict[str, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(RANDOM_BASELINE_SEED))
    return {
        name: generator.integers(
            0, 2, size=SPLIT_COUNTS[name], dtype=np.int8
        )
        for name in ("validation", "test")
    }


def _positive_gate(
    *,
    train_mean: float,
    train_regret: int,
    validation_macro: float,
    test_macro: float,
    minimum_heldout_regime: float,
    best_constant_validation: float,
    best_constant_test: float,
    random_test: float,
) -> bool:
    return (
        train_mean >= THRESHOLDS["minimum_train_mean_reward"]
        and train_regret <= THRESHOLDS["maximum_train_regret"]
        and validation_macro >= THRESHOLDS["minimum_validation_macro_reward"]
        and test_macro >= THRESHOLDS["minimum_test_macro_reward"]
        and minimum_heldout_regime
        >= THRESHOLDS["minimum_heldout_regime_reward"]
        and validation_macro - best_constant_validation
        >= THRESHOLDS["minimum_validation_gain_constant"]
        and test_macro - best_constant_test
        >= THRESHOLDS["minimum_test_gain_constant"]
        and test_macro - random_test
        >= THRESHOLDS["minimum_test_gain_random"]
    )


def _variant_split(split: SplitData, contexts: np.ndarray) -> SplitData:
    return SplitData(
        contexts=contexts,
        preferred_actions=split.preferred_actions.copy(),
        regime_codes=split.regime_codes.copy(),
        sample_keys=split.sample_keys,
        done=split.done.copy(),
    )


def _shuffled_train(split: SplitData) -> SplitData:
    contexts = (
        split.contexts.reshape(-1, STEPS_PER_TRAJECTORY, len(CONTEXT_FIELDS))[
            :, SHUFFLE_PERMUTATION, :
        ]
        .reshape(-1, len(CONTEXT_FIELDS))
        .copy()
    )
    return _variant_split(split, contexts)


def _ablated_splits(splits: dict[str, SplitData]) -> dict[str, SplitData]:
    result: dict[str, SplitData] = {}
    for name, split in splits.items():
        contexts = split.contexts.copy()
        contexts[:, 0] = 0.0
        result[name] = _variant_split(split, contexts)
    return result


def _event_order_exact(events: tuple[str, ...]) -> bool:
    expected = LOGGING_EVENT_ORDER * SPLIT_COUNTS["train"]
    return events == expected


def _online_sentinel_rejections(train: SplitData) -> tuple[int, int]:
    checked = 7
    rejected = 0
    context = train.contexts[0].copy()
    record = {"context": context}
    action_zero = np.asarray(0, dtype=np.int8)
    reward_one = np.asarray(1.0, dtype=np.float64)

    state = _new_state()
    try:
        _update(state, record, action_zero, reward_one)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    _select(state, record)
    try:
        _select(state, record)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    action = _select(state, record)
    _update(state, record, action, reward_one)
    try:
        _update(state, record, action, reward_one)
    except RuntimeError:
        rejected += 1

    state = _new_state()
    action = _select(state, record)
    wrong_context = context.copy()
    wrong_context[1] += 1.0
    try:
        _update(state, {"context": wrong_context}, action, reward_one)
    except ValueError:
        rejected += 1

    state = _new_state()
    action = _select(state, record)
    wrong_action = np.asarray(1 - int(action), dtype=np.int8)
    try:
        _update(state, record, wrong_action, reward_one)
    except ValueError:
        rejected += 1

    state = _new_state()
    _select(state, record)
    try:
        _update(state, record, np.asarray(0.0), reward_one)
    except TypeError:
        rejected += 1

    state = _new_state()
    action = _select(state, record)
    try:
        _update(state, record, action, np.asarray(0.5, dtype=np.float64))
    except TypeError:
        rejected += 1
    return checked, rejected


def _forbidden_policy_rejections(context: np.ndarray) -> int:
    rejected = 0
    for field in FORBIDDEN_POLICY_FIELDS:
        try:
            _policy_context(
                {"context": context.copy(), field: "forbidden-sentinel"}
            )
        except ValueError:
            rejected += 1
    return rejected


def _train_scope_sentinel_rejections(
    splits: dict[str, SplitData], train_metadata_sha256: str
) -> int:
    rejected = 0
    for name in ("validation", "test"):
        try:
            _train_policy(
                splits[name], expected_metadata_sha256=train_metadata_sha256
            )
        except ValueError:
            rejected += 1
    return rejected


def _log_sentinel_rejections(
    logs: tuple[StepLog, ...], split: SplitData
) -> tuple[int, int, bool]:
    duplicate = list(logs)
    duplicate[1] = duplicate[0]
    duplicate_rejected = 0
    try:
        _score_logs(tuple(duplicate), split)
    except ValueError:
        duplicate_rejected = 1

    wrong_reward = list(logs)
    first = wrong_reward[0]
    wrong_reward[0] = StepLog(
        sample_key=first.sample_key,
        context_sha256=first.context_sha256,
        action=first.action,
        reward=1.0 - first.reward,
        update_before=first.update_before,
        update_after=first.update_after,
        exploration=first.exploration,
        done=first.done,
    )
    wrong_reward_rejected = 0
    try:
        _score_logs(tuple(wrong_reward), split)
    except ValueError:
        wrong_reward_rejected = 1

    canonical_score, _, _ = _score_logs(logs, split)
    swapped = np.asarray([1 - entry.action for entry in logs], dtype=np.int8)
    swapped_score, _, _ = _score_actions(swapped, split)
    return duplicate_rejected, wrong_reward_rejected, swapped_score != canonical_score


def _context_multiset_sha256(contexts: np.ndarray) -> str:
    return _json_sha256(
        sorted(
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in contexts
        )
    )


def _non_process_projection() -> dict[str, object]:
    splits = _generate_all()
    replay = _generate_all()
    dataset_sha256 = _dataset_commitment(splits)
    replay_sha256 = _dataset_commitment(replay)
    train_metadata_sha256 = _metadata_commitment(splits["train"])
    partition_flags = _partition_flags(splits)

    true_state, train_logs, train_events = _train_policy(
        splits["train"], expected_metadata_sha256=train_metadata_sha256
    )
    validation_state_before = _state_commitment(true_state)
    validation_logs = _frozen_logs(true_state, splits["validation"])
    validation_state_after = _state_commitment(true_state)
    test_logs = _frozen_logs(true_state, splits["test"])
    heldout_state_after = _state_commitment(true_state)

    action_values = {entry.action for entry in train_logs}
    reward_values = {entry.reward for entry in train_logs}
    preferred_values = {
        int(value)
        for split in splits.values()
        for value in split.preferred_actions
    }
    done_pattern = np.asarray(
        [False] * (STEPS_PER_TRAJECTORY - 1) + [True], dtype=np.bool_
    )
    immutable_context = _policy_context(
        {"context": splits["train"].contexts[0].copy()}
    )
    immutable_context_rejected = False
    try:
        immutable_context[0] = 0.0
    except ValueError:
        immutable_context_rejected = True
    typed_passed = (
        all(split.contexts.dtype == np.float64 for split in splits.values())
        and all(
            split.contexts.shape == (SPLIT_COUNTS[name], len(CONTEXT_FIELDS))
            for name, split in splits.items()
        )
        and all(
            split.preferred_actions.dtype == np.int8 for split in splits.values()
        )
        and all(split.regime_codes.dtype == np.int32 for split in splits.values())
        and all(split.done.dtype == np.bool_ for split in splits.values())
        and action_values == {0, 1}
        and reward_values == {0.0, 1.0}
        and preferred_values == {0, 1}
        and all(
            np.array_equal(
                split.done.reshape(-1, STEPS_PER_TRAJECTORY)[0], done_pattern
            )
            for split in splits.values()
        )
        and immutable_context_rejected
        and _event_order_exact(train_events)
        and STRUCTURE_KIND == "none"
    )
    typed_case = {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "context_dtype": "float64",
        "context_fields": list(CONTEXT_FIELDS),
        "context_immutable": immutable_context_rejected,
        "context_shape": [3],
        "done_dtype": "bool",
        "done_pattern_exact": bool(
            all(
                np.all(
                    split.done.reshape(-1, STEPS_PER_TRAJECTORY)
                    == done_pattern
                )
                for split in splits.values()
            )
        ),
        "horizon": STEPS_PER_TRAJECTORY,
        "logging_event_order": list(LOGGING_EVENT_ORDER),
        "passed": bool(typed_passed),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "preferred_action_dtype": "int8",
        "preferred_action_values": [0, 1],
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
    }

    (
        validated_trajectories,
        exact_regime_order,
        exact_key_order,
        within_split_keys_unique,
        done_pattern_exact,
    ) = _generator_contract_checks(splits)
    balanced_regimes = 0
    for split in splits.values():
        for code in sorted({int(value) for value in split.regime_codes}):
            values = split.preferred_actions[split.regime_codes == code]
            if int(np.sum(values == 0)) == int(np.sum(values == 1)) == 128:
                balanced_regimes += 1
    context_overlap_count = _cross_split_context_overlap_count(splits)
    generator_passed = (
        dataset_sha256 == EXPECTED_DATASET_SHA256
        and dataset_sha256 == replay_sha256
        and all(partition_flags.values())
        and balanced_regimes == len(REGIMES)
        and validated_trajectories
        == len(REGIMES) * TRAJECTORIES_PER_REGIME
        and exact_regime_order
        and exact_key_order
        and within_split_keys_unique
        and done_pattern_exact
        and context_overlap_count == 0
        and all(np.isfinite(split.contexts).all() for split in splits.values())
    )
    generator_case = {
        "balanced_regimes": balanced_regimes,
        "context_overlap_count": context_overlap_count,
        "contexts_disjoint": partition_flags["contexts"],
        "dataset_sha256": dataset_sha256,
        "deterministic_replay": dataset_sha256 == replay_sha256,
        "done_pattern_exact": done_pattern_exact,
        "exact_key_order": exact_key_order,
        "exact_regime_order": exact_regime_order,
        "expected_dataset_commitment": dataset_sha256
        == EXPECTED_DATASET_SHA256,
        "finite_contexts": bool(
            all(np.isfinite(split.contexts).all() for split in splits.values())
        ),
        "passed": bool(generator_passed),
        "regime_counts": REGIME_COUNTS,
        "regimes_disjoint": partition_flags["regimes"],
        "sample_keys_disjoint": partition_flags["sample_keys"],
        "split_counts": SPLIT_COUNTS,
        "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
        "trajectory_counts": TRAJECTORY_COUNTS,
        "validated_trajectories": validated_trajectories,
        "within_split_keys_unique": within_split_keys_unique,
    }

    select_count = len(train_logs)
    reward_count = len(train_logs)
    update_count = len(train_logs)
    log_count = len(train_logs)
    exploration_count = sum(entry.exploration for entry in train_logs)
    exploitation_count = len(train_logs) - exploration_count
    terminal_rewards_incorporated = sum(
        entry.done and entry.update_after == entry.update_before + 1
        for entry in train_logs
    )
    sentinels_checked, sentinels_rejected = _online_sentinel_rejections(
        splits["train"]
    )
    event_order_exact = _event_order_exact(train_events)
    online_passed = (
        select_count == reward_count == update_count == log_count == 1024
        and exploration_count == exploitation_count == 512
        and true_state.update_count == 1024
        and true_state.chosen_cell_checks == 1024
        and terminal_rewards_incorporated == 128
        and event_order_exact
        and sentinels_checked == sentinels_rejected == 7
        and all(
            entry.update_after == entry.update_before + 1
            for entry in train_logs
        )
    )
    online_case = {
        "chosen_cell_only_checks": true_state.chosen_cell_checks,
        "event_order_exact": event_order_exact,
        "exploitation_count": exploitation_count,
        "exploration_count": exploration_count,
        "final_update_count": true_state.update_count,
        "log_count": log_count,
        "passed": bool(online_passed),
        "reward_count": reward_count,
        "select_count": select_count,
        "sentinels_checked": sentinels_checked,
        "sentinels_rejected": sentinels_rejected,
        "terminal_rewards_incorporated": terminal_rewards_incorporated,
        "trace_sha256": _json_sha256(_log_projection(train_logs)),
        "update_count": update_count,
    }

    forbidden_rejected = _forbidden_policy_rejections(
        splits["train"].contexts[0]
    )
    scope_rejected = _train_scope_sentinel_rejections(
        splits, train_metadata_sha256
    )
    canonical_validation = _score_logs(validation_logs, splits["validation"])
    canonical_test = _score_logs(test_logs, splits["test"])
    reversed_validation = _score_logs(
        tuple(reversed(validation_logs)), splits["validation"]
    )
    reversed_test = _score_logs(tuple(reversed(test_logs)), splits["test"])
    reorder_exact = (
        canonical_validation == reversed_validation
        and canonical_test == reversed_test
    )
    duplicate_rejected, wrong_reward_rejected, swapped_changed = (
        _log_sentinel_rejections(test_logs, splits["test"])
    )
    heldout_updates = true_state.update_count - SPLIT_COUNTS["train"]
    state_unchanged = (
        validation_state_before
        == validation_state_after
        == heldout_state_after
    )
    leakage_passed = (
        forbidden_rejected == len(FORBIDDEN_POLICY_FIELDS)
        and scope_rejected == 2
        and heldout_updates == 0
        and state_unchanged
        and reorder_exact
        and duplicate_rejected == wrong_reward_rejected == 1
        and swapped_changed
    )
    leakage_case = {
        "canonical_reorder_exact": reorder_exact,
        "duplicate_key_sentinel_rejected": bool(duplicate_rejected),
        "forbidden_fields_checked": len(FORBIDDEN_POLICY_FIELDS),
        "forbidden_fields_rejected": forbidden_rejected,
        "heldout_state_unchanged": state_unchanged,
        "heldout_updates": heldout_updates,
        "passed": bool(leakage_passed),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "swapped_action_score_changed": swapped_changed,
        "trace_sha256": _json_sha256(
            {
                "test": _log_projection(test_logs),
                "validation": _log_projection(validation_logs),
            }
        ),
        "train_scope_sentinels_checked": 2,
        "train_scope_sentinels_rejected": scope_rejected,
        "wrong_reward_sentinel_rejected": bool(wrong_reward_rejected),
    }

    constant_scores: dict[int, dict[str, float]] = {}
    for action in (0, 1):
        constant_scores[action] = {}
        for name in ("validation", "test"):
            score, _, _ = _score_actions(
                np.full(SPLIT_COUNTS[name], action, dtype=np.int8),
                splits[name],
            )
            constant_scores[action][name] = score
    best_constant_validation = max(
        constant_scores[action]["validation"] for action in (0, 1)
    )
    best_constant_test = max(
        constant_scores[action]["test"] for action in (0, 1)
    )
    random_actions = _random_baseline(splits)
    random_replay = _random_baseline(splits)
    random_validation, _, _ = _score_actions(
        random_actions["validation"], splits["validation"]
    )
    random_test, _, _ = _score_actions(random_actions["test"], splits["test"])
    random_replay_exact = all(
        np.array_equal(random_actions[name], random_replay[name])
        for name in ("validation", "test")
    )
    heldout_commitment = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    replay_heldout_commitment = _json_sha256(
        {
            name: _split_projection(replay[name])
            for name in ("validation", "test")
        }
    )
    same_evaluation_rows = heldout_commitment == replay_heldout_commitment
    baseline_values = (
        constant_scores[0]["validation"],
        constant_scores[1]["validation"],
        constant_scores[0]["test"],
        constant_scores[1]["test"],
        random_validation,
        random_test,
    )
    baseline_case = {
        "best_constant_test_macro_reward": best_constant_test,
        "best_constant_validation_macro_reward": best_constant_validation,
        "constant_one_test_macro_reward": constant_scores[1]["test"],
        "constant_one_validation_macro_reward": constant_scores[1]["validation"],
        "constant_zero_test_macro_reward": constant_scores[0]["test"],
        "constant_zero_validation_macro_reward": constant_scores[0]["validation"],
        "finite_metrics": bool(all(np.isfinite(value) for value in baseline_values)),
        "passed": bool(
            random_replay_exact
            and same_evaluation_rows
            and all(np.isfinite(value) for value in baseline_values)
        ),
        "random_seed": RANDOM_BASELINE_SEED,
        "random_test_macro_reward": random_test,
        "random_validation_macro_reward": random_validation,
        "replay_exact": random_replay_exact,
        "same_evaluation_rows": same_evaluation_rows,
        "trace_sha256": _json_sha256(
            {
                "random_test": _array_identity(random_actions["test"]),
                "random_validation": _array_identity(random_actions["validation"]),
                "rows_sha256": heldout_commitment,
            }
        ),
    }

    train_reward_total = int(sum(entry.reward for entry in train_logs))
    train_mean = train_reward_total / SPLIT_COUNTS["train"]
    train_regret = SPLIT_COUNTS["train"] - train_reward_total
    validation_macro, validation_minimum, _ = canonical_validation
    test_macro, test_minimum, _ = canonical_test
    minimum_heldout = min(validation_minimum, test_minimum)
    recovery_passed = _positive_gate(
        train_mean=train_mean,
        train_regret=train_regret,
        validation_macro=validation_macro,
        test_macro=test_macro,
        minimum_heldout_regime=minimum_heldout,
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        random_test=random_test,
    )
    recovery_case = {
        "exploitation_count": exploitation_count,
        "exploration_count": exploration_count,
        "learned_projection_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(true_state),
                "test": _log_projection(test_logs),
                "train": _log_projection(train_logs),
                "validation": _log_projection(validation_logs),
            }
        ),
        "min_heldout_regime_reward": minimum_heldout,
        "passed": bool(recovery_passed),
        "test_gain_over_constant": test_macro - best_constant_test,
        "test_gain_over_random": test_macro - random_test,
        "test_macro_reward": test_macro,
        "train_cumulative_regret": train_regret,
        "train_mean_reward": train_mean,
        "validation_gain_over_constant": validation_macro
        - best_constant_validation,
        "validation_macro_reward": validation_macro,
    }

    shuffled_train = _shuffled_train(splits["train"])
    shuffled_state, shuffled_train_logs, _ = _train_policy(
        shuffled_train, expected_metadata_sha256=train_metadata_sha256
    )
    shuffled_validation_logs = _frozen_logs(
        shuffled_state, splits["validation"]
    )
    shuffled_test_logs = _frozen_logs(shuffled_state, splits["test"])
    shuffled_validation, shuffled_validation_minimum, _ = _score_logs(
        shuffled_validation_logs, splits["validation"]
    )
    shuffled_test, shuffled_test_minimum, _ = _score_logs(
        shuffled_test_logs, splits["test"]
    )
    shuffled_train_total = int(sum(entry.reward for entry in shuffled_train_logs))
    shuffled_train_mean = shuffled_train_total / SPLIT_COUNTS["train"]
    shuffled_train_regret = SPLIT_COUNTS["train"] - shuffled_train_total
    shuffled_positive_gate = _positive_gate(
        train_mean=shuffled_train_mean,
        train_regret=shuffled_train_regret,
        validation_macro=shuffled_validation,
        test_macro=shuffled_test,
        minimum_heldout_regime=min(
            shuffled_validation_minimum, shuffled_test_minimum
        ),
        best_constant_validation=best_constant_validation,
        best_constant_test=best_constant_test,
        random_test=random_test,
    )
    context_multiset_unchanged = _context_multiset_sha256(
        shuffled_train.contexts
    ) == _context_multiset_sha256(splits["train"].contexts)
    rowwise_context_changed = not np.array_equal(
        shuffled_train.contexts, splits["train"].contexts
    )
    metadata_unchanged = _metadata_commitment(shuffled_train) == train_metadata_sha256
    heldout_control_commitment = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    heldout_unchanged = heldout_control_commitment == heldout_commitment
    independence_table_exact = True
    shaped_contexts = shuffled_train.contexts.reshape(
        -1, STEPS_PER_TRAJECTORY, len(CONTEXT_FIELDS)
    )
    shaped_targets = shuffled_train.preferred_actions.reshape(
        -1, STEPS_PER_TRAJECTORY
    )
    for trajectory, (contexts, targets) in enumerate(
        zip(shaped_contexts, shaped_targets, strict=True)
    ):
        bins = (contexts[:, 0] > 0.0).astype(np.int8)
        for context_bin in (0, 1):
            bin_targets = targets[bins == context_bin]
            independence_table_exact = independence_table_exact and (
                int(np.sum(bin_targets == 0))
                == int(np.sum(bin_targets == 1))
                == 2
            )
            if trajectory % 2 == 0:
                forced = np.asarray(FORCED_ACTION_PATTERN, dtype=np.int8)
                for action in (0, 1):
                    cell_targets = targets[(bins == context_bin) & (forced == action)]
                    independence_table_exact = independence_table_exact and (
                        int(np.sum(cell_targets == 0))
                        == int(np.sum(cell_targets == 1))
                        == 1
                    )
    true_shuffle_gap = test_macro - shuffled_test
    shuffle_passed = (
        context_multiset_unchanged
        and rowwise_context_changed
        and metadata_unchanged
        and heldout_unchanged
        and independence_table_exact
        and not shuffled_positive_gate
        and shuffled_validation
        <= THRESHOLDS["maximum_shuffle_validation_macro_reward"]
        and shuffled_test <= THRESHOLDS["maximum_shuffle_test_macro_reward"]
        and true_shuffle_gap >= THRESHOLDS["minimum_true_shuffle_gap"]
    )
    shuffle_case = {
        "context_multiset_unchanged": context_multiset_unchanged,
        "heldout_commitment_unchanged": heldout_unchanged,
        "independence_table_exact": bool(independence_table_exact),
        "metadata_commitment_unchanged": metadata_unchanged,
        "nonidentity_permutation": SHUFFLE_PERMUTATION != tuple(
            range(STEPS_PER_TRAJECTORY)
        ),
        "passed": bool(shuffle_passed),
        "positive_gate_rejected": not shuffled_positive_gate,
        "rowwise_context_commitment_changed": rowwise_context_changed,
        "test_macro_reward": shuffled_test,
        "trace_sha256": _json_sha256(
            {
                "state_sha256": _state_commitment(shuffled_state),
                "test": _log_projection(shuffled_test_logs),
                "train": _log_projection(shuffled_train_logs),
                "validation": _log_projection(shuffled_validation_logs),
            }
        ),
        "true_reward_gap": true_shuffle_gap,
        "validation_macro_reward": shuffled_validation,
    }

    ablated = _ablated_splits(splits)
    only_signal_changed = bool(
        all(
            np.all(ablated[name].contexts[:, 0] == 0.0)
            and np.array_equal(
                ablated[name].contexts[:, 1:], splits[name].contexts[:, 1:]
            )
            and np.any(splits[name].contexts[:, 0] != 0.0)
            for name in splits
        )
    )
    attribution_metadata_unchanged = all(
        _metadata_commitment(ablated[name]) == _metadata_commitment(splits[name])
        for name in splits
    )
    ablated_state, ablated_train_logs, _ = _train_policy(
        ablated["train"], expected_metadata_sha256=train_metadata_sha256
    )
    ablated_test_logs = _frozen_logs(ablated_state, ablated["test"])
    true_zeroed_test_logs = _frozen_logs(true_state, ablated["test"])
    ablated_test_macro, _, _ = _score_logs(
        ablated_test_logs, ablated["test"]
    )
    true_zeroed_test_macro, _, _ = _score_logs(
        true_zeroed_test_logs, ablated["test"]
    )
    attribution_passed = (
        only_signal_changed
        and attribution_metadata_unchanged
        and ablated_test_macro
        <= THRESHOLDS["maximum_attribution_test_macro_reward"]
        and true_zeroed_test_macro
        <= THRESHOLDS["maximum_attribution_test_macro_reward"]
    )
    attribution_case = {
        "evaluator_metadata_unchanged": attribution_metadata_unchanged,
        "only_signal_changed": only_signal_changed,
        "passed": bool(attribution_passed),
        "refit_test_macro_reward": ablated_test_macro,
        "trace_sha256": _json_sha256(
            {
                "refit_state_sha256": _state_commitment(ablated_state),
                "refit_test": _log_projection(ablated_test_logs),
                "refit_train": _log_projection(ablated_train_logs),
                "true_zeroed_test": _log_projection(true_zeroed_test_logs),
            }
        ),
        "true_policy_zeroed_test_macro_reward": true_zeroed_test_macro,
    }

    return {
        "baseline_replay": baseline_case,
        "contextual_recovery": recovery_case,
        "generator_partition": generator_case,
        "leakage_guards": leakage_case,
        "online_update_order": online_case,
        "shuffled_context_control": shuffle_case,
        "signal_attribution_control": attribution_case,
        "typed_bandit_contract": typed_case,
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
            "experiments.local_lab.contextual_bandit_toy_signal_worker",
            "--mode",
            "contextual-bandit-toy-signal-trace",
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
        raise RuntimeError("the contextual-bandit study requires a CPU backend")

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
            "synthetic_contextual_bandit_signal_recovered_for_harness"
            if passed
            else (
                "park_contextual_bandit_research"
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
