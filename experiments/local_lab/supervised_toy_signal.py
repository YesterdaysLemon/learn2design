"""Deterministic supervised toy-signal learning-contract fixture."""

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


STUDY_ID = "supervised-toy-signal-v1"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).parents[2]
OBSERVATION_FIELDS = (
    "signal",
    "nuisance_pair",
    "nuisance_regime",
    "nuisance_cycle",
)
POLICY_INPUT_FIELDS = ("observation",)
STRUCTURE_KIND = "none"
FORBIDDEN_POLICY_FIELDS = (
    "target",
    "reward",
    "regime_code",
    "split",
    "sample_key",
)
BLOCKS_PER_REGIME = 32
ROWS_PER_BLOCK = 4
TRAJECTORIES_PER_REGIME = BLOCKS_PER_REGIME * ROWS_PER_BLOCK
RIDGE_COEFFICIENT = 1e-6
RANDOM_BASELINE_SEED = 2026082707
LABEL_SHUFFLE_PERMUTATION = (0, 2, 1, 3)
EXPECTED_DATASET_SHA256 = (
    "f6af01db64aab94363e964f73b8e46b9341b680f63338f9fdff059ed14f1a38e"
)
REGIMES = (
    {
        "code": 101,
        "nuisance_scale": 0.90,
        "nuisance_shift": -0.75,
        "signal_scale": 0.80,
        "split": "train",
    },
    {
        "code": 103,
        "nuisance_scale": 1.10,
        "nuisance_shift": -0.25,
        "signal_scale": 0.95,
        "split": "train",
    },
    {
        "code": 107,
        "nuisance_scale": 0.80,
        "nuisance_shift": 0.25,
        "signal_scale": 1.05,
        "split": "train",
    },
    {
        "code": 109,
        "nuisance_scale": 1.20,
        "nuisance_shift": 0.75,
        "signal_scale": 1.20,
        "split": "train",
    },
    {
        "code": 211,
        "nuisance_scale": 0.70,
        "nuisance_shift": -1.25,
        "signal_scale": 0.70,
        "split": "validation",
    },
    {
        "code": 223,
        "nuisance_scale": 1.30,
        "nuisance_shift": 1.25,
        "signal_scale": 1.30,
        "split": "validation",
    },
    {
        "code": 307,
        "nuisance_scale": 0.60,
        "nuisance_shift": -1.75,
        "signal_scale": 0.60,
        "split": "test",
    },
    {
        "code": 331,
        "nuisance_scale": 1.40,
        "nuisance_shift": 1.75,
        "signal_scale": 1.40,
        "split": "test",
    },
)
REGIME_COUNTS = {"test": 2, "train": 4, "validation": 2}
SPLIT_COUNTS = {
    name: count * TRAJECTORIES_PER_REGIME
    for name, count in REGIME_COUNTS.items()
}
THRESHOLDS = {
    "minimum_heldout_regime_accuracy": 0.98,
    "minimum_test_gain_constant": 0.30,
    "minimum_test_gain_random": 0.25,
    "minimum_train_macro_accuracy": 0.99,
    "minimum_validation_gain_constant": 0.30,
    "minimum_validation_macro_accuracy": 0.99,
    "minimum_test_macro_accuracy": 0.99,
    "maximum_attribution_macro_accuracy": 0.55,
    "maximum_shuffle_macro_accuracy": 0.55,
    "minimum_true_shuffle_gap": 0.40,
}
CASE_CONTRACT = {
    "typed_task_contract": {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "done": True,
        "horizon": 1,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [4],
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
        "target_dtype": "int8",
        "target_values": [0, 1],
    },
    "generator_partition": {
        "blocks_per_regime": BLOCKS_PER_REGIME,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "regime_counts": REGIME_COUNTS,
        "regimes": list(REGIMES),
        "rows_per_block": ROWS_PER_BLOCK,
        "split_counts": SPLIT_COUNTS,
        "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
    },
    "leakage_guards": {
        "forbidden_policy_fields": list(FORBIDDEN_POLICY_FIELDS),
        "overlap_sentinels": ["regime", "sample_key", "observation"],
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reorder": "reverse_each_heldout_split",
    },
    "baseline_replay": {
        "constant_action": 0,
        "random_seed": RANDOM_BASELINE_SEED,
        "scoring": "macro_accuracy_equal_weight_by_regime",
    },
    "supervised_recovery": {
        "fit_split": "train_only",
        "model": "float64_linear_ridge_score",
        "ridge_coefficient": RIDGE_COEFFICIENT,
        "thresholds": THRESHOLDS,
        "tie_action": 0,
    },
    "label_shuffle_control": {
        "permutation": list(LABEL_SHUFFLE_PERMUTATION),
        "scope": "train_labels_within_each_four_row_block",
        "thresholds": {
            "maximum_shuffle_macro_accuracy": THRESHOLDS[
                "maximum_shuffle_macro_accuracy"
            ],
            "minimum_true_shuffle_gap": THRESHOLDS["minimum_true_shuffle_gap"],
        },
    },
    "signal_attribution_control": {
        "maximum_macro_accuracy": THRESHOLDS[
            "maximum_attribution_macro_accuracy"
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
    "blocks_per_regime": BLOCKS_PER_REGIME,
    "claim_boundary": "synthetic_cpu_supervised_harness_only",
    "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
    "generator_regimes": list(REGIMES),
    "horizon": 1,
    "label_shuffle_permutation": list(LABEL_SHUFFLE_PERMUTATION),
    "observation_dtype": "float64",
    "observation_fields": list(OBSERVATION_FIELDS),
    "observation_shape": [4],
    "policy_input_fields": list(POLICY_INPUT_FIELDS),
    "regime_counts": REGIME_COUNTS,
    "reward_dtype": "float64",
    "reward_values": [0.0, 1.0],
    "ridge_coefficient": RIDGE_COEFFICIENT,
    "rows_per_block": ROWS_PER_BLOCK,
    "split_counts": SPLIT_COUNTS,
    "target_dtype": "int8",
    "target_values": [0, 1],
    "thresholds": THRESHOLDS,
    "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
    "structure_kind": STRUCTURE_KIND,
}


@dataclass(frozen=True)
class SplitData:
    observations: np.ndarray
    targets: np.ndarray
    regime_codes: np.ndarray
    sample_keys: tuple[str, ...]
    done: np.ndarray


@dataclass(frozen=True)
class OneStepTransition:
    observation: np.ndarray
    action: np.ndarray
    target: np.ndarray
    reward: np.ndarray
    done: np.ndarray


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


def _split_projection(split: SplitData) -> dict[str, object]:
    return {
        "keys_sha256": _json_sha256(list(split.sample_keys)),
        "observations": _array_identity(split.observations),
        "regime_codes": _array_identity(split.regime_codes),
        "targets": _array_identity(split.targets),
        "done": _array_identity(split.done),
    }


def _generate_split(split_name: str) -> SplitData:
    observations: list[list[float]] = []
    targets: list[int] = []
    regime_codes: list[int] = []
    sample_keys: list[str] = []
    for regime in REGIMES:
        if regime["split"] != split_name:
            continue
        code = int(regime["code"])
        signal_scale = float(regime["signal_scale"])
        nuisance_shift = float(regime["nuisance_shift"])
        nuisance_scale = float(regime["nuisance_scale"])
        for block in range(BLOCKS_PER_REGIME):
            magnitude = signal_scale * (1.0 + 0.05 * (block % 7))
            nuisance_pair = nuisance_shift + nuisance_scale * (
                ((block % 9) - 4) / 4.0
            )
            nuisance_regime = nuisance_scale * (
                (((5 * block + code) % 13) - 6) / 6.0
            )
            nuisance_cycle = 0.5 * nuisance_shift + nuisance_scale * (
                (((7 * block + code) % 17) - 8) / 8.0
            )
            for row, sign in enumerate((-1.0, 1.0, -1.0, 1.0)):
                observations.append(
                    [
                        sign * magnitude,
                        nuisance_pair,
                        nuisance_regime,
                        nuisance_cycle,
                    ]
                )
                targets.append(1 if sign > 0 else 0)
                regime_codes.append(code)
                sample_keys.append(f"{split_name}:{code}:{block:02d}:{row}")
    return SplitData(
        observations=np.asarray(observations, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.int8),
        regime_codes=np.asarray(regime_codes, dtype=np.int32),
        sample_keys=tuple(sample_keys),
        done=np.ones(len(targets), dtype=np.bool_),
    )


def _generate_all() -> dict[str, SplitData]:
    return {
        name: _generate_split(name) for name in ("train", "validation", "test")
    }


def _row_hashes(observations: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in observations
    }


def _partition_flags(splits: dict[str, SplitData]) -> dict[str, bool]:
    names = ("train", "validation", "test")
    regime_sets = {
        name: {int(value) for value in splits[name].regime_codes} for name in names
    }
    key_sets = {name: set(splits[name].sample_keys) for name in names}
    observation_sets = {
        name: _row_hashes(splits[name].observations) for name in names
    }

    def pairwise_disjoint(values: dict[str, set[object]]) -> bool:
        return all(
            values[left].isdisjoint(values[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )

    return {
        "observations": pairwise_disjoint(observation_sets),
        "regimes": pairwise_disjoint(regime_sets),
        "sample_keys": pairwise_disjoint(key_sets),
    }


def _cross_split_observation_overlap_count(
    splits: dict[str, SplitData],
) -> int:
    names = ("train", "validation", "test")
    observation_sets = {
        name: _row_hashes(splits[name].observations) for name in names
    }
    return sum(
        len(observation_sets[left] & observation_sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _generator_contract_checks(
    splits: dict[str, SplitData],
) -> tuple[int, bool, bool, bool]:
    """Validate every generated row against the independently expanded contract."""
    expected_targets = np.asarray([0, 1, 0, 1], dtype=np.int8)
    expected_signs = (-1.0, 1.0, -1.0, 1.0)
    validated_blocks = 0
    exact_regime_order = True
    exact_sample_key_order = True
    within_split_keys_unique = True

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
            for block in range(BLOCKS_PER_REGIME):
                magnitude = signal_scale * (1.0 + 0.05 * (block % 7))
                nuisance_pair = nuisance_shift + nuisance_scale * (
                    ((block % 9) - 4) / 4.0
                )
                nuisance_regime = nuisance_scale * (
                    (((5 * block + code) % 13) - 6) / 6.0
                )
                nuisance_cycle = 0.5 * nuisance_shift + nuisance_scale * (
                    (((7 * block + code) % 17) - 8) / 8.0
                )
                expected_observations = np.asarray(
                    [
                        [
                            sign * magnitude,
                            nuisance_pair,
                            nuisance_regime,
                            nuisance_cycle,
                        ]
                        for sign in expected_signs
                    ],
                    dtype=np.float64,
                )
                stop = cursor + ROWS_PER_BLOCK
                if (
                    np.array_equal(split.observations[cursor:stop], expected_observations)
                    and np.array_equal(split.targets[cursor:stop], expected_targets)
                    and np.array_equal(
                        split.regime_codes[cursor:stop],
                        np.full(ROWS_PER_BLOCK, code, dtype=np.int32),
                    )
                    and bool(np.all(split.done[cursor:stop]))
                ):
                    validated_blocks += 1
                expected_codes.extend([code] * ROWS_PER_BLOCK)
                expected_keys.extend(
                    f"{split_name}:{code}:{block:02d}:{row}"
                    for row in range(ROWS_PER_BLOCK)
                )
                cursor = stop

        exact_regime_order = exact_regime_order and np.array_equal(
            split.regime_codes,
            np.asarray(expected_codes, dtype=np.int32),
        )
        exact_sample_key_order = exact_sample_key_order and (
            split.sample_keys == tuple(expected_keys)
        )
        within_split_keys_unique = within_split_keys_unique and (
            len(set(split.sample_keys)) == len(split.sample_keys)
        )

    return (
        validated_blocks,
        bool(exact_regime_order),
        bool(exact_sample_key_order),
        bool(within_split_keys_unique),
    )


def _overlap_sentinel_rejections(splits: dict[str, SplitData]) -> int:
    train = splits["train"]
    test = splits["test"]

    regime_codes = test.regime_codes.copy()
    regime_codes[0] = train.regime_codes[0]
    regime_sentinel = {
        **splits,
        "test": SplitData(
            observations=test.observations,
            targets=test.targets,
            regime_codes=regime_codes,
            sample_keys=test.sample_keys,
            done=test.done,
        ),
    }

    sample_keys = (train.sample_keys[0], *test.sample_keys[1:])
    key_sentinel = {
        **splits,
        "test": SplitData(
            observations=test.observations,
            targets=test.targets,
            regime_codes=test.regime_codes,
            sample_keys=sample_keys,
            done=test.done,
        ),
    }

    observations = test.observations.copy()
    observations[0] = train.observations[0]
    observation_sentinel = {
        **splits,
        "test": SplitData(
            observations=observations,
            targets=test.targets,
            regime_codes=test.regime_codes,
            sample_keys=test.sample_keys,
            done=test.done,
        ),
    }

    return sum(
        (
            not _partition_flags(regime_sentinel)["regimes"],
            not _partition_flags(key_sentinel)["sample_keys"],
            not _partition_flags(observation_sentinel)["observations"],
        )
    )


def _policy_observation(record: dict[str, object]) -> np.ndarray:
    if tuple(record) != POLICY_INPUT_FIELDS:
        raise ValueError("the policy record contains a forbidden input field")
    observation = record["observation"]
    if (
        not isinstance(observation, np.ndarray)
        or observation.dtype != np.float64
        or observation.shape != (len(OBSERVATION_FIELDS),)
        or not np.isfinite(observation).all()
    ):
        raise TypeError("the policy record contains a malformed observation")
    return observation


def _one_step_transition(
    split: SplitData,
    *,
    index: int,
    action: int,
) -> OneStepTransition:
    action_value = np.asarray(action, dtype=np.int8)
    target_value = np.asarray(split.targets[index], dtype=np.int8)
    reward_value = np.asarray(action_value == target_value, dtype=np.float64)
    return OneStepTransition(
        observation=_policy_observation(
            {"observation": split.observations[index].copy()}
        ),
        action=action_value,
        target=target_value,
        reward=reward_value,
        done=np.asarray(split.done[index], dtype=np.bool_),
    )


def _fit_source_commitment(
    split: SplitData,
    observations: np.ndarray,
) -> str:
    return _json_sha256(
        {
            "done": _array_identity(split.done),
            "keys_sha256": _json_sha256(list(split.sample_keys)),
            "observations": _array_identity(observations),
            "regime_codes": _array_identity(split.regime_codes),
        }
    )


def _fit_policy(
    split: SplitData,
    targets: np.ndarray,
    *,
    observations: np.ndarray | None = None,
    expected_source_sha256: str,
    expected_target_sha256: str,
    variant: str,
    audit_log: list[dict[str, str]],
) -> np.ndarray:
    fit_observations = split.observations if observations is None else observations
    source_sha256 = _fit_source_commitment(split, fit_observations)
    target_sha256 = _json_sha256(_array_identity(targets))
    key_scopes = {key.split(":", 1)[0] for key in split.sample_keys}
    if key_scopes != {"train"}:
        raise ValueError("the frozen learner may fit only on authenticated train rows")
    if source_sha256 != expected_source_sha256:
        raise ValueError("the fit source does not match its frozen commitment")
    if target_sha256 != expected_target_sha256:
        raise ValueError("the fit targets do not match their frozen commitment")
    if fit_observations.dtype != np.float64 or targets.dtype != np.int8:
        raise TypeError("the frozen learner requires float64 observations and int8 targets")
    if (
        fit_observations.ndim != 2
        or fit_observations.shape[1] != len(OBSERVATION_FIELDS)
    ):
        raise ValueError("the frozen learner received the wrong observation shape")
    if targets.shape != (fit_observations.shape[0],):
        raise ValueError("the frozen learner received the wrong target shape")
    design = np.concatenate(
        [
            np.ones((fit_observations.shape[0], 1), dtype=np.float64),
            fit_observations,
        ],
        axis=1,
    )
    encoded_targets = targets.astype(np.float64) * 2.0 - 1.0
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_COEFFICIENT
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ encoded_targets,
    )
    audit_log.append(
        {
            "scope": "train",
            "source_sha256": source_sha256,
            "target_sha256": target_sha256,
            "variant": variant,
        }
    )
    return weights


def _predict(observations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [np.ones((observations.shape[0], 1), dtype=np.float64), observations],
        axis=1,
    )
    return (design @ weights > 0.0).astype(np.int8)


def _reward(actions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (actions == targets).astype(np.float64)


def _keyed_action_map(
    actions: np.ndarray,
    action_keys: tuple[str, ...],
) -> dict[str, int]:
    if actions.shape != (len(action_keys),):
        raise ValueError("action count does not match the canonical sample keys")
    if len(set(action_keys)) != len(action_keys):
        raise ValueError("canonical sample keys must be unique within a split")
    return {
        key: int(action) for key, action in zip(action_keys, actions, strict=True)
    }


def _macro_accuracy(
    actions: np.ndarray,
    split: SplitData,
    *,
    action_keys: tuple[str, ...] | None = None,
) -> tuple[float, float]:
    keyed = _keyed_action_map(
        actions,
        split.sample_keys if action_keys is None else action_keys,
    )
    aligned_actions = np.asarray(
        [keyed[key] for key in split.sample_keys],
        dtype=np.int8,
    )
    regime_accuracies = [
        float(
            np.mean(
                _reward(
                    aligned_actions[split.regime_codes == code],
                    split.targets[split.regime_codes == code],
                )
            )
        )
        for code in sorted({int(value) for value in split.regime_codes})
    ]
    return float(np.mean(regime_accuracies)), float(min(regime_accuracies))


def _accuracy_by_key(
    actions: np.ndarray,
    split: SplitData,
    *,
    action_keys: tuple[str, ...] | None = None,
) -> tuple[float, str]:
    keyed = _keyed_action_map(
        actions,
        split.sample_keys if action_keys is None else action_keys,
    )
    score = float(
        np.mean(
            [
                keyed[key] == int(target)
                for key, target in zip(split.sample_keys, split.targets, strict=True)
            ]
        )
    )
    return score, _json_sha256(keyed)


def _reversed_accuracy_by_key(
    weights: np.ndarray,
    split: SplitData,
) -> tuple[float, str]:
    reversed_observations = split.observations[::-1].copy()
    reversed_actions = _predict(reversed_observations, weights)
    reversed_keys = tuple(reversed(split.sample_keys))
    score, digest = _accuracy_by_key(
        reversed_actions,
        split,
        action_keys=reversed_keys,
    )
    macro, _ = _macro_accuracy(
        reversed_actions,
        split,
        action_keys=reversed_keys,
    )
    if score != macro:
        raise RuntimeError("keyed aggregate and macro accuracy disagree")
    return score, digest


def _random_baseline(splits: dict[str, SplitData]) -> dict[str, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(RANDOM_BASELINE_SEED))
    return {
        name: generator.integers(
            0,
            2,
            size=splits[name].targets.shape[0],
            dtype=np.int8,
        )
        for name in ("validation", "test")
    }


def _positive_gate(
    *,
    train_macro: float,
    validation_macro: float,
    test_macro: float,
    minimum_heldout_regime: float,
    constant_validation_macro: float,
    constant_test_macro: float,
    random_test_macro: float,
) -> bool:
    return (
        train_macro >= THRESHOLDS["minimum_train_macro_accuracy"]
        and validation_macro >= THRESHOLDS["minimum_validation_macro_accuracy"]
        and test_macro >= THRESHOLDS["minimum_test_macro_accuracy"]
        and minimum_heldout_regime
        >= THRESHOLDS["minimum_heldout_regime_accuracy"]
        and validation_macro - constant_validation_macro
        >= THRESHOLDS["minimum_validation_gain_constant"]
        and test_macro - constant_test_macro
        >= THRESHOLDS["minimum_test_gain_constant"]
        and test_macro - random_test_macro
        >= THRESHOLDS["minimum_test_gain_random"]
    )


def _shuffled_targets(targets: np.ndarray) -> np.ndarray:
    return (
        targets.reshape(-1, ROWS_PER_BLOCK)[:, LABEL_SHUFFLE_PERMUTATION]
        .reshape(-1)
        .astype(np.int8)
    )


def _dataset_commitment(splits: dict[str, SplitData]) -> str:
    return _json_sha256(
        {name: _split_projection(split) for name, split in sorted(splits.items())}
    )


def _non_process_projection() -> dict[str, object]:
    splits = _generate_all()
    replay = _generate_all()
    fit_audit: list[dict[str, str]] = []
    dataset_sha256 = _dataset_commitment(splits)
    replay_sha256 = _dataset_commitment(replay)
    partition_flags = _partition_flags(splits)

    transition_zero = _one_step_transition(splits["train"], index=0, action=0)
    transition_one = _one_step_transition(splits["train"], index=0, action=1)
    action_probe = np.asarray(
        [transition_zero.action, transition_one.action],
        dtype=np.int8,
    )
    reward_probe = np.asarray(
        [transition_zero.reward, transition_one.reward],
        dtype=np.float64,
    )
    done_valid = all(
        split.done.dtype == np.bool_
        and split.done.shape == split.targets.shape
        and bool(np.all(split.done))
        for split in splits.values()
    )
    typed_passed = (
        all(split.observations.dtype == np.float64 for split in splits.values())
        and all(
            split.observations.shape == (SPLIT_COUNTS[name], 4)
            for name, split in splits.items()
        )
        and all(split.targets.dtype == np.int8 for split in splits.values())
        and all(
            set(int(value) for value in split.targets) == {0, 1}
            for split in splits.values()
        )
        and action_probe.dtype == np.int8
        and set(int(value) for value in action_probe) == {0, 1}
        and reward_probe.dtype == np.float64
        and set(float(value) for value in reward_probe) == {0.0, 1.0}
        and done_valid
        and transition_zero.observation.shape == (4,)
        and transition_zero.action.shape == ()
        and transition_zero.target.shape == ()
        and transition_zero.reward.shape == ()
        and transition_zero.done.shape == ()
        and bool(transition_zero.done)
        and STRUCTURE_KIND == "none"
    )
    typed_case = {
        "action_dtype": "int8",
        "action_values": [0, 1],
        "done": done_valid,
        "horizon": 1,
        "observation_dtype": "float64",
        "observation_fields": list(OBSERVATION_FIELDS),
        "observation_shape": [4],
        "passed": bool(typed_passed),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
        "reward_dtype": "float64",
        "reward_values": [0.0, 1.0],
        "structure_kind": STRUCTURE_KIND,
        "target_dtype": "int8",
        "target_values": [0, 1],
    }

    balanced_regimes = 0
    for split in splits.values():
        for code in sorted({int(value) for value in split.regime_codes}):
            values = split.targets[split.regime_codes == code]
            if int(np.sum(values == 0)) == int(np.sum(values == 1)) == (
                TRAJECTORIES_PER_REGIME // 2
            ):
                balanced_regimes += 1
    (
        validated_blocks,
        exact_regime_order,
        exact_sample_key_order,
        within_split_keys_unique,
    ) = _generator_contract_checks(splits)
    observation_overlap_count = _cross_split_observation_overlap_count(splits)
    generator_passed = (
        dataset_sha256 == EXPECTED_DATASET_SHA256
        and dataset_sha256 == replay_sha256
        and all(partition_flags.values())
        and balanced_regimes == len(REGIMES)
        and validated_blocks == len(REGIMES) * BLOCKS_PER_REGIME
        and exact_regime_order
        and exact_sample_key_order
        and within_split_keys_unique
        and observation_overlap_count == 0
        and all(np.isfinite(split.observations).all() for split in splits.values())
        and all(
            split.observations.shape[0] == SPLIT_COUNTS[name]
            for name, split in splits.items()
        )
    )
    generator_case = {
        "balanced_regimes": balanced_regimes,
        "dataset_sha256": dataset_sha256,
        "deterministic_replay": dataset_sha256 == replay_sha256,
        "exact_regime_order": exact_regime_order,
        "exact_sample_key_order": exact_sample_key_order,
        "expected_dataset_commitment": dataset_sha256
        == EXPECTED_DATASET_SHA256,
        "finite_observations": bool(
            all(np.isfinite(split.observations).all() for split in splits.values())
        ),
        "observation_overlap_count": observation_overlap_count,
        "passed": bool(generator_passed),
        "regime_counts": REGIME_COUNTS,
        "regimes_disjoint": partition_flags["regimes"],
        "sample_keys_disjoint": partition_flags["sample_keys"],
        "split_counts": SPLIT_COUNTS,
        "trajectories_per_regime": TRAJECTORIES_PER_REGIME,
        "validated_blocks": validated_blocks,
        "within_split_keys_unique": within_split_keys_unique,
    }

    forbidden_rejections = 0
    for field in FORBIDDEN_POLICY_FIELDS:
        sentinel = {
            "observation": splits["train"].observations[0].copy(),
            field: "forbidden-sentinel",
        }
        try:
            _policy_observation(sentinel)
        except ValueError:
            forbidden_rejections += 1
    overlap_sentinel_rejections = _overlap_sentinel_rejections(splits)

    train_source_sha256 = _fit_source_commitment(
        splits["train"],
        splits["train"].observations,
    )
    true_target_sha256 = _json_sha256(
        _array_identity(splits["train"].targets)
    )
    fit_scope_sentinel_rejections = 0
    for sentinel_name in ("validation", "test"):
        try:
            _fit_policy(
                splits[sentinel_name],
                splits[sentinel_name].targets,
                expected_source_sha256=train_source_sha256,
                expected_target_sha256=true_target_sha256,
                variant="fit-scope-sentinel",
                audit_log=[],
            )
        except ValueError:
            fit_scope_sentinel_rejections += 1
    true_weights = _fit_policy(
        splits["train"],
        splits["train"].targets,
        expected_source_sha256=train_source_sha256,
        expected_target_sha256=true_target_sha256,
        variant="true-labels",
        audit_log=fit_audit,
    )
    true_actions = {
        name: _predict(split.observations, true_weights)
        for name, split in splits.items()
    }
    canonical_scores = {
        name: _accuracy_by_key(true_actions[name], splits[name])
        for name in ("validation", "test")
    }
    reversed_scores = {
        name: _reversed_accuracy_by_key(true_weights, splits[name])
        for name in ("validation", "test")
    }
    reorder_exact = canonical_scores == reversed_scores
    leakage_passed = (
        forbidden_rejections == len(FORBIDDEN_POLICY_FIELDS)
        and overlap_sentinel_rejections == 3
        and all(partition_flags.values())
        and reorder_exact
    )
    leakage_case = {
        "canonical_reorder_exact": reorder_exact,
        "fit_scopes": [],
        "fit_scope_sentinels_checked": 2,
        "fit_scope_sentinels_rejected": fit_scope_sentinel_rejections,
        "forbidden_fields_checked": len(FORBIDDEN_POLICY_FIELDS),
        "forbidden_fields_rejected": forbidden_rejections,
        "overlap_sentinels_checked": 3,
        "overlap_sentinels_rejected": overlap_sentinel_rejections,
        "passed": bool(leakage_passed),
        "policy_input_fields": list(POLICY_INPUT_FIELDS),
    }

    constant_actions = {
        name: np.zeros(split.targets.shape, dtype=np.int8)
        for name, split in splits.items()
    }
    random_actions = _random_baseline(splits)
    random_replay = _random_baseline(splits)
    constant_validation_macro, _ = _macro_accuracy(
        constant_actions["validation"], splits["validation"]
    )
    constant_test_macro, _ = _macro_accuracy(
        constant_actions["test"], splits["test"]
    )
    random_validation_macro, _ = _macro_accuracy(
        random_actions["validation"], splits["validation"]
    )
    random_test_macro, _ = _macro_accuracy(random_actions["test"], splits["test"])
    random_replay_exact = all(
        np.array_equal(random_actions[name], random_replay[name])
        for name in random_actions
    )
    heldout_row_commitment = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    replay_row_commitment = _json_sha256(
        {
            name: _split_projection(replay[name])
            for name in ("validation", "test")
        }
    )
    same_evaluation_rows = (
        heldout_row_commitment == replay_row_commitment
        and all(
            constant_actions[name].shape
            == random_actions[name].shape
            == splits[name].targets.shape
            for name in ("validation", "test")
        )
    )
    baseline_projection = {
        "constant_validation": _array_identity(constant_actions["validation"]),
        "constant_test": _array_identity(constant_actions["test"]),
        "heldout_rows_sha256": heldout_row_commitment,
        "random_validation": _array_identity(random_actions["validation"]),
        "random_test": _array_identity(random_actions["test"]),
    }
    baseline_case = {
        "constant_test_macro_accuracy": constant_test_macro,
        "constant_validation_macro_accuracy": constant_validation_macro,
        "passed": bool(
            random_replay_exact
            and same_evaluation_rows
            and all(
                np.isfinite(value)
                for value in (
                    constant_validation_macro,
                    constant_test_macro,
                    random_validation_macro,
                    random_test_macro,
                )
            )
        ),
        "random_seed": RANDOM_BASELINE_SEED,
        "random_test_macro_accuracy": random_test_macro,
        "random_validation_macro_accuracy": random_validation_macro,
        "replay_exact": random_replay_exact,
        "same_evaluation_rows": same_evaluation_rows,
        "trace_sha256": _json_sha256(baseline_projection),
    }

    train_macro, _ = _macro_accuracy(true_actions["train"], splits["train"])
    validation_macro, validation_minimum = _macro_accuracy(
        true_actions["validation"], splits["validation"]
    )
    test_macro, test_minimum = _macro_accuracy(
        true_actions["test"], splits["test"]
    )
    minimum_heldout = min(validation_minimum, test_minimum)
    recovery_passed = _positive_gate(
        train_macro=train_macro,
        validation_macro=validation_macro,
        test_macro=test_macro,
        minimum_heldout_regime=minimum_heldout,
        constant_validation_macro=constant_validation_macro,
        constant_test_macro=constant_test_macro,
        random_test_macro=random_test_macro,
    )
    recovery_projection = {
        "model": _array_identity(true_weights),
        "train_actions": _array_identity(true_actions["train"]),
        "validation_actions": _array_identity(true_actions["validation"]),
        "test_actions": _array_identity(true_actions["test"]),
    }
    recovery_case = {
        "learned_projection_sha256": _json_sha256(recovery_projection),
        "min_heldout_regime_accuracy": minimum_heldout,
        "passed": bool(recovery_passed),
        "test_gain_over_constant": test_macro - constant_test_macro,
        "test_gain_over_random": test_macro - random_test_macro,
        "test_macro_accuracy": test_macro,
        "train_macro_accuracy": train_macro,
        "validation_gain_over_constant": validation_macro
        - constant_validation_macro,
        "validation_macro_accuracy": validation_macro,
    }

    train_feature_commitment = train_source_sha256
    shuffled_train_targets = _shuffled_targets(splits["train"].targets)
    shuffled_target_sha256 = _json_sha256(
        _array_identity(shuffled_train_targets)
    )
    shuffled_weights = _fit_policy(
        splits["train"],
        shuffled_train_targets,
        expected_source_sha256=train_source_sha256,
        expected_target_sha256=shuffled_target_sha256,
        variant="within-block-label-shuffle",
        audit_log=fit_audit,
    )
    shuffled_actions = {
        name: _predict(split.observations, shuffled_weights)
        for name, split in splits.items()
    }
    shuffled_train_macro, _ = _macro_accuracy(
        shuffled_actions["train"], splits["train"]
    )
    shuffled_validation_macro, shuffled_validation_minimum = _macro_accuracy(
        shuffled_actions["validation"], splits["validation"]
    )
    shuffled_test_macro, shuffled_test_minimum = _macro_accuracy(
        shuffled_actions["test"], splits["test"]
    )
    shuffled_positive_gate = _positive_gate(
        train_macro=shuffled_train_macro,
        validation_macro=shuffled_validation_macro,
        test_macro=shuffled_test_macro,
        minimum_heldout_regime=min(
            shuffled_validation_minimum,
            shuffled_test_minimum,
        ),
        constant_validation_macro=constant_validation_macro,
        constant_test_macro=constant_test_macro,
        random_test_macro=random_test_macro,
    )
    label_marginal_preserved = np.array_equal(
        np.bincount(splits["train"].targets, minlength=2),
        np.bincount(shuffled_train_targets, minlength=2),
    )
    shuffled_block_pattern_exact = bool(
        np.all(
            shuffled_train_targets.reshape(-1, ROWS_PER_BLOCK)
            == np.asarray([0, 0, 1, 1], dtype=np.int8)
        )
    )
    shuffled_feature_commitment = _fit_source_commitment(
        splits["train"],
        splits["train"].observations,
    )
    feature_commitment_unchanged = (
        train_feature_commitment == shuffled_feature_commitment
    )
    shuffled_heldout_commitment = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    heldout_commitment_unchanged = (
        heldout_row_commitment == shuffled_heldout_commitment
    )
    shuffle_gap = test_macro - shuffled_test_macro
    shuffle_passed = (
        not shuffled_positive_gate
        and shuffled_test_macro <= THRESHOLDS["maximum_shuffle_macro_accuracy"]
        and shuffle_gap >= THRESHOLDS["minimum_true_shuffle_gap"]
        and label_marginal_preserved
        and shuffled_block_pattern_exact
        and feature_commitment_unchanged
        and heldout_commitment_unchanged
        and not np.array_equal(shuffled_train_targets, splits["train"].targets)
    )
    shuffle_case = {
        "feature_commitment_unchanged": feature_commitment_unchanged,
        "heldout_commitment_unchanged": heldout_commitment_unchanged,
        "label_marginal_preserved": bool(label_marginal_preserved),
        "nonidentity_permutation": bool(
            not np.array_equal(shuffled_train_targets, splits["train"].targets)
        ),
        "passed": bool(shuffle_passed),
        "positive_gate_rejected": not shuffled_positive_gate,
        "shuffled_block_pattern_exact": shuffled_block_pattern_exact,
        "test_macro_accuracy": shuffled_test_macro,
        "trace_sha256": _json_sha256(
            {
                "model": _array_identity(shuffled_weights),
                "targets": _array_identity(shuffled_train_targets),
                "test_actions": _array_identity(shuffled_actions["test"]),
            }
        ),
        "true_accuracy_gap": shuffle_gap,
    }

    ablated_observations = {
        name: split.observations.copy() for name, split in splits.items()
    }
    for observations in ablated_observations.values():
        observations[:, 0] = 0.0
    only_signal_changed = bool(
        all(
            np.all(ablated_observations[name][:, 0] == 0.0)
            and np.array_equal(
                ablated_observations[name][:, 1:],
                split.observations[:, 1:],
            )
            and np.any(split.observations[:, 0] != 0.0)
            for name, split in splits.items()
        )
    )
    ablated_train_source_sha256 = _fit_source_commitment(
        splits["train"],
        ablated_observations["train"],
    )
    ablated_weights = _fit_policy(
        splits["train"],
        splits["train"].targets,
        observations=ablated_observations["train"],
        expected_source_sha256=ablated_train_source_sha256,
        expected_target_sha256=true_target_sha256,
        variant="signal-ablation",
        audit_log=fit_audit,
    )
    nuisance_only_actions = _predict(ablated_observations["test"], ablated_weights)
    true_policy_zeroed_actions = _predict(
        ablated_observations["test"],
        true_weights,
    )
    nuisance_only_macro, _ = _macro_accuracy(
        nuisance_only_actions,
        splits["test"],
    )
    true_policy_zeroed_macro, _ = _macro_accuracy(
        true_policy_zeroed_actions,
        splits["test"],
    )
    attribution_heldout_commitment = _json_sha256(
        {
            name: _split_projection(splits[name])
            for name in ("validation", "test")
        }
    )
    attribution_heldout_unchanged = (
        heldout_row_commitment == attribution_heldout_commitment
    )
    attribution_passed = (
        nuisance_only_macro <= THRESHOLDS["maximum_attribution_macro_accuracy"]
        and true_policy_zeroed_macro
        <= THRESHOLDS["maximum_attribution_macro_accuracy"]
        and only_signal_changed
        and attribution_heldout_unchanged
    )
    attribution_case = {
        "heldout_commitment_unchanged": attribution_heldout_unchanged,
        "nuisance_only_test_macro_accuracy": nuisance_only_macro,
        "only_signal_changed": only_signal_changed,
        "passed": bool(attribution_passed),
        "signal_zeroed_true_policy_test_macro_accuracy": true_policy_zeroed_macro,
        "trace_sha256": _json_sha256(
            {
                "nuisance_model": _array_identity(ablated_weights),
                "nuisance_test_actions": _array_identity(nuisance_only_actions),
                "true_zeroed_test_actions": _array_identity(
                    true_policy_zeroed_actions
                ),
            }
        ),
    }

    fit_scopes = sorted({entry["scope"] for entry in fit_audit})
    leakage_case["fit_scopes"] = fit_scopes
    leakage_case["passed"] = bool(
        leakage_case["passed"]
        and fit_scope_sentinel_rejections == 2
        and fit_scopes == ["train"]
        and len(fit_audit) == 3
        and {entry["variant"] for entry in fit_audit}
        == {
            "signal-ablation",
            "true-labels",
            "within-block-label-shuffle",
        }
    )

    return {
        "baseline_replay": baseline_case,
        "generator_partition": generator_case,
        "label_shuffle_control": shuffle_case,
        "leakage_guards": leakage_case,
        "signal_attribution_control": attribution_case,
        "supervised_recovery": recovery_case,
        "typed_task_contract": typed_case,
    }


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
            "experiments.local_lab.supervised_toy_signal_worker",
            "--mode",
            "supervised-toy-signal-trace",
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
        raise RuntimeError("the supervised toy-signal study requires a CPU backend")

    local_projection = isolated_worker_trace()
    cases = dict(local_projection["cases"])
    if include_process_isolation:
        isolated_left = _isolated_trace()
        isolated_right = _isolated_trace()
        isolation_passed = (
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

    completed = all(case["passed"] is not None for case in cases.values())
    passed = completed and all(bool(case["passed"]) for case in cases.values())
    return {
        "action": (
            "synthetic_supervised_toy_signal_recovered_for_harness"
            if passed
            else (
                "park_learning_contract_research"
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
